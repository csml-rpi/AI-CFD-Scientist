"""Run an OpenFOAM case on NERSC Perlmutter through the Superfacility API.

The case is prepared locally as usual, packed, copied to Perlmutter's scratch,
run there by a Slurm job, and its results copied back into the same local
case folder, so every local tool (scorers, analysis, plots) reads them as if
the case had run here.

Settings, all from the environment:
  CFD_HPC_ALLOW_SUBMIT   "1" to submit jobs. Unset, submit_case writes the job
                         script and stops (a dry run). A study submits only when
                         the person who launched it said so.
  CFD_HPC_ACCOUNT        the NERSC project charged (e.g. m4756). Required to submit.
  CFD_HPC_MAX_NODE_HOURS most node-hours one study may request in total (default 20).
  CFD_HPC_MAX_NODES      most nodes one job may ask for (default 4).
  CFD_HPC_MAX_WALLTIME   longest time limit one job may ask for (default 12:00:00).
  CFD_HPC_NERSC_KEY      the SF API private key (default: the one .pem file in
                         ~/.superfacility). Read by this module only, never shown.
  CFD_HPC_REMOTE_ROOT    where cases go on Perlmutter (default $SCRATCH/cfd_scientist).
  CFD_HPC_OPENFOAM_SETUP shell line run in the job before the case (default: none;
                         the job is a login shell, so ~/.bashrc's OpenFOAM loads).
  CFD_HPC_MPI_LAUNCH     launcher for parallel runs (default "srun --mpi=pmix": the
                         OpenMPI 5 that OpenFOAM is built with starts through PMIx).

Every job is kept in <study>/hpc/jobs.json: what was asked, what it was
charged and where its files are.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shlex
import tarfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

MACHINE = "perlmutter"
CORES_PER_NODE = 128          # Perlmutter CPU node, physical cores
CHUNK = 16 * 1024 * 1024      # the SF API refuses an upload body over ~16-32 MB (413)
QOS = ("regular", "debug", "shared", "preempt")
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE")
# What `fetch_results` brings back. Field files of a large 3-D case are
# gigabytes per time; the light set is what scoring and analysis usually read.
FETCH = {
    "light": "log.* postProcessing slurm-*.out hpc_exit.txt",
    "latest": "log.* postProcessing slurm-*.out hpc_exit.txt @LATEST@",
    "all": ". --exclude=./processor*",
}


@dataclass
class HpcConfig:
    allow_submit: bool
    account: str
    max_node_hours: float
    max_nodes: int
    max_walltime_s: int
    key: Optional[Path]
    remote_root: str
    openfoam_setup: str
    mpi_launch: str

    @classmethod
    def from_env(cls) -> "HpcConfig":
        key = os.environ.get("CFD_HPC_NERSC_KEY", "").strip()
        if key:
            key_path: Optional[Path] = Path(key).expanduser()
        else:
            pems = sorted((Path.home() / ".superfacility").glob("*.pem"))
            key_path = pems[0] if len(pems) == 1 else None
        return cls(
            allow_submit=os.environ.get("CFD_HPC_ALLOW_SUBMIT", "") == "1",
            account=os.environ.get("CFD_HPC_ACCOUNT", "").strip(),
            max_node_hours=float(os.environ.get("CFD_HPC_MAX_NODE_HOURS", "20")),
            max_nodes=int(os.environ.get("CFD_HPC_MAX_NODES", "4")),
            max_walltime_s=parse_walltime(os.environ.get("CFD_HPC_MAX_WALLTIME", "12:00:00")),
            key=key_path,
            remote_root=os.environ.get("CFD_HPC_REMOTE_ROOT", "").strip(),
            openfoam_setup=os.environ.get("CFD_HPC_OPENFOAM_SETUP", "").strip(),
            mpi_launch=os.environ.get("CFD_HPC_MPI_LAUNCH", "srun --mpi=pmix").strip() or "srun --mpi=pmix",
        )


def parse_walltime(value: Any) -> int:
    """Seconds in a Slurm time limit: minutes, MM:SS, HH:MM:SS or D-HH:MM:SS."""
    text = str(value).strip()
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 1:
        secs = parts[0] * 60
    elif len(parts) == 2:
        secs = parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        secs = parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        raise ValueError(f"not a time limit: {value!r}")
    return days * 86400 + secs


def format_walltime(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h:02d}:{rem // 60:02d}:{rem % 60:02d}"


def node_hours(nodes: int, ntasks: int, qos: str, walltime_s: int) -> float:
    """What a job can be charged at most: the shared QOS bills the cores it
    holds as a fraction of a node, every other QOS whole nodes."""
    share = min(1.0, ntasks / CORES_PER_NODE) if qos == "shared" else float(nodes)
    return round(share * walltime_s / 3600.0, 3)


# ---------------------------------------------------------------- the ledger

def _ledger_path(out_dir: Path) -> Path:
    return Path(out_dir) / "hpc" / "jobs.json"


def load_jobs(out_dir: Path) -> List[Dict[str, Any]]:
    p = _ledger_path(out_dir)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return []


def _save_jobs(out_dir: Path, jobs: List[Dict[str, Any]]) -> None:
    p = _ledger_path(out_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=2))
    tmp.replace(p)


def _job_record(out_dir: Path, job_id: str) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    jobs = load_jobs(out_dir)
    return jobs, next((j for j in jobs if str(j.get("job_id")) == str(job_id)), None)


def charged_node_hours(jobs: List[Dict[str, Any]]) -> float:
    """Node-hours this study has spent or holds: the measured charge of a
    finished job, the requested maximum of one still queued or running."""
    total = 0.0
    for j in jobs:
        if j.get("job_id") is None:
            continue
        total += float(j["used_node_hours"] if j.get("used_node_hours") is not None
                       else j.get("requested_node_hours", 0.0))
    return round(total, 3)


# ------------------------------------------------------------ the connection

@contextmanager
def connect(cfg: HpcConfig) -> Iterator[Any]:
    """The Perlmutter compute resource, authenticated with the SF API key."""
    from sfapi_client import Client
    from sfapi_client.compute import Machine

    if cfg.key is None or not cfg.key.is_file():
        raise RuntimeError(
            "No NERSC Superfacility API key: set CFD_HPC_NERSC_KEY, or keep exactly one .pem "
            "file in ~/.superfacility")
    # A bare Client() does not load the key file; it has to be passed.
    with Client(key=cfg.key) as client:
        yield client, client.compute(Machine.perlmutter)


def run_remote(pm: Any, command: str) -> Tuple[int, str]:
    """One command on a Perlmutter login node, as a login shell (so ~/.bashrc
    and its modules load). Returns (exit code, output).

    Not Compute.run(): it raises on a non-zero exit, and on stderr noise when
    the API gives no exit code, which Lmod's module messages are."""
    task = pm._run_task(f"bash -lc {shlex.quote(command)}")
    task._wait()
    result = task.result
    code = getattr(result, "exit_code", None)
    out = (getattr(result, "output", "") or "") + (getattr(result, "error", "") or "")
    return (0 if code is None else int(code)), out


def _remote_root(pm: Any, cfg: HpcConfig) -> str:
    if cfg.remote_root:
        return cfg.remote_root.rstrip("/")
    code, out = run_remote(pm, 'echo "ROOT=$SCRATCH"')
    root = next((l[5:].strip() for l in out.splitlines() if l.startswith("ROOT=")), "")
    if code != 0 or not root:
        raise RuntimeError(f"could not read $SCRATCH on Perlmutter: {out[-500:]}")
    return root.rstrip("/") + "/cfd_scientist"


# ------------------------------------------------------------- file transfer

def pack_case(case_dir: Path, decomposed: bool = False) -> bytes:
    """The case as a .tar.gz; its processor* folders only when decomposed
    (a large mesh split here saves the job the serial decomposePar)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for child in sorted(Path(case_dir).iterdir()):
            if child.name.startswith("processor") and not decomposed:
                continue
            tf.add(child, arcname=f"{Path(case_dir).name}/{child.name}")
    return buf.getvalue()


def upload_bytes(pm: Any, data: bytes, remote_dir: str, name: str) -> None:
    """Copy data to remote_dir/name in CHUNK pieces, joined and checked there."""
    [d] = pm.ls(remote_dir, directory=True)
    parts = []
    for i in range(0, max(1, len(data)), CHUNK):
        piece = io.BytesIO(data[i:i + CHUNK])
        piece.filename = f"{name}.part{len(parts):04d}"
        d.upload(piece)
        parts.append(piece.filename)
    want = hashlib.sha256(data).hexdigest()
    code, out = run_remote(pm, f"cd {shlex.quote(remote_dir)} && cat {' '.join(parts)} > {shlex.quote(name)}"
                               f" && rm -f {' '.join(parts)} && sha256sum {shlex.quote(name)}")
    if code != 0 or want not in out:
        raise RuntimeError(f"upload of {name} did not arrive intact: {out[-500:]}")


def download_file(pm: Any, remote_file: str) -> bytes:
    """remote_file's bytes, moved in CHUNK pieces and checked."""
    q = shlex.quote(remote_file)
    code, out = run_remote(pm, f"rm -f {q}.part* && split -b {CHUNK} -d -a 4 {q} {q}.part"
                               f" && sha256sum {q} && ls -1 {q}.part*")
    if code != 0:
        raise RuntimeError(f"could not prepare {remote_file} for download: {out[-500:]}")
    lines = out.split()
    want = lines[0]
    parts = sorted(l for l in lines if l.startswith(remote_file + ".part"))
    data = b""
    for part in parts:
        [f] = pm.ls(part)
        data += f.download(binary=True).read()
    run_remote(pm, f"rm -f {q}.part*")
    if hashlib.sha256(data).hexdigest() != want:
        raise RuntimeError(f"download of {remote_file} arrived corrupted")
    return data


# ---------------------------------------------------------------- the job

def default_commands(case_dir: Path, ntasks: int, mpi_launch: str) -> str:
    """Allrun when the case has one; otherwise its controlDict application,
    decomposed and run in parallel when it has more than one task."""
    if (Path(case_dir) / "Allrun").is_file():
        return "chmod +x Allrun && ./Allrun"
    app = ""
    cd = Path(case_dir) / "system" / "controlDict"
    if cd.is_file():
        m = re.search(r"^\s*application\s+(\w+)\s*;", cd.read_text(errors="ignore"), re.M)
        app = m.group(1) if m else ""
    if not app:
        raise ValueError("the case has neither an Allrun nor an application in system/controlDict; "
                         "give the commands to run")
    if ntasks <= 1:
        return f"{app} > log.{app} 2>&1"
    return "\n".join([
        f"if [ ! -d processor{ntasks - 1} ] || [ -d processor{ntasks} ]; then",
        f"  foamDictionary -entry numberOfSubdomains -set {ntasks} system/decomposeParDict",
        "  decomposePar -force > log.decomposePar 2>&1",
        "fi",
        f"{mpi_launch} -n {ntasks} {app} -parallel > log.{app} 2>&1",
        "reconstructPar > log.reconstructPar 2>&1",
    ])


def job_script(*, account: str, qos: str, nodes: int, ntasks: int, walltime_s: int, name: str,
               remote_case: str, commands: str, openfoam_setup: str, after: str = "") -> str:
    """The Slurm script. The run's exit code goes to hpc_exit.txt so a failed
    solver is told apart from a job that was killed. after: a job id this one
    waits for (it starts once that job ends, however it ended)."""
    sizing = ([f"#SBATCH -n {ntasks}", "#SBATCH -c 2"] if qos == "shared"
              else [f"#SBATCH -N {nodes}", f"#SBATCH --ntasks-per-node {max(1, math.ceil(ntasks / nodes))}"])
    return "\n".join([
        "#!/bin/bash -l",
        f"#SBATCH -A {account}",
        "#SBATCH -C cpu",
        f"#SBATCH -q {qos}",
        *sizing,
        f"#SBATCH -t {format_walltime(walltime_s)}",
        f"#SBATCH -J {name}",
        *([f"#SBATCH --dependency=afterany:{int(after)}"] if after else []),
        f"#SBATCH -o {remote_case}/slurm-%j.out",
        f"cd {shlex.quote(remote_case)}",
        openfoam_setup,
        'echo "host $(hostname) start $(date -Is) OpenFOAM ${WM_PROJECT_VERSION:-not loaded}"',
        "(",
        "set -e",
        commands,
        ")",
        "rc=$?",
        'echo "$rc" > hpc_exit.txt',
        'echo "exit $rc end $(date -Is)"',
        "exit $rc",
        "",
    ])


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:60] or "case"


def submit_case(out_dir: Path, case_dir: Path, *, nodes: int = 1, ntasks: int = 0, time_limit: str = "01:00:00",
                qos: str = "regular", commands: str = "", label: str = "",
                decomposed: bool = False, rerun_of: str = "", after: str = "") -> Dict[str, Any]:
    """Copy the case to Perlmutter and submit it. See the module docstring for
    the settings that gate and bound it. rerun_of: the id of an earlier job of
    this study whose copy on Perlmutter is run again (no new upload); the
    commands then run in that folder, after whatever the earlier job left.
    after: the id of a job this one waits for, to chain long runs past the
    wall-clock limit (use with rerun_of so both run in the same folder)."""
    cfg = HpcConfig.from_env()
    case_dir = Path(case_dir)
    if qos not in QOS:
        return {"ok": False, "error": f"qos must be one of {QOS}"}
    walltime_s = parse_walltime(time_limit)
    if walltime_s > cfg.max_walltime_s:
        return {"ok": False, "error": f"time limit {time_limit} is over this study's maximum "
                                      f"{format_walltime(cfg.max_walltime_s)} (CFD_HPC_MAX_WALLTIME)"}
    if qos == "shared":
        nodes = 1
        ntasks = ntasks or 1
        if ntasks > CORES_PER_NODE // 2:
            return {"ok": False, "error": "the shared QOS holds at most half a node; use regular"}
    else:
        ntasks = ntasks or nodes * CORES_PER_NODE
    if nodes > cfg.max_nodes:
        return {"ok": False, "error": f"{nodes} nodes is over this study's maximum {cfg.max_nodes} "
                                      "(CFD_HPC_MAX_NODES)"}
    cost = node_hours(nodes, ntasks, qos, walltime_s)
    jobs = load_jobs(out_dir)
    spent = charged_node_hours(jobs)
    if spent + cost > cfg.max_node_hours:
        return {"ok": False, "error": (
            f"this job could cost {cost} node-hours; the study has {spent} of its "
            f"{cfg.max_node_hours} committed (CFD_HPC_MAX_NODE_HOURS). Ask for fewer nodes or less "
            "time, or finish with the results you have.")}
    try:
        cmds = commands.strip() or default_commands(case_dir, ntasks, cfg.mpi_launch)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = _safe_name(label or case_dir.name)
    record: Dict[str, Any] = {
        "job_id": None, "case": str(case_dir), "label": label, "qos": qos, "nodes": nodes,
        "ntasks": ntasks, "time_limit": format_walltime(walltime_s), "account": cfg.account,
        "requested_node_hours": cost, "used_node_hours": None, "commands": cmds,
        "created": stamp, "state": "DRY_RUN", "fetched": [],
    }
    if not cfg.allow_submit or not cfg.account:
        script = job_script(account=cfg.account or "<CFD_HPC_ACCOUNT>", qos=qos, nodes=nodes, ntasks=ntasks,
                            walltime_s=walltime_s, name=name, remote_case="<remote case folder>",
                            commands=cmds, openfoam_setup=cfg.openfoam_setup)
        dest = Path(out_dir) / "hpc" / f"dry_run_{name}_{stamp}.sh"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(script)
        jobs.append({**record, "script": str(dest)})
        _save_jobs(out_dir, jobs)
        why = ("CFD_HPC_ALLOW_SUBMIT is not 1" if not cfg.allow_submit else "CFD_HPC_ACCOUNT is not set")
        return {"ok": False, "dry_run": True, "script": str(dest), "requested_node_hours": cost,
                "error": f"not submitted: {why}. Submitting to Perlmutter is switched on by the person "
                         "who launches the study; report that the case is ready and how much it would cost."}
    earlier = None
    if rerun_of:
        earlier = next((j for j in jobs if str(j.get("job_id")) == str(rerun_of) and j.get("remote_dir")), None)
        if earlier is None:
            return {"ok": False, "error": f"job {rerun_of} is not one this study submitted"}
        record["rerun_of"] = str(rerun_of)
    if after:
        record["after"] = str(after)
    data = b"" if earlier else pack_case(case_dir, decomposed=decomposed)
    with connect(cfg) as (client, pm):
        projects = {p.repo_name for p in client.user().projects()}
        if cfg.account not in projects:
            return {"ok": False, "error": f"account {cfg.account} is not one of this user's projects {sorted(projects)}"}
        if earlier:
            remote = earlier["remote_dir"]
            code, out = run_remote(pm, f"test -f {shlex.quote(remote)}/system/controlDict")
            if code != 0:
                return {"ok": False, "error": f"the copy of job {rerun_of} is gone from Perlmutter ({remote})"}
        else:
            remote = f"{_remote_root(pm, cfg)}/{_safe_name(Path(out_dir).name)}/{name}_{stamp}"
            code, out = run_remote(pm, f"mkdir -p {shlex.quote(remote)}")
            if code != 0:
                return {"ok": False, "error": f"could not create {remote}: {out[-500:]}"}
            upload_bytes(pm, data, remote, "case.tgz")
            code, out = run_remote(pm, f"cd {shlex.quote(remote)} && tar xzf case.tgz --strip-components=1 && rm case.tgz")
            if code != 0:
                return {"ok": False, "error": f"could not unpack the case on Perlmutter: {out[-500:]}"}
        script = job_script(account=cfg.account, qos=qos, nodes=nodes, ntasks=ntasks, walltime_s=walltime_s,
                            name=name, remote_case=remote, commands=cmds, openfoam_setup=cfg.openfoam_setup,
                            after=after)
        job = pm.submit_job(script)
        record.update(job_id=str(job.jobid), remote_dir=remote, state="PENDING", script=script,
                      uploaded_mb=round(len(data) / 1e6, 1))
    jobs.append(record)
    _save_jobs(out_dir, jobs)
    return {"ok": True, "job_id": record["job_id"], "remote_dir": remote, "requested_node_hours": cost,
            "study_node_hours_committed": round(spent + cost, 3), "study_node_hours_max": cfg.max_node_hours}


def _sacct(pm: Any, job_id: str) -> Dict[str, str]:
    code, out = run_remote(pm, f"sacct -j {int(job_id)} -X -n -P "
                               "--format=JobID,State,Elapsed,ElapsedRaw,NNodes,ExitCode,Reason")
    for line in out.splitlines():
        f = line.split("|")
        if len(f) >= 7 and f[0] == str(job_id):
            return {"state": f[1].split()[0], "elapsed": f[2], "elapsed_s": f[3], "nodes": f[4],
                    "exit_code": f[5], "reason": f[6]}
    return {}


def job_status(out_dir: Path, job_id: str, tail_lines: int = 30) -> Dict[str, Any]:
    """The job's Slurm state, and the tail of its output once it has started."""
    jobs, rec = _job_record(out_dir, job_id)
    if rec is None:
        return {"ok": False, "error": f"job {job_id} is not one this study submitted"}
    cfg = HpcConfig.from_env()
    with connect(cfg) as (client, pm):
        info = _sacct(pm, job_id)
        tail = ""
        if info.get("state") not in (None, "PENDING"):
            _, tail = run_remote(pm, f"cd {shlex.quote(rec['remote_dir'])} && tail -n {int(tail_lines)} "
                                     f"slurm-{int(job_id)}.out; echo; cat hpc_exit.txt 2>/dev/null "
                                     "| sed 's/^/run exit code: /'")
    if not info:
        return {"ok": False, "error": f"sacct has no record of job {job_id} yet; try again shortly"}
    rec["state"] = info["state"]
    if info["state"] in TERMINAL and info.get("elapsed_s", "").isdigit():
        share = (float(rec["ntasks"]) / CORES_PER_NODE if rec["qos"] == "shared" else float(info["nodes"] or rec["nodes"]))
        rec["used_node_hours"] = round(share * int(info["elapsed_s"]) / 3600.0, 3)
    _save_jobs(out_dir, jobs)
    return {"ok": True, "job_id": str(job_id), **info, "finished": info["state"] in TERMINAL,
            "output_tail": tail[-4000:], "used_node_hours": rec.get("used_node_hours")}


def wait_for_job(out_dir: Path, job_id: str, max_wait_s: int = 3600, poll_s: int = 60) -> Dict[str, Any]:
    """job_status every poll_s until the job ends or max_wait_s passes."""
    deadline = time.time() + max(0, int(max_wait_s))
    while True:
        status = job_status(out_dir, job_id)
        if not status.get("ok") and "no record" not in status.get("error", ""):
            return status
        if status.get("finished") or time.time() >= deadline:
            status["waited_until_end"] = bool(status.get("finished"))
            return status
        time.sleep(max(15, int(poll_s)))


def fetch_results(out_dir: Path, job_id: str, what: str = "light") -> Dict[str, Any]:
    """Copy the job's results back into the local case folder.
    what: light (logs, postProcessing, Slurm output), latest (light plus the
    last time folder) or all (everything except processor* folders)."""
    if what not in FETCH:
        return {"ok": False, "error": f"what must be one of {sorted(FETCH)}"}
    jobs, rec = _job_record(out_dir, job_id)
    if rec is None:
        return {"ok": False, "error": f"job {job_id} is not one this study submitted"}
    cfg = HpcConfig.from_env()
    remote = rec["remote_dir"]
    sel = FETCH[what]
    if "@LATEST@" in sel:
        latest = ("$(ls -d [0-9]* 2>/dev/null | sort -g | tail -1)")
        sel = sel.replace("@LATEST@", latest)
    name = f"results_{int(job_id)}.tgz"
    with connect(cfg) as (client, pm):
        code, out = run_remote(pm, f"cd {shlex.quote(remote)} && rm -f {name} && "
                                   f"tar czf {name} --ignore-failed-read {sel} 2>&1 | tail -5; ls -l {name}")
        if code != 0 or name not in out:
            return {"ok": False, "error": f"could not pack the results on Perlmutter: {out[-800:]}"}
        data = download_file(pm, f"{remote}/{name}")
        run_remote(pm, f"rm -f {shlex.quote(remote)}/{name}")
    case_dir = Path(rec["case"])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = [m for m in tf.getmembers() if not m.name.startswith(("processor", "./processor"))]
        tf.extractall(case_dir, members=members, filter="data")
    rec.setdefault("fetched", []).append({"what": what, "at": time.strftime("%Y%m%d_%H%M%S"),
                                          "mb": round(len(data) / 1e6, 1)})
    _save_jobs(out_dir, jobs)
    return {"ok": True, "case": str(case_dir), "mb": round(len(data) / 1e6, 1), "files": len(members)}


def cancel_job(out_dir: Path, job_id: str) -> Dict[str, Any]:
    jobs, rec = _job_record(out_dir, job_id)
    if rec is None:
        return {"ok": False, "error": f"job {job_id} is not one this study submitted"}
    with connect(HpcConfig.from_env()) as (client, pm):
        code, out = run_remote(pm, f"scancel {int(job_id)}")
    return {"ok": code == 0, "job_id": str(job_id), "output": out[-500:]}


def hpc_overview(out_dir: Path) -> Dict[str, Any]:
    """Whether Perlmutter is reachable and set up, what this study may spend and has spent."""
    cfg = HpcConfig.from_env()
    jobs = load_jobs(out_dir)
    result: Dict[str, Any] = {
        "submit_allowed": cfg.allow_submit and bool(cfg.account), "account": cfg.account or None,
        "study_node_hours_committed": charged_node_hours(jobs), "study_node_hours_max": cfg.max_node_hours,
        "max_nodes_per_job": cfg.max_nodes, "max_time_limit": format_walltime(cfg.max_walltime_s),
        "jobs": [{k: j.get(k) for k in ("job_id", "label", "case", "state", "requested_node_hours",
                                         "used_node_hours")} for j in jobs],
    }
    try:
        with connect(cfg) as (client, pm):
            result["perlmutter"] = getattr(pm.status, "value", str(pm.status))
            projects = client.user().projects()
            result["projects"] = {p.repo_name: {"cpu_hours_used": p.cpu_hours_used,
                                                "cpu_hours_given": p.cpu_hours_given} for p in projects}
            code, out = run_remote(pm, 'echo "OF=${WM_PROJECT_VERSION:-none}"; command -v simpleFoam || true')
            result["openfoam_version"] = next((l[3:] for l in out.splitlines() if l.startswith("OF=")), "unknown")
            result["remote_root"] = _remote_root(pm, cfg)
    except Exception as exc:  # reported, not raised: the agent must see why
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
