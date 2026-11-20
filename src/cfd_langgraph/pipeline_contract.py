"""The two scripts a fitted-model candidate leaves, which the framework runs itself.

Scoring the files a candidate wrote trusted two things the candidate controlled.
Its timing files: on airfrans_codex_20260913 and _20261008 the inference times
the scorer turns into a speed-up were written by hand. And its seeds: the same
runs submitted seeds that were copies, or 99% copies, of one seed, and the
scorer, which requires every seed to pass, never checked that they differed.

So in a study with a validation split (cfd_langgraph.validation_split) every
candidate leaves
    pipeline/train.sh   SEED MODEL_DIR
    pipeline/predict.sh SEED MODEL_DIR OUT_DIR SET
and its trained seeds under pipeline/models/seed<SEED>/. The framework runs
predict.sh for every seed and evaluation set in the candidate's sandbox, times
each run, writes the timing files the scorer reads itself, rejects seeds whose
predictions are copies of each other, and scores those files. At the end of the
study it also runs train.sh, for the candidate it picked, on the full training
data, and scores it once on the test set.

What the scorer reads per seed (the prediction files of each evaluation set, and
the file reporting measured times under which keys) is proposed once per study
by an LLM from the scorer's source and checked against the supplied baseline.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SPEC_FILE = "pipeline_spec.json"
PIPELINE_DIR = "pipeline"
MODELS_DIR = "models"
NEAR_COPY = 1e-3
_SCORER_SOURCE_CHARS = 14000

PROPOSE_SYSTEM = "You read scoring code precisely. Answer with one JSON object and nothing else."

PROPOSE_USER = """A study's scorer reads, for every seed (or run) of a result, one or more \
prediction files for each evaluation set, and possibly a file in which the run reports its own \
measured times. From the scorer's source, the task brief and the supplied baseline result below, \
describe exactly what it reads.

Reply with one JSON object:
{{"seeds": [every seed number the study requires a result to have],
  "sets": [{{"name": "<evaluation set name, as the scorer names it>",
             "files": ["<file name with {{seed}} where the seed number goes>", ...]}}],
  "timing": null, or {{"file": "<file name with {{seed}}>",
                      "seconds": {{"<key>": "<name of the set whose prediction time it holds>"}},
                      "text": {{"<key>": "<what the text describes>"}}}}}}
Rules: every file the scorer reads for one seed appears exactly once, either in one set's files
or as the timing file. File names are relative to the folder the scorer is given. Use "timing"
only for a file whose numbers the run measures itself; "seconds" maps each such number to the
evaluation set it times.

TASK BRIEF:
{brief}

THE SCORER ({scorer_path}):
{scorer_source}

THE SUPPLIED BASELINE RESULT ({baseline_dir}):
{baseline_listing}
{feedback}"""


# --------------------------------------------------------------------------- #
# The spec                                                                      #
# --------------------------------------------------------------------------- #

def _json_object(text: str) -> Optional[Dict[str, Any]]:
    text = str(text or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def file_name(pattern: str, seed: int) -> str:
    return pattern.replace("{seed}", str(seed))


def check_spec(spec: Dict[str, Any], baseline_dir: Path) -> List[str]:
    """What is wrong with a spec, judged against the supplied baseline result."""
    problems: List[str] = []
    seeds = spec.get("seeds")
    if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) and s >= 0 for s in seeds):
        problems.append("'seeds' must be a non-empty list of non-negative integers.")
    sets = spec.get("sets")
    if not isinstance(sets, list) or not sets:
        return problems + ["'sets' must be a non-empty list."]
    names = [str(s.get("name", "")) for s in sets if isinstance(s, dict)]
    if len(names) != len(sets) or len(set(names)) != len(names) or not all(n.strip() for n in names):
        problems.append("every set needs a distinct, non-empty name.")
    patterns: List[str] = []
    for s in sets:
        files = s.get("files") if isinstance(s, dict) else None
        if not isinstance(files, list) or not files:
            problems.append(f"set {s.get('name') if isinstance(s, dict) else s!r} lists no files.")
            continue
        for f in files:
            if "{seed}" not in str(f) or "/" in str(f):
                problems.append(f"file name {f!r} must contain {{seed}} and no folder.")
            patterns.append(str(f))
    timing = spec.get("timing")
    if timing is not None:
        if not isinstance(timing, dict) or "{seed}" not in str(timing.get("file", "")):
            problems.append("'timing' must be null or name a file containing {seed}.")
        else:
            seconds = timing.get("seconds") or {}
            if not isinstance(seconds, dict) or not seconds:
                problems.append("'timing.seconds' must map at least one key to a set name.")
            elif any(v not in names for v in seconds.values()):
                problems.append(f"'timing.seconds' names a set that is not in 'sets': {list(seconds.values())}.")
    if problems:
        return problems
    present = [s for s in range(0, 100) if all((Path(baseline_dir) / file_name(p, s)).is_file() for p in patterns)]
    if not present:
        problems.append("for no seed number does the supplied baseline result hold every file named "
                        f"in 'sets' ({patterns}).")
    elif timing:
        tf = Path(baseline_dir) / file_name(str(timing["file"]), present[0])
        try:
            doc = json.loads(tf.read_text())
        except (OSError, ValueError):
            doc = None
        if not isinstance(doc, dict):
            problems.append(f"the baseline result's timing file {tf.name} is missing or not a JSON object.")
        else:
            missing = [k for k in list(timing["seconds"]) + list(timing.get("text") or {}) if k not in doc]
            if missing:
                problems.append(f"the baseline's {tf.name} has no keys {missing}.")
    return problems


def propose_spec(*, llm_invoke: Callable[[List[Tuple[str, str]]], str], brief: str, scorer_path: Path,
                 baseline_dir: Path, baseline_listing: str, attempts: int = 3) -> Tuple[Dict[str, Any], List[str]]:
    try:
        source = Path(scorer_path).read_text(errors="replace")[:_SCORER_SOURCE_CHARS]
    except OSError as exc:
        source = f"(could not read: {exc})"
    feedback, problems = "", ["not attempted"]
    spec: Dict[str, Any] = {}
    for _ in range(attempts):
        prompt = PROPOSE_USER.format(brief=brief[:12000], scorer_path=scorer_path, scorer_source=source,
                                     baseline_dir=baseline_dir, baseline_listing=baseline_listing,
                                     feedback=feedback)
        try:
            spec = _json_object(llm_invoke([("system", PROPOSE_SYSTEM), ("user", prompt)])) or {}
        except Exception as exc:  # noqa: BLE001
            problems = [f"the call failed: {exc}"]
            continue
        problems = check_spec(spec, baseline_dir) if spec else ["the reply held no JSON object."]
        if not problems:
            return spec, []
        feedback = ("\n\nYOUR PREVIOUS ANSWER WAS WRONG:\n- " + "\n- ".join(problems)
                    + f"\nIt was: {json.dumps(spec)[:3000]}\nAnswer again.")
    return {}, problems


def load_spec(path: Path) -> Dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


# --------------------------------------------------------------------------- #
# What a candidate must leave                                                    #
# --------------------------------------------------------------------------- #

def scripts(run_dir: Path) -> Tuple[Path, Path]:
    d = Path(run_dir) / PIPELINE_DIR
    return d / "train.sh", d / "predict.sh"


def model_dir(run_dir: Path, seed: int) -> Path:
    return Path(run_dir) / PIPELINE_DIR / MODELS_DIR / f"seed{seed}"


def missing_parts(run_dir: Path, spec: Dict[str, Any]) -> List[str]:
    train, predict = scripts(run_dir)
    out = [str(p) for p in (train, predict) if not p.is_file()]
    for s in spec.get("seeds") or []:
        d = model_dir(run_dir, int(s))
        if not d.is_dir() or not any(f.is_file() for f in d.rglob("*")):
            out.append(f"{d} (trained seed {s})")
    return out


# --------------------------------------------------------------------------- #
# Running it                                                                    #
# --------------------------------------------------------------------------- #

def hardware() -> str:
    gpu = ""
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        names = [n.strip() for n in proc.stdout.splitlines() if n.strip()]
        if visible.strip() and names:
            idx = [int(i) for i in visible.split(",") if i.strip().isdigit() and int(i) < len(names)]
            names = [names[i] for i in idx] or names
        gpu = ", ".join(names[:2])
    except Exception:  # noqa: BLE001
        pass
    cpu = platform.processor() or platform.machine()
    return f"{gpu or 'no GPU'}; CPU {cpu}; {os.cpu_count()} cores (timed by the framework)"


def _tail(text: str, n: int = 1500) -> str:
    return (text or "")[-n:]


def _failed(res: Dict[str, Any]) -> bool:
    """A Sandbox.run_bash result for a command that was refused, timed out or
    exited non-zero."""
    return bool(res.get("error")) or bool(res.get("timeout")) or res.get("rc", 0) != 0


def run_predictions(*, sandbox: Any, run_dir: Path, spec: Dict[str, Any], models_root: Path,
                    out_dir: Path, timeout_s: int = 7200) -> Dict[str, Any]:
    """predict.sh for every seed and set, each timed and each in a fresh folder;
    only the files the scorer reads are kept. ``sandbox`` is a
    code_mod_agentic.Sandbox whose run_dir contains ``out_dir``."""
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    _, predict = scripts(run_dir)
    runs: List[Dict[str, Any]] = []
    errors: List[str] = []
    hw = hardware()
    for seed in spec["seeds"]:
        seconds: Dict[str, float] = {}
        for s in spec["sets"]:
            scratch = out_dir / f".run_seed{seed}_{s['name']}"
            scratch.mkdir(parents=True)
            cmd = f"bash {predict} {seed} {Path(models_root) / f'seed{seed}'} {scratch} {s['name']}"
            started = time.monotonic()
            res = sandbox.run_bash(cmd, cwd=str(run_dir), timeout=timeout_s)
            took = time.monotonic() - started
            wanted = [file_name(f, seed) for f in s["files"]]
            absent = [w for w in wanted if not (scratch / w).is_file()]
            runs.append({"seed": seed, "set": s["name"], "seconds": round(took, 3),
                         "rc": res.get("rc"), "timeout": bool(res.get("timeout")), "missing": absent})
            if absent or _failed(res):
                why = (res.get("error") or ("timed out" if res.get("timeout") else f"exit code {res.get('rc')}")
                       if _failed(res) else f"wrote no {absent}")
                errors.append(f"predict.sh seed {seed} set {s['name']}: {str(why)[:300]}; "
                              f"stderr: {_tail(res.get('stderr', ''), 600)}")
            for w in wanted:
                if (scratch / w).is_file():
                    os.replace(scratch / w, out_dir / w)
            shutil.rmtree(scratch, ignore_errors=True)
            seconds[s["name"]] = took
        timing = spec.get("timing")
        if timing:
            doc: Dict[str, Any] = {k: seconds[v] for k, v in timing["seconds"].items()}
            for k in (timing.get("text") or {}):
                doc[k] = hw
            (out_dir / file_name(timing["file"], seed)).write_text(json.dumps(doc, indent=2))
    return {"ok": not errors, "errors": errors, "runs": runs, "hardware": hw}


def screen_seeds(spec: Dict[str, Any]) -> List[int]:
    """The seeds a candidate is first scored on during the search: the first
    CFD_SCIENTIST_SCREEN_SEEDS (default 3) of the study's seeds. The rest are
    trained only for a candidate whose score on these comes near the best
    one's; ten seeds trained one after another took 6-9 h per candidate
    (dlr_airfoil_codex_20261009)."""
    seeds = [int(s) for s in spec.get("seeds") or []]
    try:
        n = int(os.environ.get("CFD_SCIENTIST_SCREEN_SEEDS") or 3)
    except ValueError:
        n = 3
    return seeds[:n] if 0 < n < len(seeds) else seeds


def with_seeds(spec: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    return {**spec, "seeds": [int(s) for s in seeds]}


def visible_gpu_memory() -> Dict[str, Tuple[float, float]]:
    """{nvidia-smi index: (total MB, used MB)} for the GPUs this process may
    use (CUDA_VISIBLE_DEVICES, read as nvidia-smi's numbering); empty without
    a GPU, or when they are named some other way."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and not visible.strip():
        return {}
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
    except Exception:  # noqa: BLE001
        return {}
    found: Dict[str, Tuple[float, float]] = {}
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            found[parts[0]] = (float(parts[1]), float(parts[2]))
        except (IndexError, ValueError):
            continue
    if visible:
        keep = [v.strip() for v in visible.split(",")]
        found = {k: v for k, v in found.items() if k in keep}
    return found


def gpu_processes() -> Dict[str, int]:
    """{nvidia-smi index: compute processes on that GPU now}, or {} when it
    cannot be read."""
    try:
        listed = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=20).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return {}
    index: Dict[str, str] = {}
    for line in listed.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            index[parts[1]] = parts[0]
    counts = {i: 0 for i in index.values()}
    for line in apps.splitlines():
        if line.strip() in index:
            counts[index[line.strip()]] += 1
    return counts


def least_busy_gpu(gpus: Dict[str, Tuple[float, float]], procs: Optional[Dict[str, int]] = None) -> Optional[str]:
    """Of ``gpus`` (visible_gpu_memory), the one with the fewest processes on
    it now, then the least memory in use; None without a GPU."""
    if not gpus:
        return None
    procs = gpu_processes() if procs is None else procs
    return min(gpus, key=lambda g: (procs.get(g, 0), gpus[g][1], g))


def _seed_slots(before: Dict[str, Tuple[float, float]], peak: Dict[str, float], n: int,
                procs: Optional[Dict[str, int]] = None) -> List[str]:
    """One GPU per seed that may train at once: as many as fit in 80% of each
    visible GPU's free memory, at the memory one seed took, spread across the
    GPUs, the least busy first (``procs``, gpu_processes), at most
    CFD_SCIENTIST_PARALLEL_SEEDS (default 6) and ``n``."""
    if not before or n <= 0:
        return []
    taken = max((peak.get(k, used) - used for k, (_, used) in before.items()), default=0.0)
    per_seed = max(taken, 1024.0)
    try:
        cap = int(os.environ.get("CFD_SCIENTIST_PARALLEL_SEEDS") or 6)
    except ValueError:
        cap = 6
    counts = {k: int(max(total - used, 0.0) * 0.8 // per_seed)
              for k, (total, used) in (visible_gpu_memory() or before).items()}
    slots: List[str] = []
    while len(slots) < min(cap, n) and any(counts.values()):
        for k in sorted(counts, key=lambda g: ((procs or {}).get(g, 0), -counts[g])):
            if counts[k] > 0 and len(slots) < min(cap, n):
                slots.append(k)
                counts[k] -= 1
    return slots


def run_training(*, sandbox: Any, run_dir: Path, spec: Dict[str, Any], models_root: Path,
                 timeout_s: int) -> Dict[str, Any]:
    """train.sh for every seed. The first trains alone while the GPU memory it
    takes is measured; the rest then train side by side, as many at once as
    the visible GPUs' free memory holds (_seed_slots). One after another, ten
    seeds of a small model took 9.4 h on one GPU while it held 1-2 GB of 48
    and a second GPU sat idle (dlr_airfoil_codex_20261009, final test).
    Every seed is given a GPU, the least busy: run with none named, the first
    seed and any trained alone went to GPU 0, so the framework's extra seeds
    for AirfRANS candidates shared it with five other runs while GPUs 1 and 2
    sat idle (airfrans_codex_20261009, 2026-10-19)."""
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    train, _ = scripts(run_dir)
    seeds = [int(s) for s in spec["seeds"]]
    runs: List[Dict[str, Any]] = []
    errors: List[str] = []
    lock = threading.Lock()

    def one(seed: int, gpu: Optional[str] = None) -> None:
        target = Path(models_root) / f"seed{seed}"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        prefix = f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
        started = time.monotonic()
        res = sandbox.run_bash(f"{prefix}bash {train} {seed} {target}", cwd=str(run_dir), timeout=timeout_s)
        took = time.monotonic() - started
        ok = not _failed(res) and any(f.is_file() for f in target.rglob("*"))
        with lock:
            runs.append({"seed": seed, "seconds": round(took, 1), "ok": ok,
                         **({"gpu": gpu} if gpu is not None else {})})
            if not ok:
                why = res.get("error") or ("timed out" if res.get("timeout") else
                                           f"exit code {res.get('rc')}" if res.get("rc") else "wrote nothing")
                errors.append(f"train.sh seed {seed}: {str(why)[:300]}; "
                              f"stderr: {_tail(res.get('stderr', ''), 800)}")

    if not seeds:
        return {"ok": True, "errors": [], "runs": []}
    before = visible_gpu_memory() if len(seeds) > 1 else {}
    peak: Dict[str, float] = {}
    stop = threading.Event()

    def sample() -> None:
        while not stop.wait(5):
            for k, (_, used) in visible_gpu_memory().items():
                peak[k] = max(peak.get(k, 0.0), used)

    sampler = threading.Thread(target=sample, daemon=True) if before else None
    if sampler:
        sampler.start()
    one(seeds[0], least_busy_gpu(before or visible_gpu_memory()))
    if sampler:
        stop.set()
        sampler.join()
    rest = seeds[1:]
    slots = _seed_slots(before, peak, len(rest), gpu_processes() if before else {})
    if len(slots) > 1:
        print(f"[pipeline] training seeds {rest} {len(slots)} at a time on GPU(s) "
              f"{sorted(set(slots))}", flush=True)
        free: "queue.Queue[str]" = queue.Queue()
        for g in slots:
            free.put(g)

        def on_a_gpu(seed: int) -> None:
            gpu = free.get()
            try:
                one(seed, gpu)
            finally:
                free.put(gpu)

        with ThreadPoolExecutor(max_workers=len(slots)) as pool:
            list(pool.map(on_a_gpu, rest))
    else:
        for seed in rest:
            one(seed, least_busy_gpu(visible_gpu_memory()))
    runs.sort(key=lambda r: seeds.index(r["seed"]))
    return {"ok": not errors, "errors": errors, "runs": runs}


# --------------------------------------------------------------------------- #
# Are the seeds independent                                                     #
# --------------------------------------------------------------------------- #

def _arrays(path: Path) -> Optional[Dict[str, Any]]:
    import numpy as np

    try:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        if path.suffix == ".npy":
            return {"": np.load(path, allow_pickle=False)}
        if path.suffix == ".csv":
            import pandas as pd

            frame = pd.read_csv(path).select_dtypes("number")
            return {c: frame[c].to_numpy() for c in frame.columns}
    except Exception:  # noqa: BLE001
        return None
    return None


def difference(a: Path, b: Path) -> Optional[float]:
    """How far apart two seeds' predictions are: the median, over the arrays
    they share, of the RMS difference relative to the array's own spread. None
    when the files cannot be compared as numbers."""
    import numpy as np

    x, y = _arrays(a), _arrays(b)
    if not x or not y:
        return None
    ratios = []
    for k in x:
        if k not in y or x[k].shape != y[k].shape or not np.issubdtype(x[k].dtype, np.number):
            continue
        u, v = x[k].astype(np.float64), y[k].astype(np.float64)
        finite = np.isfinite(u) & np.isfinite(v)
        if not finite.any():
            continue
        spread = float(np.std(u[finite]))
        ratios.append(float(np.sqrt(np.mean((u[finite] - v[finite]) ** 2))) / (spread if spread > 0 else 1.0))
    return float(np.median(ratios)) if ratios else None


def seed_copies(out_dir: Path, spec: Dict[str, Any]) -> List[str]:
    """Pairs of seeds whose prediction files are the same file, the same bytes,
    or numbers less than NEAR_COPY of their spread apart."""
    out_dir = Path(out_dir)
    seeds = list(spec.get("seeds") or [])
    found: List[str] = []
    for s in spec.get("sets") or []:
        for pattern in s["files"]:
            files = {k: out_dir / file_name(pattern, k) for k in seeds}
            for i, a in enumerate(seeds):
                for b in seeds[i + 1:]:
                    fa, fb = files[a], files[b]
                    if not fa.is_file() or not fb.is_file():
                        continue
                    if os.path.samefile(fa, fb):
                        found.append(f"{fa.name} and {fb.name} are the same file")
                    elif fa.stat().st_size == fb.stat().st_size and fa.read_bytes() == fb.read_bytes():
                        found.append(f"{fa.name} and {fb.name} are identical")
                    else:
                        d = difference(fa, fb)
                        if d is not None and d < NEAR_COPY:
                            found.append(f"{fa.name} and {fb.name} differ by only {d:.2e} of their spread")
    return found
