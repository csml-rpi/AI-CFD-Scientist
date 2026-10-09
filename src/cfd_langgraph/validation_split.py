"""The validation split a data-fitted study's search is scored on.

The test set is scored once, on the candidate the search finally picks. Until
then candidates are trained on part of the training data and scored on the rest.
Withholding the test targets from the build agents (cfd_langgraph.withheld_data)
was not enough on its own: the framework still scored every candidate on the
test set and wrote the result where the next candidates could read it, so test
scores chose what was tried next and which candidate won.

Once per study an LLM writes a script that, from the starter folder's own
documentation and files, writes a changed copy of each data file whose content
must change: the evaluation sets the scorer reads now hold validation stand-ins
drawn from the training samples, and the training files hold the rest. The
copy is laid over the starter's originals at their own paths, so the study's
scorer, its file formats and the candidates' code all stay as they are. Code
checks the copy and runs the study's scorer on a reference result for the
stand-ins before anything uses it.

Three views come out of it, each a manifest in the withheld_data format:
  scorer    the copy over the originals, targets included: how the framework
            scores a candidate during the search.
  candidate the same, with the stand-ins' targets withheld the way the test
            set's are, and the framework's own files hidden: what a build agent
            sees during the search (validation_view/manifest.json).
  final     the real starter with the test targets withheld and every training
            sample visible: how the picked candidate is re-run at the end.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

from cfd_langgraph.withheld_data import (
    VALIDATION_VIEW_DIR,
    data_listing,
    documentation,
    input_only_copy,
    load_manifest,
)

SPLIT_FILE = "validation_split.json"
PRIVATE_DIR = "framework_private"
MAX_ATTEMPTS = 4
_BUILD_TIMEOUT_S = 3600
_SCORER_SOURCE_CHARS = 14000

BUILDER_SYSTEM = (
    "You write careful Python data-preparation scripts. Output ONLY raw Python code. "
    "No markdown, no code fences, no commentary."
)

BUILDER_USER = """Write a Python script that prepares a validation version of a \
machine-learning study's data, so that candidate models can be scored during a search \
without the real test set being used at all.

The starter folder ({starter}) holds the study's training data, its test data (and possibly \
further held-out evaluation sets, an out-of-distribution set say), the study's own scorer and \
its documentation. The scorer reads its data from fixed places inside the starter folder.

CONTRACT
  python <script> --starter <S> --out <V> --baseline-out <B>

1. Choose validation samples from the TRAINING samples only, with a fixed random seed. For
   every held-out evaluation set the scorer reads (the test set, and each further one),
   choose a group of training samples to stand in for it. Where the documentation says how
   an evaluation set differs from the training set (a parameter outside the training range,
   say), choose its stand-ins so that they differ from the remaining training samples in the
   same way (the training samples at the extremes of that parameter, say); otherwise choose
   them at random. Hold out roughly a quarter of the training samples in total, and at least
   {min_per_set} per evaluation set where the training set allows. The groups may overlap
   each other only where the real evaluation sets overlap.
2. Write into V a changed copy of every starter file whose content must change, at the same
   path relative to the starter folder as the original and in exactly the original's format
   (same file type, keys, columns, array layout and dtypes):
   - every file that defines or holds an evaluation set (its list of samples, its inputs,
     its targets) now holds that set's validation stand-ins instead, targets included;
   - every file that defines or holds the training set now holds only the remaining
     training samples;
   - a statistics file the scorer reads may stay as it is.
   Write nothing else into V and leave out files that do not change. With these copies in
   place of the originals, the scorer must score predictions for the stand-ins exactly as it
   scores predictions for the test set.
3. No real test sample, and no sample of any other held-out evaluation set, may appear
   anywhere in V.
4. Write V/{split_file}:
   {{"train": [ids of the remaining training samples],
    "validation": {{"<evaluation set name>": [ids of its stand-ins], ...}},
    "hide": [paths, relative to the starter folder, of files or folders that are NOT
             replaced in V but hold data of a stand-in (its own raw-data folder or file,
             say)],
    "notes": "how the stand-ins were chosen, in two or three sentences"}}
   Ids are the names or numbers the starter's own files use for samples, as strings.
5. Write into B a reference result for the stand-ins, with exactly the file names and format
   of the supplied baseline result listed below, that predicts every target quantity of every
   stand-in as its mean over the remaining training samples. If the baseline result holds a
   file reporting measured run times, write it with the seconds your script spent producing
   each set. Do not copy the baseline result's other files (a summary, notes).

Use only the Python standard library, numpy and pandas; read the data files directly. Do not
import anything from the starter folder and never write inside it. Print a short summary at
the end. Exit non-zero on any error.

STARTER DOCUMENTATION:
{docs}

DATA FILES (archive keys and shapes, table columns):
{listing}

JSON FILES (structure):
{json_peek}

WHICH STARTER FILES HOLD TEST TARGETS (already withheld from candidates as follows):
{test_manifest}

THE STUDY'S SCORER ({scorer_path}):
{scorer_source}

THE SUPPLIED BASELINE RESULT ({baseline_dir}):
{baseline_listing}
{feedback}"""


# --------------------------------------------------------------------------- #
# What the builder is shown                                                     #
# --------------------------------------------------------------------------- #

def _json_peek(root: Path, limit_files: int = 20, max_bytes: int = 5_000_000) -> str:
    lines: List[str] = []
    for p in sorted(Path(root).rglob("*.json")):
        if len(lines) >= limit_files:
            break
        try:
            if p.stat().st_size > max_bytes:
                lines.append(f"{p.relative_to(root)}: {p.stat().st_size} bytes (not opened)")
                continue
            doc = json.loads(p.read_text(errors="replace"))
        except (OSError, ValueError):
            continue
        lines.append(f"{p.relative_to(root)}: {_describe(doc)}")
    return "\n".join(lines) or "(none)"


def _describe(doc: Any, depth: int = 0) -> str:
    if isinstance(doc, dict):
        if depth >= 2:
            return f"object with {len(doc)} keys"
        items = list(doc.items())[:12]
        inner = ", ".join(f"{k!r}: {_describe(v, depth + 1)}" for k, v in items)
        return "{" + inner + (", ..." if len(doc) > 12 else "") + "}"
    if isinstance(doc, list):
        first = _describe(doc[0], depth + 1) if doc else ""
        return f"list of {len(doc)}" + (f", first: {first}" if first else "")
    return repr(doc)[:80]


def _manifest_summary(manifest: Dict[str, Any]) -> str:
    root = Path(manifest.get("starter_root", "/"))
    lines = []
    for e in manifest.get("entries") or []:
        try:
            rel = Path(e["path"]).relative_to(root)
        except ValueError:
            rel = Path(e["path"])
        if e.get("action") == "replace":
            lines.append(f"  {rel}: candidates see only {e.get('kept')}; withheld {e.get('withheld')}")
        elif e.get("action") == "hide":
            lines.append(f"  {rel}: hidden from candidates")
    return "\n".join(lines) or "  (none)"


def _baseline_listing(result_dir: Path) -> str:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from surrogate_setup import _result_listing  # type: ignore

        return _result_listing(result_dir)
    except Exception as exc:  # noqa: BLE001
        return f"(could not list {result_dir}: {exc})"


def builder_prompt(*, starter: Path, scorer_path: Path, baseline_dir: Path,
                   test_manifest: Dict[str, Any], feedback: str = "", min_per_set: int = 10) -> str:
    try:
        scorer_source = Path(scorer_path).read_text(errors="replace")[:_SCORER_SOURCE_CHARS]
    except OSError as exc:
        scorer_source = f"(could not read: {exc})"
    return BUILDER_USER.format(
        starter=starter, min_per_set=min_per_set, split_file=SPLIT_FILE,
        docs=documentation(starter, limit=30000, per_doc=16000),
        listing=data_listing(starter), json_peek=_json_peek(starter),
        test_manifest=_manifest_summary(test_manifest),
        scorer_path=scorer_path, scorer_source=scorer_source,
        baseline_dir=baseline_dir, baseline_listing=_baseline_listing(baseline_dir),
        feedback=feedback,
    )


# --------------------------------------------------------------------------- #
# Checking what it wrote                                                        #
# --------------------------------------------------------------------------- #

def _ids(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(v) for v in values if str(v).strip()]


def read_split(copy_dir: Path) -> Dict[str, Any]:
    try:
        doc = json.loads((Path(copy_dir) / SPLIT_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def split_ids(split: Dict[str, Any]) -> Tuple[Set[str], Dict[str, Set[str]]]:
    train = set(_ids(split.get("train")))
    val_raw = split.get("validation")
    val = {str(k): set(_ids(v)) for k, v in val_raw.items()} if isinstance(val_raw, dict) else {}
    return train, val


def copied_files(copy_dir: Path) -> List[Path]:
    """The copy's files, relative to it, except the split record."""
    copy_dir = Path(copy_dir)
    return sorted(p.relative_to(copy_dir) for p in copy_dir.rglob("*")
                  if p.is_file() and p != copy_dir / SPLIT_FILE)


def _under(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == r or r in path.parents for r in roots)


def _npz_keys(path: Path) -> List[str]:
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as z:
            return list(z.files)
    except Exception:  # noqa: BLE001
        return []


def _csv_id_rows(path: Path, all_ids: Set[str], wanted: Set[str]) -> List[str]:
    """Ids from ``wanted`` that name a row of the table: the id column is one
    whose values cover at least half of every declared id."""
    try:
        with path.open(newline="") as f:
            rows = list(csv.reader(f))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []
    if len(rows) < 2:
        return []
    header, body = rows[0], rows[1:]
    for i in range(len(header)):
        column = {r[i].strip() for r in body if i < len(r)}
        if all_ids and len(column & all_ids) >= 0.5 * len(all_ids):
            return sorted(column & wanted)
    return []


def named_paths(root: Path, ids: Set[str], depth: int = 4) -> List[Path]:
    """Files and folders inside ``root`` named by one of ``ids`` (by name or stem)."""
    out: List[Path] = []
    stack: List[Tuple[Path, int]] = [(Path(root), 0)]
    while stack:
        folder, level = stack.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for e in entries:
            p = Path(e.path)
            if e.name in ids or p.stem in ids:
                out.append(p)
                continue
            if e.is_dir(follow_symlinks=False) and level + 1 < depth:
                stack.append((p, level + 1))
    return sorted(out)


def check_copy(starter: Path, copy_dir: Path, test_manifest: Dict[str, Any]) -> List[str]:
    """Everything wrong with a copy, in words the builder can act on. Empty when
    the copy may be used."""
    import numpy as np

    starter, copy_dir = Path(starter).resolve(), Path(copy_dir)
    problems: List[str] = []
    split = read_split(copy_dir)
    if not split:
        return [f"{SPLIT_FILE} is missing from --out or is not a JSON object."]
    train, val = split_ids(split)
    all_val = set().union(*val.values()) if val else set()
    if not train:
        problems.append(f"{SPLIT_FILE} lists no remaining training samples under 'train'.")
    if not val or not all(val.values()):
        problems.append(f"{SPLIT_FILE} 'validation' must map each evaluation set to a non-empty list of ids.")
    both = sorted(train & all_val)
    if both:
        problems.append(f"{len(both)} ids are both training and validation samples, e.g. {both[:5]}.")

    files = copied_files(copy_dir)
    if not files:
        problems.append("The script wrote no changed starter file into --out.")
    for rel in files:
        if not (starter / rel).is_file():
            problems.append(f"{rel} in --out is not a file of the starter folder; --out may hold only "
                            "changed versions of existing starter files, at their own relative paths.")
    replaced = {starter / rel for rel in files}

    for e in test_manifest.get("entries") or []:
        held = Path(e["path"])
        if e.get("action") != "replace":
            continue
        rel = held.relative_to(starter)
        mine = copy_dir / rel
        if not mine.is_file():
            problems.append(f"{rel} holds the test set's targets, but --out has no changed version of it, "
                            "so the scorer would score the real test set.")
            continue
        if held.suffix != ".npz":
            continue
        ours, theirs = _npz_keys(mine), _npz_keys(held)
        if ours and set(ours) <= (train | all_val):
            # One array per sample: the copy must hold none of the real ones.
            common = sorted(set(ours) & set(theirs))
            if common:
                problems.append(f"{rel} in --out still holds real evaluation samples, e.g. {common[:5]}.")
            continue
        cols = e.get("masked_columns") or []
        names = [k for k in ours if k in theirs] if cols else \
            [k for k in (e.get("withheld") or []) if k in ours and k in theirs]
        with np.load(mine, allow_pickle=False) as a, np.load(held, allow_pickle=False) as b:
            for k in names:
                x, y = a[k], b[k]
                if cols and x.ndim and x.shape == y.shape and x.shape[-1] > max(cols):
                    x, y = x[..., cols], y[..., cols]
                if x.shape == y.shape and np.array_equal(x, y):
                    problems.append(f"{rel} in --out: '{k}' is identical to the real test targets.")
                    break

    hidden_roots = [Path(e["path"]) for e in test_manifest.get("entries") or [] if e.get("action") == "hide"]
    hidden_roots += [(starter / str(h).lstrip("/")).resolve() for h in _ids(split.get("hide"))]
    leaks: List[str] = []
    for p in sorted(starter.rglob("*")):
        if not p.is_file() or p in replaced or _under(p, hidden_roots):
            continue
        if p.suffix == ".npz":
            hit = sorted(set(_npz_keys(p)) & all_val)
        elif p.suffix == ".csv":
            hit = _csv_id_rows(p, train | all_val, all_val)
        else:
            continue
        if hit:
            leaks.append(f"{p.relative_to(starter)} (e.g. {hit[:3]})")
        if len(leaks) >= 8:
            break
    if leaks:
        problems.append("These starter files still hold validation samples and are neither replaced in "
                        "--out nor listed under 'hide': " + "; ".join(leaks))
    return problems


# --------------------------------------------------------------------------- #
# The three views                                                               #
# --------------------------------------------------------------------------- #

def _reveals(hides: List[Path], allowed: Set[str]) -> List[Dict[str, Any]]:
    """Inside each hidden folder, the children named by an allowed sample id."""
    out = []
    for folder in hides:
        if not folder.is_dir():
            continue
        try:
            children = sorted(Path(c.path) for c in os.scandir(folder))
        except OSError:
            continue
        for c in children:
            if c.name in allowed or c.stem in allowed:
                out.append({"path": str(c), "action": "reveal", "is_dir": c.is_dir(),
                            "reason": "a training sample's own data"})
    return out


def build_views(*, starter: Path, copy_dir: Path, view_dir: Path, private_dir: Path,
                test_manifest: Dict[str, Any], empty_file: str) -> Dict[str, Dict[str, Any]]:
    """Write the scorer, candidate and final manifests into ``view_dir``."""
    starter, copy_dir, view_dir = Path(starter).resolve(), Path(copy_dir), Path(view_dir)
    view_dir.mkdir(parents=True, exist_ok=True)
    split = read_split(copy_dir)
    train, val = split_ids(split)
    all_val = set().union(*val.values()) if val else set()
    test_entries = {Path(e["path"]): e for e in test_manifest.get("entries") or []}
    files = copied_files(copy_dir)

    scorer = {"starter_root": str(starter), "empty_file": empty_file, "entries": [
        {"path": str(starter / rel), "action": "replace", "source": str(copy_dir / rel),
         "reason": "validation copy"} for rel in files]}

    entries: List[Dict[str, Any]] = []
    for rel in files:
        original = starter / rel
        test_entry = test_entries.get(original)
        if test_entry is None:
            entries.append({"path": str(original), "action": "replace", "source": str(copy_dir / rel),
                            "reason": "validation copy: remaining training samples"})
            continue
        if test_entry.get("action") == "replace":
            dest = view_dir / "inputs" / rel
            kept, withheld = input_only_copy(
                copy_dir / rel, dest,
                keep=None if test_entry.get("masked_columns") else test_entry.get("kept"),
                mask=test_entry.get("masked_columns"))
            if kept:
                entries.append({"path": str(original), "action": "replace", "source": str(dest),
                                "kept": kept, "withheld": withheld,
                                "reason": "validation stand-ins, inputs only"})
                continue
        entries.append({"path": str(original), "action": "hide", "is_dir": False,
                        "reason": "validation stand-ins' targets"})
    replaced = {Path(e["path"]) for e in entries}
    entries += [dict(e) for p, e in test_entries.items() if p not in replaced and e.get("action") == "hide"]
    extra = [(starter / str(h).lstrip("/")).resolve() for h in _ids(split.get("hide"))]
    extra += named_paths(starter, all_val)
    covered = [Path(e["path"]) for e in entries if e.get("action") == "hide" and e.get("is_dir")]
    for p in sorted(set(extra)):
        if p.exists() and str(p).startswith(str(starter)) and p not in replaced and not _under(p, covered):
            entries.append({"path": str(p), "action": "hide", "is_dir": p.is_dir(),
                            "reason": "a validation stand-in's own data"})
    unique: Dict[str, Dict[str, Any]] = {}
    for e in entries:
        unique.setdefault(e["path"], e)
    entries = list(unique.values())
    private = {"path": str(Path(private_dir).resolve()), "action": "hide", "is_dir": True,
               "reason": "the framework's validation targets and scores"}
    hidden_dirs = [Path(e["path"]) for e in entries if e.get("action") == "hide" and e.get("is_dir")]
    candidate = {"starter_root": str(starter), "empty_file": empty_file,
                 "entries": entries + [private] + _reveals(hidden_dirs, train)}

    test_hidden_dirs = [Path(e["path"]) for e in test_entries.values()
                        if e.get("action") == "hide" and e.get("is_dir")]
    final = {"starter_root": str(starter), "empty_file": empty_file,
             "entries": [dict(e) for e in test_entries.values()] + [private]
             + _reveals(test_hidden_dirs, train | all_val)}

    for name, doc in (("scorer_manifest.json", scorer), ("manifest.json", candidate),
                      ("final_manifest.json", final)):
        (view_dir / name).write_text(json.dumps(doc, indent=2))
    shutil.copy2(copy_dir / SPLIT_FILE, view_dir / SPLIT_FILE)
    return {"scorer": scorer, "candidate": candidate, "final": final}


# --------------------------------------------------------------------------- #
# Once per study                                                                #
# --------------------------------------------------------------------------- #

def paths(study_dir: Path) -> Dict[str, Path]:
    study_dir = Path(study_dir)
    private = study_dir / PRIVATE_DIR
    view = study_dir / VALIDATION_VIEW_DIR
    return {
        "private": private,
        "copy": private / "validation_copy",
        "baseline": private / "validation_baseline",
        "builder": private / "validation_builder",
        "scores": private / "validation_scores",
        "view": view,
        "scorer_manifest": view / "scorer_manifest.json",
        "candidate_manifest": view / "manifest.json",
        "final_manifest": view / "final_manifest.json",
        "status": view / "status.json",
    }


def ready(study_dir: Path) -> bool:
    status = load_manifest(str(paths(study_dir)["status"]))
    return bool(status.get("ok")) and paths(study_dir)["scorer_manifest"].is_file()


def build(
    *,
    study_dir: Path,
    starter: Path,
    scorer_path: Path,
    baseline_dir: Path,
    test_manifest_path: Path,
    llm_invoke: Callable[[List[Tuple[str, str]]], str],
    smoke_test: Callable[[Path, str], Tuple[bool, str]],
    max_attempts: int = MAX_ATTEMPTS,
    strip_fences: Callable[[str], str] = lambda s: s,
) -> Dict[str, Any]:
    """Author, run and check the builder until a copy passes, then write the views.

    ``smoke_test(case_dir, scorer_manifest)`` scores the reference result in
    ``case_dir`` against the copy and says whether the scorer produced a number.
    """
    p = paths(study_dir)
    if ready(study_dir):
        return load_manifest(str(p["status"]))
    test_manifest = load_manifest(str(test_manifest_path))
    p["builder"].mkdir(parents=True, exist_ok=True)
    feedback = ""
    log: List[Dict[str, Any]] = []
    status: Dict[str, Any] = {"ok": False}
    for attempt in range(1, max_attempts + 1):
        prompt = builder_prompt(starter=starter, scorer_path=scorer_path, baseline_dir=baseline_dir,
                                test_manifest=test_manifest, feedback=feedback)
        try:
            code = strip_fences(llm_invoke([("system", BUILDER_SYSTEM), ("user", prompt)]))
        except Exception as exc:  # noqa: BLE001
            log.append({"attempt": attempt, "error": f"authoring failed: {exc}"})
            continue
        script = p["builder"] / f"build_validation_attempt{attempt}.py"
        script.write_text(code)
        for d in (p["copy"], p["baseline"]):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        try:
            proc = subprocess.run(
                [sys.executable, str(script), "--starter", str(starter), "--out", str(p["copy"]),
                 "--baseline-out", str(p["baseline"])],
                capture_output=True, text=True, timeout=_BUILD_TIMEOUT_S, cwd=str(p["builder"]))
            output = (proc.stdout or "")[-3000:] + "\n" + (proc.stderr or "")[-3000:]
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            output, rc = f"timed out after {_BUILD_TIMEOUT_S}s", -1
        problems = [f"the script exited with code {rc}"] if rc != 0 else \
            check_copy(starter, p["copy"], test_manifest)
        if not problems and not any(f.is_file() for f in p["baseline"].rglob("*")):
            problems = ["the script wrote no reference result into --baseline-out."]
        smoke = ""
        if not problems:
            build_views(starter=starter, copy_dir=p["copy"], view_dir=p["view"],
                        private_dir=p["private"], test_manifest=test_manifest,
                        empty_file=str(test_manifest.get("empty_file") or "/dev/null"))
            ok, smoke = smoke_test(p["baseline"], str(p["scorer_manifest"]))
            if not ok:
                problems = ["the study's scorer, run on --baseline-out with --out laid over the "
                            f"starter's files, did not produce a score: {smoke}"]
                for name in ("scorer_manifest", "candidate_manifest", "final_manifest"):
                    p[name].unlink(missing_ok=True)
        log.append({"attempt": attempt, "script": str(script), "rc": rc, "problems": problems,
                    "smoke_test": smoke, "output_tail": output[-2000:]})
        print(f"[validation] builder attempt {attempt}: "
              + ("ok" if not problems else "; ".join(problems)[:600]), flush=True)
        if not problems:
            split = read_split(p["copy"])
            train, val = split_ids(split)
            status = {"ok": True, "attempt": attempt, "script": str(script),
                      "train_count": len(train), "validation_counts": {k: len(v) for k, v in val.items()},
                      "notes": str(split.get("notes", ""))[:1000], "smoke_test": smoke,
                      "baseline_dir": str(p["baseline"])}
            break
        feedback = (
            "\n\nYOUR PREVIOUS SCRIPT DID NOT PASS.\nProblems:\n- " + "\n- ".join(problems)
            + f"\n\nWhat it printed (tail):\n{output[-3000:]}\n\nThe script:\n{code[:14000]}\n\n"
            "Fix the cause. Return the complete corrected script."
        )
    status["attempts"] = log
    p["view"].mkdir(parents=True, exist_ok=True)
    p["status"].write_text(json.dumps(status, indent=2, default=str))
    return status
