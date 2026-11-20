"""The checks that hold OpenFOAM to a model card. A network used inside a case
counts only once both have passed:

check_parity  OpenFOAM's build of the network (nnModel.H, compiled with
              OpenFOAM's headers) reproduces the outputs stored in the card
              for its stored raw inputs: the scaling, bounds, feature order
              and precision are the trained ones.
check_case    in the case itself, the features the case computed and the
              output it used, both written as fields, agree with the card's
              network evaluated in Python on those same features: the case
              feeds the network what it computes, in the card's order. It also
              says how many cells lie outside the training range.

What neither can check is whether a feature is computed the way the card's
definition says; the case's code must be read against the definition.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .card import ModelCard
from .export import evaluate, sha256
from .openfoam import build_parity_app, openfoam_env, run_parity_app

# Agreement asked of two evaluations of one network in double precision.
TOLERANCE = 1e-9


def _within(got: np.ndarray, want: np.ndarray) -> tuple:
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    if got.shape != want.shape:
        return False, float("inf")
    diff = float(np.abs(got - want).max()) if got.size else 0.0
    return diff <= TOLERANCE * max(1.0, float(np.abs(want).max()) if want.size else 1.0), diff


def check_parity(card_path: Path, openfoam_path: str = "", build_root: Optional[Path] = None) -> Dict:
    card_path = Path(card_path)
    card = ModelCard.load(card_path)
    problems = card.problems()
    model = card_path.parent / card.torchscript
    if not model.is_file():
        problems.append(f"{model} is missing")
    elif card.sha256 and sha256(model) != card.sha256:
        problems.append(f"{model.name} is not the file this card was written for (its sha256 differs)")
    if len(card.parity_inputs) != len(card.parity_outputs) or not card.parity_inputs:
        problems.append("the card holds no parity set: export the network with nn_export first")
    if problems:
        return {"ok": False, "problems": problems}
    env = openfoam_env(openfoam_path)
    if not env.get("WM_PROJECT_DIR"):
        return {"ok": False, "problems": ["no OpenFOAM is loaded: pass openfoam_path"]}
    want = np.asarray(card.parity_outputs, dtype=float)
    python_ok, python_diff = _within(evaluate(card_path, card.parity_inputs, card), want)
    try:
        exe = build_parity_app(env, Path(build_root or Path.home() / ".cfd_scientist" / "nn_coupling"))
        got = run_parity_app(exe, card_path, env)
    except RuntimeError as exc:
        return {"ok": False, "problems": [str(exc)]}
    ok, diff = _within(got, want)
    return {
        "ok": ok and python_ok,
        "openfoam": env.get("WM_PROJECT_DIR"),
        "rows": len(want),
        "max_difference_openfoam": diff,
        "max_difference_python": python_diff,
        "tolerance": TOLERANCE,
        "note": ("OpenFOAM's build of the network reproduces the card's stored outputs."
                 if ok and python_ok else
                 "OpenFOAM's build does NOT reproduce the card's stored outputs: do not use it."),
    }


def _read_text(path: Path) -> str:
    if path.is_file():
        return path.read_text(errors="ignore")
    gz = path.with_name(path.name + ".gz")
    if gz.is_file():
        with gzip.open(gz, "rt", errors="ignore") as f:
            return f.read()
    raise FileNotFoundError(f"{path} (nor {gz.name})")


def read_internal_field(path: Path, n_cells: Optional[int] = None) -> np.ndarray:
    """A volScalarField's or volVectorField's internal values from an ASCII
    field file (plain or .gz): one value, or one row of three, per cell."""
    text = _read_text(Path(path))
    header = text[:text.find("}") + 1] if "}" in text else ""
    if "binary" in header:
        raise ValueError(f"{path} is written in binary: set writeFormat ascii")
    start = text.find("internalField")
    if start < 0:
        raise ValueError(f"{path} has no internalField")
    rest = text[start + len("internalField"):].lstrip()
    is_vector = "vector" in rest[:40] or rest.startswith("uniform (")
    if rest.startswith("uniform"):
        value = rest[len("uniform"):].split(";", 1)[0].strip().strip("()").split()
        if n_cells is None:
            raise ValueError(f"{path} is uniform: the cell count is needed")
        row = np.array([float(v) for v in value])
        return np.tile(row, (n_cells, 1)) if row.size > 1 else np.full(n_cells, row[0])
    after = rest.split("nonuniform", 1)[1].split(None, 1)[1]  # past List<scalar> / List<vector>
    count, body = after.split("(", 1)
    n = int(count.split()[-1])
    width = 3 if is_vector else 1
    tokens = body.replace("(", " ").replace(")", " ").split()[:n * width]
    values = np.array([float(t) for t in tokens])
    return values.reshape(n, 3) if is_vector else values


def _is_time(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def _latest_time(root: Path) -> str:
    times = sorted((p.name for p in root.iterdir() if p.is_dir() and _is_time(p.name)), key=float)
    if not times:
        raise FileNotFoundError(f"no time folder in {root}")
    return times[-1]


def read_case_field(case_dir: Path, time: str, name: str) -> np.ndarray:
    """``name``'s cell values at ``time``: from the case, or, while the case is
    decomposed, from its processors in order (each field the same way, so the
    cells of different fields still line up)."""
    case_dir = Path(case_dir)
    if (case_dir / time).is_dir() and any((case_dir / time / n).is_file() for n in (name, name + ".gz")):
        roots = [case_dir]
    else:
        roots = sorted((p for p in case_dir.glob("processor*") if p.name[9:].isdigit()),
                       key=lambda p: int(p.name[9:]))
    parts: List[np.ndarray] = []
    for root in roots:
        cells = None
        owner = root / "constant" / "polyMesh" / "owner"
        if owner.is_file() or owner.with_name("owner.gz").is_file():
            note = _read_text(owner)[:2000]
            mark = note.find("nCells:")
            if mark >= 0:
                cells = int(note[mark + len("nCells:"):].split()[0])
        parts.append(read_internal_field(root / time / name, cells))
    if not parts:
        raise FileNotFoundError(f"{name} at time {time} in {case_dir}")
    return np.concatenate(parts)


def _write_precision(case_dir: Path) -> Optional[int]:
    """The case's controlDict writePrecision, or None when it sets none."""
    control = Path(case_dir) / "system" / "controlDict"
    if not control.is_file():
        return None
    for line in control.read_text(errors="ignore").splitlines():
        words = line.replace(";", " ").split()
        if len(words) >= 2 and words[0] == "writePrecision":
            try:
                return int(words[1])
            except ValueError:
                return None
    return None


# Significant digits a case must write its fields with for check_case: the
# network ran on full-precision features, and read back from a cavity written
# to 6 digits its output differed from Python's by 8e-8 although the case was
# right (test_nn_coupling, 2026-10-22).
MIN_WRITE_PRECISION = 15


def check_case(case_dir: Path, card_path: Path, time: str = "", prefix: str = "nn_",
               output_field: str = "", features_normalised: bool = False) -> Dict:
    """The case's written features (fields ``prefix`` + each input name) and
    output field against the card's network evaluated in Python on them.
    ``features_normalised``: the written features are already shifted and
    scaled by the card's input normalisation, as DAFoam writes them
    (regressionModel writeFeatures, fields named after the inputs)."""
    case_dir, card_path = Path(case_dir), Path(card_path)
    card = ModelCard.load(card_path)
    precision = _write_precision(case_dir)
    if precision is None or precision < MIN_WRITE_PRECISION:
        return {"ok": False, "problems": [
            f"the case writes its fields with {precision or 'the default 6'} significant digits; the network "
            f"ran on full-precision values, so the check cannot compare. Set writePrecision "
            f"{MIN_WRITE_PRECISION + 2} (and writeFormat ascii) in system/controlDict and run it again."]}
    root = case_dir if any(p.is_dir() and _is_time(p.name) and p.name != "0" for p in case_dir.iterdir()) \
        else case_dir / "processor0"
    time = time or _latest_time(root)
    names = [prefix + f.name for f in card.inputs]
    try:
        columns = [read_case_field(case_dir, time, n) for n in names]
        output = read_case_field(case_dir, time, output_field or card.output)
    except (FileNotFoundError, ValueError) as exc:
        return {"ok": False, "time": time, "problems": [
            f"{exc}. The case must write each feature it gives the network as a field named "
            f"{names} and the network's output as {output_field or card.output!r}, at the same time."]}
    raw = np.column_stack(columns)
    if features_normalised:
        raw = raw / np.asarray(card.input_scale) - np.asarray(card.input_shift)
    python = evaluate(card_path, raw, card)
    ok, diff = _within(output, python)
    report = {
        "ok": ok,
        "time": time,
        "cells": int(raw.shape[0]),
        "max_difference": diff,
        "tolerance": TOLERANCE,
        "feature_range": {f.name: [float(raw[:, j].min()), float(raw[:, j].max())]
                          for j, f in enumerate(card.inputs)},
        "output_range": [float(output.min()), float(output.max())],
    }
    if card.training_range:
        lo = np.array([r[0] for r in card.training_range])
        hi = np.array([r[1] for r in card.training_range])
        outside = ((raw < lo) | (raw > hi)).any(axis=1)
        report["cells_outside_training_range"] = float(outside.mean())
    report["note"] = (
        "The case feeds the network its features in the card's order and uses its output as the "
        "card defines it." if ok else
        "The case's output is NOT the card's network on the case's own features: the feature order, "
        "normalisation, output scaling or the field used differs from the card. Do not use the result.")
    return report
