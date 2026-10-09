#!/usr/bin/env python3
"""Run a fitted-model candidate's own pipeline the way the framework scores it.

  python scripts/pipeline_run.py --run-dir <candidate> --starter <starter> \
      --spec <pipeline_spec.json> --manifest <view manifest> --out <dir> --result <json> \
      [--models <dir>] [--train --train-timeout <s>]

Without --train: predict.sh for every seed and evaluation set, on the models the
candidate trained (<candidate>/pipeline/models), each run timed. With --train:
train.sh for every seed first, into --models, with the evaluation sets' files
hidden, as the final re-run of the picked candidate does. Both run in the same
sandbox the candidate was built in, with the view in --manifest (see
cfd_langgraph.validation_split). The timing files the scorer reads are written
from the measured times, and seeds whose predictions are copies of each other
are reported. See cfd_langgraph.pipeline_contract.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_SCRIPTS), str(_SCRIPTS.parent / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cfd_langgraph import pipeline_contract as pc  # noqa: E402
from cfd_langgraph.withheld_data import MANIFEST_ENV, load_manifest  # noqa: E402


def _sandbox(run_dir: Path, starter: Path, manifest: str, max_timeout: int):
    os.environ[MANIFEST_ENV] = manifest
    from code_mod_agentic import Sandbox  # type: ignore

    return Sandbox(run_dir=run_dir, starter_case=starter, wm_project_dir=None, starter_root=starter,
                   max_bash_timeout=max_timeout)


def _training_view(manifest_path: str, folder: Path) -> str:
    """The view with every evaluation set's files hidden, inputs included:
    training never sees the sets it is scored on."""
    doc = load_manifest(manifest_path)
    entries = []
    for e in doc.get("entries") or []:
        if e.get("action") == "replace":
            e = {k: v for k, v in e.items() if k not in ("source", "kept", "withheld", "masked_columns")}
            e.update(action="hide", is_dir=False, reason="an evaluation set, hidden while training")
        entries.append(e)
    path = folder / "training_view.json"
    path.write_text(json.dumps({**doc, "entries": entries}, indent=2))
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--starter", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--result", required=True)
    ap.add_argument("--models", default="")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--train-timeout", type=int, default=6 * 3600)
    ap.add_argument("--predict-timeout", type=int, default=2 * 3600)
    a = ap.parse_args()

    run_dir, starter, out = Path(a.run_dir).resolve(), Path(a.starter).resolve(), Path(a.out).resolve()
    spec = pc.load_spec(Path(a.spec))
    models = Path(a.models).resolve() if a.models else run_dir / pc.PIPELINE_DIR / pc.MODELS_DIR
    result: dict = {"run_dir": str(run_dir), "manifest": a.manifest, "out": str(out), "models": str(models)}
    if not spec:
        result.update(ok=False, errors=[f"no pipeline spec at {a.spec}"])
    else:
        missing = [m for m in pc.missing_parts(run_dir, spec) if not a.train or m.endswith(".sh")]
        if missing:
            result.update(ok=False, errors=[f"missing: {', '.join(missing)}"])
        else:
            ok = True
            if a.train:
                with tempfile.TemporaryDirectory() as td:
                    view = _training_view(a.manifest, Path(td))
                    sb = _sandbox(run_dir, starter, view, a.train_timeout)
                    result["training"] = pc.run_training(sandbox=sb, run_dir=run_dir, spec=spec,
                                                         models_root=models, timeout_s=a.train_timeout)
                ok = result["training"]["ok"]
            if ok:
                sb = _sandbox(run_dir, starter, a.manifest, a.predict_timeout)
                result["prediction"] = pc.run_predictions(sandbox=sb, run_dir=run_dir, spec=spec,
                                                          models_root=models, out_dir=out,
                                                          timeout_s=a.predict_timeout)
                ok = result["prediction"]["ok"]
            copies = pc.seed_copies(out, spec) if ok else []
            result["seed_copies"] = copies
            errors = (result.get("training") or {}).get("errors", []) + \
                (result.get("prediction") or {}).get("errors", [])
            if copies:
                errors.append("seeds are not independent: " + "; ".join(copies[:6]))
            result.update(ok=ok and not copies, errors=errors)
    Path(a.result).parent.mkdir(parents=True, exist_ok=True)
    Path(a.result).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"ok": result.get("ok"), "errors": result.get("errors", [])[:5]}, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
