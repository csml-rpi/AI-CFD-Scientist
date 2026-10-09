"""The validation split a data-fitted study's search is scored on, end to end on
a small synthetic study: the copy and its checks, the scorer, candidate and final
views in the sandbox, the framework running a candidate's predict.sh and train.sh,
its timing files, and the seed-copy check.

Run: python scripts/test_validation_split.py   (needs bubblewrap)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cfd_langgraph import pipeline_contract as pc  # noqa: E402
from cfd_langgraph import validation_split as vs  # noqa: E402
from cfd_langgraph.withheld_data import MANIFEST_ENV, build_view, find_manifest  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: object, detail: object = "") -> None:
    if cond:
        print(f"[PASS] {name}")
    else:
        FAILURES.append(name)
        print(f"[FAIL] {name}" + (f" — {str(detail)[:1500]}" if detail != "" else ""))


SCORER = '''
import argparse, glob, json, os, re, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
ap = argparse.ArgumentParser(); ap.add_argument("--pred-dir"); ap.add_argument("--out"); a = ap.parse_args()
splits = json.load(open(os.path.join(D, "splits.json")))
seeds = sorted({int(re.search(r"seed(\\d+)", f).group(1)) for f in glob.glob(os.path.join(a.pred_dir, "pred_test_seed*.npz"))})
out = {"names": {}, "per_seed": []}
for sd in seeds:
    t = json.load(open(os.path.join(a.pred_dir, f"timing_seed{sd}.json")))
    row = {"seed": sd, "timing": t}
    for k in ("test", "ood"):
        ref = np.load(os.path.join(D, f"fields_{k}.npz")); pred = np.load(os.path.join(a.pred_dir, f"pred_{k}_seed{sd}.npz"))
        names = splits[k]; out["names"][k] = names
        row[k] = float(np.mean([np.mean((pred[n] - ref[n][:, 3:5]) ** 2) for n in names]))
    out["per_seed"].append(row)
out["mse"] = float(np.mean([r["test"] + r["ood"] for r in out["per_seed"]]))
json.dump(out, open(a.out, "w"))
print("MSE", out["mse"])
'''

ADAPTER = '''
import argparse, json, os, subprocess, sys, tempfile
ap = argparse.ArgumentParser(); ap.add_argument("--case"); ap.add_argument("--reference"); ap.add_argument("--metric-name"); ap.add_argument("--baseline-time")
a = ap.parse_args()
pred = os.path.join(a.case, "predictions") if os.path.isdir(os.path.join(a.case, "predictions")) else a.case
out = os.path.join(tempfile.mkdtemp(), "summary.json")
r = subprocess.run([sys.executable, "SCORER_PATH", "--pred-dir", pred, "--out", out], capture_output=True, text=True)
if r.returncode:
    print("ADAPTER_WARNING:", r.stderr[-800:]); print("METRIC mse: nan"); sys.exit(2)
doc = json.load(open(out))
dest = os.environ.get("CFD_SCIENTIST_SCORER_OUTPUT_DIR")
if dest:
    os.makedirs(dest, exist_ok=True); json.dump(doc, open(os.path.join(dest, "summary.json"), "w"))
print("NAMES", json.dumps(doc["names"]))
print("METRIC mse:", doc["mse"])
'''

BUILDER = '''
import argparse, json, os, time
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--starter"); ap.add_argument("--out"); ap.add_argument("--baseline-out"); a = ap.parse_args()
D = os.path.join(a.starter, "data"); splits = json.load(open(os.path.join(D, "splits.json")))
train = np.load(os.path.join(D, "fields_train.npz")); names = list(splits["train"])
rng = np.random.default_rng(0); order = list(rng.permutation(names))
val_test, val_ood, rest = sorted(order[:3]), sorted(order[3:6]), sorted(order[6:])
os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
np.savez(os.path.join(a.out, "data", "fields_train.npz"), **{n: train[n] for n in rest})
np.savez(os.path.join(a.out, "data", "fields_test.npz"), **{n: train[n] for n in val_test})
np.savez(os.path.join(a.out, "data", "fields_ood.npz"), **{n: train[n] for n in val_ood})
json.dump({"train": rest, "test": val_test, "ood": val_ood}, open(os.path.join(a.out, "data", "splits.json"), "w"))
json.dump({"train": rest, "validation": {"test": val_test, "ood": val_ood}, "hide": [], "notes": "random"},
          open(os.path.join(a.out, "validation_split.json"), "w"))
mean = np.mean(np.concatenate([train[n][:, 3:5] for n in rest]), axis=0)
os.makedirs(a.baseline_out, exist_ok=True); t = {}
for k, group in (("test", val_test), ("ood", val_ood)):
    t0 = time.time(); np.savez(os.path.join(a.baseline_out, f"pred_{k}_seed0.npz"), **{n: np.tile(mean, (len(train[n]), 1)) for n in group}); t[k + "_inference_s"] = time.time() - t0
t["hardware"] = "cpu"; json.dump(t, open(os.path.join(a.baseline_out, "timing_seed0.json"), "w"))
print("ok", len(rest), len(val_test), len(val_ood))
'''

TRAIN_SH = '''#!/bin/bash
set -e
python "$(dirname "$0")/../model/train.py" --starter STARTER --seed "$1" --out "$2"
'''
PREDICT_SH = '''#!/bin/bash
set -e
python "$(dirname "$0")/../model/predict.py" --starter STARTER --seed "$1" --model "$2" --out "$3" --set "$4"
'''
TRAIN_PY = '''
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--starter"); ap.add_argument("--seed", type=int); ap.add_argument("--out"); a = ap.parse_args()
D = os.path.join(a.starter, "data")
z = np.load(os.path.join(D, "fields_train.npz"))
mean = np.mean(np.concatenate([z[n][:, 3:5] for n in z.files]), axis=0)
mean = mean + np.random.default_rng(a.seed).normal(0, 0.05, size=2)
saw_test = os.path.isfile(os.path.join(D, "fields_test.npz")) and os.path.getsize(os.path.join(D, "fields_test.npz")) > 0
os.makedirs(a.out, exist_ok=True)
json.dump({"mean": mean.tolist(), "n_train": len(z.files), "saw_test_file": saw_test}, open(os.path.join(a.out, "model.json"), "w"))
'''
PREDICT_PY = '''
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--starter"); ap.add_argument("--seed", type=int); ap.add_argument("--model"); ap.add_argument("--out"); ap.add_argument("--set"); a = ap.parse_args()
m = json.load(open(os.path.join(a.model, "model.json")))
mean = np.array(m["mean"]) if not COPY else np.array([0.5, 0.5])
z = np.load(os.path.join(a.starter, "data", f"fields_{a.set}.npz"))
assert all(np.isnan(z[n][:, 3:5]).all() for n in z.files), "targets visible"
np.savez(os.path.join(a.out, f"pred_{a.set}_seed{a.seed}.npz"), **{n: np.tile(mean, (len(z[n]), 1)) for n in z.files})
'''


def make_candidate(cand: Path, starter: Path, copy_seeds: bool = False) -> None:
    (cand / "pipeline").mkdir(parents=True)
    (cand / "model").mkdir()
    (cand / "pipeline" / "train.sh").write_text(TRAIN_SH.replace("STARTER", str(starter)))
    (cand / "pipeline" / "predict.sh").write_text(PREDICT_SH.replace("STARTER", str(starter)))
    (cand / "model" / "train.py").write_text(TRAIN_PY)
    (cand / "model" / "predict.py").write_text(PREDICT_PY.replace("COPY", str(copy_seeds)))


with (tempfile.TemporaryDirectory() if not os.environ.get("KEEP_TMP") else __import__("contextlib").nullcontext(tempfile.mkdtemp())) as td:
    tmp = Path(td)
    starter = tmp / "starter"
    (starter / "data").mkdir(parents=True)
    (starter / "scripts").mkdir()
    rng = np.random.default_rng(1)
    names = {"train": [f"tr{i:02d}" for i in range(20)], "test": [f"te{i:02d}" for i in range(6)],
             "ood": [f"oo{i:02d}" for i in range(6)]}
    arrays = {k: {n: rng.normal(size=(30 + i, 5)).astype(np.float32) for i, n in enumerate(v)} for k, v in names.items()}
    for k, v in arrays.items():
        np.savez(starter / "data" / f"fields_{k}.npz", **v)
    (starter / "data" / "splits.json").write_text(json.dumps(names))
    for k, v in names.items():
        for n in v:
            (starter / "raw" / "Dataset" / n).mkdir(parents=True)
            (starter / "raw" / "Dataset" / n / "field.txt").write_text(f"{n} targets")
    (starter / "TASK.md").write_text("Columns 0-2 are inputs, 3-4 targets. Train/test/ood listed in data/splits.json.")
    (starter / "scripts" / "evaluate.py").write_text(SCORER)
    base = starter / "results" / "baseline"
    base.mkdir(parents=True)
    for k in ("test", "ood"):
        np.savez(base / f"pred_{k}_seed0.npz", **{n: np.zeros((len(a), 2), np.float32) for n, a in arrays[k].items()})
    (base / "timing_seed0.json").write_text(json.dumps({"test_inference_s": 1.0, "ood_inference_s": 1.0, "hardware": "x"}))

    study = tmp / "study"
    disc = study / "open_ended_discovery"
    disc.mkdir(parents=True)
    (study / "state.json").write_text("{}")
    test_manifest = build_view(starter, [
        {"path": "data/fields_test.npz", "action": "mask_columns", "columns_to_mask": [3, 4]},
        {"path": "data/fields_ood.npz", "action": "mask_columns", "columns_to_mask": [3, 4]},
        {"path": "raw/Dataset", "action": "hide"},
    ], study / "withheld_test_targets")
    adapter = study / "adapter.py"
    adapter.write_text(ADAPTER.replace("SCORER_PATH", str(starter / "scripts" / "evaluate.py")))

    print("== 1. masked columns in the test manifest")
    by = {Path(e["path"]).name: e for e in test_manifest["entries"]}
    with np.load(by["fields_test.npz"]["source"]) as z:
        check("test inputs kept, targets NaN", np.isfinite(z["te00"][:, :3]).all() and np.isnan(z["te00"][:, 3:]).all())

    print("== 2. the pipeline spec against the baseline")
    spec = {"seeds": [0, 1, 2],
            "sets": [{"name": "test", "files": ["pred_test_seed{seed}.npz"]},
                     {"name": "ood", "files": ["pred_ood_seed{seed}.npz"]}],
            "timing": {"file": "timing_seed{seed}.json", "seconds": {"test_inference_s": "test", "ood_inference_s": "ood"},
                       "text": {"hardware": "hardware"}}}
    check("a correct spec passes", pc.check_spec(spec, base) == [], pc.check_spec(spec, base))
    bad = json.loads(json.dumps(spec))
    bad["sets"][0]["files"] = ["preds_test_{seed}.npz"]
    check("a spec naming files the baseline lacks fails", pc.check_spec(bad, base))
    (disc / pc.SPEC_FILE).write_text(json.dumps(spec))

    print("== 3. the builder and its checks")
    import oed_extensions as ox  # noqa: E402

    def smoke(case_dir: Path, scorer_manifest: str):
        r = ox._run_comparator_with_optional_baseline_time(
            comparator=adapter, case_dir=case_dir, reference_file=starter, baseline_time=None,
            timeout_s=120, scorer_view=scorer_manifest)
        line = next((l for l in r.stdout.splitlines() if l.startswith("METRIC mse:")), "")
        names_line = next((l for l in r.stdout.splitlines() if l.startswith("NAMES")), "")
        smoke.names = json.loads(names_line[6:]) if names_line else {}
        ok = bool(line) and line.split(":")[1].strip() != "nan"
        return ok, line or r.stdout[-500:] + r.stderr[-500:]

    leaky = BUILDER.replace('np.savez(os.path.join(a.out, "data", "fields_train.npz"), **{n: train[n] for n in rest})',
                            'pass')
    replies = iter([leaky, BUILDER])
    status = vs.build(study_dir=study, starter=starter, scorer_path=starter / "scripts" / "evaluate.py",
                      baseline_dir=base, test_manifest_path=study / "withheld_test_targets" / "manifest.json",
                      llm_invoke=lambda m: next(replies), smoke_test=smoke)
    first = status["attempts"][0]["problems"]
    check("a copy that leaves the stand-ins in the training file is refused",
          any("fields_train.npz" in p for p in first), first)
    check("the second script passes", status.get("ok") and status.get("attempt") == 2, status)
    check("the scorer, run with the copy laid over, scores the stand-ins",
          set(smoke.names.get("test", [])) <= set(names["train"]) and smoke.names.get("test"), smoke.names)
    check("the starter's own files are untouched",
          json.loads((starter / "data" / "splits.json").read_text()) == names)
    split = vs.read_split(vs.paths(study)["copy"])
    val_names = split["validation"]["test"] + split["validation"]["ood"]

    print("== 4. what a candidate sees during the search")
    check("a candidate folder finds the validation view",
          find_manifest(disc / "cand_a").endswith("validation_view/manifest.json"))
    cand = disc / "cand_a"
    make_candidate(cand, starter)
    os.environ.pop(MANIFEST_ENV, None)
    from code_mod_agentic import Sandbox  # noqa: E402

    sb = Sandbox(run_dir=cand, starter_case=starter, wm_project_dir=None, starter_root=starter)
    py = sys.executable
    probe = textwrap.dedent(f"""
        import json, os, numpy as np
        D = '{starter}/data'
        te = np.load(D + '/fields_test.npz'); tr = np.load(D + '/fields_train.npz')
        print(json.dumps({{"test": sorted(te.files), "nan": bool(np.isnan(te[te.files[0]][:, 3:]).all()),
            "train": len(tr.files), "split": json.load(open(D + '/splits.json'))["test"]}}))
    """)
    (cand / "probe.py").write_text(probe)
    r = sb.run_bash(f"{py} probe.py", cwd=str(cand))
    seen = json.loads(r["stdout"].strip().splitlines()[-1]) if r.get("stdout", "").strip() else {}
    check("the evaluation file holds the stand-ins, targets withheld",
          seen.get("test") == sorted(split["validation"]["test"]) and seen.get("nan"), r)
    check("the training file holds only the remaining samples", seen.get("train") == len(split["train"]), seen)
    r = sb.run_bash(f"cat {starter}/raw/Dataset/{split['train'][0]}/field.txt; "
                    f"cat {starter}/raw/Dataset/{val_names[0]}/field.txt; "
                    f"cat {starter}/raw/Dataset/te00/field.txt; echo; ls {study}/framework_private | wc -l", cwd=str(cand))
    out = r.get("stdout", "")
    check("a remaining training sample's raw folder is readable", f"{split['train'][0]} targets" in out, r)
    check("a stand-in's raw folder is hidden", f"{val_names[0]} targets" not in out, out)
    check("a test sample's raw folder is hidden", "te00 targets" not in out, out)
    check("the framework's private folder is empty to the candidate", out.strip().splitlines()[-1] == "0", out)
    rf = sb.read_file(str(starter / "raw" / "Dataset" / split["train"][0] / "field.txt"))
    check("read_file serves a training sample's raw file", rf.get("ok"), rf)
    rf = sb.read_file(str(starter / "raw" / "Dataset" / val_names[0] / "field.txt"))
    check("read_file refuses a stand-in's raw file", not rf.get("ok"), rf)

    print("== 5. the framework runs the candidate's pipeline")
    for k in spec["seeds"]:
        r = sb.run_bash(f"bash pipeline/train.sh {k} pipeline/models/seed{k}", cwd=str(cand))
        check(f"candidate trains seed {k} itself", r.get("rc") == 0, r)
    view = vs.paths(study)

    def pipeline_run(run_dir: Path, manifest: str, out: Path, *extra: str) -> dict:
        res = run_dir / "pr.json"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "pipeline_run.py"), "--run-dir", str(run_dir),
                        "--starter", str(starter), "--spec", str(disc / pc.SPEC_FILE), "--manifest", manifest,
                        "--out", str(out), "--result", str(res), *extra],
                       capture_output=True, text=True, timeout=600,
                       env={k: v for k, v in os.environ.items() if k != MANIFEST_ENV})
        return json.loads(res.read_text())

    doc = pipeline_run(cand, str(view["candidate_manifest"]), cand / "framework_scored" / "predictions")
    check("predict.sh ran for every seed and set", doc.get("ok") and len(doc["prediction"]["runs"]) == 6, doc)
    timing = json.loads((cand / "framework_scored" / "predictions" / "timing_seed1.json").read_text())
    check("the framework wrote the timing file from measured times",
          timing["test_inference_s"] > 0 and "timed by the framework" in timing["hardware"], timing)
    ok, line = smoke(cand / "framework_scored", str(view["scorer_manifest"]))
    check("the framework's predictions score on the validation split", ok, line)

    copier = disc / "cand_b"
    make_candidate(copier, starter, copy_seeds=True)
    sb2 = Sandbox(run_dir=copier, starter_case=starter, wm_project_dir=None, starter_root=starter)
    for k in spec["seeds"]:
        sb2.run_bash(f"bash pipeline/train.sh {k} pipeline/models/seed{k}", cwd=str(copier))
    doc = pipeline_run(copier, str(view["candidate_manifest"]), copier / "framework_scored" / "predictions")
    check("seeds that are copies are rejected", not doc.get("ok") and doc.get("seed_copies"), doc)
    p1 = cand / "framework_scored" / "predictions" / "pred_test_seed0.npz"
    p2 = cand / "framework_scored" / "predictions" / "pred_test_seed1.npz"
    check("independent seeds are not flagged", (pc.difference(p1, p2) or 0) > pc.NEAR_COPY, pc.difference(p1, p2))

    print("== 6. the final re-run on the real data")
    final_out = cand / "final_test" / "predictions"
    doc = pipeline_run(cand, str(view["final_manifest"]), final_out, "--train",
                       "--models", str(cand / "final_test" / "models"))
    check("train.sh and predict.sh ran on the real data", doc.get("ok"), doc)
    model = json.loads((cand / "final_test" / "models" / "seed0" / "model.json").read_text())
    check("the final training saw every training sample", model["n_train"] == len(names["train"]), model)
    check("the final training did not see the evaluation set", model["saw_test_file"] is False, model)
    with np.load(final_out / "pred_test_seed0.npz") as z:
        check("the final predictions cover the real test set", sorted(z.files) == names["test"], z.files)
    r = subprocess.run([sys.executable, str(adapter), "--case", str(cand / "final_test"), "--reference", "x"],
                       capture_output=True, text=True)
    check("the real scorer scores the final predictions on the test set",
          "METRIC mse:" in r.stdout and json.loads(r.stdout.split("NAMES ")[1].splitlines()[0])["test"] == names["test"],
          r.stdout + r.stderr)

    print("== 7. the brief")
    import surrogate_agentic as sa  # noqa: E402

    brief = sa.build_surrogate_prompt(topic="t", hypothesis="h", variant_name="v", run_dir=cand, starter_case=starter)
    check("the brief asks for train.sh and predict.sh and explains the validation data",
          "train.sh SEED MODEL_DIR" in brief and "validation version of the study's data" in brief
          and "Do not write timing_seed{seed}.json" in brief, brief[-3000:])
    check("the candidate result counts the pipeline as produced",
          pc.missing_parts(cand, spec) == [], pc.missing_parts(cand, spec))

    print("== 8. the manager's tools: scoring a candidate, then the one test")
    from cfd_langgraph.config import Settings  # noqa: E402
    import cfd_langgraph.manager.tools as tools  # noqa: E402

    (study / "study_mode.json").write_text(json.dumps({"mode": "surrogate"}))
    (study / "starter_understanding.json").write_text(json.dumps({"starter_dir": str(starter)}))
    ok, line = smoke(view["baseline"], str(view["scorer_manifest"]))
    base_value = float(line.split(":")[1])
    (disc / "search_config.json").write_text(json.dumps({"topic": "t", "total_budget": 10, "baseline_direction": "min",
                                                         "target_value": 0.01, "target_kind": "absolute",
                                                         "target_improvement_pct": 10}))
    (disc / "baseline_score.json").write_text(json.dumps({"metric": "mse", "value": base_value, "direction": "min",
                                                          "verified": True}))
    (disc / "bound_comparators.json").write_text(json.dumps({"mse": {
        "path": str(adapter), "origin": "surrogate_adapter", "selftest_ok": True, "selftest_value": 2.0,
        "scorer_seconds": 5}}))
    (disc / "metric_specs.json").write_text(json.dumps([{"name": "mse", "direction": "min", "primary": True}]))
    (disc / "objective_contract.json").write_text(json.dumps({"reference_files": []}))
    (disc / "proposals.json").write_text(json.dumps({"cand_a": {"strategy": "analytic", "target_family": "mean"}}))
    (cand / "agentic_result.json").write_text(json.dumps({
        "status": "OK", "case_dir": str(cand), "produced_predictions": True, "finished_cleanly": True}))
    shutil.rmtree(cand / "framework_scored", ignore_errors=True)
    shutil.rmtree(cand / "final_test", ignore_errors=True)
    built = tools.build_manager_tools(Settings(), study)
    every = [f for v in built.values() if isinstance(v, list) for f in v]
    tool = {getattr(f, "__name__", ""): f for f in every}
    rec = tool["oed_score_candidate"](candidate_dir=str(cand), case_dir=str(cand), action_type="code_mod",
                                      variant_name="cand_a", model_description="mean model")
    check("the candidate is scored on the validation split",
          rec.get("ok") and rec.get("score") and "validation" in str(rec.get("scored_on")), rec)
    card = json.loads((cand / "validation_score.json").read_text()) if (cand / "validation_score.json").is_file() else {}
    check("its folder holds the aggregate validation score", card.get("value") is not None and "metrics" in card, card)
    check("the scorer's full output is kept where candidates cannot read it",
          (view["scores"] / "cand_a" / "summary.json").is_file() and not (cand / "scorer_output").exists(),
          [str(x) for x in view["private"].rglob("*") if "scores" in str(x)] + [str(rec.get("metric_vector"))[:600]])
    check("the framework's predictions were removed after scoring",
          not (cand / "framework_scored" / "predictions").exists())
    check("the timing it measured is in the record",
          len((rec.get("pipeline_run") or {}).get("predict_seconds") or []) == 6, rec.get("pipeline_run"))

    (disc / "history.json").write_text(json.dumps([{
        "candidate_dir": str(cand), "case_dir": str(cand), "valid_case": True, "status": rec.get("status"),
        "score": rec.get("score")}]))
    result = tool["oed_final_test"]()
    check("the final test re-trains the best candidate and scores it on the test set",
          result.get("ok") and isinstance(result.get("test_value"), float) and result.get("candidate") == "cand_a",
          result)
    final_summary = json.loads((study / "final_test" / "scorer_output" / "summary.json").read_text())
    check("the final score is on the real test samples", final_summary["names"]["test"] == names["test"],
          final_summary.get("names"))
    again = tool["oed_final_test"]()
    check("a second call returns the recorded result", again.get("already_done") and
          again.get("test_value") == result.get("test_value"), again)
    proposed = tool["oed_propose_candidates"](topic="t", num_candidates=1)
    check("no candidate can be proposed after the final test", "search is closed" in str(proposed), proposed)

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
