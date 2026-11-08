"""python -m cfd_langgraph.nn_coupling <command> ...: each command prints one
JSON object, and exits 1 when what it checks does not hold."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _export(args) -> dict:
    from .card import ModelCard
    from .export import export

    spec = importlib.util.spec_from_file_location("nn_model_script", args.model_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    net = module.build_model()
    sample = np.loadtxt(args.sample, ndmin=2) if args.sample else np.asarray(module.sample_inputs(), dtype=float)
    card = export(net, ModelCard.load(Path(args.card)), sample, Path(args.out))
    return {"ok": True, "card": str(Path(args.out) / "card.json"),
            "torchscript": str(Path(args.out) / card.torchscript), "parity_rows": len(card.parity_inputs)}


def _from_dafoam(args) -> dict:
    from .dafoam import card_from_dafoam, reference_output
    from .export import export
    from .verify import check_case, read_case_field

    options = json.loads(Path(args.options).read_text())
    parameters = json.loads(Path(args.parameters).read_text())
    if not (isinstance(parameters, list) and all(isinstance(v, (int, float)) for v in parameters)):
        found = (f"a JSON object with keys {sorted(parameters)}" if isinstance(parameters, dict)
                 else f"a JSON {type(parameters).__name__}")
        raise ValueError(f"--parameters {args.parameters} holds {found}; it must be a bare JSON list of "
                         "numbers, the weight vector in DAFoam's own order (what prob.get_val returns)")
    definitions = json.loads(Path(args.definitions).read_text()) if args.definitions else None
    net, card = card_from_dafoam(args.name, options, parameters, definitions)
    if args.case:
        # The rows the trained network actually saw, read back from the
        # features DAFoam wrote (already shifted and scaled).
        columns = [read_case_field(Path(args.case), args.time, f.name) for f in card.inputs]
        normalised = np.column_stack(columns)
        raw = normalised / np.asarray(card.input_scale) - np.asarray(card.input_shift)
        card.training_range = [[float(raw[:, j].min()), float(raw[:, j].max())] for j in range(raw.shape[1])]
        rows = np.unique(np.linspace(0, raw.shape[0] - 1, min(64, raw.shape[0])).astype(int))
        sample = raw[rows]
    else:
        sample = np.random.default_rng(0).uniform(0.0, 1.0, size=(32, len(card.inputs)))
    hidden = [int(h) for h in options["hiddenLayerNeurons"]]
    activation = str(options.get("activationFunction", "sigmoid"))
    reference = card.finish(reference_output(len(card.inputs), hidden, activation, parameters,
                                             card.normalise(sample), float(options.get("leakyCoeff", 0.0))))
    card = export(net, card, sample, Path(args.out))
    mapped = np.allclose(card.parity_outputs, reference, rtol=1e-12, atol=1e-13)
    report = {"ok": mapped, "card": str(Path(args.out) / "card.json"),
              "matches_dafoam_formula": mapped,
              "max_difference_from_dafoam_formula": float(np.abs(np.asarray(card.parity_outputs) - reference).max())}
    if args.case:
        report["dafoam_case_check"] = check_case(Path(args.case), Path(args.out) / "card.json", args.time,
                                                 prefix="", features_normalised=True)
        report["ok"] = report["ok"] and report["dafoam_case_check"]["ok"]
    return report


def _check_parity(args) -> dict:
    from .verify import check_parity

    return check_parity(Path(args.card), args.openfoam, Path(args.build_root) if args.build_root else None)


def _check_case(args) -> dict:
    from .verify import check_case

    return check_case(Path(args.case), Path(args.card), args.time, args.prefix, args.output_field,
                      args.normalised)


def _coded_entries(args) -> dict:
    from .openfoam import HELPER, coded_entries

    return {"ok": True, "helper": str(Path(args.dir) / HELPER), "entries": coded_entries(Path(args.dir))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cfd_langgraph.nn_coupling")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export", help="a trained PyTorch network and its card as TorchScript + parity set")
    p.add_argument("--model-script", required=True,
                   help="a .py defining build_model() (the trained network) and sample_inputs() (raw rows)")
    p.add_argument("--card", required=True, help="the card JSON: inputs, output, normalisation, bounds")
    p.add_argument("--sample", default="", help="raw input rows (text), instead of sample_inputs()")
    p.add_argument("--out", required=True)
    p.set_defaults(run=_export)
    p = sub.add_parser("from-dafoam", help="a network trained inside DAFoam as PyTorch + card")
    p.add_argument("--name", required=True, help="the regression model's name in daOptions")
    p.add_argument("--options", required=True, help="JSON of daOptions['regressionModel'][name]")
    p.add_argument("--parameters", required=True, help="JSON list: the trained weight vector")
    p.add_argument("--definitions", default="", help="JSON {input: definition} for non-DAFoam features")
    p.add_argument("--case", default="", help="the DAFoam case that wrote its features (writeFeatures)")
    p.add_argument("--time", default="", help="the time folder holding those features and the output")
    p.add_argument("--out", required=True)
    p.set_defaults(run=_from_dafoam)
    p = sub.add_parser("check-parity", help="OpenFOAM's build reproduces the card's stored outputs")
    p.add_argument("--card", required=True)
    p.add_argument("--openfoam", default="", help="the OpenFOAM installation (default: the loaded one)")
    p.add_argument("--build-root", default="", help="where the parity program is built and kept")
    p.set_defaults(run=_check_parity)
    p = sub.add_parser("check-case", help="a case's written features and output against the card")
    p.add_argument("--case", required=True)
    p.add_argument("--card", required=True)
    p.add_argument("--time", default="")
    p.add_argument("--prefix", default="nn_")
    p.add_argument("--output-field", default="")
    p.add_argument("--normalised", action="store_true", help="the features are written already scaled")
    p.set_defaults(run=_check_case)
    p = sub.add_parser("coded-entries", help="nnModel.H and the entries a coded object needs")
    p.add_argument("--dir", required=True)
    p.set_defaults(run=_coded_entries)
    args = parser.parse_args(argv)
    try:
        result = args.run(args)
    except (ValueError, TypeError, RuntimeError, FileNotFoundError, KeyError) as exc:
        result = {"ok": False, "problems": [f"{type(exc).__name__}: {exc}"]}
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
