#!/usr/bin/env python3
"""Neural networks inside OpenFOAM (cfd_langgraph.nn_coupling).

1. DAFoam's weight vector becomes the identical PyTorch network; read in
   PyTorch's own order the same numbers make another network.
2. The card refuses what would make a network unusable.
3. The exported TorchScript carries the card's normalisation, output scaling,
   finite check and bounds.
4. OpenFOAM's build of a network reproduces the card's stored outputs.
5. A real OpenFOAM case evaluating a network through nnModel.H passes the
   case check; the same case feeding its features in the wrong order fails it.

Parts 4 and 5 need an OpenFOAM (default /mnt/sda1/openfoam10, or
$CFD_SCIENTIST_TEST_OPENFOAM) and compile against it, a minute or two.

Run: python scripts/test_nn_coupling.py
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfd_langgraph.nn_coupling import dafoam, export as E, openfoam as O, verify as V  # noqa: E402
from cfd_langgraph.nn_coupling.card import Feature, ModelCard  # noqa: E402

FAILURES: list[str] = []
OPENFOAM = os.environ.get("CFD_SCIENTIST_TEST_OPENFOAM", "/mnt/sda1/openfoam10")


def check(name: str, got, want=True) -> None:
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f" — got {got!r}, want {want!r}"))
    if not ok:
        FAILURES.append(name)


rng = np.random.default_rng(3)

print("== 1. DAFoam's weight layout")
for activation in ("tanh", "sigmoid", "relu"):
    params = rng.normal(size=dafoam.n_parameters(3, [5, 4]))
    net = dafoam.mlp_from_dafoam(3, [5, 4], activation, params, leaky_coeff=0.1)
    x = rng.uniform(-1, 2, size=(40, 3))
    with torch.no_grad():
        got = net(torch.tensor(x)).numpy().reshape(-1)
    want = dafoam.reference_output(3, [5, 4], activation, params, x, leaky_coeff=0.1)
    check(f"{activation}: the mapped network is DAFoam's, to rounding", float(abs(got - want).max()) < 1e-13)
params = rng.normal(size=dafoam.n_parameters(3, [5, 4]))
plain = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 4), torch.nn.Tanh(),
                            torch.nn.Linear(4, 1)).double()
k = 0
with torch.no_grad():
    for layer in (plain[0], plain[2], plain[4]):
        for t in (layer.weight, layer.bias):
            t.copy_(torch.tensor(params[k:k + t.numel()]).reshape(t.shape))
            k += t.numel()
    x = rng.uniform(-1, 2, size=(40, 3))
    misread = plain(torch.tensor(x)).numpy().reshape(-1)
check("the same numbers read in PyTorch's order make another network",
      float(abs(misread - dafoam.reference_output(3, [5, 4], "tanh", params, x)).max()) > 1e-3)
try:
    dafoam.mlp_from_dafoam(3, [5], "tanh", [0.0] * 5)
    check("a weight vector of the wrong length is refused", False)
except ValueError:
    check("a weight vector of the wrong length is refused", True)

print("== 2. the card")
card = ModelCard("m", [Feature("a", "the a field"), Feature("b", "")], "out", input_shift=[0.0],
                 input_scale=[1.0, 0.0])
problems = " | ".join(card.problems())
check("an input without a definition is named", "'b' has no definition" in problems)
check("shift and scale must match the inputs", "need one value per input" in problems)
check("a zero scale is refused", "zero or non-finite" in problems)

print("== 3. the exported network carries its scaling")


def two_input_net() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)).double()


def demo_card(**bounds) -> ModelCard:
    return ModelCard("demo", [Feature("speed", "|U| per cell, m/s"), Feature("x", "cell centre x, m")], "nnOut",
                     input_shift=[-0.3, -0.05], input_scale=[4.0, 10.0], output_shift=0.5, output_scale=2.0,
                     **bounds)


with tempfile.TemporaryDirectory() as tmp:
    out = Path(tmp)
    net = two_input_net()
    card = demo_card(lower_bound=-1.0, upper_bound=3.0, default_value=0.25, training_range=[[0.0, 1.0], [0.0, 0.1]])
    sample = np.column_stack([rng.uniform(0, 1, 16), rng.uniform(0, 0.1, 16)])
    done = E.export(net, card, sample, out)
    with torch.no_grad():
        by_hand = card.finish(net(torch.tensor(card.normalise(sample))).numpy())
    check("its outputs are the card's formula around the network",
          bool(np.allclose(done.parity_outputs, by_hand, rtol=1e-13, atol=1e-14)))
    check("the card records the file it was written for", done.sha256 == E.sha256(out / "model.pt"))
    y = E.evaluate(out / "card.json", np.array([[float("nan"), 0.05]]))
    check("a non-finite output becomes the default", float(y[0]), 0.25)
    narrow = E.export(net, demo_card(lower_bound=-0.01, upper_bound=0.01), sample, out / "narrow")
    check("an output past a bound is held at it",
          (max(narrow.parity_outputs) <= 0.01, min(narrow.parity_outputs) >= -0.01,
           any(abs(v) == 0.01 for v in narrow.parity_outputs)), (True, True, True))
    loaded = ModelCard.load(out / "card.json")
    check("the card survives being saved and read back", (loaded.upper_bound, loaded.inputs[1].name), (3.0, "x"))
    open_card = ModelCard("o", [Feature("a", "d")], "y", input_shift=[0.0], input_scale=[1.0])
    open_card.save(out / "open.json")
    check("an open bound is kept open", math.isinf(ModelCard.load(out / "open.json").upper_bound))

if not Path(OPENFOAM, "etc", "bashrc").is_file():
    print(f"[SKIP] no OpenFOAM at {OPENFOAM}: parts 4 and 5 not run")
else:
    env = O.openfoam_env(OPENFOAM)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        card_dir = out / "card"
        E.export(two_input_net(), demo_card(), np.column_stack([rng.uniform(0, 1, 16), rng.uniform(0, 0.1, 16)]),
                 card_dir)

        print("== 4. OpenFOAM's build reproduces the card")
        report = V.check_parity(card_dir / "card.json", OPENFOAM, build_root=out / "build")
        check("parity holds", report.get("ok"), True)
        tampered = ModelCard.load(card_dir / "card.json")
        tampered.parity_outputs = [v + 1e-3 for v in tampered.parity_outputs]
        tampered.save(card_dir / "tampered.json")
        check("a card whose stored outputs are not the network's fails parity",
              V.check_parity(card_dir / "tampered.json", OPENFOAM, build_root=out / "build").get("ok"), False)

        print("== 5. a real case evaluating the network")

        def run_cavity(name: str, order: tuple) -> Path:
            case = out / name
            shutil.copytree(Path(env["FOAM_TUTORIALS"]) / "incompressible" / "icoFoam" / "cavity" / "cavity", case)
            entries = O.coded_entries(case / "nn")
            code = f"""
functions
{{
    nnDemo
    {{
        type            coded;
        libs            ("libutilityFunctionObjects.so");
        name            nnDemo;
{entries}
        codeWrite
        #{{
            static nnModel model("{card_dir / 'model.pt'}", 2);
            const volVectorField& U = mesh().lookupObject<volVectorField>("U");
            volScalarField speed(IOobject("nn_speed", mesh().time().timeName(), mesh()), mag(U));
            volScalarField x(IOobject("nn_x", mesh().time().timeName(), mesh()), mesh().C().component(0));
            List<const scalarField*> features(2);
            features[0] = &{order[0]}.primitiveField();
            features[1] = &{order[1]}.primitiveField();
            volScalarField nnOut
            (
                IOobject("nnOut", mesh().time().timeName(), mesh()),
                mesh(),
                dimensionedScalar(dimless, 0)
            );
            nnOut.primitiveFieldRef() = model.evaluate(features);
            speed.write();
            x.write();
            nnOut.write();
        #}};
    }}
}}
"""
            control = case / "system" / "controlDict"
            mark = "// ************************************************************************* //"
            text = control.read_text().replace(mark, code + "\n" + mark, 1)
            # The protocol's full-precision fields, so the check compares like with like.
            control.write_text(text.replace("writePrecision  6;", "writePrecision  17;"))
            for cmd in (["blockMesh"], ["icoFoam"]):
                proc = subprocess.run(cmd, cwd=case, env=env, capture_output=True, text=True, timeout=1200)
                if proc.returncode != 0:
                    print((proc.stdout + proc.stderr)[-3000:])
                    raise RuntimeError(f"{cmd[0]} failed in {name}")
            return case

        good = run_cavity("cavity_good", ("speed", "x"))
        low = out / "cavity_low_precision"
        shutil.copytree(good, low)
        control = low / "system" / "controlDict"
        control.write_text(control.read_text().replace("writePrecision  17;", "writePrecision  6;"))
        report = V.check_case(low, card_dir / "card.json")
        check("a case writing its fields to 6 digits is told to write them in full",
              (report.get("ok"), "writePrecision" in " ".join(report.get("problems", []))), (False, True))
        report = V.check_case(good, card_dir / "card.json")
        check("the case's output is the card's network on the case's features", report.get("ok"), True)
        check("it reads every cell", report.get("cells"), 400)
        swapped = run_cavity("cavity_swapped", ("x", "speed"))
        report = V.check_case(swapped, card_dir / "card.json")
        check("a case feeding its features in the wrong order is caught", report.get("ok"), False)
        check("and told why", "feature order, normalisation" in report.get("note", ""))

print()
print("FAILURES:", FAILURES if FAILURES else "none")
sys.exit(1 if FAILURES else 0)
