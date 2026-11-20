"""The model card: everything OpenFOAM must reproduce to evaluate a network as
it was trained.

A network's weights are half of it. The other half is what it was fed and how:
which features, in which order, computed how, scaled how, and what its output
means and how it is scaled back. Get any of it wrong inside a solver and
nothing crashes: the network answers inputs it never saw with numbers that
look plausible. The card holds all of it, beside the TorchScript file, and the
checks (verify.py) hold OpenFOAM to it.

Conventions, DAFoam's (DARegression.C, v5.1.1), so a network trained inside
DAFoam needs no conversion:

    normalised input   x_i = (raw_i + input_shift_i) * input_scale_i
    output             y   = output_scale * (net(x) + output_shift)

then a y that is not finite becomes default_value, and y is clipped to
[lower_bound, upper_bound]. A network trained on standardised inputs (mean m,
standard deviation s) has input_shift = -m and input_scale = 1/s.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# The fewest stored inputs a card's parity set may hold: enough rows to tell
# a wrong feature order or scaling from a coincidence.
MIN_PARITY_ROWS = 8


@dataclass
class Feature:
    name: str
    # How the raw value is computed from the flow, exactly as training computed
    # it: the formula, which fields, cell or face values, any small constants.
    definition: str
    units: str = "-"


@dataclass
class ModelCard:
    name: str
    inputs: List[Feature]
    output: str
    output_meaning: str = ""
    input_shift: List[float] = field(default_factory=list)
    input_scale: List[float] = field(default_factory=list)
    output_shift: float = 0.0
    output_scale: float = 1.0
    lower_bound: float = -math.inf
    upper_bound: float = math.inf
    default_value: float = 0.0
    # [min, max] of each raw input over the training data, when known: a case
    # whose features fall outside it is asking the network to extrapolate.
    training_range: List[List[float]] = field(default_factory=list)
    # "pytorch", or "dafoam-adjoint" for a network trained inside DAFoam.
    source: str = "pytorch"
    torchscript: str = "model.pt"
    sha256: str = ""
    parity_inputs: List[List[float]] = field(default_factory=list)
    parity_outputs: List[float] = field(default_factory=list)
    notes: str = ""

    def problems(self) -> List[str]:
        """What keeps this card from being used, said plainly; empty when none."""
        out: List[str] = []
        if not self.name.strip():
            out.append("name is empty")
        if not self.inputs:
            out.append("inputs is empty")
        for f in self.inputs:
            if not f.definition.strip():
                out.append(f"input {f.name!r} has no definition: say exactly how it is computed")
        names = [f.name for f in self.inputs]
        if len(set(names)) != len(names):
            out.append(f"input names repeat: {names}")
        n = len(self.inputs)
        if len(self.input_shift) != n or len(self.input_scale) != n:
            out.append(f"input_shift ({len(self.input_shift)}) and input_scale ({len(self.input_scale)}) "
                       f"need one value per input ({n})")
        if any(not math.isfinite(s) or s == 0 for s in self.input_scale):
            out.append("input_scale holds a zero or non-finite value")
        if not math.isfinite(self.output_scale) or self.output_scale == 0:
            out.append("output_scale is zero or not finite")
        if not self.output.strip():
            out.append("output is empty: name the field or quantity the network gives")
        if not self.lower_bound <= self.upper_bound:
            out.append("lower_bound is above upper_bound")
        if self.training_range and (len(self.training_range) != n
                                    or any(len(r) != 2 or r[0] > r[1] for r in self.training_range)):
            out.append("training_range needs one [min, max] per input")
        return out

    def normalise(self, raw: np.ndarray) -> np.ndarray:
        """The network's input from raw features (one row per point, inputs in order)."""
        return (np.asarray(raw, dtype=float) + np.asarray(self.input_shift)) * np.asarray(self.input_scale)

    def finish(self, net_out: np.ndarray) -> np.ndarray:
        """The card's output from the network's own output: scaled, then made
        finite, then bounded, in DAFoam's order."""
        y = self.output_scale * (np.asarray(net_out, dtype=float).reshape(-1) + self.output_shift)
        y = np.where(np.isfinite(y), y, self.default_value)
        return np.clip(y, self.lower_bound, self.upper_bound)

    def save(self, path: Path) -> Path:
        data = asdict(self)
        # JSON has no infinity: an open bound is written as null.
        for key in ("lower_bound", "upper_bound"):
            if not math.isfinite(data[key]):
                data[key] = None
        Path(path).write_text(json.dumps(data, indent=2))
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "ModelCard":
        data = json.loads(Path(path).read_text())
        data["inputs"] = [Feature(**f) for f in data.get("inputs") or []]
        if data.get("lower_bound") is None:
            data["lower_bound"] = -math.inf
        if data.get("upper_bound") is None:
            data["upper_bound"] = math.inf
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})
