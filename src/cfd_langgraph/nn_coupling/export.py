"""A trained PyTorch network and its card as one TorchScript file that carries
its own normalisation, output scaling and bounds, so OpenFOAM is given the raw
features and evaluates exactly what was trained."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .card import MIN_PARITY_ROWS, ModelCard


class Scaled(torch.nn.Module):
    """``net`` with the card's input normalisation in front and its output
    scaling, finite check and bounds behind (card.py has the conventions)."""

    def __init__(self, net: torch.nn.Module, card: ModelCard) -> None:
        super().__init__()
        self.net = net
        f64 = torch.float64
        self.register_buffer("in_shift", torch.tensor(card.input_shift, dtype=f64))
        self.register_buffer("in_scale", torch.tensor(card.input_scale, dtype=f64))
        self.register_buffer("out_shift", torch.tensor(float(card.output_shift), dtype=f64))
        self.register_buffer("out_scale", torch.tensor(float(card.output_scale), dtype=f64))
        self.register_buffer("lower", torch.tensor(float(card.lower_bound), dtype=f64))
        self.register_buffer("upper", torch.tensor(float(card.upper_bound), dtype=f64))
        self.register_buffer("default", torch.tensor(float(card.default_value), dtype=f64))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        y = self.net((raw + self.in_shift) * self.in_scale).reshape(-1)
        y = self.out_scale * (y + self.out_shift)
        y = torch.where(torch.isfinite(y), y, torch.zeros_like(y) + self.default)
        return torch.minimum(torch.maximum(y, self.lower), self.upper)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(net: torch.nn.Module, card: ModelCard, sample_raw: np.ndarray, out_dir: Path,
           card_name: str = "card.json") -> ModelCard:
    """Write ``net`` with ``card``'s scaling as TorchScript (double precision)
    beside the card, and record its outputs for ``sample_raw`` (raw inputs,
    one row per point, e.g. rows of the training data) as the parity set every
    OpenFOAM build of it must reproduce. Returns the completed card."""
    problems = card.problems()
    if problems:
        raise ValueError("the card is incomplete: " + "; ".join(problems))
    sample = np.asarray(sample_raw, dtype=float)
    if sample.ndim != 2 or sample.shape[1] != len(card.inputs) or sample.shape[0] < MIN_PARITY_ROWS:
        raise ValueError(f"sample_raw must be at least {MIN_PARITY_ROWS} rows of {len(card.inputs)} raw "
                         f"inputs in the card's order; got shape {sample.shape}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapped = Scaled(net.double().eval(), card).eval()
    try:
        scripted = torch.jit.script(wrapped)
    except Exception:  # noqa: BLE001 -- a network TorchScript cannot compile is traced instead
        scripted = torch.jit.trace(wrapped, torch.tensor(sample[:2], dtype=torch.float64))
    path = out_dir / card.torchscript
    scripted.save(str(path))
    x = torch.tensor(sample, dtype=torch.float64)
    with torch.no_grad():
        eager = wrapped(x).numpy()
        saved = torch.jit.load(str(path))(x).numpy()
    if not np.allclose(eager, saved, rtol=1e-12, atol=1e-14):
        raise RuntimeError("the saved TorchScript does not reproduce the network: "
                           f"largest difference {np.abs(eager - saved).max():.3e}")
    card.parity_inputs = sample.tolist()
    card.parity_outputs = saved.tolist()
    card.sha256 = sha256(path)
    card.save(out_dir / card_name)
    return card


def evaluate(card_path: Path, raw: np.ndarray, card: Optional[ModelCard] = None) -> np.ndarray:
    """The card's network on ``raw`` inputs, as Python evaluates it."""
    card_path = Path(card_path)
    card = card or ModelCard.load(card_path)
    model = torch.jit.load(str(card_path.parent / card.torchscript))
    with torch.no_grad():
        return model(torch.tensor(np.asarray(raw, dtype=float), dtype=torch.float64)).numpy()
