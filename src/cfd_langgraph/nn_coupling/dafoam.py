"""A network trained inside DAFoam (daOptions "regressionModel", modelType
"neuralNetwork") as an identical PyTorch network with its card.

DAFoam trains such a network through its adjoint: the weights are a design
variable (inputInfo type "regressionPar") and the gradient of a flow objective
with respect to every weight comes from one adjoint solve. It keeps the
weights in one vector, neuron by neuron: for each neuron of each hidden layer,
its weights from every input (or neuron) of the layer before and then its
bias; then the output neuron's weights and bias (DARegression.C, v5.1.1).
PyTorch keeps a layer's whole weight matrix and then its bias vector, so the
same numbers read in PyTorch's order make another network. mlp_from_dafoam
maps one to the other, and reference_output is DARegression's own loop in
NumPy, to prove the mapping on any inputs.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .card import Feature, ModelCard

# DAFoam's built-in features as DARegression.C computes them, before its
# shift and scale. Every ratio is A / (A + B + 1e-16), so it lies in [0, 1].
DAFOAM_FEATURES: Dict[str, str] = {
    "VoS": ("|Omega| / (|S| + |Omega| + 1e-16) per cell, with gradU = fvc::grad(U) (the case's "
            "gradSchemes), Omega = skew(gradU), S = symm(gradU) and |T| = sqrt(T && T), OpenFOAM's mag"),
    "chiSA": "nuTilda / (nu + nuTilda + 1e-16) per cell, nu the laminar kinematic viscosity",
    "PoD": ("P / (D + P + 1e-16) per cell from the Spalart-Allmaras model's own terms: "
            "P = Cb1 * Stilda * nuTilda and D = Cw1 * fw(Stilda) * (nuTilda / y)^2, with its chi, "
            "fv1, Stilda and fw and y the wall distance (rho = 1 for an incompressible case)"),
}


def n_parameters(n_inputs: int, hidden: Sequence[int]) -> int:
    """The length of DAFoam's weight vector for this network."""
    sizes = [n_inputs, *hidden, 1]
    return sum((sizes[i] + 1) * sizes[i + 1] for i in range(len(sizes) - 1))


def _activation(name: str, leaky_coeff: float) -> torch.nn.Module:
    if name == "tanh":
        return torch.nn.Tanh()
    if name == "sigmoid":
        return torch.nn.Sigmoid()
    if name == "relu":
        # DAFoam's relu is leaky, with leakyCoeff (default 0, a plain relu).
        return torch.nn.LeakyReLU(negative_slope=leaky_coeff)
    raise ValueError(f"activation {name!r} is not one of DAFoam's: tanh, sigmoid, relu")


def mlp_from_dafoam(n_inputs: int, hidden: Sequence[int], activation: str, parameters: Sequence[float],
                    leaky_coeff: float = 0.0) -> torch.nn.Sequential:
    """DAFoam's network as torch layers (double precision), weights in place."""
    p = np.asarray(parameters, dtype=float)
    if p.size != n_parameters(n_inputs, hidden):
        raise ValueError(f"{p.size} parameters for a network that has {n_parameters(n_inputs, hidden)}")
    sizes = [n_inputs, *hidden, 1]
    layers: List[torch.nn.Module] = []
    k = 0
    for i in range(len(sizes) - 1):
        n_in, n_out = sizes[i], sizes[i + 1]
        lin = torch.nn.Linear(n_in, n_out).double()
        weight = np.empty((n_out, n_in))
        bias = np.empty(n_out)
        for neuron in range(n_out):  # DAFoam's order: a neuron's weights, then its bias
            weight[neuron] = p[k:k + n_in]
            bias[neuron] = p[k + n_in]
            k += n_in + 1
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(weight))
            lin.bias.copy_(torch.from_numpy(bias))
        layers.append(lin)
        if i < len(sizes) - 2:  # no activation on the output layer
            layers.append(_activation(activation, leaky_coeff))
    return torch.nn.Sequential(*layers)


def reference_output(n_inputs: int, hidden: Sequence[int], activation: str, parameters: Sequence[float],
                     normalised: np.ndarray, leaky_coeff: float = 0.0) -> np.ndarray:
    """The network's raw output on normalised features, computed as
    DARegression::compute does it, loop for loop (its tanh written as
    (1 - e^-2x) / (1 + e^-2x)). Independent of torch, to check the mapping."""
    p = np.asarray(parameters, dtype=float)
    x = np.atleast_2d(np.asarray(normalised, dtype=float))
    out = np.empty(x.shape[0])
    for c, row in enumerate(x):
        k = 0
        prev = list(row)
        for n_neurons in hidden:
            layer = []
            for _ in range(n_neurons):
                v = 0.0
                for value in prev:
                    v += value * p[k]
                    k += 1
                v += p[k]
                k += 1
                if activation == "sigmoid":
                    v = 1.0 / (1.0 + math.exp(-v))
                elif activation == "tanh":
                    v = (1.0 - math.exp(-2.0 * v)) / (1.0 + math.exp(-2.0 * v))
                elif activation == "relu":
                    v = v if v >= 0 else leaky_coeff * v
                layer.append(v)
            prev = layer
        v = 0.0
        for value in prev:
            v += value * p[k]
            k += 1
        out[c] = v + p[k]
    return out


def card_from_dafoam(name: str, options: dict, parameters: Sequence[float],
                     definitions: Optional[Dict[str, str]] = None) -> Tuple[torch.nn.Sequential, ModelCard]:
    """The PyTorch network and card of DAFoam regression model ``name``, from
    its daOptions["regressionModel"][name] dictionary and its trained weight
    vector. ``definitions`` gives each input's definition where DAFoam's own
    (DAFOAM_FEATURES) does not."""
    if options.get("modelType", "neuralNetwork") != "neuralNetwork":
        raise ValueError(f"modelType {options.get('modelType')!r}: only neuralNetwork models are trained "
                         "through DAFoam's adjoint and exported here")
    inputs = list(options["inputNames"])
    hidden = [int(h) for h in options["hiddenLayerNeurons"]]
    activation = str(options.get("activationFunction", "sigmoid"))
    leaky = float(options.get("leakyCoeff", 0.0))
    net = mlp_from_dafoam(len(inputs), hidden, activation, parameters, leaky)
    defs = {**DAFOAM_FEATURES, **(definitions or {})}
    missing = [n for n in inputs if n not in defs]
    if missing:
        raise ValueError(f"no definition for input(s) {missing}: pass them in definitions")
    card = ModelCard(
        name=name,
        inputs=[Feature(n, defs[n]) for n in inputs],
        output=str(options["outputName"]),
        output_meaning=f"DAFoam regression model {name!r} output",
        input_shift=[float(v) for v in options["inputShift"]],
        input_scale=[float(v) for v in options["inputScale"]],
        output_shift=float(options.get("outputShift", 0.0)),
        output_scale=float(options.get("outputScale", 1.0)),
        lower_bound=float(options.get("outputLowerBound", -math.inf)),
        upper_bound=float(options.get("outputUpperBound", math.inf)),
        default_value=float(options.get("defaultOutputValue", 0.0)),
        source="dafoam-adjoint",
        notes=f"hidden layers {hidden}, activation {activation}"
              + (f", leakyCoeff {leaky}" if activation == "relu" else ""),
    )
    return net, card
