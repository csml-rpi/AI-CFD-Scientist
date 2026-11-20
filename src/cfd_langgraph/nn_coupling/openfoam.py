"""Evaluating a card's network inside OpenFOAM through libtorch.

The libtorch used is the one inside this Python environment's PyTorch, so
nothing is downloaded or installed. Three things had to be right, found
building a network into OpenFOAM 10 (2026-10-22):

- OpenFOAM's TypeName macro breaks libtorch's headers ("two or more data types
  in declaration of 'typeName_'"): it is hidden while they load;
- OpenFOAM 10 compiles as C++14 and libtorch needs C++17: -std=c++17 comes
  after OpenFOAM's own flag and wins;
- libtorch starts threads of its own in every MPI rank, 71 s of CPU for a
  27 s run on four ranks: one thread per rank.

nnModel.H holds all three. A case uses it from a coded object (coded function
object, coded fvModel or fvConstraint), compiled when the case starts, with
the entries coded_entries() returns, or from a case-local library with the
same flags.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .card import ModelCard

HELPER = "nnModel.H"

NN_MODEL_H = r"""// Evaluates a network exported with its model card by CFD Scientist
// (cfd_langgraph.nn_coupling). The TorchScript file carries the card's input
// normalisation, output scaling and bounds, so evaluate() takes the RAW
// features, in the card's order, and returns the card's output.
#ifndef nnModel_H
#define nnModel_H

// OpenFOAM's TypeName macro breaks libtorch's headers.
#pragma push_macro("TypeName")
#undef TypeName
#include <torch/script.h>
#include <ATen/Parallel.h>
#pragma pop_macro("TypeName")

#include "scalarField.H"
#include "List.H"
#include "fileName.H"
#include "error.H"

namespace Foam
{

class nnModel
{
    mutable torch::jit::script::Module module_;
    const label nInputs_;

public:

    nnModel(const fileName& path, const label nInputs)
    :
        nInputs_(nInputs)
    {
        // Each MPI rank is a process already: libtorch's own threads only
        // compete with the other ranks for the same cores.
        at::set_num_threads(1);
        try
        {
            module_ = torch::jit::load(path);
        }
        catch (const c10::Error& e)
        {
            FatalErrorInFunction
                << "cannot load TorchScript network " << path << nl << e.what()
                << exit(FatalError);
        }
        module_.eval();
    }

    //- The card's output for each element, from the raw features in the
    //  card's order, one field per feature, all of the same size
    tmp<scalarField> evaluate(const List<const scalarField*>& features) const
    {
        if (features.size() != nInputs_)
        {
            FatalErrorInFunction
                << "the network takes " << nInputs_ << " features, given "
                << features.size() << exit(FatalError);
        }
        const label n = features[0]->size();
        torch::Tensor x = torch::empty({n, nInputs_}, torch::kFloat64);
        double* xp = x.data_ptr<double>();
        for (label j = 0; j < nInputs_; j++)
        {
            const scalarField& f = *features[j];
            if (f.size() != n)
            {
                FatalErrorInFunction
                    << "feature " << j << " has " << f.size() << " values, feature 0 has " << n
                    << exit(FatalError);
            }
            for (label i = 0; i < n; i++)
            {
                xp[i*nInputs_ + j] = f[i];
            }
        }
        torch::NoGradGuard noGrad;
        torch::Tensor y =
            module_.forward({x}).toTensor().to(torch::kFloat64).contiguous().reshape({-1});
        if (y.size(0) != n)
        {
            FatalErrorInFunction
                << "the network returned " << y.size(0) << " values for " << n << " inputs"
                << exit(FatalError);
        }
        tmp<scalarField> tresult(new scalarField(n));
        scalarField& result = tresult.ref();
        const double* yp = y.data_ptr<double>();
        forAll(result, i)
        {
            result[i] = yp[i];
        }
        return tresult;
    }
};

} // End namespace Foam

#endif
"""

# Reads raw inputs (one row per point) and writes the network's outputs, one
# per line: OpenFOAM's own headers and libtorch compiled together, the way a
# case compiles them, with no case needed.
PARITY_APP = r"""#include "nnModel.H"
#include <fstream>
#include <iostream>
#include <vector>

using namespace Foam;

int main(int argc, char* argv[])
{
    if (argc != 5)
    {
        std::cerr << "usage: nnParity <model.pt> <nInputs> <inputs.txt> <outputs.txt>" << std::endl;
        return 2;
    }
    const label nIn = std::atoi(argv[2]);
    std::ifstream in(argv[3]);
    std::vector<double> values;
    double v;
    while (in >> v)
    {
        values.push_back(v);
    }
    const label n = values.size()/nIn;
    List<scalarField> columns(nIn, scalarField(n));
    for (label i = 0; i < n; i++)
    {
        for (label j = 0; j < nIn; j++)
        {
            columns[j][i] = values[i*nIn + j];
        }
    }
    List<const scalarField*> features(nIn);
    forAll(columns, j)
    {
        features[j] = &columns[j];
    }
    nnModel model(argv[1], nIn);
    tmp<scalarField> tout = model.evaluate(features);
    std::ofstream out(argv[4]);
    out.precision(17);
    forAll(tout(), i)
    {
        out << tout()[i] << '\n';
    }
    return 0;
}
"""


def torch_paths() -> Dict[str, str]:
    """libtorch's headers and libraries inside this environment's PyTorch."""
    import torch

    root = Path(torch.__file__).resolve().parent
    return {"include": str(root / "include"),
            "api_include": str(root / "include" / "torch" / "csrc" / "api" / "include"),
            "lib": str(root / "lib"), "version": torch.__version__}


def compile_flags(helper_dir: Optional[Path] = None) -> str:
    t = torch_paths()
    return " ".join(["-std=c++17", f"-I{t['include']}", f"-I{t['api_include']}"]
                    + ([f"-I{Path(helper_dir).resolve()}"] if helper_dir else []))


def link_flags() -> str:
    lib = torch_paths()["lib"]
    return f"-L{lib} -Wl,-rpath,{lib} -ltorch -ltorch_cpu -lc10"


def write_helper(dest_dir: Path) -> Path:
    """nnModel.H in ``dest_dir``."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / HELPER).write_text(NN_MODEL_H)
    return dest / HELPER


def coded_entries(helper_dir: Path) -> str:
    """The codeOptions, codeLibs and codeInclude entries that let a coded
    OpenFOAM object use nnModel.H (written into ``helper_dir``)."""
    write_helper(helper_dir)
    return ("codeOptions\n#{\n    " + compile_flags(helper_dir) + "\n#};\n\n"
            "codeLibs\n#{\n    " + link_flags() + "\n#};\n\n"
            "codeInclude\n#{\n    #include \"" + HELPER + "\"\n#};\n")


def build_parity_app(env: Dict[str, str], build_root: Path) -> Path:
    """The nnParity program built against the OpenFOAM ``env`` sets up, once
    per OpenFOAM installation and helper version (kept under build_root)."""
    project = Path(env.get("WM_PROJECT_DIR", "unknown")).name
    stamp = hashlib.sha256((NN_MODEL_H + PARITY_APP + compile_flags() + link_flags()
                            + env.get("WM_PROJECT_DIR", "")).encode()).hexdigest()[:12]
    work = Path(build_root) / f"nnParity_{project}_{stamp}"
    exe = work / "nnParity"
    if exe.is_file():
        return exe
    (work / "Make").mkdir(parents=True, exist_ok=True)
    write_helper(work)
    (work / "nnParity.C").write_text(PARITY_APP)
    (work / "Make" / "files").write_text(f"nnParity.C\n\nEXE = {exe}\n")
    (work / "Make" / "options").write_text(
        f"EXE_INC = {compile_flags(work)}\n\nEXE_LIBS = {link_flags()}\n")
    proc = subprocess.run(["wmake"], cwd=str(work), env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not exe.is_file():
        tail = (proc.stdout + proc.stderr)[-4000:]
        raise RuntimeError(f"nnParity did not build against {env.get('WM_PROJECT_DIR')}:\n{tail}")
    return exe


def run_parity_app(exe: Path, card_path: Path, env: Dict[str, str]) -> np.ndarray:
    """The card's parity inputs through nnParity: OpenFOAM's evaluation."""
    card_path = Path(card_path)
    card = ModelCard.load(card_path)
    work = Path(exe).parent
    inputs = work / f"{card.name}_inputs.txt"
    outputs = work / f"{card.name}_outputs.txt"
    np.savetxt(inputs, np.asarray(card.parity_inputs, dtype=float), fmt="%.17g")
    run_env = {**env, "OMP_NUM_THREADS": "1"}
    proc = subprocess.run([str(exe), str(card_path.parent / card.torchscript), str(len(card.inputs)),
                           str(inputs), str(outputs)], env=run_env, capture_output=True, text=True,
                          timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"nnParity failed: {(proc.stdout + proc.stderr)[-2000:]}")
    return np.loadtxt(outputs, ndmin=1)


def openfoam_env(openfoam_path: str = "") -> Dict[str, str]:
    """The environment of the OpenFOAM at ``openfoam_path``, or of the one
    already loaded in this process."""
    if openfoam_path:
        from cfd_langgraph.foam_native.openfoam_env import resolve_openfoam_env

        return resolve_openfoam_env(openfoam_path)
    return dict(os.environ)
