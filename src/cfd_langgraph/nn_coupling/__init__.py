"""Neural networks inside OpenFOAM: trained through the solver's adjoint, or
trained elsewhere in PyTorch and evaluated inside the solver.

Two ways in, one card:

  train in the solver   DAFoam trains a network (regressionModel, modelType
                        neuralNetwork) through its adjoint: the weights are a
                        design variable and one adjoint solve gives the
                        gradient of a flow objective for all of them.
                        dafoam.card_from_dafoam turns the trained weight
                        vector into the identical PyTorch network.
  bring a trained one   export.export wraps a PyTorch network with its input
                        normalisation, output scaling and bounds into one
                        TorchScript file; openfoam.nnModel.H evaluates it in
                        any OpenFOAM case from the raw features.

The card (card.py) holds everything the network was trained with that OpenFOAM
must reproduce, and verify.py holds OpenFOAM to it: check_parity on the
card's stored inputs, check_case on the case's own fields.

Command line: python -m cfd_langgraph.nn_coupling --help
"""

from .card import Feature, ModelCard  # noqa: F401
