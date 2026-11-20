#!/bin/bash
# Build olaFlow onto the shelf (run from the checkout, with the OpenFOAM
# environment sourced and FOAM_USER_APPBIN / FOAM_USER_LIBBIN pointing at the
# shelf build). Build configuration only; no source file is changed.
#
# The solver folder carries its own versions of two OpenFOAM libraries under
# the stock names (libincompressibleInterPhaseTransportModel, adding mut(), and
# libVoFTurbulenceDamping) and installs them into $(FOAM_LIBBIN), overwriting
# the installation's own. Here they are built into the shelf under names of
# their own, and the solver links those, so the stock libraries -- and every
# stock solver that loads them -- stay as they are.
set -e
for solver in solvers/olaFlowOF_org_latest solvers/olaFlowOFv_com_latest; do
    [ -d "$solver" ] || continue
    for lib in incompressibleInterPhaseTransportModel VoFTurbulenceDamping; do
        files="$solver/$lib/Make/files"
        [ -f "$files" ] || continue
        sed -i "s|^LIB *=.*|LIB = \$(FOAM_USER_LIBBIN)/libolaFlow_$lib|" "$files"
        sed -i "s|-l$lib\b|-lolaFlow_$lib|" "$solver/Make/options"
    done
done
grep -rl --include=files 'FOAM_LIBBIN)' genAbs applications | xargs -r sed -i 's|$(FOAM_LIBBIN)|$(FOAM_USER_LIBBIN)|'
./allMake
