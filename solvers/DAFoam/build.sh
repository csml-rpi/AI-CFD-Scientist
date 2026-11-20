#!/bin/bash
# Build DAFoam v5.1.1 locally, with no root and no container, following its
# from-source instructions (DAFoam.github.io pages/mydoc/mydoc_installation_source.md):
# its own OpenFOAM v2506 and the reverse-mode AD copy of it (OpenFOAM-AD, the
# Jacobian-free adjoint), PETSc, CGNS, IPOPT, the MACH-Aero packages, pyOFM,
# Hisa4DAFoam, OpenMDAO and Mphys, all inside this build folder. Optional parts
# the shelf does not use are left out: OpenVSP, SMT, Uno, FUNtoFEM, TACS and the
# forward-mode AD copy.
#
# Run by scripts/shelf_install.py from the DAFoam checkout (the build's src/),
# in a sandbox where only the build folder is writable. Every step leaves a
# stamp, so a build that stopped resumes where it stopped. DAFoam brings its
# own OpenFOAM: the installed one is never used, so the build starts from a
# clean environment.
set -eo pipefail

if [ -z "${DAFOAM_CLEAN_ENV:-}" ]; then
    exec env -i DAFOAM_CLEAN_ENV=1 USER="${USER:-}" LOGNAME="${LOGNAME:-}" TERM=dumb LANG=C.UTF-8 \
        PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp JOBS="${WM_NCOMPPROCS:-16}" \
        SRC="$(pwd)" bash "$0" "$@"
fi

BUILD="$(dirname "$SRC")"
export HOME="$BUILD/home"                 # pip, conda and MPI write here, not to the real home
export DAFOAM_ROOT_PATH="$BUILD/dafoam"
mkdir -p "$HOME" "$DAFOAM_ROOT_PATH"/{packages,repos,OpenFOAM/sharedBins,OpenFOAM/sharedLibs}
cd "$DAFOAM_ROOT_PATH"

done_() { [ -f "$DAFOAM_ROOT_PATH/.done_$1" ]; }
mark() { touch "$DAFOAM_ROOT_PATH/.done_$1"; echo "=== done: $1"; }
# OpenFOAM's bashrc may end on a non-zero status; that is not a failure.
load() { set +u; . "$DAFOAM_ROOT_PATH/loadDAFoam.sh" || true; }

# The environment file, as the instructions leave it; each part is skipped
# until it is built.
cat > loadDAFoam.sh <<'EOF'
#!/bin/bash
# DAFoam root path
export DAFOAM_ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The original (non-AD) mode. Sourcing OpenFOAM-AD's bashrc exports
# WM_AD_MODE=ADR, and DAFoam's and Hisa4DAFoam's Allmake build their
# original mode right after loading this file: a WM_AD_MODE left over turns
# that pass into a second ADR build, and the original libraries
# (libDASolver, libHisaSolver, pyDASolvers) are never made.
unset WM_AD_MODE
# Miniconda3
export PATH=$DAFOAM_ROOT_PATH/packages/miniconda3/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DAFOAM_ROOT_PATH/packages/miniconda3/lib
export PYTHONUSERBASE=no-local-libs
# Petsc
export PETSC_DIR=$DAFOAM_ROOT_PATH/packages/petsc-3.21.6
export PETSC_ARCH=real-opt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/$PETSC_ARCH/lib
export PETSC_LIB=$PETSC_DIR/$PETSC_ARCH/lib
# CGNS
export CGNS_HOME=$DAFOAM_ROOT_PATH/packages/CGNS-4.5.0/opt-gfortran
export PATH=$PATH:$CGNS_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CGNS_HOME/lib
# Ipopt
export IPOPT_DIR=$DAFOAM_ROOT_PATH/packages/Ipopt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$IPOPT_DIR/lib
export PKG_CONFIG_PATH=$IPOPT_DIR/lib/pkgconfig:$PKG_CONFIG_PATH
# OpenFOAM-v2506
if [ -f $DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-v2506/etc/bashrc ]; then
    source $DAFOAM_ROOT_PATH/OpenFOAM/OpenFOAM-v2506/etc/bashrc
fi
export LD_LIBRARY_PATH=$DAFOAM_ROOT_PATH/OpenFOAM/sharedLibs:$LD_LIBRARY_PATH
export PATH=$DAFOAM_ROOT_PATH/OpenFOAM/sharedBins:$PATH
EOF
chmod 755 loadDAFoam.sh
load

# ---------------------------------------------------------------- Python
if ! done_ python; then
    cd "$DAFOAM_ROOT_PATH/packages"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py312_26.1.1-1-Linux-x86_64.sh
    bash ./Miniconda3-py312_26.1.1-1-Linux-x86_64.sh -b -p "$DAFOAM_ROOT_PATH/packages/miniconda3"
    # Its own libstdc++, libtinfo, linker, libgcc_s and libquadmath clash with the system's.
    m="$DAFOAM_ROOT_PATH/packages/miniconda3"
    for f in lib/libstdc++.so.6 lib/libtinfo.so.6 compiler_compat/ld lib/libgcc_s.so.1 lib/libquadmath.so.0; do
        [ -e "$m/$f" ] && mv "$m/$f" "$m/$f.backup"
    done
    load
    pip install --upgrade pip
    pip install numpy==2.3.5 scipy==1.17.1 cython==3.0.5 numpy-stl==2.16.0 nptyping==1.4.4
    wget -q https://github.com/mpi4py/mpi4py/releases/download/4.1.1/mpi4py-4.1.1.tar.gz
    tar -xf mpi4py-4.1.1.tar.gz && (cd mpi4py-4.1.1 && pip install .)
    mark python
fi
load

# ---------------------------------------------------------------- PETSc
if ! done_ petsc; then
    cd "$DAFOAM_ROOT_PATH/packages"
    wget -q https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-3.21.6.tar.gz
    tar -xf petsc-3.21.6.tar.gz
    cd petsc-3.21.6
    ./configure --PETSC_ARCH=real-opt --with-scalar-type=real --with-debugging=0 --download-metis=yes \
        --download-parmetis=yes --download-superlu_dist=yes --download-fblaslapack=yes \
        --download-f2cblaslapack=yes --with-shared-libraries=yes --with-fortran-bindings=1 \
        --with-cxx-dialect=C++11 --with-make-np="$JOBS"
    make PETSC_DIR="$PETSC_DIR" PETSC_ARCH=real-opt all
    (cd src/binding/petsc4py && pip install . --no-build-isolation)
    mark petsc
fi

# ---------------------------------------------------------------- CGNS
if ! done_ cgns; then
    cd "$DAFOAM_ROOT_PATH/packages"
    wget -q https://github.com/CGNS/CGNS/archive/v4.5.0.tar.gz
    tar -xaf v4.5.0.tar.gz
    mkdir -p CGNS-4.5.0/build && cd CGNS-4.5.0/build
    cmake .. -DCGNS_ENABLE_FORTRAN=1 -DCMAKE_INSTALL_PREFIX="$CGNS_HOME" -DCGNS_BUILD_CGNSTOOLS=0 \
        -DCGNS_ENABLE_HDF5=OFF -DCGNS_ENABLE_64BIT=OFF -DCMAKE_C_FLAGS="-fPIC" -DCMAKE_Fortran_FLAGS="-fPIC"
    make -j "$JOBS" all install
    mark cgns
fi

# ---------------------------------------------------------------- IPOPT
if ! done_ ipopt; then
    cd "$DAFOAM_ROOT_PATH/packages"
    git clone -q --depth 1 -b stable/3.13 https://github.com/coin-or/Ipopt.git
    cd "$IPOPT_DIR"
    git clone -q --depth 1 -b stable/1.3 https://github.com/coin-or-tools/ThirdParty-Blas.git
    (cd ThirdParty-Blas && ./get.Blas && ./configure --prefix="$IPOPT_DIR" && make && make install)
    git clone -q --depth 1 -b stable/1.5 https://github.com/coin-or-tools/ThirdParty-Lapack.git
    (cd ThirdParty-Lapack && ./get.Lapack && ./configure --prefix="$IPOPT_DIR" \
        --with-blas-lflags="-L${IPOPT_DIR}/lib -lcoinblas" && make && make install)
    git clone -q --depth 1 -b stable/1.3 https://github.com/coin-or-tools/ThirdParty-Metis.git
    (cd ThirdParty-Metis && ./get.Metis && ./configure --prefix="$IPOPT_DIR" && make && make install)
    git clone -q --depth 1 -b stable/2.1 https://github.com/coin-or-tools/ThirdParty-Mumps.git
    (cd ThirdParty-Mumps && ./get.Mumps && ./configure --prefix="$IPOPT_DIR" \
        --with-blas-lflags="-L${IPOPT_DIR}/lib -lcoinblas" \
        --with-metis-lflags="-L${IPOPT_DIR}/lib -lcoinmetis" \
        --with-metis-cflags="-I${IPOPT_DIR}/include/coin/ThirdParty" \
        --with-lapack-lflags="-L${IPOPT_DIR}/lib -lcoinlapack" && make && make install)
    mkdir -p build && cd build
    ../configure --prefix="${IPOPT_DIR}" --disable-java --with-mumps \
        --with-mumps-lflags="-L${IPOPT_DIR}/lib -lcoinmumps" \
        --with-mumps-cflags="-I${IPOPT_DIR}/include/coin-or/mumps" \
        --with-blas-lflags="-L${IPOPT_DIR}/lib -lcoinblas" \
        --with-metis-lflags="-L${IPOPT_DIR}/lib -lcoinmetis" \
        --with-metis-cflags="-I${IPOPT_DIR}/include/coin/ThirdParty" \
        --with-lapack-lflags="-L${IPOPT_DIR}/lib -lcoinlapack"
    make && make install
    wget -q https://github.com/mechmotum/cyipopt/archive/refs/tags/v1.7.0.tar.gz -O cyipopt.tar.gz
    tar -xf cyipopt.tar.gz && mv cyipopt-* cyipopt && (cd cyipopt && pip install .)
    mark ipopt
fi

# ---------------------------------------------------------------- MACH-Aero
get() {  # get <url> <archive name>: download and unpack into repos/
    cd "$DAFOAM_ROOT_PATH/repos" && wget -q "$1" -O "$2" && tar -xf "$2"
}
if ! done_ machaero; then
    get https://github.com/mdolab/baseclasses/archive/v1.9.0.tar.gz baseclasses.tar.gz
    (cd baseclasses-1.9.0 && pip install .)
    get https://github.com/mdolab/pyspline/archive/v1.5.4.tar.gz pyspline.tar.gz
    (cd pyspline-1.5.4 && cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && make && pip install .)
    get https://github.com/mdolab/pygeo/archive/v1.17.0.tar.gz pygeo.tar.gz
    (cd pygeo-1.17.0 && pip install .)
    get https://github.com/mdolab/multipoint/archive/v1.4.2.tar.gz multipoint.tar.gz
    (cd multipoint-1.4.2 && pip install .)
    get https://github.com/mdolab/cgnsutilities/archive/v2.9.0.tar.gz cgnsutilities.tar.gz
    (cd cgnsutilities-2.9.0 && cp config/defaults/config.LINUX_GFORTRAN.mk config/config.mk && make && pip install .)
    get https://github.com/mdolab/idwarp/archive/v2.6.4.tar.gz idwarp.tar.gz
    (cd idwarp-2.6.4 && cp -r config/defaults/config.LINUX_GFORTRAN.mk config/config.mk \
        && sed -i "s/mpifort/mpif90/g" config/config.mk && make && pip install .)
    get https://github.com/mdolab/prefoil/archive/v2.0.1.tar.gz prefoil.tar.gz
    (cd prefoil-2.0.1 && pip install .)
    get https://github.com/mdolab/pyoptsparse/archive/v2.16.0.tar.gz pyoptsparse.tar.gz
    (cd pyoptsparse-2.16.0 && pip install .)
    mark machaero
fi

# ---------------------------------------------------------------- OpenFOAM v2506
if ! done_ openfoam; then
    cd "$DAFOAM_ROOT_PATH/OpenFOAM"
    if [ ! -f OpenFOAM-v2506/.unpacked ]; then
        wget -q https://sourceforge.net/projects/openfoam/files/v2506/OpenFOAM-v2506.tgz/download -O OpenFOAM-v2506.tgz
        wget -q https://sourceforge.net/projects/openfoam/files/v2506/ThirdParty-v2506.tgz/download -O ThirdParty-v2506.tgz
        tar -xf OpenFOAM-v2506.tgz && tar -xf ThirdParty-v2506.tgz
        wget -q https://github.com/DAFoam/files/releases/download/v1.0.0/UPstream_OF2506_Patch.C \
            -O OpenFOAM-v2506/src/Pstream/mpi/UPstream.C
        touch OpenFOAM-v2506/.unpacked
    fi
    cd OpenFOAM-v2506
    load
    export WM_QUIET=true WM_NCOMPPROCS="$JOBS"
    ./Allwmake
    simpleFoam -help > /dev/null
    mark openfoam
fi
load

# ---------------------------------------------------------------- OpenFOAM-AD, reverse mode
if ! done_ openfoam_adr; then
    cd "$DAFOAM_ROOT_PATH/OpenFOAM"
    if [ ! -d OpenFOAM-AD ]; then
        wget -q https://github.com/DAFoam/OpenFOAM-AD/archive/2e519c1d4d5ad32cb2f42b7298f9087ca243b9c3.tar.gz -O OpenFOAM-AD.tgz
        tar -xf OpenFOAM-AD.tgz && mv OpenFOAM-AD-2e519c1d4d5ad32cb2f42b7298f9087ca243b9c3 OpenFOAM-AD
    fi
    cd OpenFOAM-AD
    sed -i 's/export WM_AD_MODE=.*/export WM_AD_MODE=ADR/g' etc/bashrc
    set +u; source etc/bashrc || true
    export WM_QUIET=true WM_NCOMPPROCS="$JOBS"
    ./Allwmake
    simpleFoamADR -help > /dev/null
    # renameAD.sh needs patchelf, which the instructions take from apt; without
    # root it comes from PyPI as a prebuilt binary into DAFoam's own Python.
    command -v patchelf > /dev/null || pip install patchelf==0.17.2.1
    ./renameAD.sh platforms/linux*ADR --ADR --commit
    load
    cd "$DAFOAM_ROOT_PATH"/OpenFOAM/OpenFOAM-v2506/platforms/*/lib
    ln -s ../../../../OpenFOAM-AD/platforms/linux*ADR/lib/*.so .
    cd "$DAFOAM_ROOT_PATH"/OpenFOAM/OpenFOAM-v2506/platforms/*/lib/dummy
    ln -s ../../../../../OpenFOAM-AD/platforms/linux*ADR/lib/dummy/*.so .
    cd "$DAFOAM_ROOT_PATH"/OpenFOAM/OpenFOAM-v2506/platforms/*/lib/$FOAM_MPI
    ln -s ../../../../../OpenFOAM-AD/platforms/linux*ADR/lib/$FOAM_MPI/*.so .
    mark openfoam_adr
fi
load

# ---------------------------------------------------------------- pyOFM, Hisa4DAFoam
if ! done_ pyofm; then
    get https://github.com/mdolab/pyofm/archive/refs/tags/v1.2.3.tar.gz pyofm.tar.gz
    (cd pyofm-1.2.3 && make && pip install .)
    mark pyofm
fi
if ! done_ hisa; then
    cd "$DAFOAM_ROOT_PATH/OpenFOAM"
    if [ ! -d Hisa4DAFoam ]; then
        wget -q https://github.com/DAFoam/Hisa4DAFoam/archive/73ced10b6dffc8aa2fb5433a787c74dc21f313da.tar.gz -O Hisa4DAFoam.tgz
        tar -xf Hisa4DAFoam.tgz && mv Hisa4DAFoam-73ced10b6dffc8aa2fb5433a787c74dc21f313da Hisa4DAFoam
    fi
    (cd Hisa4DAFoam && ./Allclean && ./Allmake)
    mark hisa
fi

# ---------------------------------------------------------------- DAFoam (original and reverse-mode AD)
load
cd "$SRC"
./Allmake
load

# ---------------------------------------------------------------- MDO packages
if ! done_ mdo; then
    pip install openmdao==3.45
    get https://github.com/OpenMDAO/mphys/archive/2080c39e86eee4c7069a41898181fb0e92cd4b93.tar.gz mphys.tar.gz
    mv mphys-2080c39e86eee4c7069a41898181fb0e92cd4b93 mphys && (cd mphys && pip install .)
    mark mdo
fi

# ---------------------------------------------------------------- regression cases
# The shelf gives studies DAFoam's tests/ as its example cases, and those tests
# run on cases their Allrun downloads and unpacks there as reg_test_files-main/.
# Unpack the pinned copy the shelf's checks use (checks.py) in the same place,
# so a study finds the cases where the tests' scripts look for them.
REG_FILES=927974649273962a7f60388a781e330e2a85e6f9
if [ ! -d "$SRC/tests/reg_test_files-main" ]; then
    cd "$SRC/tests"
    wget -q "https://github.com/DAFoam/reg_test_files/archive/$REG_FILES.tar.gz" -O reg_test_files.tar.gz
    echo "af21851e0ba08f6668bbed62447cc4f1096a935d884f2371b2d6e3d88e2ddff5  reg_test_files.tar.gz" \
        | sha256sum -c -
    tar -xf reg_test_files.tar.gz
    mv "reg_test_files-$REG_FILES" reg_test_files-main
    rm reg_test_files.tar.gz
    cd "$DAFOAM_ROOT_PATH"
fi
# Those cases are variants each regression script assembles before it runs,
# so none runs as shipped. ConvergentChannel is kept ready to run, assembled
# as runRegTests_DASimpleFoamForward.py does, with the case's own blockMeshDict
# (the 343-cell mesh it ships) and a runScript.py that solves it once.
CHANNEL="$BUILD/tutorials_ported/tests/reg_test_files-main/ConvergentChannel"
if [ ! -d "$CHANNEL" ]; then
    REG="$SRC/tests/reg_test_files-main/ConvergentChannel"
    mkdir -p "$CHANNEL"
    cp -r "$REG/constant" "$REG/system" "$REG/FFD" "$CHANNEL/"
    rm -rf "$CHANNEL/constant/polyMesh"
    cp -r "$REG/0.incompressible" "$CHANNEL/0"
    for f in "$REG"/system.incompressible/*; do
        [ "$(basename "$f")" = blockMeshDict ] || cp "$f" "$CHANNEL/system/"
    done
    cp "$CHANNEL/constant/turbulenceProperties.sa" "$CHANNEL/constant/turbulenceProperties"
    cp "$(dirname "$0")/convergent_channel_runScript.py" "$CHANNEL/runScript.py"
    printf '{"source": "%s", "case": "%s", "status": "translated"}\n' "$REG" "$CHANNEL" \
        > "$CHANNEL.translation.json"
fi

# ---------------------------------------------------------------- what studies call
# A case runs DAFoam through bin/dafoam-run, which starts from a clean
# environment and loads DAFoam's own: it never mixes with the installed OpenFOAM.
mkdir -p "$BUILD/bin" "$BUILD/lib"
# OpenFOAM's bashrc reads the shell's positional arguments as its own settings,
# so the wrapper sets the command aside before sourcing it.
cat > "$BUILD/bin/dafoam-run" <<EOF
#!/bin/bash
# Run a command in DAFoam's environment (its own OpenFOAM v2506), e.g.
#   dafoam-run mpirun -np 4 python runScript.py
# Already inside (an Allrun the wrapper runs calls it again): run as it is.
[ "\${CFD_SHELF_RUNTIME:-}" = DAFoam ] && exec "\$@"
exec env -i CFD_SHELF_RUNTIME=DAFoam HOME="\${HOME}" USER="\${USER:-}" TERM="\${TERM:-dumb}" LANG=C.UTF-8 \\
    PATH="$BUILD/bin":/usr/local/bin:/usr/bin:/bin TMPDIR="\${TMPDIR:-/tmp}" \\
    OMP_NUM_THREADS="\${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="\${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="\${MKL_NUM_THREADS:-1}" \\
    bash -c 'cmd=("\$@"); set --; set +u; . "$DAFOAM_ROOT_PATH/loadDAFoam.sh"; exec "\${cmd[@]}"' dafoam-run "\$@"
EOF
chmod 755 "$BUILD/bin/dafoam-run"
"$BUILD/bin/dafoam-run" python -c "import dafoam, mphys, openmdao, pygeo, idwarp, pyofm; print('dafoam', dafoam.__file__)"
echo "*** DAFoam build complete ***"
