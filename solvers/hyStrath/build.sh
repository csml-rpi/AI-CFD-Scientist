#!/bin/bash
# Build hyStrath (CFD: hy2Foam; DSMC: dsmcFoam+; hybrid PIC-DSMC: pdFoam)
# locally, with no root and no container, on the OpenFOAM it was written for:
# OpenFOAM v1706, built here from the release sources. Nobody has ported
# hyStrath past v1706, so it brings its own OpenFOAM, as DAFoam does.
#
# Run by scripts/shelf_install.py from the hyStrath checkout (the build's src/),
# in a sandbox where only the build folder is writable. HOME is the build's
# home/, so OpenFOAM v1706 sits where its bashrc expects it
# ($HOME/OpenFOAM/OpenFOAM-v1706) and hyStrath's installers put their
# programs and libraries in $HOME/OpenFOAM/$USER-v1706, never on the
# installed OpenFOAM's path. Every step leaves a stamp, so a build that
# stopped resumes where it stopped.
set -eo pipefail

if [ -z "${HYSTRATH_CLEAN_ENV:-}" ]; then
    exec env -i HYSTRATH_CLEAN_ENV=1 USER="${USER:-$(id -un)}" LOGNAME="${LOGNAME:-}" TERM=dumb \
        LANG=C.UTF-8 PATH=/usr/local/bin:/usr/bin:/bin TMPDIR=/tmp JOBS="${WM_NCOMPPROCS:-16}" \
        SHELF_RECIPE_DIR="${SHELF_RECIPE_DIR:-}" SRC="$(pwd)" bash "$0" "$@"
fi

BUILD="$(dirname "$SRC")"
export HOME="$BUILD/home"
OF="$HOME/OpenFOAM"
mkdir -p "$OF"
cd "$OF"

done_() { [ -f "$OF/.done_$1" ]; }
mark() { touch "$OF/.done_$1"; echo "=== done: $1"; }
# OpenFOAM's bashrc may end on a non-zero status; that is not a failure.
load() { set +u; . "$OF/OpenFOAM-v1706/etc/bashrc" || true; }

# ---------------------------------------------------------------- OpenFOAM v1706
if ! done_ openfoam; then
    if [ ! -d OpenFOAM-v1706 ]; then
        wget -q https://sourceforge.net/projects/openfoam/files/v1706/OpenFOAM-v1706.tgz/download -O OpenFOAM-v1706.tgz
        wget -q https://sourceforge.net/projects/openfoam/files/v1706/ThirdParty-v1706.tgz/download -O ThirdParty-v1706.tgz
        tar -xzf OpenFOAM-v1706.tgz && tar -xzf ThirdParty-v1706.tgz
    fi
    # Changes a 2017 release needs to compile with today's compilers, each
    # explained in the script; safe to apply again on a resumed build.
    bash "$SHELF_RECIPE_DIR/patches_v1706.sh" "$OF/OpenFOAM-v1706" "$OF/ThirdParty-v1706"
    load
    export WM_NCOMPPROCS="$JOBS"
    (cd "$OF/ThirdParty-v1706" && ./Allwmake)
    load
    (cd "$OF/OpenFOAM-v1706" && ./Allwmake -j "$JOBS")
    load
    simpleFoam -help > /dev/null
    rhoCentralFoam -help > /dev/null
    mark openfoam
fi
load

# ---------------------------------------------------------------- hyStrath
# Its own installers: each copies its sources into $WM_PROJECT_USER_DIR and
# compiles them there, ending with "... compiled successfully".
# hyStrath's own copies of OpenFOAM classes carry the same gcc 11 fixes.
bash "$SHELF_RECIPE_DIR/patches_v1706.sh" "$OF/OpenFOAM-v1706" "$OF/ThirdParty-v1706" "$SRC"
for module in CFD DSMC hybridPICDSMC; do
    if ! done_ "hystrath_$module"; then
        (cd "$SRC" && bash ./build/install-$module.sh "$JOBS" > "$OF/install-$module.log" 2>&1) || true
        tail -n 3 "$OF/install-$module.log"
        grep -q "compiled successfully" "$OF/install-$module.log"
        mark "hystrath_$module"
    fi
done
load
for program in hy2Foam dsmcFoam+ dsmcInitialise+ pdFoam; do
    command -v "$program" > /dev/null || { echo "missing after the build: $program"; exit 1; }
done

# ---------------------------------------------------------------- what studies call
# A case runs hyStrath through bin/hystrath-run, which starts from a clean
# environment and loads OpenFOAM v1706 with hyStrath: it never mixes with the
# installed OpenFOAM.
mkdir -p "$BUILD/bin" "$BUILD/lib"
cat > "$BUILD/bin/hystrath-run" <<EOF
#!/bin/bash
# Run a command with hyStrath on its own OpenFOAM v1706, e.g.
#   hystrath-run bash ./Allrun        hystrath-run dsmcFoam+
# Already inside (an Allrun the wrapper runs calls it again): run as it is.
[ "\${CFD_SHELF_RUNTIME:-}" = hyStrath ] && exec "\$@"
exec env -i CFD_SHELF_RUNTIME=hyStrath HOME="$HOME" USER="\${USER:-\$(id -un)}" TERM="\${TERM:-dumb}" LANG=C.UTF-8 \\
    PATH="$BUILD/bin":/usr/local/bin:/usr/bin:/bin TMPDIR="\${TMPDIR:-/tmp}" \\
    bash -c 'cmd=("\$@"); set --; set +u; . "$OF/OpenFOAM-v1706/etc/bashrc"; exec "\${cmd[@]}"' hystrath-run "\$@"
EOF
chmod 755 "$BUILD/bin/hystrath-run"
"$BUILD/bin/hystrath-run" bash -c 'command -v dsmcFoam+ hy2Foam'
echo "*** hyStrath build complete ***"
