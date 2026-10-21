#!/bin/bash
# Launch one solver-shelf demonstration study (topic_shelf_<name>.md) in tmux.
#
#   bash scripts/run_shelf_demo.sh olaflow|additivefoam|solids4foam|turbinesfoam|dafoam|hystrath
#   bash scripts/run_shelf_demo.sh <name> --resume <run dir>    (then type 'continue' in the session)
#
# Each study runs from this repository (it has the shelf integration), with the
# OpenFOAM the solver's shelf build needs loaded: v2512 for solids4foam and
# turbinesFoam, OpenFOAM 10 for the others (DAFoam and hyStrath bring their own
# OpenFOAM and run through their wrappers whatever is loaded).
set -euo pipefail
name=${1:?usage: run_shelf_demo.sh <olaflow|additivefoam|solids4foam|turbinesfoam|dafoam|hystrath>}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OF10=/mnt/sda1/openfoam10
V2512=/mnt/sda1/openfoam_versions/v2512/usr/lib/openfoam/openfoam2512
case "$name" in
    olaflow)      topic=topic_shelf_olaflow_wave_flume.md;          of=$OF10 ;;
    additivefoam) topic=topic_shelf_additivefoam_melt_pool.md;      of=$OF10 ;;
    solids4foam)  topic=topic_shelf_solids4foam_turek_hron.md;      of=$V2512 ;;
    turbinesfoam) topic=topic_shelf_turbinesfoam_rvat.md;           of=$V2512 ;;
    dafoam)       topic=topic_shelf_dafoam_field_inversion.md; of=$OF10 ;;
    hystrath)     topic=topic_shelf_hystrath_continuum_breakdown.md;    of=$OF10 ;;
    *) echo "unknown demo: $name" >&2; exit 2 ;;
esac
if [ "${2:-}" = "--resume" ]; then
    out=${3:?--resume needs the run directory}
    cli="resume --out-dir $out"
else
    out=/mnt/sda1/somasn_cfd_scientist_experiments/shelf_demo_${name}_$(date +%Y%m%d_%H%M)
    cli="run --topic-file $topic --out-dir $out"
fi
session=shelfdemo_$name
tmux has-session -t "$session" 2>/dev/null && { echo "tmux session $session already exists" >&2; exit 2; }
tmux new-session -d -s "$session" -c "$REPO" \
  "source ~/anaconda3/etc/profile.d/conda.sh && conda activate cfd-scientist \
   && set +u && source $of/etc/bashrc; set -u; \
   export CFD_SCIENTIST_SHELF=/mnt/sda1/cfd_solver_shelf \
          CFD_SCIENTIST_LLM_PROVIDER=openai-codex CFD_SCIENTIST_MODEL=gpt-5.6-sol CFD_SCIENTIST_EFFORT=high \
   && python scripts/cfd_cli.py $cli 2>&1 | tee -a $out.log; sleep infinity"
echo "launched $name: tmux $session, run dir $out"
