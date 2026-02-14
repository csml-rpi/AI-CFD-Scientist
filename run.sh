python src/main.py --auto-rerun --auto-rerun-threshold 5.0 --max-rerun-iterations 1 --auto-execute-recommendations

BATCH="$(ls -dt data/experiments/batch_* | head -n 1 | xargs basename)"
python agents/analysis_ag.py --batch "$BATCH" --auto-rerun --auto-rerun-threshold 5.0 --max-rerun-iterations 1

python src/latexpaper.py --batch "$BATCH"
