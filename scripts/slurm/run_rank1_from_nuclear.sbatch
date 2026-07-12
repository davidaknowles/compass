#!/bin/bash
#SBATCH --job-name=compass-rank1-from-nuclear
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/rank1-from-nuclear-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/rank1-from-nuclear-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G

set -euo pipefail

: "${COMPASS_NUCLEAR_PREFIX:?set COMPASS_NUCLEAR_PREFIX}"
: "${COMPASS_RANK1_RUN_NAME:?set COMPASS_RANK1_RUN_NAME}"

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

lambda=$(python - "$COMPASS_NUCLEAR_PREFIX.metadata.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["best_lambda"])
PY
)

python scripts/run.py \
  --data-root /gpfs/commons/home/daknowles/knowles_lab/data/compass \
  --abc-context-panel ad_with_controls \
  --device cuda \
  --method rank1 \
  --init-b-tsv "$COMPASS_NUCLEAR_PREFIX.B.tsv" \
  --lambdas "$lambda" \
  --no-cv \
  --model-dtype float32 \
  --lr 0.001 \
  --max-iter 500 \
  --progress-every 10 \
  --run-name "$COMPASS_RANK1_RUN_NAME"
