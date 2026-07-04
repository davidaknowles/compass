#!/bin/bash
#SBATCH --job-name=compass-run
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/run-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/run-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
mkdir -p /gpfs/commons/home/daknowles/knowles_lab/data/compass/logs
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/run.py \
  --data-root /gpfs/commons/home/daknowles/knowles_lab/data/compass \
  ${COMPASS_RUN_ARGS:-}
