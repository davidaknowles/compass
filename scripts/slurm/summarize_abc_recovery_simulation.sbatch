#!/bin/bash
#SBATCH --job-name=compass-abc-sim-summary
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/abc-sim-summary-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/abc-sim-summary-%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/summarize_abc_recovery_simulations.py \
  --results-dir /gpfs/commons/home/daknowles/knowles_lab/data/compass/results/abc_recovery_simulation
