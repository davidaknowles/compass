#!/bin/bash
#SBATCH --job-name=compass-abc-sldsc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/abc-sldsc-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/abc-sldsc-%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/run_sldsc_baseline.py \
  --simulation-dir /gpfs/commons/home/daknowles/knowles_lab/data/compass/results/abc_recovery_simulation \
  --max-seed 2
