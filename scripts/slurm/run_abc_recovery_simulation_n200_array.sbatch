#!/bin/bash
#SBATCH --job-name=compass-abc-sim-n200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --array=0-2%2
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/abc-sim-n200-%A_%a.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/abc-sim-n200-%A_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/simulate_abc_recovery.py --n-eff 200000 --seed "${SLURM_ARRAY_TASK_ID:?}" --device cuda
