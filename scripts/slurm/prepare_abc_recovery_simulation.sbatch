#!/bin/bash
#SBATCH --job-name=compass-abc-sim-prepare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/abc-sim-prepare-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/abc-sim-prepare-%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/simulate_abc_recovery.py --prepare-only --device cuda
