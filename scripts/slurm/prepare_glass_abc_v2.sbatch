#!/bin/bash
#SBATCH --job-name=compass-glass-abc-prep
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/glass-abc-prep-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/glass-abc-prep-%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
module load Kent_tools/461-GCC-12.2.0
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/prepare_glass_abc_v2.py
