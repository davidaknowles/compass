#!/bin/bash
#SBATCH --job-name=compass-peak-annot
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/peak-annot-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/peak-annot-%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/build_baselineld_peak_annotations.py
