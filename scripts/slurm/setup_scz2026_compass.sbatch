#!/bin/bash
#SBATCH --job-name=compass-scz2026-setup
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/scz2026-setup-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/scz2026-setup-%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/run.py \
  --data-root /gpfs/commons/home/daknowles/knowles_lab/data/compass \
  --gwas /gpfs/commons/home/daknowles/knowles_lab/data/compass/raw/scz_gwas_2026/SCZ_EUR_2026.hg19.tsv.gz \
  --abc-path /gpfs/commons/home/daknowles/knowles_lab/data/compass/raw/abc/glass_brain_v2.hg19.tsv.gz \
  --abc-cell-types astrocyte,microglia,neuron,oligodendrocyte \
  --device cpu \
  --ld-jobs 8 \
  --setup-only \
  --run-name compass-scz2026-glass-abc-setup
