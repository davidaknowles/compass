#!/bin/bash
#SBATCH --job-name=compass-ad2026-setup
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/ad2026-setup-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/ad2026-setup-%j.err
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/run.py \
  --data-root /gpfs/commons/home/daknowles/knowles_lab/data/compass \
  --gwas /gpfs/commons/home/daknowles/knowles_lab/data/compass/raw/ad_gwas_2026/GCST90704647.hg19.tsv.gz \
  --abc-path /gpfs/commons/home/daknowles/knowles_lab/data/compass/raw/abc/glass_brain_v2.hg19.tsv.gz \
  --abc-cell-types astrocyte,microglia,neuron,oligodendrocyte \
  --device cpu \
  --ld-jobs 8 \
  --setup-only \
  --run-name compass-ad2026-glass-abc-setup
