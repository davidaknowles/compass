#!/bin/bash
#SBATCH --job-name=compass-ad2026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/ad2026-compass-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/ad2026-compass-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: run_ad2026_compass_b6k.sbatch TARGET" >&2
  exit 2
fi
target=$1
data_root=/gpfs/commons/home/daknowles/knowles_lab/data/compass
gwas="$data_root/raw/ad_gwas_2026/GCST90704647.hg19.tsv.gz"

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

case "$target" in
  glass)
    args=(--abc-path "$data_root/raw/abc/glass_brain_v2.hg19.tsv.gz" --abc-cell-types astrocyte,microglia,neuron,oligodendrocyte --run-name compass-ad2026-glass-abc-ldcv10-fp32)
    ;;
  open_chromatin)
    args=(--annotation-source open_chromatin --open-chromatin-tss-window 100000 --run-name compass-ad2026-open-chromatin-tss100kb-ldcv10-fp32)
    ;;
  public_abc)
    args=(--abc-context-panel ad_proximal --run-name compass-ad2026-public-abc-ldcv10-fp32)
    ;;
  public_abc_controls)
    args=(--abc-context-panel ad_with_controls --run-name compass-ad2026-public-abc-controls-ldcv10-fp32)
    ;;
  *)
    echo "Unknown target: $target" >&2
    exit 2
    ;;
esac

python scripts/run.py \
  --data-root "$data_root" \
  --gwas "$gwas" \
  --device cuda \
  --model-dtype float32 \
  --ld-jobs 8 \
  "${args[@]}"
