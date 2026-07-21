#!/bin/bash
#SBATCH --job-name=compass-hier-fold-2026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --exclude=ne1dg7-011
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/hierarchical-fold-2026-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/hierarchical-fold-2026-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: run_hierarchical_2026_fold_b6k.sbatch TRAIT FOLD" >&2
  exit 2
fi

trait=$1
fold=$2
data_root=/gpfs/commons/home/daknowles/knowles_lab/data/compass
case "$trait" in
  ad)
    gwas="$data_root/raw/ad_gwas_2026/GCST90704647.hg19.tsv.gz"
    effects="$data_root/results/ad2026_h3k27ac_sldsc_context_effects.tsv"
    ;;
  scz)
    gwas="$data_root/raw/scz_gwas_2026/SCZ_EUR_2026.hg19.tsv.gz"
    effects="$data_root/results/scz2026_h3k27ac_sldsc_context_effects.tsv"
    ;;
  *)
    echo "unknown trait: $trait" >&2
    exit 2
    ;;
esac

run_name="compass-${trait}2026-h3k27ac-flat-hierarchical-ldcv10-fp32-fold${fold}"
cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/run.py \
  --data-root "$data_root" \
  --gwas "$gwas" \
  --annotation-source open_chromatin \
  --peak-assay H3K27ac \
  --method hierarchical \
  --context-annotation binary \
  --context-annotation-source peak \
  --context-effects-tsv "$effects" \
  --context-effects-mode scaled_frozen \
  --cv-fold-subset "$fold" \
  --device cuda \
  --model-dtype float32 \
  --lr 3e-7 \
  --lambdas 1e6,1e5,1e4,1e3,3e2,1e2,3e1,1e1 \
  --max-lambda-extensions 0 \
  --lambda-extension-factor 3 \
  --ld-jobs 8 \
  --run-name "$run_name"
