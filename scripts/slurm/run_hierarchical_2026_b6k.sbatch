#!/bin/bash
#SBATCH --job-name=compass-hierarchical-2026
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --exclude=ne1dg7-011
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/hierarchical-2026-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/hierarchical-2026-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: run_hierarchical_2026_b6k.sbatch TRAIT" >&2
  exit 2
fi

trait=$1
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
  --context-effects-mode scaled \
  --device cuda \
  --model-dtype float32 \
  --lambdas 1e6,1e5,1e4,1e3,3e2,1e2,3e1,1e1,3,1,3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5 \
  --max-lambda-extensions 4 \
  --lambda-extension-factor 3 \
  --ld-jobs 8 \
  --run-name "compass-${trait}2026-h3k27ac-flat-hierarchical-ldcv10-fp32"
