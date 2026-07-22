#!/bin/bash
#SBATCH --job-name=compass-hier-genesum-fold
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --exclude=ne1dg7-011
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/hierarchical-genesum-fold-2026-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/hierarchical-genesum-fold-2026-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: run_hierarchical_genesum_2026_fold_b6k.sbatch TRAIT FOLD DEVIATION_CONSTRAINT" >&2
  exit 2
fi

trait=$1
fold=$2
constraint=$3
data_root=/gpfs/commons/home/daknowles/knowles_lab/data/compass

case "$constraint" in
  total_nonnegative) constraint_label=signed ;;
  nonnegative) constraint_label=nonnegative ;;
  *)
    echo "unknown deviation constraint: $constraint" >&2
    exit 2
    ;;
esac

case "$trait" in
  ad)
    analysis_label=ad2026
    gwas="$data_root/raw/ad_gwas_2026/GCST90704647.hg19.tsv.gz"
    effects="$data_root/results/ad2026_h3k27ac_sldsc_context_effects.tsv"
    ;;
  scz)
    analysis_label=scz2026
    gwas="$data_root/raw/scz_gwas_2026/SCZ_EUR_2026.hg19.tsv.gz"
    effects="$data_root/results/scz2026_h3k27ac_sldsc_context_effects.tsv"
    ;;
  pd)
    analysis_label=pd2025
    gwas="$data_root/raw/neuropsychiatric_gwas/pd/pd.hg19.tsv.gz"
    effects="$data_root/results/pd2025_h3k27ac_sldsc_context_effects.tsv"
    ;;
  bipolar)
    analysis_label=bipolar2025
    gwas="$data_root/raw/neuropsychiatric_gwas/bipolar/bipolar.hg19.tsv.gz"
    effects="$data_root/results/bipolar2025_h3k27ac_sldsc_context_effects.tsv"
    ;;
  mdd)
    analysis_label=mdd2025
    gwas="$data_root/raw/neuropsychiatric_gwas/mdd/mdd.hg19.tsv.gz"
    effects="$data_root/results/mdd2025_h3k27ac_sldsc_context_effects.tsv"
    ;;
  als)
    analysis_label=als2021
    gwas="$data_root/raw/neuropsychiatric_gwas/als/als.hg19.tsv.gz"
    effects="$data_root/results/als2021_h3k27ac_sldsc_context_effects.tsv"
    ;;
  anxiety)
    analysis_label=anxiety2024
    gwas="$data_root/raw/neuropsychiatric_gwas/anxiety/anxiety.hg19.tsv.gz"
    effects="$data_root/results/anxiety2024_h3k27ac_sldsc_context_effects.tsv"
    ;;
  *)
    echo "unknown trait: $trait" >&2
    exit 2
    ;;
esac

fold_args=()
if [[ "$fold" == all ]]; then
  run_name="compass-${analysis_label}-h3k27ac-genesum-hierarchical-${constraint_label}-ldcv10-fp32"
else
  run_name="compass-${analysis_label}-h3k27ac-genesum-hierarchical-${constraint_label}-ldcv10-fp32-fold${fold}"
  fold_args=(--cv-fold-subset "$fold")
fi
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
  --context-annotation sum \
  --context-annotation-source gene_aggregate \
  --context-effects-tsv "$effects" \
  --context-effects-mode scaled_frozen \
  --deviation-constraint "$constraint" \
  "${fold_args[@]}" \
  --device cuda \
  --model-dtype float32 \
  --lr 3e-7 \
  --lambdas 1e6,1e5,1e4,1e3,3e2,1e2,3e1,1e1 \
  --max-lambda-extensions 0 \
  --ld-jobs 4 \
  --run-name "$run_name"
