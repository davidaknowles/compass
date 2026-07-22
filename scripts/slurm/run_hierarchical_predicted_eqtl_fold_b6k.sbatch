#!/bin/bash
#SBATCH --job-name=compass-pred-eqtl-fold
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --exclude=ne1dg7-011
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/pred-eqtl-fold-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/pred-eqtl-fold-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G

set -euo pipefail
trait=${1:?usage: run_hierarchical_predicted_eqtl_fold_b6k.sbatch TRAIT FOLD}
fold=${2:?usage: run_hierarchical_predicted_eqtl_fold_b6k.sbatch TRAIT FOLD}
data_root=/gpfs/commons/home/daknowles/knowles_lab/data/compass
case "$trait" in
  pd) label=pd2025; gwas="$data_root/raw/neuropsychiatric_gwas/pd/pd.hg19.tsv.gz" ;;
  bipolar) label=bipolar2025; gwas="$data_root/raw/neuropsychiatric_gwas/bipolar/bipolar.hg19.tsv.gz" ;;
  mdd) label=mdd2025; gwas="$data_root/raw/neuropsychiatric_gwas/mdd/mdd.hg19.tsv.gz" ;;
  als) label=als2021; gwas="$data_root/raw/neuropsychiatric_gwas/als/als.hg19.tsv.gz" ;;
  anxiety) label=anxiety2024; gwas="$data_root/raw/neuropsychiatric_gwas/anxiety/anxiety.hg19.tsv.gz" ;;
  *) echo "unknown trait: $trait" >&2; exit 2 ;;
esac
effects="$data_root/results/${label}_predicted_eqtl_sldsc_context_effects.tsv"
base="compass-${label}-predicted-eqtl-min0.9-hierarchical-signed-ldcv10-fp32"
fold_args=()
if [[ "$fold" == all ]]; then
  run_name="$base"
else
  run_name="$base-fold${fold}"
  fold_args=(--cv-fold-subset "$fold")
fi

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python scripts/run.py \
  --data-root "$data_root" \
  --gwas "$gwas" \
  --annotation-source predicted_eqtl \
  --predicted-eqtl-min-score 0.9 \
  --method hierarchical \
  --context-annotation sum \
  --context-annotation-source gene_aggregate \
  --context-effects-tsv "$effects" \
  --context-effects-mode scaled_frozen \
  --deviation-constraint total_nonnegative \
  "${fold_args[@]}" \
  --device cuda \
  --model-dtype float32 \
  --lr 3e-7 \
  --lambdas 1e6,1e5,1e4,1e3,3e2,1e2,3e1,1e1 \
  --max-lambda-extensions 0 \
  --ld-jobs 4 \
  --run-name "$run_name"
