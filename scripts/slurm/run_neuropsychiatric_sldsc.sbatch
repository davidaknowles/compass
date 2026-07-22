#!/bin/bash
#SBATCH --job-name=compass-neuro-sldsc
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/neuro-sldsc-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/neuro-sldsc-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: run_neuropsychiatric_sldsc.sbatch TRAIT STAGE" >&2
  exit 2
fi

trait=$1
stage=$2
data_root=/gpfs/commons/home/daknowles/knowles_lab/data/compass
case "$trait" in
  pd) analysis_label=pd2019 ;;
  bipolar) analysis_label=bipolar2025 ;;
  mdd) analysis_label=mdd2025 ;;
  als) analysis_label=als2021 ;;
  anxiety) analysis_label=anxiety2024 ;;
  *) echo "unknown trait: $trait" >&2; exit 2 ;;
esac

gwas="$data_root/raw/neuropsychiatric_gwas/$trait/$trait.hg19.tsv.gz"
input_dir="$data_root/results/${analysis_label}_sldsc_peaks_inputs"
output_dir="$data_root/results/${analysis_label}_sldsc_peaks"
prefix=peaks_baselineld

cd /gpfs/commons/home/daknowles/projects/compass
case "$stage" in
  build)
    source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
    python scripts/build_baselineld_peak_annotations.py --out-dir "$input_dir" --gwas "$gwas"
    ;;
  ldscores)
    bash scripts/compute_baselineld_ldscores_chrom.sh "$input_dir" "$prefix"
    ;;
  fit)
    bash scripts/run_official_sldsc_baselineld_annotations.sh \
      "$input_dir" "$prefix" "$output_dir" "$input_dir/sumstats.gz" "${trait}_joint"
    ;;
  extract)
    source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
    python scripts/extract_sldsc_context_effects.py \
      "$output_dir/${trait}_joint.results" \
      "$data_root/results/${analysis_label}_h3k27ac_sldsc_context_effects.tsv" \
      --panel h3k27ac
    ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
