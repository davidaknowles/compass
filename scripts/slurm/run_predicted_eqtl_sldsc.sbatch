#!/bin/bash
#SBATCH --job-name=compass-pred-eqtl-sldsc
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/pred-eqtl-sldsc-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/pred-eqtl-sldsc-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail
trait=${1:?usage: run_predicted_eqtl_sldsc.sbatch TRAIT STAGE [THRESHOLD]}
stage=${2:?usage: run_predicted_eqtl_sldsc.sbatch TRAIT STAGE [THRESHOLD]}
threshold=${3:-0.9}
data_root=/gpfs/commons/home/daknowles/knowles_lab/data/compass
prefix=predicted_eqtl_baselineld

if [[ "$threshold" == 0.9 ]]; then
  shared_input="$data_root/results/predicted_eqtl_sldsc_shared_inputs"
  result_suffix=predicted_eqtl
else
  shared_input="$data_root/results/predicted_eqtl_sldsc_min${threshold}_shared_inputs"
  result_suffix="predicted_eqtl_min${threshold}"
fi

case "$trait" in
  ad) label=ad2026; gwas="$data_root/raw/ad_gwas_2026/GCST90704647.hg19.tsv.gz" ;;
  scz) label=scz2026; gwas="$data_root/raw/scz_gwas_2026/SCZ_EUR_2026.hg19.tsv.gz" ;;
  pd) label=pd2025; gwas="$data_root/raw/neuropsychiatric_gwas/pd/pd.hg19.tsv.gz" ;;
  bipolar) label=bipolar2025; gwas="$data_root/raw/neuropsychiatric_gwas/bipolar/bipolar.hg19.tsv.gz" ;;
  mdd) label=mdd2025; gwas="$data_root/raw/neuropsychiatric_gwas/mdd/mdd.hg19.tsv.gz" ;;
  als) label=als2021; gwas="$data_root/raw/neuropsychiatric_gwas/als/als.hg19.tsv.gz" ;;
  anxiety) label=anxiety2024; gwas="$data_root/raw/neuropsychiatric_gwas/anxiety/anxiety.hg19.tsv.gz" ;;
  *) echo "unknown trait: $trait" >&2; exit 2 ;;
esac
output_dir="$data_root/results/${label}_sldsc_${result_suffix}"
sumstats="$output_dir/sumstats.gz"
cd /gpfs/commons/home/daknowles/projects/compass

case "$stage" in
  build)
    source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
    python scripts/build_baselineld_predicted_eqtl_annotations.py \
      --gwas "$gwas" --out-dir "$shared_input" --min-score "$threshold" \
      --predicted-eqtl-path "$data_root/raw/predicted_eqtl/links.hg19.min${threshold}"
    ;;
  ldscores)
    bash scripts/compute_baselineld_ldscores_chrom.sh "$shared_input" "$prefix"
    ;;
  fit)
    mkdir -p "$output_dir"
    source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
    python - "$gwas" "$sumstats" "$data_root/raw/ldsc_1000g" <<'PY'
import sys
from pathlib import Path
from compass.baselineld import read_bim, write_hapmap3_sumstats
root = Path(sys.argv[3])
bims = {}
for chrom in range(1, 23):
    matches = list(root.rglob(f"1000G.EUR.QC.{chrom}.bim"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one chromosome {chrom} BIM, found {len(matches)}")
    bims[chrom] = read_bim(matches[0])
write_hapmap3_sumstats(sys.argv[1], sys.argv[2], bims)
PY
    bash scripts/run_official_sldsc_baselineld_annotations.sh \
      "$shared_input" "$prefix" "$output_dir" "$sumstats" "${trait}_joint"
    ;;
  extract)
    source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
    python scripts/extract_sldsc_context_effects.py \
      "$output_dir/${trait}_joint.results" \
      "$data_root/results/${label}_${result_suffix}_sldsc_context_effects.tsv" \
      --panel predicted_eqtl
    ;;
  *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
