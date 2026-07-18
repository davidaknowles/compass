#!/bin/bash
#SBATCH --job-name=sldsc-ad2026
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/ad2026-sldsc-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/ad2026-sldsc-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: run_ad2026_sldsc.sbatch TARGET STAGE" >&2
  exit 2
fi
target=$1
stage=$2
data_root=/gpfs/commons/home/daknowles/knowles_lab/data/compass
gwas="$data_root/raw/ad_gwas_2026/GCST90704647.hg19.tsv.gz"

case "$target" in
  glass)
    input_dir="$data_root/results/ad2026_sldsc_glass_abc_inputs"
    output_dir="$data_root/results/ad2026_sldsc_glass_abc"
    prefix=glass_abc_baselineld
    build=(scripts/build_baselineld_glass_abc_annotations.py --out-dir "$input_dir" --gwas "$gwas")
    ;;
  glass_binary)
    input_dir="$data_root/results/ad2026_sldsc_glass_abc_binary_inputs"
    output_dir="$data_root/results/ad2026_sldsc_glass_abc_binary"
    prefix=glass_abc_baselineld
    build=(scripts/build_baselineld_glass_abc_annotations.py --out-dir "$input_dir" --gwas "$gwas" --binary)
    ;;
  glass_distal)
    input_dir="$data_root/results/ad2026_sldsc_glass_abc_distal_inputs"
    output_dir="$data_root/results/ad2026_sldsc_glass_abc_distal"
    prefix=glass_abc_baselineld
    build=(scripts/build_baselineld_glass_abc_annotations.py --out-dir "$input_dir" --gwas "$gwas" --exclude-self-promoter)
    ;;
  peaks)
    input_dir="$data_root/results/ad2026_sldsc_peaks_inputs"
    output_dir="$data_root/results/ad2026_sldsc_peaks"
    prefix=peaks_baselineld
    build=(scripts/build_baselineld_peak_annotations.py --out-dir "$input_dir" --gwas "$gwas")
    ;;
  public_abc)
    input_dir="$data_root/results/ad2026_sldsc_public_abc_inputs"
    output_dir="$data_root/results/ad2026_sldsc_public_abc"
    prefix=abc_baselineld
    build=(scripts/build_baselineld_abc_annotations.py --out-dir "$input_dir" --gwas "$gwas")
    ;;
  *)
    echo "Unknown target: $target" >&2
    exit 2
    ;;
esac

cd /gpfs/commons/home/daknowles/projects/compass
case "$stage" in
  build)
    source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
    export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
    python "${build[@]}"
    ;;
  ldscores)
    bash scripts/compute_baselineld_ldscores_chrom.sh "$input_dir" "$prefix"
    ;;
  fit)
    bash scripts/run_official_sldsc_baselineld_annotations.sh "$input_dir" "$prefix" "$output_dir"
    ;;
  *)
    echo "Unknown stage: $stage" >&2
    exit 2
    ;;
esac
