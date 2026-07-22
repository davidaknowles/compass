#!/bin/bash
# Run official BaselineLD-adjusted S-LDSC for a custom annotation prefix.
set -euo pipefail

input_dir=${1:?usage: run_official_sldsc_baselineld_annotations.sh INPUT_DIR PREFIX OUTPUT_DIR}
prefix=${2:?usage: run_official_sldsc_baselineld_annotations.sh INPUT_DIR PREFIX OUTPUT_DIR}
output_dir=${3:?usage: run_official_sldsc_baselineld_annotations.sh INPUT_DIR PREFIX OUTPUT_DIR}
sumstats=${4:-"$input_dir/sumstats.gz"}
output_name=${5:-ad_joint}
if [[ ! -f "$sumstats" && -f "$input_dir/ad.sumstats.gz" ]]; then
  sumstats="$input_dir/ad.sumstats.gz"
fi
data_root=/gpfs/commons/home/daknowles/knowles_lab/data/compass
reference_root="$data_root/raw/ldsc_1000g"
mkdir -p "$output_dir"

baseline=$(find "$reference_root" -name 'baselineLD.1.l2.ldscore.gz' -print -quit)
weights=$(find "$reference_root" -name 'weights.hm3_noMHC.1.l2.ldscore.gz' -print -quit)
frq=$(find "$reference_root" -name '1000G.EUR.QC.1.frq' -print -quit)
[[ -n "$baseline" && -n "$weights" && -n "$frq" ]]

module purge
module load Python/2.7.16-GCCcore-8.3.0
export PYTHONPATH="$HOME/knowles_lab/software/ldsc_py2_deps_portable:$HOME/knowles_lab/software/ldsc"

python "$HOME/knowles_lab/software/ldsc/ldsc.py" \
  --h2 "$sumstats" \
  --ref-ld-chr "$input_dir/ldscores/${prefix}.,${baseline%1.l2.ldscore.gz}" \
  --w-ld-chr "${weights%1.l2.ldscore.gz}" \
  --frqfile-chr "${frq%1.frq}" \
  --overlap-annot \
  --print-coefficients \
  --n-blocks 200 \
  --out "$output_dir/$output_name"
