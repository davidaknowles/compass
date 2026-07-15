#!/bin/bash
# Compute one chromosome of LDSC annotation scores using a standard reference.
set -euo pipefail

input_dir=${1:?usage: compute_baselineld_ldscores_chrom.sh INPUT_DIR PREFIX}
prefix=${2:?usage: compute_baselineld_ldscores_chrom.sh INPUT_DIR PREFIX}
chrom=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
data_root=/gpfs/commons/home/daknowles/knowles_lab/data/compass
reference_root="$data_root/raw/ldsc_1000g"
output_dir="$input_dir/ldscores"
mkdir -p "$output_dir"
ln -sfn "$input_dir/annotation/${prefix}.${chrom}.annot.gz" "$output_dir/${prefix}.${chrom}.annot.gz"

bim=$(find "$reference_root" -name "1000G.EUR.QC.${chrom}.bim" -print -quit)
[[ -n "$bim" ]]

module purge
module load Python/2.7.16-GCCcore-8.3.0
export PYTHONPATH="$HOME/knowles_lab/software/ldsc_py2_deps_portable:$HOME/knowles_lab/software/ldsc"

python "$HOME/knowles_lab/software/ldsc/ldsc.py" \
  --l2 \
  --bfile "${bim%.bim}" \
  --annot "$input_dir/annotation/${prefix}.${chrom}.annot.gz" \
  --print-snps "$input_dir/regression_snps/${chrom}.snps" \
  --ld-wind-cm 1 \
  --out "$output_dir/${prefix}.${chrom}"
