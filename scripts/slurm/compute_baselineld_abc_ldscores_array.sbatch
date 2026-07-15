#!/bin/bash
#SBATCH --job-name=compass-baselineld-l2
#SBATCH --partition=cpu
#SBATCH --array=1-22%6
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/baselineld-l2-%A_%a.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/baselineld-l2-%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

chrom=${SLURM_ARRAY_TASK_ID:?}
data_root=/gpfs/commons/home/daknowles/knowles_lab/data/compass
reference_root="$data_root/raw/ldsc_1000g"
input_dir="$data_root/results/official_sldsc_baselineld_inputs"
output_dir="$input_dir/ldscores"
mkdir -p "$output_dir"
ln -sfn "$input_dir/annotation/abc_baselineld.${chrom}.annot.gz" \
  "$output_dir/abc_baselineld.${chrom}.annot.gz"

bim=$(find "$reference_root" -name "1000G.EUR.QC.${chrom}.bim" -print -quit)
[[ -n "$bim" ]]

module purge
module load Python/2.7.16-GCCcore-8.3.0
export PYTHONPATH="$HOME/knowles_lab/software/ldsc_py2_deps_portable:$HOME/knowles_lab/software/ldsc"

python "$HOME/knowles_lab/software/ldsc/ldsc.py" \
  --l2 \
  --bfile "${bim%.bim}" \
  --annot "$input_dir/annotation/abc_baselineld.${chrom}.annot.gz" \
  --print-snps "$input_dir/regression_snps/${chrom}.snps" \
  --ld-wind-cm 1 \
  --out "$output_dir/abc_baselineld.${chrom}"
