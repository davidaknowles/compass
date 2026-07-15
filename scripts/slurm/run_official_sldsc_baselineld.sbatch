#!/bin/bash
#SBATCH --job-name=compass-sldsc-baselineld
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/sldsc-baselineld-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/sldsc-baselineld-%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

data_root=/gpfs/commons/home/daknowles/knowles_lab/data/compass
reference_root="$data_root/raw/ldsc_1000g"
input_dir="$data_root/results/official_sldsc_baselineld_inputs"
output_dir="$data_root/results/official_sldsc_baselineld"
mkdir -p "$output_dir"

baseline=$(find "$reference_root" -name 'baselineLD.1.l2.ldscore.gz' -print -quit)
weights=$(find "$reference_root" -name 'weights.hm3_noMHC.1.l2.ldscore.gz' -print -quit)
frq=$(find "$reference_root" -name '1000G.EUR.QC.1.frq' -print -quit)
[[ -n "$baseline" && -n "$weights" && -n "$frq" ]]

module purge
module load Python/2.7.16-GCCcore-8.3.0
export PYTHONPATH="$HOME/knowles_lab/software/ldsc_py2_deps:$HOME/knowles_lab/software/ldsc"

python "$HOME/knowles_lab/software/ldsc/ldsc.py" \
  --h2 "$input_dir/ad.sumstats.gz" \
  --ref-ld-chr "$input_dir/ldscores/abc_baselineld.,${baseline%1.l2.ldscore.gz}" \
  --w-ld-chr "${weights%1.l2.ldscore.gz}" \
  --frqfile-chr "${frq%1.frq}" \
  --overlap-annot \
  --print-coefficients \
  --n-blocks 200 \
  --out "$output_dir/ad_abc_joint"
