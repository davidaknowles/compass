#!/bin/bash
#SBATCH --job-name=compass-official-sldsc
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/official-sldsc-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/official-sldsc-%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

input_dir=/gpfs/commons/home/daknowles/knowles_lab/data/compass/results/official_sldsc_inputs
output_dir=/gpfs/commons/home/daknowles/knowles_lab/data/compass/results/official_sldsc
mkdir -p "$output_dir"

module purge
module load Python/2.7.16-GCCcore-8.3.0
export PYTHONPATH="$HOME/knowles_lab/software/ldsc_py2_deps:$HOME/knowles_lab/software/ldsc"

python "$HOME/knowles_lab/software/ldsc/ldsc.py" \
  --h2 "$input_dir/ad.sumstats.gz" \
  --ref-ld-chr "$input_dir/reference/abc_ukbb." \
  --w-ld-chr "$input_dir/weights/weights_ukbb." \
  --n-blocks 200 \
  --out "$output_dir/ad_abc_joint"
