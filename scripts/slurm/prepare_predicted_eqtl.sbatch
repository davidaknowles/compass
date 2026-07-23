#!/bin/bash
#SBATCH --job-name=compass-pred-eqtl-prep
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/pred-eqtl-prep-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/pred-eqtl-prep-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail
threshold=${1:-0.9}
cd /gpfs/commons/home/daknowles/projects/compass
module load Kent_tools/461-GCC-12.2.0
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/prepare_predicted_eqtl_annotations.py \
  --source-root /gpfs/commons/groups/knowles_lab/gcavalca/mlxqtl_analysis/data/training_data_04_2024 \
  --chain /gpfs/commons/groups/knowles_lab/data/ADSP_reguloML/ADSP_vcf/liftover/hg38ToHg19.over.chain.gz \
  --min-score "$threshold"
