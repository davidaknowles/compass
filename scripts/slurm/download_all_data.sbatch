#!/bin/bash
#SBATCH --job-name=compass-download
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/download-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/download-%j.err
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
mkdir -p /gpfs/commons/home/daknowles/knowles_lab/data/compass/logs
module load Python/3.12.3-GCCcore-13.3.0
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/download_required_data.py \
  --data-root /gpfs/commons/home/daknowles/knowles_lab/data/compass \
  --download-top-assoc \
  --download-ad \
  --download-abc \
  --download-ld-metadata \
  --download-ld-npz \
  --jobs 8
