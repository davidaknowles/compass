#!/bin/bash
#SBATCH --job-name=compass-ld-components
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/ld-components-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/ld-components-%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
mkdir -p /gpfs/commons/home/daknowles/knowles_lab/data/compass/logs
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python scripts/diagnose_ld_components.py \
  --data-root /gpfs/commons/home/daknowles/knowles_lab/data/compass \
  ${COMPASS_COMPONENT_ARGS:-}
