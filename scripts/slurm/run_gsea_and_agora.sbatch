#!/bin/bash
#SBATCH --job-name=compass-postfit
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/postfit-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/postfit-%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

: "${COMPASS_B_TSV:?set COMPASS_B_TSV}"
: "${COMPASS_POSTFIT_DIR:?set COMPASS_POSTFIT_DIR}"

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate

AGORA_JSON="${COMPASS_AGORA_JSON:-$HOME/knowles_lab/data/compass/raw/agora/nominated_targets.json}"

python scripts/run_gsea.py \
  --b-tsv "$COMPASS_B_TSV" \
  --out-dir "$COMPASS_POSTFIT_DIR/gsea"
python scripts/crossref_agora_targets.py \
  --b-tsv "$COMPASS_B_TSV" \
  --agora-json "$AGORA_JSON" \
  --out-dir "$COMPASS_POSTFIT_DIR/agora"
