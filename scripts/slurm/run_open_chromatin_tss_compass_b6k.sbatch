#!/bin/bash
#SBATCH --job-name=compass-open-chromatin-tss
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b6k:1
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/open-chromatin-tss-compass-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/open-chromatin-tss-compass-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
source /gpfs/commons/home/daknowles/venv/torchfix/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python scripts/run.py \
  --data-root /gpfs/commons/home/daknowles/knowles_lab/data/compass \
  --annotation-source open_chromatin \
  --open-chromatin-tss-window 100000 \
  --device cuda \
  --model-dtype float32 \
  --ld-jobs 8 \
  --run-name compass-open-chromatin-tss100kb-ldcv10-fp32
