#!/bin/bash
#SBATCH --job-name=compass-glass-abc-l2
#SBATCH --partition=cpu
#SBATCH --array=1-22%6
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/glass-abc-l2-%A_%a.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/glass-abc-l2-%A_%a.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

bash /gpfs/commons/home/daknowles/projects/compass/scripts/compute_baselineld_ldscores_chrom.sh \
  /gpfs/commons/home/daknowles/knowles_lab/data/compass/results/official_sldsc_glass_abc_inputs \
  glass_abc_baselineld
