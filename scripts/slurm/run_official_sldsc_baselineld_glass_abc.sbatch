#!/bin/bash
#SBATCH --job-name=compass-sldsc-glass-abc
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/sldsc-glass-abc-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/sldsc-glass-abc-%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

bash /gpfs/commons/home/daknowles/projects/compass/scripts/run_official_sldsc_baselineld_annotations.sh \
  /gpfs/commons/home/daknowles/knowles_lab/data/compass/results/official_sldsc_glass_abc_inputs \
  glass_abc_baselineld \
  /gpfs/commons/home/daknowles/knowles_lab/data/compass/results/official_sldsc_glass_abc
