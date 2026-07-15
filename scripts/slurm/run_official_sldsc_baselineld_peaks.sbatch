#!/bin/bash
#SBATCH --job-name=compass-sldsc-peaks
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/sldsc-peaks-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/sldsc-peaks-%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

bash /gpfs/commons/home/daknowles/projects/compass/scripts/run_official_sldsc_baselineld_annotations.sh \
  /gpfs/commons/home/daknowles/knowles_lab/data/compass/results/official_sldsc_peaks_inputs \
  peaks_baselineld \
  /gpfs/commons/home/daknowles/knowles_lab/data/compass/results/official_sldsc_peaks
