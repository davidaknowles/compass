#!/bin/bash
#SBATCH --job-name=compass-baselineld-download
#SBATCH --partition=cpu
#SBATCH --output=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/baselineld-download-%j.out
#SBATCH --error=/gpfs/commons/home/daknowles/knowles_lab/data/compass/logs/baselineld-download-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass
mkdir -p /gpfs/commons/home/daknowles/knowles_lab/data/compass/logs
bash scripts/download_baselineld_reference.sh /gpfs/commons/home/daknowles/knowles_lab/data/compass
