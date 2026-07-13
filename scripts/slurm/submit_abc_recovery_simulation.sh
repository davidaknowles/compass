#!/bin/bash
set -euo pipefail

cd /gpfs/commons/home/daknowles/projects/compass

prepare=$(sbatch --parsable scripts/slurm/prepare_abc_recovery_simulation.sbatch)
array=$(sbatch --parsable --dependency="afterok:${prepare}" scripts/slurm/run_abc_recovery_simulation_array.sbatch)
summary=$(sbatch --parsable --dependency="afterok:${array}" scripts/slurm/summarize_abc_recovery_simulation.sbatch)
printf 'prepare=%s\narray=%s\nsummary=%s\n' "$prepare" "$array" "$summary"
