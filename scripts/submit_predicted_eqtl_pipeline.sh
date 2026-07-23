#!/bin/bash
# Submit shared predicted-eQTL preparation/S-LDSC and trait-specific COMPASS fits.
set -euo pipefail

traits=("${@:-pd bipolar mdd als anxiety}")
repo=$(cd "$(dirname "$0")/.." && pwd)
data_root="$HOME/knowles_lab/data/compass"
log_dir="$data_root/logs"
mkdir -p "$log_dir"

job_id() { awk '{print $4}' <<<"$1"; }

cd "$repo"
prep=$(job_id "$(sbatch scripts/slurm/prepare_predicted_eqtl.sbatch)")
build=$(job_id "$(sbatch --dependency="afterok:$prep" \
  scripts/slurm/run_predicted_eqtl_sldsc.sbatch pd build)")
ldscores=$(job_id "$(sbatch --dependency="afterok:$build" --array=1-22%8 \
  scripts/slurm/run_predicted_eqtl_sldsc.sbatch pd ldscores)")
printf 'shared\tprepare\t%s\nshared\tbuild\t%s\nshared\tldscores\t%s\n' "$prep" "$build" "$ldscores"

for trait in ${traits[*]}; do
  case "$trait" in
    pd) label=pd2025 ;;
    bipolar) label=bipolar2025 ;;
    mdd) label=mdd2025 ;;
    als) label=als2021 ;;
    anxiety) label=anxiety2024 ;;
    *) echo "unknown trait: $trait" >&2; exit 2 ;;
  esac
  fit=$(job_id "$(sbatch --dependency="afterok:$ldscores" \
    scripts/slurm/run_predicted_eqtl_sldsc.sbatch "$trait" fit)")
  extract=$(job_id "$(sbatch --dependency="afterok:$fit" \
    scripts/slurm/run_predicted_eqtl_sldsc.sbatch "$trait" extract)")

  # Fold 0 is the sole cache builder. Remaining folds start after its cache is complete.
  fold0=$(job_id "$(sbatch --dependency="afterok:$extract" \
    scripts/slurm/run_hierarchical_predicted_eqtl_fold_b6k.sbatch "$trait" 0)")
  fold_jobs=("$fold0")
  shard_args=(--shard "$data_root/results/compass-${label}-predicted-eqtl-min0.9-hierarchical-signed-ldcv10-fp32-fold0.hierarchical_cv_checkpoint.npz")
  for fold in $(seq 1 9); do
    job=$(job_id "$(sbatch --dependency="afterok:$fold0" \
      scripts/slurm/run_hierarchical_predicted_eqtl_fold_b6k.sbatch "$trait" "$fold")")
    fold_jobs+=("$job")
    shard_args+=(--shard "$data_root/results/compass-${label}-predicted-eqtl-min0.9-hierarchical-signed-ldcv10-fp32-fold${fold}.hierarchical_cv_checkpoint.npz")
  done
  fold_dependency=$(IFS=:; echo "${fold_jobs[*]}")
  merged="$data_root/results/compass-${label}-predicted-eqtl-min0.9-hierarchical-signed-ldcv10-fp32.hierarchical_cv_checkpoint.npz"
  merge_command=(python scripts/merge_hierarchical_cv_checkpoints.py \
    --folds 0,1,2,3,4,5,6,7,8,9 "${shard_args[@]}" --output "$merged")
  merge=$(job_id "$(sbatch -p cpu --dependency="afterany:$fold_dependency" --time=00:30:00 \
    --mem=8G --cpus-per-task=1 --job-name="pred-eqtl-${trait}-merge" \
    --output="$log_dir/pred-eqtl-${trait}-merge-%j.out" \
    --error="$log_dir/pred-eqtl-${trait}-merge-%j.err" \
    --wrap "bash -lc 'cd $repo; source $HOME/venv/torchfix/bin/activate; ${merge_command[*]}'")")
  refit=$(job_id "$(sbatch --dependency="afterok:$merge" \
    scripts/slurm/run_hierarchical_predicted_eqtl_fold_b6k.sbatch "$trait" all)")

  b_tsv="$data_root/results/compass-${label}-predicted-eqtl-min0.9-hierarchical-signed-ldcv10-fp32.B.tsv"
  for fraction in 25 50; do
    gsea=$(job_id "$(sbatch -p cpu --dependency="afterok:$refit" --time=01:00:00 --mem=24G \
      --cpus-per-task=1 --job-name="pred-eqtl-${trait}-gsea${fraction}" \
      --output="$log_dir/pred-eqtl-${trait}-gsea${fraction}-%j.out" \
      --error="$log_dir/pred-eqtl-${trait}-gsea${fraction}-%j.err" \
      --wrap "bash -lc 'cd $repo; source $HOME/venv/torchfix/bin/activate; export PYTHONPATH=\$PWD/src:\${PYTHONPATH:-}; annotation=\$(find $data_root/cache -maxdepth 1 -name \"predictedeqtl.min0.9.r2ge0.01.chromfp16.*.intercept.${trait}.hg19.tsv.gz.A.npz\" -print -quit); test -f \"\$annotation\"; python scripts/run_gsea.py --b-tsv $b_tsv --annotation-npz \"\$annotation\" --cumulative-score-fraction 0.$fraction --out-dir $data_root/results/${label}_predicted_eqtl_hierarchical_signed_gsea_c${fraction}'")")
    printf '%s\tgsea%s\t%s\n' "$trait" "$fraction" "$gsea"
  done
  printf '%s\tsldsc_fit\t%s\n%s\tsldsc_extract\t%s\n' "$trait" "$fit" "$trait" "$extract"
  printf '%s\tfolds\t%s\n%s\tmerge\t%s\n%s\trefit\t%s\n' \
    "$trait" "$(IFS=,; echo "${fold_jobs[*]}")" "$trait" "$merge" "$trait" "$refit"
done
