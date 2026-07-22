#!/bin/bash
# Submit BaselineLD S-LDSC, signed COMPASS, and GSEA for public GWAS panels.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: submit_neuropsychiatric_pipeline.sh DOWNLOAD_JOB [TRAIT ...]" >&2
  exit 2
fi

download_job=$1
shift
traits=("${@:-pd bipolar mdd als anxiety}")
repo=$(cd "$(dirname "$0")/.." && pwd)
data_root="$HOME/knowles_lab/data/compass"
log_dir="$data_root/logs"
mkdir -p "$log_dir"

job_id() {
  awk '{print $4}' <<<"$1"
}

for trait in ${traits[*]}; do
  case "$trait" in
    pd) label=pd2025 ;;
    bipolar) label=bipolar2025 ;;
    mdd) label=mdd2025 ;;
    als) label=als2021 ;;
    anxiety) label=anxiety2024 ;;
    *) echo "unknown trait: $trait" >&2; exit 2 ;;
  esac

  build=$(job_id "$(sbatch --dependency="afterok:$download_job" \
    --job-name="sldsc-${trait}-build" scripts/slurm/run_neuropsychiatric_sldsc.sbatch "$trait" build)")
  ldscores=$(job_id "$(sbatch --dependency="afterok:$build" --array=1-22%8 \
    --job-name="sldsc-${trait}-ld" scripts/slurm/run_neuropsychiatric_sldsc.sbatch "$trait" ldscores)")
  fit=$(job_id "$(sbatch --dependency="afterok:$ldscores" \
    --job-name="sldsc-${trait}-fit" scripts/slurm/run_neuropsychiatric_sldsc.sbatch "$trait" fit)")
  extract=$(job_id "$(sbatch --dependency="afterok:$fit" \
    --job-name="sldsc-${trait}-extract" scripts/slurm/run_neuropsychiatric_sldsc.sbatch "$trait" extract)")

  fold_jobs=()
  watch_args=()
  shard_args=()
  for fold in $(seq 0 9); do
    fold_job=$(job_id "$(sbatch --dependency="afterok:$extract" \
      scripts/slurm/run_hierarchical_genesum_2026_fold_b6k.sbatch "$trait" "$fold" total_nonnegative)")
    checkpoint="$data_root/results/compass-${label}-h3k27ac-genesum-hierarchical-signed-ldcv10-fp32-fold${fold}.hierarchical_cv_checkpoint.npz"
    fold_jobs+=("$fold_job")
    watch_args+=(--job "$fold_job" "$checkpoint")
    shard_args+=(--shard "$checkpoint")
  done
  fold_dependency=$(IFS=:; echo "${fold_jobs[*]}")

  watch_command=(python scripts/watch_hierarchical_cv_jobs.py --poll-seconds 60 "${watch_args[@]}")
  watch=$(job_id "$(sbatch -p cpu --dependency="afterok:$extract" --time=24:00:00 --mem=2G \
    --cpus-per-task=1 --job-name="compass-${trait}-cv-watch" \
    --output="$log_dir/${trait}-signed-watch-%j.out" --error="$log_dir/${trait}-signed-watch-%j.err" \
    --wrap "bash -lc 'cd $repo; source $HOME/venv/torchfix/bin/activate; ${watch_command[*]}'")")

  merged="$data_root/results/compass-${label}-h3k27ac-genesum-hierarchical-signed-ldcv10-fp32.hierarchical_cv_checkpoint.npz"
  merge_command=(python scripts/merge_hierarchical_cv_checkpoints.py --folds 0,1,2,3,4,5,6,7,8,9 "${shard_args[@]}" --output "$merged")
  merge=$(job_id "$(sbatch -p cpu --dependency="afterany:$fold_dependency" --time=00:30:00 --mem=8G \
    --cpus-per-task=1 --job-name="compass-${trait}-merge" \
    --output="$log_dir/${trait}-signed-merge-%j.out" --error="$log_dir/${trait}-signed-merge-%j.err" \
    --wrap "bash -lc 'cd $repo; source $HOME/venv/torchfix/bin/activate; ${merge_command[*]}'")")
  refit=$(job_id "$(sbatch --dependency="afterok:$merge" --job-name="compass-${trait}-refit" \
    scripts/slurm/run_hierarchical_genesum_2026_fold_b6k.sbatch "$trait" all total_nonnegative)")

  b_tsv="$data_root/results/compass-${label}-h3k27ac-genesum-hierarchical-signed-ldcv10-fp32.B.tsv"
  annotation="$data_root/cache/peaktss.H3K27ac.tsswin100000.6dce59cb230d.intercept.${trait}.hg19.tsv.gz.A.npz"
  for fraction in 25 50; do
    value="0.$fraction"
    gsea=$(job_id "$(sbatch -p cpu --dependency="afterok:$refit" --time=01:00:00 --mem=24G \
      --cpus-per-task=1 --job-name="compass-${trait}-gsea${fraction}" \
      --output="$log_dir/${trait}-signed-gsea${fraction}-%j.out" \
      --error="$log_dir/${trait}-signed-gsea${fraction}-%j.err" \
      --wrap "bash -lc 'cd $repo; source $HOME/venv/torchfix/bin/activate; export PYTHONPATH=\$PWD/src:\${PYTHONPATH:-}; python scripts/run_gsea.py --b-tsv $b_tsv --annotation-npz $annotation --cumulative-score-fraction $value --out-dir $data_root/results/${label}_h3k27ac_genesum_hierarchical_signed_gsea_c${fraction}'")")
    printf '%s\t%s\t%s\n' "$trait" "gsea${fraction}" "$gsea"
  done
  printf '%s\tbuild\t%s\n%s\tldscores\t%s\n%s\tfit\t%s\n%s\textract\t%s\n' \
    "$trait" "$build" "$trait" "$ldscores" "$trait" "$fit" "$trait" "$extract"
  printf '%s\tfolds\t%s\n%s\twatch\t%s\n%s\tmerge\t%s\n%s\trefit\t%s\n' \
    "$trait" "$(IFS=,; echo "${fold_jobs[*]}")" "$trait" "$watch" "$trait" "$merge" "$trait" "$refit"
done
