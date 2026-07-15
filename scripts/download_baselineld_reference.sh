#!/bin/bash
# Download the standard EUR Phase 3 LDSC reference resources from Zenodo.
set -euo pipefail

data_root=${1:?usage: download_baselineld_reference.sh DATA_ROOT}
root="$data_root/raw/ldsc_1000g"
archive_dir="$root/archives"
mkdir -p "$archive_dir"

base_url=https://zenodo.org/api/records/10515792/files
archives=(
  1000G_Phase3_plinkfiles.tgz
  1000G_Phase3_baselineLD_v2.2_ldscores.tgz
  1000G_Phase3_weights_hm3_no_MHC.tgz
  1000G_Phase3_frq.tgz
)

for archive in "${archives[@]}"; do
  target="$archive_dir/$archive"
  if [[ ! -s "$target" ]]; then
    curl --fail --location --retry 5 --retry-delay 10 --continue-at - \
      "$base_url/$archive/content" -o "$target"
  fi
  tar -xzf "$target" -C "$root"
done

for chrom in 1 22; do
  find "$root" -name "1000G.EUR.QC.${chrom}.bim" -print -quit | grep -q .
  find "$root" -name "baselineLD.${chrom}.l2.ldscore.gz" -print -quit | grep -q .
  find "$root" -name "weights.hm3_noMHC.${chrom}.l2.ldscore.gz" -print -quit | grep -q .
  find "$root" -name "1000G.EUR.QC.${chrom}.frq" -print -quit | grep -q .
done
