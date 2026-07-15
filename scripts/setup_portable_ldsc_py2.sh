#!/bin/bash
# Build the one Python 2 LDSC dependency that must avoid host-specific SIMD.
set -euo pipefail

target="$HOME/knowles_lab/software/ldsc_py2_deps_portable"
module purge
module load Python/2.7.16-GCCcore-8.3.0 GCC/8.3.0
mkdir -p "$target"
CFLAGS='-O3 -march=x86-64 -mtune=generic' \
  python -m pip install --no-cache-dir --no-binary=bitarray --target "$target" 'bitarray==0.8.3'
