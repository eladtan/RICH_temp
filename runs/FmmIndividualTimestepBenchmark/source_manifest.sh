#!/bin/bash

set -euo pipefail

repo_root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${repo_root}"

required_paths=(
  build_rich.sh
  runs/FmmIndividualTimestepBenchmark/README.md
  runs/FmmIndividualTimestepBenchmark/analyze.py
  runs/FmmIndividualTimestepBenchmark/build_benchmark.sh
  runs/FmmIndividualTimestepBenchmark/run_campaign.sh
  runs/FmmIndividualTimestepBenchmark/source_manifest.sh
  runs/FmmIndividualTimestepBenchmark/submit.sbatch
  runs/FmmIndividualTimestepBenchmark/test.cpp
)
source_roots=(source config)

for path in "${required_paths[@]}" "${source_roots[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing runtime source path: ${path}" >&2
    exit 2
  fi
done

{
  printf '%s\0' "${required_paths[@]}"
  find "${source_roots[@]}" -type f -print0
} | LC_ALL=C sort -z | while IFS= read -r -d '' path; do
  sha256sum "${path}"
done
