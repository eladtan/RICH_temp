#!/bin/bash
set -euo pipefail

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${case_dir}/../../.." && pwd)"
campaign_id="${RICH_CAMPAIGN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
campaign_root="${1:-${repo_root}/regression_tests/results/lane_radiation_shock_${campaign_id}}"
mkdir -p "${campaign_root}"

mg_preconditioner="${RICH_TEST_MG_PRECONDITIONER:-cell_block}"
case "${mg_preconditioner}" in
  scalar|cell_block|cell_block_two_sweep|cell_block_four_sweep|cell_block_eight_sweep) ;;
  *)
    printf 'invalid RICH_TEST_MG_PRECONDITIONER: %s\n' \
      "${mg_preconditioner}" >&2
    exit 2
    ;;
esac

source_binary="${RICH_SOURCE_BINARY:-${repo_root}/build/intelReleaseMPI/rich_intelReleaseMPI}"
test -x "${source_binary}"
binary_hash="$(sha256sum "${source_binary}" | awk '{print $1}')"
campaign_binary="${campaign_root}/rich_intelReleaseMPI_${binary_hash}"
if [[ -e "${campaign_binary}" ]]; then
  test "$(sha256sum "${campaign_binary}" | awk '{print $1}')" = "${binary_hash}"
else
  install -m 0555 "${source_binary}" "${campaign_binary}"
fi
sha256sum "${campaign_binary}" > "${campaign_root}/binary.sha256"

export_arg="ALL,RICH_CAMPAIGN_ROOT=${campaign_root},RICH_CASE_DIR=${case_dir},RICH_REPO_ROOT=${repo_root},RICH_CAMPAIGN_BINARY=${campaign_binary},RICH_TEST_MG_PRECONDITIONER=${mg_preconditioner}"
calibration_job="$(sbatch --parsable --export="${export_arg}" "${case_dir}/calibrate_quantum.sbatch")"
calibration_dependency="afterok:${calibration_job}"
global_job="$(sbatch --parsable --dependency="${calibration_dependency}" --export="${export_arg}" "${case_dir}/submit_global.sbatch")"
full_variable_job="$(sbatch --parsable --dependency="${calibration_dependency}" --export="${export_arg}" "${case_dir}/submit_full_variable.sbatch")"
partial_job="$(sbatch --parsable --dependency="${calibration_dependency}" --export="${export_arg}" "${case_dir}/submit_partial.sbatch")"
dependency="afterany:${global_job}:${full_variable_job}:${partial_job}"
comparison_job="$(sbatch --parsable --dependency="${dependency}" \
  --export="${export_arg}" "${case_dir}/compare_campaign.sbatch")"

{
  printf 'campaign_root %s\n' "${campaign_root}"
  printf 'campaign_binary %s\n' "${campaign_binary}"
  printf 'binary_sha256 %s\n' "${binary_hash}"
  printf 'mg_preconditioner %s\n' "${mg_preconditioner}"
  printf 'calibration_job %s\n' "${calibration_job}"
  printf 'global_job %s\n' "${global_job}"
  printf 'full_variable_job %s\n' "${full_variable_job}"
  printf 'partial_job %s\n' "${partial_job}"
  printf 'comparison_job %s\n' "${comparison_job}"
} | tee "${campaign_root}/submitted_jobs.txt"
