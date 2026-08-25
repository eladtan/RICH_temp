#!/bin/bash

set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: bash build_benchmark.sh BUILD_SUBDIRECTORY [JOBS]" >&2
  exit 2
fi

build_subdirectory="$1"
build_jobs="${2:-16}"
if [[ ! "${build_subdirectory}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Build subdirectory may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
if [[ ! "${build_jobs}" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS must be a positive integer" >&2
  exit 2
fi

case_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${case_dir}/../.." && pwd)"
build_directory="${repo_root}/build/gnuReleaseMPI/${build_subdirectory}"
binary="${build_directory}/rich_gnuReleaseMPI"
if [[ -e "${build_directory}" ]]; then
  echo "Build directory must not exist: ${build_directory}" >&2
  exit 2
fi

source_manifest_before="$(mktemp)"
source_manifest_after="$(mktemp)"
build_environment_tmp="$(mktemp)"
cleanup() {
  rm -f "${source_manifest_before}" "${source_manifest_after}" \
    "${build_environment_tmp}"
}
trap cleanup EXIT

bash "${case_dir}/source_manifest.sh" "${repo_root}" > "${source_manifest_before}"
source_manifest_sha="$(sha256sum "${source_manifest_before}" | awk '{print $1}')"
source_head="$(git -C "${repo_root}" rev-parse HEAD)"
build_started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +u
source /etc/profile.d/modules.sh
module purge
module load gcc/12.3.0
module load openmpi/4.1.6/gcc/12.3.0
module load hdf5/1.14.2/gcc/12.3.0_cxx
module load fftw/3.3.7
module load vtk/9.3.0/gcc/12.3.0/with_mesa
module load jsoncpp/1.9.5
module load boost/1.88.0
set -u

{
  module -t list 2>&1
  command -v gcc
  gcc --version | head -n 1
  command -v mpic++
  mpic++ --version | head -n 1
  command -v mpirun
  mpirun --version | head -n 2
} > "${build_environment_tmp}"

cd "${repo_root}"
./build_rich.sh gnuReleaseMPI \
  --test_name=FmmIndividualTimestepBenchmark \
  --build-subdir="${build_subdirectory}" \
  --jobs="${build_jobs}"

bash "${case_dir}/source_manifest.sh" "${repo_root}" > "${source_manifest_after}"
if ! cmp -s "${source_manifest_before}" "${source_manifest_after}"; then
  echo "Benchmark sources changed during the build; refusing provenance" >&2
  exit 2
fi
if [[ ! -x "${binary}" ]]; then
  echo "Build did not produce the expected executable: ${binary}" >&2
  exit 2
fi

cp "${source_manifest_before}" \
  "${build_directory}/FMM_INDIVIDUAL_SOURCE_MANIFEST.sha256"
cp "${build_environment_tmp}" \
  "${build_directory}/FMM_INDIVIDUAL_BUILD_ENVIRONMENT.txt"
binary_sha="$(sha256sum "${binary}" | awk '{print $1}')"
build_environment_sha="$(sha256sum \
  "${build_directory}/FMM_INDIVIDUAL_BUILD_ENVIRONMENT.txt" | awk '{print $1}')"
build_finished_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
provenance_tmp="$(mktemp "${build_directory}/.fmm-build-provenance.XXXXXX")"
{
  echo "binary_sha256=${binary_sha}"
  echo "source_manifest_sha256=${source_manifest_sha}"
  echo "build_environment_sha256=${build_environment_sha}"
  echo "source_head=${source_head}"
  echo "build_started_utc=${build_started_utc}"
  echo "build_finished_utc=${build_finished_utc}"
  echo "build_subdirectory=${build_subdirectory}"
  echo "build_jobs=${build_jobs}"
  echo "build_command=./build_rich.sh gnuReleaseMPI --test_name=FmmIndividualTimestepBenchmark --build-subdir=${build_subdirectory} --jobs=${build_jobs}"
} > "${provenance_tmp}"
mv "${provenance_tmp}" \
  "${build_directory}/FMM_INDIVIDUAL_BUILD_PROVENANCE.txt"
chmod 0444 \
  "${build_directory}/FMM_INDIVIDUAL_SOURCE_MANIFEST.sha256" \
  "${build_directory}/FMM_INDIVIDUAL_BUILD_ENVIRONMENT.txt" \
  "${build_directory}/FMM_INDIVIDUAL_BUILD_PROVENANCE.txt"

echo "binary=${binary}"
echo "binary_sha256=${binary_sha}"
echo "source_manifest_sha256=${source_manifest_sha}"
