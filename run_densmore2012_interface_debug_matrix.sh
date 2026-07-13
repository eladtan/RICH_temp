#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-intelReleaseMPI}"
NP="${NP:-32}"
GROUPS="${GROUPS:-30}"
PARTITION="${PARTITION:-bigrun}"
SLURM_TIME="${SLURM_TIME:-30}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1500}"
RESUME="${RESUME:-1}"

if command -v ml >/dev/null 2>&1; then
    ml openmpi/4.1.6/Intel/OneApi/2024.2.1
fi

run_case() {
    local case_dir="$1"
    local prefix="$2"
    local expected_rows="$3"
    local cells_file="${case_dir}/${prefix}_cells.tsv"

    if [[ "${RESUME}" == "1" && -s "${cells_file}" ]]; then
        local rows
        rows="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "${cells_file}")"
        if [[ "${rows}" == "${expected_rows}" ]]; then
            echo "=== Reusing ${prefix} (${rows} cells) ==="
            return
        fi
    fi

    echo "=== Building ${prefix} ==="
    ./build_rich.sh "${CONFIG}" \
        --test_name="${case_dir}" \
        --energy_groups_num="${GROUPS}"

    rm -f "${case_dir}/${prefix}"_{cells.tsv,profile.txt,interface_history.tsv,global_diagnostics.tsv}
    rm -f "${case_dir}/${prefix}"_rank*_ddmc_face_history.tsv
    rm -f "${case_dir}/${prefix}"_rank*_ddmc_interface_events.tsv
    rm -f "${case_dir}/run.stdout.log" "${case_dir}/run.stderr.log"

    echo "=== Running ${prefix} on ${NP} ranks ==="
    sbatch --wait --exclusive \
        --partition="${PARTITION}" \
        --ntasks="${NP}" \
        --time="${SLURM_TIME}" \
        --output="${case_dir}/run.stdout.log" \
        --error="${case_dir}/run.stderr.log" \
        --wrap="timeout ${TIMEOUT_SECONDS} mpirun -np ${NP} ./build/${CONFIG}/rich"

    if [[ ! -s "${cells_file}" ]]; then
        echo "Missing ${cells_file}" >&2
        exit 1
    fi
    local rows
    rows="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "${cells_file}")"
    if [[ "${rows}" != "${expected_rows}" ]]; then
        echo "${prefix}: expected ${expected_rows} cells, found ${rows}" >&2
        exit 1
    fi
    if grep -Eqi 'UniversalError|terminate called|Segmentation fault|MPI_ABORT' \
        "${case_dir}/run.stdout.log" "${case_dir}/run.stderr.log"; then
        echo "Fatal marker found in ${prefix} logs" >&2
        exit 1
    fi
}

compare_pair() {
    local mc_dir="$1"
    local mc_prefix="$2"
    local ddmc_dir="$3"
    local ddmc_prefix="$4"
    local output_stem="$5"

    python3 regression_tests/cases/compare_densmore2012_interface.py \
        --mc "${mc_dir}/${mc_prefix}_cells.tsv" \
        --ddmc "${ddmc_dir}/${ddmc_prefix}_cells.tsv" \
        --output "regression_tests/cases/${output_stem}.tsv" \
        | tee "regression_tests/cases/${output_stem}.txt"
}

analyze_events() {
    local case_dir="$1"
    local prefix="$2"
    local count
    count="$(find "${case_dir}" -maxdepth 1 \
        -name "${prefix}_rank*_ddmc_interface_events.tsv" | wc -l)"
    if [[ "${count}" != "${NP}" ]]; then
        echo "${prefix}: expected ${NP} event ledgers, found ${count}" >&2
        exit 1
    fi
    python3 regression_tests/cases/analyze_densmore2012_interface_events.py \
        "${case_dir}" --prefix "${prefix}" \
        | tee "${case_dir}/${prefix}_interface_event_analysis.txt"
}

EXPOSED_MC="regression_tests/cases/desmore2012_interface_mc"
EXPOSED_DDMC="regression_tests/cases/desmore2012_interface_ddmc"
EXPOSED_C17="regression_tests/cases/desmore2012_interface_ddmc_cutoff17"
CENTERED_MC="regression_tests/cases/desmore2012_interface_mc_centered"
CENTERED_DDMC="regression_tests/cases/desmore2012_interface_ddmc_centered"
CENTERED_C17="regression_tests/cases/desmore2012_interface_ddmc_centered_cutoff17"

run_case "${EXPOSED_MC}" desmore2012_interface_mc 300
run_case "${EXPOSED_DDMC}" desmore2012_interface_ddmc 300
run_case "${EXPOSED_C17}" desmore2012_interface_ddmc_cutoff17 300
run_case "${CENTERED_MC}" desmore2012_interface_mc_centered 315
run_case "${CENTERED_DDMC}" desmore2012_interface_ddmc_centered 315
run_case "${CENTERED_C17}" desmore2012_interface_ddmc_centered_cutoff17 315

compare_pair "${EXPOSED_MC}" desmore2012_interface_mc \
    "${EXPOSED_DDMC}" desmore2012_interface_ddmc \
    densmore2012_interface_comparison_baseline
compare_pair "${EXPOSED_MC}" desmore2012_interface_mc \
    "${EXPOSED_C17}" desmore2012_interface_ddmc_cutoff17 \
    densmore2012_interface_comparison_cutoff17
compare_pair "${CENTERED_MC}" desmore2012_interface_mc_centered \
    "${CENTERED_DDMC}" desmore2012_interface_ddmc_centered \
    densmore2012_interface_comparison_centered
compare_pair "${CENTERED_MC}" desmore2012_interface_mc_centered \
    "${CENTERED_C17}" desmore2012_interface_ddmc_centered_cutoff17 \
    densmore2012_interface_comparison_centered_cutoff17

analyze_events "${EXPOSED_DDMC}" desmore2012_interface_ddmc
analyze_events "${EXPOSED_C17}" desmore2012_interface_ddmc_cutoff17
analyze_events "${CENTERED_DDMC}" desmore2012_interface_ddmc_centered
analyze_events "${CENTERED_C17}" desmore2012_interface_ddmc_centered_cutoff17

echo "=== Densmore interface debug matrix complete ==="
echo "Upload the four comparison .txt/.tsv pairs and the four"
echo "*_interface_event_analysis.txt / *_interface_net_energy.tsv outputs."
