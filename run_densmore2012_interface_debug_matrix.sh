#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-intelReleaseMPI}"
NP="${NP:-32}"
GROUPS="${GROUPS:-30}"
PARTITION="${PARTITION:-bigrun}"
SLURM_TIME="${SLURM_TIME:-30}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-71500}"
RESUME="${RESUME:-1}"

# The cutoff17 cases mean group 17 of the original 30-group discretization.
# With another compile-time group count, especially 1100, cutoff 17 represents
# a completely different energy and creates an extremely expensive IMC band.
if [[ "${GROUPS}" != "30" ]]; then
    echo "This debug matrix requires GROUPS=30; got GROUPS=${GROUPS}." >&2
    echo "The cutoff17 ablation is an index-based 30-group experiment." >&2
    exit 2
fi

if command -v ml >/dev/null 2>&1; then
    ml openmpi/4.1.6/Intel/OneApi/2024.2.1
fi

BINARY_DIR="${ROOT}/.densmore2012_interface_bins/${CONFIG}_g${GROUPS}"
mkdir -p "${BINARY_DIR}"

EXPOSED_MC="regression_tests/cases/desmore2012_interface_mc"
EXPOSED_DDMC="regression_tests/cases/desmore2012_interface_ddmc"
EXPOSED_C17="regression_tests/cases/desmore2012_interface_ddmc_cutoff17"
CENTERED_MC="regression_tests/cases/desmore2012_interface_mc_centered"
CENTERED_DDMC="regression_tests/cases/desmore2012_interface_ddmc_centered"
CENTERED_C17="regression_tests/cases/desmore2012_interface_ddmc_centered_cutoff17"

CASE_DIRS=(
    "${EXPOSED_MC}"
    "${EXPOSED_DDMC}"
    "${EXPOSED_C17}"
    "${CENTERED_MC}"
    "${CENTERED_DDMC}"
    "${CENTERED_C17}"
)
CASE_PREFIXES=(
    desmore2012_interface_mc
    desmore2012_interface_ddmc
    desmore2012_interface_ddmc_cutoff17
    desmore2012_interface_mc_centered
    desmore2012_interface_ddmc_centered
    desmore2012_interface_ddmc_centered_cutoff17
)
CASE_ROWS=(300 300 300 315 315 315)
CASE_HAS_EVENTS=(0 1 1 0 1 1)

case_complete() {
    local case_dir="$1"
    local prefix="$2"
    local expected_rows="$3"
    local has_events="$4"
    local cells_file="${case_dir}/${prefix}_cells.tsv"

    [[ -s "${cells_file}" ]] || return 1

    local rows
    rows="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "${cells_file}")"
    [[ "${rows}" == "${expected_rows}" ]] || return 1

    [[ -s "${case_dir}/run.stdout.log" ]] || return 1
    grep -q "G=${GROUPS}" "${case_dir}/run.stdout.log" || return 1
    grep -q "new/cell=25" "${case_dir}/run.stdout.log" || return 1
    grep -q "max/cell=100" "${case_dir}/run.stdout.log" || return 1

    if grep -Eqi 'UniversalError|terminate called|Segmentation fault|MPI_ABORT|StuckParticle' \
        "${case_dir}/run.stdout.log" "${case_dir}/run.stderr.log"; then
        return 1
    fi

    if [[ "${has_events}" == "1" ]]; then
        local count
        count="$(find "${case_dir}" -maxdepth 1 \
            -name "${prefix}_rank*_ddmc_interface_events.tsv" | wc -l)"
        [[ "${count}" == "${NP}" ]] || return 1
    fi

    return 0
}

clean_case_outputs() {
    local case_dir="$1"
    local prefix="$2"

    rm -f "${case_dir}/${prefix}"_{cells.tsv,profile.txt,interface_history.tsv,global_diagnostics.tsv}
    rm -f "${case_dir}/${prefix}"_rank*_acceleration_debug.txt
    rm -f "${case_dir}/${prefix}"_rank*_ddmc_face_history.tsv
    rm -f "${case_dir}/${prefix}"_rank*_ddmc_interface_events.tsv
    rm -f "${case_dir}/${prefix}"_interface_{event_analysis.txt,event_summary.tsv,admission_balance.tsv,net_energy.tsv}
    rm -f "${case_dir}/run.stdout.log" "${case_dir}/run.stderr.log"
}

build_case_binary() {
    local case_dir="$1"
    local prefix="$2"
    local binary="${BINARY_DIR}/${prefix}"

    echo "=== Building ${prefix} ==="
    ./build_rich.sh "${CONFIG}" \
        --test_name="${case_dir}" \
        --energy_groups_num="${GROUPS}"

    cp -L "./build/${CONFIG}/rich" "${binary}"
    chmod +x "${binary}"
}

submit_case() {
    local case_dir="$1"
    local prefix="$2"
    local binary="${BINARY_DIR}/${prefix}"

    echo "=== Submitting ${prefix} on ${NP} ranks ==="
    sbatch --wait --exclusive \
        --job-name="d12_${prefix#desmore2012_interface_}" \
        --partition="${PARTITION}" \
        --ntasks="${NP}" \
        --time="${SLURM_TIME}" \
        --chdir="${ROOT}" \
        --output="${case_dir}/run.stdout.log" \
        --error="${case_dir}/run.stderr.log" \
        --wrap="timeout ${TIMEOUT_SECONDS} mpirun -np ${NP} ${binary}"
}

NEEDS_RUN=()
for i in "${!CASE_DIRS[@]}"; do
    case_dir="${CASE_DIRS[$i]}"
    prefix="${CASE_PREFIXES[$i]}"
    expected_rows="${CASE_ROWS[$i]}"
    has_events="${CASE_HAS_EVENTS[$i]}"

    if [[ "${RESUME}" == "1" ]] && \
       case_complete "${case_dir}" "${prefix}" "${expected_rows}" "${has_events}"; then
        echo "=== Reusing ${prefix} (${expected_rows} cells, G=${GROUPS}) ==="
        continue
    fi

    clean_case_outputs "${case_dir}" "${prefix}"
    build_case_binary "${case_dir}" "${prefix}"
    NEEDS_RUN+=("${i}")
done

# Submit every required case before waiting for any one of them.  Each case has
# a private copied executable, so later builds cannot replace a queued job's
# binary.  Slurm may still queue jobs if the cluster lacks six allocations.
PIDS=()
PID_CASES=()
for i in "${NEEDS_RUN[@]}"; do
    submit_case "${CASE_DIRS[$i]}" "${CASE_PREFIXES[$i]}" &
    PIDS+=("$!")
    PID_CASES+=("${CASE_PREFIXES[$i]}")
done

FAILED=0
for j in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$j]}"; then
        echo "Slurm job failed: ${PID_CASES[$j]}" >&2
        FAILED=1
    fi
done
[[ "${FAILED}" == "0" ]] || exit 1

for i in "${!CASE_DIRS[@]}"; do
    if ! case_complete "${CASE_DIRS[$i]}" "${CASE_PREFIXES[$i]}" \
            "${CASE_ROWS[$i]}" "${CASE_HAS_EVENTS[$i]}"; then
        echo "Incomplete or invalid output: ${CASE_PREFIXES[$i]}" >&2
        exit 1
    fi
done

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
    python3 regression_tests/cases/analyze_densmore2012_interface_events.py \
        "${case_dir}" --prefix "${prefix}" \
        | tee "${case_dir}/${prefix}_interface_event_analysis.txt"
}

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
