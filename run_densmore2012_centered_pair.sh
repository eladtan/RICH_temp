#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

CONFIG="${CONFIG:-intelReleaseMPI}"
NP="${NP:-32}"
D12_ENERGY_GROUPS="${D12_ENERGY_GROUPS:-30}"
PARTITION="${PARTITION:-bigrun}"
SLURM_TIME="${SLURM_TIME:-30}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-71500}"
RESUME="${RESUME:-0}"

if command -v ml >/dev/null 2>&1; then
    ml openmpi/4.1.6/Intel/OneApi/2024.2.1
fi

MC_DIR="regression_tests/cases/desmore2012_interface_mc_centered"
MC_PREFIX="desmore2012_interface_mc_centered"
DDMC_DIR="regression_tests/cases/desmore2012_interface_ddmc_centered"
DDMC_PREFIX="desmore2012_interface_ddmc_centered"
EXPECTED_ROWS=315

BINARY_DIR="${ROOT}/.densmore2012_centered_pair_bins/${CONFIG}_g${D12_ENERGY_GROUPS}"
mkdir -p "${BINARY_DIR}"

case_complete() {
    local case_dir="$1"
    local prefix="$2"
    local expect_events="$3"
    local cells_file="${case_dir}/${prefix}_cells.tsv"

    [[ -s "${cells_file}" ]] || return 1

    local rows
    rows="$(awk 'END { print (NR > 0 ? NR - 1 : 0) }' "${cells_file}")"
    [[ "${rows}" == "${EXPECTED_ROWS}" ]] || return 1

    [[ -s "${case_dir}/run.stdout.log" ]] || return 1
    grep -q "G=${D12_ENERGY_GROUPS}" "${case_dir}/run.stdout.log" || return 1
    grep -q "new/cell=25" "${case_dir}/run.stdout.log" || return 1
    grep -q "max/cell=100" "${case_dir}/run.stdout.log" || return 1

    if grep -Eqi 'UniversalError|terminate called|Segmentation fault|MPI_ABORT|StuckParticle' \
        "${case_dir}/run.stdout.log" "${case_dir}/run.stderr.log"; then
        return 1
    fi

    if [[ "${expect_events}" == "1" ]]; then
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

build_case() {
    local case_dir="$1"
    local prefix="$2"
    local binary="${BINARY_DIR}/${prefix}"

    echo "=== Building ${prefix} ==="
    ./build_rich.sh "${CONFIG}" \
        --test_name="${case_dir}" \
        --energy_groups_num="${D12_ENERGY_GROUPS}"

    cp -L "./build/${CONFIG}/rich" "${binary}"
    chmod +x "${binary}"
}

run_case() {
    local case_dir="$1"
    local prefix="$2"
    local binary="${BINARY_DIR}/${prefix}"

    echo "=== Running ${prefix} on ${NP} ranks ==="
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

CASE_DIRS=("${MC_DIR}" "${DDMC_DIR}")
CASE_PREFIXES=("${MC_PREFIX}" "${DDMC_PREFIX}")
CASE_EVENTS=(0 1)
NEEDS_RUN=()

for i in "${!CASE_DIRS[@]}"; do
    if [[ "${RESUME}" == "1" ]] && \
       case_complete "${CASE_DIRS[$i]}" "${CASE_PREFIXES[$i]}" "${CASE_EVENTS[$i]}"; then
        echo "=== Reusing ${CASE_PREFIXES[$i]} ==="
        continue
    fi

    clean_case_outputs "${CASE_DIRS[$i]}" "${CASE_PREFIXES[$i]}"
    build_case "${CASE_DIRS[$i]}" "${CASE_PREFIXES[$i]}"
    NEEDS_RUN+=("${i}")
done

PIDS=()
PID_CASES=()
for i in "${NEEDS_RUN[@]}"; do
    run_case "${CASE_DIRS[$i]}" "${CASE_PREFIXES[$i]}" &
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
    if ! case_complete "${CASE_DIRS[$i]}" "${CASE_PREFIXES[$i]}" "${CASE_EVENTS[$i]}"; then
        echo "Incomplete or invalid output: ${CASE_PREFIXES[$i]}" >&2
        exit 1
    fi
done

OUTPUT_STEM="densmore2012_interface_comparison_centered_equal_cells"
python3 regression_tests/cases/compare_densmore2012_interface.py \
    --mc "${MC_DIR}/${MC_PREFIX}_cells.tsv" \
    --ddmc "${DDMC_DIR}/${DDMC_PREFIX}_cells.tsv" \
    --output "regression_tests/cases/${OUTPUT_STEM}.tsv" \
    | tee "regression_tests/cases/${OUTPUT_STEM}.txt"

python3 regression_tests/cases/analyze_densmore2012_interface_events.py \
    "${DDMC_DIR}" --prefix "${DDMC_PREFIX}" \
    | tee "${DDMC_DIR}/${DDMC_PREFIX}_interface_event_analysis.txt"

PLOT_STEM="regression_tests/cases/densmore2012_centered_equal_cells_overlay"
python3 regression_tests/cases/plot_densmore2012_centered_pair.py \
    --mc "${MC_DIR}/${MC_PREFIX}_profile.txt" \
    --ddmc "${DDMC_DIR}/${DDMC_PREFIX}_profile.txt" \
    --reference "regression_tests/cases/desmore2012_mc/data/densmore2012_fig4_mc.csv" \
    --output-stem "${PLOT_STEM}"

echo "=== Centered equal-interface-cell IMC/DDMC pair complete ==="
echo "Mesh at x=2: [1.995,2.000] and [2.000,2.005] cm; both widths 0.005 cm."
echo "Comparison outputs:"
echo "  regression_tests/cases/${OUTPUT_STEM}.txt"
echo "  regression_tests/cases/${OUTPUT_STEM}.tsv"
echo "Plot outputs:"
echo "  ${PLOT_STEM}.png"
echo "  ${PLOT_STEM}.pdf"
echo "  ${PLOT_STEM}_interface_zoom.png"
echo "  ${PLOT_STEM}_interface_zoom.pdf"
