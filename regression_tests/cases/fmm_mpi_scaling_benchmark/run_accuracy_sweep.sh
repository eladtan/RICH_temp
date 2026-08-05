#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <rich-binary> <result-directory>" >&2
    exit 2
fi

RICH_BIN="$(readlink -f "$1")"
RESULT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PARTICLES="${FMM_TUNE_PARTICLES:-10000000}"
NODES="${FMM_TUNE_NODES:-6}"
RANKS_PER_NODE="${FMM_TUNE_RANKS_PER_NODE:-192}"
REPEATS="${FMM_TUNE_REPEATS:-1}"
MAX_REMOTE_MIB="${FMM_TUNE_MAX_REMOTE_MIB:-512}"
OPERATOR_CACHE_MIB="${FMM_TUNE_OPERATOR_CACHE_MIB:-128}"
MEAN_ERROR_TARGET="${FMM_TUNE_MEAN_ERROR_TARGET:-1e-3}"
MAX_ERROR_TARGET="${FMM_TUNE_MAX_ERROR_TARGET:-5e-3}"
QUADRUPOLE_THETA="${FMM_TUNE_QUADRUPOLE_THETA:-0.5}"
RANKS=$((NODES * RANKS_PER_NODE))

if [[ ! -x "${RICH_BIN}" ]]; then
    echo "Benchmark binary is not executable: ${RICH_BIN}" >&2
    exit 2
fi
for value in "${PARTICLES}" "${NODES}" "${RANKS_PER_NODE}" "${REPEATS}" \
             "${MAX_REMOTE_MIB}" "${OPERATOR_CACHE_MIB}"; do
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Integer tuning parameters must be positive: ${value}" >&2
        exit 2
    fi
done
if [[ -n "${SLURM_JOB_NUM_NODES:-}" && "${SLURM_JOB_NUM_NODES}" -ne "${NODES}" ]]; then
    echo "Expected ${NODES} allocated nodes; got ${SLURM_JOB_NUM_NODES}" >&2
    exit 2
fi
if [[ -n "${SLURM_NTASKS:-}" && "${SLURM_NTASKS}" -lt "${RANKS}" ]]; then
    echo "Expected at least ${RANKS} allocated tasks; got ${SLURM_NTASKS}" >&2
    exit 2
fi

mkdir -p "${RESULT_DIR}"
RESULT_DIR="$(readlink -f "${RESULT_DIR}")"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"
unset OMPI_MCA_topo || true

DEFAULT_CONFIGS="2:0.50:32,2:0.70:32,2:0.90:32,3:0.70:32,3:0.90:32,3:1.00:32,4:0.90:32,4:1.00:32"
CONFIG_LIST="${FMM_TUNE_CONFIGS:-${DEFAULT_CONFIGS}}"
IFS=',' read -r -a configs <<< "${CONFIG_LIST}"

for config in "${configs[@]}"; do
    IFS=':' read -r order theta leaf <<< "${config}"
    if [[ -z "${order}" || -z "${theta}" || -z "${leaf}" ]]; then
        echo "Invalid FMM_TUNE_CONFIGS entry: ${config}" >&2
        exit 2
    fi
    theta_tag="${theta//./}"
    tag="p${order}_t${theta_tag}_l${leaf}"
    output="${RESULT_DIR}/${tag}.txt"
    stdout_log="${RESULT_DIR}/${tag}.stdout.log"
    stderr_log="${RESULT_DIR}/${tag}.stderr.log"
    rm -f "${output}" "${stdout_log}" "${stderr_log}"

    echo "=== ${tag}: particles=${PARTICLES}, ranks=${RANKS}, cache_mib=${OPERATOR_CACHE_MIB} ==="
    set +e
    mpirun -np "${RANKS}" \
        --map-by "ppr:${RANKS_PER_NODE}:node" \
        --bind-to core \
        "${RICH_BIN}" \
        --particles "${PARTICLES}" \
        --expected-nodes "${NODES}" \
        --expected-ranks-per-node "${RANKS_PER_NODE}" \
        --repeats "${REPEATS}" \
        --warm-only \
        --fmm-order "${order}" \
        --fmm-theta "${theta}" \
        --quadrupole-theta "${QUADRUPOLE_THETA}" \
        --fmm-leaf-capacity "${leaf}" \
        --fmm-mean-error-target "${MEAN_ERROR_TARGET}" \
        --fmm-max-error-target "${MAX_ERROR_TARGET}" \
        --fmm-max-remote-mib "${MAX_REMOTE_MIB}" \
        --fmm-operator-cache-mib "${OPERATOR_CACHE_MIB}" \
        --output "${output}" \
        >"${stdout_log}" 2>"${stderr_log}"
    status=$?
    set -e

    if [[ ! -s "${output}" ]] || ! grep -q '^pass [01]$' "${output}"; then
        echo "${tag} failed before producing a complete result (mpirun status ${status})." >&2
        tail -n 40 "${stdout_log}" >&2 || true
        tail -n 40 "${stderr_log}" >&2 || true
        exit 1
    fi

    pass_value="$(awk '$1 == "pass" { value=$2 } END { print value }' "${output}")"
    echo "${tag}: benchmark_pass=${pass_value}, mpirun_status=${status}"
done

python3 "${SCRIPT_DIR}/summarize_accuracy_sweep.py" \
    "${RESULT_DIR}" \
    --output "${RESULT_DIR}/accuracy_sweep_summary.tsv"
