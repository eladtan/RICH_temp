#!/usr/bin/env bash

set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKS_SCRIPT="${ROOT_DIR}/regression_tests/lib/regression_checks.sh"
TESTS_DIR="${ROOT_DIR}/regression_tests/tests"

if [[ ! -f "${CHECKS_SCRIPT}" ]]; then
    echo "Missing checks script: ${CHECKS_SCRIPT}" >&2
    exit 2
fi
if [[ ! -d "${TESTS_DIR}" ]]; then
    echo "Missing tests directory: ${TESTS_DIR}" >&2
    exit 2
fi
source "${CHECKS_SCRIPT}"

CONFIG="gnuReleaseMPI"
MPI_NP=4
KEEP_ARTIFACTS=0
VERBOSE=0
TEST_FILTER=""
CLEAN_RESULTS=0

ARTIFACT_ROOT="${ROOT_DIR}/regression_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ARTIFACT_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}"

declare -a RESULT_NAMES=()
declare -a RESULT_STATUS=()
declare -a RESULT_DETAILS=()
declare -a RESULT_LOG_PATH=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --config <name>          Build configuration (default: ${CONFIG})
  --mpi-np <N>             MPI ranks for sedov_3d test (default: ${MPI_NP})
  --test <id>              Run only one test id (sod_1d|sedov_3d_mpi|till_compton)
  --clean-results          Delete regression_results directory and exit
  --keep-artifacts         Keep all logs even if all tests pass
  --verbose                Stream run output to terminal as well
  -h, --help               Show this help

Examples:
  ./regression_tests/run_all.sh
  ./regression_tests/run_all.sh --config gnuDebugMPI --mpi-np 8 --verbose
  ./regression_tests/run_all.sh --test sod_1d
  ./regression_tests/run_all.sh --clean-results
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="${2:-}"
            shift 2
            ;;
        --mpi-np)
            MPI_NP="${2:-}"
            shift 2
            ;;
        --test)
            TEST_FILTER="${2:-}"
            shift 2
            ;;
        --clean-results)
            CLEAN_RESULTS=1
            shift
            ;;
        --keep-artifacts)
            KEEP_ARTIFACTS=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "${CONFIG}" ]]; then
    echo "--config requires a non-empty value" >&2
    exit 2
fi
if ! [[ "${MPI_NP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--mpi-np must be a positive integer" >&2
    exit 2
fi
if [[ -n "${TEST_FILTER}" ]]; then
    case "${TEST_FILTER}" in
        sod_1d|sedov_3d_mpi|till_compton) ;;
        *)
            echo "--test must be one of: sod_1d, sedov_3d_mpi, till_compton" >&2
            exit 2
            ;;
    esac
fi

if [[ "${CLEAN_RESULTS}" -eq 1 ]]; then
    if [[ -d "${ARTIFACT_ROOT}" ]]; then
        rm -rf "${ARTIFACT_ROOT}"
        echo "Removed ${ARTIFACT_ROOT}"
    else
        echo "No results directory to clean (${ARTIFACT_ROOT})"
    fi
    exit 0
fi

mkdir -p "${RUN_ARTIFACT_DIR}"

echo "Running local regression suite"
echo "Root: ${ROOT_DIR}"
echo "Config: ${CONFIG}"
if [[ -z "${TEST_FILTER}" || "${TEST_FILTER}" == "sedov_3d_mpi" ]]; then
    echo "Sedov MPI ranks: ${MPI_NP}"
fi
echo "Artifacts: ${RUN_ARTIFACT_DIR}"
echo

run_case() {
    local case_name="$1"
    local test_name="$2"
    local run_dir_rel="$3"
    local run_cmd="$4"
    local check_fn="$5"
    local build_args_str="${6-}"
    local run_mode="${7-direct}"
    local slurm_ntasks="${8-32}"
    local slurm_partition="${9-bigrun}"
    local slurm_exclusive="${10-1}"
    local case_dir="${RUN_ARTIFACT_DIR}/${case_name}"
    local build_stdout="${case_dir}/build.stdout.log"
    local build_stderr="${case_dir}/build.stderr.log"
    local run_stdout="${case_dir}/run.stdout.log"
    local run_stderr="${case_dir}/run.stderr.log"
    local run_dir_abs="${ROOT_DIR}/${run_dir_rel}"
    local run_start_epoch
    local status
    local detail
    local cmd_exit

    mkdir -p "${case_dir}"
    echo "==> [${case_name}] building (${test_name})"

    local -a build_cmd=("${ROOT_DIR}/build_rich.sh" "${CONFIG}" "--test_name=${test_name}")
    if [[ -n "${build_args_str}" ]]; then
        local -a extra_args=()
        read -r -a extra_args <<< "${build_args_str}"
        build_cmd+=("${extra_args[@]}")
    fi
    "${build_cmd[@]}" >"${build_stdout}" 2>"${build_stderr}"
    if [[ $? -ne 0 ]]; then
        status="FAIL"
        detail="build failed (see build.stderr.log)"
        RESULT_NAMES+=("${case_name}")
        RESULT_STATUS+=("${status}")
        RESULT_DETAILS+=("${detail}")
        RESULT_LOG_PATH+=("${case_dir}")
        echo "    ${status}: ${detail}"
        return 1
    fi

    if [[ ! -x "${ROOT_DIR}/build/${CONFIG}/rich" && ! -L "${ROOT_DIR}/build/${CONFIG}/rich" ]]; then
        status="FAIL"
        detail="built binary not found at build/${CONFIG}/rich"
        RESULT_NAMES+=("${case_name}")
        RESULT_STATUS+=("${status}")
        RESULT_DETAILS+=("${detail}")
        RESULT_LOG_PATH+=("${case_dir}")
        echo "    ${status}: ${detail}"
        return 1
    fi

    if [[ ! -d "${run_dir_abs}" ]]; then
        status="FAIL"
        detail="run directory does not exist: ${run_dir_rel}"
        RESULT_NAMES+=("${case_name}")
        RESULT_STATUS+=("${status}")
        RESULT_DETAILS+=("${detail}")
        RESULT_LOG_PATH+=("${case_dir}")
        echo "    ${status}: ${detail}"
        return 1
    fi

    echo "==> [${case_name}] running"
    run_start_epoch="$(date +%s)"
    pushd "${run_dir_abs}" >/dev/null || {
        status="FAIL"
        detail="failed to enter run directory"
        RESULT_NAMES+=("${case_name}")
        RESULT_STATUS+=("${status}")
        RESULT_DETAILS+=("${detail}")
        RESULT_LOG_PATH+=("${case_dir}")
        echo "    ${status}: ${detail}"
        return 1
    }

    if [[ "${run_mode}" == "slurm" ]]; then
        local escaped_run_cmd="${run_cmd//\'/\'\\\'\'}"
        local sbatch_wrap_cmd
        sbatch_wrap_cmd="ROOT_DIR=\"${ROOT_DIR}\" CONFIG=\"${CONFIG}\" MPI_NP=\"${MPI_NP}\" SLURM_NTASKS=\"${slurm_ntasks}\" bash -c '${escaped_run_cmd}'"
        local -a sbatch_cmd=(
            sbatch
            --wait
            --ntasks="${slurm_ntasks}"
            --partition="${slurm_partition}"
            --output="${run_stdout}"
            --error="${run_stderr}"
            --chdir="${run_dir_abs}"
            --wrap "${sbatch_wrap_cmd}"
        )
        if [[ "${slurm_exclusive}" == "1" ]]; then
            sbatch_cmd+=(--exclusive)
        fi
        "${sbatch_cmd[@]}"
        cmd_exit=$?
    else
        if [[ "${VERBOSE}" -eq 1 ]]; then
            ROOT_DIR="${ROOT_DIR}" CONFIG="${CONFIG}" MPI_NP="${MPI_NP}" SLURM_NTASKS="${slurm_ntasks}" \
                bash -c "${run_cmd}" > >(tee "${run_stdout}") 2> >(tee "${run_stderr}" >&2)
            cmd_exit=$?
        else
            ROOT_DIR="${ROOT_DIR}" CONFIG="${CONFIG}" MPI_NP="${MPI_NP}" SLURM_NTASKS="${slurm_ntasks}" \
                bash -c "${run_cmd}" >"${run_stdout}" 2>"${run_stderr}"
            cmd_exit=$?
        fi
    fi
    popd >/dev/null || true

    if [[ ${cmd_exit} -ne 0 ]]; then
        status="FAIL"
        detail="run command failed with exit code ${cmd_exit}"
        RESULT_NAMES+=("${case_name}")
        RESULT_STATUS+=("${status}")
        RESULT_DETAILS+=("${detail}")
        RESULT_LOG_PATH+=("${case_dir}")
        echo "    ${status}: ${detail}"
        return 1
    fi

    if "${check_fn}" "${run_dir_abs}" "${run_start_epoch}" "${run_stdout}" "${run_stderr}"; then
        status="PASS"
        detail="${REGRESSION_CHECK_MSG}"
        RESULT_NAMES+=("${case_name}")
        RESULT_STATUS+=("${status}")
        RESULT_DETAILS+=("${detail}")
        RESULT_LOG_PATH+=("${case_dir}")
        echo "    ${status}: ${detail}"
        return 0
    fi

    status="FAIL"
    detail="${REGRESSION_CHECK_MSG}"
    RESULT_NAMES+=("${case_name}")
    RESULT_STATUS+=("${status}")
    RESULT_DETAILS+=("${detail}")
    RESULT_LOG_PATH+=("${case_dir}")
    echo "    ${status}: ${detail}"
    return 1
}

run_definition() {
    local def_file="$1"
    local TEST_ID=""
    local BUILD_TEST_NAME=""
    local RUN_DIR_REL=""
    local RUN_COMMAND=""
    local CHECK_FUNCTION=""
    local BUILD_ARGS=""
    local RUN_MODE="direct"
    local SLURM_NTASKS="32"
    local SLURM_PARTITION="bigrun"
    local SLURM_EXCLUSIVE="1"

    source "${def_file}"

    if [[ -z "${TEST_ID}" || -z "${BUILD_TEST_NAME}" || -z "${RUN_DIR_REL}" || -z "${RUN_COMMAND}" || -z "${CHECK_FUNCTION}" ]]; then
        echo "Invalid test definition: ${def_file}" >&2
        RESULT_NAMES+=("$(basename "${def_file}")")
        RESULT_STATUS+=("FAIL")
        RESULT_DETAILS+=("invalid test definition")
        RESULT_LOG_PATH+=("${RUN_ARTIFACT_DIR}")
        return 1
    fi

    run_case \
        "${TEST_ID}" \
        "${BUILD_TEST_NAME}" \
        "${RUN_DIR_REL}" \
        "${RUN_COMMAND}" \
        "${CHECK_FUNCTION}" \
        "${BUILD_ARGS-}" \
        "${RUN_MODE}" \
        "${SLURM_NTASKS}" \
        "${SLURM_PARTITION}" \
        "${SLURM_EXCLUSIVE}"
}

TOTAL_FAILURES=0

mapfile -t TEST_FILES < <(printf '%s\n' "${TESTS_DIR}"/*.sh | sort)
for test_file in "${TEST_FILES[@]}"; do
    if [[ ! -f "${test_file}" ]]; then
        continue
    fi
    if [[ -n "${TEST_FILTER}" ]]; then
        test_id="$(basename "${test_file}" .sh)"
        if [[ "${test_id}" != "${TEST_FILTER}" ]]; then
            continue
        fi
    fi
    run_definition "${test_file}" || TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
done

echo
echo "Regression Summary"
printf "%-16s  %-6s  %-52s  %s\n" "Case" "Status" "Details" "Logs"
printf "%-16s  %-6s  %-52s  %s\n" "----------------" "------" "----------------------------------------------------" "----"

for i in "${!RESULT_NAMES[@]}"; do
    printf "%-16s  %-6s  %-52s  %s\n" \
        "${RESULT_NAMES[$i]}" \
        "${RESULT_STATUS[$i]}" \
        "${RESULT_DETAILS[$i]}" \
        "${RESULT_LOG_PATH[$i]}"
done

echo
if [[ ${TOTAL_FAILURES} -eq 0 ]]; then
    echo "All regression tests passed."
    if [[ "${KEEP_ARTIFACTS}" -eq 0 ]]; then
        rm -rf "${RUN_ARTIFACT_DIR}"
        echo "Removed success artifacts (use --keep-artifacts to retain logs)."
    fi
    exit 0
fi

echo "${TOTAL_FAILURES} test(s) failed."
exit 1
