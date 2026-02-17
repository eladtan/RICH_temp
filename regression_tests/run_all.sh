#!/usr/bin/env bash

set -u

# ==================== Colors ====================
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
ORANGE=$'\033[0;33m'
CYAN=$'\033[0;36m'
BOLD=$'\033[1m'
NC=$'\033[0m'

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

# ==================== Defaults ====================
CONFIG=""
CONFIG_EXPLICIT=0
MPI_NP=4
KEEP_ARTIFACTS=0
VERBOSE=0
TEST_FILTER=""
CLEAN_RESULTS=0
MODE="all"

ARTIFACT_ROOT="${ROOT_DIR}/regression_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ARTIFACT_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}"

# ==================== Result arrays ====================
declare -a RESULT_NAMES=()
declare -a RESULT_STATUS=()
declare -a RESULT_DETAILS=()
declare -a RESULT_LOG_PATH=()

# ==================== Discover valid test IDs ====================
discover_test_ids() {
    local ids=""
    for f in "${TESTS_DIR}"/*.sh; do
        [[ -f "$f" ]] || continue
        ids+="$(basename "$f" .sh)|"
    done
    echo "${ids%|}"
}
VALID_TEST_IDS="$(discover_test_ids)"

# ==================== Usage ====================
usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --mode <mode>            Run mode (default: all). See Modes below.
  --config <name>          Build configuration (auto-derived from --mode if omitted)
  --mpi-np <N>             MPI ranks for sedov_3d test (default: ${MPI_NP})
  --test <id>              Run only one test id (${VALID_TEST_IDS//|/, })
  --clean-results          Delete regression_results directory and exit
  --keep-artifacts         Keep all logs even if all tests pass
  --verbose                Stream run output to terminal as well
  -h, --help               Show this help

Modes:
  serial           Run tests tagged "serial" (default config: gnuRelease)
  mpi              Run tests tagged "mpi"    (default config: gnuReleaseMPI)
  all              Run all tests             (default config: gnuReleaseMPI)
  serial_then_mpi  Run serial tests first (gnuRelease), then MPI tests (gnuReleaseMPI)

Examples:
  ./regression_tests/run_all.sh --mode serial
  ./regression_tests/run_all.sh --mode mpi --config intelReleaseMPI
  ./regression_tests/run_all.sh --mode serial_then_mpi
  ./regression_tests/run_all.sh --test sod_1d --config gnuRelease
  ./regression_tests/run_all.sh --clean-results
EOF
}

# ==================== Parse arguments ====================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:-}"
            shift 2
            ;;
        --config)
            CONFIG="${2:-}"
            CONFIG_EXPLICIT=1
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

# ==================== Validate arguments ====================
case "${MODE}" in
    serial|mpi|all|serial_then_mpi) ;;
    *)
        echo "--mode must be one of: serial, mpi, all, serial_then_mpi" >&2
        exit 2
        ;;
esac

# serial_then_mpi: re-invoke ourselves twice and merge results
if [[ "${MODE}" == "serial_then_mpi" ]]; then
    echo "${BOLD}=== serial_then_mpi: running serial pass ===${NC}"
    passthrough_args=()
    [[ "${KEEP_ARTIFACTS}" -eq 1 ]] && passthrough_args+=(--keep-artifacts)
    [[ "${VERBOSE}" -eq 1 ]]       && passthrough_args+=(--verbose)
    [[ -n "${TEST_FILTER}" ]]      && passthrough_args+=(--test "${TEST_FILTER}")

    mpi_config="${CONFIG}"
    if [[ "${CONFIG_EXPLICIT}" -eq 0 ]]; then
        serial_config="gnuRelease"
        mpi_config="gnuReleaseMPI"
    elif [[ "${CONFIG}" == *MPI* ]]; then
        # Derive the serial config by stripping "MPI" from the config name
        serial_config="${CONFIG//MPI/}"
        echo "  Derived serial config: ${serial_config} (from ${CONFIG})"
    else
        serial_config="${CONFIG}"
    fi

    serial_rc=0
    "${BASH_SOURCE[0]}" --mode serial --config "${serial_config}" \
        --mpi-np "${MPI_NP}" "${passthrough_args[@]}" || serial_rc=$?

    echo
    echo "${BOLD}=== serial_then_mpi: running MPI pass ===${NC}"

    mpi_rc=0
    "${BASH_SOURCE[0]}" --mode mpi --config "${mpi_config}" \
        --mpi-np "${MPI_NP}" "${passthrough_args[@]}" || mpi_rc=$?

    echo
    if [[ ${serial_rc} -eq 0 && ${mpi_rc} -eq 0 ]]; then
        echo "${GREEN}serial_then_mpi: all passes succeeded.${NC}"
        exit 0
    else
        echo "${RED}serial_then_mpi: failures detected (serial=${serial_rc}, mpi=${mpi_rc}).${NC}"
        exit 1
    fi
fi

# Auto-derive config from mode if not explicitly set
if [[ "${CONFIG_EXPLICIT}" -eq 0 ]]; then
    case "${MODE}" in
        serial) CONFIG="gnuRelease" ;;
        mpi)    CONFIG="gnuReleaseMPI" ;;
        all)    CONFIG="gnuReleaseMPI" ;;
    esac
fi

if [[ -z "${CONFIG}" ]]; then
    echo "--config requires a non-empty value" >&2
    exit 2
fi

# Warn about mode/config mismatch
if [[ "${MODE}" == "serial" && "${CONFIG}" == *MPI* ]]; then
    echo "${ORANGE}Warning: --mode serial with MPI config '${CONFIG}'. Serial tests will still run directly.${NC}" >&2
fi
if [[ "${MODE}" == "mpi" && "${CONFIG}" != *MPI* ]]; then
    echo "${RED}Error: --mode mpi requires an MPI config (name must contain 'MPI'), got '${CONFIG}'${NC}" >&2
    exit 2
fi

if ! [[ "${MPI_NP}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--mpi-np must be a positive integer" >&2
    exit 2
fi

if [[ -n "${TEST_FILTER}" ]]; then
    if ! echo "${TEST_FILTER}" | grep -qE "^(${VALID_TEST_IDS})$"; then
        echo "--test must be one of: ${VALID_TEST_IDS//|/, }" >&2
        exit 2
    fi
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

# ==================== Helper: check if test matches mode ====================
test_matches_mode() {
    local tags="$1"
    case "${MODE}" in
        all) return 0 ;;
        serial)
            [[ " ${tags} " == *" serial "* ]] && return 0
            return 1
            ;;
        mpi)
            [[ " ${tags} " == *" mpi "* ]] && return 0
            return 1
            ;;
    esac
    return 1
}

# ==================== Helper: formatted status line ====================
print_status() {
    local phase="$1"
    local test_id="$2"
    local msg="$3"
    local color="${4:-${NC}}"
    printf "${color}[%-5s] %-20s %s${NC}\n" "${phase}" "${test_id}" "${msg}"
}

# ==================== Collect test definitions ====================
# Each test is stored in parallel arrays indexed by position.
declare -a ALL_TEST_IDS=()
declare -a ALL_BUILD_TEST_NAMES=()
declare -a ALL_RUN_DIR_RELS=()
declare -a ALL_RUN_COMMANDS=()
declare -a ALL_CHECK_FUNCTIONS=()
declare -a ALL_BUILD_ARGS=()
declare -a ALL_RUN_MODES=()
declare -a ALL_SLURM_NTASKS=()
declare -a ALL_SLURM_PARTITIONS=()
declare -a ALL_SLURM_EXCLUSIVES=()
declare -a ALL_CASE_DIRS=()

load_test_definition() {
    local def_file="$1"
    # Reset variables before sourcing
    local TEST_ID=""
    local TAGS=""
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

    # Validate required fields
    if [[ -z "${TEST_ID}" || -z "${BUILD_TEST_NAME}" || -z "${RUN_DIR_REL}" || -z "${RUN_COMMAND}" || -z "${CHECK_FUNCTION}" ]]; then
        echo "Invalid test definition: ${def_file}" >&2
        return 1
    fi

    # Filter by --test
    if [[ -n "${TEST_FILTER}" && "${TEST_ID}" != "${TEST_FILTER}" ]]; then
        return 2  # skipped, not an error
    fi

    # Filter by --mode using TAGS
    if [[ -z "${TAGS}" ]]; then
        TAGS="serial"  # default to serial if no tags
    fi
    if ! test_matches_mode "${TAGS}"; then
        return 2  # skipped by mode filter
    fi

    local case_dir="${RUN_ARTIFACT_DIR}/${TEST_ID}"

    ALL_TEST_IDS+=("${TEST_ID}")
    ALL_BUILD_TEST_NAMES+=("${BUILD_TEST_NAME}")
    ALL_RUN_DIR_RELS+=("${RUN_DIR_REL}")
    ALL_RUN_COMMANDS+=("${RUN_COMMAND}")
    ALL_CHECK_FUNCTIONS+=("${CHECK_FUNCTION}")
    ALL_BUILD_ARGS+=("${BUILD_ARGS}")
    ALL_RUN_MODES+=("${RUN_MODE}")
    ALL_SLURM_NTASKS+=("${SLURM_NTASKS}")
    ALL_SLURM_PARTITIONS+=("${SLURM_PARTITION}")
    ALL_SLURM_EXCLUSIVES+=("${SLURM_EXCLUSIVE}")
    ALL_CASE_DIRS+=("${case_dir}")
    return 0
}

# Discover and load all matching tests
mapfile -t TEST_FILES < <(printf '%s\n' "${TESTS_DIR}"/*.sh | sort)
for test_file in "${TEST_FILES[@]}"; do
    [[ -f "${test_file}" ]] || continue
    load_test_definition "${test_file}" || {
        rc=$?
        if [[ ${rc} -eq 1 ]]; then
            # Invalid definition — record as failure
            RESULT_NAMES+=("$(basename "${test_file}" .sh)")
            RESULT_STATUS+=("FAIL")
            RESULT_DETAILS+=("invalid test definition")
            RESULT_LOG_PATH+=("${RUN_ARTIFACT_DIR}")
        fi
        # rc==2 means skipped by filter, not an error
    }
done

NUM_TESTS=${#ALL_TEST_IDS[@]}
if [[ ${NUM_TESTS} -eq 0 ]]; then
    echo "No tests matched the current filters (mode=${MODE}, test=${TEST_FILTER:-<all>})." >&2
    exit 0
fi

# ==================== Print header ====================
mkdir -p "${RUN_ARTIFACT_DIR}"

echo "${BOLD}Running regression suite${NC}"
echo "  Mode:      ${MODE}"
echo "  Config:    ${CONFIG}"
echo "  Tests:     ${ALL_TEST_IDS[*]}"
if [[ "${MODE}" != "serial" ]]; then
    echo "  MPI ranks: ${MPI_NP}"
fi
echo "  Artifacts: ${RUN_ARTIFACT_DIR}"
echo

# ==================== Track which tests built successfully ====================
declare -a BUILD_OK=()  # 1=success, 0=failure for each index

# Record epoch before any compilation starts.  Used later by check functions
# to verify output files are fresh (is_nonempty_and_newer).  We use this early
# timestamp rather than a per-test one so that SLURM clock skew between the
# head node and compute nodes does not cause false "stale" failures.
SUITE_START_EPOCH="$(date +%s)"

# ==========================================================================
#  PHASE 1: BUILD (sequential)
# ==========================================================================
echo "${BOLD}=== BUILD PHASE ===${NC}"

for i in "${!ALL_TEST_IDS[@]}"; do
    test_id="${ALL_TEST_IDS[$i]}"
    build_test_name="${ALL_BUILD_TEST_NAMES[$i]}"
    build_args="${ALL_BUILD_ARGS[$i]}"
    case_dir="${ALL_CASE_DIRS[$i]}"

    mkdir -p "${case_dir}"

    local_build_stdout="${case_dir}/build.stdout.log"
    local_build_stderr="${case_dir}/build.stderr.log"

    extra_info=""
    if [[ -n "${build_args}" ]]; then
        extra_info=" (${build_args})"
    fi
    print_status "BUILD" "${test_id}" "compiling${extra_info}..." "${CYAN}"

    # Assemble build command
    build_cmd=("${ROOT_DIR}/build_rich.sh" "${CONFIG}" "--test_name=${build_test_name}")
    if [[ -n "${build_args}" ]]; then
        declare -a extra_args=()
        read -r -a extra_args <<< "${build_args}"
        build_cmd+=("${extra_args[@]}")
    fi

    # Run the build
    if ! "${build_cmd[@]}" >"${local_build_stdout}" 2>"${local_build_stderr}"; then
        print_status "BUILD" "${test_id}" "FAIL (build failed, see build.stderr.log)" "${RED}"
        RESULT_NAMES+=("${test_id}")
        RESULT_STATUS+=("FAIL")
        RESULT_DETAILS+=("build failed (see build.stderr.log)")
        RESULT_LOG_PATH+=("${case_dir}")
        BUILD_OK+=("0")
        continue
    fi

    # Verify binary exists
    if [[ ! -x "${ROOT_DIR}/build/${CONFIG}/rich" && ! -L "${ROOT_DIR}/build/${CONFIG}/rich" ]]; then
        print_status "BUILD" "${test_id}" "FAIL (binary not found at build/${CONFIG}/rich)" "${RED}"
        RESULT_NAMES+=("${test_id}")
        RESULT_STATUS+=("FAIL")
        RESULT_DETAILS+=("binary not found at build/${CONFIG}/rich")
        RESULT_LOG_PATH+=("${case_dir}")
        BUILD_OK+=("0")
        continue
    fi

    # Copy the binary (dereference symlink) so the next build doesn't overwrite it
    cp -L "${ROOT_DIR}/build/${CONFIG}/rich" "${case_dir}/rich"
    chmod +x "${case_dir}/rich"

    print_status "BUILD" "${test_id}" "OK (binary copied)" "${GREEN}"
    BUILD_OK+=("1")
done

echo

# ==========================================================================
#  PHASE 2: RUN (parallel)
# ==========================================================================
echo "${BOLD}=== RUN PHASE (parallel) ===${NC}"

declare -A RUN_PIDS=()         # test_id -> PID
declare -A RUN_START_EPOCHS=() # test_id -> epoch seconds (wall clock, for elapsed display)
declare -A RUN_INDICES=()      # test_id -> index into ALL_* arrays

for i in "${!ALL_TEST_IDS[@]}"; do
    if [[ "${BUILD_OK[$i]}" != "1" ]]; then
        continue
    fi

    test_id="${ALL_TEST_IDS[$i]}"
    run_dir_rel="${ALL_RUN_DIR_RELS[$i]}"
    run_cmd="${ALL_RUN_COMMANDS[$i]}"
    run_mode="${ALL_RUN_MODES[$i]}"
    slurm_ntasks="${ALL_SLURM_NTASKS[$i]}"
    slurm_partition="${ALL_SLURM_PARTITIONS[$i]}"
    slurm_exclusive="${ALL_SLURM_EXCLUSIVES[$i]}"
    case_dir="${ALL_CASE_DIRS[$i]}"

    run_dir_abs="${ROOT_DIR}/${run_dir_rel}"
    run_stdout="${case_dir}/run.stdout.log"
    run_stderr="${case_dir}/run.stderr.log"
    rich_bin="${case_dir}/rich"

    if [[ ! -d "${run_dir_abs}" ]]; then
        print_status "RUN" "${test_id}" "FAIL (run directory missing: ${run_dir_rel})" "${RED}"
        RESULT_NAMES+=("${test_id}")
        RESULT_STATUS+=("FAIL")
        RESULT_DETAILS+=("run directory does not exist: ${run_dir_rel}")
        RESULT_LOG_PATH+=("${case_dir}")
        continue
    fi

    print_status "RUN" "${test_id}" "started" "${CYAN}"
    RUN_START_EPOCHS["${test_id}"]="$(date +%s)"
    RUN_INDICES["${test_id}"]="${i}"

    # Launch the test in the background
    (
        cd "${run_dir_abs}" || exit 1

        export ROOT_DIR CONFIG MPI_NP SLURM_NTASKS="${slurm_ntasks}"
        export RICH_BIN="${rich_bin}"

        if [[ "${run_mode}" == "slurm" ]]; then
            local_escaped_run_cmd="${run_cmd//\'/\'\\\'\'}"
            sbatch_wrap_cmd="ROOT_DIR=\"${ROOT_DIR}\" CONFIG=\"${CONFIG}\" MPI_NP=\"${MPI_NP}\" SLURM_NTASKS=\"${slurm_ntasks}\" RICH_BIN=\"${rich_bin}\" bash -c '${local_escaped_run_cmd}'"
            sbatch_args=(
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
                sbatch_args+=(--exclusive)
            fi
            "${sbatch_args[@]}"
        else
            if [[ "${VERBOSE}" -eq 1 ]]; then
                bash -c "${run_cmd}" > >(tee "${run_stdout}") 2> >(tee "${run_stderr}" >&2)
            else
                bash -c "${run_cmd}" >"${run_stdout}" 2>"${run_stderr}"
            fi
        fi
    ) &
    RUN_PIDS["${test_id}"]=$!
done

echo

# ==========================================================================
#  PHASE 3: WAIT & CHECK
# ==========================================================================
echo "${BOLD}=== RESULTS ===${NC}"

TOTAL_FAILURES=0

# Wait for each running test and check results
for test_id in "${!RUN_PIDS[@]}"; do
    pid="${RUN_PIDS[${test_id}]}"
    idx="${RUN_INDICES[${test_id}]}"
    case_dir="${ALL_CASE_DIRS[$idx]}"
    run_dir_rel="${ALL_RUN_DIR_RELS[$idx]}"
    run_dir_abs="${ROOT_DIR}/${run_dir_rel}"
    check_fn="${ALL_CHECK_FUNCTIONS[$idx]}"
    run_stdout="${case_dir}/run.stdout.log"
    run_stderr="${case_dir}/run.stderr.log"
    run_wall_start="${RUN_START_EPOCHS[${test_id}]}"

    wait "${pid}"
    cmd_exit=$?
    elapsed=$(( $(date +%s) - run_wall_start ))

    if [[ ${cmd_exit} -ne 0 ]]; then
        print_status "RUN" "${test_id}" "FAIL after ${elapsed}s (exit code ${cmd_exit})" "${RED}"
        RESULT_NAMES+=("${test_id}")
        RESULT_STATUS+=("FAIL")
        RESULT_DETAILS+=("run failed with exit code ${cmd_exit}")
        RESULT_LOG_PATH+=("${case_dir}")
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
        continue
    fi

    print_status "RUN" "${test_id}" "finished (${elapsed}s)" "${GREEN}"

    # For SLURM runs, invalidate the NFS attribute cache so the head node
    # sees output files freshly created by compute nodes.  Without this,
    # stat may return stale metadata and is_nonempty_and_newer can
    # incorrectly report files as missing or old.
    if [[ "${ALL_RUN_MODES[$idx]}" == "slurm" ]]; then
        sync
        stat "${run_dir_abs}"/* > /dev/null 2>&1 || true
        sleep 1
    fi

    # Run the check function (use SUITE_START_EPOCH for staleness checks to
    # tolerate clock skew between head node and SLURM compute nodes)
    if "${check_fn}" "${run_dir_abs}" "${SUITE_START_EPOCH}" "${run_stdout}" "${run_stderr}"; then
        print_status "CHECK" "${test_id}" "PASS: ${REGRESSION_CHECK_MSG}" "${GREEN}"
        RESULT_NAMES+=("${test_id}")
        RESULT_STATUS+=("PASS")
        RESULT_DETAILS+=("${REGRESSION_CHECK_MSG}")
        RESULT_LOG_PATH+=("${case_dir}")
    else
        print_status "CHECK" "${test_id}" "FAIL: ${REGRESSION_CHECK_MSG}" "${RED}"
        RESULT_NAMES+=("${test_id}")
        RESULT_STATUS+=("FAIL")
        RESULT_DETAILS+=("${REGRESSION_CHECK_MSG}")
        RESULT_LOG_PATH+=("${case_dir}")
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
    fi
done

# Count build failures too
for i in "${!BUILD_OK[@]}"; do
    if [[ "${BUILD_OK[$i]}" == "0" ]]; then
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
    fi
done

# ==========================================================================
#  PHASE 4: SUMMARY
# ==========================================================================
echo
echo "${BOLD}=== SUMMARY ===${NC}"
printf "%-20s  %-6s  %-52s  %s\n" "Test" "Status" "Details" "Logs"
printf "%-20s  %-6s  %-52s  %s\n" "--------------------" "------" "----------------------------------------------------" "----"

for i in "${!RESULT_NAMES[@]}"; do
    status="${RESULT_STATUS[$i]}"
    if [[ "${status}" == "PASS" ]]; then
        color="${GREEN}"
    else
        color="${RED}"
    fi
    printf "${color}%-20s  %-6s${NC}  %-52s  %s\n" \
        "${RESULT_NAMES[$i]}" \
        "${status}" \
        "${RESULT_DETAILS[$i]}" \
        "${RESULT_LOG_PATH[$i]}"
done

echo
if [[ ${TOTAL_FAILURES} -eq 0 ]]; then
    echo "${GREEN}All regression tests passed.${NC}"
    if [[ "${KEEP_ARTIFACTS}" -eq 0 ]]; then
        rm -rf "${RUN_ARTIFACT_DIR}"
        echo "Removed success artifacts (use --keep-artifacts to retain logs)."
    fi
    exit 0
fi

echo "${RED}${TOTAL_FAILURES} test(s) failed.${NC}"
exit 1
