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

# Shorten a test_id to at most 8 characters for SLURM --job-name,
# keeping the distinguishing part visible in default squeue output.
slurm_job_name() {
    local n="$1"
    n="${n/eulerian_diffusion_freefree_multigroup_suite/edf_mg}"
    n="${n/eulerian_diffusion_freefree_suite/edf_gray}"
    n="${n/desmore2012_mc_serial/dsm_ser}"
    n="${n/desmore2012_mc/dsm_mpi}"
    n="${n/spherical_collapse_hires/sphc_hi}"
    n="${n/spherical_collapse/sphc}"
    n="${n/spherical_gauss_tangential/sph_gtan}"
    n="${n/spherical_gauss_linear/sph_glin}"
    n="${n/cartesian_gauss_linear/crt_glin}"
    n="${n/rayleigh_taylor_mpi/rt_mpi}"
    n="${n/marshak_wave_/mw}"
    n="${n/yee_vortex_/yv}"
    printf '%s' "${n:0:8}"
}

if [[ ! -f "${CHECKS_SCRIPT}" ]]; then
    echo "Missing checks script: ${CHECKS_SCRIPT}" >&2
    exit 2
fi
if [[ ! -d "${TESTS_DIR}" ]]; then
    echo "Missing tests directory: ${TESTS_DIR}" >&2
    exit 2
fi
source "${CHECKS_SCRIPT}"

# ==================== Recheck helpers ====================
# Find the newest regression_results/<timestamp>/[<mode>/]<test_id>/ that has run logs.
# Searches both the flat layout (<timestamp>/<test_id>) and the serial_then_mpi
# layout (<timestamp>/{serial,mpi}/<test_id>), returning the most recent match.
find_latest_regression_artifact_for_test() {
    local test_id="$1"
    local artifact_root="$2"
    local ts_dir

    if [[ ! -d "$artifact_root" ]]; then
        return 1
    fi
    while IFS= read -r ts_dir; do
        [[ -n "$ts_dir" ]] || continue
        # Check serial_then_mpi layout: <timestamp>/{serial,mpi}/<test_id>
        for mode_sub in serial mpi; do
            local case_dir="${ts_dir}/${mode_sub}/${test_id}"
            if [[ -d "$case_dir" && -f "$case_dir/run.stdout.log" && -f "$case_dir/run.stderr.log" ]]; then
                printf '%s\n' "$case_dir"
                return 0
            fi
        done
        # Check flat layout: <timestamp>/<test_id>
        local case_dir="${ts_dir}/${test_id}"
        if [[ -d "$case_dir" && -f "$case_dir/run.stdout.log" && -f "$case_dir/run.stderr.log" ]]; then
            printf '%s\n' "$case_dir"
            return 0
        fi
    done < <(find "$artifact_root" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | LC_ALL=C sort -r)
    return 1
}

run_recheck_analysis_for_test() {
    local test_id="$1"
    local artifact_case
    local def_file="${TESTS_DIR}/${test_id}.sh"
    local run_dir_abs
    local run_stdout
    local run_stderr
    local rc_stored

    if ! artifact_case="$(find_latest_regression_artifact_for_test "${test_id}" "${ARTIFACT_ROOT}")"; then
        echo "No regression artifact for test '${test_id}' under ${ARTIFACT_ROOT}." >&2
        echo "Expected a timestamped directory containing ${test_id}/run.stdout.log and run.stderr.log." >&2
        return 1
    fi

    if [[ ! -f "$def_file" ]]; then
        echo "Missing test definition: ${def_file}" >&2
        return 2
    fi

    local TEST_ID=""
    local TAGS=""
    local BUILD_TEST_NAME=""
    local RUN_DIR_REL=""
    local RUN_COMMAND=""
    local CHECK_FUNCTION=""
    local BUILD_ARGS=""
    local RUN_MODE=""
    local SLURM_NTASKS=""
    local SLURM_PARTITION=""
    local SLURM_EXCLUSIVE=""
    local SLURM_NODES=""
    local SLURM_TIME_LIMIT=""

    # shellcheck source=/dev/null
    source "${def_file}"

    if [[ -z "${CHECK_FUNCTION}" || -z "${RUN_DIR_REL}" ]]; then
        echo "Invalid test definition (need CHECK_FUNCTION and RUN_DIR_REL): ${def_file}" >&2
        return 2
    fi

    run_dir_abs="${ROOT_DIR}/${RUN_DIR_REL}"
    if [[ ! -d "$run_dir_abs" ]]; then
        echo "Run directory does not exist: ${RUN_DIR_REL}" >&2
        return 2
    fi

    run_stdout="${artifact_case}/run.stdout.log"
    run_stderr="${artifact_case}/run.stderr.log"

    if [[ -f "${artifact_case}/run_exit_code.txt" ]]; then
        rc_stored="$(< "${artifact_case}/run_exit_code.txt")"
        if [[ "$rc_stored" != "0" ]]; then
            echo "${ORANGE}Warning: stored run exit code was ${rc_stored} (re-running analysis anyway).${NC}" >&2
        fi
    fi

    echo "${BOLD}Recheck analysis for '${test_id}'${NC}"
    echo "  Artifact:  ${artifact_case}"
    echo "  Case data: ${run_dir_abs}"
    echo

    # Second argument is suite start epoch for staleness checks; use 0 to accept existing outputs.
    if "${CHECK_FUNCTION}" "${run_dir_abs}" "0" "${run_stdout}" "${run_stderr}"; then
        echo "${GREEN}CHECK PASS: ${REGRESSION_CHECK_MSG}${NC}"
        return 0
    fi
    echo "${RED}CHECK FAIL: ${REGRESSION_CHECK_MSG}${NC}"
    return 1
}

# ==================== Defaults ====================
CONFIG=""
CONFIG_EXPLICIT=0
MPI_NP=4
KEEP_ARTIFACTS=0
VERBOSE=0
TEST_FILTER=""
CLEAN_RESULTS=0
RECHECK_MODE=0
MODE="all"
NPROC_OVERRIDE=""
SLURM_PARTITION_OVERRIDE=""
RUN_LOCAL=0
SEQUENTIAL=0
NO_EXCLUSIVE=0

ARTIFACT_ROOT="${ROOT_DIR}/regression_results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ARTIFACT_DIR="${ARTIFACT_ROOT}/${TIMESTAMP}"
ARTIFACT_DIR_OVERRIDE=""

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
  --partition <name>       Override SLURM partition for all MPI tests (default per-test, usually bigrun)
  --local                  Run MPI tests locally via mpirun instead of submitting through SLURM
  --sequential              Run tests one at a time instead of in parallel
  --no-exclusive            Do not pass --exclusive to SLURM (allow node sharing)
  --clean-results          Delete regression_results, generated figures, and run artifacts from cases, then exit
  --nproc <N>              Override detected core count (default: $(nproc))
  --keep-artifacts         Keep all logs even if all tests pass
  --verbose                Stream run output to terminal as well
  --recheck                Re-run the CHECK step only for --test, using the newest
                           regression_results/<timestamp>/<test>/ logs (no build/run)
  -h, --help               Show this help

Modes:
  serial           Run tests tagged "serial" (default config: gnuRelease)
  mpi              Run tests tagged "mpi"    (default config: gnuReleaseMPI)
  all              Run all tests             (default config: gnuReleaseMPI)
  serial_then_mpi  Run serial (gnuRelease) and MPI (gnuReleaseMPI) tests in parallel

Examples:
  ./regression_tests/run_all.sh --mode serial
  ./regression_tests/run_all.sh --mode mpi --config intelReleaseMPI
  ./regression_tests/run_all.sh --mode mpi --partition short
  ./regression_tests/run_all.sh --mode mpi --local
  ./regression_tests/run_all.sh --mode serial_then_mpi
  ./regression_tests/run_all.sh --test sod_1d --config gnuRelease
  ./regression_tests/run_all.sh --recheck --test moving_slab_mc
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
        --partition)
            SLURM_PARTITION_OVERRIDE="${2:-}"
            shift 2
            ;;
        --local)
            RUN_LOCAL=1
            shift
            ;;
        --sequential)
            SEQUENTIAL=1
            shift
            ;;
        --no-exclusive)
            NO_EXCLUSIVE=1
            shift
            ;;
        --nproc)
            NPROC_OVERRIDE="${2:-}"
            shift 2
            ;;
        --keep-artifacts)
            KEEP_ARTIFACTS=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --recheck)
            RECHECK_MODE=1
            shift
            ;;
        --_artifact-dir)
            ARTIFACT_DIR_OVERRIDE="${2:-}"
            shift 2
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

# Apply artifact directory override (used by serial_then_mpi to isolate passes)
if [[ -n "${ARTIFACT_DIR_OVERRIDE}" ]]; then
    RUN_ARTIFACT_DIR="${ARTIFACT_DIR_OVERRIDE}"
fi

# ==================== Recheck-only (no build/run) ====================
if [[ "${RECHECK_MODE}" -eq 1 ]]; then
    if [[ "${CLEAN_RESULTS}" -eq 1 ]]; then
        echo "--recheck cannot be combined with --clean-results" >&2
        exit 2
    fi
    if [[ -z "${TEST_FILTER}" ]]; then
        echo "--recheck requires --test <id>" >&2
        exit 2
    fi
    if ! echo "${TEST_FILTER}" | grep -qE "^(${VALID_TEST_IDS})$"; then
        echo "--test must be one of: ${VALID_TEST_IDS//|/, }" >&2
        exit 2
    fi
    if [[ "${RUN_LOCAL}" -eq 1 || "${CONFIG_EXPLICIT}" -eq 1 || -n "${SLURM_PARTITION_OVERRIDE}" || -n "${NPROC_OVERRIDE}" ]]; then
        echo "${ORANGE}Warning: --recheck ignores build/run options (--config, --local, --partition, --nproc, --mpi-np, --mode).${NC}" >&2
    fi
    run_recheck_analysis_for_test "${TEST_FILTER}"
    exit $?
fi

# ==================== Validate arguments ====================
case "${MODE}" in
    serial|mpi|all|serial_then_mpi) ;;
    *)
        echo "--mode must be one of: serial, mpi, all, serial_then_mpi" >&2
        exit 2
        ;;
esac

# serial_then_mpi: launch both passes in parallel so a long-running serial
# test does not delay the MPI pass.  Each pass writes to its own artifact
# subdirectory (serial/ and mpi/) to prevent clobbering.
if [[ "${MODE}" == "serial_then_mpi" ]]; then
    echo "${BOLD}=== serial_then_mpi: launching serial and MPI passes in parallel ===${NC}"
    passthrough_args=()
    [[ "${KEEP_ARTIFACTS}" -eq 1 ]]            && passthrough_args+=(--keep-artifacts)
    [[ "${VERBOSE}" -eq 1 ]]                   && passthrough_args+=(--verbose)
    [[ -n "${TEST_FILTER}" ]]                  && passthrough_args+=(--test "${TEST_FILTER}")
    [[ -n "${NPROC_OVERRIDE}" ]]               && passthrough_args+=(--nproc "${NPROC_OVERRIDE}")
    [[ -n "${SLURM_PARTITION_OVERRIDE}" ]]     && passthrough_args+=(--partition "${SLURM_PARTITION_OVERRIDE}")
    [[ "${RUN_LOCAL}" -eq 1 ]]                 && passthrough_args+=(--local)
    [[ "${SEQUENTIAL}" -eq 1 ]]                && passthrough_args+=(--sequential)
    [[ "${NO_EXCLUSIVE}" -eq 1 ]]              && passthrough_args+=(--no-exclusive)

    mpi_config="${CONFIG}"
    if [[ "${CONFIG_EXPLICIT}" -eq 0 ]]; then
        serial_config="gnuRelease"
        mpi_config="gnuReleaseMPI"
    elif [[ "${CONFIG}" == *MPI* ]]; then
        serial_config="${CONFIG//MPI/}"
        echo "  Derived serial config: ${serial_config} (from ${CONFIG})"
    else
        serial_config="${CONFIG}"
    fi

    serial_artifact_dir="${RUN_ARTIFACT_DIR}/serial"
    mpi_artifact_dir="${RUN_ARTIFACT_DIR}/mpi"
    mkdir -p "${serial_artifact_dir}" "${mpi_artifact_dir}"

    echo "  Artifacts: ${RUN_ARTIFACT_DIR}"
    echo "    serial → ${serial_artifact_dir}"
    echo "    mpi    → ${mpi_artifact_dir}"

    # Children must always keep artifacts; the parent handles final cleanup.
    echo "${BOLD}--- serial pass (background) ---${NC}"
    "${BASH_SOURCE[0]}" --mode serial --config "${serial_config}" \
        --mpi-np "${MPI_NP}" --_artifact-dir "${serial_artifact_dir}" \
        --keep-artifacts "${passthrough_args[@]}" &
    serial_pid=$!

    echo "${BOLD}--- MPI pass (background) ---${NC}"
    "${BASH_SOURCE[0]}" --mode mpi --config "${mpi_config}" \
        --mpi-np "${MPI_NP}" --_artifact-dir "${mpi_artifact_dir}" \
        --keep-artifacts "${passthrough_args[@]}" &
    mpi_pid=$!

    serial_rc=0
    wait "${serial_pid}" || serial_rc=$?
    mpi_rc=0
    wait "${mpi_pid}" || mpi_rc=$?

    echo
    if [[ ${serial_rc} -eq 0 && ${mpi_rc} -eq 0 ]]; then
        echo "${GREEN}serial_then_mpi: all passes succeeded.${NC}"
        if [[ "${KEEP_ARTIFACTS}" -eq 0 ]]; then
            rm -rf "${RUN_ARTIFACT_DIR}"
            echo "Removed success artifacts (use --keep-artifacts to retain logs)."
        fi
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

if [[ -n "${SLURM_PARTITION_OVERRIDE}" && "${RUN_LOCAL}" -eq 1 ]]; then
    echo "${ORANGE}Warning: --partition is ignored when --local is set${NC}" >&2
fi

if [[ -n "${NPROC_OVERRIDE}" ]] && ! [[ "${NPROC_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--nproc must be a positive integer" >&2
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
    # Also clean run-generated data and figures from regression_tests/.
    local_regression_dir="${ROOT_DIR}/regression_tests"
    local_cases_dir="${local_regression_dir}/cases"
    if [[ -d "${local_cases_dir}" ]]; then
        cleaned=0
        while IFS= read -r -d '' f; do
            rm -f "$f"
            cleaned=$((cleaned + 1))
        done < <(find "${local_cases_dir}" -maxdepth 2 \
            \( -name "*.log" -o -name "*.txt" -o -name "*.h5" -o -name "*.vtu" -o -name "rich" \) \
            ! -name "test.cpp" -print0)
        if [[ ${cleaned} -gt 0 ]]; then
            echo "Cleaned ${cleaned} run-generated files from ${local_cases_dir}"
        fi

        # Remove run-generated subdirectories inside case dirs:
        #   snap_*  - HDF5/VTU snapshot directories
        #   build/  - stale in-case build artifacts
        #   regression_tests/ - accidental nested run artifacts
        #   rt_final/ and similar VTU output dirs from WriteSnapshot3D
        cleaned_dirs=0
        while IFS= read -r -d '' d; do
            rm -rf "$d"
            cleaned_dirs=$((cleaned_dirs + 1))
        done < <(find "${local_cases_dir}" -mindepth 2 -maxdepth 2 -type d \
            \( -name "snap_*" -o -name "build" -o -name "regression_tests" \
               -o -name "regression_results" -o -name "rt_final" \
               -o -name "__pycache__" \) -print0)
        if [[ ${cleaned_dirs} -gt 0 ]]; then
            echo "Cleaned ${cleaned_dirs} run-generated directories from ${local_cases_dir}"
        fi
    fi

    if [[ -d "${local_regression_dir}" ]]; then
        cleaned_figures=0
        while IFS= read -r -d '' f; do
            rm -f "$f"
            cleaned_figures=$((cleaned_figures + 1))
        done < <(find "${local_regression_dir}" -type f \
            \( -name "*.png" -o -name "*.pdf" -o -name "*.jpg" -o -name "*.jpeg" -o -name "*.svg" \) -print0)
        if [[ ${cleaned_figures} -gt 0 ]]; then
            echo "Cleaned ${cleaned_figures} figure file(s) from ${local_regression_dir}"
        fi

        # Remove top-level generated directories
        for d in "${local_regression_dir}/plots" \
                 "${local_regression_dir}/__pycache__"; do
            if [[ -d "$d" ]]; then
                rm -rf "$d"
                echo "Removed ${d}"
            fi
        done
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
declare -a ALL_SLURM_NODES=()
declare -a ALL_SLURM_NTASKS_PER_NODE=()
declare -a ALL_SLURM_TIME_LIMITS=()
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
    local SLURM_NODES=""
    local SLURM_NTASKS_PER_NODE=""
    local SLURM_TIME_LIMIT="02:00:00"

    source "${def_file}"

    # Validate required fields. Static checks do not need a build target or run command.
    if [[ "${RUN_MODE}" == "static" ]]; then
        if [[ -z "${TEST_ID}" || -z "${RUN_DIR_REL}" || -z "${CHECK_FUNCTION}" ]]; then
            echo "Invalid static test definition: ${def_file}" >&2
            return 1
        fi
    elif [[ -z "${TEST_ID}" || -z "${BUILD_TEST_NAME}" || -z "${RUN_DIR_REL}" || -z "${RUN_COMMAND}" || -z "${CHECK_FUNCTION}" ]]; then
        echo "Invalid test definition: ${def_file}" >&2
        return 1
    fi

    # Filter by --test
    if [[ -n "${TEST_FILTER}" && "${TEST_ID}" != "${TEST_FILTER}" ]]; then
        return 2  # skipped, not an error
    fi

    # Skip manual-only tests unless explicitly selected with --test
    if [[ " ${TAGS} " == *" manual "* && "${TEST_ID}" != "${TEST_FILTER}" ]]; then
        return 2
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
    ALL_SLURM_NODES+=("${SLURM_NODES}")
    ALL_SLURM_NTASKS_PER_NODE+=("${SLURM_NTASKS_PER_NODE}")
    ALL_SLURM_TIME_LIMITS+=("${SLURM_TIME_LIMIT}")
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

# ==================== Apply --local and --partition overrides ====================
if [[ "${RUN_LOCAL}" -eq 1 ]]; then
    for i in "${!ALL_RUN_MODES[@]}"; do
        if [[ "${ALL_RUN_MODES[$i]}" == "slurm" ]]; then
            ALL_RUN_MODES[$i]="direct"
        fi
    done
fi
if [[ -n "${SLURM_PARTITION_OVERRIDE}" ]]; then
    for i in "${!ALL_SLURM_PARTITIONS[@]}"; do
        ALL_SLURM_PARTITIONS[$i]="${SLURM_PARTITION_OVERRIDE}"
    done
fi
if [[ "${NO_EXCLUSIVE}" -eq 1 ]]; then
    for i in "${!ALL_SLURM_EXCLUSIVES[@]}"; do
        ALL_SLURM_EXCLUSIVES[$i]="0"
    done
fi

# Do not alter modules inside SLURM jobs. The submit environment and the
# explicit variables below define the run environment.
SLURM_MODULE_SETUP="true"

# ==================== Print header ====================
mkdir -p "${RUN_ARTIFACT_DIR}"

echo "${BOLD}Running regression suite${NC}"
echo "  Mode:      ${MODE}"
echo "  Config:    ${CONFIG}"
echo "  Tests:     ${ALL_TEST_IDS[*]}"
if [[ "${MODE}" != "serial" ]]; then
    echo "  MPI ranks: ${MPI_NP}"
    if [[ "${RUN_LOCAL}" -eq 1 ]]; then
        echo "  Execution: local (mpirun, no SLURM)"
    else
        echo "  Execution: SLURM"
        if [[ -n "${SLURM_PARTITION_OVERRIDE}" ]]; then
            echo "  Partition: ${SLURM_PARTITION_OVERRIDE} (override)"
        fi
    fi
fi
echo "  Cores:     ${NPROC_OVERRIDE:-$(nproc)} (override with --nproc)"
echo "  SLURM env: ${SLURM_MODULE_SETUP}"
echo "  Artifacts: ${RUN_ARTIFACT_DIR}"
echo

# Record epoch before any work starts.  Used later by check functions
# to verify output files are fresh (is_nonempty_and_newer).  We use this early
# timestamp rather than a per-test one so that SLURM clock skew between the
# head node and compute nodes does not cause false "stale" failures.
SUITE_START_EPOCH="$(date +%s)"

# ==================== Parallel build configuration ====================
MAX_PARALLEL_BUILDS=4
MAX_TOTAL_MAKE_JOBS=32
TOTAL_CORES="${NPROC_OVERRIDE:-$(nproc)}"

# Allocate build slots only for tests that are actually selected. With
# --test, one build receives the full job budget instead of one quarter of it.
ACTIVE_PARALLEL_BUILDS=${MAX_PARALLEL_BUILDS}
(( NUM_TESTS < ACTIVE_PARALLEL_BUILDS )) && ACTIVE_PARALLEL_BUILDS=${NUM_TESTS}
(( SEQUENTIAL == 1 )) && ACTIVE_PARALLEL_BUILDS=1

JOBS_PER_BUILD=$(( TOTAL_CORES / ACTIVE_PARALLEL_BUILDS ))
(( JOBS_PER_BUILD < 1 )) && JOBS_PER_BUILD=1
MAX_JOBS_PER_BUILD=$(( MAX_TOTAL_MAKE_JOBS / ACTIVE_PARALLEL_BUILDS ))
(( MAX_JOBS_PER_BUILD < 1 )) && MAX_JOBS_PER_BUILD=1
(( JOBS_PER_BUILD > MAX_JOBS_PER_BUILD )) && JOBS_PER_BUILD=$MAX_JOBS_PER_BUILD

# FIFO-based semaphore to cap concurrent builds
BUILD_FIFO="$(mktemp -u)"
mkfifo "${BUILD_FIFO}"
exec 7<>"${BUILD_FIFO}"
rm "${BUILD_FIFO}"
for (( s=0; s<ACTIVE_PARALLEL_BUILDS; s++ )); do
    echo >&7
done

# ==========================================================================
#  PHASE 1: BUILD & RUN (pipelined, max ${ACTIVE_PARALLEL_BUILDS} concurrent builds)
# ==========================================================================
echo "${BOLD}=== BUILD & RUN PHASE (max ${ACTIVE_PARALLEL_BUILDS} concurrent builds, ${JOBS_PER_BUILD} make-jobs each, ${MAX_TOTAL_MAKE_JOBS} total cap) ===${NC}"

declare -A JOB_PIDS=()    # test_id -> PID
declare -A JOB_INDICES=()  # test_id -> index into ALL_* arrays

for i in "${!ALL_TEST_IDS[@]}"; do
    test_id="${ALL_TEST_IDS[$i]}"
    build_test_name="${ALL_BUILD_TEST_NAMES[$i]}"
    build_args="${ALL_BUILD_ARGS[$i]}"
    run_dir_rel="${ALL_RUN_DIR_RELS[$i]}"
    run_cmd="${ALL_RUN_COMMANDS[$i]}"
    run_mode="${ALL_RUN_MODES[$i]}"
    slurm_ntasks="${ALL_SLURM_NTASKS[$i]}"
    slurm_partition="${ALL_SLURM_PARTITIONS[$i]}"
    slurm_exclusive="${ALL_SLURM_EXCLUSIVES[$i]}"
    slurm_nodes="${ALL_SLURM_NODES[$i]}"
    slurm_ntasks_per_node="${ALL_SLURM_NTASKS_PER_NODE[$i]}"
    slurm_time_limit="${ALL_SLURM_TIME_LIMITS[$i]}"
    case_dir="${ALL_CASE_DIRS[$i]}"

    mkdir -p "${case_dir}"
    JOB_INDICES["${test_id}"]="${i}"

    # Launch a background subshell that builds then immediately runs the test
    (
        if [[ "${run_mode}" == "static" ]]; then
            echo "0" > "${case_dir}/build_status.txt"
            echo "static check; build skipped" > "${case_dir}/build_detail.txt"
            : > "${case_dir}/build.stdout.log"
            : > "${case_dir}/build.stderr.log"
            date +%s > "${case_dir}/run_start_epoch.txt"
            print_status "RUN" "${test_id}" "static check" "${CYAN}"
            echo "0" > "${case_dir}/run_exit_code.txt"
            date +%s > "${case_dir}/run_end_epoch.txt"
            : > "${case_dir}/run.stdout.log"
            : > "${case_dir}/run.stderr.log"
            exit 0
        fi

        # ---- BUILD ----
        read -u 7  # acquire build slot (blocks until a slot is free)

        local_build_stdout="${case_dir}/build.stdout.log"
        local_build_stderr="${case_dir}/build.stderr.log"

        extra_info=""
        if [[ -n "${build_args}" ]]; then
            extra_info=" (${build_args})"
        fi
        print_status "BUILD" "${test_id}" "compiling${extra_info}..." "${CYAN}"

        build_cmd=("${ROOT_DIR}/build_rich.sh" "${CONFIG}" "--test_name=${build_test_name}" "--build-subdir=${test_id}" "--jobs=${JOBS_PER_BUILD}")
        if [[ -n "${build_args}" ]]; then
            read -r -a extra_build_args <<< "${build_args}"
            build_cmd+=("${extra_build_args[@]}")
        fi

        if ! VERBOSE=1 "${build_cmd[@]}" >"${local_build_stdout}" 2>"${local_build_stderr}"; then
            print_status "BUILD" "${test_id}" "FAIL (build failed, see build.stderr.log)" "${RED}"
            echo "1" > "${case_dir}/build_status.txt"
            echo "build failed (see build.stderr.log)" > "${case_dir}/build_detail.txt"
            echo >&7  # release build slot
            exit 1
        fi

        build_bin="${ROOT_DIR}/build/${CONFIG}/${test_id}/rich"
        if [[ ! -x "${build_bin}" && ! -L "${build_bin}" ]]; then
            print_status "BUILD" "${test_id}" "FAIL (binary not found)" "${RED}"
            echo "1" > "${case_dir}/build_status.txt"
            echo "binary not found at build/${CONFIG}/${test_id}/rich" > "${case_dir}/build_detail.txt"
            echo >&7  # release build slot
            exit 1
        fi

        cp -L "${build_bin}" "${case_dir}/rich"
        chmod +x "${case_dir}/rich"
        echo "0" > "${case_dir}/build_status.txt"
        print_status "BUILD" "${test_id}" "OK" "${GREEN}"

        echo >&7  # release build slot

        # ---- RUN ----
        run_dir_abs="${ROOT_DIR}/${run_dir_rel}"
        run_stdout="${case_dir}/run.stdout.log"
        run_stderr="${case_dir}/run.stderr.log"
        rich_bin="${case_dir}/rich"

        if [[ ! -d "${run_dir_abs}" ]]; then
            echo "run directory does not exist: ${run_dir_rel}" > "${case_dir}/run_detail.txt"
            exit 1
        fi

        date +%s > "${case_dir}/run_start_epoch.txt"
        print_status "RUN" "${test_id}" "started" "${CYAN}"

        cd "${run_dir_abs}" || exit 1
        export ROOT_DIR CONFIG MPI_NP SLURM_NTASKS="${slurm_ntasks}"
        if [[ -n "${slurm_ntasks_per_node}" ]]; then
            export SLURM_NTASKS_PER_NODE="${slurm_ntasks_per_node}"
        fi
        export RICH_BIN="${rich_bin}"
        export RUN_LOCAL="${RUN_LOCAL}"
        if [[ -n "${SLURM_PARTITION_OVERRIDE}" ]]; then
            export SLURM_PARTITION="${slurm_partition}"
        fi

        run_rc=0
        if [[ "${run_mode}" == "slurm" ]]; then
            local_escaped_run_cmd="${run_cmd//\'/\'\\\'\'}"
            sbatch_wrap_cmd="${SLURM_MODULE_SETUP} && ROOT_DIR=\"${ROOT_DIR}\" CONFIG=\"${CONFIG}\" MPI_NP=\"${MPI_NP}\" SLURM_NTASKS=\"${slurm_ntasks}\" RICH_BIN=\"${rich_bin}\" bash -c '${local_escaped_run_cmd}'"
            sbatch_args=(
                sbatch
                --export=ALL
                --wait
                --time="${slurm_time_limit}"
                --job-name="$(slurm_job_name "${test_id}")"
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
            if [[ -n "${slurm_nodes}" ]]; then
                sbatch_args+=(--nodes="${slurm_nodes}")
            fi
            if [[ -n "${slurm_ntasks_per_node}" ]]; then
                sbatch_args+=(--ntasks-per-node="${slurm_ntasks_per_node}")
            fi
            "${sbatch_args[@]}" || run_rc=$?
        else
            # Convert HH:MM:SS to seconds for timeout
            IFS=: read -r _h _m _s <<< "${slurm_time_limit}"
            timeout_secs=$(( 10#${_h} * 3600 + 10#${_m} * 60 + 10#${_s} ))
            if [[ "${VERBOSE}" -eq 1 ]]; then
                timeout "${timeout_secs}" bash -c "${run_cmd}" > >(tee "${run_stdout}") 2> >(tee "${run_stderr}" >&2) || run_rc=$?
            else
                timeout "${timeout_secs}" bash -c "${run_cmd}" >"${run_stdout}" 2>"${run_stderr}" || run_rc=$?
            fi
        fi

        echo "${run_rc}" > "${case_dir}/run_exit_code.txt"
        date +%s > "${case_dir}/run_end_epoch.txt"
        exit "${run_rc}"
    ) &
    JOB_PIDS["${test_id}"]=$!

    if [[ "${SEQUENTIAL}" -eq 1 ]]; then
        wait "${JOB_PIDS[${test_id}]}"
    fi
done

echo

# ==========================================================================
#  PHASE 2: WAIT & CHECK
# ==========================================================================
echo "${BOLD}=== RESULTS ===${NC}"

TOTAL_FAILURES=0

for test_id in "${!JOB_PIDS[@]}"; do
    pid="${JOB_PIDS[${test_id}]}"
    idx="${JOB_INDICES[${test_id}]}"
    case_dir="${ALL_CASE_DIRS[$idx]}"
    run_dir_rel="${ALL_RUN_DIR_RELS[$idx]}"
    run_dir_abs="${ROOT_DIR}/${run_dir_rel}"
    check_fn="${ALL_CHECK_FUNCTIONS[$idx]}"
    run_stdout="${case_dir}/run.stdout.log"
    run_stderr="${case_dir}/run.stderr.log"

    wait "${pid}"

    # --- Check build status ---
    build_status_file="${case_dir}/build_status.txt"
    if [[ ! -f "${build_status_file}" ]]; then
        print_status "BUILD" "${test_id}" "FAIL (build status unknown)" "${RED}"
        RESULT_NAMES+=("${test_id}")
        RESULT_STATUS+=("FAIL")
        RESULT_DETAILS+=("build status unknown (no marker file)")
        RESULT_LOG_PATH+=("${case_dir}")
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
        continue
    fi
    if [[ "$(< "${build_status_file}")" != "0" ]]; then
        build_detail="build failed"
        if [[ -f "${case_dir}/build_detail.txt" ]]; then
            build_detail="$(< "${case_dir}/build_detail.txt")"
        fi
        RESULT_NAMES+=("${test_id}")
        RESULT_STATUS+=("FAIL")
        RESULT_DETAILS+=("${build_detail}")
        RESULT_LOG_PATH+=("${case_dir}")
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
        continue
    fi

    # --- Check for pre-run failures (e.g. missing run directory) ---
    if [[ -f "${case_dir}/run_detail.txt" ]]; then
        run_detail="$(< "${case_dir}/run_detail.txt")"
        print_status "RUN" "${test_id}" "FAIL (${run_detail})" "${RED}"
        RESULT_NAMES+=("${test_id}")
        RESULT_STATUS+=("FAIL")
        RESULT_DETAILS+=("${run_detail}")
        RESULT_LOG_PATH+=("${case_dir}")
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
        continue
    fi

    # --- Check run exit code ---
    run_start_file="${case_dir}/run_start_epoch.txt"
    run_end_file="${case_dir}/run_end_epoch.txt"
    run_exit_file="${case_dir}/run_exit_code.txt"

    if [[ -f "${run_start_file}" ]]; then
        run_wall_start="$(< "${run_start_file}")"
    else
        run_wall_start="$(date +%s)"
    fi
    if [[ -f "${run_end_file}" ]]; then
        run_wall_end="$(< "${run_end_file}")"
    else
        run_wall_end="$(date +%s)"
    fi
    elapsed=$(( run_wall_end - run_wall_start ))

    if [[ -f "${run_exit_file}" ]]; then
        run_exit="$(< "${run_exit_file}")"
    else
        run_exit="1"
    fi

    if [[ "${run_exit}" != "0" ]]; then
        print_status "RUN" "${test_id}" "FAIL after ${elapsed}s (exit code ${run_exit})" "${RED}"
        RESULT_NAMES+=("${test_id}")
        RESULT_STATUS+=("FAIL")
        RESULT_DETAILS+=("run failed with exit code ${run_exit}")
        RESULT_LOG_PATH+=("${case_dir}")
        TOTAL_FAILURES=$((TOTAL_FAILURES + 1))
        continue
    fi

    print_status "RUN" "${test_id}" "finished (${elapsed}s)" "${GREEN}"

    # For SLURM runs, invalidate the NFS attribute cache so the head node
    # sees output files freshly created by compute nodes.
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

# Close the FIFO semaphore file descriptor
exec 7>&-

# ==========================================================================
#  PHASE 3: SUMMARY
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
