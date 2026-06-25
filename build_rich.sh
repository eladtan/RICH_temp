#!/bin/bash

# ==================== Auto-completion ====================
# When sourced with --completions, registers tab-completion and exits.
# Completions are derived from the case-statement below, so adding a
# new flag there automatically makes it completable.
if [[ "$1" == "--completions" ]]; then
    _build_rich_completions() {
        local cword=$COMP_CWORD
        # Reconstruct the full current word from the command line,
        # because bash splits on '=' (in COMP_WORDBREAKS).
        local before_cursor="${COMP_LINE:0:$COMP_POINT}"
        local cur_full="${before_cursor##* }"
        local cur="${COMP_WORDS[COMP_CWORD]}"

        local configs="gnuReleaseMPI gnuDebugMPI intelReleaseMPI intelDebugMPI gnuRelease gnuDebug intelRelease intelDebug gnuMixedMPI intelMixedMPI"

        # Extract flags directly from this script's case-statement
        local script="${COMP_WORDS[0]}"
        local flags
        local flags="--test_name="
        flags="$flags $(grep -oP '^\s+--[a-zA-Z_-]+(=\*?)?\)' "$script" 2>/dev/null \
            | sed 's/[)*]//g; s/^[[:space:]]*//' | sort -u)"

        if [[ $cword -eq 1 ]]; then
            COMPREPLY=($(compgen -W "$configs" -- "$cur"))
            return
        fi

        local script_dir
        script_dir="$(cd "$(dirname "$script")" && pwd)"

        if [[ "$cur_full" == --test_name=* ]]; then
            local prefix="${cur_full#--test_name=}"
            local test_dirs=""
            if [[ -d "$script_dir/runs" ]]; then
                test_dirs=$(find "$script_dir/runs" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null)
            fi
            COMPREPLY=($(compgen -W "$test_dirs" -- "$prefix"))
            return
        fi

        if [[ "$cur_full" == --debug_files=* ]]; then
            local prefix="${cur_full#--debug_files=}"
            COMPREPLY=($(compgen -f -- "$prefix"))
            return
        fi

        COMPREPLY=($(compgen -W "$flags" -- "$cur"))
        [[ ${#COMPREPLY[@]} -eq 1 && "${COMPREPLY[0]}" == *= ]] && compopt -o nospace
    }
    complete -F _build_rich_completions ./build_rich.sh
    complete -F _build_rich_completions build_rich.sh
    return 0 2>/dev/null || exit 0
fi

# ==================== Colors ====================
RED=$'\033[0;31m'
FAIL=$'\e[7;31;47m'
SUCCESS=$'\e[7;32;47m'
GREEN=$'\033[0;32m'
ORANGE=$'\033[0;33m'
PURPLE=$'\033[0;35m'
CYAN=$'\033[0;36m'
BLUE=$'\033[0;34m'
NC=$'\033[0m'

# probably causes no print races in make
export CCACHE_CPP2=yes

# ==================== Parse Arguments ====================
CONFIG="$1"
TEST_NAME=""
CMAKE_FLAGS=""
MIXED_DEBUG_FILES=""
BUILD_SUBDIR=""
MAKE_JOBS="$(nproc)"

for arg in "${@:2}"; do
    case "$arg" in
        --test_name=*)
            TEST_NAME="${arg#--test_name=}"
            ;;
        --with_asan)
            CMAKE_FLAGS+=" -DASAN=1 "
            ;;
        --energy_groups_num=*)
            val="${arg#--energy_groups_num=}"
            CMAKE_FLAGS+=" -DENERGY_GROUPS_NUM=$val "
            ;;
        --debug_files=*)
            val="${arg#--debug_files=}"
            MIXED_DEBUG_FILES=$(realpath "$val")
            CMAKE_FLAGS+=" -DDEBUG_FILES=$val "
            ;;
        --mc_debug)
            CMAKE_FLAGS+=" -DMC_DEBUG=1 "
            ;;
        --mc_trace_debug=*)
            val="${arg#--mc_trace_debug=}"
            CMAKE_FLAGS+=" -DMC_TRACE_DEBUG=$val "
            ;;
        --shared)
            CMAKE_FLAGS+=" -DDYNAMIC_LIBS=1 "
            ;;
        --high-res)
            CMAKE_FLAGS+=" -DHIGH_RES=1 "
            ;;
        --memory_debug)
            CMAKE_FLAGS+=" -DMEMORY_DEBUG=1 "
            ;;
        --memory_profile)
            CMAKE_FLAGS+=" -DMEMORY_PROFILE=1 "
            ;;
        --assert)
            CMAKE_FLAGS+=" -DFORCE_ASSERT=1 "
            ;;
        --timing)
            CMAKE_FLAGS+=" -DTIMING=1 "
            ;;
        --montecarlo-polarization)
            CMAKE_FLAGS+=" -DRICH_MONTECARLO_POLARIZATION=ON "
            ;;
        --build-subdir=*)
            BUILD_SUBDIR="${arg#--build-subdir=}"
            ;;
        --jobs=*)
            MAKE_JOBS="${arg#--jobs=}"
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            exit 1
            ;;
    esac
done

# Comparable command for change detection (excludes --build-subdir and --jobs
# since they don't affect build output and shouldn't trigger a full rebuild)
COMPARABLE_CMD="$0"
for arg in "$@"; do
    case "$arg" in
        --build-subdir=*|--jobs=*) ;;
        *) COMPARABLE_CMD+=" $arg" ;;
    esac
done

# ==================== Paths ====================
ORIG_DIR="$(pwd)"
if [[ -n "${BUILD_SUBDIR}" ]]; then
    BUILD_DIR="$ORIG_DIR/build/$CONFIG/$BUILD_SUBDIR"
else
    BUILD_DIR="$ORIG_DIR/build/$CONFIG"
fi
CMD_FILE="$BUILD_DIR/.build_cmd"
DEBUG_FILES_FILE="$BUILD_DIR/.debug_files"
SOURCE_FILES_FILE="$BUILD_DIR/.source_files"
CMAKE_MTIMES_FILE="$BUILD_DIR/.cmake_mtimes"
ENV_PATHS_FILE="$BUILD_DIR/.env_paths"

MAKE_OUT="$BUILD_DIR/${CONFIG}_build.out"
MAKE_ERR="$BUILD_DIR/${CONFIG}_build.err"
CMAKE_OUT="$BUILD_DIR/${CONFIG}_cmake.out"
CMAKE_ERR="$BUILD_DIR/${CONFIG}_cmake.err"

# ==================== Validate arguments ====================

if [[ $# -lt 2 || -z "$TEST_NAME" ]]; then
    echo -e "${RED}Usage: $0 <config> --test_name=<name> [--with_asan] [--energy_groups_num=<N>] [--mc_debug] [--mc_trace_debug=<N>] [--shared] [--high-res] [--memory_debug] [--memory_profile] [--assert] [--timing] [--montecarlo-polarization] [--build-subdir=<name>] [--jobs=<N>]${NC}"
    exit 1
fi


# ==================== Reset Build if Test Name Changed ====================

# Reset build directory if build-affecting arguments changed
if [[ -f "$CMD_FILE" ]]; then
    OLD_CMD=$(<"$CMD_FILE")
    NORMALIZED_CMD="$(echo "$COMPARABLE_CMD" | tr -s '[:space:]' ' ' | sed 's/ *$//')"
    OLD_CMD="$(echo "$OLD_CMD" | tr -s '[:space:]' ' ' | sed 's/ *$//')"
    if [[ "$OLD_CMD" != "$NORMALIZED_CMD" ]]; then
        echo -e "${PURPLE}Build command changed. Cleaning $BUILD_DIR...${NC}"
        rm -rf "$BUILD_DIR"
    fi
fi

# Always ensure build directory exists
mkdir -p "$BUILD_DIR" || { echo -e "${RED}Failed to create $BUILD_DIR${NC}"; exit 1; }

# Change into build directory
cd "$BUILD_DIR" || { echo -e "${RED}Failed to cd into $BUILD_DIR${NC}"; exit 1; }

# Save comparable command for future change detection
echo "$COMPARABLE_CMD" > "$CMD_FILE"

# Run CMake if Makefile doesn't exist, or if "MIXED_DEBUG_FILES" is not empty, and '$BUILD_DIR/debug.txt', '$MIXED_DEBUG_FILES' are different
if [[ $MIXED_DEBUG_FILES && -f "$DEBUG_FILES_FILE" ]]; then
    if ! cmp -s "$MIXED_DEBUG_FILES" "$DEBUG_FILES_FILE";
    then
        echo -e "${PURPLE}Debug files list changed. Cleaning $BUILD_DIR...${NC}"
        rm -rf "$BUILD_DIR"
        mkdir -p "$BUILD_DIR" || { echo -e "${RED}Failed to create $BUILD_DIR${NC}"; exit 1; }
        cd "$BUILD_DIR" || { echo -e "${RED}Failed to cd into $BUILD_DIR${NC}"; exit 1; }
    fi
fi

if [[ $MIXED_DEBUG_FILES ]]; then
    cp "$MIXED_DEBUG_FILES" "$DEBUG_FILES_FILE"
fi

# ==================== Track Source Files (detect add/remove) ====================
RERUN_CMAKE=0

TEST_SOURCE_DIR="$ORIG_DIR/runs/$TEST_NAME"
if [[ ! -d "$TEST_SOURCE_DIR" ]]; then
    TEST_SOURCE_DIR="$ORIG_DIR/$TEST_NAME"
fi
if [[ ! -d "$TEST_SOURCE_DIR" ]]; then
    echo -e "${RED}Test directory not found. Checked:$NC"
    echo -e "${RED}  $ORIG_DIR/runs/$TEST_NAME$NC"
    echo -e "${RED}  $ORIG_DIR/$TEST_NAME$NC"
    exit 1
fi

# Generate current list of source files (sorted for consistent comparison)
CURRENT_SOURCE_FILES=$(find "$ORIG_DIR/source" "$TEST_SOURCE_DIR" \
    -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.cxx" -o -name "*.f90" \) \
    2>/dev/null | sort)

if [[ -f "$SOURCE_FILES_FILE" ]]; then
    OLD_SOURCE_FILES=$(<"$SOURCE_FILES_FILE")
    if [[ "$OLD_SOURCE_FILES" != "$CURRENT_SOURCE_FILES" ]]; then
        echo -e "${PURPLE}Source files changed (added/removed). Will re-run CMake...${NC}"
        RERUN_CMAKE=1
    fi
else
    # First build - need to run cmake anyway
    RERUN_CMAKE=1
fi

# ==================== Track CMake Files (detect modifications) ====================
# Dynamically find all CMake files to track
mapfile -t CMAKE_FILES < <(find "$ORIG_DIR/source" "$ORIG_DIR/config" \
    -type f \( -name "CMakeLists.txt" -o -name "*.cmake" \) \
    2>/dev/null | sort)

# Generate current mtimes for cmake files (fix: avoid having a trailing newline difference)
CURRENT_CMAKE_MTIMES=""
for cmake_file in "${CMAKE_FILES[@]}"; do
    if [[ -f "$cmake_file" ]]; then
        mtime=$(stat -c %Y "$cmake_file" 2>/dev/null || stat -f %m "$cmake_file" 2>/dev/null)
        CURRENT_CMAKE_MTIMES+="$cmake_file:$mtime
"
    fi
done

# Strip trailing newlines (important: both old and current must be normalized)
strip_trailing_newlines() {
    sed ':a;N;$!ba;s/\n*$//'
}
if [[ -f "$CMAKE_MTIMES_FILE" ]]; then
    OLD_CMAKE_MTIMES=$(<"$CMAKE_MTIMES_FILE")
    # strip trailing newlines before comparing!
    if [[ "$(echo "$OLD_CMAKE_MTIMES" | strip_trailing_newlines)" != "$(echo "$CURRENT_CMAKE_MTIMES" | strip_trailing_newlines)" ]]; then
        echo -e "${PURPLE}CMake files modified. Will re-run CMake...${NC}"
        RERUN_CMAKE=1
    fi
fi

# ==================== Track Environment Paths (PATH / LD_LIBRARY_PATH) ====================
CURRENT_ENV_PATHS="$(echo "PATH:"; echo "$PATH" | tr ':' '\n' | sort; echo "LD_LIBRARY_PATH:"; echo "$LD_LIBRARY_PATH" | tr ':' '\n' | sort)"

if [[ -f "$ENV_PATHS_FILE" ]]; then
    OLD_ENV_PATHS=$(<"$ENV_PATHS_FILE")
    if [[ "$OLD_ENV_PATHS" != "$CURRENT_ENV_PATHS" ]]; then
        echo -e "${PURPLE}PATH or LD_LIBRARY_PATH changed. Will re-run CMake...${NC}"
        RERUN_CMAKE=1
    fi
fi

# Intel OneAPI modules set CPATH to include Intel MPI headers.
# icpx gives CPATH higher priority than -isystem, which can shadow
# the OpenMPI mpi.h when both -I and -isystem point to the same dir.
# Unsetting CPATH avoids the wrong mpi.h being picked up.
unset CPATH

# ==================== Run CMake if needed ====================
if [[ ! -f Makefile || $RERUN_CMAKE -eq 1 ]]; then
    echo -e "${ORANGE}Running CMake...${NC}"
    cmake -S "$ORIG_DIR/source" -DCONFIG="$CONFIG" -DTEST_DIR="$TEST_NAME" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON $CMAKE_FLAGS > "$CMAKE_OUT" 2> "$CMAKE_ERR"
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}CMake failed. See $CMAKE_ERR${NC}"
        exit 1
    fi
    # Symlink compile_commands.json to project root for clangd / IntelliSense
    if [[ -f "$BUILD_DIR/compile_commands.json" ]]; then
        ln -sf "$BUILD_DIR/compile_commands.json" "$ORIG_DIR/compile_commands.json"
    fi
    # Save current source files list, cmake mtimes, and env paths after successful cmake
    echo "$CURRENT_SOURCE_FILES" > "$SOURCE_FILES_FILE"
    # Avoid a trailing newline to keep comparison precise
    echo -n "$CURRENT_CMAKE_MTIMES" | strip_trailing_newlines > "$CMAKE_MTIMES_FILE"
    echo "$CURRENT_ENV_PATHS" > "$ENV_PATHS_FILE"
else
    echo -e "${BLUE}CMake skipped: Makefile already exists and no changes detected.${NC}"
fi

# ==================== Linking Filter ====================
linking_filter() {
    while IFS= read -r line; do
        if [[ "$line" == *"Linking CXX executable"* ]]; then
            echo "[LINKING] Linking..."
        else
            echo "$line"
        fi
    done
}

# ==================== Progress Bar Output ====================
PROGRESS_STATE_FILE=$(mktemp)
echo 0 > "$PROGRESS_STATE_FILE"

progress_bar_and_filtered_output() {
    local width=50
    local linking_printed=0
    local progress_done=0
    local max_percent=0

    while IFS= read -r line; do
        # Detect linking
        if [[ "$line" == *"[LINKING] Linking..."* && $linking_printed -eq 0 ]]; then
            percent=100
            linking_printed=1
            progress_done=1
        elif [[ "$line" =~ \[\ *([0-9]{1,3})%\] && $progress_done -eq 0 ]]; then
            percent=${BASH_REMATCH[1]}
            # With parallel make, percentages arrive out of order; only advance forward
            (( percent <= max_percent )) && continue
            max_percent=$percent
        else
            continue
        fi

        (( percent > 100 )) && percent=100
        local blocks=$((percent * width / 100))
        (( blocks < 1 )) && blocks=1
        local empty=$((width - blocks))

        local bar=""
        if (( blocks > 1 )); then
            bar="$(printf '=%.0s' $(seq 1 $((blocks - 1))))=>"
        else
            bar=">"
        fi
        bar="$(printf "%-*s" "$width" "$bar")"

        if [[ $progress_done -eq 0 ]]; then
            printf "\rProgress: [${GREEN}%s${NC}] %3s%%\033[K" "$bar" "$percent"
        elif [[ $progress_done -eq 1 ]]; then
            printf "\rProgress: [${GREEN}%s${NC}] %3s%%\033[K\n" "$bar" "$percent"
            printf "${PURPLE}Linking...${NC}"
            progress_done=2
        fi
    done

    echo "$progress_done" > "$PROGRESS_STATE_FILE"
}

# ==================== Run Make ====================
echo -e "${CYAN}Running Make...${NC}"
PROGRESS_FIFO=$(mktemp -u)
mkfifo "$PROGRESS_FIFO"

linking_filter < "$PROGRESS_FIFO" | progress_bar_and_filtered_output &
PROGRESS_PID=$!

if [[ "${VERBOSE:-0}" == "1" ]]; then
    # Verbose mode: stream raw make output (including compile commands).
    # Also tee into the progress FIFO so the background reader can finish.
    stdbuf -oL make VERBOSE=1 -j"${MAKE_JOBS}" --output-sync=target 2> "$MAKE_ERR" \
        | tee "$PROGRESS_FIFO" "$MAKE_OUT"
    MAKE_EXIT_CODE=${PIPESTATUS[0]}
else
    # Default mode: compact progress + linking indicator.
    stdbuf -oL make -j"${MAKE_JOBS}" --output-sync=target 2> "$MAKE_ERR" \
        | tee "$PROGRESS_FIFO" > "$MAKE_OUT"
    MAKE_EXIT_CODE=${PIPESTATUS[0]}
fi

wait "$PROGRESS_PID" 2>/dev/null
rm -f "$PROGRESS_FIFO"

PROGRESS_DONE=$(<"$PROGRESS_STATE_FILE")
rm -f "$PROGRESS_STATE_FILE"

# ==================== Final Status ====================

if [[ $MAKE_EXIT_CODE -ne 0 ]]; then
    if [[ $PROGRESS_DONE -eq 0 ]]; then
        echo
    fi
    echo -e "${FAIL}Make failed. See $MAKE_ERR${NC}"
    exit 1
else
    # delete old symlink if it exists
    if [[ -L rich ]]; then
        rm rich
    fi
    ln -s rich_$CONFIG rich
    if [[ $PROGRESS_DONE -eq 0 ]]; then
        echo
    fi
    echo -e "${SUCCESS}Done!${NC}"
fi
