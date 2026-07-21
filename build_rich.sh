#!/bin/bash

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
TEST_ARG="$2"
TEST_NAME="${TEST_ARG#--test_name=}"

CMAKE_FLAGS=""
MIXED_DEBUG_FILES=""
BUILD_SUBDIR=""
MAKE_JOBS="$(nproc)"
# Parse remaining optional args
for arg in "${@:3}"; do
    case "$arg" in
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
        --high-res)
            CMAKE_FLAGS+=" -DHIGH_RES=1 "
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

MAKE_OUT="$BUILD_DIR/${CONFIG}_build.out"
MAKE_ERR="$BUILD_DIR/${CONFIG}_build.err"
CMAKE_OUT="$BUILD_DIR/${CONFIG}_cmake.out"
CMAKE_ERR="$BUILD_DIR/${CONFIG}_cmake.err"

# ==================== Validate arguments ====================

if [[ $# -lt 2 || "$2" != --test_name=* ]]; then
    echo -e "${RED}Usage: $0 <config> --test_name=<name> [--with_asan] [--energy_groups_num=<N>] [--mc_debug] [--high-res] [--build-subdir=<name>] [--jobs=<N>]${NC}"
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
    -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.hpp" -o -name "*.h" \) \
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
    # Save current source files list and cmake mtimes after successful cmake
    echo "$CURRENT_SOURCE_FILES" > "$SOURCE_FILES_FILE"
    # Avoid a trailing newline to keep comparison precise
    echo -n "$CURRENT_CMAKE_MTIMES" | strip_trailing_newlines > "$CMAKE_MTIMES_FILE"
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
progress_bar_and_filtered_output() {
    local width=50
    local linking_printed=0
    local progress_done=0

    while IFS= read -r line; do
        # Detect linking
        if [[ "$line" == *"[LINKING] Linking..."* && $linking_printed -eq 0 ]]; then
            percent=100
            linking_printed=1
            progress_done=1
        elif [[ "$line" =~ \[\ *([0-9]{1,3})%\] && $progress_done -eq 0 ]]; then
            percent=${BASH_REMATCH[1]}
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

        # Print progress bar once
        if [[ $progress_done -eq 0 ]]; then
            printf "\rProgress: [${GREEN}%s${NC}] %3s%%\033[K" "$bar" "$percent"
        elif [[ $progress_done -eq 1 ]]; then
            printf "\rProgress: [${GREEN}%s${NC}] %3s%%\033[K\n" "$bar" "$percent"
            printf "${PURPLE}Linking...${NC}"
            progress_done=2
        fi
    done
}

# ==================== Run Make ====================
echo -e "${CYAN}Running Make...${NC}"
if [[ "${VERBOSE:-0}" == "1" ]]; then
    # Verbose mode: stream raw make output (including compile commands).
    stdbuf -oL make VERBOSE=1 -j"${MAKE_JOBS}" --output-sync=target 2> "$MAKE_ERR" \
        | tee "$MAKE_OUT"
    MAKE_EXIT_CODE=${PIPESTATUS[0]}
else
    # Default mode: compact progress + linking indicator.
    stdbuf -oL make -j"${MAKE_JOBS}" --output-sync=target 2> "$MAKE_ERR" \
        | tee >(linking_filter | progress_bar_and_filtered_output) > "$MAKE_OUT"
    MAKE_EXIT_CODE=${PIPESTATUS[0]}
fi

# ==================== Final Status ====================

if [[ $MAKE_EXIT_CODE -ne 0 ]]; then
    if [[ $progress_done -eq 0 ]]; then
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
    # if `progress_done` is 0 or not define, print an empty line
    if [[ $progress_done -eq 0 ]]; then
        echo
    fi
    echo -e "${SUCCESS}Done!${NC}"
fi
