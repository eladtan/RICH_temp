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

export CCACHE_CPP2=yes

# ==================== Parse Arguments ====================
CONFIG="${1:-gnuRelease}"
MAKE_JOBS="$(nproc)"
CMAKE_FLAGS=""

for arg in "${@:2}"; do
    case "$arg" in
        --jobs=*)
            MAKE_JOBS="${arg#--jobs=}"
            ;;
        --with_asan)
            CMAKE_FLAGS+=" -DASAN=1 "
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            echo -e "Usage: $0 [config] [--jobs=<N>] [--with_asan]"
            exit 1
            ;;
    esac
done

# ==================== Paths ====================
ORIG_DIR="$(pwd)"
BUILD_DIR="$ORIG_DIR/build/${CONFIG}_tests"

MAKE_OUT="$BUILD_DIR/tests_build.out"
MAKE_ERR="$BUILD_DIR/tests_build.err"
CMAKE_OUT="$BUILD_DIR/tests_cmake.out"
CMAKE_ERR="$BUILD_DIR/tests_cmake.err"

# ==================== Track Source Files (detect add/remove) ====================
SOURCE_FILES_FILE="$BUILD_DIR/.source_files"
CMAKE_MTIMES_FILE="$BUILD_DIR/.cmake_mtimes"
RERUN_CMAKE=0

mkdir -p "$BUILD_DIR" || { echo -e "${RED}Failed to create $BUILD_DIR${NC}"; exit 1; }
cd "$BUILD_DIR" || { echo -e "${RED}Failed to cd into $BUILD_DIR${NC}"; exit 1; }

CURRENT_SOURCE_FILES=$(find "$ORIG_DIR/source" "$ORIG_DIR/tests_catch2" \
    -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.hpp" -o -name "*.h" \) \
    2>/dev/null | sort)

if [[ -f "$SOURCE_FILES_FILE" ]]; then
    OLD_SOURCE_FILES=$(<"$SOURCE_FILES_FILE")
    if [[ "$OLD_SOURCE_FILES" != "$CURRENT_SOURCE_FILES" ]]; then
        echo -e "${PURPLE}Source files changed (added/removed). Will re-run CMake...${NC}"
        RERUN_CMAKE=1
    fi
else
    RERUN_CMAKE=1
fi

# ==================== Track CMake Files (detect modifications) ====================
mapfile -t CMAKE_FILES < <(find "$ORIG_DIR/source" "$ORIG_DIR/config" "$ORIG_DIR/tests_catch2" \
    -type f \( -name "CMakeLists.txt" -o -name "*.cmake" \) \
    2>/dev/null | sort)

CURRENT_CMAKE_MTIMES=""
for cmake_file in "${CMAKE_FILES[@]}"; do
    if [[ -f "$cmake_file" ]]; then
        mtime=$(stat -c %Y "$cmake_file" 2>/dev/null || stat -f %m "$cmake_file" 2>/dev/null)
        CURRENT_CMAKE_MTIMES+="$cmake_file:$mtime
"
    fi
done

strip_trailing_newlines() {
    sed ':a;N;$!ba;s/\n*$//'
}

if [[ -f "$CMAKE_MTIMES_FILE" ]]; then
    OLD_CMAKE_MTIMES=$(<"$CMAKE_MTIMES_FILE")
    if [[ "$(echo "$OLD_CMAKE_MTIMES" | strip_trailing_newlines)" != "$(echo "$CURRENT_CMAKE_MTIMES" | strip_trailing_newlines)" ]]; then
        echo -e "${PURPLE}CMake files modified. Will re-run CMake...${NC}"
        RERUN_CMAKE=1
    fi
fi

# ==================== Run CMake if needed ====================
if [[ ! -f Makefile || $RERUN_CMAKE -eq 1 ]]; then
    echo -e "${ORANGE}Running CMake...${NC}"
    cmake -S "$ORIG_DIR/source" -DCONFIG="$CONFIG" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON $CMAKE_FLAGS > "$CMAKE_OUT" 2> "$CMAKE_ERR"
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}CMake failed. See $CMAKE_ERR${NC}"
        cat "$CMAKE_ERR"
        exit 1
    fi
    echo "$CURRENT_SOURCE_FILES" > "$SOURCE_FILES_FILE"
    echo -n "$CURRENT_CMAKE_MTIMES" | strip_trailing_newlines > "$CMAKE_MTIMES_FILE"
else
    echo -e "${BLUE}CMake skipped: Makefile already exists and no changes detected.${NC}"
fi

# ==================== Progress Bar Output ====================
linking_filter() {
    while IFS= read -r line; do
        if [[ "$line" == *"Linking CXX"* ]]; then
            echo "[LINKING] Linking..."
        else
            echo "$line"
        fi
    done
}

progress_bar_and_filtered_output() {
    local width=50
    local linking_printed=0
    local progress_done=0

    while IFS= read -r line; do
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

        if [[ $progress_done -eq 0 ]]; then
            printf "\rProgress: [${GREEN}%s${NC}] %3s%%\033[K" "$bar" "$percent"
        elif [[ $progress_done -eq 1 ]]; then
            printf "\rProgress: [${GREEN}%s${NC}] %3s%%\033[K\n" "$bar" "$percent"
            printf "${PURPLE}Linking...${NC}"
            progress_done=2
        fi
    done
}

# ==================== Run Make (only rich_tests target) ====================
echo -e "${CYAN}Building unit tests...${NC}"
if [[ "${VERBOSE:-0}" == "1" ]]; then
    stdbuf -oL make VERBOSE=1 -j"${MAKE_JOBS}" rich_tests --output-sync=target 2> "$MAKE_ERR" \
        | tee "$MAKE_OUT"
    MAKE_EXIT_CODE=${PIPESTATUS[0]}
else
    stdbuf -oL make -j"${MAKE_JOBS}" rich_tests --output-sync=target 2> "$MAKE_ERR" \
        | tee >(linking_filter | progress_bar_and_filtered_output) > "$MAKE_OUT"
    MAKE_EXIT_CODE=${PIPESTATUS[0]}
fi

# ==================== Final Status ====================
if [[ $MAKE_EXIT_CODE -ne 0 ]]; then
    echo
    echo -e "${FAIL}Build failed. See $MAKE_ERR${NC}"
    exit 1
else
    echo
    echo -e "${SUCCESS}Unit tests built successfully!${NC}"
    echo -e "${GREEN}Executable: $BUILD_DIR/tests_catch2/rich_tests${NC}"
fi
