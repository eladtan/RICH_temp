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
CURRENT_CMD="$0 $*"
CONFIG="$1"
TEST_ARG="$2"
TEST_NAME="${TEST_ARG#--test_name=}"

CMAKE_FLAGS=""
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
        --mc_debug)
            CMAKE_FLAGS+=" -DMC_DEBUG=1 "
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            exit 1
            ;;
    esac
done

# ==================== Paths ====================
ORIG_DIR="$(pwd)"
BUILD_DIR="$ORIG_DIR/build/$CONFIG"
CMD_FILE="$BUILD_DIR/.build_cmd"

MAKE_OUT="$BUILD_DIR/${CONFIG}_build.out"
MAKE_ERR="$BUILD_DIR/${CONFIG}_build.err"
CMAKE_OUT="$BUILD_DIR/${CONFIG}_cmake.out"
CMAKE_ERR="$BUILD_DIR/${CONFIG}_cmake.err"

# ==================== Validate arguments ====================

if [[ $# -lt 2 || "$2" != --test_name=* ]]; then
    echo -e "${RED}Usage: $0 <config> --test_name=<name> [--with_asan] [--energy_groups_num=<N>] [--mc_debug]${NC}"
    exit 1
fi


# ==================== Reset Build if Test Name Changed ====================

# Reset build directory if test name changed
if [[ -f "$CMD_FILE" ]]; then
    OLD_CMD=$(<"$CMD_FILE")
    CURRENT_CMD="$(echo "$CURRENT_CMD" | tr -s '[:space:]' ' ' | sed 's/ *$//')"
    OLD_CMD="$(echo "$OLD_CMD" | tr -s '[:space:]' ' ' | sed 's/ *$//')"
    if [[ "$OLD_CMD" != "$CURRENT_CMD" ]]; then
        echo -e "${PURPLE}Build command changed. Cleaning $BUILD_DIR...${NC}"
        rm -rf "$BUILD_DIR"
    fi
fi

# Always ensure build directory exists
mkdir -p "$BUILD_DIR" || { echo -e "${RED}Failed to create $BUILD_DIR${NC}"; exit 1; }

# Change into build directory
cd "$BUILD_DIR" || { echo -e "${RED}Failed to cd into $BUILD_DIR${NC}"; exit 1; }

# Save command for future comparison
echo "$CURRENT_CMD" > "$CMD_FILE"


# Run CMake if Makefile doesn't exist
if [[ ! -f Makefile ]]; then
    echo -e "${ORANGE}Running CMake...${NC}"
    cmake -S "$ORIG_DIR/source" -DCONFIG="$CONFIG" -DTEST_DIR="$TEST_NAME" $CMAKE_FLAGS > "$CMAKE_OUT" 2> "$CMAKE_ERR"
    if [[ $? -ne 0 ]]; then
        echo -e "${RED}CMake failed. See $CMAKE_ERR${NC}"
        exit 1
    fi
else
    echo -e "${BLUE}CMake skipped: Makefile already exists.${NC}"
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

# ==================== Run Make with Tee & Progress ====================
echo -e "${CYAN}Running Make...${NC}"
stdbuf -oL make -j12 2> "$MAKE_ERR" \
    | tee >(linking_filter | progress_bar_and_filtered_output) > "$MAKE_OUT"

MAKE_EXIT_CODE=${PIPESTATUS[0]}

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
