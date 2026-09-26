#!/bin/bash
#SBATCH --job-name=marshak-build
#SBATCH --partition=bigrun
#SBATCH --ntasks=16
#SBATCH --time=01:00:00
#SBATCH --output=buildjob-%j.out
#SBATCH --error=buildjob-%j.err

set -euo pipefail

# build_rich.sh wipes its build directory whenever the build command changes, so
# two tests building concurrently in the default directory delete each other's
# objects.  --build-subdir gives this benchmark its own directory and is excluded
# from the change detection.
cd /home/maorm/RICH
./build_rich.sh gnuReleaseMPI --test_name=converging_marshak --build-subdir=converging_marshak

BINARY="$(find build/gnuReleaseMPI/converging_marshak -name 'rich_gnuReleaseMPI' -type f -print -quit)"
if [[ -z "${BINARY}" ]]; then
    echo "Could not find the built executable" >&2
    exit 1
fi
cp "${BINARY}" runs/converging_marshak/rich.new
echo "BUILD DONE: ${BINARY}"
