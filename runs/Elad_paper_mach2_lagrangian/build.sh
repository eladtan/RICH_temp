#!/bin/bash
# Build the Lagrangian Mach-2 driver on a compute node and copy the executable here.
# Submit from anywhere:  sbatch runs/Elad_paper_mach2_lagrangian/build.sh
# Then submit the run separately (not from inside this job, ORTE inherits SLURM_*):
#   sbatch --dependency=afterok:<this job id> run.sh
#SBATCH --partition=socket
#SBATCH --job-name=build-mach2-lag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/maorm/RICH/runs/Elad_paper_mach2_lagrangian/build_%j.out
#SBATCH --error=/home/maorm/RICH/runs/Elad_paper_mach2_lagrangian/build_%j.err

set -euo pipefail

if ! type -t module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi
module restore
hash -r

export CMAKE_PREFIX_PATH="$(python3 -c 'import sys; print(sys.prefix)')"
export PYTHON_FOR_BUILD="$(which python3)"

RICH_ROOT="/home/maorm/RICH"
RUN_DIR="$RICH_ROOT/runs/Elad_paper_mach2_lagrangian"
SUBDIR="imc_mach2_lagrangian"
cd "$RICH_ROOT"

./build_rich.sh gnuReleaseMPI \
    --test_name=Elad_paper_mach2_lagrangian \
    --build-subdir="$SUBDIR" \
    --jobs="${SLURM_CPUS_PER_TASK:-16}"

build_dir="$RICH_ROOT/build/gnuReleaseMPI/$SUBDIR"
exe=""
for candidate in "$build_dir/rich" "$build_dir/rich_gnuReleaseMPI"; do
    if [[ -x "$candidate" ]]; then exe="$candidate"; break; fi
done
if [[ ! -x "$exe" ]]; then
    echo "ERROR: no executable found in $build_dir" >&2
    exit 1
fi

cp "$exe" "$RUN_DIR/rich"
chmod +x "$RUN_DIR/rich"
sha256sum "$RUN_DIR/rich"
echo "Copied $exe -> $RUN_DIR/rich"
