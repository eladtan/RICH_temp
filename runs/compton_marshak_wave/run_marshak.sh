#!/bin/bash
set -euo pipefail

cd /home/elads/RICH_monte

ml openmpi/4.1.6/Intel/OneApi/2024.2.1

CASE_DIR="runs/compton_marshak_wave"
NP=16

echo "=== Building (MPI) ==="
./build_rich.sh intelReleaseMPI --test_name="${CASE_DIR}" --energy_groups_num=32

rm -f "${CASE_DIR}"/{x_pos,Tgas,Trad}_{mc,mc_iso,diffusion}.txt "${CASE_DIR}"/min_fleck.txt

echo "=== Running diffusion on SLURM (${NP} ranks) ==="
sbatch --wait --exclusive --partition=bigrun --ntasks="${NP}" --time=15 \
    --output="${CASE_DIR}/slurm_diffusion.out" \
    --error="${CASE_DIR}/slurm_diffusion.err" \
    --wrap "mpirun --timeout 600 -np ${NP} ./build/intelReleaseMPI/rich --diffusion" || true
if [[ ! -f "${CASE_DIR}/Trad_diffusion.txt" ]]; then
    echo "ERROR: diffusion run did not produce output" >&2; exit 1
fi

NP=32
echo "=== Running MC angle-dependent on SLURM (${NP} ranks) ==="
sbatch --wait --exclusive --partition=bigrun --ntasks="${NP}" \
    --output="${CASE_DIR}/slurm_mc.out" \
    --error="${CASE_DIR}/slurm_mc.err" \
    --wrap "mpirun --timeout 4600 -np ${NP} ./build/intelReleaseMPI/rich --mc" || true
if [[ ! -f "${CASE_DIR}/Trad_mc.txt" ]]; then
    echo "ERROR: MC angle-dependent run did not produce output" >&2; exit 1
fi

echo "=== Running MC isotropic on SLURM (${NP} ranks) ==="
sbatch --wait --exclusive --partition=bigrun --ntasks="${NP}" \
    --output="${CASE_DIR}/slurm_mc_iso.out" \
    --error="${CASE_DIR}/slurm_mc_iso.err" \
    --wrap "mpirun --timeout 4600 -np ${NP} ./build/intelReleaseMPI/rich --mc-isotropic" || true
if [[ ! -f "${CASE_DIR}/Trad_mc_iso.txt" ]]; then
    echo "ERROR: MC isotropic run did not produce output" >&2; exit 1
fi

echo "=== Plotting ==="
python3 "${CASE_DIR}/plot_marshak.py"

echo "=== All done ==="
