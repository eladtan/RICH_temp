#!/bin/bash
# Mach-45 radiative shock (Steinberg & Heizler 2022, Sec. 5.2).
# Submit from this directory:  sbatch run.sh
#SBATCH --partition=bigrun
#SBATCH --job-name=Mach45_IMC
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --output=mach45_%j.out
#SBATCH --error=mach45_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -x ./rich ]]; then
    echo "Missing executable ./rich in $(pwd)" >&2
    exit 1
fi

export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/Mach45/$(date +%Y-%m-%d_%H-%M-%S)"
run_timestamp="$(basename "$RICH_OUTPUT_DIR")"
mpi_ranks="${SLURM_NTASKS:-64}"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"

run_prefix="${RICH_OUTPUT_DIR}/mach45"
Np="${NP:-4000}"

parameter_file="$RICH_OUTPUT_DIR/mach45_parameters.txt"
mesh_dx_cm="$(awk -v n="$Np" "BEGIN {printf \"%.17g\", 500.0 / n}")"
total_initial_particles="$((Np * 50))"
total_new_photons_per_step="$((Np * 25))"
nominal_population_control_slots="$((Np * 100))"
{
    printf "%s\n" "# Mach45 launch parameters"
    printf "run_timestamp=%s\n" "$run_timestamp"
    printf "output_directory=%s\n" "$RICH_OUTPUT_DIR"
    printf "mpi_ranks=%s\n" "$mpi_ranks"
    printf "mesh_cells_x=%s\n" "$Np"
    printf "%s\n" "mesh_cells_y=1"
    printf "%s\n" "mesh_cells_z=1"
    printf "%s\n" "domain_xmin_cm=1950.0"
    printf "%s\n" "domain_xmax_cm=2450.0"
    printf "%s\n" "domain_length_x_cm=500.0"
    printf "mesh_dx_cm=%s\n" "$mesh_dx_cm"
    printf "%s\n" "shock_x_cm=2300.0"
    printf "%s\n" "gamma_gas=5/3"
    printf "%s\n" "Cv_erg_per_g_K=1.45e15/1.160451812e7"
    printf "%s\n" "sigma_a_coefficient=0.0142"
    printf "%s\n" "sigma_a_rho_exponent=2"
    printf "%s\n" "sigma_a_temperature_keV_exponent=-3.5"
    printf "%s\n" "sigma_s_coefficient=0.4006"
    printf "%s\n" "sigma_s_rho_exponent=1"
    printf "%s\n" "rho_upstream_g_per_cc=1.0"
    printf "%s\n" "rho_downstream_g_per_cc=6.43"
    printf "%s\n" "T_upstream_keV=0.1"
    printf "%s\n" "T_downstream_keV=8.36"
    printf "%s\n" "v_upstream_shock_frame_cm_s=5.71e8"
    printf "%s\n" "v_downstream_shock_frame_cm_s=0.89e8"
    printf "%s\n" "v_downstream_lab_cm_s=-4.82e8"
    printf "%s\n" "shock_speed_cm_s=5.71e8"
    printf "%s\n" "t_final_s=3e-6"
    printf "%s\n" "cfl_factor=0.3"
    printf "%s\n" "initial_dt_factor=0.001"
    printf "%s\n" "timestep_ramp_start_cycle=750"
    printf "%s\n" "timestep_ramp_factor=1.01"
    printf "%s\n" "dump_interval_cycles=50"
    printf "%s\n" "vtk_interval_cycles=200"
    printf "%s\n" "new_photons_per_cell=25"
    printf "%s\n" "max_photons_per_cell=100"
    printf "%s\n" "initial_particles_per_cell=50"
    printf "%s\n" "boundary_photons_per_cell=50"
    printf "total_initial_particles=%s\n" "$total_initial_particles"
    printf "total_new_photons_per_step=%s\n" "$total_new_photons_per_step"
    printf "nominal_population_control_slots=%s\n" "$nominal_population_control_slots"
    printf "%s\n" "with_hydro=true"
    printf "%s\n" "diffusion_pressure_gradient=false"
    printf "%s\n" "MMC=false"
    printf "%s\n" "multigroup_opacity=false"
    printf "%s\n" "random_walk=false"
    printf "%s\n" "energy_boundary_min=0.0"
    printf "%s\n" "energy_boundary_max=1.0e30"
    printf "%s\n" "population_control_comb_parameter=10"
    printf "%s\n" "manager=new-rdma-auto"
    printf "%s\n" "profile_file=mach45_analytic.dat"
} > "$parameter_file"
echo "Simulation parameters: $parameter_file"

exec mpirun -np "${SLURM_NTASKS:-64}" ./rich \
    "$Np" "$run_prefix" 25 100 \
    --profile mach45_analytic.dat \
    --manager new-rdma-auto
