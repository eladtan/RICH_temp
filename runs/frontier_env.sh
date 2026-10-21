# Frontier (OLCF) environment for RICH builds and runs. Source it, don't run it:
#   source runs/frontier_env.sh [hip|omp]
#
# The user-built dependencies (~/modulefiles: Open MPI 5.0.10, HDF5, VTK, Boost)
# were built with GCC 12.2; the host compiler is gcc-native 14.2, whose libstdc++
# is backward compatible with them. hipcc is pointed at the same GCC 14 install by
# source/CMakeLists.txt, so g++ and hipcc objects share one libstdc++.
# openmpi/5.0.10 asks for gcc/12.2.0; loading gcc-native/14.2 after it replaces
# that and Lmod prints a harmless "dependent module(s) are not currently loaded".
#
# The argument picks the Kokkos install used by --with-gpu builds:
#   hip  HIP + OpenMP, AMD_GFX90A + ZEN3 (default)
#   omp  OpenMP only, ZEN3 (CPU-thread runs)
# Override the installs with RICH_KOKKOS_HIP / RICH_KOKKOS_OMP.

module use "$HOME/modulefiles"
# The site Kokkos module (kokkos/4.6.02-gpu) has no OpenMP backend
module unload kokkos 2>/dev/null
module load openmpi/5.0.10 hdf5/2.1.1 vtk/9.7.0 boost/1.85.0 rocm/6.4.2 gcc-native/14.2 2>/dev/null

: "${RICH_KOKKOS_HIP:=$HOME/opt/kokkos/4.7.03-hip6.4.2-gcc14.2}"
: "${RICH_KOKKOS_OMP:=$HOME/opt/kokkos/4.7.03-openmp-gcc14.2}"
case "${1:-hip}" in
    hip) export Kokkos_ROOT="$RICH_KOKKOS_HIP" ;;
    omp) export Kokkos_ROOT="$RICH_KOKKOS_OMP" ;;
    *) echo "frontier_env.sh: unknown Kokkos variant '$1' (expected hip or omp)" >&2; return 1 ;;
esac

# Slingshot: Open MPI goes through OFI (set by the openmpi module); hybrid
# matching avoids running out of CXI hardware match entries with many messages.
export FI_CXI_RX_MATCH_MODE=hybrid
# Jobs without Slingshot VNIs (the service partition) cannot open a CXI domain
# (fi_domain: Function not implemented), so MPI_Init fails. Such jobs are single
# node: use Open MPI's shared-memory path instead.
if [[ -n "${SLURM_JOB_ID:-}" && -z "${SLINGSHOT_VNIS:-}" ]]; then
    export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm
    unset OMPI_MCA_mtl
fi
# Particle queues resize repeatedly; don't keep deregistered memory pinned.
export OMPI_MCA_mpi_leave_pinned=0
