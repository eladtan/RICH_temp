#!/bin/bash
#SBATCH --account=EUHPC_D35_204
#SBATCH --job-name="Space3D-StrongScale"
#SBATCH --partition=dcgp_usr_prod
#SBATCH -N 1
#SBATCH --ntasks-per-node=112
#SBATCH --cpus-per-task=1
#SBATCH --time=02:30:00
#SBATCH --exclusive
#SBATCH --out=space_%j_run.out
#SBATCH --err=space_%j_run.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il
#SBATCH --gres=none
#SBATCH --qos=dcgp_qos_bprod

module restore rich

ln -s space_${SLURM_JOB_ID}_run.out space_SS_${SLURM_JOB_ID}_n${SLURM_NTASKS}.out
ln -s space_${SLURM_JOB_ID}_run.err space_SS_${SLURM_JOB_ID}_n${SLURM_NTASKS}.err

VTUNE="/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/intel-oneapi-vtune-2024.1.0-6e2hsnvveq46e5z36kaygi54goqxhelh/vtune/2024.1/bin64/vtune"

# Keep background cells per node fixed, as in weak scaling. This preserves
# the rank-local mesh footprint while the global mesh grows with node count.
#
# In a fixed physical domain, h ~ nodes^(-1/3). Since emission is per cell,
# the number of emitted photons grows as nodes. For an iso-work strong-scaling
# benchmark, make crossings/photon scale as 1/nodes:
#
#   crossings/photon ~ dt / h
#   dt / h ~ nodes^(-1)
#   dt ~ nodes^(-4/3)
#
# Anchor the scaling at the old 20-node strong-scale setup:
#   20 nodes * 112 ranks/node * 5000 cells/rank = 11200000 cells
#   dt = 2e-10
# Override BASE_NBASE_PER_NODE=..., DT_REF_NODES=..., DT_REF=..., or DT=...
# when submitting if you want a different anchor.
BASE_NBASE_PER_NODE=${BASE_NBASE_PER_NODE:-560000}
DT_REF_NODES=${DT_REF_NODES:-20}
DT_REF=${DT_REF:-2e-10}
NODES=${SLURM_NNODES:-1}

NBASE=${NBASE:-$(( BASE_NBASE_PER_NODE * NODES ))}
N_EMIT=${N_EMIT:-5}
CYCLES=${CYCLES:-5}
DT=${DT:-$(python3 - <<PY
dt_ref_nodes = float("${DT_REF_NODES}")
nodes = float("${NODES}")
dt_ref = float("${DT_REF}")
print("{:.12g}".format(dt_ref * (dt_ref_nodes / nodes) ** (4.0 / 3.0)))
PY
)}

echo "Strong scaling parameters:"
echo "  nodes=${NODES} ranks=${SLURM_NTASKS}"
echo "  NBASE=${NBASE}"
echo "  DT=${DT}"
echo "  DT_REF_NODES=${DT_REF_NODES}"
echo "  DT_REF=${DT_REF}"
echo "  N_EMIT=${N_EMIT}"
echo "  CYCLES=${CYCLES}"

ARGS="${NBASE} ${N_EMIT} ${CYCLES} --dt ${DT} --new_ibv" # --profiling-dir prof_strong_${NODES}"
EXEC="./rich"

VTUNE_TYPE="hotspots"
VTUNE_ROOT="${VTUNE_ROOT:-/leonardo_work/EUHPC_D35_204/vtune_results_$VTUNE_TYPE}"
VTUNE_LABEL="${VTUNE_LABEL:-space_${SLURM_JOB_ID}_n${SLURM_NTASKS}}"
VTUNE_DIR="${VTUNE_ROOT}/${VTUNE_LABEL}"
mkdir -p "$VTUNE_ROOT"

# command="mpirun $VTUNE -collect $VTUNE_TYPE -knob enable-stack-collection=true -knob stack-size=0 -trace-mpi -result-dir $VTUNE_DIR -- $EXEC $ARGS"
command="mpirun $EXEC $ARGS"
echo $command
$command
