#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --job-name=branson-paper
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --nodelist=d25g[133-140]
#SBATCH --time=00:40:00
#SBATCH --output=/home/maorm/RICH/build/branson_paper128/job_%j.out
#SBATCH --error=/home/maorm/RICH/build/branson_paper128/job_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export OMPI_MCA_btl=^openib OMPI_MCA_osc=ucx FI_PROVIDER=verbs
archive=/home/maorm/RICH/build/branson_paper128/run_${SLURM_JOB_ID}
mkdir -p "$archive"
export BRANSON_PAPER_LOCAL=/tmp/branson_${USER}_paper_${SLURM_JOB_ID}
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 bash -c '
 mkdir -p "$BRANSON_PAPER_LOCAL"
 cp /home/maorm/RICH/build/branson_cp128/diagnosis/branson_o2/BRANSON "$BRANSON_PAPER_LOCAL/"
 cp /home/maorm/RICH/build/branson_paper128/paper_cartesian.xml "$BRANSON_PAPER_LOCAL/"
'
cd "$BRANSON_PAPER_LOCAL"
sha256sum BRANSON paper_cartesian.xml > "$archive/inputs.sha256"
printf 'nodes=%s\n' "$SLURM_JOB_NODELIST" > "$archive/allocation.txt"
TIMEFORMAT='%R'
rc=0
{ time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 ./BRANSON paper_cartesian.xml > "$archive/branson.out" 2> "$archive/branson.err"; } 2> "$archive/branson.elapsed" || rc=$?
echo "Branson exit=$rc"
exit "$rc"
