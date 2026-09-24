"""Site-specific launcher construction; no changes to the benchmarked libraries."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NODES = ['d25g7', 'd25g8', 'd25g9', 'd25g11']
JOB = '10185981'
INTEL = '/software/x86_64/5.14.0/Intel/OneApi-2024.0.1/mpi/2021.11'
OPENMPI = '/software/x86_64/5.14.0/openmpi/4.1.6/gcc/12.3.0'
FABRIC = '/home/maorm/opt/libfabric-2.6.0/current'
PIN = str(ROOT / 'regression_tests/mpi_rma/timing/pin_rank.sh')


def environment(mpi, debug=False, intel_provider='verbs'):
    env = dict(os.environ)
    # MPI-specific settings are applied only to the matching executable.
    for name in list(env):
        if name.startswith(('I_MPI_', 'OMPI_MCA_')):
            del env[name]
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', UCX_TLS='all',
               FI_PROVIDER='verbs', BENCHMARK_CPUS='0,2,4,6,8,10,12,14')
    if mpi == 'intelmpi':
        env.update(I_MPI_ROOT=INTEL, I_MPI_PIN='0', I_MPI_FABRICS='shm:ofi',
                   I_MPI_OFI_LIBRARY_INTERNAL='0', I_MPI_OFI_LIBRARY=FABRIC+'/lib/libfabric.so.1',
                   I_MPI_OFI_PROVIDER='verbs', I_MPI_DEBUG='5' if debug else '0')
        env['LD_LIBRARY_PATH'] = INTEL+'/lib:'+env.get('LD_LIBRARY_PATH','')
        if intel_provider == 'mlx':
            # Use one libfabric runtime for MPI and EasyRMA in this process.
            fabric = INTEL+'/libfabric/lib'
            env.pop('FI_PROVIDER', None)
            env.update(I_MPI_OFI_PROVIDER='mlx', I_MPI_OFI_LIBRARY=fabric+'/libfabric.so.1',
                       FI_PROVIDER_PATH=fabric+'/prov')
            env['LD_LIBRARY_PATH'] = fabric+':'+env['LD_LIBRARY_PATH']
    return env


def launcher(mpi, ranks):
    ppn = ranks // len(NODES)
    if mpi == 'openmpi':
        return [OPENMPI+'/bin/mpirun', '--prefix', OPENMPI, '--mca', 'plm', 'rsh',
                '--mca', 'pml', 'ucx', '--mca', 'osc', 'ucx', '--mca', 'op', '^avx',
                '-x', 'LD_LIBRARY_PATH', '-x', 'OMP_NUM_THREADS', '-x', 'OPENBLAS_NUM_THREADS',
                '-x', 'UCX_TLS', '-x', 'FI_PROVIDER', '-x', 'BENCHMARK_CPUS',
                '--host', ','.join(f'{node}:{ppn}' for node in NODES),
                '--map-by', f'ppr:{ppn}:node', '--bind-to', 'none', '-np', str(ranks), PIN]
    if mpi == 'mpich':
        return ['srun', '--jobid='+JOB, '--mpi=pmi2', '-N4', '-n'+str(ranks),
                '--ntasks-per-node='+str(ppn), '--distribution=block:block', '--cpu-bind=none', PIN]
    return [INTEL+'/bin/mpiexec.hydra', '-bootstrap', 'ssh', '-hosts', ','.join(NODES),
            '-ppn', str(ppn), '-n', str(ranks), PIN]


def binary(mpi, name='marshak_mpi_rma'):
    return str(ROOT / ('build/mpi_rma_validation/timing_'+mpi) / name)
