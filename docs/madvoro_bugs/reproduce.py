#!/usr/bin/env python3
"""Rebuild bounded baseline witnesses; inspect results, do not equate exit 0 with correctness."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVIDENCE = HERE / 'evidence'
MPI_DEFINES = ['-DMADVORO_WITH_MPI', '-DSPATIAL_DS_WITH_MPI', '-DRICH_MPI', '-D__WITH_MPI']
INCLUDES = ['-I.', '-Isource', '-Isource/3D/tessellation/voronoi', '-Isource/3D/tessellation', '-Isource/utils']
BOOST = os.environ.get('BOOST_INCLUDE', '/software/x86_64/5.14.0/boost/1.78.0/include')
INCLUDES += ['-I' + BOOST] if BOOST else []
EXCEPTIONS = ['source/3D/tessellation/voronoi/exception/' + n + '.cpp' for n in ['MadVoroException', 'InvalidArgumentException', 'SizeException']]
AMOUNT = ['source/utils/mpi_utils/AmountManager.cpp']
# name: (source filename, MPI compilation, full mesh exception sources, extra sources, extra flags)
PROGRAMS = {
    'root_serial': ('root_api_repro.cpp', False, True, [], ['-D_GLIBCXX_ASSERTIONS']),
    'root_mpi': ('root_api_repro.cpp', True, True, AMOUNT, ['-D_GLIBCXX_ASSERTIONS']),
    'geometry': ('geometry_repro.cpp', True, True, AMOUNT, []),
    'partial_volume': ('geometry_partial_volume_repro.cpp', True, True, AMOUNT, ['-O0']),
    'centroid': ('geometry_face_centroid.cpp', False, False, [], []),
    'extracted': ('mpi_extracted_helpers.cpp', False, False, [], []),
    'mpi_dependency': ('mpi_dependency_repro.cpp', True, False, AMOUNT, ['-DNDEBUG']),
    'mixed_suppression': ('integration_mixed_suppression.cpp', True, True, AMOUNT, ['-D_GLIBCXX_ASSERTIONS']),
    'lifecycle': ('integration_load_balancer_lifecycle.cpp', True, False, [], ['-D_GLIBCXX_ASSERTIONS']),
    'communicator': ('integration_points_manager_comm.cpp', True, False, [], ['-D_GLIBCXX_ASSERTIONS']),
    'chain': ('integration_exchange_chain.cpp', True, False, ['source/mpi/ExchangeChain.cpp'], []),
}
# test name, program, ranks (0=non-MPI), arguments, time limit, expected baseline interpretation
CASES = [
    ('partial_volume', 'partial_volume', 1, [], 20, 'exit 5; all 8 volumes differ, all points drop 32->8'),
    ('serial_subset', 'root_serial', 0, ['subset'], 20, 'bounds assertion / abort'),
    ('serial_permutation', 'root_serial', 0, ['permutation'], 20, 'exit 4; active0 maps to31 instead of1'),
    ('serial_empty', 'root_serial', 0, ['empty'], 20, 'bounds assertion / abort'),
    ('serial_identity_control', 'root_serial', 0, ['identity'], 20, 'exit 0; full identity control'),
    ('mpi_identity_control', 'root_mpi', 1, ['identity'], 20, 'exit 0; warm suppressed full build'),
    ('mpi_subset', 'root_mpi', 1, ['subset'], 20, 'exit 0 but owned8/all8: incorrect retention'),
    ('mpi_serial_api', 'root_mpi', 1, ['mpi_serial'], 20, 'exit 3; missing environment exception'),
    ('first_suppressed', 'root_mpi', 1, ['first_suppressed'], 20, 'segmentation fault / launcher exit 139'),
    ('empty_continuity', 'root_mpi', 1, ['empty_continuity'], 20, 'bounds assertion after owned0 marker'),
    ('periodic_resolution', 'root_mpi', 2, ['periodic_resolution'], 20, 'exit 5; wrong physical preimage'),
    ('copy', 'geometry', 1, ['copy'], 20, 'segmentation fault after copied mesh count'),
    ('unbuilt_copy', 'geometry', 1, ['unbuilt_copy'], 20, 'segmentation fault in manager clone'),
    ('custom_release', 'geometry', 1, ['custom_release'], 20, 'exit 0 but outside predicate flips 1->0'),
    ('release_control', 'geometry', 1, ['release'], 20, 'exit 0; rectangular rebuild volume1'),
    ('centroid', 'centroid', 0, [], 10, 'exit 0 but centroid is twice expected'),
    ('extracted_helpers', 'extracted', 0, [], 10, 'exit 0; duplicated shifts and one-way peer removed'),
    ('compaction', 'mpi_dependency', 1, ['compaction'], 20, 'exit 1; injected completion order throws'),
    ('overflowfit', 'mpi_dependency', 2, ['overflowfit'], 20, 'exit 1; only48/80 payload bytes delivered'),
    ('oversize', 'mpi_dependency', 2, ['oversize'], 20, 'exit 1; only56/200 payload bytes delivered'),
    ('scalar8', 'mpi_dependency', 2, ['scalar8'], 20, 'exit 1; pending1/sent0 on sender'),
    ('self', 'mpi_dependency', 1, ['self'], 20, 'exit 1; flattened0 although per-query1'),
    ('mixed_suppression', 'mixed_suppression', 2, ['mixed'], 20, 'vector<double> assertion on nonsuppressor'),
    ('all_suppression_control', 'mixed_suppression', 2, ['all_suppress'], 20, 'exit 0; both retain65 points'),
    ('kernel_change', 'lifecycle', 1, ['kernel'], 20, 'exit 3; converter not initialized'),
    ('onedim', 'lifecycle', 1, ['onedim'], 20, 'exit 3; zero bins but MPI size1'),
    ('communicator', 'communicator', 2, [], 20, 'out-of-range destination assertion'),
    ('chain_permutation', 'chain', 1, ['permutation'], 20, 'exit 3; actual100,101 expected101,100'),
    ('chain_empty', 'chain', 2, ['empty'], 8, 'timeout124 after receiver early-return marker'),
]

def invoke(command, logfile, timeout, env):
    with logfile.open('w') as log:
        log.write('COMMAND: ' + shlex.join(command) + '\n')
        log.flush()
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
        log.write(f'\nHARNESS_EXIT_STATUS={code}\n')
    return code

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--case', action='append', choices=[x[0] for x in CASES], help='Repeat to select witnesses; default all.')
    parser.add_argument('--reuse', type=Path, help='Existing rerun directory whose build/ binaries should be used.')
    args = parser.parse_args()
    cases = [x for x in CASES if not args.case or x[0] in args.case]
    if args.list:
        for name, prog, ranks, argv, timeout, expected in cases:
            print(f'{name}: ranks={ranks} program={prog} args={argv} timeout={timeout}s; BASELINE {expected}')
        return 0
    if not args.build and not args.run:
        parser.print_help()
        return 0
    if args.run and not args.build and not args.reuse:
        parser.error('--run requires --build or --reuse <previous-rerun-directory>')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = EVIDENCE / ('rerun_' + stamp)
    output.mkdir()
    build = (args.reuse.resolve() / 'build') if args.reuse else output / 'build'
    build.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault('UCX_TLS', 'self,sm,tcp')
    env['OMP_NUM_THREADS'] = '1'
    results = []
    for prog in dict.fromkeys(c[1] for c in cases):
        src, mpi, mesh, extras, flags = PROGRAMS[prog]
        if not args.build:
            if not (build / prog).is_file():
                parser.error('Missing executable: ' + str(build / prog))
            continue
        compiler = shlex.split(os.environ.get('MPICXX' if mpi else 'CXX', 'mpicxx' if mpi else 'c++'))
        command = compiler + ['-std=c++17', '-O1', '-g1', '-fno-omit-frame-pointer', '-fopenmp'] + flags
        command += (MPI_DEFINES if mpi else []) + INCLUDES + [str(EVIDENCE / src)]
        command += (EXCEPTIONS if mesh else []) + extras + ['-o', str(build / prog)]
        print('Building ' + prog, flush=True)
        code = invoke(command, output / (prog + '_compile.log'), 300, env)
        results.append({'kind': 'build', 'program': prog, 'command': command, 'exit': code})
        (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        if code:
            print('Compilation failed; see ' + str(output), file=sys.stderr)
            return 2
    if args.run:
        for name, prog, ranks, argv, timeout, expected in cases:
            command = [str(build / prog)] + argv
            if ranks:
                command = shlex.split(env.get('MPIEXEC', 'mpiexec')) + ['--bind-to', 'none', '-np', str(ranks)] + command
            code = invoke(command, output / (name + '.log'), timeout, env)
            results.append({'kind': 'witness', 'case': name, 'command': command, 'exit': code, 'baseline_interpretation': expected, 'timeout_seconds': timeout})
            (output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
            print(f'{name}: exit {code}; inspect {output / (name + ".log")}', flush=True)
    print('Results: ' + str(output))
    print('No automatic correctness verdict: witnesses intentionally include crashes and wrong results with exit 0.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
