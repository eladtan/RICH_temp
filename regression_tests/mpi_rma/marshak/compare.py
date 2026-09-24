#!/usr/bin/env python3
"""Compare MPI libraries and STORM backends at selected rank counts."""
import argparse
import bisect
import json
import math
from pathlib import Path
import random
import re
import signal
import statistics
import subprocess
import time
import os
import launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--new', type=int, default=2)
    parser.add_argument('--boundary', type=int, default=128)
    parser.add_argument('--dt-factor', type=float, default=1)
    parser.add_argument('--ranks', type=int, nargs='+', default=[8])
    parser.add_argument('--mpi', nargs='+', choices=['openmpi','mpich','intelmpi'],
                        default=['openmpi','mpich','intelmpi'])
    parser.add_argument('--intel-provider', choices=['verbs','mlx'], default='verbs')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--record-profile-variation', action='store_true',
                        help='Retain and flag profile discrepancies; physical checks remain mandatory')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, references = [], {}
    diffusion = [[float(v) for v in line.split()] for line in
                 (launch.ROOT/'source/monte/examples/marshak_wave_2/reference.txt').read_text().splitlines()]
    ref_x = [row[0] for row in diffusion]
    cases = [(n, mpi, backend) for n in args.ranks
             for mpi in args.mpi for backend in ['p2p', 'ofi', 'mpi']]
    config = {**vars(args), 'output': str(output), 'nodes': launch.NODES, 'job': launch.JOB,
              'timing': 'Maximum rank cycle-loop duration including its final barrier; initialization and validation excluded',
              'order': 'One full warmup per case; seeded shuffle of all cases in each measured repetition',
              'source': str(Path(__file__).with_name('main.cpp').resolve())}
    (output/'config.json').write_text(json.dumps(config, indent=2)+'\n')

    def run(case, phase, repetition):
        ranks, mpi, backend = case
        name = f'{phase}_{repetition}_{mpi}_{backend}_{ranks}'
        profile, log = output/(name+'_profile.txt'), output/(name+'.log')
        command = launch.launcher(mpi, ranks)+[launch.binary(mpi), str(args.nx), str(args.new),
                  str(args.boundary), str(args.dt_factor), backend, str(profile)]
        env = launch.environment(mpi, debug=(phase=='warmup' and mpi=='intelmpi'),
                                 intel_provider=args.intel_provider)
        started = time.monotonic()
        with log.open('w') as stream:
            process = subprocess.Popen(command, env=env, cwd=output, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError(f'{name}: exceeded {args.timeout}s; {log}')
        elapsed = time.monotonic()-started
        text = log.read_text()
        if code: raise RuntimeError(f'{name}: exit {code}; {log}')
        if phase=='warmup' and mpi=='intelmpi':
            expected_provider = 'mlx' if args.intel_provider=='mlx' else 'verbs;ofi_rxm'
            if f'libfabric provider: {expected_provider}' not in text:
                raise RuntimeError(f'{name}: Intel MPI provider was not verified')
        match = re.search(r'MARSHAK_RESULT cycles=(\d+) time=(\S+) seconds=(\S+) energy_residual=(\S+) diffusion_l1=(\S+) valid=(\d+)', text)
        if not match: raise RuntimeError(f'{name}: missing result')
        cycles, end, seconds, energy, l1, valid = match.groups()
        cycles, end, seconds, energy, l1 = int(cycles), float(end), float(seconds), float(energy), float(l1)
        if valid != '1' or not all(map(math.isfinite,[end,seconds,energy,l1])) or not math.isclose(end,1e-9,rel_tol=1e-10):
            raise RuntimeError(f'{name}: invalid endpoint/result')
        if abs(energy)>1e-6 or l1<0:
            raise RuntimeError(f'{name}: energy/reference check failed: energy={energy}, L1={l1}')
        rows = [[float(v) for v in line.split()] for line in profile.read_text().splitlines()]
        if len(rows)!=args.nx or any(len(row)!=3 or not all(map(math.isfinite,row)) or row[1]<=0 or row[2]<0 for row in rows):
            raise RuntimeError(f'{name}: invalid profile')
        if any(a[0]>=b[0] for a,b in zip(rows,rows[1:])):
            raise RuntimeError(f'{name}: unordered/duplicate cells')
        # Uniform-grid integral-relative L1, for gas and radiation. The example's
        # legacy mean pointwise ratio overweights the different cold floors.
        interpolated = []
        for row in rows:
            j = max(0, min(len(diffusion)-2, bisect.bisect_right(ref_x,row[0])-1))
            fraction = (row[0]-ref_x[j])/(ref_x[j+1]-ref_x[j])
            interpolated.append([diffusion[j][k]+fraction*(diffusion[j+1][k]-diffusion[j][k]) for k in [1,2]])
        diffusion_errors = [sum(abs(row[k+1]-ref[k]) for row,ref in zip(rows,interpolated))/sum(abs(ref[k]) for ref in interpolated) for k in [0,1]]
        if max(diffusion_errors)>0.10:
            raise RuntimeError(f'{name}: diffusion profile error {diffusion_errors}')
        if ranks not in references:
            references[ranks] = (rows,cycles)
        ref, expected_cycles = references[ranks]
        if cycles!=expected_cycles or any(a[0]!=b[0] for a,b in zip(rows,ref)):
            raise RuntimeError(f'{name}: mesh/cycle mismatch')
        differences = [sum(abs(a[k]-b[k]) for a,b in zip(rows,ref))/sum(abs(b[k]) for b in ref) for k in [1,2]]
        profile_matches = max(differences)<=0.005
        if not profile_matches and not args.record_profile_variation:
            raise RuntimeError(f'{name}: backend profile discrepancy {differences}')
        provider = re.search(r'\[OFI\] component: (.+)', text)
        allowed_providers = ['verbs', 'mlx'] if mpi=='intelmpi' and args.intel_provider=='mlx' else ['verbs']
        provider_name = re.search(r'provider: ([^,)]+)', provider[1]) if provider else None
        if backend=='ofi' and (not provider_name or provider_name[1] not in allowed_providers):
            raise RuntimeError(f'{name}: missing native OFI provider')
        if ('Using P2P' in text)!=(backend=='p2p'):
            raise RuntimeError(f'{name}: unexpected backend selection')
        if 'pending 2 references' in text:
            raise RuntimeError(f'{name}: unfinished receive requests')
        phases = {}
        for key, pattern in {
            'setup':r'MC step max-rank time: setup=(\S+) s',
            'generation':r'MC step max-rank time:.*generation=(\S+) s',
            'loop':r'MC step max-rank time:.*loop=(\S+) s',
            'communication':r'MC loop split max-rank: rma=(\S+) s',
            'handle':r'MC loop split max-rank:.*?handle=(\S+) s',
        }.items():
            samples = list(map(float,re.findall(pattern,text)))
            if len(samples)!=cycles: raise RuntimeError(f'{name}: incomplete phase history')
            phases[key] = sum(samples)
        record = dict(mpi=mpi, backend=backend, ranks=ranks, phase=phase, repetition=repetition,
                      simulation_seconds=seconds, launch_to_exit_seconds=elapsed, cycles=cycles,
                      energy_residual=energy, legacy_pointwise_l1=l1,
                      diffusion_l1=diffusion_errors, profile_relative_l1=differences,
                      profile_matches=profile_matches, physical_checks_passed=True,
                      native_ofi_provider=provider[1] if provider else None,
                      phase_totals=phases, command=command, log=str(log), profile=str(profile),
                      environment={k:v for k,v in env.items() if k.startswith(('I_MPI_','OMPI_MCA_')) or k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','UCX_TLS','FI_PROVIDER','FI_PROVIDER_PATH','BENCHMARK_CPUS','LD_LIBRARY_PATH']})
        records.append(record)
        (output/'runs.json').write_text(json.dumps(records,indent=2)+'\n')
        print(f'{phase} {repetition}: {mpi}+{backend}, {ranks} ranks: {seconds:.3f}s; L1={max(diffusion_errors):.4f}, energy={energy:.2g}, physics PASS; profile delta={max(differences):.4f}'+(' FLAGGED' if not profile_matches else ''),flush=True)

    for case in cases: run(case,'warmup',0)
    rng=random.Random(20260916)
    for repetition in range(1,args.repetitions+1):
        order=list(cases); rng.shuffle(order)
        for case in order: run(case,'measured',repetition)
    summary=[]
    for ranks,mpi,backend in cases:
        selected=[r for r in records if (r['ranks'],r['mpi'],r['backend'],r['phase'])==(ranks,mpi,backend,'measured')]
        row=dict(ranks=ranks,mpi=mpi,backend=backend,samples=len(selected))
        for key in ['simulation_seconds','launch_to_exit_seconds']:
            values=[r[key] for r in selected]
            row[key]=dict(median=statistics.median(values),min=min(values),max=max(values),values=values)
        row['phase_medians']={key:statistics.median(r['phase_totals'][key] for r in selected) for key in selected[0]['phase_totals']}
        row['profile_discrepancies']=sum(not r['profile_matches'] for r in selected)
        row['max_profile_relative_l1']=max(max(r['profile_relative_l1']) for r in selected)
        summary.append(row)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__': main()
