from pathlib import Path
import json,re,statistics,subprocess,hashlib
work=Path('/home/maorm/RICH/build/rdma128_followup');out=Path('/home/maorm/RICH/docs/rdma128_followup');run=work/'final_10176473'
result=json.loads(subprocess.check_output(['python3',str(work/'summarize.py'),str(run)],text=True))
assert len(result['runs'])==6,'final comparison incomplete'
assert all(r['cycles']==120 and r['final_time']==8.15719e-8 for r in result['runs'])
for row in result['runs']:
 s=(run/(row['label']+'.out')).read_text()
 row['sum_rank_average_events']=sum(float(x) for x in re.findall(r'avg steps: (\S+)',s))
 row['sum_rank_maximum_events']=sum(float(x) for x in re.findall(r'max steps: (\S+),',s))
ledgers={}
for name in ['candidate_final','vector_debug']:
 s=(run/(name+'_ledger.out')).read_text();balances=[float(x) for x in re.findall(r'balance\(inj-rem-stored\)/inj=(\S+)',s)]
 assert len(balances)==24 and 'after 24 cycles' in s
 ledgers[name]={'cycles':24,'max_abs_normalized_residual':max(map(abs,balances))}
result['ledgers']=ledgers
out.joinpath('results.json').write_text(json.dumps(result,indent=2)+'\n')
metric=result['total_seconds'];b=metric['baseline_median'];c=metric['candidate_median'];pct=metric['time_reduction_percent']
lines=['# Crooked Pipe CPU/RDMA follow-up: measured results','',f'The final candidate reduces total launcher wall time by **{pct:.2f}%**: median **{b:.3f} s → {c:.3f} s** ({b/c:.3f}× speedup). This meets the 15–20% time-reduction target on the tested segment. The additional gain comes primarily from CPU intersection/arithmetic and compiler optimization; it is not an RDMA-only speedup.','',
'## Final controlled comparison','',
'Slurm **10176473**, eight exclusive nodes `d25g[133-140]`, 128 MPI ranks, 16 ranks/node, one core/rank. Three runs per binary, interleaved in one allocation. Identical mesh, photon budgets, seed policy, timestep controls and 120-cycle endpoint. Bash times the complete `mpirun`, including setup, output and shutdown. No profiler is loaded in timing runs.','',
'| Run | Original baseline, s | Final CPU + RDMA candidate, s |','|---|---:|---:|']
rows={r['label']:r for r in result['runs']}
for i in range(1,4):lines.append(f'| {i} | {rows[f"baseline{i}"]["total_seconds"]:.3f} | {rows[f"candidate{i}"]["total_seconds"]:.3f} |')
lines += [f'| Median | **{b:.3f}** | **{c:.3f}** |','',
'| Secondary metric | Baseline median, s | Candidate median, s | Reduction |','|---|---:|---:|---:|']
for key,name in [('cycle_seconds','Application cycle interval'),('transport_seconds','Sum of max-rank transport-loop times')]:
 d=result[key];lines.append(f'| {name} | {d["baseline_median"]:.3f} | {d["candidate_median"]:.3f} | {d["time_reduction_percent"]:.2f}% |')
bwork=statistics.median(r['sum_rank_average_events'] for r in result['runs'] if r['label'].startswith('baseline'))
cwork=statistics.median(r['sum_rank_average_events'] for r in result['runs'] if r['label'].startswith('candidate'))
lines += ['',f'The summed rank-average event count changes by {100*(cwork/bwork-1):+.2f}%. This is a work-count check, not a physics-equivalence proof. All runs complete 120 cycles at **81.5719 ns**. The full 1000 ns endpoint was not benchmarked.','',
'## What was tried','',
'- Earlier queue/scheduling-only work: 2.59% total-time reduction in the preceding final comparison (10176467); opt-in retained.',
'- All-rank profile (10176469): face intersection is 23.55% of baseline CPU samples, including startup/polling CPU time. This motivates the CPU work; it is not a wall-time attribution.',
'- Unrestricted `-O3 -march=x86-64-v3` build (10176470): failed during mesh construction and rejected.',
'- Strict `-O3 -march=x86-64-v3 -ffp-contract=off` plus exponential reuse (10176471): baseline 79.609/79.460 s; optimized 73.259/72.008 s; default RDMA scheduling with the same CPU build 73.602 s (single ablation).',
'- Strict compiler-only and SIMD trial (10176472): strict compiler-only 73.680 s; SIMD + exponential reuse 66.213/65.189 s. All three use the earlier RDMA opt-in. The final CMake-built executable is measured separately above.','',
'## Correctness gates','',
'- Final native OFI and MPI RMA tests passed: full rings, partial sends, wrap, delayed consumer, 12 grow/shrink epochs, lifecycle guards, send registration and legacy reuse.',
'- Final release and STORM_DEBUG/assertion builds each completed the 24-cycle native OFI conservation case with 7-entry initial rings, a 64-entry growth limit, 4-event slices and 1 μs cooperative flush age.',
f'- Maximum absolute normalized energy residual was {max(v["max_abs_normalized_residual"] for v in ledgers.values()):.8g}.',
'- 800,000 scalar/vector intersections and 300,000 full event differential cases passed exact comparisons; standalone ASan/UBSan runs also passed.',
'- All manager/reallocation sources match their previous reviewed hashes. CPU optimization adds no geometry allocation, remote descriptor or packet-layout change. See [review.md](review.md) for invariants and test limits.','',
'## Reproduce','',
'Configure the existing CPU/MPI build with `-DSTORM_OPTIMIZE_CPU_TRANSPORT=ON`, then build `crooked_pipe`. Execution nodes must support x86-64-v3. The option adds release O3 and AVX2 with floating-point contraction disabled; it does not enable fast math. The CPU option remains off by default for portability.','',
'```bash',
'mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 \\',
'  /home/maorm/RICH/source/monte/examples/crooked_pipe/crooked_pipe \\',
'  20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi \\',
'  --max-steps 120 --optimized-rdma --output-probes probes.txt','```','',
'Use the allocation/environment in [final.sbatch](final.sbatch). The canonical executable was rebuilt with that CPU option enabled. Frozen executable and logs: `/home/maorm/RICH/build/rdma128_followup/`. Original pre-task working-source baseline: `/home/maorm/RICH/build/rdma128_20260912/baseline`. Build/source hashes and scripts are saved beside this report.','',
'GPU, CXI, full-endpoint performance, and ensemble time-to-accuracy are outside this measurement. Finite testing does not establish perfect correctness for every configuration.']
out.joinpath('results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result,indent=2))
