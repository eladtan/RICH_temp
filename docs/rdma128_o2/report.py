from pathlib import Path
import json,re,statistics,hashlib
root=Path('/home/maorm/RICH');work=root/'build/rdma128_o2';out=root/'docs/rdma128_o2';run=work/'compare_10176474'
labels=['baseline1','o3_1','o2_1','o2_2','o3_2','o3_3','o2_3','baseline2'];rows=[]
for name in labels:
 s=(run/(name+'.out')).read_text();m=re.search(r'Finished at t=(\S+) s after (\d+) cycles in (\S+) s',s)
 assert m and int(m[2])==120 and float(m[1])==8.15719e-8,name
 rows.append({'label':name,'total_seconds':float((run/(name+'.elapsed')).read_text()),'cycle_seconds':float(m[3]),'transport_seconds':sum(map(float,re.findall(r'^Loop time: (\S+) seconds',s,re.M))),'sum_rank_average_events':sum(map(float,re.findall(r'avg steps: (\S+)',s)))})
metrics={}
for prefix in ['baseline','o3_','o2_']:
 group=[r for r in rows if r['label'].startswith(prefix)]
 metrics[prefix]={'runs':len(group)}
 for key in ['total_seconds','cycle_seconds','transport_seconds','sum_rank_average_events']:
  a=[r[key] for r in group];metrics[prefix][key]={'median':statistics.median(a),'min':min(a),'max':max(a)}
b=metrics['baseline']['total_seconds']['median'];o3=metrics['o3_']['total_seconds']['median'];o2=metrics['o2_']['total_seconds']['median']
vs_o3=100*(o2/o3-1);vs_base=100*(1-o2/b)
s=(run/'candidate_o2_ledger.out').read_text();res=[float(x) for x in re.findall(r'balance\(inj-rem-stored\)/inj=(\S+)',s)];assert len(res)==24 and 'after 24 cycles' in s
result={'job':10176474,'runs':rows,'metrics':metrics,'o2_runtime_increase_vs_o3_percent':vs_o3,'o2_time_reduction_vs_baseline_percent':vs_base,'conservation_max_abs_normalized_residual':max(map(abs,res))}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Returning optimized STORM to -O2','',f'The optimized `-O2` executable takes **{o2:.3f} s** median, versus **{o3:.3f} s** at `-O3`: a **{vs_o3:+.2f}% runtime change**. It retains a **{vs_base:.2f}% reduction** against the original `{b:.3f} s` baseline. The canonical executable and CMake configuration now use `-O2`, as requested.','',
'## Controlled measurements','',
'Slurm 10176474: eight exclusive nodes `d25g[133-140]`, 128 ranks, 16 ranks/node, one core/rank. Three trials each of the optimized O2 and O3 binaries, interleaved; two original baseline runs bracket them. All timings include the complete `mpirun`, startup, output and shutdown. No profiler.','',
'| Build | Full runtime trials, s | Median, s |','|---|---|---:|']
for prefix,title in [('baseline','Original baseline, O2'),('o3_','Optimized, O3'),('o2_','Optimized, O2')]:
 vals=', '.join(f'{r["total_seconds"]:.3f}' for r in rows if r['label'].startswith(prefix));lines.append(f'| {title} | {vals} | {metrics[prefix]["total_seconds"]["median"]:.3f} |')
lines += ['', 'Same 120-cycle Crooked Pipe deck: `20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi --max-steps 120`. Both optimized builds use `--optimized-rdma`. All finish at 81.5719 ns. This remains a segment benchmark, not a full 1000 ns endpoint measurement.','',
'## What changed','',
'Only the CPU build option\'s forced `-O3` was removed. Standalone release compilation returns to its existing `-O2` flags. `-march=x86-64-v3`, `-ffp-contract=off`, four-face SIMD intersection, attenuation reuse, and the RDMA changes are retained. Runtime source hashes match the measured O3 version. No communication/reallocation code was changed. The actual CMake-generated compile flags are saved in `flags.make`; executable and source hashes are recorded beside this report.','',
'GCC\'s `-O3` does not itself enable `-ffast-math` or `-funsafe-math-optimizations`. Floating-point contraction is controlled separately, so returning to O2 is not a substitute for `-ffp-contract=off`. The earlier architecture-tuned/default-contraction build failed mesh construction; the strict configurations were tested successfully. This does not establish that O3 alone caused the earlier failure. [GCC 15.1 optimization documentation](https://gcc.gnu.org/onlinedocs/gcc-15.1.0/gcc/Optimize-Options.html).','',
'## Validation and review','',
'- CMake rebuilt the application, queue protocol test, and intersection differential test at O2; compiler output has no remaining O3 override.',
'- The O2 CMake-built 800,000-case scalar/vector intersection test passed exact result comparisons.',
'- Native OFI and MPI RMA protocol tests passed saturation, partial sends, wrap, delayed consumption, 12 resize epochs, forbidden-operation guards, registration and legacy reuse.',
f'- The rebuilt O2 application completed the native OFI 24-cycle conservation run with 7-entry initial rings, a 64-entry growth limit, 4-event slices and 1 μs cooperative flush age. Maximum absolute normalized residual: {max(map(abs,res)):.8g}.',
'- All eight production trials completed successfully. This validates the measured configuration; it does not prove correctness for every backend or input.',
'- The preceding ASan/UBSan differential tests were already compiled at O2 with SIMD enabled. They were not rerun because no runtime source changed in this step.','',
'## Current configuration','',
'Keep `STORM_OPTIMIZE_CPU_TRANSPORT=ON` to retain the explicit SIMD/ISA work. It now preserves the configured optimization level instead of overriding it to O3. The current standalone release build uses:','',
'```text','-O2 -march=x86-64-v3 -ffp-contract=off','```','',
'The canonical binary is `/home/maorm/RICH/source/monte/examples/crooked_pipe/crooked_pipe`; its frozen O2 copy is `/home/maorm/RICH/build/rdma128_o2/candidate_o2`. The preceding O3 binary remains at `/home/maorm/RICH/build/rdma128_followup/candidate_final`. Execution nodes must support x86-64-v3. The precise allocation, flags and commands are in [compare.sbatch](compare.sbatch). Source/build change: [changes.patch](changes.patch).']
(out/'results.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(result,indent=2))
