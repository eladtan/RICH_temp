from pathlib import Path
import json,re,statistics,subprocess
root=Path('/home/maorm/RICH')
runs=[root/'build/branson_cp128/run_10176483',root/'build/branson_cp128/run_10176484']
all_results=[]
for folder in runs:
 subprocess.run(['python3',str(root/'docs/branson_cp128/analyze.py'),str(folder)],check=True,stdout=subprocess.DEVNULL)
 for label,row in json.loads((folder/'summary.json').read_text()).items():
  row.update(label=label,job=folder.name,log=str(folder/(label+'.out')))
  all_results.append(row)
assert len(all_results)==6 and all(r['completed'] for r in all_results), 'All six complete runs are required'
summary={'runs':all_results}
for code in ['branson','storm']:
 rows=[r for r in all_results if r['label'].startswith(code)]
 assert len(rows)==3
 summary[code]={key:[r[key] for r in rows] for key in ['elapsed_s','transport_s','generated_histories','transported_histories']}
 for key in list(summary[code]):summary[code]['median_'+key]=statistics.median(summary[code][key])
b,s=summary['branson'],summary['storm']
summary['runtime_ratio']=b['median_elapsed_s']/s['median_elapsed_s']
summary['storm_runtime_reduction_percent']=100*(1-s['median_elapsed_s']/b['median_elapsed_s'])
summary['branson_fewer_generated_percent']=100*(1-b['median_generated_histories']/s['median_generated_histories'])
summary['timing_definition']='Full mpirun invocation, node-local binaries and output; staging/archival excluded'
summary['configuration']=json.loads((root/'build/branson_cp128/configuration.json').read_text())
probe={}
for code in ['branson','storm']:
 curves=[]
 for r in all_results:
  if not r['label'].startswith(code):continue
  if code=='branson':curves.append([float(v) for v in r['final_probe_line'][0].split()[3:]])
  else:
   p=Path(r['log']).with_name(r['label']+'_probes.csv')
   last=p.read_text().splitlines()[-1]
   curves.append([float(v.strip()) for v in last.split(',')[2:]])
 probe[code]={'runs':curves,'mean':[statistics.mean(c[i] for c in curves) for i in range(5)]}
summary['final_probes_keV']=probe
(root/'docs/branson_cp128/results.json').write_text(json.dumps(summary,indent=2)+'\n')
text=f'''# Crooked Pipe: STORM Voronoi versus Branson Cartesian

**Follow-up correction:** Branson's enabled synthetic scattering workload was missed in the initial review. These are as-run timings, not matched-kernel timings. See discrepancy.md for diagnosis.

Completed jobs 10176483 and 10176484, three runs per code on d25g133–140.
Both use 128 ranks, eight exclusive nodes, 16 ranks/node, one core/rank,
strict CPU O2/x86-64-v3/no contraction, and 120 cycles to 81.5718957163 ns.
Times include the full MPI invocation. Binaries and output were node-local;
staging and archival were outside the timer.

| Code | Mesh | Whole-run trials (s) | Median (s) | Median transport loop (s) |
|---|---|---|---:|---:|
| STORM RDMA/OFI, accepted optimized binary | 280606 refined Voronoi cells | {', '.join(f'{x:.3f}' for x in s['elapsed_s'])} | {s['median_elapsed_s']:.3f} | {s['median_transport_s']:.3f} |
| Branson 0.83, history/SoA, MPI particle passing | 280000 Cartesian cells | {', '.join(f'{x:.3f}' for x in b['elapsed_s'])} | {b['median_elapsed_s']:.3f} | {b['median_transport_s']:.3f} |

The Branson/STORM wall-time ratio is {summary['runtime_ratio']:.3f}; STORM's
runtime was {summary['storm_runtime_reduction_percent']:.2f}% lower for these configurations.
Its new measurements reproduce the earlier 65.153-second result to about one percent.

## Actual histories, median across repeats

| Code | Generated, including initial radiation where present | Transported, including carried census on every step |
|---|---:|---:|
| STORM | {s['median_generated_histories']:,} | {s['median_transported_histories']:,} |
| Branson | {b['median_generated_histories']:,} | {b['median_transported_histories']:,} |

Branson generated {summary['branson_fewer_generated_percent']:.2f}% fewer histories.
Its nominal budget is the rounded mean of the earlier measured STORM generation
count, but native source allocation and census policies produce different actual
counts. Histories do not specify equal event work or equal statistical error.

## Accuracy and interpretation

The user chose each code's own mesh. Total cell counts are close, but STORM has
114284 cells in the thin channel and 0.01-cm first wall layers; Branson has 37568
thin-channel cells and spacings 0.0625 x 0.08 x 0.08 cm. The resulting probe
histories differ materially. This experiment does not establish a speedup at
equal spatial accuracy or equal statistical error, nor a speedup from RDMA alone.
STORM retains random walk, census population control, and dynamic balancing.
Branson retains its native history transport and static METIS partitioning.
The initialization and staircase-boundary differences are detailed in review.md.

Mean final material temperatures, keV, after 120 cycles:

| Probe (r,z), cm | STORM | Branson |
|---|---:|---:|
'''
for i,loc in enumerate(['(0,0.25)','(0,2.75)','(1.25,3.5)','(0,4.25)','(0,6.75)']):
 text+=f"| {loc} | {probe['storm']['mean'][i]:.6f} | {probe['branson']['mean'][i]:.6f} |\n"
text+='''
## Validation and artifacts

All six runs exited successfully and completed 120 cycles. Branson's conservation
residuals are included in results.json. The paired STORM binary is byte-identical
to the accepted optimized O2 executable. MPI diagnostics on separate compute
nodes selected UCX and rc_mlx5 inter-node transport with the same environment.
No transport tuning, MPI protocol changes, or RDMA reallocation changes were
made for this comparison.

- `prepare.py`, `build.sh`: reproduce the isolated Branson case and strict O2 build.
- `run_128.sbatch`: stage and run pairs; pass 3 for three trials each.
- `branson_case.patch`, `review.md`: precise changes and review.
- `inputs_sha256.json`, `upstream_commit.txt`: provenance.
- `results.json`: exact timings, counts, and probe values.
- Raw logs: `/home/maorm/RICH/build/branson_cp128/run_10176483/` and
  `/home/maorm/RICH/build/branson_cp128/run_10176484/`.
'''
(root/'docs/branson_cp128/results.md').write_text(text)
print(json.dumps({k:summary[k] for k in ['runtime_ratio','storm_runtime_reduction_percent','branson_fewer_generated_percent']},indent=2))
