from pathlib import Path
import re,json,statistics,sys
folder=Path(sys.argv[1])
def numbers(pattern,s):return [float(x) for x in re.findall(pattern,s)]
results={}
for p in sorted(folder.glob('*.out')):
 if not p.stem.startswith(('branson','storm')):continue
 s=p.read_text();elapsed=p.with_suffix('.elapsed')
 if not elapsed.exists():continue
 r={'elapsed_s':float(elapsed.read_text().strip())}
 if p.stem.startswith('branson'):
  transport=numbers(r'Total transport: ([\d.eE+-]+)',s)
  r['transport_s']=transport[-1] if transport else None
  r['steps']=len(re.findall(r'^Step:',s,re.M))
  active=numbers(r'Total Photons transported: (\d+)',s)
  census=numbers(r'Post census Size: (\d+)',s)
  r['transported_histories']=int(sum(active))
  r['generated_histories']=int(sum(active)-sum(census[:-1]))
  r['final_census']=int(census[-1]) if census else None
  r['radiation_residual_max_jerk']=max(map(abs,numbers(r'Radiation conservation: ([\d.eE+-]+)',s)),default=None)
  r['material_residual_max_jerk']=max(map(abs,numbers(r'Material conservation: ([\d.eE+-]+)',s)),default=None)
  r['final_probe_line']=re.findall(r'^CP_PROBE .+$',s,re.M)[-1:] or None
  r['completed']=bool(transport) and r['steps']==120
 else:
  r['steps']=len(re.findall(r'^Cycle \d+,',s,re.M))
  r['transport_s']=sum(numbers(r'Loop time: ([\d.eE+-]+)',s))
  r['transported_histories']=int(sum(numbers(r'active_after_prestep=(\d+)',s)))
  r['generated_histories']=int(sum(numbers(r'prestep_generated=(\d+)',s)))
  r['completed']='after 120 cycles' in s
 results[p.stem]=r
print(json.dumps(results,indent=2))
(folder/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
