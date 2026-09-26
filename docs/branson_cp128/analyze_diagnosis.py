from pathlib import Path
import json,re
folder=Path('/home/maorm/RICH/build/branson_cp128/diagnosis/run_10176489')
result={}
for label in ['coarse_warm','coarse_empty','fine_empty','storm_no_rw']:
 p=folder/(label+'.out');elapsed=folder/(label+'.elapsed')
 if not elapsed.exists():continue
 s=p.read_text();r={'elapsed_s':float(elapsed.read_text().strip()),'log':str(p)}
 if label=='storm_no_rw':
  r['completed']='after 120 cycles' in s
  line=(folder/(label+'_probes.csv')).read_text().splitlines()[-1]
  r['final_probes_keV']=[float(x) for x in line.split(',')[2:]]
  r['transport_s']=sum(float(x) for x in re.findall(r'Loop time: ([\d.eE+-]+)',s))
 else:
  lines=re.findall(r'^CP_PROBE .+$',s,re.M)
  r['completed']=len(lines)==120 and 'Total transport:' in s
  r['final_probes_keV']=[float(x) for x in lines[-1].split()[3:]]
  r['transport_s']=float(re.findall(r'Total transport: ([\d.eE+-]+)',s)[-1])
  for field,pattern in [('radiation_residual_max_jerk','Radiation conservation'),('material_residual_max_jerk','Material conservation')]:
   r[field]=max(abs(float(x)) for x in re.findall(pattern+r': ([\d.eE+-]+)',s))
 result[label]=r
(folder/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
