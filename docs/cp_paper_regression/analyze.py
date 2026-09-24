from pathlib import Path
import hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path('/home/maorm/RICH')
OUT = ROOT/'docs/cp_paper_regression'
paths = {
 'Timed original 1': ROOT/'build/rdma128_20260912/final_10176467/baseline1_probes.txt',
 'Timed original 2': ROOT/'build/rdma128_20260912/final_10176467/baseline2_probes.txt',
 'Timed optimized 1': ROOT/'build/branson_cp128/run_10176483/storm1_probes.csv',
 'Timed optimized 2': ROOT/'build/branson_cp128/run_10176484/storm1_probes.csv',
 'Timed optimized 3': ROOT/'build/branson_cp128/run_10176484/storm2_probes.csv',
 'Paper DDMC': Path('/data/shared/maorm/CrookedPipe_fixed/10176427/crookedpipe_probes.csv'),
 'Paper mesh without DDMC': Path('/data/shared/maorm/CrookedPipe_fixed/10176424/crookedpipe_probes.csv'),
}
for label in ['baseline','optimized']:
 p=ROOT/f'build/cp_paper_regression/run_10176491/{label}_probes.csv'
 log=p.with_name(label+'.out')
 if p.exists() and log.exists() and 'Finished at t=' in log.read_text():
  paths['Paper replay '+label]=p
refpath=ROOT/'source/monte/examples/crooked_pipe/data/fig8_gentile.csv'
ref=np.loadtxt(refpath,delimiter=',')
data={k:np.loadtxt(p,delimiter=',') for k,p in paths.items()}
def arrival(t,y):
 idx=np.flatnonzero(y>=.08)
 if not len(idx): return None
 i=idx[0]
 if i==0:return float(t[0])
 return float(10**np.interp(.08,y[i-1:i+1],np.log10(t[i-1:i+1])))
result={}
for k,d in data.items():
 result[k]={'path':str(paths[k]),'sha256':hashlib.sha256(paths[k].read_bytes()).hexdigest(),
 'end_ns':float(d[-1,0]),'arrivals_ns':[arrival(d[:,0],d[:,i+2]) for i in range(5)],
 'temperatures_at_81_5719_ns_keV': [float(np.interp(np.log10(81.5719),np.log10(d[1:,0]),d[1:,i+2])) for i in range(5)] if d[-1,0]>=81.5719 else None}
result['reference_arrivals_ns']=[arrival(ref[:,0],ref[:,i+1]) for i in range(5)]
if 'Paper replay optimized' in data and 'Paper replay baseline' in data:
 a,b=data['Paper replay baseline'],data['Paper replay optimized']
 assert np.array_equal(a[:,:2],b[:,:2])
 result['paired_paper_replay']={'max_abs_difference_keV':np.max(np.abs(a[:,2:]-b[:,2:]),axis=0).tolist(),'rms_difference_keV':np.sqrt(np.mean((a[:,2:]-b[:,2:])**2,axis=0)).tolist()}
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
for i,ax in enumerate(axes.flat):
 if i==5:
  ax.axis('off');ax.text(0,.9,'Same geometry; different numerical cases\n\nPaper: 771,345 cells, DDMC on\nWall layer: 0.0001 cm\nAdded channel points: 400,000\n\nTimed: 280,606 cells, DDMC off\nWall layer: 0.01 cm\nAdded channel points: 0\n\nOriginal and optimized timed runs\nboth show the delayed heating.',va='top',linespacing=1.6);continue
 ax.plot(ref[:,0],ref[:,i+1],color='black',ls='--',label='Gentile (digitized)')
 for k,color,ls in [('Paper DDMC','#15914b','-'),('Timed original 1','#ed8b22',':'),('Timed optimized 1','#1676ba','-')]:
  d=data[k];ax.plot(d[1:,0],d[1:,i+2],color=color,ls=ls,lw=2,label=k)
 for k,color in [('Paper replay baseline','#bb5498'),('Paper replay optimized','#7b34ab')]:
  if k in data:
   d=data[k];ax.plot(d[1:,0],d[1:,i+2],color=color,lw=1.5,ls='-.',label=k)
 ax.set(xscale='log',xlim=(.02,1000),ylim=(.035,.51),title=f'Probe {i+1}',xlabel='Time (ns)',ylabel='Material temperature (keV)');ax.grid(alpha=.2)
 if i==0:ax.legend(fontsize=8,loc='lower right')
fig.suptitle('Crooked Pipe: paper case versus the performance case',fontsize=17)
fig.savefig(OUT/'comparison.png',dpi=170);fig.savefig(OUT/'comparison.pdf')
print(json.dumps(result,indent=2))
