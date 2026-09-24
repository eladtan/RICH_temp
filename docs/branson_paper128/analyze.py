from pathlib import Path
import re,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/home/maorm/RICH');folder=root/'build/branson_paper128/run_10176492';out=root/'docs/branson_paper128'
s=(folder/'branson.out').read_text()
assert 'Total transport:' in s,'Run has not completed'
rows=[[float(x) for x in line.split()[1:]] for line in re.findall(r'^CP_PROBE .+$',s,re.M)]
b=np.array([[0,0,.05,.05,.05,.05,.05]]+[[r[1],r[0],*r[2:]] for r in rows])
a=np.loadtxt(root/'build/cp_paper_regression/run_10176491/optimized_probes.csv',delimiter=',')
assert len(rows)==63 and b[-1,0]==40
assert np.array_equal(a[:,1],b[:,1]) and np.allclose(a[:,0],b[:,0],rtol=1e-5,atol=1e-6)
np.savetxt(folder/'branson_probes.csv',b,delimiter=',',header='t_ns,cycle,T1_keV,T2_keV,T3_keV,T4_keV,T5_keV')
def vals(pattern):return [float(x) for x in re.findall(pattern,s)]
active=vals(r'Total Photons transported: (\d+)');census=vals(r'Post census Size: (\d+)')
assert len(active)==len(census)==63
elapsed=float((folder/'branson.elapsed').read_text())
def arrival(d,k):
 ix=np.flatnonzero(d[:,k+2]>=.08)
 if not len(ix):return None
 i=ix[0];return float(10**np.interp(.08,d[i-1:i+1,k+2],np.log10(d[i-1:i+1,0])))
r={'job':10176492,'cycles':63,'end_time_ns':40,'total_s':elapsed,'total_s_per_cycle':elapsed/63,'transport_s':vals(r'Total transport: ([\d.eE+-]+)')[-1],'generated_histories':int(sum(active)-sum(census[:-1])),'transported_histories':int(sum(active)),'branson_final_temperatures_keV':b[-1,2:].tolist(),'storm_final_temperatures_keV':a[-1,2:].tolist(),'branson_arrivals_ns':[arrival(b,k) for k in range(5)],'storm_arrivals_ns':[arrival(a,k) for k in range(5)],'radiation_conservation_max_abs_jerk':max(abs(v) for v in vals(r'Radiation conservation: ([\d.eE+-]+)')),'material_conservation_max_abs_jerk':max(abs(v) for v in vals(r'Material conservation: ([\d.eE+-]+)'))}
(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
ref=np.loadtxt(root/'source/monte/examples/crooked_pipe/data/fig8_gentile.csv',delimiter=',')
fig,axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
for k,ax in enumerate(axes.flat):
 if k==5:
  ax.axis('off');ax.text(0,.92,'128 ranks · strict −O2\n63 cycles · endpoint 40 ns\n\nSTORM: paper Voronoi mesh + DDMC\nBranson: 2.24M Cartesian cells, IMC\n13.20M nominal new photons/cycle\n\nMatching physics and time schedule\ndoes not match interface resolution\nor native particle population controls.',va='top',linespacing=1.6);continue
 ax.plot(a[1:,0],a[1:,k+2],label='Optimized STORM, paper settings',color='#147bc2')
 ax.plot(b[1:,0],b[1:,k+2],label='Branson, Cartesian IMC',color='#d36e1b')
 ax.plot(ref[:,0],ref[:,k+1],ls='--',color='black',label='Gentile reference (digitized)')
 ax.set(xscale='log',xlim=(.01,40),ylim=(.035,.5),xlabel='Time (ns)',ylabel='Material temperature (keV)',title=f'Probe {k+1}');ax.grid(alpha=.2)
 if k==0:ax.legend(fontsize=8)
fig.suptitle('Crooked Pipe: matched time schedule, different spatial discretizations',fontsize=15)
fig.savefig(out/'comparison.png',dpi=170);fig.savefig(out/'comparison.pdf')
print(json.dumps(r,indent=2))
