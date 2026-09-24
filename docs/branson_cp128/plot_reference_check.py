from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('/home/maorm/RICH')
paths=[root/'build/branson_cp128/run_10176483/storm1_probes.csv',root/'build/branson_cp128/run_10176484/storm1_probes.csv',root/'build/branson_cp128/run_10176484/storm2_probes.csv']
data=[np.loadtxt(p,delimiter=',',comments='#') for p in paths]
ref=np.loadtxt(root/'source/monte/examples/crooked_pipe/data/fig8_gentile.csv',delimiter=',',comments='#')
t=data[0][:,0]; assert all(np.array_equal(d[:,0],t) for d in data)
values=np.stack([d[:,2:] for d in data]); mean=values.mean(axis=0)
valid=t>0; end=t[-1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
locations=['(0, 0.25)','(0, 2.75)','(1.25, 3.5)','(0, 4.25)','(0, 6.75)']
for k,ax in enumerate(axes.flat):
 if k==5:
  ax.axis('off')
  ax.text(.03,.9,'Comparison details',fontsize=14,weight='bold',transform=ax.transAxes)
  ax.text(.03,.77,'STORM: mean of 3 runs\nShading: min–max across runs\n128 MPI ranks · strict −O2\n120 cycles · ends at 81.57 ns\n\nReference: Gentile (2001), digitized\nfrom Steinberg & Heizler, Fig. 8(a).\nPlot-digitization uncertainty is\napproximately 0.005–0.01 keV.\n\nOnly the simulated time interval\nis shown; no extrapolation.',va='top',linespacing=1.55,transform=ax.transAxes)
  continue
 rv=(ref[:,0]>=max(t[valid][0],ref[0,0]))&(ref[:,0]<=end)
 rt=ref[rv,0]; ry=ref[rv,k+1]
 if ref[0,0]<=end<=ref[-1,0]:
  rt=np.append(rt,end);ry=np.append(ry,np.interp(np.log(end),np.log(ref[:,0]),ref[:,k+1]))
 ax.plot(rt,ry,color='#242424',lw=2.1,ls='--',label='Gentile reference (digitized)')
 ax.fill_between(t[valid],values[:,valid,k].min(axis=0),values[:,valid,k].max(axis=0),color='#0079b8',alpha=.22)
 ax.plot(t[valid],mean[valid,k],color='#0079b8',lw=2.1,label='STORM mean (3 runs)')
 ax.set_xscale('log');ax.set_xlim(.01,end);ax.set_ylim(0,.52)
 ax.set_title(f'Probe {k+1}   (r, z) = {locations[k]} cm',loc='left',fontsize=12,weight='bold')
 ax.set_xlabel('Time (ns, logarithmic scale)');ax.set_ylabel('Material temperature (keV)')
 ax.grid(True,alpha=.18,which='major')
 if k==0:ax.legend(loc='upper left',fontsize=9,frameon=False)
fig.suptitle('Crooked Pipe: timed STORM configuration versus reference',fontsize=18,weight='bold')
out=root/'docs/branson_cp128/figures/crooked_pipe_reference_comparison'
fig.savefig(str(out)+'.png',dpi=170)
fig.savefig(str(out)+'.pdf')
print(str(out)+'.png')
