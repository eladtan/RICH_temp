#!/usr/bin/env python3
"""Compare output to the same conservative cell-averaged RMTV reference.

No automatic physics pass threshold is invented here. --smoke asserts basic
integrity/conservation, while the JSON records actual profile errors.
"""
from pathlib import Path
import argparse
import json
import io
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
    p.add_argument('--label',default='final');p.add_argument('--smoke',action='store_true')
    p.add_argument('--require-complete',action='store_true');p.add_argument('--plot',action='store_true');args=p.parse_args();d=args.directory
    files=sorted(d.glob(args.label+'_rank*.csv'))
    if not files:raise RuntimeError('No snapshot files')
    a=np.concatenate([np.atleast_1d(np.genfromtxt(f,delimiter=',',names=True,skip_header=1)) for f in files])
    config=json.loads((d/'config.json').read_text())
    assert len(files)==config['ranks'] and len(a)==config['n']**3
    for name in a.dtype.names:assert np.all(np.isfinite(a[name])),name
    V=a['volume']; r=np.sqrt(a['x']**2+a['y']**2+a['z']**2)
    def error(value,ref):return float(np.sum(V*np.abs(value-ref))/max(np.sum(V*np.abs(ref)),1e-300))
    vr=(a['vx']*a['x']+a['vy']*a['y']+a['vz']*a['z'])/r
    vr_ref=(a['vx_ref']*a['x']+a['vy_ref']*a['y']+a['vz_ref']*a['z'])/r
    transverse=np.maximum(0,a['vx']**2+a['vy']**2+a['vz']**2-vr**2)
    report={f'L1_{n}':error(a[n],a[n+'_ref']) for n in ['rho','T','p','e']}
    report['snapshot']=args.label
    report['L1_radial_velocity']=error(vr,vr_ref)
    reference_transverse=np.maximum(0,a['vx_ref']**2+a['vy_ref']**2+a['vz_ref']**2-vr_ref**2)
    report['reference_transverse_to_radial_rms']=float(np.sqrt(np.sum(V*reference_transverse)/max(np.sum(V*vr_ref**2),1e-300)))
    report['transverse_to_radial_rms']=float(np.sqrt(np.sum(V*transverse)/max(np.sum(V*vr**2),1e-300)))
    snapshot_age=float(files[0].read_text().splitlines()[0].split('=')[1])
    lines=(d/'diagnostics.csv').read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):lines.pop()
    diag=np.atleast_1d(np.genfromtxt(io.StringIO(''.join(lines)),delimiter=',',names=True))
    diag=diag[diag['time']<=snapshot_age*(1+1e-13)]
    if not len(diag):raise RuntimeError('No complete diagnostics yet for this snapshot')
    report['max_energy_drift']=float(np.max(abs(diag['energy_drift'])))
    report['max_mass_drift']=float(np.max(abs(diag['mass_drift'])))
    report['max_radiation_heat_capacity_ratio']=float(np.max(diag['max_radiation_heat_capacity_ratio']))
    report['max_boundary_T']=float(np.max(diag['max_boundary_T']))
    report['initial_floor_energy_fraction']=float(diag['initial_floor_energy'][0]/diag['gas_energy'][0])
    report['physical_age_s']=snapshot_age
    report['octant']=config.get('octant',False)
    dx=(1 if report['octant'] else 2)*config['box']/config['n']
    edges=np.arange(0,config['box']+dx*.5,dx)
    centers=.5*(edges[:-1]+edges[1:])
    weights=np.histogram(r,bins=edges,weights=V)[0]
    def shells(values):
        return np.divide(np.histogram(r,bins=edges,weights=V*values)[0],weights,
                         out=np.full(len(weights),np.nan),where=weights>0)
    density_shell=shells(a['rho']);temperature_shell=shells(a['T'])
    report['shock_radius_shell_peak']=float(centers[np.nanargmax(density_shell)])
    report['shock_radius_reference_shell_peak']=float(centers[np.nanargmax(shells(a['rho_ref']))])
    report['heat_front_thresholds']={}
    for fraction in [.01,.05,.1]:
        def front(values):
            found=np.flatnonzero(values>=fraction*np.nanmax(values))
            return float(edges[found[-1]+1]) if len(found) else None
        report['heat_front_thresholds'][str(fraction)]={
            'simulation':front(temperature_shell),'averaged_reference':front(shells(a['T_ref']))}
    rf=config['rf_start']*(report['physical_age_s']/config['t_start'])**(9/13)
    report['exact_heat_front']=rf;report['exact_shock']=.5*rf

    if 'ddmc_steps' in diag.dtype.names:report['ddmc_steps']=float(np.sum(diag['ddmc_steps']))
    if args.label=='final' and (d/'status.json').exists():report.update(json.loads((d/'status.json').read_text()))
    if args.require_complete:assert report.get('reached_end',False),'Run did not reach the requested end time'
    if args.smoke:
        assert np.all(a['rho']>0) and np.all(a['T']>0) and np.all(a['Er']>=0)
        assert report['max_mass_drift']<1e-9,report
        assert report['max_energy_drift']<1e-3,report
        if len(diag)>1:assert np.all(np.diff(diag['time'])>0)
    encoded=json.dumps(report,indent=2)+'\n'
    print(encoded,end='');(d/'analysis.json').write_text(encoded)
    (d/('analysis_'+args.label+'.json')).write_text(encoded)
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,1,figsize=(7,9),sharex=True)
        for ax,(field,value,exact) in zip(axes,[('Density',a['rho'],a['rho_ref']),('Temperature [K]',a['T'],a['T_ref']),('Radial velocity [cm/s]',vr,vr_ref)]):
            order=np.argsort(r);ax.scatter(r,value,s=2,alpha=.15,label='Simulation')
            ax.scatter(r[order],exact[order],s=1,color='black',label='Cell-averaged reference')
            ax.set_ylabel(field);ax.legend()
        axes[-1].set_xlabel('Radius [cm]');axes[-1].set_xlim(0,config['box'])
        fig.suptitle(f"RMTV {args.label}, age = {snapshot_age:.6g} s")
        fig.tight_layout();fig.savefig(d/'profiles.png',dpi=160)
        fig.savefig(d/('profiles_'+args.label+'.png'),dpi=160)
if __name__=='__main__':main()
