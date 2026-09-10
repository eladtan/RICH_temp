#!/usr/bin/env python3
"""Evaluate the radiation/conduction mapping on the reference, before a run.

Kn is lambda / (T/|dT/dr|). The exact sharp heat front is excluded explicitly;
no finite-speed radiation model has the same singular front pointwise.
"""
from pathlib import Path
import argparse
import json
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--time-unit',type=float,default=0.0256)
    p.add_argument('--temperature-unit',type=float,default=937.5)
    args=p.parse_args();H=args.time_unit;K=args.temperature_unit
    if H<=0 or K<=0:raise ValueError('Positive scales required')
    a=np.loadtxt(ROOT/'shape.dat');m=json.loads((ROOT/'metadata.json').read_text())
    branches=(a[1:m['inner_rows']],a[m['inner_rows']:])
    c=2.99792458e10;arad=4*5.670374419e-5/c;Cv=4/(H*H*K)
    chi0=1/(H**3*K**7.5);out=[]
    for rf in [.45,.9]:
        age=H*(rf/(2*m['zeta']))**(13/9);R=rf/2;vs=(9/13)*R/(age/H)
        heat=[];kn=[];beta=[];reduced_flux=[];relax=[]
        for b in branches:
            r=R*b[:,0];rho=R**(-19/9)*b[:,1];T=K*vs**2*b[:,3]
            v=vs/H*b[:,2];Sigma=4*arad*c/(3*chi0)*rho*rho*T**-3.5
            gradient=np.gradient(T,r)
            mask=(b[:,0]>=.02)&(b[:,0]<=1.99)
            heat.extend((4*arad*T**3/(rho*Cv))[mask])
            kn.extend((abs(gradient)/(Sigma*T))[mask])
            beta.extend((abs(v)/c)[mask])
            reduced_flux.extend((4*abs(gradient)/(3*Sigma*T))[mask])
            relax.extend((1/(c*Sigma*(1+4*arad*T**3/(rho*Cv)))/age)[mask])
        out.append(dict(heat_front=rf,age=age,max_radiation_heat_capacity_ratio=max(heat),
                        max_v_over_c=max(beta),max_Kn_excluding_front=max(kn),
                        max_F_over_cEr_excluding_front=max(reduced_flux),
                        max_equilibration_time_over_age=max(relax)))
    print(json.dumps(dict(time_unit=H,temperature_unit=K,
                         sampled_xi_interval=[.02,1.99],profiles=out),indent=2))
if __name__=='__main__':main()
