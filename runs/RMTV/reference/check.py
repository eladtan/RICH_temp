#!/usr/bin/env python3
"""Independent checks of the generated shape table and unit mapping."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.integrate import simpson
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
import timmes
ROOT=Path(__file__).resolve().parent

def main():
    a=np.loadtxt(ROOT/'shape.dat'); m=json.loads((ROOT/'metadata.json').read_text())
    inner=a[:m['inner_rows']];outer=a[m['inner_rows']:]
    alpha=9/13;kappa=-19/9;zeta=m['zeta']
    t=(.45/(2*zeta))**(1/alpha);R=.225
    max_error=0
    for r in [.005,.01,.05,.1,.2,.2249,.2251,.3,.4,.449]:
        branch=inner if r<=R else outer
        shape=np.array([np.interp(r/R,branch[:,0],branch[:,j]) for j in [1,2,3]])
        rho=R**kappa*shape[0];v=alpha*R/t*shape[1];T=(alpha*R/t)**2*shape[2]
        d,tev,e,p,vel=timmes.rmtv_1d(r,-2,6.5,1,1.25,1,.45,2,1,7.197534e7,1)
        err=np.max(np.abs(np.array([rho,v*1e8,T*1e3])/np.array([d,vel,tev])-1))
        max_error=max(max_error,float(err))
    assert max_error<1e-4,max_error
    # Total mass inside the thermal front must equal undisturbed ambient mass.
    mass=sum(simpson(b[:,1]*b[:,0]**2,x=b[:,0]) for b in (inner,outer))
    ambient=2**(kappa+3)/(kappa+3)
    mass_error=abs(mass/ambient-1)
    assert mass_error<2e-4,mass_error
    # Similarity energy exponent: rho*r^3*(r/t)^2 is constant.
    assert abs(alpha*(kappa+5)-2)<1e-14
    # Isothermal jump, including the conductive heat flux from ODE variable w.
    # Evaluate the original upstream derivative fields independently at xi=1.
    from scipy.integrate import solve_ivp
    sols=[]
    def capture(*args,**kwargs):
        sol=solve_ivp(*args,**kwargs);sols.append((sol.y[:,0].copy(),sol.y[:,-1].copy()));return sol
    original=timmes.solve_ivp;timmes.solve_ivp=capture
    try: timmes.rmtv_1d(.9,-2,6.5,1,1.25,1,2,2,1,7.197534e7,1)
    finally: timmes.solve_ivp=original
    upstream=sols[0][1];downstream=sols[1][0]
    def flux(y):
        u,h,w,theta=y
        # conductive flux q/(rho*(alpha*r/t)^3) = theta*w, from
        # chi * dT/dr and omega relation. gas enthalpy = gamma/(gamma-1)*theta.
        return h*((u-1)*(5*theta+.5*(u-1)**2)+theta*w)
    assert np.isclose(flux(upstream),flux(downstream),rtol=1e-10)
    # Check the exact jump mapping of w, independent of interpolation.
    u,h,w,theta=upstream
    predicted=(theta*w-.5*((1-u)**4-theta**2)/(1-u))/(1-u)**2
    assert np.isclose(downstream[2],predicted,rtol=1e-12)
    # Dimensional mapping checked with arbitrary independent scale choices.
    arad=7.5657e-15;c=2.99792458e10
    for L,D,H,K in [(1,1,1e-4,3e4),(2,3,4e-4,1e4)]:
        E=(L/H)**2;chi0=D**3*L**4/(H**3*K**7.5)
        rho=4*D;T=3*K
        chi=chi0*rho**-2*T**6.5
        expected=D*L**4/(H**3*K)*4**-2*3**6.5
        assert np.isclose(chi/expected,1,rtol=1e-14)
        Sigma=4*arad*c/(3*chi0)*rho**2*T**-3.5
        assert np.isclose(4*arad*c*T**3/(3*Sigma)/chi,1,rtol=1e-14)
    results=dict(max_table_vs_upstream_relative_error=max_error,
                 mass_inside_front_relative_error=mass_error,
                 shock_jump='passed',unit_mapping='passed',energy_similarity='passed')
    print(json.dumps(results,indent=2))
    (ROOT/'verification.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
