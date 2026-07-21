"""
A wrapper to the Noebauer NLTE radiative shock Lowrie solver - written by M. Krief

The Noebauer solver was taken from:

https://github.com/unoebauer/public-astro-tools/blob/2aae2fc9b9229fdc424126299b7fdc7b1633bf92/radiative_shock/radiative_shock_calculator.py

The theory is based on:
    Lowrie & Rauenzahn 2007, Shock Waves (2007) 16:445–453
            DOI 10.1007/s00193-007-0081-2
    Lowrie & Edwards 2008, Shock Waves (2008) 18:129–143
            DOI 10.1007/s00193-008-0143-0
"""

import numpy as np
from Noebauer_nlte_solver import noeqdiff_shock
from radiative_shock import RadiativeShock, Units

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('NLTERadiativeShock')

class NLTERadiativeShock(RadiativeShock):
    def __init__(
        self, *,
        M0,
        gamma,
        sigma_ross,   # Rosseland coefficient (total macroscopic cross section) [1/cm] as a function of (T, rho) 
        sigma_abs,    # absorption (planck) coefficient (total macroscopic cross section) [1/cm] as a function of (T, rho) 
        cv,           # constant cv erg/g/kelvin
        rho_left,     # downstream (unshocked) density [g/cc]
        v_left,       # downstream (unshocked) velocity [cm/s]
        T_left,       # downstream (unshocked) temperature [K]
        eps_nlte_solver=1e-6, # numerical parameter for Noebauer solver - for some cases it needs to be alrger than 1e-6
        plot_dimensionless=False, # plot the dimensionless profiles
    ):
        super().__init__(
            M0=M0,
            gamma=gamma,
            cv=cv,
            rho_left=rho_left,
            v_left=v_left,
            T_left=T_left,
        )

        logger.info(f"creating a NLTERadiativeShock calculator...")

        assert callable(sigma_ross)
        assert callable(sigma_abs)

        # -------------- dimensional quantities
        self.sigma_ross = sigma_ross
        self.sigma_abs = sigma_abs

        # -------------- dimensionless quantities

        # dimensionless diffusion coefficient
        # NOTE! input T,rho for the dimensionless kappa function are *dimensionless* T/T_left, rho/rho_left
        # also - this kappa definition differs than the LTE one by a factor of P0
        self.kappa = lambda T,rho: Units.clight/(3.*self.sigma_ross(T*self.T_left, rho*self.rho_left)*self.L_tilde*self.cs_left)
        self.sigma_abs_nondim = lambda T,rho: self.sigma_abs(T*self.T_left, rho*self.rho_left)*self.L_tilde*Units.clight/self.cs_left
        
        # set up the Noebauer solver
        self.nlte_solver = noeqdiff_shock(
            M0=self.M0, P0=self.P0, kappa=self.kappa, sigma=self.sigma_abs_nondim, gamma=self.gamma, 
            eps=eps_nlte_solver,
        )

        # make sure Jump condition results are the same in our Hugoniot solver and in Noebauer solver
        assert abs(self.nlte_solver.T1-self.T1) < 1e-7, abs(self.nlte_solver.T1-self.T1)
        assert abs(self.nlte_solver.rho1-self.rho1) < 1e-7
        assert abs(self.nlte_solver.v1-self.v1) < 1e-7
        
        if plot_dimensionless:
            self.nlte_solver.show_profile()

        self.x_profile = self.nlte_solver.shock_profile.x
        self.T_profile_dimensionless = self.nlte_solver.shock_profile.T
        self.Trad_profile_dimensionless = self.nlte_solver.shock_profile.theta
        self.rho_profile_dimensionless = self.nlte_solver.shock_profile.rho
        self.vel_profile_dimensionless = -self.nlte_solver.shock_profile.v

        # get the dimensions back
        self.set_dimensional_profiles()

if __name__ == "__main__":

    # ------------- dimensional examples

    # Gentile LLNL-PRES-789681 
    # Mach45 slide 44/50
    sigma_abs = lambda T,rho: 0.0142*rho*rho*(T/Units.kev_kelvin)**(-3.5)
    sigma_s = lambda T,rho : 0.4006*rho
    sigma_ross = lambda T,rho: sigma_abs(T,rho) + sigma_s(T,rho)
    solver = NLTERadiativeShock(
        M0=45.,
        gamma=5./3.,
        sigma_ross=sigma_ross,
        sigma_abs=sigma_abs,
        cv=1.45e15/Units.kev_kelvin,
        rho_left=1.,
        v_left=0.,
        T_left=0.1 * Units.kev_kelvin,
        plot_dimensionless=True,
        eps_nlte_solver=1e-5,
    ).plot_profiles()

    # Gentile LLNL-PRES-789681 
    # Mach2 slide 45/50
    sigma_abs = lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5)
    solver = NLTERadiativeShock(
        M0=2.,
        gamma=5./3.,
        sigma_ross=sigma_abs,
        sigma_abs=sigma_abs,
        cv=2.22e15/Units.kev_kelvin,
        rho_left=1.,
        v_left=0.,
        T_left=0.122*Units.kev_kelvin,
        plot_dimensionless=True,
        eps_nlte_solver=1e-5,
    ).plot_profiles()

    # Krief changed Gentile Mach2 to Mach5 and higher downstream temperature to get a higher P0 -
    # dynamics where the radiation pressure and energy are not negligible
    sigma_abs = lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5)
    solver = NLTERadiativeShock(
        M0=5.,
        gamma=5./3.,
        sigma_ross=sigma_abs,
        sigma_abs=sigma_abs,
        cv=2.22e15/Units.kev_kelvin,
        rho_left=1.,
        v_left=0.,
        T_left=0.55*Units.kev_kelvin, ### Krief CHANGED IT TO HAVE non-negligible P0 - that works also in NLTE
        plot_dimensionless=True,
        eps_nlte_solver=1e-5,
    ).plot_profiles()

    # Skinner2019 - shock speed is small, downstream is moving
    # has a nice Zel'dovich spike
    Navog = 6.02214076e23
    k_boltz = 1.380649e-16
    gas_constant = k_boltz*Navog
    molar_mass_H = 1.
    solver = NLTERadiativeShock(
        M0=3.,
        gamma=5./3.,
        sigma_ross=lambda T,rho:577.,
        sigma_abs=lambda T,rho:577.,
        cv=gas_constant/(5./3.-1.)/molar_mass_H,
        rho_left=5.69,
        v_left=5.19e7,
        T_left=2.18e6,
        plot_dimensionless=True,
        eps_nlte_solver=1e-6,
    ).plot_profiles()

    # Zhang et al. ApJ 204:7 27pp 2013 section 4.1.5
    # has a nice Zel'dovich spike
    Navog = 6.02214076e23
    k_boltz = 1.380649e-16
    gas_constant = k_boltz*Navog
    molar_mass_H = 1.
    solver = NLTERadiativeShock(
        M0=5.,
        gamma=5./3.,
        sigma_ross=lambda T,rho:0.848903,
        sigma_abs=lambda T,rho:3.92664e-5,
        cv=gas_constant/(5./3.-1.)/molar_mass_H,
        rho_left=5.45969e-13,
        v_left=5.88588e5,
        T_left=100.,
        plot_dimensionless=True,
        eps_nlte_solver=0.5e-4,
    ).plot_profiles()

    pass