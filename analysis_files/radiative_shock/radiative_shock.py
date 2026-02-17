"""
A base class radiative shock solver
It handles the calculation of jump condition and interpolates the solution
on a given grid.

The theory is based on:
    Lowrie & Rauenzahn 2007, Shock Waves (2007) 16:445–453
            DOI 10.1007/s00193-007-0081-2
    Lowrie & Edwards 2008, Shock Waves (2008) 18:129–143
            DOI 10.1007/s00193-008-0143-0
"""
import sys
import numpy as np
import scipy.optimize
from scipy.interpolate import Akima1DInterpolator as akima

class Units:
    sigma_sb = 5.670374419e-5
    clight = 2.99792458e10
    arad = 4. * sigma_sb / clight 
    ev_kelvin = 1.160451812e4
    hev_kelvin = 100. * ev_kelvin
    kev_kelvin = 1000. * ev_kelvin

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('RadiativeShock')

class RadiativeShock():
    """
    Base class for a Radiative Shock (LTE and NLTE)
    Given the downstream (unshocked) values and the Mach number
    it calculates the jump condition to get the downstream values.
    In addition, this class handles the transformation from dimensional <-> nondimensional
    quantities that are used in the analytic solution By
    """
    def __init__(
        self, *,
        M0,         # downstream Mach number
        gamma,      # ideal gas adiabatic index
        cv,         # constant cv erg/g/kelvin
        rho_left,   # downstream (unshocked) density [g/cc]
        v_left,     # downstream (unshocked) velocity [cm/s]
        T_left,     # downstream (unshocked) temperature [K]
    ):

        logger.info(f"creating a RadiativeShock calculator...")

        self.gamma = gamma
        self.M0 = M0

        # -------------- dimensional quantities
        self.L_tilde = 1. # we assume here length scale L_tilde = 1cm - we work in c.g.s exclusivly
        self.cv = cv
        self.v_left = v_left
        self.rho_left = rho_left
        self.T_left = T_left
        self.sie_left = self.cv * self.T_left
        self.p_left = (self.gamma-1.)*self.rho_left*self.sie_left + 1./3.*Units.arad*self.T_left**4 
        
        # note - this is the *material* sound speed so we use *material* sie and not the total pressure (p_left above, which includes the radiation pressure)
        self.cs_left = (self.gamma*(self.gamma-1.)*self.sie_left)**0.5

        # -------------- dimensionless quantities
        # Lowrie2007 equations 2-3
        self.P0 = Units.arad*self.T_left **4 / (self.rho_left*self.cs_left**2) 

        # solve the overal dimensionless jump condition of the shock
        self.rho1, self.T1, self.v1 = RadiativeShock.shock_jump_dimensionless(gamma=self.gamma, M0=self.M0, P0=self.P0)

        # -------------- dimensional downstream quantities
        self.rho_right = self.rho_left * self.rho1
        self.T_right = self.T_left * self.T1
        self.sie_right = self.cv * self.T_right
        self.p_right = (gamma-1.)*self.rho_right*self.sie_right + 1./3.*Units.arad*self.T_right**4 
        self.Mdot = np.sqrt((self.p_right-self.p_left)/(1./self.rho_left-1./self.rho_right)) # shock mass flux
        
        self.v_shock = self.v_left - 1./self.rho_left * self.Mdot
        self.v_right = self.v_shock + self.Mdot/self.rho_right

        logger.info(f"gamma={self.gamma:g}")
        logger.info(f"M0={self.M0:g}")
        logger.info(f"P0={self.P0:g}")
        logger.info(f"downstream (unshocked, left) state: rho={self.rho_left:.7g}[g/cc] T={self.T_left:.7g}[K]={self.T_left/Units.kev_kelvin:.7g}[keV] v={self.v_left:.7g}[cm/s]")
        logger.info(f"upstream   (shocked, right)  state: rho={self.rho_right:.7g}[g/cc] T={self.T_right:.7g}[K]={self.T_right/Units.kev_kelvin:.7g}[keV] v={self.v_right:.7g}[cm/s]")
        logger.info(f"v_shock={self.v_shock:g}[cm/s]")
        
        logger.info(f"compression {self.rho_right/self.rho_left:g}")
        logger.info(f"matter ideal gas (gamma={self.gamma:g}) dominated compression {(self.p_right*(self.gamma+1)+self.p_left*(self.gamma-1.))/(self.p_right*(self.gamma-1)+self.p_left*(self.gamma+1)):g}")
        gamma_rad = 4./3.
        logger.info(f"radiation ideal gas (gamma=4/3) dominated compression {(self.p_right*(gamma_rad+1)+self.p_left*(gamma_rad-1.))/(self.p_right*(gamma_rad-1)+self.p_left*(gamma_rad+1)):g}")

        # check that the given Mach number matches the calculated shock jump
        M0_calc = (self.v_left-self.v_shock)/self.cs_left
        logger.info(f"calculated mach=(v_shock-v_left)/cs_left={M0_calc:g} while given mach M0={self.M0:g} err={abs(M0_calc/self.M0-1.):g}")
        assert abs(M0_calc/self.M0-1.) < 1e-12, abs(M0_calc/self.M0-1.)

        self.title = f"$\\mathcal{{M}}_0={self.M0:g}, \\ \\  \\mathcal{{P}}_0={self.P0:g}$"

        self.dimensionless_profile = dict()
        

    @staticmethod
    def shock_jump_dimensionless(*, gamma, M0, P0):
        """
        Compute the overal shock jump conditions for a radiative shock
        relative to the dimensionless downstream values:
        rho0 = 1, and T0 = 1.
        """
        f1 = lambda T1: 3.0*(gamma + 1.0)*(T1 - 1.0) - P0*gamma*(gamma - 1.0)*(7.0 + T1**4)
        f2 = lambda T1:  12.0 * (gamma - 1.0)**2 * T1 * (3.0 + gamma*P0*(1.0 + 7.0*T1**4))
        rho_func = lambda T1:  (f1(T1) + np.sqrt(f1(T1)**2 + f2(T1))) / (6.0*(gamma - 1.0)*T1)

        def func(T1):
            rho1 = rho_func(T1)
            lhs = 3.0*rho1*(rho1*T1 - 1.0) + gamma*P0*rho1*(T1**4 - 1.0)
            rhs = 3.0*gamma*(rho1 - 1.0) * M0*M0
            return lhs - rhs

        T1_guess = M0*M0
        T1 = scipy.optimize.newton(func, T1_guess, tol=1e-10, rtol=1e-15)
        rho1 = rho_func(T1)
        v1 = M0 / rho1
        return rho1, T1, v1

    def set_dimensional_profiles(self):
        assert not np.isnan(self.T_profile_dimensionless).any()
        assert not np.isnan(self.Trad_profile_dimensionless).any()

        # remove duplicates of x
        inds_eq = [i for i in range(1, len(self.x_profile)) if np.abs(self.x_profile[i]-self.x_profile[i-1])<1e-50]
        self.x_profile = np.delete(self.x_profile, inds_eq)
        self.rho_profile_dimensionless = np.delete(self.rho_profile_dimensionless, inds_eq)
        self.T_profile_dimensionless = np.delete(self.T_profile_dimensionless, inds_eq)
        self.Trad_profile_dimensionless = np.delete(self.Trad_profile_dimensionless, inds_eq)
        self.vel_profile_dimensionless = np.delete(self.vel_profile_dimensionless, inds_eq)

        # get dimensional profiles
        self.x_profile_dimensionless = np.copy(self.x_profile)
        self.x_profile *= self.L_tilde
        self.rho_profile = self.rho_left * self.rho_profile_dimensionless
        self.T_profile = self.T_left * self.T_profile_dimensionless
        self.Trad_profile = self.T_left * self.Trad_profile_dimensionless
        self.vel_profile = self.v_shock - self.cs_left*self.vel_profile_dimensionless

        # --- set interpolating functions
        x1, x2 = self.x_profile[0], self.x_profile[-1]
        shock_interp = np.vectorize(lambda x, data_left, data_right, interpolator : data_left if x<x1 else interpolator(x) if x1<=x<=x2 else data_right)
        
        rho_akima = akima(self.x_profile, self.rho_profile)
        Tmat_akima = akima(self.x_profile, self.T_profile)
        Trad_akima = akima(self.x_profile, self.Trad_profile)
        vel_akima = akima(self.x_profile, self.vel_profile)

        self.rho_interp = lambda x: shock_interp(x, self.rho_left, self.rho_right, rho_akima)
        self.Tmat_interp = lambda x: shock_interp(x, self.T_left, self.T_right, Tmat_akima)
        self.Trad_interp = lambda x: shock_interp(x, self.T_left, self.T_right, Trad_akima)
        self.vel_interp = lambda x: shock_interp(x, self.v_left, self.v_right, vel_akima)

    def solve_profiles(self, *, time, x):
        """
        returns the profiles at a given `time` [sec] (assuming the shock front is at the origin at t=0)
        on the given spatial grid `x` [cm]
        """
        assert time >= 0.
        assert len(x) > 1.
        x0 = x - self.v_shock * time
        return dict(
            temperature=self.Tmat_interp(x0),
            radiation_temperature=self.Trad_interp(x0),
            density=self.rho_interp(x0),
            velocity=self.vel_interp(x0),
        )

    def plot_profiles(self):

        from matplotlib import pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({'font.size': 13})

        # dimensional profiles
        # Tunits, Tunits_label = Units.kev_kelvin, "KeV"
        Tunits, Tunits_label = 1., "K"

        
        # add a plot of the wave after it has traveled some distance
        time = self.x_profile[0]*0.2/self.v_shock
        x = np.asfarray(sorted(
            # shift original solver grid to get good spike resolution at its new position
            list(self.x_profile+self.v_shock*time)+\
            # add more spatial extent since the wave is traveling
            list(np.linspace(self.x_profile[0]*1.3,self.x_profile[-1]*1.3, 4000) 
        )))

        # interpolated solution at the shifted time
        solution = self.solve_profiles(time=time, x=x)

        plt.figure("T")
        plt.plot(self.x_profile, self.T_profile/Tunits, c="r", ls="-", label="$T_{{m}}(x)$ t=0")
        plt.plot(self.x_profile, self.Trad_profile/Tunits, c="b", ls="--", label="$T_{{r}}(x)$ t=0")
        plt.plot(x, solution["temperature"]/Tunits, c="r", ls="-.", label=f"$T_{{m}}(x)$ t={time/1e-9:.3g}ns")
        plt.plot(x, solution["radiation_temperature"]/Tunits, c="b", ls=":", label=f"$T_{{r}}(x)$ t={time/1e-9:.3g}ns")
        plt.axhline(y=self.T_left/Tunits, c="k", ls="--", label=f"$T_{{\\mathrm{{left}}}}={self.T_left/Tunits:g}$ {Tunits_label}")
        plt.axhline(y=self.T_right/Tunits, c="k", ls="--", label=f"$T_{{\\mathrm{{right}}}}={self.T_right/Tunits:g}$ {Tunits_label}")
        plt.grid()
        plt.legend()
        plt.xlabel("$x$ [cm]")
        plt.ylabel(f"$T(x,t)$ [{Tunits_label}]")
        plt.title(self.title)

        plt.figure("rho")
        plt.plot(self.x_profile, self.rho_profile, c="r", ls="-", label=f"$\\rho(x)$ t=0")
        plt.plot(x, solution["density"], c="r", ls="-.",  label=f"$\\rho(x)$ t={time/1e-9:.3g}ns")
        plt.axhline(y=self.rho_left, c="k", ls="--", label=f"$\\rho_{{\\mathrm{{left}}}}={self.rho_left:g}$")
        plt.axhline(y=self.rho_right, c="k", ls="--", label=f"$\\rho_{{\\mathrm{{right}}}}={self.rho_right:g}$")
        
        # for LTE solver
        if hasattr(self, "rhop"):
            plt.scatter([0.], [self.rhop], c="b", label=f"$\\rho_p={self.rhop:g}$")

        plt.grid()
        plt.legend()
        plt.xlabel("$x$ [cm]")
        plt.ylabel("$\\rho(x,t)$ [g/cc]")
        plt.title(self.title)

        plt.figure("vel")
        plt.plot(self.x_profile, self.vel_profile, c="r", ls="-", label=f"$v(x)$ t=0")
        plt.plot(x, solution["velocity"], c="r", ls="-.",  label=f"$v(x)$ t={time/1e-9:.3g}ns")
        plt.axhline(y=self.v_left, c="b", ls="--", label=f"$v_{{\\mathrm{{left}}}}={self.v_left:g}$")
        plt.axhline(y=self.v_right, c="k", ls="--", label=f"$v_{{\\mathrm{{right}}}}={self.v_right:g}$")

        # for LTE solver
        if hasattr(self, "velp"):
            plt.scatter([0.], [self.velp], c="b", label=f"$v_p={self.velp:g}$")

        plt.grid()
        plt.legend()
        plt.xlabel("$x$ [cm]")
        plt.ylabel("$v(x,t)$ [cm/s]")
        plt.title(self.title)
        
        plt.show()

        return self