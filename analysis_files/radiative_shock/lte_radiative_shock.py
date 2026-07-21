"""
An LTE radiative shock solver - written by M. Krief
The theory is based on:
    Robert B. Lowrie · Rick M. Rauenzahn
    Shock Waves (2007) 16:445–453
    DOI 10.1007/s00193-007-0081-2
"""
import sys
import numpy as np
import scipy.optimize
import scipy.integrate

from radiative_shock import RadiativeShock, Units

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('LTERadiativeShock')

class LTERadiativeShock(RadiativeShock):
    def __init__(
        self, *,
        M0,
        gamma,
        sigma_ross, # Rosseland coefficient (total macroscopic cross section) [1/cm] as a function of (T, rho) 
        cv,         # constant cv erg/g/kelvin
        rho_left,   # downstream (unshocked) density [g/cc]
        v_left,     # downstream (unshocked) velocity [cm/s]
        T_left,     # downstream (unshocked) temperature [K]
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

        logger.info(f"creating a LTERadiativeShock calculator...")

        assert callable(sigma_ross)
        self.sigma_ross = sigma_ross

        # dimensionless diffusion coefficient
        # NOTE! input T,rho for the dimensionless kappa function are *dimensionless* T/T_left, rho/rho_left
        self.kappa = lambda T,rho: Units.clight*self.P0/(3.*self.sigma_ross(T*self.T_left, rho*self.rho_left)*self.L_tilde*self.cs_left)
        
        # solve the precursor (or isothermal shock) dimensionless profiles
        self.rhop_dimensionless, self.vp_dimensionless, self.x_profile, self.T_profile_dimensionless, self.rho_profile_dimensionless, self.vel_profile_dimensionless, self.isothermal = LTERadiativeShock.solve_dimensionless_profiles(
            M0=self.M0,
            gamma=self.gamma,
            P0=self.P0,
            kappa=self.kappa,
            rho1=self.rho1,
            T1=self.T1,
            v1=self.v1,
            plot=plot_dimensionless,
        )

        self.Trad_profile_dimensionless = np.copy(self.T_profile_dimensionless) #in LTE radiation and matter temperatures are equal

        # get the dimensions back
        self.set_dimensional_profiles()

        # dimensional quantities
        self.rhop = self.rho_left*self.rhop_dimensionless
        self.velp = self.v_shock - self.cs_left*self.vp_dimensionless

        self.title = "LTE " + self.title
        if self.isothermal: self.title = "isothermal " + self.title
        
    @staticmethod
    def solve_dimensionless_profiles(*, gamma, M0, P0, kappa, rho1, T1, v1, plot=False):

        # Eq. 26 
        m_T_fun = lambda T: 0.5*(gamma*M0*M0+1.)+gamma*P0/6.*(1.-T*T*T*T)

        # Eq. 25
        def rho_T_fun(T):
            m = m_T_fun(T)
            return (m-np.sqrt(m*m-gamma*T*M0*M0))/T
            
        # Eq. 24 - the ODE to be solver numerically
        def dTdx_fun(x, T): 
            rho = rho_T_fun(T)
            rho2 = rho*rho
            f3 = 6.*rho2*(T-1.)/(gamma-1.)+3.*(1.-rho2)*M0*M0+8.*P0*(T*T*T*T-rho)*rho
            return M0*f3/(24.*kappa(T, rho)*rho2*T*T*T)

        ode_scheme = "lsoda"
        assert ode_scheme in {"dopri5", "dop853", "lsoda", "zvode", "vode"}
        ode_solver = scipy.integrate.ode(dTdx_fun).set_integrator(ode_scheme)

        logger.info("Integrating precursor region...")

        # precursor density
        rhop = rho_T_fun(T1)
        
        isothermal = abs(rhop/rho1-1.) < 1e-10
        logger.info(f"rhop={rhop:g} rho1={rho1:g} T={T1:g}")
        if isothermal:
            logger.info("Radiative Shock is isothermal - all profiles are continuous!")
        else:
            logger.info("Radiative Shock is not isothermal - will have jumps in density and velocity")
        
        # guess the scale of dx on which to solve the ODE
        num_points_guess = 1000
        dx_nominal = 0.5*(T1+1.)/dTdx_fun(0., 0.5*(T1+1.))/num_points_guess
        logger.info(f"dx={dx_nominal:g}")
        
        if not isothermal:
            assert rhop <= rho1, rhop-rho1

            # integration starts from upstream temperature
            xsol = [0.]
            Tsol = [T1]
            ode_solver.set_initial_value(y=Tsol[-1], t=xsol[-1])

            dx = -dx_nominal
            xmax_limit = -1e10
            while ode_solver.successful() and abs(Tsol[-1]-1.)>1e-10 and ode_solver.t > xmax_limit:
                Tsol.append(ode_solver.integrate(ode_solver.t+dx)[0])
                xsol.append(ode_solver.t+dx)
                # print("ODE", "x=",xsol[-1], "T=", Tsol[-1], "T-1=", Tsol[-1]-1., "succ", ode_solver.successful(), kappa(Tsol[-1], rho_T_fun(Tsol[-1])))
                if np.isnan(Tsol[-1]):
                    logger.fatal("ODE solver gives NAN")
                    # sys.exit(1)
                    break

            # reverse the solution to start from x=-inf
            xsol = xsol[::-1]
            Tsol = Tsol[::-1]
        else:
            # isothermal shock - all quantities are continuous
            # we integrate twice in positive and negative x directions
            # initial conditions are from the middle point between the overal temperature jump T1 and 1
            

            # -----integrate to x=-inf
            # integration starts from the middle of the overall temperature jump
            xsol = [0.]
            Tode_init = 0.5*(T1+1.)
            Tsol = [Tode_init]
            ode_solver.set_initial_value(y=Tsol[-1], t=xsol[-1])

            dx = -dx_nominal
            xmax_limit = -1e10
            while ode_solver.successful() and abs(Tsol[-1]-1.)>1e-10 and ode_solver.t > xmax_limit:
                Tsol.append(ode_solver.integrate(ode_solver.t+dx)[0])
                xsol.append(ode_solver.t+dx)
                # print("ODE", "x=",xsol[-1], "T=", Tsol[-1], "T-1=", Tsol[-1]-1., "succ", ode_solver.successful())
                if np.isnan(Tsol[-1]):
                    logger.fatal("ODE solver gives NAN")
                    sys.exit(1)
            
            # reverse the solution to start from x=-inf
            xsol = xsol[::-1]
            Tsol = Tsol[::-1]

            # -----integrate to x=inf
            dx = dx_nominal
            xmax_limit = 1e10
            ode_solver.set_initial_value(y=Tsol[-1], t=xsol[-1])
            while ode_solver.successful() and abs(Tsol[-1]-T1)>1e-10 and ode_solver.t < xmax_limit:
                Tsol.append(ode_solver.integrate(ode_solver.t+dx)[0])
                xsol.append(ode_solver.t+dx)
                # print("ODE", "x=",xsol[-1], "T=", Tsol[-1], "T-1=", Tsol[-1]-1., "succ", ode_solver.successful())
                if np.isnan(Tsol[-1]):
                    logger.fata("ODE solver gives NAN")
                    sys.exit(1)

        xsol = np.array(xsol)
        Tsol = np.array(Tsol)
        rhosol = np.array([rho_T_fun(Ti) for Ti in Tsol])
        velsol = -M0/rhosol
        vp = -M0/rhop

        logger.info(f"points on profile: N={len(xsol):g}")

        # plot dimensionless profiles
        if plot:
            from matplotlib import pyplot as plt
            title = f"$\\mathcal{{M}}_0={M0:g}, \\ \\  \\mathcal{{P}}_0={P0:g} $ Np={len(xsol)}"
            if isothermal:
                title = "isothermal " + title

            plt.figure("T")
            plt.plot(xsol, Tsol, c="r", ls="-")
            plt.axhline(y=1., c="k", ls="--", label=f"$T_0={1:g}$")
            plt.axhline(y=T1, c="g", ls="--", label=f"$T_1={T1:g}$")
            plt.grid()
            plt.legend()
            plt.title(title)
            plt.ylabel("$T(x)$")
            plt.xlabel("$x$")
            plt.autoscale(enable=True, axis='x', tight=True)

            plt.figure("rho")
            plt.plot(xsol, rhosol, c="r", ls="-")
            plt.axhline(y=1., c="k", ls="--", label=f"$\\rho_0={1:g}$")
            plt.axhline(y=rho1, c="g", ls="--", label=f"$\\rho_1={rho1:g}$")
            plt.scatter([0.], [rhop], c="b", label=f"$\\rho_p={rhop:g}$")
            plt.grid()
            plt.legend()
            plt.title(title)
            plt.ylabel("$\\rho(x)$")
            plt.xlabel("$x$")
            plt.autoscale(enable=True, axis='x', tight=True)

            plt.figure("M")
            mach = M0/(rhosol*np.sqrt(Tsol))
            M1 = M0/(rho1*np.sqrt(T1))
            Mp = M0/(rhop*np.sqrt(T1))
            plt.plot(xsol, mach, c="r", ls="-")
            plt.axhline(y=M0, c="k", ls="--", label=f"$\\mathcal{{M}}_0={M0:g}$")
            plt.axhline(y=M1, c="g", ls="--", label=f"$\\mathcal{{M}}_1={M1:g}$")
            plt.scatter([0.], [Mp], c="b", label=f"$\\mathcal{{M}}_p={Mp:g}$")
            plt.grid()
            plt.legend()
            plt.title(title)
            plt.ylabel("$\\mathcal{{M}}(x)$")
            plt.xlabel("$x$")
            plt.autoscale(enable=True, axis='x', tight=True)

            plt.show()

        return rhop, vp, xsol, Tsol, rhosol, velsol, isothermal
    
def example_dimensionless_profiles(M0, gamma, P0, kappa):
    rho1, T1, v1 = RadiativeShock.shock_jump_dimensionless(gamma=gamma, M0=M0, P0=P0)
    LTERadiativeShock.solve_dimensionless_profiles(
        M0=M0, gamma=gamma, P0=P0, kappa=kappa,
        rho1=rho1, T1=T1, v1=v1,
        plot_dimensionless=True,
    )

if __name__ == "__main__":

    # --------------- nondimensional examples
    # FROM: Lowrie 2007 LA-UR-07-5986 presentation

    # # slide 22
    # example_dimensionless_profiles(
    #     M0=1.2,
    #     gamma=5./3.,
    #     P0=1e-4,
    #     kappa=lambda T,rho: 1.,
    # )

    # # slide 23
    # example_dimensionless_profiles(
    #     M0=2.,
    #     gamma=5./3.,
    #     P0=1e-4,
    #     kappa=lambda T,rho: 1.,
    # )

    # # slide 24
    # example_dimensionless_profiles(
    #     M0=3.,
    #     gamma=5./3.,
    #     P0=1e-4,
    #     kappa=lambda T,rho: 1.,
    # )

    # # slide 26
    # example_dimensionless_profiles(
    #     M0=5.,
    #     gamma=5./3.,
    #     P0=1e-4,
    #     kappa=lambda T,rho: 1.,
    # )

    # # slide 28
    # example_dimensionless_profiles(
    #     M0=27.,
    #     gamma=5./3.,
    #     P0=1e-4,
    #     kappa=lambda T,rho: 1.,
    # )

    # # slide 30
    # example_dimensionless_profiles(
    #     M0=30.,
    #     gamma=5./3.,
    #     P0=1e-4,
    #     kappa=lambda T,rho: 1.,
    # )

    # # slide 32
    # example_dimensionless_profiles(
    #     M0=50.,
    #     gamma=5./3.,
    #     P0=1e-4,
    #     kappa=lambda T,rho: 1.,
    # )

    # # slide 33
    # example_dimensionless_profiles(
    #     M0=5.,
    #     gamma=5./3.,
    #     P0=1e-4,
    #     kappa=lambda T,rho: 0.00175/rho*T**(7./2.),
    # )

    # --------------- dimensional examples

    # Gentile LLNL-PRES-789681 
    # Mach45 slide 44/50
    solver = LTERadiativeShock(
        M0=45.,
        gamma=5./3.,
        sigma_ross=lambda T,rho: 0.4006*rho+0.0142*rho*rho*(T/Units.kev_kelvin)**(-3.5),
        cv=1.45e15/Units.kev_kelvin,
        rho_left=1.,
        v_left=0.,
        T_left=0.1 * Units.kev_kelvin,
        plot_dimensionless=True,
    ).plot_profiles()

    # Gentile LLNL-PRES-789681 
    # Mach2 slide 45/50
    solver = LTERadiativeShock(
        M0=2.,
        gamma=5./3.,
        sigma_ross=lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5),
        cv=2.22e15/Units.kev_kelvin,
        rho_left=1.,
        v_left=0.,
        T_left=0.122*Units.kev_kelvin,
        plot_dimensionless=True,
    ).plot_profiles()

    # Krief changed Gentil Mach2 to Mach 1.3 to be and isothermal shock
    # same as Gentile Mach2
    solver = LTERadiativeShock(
        # M0=2.,
        M0=1.3, ### Krief CHANGED IT IS ISOTHERMAL
        gamma=5./3.,
        sigma_ross=lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5),
        cv=2.22e15/Units.kev_kelvin,
        rho_left=1.,
        v_left=0.,
        T_left=0.122*Units.kev_kelvin,
        plot_dimensionless=True,
    ).plot_profiles()


    # Krief changed Gentile Mach2 to Mach5 and higher downstream temperature to get a higher P0 -
    # dynamics where the radiation pressure and energy are not negligible
    solver = LTERadiativeShock(
        M0=5,
        gamma=5./3.,
        sigma_ross=lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5),
        cv=2.22e15/Units.kev_kelvin,
        rho_left=1.,
        v_left=0.,
        T_left=0.55*Units.kev_kelvin, ### Krief CHANGED IT TO HAVE non-negligible P0 - that works also in NLTE
        plot_dimensionless=True,
    ).plot_profiles()

    # Krief changed to very high downstream temperature to get a high P0 -
    # dynamics where the radiation pressure and energy dominates
    solver = LTERadiativeShock(
        M0=5,
        gamma=5./3.,
        sigma_ross=lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5),
        cv=2.22e15/Units.kev_kelvin,
        rho_left=1.,
        v_left=0.,
        T_left=3.*Units.kev_kelvin, ### Krief CHANGED IT TO HAVE HIGH P0 - THIS CASE DOES NOT WORK FOR NLTE
        plot_dimensionless=True,
    ).plot_profiles()

    # Skinner2019
    Navog = 6.02214076e23
    k_boltz = 1.380649e-16
    gas_constant = k_boltz*Navog
    molar_mass_H = 1.
    solver = LTERadiativeShock(
        M0=3.,
        gamma=5./3.,
        sigma_ross=lambda T,rho:577.,
        cv=gas_constant/(5./3.-1.)/molar_mass_H,
        rho_left=5.69,
        v_left=5.19e7,
        T_left=2.18e6,
        plot_dimensionless=True,
    ).plot_profiles()


    # # Ramsey2014 A&A 574, A81 (2015) Mach2
    Navog = 6.02214076e23
    k_boltz = 1.380649e-16
    gas_constant = k_boltz*Navog
    molar_mass_H = 1.
    solver = LTERadiativeShock(
        M0=2.,
        gamma=5./3.,
        sigma_ross=lambda T,rho:3.92664e-5,
        cv=gas_constant/(5./3.-1.)/molar_mass_H,
        rho_left=5.45887e-13,
        v_left=2.35435e5,
        T_left=100.,
        plot_dimensionless=True,
    ).plot_profiles()

    # # Ramsey2014 A&A 574, A81 (2015) Mach5
    Navog = 6.02214076e23
    k_boltz = 1.380649e-16
    gas_constant = k_boltz*Navog
    molar_mass_H = 1.
    solver = LTERadiativeShock(
        M0=5.,
        gamma=5./3.,
        sigma_ross=lambda T,rho:3.92664e-5,
        cv=gas_constant/(5./3.-1.)/molar_mass_H,
        rho_left=5.45887e-13,
        v_left=5.88588e5,
        T_left=100.,
        plot_dimensionless=True,
    ).plot_profiles()

    pass