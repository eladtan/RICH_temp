import numpy as np
from matplotlib import pyplot as plt
import os
from os import path
import h5py

import sys
sys.path.insert(0, "/home/itamarg/workspace/RICH/analysis_files/radiative_shock")

from nlte_radiative_shock import NLTERadiativeShock
from radiative_shock import Units

import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('Mach2Analysis')

Navog = 6.02214076e23
k_boltz = 1.380649e-16
gas_constant = k_boltz*Navog
molar_mass_H = 1.
speed_of_light = 2.99792458e10
stefan_boltzman = 5.670374e-5
radiation_constant = 4 * stefan_boltzman / speed_of_light
boltzmann_constant = 1.380649e-16
ev = 1.602176634e-12
kev = 1e3*ev
kev_to_kelvin = kev / boltzmann_constant

cases = {
    "1" : {
        "M0": 45.,
        "gamma": 5./3.,
        "sigma_ross": lambda T,rho: 0.4006*rho + 0.0142*rho*rho*(T/Units.kev_kelvin)**(-3.5),
        "sigma_abs": lambda T,rho: 0.0142*rho*rho*(T/Units.kev_kelvin)**(-3.5),
        "cv" : 1.45e15/Units.kev_kelvin,
        "rho_left" : 1.0,
        "v_left" : 0.0,
        "T_left" : 0.1 * Units.kev_kelvin,
        "plot_dimensionless": True,
        "eps_nlte_solver": 1e-5
    },

    "2" : {
        "M0": 2.,
        "gamma": 5./3.,
        "sigma_ross": lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5),
        "sigma_abs": lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5),
        "cv" : 2.22e15/Units.kev_kelvin,
        "rho_left" : 1.0,
        "v_left" : 0.0,
        "T_left" : 0.122*Units.kev_kelvin,
        "plot_dimensionless": True,
        "eps_nlte_solver": 1e-5
    },
    "3" : {
        "M0": 2.,
        "gamma": 5./3.,
        "sigma_ross": lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5),
        "sigma_abs": lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5),
        "cv" : 2.22e15/Units.kev_kelvin,
        "rho_left" : 1.0,
        "v_left" : 0.0,
        "T_left" : 0.122*Units.kev_kelvin,
        "plot_dimensionless": True,
        "eps_nlte_solver": 1e-5
    },
    "2" : {
        "M0": 5.,
        "gamma": 5./3.,
        "sigma_ross": lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5),
        "sigma_abs": lambda T,rho: 0.362*rho*(T/Units.kev_kelvin)**(-3.5),
        "cv" : 2.22e15/Units.kev_kelvin,
        "rho_left" : 1.0,
        "v_left" : 0.0,
        "T_left" : 0.55*Units.kev_kelvin,
        "plot_dimensionless": True,
        "eps_nlte_solver": 1e-5
    },
    "4" : {
        "M0": 3.,
        "gamma": 5./3.,
        "sigma_ross": lambda T,rho: 577.,
        "sigma_abs": lambda T,rho: 577.,
        "cv" : gas_constant/(5./3.-1.)/molar_mass_H,
        "rho_left" : 5.69,
        "v_left" : 5.19e7,
        "T_left" : 2.18e6,
        "plot_dimensionless": True,
        "eps_nlte_solver": 1e-6
    }, 
    "5" : {
        "M0": 5.,
        "gamma": 5./3.,
        "sigma_ross": lambda T,rho: 0.848903,
        "sigma_abs": lambda T,rho: 3.92664e-5,
        "cv" : gas_constant/(5./3.-1.)/molar_mass_H,
        "rho_left" : 5.45969e-13,
        "v_left" : 5.88588e5,
        "T_left" : 100.,
        "plot_dimensionless": True,
        "eps_nlte_solver": 1e-6
    },
    "6" : {
        "M0": 2.,
        "gamma": 5./3.,
        "sigma_ross": lambda T,rho: 0.848903,
        "sigma_abs": lambda T,rho: 3.92664e-5,
        "cv" : gas_constant/(5./3.-1.)/molar_mass_H,
        "rho_left" : 5.45969e-13,
        "v_left" : 2.3547e5,
        "T_left" : 100.,
        "plot_dimensionless": True,
        "eps_nlte_solver": 0.5e-4
    }
}


def compare_profiles(*, case, output_folder, Emin, Emax, MPI=False):
    logger.info(f"case = {case}")
    logger.info(f"output_folder = {output_folder}")
    logger.info(f"Emin = {Emin:g}")
    logger.info(f"Emax = {Emax:g}")
    logger.info(f"MPI = {MPI}")
    
    path_to_initial_snap = path.join(output_folder, f"Mach2_{0}.h5")
    snaps = [file for file in os.listdir(output_folder) if file.endswith(".h5") and file.startswith("Mach2") and (not file.endswith("final.h5"))]

    num_of_snaps = len([file for file in os.listdir(output_folder) if file.endswith(".h5") and file.startswith("Mach2") and (not file.endswith("final.h5"))])
    logger.info(f"num_of_snaps = {num_of_snaps}")
    
    num_groups = 0
    num_ranks = 0

    with h5py.File(path_to_initial_snap) as h5_file:
        # print(h5_file.keys())
        if MPI:
            num_groups = len([key for key in h5_file["rank0"].keys() if "Eg" in key])
            num_ranks = len([key for key in h5_file.keys() if "rank" in key])
        else:
            num_groups = len([key for key in h5_file.keys() if "Eg" in key])

    logger.info(f"num_groups = {num_groups}")
    if MPI: logger.info(f"num_ranks = {num_ranks}")
    
    # G = num_groups
    # energy_groups_boundary = np.geomspace(Emin, Emax, G+1)
    # energy_groups_boundary_kev = energy_groups_boundary / kev
    # energy_groups_width = np.diff(energy_groups_boundary, n=1)

    dir_figs = path.join(output_folder, "material_temperature")
    os.makedirs(dir_figs, exist_ok=True)

    solver = NLTERadiativeShock(**cases[case])#.plot_profiles()
    plt.close()

    x = solver.x_profile
    sol = solver.solve_profiles(time=0.0, x=x)

    for i in range(num_of_snaps):
        print(f"{i}/{num_of_snaps}")
        path_to_snap = path.join(output_folder, f"Mach2_{i*100}.h5")
        time = None
        with h5py.File(path_to_snap) as h5_file:
            time = h5_file["Time"][0]
            if "rank0" not in h5_file.keys(): 
                plt.plot(np.array(h5_file["CMx"]), np.array(h5_file["Temperature"]), '+', label=f"rank 0")
            else :
                for j in range(num_ranks):
                    plt.plot(np.array(h5_file[f"rank{j}"]["CMx"]), np.array(h5_file[f"rank{j}"]["Temperature"]), '+', label=f"rank {j}")
        plt.plot(x, sol['temperature'], '--', label="Analytic Solution")

        plt.title(f"snap {i}, t={time:g}")
        plt.legend(loc="best")
        plt.grid()
        plt.savefig(path.join(dir_figs, f"snap_{i}.png"))
        # if i == num_of_snaps - 1: plt.show()
        plt.close()

if __name__ == "__main__":
    compare_profiles(case="6", output_folder=sys.argv[1], Emin=ev*1e-3, Emax=ev, MPI=False)
    # compare_profiles(case="4", output_folder="/home/itamarg/workspace/RICH/output/Mach2_step_2", Emin=ev*1e-3, Emax=ev, MPI=False)