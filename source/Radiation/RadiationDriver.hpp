#ifndef RADIATION_DRIVER_HPP
#define RADIATION_DRIVER_HPP

#include "conj_grad_solve.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "boost/math/special_functions/pow.hpp"

class RadiationDriver : public CG::MatrixBuilder {
public:
    /**
 * @brief A base class for radiation drivers.
 * 
 * This class provides an interface for radiation drivers, which are used to handle radiation effects in simulations.
 * It inherits from CG::MatrixBuilder and contains common parameters and methods for radiation drivers.
 * 
 * @param eos The equation of state used to calculate thermodynamic properties.
 * @param zero_cells_ A vector of strings representing cells that should be treated as zero-radiation cells.
 * @param flux_limiter A boolean indicating whether to use a flux limiter in the radiation transport.
 * @param hydro_on A boolean indicating whether hydrodynamic effects are turned on.
 * @param compton_on A boolean indicating whether Compton scattering is turned on.
 */
    RadiationDriver(EquationOfState const& eos,
                    std::vector<std::string> const zero_cells_ = std::vector<std::string>(),
                    bool const flux_limiter = true,
                    bool const hydro_on = true,
                    bool const compton_on = false) : 
                                                        CG::MatrixBuilder(zero_cells_),
                                                        eos_(eos),
                                                        flux_limiter_(flux_limiter),
                                                        hydro_on_(hydro_on),
                                                        compton_on_(compton_on),
                                                        mass_scale_(1.0),
                                                        length_scale_(1.0),
                                                        time_scale_(1.0),
                                                        density_scale_(1.0),
                                                        energy_density_scale_(1.0),
                                                        specific_energy_scale_(1.0),
                                                        velocity_scale_(1.0)
                                                        {}

/**
 * @brief A virtual destructor for the RadiationDriver class.
 * 
 * This destructor is declared as virtual to ensure proper cleanup of derived classes.
 */
        virtual ~RadiationDriver() = default;

/**
 * @brief A pure virtual function for performing pre-step operations.
 * 
 * This function should be implemented in derived classes to perform any necessary operations before each time step.
 * 
 * @param tess The current tessellation of the simulation domain.
 * @param cells The current state of computational cells.
 * 
 * @return A boolean indicating whether the pre-step operation was successful.
 */
        virtual bool prestep(Tessellation3D const& tess,
                             std::vector<ComputationalCell3D> const& cells) const = 0;

/**
 * @brief A pure virtual function for performing a time step.
 * 
 * This function should be implemented in derived classes to perform the main radiation transport and update the state of computational cells.
 * 
 * @param tolerance The tolerance for convergence in the iterative solver.
 * @param total_iters A reference to an integer that will be updated with the total number of iterations performed.
 * @param tess The current tessellation of the simulation domain.
 * @param cells The current state of computational cells.
 * @param extensives The current state of extensive conserved quantities.
 * @param dt The time step size.
 * @param time The current simulation time.
 * 
 * @return A boolean indicating whether the time step was successful.
 */
        virtual bool step(double const tolerance, 
                          int& total_iters, 
                          Tessellation3D const& tess, 
                          std::vector<ComputationalCell3D>& cells,
                          std::vector<Conserved3D>& extensives, 
                          double const dt,
                          double const time) const = 0;
        
/**
 * @brief A pure virtual function for performing post-step operations.
 * 
 * This function should be implemented in derived classes to perform any necessary operations after each time step.
 * 
 * @return A boolean indicating whether the post-step operation was successful.
 */
        virtual bool poststep() const = 0; 
/**
 * @brief A pure virtual function for calculating the maximum allowable time step size.
 * 
 * This function should be implemented in derived classes to determine the maximum allowable time step size based on the current state of the simulation.
 * 
 * @param dt The current time step size.
 * @param tess The current tessellation of the simulation domain.
 * @param cells The current state of computational cells.
 * 
 * @return The maximum allowable time step size.
 */
       virtual double calculate_dt(double const dt,
                                    Tessellation3D& tess, 
                                    std::vector<ComputationalCell3D>& cells) const = 0;

       void set_scales(double const mass_scale, double const length_scale, double const time_scale) const {
              mass_scale_ = mass_scale;
              length_scale_ = length_scale;
              time_scale_ = time_scale;

              density_scale_ = mass_scale / (length_scale*length_scale*length_scale);
              energy_density_scale_ = mass_scale / (time_scale * time_scale * length_scale);
              specific_energy_scale_ = length_scale * length_scale / (time_scale * time_scale);
              energy_scale_ = mass_scale * length_scale * length_scale / (time_scale * time_scale);

              velocity_scale_ = length_scale / time_scale;
              momentum_scale_ = mass_scale * length_scale / time_scale;

              area_scale_   = length_scale * length_scale;
              volume_scale_ = length_scale * length_scale * length_scale;
       }

       bool const flux_limiter_;
       bool const hydro_on_;
       bool const compton_on_;     
       
       mutable double mass_scale_;
       mutable double length_scale_;
       mutable double time_scale_;  
       
       mutable double density_scale_;
       mutable double energy_density_scale_;
       mutable double specific_energy_scale_;
       mutable double energy_scale_;
       
       mutable double velocity_scale_;
       mutable double momentum_scale_;
       
       mutable double area_scale_;
       mutable double volume_scale_;

protected:
    EquationOfState const& eos_;
};

using boost::math::pow;
static inline double get_radiation_energy_density(double const T) { return CG::radiation_constant*pow<4>(T); }
static inline double get_temperature(double const radiation_energy_density) { return std::sqrt(std::sqrt(radiation_energy_density/CG::radiation_constant)); }
static inline double get_radiation_cv(double const T) { return 4.0*CG::radiation_constant*pow<3>(T); }

#endif