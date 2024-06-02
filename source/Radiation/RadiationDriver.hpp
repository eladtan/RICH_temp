#ifndef RADIATION_DRIVER_HPP
#define RADIATION_DRIVER_HPP

#include "conj_grad_solve.hpp"
#include "source/newtonian/common/equation_of_state.hpp"

class RadiationDriver : public CG::MatrixBuilder {
public:
    RadiationDriver(EquationOfState const& eos,
                    std::vector<std::string> const zero_cells_ = std::vector<std::string>(),
                    bool const flux_limiter = true,
                    bool const hydro_on = true,
                    bool const compton_on = false) : 
                                                        eos_(eos),
                                                        flux_limiter_(flux_limiter),
                                                        hydro_on_(hydro_on),
                                                        compton_on_(compton_on),
                                                        mass_scale_(1.0),
                                                        length_scale_(1.0),
                                                        time_scale_(1.0),
                                                        CG::MatrixBuilder(zero_cells_)
                                                        {}

        virtual ~RadiationDriver() = default;

        virtual bool step(double const tolerance, 
                          int& total_iters, 
                          Tessellation3D const& tess, 
                          std::vector<ComputationalCell3D> const& cells,
                          double const dt,
                          double const time) = 0;

    bool const flux_limiter_;
    bool const hydro_on_;
    bool const compton_on_;

    double mass_scale_;
    double length_scale_;
    double time_scale_;

protected:
    EquationOfState const& eos_;
};

#endif