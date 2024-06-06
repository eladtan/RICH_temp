#ifndef MULTIGROUP_DIFFUSION_HPP
#define MULTIGROUP_DIFFUSION_HPP

#include "RadiationDriver.hpp"
#include "conj_grad_solve.hpp"

using namespace CG;

class MultigroupDiffusionCoefficientCalculator {
public:
    virtual double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const =0;

    virtual double CalcPlanckOpacityGroup(ComputationalCell3D const& cell, std::size_t const group) const = 0;

    virtual double CalcDiffusionCoefficientGray(ComputationalCell3D const& cell) const = 0;

    virtual double CalcPlanckOpacityGray(ComputationalCell3D const& cell) const = 0;
};

class MultigroupDiffusion : public RadiationDriver {
public:
    MultigroupDiffusion(EquationOfState const& eos,
                        std::vector<std::string> const zero_cells,
                        bool const flux_limiter,
                        bool const hydro_on,
                        bool const compton_on);

    ~MultigroupDiffusion();

    bool prestep(Tessellation3D const& tess) const override;

    bool step(double const tolerance, 
              int& total_iters, 
              Tessellation3D const& tess, 
              std::vector<ComputationalCell3D>& cells,
              std::vector<Conserved3D>& extensives,
              double const dt,
              double const time) const override;

    bool poststep() const override;

    void BuildMatrix(Tessellation3D const& tess, mat& A, size_t_mat& A_indeces, std::vector<ComputationalCell3D> const& cells, 
            double const dt, std::vector<double>& b, std::vector<double>& x0, double const current_time) const override;

    void PostCG(Tessellation3D const& tess, std::vector<Conserved3D>& extensives, double const dt, std::vector<ComputationalCell3D>& cells,
        std::vector<double>const& CG_result, std::vector<double> const&  full_CG_result) const override;

    std::size_t current_group;
    bool grey;

    std::vector<ComputationalCell3D> cells_temp;
    std::vector<Conserved3D> extensives_temp;
};

#endif