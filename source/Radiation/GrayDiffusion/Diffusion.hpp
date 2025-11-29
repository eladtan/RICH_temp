#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP 1

#include "source/Radiation/conj_grad_solve.hpp"
#include "source/Radiation/RadiationDriver.hpp"
#include "source/newtonian/common/equation_of_state.hpp"
#include "DiffusionBoundaryCalculator.hpp"
#include "DiffusionCoefficientCalculator.hpp"

using namespace CG;

//! \brief Class for calculating diffusion matrix data for the CG solver
class Diffusion : public RadiationDriver
{
public:
/*!
\brief Class constructor
\param D_coefficient_calc Class for calcualting the diffusion coefficients
\param eos The equation of state
\param boundary_calc Class to calcualte the values for the boundary conditions
*/
    Diffusion(DiffusionCoefficientCalculator const& D_coefficient_calc, 
              DiffusionBoundaryCalculator const& boundary_calc,
              EquationOfState const& eos, 
              std::vector<std::string> const zero_cells = std::vector<std::string> (), 
              bool const flux_limiter = true, 
              bool const hydro_on = true, 
              bool const compton_on = false);
    
    ~Diffusion() = default;

    double GetLengthScale() const override {return length_scale_;}

    bool prestep(Tessellation3D const& tess,
                 std::vector<ComputationalCell3D> const& cells) const override;

    bool step(double const tolerance, 
              int& total_iters, 
              Tessellation3D const& tess, 
              std::vector<ComputationalCell3D>& cells,
              std::vector<Conserved3D>& extensives,
              double const dt,
              double const time) const override;

    bool poststep() const override;

    double calculate_dt(double const dt,
                        Tessellation3D& tess, 
                        std::vector<ComputationalCell3D>& cells) const override;

    void BuildMatrix(Tessellation3D const& tess, mat& A, size_t_mat& A_indeces, std::vector<ComputationalCell3D> const& cells, 
            double const dt, std::vector<double>& b, std::vector<double>& x0, double const current_time) const override;

    void PostCG(Tessellation3D const& tess, std::vector<Conserved3D>& extensives, double const dt, std::vector<ComputationalCell3D>& cells,
        std::vector<double>const& CG_result, std::vector<double> const&  full_CG_result) const override;

    double GetSingleFleckFactor(
        ComputationalCell3D const& cell, 
        std::size_t const cell_index,
        double const dt
    ) const;

    virtual void PrintDebugData(size_t const index) const
    {
        std::cout<<"Diffusion debug data:"<<std::endl;
        std::cout<<"sigma_planck "<<sigma_planck[index]<<" sigma_s "<<sigma_s[index]<<
        " fleck_factor "<<fleck_factor[index]<<" D "<<D[index]<<" cell_flux_limiter "<<
        cell_flux_limiter[index]<<std::endl;
    }

    bool step_iterations(
        double const tolerance, 
        int& total_iters, 
        Tessellation3D const& tess, 
        std::vector<ComputationalCell3D>& cells,
        std::vector<Conserved3D>& extensives,
        double const dt,
        double const time
    ) const;

    bool update_energy(
        Tessellation3D const& tess, 
        std::vector<Conserved3D>& extensives, 
        double const dt, 
        double const time
    ) const;

    
    
    
    DiffusionCoefficientCalculator const& D_coefficient_calcualtor;
    DiffusionBoundaryCalculator const& boundary_calc_;
    
    mutable std::vector<double> sigma_planck;
    mutable std::vector<double> sigma_s;
    mutable std::vector<double> fleck_factor; 
    mutable std::vector<double> D; 
    mutable std::vector<double> R2; 
    mutable std::vector<double> cell_flux_limiter;
    mutable std::vector<double> new_Er;
    mutable std::vector<double> new_Er_full;
    mutable std::vector<double> old_Er;
    mutable std::vector<double> old_T;
    mutable std::vector<ComputationalCell3D> cells_temp;
    mutable std::vector<Conserved3D> extensives_temp;
    
    private:
    void load_cells_cgs(
        Tessellation3D const& tess, 
        std::vector<ComputationalCell3D> const& cells_not_cgs) const;
        
    void calculate_planck_absorption_coefficient(
        Tessellation3D const& tess
    ) const;
    
    void calculate_scattering_coefficient(
        Tessellation3D const& tess
    ) const;
    
    void calculate_fleck_factor(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells,
        double const dt_cgs
    ) const;
    
    mutable std::vector<ComputationalCell3D> cells_cgs;
    mutable bool do_iterations_on_Um;
};
#endif