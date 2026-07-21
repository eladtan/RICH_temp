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
              bool const compton_on = false,
              bool const cooling_time_limiter_on = false);
    
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

    virtual void PrintDebugData(size_t const index) const override
    {
        std::cout<<"Diffusion debug data:"<<std::endl;
        std::cout<<"sigma_planck "<<sigma_planck[index]<<" sigma_s "<<sigma_s[index]<<
        " fleck_factor "<<fleck_factor[index]<<" D "<<D[index]<<" cell_flux_limiter "<<
        cell_flux_limiter[index]<<std::endl;
    }
    
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
    bool const cooling_time_limiter_on_;
    
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

    void apply_opacity_limiters(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells,
        double const dt
    ) const;
    
    void calculate_fleck_factor(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells,
        double const dt_cgs
    ) const;

    double GetSingleFleckFactor(
        ComputationalCell3D const& cell, 
        std::size_t const cell_index,
        double const dt
    ) const;

    void calculate_cell_diffusion_coefficients(
        Tessellation3D const& tess
    ) const;

    void fix_small_negative_Er(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells
    ) const;

    double dE_absorption_emission(
        Tessellation3D const& tess,
        std::size_t i,
        double const Er,
        double const temperature,
        double const dt_cgs
    ) const;

    double dE_v_squared(
        Tessellation3D const& tess,
        std::size_t i,
        double const Er,
        Vector3D const& velocity_cgs,
        double const max_velocity_cgs,
        double const dt_cgs
    ) const;

    double dE_compton(
        Tessellation3D const& tess,
        std::size_t i,
        double const Er,
        double const temperature,
        double const old_Er,
        double const dt_cgs
    ) const;

    bool is_energy_invalid(Conserved3D const& extensive) const;

    double dE_relativity(
        Tessellation3D const& tess,
        std::size_t i,
        std::vector<double> const& CG_result,
        double const dt_cgs,
        Vector3D& gradE
    ) const;

    std::tuple<Vector3D, double, double> dP_and_dE_momentum(
        std::size_t i,
        Vector3D const& gradE,
        double const dt_cgs,
        Vector3D const& momentum,
        double const mass
    ) const;

    void compute_equilibrium_from_energy_sum(
        Tessellation3D const& tess,
        std::size_t i,
        std::vector<ComputationalCell3D> const& cells,
        std::vector<Conserved3D>& extensives,
        double const volume,
        std::vector<double> const& CG_result,
        double const dt_cgs
    ) const;

    void print_postcg1_debug(
        std::size_t i,
        std::vector<ComputationalCell3D> const& cells,
        std::vector<Conserved3D> const& extensives,
        std::vector<double> const& CG_result,
        std::vector<double> const& full_CG_result,
        double T,
        double dE,
        double old_e_therm,
        double e_emitt,
        double e_v2,
        double e_absorb_emitt,
        double compton_term,
        double old_Tr
    ) const;

    void print_postcg2_debug(
        std::size_t i,
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells,
        std::vector<Conserved3D> const& extensives,
        std::vector<double> const& CG_result,
        std::vector<double> const& full_CG_result,
        double T,
        double max_v,
        double dt_cgs,
        Vector3D const& dP,
        double Erad_dE,
        double e_absorb,
        double e_emitt,
        double e_v2,
        double total_relativity,
        double etherm_mid,
        Vector3D const& gradE,
        Vector3D const& CM,
        Vector3D const& point,
        std::vector<size_t> const& neighbors,
        face_vec const& faces
    ) const;

    bool iterations(
        double const tolerance, 
        int& total_iters, 
        Tessellation3D const& tess, 
        std::vector<ComputationalCell3D>& cells,
        std::vector<Conserved3D>& extensives,
        double const dt,
        double const time
    ) const;

    bool update_energy_iterations(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D>& cells,
        std::vector<Conserved3D>& extensives,
        double const dt,
        std::vector<double>& Er_full,
        std::vector<double>& Er,
        double& newton_raphson_error,
        std::vector<double>& error_per_cell
    ) const;
    
    mutable std::vector<ComputationalCell3D> cells_cgs;
    mutable bool do_iterations_on_Um;
    mutable bool use_new_Er_for_x0;
};
#endif