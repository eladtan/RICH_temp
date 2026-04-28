#ifndef DIFFUSION_HPP
#define DIFFUSION_HPP 1

#include "conj_grad_solve.hpp"
#include "RadiationDriver.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "OpacityCalculator.hpp"


using namespace CG;

namespace CG
{
    double CalcSingleFluxLimiter(Vector3D const& grad, double const D, double const cell_value);

    double FleckFactor(double const dt, double const beta, double const sigma_a);

}

//! \brief Class for assigning boundary conditions for diffusion
class DiffusionBoundaryCalculator
{
    public:
    /*!
\brief Sets the boundary values for the matrix build in A*x=b
\param tess The tesselation
\param index The index of the cell that is adjacent to a boundary
\param outside_point The index of the cell that is outside
\param cell The primitve cells
\param A the value in the A matrix to change, given as input and output
\param b the value in the b vector to change, given as input and output
\param Area The area of the interface between the two cells
\param dt The time step
\param face_index The index of the interface
    */
virtual void SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
    std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const face_index)const = 0;
    /*!
\brief Sets the outside values for the Froce calcualtion if needed
\param tess The tesselation
\param index The index of the cell that is adjacent to a boundary
\param outside_point The index of the cell that is outside
\param cell The primitve cells
\param E_outside The value of the energy in the outside cell, given as output
\param v_outside The value of the velocity, given as output
\param new_E The values of the new energy after the CG step
    */
virtual void GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const = 0;
   /*!
\brief Sets the boundary values for the matrix build for the momentum term
\param tess The tesselation
\param index The index of the cell that is adjacent to a boundary
\param outside_point The index of the cell that is outside
\param cell The primitve cell
\param A the value in the A matrix to change, given as input and output
\param b the value in the b vector to change, given as input and output
\param Area The area of the interface between the two cells
\param dt The time step
\param face_index The index of the interface
\param fleck_factor The fleck factor
\param flux_limiter The flux limiter
\param D The diffusion coefficient
\param sigma_planck The planck opacity
    */
virtual void SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
    ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, 
    double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck)const = 0;
};

//! \brief Class with constant blackbody temperature on the left x side and zero flux on other sides
class DiffusionSideBoundary : public DiffusionBoundaryCalculator
{
    public:
    /*!
    \brief Class constructor
    \param T Boundary temperature, in kelvin
    */
    DiffusionSideBoundary(double const T): T_(T){}

    void SetTemperature(double T) { T_ = T; }

    void SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const face_index)const override;

    void GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
        std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const override;

    void SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, 
        double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck)const override;

    private:
        double T_;
};

//! \brief Class with constant blackbody temperature on the left x side and zero flux on other sides
class DiffusionClosedBox : public DiffusionBoundaryCalculator
{
    public:
    void SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const face_index)const override;
    
    void GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
        std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const override;

    void SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, 
        double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck)const override;
};

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
    Diffusion(OpacityCalculator const& D_coefficient_calc, 
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

    double GetSingleFleckFactor(ComputationalCell3D const& cell, double const dt) const;

    virtual void PrintDebugData(size_t const index) const
    {
        std::cout<<"Diffusion debug data:"<<std::endl;
        std::cout<<"sigma_planck "<<sigma_planck[index]<<" sigma_s "<<sigma_s[index]<<
        " fleck_factor "<<fleck_factor[index]<<" D "<<D[index]<<" cell_flux_limiter "<<
        cell_flux_limiter[index]<<std::endl;
    }

    OpacityCalculator const& D_coefficient_calcualtor;
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
    mutable CG::BiCGSTABWorkspace cg_workspace_;
    bool const cooling_time_limiter_on_;
};

//! D=D0*rho^alpha*T^beta, sigma_planck=sigma_planck0*rho^alpha_planck*T^beta_planck
class PowerLawOpacity: public OpacityCalculator
{
private:
    double const D0_, alpha_, beta_, planck0_, alpha_planck_, beta_planck_;
public:
    PowerLawOpacity(double const D0, double const alpha, double const beta,
        double const planck0, double const alpha_planck, double const beta_planck)
        : D0_(D0), alpha_(alpha), beta_(beta), planck0_(planck0), alpha_planck_(alpha_planck),
        beta_planck_(beta_planck){}

    double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const override;

    double CalcPlanckOpacity(ComputationalCell3D const& cell) const override;
};
//! \brief Class with constant states on the x sides and zero flux on other sides
class DiffusionXInflowBoundary : public DiffusionBoundaryCalculator
{
    public:
    /*!
    \brief Class constructor
    \param T Boundary temperature, in kelvin
    */
    DiffusionXInflowBoundary(ComputationalCell3D const& left_state, ComputationalCell3D const& right_state,
        OpacityCalculator const& D_calc): left_state_(left_state), right_state_(right_state), D_calc_(D_calc){}

    void SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const face_index)const override;
    
    void GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
        std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const override;
    
   void SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, 
        double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck)const override;

    private:
        ComputationalCell3D const& left_state_, right_state_;
        OpacityCalculator const& D_calc_;
};

class DiffusionOpenBoundary : public DiffusionBoundaryCalculator
{
    public:
    void SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const face_index)const override;
    
    void GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
        std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const override;
    
   void SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, 
        double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck)const override;
};
#endif
