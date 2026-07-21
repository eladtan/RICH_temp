#ifndef DIFFUSION_BOUNDARY_CALCULATOR_HPP
#define DIFFUSION_BOUNDARY_CALCULATOR_HPP

#include "source/3D/tessellation/Tessellation3D.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/Radiation/GrayDiffusion/DiffusionCoefficientCalculator.hpp"

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

    void SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const face_index)const override;

    void GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
        std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const override;

    void SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, 
        double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck)const override;

    void SetTemperature(double const temperature_);

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

//! \brief Class with constant states on the x sides and zero flux on other sides
class DiffusionXInflowBoundary : public DiffusionBoundaryCalculator
{
    public:
    /*!
    \brief Class constructor
    \param T Boundary temperature, in kelvin
    */
    DiffusionXInflowBoundary(ComputationalCell3D const& left_state, ComputationalCell3D const& right_state,
        DiffusionCoefficientCalculator const& D_calc): left_state_(left_state), right_state_(right_state), D_calc_(D_calc){}

    void SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const face_index)const override;
    
    void GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
        std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const override;
    
   void SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, 
        double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck)const override;

    private:
        ComputationalCell3D const& left_state_, right_state_;
        DiffusionCoefficientCalculator const& D_calc_;
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
#endif // DIFFUSION_BOUNDARY_CALCULATOR_HPP