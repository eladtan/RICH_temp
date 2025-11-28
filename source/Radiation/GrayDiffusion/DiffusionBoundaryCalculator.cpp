#include "DiffusionBoundaryCalculator.hpp"
#include "source/Radiation/conj_grad_solve.hpp"

void DiffusionSideBoundary::SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt, 
    std::vector<ComputationalCell3D> const& /*cells*/, double const Area, double& A, double &b, size_t const /*face_index*/)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        A += 0.5 * CG::speed_of_light * dt * Area;
        b += 2 * Area * dt * CG::stefan_boltzman * T_ * T_ * T_ * T_;
    }
}

void DiffusionSideBoundary::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
    ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, double const fleck_factor,
    double const flux_limiter, double const D, double const sigma_planck)const
{
    double const R = tess.GetWidth(index);
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * 
        (2 * 3 * sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        A += momentum_relativity_term;
        b -= momentum_relativity_term * CG::radiation_constant * T_ * T_ * T_ * T_;
    }
    else
        A += 2 * momentum_relativity_term;
}

void DiffusionSideBoundary::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
        E_outside = CG::radiation_constant * T_ * T_ * T_ * T_;
    else
        E_outside = new_E[index];
    v_outside = cells[index].velocity;
}

void DiffusionSideBoundary::SetTemperature(double const temperature){
    T_ = temperature;
}

void DiffusionClosedBox::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
    ComputationalCell3D const& cell, double const Area, double& A, 
    double& /*b*/, size_t const /*face_index*/, double const fleck_factor, double const flux_limiter, 
    double const D, double const sigma_planck)const
{
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * 
        (2 * 3 * sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    A += 2 * momentum_relativity_term;
}

void DiffusionClosedBox::SetBoundaryValues(Tessellation3D const& /*tess*/, size_t const /*index*/, size_t const /*outside_point*/, double const /*dt*/, 
    std::vector<ComputationalCell3D> const& /*cells*/, double const /*Area*/, double& /*A*/, double& /*b*/, size_t const /*face_index*/)const
{}

void DiffusionClosedBox::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    E_outside = new_E[index];
    Vector3D normal = normalize(tess.GetMeshPoint(outside_point) - tess.GetMeshPoint(index));
    v_outside = cells[index].velocity;
    v_outside -= 2 * normal * ScalarProd(normal, v_outside);
}

void DiffusionXInflowBoundary::SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt, 
        std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const face_index)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        double const Er_j = left_state_.Erad * left_state_.density;
        double const Er = cells[index].Erad * cells[index].density;
        double const dx = tess.GetCellCM(index).x - tess.GetBoxCoordinates().first.x;
        Vector3D const grad_E = Vector3D(1, 0, 0) * (1.0 / (2 * dx));
        Vector3D const r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
        double mid_D = 0.5 * (D_calc_.CalcDiffusionCoefficient(cells[index]) + D_calc_.CalcDiffusionCoefficient(left_state_));
        double const flux_limiter = CG::CalcSingleFluxLimiter(grad_E * (Er - Er_j), mid_D, 0.5 * (Er + Er_j));
        mid_D *= flux_limiter;
        double const flux = ScalarProd(grad_E, r_ij * (tess.GetArea(face_index) / abs(r_ij))) * dt * mid_D; 
        A += flux;
        b += flux * Er_j;
    }
    else
    {
        if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4))
        {
            double const Er_j = right_state_.Erad * right_state_.density;
            double const Er = cells[index].Erad * cells[index].density;
            double const dx = tess.GetBoxCoordinates().second.x - tess.GetCellCM(index).x;
            Vector3D const grad_E = Vector3D(-1, 0, 0) * (1.0 / (2 * dx));
            Vector3D const r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
            double mid_D = 0.5 * (D_calc_.CalcDiffusionCoefficient(cells[index]) + D_calc_.CalcDiffusionCoefficient(right_state_));
            double const flux_limiter = CG::CalcSingleFluxLimiter(grad_E * (Er - Er_j), mid_D, 0.5 * (Er + Er_j));
            mid_D *= flux_limiter;
            double const flux = ScalarProd(grad_E, r_ij * (tess.GetArea(face_index) / abs(r_ij))) * dt * mid_D; 
            A += flux;
            b += flux * Er_j;
        }
    }
}

void DiffusionXInflowBoundary::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, double const fleck_factor,
        double const flux_limiter, double const D, double const sigma_planck)const
{
    double const R = tess.GetWidth(index);
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * 
        (2 * 3 * sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        A += momentum_relativity_term;
        b -= momentum_relativity_term * left_state_.Erad * left_state_.density;
    }
    else
    {
        if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4))
        {
            A += momentum_relativity_term;
            b -= momentum_relativity_term * right_state_.Erad * right_state_.density;
        }
        else
            A += 2 * momentum_relativity_term;
    }
}

void DiffusionXInflowBoundary::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        E_outside = left_state_.Erad * left_state_.density;
        v_outside = left_state_.velocity;
    }
    else
    {
        if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4))
        {
            E_outside = right_state_.Erad * right_state_.density;
            v_outside = right_state_.velocity;
        }
        else
        {
            E_outside = new_E[index];
            Vector3D normal = normalize(tess.GetMeshPoint(outside_point) - tess.GetMeshPoint(index));
            v_outside = cells[index].velocity;
            v_outside -= 2 * normal * ScalarProd(normal, v_outside);
        }
    }
}

void DiffusionOpenBoundary::SetBoundaryValues(Tessellation3D const& /*tess*/, size_t const /*index*/, size_t const /*outside_point*/, double const dt,
    std::vector<ComputationalCell3D> const& /*cells*/, double const Area, double& A, double& /*b*/, size_t const /*face_index*/)const
{
    A += Area * dt * 0.5 * CG::speed_of_light;
}

void DiffusionOpenBoundary::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    E_outside = new_E[index] * 1e-20;
    v_outside = cells[index].velocity;
}

void DiffusionOpenBoundary::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
    ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, 
    double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck)const
{
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * (2 * 3 * 
        sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    A += momentum_relativity_term;
}
