#include "MultigroupDiffusionBoundaryCalculator.hpp"
#include "boost/math/special_functions/pow.hpp"
#include "RadiationDriver.hpp"
#include "planck_integral/planck_integral.hpp"


MultigroupDiffusionBoundaryCalculator
::MultigroupDiffusionBoundaryCalculator(std::vector<double> const& energy_groups_center_,
                                        std::vector<double> const& energy_groups_boundary_) : energy_groups_center(energy_groups_center_),
                                                                                              energy_groups_boundary(energy_groups_boundary_) {}

MultigroupDiffusionSideBoundary
::MultigroupDiffusionSideBoundary(double const temperature_,
                                  std::vector<double> const& energy_groups_center_,
                                  std::vector<double> const& energy_groups_boundary_) : temperature(temperature_),
                                                                                        Ur(get_radiation_energy_density(temperature_)),
                                                                                        Ug(ENERGY_GROUPS_NUM, 0.0),
                                                                                        MultigroupDiffusionBoundaryCalculator(energy_groups_center_, energy_groups_boundary_) {

    for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
        Ug[g] = planck_energy_density_group_integral(energy_groups_boundary[g], energy_groups_boundary[g+1], temperature);
    }
}

void MultigroupDiffusionSideBoundary::setBoundaryValuesGray(Tessellation3D const& tess,
                                                            std::size_t const index,
                                                            std::size_t const outside_point,
                                                            double const dt,
                                                            std::vector<ComputationalCell3D> const& /*cells*/,
                                                            double const Area, 
                                                            double& A,
                                                            double& b,
                                                            std::size_t const /*face_index*/) const {
    
    using boost::math::pow;

    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4)){
        A += 0.5 * CG::speed_of_light * dt * Area;
        b += 2.0 * Area * dt * Ur;
    } 
}

void MultigroupDiffusionSideBoundary::getOutsideValuesGray(Tessellation3D const& tess,
                                                           std::size_t const index,
                                                           std::size_t const outside_point,
                                                           std::vector<ComputationalCell3D> const& cells,
                                                           std::vector<double> const& new_Er,
                                                           double& Er_outside,
                                                           Vector3D& v_outside) const {
    double const R = tess.GetWidth(index);

    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4)){
        Er_outside = Ur;
    } else {
        Er_outside = new_Er[index];
    }

    v_outside = cells[index].velocity;
}

void MultigroupDiffusionSideBoundary::setBoundaryValuesGroup(std::size_t const group,
                                                             Tessellation3D const& tess,
                                                             std::size_t const index, 
                                                             std::size_t const outside_point,
                                                             double const dt,
                                                             std::vector<ComputationalCell3D> const& /*cells*/,
                                                             double const Area,
                                                             double& A,
                                                             double& b,
                                                             std::size_t const /*face_index*/) const {
    
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R*1e-4)){
        A += 0.5 * CG::speed_of_light * dt * Area;
        b += 2.0 * Area * dt * Ug[group];
    }
}

void MultigroupDiffusionSideBoundary::getOutSideValuesGroup(std::size_t const group,
                                                            Tessellation3D const& tess,
                                                            std::vector<ComputationalCell3D> const& cells,
                                                            std::size_t const index,
                                                            std::size_t const outside_point,
                                                            std::vector<double> const& new_Eg,
                                                            double& Eg_outside,
                                                            Vector3D& /*v_outside*/) const {
    
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R*1e-4)){
        Eg_outside = Ug[group];
    } else {
        Eg_outside = new_Eg[group];
    }
}

