#include "MultigroupDiffusionBoundaryCalculator.hpp"
#include "boost/math/special_functions/pow.hpp"
#include "RadiationDriver.hpp"
#include "planck_integral/planck_integral.hpp"
#include "Diffusion.hpp" // for CalcSingleFluxLimiter


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
        Ug[g] = planck_integral::planck_energy_density_group_integral(energy_groups_boundary[g], energy_groups_boundary[g+1], temperature);
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
        double const D_dr_boundary = 0.5 * CG::speed_of_light * dt * Area;
        A += D_dr_boundary;
        b += D_dr_boundary * Ur;
    } 
}

void MultigroupDiffusionSideBoundary::getOutsideValuesGray(Tessellation3D const& tess,
                                                           std::size_t const index,
                                                           std::size_t const outside_point,
                                                           std::vector<ComputationalCell3D> const& cells,
                                                           double const Er_i,
                                                           double& Er_outside,
                                                           Vector3D& v_outside) const {
    double const R = tess.GetWidth(index);

    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4)){
        Er_outside = Ur;
    } else {
        Er_outside = Er_i;
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
        double const D_dr_boundary = 0.5 * CG::speed_of_light * dt * Area;
        A += D_dr_boundary;
        b += D_dr_boundary*Ug[group];
    }
}

void MultigroupDiffusionSideBoundary::getOutsideValuesGroup(std::size_t const group,
                                                            Tessellation3D const& tess,
                                                            std::size_t const index,
                                                            std::size_t const outside_point,
                                                            std::vector<ComputationalCell3D> const& cells,
                                                            double const Eg_i,
                                                            double& Eg_outside,
                                                            Vector3D& /*v_outside*/) const {
    
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R*1e-4)){
        Eg_outside = Ug[group];
    } else {
        Eg_outside = Eg_i;
    }
}

void MultigroupDiffusionSideBoundary::setMomentumTermBoundaryGray(Tessellation3D const& tess,
                                                                  std::size_t const index,
                                                                  std::size_t const outside_point,
                                                                  double dt,
                                                                  std::vector<ComputationalCell3D> const& cells,
                                                                  double const momentum_term_coefficient_i,
                                                                  double const momentum_term_coefficient_j,
                                                                  double& A,
                                                                  double& b) const {    
    double const R = tess.GetWidth(index);
    
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R*1e-4)){
        A += momentum_term_coefficient_i;
        b -= momentum_term_coefficient_j * Ur;
    } else {
        A +=  (momentum_term_coefficient_i + momentum_term_coefficient_j);
    }
}

void MultigroupDiffusionSideBoundary::setMomentumTermBoundaryGroup(std::size_t const group,
                                                                  Tessellation3D const& tess,
                                                                  std::size_t const index,
                                                                  std::size_t const outside_point,
                                                                  double dt,
                                                                  std::vector<ComputationalCell3D> const& cells,
                                                                  double const momentum_term_coefficient,
                                                                  double& A,
                                                                  double& b) const {
    double const R = tess.GetWidth(index);

    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R*1e-4)){
        A += momentum_term_coefficient;
        b -= momentum_term_coefficient * Ug[group];
    } else {
        A += 2.0 * momentum_term_coefficient;
    }
}

MultigroupDiffusionXInflowBoundary::MultigroupDiffusionXInflowBoundary(ComputationalCell3D const& left_state,
                                                                       ComputationalCell3D const& right_state,
                                                                       MultigroupDiffusionCoefficientCalculator const& coefficient_calculator) : left_state_(left_state),
                                                                                                 right_state_(right_state),
                                                                                                 coefficient_calculator_(coefficient_calculator),
                                                                                                 MultigroupDiffusionBoundaryCalculator(std::vector<double>(), std::vector<double>()) {}

void MultigroupDiffusionXInflowBoundary::setBoundaryValuesGroup(std::size_t const group,
                                                                Tessellation3D const& tess,
                                                                std::size_t const index, 
                                                                std::size_t const outside_point,
                                                                double const dt,
                                                                std::vector<ComputationalCell3D> const& cells,
                                                                double const Area,
                                                                double& A,
                                                                double& b,
                                                                std::size_t const face_index) const {
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R*1e-4)){
        double const Eg_j = left_state_.Eg[group] * left_state_.density;
        double const Eg_i = cells[index].Eg[group] * cells[index].density;

        double const dx = tess.GetCellCM(index).x - tess.GetBoxCoordinates().first.x;
        Vector3D const gradient = Vector3D(1.0, 0.0, 0.0) * (1.0 / (2.0 * dx));
        Vector3D const r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);

        // max temperature?
        double const Dg_i = coefficient_calculator_.CalcDiffusionCoefficientGroup(cells[index], group);
        double const Dg_j = coefficient_calculator_.CalcDiffusionCoefficientGroup(left_state_, group);
        
        // this is supposed to be harmonic probably but for the sake of sake
        double const Dg_ij = 0.5 * (Dg_i + Dg_j);
        
        // what if there is no flux limiter?    
        double const lambda_g_ij = CG::CalcSingleFluxLimiter(gradient * (Eg_i - Eg_j), Dg_ij, 0.5*(Eg_i + Eg_j));

        double const lambda_gDg_ij = lambda_g_ij * Dg_ij;

        // WARNING: This only works if the length scale is 1.0 since tess.GetArea and r_ij are not to scaled! 
        double const flux = ScalarProd(gradient, r_ij * (Area / abs(r_ij))) * dt * lambda_gDg_ij;
        
        A += flux;
        b += flux * Eg_j;
    } else if (tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R*1e-4)){
        double const Eg_j = right_state_.Eg[group] * right_state_.density;
        double const Eg_i = cells[index].Eg[group] * cells[index].density;

        
        double const dx = tess.GetBoxCoordinates().second.x - tess.GetCellCM(index).x;
        Vector3D const gradient = Vector3D(-1.0, 0.0, 0.0) * (1.0 / (2.0 * dx));
        Vector3D const r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
        
        // max temperature?
        double const Dg_i = coefficient_calculator_.CalcDiffusionCoefficientGroup(cells[index], group);
        double const Dg_j = coefficient_calculator_.CalcDiffusionCoefficientGroup(right_state_, group);
        
        // this is supposed to be harmonic probably but for the sake of sake
        double Dg_ij = 0.5 * (Dg_i + Dg_j);

        // what if there is no flux limiter?    
        double const lambda_g_ij = CG::CalcSingleFluxLimiter(gradient * (Eg_i - Eg_j), Dg_ij, 0.5*(Eg_i + Eg_j));
        double const lambda_gDg_ij = lambda_g_ij * Dg_ij;
        
        // WARNING: This only works if the length scale is 1.0 since tess.GetArea and r_ij are not to scaled! 
        double const flux = ScalarProd(gradient, r_ij * (Area / abs(r_ij))) * dt * lambda_gDg_ij;
        
        A += flux;
        b += flux * Eg_j;
    }
}

void MultigroupDiffusionXInflowBoundary::getOutsideValuesGroup(std::size_t const group,
                                                               Tessellation3D const& tess,
                                                               std::size_t const index,
                                                               std::size_t const outside_point,
                                                               std::vector<ComputationalCell3D> const& cells,
                                                               double const Eg_i,
                                                               double& Eg_outside,
                                                               Vector3D& v_outside) const {
    double const R = tess.GetWidth(index);

    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4)){
        Eg_outside = left_state_.Eg[group] * left_state_.density;
        v_outside  = left_state_.velocity;
    } else if (tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4)){
        Eg_outside = right_state_.Eg[group] * right_state_.density;
        v_outside  = right_state_.velocity;
    } else {
        Eg_outside = Eg_i;
        Vector3D const normal = normalize(tess.GetMeshPoint(outside_point) - tess.GetMeshPoint(index));

        v_outside = cells[index].velocity;
        v_outside -=  normal * (2.0 * ScalarProd(normal, v_outside));
    }       
}

void MultigroupDiffusionXInflowBoundary::setBoundaryValuesGray(Tessellation3D const& tess,
                                                               std::size_t const index,
                                                               std::size_t const outside_point,
                                                               double const dt,
                                                               std::vector<ComputationalCell3D> const& cells,
                                                               double const Area, 
                                                               double& A,
                                                               double& b,
                                                               std::size_t const face_index) const {
    double const R = tess.GetWidth(index);

    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R*1e-4)){
        double const dx = tess.GetCellCM(index).x - tess.GetBoxCoordinates().first.x;
        Vector3D const gradient = Vector3D(1.0, 0.0, 0.0) * (1.0 / (2.0 * dx));
        Vector3D const r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);

        double lambdaD_i = 0.0;
        double lambdaD_j = 0.0;
        double sum_Eg_i = 0.0;
        double sum_Eg_j = 0.0;
        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g){
            double const Eg_j = left_state_.Eg[g] * left_state_.density;
            double const Eg_i = cells[index].Eg[g] * cells[index].density;

            double const Dg_i = coefficient_calculator_.CalcDiffusionCoefficientGroup(cells[index], g);
            double const Dg_j = coefficient_calculator_.CalcDiffusionCoefficientGroup(left_state_, g);

            double Dg_ij = 0.5 * (Dg_i + Dg_j);
            double const lambda_g_ij = CG::CalcSingleFluxLimiter(gradient * (Eg_i - Eg_j), Dg_ij, 0.5*(Eg_i + Eg_j));

            double const  lambda_gDg_ij = lambda_g_ij * Dg_ij;

            lambdaD_i += lambda_gDg_ij * Eg_i;
            lambdaD_j += lambda_gDg_ij * Eg_j;

            sum_Eg_i += Eg_i;
            sum_Eg_j += Eg_j;
        }

        lambdaD_i /= sum_Eg_i;
        lambdaD_j /= sum_Eg_j;

        double const flux_coefficient = ScalarProd(gradient, r_ij * (Area / abs(r_ij))) * dt;
        
        double const flux_i_to_j = flux_coefficient * lambdaD_i;
        double const flux_j_to_i = flux_coefficient * lambdaD_j;

        A += flux_i_to_j;
        b += flux_j_to_i * left_state_.Erad * left_state_.density;

    } else if (tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R*1e-4)){
        double const dx = tess.GetBoxCoordinates().second.x - tess.GetCellCM(index).x;
        Vector3D const gradient = Vector3D(-1.0, 0.0, 0.0) * (1.0 / (2.0 * dx));
        Vector3D const r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);

        double lambdaD_i = 0.0;
        double lambdaD_j = 0.0;
        double sum_Eg_i = 0.0;
        double sum_Eg_j = 0.0;

        for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
            double const Eg_j = right_state_.Eg[g] * right_state_.density;
            double const Eg_i = cells[index].Eg[g] * cells[index].density;

            double const Dg_i = coefficient_calculator_.CalcDiffusionCoefficientGroup(cells[index], g);
            double const Dg_j = coefficient_calculator_.CalcDiffusionCoefficientGroup(right_state_, g);

            double Dg_ij = 0.5 * (Dg_i + Dg_j);
            double const lambda_g_ij = CG::CalcSingleFluxLimiter(gradient * (Eg_i - Eg_j), Dg_ij, 0.5*(Eg_i + Eg_j));

            double const  lambda_gDg_ij = lambda_g_ij * Dg_ij;

            lambdaD_i += lambda_gDg_ij * Eg_i;
            lambdaD_j += lambda_gDg_ij * Eg_j;

            sum_Eg_i += Eg_i;
            sum_Eg_j += Eg_j;
        }
        

        lambdaD_i /= sum_Eg_i;
        lambdaD_j /= sum_Eg_j;

        double const flux_coefficient = ScalarProd(gradient, r_ij * (Area / abs(r_ij))) * dt;

        double const flux_i_to_j = flux_coefficient * lambdaD_i;
        double const flux_j_to_i = flux_coefficient * lambdaD_j;

        A += flux_i_to_j;
        b += flux_j_to_i * right_state_.Erad * right_state_.density;
    }
}

void MultigroupDiffusionXInflowBoundary::getOutsideValuesGray(Tessellation3D const& tess,
                                                              std::size_t const index,
                                                              std::size_t const outside_point,
                                                              std::vector<ComputationalCell3D> const& cells,
                                                              double const Er_i,
                                                              double& Er_outside,
                                                              Vector3D& v_outside) const { 
    double const R = tess.GetWidth(index);

    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4)){
        Er_outside = left_state_.Erad * left_state_.density;
        v_outside  = left_state_.velocity;
    } else if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4)) {
        Er_outside = right_state_.Erad;
        v_outside  = right_state_.velocity;
    } else {
        Er_outside = Er_i;

        Vector3D const normal = normalize(tess.GetMeshPoint(outside_point) - tess.GetMeshPoint(index));

        v_outside = cells[index].velocity;
        v_outside -=  normal * (2.0 * ScalarProd(normal, v_outside));
    }
}

void MultigroupDiffusionXInflowBoundary::setMomentumTermBoundaryGroup(std::size_t const group,
                                                                      Tessellation3D const& tess,
                                                                      std::size_t const index,
                                                                      std::size_t const outside_point,
                                                                      double dt,
                                                                      std::vector<ComputationalCell3D> const& cells,
                                                                      double const momentum_term_coefficient,
                                                                      double& A,
                                                                      double& b) const {

    double const R = tess.GetWidth(index);

    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4)){
        A += momentum_term_coefficient;
        b -= momentum_term_coefficient * left_state_.Eg[group] * left_state_.density;
    } else if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4)){
        A += momentum_term_coefficient;
        b -= momentum_term_coefficient * right_state_.Eg[group] * right_state_.density;
    } else {
        A += 2.0 * momentum_term_coefficient;
    }
}

void MultigroupDiffusionXInflowBoundary::setMomentumTermBoundaryGray(Tessellation3D const& tess,
                                                                     std::size_t const index,
                                                                     std::size_t const outside_point,
                                                                     double dt,
                                                                     std::vector<ComputationalCell3D> const& cells,
                                                                     double const momentum_term_coefficient_i,
                                                                     double const momentum_term_coefficient_j,
                                                                     double& A,
                                                                     double& b) const {
    
    double const R = tess.GetWidth(index);
    
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R*1e-4)){
        A += momentum_term_coefficient_i;
        b -= momentum_term_coefficient_i * left_state_.Erad * left_state_.density;
    } else if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R*1e-4)) {
        A += momentum_term_coefficient_j;
        b -= momentum_term_coefficient_j * right_state_.Erad * right_state_.density;
    } else {
        A += momentum_term_coefficient_i + momentum_term_coefficient_j;
    }

}

void MultigroupDiffusionOpenBoundary::setBoundaryValuesGroup(std::size_t const group,
                            Tessellation3D const& tess,
                            std::size_t const index, 
                            std::size_t const outside_point,
                            double const dt,
                            std::vector<ComputationalCell3D> const& cells,
                            double const Area,
                            double& A,
                            double& b,
                            std::size_t const face_index) const {
    A += Area * dt * 0.5 * CG::speed_of_light;
}

void MultigroupDiffusionOpenBoundary::getOutsideValuesGroup(std::size_t const group,
                            Tessellation3D const& tess,
                            std::size_t const index,
                            std::size_t const outside_point,
                            std::vector<ComputationalCell3D> const& cells,
                            double const Eg_i,
                            double& Eg_outside,
                            Vector3D& v_outside) const {
    Eg_outside = Eg_i * 1e-20;
    v_outside = cells[index].velocity;
}

void MultigroupDiffusionOpenBoundary::setMomentumTermBoundaryGroup(std::size_t const group,
                                    Tessellation3D const& tess,
                                    std::size_t const index,
                                    std::size_t const outside_point,
                                    double dt,
                                    std::vector<ComputationalCell3D> const& cells,
                                    double const momentum_term_coefficient,
                                    double& A,
                                    double& b) const {
    A += momentum_term_coefficient;
}

void MultigroupDiffusionOpenBoundary::setBoundaryValuesGray(Tessellation3D const& tess,
                            std::size_t const index,
                            std::size_t const outside_point,
                            double const dt,
                            std::vector<ComputationalCell3D> const& cells,
                            double const Area, 
                            double& A,
                            double& b,
                            std::size_t const face_index) const {
    A += Area * dt * 0.5 * CG::speed_of_light;
}

void MultigroupDiffusionOpenBoundary::getOutsideValuesGray(Tessellation3D const& tess,
                            std::size_t const index,
                            std::size_t const outside_point,
                            std::vector<ComputationalCell3D> const& cells,
                            double const Er_i,
                            double& Er_outside,
                            Vector3D& v_outside) const {
    Er_outside = Er_i * 1e-20;
    v_outside = cells[index].velocity;
}

void MultigroupDiffusionOpenBoundary::setMomentumTermBoundaryGray(Tessellation3D const& tess,
                                    std::size_t const index,
                                    std::size_t const outside_point,
                                    double dt,
                                    std::vector<ComputationalCell3D> const& cells,
                                    double const momentum_term_coefficient_i,
                                    double const momentum_term_coefficient_j,
                                    double& A,
                                    double& b) const {
    A += momentum_term_coefficient_i;
}

    void MultigroupDiffusionClosedBoundary::setBoundaryValuesGroup(std::size_t const group,
                                Tessellation3D const& tess,
                                std::size_t const index, 
                                std::size_t const outside_point,
                                double const dt,
                                std::vector<ComputationalCell3D> const& cells,
                                double const Area,
                                double& A,
                                double& b,
                                std::size_t const face_index) const {}
    
    void MultigroupDiffusionClosedBoundary::getOutsideValuesGroup(std::size_t const group,
                               Tessellation3D const& tess,
                               std::size_t const index,
                               std::size_t const outside_point,
                               std::vector<ComputationalCell3D> const& cells,
                               double const Eg_i,
                               double& Eg_outside,
                               Vector3D& v_outside) const {
    Eg_outside = Eg_i;
}
    
    void MultigroupDiffusionClosedBoundary::setMomentumTermBoundaryGroup(std::size_t const group,
                                      Tessellation3D const& tess,
                                      std::size_t const index,
                                      std::size_t const outside_point,
                                      double dt,
                                      std::vector<ComputationalCell3D> const& cells,
                                      double const momentum_term_coefficient,
                                      double& A,
                                      double& b) const {
    A += 2 * momentum_term_coefficient;
}
    
    void MultigroupDiffusionClosedBoundary::setBoundaryValuesGray(Tessellation3D const& tess,
                               std::size_t const index,
                               std::size_t const outside_point,
                               double const dt,
                               std::vector<ComputationalCell3D> const& cells,
                               double const Area, 
                               double& A,
                               double& b,
                               std::size_t const face_index) const {}

    void MultigroupDiffusionClosedBoundary::getOutsideValuesGray(Tessellation3D const& tess,
                              std::size_t const index,
                              std::size_t const outside_point,
                              std::vector<ComputationalCell3D> const& cells,
                              double const Er_i,
                              double& Er_outside,
                              Vector3D& v_outside) const {
    Er_outside = Er_i;
    v_outside = cells[index].velocity;
}

    void MultigroupDiffusionClosedBoundary::setMomentumTermBoundaryGray(Tessellation3D const& tess,
                                     std::size_t const index,
                                     std::size_t const outside_point,
                                     double dt,
                                     std::vector<ComputationalCell3D> const& cells,
                                     double const momentum_term_coefficient_i,
                                     double const momentum_term_coefficient_j,
                                     double& A,
                                     double& b) const {
    A += (momentum_term_coefficient_i + momentum_term_coefficient_j);
}