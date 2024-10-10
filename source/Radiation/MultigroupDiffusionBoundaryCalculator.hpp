#ifndef MULTI_GROUP_DIFFUSION_BOUNDARY_CALCULATOR_HPP
#define MULTI_GROUP_DIFFUSION_BOUNDARY_CALCULATOR_HPP

#include "conj_grad_solve.hpp"
#include "MultigroupDiffusionCoefficientCalculator.hpp"
#include "planck_integral/planck_integral.hpp"

class MultigroupDiffusionBoundaryCalculator {
public:
    MultigroupDiffusionBoundaryCalculator(std::vector<double> const& energy_groups_center_,
                                          std::vector<double> const& energy_groups_boundary_);

    MultigroupDiffusionBoundaryCalculator() = default;

    virtual ~MultigroupDiffusionBoundaryCalculator() = default;

    virtual void setBoundaryValuesGroup(std::size_t const group,
                                        Tessellation3D const& tess,
                                        std::size_t const index, 
                                        std::size_t const outside_point,
                                        double const dt,
                                        std::vector<ComputationalCell3D> const& cells,
                                        double const Area,
                                        double& A,
                                        double& b,
                                        std::size_t const face_index) const = 0;
    
    virtual void getOutsideValuesGroup(std::size_t const group,
                                       Tessellation3D const& tess,
                                       std::size_t const index,
                                       std::size_t const outside_point,
                                       std::vector<ComputationalCell3D> const& cells,
                                       double const Eg_i,
                                       double& Eg_outside,
                                       Vector3D& v_outside) const = 0;
    
    
    virtual void setMomentumTermBoundaryGroup(std::size_t const group,
                                              Tessellation3D const& tess,
                                              std::size_t const index,
                                              std::size_t const outside_point,
                                              double dt,
                                              std::vector<ComputationalCell3D> const& cells,
                                              double const momentum_term_coefficient,
                                              double& A,
                                              double& b) const = 0;

    virtual void setBoundaryValuesGray(Tessellation3D const& tess,
                                       std::size_t const index,
                                       std::size_t const outside_point,
                                       double const dt,
                                       std::vector<ComputationalCell3D> const& cells,
                                       double const Area, 
                                       double& A,
                                       double& b,
                                       std::size_t const face_index) const = 0;

    virtual void getOutsideValuesGray(Tessellation3D const& tess,
                                      std::size_t const index,
                                      std::size_t const outside_point,
                                      std::vector<ComputationalCell3D> const& cells,
                                      double const Er_i,
                                      double& Er_outside,
                                      Vector3D& v_outside) const = 0;
    
    virtual void setMomentumTermBoundaryGray(Tessellation3D const& tess,
                                             std::size_t const index,
                                             std::size_t const outside_point,
                                             double dt,
                                             std::vector<ComputationalCell3D> const& cells,
                                             double const momentum_term_coefficient_i,
                                             double const momentum_term_coefficient_j, 
                                             double& A,
                                             double& b) const = 0;

    std::vector<double> const energy_groups_center;
    std::vector<double> const energy_groups_boundary;

};

class MultigroupDiffusionSideBoundary : public MultigroupDiffusionBoundaryCalculator {
public:

    MultigroupDiffusionSideBoundary(double const temperature_,
                                    std::vector<double> const& energy_group_center_,
                                    std::vector<double> const& energy_group_boundary_);

    ~MultigroupDiffusionSideBoundary() = default;

    void setBoundaryValuesGroup(std::size_t const group,
                                Tessellation3D const& tess,
                                std::size_t const index, 
                                std::size_t const outside_point,
                                double const dt,
                                std::vector<ComputationalCell3D> const& cells,
                                double const Area,
                                double& A,
                                double& b,
                                std::size_t const face_index) const override;
    
    void getOutsideValuesGroup(std::size_t const group,
                               Tessellation3D const& tess,
                               std::size_t const index,
                               std::size_t const outside_point,
                               std::vector<ComputationalCell3D> const& cells,
                               double const Eg_i,
                               double& Eg_outside,
                               Vector3D& v_outside) const override;
    
    void setMomentumTermBoundaryGroup(std::size_t const group,
                                      Tessellation3D const& tess,
                                      std::size_t const index,
                                      std::size_t const outside_point,
                                      double dt,
                                      std::vector<ComputationalCell3D> const& cells,
                                      double const momentum_term_coefficient,
                                      double& A,
                                      double& b) const override;
    
    void setBoundaryValuesGray(Tessellation3D const& tess,
                               std::size_t const index,
                               std::size_t const outside_point,
                               double const dt,
                               std::vector<ComputationalCell3D> const& cells,
                               double const Area, 
                               double& A,
                               double& b,
                               std::size_t const face_index) const override;

    void getOutsideValuesGray(Tessellation3D const& tess,
                              std::size_t const index,
                              std::size_t const outside_point,
                              std::vector<ComputationalCell3D> const& cells,
                              double const Er_i,
                              double& Er_outside,
                              Vector3D& v_outside) const override;
    
    void setMomentumTermBoundaryGray(Tessellation3D const& tess,
                                     std::size_t const index,
                                     std::size_t const outside_point,
                                     double dt,
                                     std::vector<ComputationalCell3D> const& cells,
                                     double const momentum_term_coefficient_i,
                                     double const momentum_term_coefficient_j,
                                     double& A,
                                     double& b) const override;
    
    void SetTemperature(double const temperature_);

    private:
        double temperature;
        double Ur;
        mutable std::vector<double> Ug;
};


class MultigroupDiffusionXInflowBoundary : public MultigroupDiffusionBoundaryCalculator {
    public:
        MultigroupDiffusionXInflowBoundary(ComputationalCell3D const& left_state,
                                           ComputationalCell3D const& right_state,
                                           MultigroupDiffusionCoefficientCalculator const& coefficient_calculator);

        void setBoundaryValuesGroup(std::size_t const group,
                                    Tessellation3D const& tess,
                                    std::size_t const index, 
                                    std::size_t const outside_point,
                                    double const dt,
                                    std::vector<ComputationalCell3D> const& cells,
                                    double const Area,
                                    double& A,
                                    double& b,
                                    std::size_t const face_index) const override;
        
        void getOutsideValuesGroup(std::size_t const group,
                                Tessellation3D const& tess,
                                std::size_t const index,
                                std::size_t const outside_point,
                                std::vector<ComputationalCell3D> const& cells,
                                double const Eg_i,
                                double& Eg_outside,
                                Vector3D& v_outside) const override;
        
        void setMomentumTermBoundaryGroup(std::size_t const group,
                                        Tessellation3D const& tess,
                                        std::size_t const index,
                                        std::size_t const outside_point,
                                        double dt,
                                        std::vector<ComputationalCell3D> const& cells,
                                        double const momentum_term_coefficient,
                                        double& A,
                                        double& b) const override;
        
        void setBoundaryValuesGray(Tessellation3D const& tess,
                                std::size_t const index,
                                std::size_t const outside_point,
                                double const dt,
                                std::vector<ComputationalCell3D> const& cells,
                                double const Area, 
                                double& A,
                                double& b,
                                std::size_t const face_index) const override;

        void getOutsideValuesGray(Tessellation3D const& tess,
                                std::size_t const index,
                                std::size_t const outside_point,
                                std::vector<ComputationalCell3D> const& cells,
                                double const Er_i,
                                double& Er_outside,
                                Vector3D& v_outside) const override;
        
        void setMomentumTermBoundaryGray(Tessellation3D const& tess,
                                        std::size_t const index,
                                        std::size_t const outside_point,
                                        double dt,
                                        std::vector<ComputationalCell3D> const& cells,
                                        double const momentum_term_coefficient_i,
                                        double const momentum_term_coefficient_j,
                                        double& A,
                                        double& b) const override;
    private:
        ComputationalCell3D const& left_state_, right_state_;
        MultigroupDiffusionCoefficientCalculator const& coefficient_calculator_;
};

class MultigroupDiffusionOpenBoundary : public MultigroupDiffusionBoundaryCalculator {
public:

    MultigroupDiffusionOpenBoundary() = default;

    ~MultigroupDiffusionOpenBoundary() = default;

    void setBoundaryValuesGroup(std::size_t const group,
                                Tessellation3D const& tess,
                                std::size_t const index, 
                                std::size_t const outside_point,
                                double const dt,
                                std::vector<ComputationalCell3D> const& cells,
                                double const Area,
                                double& A,
                                double& b,
                                std::size_t const face_index) const override;
    
    void getOutsideValuesGroup(std::size_t const group,
                               Tessellation3D const& tess,
                               std::size_t const index,
                               std::size_t const outside_point,
                               std::vector<ComputationalCell3D> const& cells,
                               double const Eg_i,
                               double& Eg_outside,
                               Vector3D& v_outside) const override;
    
    void setMomentumTermBoundaryGroup(std::size_t const group,
                                      Tessellation3D const& tess,
                                      std::size_t const index,
                                      std::size_t const outside_point,
                                      double dt,
                                      std::vector<ComputationalCell3D> const& cells,
                                      double const momentum_term_coefficient,
                                      double& A,
                                      double& b) const override;
    
    void setBoundaryValuesGray(Tessellation3D const& tess,
                               std::size_t const index,
                               std::size_t const outside_point,
                               double const dt,
                               std::vector<ComputationalCell3D> const& cells,
                               double const Area, 
                               double& A,
                               double& b,
                               std::size_t const face_index) const override;

    void getOutsideValuesGray(Tessellation3D const& tess,
                              std::size_t const index,
                              std::size_t const outside_point,
                              std::vector<ComputationalCell3D> const& cells,
                              double const Er_i,
                              double& Er_outside,
                              Vector3D& v_outside) const override;
    
    void setMomentumTermBoundaryGray(Tessellation3D const& tess,
                                     std::size_t const index,
                                     std::size_t const outside_point,
                                     double dt,
                                     std::vector<ComputationalCell3D> const& cells,
                                     double const momentum_term_coefficient_i,
                                     double const momentum_term_coefficient_j,
                                     double& A,
                                     double& b) const override;
};

class MultigroupDiffusionClosedBoundary : public MultigroupDiffusionBoundaryCalculator {
public:

    MultigroupDiffusionClosedBoundary() = default;

    ~MultigroupDiffusionClosedBoundary() = default;

    void setBoundaryValuesGroup(std::size_t const group,
                                Tessellation3D const& tess,
                                std::size_t const index, 
                                std::size_t const outside_point,
                                double const dt,
                                std::vector<ComputationalCell3D> const& cells,
                                double const Area,
                                double& A,
                                double& b,
                                std::size_t const face_index) const override;
    
    void getOutsideValuesGroup(std::size_t const group,
                               Tessellation3D const& tess,
                               std::size_t const index,
                               std::size_t const outside_point,
                               std::vector<ComputationalCell3D> const& cells,
                               double const Eg_i,
                               double& Eg_outside,
                               Vector3D& v_outside) const override;
    
    void setMomentumTermBoundaryGroup(std::size_t const group,
                                      Tessellation3D const& tess,
                                      std::size_t const index,
                                      std::size_t const outside_point,
                                      double dt,
                                      std::vector<ComputationalCell3D> const& cells,
                                      double const momentum_term_coefficient,
                                      double& A,
                                      double& b) const override;
    
    void setBoundaryValuesGray(Tessellation3D const& tess,
                               std::size_t const index,
                               std::size_t const outside_point,
                               double const dt,
                               std::vector<ComputationalCell3D> const& cells,
                               double const Area, 
                               double& A,
                               double& b,
                               std::size_t const face_index) const override;

    void getOutsideValuesGray(Tessellation3D const& tess,
                              std::size_t const index,
                              std::size_t const outside_point,
                              std::vector<ComputationalCell3D> const& cells,
                              double const Er_i,
                              double& Er_outside,
                              Vector3D& v_outside) const override;
    
    void setMomentumTermBoundaryGray(Tessellation3D const& tess,
                                     std::size_t const index,
                                     std::size_t const outside_point,
                                     double dt,
                                     std::vector<ComputationalCell3D> const& cells,
                                     double const momentum_term_coefficient_i,
                                     double const momentum_term_coefficient_j,
                                     double& A,
                                     double& b) const override;
};

#endif // MULTI_GROUP_DIFFUSION_BOUNDARY_CALCULATOR_HPP