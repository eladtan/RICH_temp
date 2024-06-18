#ifndef MULTI_GROUP_DIFFUSION_BOUNDARY_CALCULATOR_HPP
#define MULTI_GROUP_DIFFUSION_BOUNDARY_CALCULATOR_HPP

#include "conj_grad_solve.hpp"

class MultigroupDiffusionBoundaryCalculator {
public:
    MultigroupDiffusionBoundaryCalculator(std::vector<double> const& energy_groups_center_,
                                          std::vector<double> const& energy_groups_boundary_);

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
    
    virtual void getOutSideValuesGroup(std::size_t const group,
                                       Tessellation3D const& tess,
                                       std::vector<ComputationalCell3D> const& cells,
                                       std::size_t const index,
                                       std::size_t const outside_point,
                                       std::vector<double> const& new_Eg,
                                       double& Eg_outside,
                                       Vector3D& v_outside) const = 0;
    
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
                                      std::vector<double> const& new_Er,
                                      double& Er_outside,
                                      Vector3D& v_outside) const = 0;

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
    
    void getOutSideValuesGroup(std::size_t const group,
                               Tessellation3D const& tess,
                               std::vector<ComputationalCell3D> const& cells,
                               std::size_t const index,
                               std::size_t const outside_point,
                               std::vector<double> const& new_Eg,
                               double& Eg_outside,
                               Vector3D& v_outside) const override;
    
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
                              std::vector<double> const& new_Er,
                              double& Er_outside,
                              Vector3D& v_outside) const override;

    private:
        double const temperature;
        double const Ur;
        mutable std::vector<double> Ug;
};

#endif // MULTI_GROUP_DIFFUSION_BOUNDARY_CALCULATOR_HPP