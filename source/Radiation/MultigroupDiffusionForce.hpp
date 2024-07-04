#ifndef MULTIGROUP_DIFFUSION_FORCE_HPP
#define MULTIGROUP_DIFFUSION_FORCE_HPP

#include "../newtonian/three_dimensional/SourceTerm3D.hpp"
#include "MultigroupDiffusion.hpp"

class MultigroupDiffusionForce : public SourceTerm3D {
    public:
        MultigroupDiffusionForce(MultigroupDiffusion const& multigroup_diffusion,
                                 EquationOfState const& eos,
                                 bool const momentum_limit=true);

        void operator()(Tessellation3D const& tess, 
                        std::vector<ComputationalCell3D> const& cells,
                        std::vector<Conserved3D> const& fluxes,
                        std::vector<Vector3D> const& point_velocities, 
                        double const t,
                        double const dt,
                        std::vector<Conserved3D>& extensives) const override;

        double SuggestInverseTimeStep(void) const;
    
    private:
        MultigroupDiffusion const& multigroup_diffusion_;
        mutable double next_dt_;
        EquationOfState const& eos_;
        bool const momentum_limit_;
};

#endif // MULTIGROUP_DIFFUSION_FORCE_HPP 