#include "DiagnosticAppendix3D.hpp"
#include "source/Radiation/MultigroupDiffusion.hpp"

class EnergyGroupToLabFrame : public DiagnosticAppendix3D {
    private:
        MultigroupDiffusion const& multigroup_;
        std::size_t const group;
    public:
        EnergyGroupToLabFrame(MultigroupDiffusion const& multigroup, std::size_t const group);

        std::string getName(void) const override;        
        
        std::vector<double> operator()(HDSim3D const& sim) const override;

};