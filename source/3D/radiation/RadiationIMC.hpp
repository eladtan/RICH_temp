#ifndef RADIATION_IMC_HPP
#define RADIATION_IMC_HPP

#include <array>
#include <memory>
#include "MonteCarloPhysics3D.hpp"
#include "MultigroupOpacity.hpp"
#include "3D/monte/Voronoi3DMovement.hpp"
#include "RandomWalk.hpp"
#include "Radiation/CMMC/src/compton_matrix_mc.hpp"

struct RadiationIMCParameters
{
    size_t newPhotonsPerCell;
    bool withHydro = false;
    bool diffusionPressureGradient = false;
    bool MMC = false;
    bool withMultigroupOpacity = false;
    bool withRandomWalk = false;
    double rwMinCellOpticalDepth = 25.0;
    double rwMinParticleOpticalDepth = 5.0;
    bool noHydroFeedback = false;
    bool withEgTimeAvg = false;
    bool withCompton = false;
    bool comptonUseInduced = true;
    bool comptonAllowNZeroFallback = true;
    size_t comptonMatrixSamples = 200000;

    friend std::ostream &operator<<(std::ostream &os, const RadiationIMCParameters &parameters);
};

class RadiationIMC : public MonteCarloRadiationPhysics3D
{
public:
    using Particle = MonteCarloParticle<Vector3D, Tessellation3D>;
    using Functionality = MonteCarloFunctionality<Vector3D, Tessellation3D>;
    using BoundaryCond = BoundaryCondition<Vector3D, Tessellation3D>;
    using GroupArray = std::array<double, ENERGY_GROUPS_NUM>;
    using GroupCdf = std::array<double, ENERGY_GROUPS_NUM + 1>;
    using GroupMatrix = std::array<GroupArray, ENERGY_GROUPS_NUM>;
    using GroupCdfMatrix = std::array<GroupCdf, ENERGY_GROUPS_NUM>;

    struct ComptonCellData
    {
        GroupArray absorptionOpacity{};
        GroupArray planckFraction{};
        GroupArray baseSourceFraction{};
        GroupCdf planckCdf{};
        GroupCdf baseSourceCdf{};
        bool active = false;
        double planckOpacity = 0.0;
        double volume = 0.0;
        double temperature = 0.0;
        double Um = 0.0;
        double beta = 0.0;
        double cv = 0.0;
        double fleck = 0.0;
        double Upsilon = 0.0;
        double Gamma = 0.0;
        double betaCdtF = 0.0;
        bool useNZero = false;
        GroupArray oldRadiationEnergy{};
        GroupArray occupation{};
        GroupArray D{};
        GroupArray M{};
        GroupArray rowS{};
        GroupArray Lambda{};
        GroupArray Bbase{};
        GroupArray Bcorr{};
        GroupArray Btotal{};
        GroupArray Bpos{};
        GroupArray Bres{};
        GroupArray baseEffectiveOpacity{};
        GroupArray comptonOutRate{};
        GroupCdfMatrix comptonTargetCdf{};
        GroupMatrix tau{};
        GroupMatrix dtau_dUm{};
        GroupMatrix S{};
        GroupMatrix dSdUm{};
        GroupMatrix segmentKernel{};
        GroupMatrix residualKernel{};
        GroupMatrix Ktotal{};
        GroupArray comptonMu{};
        GroupArray comptonMh{};
    };

    RadiationIMC(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, std::shared_ptr<EquationOfState> eos, std::shared_ptr<OpacityCalculator> opacity, RadiationIMCParameters parameters);

    std::vector<Particle> preStep(double fullDt) override;

    Functionality step(Particle &particle, std::vector<Particle> &particlesToAdd) override;

    void postStep(const std::vector<Particle> &particles, double fullDt) override;

    Particle generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const override;

    std::vector<Particle> generateInitialParticles(size_t particlesPerCell) override;

    void adjustExistingParticles(std::vector<Particle> &particles, double fullDt) override;

    inline const std::vector<double> &getFactorFleck(void) const{return this->factorFleck;}

    inline const std::vector<double> &getPlanckOpacities(void) const {return this->planckOpacities;}

    inline const std::vector<ComptonCellData> &getComptonData(void) const {return this->comptonData;}

    inline const GroupArray &getComptonGroupCenters(void) const {return this->comptonGroupCenters;}

    inline const GroupArray &getComptonGroupWidths(void) const {return this->comptonGroupWidths;}

private:    
    std::vector<Particle> generateParticles(double fullDt);
    std::vector<Particle> generateComptonParticles(double fullDt);
    void precomputeComptonData(double fullDt);
    void initializeComptonGroups();
    void initializeComptonMatrixGenerator();
    void buildComptonMatricesForCell(const ComputationalCell3D &cell, size_t cellIndex, bool calculateN, ComptonCellData &cd);
    void recomputeComptonContractions(ComptonCellData &cd);
    void buildComptonEventData(size_t cellIndex, ComptonCellData &cd);
    void buildComptonSources(double fullDt, ComptonCellData &cd);
    void applyComptonScatterEvent(size_t cellIndex, const ComputationalCell3D &cell, size_t sourceGroup, const Vector3D &oldVelocity, double oldWeight, double dopplerShift, Particle &particle);
    void applyComptonEndOfStepCorrection(double fullDt);
    void reconcileComptonParticles(std::vector<Particle> &particles);
    double frequencyForComptonGroup(size_t group) const;
    size_t sampleComptonCdf(const GroupCdf &cdf, double random) const;
    static GroupCdf buildSafeComptonCdf(const GroupArray &weights);

    std::vector<double> factorFleck;
    std::vector<double> planckOpacities;
    std::shared_ptr<MultigroupOpacity> multigroupOpacity;
    std::vector<ComptonCellData> comptonData;
    GroupArray comptonGroupCenters{};
    GroupArray comptonGroupWidths{};
    bool comptonGroupsInitialized = false;
    std::unique_ptr<ComptonMatrixMC> comptonMatrixGen;

    bool withHydro;
    bool diffusionPressureGradient;
    bool MMC;
    size_t newPhotonsPerCell;
    bool withRandomWalk;
    double rwMinCellOpticalDepth;
    double rwMinParticleOpticalDepth;
    bool noHydroFeedback;
    bool withEgTimeAvg;
    bool withCompton;
    bool comptonUseInduced;
    bool comptonAllowNZeroFallback;
    size_t comptonMatrixSamples;

    std::unique_ptr<RandomWalk> randomWalk;
    std::vector<bool> rwCellEligible;
    std::vector<double> rwCellTotalOpacity;
    std::vector<PGRWCellData> rwCellData;
    size_t rwStepCount = 0;

    bool tryRandomWalkStep(Particle &particle, Functionality &functionality, double dopplerShift);
    void precomputeRandomWalkData();
};

#endif // RADIATION_IMC_HPP
