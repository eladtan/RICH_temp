#include "RadiationIMC.hpp"
#include "mpi/mpi_commands_3d.hpp"
#include "Radiation/conj_grad_solve.hpp"
#include <cmath>
#include <map>

// #define MONTECARLO_EPS 1e-7

namespace {
    inline void ClampFrequencyToBounds(double &frequency)
    {
        frequency = std::clamp(frequency,
            ComputationalCell3D::energyBoundaries[0],
            ComputationalCell3D::energyBoundaries[ENERGY_GROUPS_NUM]);
    }

    inline void SetInitialWeightFromWeight(RadiationIMC::Particle &particle)
    {
        particle.initialWeight = std::abs(particle.weight);
    }

    std::vector<double> BuildComptonTemperatures()
    {
        std::vector<double> temperatures;
        temperatures.reserve(131);
        temperatures.push_back(0.0001 * units::kev_kelvin);
        temperatures.push_back(0.001 * units::kev_kelvin);
        temperatures.push_back(0.005 * units::kev_kelvin);
        for(size_t i = 0; i < 128; i++)
        {
            double const x = -2.0 + 6.0 * static_cast<double>(i) / 127.0;
            temperatures.push_back(std::pow(10.0, x) * units::kev_kelvin);
        }
        return temperatures;
    }

    std::vector<double> BuildComptonCentersVector()
    {
        std::vector<double> centers(ENERGY_GROUPS_NUM, 0.0);
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            centers[g] = 0.5 * (ComputationalCell3D::energyBoundaries[g] +
                                ComputationalCell3D::energyBoundaries[g + 1]);
        }
        return centers;
    }

    std::vector<double> BuildComptonBoundariesVector()
    {
        std::vector<double> boundaries(ENERGY_GROUPS_NUM + 1, 0.0);
        for(size_t g = 0; g <= ENERGY_GROUPS_NUM; g++)
        {
            boundaries[g] = ComputationalCell3D::energyBoundaries[g];
        }
        return boundaries;
    }

    template<class MatrixLike>
    void ZeroGroupMatrix(MatrixLike &matrix)
    {
        for(auto &row : matrix)
        {
            row.fill(0.0);
        }
    }
}

    RadiationIMC::RadiationIMC(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, std::shared_ptr<EquationOfState> eos, std::shared_ptr<OpacityCalculator> opacity, RadiationIMCParameters parameters)
    : MonteCarloRadiationPhysics3D(grid, boundary, cells, conserved, eos, opacity), withHydro(parameters.withHydro), diffusionPressureGradient(parameters.diffusionPressureGradient), MMC(parameters.MMC), newPhotonsPerCell(parameters.newPhotonsPerCell), withRandomWalk(parameters.withRandomWalk), rwMinCellOpticalDepth(parameters.rwMinCellOpticalDepth), rwMinParticleOpticalDepth(parameters.rwMinParticleOpticalDepth), noHydroFeedback(parameters.noHydroFeedback), withEgTimeAvg(parameters.withEgTimeAvg), withCompton(parameters.withCompton), comptonUseInduced(parameters.comptonUseInduced), comptonAllowNZeroFallback(parameters.comptonAllowNZeroFallback), comptonDebugParityCheck(parameters.comptonDebugParityCheck), comptonCheckSignedTallies(parameters.comptonCheckSignedTallies), comptonDiagnostics(parameters.comptonDiagnostics), comptonSignedTallyTolerance(parameters.comptonSignedTallyTolerance), comptonMatrixSamples(parameters.comptonMatrixSamples)
{
    if(this->withCompton && this->withRandomWalk)
    {
        throw UniversalError("RadiationIMC Compton precompute is not compatible with random walk yet");
    }
    if(this->withCompton && !parameters.withMultigroupOpacity)
    {
        throw UniversalError("RadiationIMC Compton requires multigroup opacity");
    }
    if(parameters.withMultigroupOpacity)
    {
        this->multigroupOpacity = std::make_shared<MultigroupOpacity>(opacity);
    }
    else
    {
        this->multigroupOpacity = nullptr;
    }
    if(this->withRandomWalk)
    {
        this->randomWalk = std::make_unique<RandomWalk>();
    }
    int rank = 0;
    #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif
    if(rank == 0)
    {
        std::cout << parameters << std::endl;
    }
}

void RadiationIMC::setAdaptiveSourceCellScores(
    std::unordered_map<size_t, double> scores,
    double strength, double maxFactor,
    double learnedReserveFrac, double learnedMinFactor,
    double observerBudgetMultiplier)
{
    adaptiveSourceScores_ = std::move(scores);
    adaptiveSourceStrength_ = std::clamp(strength, 0.0, 1.0);
    adaptiveSourceMaxFactor_ = std::max(1.0, maxFactor);
    adaptiveSourceLearnedReserveFrac_ = std::clamp(learnedReserveFrac, 0.0, 1.0);
    adaptiveSourceLearnedMinFactor_ = std::max(1.0, learnedMinFactor);
    adaptiveSourceObserverBudgetMultiplier_ = std::max(1.0, observerBudgetMultiplier);
    adaptiveSourceScoresEnabled_ = !adaptiveSourceScores_.empty();
}

void RadiationIMC::clearAdaptiveSourceCellScores()
{
    adaptiveSourceScores_.clear();
    adaptiveSourceScoresEnabled_ = false;
    adaptiveSourceStrength_ = 0.0;
    adaptiveSourceMaxFactor_ = 1.0;
    adaptiveSourceLearnedReserveFrac_ = 0.0;
    adaptiveSourceLearnedMinFactor_ = 1.0;
    adaptiveSourceObserverBudgetMultiplier_ = 1.0;
}

void RadiationIMC::setSourceEmissionControl(
    bool useLearnedScores, bool includeUniformBase, size_t baseMultiplier,
    size_t learnedBoostFactor, size_t learnedExtraBudget)
{
    sourceEmissionControlEnabled_ = true;
    sourceEmissionUseLearnedScores_ = useLearnedScores;
    sourceEmissionIncludeUniformBase_ = includeUniformBase;
    sourceEmissionBaseMultiplier_ = std::max<size_t>(1, baseMultiplier);
    sourceEmissionLearnedBoostFactor_ = std::max<size_t>(1, learnedBoostFactor);
    sourceEmissionLearnedExtraBudget_ = learnedExtraBudget;
}

void RadiationIMC::clearSourceEmissionControl()
{
    sourceEmissionControlEnabled_ = false;
    sourceEmissionUseLearnedScores_ = false;
    sourceEmissionIncludeUniformBase_ = true;
    sourceEmissionBaseMultiplier_ = 1;
    sourceEmissionLearnedBoostFactor_ = 20;
    sourceEmissionLearnedExtraBudget_ = 0;
}

typename RadiationIMC::Functionality RadiationIMC::step(Particle &particle, std::vector<Particle> &particlesToAdd)
{
    (void)particlesToAdd;
    Functionality functionality;

    bool debug = false;

    size_t cellIndex = particle.cellIndex;
    ComputationalCell3D &cell = this->cells[cellIndex];

    auto [faceIntersect, timeIntersect, nextCellIndex] = this->getIntersectionDetails(particle);
    assert(timeIntersect >= 0);

    // todo: change opacity with doppler shift in cast of frequency dependance
    double dopplerShift = (this->withHydro && !this->MMC) ? DopplerShift(particle, cell.velocity) : 1.0;

    if(this->randomWalk && this->rwCellEligible[cellIndex])
    {
        if(this->tryRandomWalkStep(particle, functionality, dopplerShift))
        {
            ++this->rwStepCount;
            return functionality;
        }
    }

    double absorptionOpacity;
    size_t group = std::numeric_limits<size_t>::max();
    if(this->multigroupOpacity)
    {
        double shiftedFrequency = particle.frequency * dopplerShift;
        ClampFrequencyToBounds(shiftedFrequency);
        group = this->opacity->findGroup(shiftedFrequency);
        if(this->withCompton)
        {
            absorptionOpacity = this->comptonData[cellIndex].absorptionOpacity[group];
        }
        else
        {
            absorptionOpacity = this->opacity->CalcAbsorptionOpacity(cell, shiftedFrequency);
        }
    }
    else
    {
        absorptionOpacity = this->planckOpacities[cellIndex];
    }
    double elasticScatteringOpacity = this->opacity->CalcScatteringOpacity(cell);
    double effectiveAbsorptionOpacity = this->withCompton
        ? this->comptonData[cellIndex].baseEffectiveOpacity[group]
        : (1 - this->factorFleck[cellIndex]) * absorptionOpacity;
    double implicitComptonOpacity = (this->withCompton && group < ENERGY_GROUPS_NUM)
        ? this->comptonData[cellIndex].implicitEventRate[group]
        : 0.0;
    double eventOpacity = elasticScatteringOpacity + effectiveAbsorptionOpacity + implicitComptonOpacity;
    double scatteringLength = (eventOpacity > 0.0) ? 1.0 / eventOpacity : std::numeric_limits<double>::infinity();
    double _log1p = -std::log1p(this->dist(this->re) - 1); 
    distance_t scatteringDistance = scatteringLength * _log1p / dopplerShift; 
    if(scatteringDistance < 0)
    {
        UniversalError eo("Negative scattering distance in RadiationIMC::step");
        eo.addEntry("Cell scattering distance", opacity->CalcScatteringOpacity(cell));
        eo.addEntry("Factor fleck", this->factorFleck[cellIndex]);
        eo.addEntry("Planck opacity", this->planckOpacities[cellIndex]);
        eo.addEntry("log(1-randm)", _log1p);
        eo.addEntry("doppler shift", dopplerShift);
        eo.addEntry("particle", particle);
        throw eo;
    }
    dt_t timeScattering = std::isfinite(scatteringDistance) ? scatteringDistance / abs(particle.velocity) : std::numeric_limits<dt_t>::infinity();

    dt_t timeLeft = particle.timeLeft;
    std::array<std::pair<size_t, dt_t>, 3> times;
    enum Events
    { 
        INTERSECTION = 0, 
        SCATTERING = 1, 
        TIMELEFT = 2
    };
    times[Events::INTERSECTION] = {INTERSECTION, timeIntersect};
    times[Events::SCATTERING] = {SCATTERING, timeScattering};
    times[Events::TIMELEFT] = {TIMELEFT, timeLeft};

    std::pair<size_t, double> min = *std::min_element(times.begin(), times.end(), [](const std::pair<size_t, dt_t> &a, const std::pair<size_t, dt_t> &b) { return a.second < b.second; });
    if(debug)
    {
        std::cout << "min: " << min.first << " with time " << min.second << " for particle " << particle << std::endl;
    }
    dt_t dt = min.second;
    if(dt < 0)
    {
        UniversalError eo("Negative time step in RadiationIMC::step");
        eo.addEntry("time Intersect", timeIntersect);
        eo.addEntry("time Scattering", timeScattering);
        eo.addEntry("time Left", timeLeft);
        eo.addEntry("Particle", particle);
        throw eo;
    }
    particle.timeLeft -= dt;
    double weightEvolutionOpacity = absorptionOpacity * this->factorFleck[cellIndex];
    if(this->withCompton && group < ENERGY_GROUPS_NUM)
        weightEvolutionOpacity -= this->comptonData[cellIndex].implicitDiagonalCorrection[group];
    double tmp2 = weightEvolutionOpacity * units::clight;
    double tmp = -dt * tmp2;
    double expFactor1 = std::expm1(tmp * dopplerShift);
    double expFactor2 = std::expm1(tmp);
    double integratedForTally = particle.weight * dt;
    if(std::abs(tmp2 * dt) >= 1e-12)
    {
        integratedForTally = particle.weight * expFactor2 * (-1.0 / tmp2);
    }
    particle.location += particle.velocity * dt;
    if(!this->noHydroFeedback)
    {
        double const materialDeposit = -expFactor2 * particle.weight;
        this->conserved[cellIndex].internal_energy += materialDeposit;
        if(this->withCompton)
            this->comptonContinuousMaterialExchange += materialDeposit;
        if(this->withHydro)
        {
            if(not this->diffusionPressureGradient)
            {
                this->conserved[cellIndex].momentum += -expFactor1 * particle.weight * particle.velocity * units::inv_clight2;
            }
        }
    }
    this->Erad_time_avg[cellIndex] += integratedForTally;
    if(this->withEgTimeAvg && this->multigroupOpacity)
    {
        size_t g = this->opacity->findGroup(particle.frequency);
        this->Eg_time_avg[cellIndex][g] += integratedForTally;
    }
    particle.weight *= 1 + expFactor1;

    if(std::abs(particle.weight) < particle.initialWeight * 1e-3)
    {
        functionality.change = MonteCarloParticleStatus::REMOVE;
        if(!this->noHydroFeedback)
        {
            this->conserved[cellIndex].internal_energy += particle.weight;
            if(this->withCompton)
                this->comptonRemovalMaterialExchange += particle.weight;
        }
        return functionality;
    }

    if(min.first == Events::INTERSECTION)
    {
        functionality.change = MonteCarloParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = nextCellIndex;
    }
    else if(min.first == Events::SCATTERING)
    {
        Vector3D oldVelocity = particle.velocity;
        double oldWeight = particle.weight;
        double D_lab_to_co = dopplerShift;
        double eventRandom = this->dist(this->re) * eventOpacity;
        bool didImplicitCompton = false;
        bool isEffectiveScatter = false;
        if(eventRandom < elasticScatteringOpacity)
        {
            particle.velocity = opacity->getNewScatterVelocity(cell, particle);
        }
        else if((eventRandom -= elasticScatteringOpacity) < effectiveAbsorptionOpacity)
        {
            particle.velocity = opacity->getNewScatterVelocity(cell, particle);
            isEffectiveScatter = true;
        }
        else
        {
            this->applyImplicitComptonEvent(cellIndex, cell, group, oldVelocity, oldWeight, dopplerShift, particle);
            didImplicitCompton = true;
        }
        if(this->multigroupOpacity)
        {
            if(!didImplicitCompton)
            {
                particle.frequency *= dopplerShift; // lab → comoving
                ClampFrequencyToBounds(particle.frequency);
            }
            if(isEffectiveScatter)
            {
                if(this->withCompton)
                {
                    size_t targetGroup = this->sampleComptonCdf(this->comptonData[cellIndex].baseSourceCdf, this->dist(this->re));
                    particle.frequency = this->frequencyForComptonGroup(targetGroup);
                }
                else
                {
                    double reemitRandom = this->dist(this->re);
                    particle.frequency = this->multigroupOpacity->GetThermalEnergy(cell, reemitRandom);
                }
            }
            if(didImplicitCompton)
            {
                ClampFrequencyToBounds(particle.frequency);
            }
        }
        if(this->withHydro && !this->MMC && !didImplicitCompton)
        {
            double weightBefore = particle.weight;
            particle.weight *= D_lab_to_co;
            LorentzTransformation(particle, -1 * cell.velocity);
            if(this->multigroupOpacity)
            {
                ClampFrequencyToBounds(particle.frequency);
            }
            if(not this->diffusionPressureGradient && !this->noHydroFeedback)
            {
                this->conserved[cellIndex].momentum += (weightBefore * oldVelocity - particle.weight * particle.velocity) * units::inv_clight2;
            }
        }
    }
    else if(min.first == Events::TIMELEFT)
    {
        functionality.change = MonteCarloParticleStatus::DONE;
    }
    else
    {
        UniversalError eo("Unknown case in RadiationIMC::step");
        eo.addEntry("Particle", particle);
        throw eo;
    }

    return functionality;
}

void RadiationIMC::postStep(const std::vector<Particle> &particles, double fullDt)
{
    size_t Ncells = this->grid.GetPointNo();
    for(size_t i = 0; i < Ncells; i++)
    {
        this->Erad_time_avg[i] /= (fullDt * this->grid.GetVolume(i));
        if(this->withEgTimeAvg && this->multigroupOpacity)
        {
            double norm = fullDt * this->grid.GetVolume(i);
            for(size_t g = 0; g < this->Eg_time_avg[i].size(); g++)
                this->Eg_time_avg[i][g] /= norm;
        }
    }

    if(!this->noHydroFeedback)
    {
        std::vector<double> Erad_time_avg_grad = std::vector<double>(Ncells, 0);
        if(this->diffusionPressureGradient)
        {
            #ifdef RICH_MPI
            MPI_exchange_data(this->grid, this->Erad_time_avg, true);
            #endif // RICH_MPI

            // todo: fix for 3D, current is 1D!
            
            for(size_t i = 0; i < Ncells; i++)
            {
                const Vector3D &point = this->grid.GetMeshPoint(i);
                // locate neighbor from right and left
                size_t neighbor_right = std::numeric_limits<size_t>::max();
                size_t neighbor_left = std::numeric_limits<size_t>::max();
                for(size_t faceIdx : this->grid.GetCellFaces(i))
                {
                    const std::pair<size_t, size_t> &neighbors = this->grid.GetFaceNeighbors(faceIdx);
                    size_t neighborIdx = (neighbors.first == i)? neighbors.second : neighbors.first;
                    Vector3D diff = normalize(this->grid.GetMeshPoint(neighborIdx) - point);
                    if(diff.x > 0.99)
                    {
                        neighbor_right = neighborIdx;
                    }
                    else if(diff.x < -0.99)
                    {
                        neighbor_left = neighborIdx;
                    }
                }
                if(neighbor_right == std::numeric_limits<size_t>::max())
                {
                    throw UniversalError("No right neighbor found in RadiationIMC::postStep");
                }
                if(neighbor_left == std::numeric_limits<size_t>::max())
                {
                    throw UniversalError("No left neighbor found in RadiationIMC::postStep");
                }
                const Vector3D &neighbor_right_point = this->grid.GetMeshPoint(neighbor_right);
                const Vector3D &neighbor_left_point = this->grid.GetMeshPoint(neighbor_left);
                double grad;
                if(this->grid.IsPointOutsideBox(neighbor_left))
                {
                    grad = (this->Erad_time_avg[neighbor_right] - this->Erad_time_avg[i]) / (neighbor_right_point - point).x;
                }
                else if(this->grid.IsPointOutsideBox(neighbor_right))
                {
                    grad = (this->Erad_time_avg[i] - this->Erad_time_avg[neighbor_left]) / (point - neighbor_left_point).x;
                }
                else
                {
                    grad = (this->Erad_time_avg[neighbor_right] - this->Erad_time_avg[neighbor_left]) / (neighbor_right_point - neighbor_left_point).x;
                }
                Erad_time_avg_grad[i] = grad;
            }
        }
        
        for(size_t i = 0; i < Ncells; i++)
        {
            ComputationalCell3D &cell = this->cells[i];
            cell.internal_energy = this->conserved[i].internal_energy / this->conserved[i].mass;
            if(cell.internal_energy < 0)
            {
                UniversalError eo("Negative internal energy in RadiationIMC::postStep");
                eo.addEntry("Cell index", i);
                eo.addEntry("Internal energy", cell.internal_energy);
                eo.addEntry("Mass", this->conserved[i].mass);
                eo.addEntry("Density", cell.density);
                eo.addEntry("Temperature", cell.temperature);
                if(this->withCompton)
                {
                    eo.addEntry("Compton source material exchange", this->comptonSourceMaterialExchange);
                    eo.addEntry("Compton continuous material exchange", this->comptonContinuousMaterialExchange);
                    eo.addEntry("Compton event material exchange", this->comptonImplicitMaterialExchange);
                    eo.addEntry("Compton removal material exchange", this->comptonRemovalMaterialExchange);
                    eo.addEntry("Compton implicit event count", this->comptonImplicitEventCount);
                }
                throw eo;
            }
            if(this->withHydro)
            {
                if(this->diffusionPressureGradient)
                {
                    this->conserved[i].momentum.x -= fullDt * this->grid.GetVolume(i) * Erad_time_avg_grad[i] / 3;
                }
                cell.velocity = this->conserved[i].momentum / this->conserved[i].mass;
                this->conserved[i].energy = this->conserved[i].internal_energy + 0.5 * ScalarProd(this->conserved[i].momentum, this->conserved[i].momentum) / this->conserved[i].mass; // TODO: material strength
            }
            cell.temperature = this->eos->de2T(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
            cell.pressure = this->eos->de2p(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
        }
    }

    for(size_t i = 0; i < Ncells; i++)
    {
        this->conserved[i].Erad = 0;
        if(this->multigroupOpacity)
        {
            std::fill(this->conserved[i].Eg.begin(), this->conserved[i].Eg.end(), 0.0);
        }
    }
    std::vector<double> absRadiationWeight(Ncells, 0.0);
    std::vector<std::array<double, ENERGY_GROUPS_NUM>> absGroupRadiationWeight;
    if(this->multigroupOpacity)
    {
        std::array<double, ENERGY_GROUPS_NUM> zeros{};
        absGroupRadiationWeight.assign(Ncells, zeros);
    }
    for(const Particle &particle : particles)
    {
        size_t cellIndex = particle.cellIndex;
        assert(cellIndex < Ncells);
        this->conserved[cellIndex].Erad += particle.weight;
        absRadiationWeight[cellIndex] += std::abs(particle.weight);
        if(this->multigroupOpacity)
        {
            size_t g = this->opacity->findGroup(particle.frequency);
            this->conserved[cellIndex].Eg[g] += particle.weight;
            absGroupRadiationWeight[cellIndex][g] += std::abs(particle.weight);
        }
    }
    if(this->withCompton && this->multigroupOpacity)
    {
        this->comptonMinGroupEnergy = std::numeric_limits<double>::infinity();
        this->comptonMaxGroupEnergy = -std::numeric_limits<double>::infinity();
        for(size_t i = 0; i < Ncells; i++)
        {
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                this->comptonMinGroupEnergy = std::min(this->comptonMinGroupEnergy, this->conserved[i].Eg[g]);
                this->comptonMaxGroupEnergy = std::max(this->comptonMaxGroupEnergy, this->conserved[i].Eg[g]);
            }
        }
        for(size_t i = 0; i < Ncells; i++)
        {
            double radiationProjection = 0.0;
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                if(this->conserved[i].Eg[g] < 0.0)
                {
                    radiationProjection -= this->conserved[i].Eg[g];
                    this->conserved[i].Eg[g] = 0.0;
                    ++this->comptonProjectedNegativeGroupCount;
                }
            }
            if(radiationProjection > 0.0)
            {
                this->conserved[i].Erad += radiationProjection;
                this->comptonProjectedRadiationEnergy += radiationProjection;
                if(!this->noHydroFeedback)
                {
                    this->conserved[i].internal_energy -= radiationProjection;
                    this->conserved[i].energy -= radiationProjection;
                    if(this->conserved[i].internal_energy < 0.0)
                    {
                        UniversalError eo("Negative internal energy after Compton group positivity projection");
                        eo.addEntry("Cell index", i);
                        eo.addEntry("Projected radiation energy", radiationProjection);
                        eo.addEntry("Internal energy", this->conserved[i].internal_energy);
                        eo.addEntry("Raw min group energy", this->comptonMinGroupEnergy);
                        eo.addEntry("Raw max group energy", this->comptonMaxGroupEnergy);
                        throw eo;
                    }
                    ComputationalCell3D &cell = this->cells[i];
                    cell.internal_energy = this->conserved[i].internal_energy / this->conserved[i].mass;
                    cell.temperature = this->eos->de2T(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
                    cell.pressure = this->eos->de2p(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
                }
            }
        }
    }
    if(this->withCompton && this->comptonCheckSignedTallies)
    {
        for(size_t i = 0; i < Ncells; i++)
        {
            double const tolerance = this->comptonSignedTallyTolerance * std::max(absRadiationWeight[i], 1.0);
            if(this->conserved[i].Erad < -tolerance)
            {
                UniversalError eo("Negative radiation energy after signed Compton IMC tally");
                eo.addEntry("Cell index", i);
                eo.addEntry("Erad", this->conserved[i].Erad);
                eo.addEntry("Abs radiation packet weight", absRadiationWeight[i]);
                eo.addEntry("Tolerance", tolerance);
                throw eo;
            }
            if(this->multigroupOpacity)
            {
                for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
                {
                    double const groupTolerance = this->comptonSignedTallyTolerance * std::max(absGroupRadiationWeight[i][g], 1.0);
                    if(this->conserved[i].Eg[g] < -groupTolerance)
                    {
                        UniversalError eo("Negative group radiation energy after signed Compton IMC tally");
                        eo.addEntry("Cell index", i);
                        eo.addEntry("Group", g);
                        eo.addEntry("Eg", this->conserved[i].Eg[g]);
                        eo.addEntry("Abs group packet weight", absGroupRadiationWeight[i][g]);
                        eo.addEntry("Tolerance", groupTolerance);
                        throw eo;
                    }
                }
            }
        }
    }
    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        cell.Erad = this->conserved[i].Erad / this->conserved[i].mass;
        if(this->multigroupOpacity)
        {
            for(size_t g = 0; g < cell.Eg.size(); g++)
            {
                cell.Eg[g] = this->conserved[i].Eg[g] / this->conserved[i].mass;
            }
        }
    }

    if(this->withRandomWalk)
    {
        size_t globalRwSteps = this->rwStepCount;
        int rank = 0;
        #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 0)
            MPI_Reduce(MPI_IN_PLACE, &globalRwSteps, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        else
            MPI_Reduce(&globalRwSteps, nullptr, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        #endif
        if(rank == 0)
            std::cout << "RW steps: " << globalRwSteps << std::endl;
    }
    this->printComptonDiagnostics();
}

std::vector<typename RadiationIMC::Particle> RadiationIMC::generateParticles(double fullDt)
{
    if(this->withCompton)
        return this->generateComptonParticles(fullDt);

    std::vector<Particle> newParticles;
    size_t Ncells = this->grid.GetPointNo();

    std::vector<double> energyToCreateVec(Ncells);
    std::vector<double> gammaVec(Ncells);
    double localTotalEnergy = 0;
    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        gammaVec[i] = (this->withHydro && !this->MMC) ? 1 / std::sqrt(1 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2) : 1;
        energyToCreateVec[i] = this->factorFleck[i] * this->grid.GetVolume(i) * units::arad * boost::math::pow<4>(cell.temperature) * this->planckOpacities[i] * fullDt * units::clight;
        localTotalEnergy += energyToCreateVec[i];
    }

    double globalTotalEnergy = localTotalEnergy;
    size_t globalTotalCells = Ncells;
    #ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &globalTotalEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &globalTotalCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    #endif

    size_t totalParticles = globalTotalCells * this->newPhotonsPerCell * 10;
    std::vector<size_t> nPhotons(Ncells);
    for(size_t i = 0; i < Ncells; i++)
    {
        size_t proportionalShare = (globalTotalEnergy > 0)
            ? static_cast<size_t>(energyToCreateVec[i] / globalTotalEnergy * totalParticles)
            : this->newPhotonsPerCell;
        nPhotons[i] = std::max(this->newPhotonsPerCell, std::min(proportionalShare, this->newPhotonsPerCell * 20));
    }

    if(sourceEmissionControlEnabled_)
    {
        double scoreSum = 0.0;
        for(auto const &kv : adaptiveSourceScores_)
            if(std::isfinite(kv.second) && kv.second > 0.0)
                scoreSum += kv.second;

        size_t const basePhotons = this->newPhotonsPerCell * sourceEmissionBaseMultiplier_;
        size_t const maxPhotons = static_cast<size_t>(std::ceil(
            static_cast<double>(std::max<size_t>(1, this->newPhotonsPerCell)) *
            adaptiveSourceMaxFactor_ * adaptiveSourceObserverBudgetMultiplier_));
        for(size_t i = 0; i < Ncells; ++i)
        {
            auto const it = adaptiveSourceScores_.find(this->cells[i].ID);
            bool const learned = adaptiveSourceScoresEnabled_ && it != adaptiveSourceScores_.end()
                && std::isfinite(it->second) && it->second > 0.0;

            size_t photons = sourceEmissionIncludeUniformBase_ ? basePhotons : 0;
            if(sourceEmissionUseLearnedScores_ && learned)
            {
                size_t learnedPhotons = this->newPhotonsPerCell * sourceEmissionLearnedBoostFactor_;
                if(scoreSum > 0.0 && sourceEmissionLearnedExtraBudget_ > 0)
                {
                    learnedPhotons += static_cast<size_t>(std::ceil(
                        adaptiveSourceStrength_ * static_cast<double>(sourceEmissionLearnedExtraBudget_) *
                        it->second / scoreSum));
                }
                size_t const minLearned = static_cast<size_t>(std::ceil(
                    static_cast<double>(std::max<size_t>(1, this->newPhotonsPerCell)) *
                    adaptiveSourceLearnedMinFactor_));
                learnedPhotons = std::max(learnedPhotons, minLearned);
                photons = std::max(photons, learnedPhotons);
            }
            nPhotons[i] = std::min(photons, std::max<size_t>(1, maxPhotons));
        }
    }

    lastSourcePhotonsPerCell_ = nPhotons;
    lastSourceAllocationSummary_ = SourceAllocationSummary{};
    lastSourceAllocationSummary_.adaptiveEnabled =
        sourceEmissionControlEnabled_ && sourceEmissionUseLearnedScores_ && adaptiveSourceScoresEnabled_;
    lastSourceAllocationSummary_.minPhotons = std::numeric_limits<size_t>::max();
    lastSourceAllocationSummary_.learnedMinPhotons = std::numeric_limits<size_t>::max();
    for(size_t i = 0; i < Ncells; ++i)
    {
        size_t const photons = nPhotons[i];
        if(photons == 0)
            continue;
        ++lastSourceAllocationSummary_.sourceCells;
        lastSourceAllocationSummary_.totalPhotons += photons;
        lastSourceAllocationSummary_.minPhotons = std::min(lastSourceAllocationSummary_.minPhotons, photons);
        lastSourceAllocationSummary_.maxPhotons = std::max(lastSourceAllocationSummary_.maxPhotons, photons);
        if(photons > this->newPhotonsPerCell)
            ++lastSourceAllocationSummary_.boostedCells;

        auto const it = adaptiveSourceScores_.find(this->cells[i].ID);
        bool const learned = adaptiveSourceScoresEnabled_ && it != adaptiveSourceScores_.end()
            && std::isfinite(it->second) && it->second > 0.0;
        if(learned)
        {
            ++lastSourceAllocationSummary_.learnedCells;
            lastSourceAllocationSummary_.learnedPhotons += photons;
            lastSourceAllocationSummary_.adaptiveScoreSum += it->second;
            lastSourceAllocationSummary_.learnedMinPhotons =
                std::min(lastSourceAllocationSummary_.learnedMinPhotons, photons);
            lastSourceAllocationSummary_.learnedMaxPhotons =
                std::max(lastSourceAllocationSummary_.learnedMaxPhotons, photons);
            if(photons > this->newPhotonsPerCell)
            {
                ++lastSourceAllocationSummary_.learnedBoostedCells;
                lastSourceAllocationSummary_.learnedExtraPhotons += photons - this->newPhotonsPerCell;
            }
        }
    }
    if(lastSourceAllocationSummary_.minPhotons == std::numeric_limits<size_t>::max())
        lastSourceAllocationSummary_.minPhotons = 0;
    if(lastSourceAllocationSummary_.learnedMinPhotons == std::numeric_limits<size_t>::max())
        lastSourceAllocationSummary_.learnedMinPhotons = 0;

    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        double energyToCreate = energyToCreateVec[i];
        double gamma = gammaVec[i];
        size_t nPhotonsCell = nPhotons[i];
        if(nPhotonsCell == 0)
            continue;

        if(!this->noHydroFeedback)
        {
            this->conserved[i].internal_energy -= energyToCreate;
            if(this->conserved[i].internal_energy < 0)
            {
                UniversalError eo("Negative internal energy in RadiationIMC::generateParticles");
                eo.addEntry("Energy to create", energyToCreate);
                eo.addEntry("Cell index", i);
                eo.addEntry("Internal energy", this->conserved[i].internal_energy);
                eo.addEntry("Density", cell.density);
                eo.addEntry("Temperature", cell.temperature);
                eo.addEntry("Planck opacity", this->planckOpacities[i]);
                eo.addEntry("Volume", this->grid.GetVolume(i));
                eo.addEntry("Full dt", fullDt);
                eo.addEntry("Factor fleck", this->factorFleck[i]);
                eo.addEntry("Gamma", gamma);
                eo.addEntry("cv", this->eos->dT2cv(cell.density, cell.temperature, cell.tracers, cell.tracerNames));
                throw eo;
            }
            this->conserved[i].energy -= energyToCreate * gamma;
            if(this->withHydro)
            {
                if(not this->diffusionPressureGradient)
                {
                    this->conserved[i].momentum -= energyToCreate * cell.velocity * units::inv_clight2 * gamma;
                }
            }
        }
        double energyPerPhoton = energyToCreate * gamma / nPhotonsCell;
        for(size_t j = 0; j < nPhotonsCell; j++)
        {
            MCParticle particle = this->generateSingleParticle(i, cell);
            particle.cellID = cell.ID;
            particle.timeLeft = fullDt * this->dist(this->re);
            if(this->withHydro && !this->MMC)
            {
                double D = DopplerShift(particle, cell.velocity);
                if(this->multigroupOpacity)
                {
                    double rnd = this->dist(this->re);
                    double freqCo = this->multigroupOpacity->GetThermalEnergy(cell, rnd);
                    particle.frequency = freqCo / D;
                }
                particle.weight = energyToCreate / (nPhotonsCell * D);
            }
            else
            {
                if(this->multigroupOpacity)
                {
                    particle.frequency = this->multigroupOpacity->GetThermalEnergy(cell, this->dist(this->re));
                }
                particle.weight = energyPerPhoton;
            }
            SetInitialWeightFromWeight(particle);
            newParticles.push_back(particle);
        }
    }
    return newParticles;
}

std::vector<typename RadiationIMC::Particle> RadiationIMC::generateComptonParticles(double fullDt)
{
    std::vector<Particle> newParticles;
    size_t const Ncells = this->grid.GetPointNo();
    lastSourcePhotonsPerCell_.assign(Ncells, 0);
    lastSourceAllocationSummary_ = SourceAllocationSummary{};
    if(this->comptonData.size() != Ncells)
        throw UniversalError("Compton data is not initialized in RadiationIMC::generateComptonParticles");

    std::vector<double> absEnergyToCreateVec(Ncells, 0.0);
    for(size_t i = 0; i < Ncells; i++)
    {
        for(double const sourceEnergy : this->comptonData[i].Btotal)
            absEnergyToCreateVec[i] += std::abs(sourceEnergy);
    }

    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        ComptonCellData const &cd = this->comptonData[i];
        double const gamma = (this->withHydro && !this->MMC) ? 1 / std::sqrt(1 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2) : 1;

        GroupArray absGroupEnergy{};
        std::array<size_t, ENERGY_GROUPS_NUM> groupCounts{};
        GroupArray fractional{};
        size_t nonzeroGroups = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            absGroupEnergy[g] = std::abs(cd.Btotal[g]);
            if(absGroupEnergy[g] > 0.0)
                ++nonzeroGroups;
        }
        size_t const nPhotonsCell = std::max(this->newPhotonsPerCell, nonzeroGroups);
        if(nPhotonsCell == 0 || absEnergyToCreateVec[i] <= 0.0)
            continue;
        lastSourcePhotonsPerCell_[i] = nPhotonsCell;
        ++lastSourceAllocationSummary_.sourceCells;
        lastSourceAllocationSummary_.totalPhotons += nPhotonsCell;
        lastSourceAllocationSummary_.minPhotons =
            (lastSourceAllocationSummary_.minPhotons == 0)
                ? nPhotonsCell : std::min(lastSourceAllocationSummary_.minPhotons, nPhotonsCell);
        lastSourceAllocationSummary_.maxPhotons =
            std::max(lastSourceAllocationSummary_.maxPhotons, nPhotonsCell);

        size_t allocated = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(absGroupEnergy[g] <= 0.0)
                continue;
            groupCounts[g] = 1;
            ++allocated;
        }
        size_t const remainingBudget = nPhotonsCell - allocated;
        size_t extraAllocated = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(absGroupEnergy[g] <= 0.0)
                continue;
            double const exactExtra = static_cast<double>(remainingBudget) * absGroupEnergy[g] / absEnergyToCreateVec[i];
            size_t const extra = static_cast<size_t>(std::floor(exactExtra));
            groupCounts[g] += extra;
            fractional[g] = exactExtra - static_cast<double>(extra);
            extraAllocated += extra;
        }
        while(extraAllocated < remainingBudget)
        {
            size_t bestGroup = ENERGY_GROUPS_NUM;
            double bestFraction = -1.0;
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                if(absGroupEnergy[g] > 0.0 && fractional[g] > bestFraction)
                {
                    bestGroup = g;
                    bestFraction = fractional[g];
                }
            }
            if(bestGroup == ENERGY_GROUPS_NUM)
                break;
            ++groupCounts[bestGroup];
            fractional[bestGroup] = 0.0;
            ++extraAllocated;
        }

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            size_t const ng = groupCounts[g];
            double const signedGroupEnergy = cd.Btotal[g];
            if(ng == 0 || signedGroupEnergy == 0.0)
                continue;

            if(!this->noHydroFeedback)
            {
                double const materialDeposit = -signedGroupEnergy;
                this->conserved[i].internal_energy += materialDeposit;
                this->comptonSourceMaterialExchange += materialDeposit;
                if(signedGroupEnergy > 0.0 && this->conserved[i].internal_energy < 0.0)
                {
                    UniversalError eo("Negative internal energy in RadiationIMC::generateComptonParticles");
                    eo.addEntry("Energy to emit", signedGroupEnergy);
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Group", g);
                    eo.addEntry("Internal energy", this->conserved[i].internal_energy);
                    eo.addEntry("Density", cell.density);
                    eo.addEntry("Temperature", cell.temperature);
                    eo.addEntry("Planck opacity", cd.planckOpacity);
                    eo.addEntry("Factor fleck", cd.fleck);
                    throw eo;
                }
                this->conserved[i].energy -= signedGroupEnergy * gamma;
                if(this->withHydro && !this->diffusionPressureGradient)
                    this->conserved[i].momentum -= signedGroupEnergy * cell.velocity * units::inv_clight2 * gamma;
            }

            double const signedEnergy = signedGroupEnergy / static_cast<double>(ng);
            for(size_t j = 0; j < ng; j++)
            {
                Particle particle = this->generateSingleParticle(i, cell);
                particle.timeLeft = fullDt * this->dist(this->re);
                particle.cellID = cell.ID;
                if(this->withHydro && !this->MMC)
                {
                    double const D = DopplerShift(particle, cell.velocity);
                    particle.frequency = this->frequencyForComptonGroup(g) / D;
                    particle.weight = signedEnergy / D;
                }
                else
                {
                    particle.frequency = this->frequencyForComptonGroup(g);
                    particle.weight = signedEnergy;
                }
                ClampFrequencyToBounds(particle.frequency);
                SetInitialWeightFromWeight(particle);
                if(particle.initialWeight > 0.0)
                    newParticles.push_back(particle);
            }
        }
    }

    return newParticles;
}

void RadiationIMC::initializeComptonGroups()
{
    if(this->comptonGroupsInitialized)
        return;

    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        double const left = ComputationalCell3D::energyBoundaries[g];
        double const right = ComputationalCell3D::energyBoundaries[g + 1];
        if(std::isnan(left) || std::isnan(right) || left >= right)
        {
            UniversalError eo("Invalid Compton energy group boundaries");
            eo.addEntry("Group", g);
            eo.addEntry("Left boundary", left);
            eo.addEntry("Right boundary", right);
            throw eo;
        }
        this->comptonGroupCenters[g] = 0.5 * (left + right);
        this->comptonGroupWidths[g] = right - left;
    }
    this->comptonGroupsInitialized = true;
}

void RadiationIMC::initializeComptonMatrixGenerator()
{
    if(this->comptonMatrixGen)
        return;

    this->initializeComptonGroups();
    this->comptonMatrixGen = std::make_unique<ComptonMatrixMC>(
        BuildComptonCentersVector(),
        BuildComptonBoundariesVector(),
        this->comptonMatrixSamples,
        true,
        1);
    this->comptonMatrixGen->set_tables(BuildComptonTemperatures());
}

RadiationIMC::GroupCdf RadiationIMC::buildSafeComptonCdf(const GroupArray &weights)
{
    GroupCdf cdf{};
    cdf[0] = 0.0;
    double total = 0.0;
    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        total += std::max(0.0, weights[g]);
        cdf[g + 1] = total;
    }
    if(total <= 0.0)
    {
        for(size_t g = 0; g <= ENERGY_GROUPS_NUM; g++)
            cdf[g] = static_cast<double>(g) / static_cast<double>(ENERGY_GROUPS_NUM);
        return cdf;
    }
    for(double &value : cdf)
        value /= total;
    cdf[ENERGY_GROUPS_NUM] = 1.0;
    return cdf;
}

double RadiationIMC::frequencyForComptonGroup(size_t group) const
{
    if(group >= ENERGY_GROUPS_NUM)
        throw UniversalError("Invalid Compton group in RadiationIMC::frequencyForComptonGroup");
    double frequency = this->comptonGroupCenters[group];
    ClampFrequencyToBounds(frequency);
    return frequency;
}

void RadiationIMC::buildComptonMatricesForCell(const ComputationalCell3D &cell, size_t cellIndex, bool calculateN, ComptonCellData &cd)
{
    this->initializeComptonMatrixGenerator();

    double constexpr fac = boost::math::pow<3>(units::clight) / (8.0 * M_PI * units::planck_constant);
    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        if(calculateN)
        {
            double const dnu = this->comptonGroupWidths[g] / units::planck_constant;
            double const nu = this->comptonGroupCenters[g] / units::planck_constant;
            double const Eg = std::max(0.0, cell.Eg[g] * cell.density);
            cd.occupation[g] = std::min(100.0, fac * Eg / (boost::math::pow<3>(nu) * dnu));
        }
        else
        {
            cd.occupation[g] = 0.0;
        }
    }

    Matrix tau(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0));
    Matrix dtau(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0));
    double const A = 1.0;
    double const Z = 1.0;
    double const temperature = std::min(this->comptonMatrixGen->get_maximum_temperature_grid() * 0.9999, cell.temperature);
    this->comptonMatrixGen->get_tau_matrix(temperature, cell.density, A, Z, tau, dtau);

    for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            cd.tau[h][g] = tau[h][g];
            cd.dtau_dUm[h][g] = dtau[h][g];
        }
    }

    auto const lastGroup = this->comptonMatrixGen->get_last_group_upscattering_and_downscattering(temperature, cell.density, A, Z);
    double const upScatteringLast = lastGroup.first;
    double const downScatteringLast = lastGroup.second;

    ZeroGroupMatrix(cd.S);
    ZeroGroupMatrix(cd.dSdUm);

    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        for(size_t gt = 0; gt < ENERGY_GROUPS_NUM; gt++)
        {
            if(g + 1 == ENERGY_GROUPS_NUM && gt + 1 == ENERGY_GROUPS_NUM)
            {
                cd.S[g][g] += (upScatteringLast - downScatteringLast) * (1.0 + cd.occupation[g]);
                cd.dSdUm[g][g] += cd.dtau_dUm[g][g] * (1.0 + cd.occupation[g]);
                continue;
            }

            double const inScatteringFactor = this->comptonGroupCenters[g] / this->comptonGroupCenters[gt] * (1.0 + cd.occupation[g]);
            cd.S[gt][g] += cd.tau[gt][g] * inScatteringFactor;
            cd.dSdUm[gt][g] += cd.dtau_dUm[gt][g] * inScatteringFactor;

            double const outScatteringFactor = 1.0 + cd.occupation[gt];
            cd.S[g][g] -= cd.tau[g][gt] * outScatteringFactor;
            cd.dSdUm[g][g] -= cd.dtau_dUm[g][gt] * outScatteringFactor;
        }
    }

    double const UmFactor = 1.0 / (4.0 * units::arad * boost::math::pow<3>(cell.temperature));
    for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            cd.dSdUm[h][g] *= UmFactor;
        }
    }

    (void) cellIndex;
}

void RadiationIMC::recomputeComptonContractions(ComptonCellData &cd)
{
    cd.Upsilon = 0.0;
    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        cd.D[g] = 0.0;
        for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
            cd.D[g] += cd.dSdUm[h][g] * cd.oldRadiationEnergy[h];
        cd.Upsilon += cd.D[g];
        cd.M[g] = cd.absorptionOpacity[g] * cd.planckFraction[g] + cd.D[g];
    }

    for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        cd.rowS[h] = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            cd.rowS[h] += cd.S[h][g];
        cd.Lambda[h] = cd.absorptionOpacity[h] - cd.rowS[h];
    }

    cd.Gamma = cd.planckOpacity + cd.Upsilon;
}

void RadiationIMC::buildComptonSources(double fullDt, ComptonCellData &cd)
{
    double const cdt = units::clight * fullDt;
    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        double const kgbg = cd.absorptionOpacity[g] * cd.planckFraction[g];
        cd.Bbase[g] = cd.volume * cdt * cd.fleck * kgbg * cd.Um;

        if(cd.planckOpacity > 0.0)
        {
            cd.Bcorr[g] = cd.volume * cdt * cd.planckOpacity * cd.Um *
                ((kgbg / cd.planckOpacity) * (1.0 - (1.0 + cd.beta * cdt * cd.planckOpacity) * cd.fleck)
                 - cd.beta * cdt * cd.fleck * cd.D[g]);
        }
        else
        {
            cd.Bcorr[g] = 0.0;
        }
        cd.Btotal[g] = cd.Bbase[g] + cd.Bcorr[g];
    }
}

void RadiationIMC::buildComptonInPlaceKernels(size_t cellIndex, ComptonCellData &cd)
{
    (void) cellIndex;
    ZeroGroupMatrix(cd.Kmat);
    ZeroGroupMatrix(cd.Hbase);
    ZeroGroupMatrix(cd.implicitKernel);
    ZeroGroupMatrix(cd.implicitEventRateMatrix);

    for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        cd.baseEffectiveOpacity[h] = 0.0;
        cd.implicitEventRate[h] = 0.0;
        cd.implicitDiagonalCorrection[h] = 0.0;
        cd.implicitEventCdf[h] = RadiationIMC::buildSafeComptonCdf(GroupArray{});
    }

    for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        GroupArray eventRates{};
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            cd.Kmat[h][g] = cd.betaCdtF * cd.M[g] * cd.Lambda[h];
            cd.Hbase[h][g] = (1.0 - cd.fleck) * cd.absorptionOpacity[h] * cd.baseSourceFraction[g];
            cd.baseEffectiveOpacity[h] += cd.Hbase[h][g];
            cd.implicitKernel[h][g] = cd.Kmat[h][g] - cd.Hbase[h][g] + cd.S[h][g];
            if(g == h)
                continue;
            double const ratio = this->comptonGroupCenters[g] / this->comptonGroupCenters[h];
            if(ratio > 0.0)
            {
                eventRates[g] = std::abs(cd.implicitKernel[h][g]) / ratio;
                cd.implicitEventRateMatrix[h][g] = eventRates[g];
                cd.implicitEventRate[h] += eventRates[g];
            }
        }
        cd.implicitDiagonalCorrection[h] = cd.implicitKernel[h][h] + cd.implicitEventRate[h];
        cd.implicitEventCdf[h] = RadiationIMC::buildSafeComptonCdf(eventRates);
    }
}

void RadiationIMC::validateComptonParity(size_t cellIndex, const ComptonCellData &cd) const
{
    double maxAbsDiff = 0.0;
    double maxDeterministicDiff = 0.0;
    for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double const deterministicKernel =
                cd.S[h][g] + cd.betaCdtF * cd.M[g] * cd.Lambda[h] - cd.Hbase[h][g];
            maxDeterministicDiff = std::max(maxDeterministicDiff,
                std::abs(cd.implicitKernel[h][g] - deterministicKernel));

            double lhs = cd.implicitDiagonalCorrection[h] - cd.implicitEventRate[h];
            if(h != g)
            {
                double const ratio = this->comptonGroupCenters[g] / this->comptonGroupCenters[h];
                double const sign = (cd.implicitKernel[h][g] >= 0.0) ? 1.0 : -1.0;
                lhs = cd.implicitEventRateMatrix[h][g] * sign * ratio;
            }
            double const rhs = cd.implicitKernel[h][g];
            maxAbsDiff = std::max(maxAbsDiff, std::abs(lhs - rhs));
        }
    }
    if(maxAbsDiff > 1e-10)
    {
        UniversalError eo("Compton in-place parity check failed in RadiationIMC");
        eo.addEntry("Cell index", cellIndex);
        eo.addEntry("Max abs diff", maxAbsDiff);
        throw eo;
    }
    if(maxDeterministicDiff > 1e-10)
    {
        UniversalError eo("Compton deterministic matrix parity check failed in RadiationIMC");
        eo.addEntry("Cell index", cellIndex);
        eo.addEntry("Max deterministic abs diff", maxDeterministicDiff);
        throw eo;
    }
}

void RadiationIMC::applyImplicitComptonEvent(size_t cellIndex, const ComputationalCell3D &cell, size_t sourceGroup, const Vector3D &oldVelocity, double oldWeight, double dopplerShift, Particle &particle)
{
    (void) dopplerShift;
    if(sourceGroup >= ENERGY_GROUPS_NUM)
        return;

    const ComptonCellData &cd = this->comptonData[cellIndex];
    if(cd.implicitEventRate[sourceGroup] <= 0.0)
        return;

    size_t targetGroup = this->sampleComptonCdf(cd.implicitEventCdf[sourceGroup], this->dist(this->re));
    if(targetGroup == sourceGroup || cd.implicitEventRateMatrix[sourceGroup][targetGroup] <= 0.0)
    {
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(g != sourceGroup && cd.implicitEventRateMatrix[sourceGroup][g] > 0.0)
            {
                targetGroup = g;
                break;
            }
        }
    }
    if(targetGroup >= ENERGY_GROUPS_NUM)
        return;
    if(targetGroup == sourceGroup || cd.implicitEventRateMatrix[sourceGroup][targetGroup] <= 0.0)
        return;

    double const kernel = cd.implicitKernel[sourceGroup][targetGroup];
    if(kernel == 0.0)
        return;

    particle.velocity = this->opacity->getNewScatterVelocity(cell, particle);

    double const oldCenter = this->comptonGroupCenters[sourceGroup];
    double const newCenter = this->comptonGroupCenters[targetGroup];
    double const ratio = newCenter / oldCenter;
    double const sign = (kernel > 0.0) ? 1.0 : -1.0;
    double const newWeight = sign * oldWeight * ratio;
    double const materialDeposit = oldWeight - newWeight;

    particle.weight = newWeight;
    particle.frequency = this->frequencyForComptonGroup(targetGroup);

    if(!this->noHydroFeedback)
    {
        this->conserved[cellIndex].internal_energy += materialDeposit;
        this->conserved[cellIndex].energy += materialDeposit;
        if(this->withHydro && !this->diffusionPressureGradient)
        {
            this->conserved[cellIndex].momentum +=
                (oldWeight * oldVelocity - newWeight * particle.velocity) * units::inv_clight2;
        }
    }
    this->comptonImplicitMaterialExchange += materialDeposit;
    ++this->comptonImplicitEventCount;
}

void RadiationIMC::resetComptonDiagnostics()
{
    this->comptonSourceMaterialExchange = 0.0;
    this->comptonContinuousMaterialExchange = 0.0;
    this->comptonImplicitMaterialExchange = 0.0;
    this->comptonRemovalMaterialExchange = 0.0;
    this->comptonMinGroupEnergy = std::numeric_limits<double>::infinity();
    this->comptonMaxGroupEnergy = -std::numeric_limits<double>::infinity();
    this->comptonProjectedRadiationEnergy = 0.0;
    this->comptonMinFleck = std::numeric_limits<double>::infinity();
    this->comptonMaxFleck = -std::numeric_limits<double>::infinity();
    this->comptonMinGamma = std::numeric_limits<double>::infinity();
    this->comptonMaxGamma = -std::numeric_limits<double>::infinity();
    this->comptonMinUpsilon = std::numeric_limits<double>::infinity();
    this->comptonMaxUpsilon = -std::numeric_limits<double>::infinity();
    this->comptonNZeroFallbackCount = 0;
    this->comptonImplicitEventCount = 0;
    this->comptonOpacityLimitedGroupCount = 0;
    this->comptonProjectedNegativeGroupCount = 0;
}

void RadiationIMC::printComptonDiagnostics()
{
    if(!this->withCompton || !this->comptonDiagnostics)
        return;

    double sourceMaterialExchange = this->comptonSourceMaterialExchange;
    double continuousMaterialExchange = this->comptonContinuousMaterialExchange;
    double implicitMaterialExchange = this->comptonImplicitMaterialExchange;
    double removalMaterialExchange = this->comptonRemovalMaterialExchange;
    double minGroupEnergy = this->comptonMinGroupEnergy;
    double maxGroupEnergy = this->comptonMaxGroupEnergy;
    double projectedRadiationEnergy = this->comptonProjectedRadiationEnergy;
    double minFleck = this->comptonMinFleck;
    double maxFleck = this->comptonMaxFleck;
    double minGamma = this->comptonMinGamma;
    double maxGamma = this->comptonMaxGamma;
    double minUpsilon = this->comptonMinUpsilon;
    double maxUpsilon = this->comptonMaxUpsilon;
    size_t nZeroFallbackCount = this->comptonNZeroFallbackCount;
    size_t implicitEventCount = this->comptonImplicitEventCount;
    size_t opacityLimitedGroupCount = this->comptonOpacityLimitedGroupCount;
    size_t projectedNegativeGroupCount = this->comptonProjectedNegativeGroupCount;
    int rank = 0;

    #ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Allreduce(MPI_IN_PLACE, &sourceMaterialExchange, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &continuousMaterialExchange, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &implicitMaterialExchange, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &removalMaterialExchange, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minGroupEnergy, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxGroupEnergy, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &projectedRadiationEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minFleck, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxFleck, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minGamma, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxGamma, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minUpsilon, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxUpsilon, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &nZeroFallbackCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &implicitEventCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &opacityLimitedGroupCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &projectedNegativeGroupCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    #endif

    if(rank == 0)
    {
        std::cout << "Compton diagnostics:"
                  << " implicit_events=" << implicitEventCount
                  << " source_material_exchange=" << sourceMaterialExchange
                  << " continuous_material_exchange=" << continuousMaterialExchange
                  << " event_material_exchange=" << implicitMaterialExchange
                  << " removal_material_exchange=" << removalMaterialExchange
                  << " min_group_energy=" << minGroupEnergy
                  << " max_group_energy=" << maxGroupEnergy
                  << " projected_radiation_energy=" << projectedRadiationEnergy
                  << " fleck_min=" << minFleck
                  << " fleck_max=" << maxFleck
                  << " gamma_min=" << minGamma
                  << " gamma_max=" << maxGamma
                  << " upsilon_min=" << minUpsilon
                  << " upsilon_max=" << maxUpsilon
                  << " n_zero_fallback_cells=" << nZeroFallbackCount
                  << " opacity_limited_groups=" << opacityLimitedGroupCount
                  << " projected_negative_groups=" << projectedNegativeGroupCount
                  << std::endl;
    }
    this->resetComptonDiagnostics();
}

size_t RadiationIMC::sampleComptonCdf(const GroupCdf &cdf, double random) const
{
    double const value = std::clamp(random, 0.0, std::nextafter(1.0, 0.0));
    auto it = std::upper_bound(cdf.begin(), cdf.end(), value);
    if(it == cdf.begin())
        return 0;
    size_t group = static_cast<size_t>(std::distance(cdf.begin(), it)) - 1;
    if(group >= ENERGY_GROUPS_NUM)
        group = ENERGY_GROUPS_NUM - 1;
    return group;
}

void RadiationIMC::precomputeComptonData(double fullDt)
{
    if(!this->withCompton)
    {
        this->comptonData.clear();
        return;
    }

    this->initializeComptonGroups();
    this->initializeComptonMatrixGenerator();

    size_t const Ncells = this->grid.GetPointNo();
    this->comptonData.assign(Ncells, ComptonCellData{});

    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D const &cell = this->cells[i];
        ComptonCellData &data = this->comptonData[i];
        data.volume = this->grid.GetVolume(i);
        data.temperature = cell.temperature;
        data.Um = units::arad * boost::math::pow<4>(cell.temperature);
        data.cv = this->eos->dT2cv(cell.density, cell.temperature, cell.tracers, cell.tracerNames);
        if(data.cv <= 0.0)
        {
            UniversalError eo("Invalid heat capacity in RadiationIMC::precomputeComptonData");
            eo.addEntry("Cell index", i);
            eo.addEntry("cv", data.cv);
            throw eo;
        }
        data.beta = 4.0 * units::arad * boost::math::pow<3>(cell.temperature) / data.cv;

        double const kT = units::k_boltz * cell.temperature;
        double planckIntegralTotal = 0.0;
        if(kT > 0.0)
        {
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                double const a = ComputationalCell3D::energyBoundaries[g] / kT;
                double const b = ComputationalCell3D::energyBoundaries[g + 1] / kT;
                data.planckFraction[g] = planck_integral::planck_integral(a, b);
                planckIntegralTotal += data.planckFraction[g];
            }
        }

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(planckIntegralTotal > 0.0)
                data.planckFraction[g] /= planckIntegralTotal;
            else
                data.planckFraction[g] = 0.0;

            double absorptionOpacity = this->opacity->CalcAbsorptionOpacity(cell, this->comptonGroupCenters[g]);
            absorptionOpacity = std::min(absorptionOpacity, CG::max_coupling_strength / (units::clight * fullDt));
            double const groupRadiationEnergy = std::max(0.0, cell.Eg[g] * cell.density);
            double const groupExcess = groupRadiationEnergy - data.planckFraction[g] * data.Um;
            if(cell.density > 1e-12 &&
               groupExcess > 0.0 &&
               units::clight * fullDt * absorptionOpacity * groupExcess > 2.0 * data.cv * cell.temperature)
            {
                double const limitedOpacity =
                    2.0 * data.cv * cell.temperature / (units::clight * fullDt * groupExcess);
                if(limitedOpacity < absorptionOpacity)
                {
                    absorptionOpacity = limitedOpacity;
                    ++this->comptonOpacityLimitedGroupCount;
                }
            }
            if(absorptionOpacity < 0.0)
            {
                UniversalError eo("Negative absorption coefficient in RadiationIMC::precomputeComptonData");
                eo.addEntry("Cell index", i);
                eo.addEntry("Group", g);
                eo.addEntry("Absorption opacity", absorptionOpacity);
                throw eo;
            }
            data.absorptionOpacity[g] = absorptionOpacity;
            data.planckOpacity += data.absorptionOpacity[g] * data.planckFraction[g];
            data.oldRadiationEnergy[g] = groupRadiationEnergy;
        }
        this->planckOpacities[i] = data.planckOpacity;

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(data.planckOpacity > 0.0)
                data.baseSourceFraction[g] = data.absorptionOpacity[g] * data.planckFraction[g] / data.planckOpacity;
            else
                data.baseSourceFraction[g] = 0.0;
        }

        data.planckCdf = RadiationIMC::buildSafeComptonCdf(data.planckFraction);
        data.baseSourceCdf = RadiationIMC::buildSafeComptonCdf(data.baseSourceFraction);

        this->buildComptonMatricesForCell(cell, i, this->comptonUseInduced, data);
        this->recomputeComptonContractions(data);

        double const gamma = (this->withHydro && !this->MMC)
            ? 1.0 / std::sqrt(1.0 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2)
            : 1.0;
        double const cdtEff = units::clight * fullDt * gamma;
        double denom = 1.0 + data.beta * cdtEff * data.Gamma;
        bool negativeUpsilon = data.Upsilon < 0.0;
        if((denom <= 0.0 || (negativeUpsilon && std::abs(data.Upsilon) > 0.1 * data.planckOpacity)) &&
           this->comptonAllowNZeroFallback)
        {
            this->buildComptonMatricesForCell(cell, i, false, data);
            data.useNZero = true;
            ++this->comptonNZeroFallbackCount;
            this->recomputeComptonContractions(data);
            denom = 1.0 + data.beta * cdtEff * data.Gamma;
        }
        if(denom <= 0.0)
        {
            UniversalError eo("Compton Fleck denominator is nonpositive in RadiationIMC::precomputeComptonData");
            eo.addEntry("Cell index", i);
            eo.addEntry("Denominator", denom);
            eo.addEntry("Gamma", data.Gamma);
            eo.addEntry("Upsilon", data.Upsilon);
            eo.addEntry("Planck opacity", data.planckOpacity);
            eo.addEntry("Full dt", fullDt);
            throw eo;
        }
        data.fleck = 1.0 / denom;
        if(data.fleck < 0.0 || data.fleck > 1.0)
        {
            UniversalError eo("Invalid Compton-modified Fleck factor in RadiationIMC::precomputeComptonData");
            eo.addEntry("Cell index", i);
            eo.addEntry("Fleck", data.fleck);
            eo.addEntry("Gamma", data.Gamma);
            eo.addEntry("Upsilon", data.Upsilon);
            eo.addEntry("Planck opacity", data.planckOpacity);
            throw eo;
        }
        this->factorFleck[i] = data.fleck;
        this->comptonMinFleck = std::min(this->comptonMinFleck, data.fleck);
        this->comptonMaxFleck = std::max(this->comptonMaxFleck, data.fleck);
        this->comptonMinGamma = std::min(this->comptonMinGamma, data.Gamma);
        this->comptonMaxGamma = std::max(this->comptonMaxGamma, data.Gamma);
        this->comptonMinUpsilon = std::min(this->comptonMinUpsilon, data.Upsilon);
        this->comptonMaxUpsilon = std::max(this->comptonMaxUpsilon, data.Upsilon);
        data.betaCdtF = data.beta * cdtEff * data.fleck;
        if(std::abs(data.Gamma) > 1e-200)
            data.betaCdtF = (1.0 - data.fleck) / data.Gamma;

        this->buildComptonInPlaceKernels(i, data);
        if(this->comptonDebugParityCheck)
            this->validateComptonParity(i, data);
        this->buildComptonSources(fullDt, data);
        data.active = true;
    }
}

std::vector<typename RadiationIMC::Particle> RadiationIMC::preStep(double fullDt)
{
    assert(this->cells.size() >= this->grid.GetPointNo());
    assert(this->conserved.size() >= this->grid.GetPointNo());

    size_t Ncells = this->grid.GetPointNo();
    this->factorFleck = std::vector<double>(Ncells);
    this->planckOpacities = std::vector<double>(Ncells);
    this->Erad_time_avg = std::vector<double>(Ncells, 0);
    if(this->withEgTimeAvg && this->multigroupOpacity)
    {
        std::array<double, ENERGY_GROUPS_NUM> zeros{};
        this->Eg_time_avg.assign(Ncells, zeros);
    }
    if(this->multigroupOpacity)
    {
        this->multigroupOpacity->ResetCummulativeOpacityCellID();
    }
    if(this->withCompton)
    {
        this->resetComptonDiagnostics();
    }
    for(size_t i = 0; i < Ncells; i++)
    {
        const ComputationalCell3D &cell = this->cells[i];
        double gamma = (this->withHydro && !this->MMC)? 1 / std::sqrt(1 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2) : 1;

        if(this->withCompton)
        {
            this->planckOpacities[i] = 0.0;
            this->factorFleck[i] = 1.0;
        }
        else
        {
            this->planckOpacities[i] = this->opacity->CalcPlanckOpacity(this->cells[i]);
            double cv = this->eos->dT2cv(this->cells[i].density, this->cells[i].temperature, this->cells[i].tracers, this->cells[i].tracerNames);
            this->factorFleck[i] = 1 / (1 + (4 * units::arad * boost::math::pow<3>(this->cells[i].temperature) * this->planckOpacities[i] * units::clight * fullDt * gamma) / cv);
            if(this->factorFleck[i] < 0 or this->factorFleck[i] > 1)
            {
                UniversalError eo("Invalid factor fleck in RadiationIMC::preStep");
                eo.addEntry("Factor fleck", this->factorFleck[i]);
                eo.addEntry("Planck opacity", this->planckOpacities[i]);
                eo.addEntry("Temperature", this->cells[i].temperature);
                eo.addEntry("Density", this->cells[i].density);
                eo.addEntry("Gamma", gamma);
                eo.addEntry("cv", cv);
                eo.addEntry("Full dt", fullDt);
                throw eo;
            }
        }
    }

    this->precomputeComptonData(fullDt);

    if(this->withRandomWalk)
    {
        this->precomputeRandomWalkData();
        this->rwStepCount = 0;
    }

    std::vector<Particle> newParticles = this->generateParticles(fullDt);
    std::vector<Particle> newParticles2 = this->boundary->generateNewBoundaryParticles(fullDt); // todo: not here
    for(Particle &particle : newParticles2)
    {
        SetInitialWeightFromWeight(particle);
    }
    newParticles.insert(newParticles.end(), newParticles2.begin(), newParticles2.end());
    // auto printParticles = [&]()
    // {
    //     rank_t rank;
    //     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //     for(MCParticle &p : newParticles)
    //     {
    //         std::cout << "Rank " << rank << ": " << p << std::endl;
    //     }
    // };
    // MPI_Sync(MPI_COMM_WORLD, printParticles);
    return newParticles;
}

std::vector<typename RadiationIMC::Particle> RadiationIMC::generateInitialParticles(size_t particlesPerCell)
{
    std::vector<Particle> result;
    size_t Ncells = this->grid.GetPointNo();

    std::array<double, ENERGY_GROUPS_NUM + 1> cumulPlanck;
    bool hasPlanckTable = false;
    double cachedTemperature = -1;

    for(size_t i = 0; i < Ncells; i++)
    {
        double totalErad = this->cells[i].Erad * this->cells[i].density * this->grid.GetVolume(i);
        double weightPerPhoton = totalErad / particlesPerCell;
        if(weightPerPhoton <= 0)
            continue;

        if(this->multigroupOpacity && !this->withCompton && (!hasPlanckTable || this->cells[i].temperature != cachedTemperature))
        {
            double kT = units::k_boltz * this->cells[i].temperature;
            cumulPlanck[0] = 0.0;
            for(size_t g = 1; g <= ENERGY_GROUPS_NUM; g++)
            {
                double a = ComputationalCell3D::energyBoundaries[g - 1] / kT;
                double b = ComputationalCell3D::energyBoundaries[g] / kT;
                cumulPlanck[g] = planck_integral::planck_integral(a, b) + cumulPlanck[g - 1];
            }
            cachedTemperature = this->cells[i].temperature;
            hasPlanckTable = true;
        }
        GroupCdf initialGroupEnergyCdf{};
        if(this->withCompton && this->multigroupOpacity)
        {
            initialGroupEnergyCdf[0] = 0.0;
            double cumulativeEnergy = 0.0;
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                cumulativeEnergy += std::max(0.0, this->cells[i].Eg[g] * this->cells[i].density * this->grid.GetVolume(i));
                initialGroupEnergyCdf[g + 1] = cumulativeEnergy;
            }
            if(cumulativeEnergy <= 0.0)
            {
                UniversalError eo("Compton initial census has nonpositive multigroup radiation energy");
                eo.addEntry("Cell index", i);
                eo.addEntry("Erad", this->cells[i].Erad);
                throw eo;
            }
        }

        for(size_t j = 0; j < particlesPerCell; j++)
        {
            Particle p = this->generateSingleParticle(i, this->cells[i]);
            if(this->withCompton && this->multigroupOpacity)
            {
                double const r = this->dist(this->re) * initialGroupEnergyCdf.back();
                p.frequency = LinearInterpolation(initialGroupEnergyCdf, ComputationalCell3D::energyBoundaries, r);
            }
            else if(this->multigroupOpacity)
            {
                double rnd = this->dist(this->re);
                double r = rnd * cumulPlanck.back();
                p.frequency = LinearInterpolation(cumulPlanck, ComputationalCell3D::energyBoundaries, r);
            }
            p.cellID = this->cells[i].ID;
            p.weight = weightPerPhoton;
            SetInitialWeightFromWeight(p);
            result.push_back(p);
        }
    }
    return result;
}

typename RadiationIMC::Particle RadiationIMC::generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const
{
    Particle particle;
    particle.id = std::numeric_limits<size_t>::max();
    particle.frequency = 0; // TODO
    particle.location = RandomPointInCell(this->grid, cellIndex);
    // particle.location = particle.location * (1 - MONTECARLO_EPS) + MONTECARLO_EPS * this->grid.GetMeshPoint(cellIndex);
    particle.timeLeft = 0;
    assert(this->grid.IsPointInCell(particle.location, cellIndex));
    assert(not this->grid.IsPointOutsideBox(particle.location));
    particle.velocity = this->opacity->getRandomVelocity(cell);
    if(this->withHydro && !this->MMC)
    {
        LorentzTransformation(particle, -1 * cell.velocity);
    }
    particle.cellIndex = cellIndex;
    // nudge a little bit towards the cell's point
    static constexpr double nudge = 1e-10;
    particle.location = particle.location * (1 - nudge) + nudge * this->grid.GetMeshPoint(cellIndex);
    return particle;
}

void RadiationIMC::adjustExistingParticles(std::vector<Particle> &particles, double fullDt)
{
    if(!this->MMC)
    {
        return;
    }

    size_t Ncells = this->grid.GetPointNo();
    std::vector<double> divV(Ncells, 0);

    std::vector<size_t> neigh;
    for(size_t i = 0; i < Ncells; i++)
    {
        this->grid.GetNeighbors(i, neigh);
        const auto &faces = this->grid.GetCellFaces(i);
        Vector3D r_i = this->grid.GetMeshPoint(i);
        for(size_t j = 0; j < neigh.size(); j++)
        {
            size_t neighbor_j = neigh[j];
            auto r_ij = normalize(r_i - this->grid.GetMeshPoint(neighbor_j));
            double A_ij = this->grid.GetArea(faces[j]);
            Vector3D v_j = (neighbor_j >= Ncells && this->grid.IsPointOutsideBox(neighbor_j))
                           ? this->cells[i].velocity
                           : this->cells[neighbor_j].velocity;
            divV[i] -= 0.5 * ScalarProd(this->cells[i].velocity + v_j, r_ij) * A_ij;
        }
        divV[i] /= this->grid.GetVolume(i);
    }

    const auto [ll, ur] = this->grid.GetBoxCoordinates();

    auto it = particles.begin();
    while(it != particles.end())
    {
        Particle &p = *it;
        size_t ci = p.cellIndex;
        p.location += this->cells[ci].velocity * fullDt;
        p.weight += -p.weight * fullDt * divV[ci] / 3.0;

        if(this->grid.IsPointOutsideBox(p.location))
        {
            p.location.x = std::max(ll.x, std::min(ur.x, p.location.x));
            p.location.y = std::max(ll.y, std::min(ur.y, p.location.y));
            p.location.z = std::max(ll.z, std::min(ur.z, p.location.z));
            MonteCarloParticleStatus status = this->boundary->apply(p);
            if(status == MonteCarloParticleStatus::REMOVE)
            {
                it = particles.erase(it);
                continue;
            }
        }
        ++it;
    }

    UpdateNewCells(this->grid, particles, this->cells);
}

std::ostream &operator<<(std::ostream &os, const RadiationIMCParameters &parameters)
{
    os << "IMC, with parameters:" << std::endl;
    os << "\t" << "new photons per cell: " << parameters.newPhotonsPerCell << std::endl;
    os << "\t" << "with hydro: " << parameters.withHydro << std::endl;
    os << "\t" << "diffusion pressure gradient: " << parameters.diffusionPressureGradient << std::endl;
    os << "\t" << "MMC: " << parameters.MMC << std::endl;
    os << "\t" << "with multigroup opacity: " << parameters.withMultigroupOpacity << std::endl;
    os << "\t" << "with random walk: " << parameters.withRandomWalk << std::endl;
    os << "\t" << "with Compton: " << parameters.withCompton << std::endl;
    if(parameters.withCompton)
    {
        os << "\t" << "Compton induced terms: " << parameters.comptonUseInduced << std::endl;
        os << "\t" << "Compton n=0 fallback: " << parameters.comptonAllowNZeroFallback << std::endl;
        os << "\t" << "Compton debug parity check: " << parameters.comptonDebugParityCheck << std::endl;
        os << "\t" << "Compton transport mode: implicit in-place signed scattering" << std::endl;
        os << "\t" << "Compton diagnostics: " << parameters.comptonDiagnostics << std::endl;
        os << "\t" << "Compton matrix samples: " << parameters.comptonMatrixSamples << std::endl;
    }
    if(parameters.withRandomWalk)
    {
        os << "\t" << "RW min cell optical depth: " << parameters.rwMinCellOpticalDepth << std::endl;
        os << "\t" << "RW min particle optical depth: " << parameters.rwMinParticleOpticalDepth << std::endl;
    }
    os << "\t" << "no hydro feedback: " << parameters.noHydroFeedback << std::endl;
    return os;
}
