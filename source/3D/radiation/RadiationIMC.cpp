#include "RadiationIMC.hpp"
#include "mpi/mpi_commands_3d.hpp"
#include "Radiation/conj_grad_solve.hpp"
#include <cmath>
#include <limits>
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

    struct SolverDiagnostics
    {
        double minPivot = std::numeric_limits<double>::infinity();
        double maxCoeff = 0.0;
    };

    bool SolveComptonGroupSystem(RadiationIMC::GroupMatrix matrix,
                                 RadiationIMC::GroupArray rhs,
                                 RadiationIMC::GroupArray &solution,
                                 SolverDiagnostics &diag)
    {
        diag.minPivot = std::numeric_limits<double>::infinity();
        diag.maxCoeff = 0.0;
        for(size_t r = 0; r < ENERGY_GROUPS_NUM; r++)
            for(size_t c = 0; c < ENERGY_GROUPS_NUM; c++)
                diag.maxCoeff = std::max(diag.maxCoeff, std::abs(matrix[r][c]));

        for(size_t col = 0; col < ENERGY_GROUPS_NUM; col++)
        {
            size_t pivot = col;
            double pivotAbs = std::abs(matrix[col][col]);
            for(size_t row = col + 1; row < ENERGY_GROUPS_NUM; row++)
            {
                double const candidateAbs = std::abs(matrix[row][col]);
                if(candidateAbs > pivotAbs)
                {
                    pivot = row;
                    pivotAbs = candidateAbs;
                }
            }
            if(pivotAbs <= 1e-200)
                return false;
            diag.minPivot = std::min(diag.minPivot, pivotAbs);
            if(pivot != col)
            {
                std::swap(matrix[pivot], matrix[col]);
                std::swap(rhs[pivot], rhs[col]);
            }

            double const pivotValue = matrix[col][col];
            for(size_t row = col + 1; row < ENERGY_GROUPS_NUM; row++)
            {
                double const factor = matrix[row][col] / pivotValue;
                if(factor == 0.0)
                    continue;
                matrix[row][col] = 0.0;
                for(size_t j = col + 1; j < ENERGY_GROUPS_NUM; j++)
                    matrix[row][j] -= factor * matrix[col][j];
                rhs[row] -= factor * rhs[col];
            }
        }

        for(size_t rev = 0; rev < ENERGY_GROUPS_NUM; rev++)
        {
            size_t const row = ENERGY_GROUPS_NUM - 1 - rev;
            double value = rhs[row];
            for(size_t col = row + 1; col < ENERGY_GROUPS_NUM; col++)
                value -= matrix[row][col] * solution[col];
            solution[row] = value / matrix[row][row];
            if(!std::isfinite(solution[row]))
                return false;
        }
        return true;
    }
}

    RadiationIMC::RadiationIMC(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, std::shared_ptr<EquationOfState> eos, std::shared_ptr<OpacityCalculator> opacity, RadiationIMCParameters parameters)
    : MonteCarloRadiationPhysics3D(grid, boundary, cells, conserved, eos, opacity), withHydro(parameters.withHydro), diffusionPressureGradient(parameters.diffusionPressureGradient), MMC(parameters.MMC), newPhotonsPerCell(parameters.newPhotonsPerCell), withRandomWalk(parameters.withRandomWalk), rwMinCellOpticalDepth(parameters.rwMinCellOpticalDepth), rwMinParticleOpticalDepth(parameters.rwMinParticleOpticalDepth), noHydroFeedback(parameters.noHydroFeedback), withEgTimeAvg(parameters.withEgTimeAvg), withCompton(parameters.withCompton), comptonUseInduced(parameters.comptonUseInduced), comptonAllowNZeroFallback(parameters.comptonAllowNZeroFallback), comptonAngleDependent(parameters.comptonAngleDependent), comptonMatrixSamples(parameters.comptonMatrixSamples)
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
    bool const comptonMcGroup = this->withCompton && group < ENERGY_GROUPS_NUM;
    double elasticScatteringOpacity = this->withCompton ? 0.0 : this->opacity->CalcScatteringOpacity(cell);
    double effectiveAbsorptionOpacity;
    if(comptonMcGroup)
        effectiveAbsorptionOpacity = (1.0 - this->comptonData[cellIndex].fleck) * absorptionOpacity;
    else if(this->withCompton)
        effectiveAbsorptionOpacity = this->comptonData[cellIndex].baseEffectiveOpacity[group];
    else
        effectiveAbsorptionOpacity = (1 - this->factorFleck[cellIndex]) * absorptionOpacity;
    double implicitComptonOpacity =
        comptonMcGroup ? this->comptonData[cellIndex].comptonOutRate[group] : 0.0;
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
    double const fleckForDecay = (this->withCompton && group < ENERGY_GROUPS_NUM)
        ? this->comptonData[cellIndex].fleck
        : this->factorFleck[cellIndex];
    double weightEvolutionOpacity = absorptionOpacity * fleckForDecay;
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
            this->conserved[cellIndex].energy += materialDeposit;
        if(this->withHydro)
        {
            if(not this->diffusionPressureGradient)
            {
                this->conserved[cellIndex].momentum += -expFactor1 * particle.weight * particle.velocity * units::inv_clight2;
            }
        }
    }
    this->Erad_time_avg[cellIndex] += integratedForTally;
    if((this->withEgTimeAvg || this->withCompton) && this->multigroupOpacity)
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
                this->conserved[cellIndex].energy += particle.weight;
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
            this->applyComptonScatterEvent(cellIndex, cell, group, oldVelocity, oldWeight, dopplerShift, particle);
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
        if((this->withEgTimeAvg || this->withCompton) && this->multigroupOpacity)
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
                if(!this->withCompton)
                {
                    UniversalError eo("Negative internal energy in RadiationIMC::postStep");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Internal energy", cell.internal_energy);
                    eo.addEntry("Mass", this->conserved[i].mass);
                    eo.addEntry("Density", cell.density);
                    eo.addEntry("Temperature", cell.temperature);
                    throw eo;
                }
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
            if(cell.internal_energy >= 0.0)
            {
                cell.temperature = this->eos->de2T(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
                cell.pressure = this->eos->de2p(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
            }
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
    for(const Particle &particle : particles)
    {
        size_t cellIndex = particle.cellIndex;
        assert(cellIndex < Ncells);
        this->conserved[cellIndex].Erad += particle.weight;
        if(this->multigroupOpacity)
        {
            size_t g = this->opacity->findGroup(particle.frequency);
            this->conserved[cellIndex].Eg[g] += particle.weight;
        }
    }
    if(this->withCompton && this->multigroupOpacity)
    {
        this->applyComptonEndOfStepCorrection(fullDt);
        // Reconcile particles to match corrected Eg[g] immediately, rather than
        // deferring to the start of the next step. All callers pass non-const locals.
        this->reconcileComptonParticles(const_cast<std::vector<Particle>&>(particles));
        if(!this->noHydroFeedback)
        {
            for(size_t i = 0; i < Ncells; i++)
            {
                ComputationalCell3D &cell = this->cells[i];
                cell.internal_energy = this->conserved[i].internal_energy / this->conserved[i].mass;
                if(cell.internal_energy < 0.0)
                {
                    UniversalError eo("Negative internal energy after Compton end-of-step correction in RadiationIMC::postStep");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Internal energy", cell.internal_energy);
                    eo.addEntry("Mass", this->conserved[i].mass);
                    eo.addEntry("Density", cell.density);
                    eo.addEntry("Temperature", cell.temperature);
                    throw eo;
                }
                cell.temperature = this->eos->de2T(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
                cell.pressure = this->eos->de2p(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
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
}

void RadiationIMC::applyComptonEndOfStepCorrection(double fullDt)
{
    size_t const Ncells = this->grid.GetPointNo();
    if(this->comptonData.size() != Ncells)
        throw UniversalError("Compton data not initialized in applyComptonEndOfStepCorrection");

    for(size_t i = 0; i < Ncells; i++)
    {
        ComptonCellData const &cd = this->comptonData[i];
        GroupArray rawGroupEnergy{};
        GroupArray rhs{};
        GroupArray solvedGroupEnergy{};
        GroupMatrix residualMatrix{};
        double totalCorrectionToRadiation = 0.0;

        // Scale Bcorr by ratio of post-transport to pre-step Erad to prevent
        // overcorrection when transport has depleted radiation from this cell.
        double totalPreStepErad = 0.0;
        double totalPostTransportErad = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            rawGroupEnergy[g] = this->conserved[i].Eg[g];
            totalPostTransportErad += rawGroupEnergy[g];
            totalPreStepErad += cd.oldRadiationEnergy[g];
        }
        totalPostTransportErad = std::max(0.0, totalPostTransportErad);
        double const preStepExtensive = totalPreStepErad * cd.volume;
        double const bcorrScale = (preStepExtensive > 0.0)
            ? std::clamp(totalPostTransportErad / preStepExtensive, 0.01, 1.0)
            : 1.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            rhs[g] = rawGroupEnergy[g] + bcorrScale * cd.Bcorr[g];

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
            {
                residualMatrix[g][h] = ((g == h) ? 1.0 : 0.0)
                    - fullDt * units::clight * cd.residualKernel[h][g];
            }
        }

        SolverDiagnostics diag;
        if(!SolveComptonGroupSystem(residualMatrix, rhs, solvedGroupEnergy, diag))
        {
            UniversalError eo("Failed to solve end-of-step Compton group system");
            eo.addEntry("Cell index", static_cast<double>(i));
            eo.addEntry("Full dt", fullDt);
            eo.addEntry("Gamma", cd.Gamma);
            eo.addEntry("Fleck", cd.fleck);
            eo.addEntry("Upsilon", cd.Upsilon);
            eo.addEntry("Min pivot", diag.minPivot);
            eo.addEntry("Max coefficient", diag.maxCoeff);
            throw eo;
        }

        double totalErad = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            totalErad += rawGroupEnergy[g];

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double finalGroupEnergy = solvedGroupEnergy[g];
            if(finalGroupEnergy < 0.0)
            {
                // Relaxed from 1e-4 to tolerate small residuals from bcorrScale clamping
                if(std::abs(finalGroupEnergy) > 1e-2 * std::max(totalErad, rawGroupEnergy[g]))
                {
                    UniversalError eo("End-of-step correction produced significant negative group energy");
                    eo.addEntry("Cell index", static_cast<double>(i));
                    eo.addEntry("Group", static_cast<double>(g));
                    eo.addEntry("Eg_raw", rawGroupEnergy[g]);
                    eo.addEntry("Solved energy", finalGroupEnergy);
                    eo.addEntry("Total Erad", totalErad);
                    eo.addEntry("Fleck", cd.fleck);
                    eo.addEntry("Upsilon", cd.Upsilon);
                    eo.addEntry("Planck opacity", cd.planckOpacity);
                    eo.addEntry("Gamma", cd.Gamma);
                    eo.addEntry("Min pivot", diag.minPivot);
                    eo.addEntry("Max coefficient", diag.maxCoeff);
                    eo.addEntry("Cell volume", this->grid.GetVolume(i));
                    eo.addEntry("Cell", this->cells[i]);
                    throw eo;
                }
                finalGroupEnergy = 0.0;
            }

            totalCorrectionToRadiation += finalGroupEnergy - rawGroupEnergy[g];
            this->conserved[i].Eg[g] = finalGroupEnergy;
        }

        this->conserved[i].Erad += totalCorrectionToRadiation;
        if(!this->noHydroFeedback)
        {
            double const materialDeposit = -totalCorrectionToRadiation;
            this->conserved[i].internal_energy += materialDeposit;
            this->conserved[i].energy += materialDeposit;
            if(this->conserved[i].internal_energy < 0.0)
            {
                UniversalError eo("Negative internal energy after Compton end-of-step correction");
                eo.addEntry("Cell index", static_cast<double>(i));
                eo.addEntry("Material deposit", materialDeposit);
                eo.addEntry("Internal energy", this->conserved[i].internal_energy);
                eo.addEntry("Total radiation delta", totalCorrectionToRadiation);
                eo.addEntry("Fleck", cd.fleck);
                eo.addEntry("Gamma", cd.Gamma);
                throw eo;
            }
            ComputationalCell3D &cell = this->cells[i];
            cell.internal_energy = this->conserved[i].internal_energy / this->conserved[i].mass;
            cell.temperature = this->eos->de2T(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
            cell.pressure = this->eos->de2p(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
        }
    }
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

    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        double energyToCreate = energyToCreateVec[i];
        double gamma = gammaVec[i];

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
        size_t nPhotonsCell = nPhotons[i];
        double energyPerPhoton = energyToCreate * gamma / nPhotonsCell;
        for(size_t j = 0; j < nPhotonsCell; j++)
        {
            MCParticle particle = this->generateSingleParticle(i, cell);
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
    if(this->comptonData.size() != Ncells)
        throw UniversalError("Compton data is not initialized in RadiationIMC::generateComptonParticles");

    std::vector<double> absEnergyToCreateVec(Ncells, 0.0);
    for(size_t i = 0; i < Ncells; i++)
    {
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            absEnergyToCreateVec[i] += this->comptonData[i].Bbase[g];
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
            absGroupEnergy[g] = cd.Bbase[g];
            if(absGroupEnergy[g] > 0.0)
                ++nonzeroGroups;
        }
        size_t const nPhotonsCell = std::max(this->newPhotonsPerCell, nonzeroGroups);
        if(nPhotonsCell == 0 || absEnergyToCreateVec[i] <= 0.0)
            continue;

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
            double const groupEnergy = cd.Bbase[g];
            if(ng == 0 || groupEnergy == 0.0)
                continue;

            if(!this->noHydroFeedback)
            {
                double const materialDeposit = -groupEnergy;
                this->conserved[i].internal_energy += materialDeposit;
                this->conserved[i].energy -= groupEnergy * gamma;
                if(this->withHydro && !this->diffusionPressureGradient)
                    this->conserved[i].momentum -= groupEnergy * cell.velocity * units::inv_clight2 * gamma;
            }

            double const packetEnergy = groupEnergy / static_cast<double>(ng);
            for(size_t j = 0; j < ng; j++)
            {
                Particle particle = this->generateSingleParticle(i, cell);
                particle.timeLeft = fullDt * this->dist(this->re);
                particle.cellID = cell.ID;
                if(this->withHydro && !this->MMC)
                {
                    double const D = DopplerShift(particle, cell.velocity);
                    particle.frequency = this->frequencyForComptonGroup(g) / D;
                    particle.weight = packetEnergy / D;
                }
                else
                {
                    particle.frequency = this->frequencyForComptonGroup(g);
                    particle.weight = packetEnergy;
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
    double frequency = 0.5 * (ComputationalCell3D::energyBoundaries[group] +
                              ComputationalCell3D::energyBoundaries[group + 1]);
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
        cd.Bpos[g] = std::max(0.0, cd.Btotal[g]);
        cd.Bres[g] = cd.Btotal[g] - cd.Bpos[g];
    }
}

void RadiationIMC::buildComptonEventData(size_t cellIndex, ComptonCellData &cd)
{
    (void)cellIndex;
    ZeroGroupMatrix(cd.segmentKernel);
    ZeroGroupMatrix(cd.residualKernel);

    for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        GroupArray weights{};
        double outRateSum = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(g == h)
                continue;
            weights[g] = std::max(0.0, cd.tau[h][g] * (1.0 + cd.occupation[g]));
            outRateSum += weights[g];
        }
        cd.comptonOutRate[h] = outRateSum;
        cd.comptonTargetCdf[h] = RadiationIMC::buildSafeComptonCdf(weights);

        cd.baseEffectiveOpacity[h] = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double const kgbg = cd.absorptionOpacity[g] * cd.planckFraction[g];
            cd.baseEffectiveOpacity[h] += cd.betaCdtF * cd.absorptionOpacity[h] * kgbg;
        }

        double const f = cd.fleck;

        double mu = 0.0;
        if(outRateSum > 0.0)
        {
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                if(g == h)
                    continue;
                double const q_hg = weights[g] / outRateSum;
                double const r_hg = this->comptonGroupCenters[g] / this->comptonGroupCenters[h];
                mu += q_hg * r_hg;
            }
        }
        else
        {
            mu = 1.0;
        }
        cd.comptonMu[h] = mu;
        cd.comptonMh[h] = 1.0 + f * (mu - 1.0);

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double const kgbg = cd.absorptionOpacity[g] * cd.planckFraction[g];
            double const Ktotal_hg = cd.S[h][g] + cd.betaCdtF * cd.M[g] * cd.Lambda[h];
            double const Hbase_hg = cd.betaCdtF * cd.absorptionOpacity[h] * kgbg;
            double const Ktarget_hg = Ktotal_hg - Hbase_hg;
            cd.segmentKernel[h][g] = Ktarget_hg;

            double const diag = (h == g) ? -cd.absorptionOpacity[h] : 0.0;
            cd.Ktotal[h][g] = diag + Ktotal_hg;

            double Kimc_hg = (h == g)
                ? -cd.absorptionOpacity[h] + (1.0 - f) * cd.absorptionOpacity[h] * cd.baseSourceFraction[h]
                : (1.0 - f) * cd.absorptionOpacity[h] * cd.baseSourceFraction[g];
            double Kevent_new_hg;
            if(h == g)
                Kevent_new_hg = -cd.comptonOutRate[h];
            else
            {
                double const q_hg = (outRateSum > 0.0) ? weights[g] / outRateSum : 0.0;
                Kevent_new_hg = cd.comptonOutRate[h] * q_hg * cd.comptonMh[h];
            }
            cd.residualKernel[h][g] = cd.Ktotal[h][g] - Kimc_hg - Kevent_new_hg;
        }
    }
}

void RadiationIMC::applyComptonScatterEvent(size_t cellIndex, const ComputationalCell3D &cell, size_t sourceGroup, const Vector3D &oldVelocity, double oldWeight, double dopplerShift, Particle &particle)
{
    (void)dopplerShift;
    if(sourceGroup >= ENERGY_GROUPS_NUM)
        return;

    ComptonCellData &cd = this->comptonData[cellIndex];
    if(cd.comptonOutRate[sourceGroup] <= 0.0)
        return;

    size_t targetGroup = this->sampleComptonCdf(
        cd.comptonTargetCdf[sourceGroup], this->dist(this->re));
    if(targetGroup == sourceGroup || targetGroup >= ENERGY_GROUPS_NUM)
    {
        UniversalError eo("Compton CDF sampling returned invalid target group");
        eo.addEntry("Cell index", static_cast<double>(cellIndex));
        eo.addEntry("Source group", static_cast<double>(sourceGroup));
        eo.addEntry("Target group", static_cast<double>(targetGroup));
        eo.addEntry("comptonOutRate", cd.comptonOutRate[sourceGroup]);
        throw eo;
    }

    if(this->comptonAngleDependent)
    {
        static thread_local std::vector<double> angleCdf;
        this->comptonMatrixGen->get_angle_cdf(cd.temperature, sourceGroup, targetGroup, angleCdf);

        double const r = this->dist(this->re);
        std::size_t const N = ComptonMatrixMC::NUM_ANGLE_BINS;

        auto it = std::upper_bound(angleCdf.begin(), angleCdf.end(), r);
        std::size_t bin = 0;
        if(it != angleCdf.begin())
            bin = static_cast<std::size_t>(std::distance(angleCdf.begin(), it)) - 1;
        if(bin >= N)
            bin = N - 1;

        double const binWidth = 2.0 / static_cast<double>(N);
        double frac = 0.0;
        double const denom = angleCdf[bin + 1] - angleCdf[bin];
        if(denom > 0.0)
            frac = (r - angleCdf[bin]) / denom;
        double const cosTheta = -1.0 + (static_cast<double>(bin) + frac) * binWidth;
        double const sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
        double const phi = 2.0 * M_PI * this->dist(this->re);

        Vector3D const oldDir = normalize(oldVelocity);

        Vector3D perp1;
        if(std::abs(oldDir.z) < 0.9)
            perp1 = normalize(CrossProduct(oldDir, Vector3D(0, 0, 1)));
        else
            perp1 = normalize(CrossProduct(oldDir, Vector3D(1, 0, 0)));
        Vector3D const perp2 = CrossProduct(oldDir, perp1);

        Vector3D const newDir = oldDir * cosTheta
            + (perp1 * std::cos(phi) + perp2 * std::sin(phi)) * sinTheta;

        particle.velocity = normalize(newDir) * units::clight;
    }
    else
    {
        particle.velocity = this->opacity->getNewScatterVelocity(cell, particle);
    }

    double const mh = cd.comptonMh[sourceGroup];
    double const newWeight = oldWeight * mh;
    double const materialDeposit = oldWeight * (1.0 - mh);

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
        data.betaCdtF = data.beta * cdtEff * data.fleck;
        if(std::abs(data.Gamma) > 1e-200)
            data.betaCdtF = (1.0 - data.fleck) / data.Gamma;

        this->buildComptonEventData(i, data);
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
    if((this->withEgTimeAvg || this->withCompton) && this->multigroupOpacity)
    {
        std::array<double, ENERGY_GROUPS_NUM> zeros{};
        this->Eg_time_avg.assign(Ncells, zeros);
    }
    if(this->multigroupOpacity)
    {
        this->multigroupOpacity->ResetCummulativeOpacityCellID();
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

void RadiationIMC::reconcileComptonParticles(std::vector<Particle> &particles)
{
    if(!this->withCompton || !this->multigroupOpacity)
        return;

    size_t const Ncells = this->grid.GetPointNo();
    std::vector<GroupArray> rawGroupEnergy(Ncells, GroupArray{});
    for(const Particle &particle : particles)
    {
        if(particle.cellIndex >= Ncells)
            continue;
        if(particle.weight < 0.0)
        {
            UniversalError eo("Negative particle weight in positive-only Compton reconciliation");
            eo.addEntry("Cell index", particle.cellIndex);
            eo.addEntry("Particle weight", particle.weight);
            throw eo;
        }
        double frequency = particle.frequency;
        ClampFrequencyToBounds(frequency);
        size_t const g = this->opacity->findGroup(frequency);
        rawGroupEnergy[particle.cellIndex][g] += particle.weight;
    }

    std::vector<GroupArray> scale(Ncells, GroupArray{});
    for(size_t i = 0; i < Ncells; i++)
    {
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double const target = std::max(0.0, this->conserved[i].Eg[g]);
            double const raw = rawGroupEnergy[i][g];
            scale[i][g] = 1.0;
            if(raw > 0.0 && target < raw)
                scale[i][g] = target / raw;
        }
    }

    auto it = particles.begin();
    while(it != particles.end())
    {
        Particle &particle = *it;
        if(particle.cellIndex < Ncells)
        {
            double frequency = particle.frequency;
            ClampFrequencyToBounds(frequency);
            size_t const g = this->opacity->findGroup(frequency);
            particle.weight *= scale[particle.cellIndex][g];
            SetInitialWeightFromWeight(particle);
        }
        if(particle.weight <= 0.0)
            it = particles.erase(it);
        else
            ++it;
    }

    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D const &cell = this->cells[i];
        GroupArray deficits{};
        GroupArray fractional{};
        std::array<size_t, ENERGY_GROUPS_NUM> groupCounts{};
        double totalDeficit = 0.0;
        size_t nonzeroDeficitGroups = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double const target = std::max(0.0, this->conserved[i].Eg[g]);
            double const represented = (rawGroupEnergy[i][g] > target) ? target : rawGroupEnergy[i][g];
            double const deficit = target - represented;
            double const tolerance = 1e-12 * std::max(target, 1.0);
            if(deficit <= tolerance)
                continue;

            deficits[g] = deficit;
            totalDeficit += deficit;
            ++nonzeroDeficitGroups;
        }
        if(totalDeficit <= 0.0)
            continue;

        size_t const correctionBudget = std::max(this->newPhotonsPerCell, nonzeroDeficitGroups);
        size_t allocated = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(deficits[g] <= 0.0)
                continue;
            groupCounts[g] = 1;
            ++allocated;
        }

        size_t const remainingBudget = correctionBudget - allocated;
        size_t extraAllocated = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(deficits[g] <= 0.0)
                continue;
            double const exactExtra = static_cast<double>(remainingBudget) * deficits[g] / totalDeficit;
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
                if(deficits[g] > 0.0 && fractional[g] > bestFraction)
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
            if(ng == 0)
                continue;
            double const packetWeight = deficits[g] / static_cast<double>(ng);
            for(size_t j = 0; j < ng; j++)
            {
                Particle particle = this->generateSingleParticle(i, cell);
                particle.cellID = cell.ID;
                particle.frequency = this->frequencyForComptonGroup(g);
                particle.weight = packetWeight;
                SetInitialWeightFromWeight(particle);
                particles.push_back(particle);
            }
        }
    }
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
        os << "\t" << "Compton angle-dependent: " << parameters.comptonAngleDependent << std::endl;
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
