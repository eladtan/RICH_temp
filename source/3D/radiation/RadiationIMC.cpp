#include "RadiationIMC.hpp"
#include "mpi/mpi_commands_3d.hpp"

// #define MONTECARLO_EPS 1e-7

namespace {
    inline void ClampFrequencyToBounds(double &frequency)
    {
        frequency = std::clamp(frequency,
            ComputationalCell3D::energyBoundaries[0],
            ComputationalCell3D::energyBoundaries[ENERGY_GROUPS_NUM]);
    }
}

    RadiationIMC::RadiationIMC(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, std::shared_ptr<EquationOfState> eos, std::shared_ptr<OpacityCalculator> opacity, RadiationIMCParameters parameters)
    : MonteCarloRadiationPhysics3D(grid, boundary, cells, conserved, eos, opacity), withHydro(parameters.withHydro), diffusionPressureGradient(parameters.diffusionPressureGradient), MMC(parameters.MMC), newPhotonsPerCell(parameters.newPhotonsPerCell), withRandomWalk(parameters.withRandomWalk), rwMinCellOpticalDepth(parameters.rwMinCellOpticalDepth), rwMinParticleOpticalDepth(parameters.rwMinParticleOpticalDepth), noHydroFeedback(parameters.noHydroFeedback), withEgTimeAvg(parameters.withEgTimeAvg)
{
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
    (void) particlesToAdd; // particlesToAdd is not used in this physics
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
    if(this->multigroupOpacity)
    {
        double shiftedFrequency = particle.frequency * dopplerShift;
        ClampFrequencyToBounds(shiftedFrequency);
        absorptionOpacity = this->opacity->CalcAbsorptionOpacity(cell, shiftedFrequency);
    }
    else
    {
        absorptionOpacity = this->planckOpacities[cellIndex];
    }
    double scatteringOpacity = this->opacity->CalcScatteringOpacity(cell);
    double scatteringLength = 1.0 / (scatteringOpacity + (1 - this->factorFleck[cellIndex]) * absorptionOpacity);
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
    dt_t timeScattering = scatteringDistance / abs(particle.velocity);

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
    double tmp2 = absorptionOpacity * this->factorFleck[cellIndex] * units::clight;
    double tmp = -dt * tmp2;
    double expFactor1 = std::expm1(tmp * dopplerShift);
    double expFactor2 = std::expm1(tmp);
    particle.location += particle.velocity * dt;
    if(!this->noHydroFeedback)
    {
        this->conserved[cellIndex].internal_energy += -expFactor2 * particle.weight;
        if(this->withHydro)
        {
            if(not this->diffusionPressureGradient)
            {
                this->conserved[cellIndex].momentum += -expFactor1 * particle.weight * particle.velocity * units::inv_clight2;
            }
        }
    }
    this->Erad_time_avg[cellIndex] += particle.weight * expFactor2 * (-1/tmp2);
    if(this->withEgTimeAvg && this->multigroupOpacity)
    {
        size_t g = this->opacity->findGroup(particle.frequency);
        double groupEradContrib = particle.weight * expFactor2 * (-1/tmp2);
        this->Eg_time_avg[cellIndex][g] += groupEradContrib;
    }
    particle.weight *= 1 + expFactor1;

    if(particle.weight < particle.initialWeight * 1e-3)
    {
        functionality.change = MonteCarloParticleStatus::REMOVE;
        if(!this->noHydroFeedback)
        {
            this->conserved[cellIndex].internal_energy += particle.weight;
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
        double D_lab_to_co = dopplerShift;
        particle.velocity = opacity->getNewScatterVelocity(cell, particle);
        if(this->multigroupOpacity)
        {
            particle.frequency *= dopplerShift; // lab → comoving
            ClampFrequencyToBounds(particle.frequency);
            double random = this->dist(this->re);
            bool isEffectiveScatter = ((1 - this->factorFleck[cellIndex]) * absorptionOpacity) > random * ((1 - this->factorFleck[cellIndex]) * absorptionOpacity + scatteringOpacity);
            if(isEffectiveScatter)
            {
                double reemitRandom = this->dist(this->re);
                particle.frequency = this->multigroupOpacity->GetThermalEnergy(cell, reemitRandom);
            }
        }
        if(this->withHydro && !this->MMC)
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

std::vector<typename RadiationIMC::Particle> RadiationIMC::generateParticles(double fullDt)
{
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
    newParticles.reserve(Ncells * this->newPhotonsPerCell * 10);

    std::vector<size_t> nPhotons(Ncells);
    for(size_t i = 0; i < Ncells; i++)
    {
        size_t proportionalShare = (globalTotalEnergy > 0)
            ? std::llround(energyToCreateVec[i] / globalTotalEnergy * totalParticles)
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
            particle.initialWeight = particle.weight;
            newParticles.push_back(particle);
        }
    }
    return newParticles;
}

std::vector<typename RadiationIMC::Particle> RadiationIMC::preStep(double fullDt)
{
    assert(this->cells.size() >= this->grid.GetPointNo());
    assert(this->conserved.size() >= this->grid.GetPointNo());

    size_t Ncells = this->grid.GetPointNo();
    this->factorFleck.resize(Ncells);
    std::fill(this->factorFleck.begin(), this->factorFleck.end(), 0);
    this->planckOpacities.resize(Ncells);
    std::fill(this->planckOpacities.begin(), this->planckOpacities.end(), 0);
    this->Erad_time_avg.resize(Ncells);
    std::fill(this->Erad_time_avg.begin(), this->Erad_time_avg.end(), 0);
    if(this->withEgTimeAvg && this->multigroupOpacity)
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
        this->planckOpacities[i] = this->opacity->CalcPlanckOpacity(this->cells[i]);
        double gamma = (this->withHydro && !this->MMC)? 1 / std::sqrt(1 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2) : 1;

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

    if(this->withRandomWalk)
    {
        this->precomputeRandomWalkData();
        this->rwStepCount = 0;
    }

    std::vector<Particle> newParticles = this->generateParticles(fullDt);
    std::vector<Particle> newParticles2 = this->boundary->generateNewBoundaryParticles(fullDt); // todo: not here
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

        if(this->multigroupOpacity && (!hasPlanckTable || this->cells[i].temperature != cachedTemperature))
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

        for(size_t j = 0; j < particlesPerCell; j++)
        {
            Particle p = this->generateSingleParticle(i, this->cells[i]);
            if(this->multigroupOpacity)
            {
                double rnd = this->dist(this->re);
                double r = rnd * cumulPlanck.back();
                p.frequency = LinearInterpolation(cumulPlanck, ComputationalCell3D::energyBoundaries, r);
            }
            p.cellID = this->cells[i].ID;
            p.weight = weightPerPhoton;
            p.initialWeight = weightPerPhoton;
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

    for(size_t i = 0; i < Ncells; i++)
    {
        std::vector<size_t> neigh;
        this->grid.GetNeighbors(i, neigh);
        auto faces = this->grid.GetCellFaces(i);
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
    if(parameters.withRandomWalk)
    {
        os << "\t" << "RW min cell optical depth: " << parameters.rwMinCellOpticalDepth << std::endl;
        os << "\t" << "RW min particle optical depth: " << parameters.rwMinParticleOpticalDepth << std::endl;
    }
    os << "\t" << "no hydro feedback: " << parameters.noHydroFeedback << std::endl;
    return os;
}