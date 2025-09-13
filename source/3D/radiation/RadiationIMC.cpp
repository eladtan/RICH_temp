#include "RadiationIMC.hpp"

#include "mpi/Synchronize.h" // todo remove

// #define MONTECARLO_EPS 1e-7

#define INTERSECTION 0
#define SCATTERING 1
#define TIMELEFT 2

RadiationIMC::RadiationIMC(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, const EquationOfState &eos, const RadiationOpacity &opacity, size_t newPhotonsPerCell, bool withHydro)
    : MonteCarloPhysics3D(grid, boundary, cells, conserved, eos, opacity), withHydro(withHydro), newPhotonsPerCell(newPhotonsPerCell)
{}

typename RadiationIMC::Functionality RadiationIMC::step(Particle &particle)
{
    Functionality functionality;

    bool debug = false;
    // debug = debug or (particle.id == 6562628 and particle.rank == 27);

    size_t cellIndex = particle.cellIndex;
    ComputationalCell3D &cell = this->cells[cellIndex];

    auto [faceIntersect, timeIntersect, nextCellIndex] = this->getIntersectionDetails(particle);
    assert(timeIntersect >= 0);

    // todo: change opacity with doppler shift in cast of frequency dependance
    double dopplerShift = (this->withHydro) ? DopplerShift(particle, cell.velocity) : 1.0;

    double scatteringLength = 1.0 / (opacity.getScatteringOpacity(cell) + (1 - this->factorFleck[cellIndex]) * this->planckOpacities[cellIndex]);
    double _log1p = -std::log1p(this->dist(this->re) - 1); 
    distance_t scatteringDistance = scatteringLength * _log1p / dopplerShift; 
    if(scatteringDistance < 0)
    {
        UniversalError eo("Negative scattering distance in RadiationIMC::step");
        eo.addEntry("Cell scattering distance", opacity.getScatteringOpacity(cell));
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
    times[0] = {INTERSECTION, timeIntersect};
    times[1] = {SCATTERING, timeScattering};
    times[2] = {TIMELEFT, timeLeft};

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
    double tmp = -dt * this->planckOpacities[cellIndex] * this->factorFleck[cellIndex] * units::clight;
    double expFactor1 = std::expm1(tmp * dopplerShift);
    double expFactor2 = std::expm1(tmp);
    particle.location += particle.velocity * dt;
    this->conserved[cellIndex].internal_energy += -expFactor2 * particle.weight;
    if(this->withHydro)
    {
        this->conserved[cellIndex].momentum += -expFactor1 * particle.weight * particle.velocity * units::inv_clight2;
    }
    particle.weight *= 1 + expFactor1;

    if(particle.weight < particle.initialWeight * 1e-2)
    {
        functionality.change = MonteCarloParticleStatus::REMOVE;
        this->conserved[cellIndex].internal_energy += particle.weight;
        return functionality;
    }

    if(min.first == INTERSECTION)
    {
        functionality.change = MonteCarloParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = nextCellIndex;
    }
    else if(min.first == SCATTERING)
    {
        Vector3D oldVelocity = particle.velocity;
        particle.velocity = opacity.getNewScatterVelocity(cell, particle);
        if(this->withHydro)
        {
            double weightBefore = particle.weight; // to restore after lorentz transformation
            LorentzTransformation(particle, cell.velocity);
            particle.weight = weightBefore;
            this->conserved[cellIndex].momentum += particle.weight * (oldVelocity - particle.velocity) * units::inv_clight2; // todo: correct?
        }
        // todo: this needs to be changed once we'll have Compton scattering implemented
    }
    else if(min.first == TIMELEFT)
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

void RadiationIMC::postStep(const std::vector<MCParticle> &particles, double fullDt)
{
    size_t Ncells = this->grid.GetPointNo();
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
            cell.velocity = this->conserved[i].momentum / this->conserved[i].mass;
            this->conserved[i].energy = this->conserved[i].internal_energy + 0.5 * ScalarProd(this->conserved[i].momentum, this->conserved[i].momentum) / this->conserved[i].mass; // TODO: material strength
        }
        this->conserved[i].Erad = 0;
        cell.temperature = this->eos.de2T(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
    }
    for(const MCParticle &particle : particles)
    {
        size_t cellIndex = particle.cellIndex;
        assert(cellIndex < Ncells);
        this->conserved[cellIndex].Erad += particle.weight;
    }
    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        cell.Erad = this->conserved[i].Erad / this->conserved[i].mass;
    }
}

std::vector<typename RadiationIMC::MCParticle> RadiationIMC::generateParticles(double fullDt)
{
    std::vector<MCParticle> newParticles;
    size_t Ncells = this->grid.GetPointNo();
    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        double gamma = (this->withHydro)? 1 / std::sqrt(1 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2) : 1;
        double energyToCreate = this->factorFleck[i] * this->grid.GetVolume(i) * units::arad * boost::math::pow<4>(cell.temperature) * this->planckOpacities[i] * fullDt * units::clight;
        double energyPerPhoton = energyToCreate * gamma / this->newPhotonsPerCell;
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
            eo.addEntry("cv", this->eos.dT2cv(cell.density, cell.temperature, cell.tracers, cell.tracerNames));
            throw eo;
        }
        this->conserved[i].energy -= energyToCreate * gamma;
        if(this->withHydro)
        {
            this->conserved[i].momentum -= energyToCreate * cell.velocity * units::inv_clight2 * gamma;
        }
        for(size_t j = 0; j < this->newPhotonsPerCell; j++)
        {
            MCParticle particle = this->generateSingleParticle(i, cell);
            particle.timeLeft = fullDt * this->dist(this->re);
            particle.weight = energyPerPhoton;
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
    this->factorFleck = std::vector<double>(Ncells);
    this->planckOpacities = std::vector<double>(Ncells);
    
    for(size_t i = 0; i < Ncells; i++)
    {
        const ComputationalCell3D &cell = this->cells[i];
        this->planckOpacities[i] = this->opacity.getPlanckOpacity(this->cells[i]);
        double gamma = (this->withHydro)? 1 / std::sqrt(1 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2) : 1;

        double cv = this->eos.dT2cv(this->cells[i].density, this->cells[i].temperature, this->cells[i].tracers, this->cells[i].tracerNames);
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

    std::vector<MCParticle> newParticles = this->generateParticles(fullDt);
    std::vector<MCParticle> newParticles2 = this->boundary->generateNewBoundaryParticles(fullDt); // todo: not here
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

typename RadiationIMC::Particle RadiationIMC::generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const
{
    Particle particle;
    particle.id = std::numeric_limits<size_t>::max();
    particle.energy = 0; // TODO
    particle.location = RandomPointInCell(this->grid, cellIndex);
    // particle.location = particle.location * (1 - MONTECARLO_EPS) + MONTECARLO_EPS * this->grid.GetMeshPoint(cellIndex);
    particle.timeLeft = 0;
    assert(this->grid.IsPointInCell(particle.location, cellIndex));
    assert(not this->grid.IsPointOutsideBox(particle.location));
    particle.velocity = this->opacity.getRandomVelocity(cell);
    if(this->withHydro)
    {
        LorentzTransformation(particle, -1 * cell.velocity);
    }
    particle.cellIndex = cellIndex;
    return particle;
}
