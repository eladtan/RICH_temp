#include "RadiationIMC.hpp"

#define INTERSECTION 0
#define SCATTERING 1
#define TIMELEFT 2

RadiationIMC::RadiationIMC(Tessellation3D &grid, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, const EquationOfState &eos, const RadiationOpacity &opacity, size_t newPhotonsPerCell):
    MonteCarloPhysics(grid), cells(cells), conserved(conserved), eos(eos), opacity(opacity), newPhotonsPerCell(newPhotonsPerCell)
{
    this->dist = std::uniform_real_distribution<double>(std::numeric_limits<double>::epsilon(), 1 - std::numeric_limits<double>::epsilon());
    int rank = 0;
    #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif // RICH_MPI
    this->re = std::mt19937_64(rank);
}

typename RadiationIMC::Functionality RadiationIMC::step(Particle &particle)
{
    Functionality functionality;

    size_t cellIndex = particle.cellIndex;
    ComputationalCell3D &cell = this->cells[cellIndex];

    auto [faceIntersect, timeIntersect, nextCellIndex] = this->getIntersectionDetails(particle);
    assert(timeIntersect >= 0);

    double scatteringLength = 1.0 / (opacity.getScatteringOpacity(cell) + (1 - this->factorFleck[cellIndex]) * this->planckOpacities[cellIndex]);
    distance_t scatteringDistance = (-1 * scatteringLength) * std::log1p(this->dist(this->re)); 
    dt_t timeScattering = scatteringDistance / abs(particle.velocity);

    dt_t timeLeft = particle.timeLeft;
    std::array<std::pair<size_t, dt_t>, 3> times;
    times[0] = {INTERSECTION, timeIntersect};
    times[1] = {SCATTERING, timeScattering};
    times[2] = {TIMELEFT, timeLeft};

    std::pair<size_t, double> min = *std::min_element(times.begin(), times.end(), [](const std::pair<size_t, dt_t> &a, const std::pair<size_t, dt_t> &b) { return a.second < b.second; });
    dt_t dt = min.second;
    particle.timeLeft -= dt;
    double expFactor = std::expm1(-dt * this->planckOpacities[cellIndex] * this->factorFleck[cellIndex] * units::clight);
    particle.location += particle.velocity * dt;
    this->conserved[cellIndex].internal_energy += -expFactor * particle.weight;
    particle.weight *= 1 - expFactor;

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
        particle.velocity = opacity.getNewScatterVelocity(cell, particle);
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

void RadiationIMC::postStep(const std::vector<MCParticle> &particles)
{
    size_t Ncells = this->grid.GetPointNo();
    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        cell.internal_energy = this->conserved[i].internal_energy / this->conserved[i].mass;
        cell.temperature = this->eos.de2T(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
    }
}

std::vector<typename RadiationIMC::MCParticle> RadiationIMC::generateParticles(double fullDt)
{
    std::vector<MCParticle> newParticles;
    size_t Ncells = this->grid.GetPointNo();
    for(size_t i = 0; i < Ncells; i++)
    {
        double energyToCreate = this->grid.GetVolume(i) * units::arad * boost::math::pow<4>(this->cells[i].temperature) * this->planckOpacities[i] * fullDt * units::clight;
        double energyPerPhoton = energyToCreate / this->newPhotonsPerCell;
        for(size_t j = 0; j < this->newPhotonsPerCell; j++)
        {
            MCParticle particle;
            particle.id = std::numeric_limits<size_t>::max(); // TODO: determine ID!
            particle.weight = energyPerPhoton;
            particle.energy = 0; // TODO
            particle.initialWeight = particle.weight;
            particle.location = RandomPointInCell(this->grid, i);
            particle.timeLeft = fullDt * this->dist(this->re);
            particle.velocity = this->opacity.getRandomVelocity(this->cells[i]);
            particle.cellIndex = i;
            newParticles.push_back(particle);
        }
    }
    return newParticles;
}

std::vector<typename RadiationIMC::Particle> RadiationIMC::preStep(double fullDt)
{
    size_t Ncells = this->grid.GetPointNo();
    this->factorFleck = std::vector<double>(Ncells);
    this->planckOpacities = std::vector<double>(Ncells);
    
    for(size_t i = 0; i < Ncells; i++)
    {
        this->planckOpacities[i] = this->opacity.getPlanckOpacity(this->cells[i]);
        
        double cv = this->eos.dT2cv(this->cells[i].density, this->cells[i].temperature, this->cells[i].tracers, this->cells[i].tracerNames);
        this->factorFleck[i] = 1 / (1 + (4 * units::arad * boost::math::pow<3>(this->cells[i].temperature) * this->planckOpacities[i] * units::clight * fullDt) / cv);
    }    

    std::vector<MCParticle> newParticles = this->generateParticles(fullDt);
    return newParticles;
}