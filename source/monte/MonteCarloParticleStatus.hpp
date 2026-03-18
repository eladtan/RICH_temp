#ifndef MONTE_CARLO_PARTICLE_STATUS_HPP
#define MONTE_CARLO_PARTICLE_STATUS_HPP

#include <string>

enum MonteCarloParticleStatus
{
    NO_CELL_MOVE,
    CELL_MOVE,
    DONE,
    REMOVE,
    REFLECT
};

inline std::string MonteCarloParticleStatusToString(int status)
{
    switch(status)
    {
        case NO_CELL_MOVE: return "NO_CELL_MOVE";
        case CELL_MOVE:    return "CELL_MOVE";
        case DONE:         return "DONE";
        case REMOVE:       return "REMOVE";
        case REFLECT:      return "REFLECT";
        default:           return "UNKNOWN(" + std::to_string(status) + ")";
    }
}

#endif // MONTE_CARLO_PARTICLE_STATUS_HPP