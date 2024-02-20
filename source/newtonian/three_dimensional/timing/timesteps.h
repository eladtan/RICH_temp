#ifndef TIMESTEPS_H
#define TIMESTEPS_H

#include <limits>

typedef double dt_t;
typedef unsigned int time_ratio_t;

#define MAX_TIME (std::numeric_limits<dt_t>::max())

#endif // TIMESTEPS_H