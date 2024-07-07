#ifndef TIMESTEPS_H
#define TIMESTEPS_H

#include <limits>

typedef double dt_t;
typedef unsigned int time_ratio_t;

#define MAX_TIME (std::numeric_limits<dt_t>::max())

#define MIN_TIME (std::numeric_limits<dt_t>::lowest()) // TODO: change to 0?

#endif // TIMESTEPS_H