#ifndef PROGRESS_TRACKER_HPP
#define PROGRESS_TRACKER_HPP

#include <stddef.h>

//! \brief Tracks the progress of a simulation
class ProgressTracker
{
public:

//! \brief Class constructor
ProgressTracker(void);

/*! \brief Update the progress tracker
    \param dt Time step
    */
void updateTime(double dt);

/*! \brief Updates the cycle number
    */
void updateCycle();

/*! \brief Returns the current time of the simulation
    \return Time of the simulation
    */
double getTime(void) const;

/*! \brief Returns the number of times time advance was called
    \return Cycle number
    */
size_t getCycle(void) const;

//! \brief Simulation time
double time;
//! \brief Tracks the number of times time advance was called
size_t cycle;
};

#endif // PROGRESS_TRACKER_HPP