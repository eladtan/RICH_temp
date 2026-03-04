#include "ProgressTracker.hpp"

ProgressTracker::ProgressTracker(void) :
	time(0), cycle(0) {}

void ProgressTracker::updateTime(double dt)
{
	time += dt;
}

void ProgressTracker::updateCycle()
{
	++cycle;
}

double ProgressTracker::getTime(void) const
{
	return time;
}

size_t ProgressTracker::getCycle(void) const
{
	return cycle;
}
