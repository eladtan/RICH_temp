#ifndef UPDATE_BOX_HPP
#define UPDATE_BOX_HPP 1

#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "3D/tessellation/Voronoi3D.hpp"

/*! \brief Grow the box when fast cells approach a wall (collective).
 *
 * Cells faster than `min_velocity` within five times the largest such cell
 * width of a wall move that wall out by the same distance; random cells with
 * `reference_cell`'s primitives (fraction `volume_fraction` of the new box
 * volume each) fill the new region, and every extensive is recomputed from
 * its primitive on the rebuilt mesh.  Global stepping: after any step.
 * Individual stepping: only at a synchronized individual state, through
 * UpdateBoxSynchronized.
 */
void UpdateBox(Voronoi3D &tess, Simulation &sim, double const min_velocity, double const volume_fraction, ComputationalCell3D const& reference_cell);

/*! \brief UpdateBox at a synchronized state of either mode (collective).
 *
 * Global mode: UpdateBox.  Individual mode: the same decision on the
 * synchronized state, and when it grows,
 * Simulation::GrowDomainAtSynchronizedIndividualState with the points and
 * cells the global resize would add.  Returns whether the box grew; fills
 * `report` for an individual growth.
 */
bool UpdateBoxSynchronized(Voronoi3D &tess, Simulation &sim, double min_velocity,
	double volume_fraction, ComputationalCell3D const& reference_cell,
	Simulation::DomainGrowthReport* report = nullptr);

/*! \brief Whether UpdateBox would grow the box, from the committed state of an
 * individual run between events (collective).
 *
 * The decision of UpdateBox evaluated on each owned cell's committed
 * generator position, velocity, and width from its committed volume
 * (mass / density; a passive cell's flux updates since its last primitive
 * make this approximate).  A trigger for requesting a synchronized event:
 * UpdateBoxSynchronized decides on the synchronized state.
 */
bool BoxGrowthDue(Simulation const& sim, double min_velocity);

#endif // UPDATE_BOX_HPP
