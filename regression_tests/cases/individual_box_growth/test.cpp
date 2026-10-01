// Box growth during individual (per-cell, power-of-two bin) time stepping.
//
// A Sedov-Taylor blast (E = 1, ambient density 1) in the rigid box [-1,1]^3
// runs with individual time steps from the first event.  After every event
// the driver idiom of runs/BaseTDEComptonIndividual ("Box growth, after the
// step and outside it") runs unchanged:
//
//   synchronized state          -> UpdateBoxSynchronized (the exact decision)
//   pending || BoxGrowthDue(v)  -> RequestSynchronizedIndividualEvent, pending
//
// The blast reaches the walls well before the end time, so the box grows
// several times in individual mode and the run continues after each growth.
// At every growth the test checks, collectively:
//   - counts: cells_after == cells_before + added_cells (report and direct
//     counts), added_cells > 0, IDs globally unique, new IDs fresh, MaxID;
//   - state: IndividualStateSynchronized() afterwards; every scheduler state
//     has begin_tick == last_primitive_tick == currentTick(), its own cell's
//     ID, a positive interval no longer than its bin, and an interval no
//     longer than its hydro CFL limit recomputed on the rebuilt mesh
//     (CourantFriedrichsLewy::CellTimeSteps with the states' point
//     velocities); new and volume-changed cells no coarser than seed_bin;
//   - geometry: the new box contains the old one and is larger, the moved
//     walls moved by one common distance >= the legacy floor, every owned
//     generator (mesh and committed) lies strictly inside the new box, old
//     generators did not move, new generators lie outside the old box;
//   - state preservation: every old cell keeps every serialized primitive
//     field bit for bit (CellDigest; `dt` is scratch and excluded) and its
//     committed point velocity, and no bin gets coarser; every new cell has
//     every primitive field of the reference (ambient) state and no point
//     velocity; every extensive field equals PrimitiveToConserved of its
//     primitive on the rebuilt mesh; without gravity the acceleration cache
//     is invalidated;
//   - bookkeeping: report totals equal the gathered per-cell sums, the new
//     mesh volumes sum to the new box volume, and
//       mass_after - mass_before - inserted_mass
//         == sum over old cells of density * (V_new - V_committed)
//     to 1e-10 of mass_before.  The legacy semantics recompute every
//     extensive from its primitive on the rebuilt mesh, so old cells next to
//     a moved wall change volume and mass: mass_after - mass_before -
//     inserted_mass alone is not small, and is reported (naive residual)
//     together with the box-volume form of the same balance.  The energy
//     residual of the same decomposition is reported and bounded loosely.
// At every synchronized state that does not grow the box the test checks
// that nothing changed: box, MaxID, time and tick, mesh and committed
// generators, and every field of every cell, extensive and scheduler state
// (cached accelerations, pending kicks, point velocities, pending neighbour
// bins): the zero-growth path.  Every growth must be followed by an event
// (the loop runs past the end time for it) with finite totals.  Over the run it checks mass conservation between
// growths, and at the end that the end time was reached with finite totals.
//
// Rank 0 writes individual_box_growth_metrics.txt ("key value" lines, last
// "pass 0|1") and one RICH_TEST_BOX_GROWTH line per growth on stdout.  The
// process returns nonzero on any violation.
//
// Environment overrides (defaults in parentheses; serial / MPI):
//   RICH_TEST_POINT_COUNT      initial cells (2e4 / 2e5)
//   RICH_TEST_FINAL_TIME       end time (0.3 / 0.35)
//   RICH_TEST_MIN_VELOCITY     growth speed threshold (0.1)
//   RICH_TEST_VOLUME_FRACTION  new-cell volume fraction (1e-4 / 1e-5)
//   RICH_TEST_MAX_EVENTS       event budget (20000)
//   RICH_TEST_MIN_GROWTHS      growths required (2)
//   RICH_TEST_INITIAL_BIN, RICH_TEST_MAXIMUM_BIN  scheduler bins (30, 40)
//   RICH_TEST_FORCE_SYNCHRONIZED=1  one shared bin for every cell
//     (force_synchronized, full meshes): every event is synchronized, the
//     request path is not required, and after each growth every cell must
//     share one bin and one interval.
//   RICH_TEST_BOX_GROWTH_GRAVITY=1  self-gravity (ConservativeForce3D over
//     FastMultipoleAcceleration3D, G = RICH_TEST_GRAVITY_G, 0.03; FMM order,
//     theta, leaf from RICH_TEST_FMM_ORDER / _THETA / _LEAF, 2 / 1.0 / 64):
//     after each growth the cached accelerations of RICH_TEST_GRAVITY_TARGETS
//     (8000) sampled cells must match a direct sum to
//     RICH_TEST_GRAVITY_TOLERANCE (0.1, max error over the force scale
//     G sum m/r^2), the pre-growth (stale) cache must miss it by more than
//     that tolerance, every half kick must be pending, and every sampled
//     interval must fit the unrelaxed source limit of the direct
//     acceleration.  The event after each growth must kick every first half
//     from the cache on every rank (no geometry-path first half) with the
//     refreshed values bit for bit.  The energy drift is not bounded
//     (potential energy is not in the totals).
//   RICH_TEST_GRAVITY_STALE_CACHE=1  negative control (with gravity): the
//     pre-growth cache is put back after each growth; the gravity check must
//     then fail at every growth (stale_control_growths_caught == growths),
//     and the run reports pass 0.
// After each growth every interval and bin must also fit the per-cell limit
// the test computes itself on the rebuilt mesh: wave-speed CFL, the source
// limit source_cfl * sqrt(width / |a|), and the mesh-drift guard
// (RICH_INDIVIDUAL_MESH_DRIFT_FRACTION, 0.25); the drift term must tighten
// some cell at some growth.

#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/3D/GeometryCommon/UpdateBox.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/newtonian/three_dimensional/timing/advance/TimeStepUtils.hpp"

namespace
{
int g_rank = 0;
int g_size = 1;

double EnvironmentDouble(char const* name, double fallback)
{
	char const* value = std::getenv(name);
	if(value == nullptr || value[0] == '\0')
		return fallback;
	return std::stod(value);
}

size_t EnvironmentSize(char const* name, size_t fallback)
{
	char const* value = std::getenv(name);
	if(value == nullptr || value[0] == '\0')
		return fallback;
	return static_cast<size_t>(std::stod(value));
}

bool AllRanks(bool const local)
{
	int value = local ? 1 : 0;
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
	return value != 0;
}

double SumAll(double value)
{
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	return value;
}

double MinAll(double value)
{
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
	return value;
}

unsigned long long SumAll(unsigned long long value)
{
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
		MPI_COMM_WORLD);
#endif
	return value;
}

bool BroadcastFromRoot(bool value)
{
	int flag = value ? 1 : 0;
#ifdef RICH_MPI
	MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
	return flag != 0;
}

double RelativeDifference(double value, double reference)
{
	return std::abs(value - reference) / std::max(std::abs(reference), 1e-300);
}

bool SameBits(double a, double b)
{
	return std::memcmp(&a, &b, sizeof(double)) == 0;
}

bool SameVector(Vector3D const& a, Vector3D const& b)
{
	return SameBits(a.x, b.x) && SameBits(a.y, b.y) && SameBits(a.z, b.z);
}

// Every serialized primitive field (ComputationalCell3D::dump) bit for bit;
// `dt` is per-step scratch, neither serialized nor compared.
bool SameCell(ComputationalCell3D const& a, ComputationalCell3D const& b, bool compare_id)
{
	bool same = (!compare_id || a.ID == b.ID) && SameBits(a.density, b.density) &&
		SameBits(a.pressure, b.pressure) && SameVector(a.velocity, b.velocity) &&
		SameBits(a.internal_energy, b.internal_energy) &&
		SameBits(a.temperature, b.temperature) && SameBits(a.Erad, b.Erad) &&
		SameBits(a.Erad_dt, b.Erad_dt) && SameBits(a.Erad_dt_dt, b.Erad_dt_dt) &&
		SameBits(a.cs, b.cs) && a.stickers == b.stickers && a.Eg.size() == b.Eg.size();
	for(size_t k = 0; same && k < a.tracers.size(); ++k)
		same = SameBits(a.tracers[k], b.tracers[k]);
	for(size_t k = 0; same && k < a.Eg.size(); ++k)
		same = SameBits(a.Eg[k], b.Eg[k]);
	return same;
}

// Every Conserved3D field bit for bit.
bool SameExtensive(Conserved3D const& a, Conserved3D const& b)
{
	bool same = SameBits(a.mass, b.mass) && SameVector(a.momentum, b.momentum) &&
		SameBits(a.energy, b.energy) && SameBits(a.internal_energy, b.internal_energy) &&
		SameBits(a.Erad, b.Erad) && SameBits(a.Erad_dt, b.Erad_dt) &&
		SameBits(a.Erad_dt_dt, b.Erad_dt_dt) && a.Eg.size() == b.Eg.size();
	for(size_t k = 0; same && k < a.tracers.size(); ++k)
		same = SameBits(a.tracers[k], b.tracers[k]);
	for(size_t k = 0; same && k < a.Eg.size(); ++k)
		same = SameBits(a.Eg[k], b.Eg[k]);
	return same;
}

// Every CellTimeState field.
bool SameState(CellTimeState const& a, CellTimeState const& b)
{
	return a.cell_id == b.cell_id && a.begin_tick == b.begin_tick &&
		a.end_tick == b.end_tick && a.last_primitive_tick == b.last_primitive_tick &&
		a.time_bin == b.time_bin && a.pending_neighbor_bin == b.pending_neighbor_bin &&
		SameVector(a.point_velocity, b.point_velocity) &&
		SameVector(a.cached_acceleration, b.cached_acceleration) &&
		a.gravity_half_kick_pending == b.gravity_half_kick_pending;
}

// FNV-1a over every field SameCell compares except the ID, so cells can be
// compared by stable ID after migration without gathering every field.
std::uint64_t CellDigest(ComputationalCell3D const& c)
{
	std::uint64_t h = 1469598103934665603ULL;
	auto mix = [&h](void const* data, size_t bytes)
	{
		unsigned char const* p = static_cast<unsigned char const*>(data);
		for(size_t i = 0; i < bytes; ++i)
		{
			h ^= p[i];
			h *= 1099511628211ULL;
		}
	};
	double const scalars[11] = {c.density, c.pressure, c.velocity.x, c.velocity.y,
		c.velocity.z, c.internal_energy, c.temperature, c.Erad, c.Erad_dt, c.Erad_dt_dt,
		c.cs};
	mix(scalars, sizeof scalars);
	for(double const t : c.tracers)
		mix(&t, sizeof t);
	for(bool const st : c.stickers)
	{
		unsigned char const b = st ? 1 : 0;
		mix(&b, 1);
	}
	std::uint64_t const groups = c.Eg.size();
	mix(&groups, sizeof groups);
	for(double const e : c.Eg)
		mix(&e, sizeof e);
	return h;
}

// One cell as the growth checks see it; rank 0 gathers these by stable ID.
struct CellRecord
{
	unsigned long long id = 0;
	double mass = 0;
	double energy = 0;
	double density = 0;
	double internal_energy = 0;
	double vx = 0;
	double vy = 0;
	double vz = 0;
	double volume = 0;
	double px = 0;
	double py = 0;
	double pz = 0;
	unsigned long long bin = 0;
	unsigned long long interval = 0;
	std::uint64_t digest = 0;       // CellDigest: every primitive field but the ID
	double wx = 0;                  // scheduler point velocity
	double wy = 0;
	double wz = 0;
	double ax = 0;                  // scheduler cached acceleration
	double ay = 0;
	double az = 0;
	unsigned long long pending = 0; // gravity_half_kick_pending
};

template <typename T>
std::vector<T> GatherToRoot(std::vector<T> const& local)
{
#ifdef RICH_MPI
	unsigned long long const local_bytes =
		static_cast<unsigned long long>(local.size()) * sizeof(T);
	if(local_bytes > static_cast<unsigned long long>(INT_MAX))
		throw std::runtime_error("GatherToRoot: local block too large");
	int const bytes = static_cast<int>(local_bytes);
	std::vector<int> counts(static_cast<size_t>(g_size), 0);
	MPI_Gather(&bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
	std::vector<int> displacements(static_cast<size_t>(g_size), 0);
	std::vector<T> result;
	int overflow = 0;
	if(g_rank == 0)
	{
		unsigned long long total = 0;
		for(int r = 0; r < g_size; ++r)
		{
			displacements[static_cast<size_t>(r)] = static_cast<int>(total);
			total += static_cast<unsigned long long>(counts[static_cast<size_t>(r)]);
			if(total > static_cast<unsigned long long>(INT_MAX))
				overflow = 1;
		}
		if(overflow == 0)
			result.resize(static_cast<size_t>(total / sizeof(T)));
	}
	MPI_Bcast(&overflow, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if(overflow != 0)
		throw std::runtime_error("GatherToRoot: gathered block too large");
	MPI_Gatherv(local.data(), bytes, MPI_BYTE, result.data(), counts.data(),
		displacements.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
	return result;
#else
	return local;
#endif
}

struct Totals
{
	double mass = 0;
	double energy = 0;
	double momentum_x = 0;
	double momentum_y = 0;
	double momentum_z = 0;
	double momentum_scale = 0;
	bool finite = true;
	bool positive = true;
};

// Totals of the committed extensives (collective).
Totals ComputeTotals(Simulation const& simulation)
{
	std::vector<Conserved3D> const& extensives = simulation.getExtensives();
	std::vector<ComputationalCell3D> const& cells = simulation.getCells();
	size_t const n = std::min(extensives.size(), cells.size());
	std::array<long double, 6> sums = {0, 0, 0, 0, 0, 0};
	bool finite = extensives.size() == cells.size();
	bool positive = true;
	for(size_t i = 0; i < n; ++i)
	{
		Conserved3D const& value = extensives[i];
		sums[0] += value.mass;
		sums[1] += value.energy;
		sums[2] += value.momentum.x;
		sums[3] += value.momentum.y;
		sums[4] += value.momentum.z;
		sums[5] += abs(value.momentum);
		finite = finite && std::isfinite(value.mass) && std::isfinite(value.energy) &&
			std::isfinite(value.momentum.x) && std::isfinite(value.momentum.y) &&
			std::isfinite(value.momentum.z) && std::isfinite(cells[i].density) &&
			std::isfinite(cells[i].internal_energy) && std::isfinite(cells[i].pressure);
		positive = positive && value.mass > 0 && cells[i].density > 0 &&
			cells[i].internal_energy > 0 && cells[i].pressure > 0;
	}
	std::array<double, 6> reduced;
	for(size_t k = 0; k < 6; ++k)
		reduced[k] = static_cast<double>(sums[k]);
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, reduced.data(), 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
	Totals totals;
	totals.mass = reduced[0];
	totals.energy = reduced[1];
	totals.momentum_x = reduced[2];
	totals.momentum_y = reduced[3];
	totals.momentum_z = reduced[4];
	totals.momentum_scale = reduced[5];
	totals.finite = AllRanks(finite && std::isfinite(totals.mass) &&
		std::isfinite(totals.energy));
	totals.positive = AllRanks(positive);
	return totals;
}

// The owned state at a synchronized individual state, in canonical order.
struct Snapshot
{
	std::pair<Vector3D, Vector3D> box;
	size_t max_id = 0;
	std::vector<ComputationalCell3D> cells;
	std::vector<Conserved3D> extensives;
	std::vector<CellTimeState> states;
	std::vector<Vector3D> committed_points;
	std::vector<Vector3D> mesh_points;
	std::uint64_t current_tick = 0;
	double time = 0;
};

Snapshot TakeSnapshot(Simulation const& simulation, Voronoi3D const& tess)
{
	Snapshot snapshot;
	snapshot.box = tess.GetBoxCoordinates();
	snapshot.max_id = simulation.GetMaxID();
	snapshot.cells = simulation.getCells();
	snapshot.extensives = simulation.getExtensives();
	IndividualTimeStepScheduler const* scheduler =
		simulation.GetIndividualTimeStepScheduler();
	snapshot.states = scheduler->states();
	snapshot.current_tick = scheduler->currentTick();
	snapshot.committed_points = simulation.CommittedGeneratorPoints();
	snapshot.mesh_points.resize(tess.GetPointNo());
	for(size_t i = 0; i < snapshot.mesh_points.size(); ++i)
		snapshot.mesh_points[i] = tess.GetMeshPoint(i);
	snapshot.time = simulation.GetTime();
	return snapshot;
}

std::vector<CellRecord> BeforeRecords(Snapshot const& snapshot)
{
	std::vector<CellRecord> records(snapshot.cells.size());
	for(size_t i = 0; i < snapshot.cells.size(); ++i)
	{
		ComputationalCell3D const& cell = snapshot.cells[i];
		Conserved3D const& extensive = snapshot.extensives[i];
		CellRecord& record = records[i];
		record.id = static_cast<unsigned long long>(cell.ID);
		record.mass = extensive.mass;
		record.energy = extensive.energy;
		record.density = cell.density;
		record.internal_energy = cell.internal_energy;
		record.vx = cell.velocity.x;
		record.vy = cell.velocity.y;
		record.vz = cell.velocity.z;
		// The committed volume at a synchronized state, as the growth uses it.
		record.volume = extensive.mass / cell.density;
		record.px = snapshot.committed_points[i].x;
		record.py = snapshot.committed_points[i].y;
		record.pz = snapshot.committed_points[i].z;
		CellTimeState const& state = snapshot.states[i];
		record.bin = state.time_bin;
		record.interval = state.end_tick - state.begin_tick;
		record.digest = CellDigest(cell);
		record.wx = state.point_velocity.x;
		record.wy = state.point_velocity.y;
		record.wz = state.point_velocity.z;
		record.ax = state.cached_acceleration.x;
		record.ay = state.cached_acceleration.y;
		record.az = state.cached_acceleration.z;
		record.pending = state.gravity_half_kick_pending ? 1 : 0;
	}
	return records;
}

std::vector<CellRecord> AfterRecords(Simulation const& simulation, Voronoi3D const& tess)
{
	std::vector<ComputationalCell3D> const& cells = simulation.getCells();
	std::vector<Conserved3D> const& extensives = simulation.getExtensives();
	std::vector<CellTimeState> const& states =
		simulation.GetIndividualTimeStepScheduler()->states();
	size_t const n = std::min(std::min(cells.size(), extensives.size()),
		std::min(states.size(), tess.GetPointNo()));
	std::vector<CellRecord> records(n);
	for(size_t i = 0; i < n; ++i)
	{
		CellRecord& record = records[i];
		record.id = static_cast<unsigned long long>(cells[i].ID);
		record.mass = extensives[i].mass;
		record.energy = extensives[i].energy;
		record.density = cells[i].density;
		record.internal_energy = cells[i].internal_energy;
		record.vx = cells[i].velocity.x;
		record.vy = cells[i].velocity.y;
		record.vz = cells[i].velocity.z;
		record.volume = tess.GetVolume(i);
		Vector3D const& point = tess.GetMeshPoint(i);
		record.px = point.x;
		record.py = point.y;
		record.pz = point.z;
		CellTimeState const& state = states[i];
		record.bin = state.time_bin;
		record.interval = state.end_tick - state.begin_tick;
		record.digest = CellDigest(cells[i]);
		record.wx = state.point_velocity.x;
		record.wy = state.point_velocity.y;
		record.wz = state.point_velocity.z;
		record.ax = state.cached_acceleration.x;
		record.ay = state.cached_acceleration.y;
		record.az = state.cached_acceleration.z;
		record.pending = state.gravity_half_kick_pending ? 1 : 0;
	}
	return records;
}

double BoxVolume(std::pair<Vector3D, Vector3D> const& box)
{
	return (box.second.x - box.first.x) * (box.second.y - box.first.y) *
		(box.second.z - box.first.z);
}

bool StrictlyInside(Vector3D const& p, std::pair<Vector3D, Vector3D> const& box)
{
	return p.x > box.first.x && p.y > box.first.y && p.z > box.first.z &&
		p.x < box.second.x && p.y < box.second.y && p.z < box.second.z;
}

bool OutsideBox(double x, double y, double z, std::pair<Vector3D, Vector3D> const& box)
{
	return x < box.first.x || y < box.first.y || z < box.first.z ||
		x > box.second.x || y > box.second.y || z > box.second.z;
}

// Zero-growth path: after UpdateBoxSynchronized returned false nothing may
// have changed (collective): box, MaxID, time and tick, mesh and committed
// generator positions, and every field of every cell, extensive and
// scheduler state (cached accelerations, pending kicks, point velocities and
// pending neighbour bins included).
bool NothingChanged(Snapshot const& before, Simulation const& simulation,
	Voronoi3D const& tess)
{
	std::pair<Vector3D, Vector3D> const box = tess.GetBoxCoordinates();
	bool same = SameVector(box.first, before.box.first) &&
		SameVector(box.second, before.box.second) &&
		simulation.GetMaxID() == before.max_id &&
		tess.GetPointNo() == before.mesh_points.size() &&
		SameBits(simulation.GetTime(), before.time);
	std::vector<ComputationalCell3D> const& cells = simulation.getCells();
	std::vector<Conserved3D> const& extensives = simulation.getExtensives();
	IndividualTimeStepScheduler const* scheduler =
		simulation.GetIndividualTimeStepScheduler();
	std::vector<CellTimeState> const& states = scheduler->states();
	std::vector<Vector3D> const committed = simulation.CommittedGeneratorPoints();
	same = same && cells.size() == before.cells.size() &&
		extensives.size() == before.extensives.size() &&
		states.size() == before.states.size() &&
		committed.size() == before.committed_points.size() &&
		scheduler->currentTick() == before.current_tick;
	for(size_t i = 0; same && i < cells.size(); ++i)
		same = SameCell(cells[i], before.cells[i], true) &&
			SameExtensive(extensives[i], before.extensives[i]) &&
			SameState(states[i], before.states[i]) &&
			SameVector(committed[i], before.committed_points[i]);
	for(size_t i = 0; same && i < before.mesh_points.size(); ++i)
		same = SameVector(tess.GetMeshPoint(i), before.mesh_points[i]);
	return AllRanks(same);
}

struct GrowthResult
{
	bool ok = true;
	std::vector<std::string> failures;
	double mass_residual = 0;          // decomposed: fails above 1e-10
	double mass_residual_naive = 0;    // (after - before - inserted) / before
	double mass_residual_box = 0;      // (after - before - rho_ref dV_box) / before
	double energy_residual = 0;        // decomposed: loose bound
	double energy_residual_naive = 0;
	double volume_closure_after = 0;
	double volume_closure_before = 0;
	double max_changed_density_deviation = 0;
	double cfl_margin = std::numeric_limits<double>::infinity();
	double expected_margin = std::numeric_limits<double>::infinity();
	double smallest_cfl_limit = std::numeric_limits<double>::infinity();
	unsigned long long drift_tightened = 0;
	unsigned long long source_tightened = 0;
	// Gravity variant: cached accelerations against a direct sum.
	unsigned long long gravity_targets = 0;
	double gravity_max_scaled_error = 0;
	double gravity_rms_scaled_error = 0;
	double gravity_max_relative_error = 0;
	double gravity_rms_relative_error = 0;   // rms |error| / rms |a_direct|
	double gravity_source_margin = std::numeric_limits<double>::infinity();
	// Sensitivity: the same comparison for the pre-growth cache (new cells 0)
	// and the size of the new cells' pull on old targets.
	double gravity_stale_max_scaled_error = 0;
	double gravity_new_mass_max_scaled_effect = 0;
	unsigned shared_bin_min = 0;
	unsigned shared_bin_max = 0;
	unsigned long long changed_old_cells = 0;
	unsigned walls_moved = 0;
	double wall_shift = 0;
};

struct LimitSettings
{
	bool gravity = false;
	double G = 0;
	double source_cfl = 1;
	double drift_fraction = 0.25;
	size_t gravity_targets = 8000;
	double gravity_tolerance = 0.1;
};

void Require(GrowthResult& result, bool condition, std::string const& what)
{
	if(!condition)
	{
		result.ok = false;
		result.failures.push_back(what);
	}
}

GrowthResult VerifyGrowth(Snapshot const& before, Simulation::DomainGrowthReport const& growth,
	Simulation& simulation, Voronoi3D& tess, CourantFriedrichsLewy const& cfl,
	EquationOfState const& eos, ComputationalCell3D const& reference_cell,
	LimitSettings const& settings)
{
	GrowthResult result;
	IndividualTimeStepScheduler const* scheduler =
		simulation.GetIndividualTimeStepScheduler();
	std::pair<Vector3D, Vector3D> const old_box = before.box;
	std::pair<Vector3D, Vector3D> const new_box = tess.GetBoxCoordinates();
	double const old_volume = BoxVolume(old_box);
	double const new_volume = BoxVolume(new_box);

	// Box: contains the old one, larger, moved walls moved by one distance.
	std::array<double, 6> const shifts = {
		old_box.first.x - new_box.first.x, old_box.first.y - new_box.first.y,
		old_box.first.z - new_box.first.z, new_box.second.x - old_box.second.x,
		new_box.second.y - old_box.second.y, new_box.second.z - old_box.second.z};
	bool contains = true;
	double largest_shift = 0;
	for(double const shift : shifts)
	{
		contains = contains && shift >= 0;
		largest_shift = std::max(largest_shift, shift);
	}
	bool common_shift = largest_shift > 0;
	for(double const shift : shifts)
		if(shift > 0)
		{
			++result.walls_moved;
			common_shift = common_shift &&
				std::abs(shift - largest_shift) <= 1e-9 * largest_shift;
		}
	result.wall_shift = largest_shift;
	Require(result, contains, "new box does not contain the old box");
	Require(result, new_volume > old_volume, "new box is not larger");
	Require(result, common_shift, "moved walls moved by different distances");
	Require(result, largest_shift >= 5 * 0.03 * std::pow(old_volume, 0.333333333) * (1 - 1e-9),
		"wall shift below the legacy floor 5 * 0.03 * V^(1/3)");

	// State after growth.
	Require(result, simulation.IndividualStateSynchronized(),
		"state not synchronized after growth");
	std::vector<ComputationalCell3D> const& cells = simulation.getCells();
	std::vector<Conserved3D> const& extensives = simulation.getExtensives();
	std::vector<CellTimeState> const& states = scheduler->states();
	size_t const n = cells.size();
	bool const aligned_sizes = extensives.size() == n && states.size() == n &&
		tess.GetPointNo() == n;
	Require(result, AllRanks(aligned_sizes),
		"cells, extensives, states and mesh differ in size after growth");
	unsigned long long const count_before =
		SumAll(static_cast<unsigned long long>(before.cells.size()));
	unsigned long long const count_after = SumAll(static_cast<unsigned long long>(n));
	Require(result, growth.added_cells > 0, "report added_cells is zero");
	Require(result, growth.cells_after == growth.cells_before + growth.added_cells,
		"report cells_after != cells_before + added_cells");
	Require(result, growth.cells_before == count_before, "report cells_before != counted cells");
	Require(result, growth.cells_after == count_after, "report cells_after != counted cells");

	// Scheduler states and the CFL cap on the rebuilt mesh.
	std::uint64_t const tick = scheduler->currentTick();
	double const quantum = scheduler->timeQuantum();
	bool states_ok = aligned_sizes;
	for(size_t i = 0; states_ok && i < n; ++i)
	{
		CellTimeState const& state = states[i];
		states_ok = state.cell_id == cells[i].ID && state.begin_tick == tick &&
			state.last_primitive_tick == tick && state.end_tick > state.begin_tick &&
			state.end_tick - state.begin_tick <= scheduler->binTicks(state.time_bin) &&
			state.time_bin <= scheduler->options().maximum_bin;
	}
	Require(result, AllRanks(states_ok),
		"a scheduler state is not aligned at the current tick after growth");
	{
		// Every extensive field is PrimitiveToConserved of its (unchanged)
		// primitive on the rebuilt mesh; without a cache-keeping source the
		// acceleration cache is invalidated (zero, no pending kick).
		bool extensives_ok = aligned_sizes;
		bool cache_invalidated = aligned_sizes;
		for(size_t i = 0; extensives_ok && i < n; ++i)
		{
			Conserved3D expected;
			PrimitiveToConserved(cells[i], tess.GetVolume(i), expected);
			extensives_ok = SameExtensive(expected, extensives[i]);
		}
		for(size_t i = 0; cache_invalidated && i < n; ++i)
			cache_invalidated = !states[i].gravity_half_kick_pending &&
				states[i].cached_acceleration.x == 0 && states[i].cached_acceleration.y == 0 &&
				states[i].cached_acceleration.z == 0;
		Require(result, AllRanks(extensives_ok),
			"an extensive differs from PrimitiveToConserved of its primitive on the new mesh");
		if(!settings.gravity)
			Require(result, AllRanks(cache_invalidated),
				"acceleration cache not invalidated after growth without a cache-keeping source");
	}
	{
		// The per-cell limit the growth must honour, computed here without the
		// library's synchronized-limit functions: the wave-speed CFL
		// (CourantFriedrichsLewy::CellTimeSteps, face velocities from the
		// states' point velocities, ghost-exchanged under MPI); the source
		// limit source_cfl * sqrt(width / |a|) from each state's cached
		// acceleration (gravity variant; the cache is checked against a direct
		// sum below); and the mesh-drift guard: over non-boundary faces a
		// generator may close at most a fraction f of the distance to its
		// neighbour, closing = -(w_i - w_j).(x_i - x_j) / |x_i - x_j|, the
		// bound f d / closing floored at 1/16 of the cell's own limit.
		std::vector<Vector3D> mesh_velocities(n);
		for(size_t i = 0; i < n && aligned_sizes; ++i)
			mesh_velocities[i] = states[i].point_velocity;
#ifdef RICH_MPI
		MPI_exchange_data(tess, mesh_velocities, true);
#endif
		std::vector<Vector3D> face_velocities;
		CalcFaceVelocities(tess, mesh_velocities, face_velocities);
		std::vector<double> limits;
		cfl.CellTimeSteps(tess, cells, eos, face_velocities, limits, false);
		double margin = std::numeric_limits<double>::infinity();
		double expected_margin = std::numeric_limits<double>::infinity();
		double smallest = std::numeric_limits<double>::infinity();
		unsigned long long drift_tightened = 0;
		unsigned long long source_tightened = 0;
		unsigned bin_min = scheduler->options().maximum_bin;
		unsigned bin_max = 0;
		for(size_t i = 0; i < n && i < limits.size() && aligned_sizes; ++i)
		{
			CellTimeState const& state = states[i];
			double const interval = static_cast<double>(state.end_tick - state.begin_tick) * quantum;
			double const bin_length = static_cast<double>(scheduler->binTicks(state.time_bin)) * quantum;
			double own = limits[i];
			if(settings.gravity)
			{
				double const a = abs(state.cached_acceleration);
				if(a > 0)
				{
					double const source = settings.source_cfl * std::sqrt(tess.GetWidth(i) / a);
					if(source < own)
						++source_tightened;
					own = std::min(own, source);
				}
			}
			double expected = own;
			if(settings.drift_fraction > 0)
			{
				double drift = std::numeric_limits<double>::infinity();
				Vector3D const& point = tess.GetMeshPoint(i);
				for(size_t const face : tess.GetCellFaces(i))
				{
					if(tess.BoundaryFace(face))
						continue;
					std::pair<size_t, size_t> const neighbors = tess.GetFaceNeighbors(face);
					size_t const other = neighbors.first == i ? neighbors.second : neighbors.first;
					if(other >= mesh_velocities.size())
						continue;
					Vector3D const separation = point - tess.GetMeshPoint(other);
					double const distance = abs(separation);
					if(!(distance > 0))
						continue;
					double const closing = -ScalarProd(mesh_velocities[i] - mesh_velocities[other],
						separation) / distance;
					if(closing > 0)
						drift = std::min(drift, settings.drift_fraction * distance / closing);
				}
				if(std::isfinite(own) && own > 0)
					drift = std::max(drift, 0.0625 * own);
				if(drift < own)
					++drift_tightened;
				expected = std::min(own, drift);
			}
			margin = std::min(margin, limits[i] / interval);
			margin = std::min(margin, limits[i] / bin_length);
			expected_margin = std::min(expected_margin, expected / interval);
			expected_margin = std::min(expected_margin, expected / bin_length);
			smallest = std::min(smallest, expected);
			bin_min = std::min<unsigned>(bin_min, state.time_bin);
			bin_max = std::max<unsigned>(bin_max, state.time_bin);
		}
		result.cfl_margin = MinAll(margin);
		result.expected_margin = MinAll(expected_margin);
		result.smallest_cfl_limit = MinAll(smallest);
		result.drift_tightened = SumAll(drift_tightened);
		result.source_tightened = SumAll(source_tightened);
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &bin_min, 1, MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &bin_max, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
#endif
		result.shared_bin_min = bin_min;
		result.shared_bin_max = bin_max;
		Require(result, result.cfl_margin >= 1 - 1e-12,
			"an interval after growth exceeds its hydro CFL limit on the rebuilt mesh");
		Require(result, result.expected_margin >= 1 - 1e-12,
			"an interval after growth exceeds the expected CFL/source/drift limit on the rebuilt mesh");
		// The report's smallest limit is the smallest expected limit (same
		// rule), and never below the quantum (else growth throws).
		Require(result, std::isfinite(growth.smallest_limit) &&
			growth.smallest_limit >= quantum &&
			RelativeDifference(growth.smallest_limit, result.smallest_cfl_limit) <= 1e-9,
			"report smallest_limit differs from the smallest expected limit");
		if(scheduler->options().force_synchronized)
		{
			bool one_interval = true;
			for(size_t i = 1; one_interval && i < n && aligned_sizes; ++i)
				one_interval = states[i].end_tick == states[0].end_tick;
			std::uint64_t end_min = n > 0 ? states[0].end_tick :
				std::numeric_limits<std::uint64_t>::max();
			std::uint64_t end_max = n > 0 ? states[0].end_tick : 0;
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, &end_min, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &end_max, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
#endif
			Require(result, bin_min == bin_max && AllRanks(one_interval) && end_min == end_max,
				"force_synchronized: cells do not share one bin and interval after growth");
		}
	}

	// Gravity variant: the refreshed cache against a direct sum
	// G sum_j m_j (x_j - x_i) / |x_j - x_i|^3 over every cell's centroid and
	// mass on the rebuilt mesh (the sources the conservative force uses), on a
	// sample of stable IDs: the maximum error over the force scale
	// G sum_j m_j / r^2 must stay within the tolerance; every half kick is
	// pending; and every sampled interval and bin stays within the UNRELAXED
	// source limit source_cfl * sqrt(width / |a_direct|).  Sensitivity: the
	// same error for the pre-growth cache by ID (new cells zero), which must
	// exceed the tolerance, and the new cells' share of the direct pull.
	if(settings.gravity)
	{
		bool pending = aligned_sizes;
		for(size_t i = 0; pending && i < n; ++i)
			pending = states[i].gravity_half_kick_pending &&
				std::isfinite(states[i].cached_acceleration.x) &&
				std::isfinite(states[i].cached_acceleration.y) &&
				std::isfinite(states[i].cached_acceleration.z);
		Require(result, growth.accelerations_refreshed,
			"gravity: report says the acceleration cache was not refreshed");
		Require(result, AllRanks(pending),
			"gravity: a state has no pending half kick or a non-finite cached acceleration");
		unsigned long long const total = count_after;
		unsigned long long const stride = std::max<unsigned long long>(1,
			(total + settings.gravity_targets - 1) / settings.gravity_targets);
		// source: x, y, z, mass, id; target: x, y, z, cached a, width,
		// interval, bin length, id.
		std::vector<std::array<double, 5> > sources(n);
		std::vector<std::array<double, 10> > targets;
		for(size_t i = 0; i < n && aligned_sizes; ++i)
		{
			Vector3D const cm = tess.GetCellCM(i);
			sources[i] = {{cm.x, cm.y, cm.z, extensives[i].mass,
				static_cast<double>(cells[i].ID)}};
			if(static_cast<unsigned long long>(cells[i].ID) % stride != 0)
				continue;
			CellTimeState const& state = states[i];
			targets.push_back({{cm.x, cm.y, cm.z, state.cached_acceleration.x,
				state.cached_acceleration.y, state.cached_acceleration.z, tess.GetWidth(i),
				static_cast<double>(state.end_tick - state.begin_tick) * quantum,
				static_cast<double>(scheduler->binTicks(state.time_bin)) * quantum,
				static_cast<double>(cells[i].ID)}});
		}
		// The pre-growth cache by stable ID.
		std::vector<std::array<double, 4> > old_cache(before.cells.size());
		for(size_t i = 0; i < before.cells.size() && i < before.states.size(); ++i)
			old_cache[i] = {{static_cast<double>(before.cells[i].ID),
				before.states[i].cached_acceleration.x, before.states[i].cached_acceleration.y,
				before.states[i].cached_acceleration.z}};
		double max_old_id_local = 0;
		for(ComputationalCell3D const& cell : before.cells)
			max_old_id_local = std::max(max_old_id_local, static_cast<double>(cell.ID));
		double max_old_id = max_old_id_local;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &max_old_id, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
		std::vector<std::array<double, 5> > const all_sources = GatherToRoot(sources);
		std::vector<std::array<double, 10> > const all_targets = GatherToRoot(targets);
		std::vector<std::array<double, 4> > const all_old_cache = GatherToRoot(old_cache);
		bool gravity_ok = true;
		bool stale_detected = true;
		if(g_rank == 0)
		{
			std::unordered_map<unsigned long long, Vector3D> stale;
			stale.reserve(all_old_cache.size());
			for(std::array<double, 4> const& entry : all_old_cache)
				stale[static_cast<unsigned long long>(entry[0])] =
					Vector3D(entry[1], entry[2], entry[3]);
			long double squared = 0;
			long double error_squared = 0;
			long double direct_squared = 0;
			for(std::array<double, 10> const& t : all_targets)
			{
				double ax = 0, ay = 0, az = 0, scale = 0;
				double nx = 0, ny = 0, nz = 0;
				for(std::array<double, 5> const& src : all_sources)
				{
					double const dx = src[0] - t[0];
					double const dy = src[1] - t[1];
					double const dz = src[2] - t[2];
					double const r2 = dx * dx + dy * dy + dz * dz;
					if(r2 == 0)
						continue;
					double const inv_r = 1.0 / std::sqrt(r2);
					double const f = src[3] * inv_r * inv_r * inv_r;
					ax += f * dx;
					ay += f * dy;
					az += f * dz;
					scale += src[3] * inv_r * inv_r;
					if(src[4] > max_old_id)
					{
						nx += f * dx;
						ny += f * dy;
						nz += f * dz;
					}
				}
				ax *= settings.G;
				ay *= settings.G;
				az *= settings.G;
				scale *= settings.G;
				double const ex = t[3] - ax, ey = t[4] - ay, ez = t[5] - az;
				double const error = std::sqrt(ex * ex + ey * ey + ez * ez);
				double const direct = std::sqrt(ax * ax + ay * ay + az * az);
				double const scaled = scale > 0 ? error / scale : 0;
				squared += static_cast<long double>(scaled) * scaled;
				error_squared += static_cast<long double>(error) * error;
				direct_squared += static_cast<long double>(direct) * direct;
				result.gravity_max_scaled_error = std::max(result.gravity_max_scaled_error, scaled);
				if(direct > 0)
					result.gravity_max_relative_error = std::max(result.gravity_max_relative_error,
						error / direct);
				// Stale cache: the pre-growth value of this ID, zero for a new cell.
				unsigned long long const id = static_cast<unsigned long long>(t[9]);
				auto const old_value = stale.find(id);
				Vector3D const stale_a = old_value == stale.end() ? Vector3D() : old_value->second;
				double const sx = stale_a.x - ax, sy = stale_a.y - ay, sz = stale_a.z - az;
				if(scale > 0)
				{
					result.gravity_stale_max_scaled_error = std::max(
						result.gravity_stale_max_scaled_error,
						std::sqrt(sx * sx + sy * sy + sz * sz) / scale);
					if(t[9] <= max_old_id)
						result.gravity_new_mass_max_scaled_effect = std::max(
							result.gravity_new_mass_max_scaled_effect,
							settings.G * std::sqrt(nx * nx + ny * ny + nz * nz) / scale);
				}
				if(direct > 0)
				{
					double const limit = settings.source_cfl * std::sqrt(t[6] / direct);
					double const longest = std::max(t[7], t[8]);
					result.gravity_source_margin = std::min(result.gravity_source_margin,
						limit / longest);
					if(longest > limit * (1 + 1e-12))
						gravity_ok = false;
				}
			}
			result.gravity_targets = all_targets.size();
			result.gravity_rms_scaled_error = all_targets.empty() ? 0 :
				static_cast<double>(std::sqrt(squared / all_targets.size()));
			result.gravity_rms_relative_error = direct_squared > 0 ?
				static_cast<double>(std::sqrt(error_squared / direct_squared)) : 0;
			gravity_ok = gravity_ok && !all_targets.empty() &&
				result.gravity_max_scaled_error <= settings.gravity_tolerance;
			stale_detected = result.gravity_stale_max_scaled_error > settings.gravity_tolerance;
		}
		double values[8] = {static_cast<double>(result.gravity_targets),
			result.gravity_max_scaled_error, result.gravity_rms_scaled_error,
			result.gravity_max_relative_error, result.gravity_source_margin,
			result.gravity_rms_relative_error, result.gravity_stale_max_scaled_error,
			result.gravity_new_mass_max_scaled_effect};
#ifdef RICH_MPI
		MPI_Bcast(values, 8, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
		result.gravity_targets = static_cast<unsigned long long>(values[0]);
		result.gravity_max_scaled_error = values[1];
		result.gravity_rms_scaled_error = values[2];
		result.gravity_max_relative_error = values[3];
		result.gravity_source_margin = values[4];
		result.gravity_rms_relative_error = values[5];
		result.gravity_stale_max_scaled_error = values[6];
		result.gravity_new_mass_max_scaled_effect = values[7];
		Require(result, BroadcastFromRoot(gravity_ok),
			"gravity: cached accelerations off the direct sum beyond tolerance, or an interval above the unrelaxed direct source limit");
		Require(result, BroadcastFromRoot(stale_detected),
			"gravity: the direct comparison would not detect the pre-growth (stale) cache");
	}

	// Generators strictly inside the new box, mesh and committed positions agree.
	std::vector<Vector3D> const committed = simulation.CommittedGeneratorPoints();
	bool inside = committed.size() == n;
	for(size_t i = 0; inside && i < n; ++i)
		inside = StrictlyInside(tess.GetMeshPoint(i), new_box) &&
			StrictlyInside(committed[i], new_box) &&
			committed[i].x == tess.GetMeshPoint(i).x &&
			committed[i].y == tess.GetMeshPoint(i).y &&
			committed[i].z == tess.GetMeshPoint(i).z;
	Require(result, AllRanks(inside),
		"an owned generator lies outside the new box or off its committed position");

	// Per-cell checks by stable ID on rank 0.
	std::vector<CellRecord> old_records = GatherToRoot(BeforeRecords(before));
	std::vector<CellRecord> new_records = GatherToRoot(AfterRecords(simulation, tess));
	bool root_ok = true;
	std::vector<std::string> root_failures;
	auto root_require = [&root_ok, &root_failures](bool condition, std::string const& what)
	{
		if(!condition)
		{
			root_ok = false;
			root_failures.push_back(what);
		}
	};
	if(g_rank == 0)
	{
		auto const by_id = [](CellRecord const& a, CellRecord const& b) { return a.id < b.id; };
		std::sort(old_records.begin(), old_records.end(), by_id);
		std::sort(new_records.begin(), new_records.end(), by_id);
		bool unique_old = true;
		bool unique_new = true;
		for(size_t i = 1; i < old_records.size(); ++i)
			unique_old = unique_old && old_records[i - 1].id != old_records[i].id;
		for(size_t i = 1; i < new_records.size(); ++i)
			unique_new = unique_new && new_records[i - 1].id != new_records[i].id;
		root_require(unique_old, "cell IDs not unique before growth");
		root_require(unique_new, "cell IDs not unique after growth");
		unsigned long long const max_old_id = old_records.empty() ? 0 : old_records.back().id;
		root_require(!new_records.empty() &&
			new_records.back().id == static_cast<unsigned long long>(simulation.GetMaxID()),
			"largest cell ID after growth != GetMaxID()");

		long double mass_before = 0, energy_before = 0, volume_before = 0;
		long double mass_after = 0, energy_after = 0, volume_after = 0;
		long double inserted_mass = 0, inserted_energy = 0;
		long double mass_volume_change = 0, energy_volume_change = 0;
		unsigned long long lost = 0, added = 0, stale_new_ids = 0;
		unsigned long long changed_primitives = 0, moved_generators = 0;
		unsigned long long inconsistent_mass = 0, coarse_reseeded = 0;
		unsigned long long wrong_new_state = 0, new_inside_old_box = 0;
		unsigned long long changed_point_velocity = 0, coarsened_bins = 0;
		unsigned long long new_point_velocity = 0;
		std::uint64_t const reference_digest = CellDigest(reference_cell);
		for(CellRecord const& record : old_records)
		{
			mass_before += record.mass;
			energy_before += record.energy;
			volume_before += record.volume;
		}
		size_t j = 0;
		for(CellRecord const& after : new_records)
		{
			mass_after += after.mass;
			energy_after += after.energy;
			volume_after += after.volume;
			// PrimitiveToConserved on the new mesh.
			if(!(RelativeDifference(after.mass, after.density * after.volume) <= 1e-14))
				++inconsistent_mass;
			while(j < old_records.size() && old_records[j].id < after.id)
			{
				++lost;
				++j;
			}
			if(j < old_records.size() && old_records[j].id == after.id)
			{
				CellRecord const& old = old_records[j];
				++j;
				// Every primitive field (CellDigest), the committed point
				// velocity, and a bin that can only be shortened.
				if(old.digest != after.digest || !(SameBits(old.density, after.density) &&
					 SameBits(old.internal_energy, after.internal_energy) &&
					 SameBits(old.vx, after.vx) && SameBits(old.vy, after.vy) &&
					 SameBits(old.vz, after.vz)))
					++changed_primitives;
				if(!(SameBits(old.wx, after.wx) && SameBits(old.wy, after.wy) &&
					 SameBits(old.wz, after.wz)))
					++changed_point_velocity;
				if(after.bin > old.bin)
					++coarsened_bins;
				if(!(SameBits(old.px, after.px) && SameBits(old.py, after.py) &&
					 SameBits(old.pz, after.pz)))
					++moved_generators;
				double const dv = after.volume - old.volume;
				mass_volume_change += static_cast<long double>(old.density) * dv;
				double const specific = old.internal_energy +
					0.5 * (old.vx * old.vx + old.vy * old.vy + old.vz * old.vz);
				energy_volume_change += static_cast<long double>(old.density * specific) * dv;
				bool const changed = !(std::isfinite(old.volume) && old.volume > 0) ||
					std::abs(dv) > 1e-8 * old.volume;
				if(changed)
				{
					++result.changed_old_cells;
					result.max_changed_density_deviation = std::max(
						result.max_changed_density_deviation,
						RelativeDifference(old.density, reference_cell.density));
					if(after.bin > growth.seed_bin)
						++coarse_reseeded;
				}
			}
			else
			{
				++added;
				inserted_mass += after.mass;
				inserted_energy += after.energy;
				if(after.id <= max_old_id)
					++stale_new_ids;
				if(after.digest != reference_digest)
					++wrong_new_state;
				if(!(after.wx == 0 && after.wy == 0 && after.wz == 0))
					++new_point_velocity;
				if(!OutsideBox(after.px, after.py, after.pz, old_box))
					++new_inside_old_box;
				if(after.bin > growth.seed_bin)
					++coarse_reseeded;
			}
		}
		lost += static_cast<unsigned long long>(old_records.size() - j);
		root_require(lost == 0, "old cells lost in growth: " + std::to_string(lost));
		root_require(added == growth.added_cells,
			"new IDs after growth (" + std::to_string(added) + ") != added_cells");
		root_require(stale_new_ids == 0, "a new cell reuses an old ID range");
		root_require(changed_primitives == 0,
			"old cells changed primitive state: " + std::to_string(changed_primitives));
		root_require(moved_generators == 0,
			"old generators moved in growth: " + std::to_string(moved_generators));
		root_require(changed_point_velocity == 0,
			"old cells changed committed point velocity: " +
			std::to_string(changed_point_velocity));
		root_require(coarsened_bins == 0,
			"old cells moved to a coarser bin: " + std::to_string(coarsened_bins));
		root_require(new_point_velocity == 0,
			"new cells with a nonzero point velocity: " + std::to_string(new_point_velocity));
		root_require(inconsistent_mass == 0,
			"extensive mass != density * new volume: " + std::to_string(inconsistent_mass));
		root_require(wrong_new_state == 0,
			"new cells without the reference state (all primitive fields): " +
			std::to_string(wrong_new_state));
		root_require(new_inside_old_box == 0,
			"new generators inside the old box: " + std::to_string(new_inside_old_box));
		root_require(coarse_reseeded == 0,
			"new or volume-changed cells coarser than seed_bin: " + std::to_string(coarse_reseeded));
		root_require(growth.reseeded_cells == added + result.changed_old_cells,
			"report reseeded_cells (" + std::to_string(growth.reseeded_cells) +
			") != new + volume-changed cells (" +
			std::to_string(added + result.changed_old_cells) + ")");

		// Report totals against the gathered per-cell sums.
		root_require(RelativeDifference(growth.mass_before, static_cast<double>(mass_before)) <= 1e-12,
			"report mass_before != gathered sum");
		root_require(RelativeDifference(growth.mass_after, static_cast<double>(mass_after)) <= 1e-12,
			"report mass_after != gathered sum");
		root_require(RelativeDifference(growth.inserted_mass, static_cast<double>(inserted_mass)) <= 1e-12,
			"report inserted_mass != gathered sum over new IDs");
		root_require(RelativeDifference(growth.energy_before, static_cast<double>(energy_before)) <= 1e-12,
			"report energy_before != gathered sum");
		root_require(RelativeDifference(growth.energy_after, static_cast<double>(energy_after)) <= 1e-12,
			"report energy_after != gathered sum");
		root_require(RelativeDifference(growth.inserted_energy, static_cast<double>(inserted_energy)) <= 1e-12,
			"report inserted_energy != gathered sum over new IDs");

		// Volume closure and the conservation bookkeeping.
		result.volume_closure_after =
			RelativeDifference(static_cast<double>(volume_after), new_volume);
		result.volume_closure_before =
			RelativeDifference(static_cast<double>(volume_before), old_volume);
		root_require(result.volume_closure_after <= 1e-9,
			"new mesh volumes do not sum to the new box volume");
		double const mb = growth.mass_before;
		result.mass_residual = std::abs(static_cast<double>(
			static_cast<long double>(growth.mass_after) - growth.mass_before -
			growth.inserted_mass - mass_volume_change)) / mb;
		result.mass_residual_naive = std::abs(
			growth.mass_after - growth.mass_before - growth.inserted_mass) / mb;
		result.mass_residual_box = std::abs(growth.mass_after - growth.mass_before -
			reference_cell.density * (new_volume - old_volume)) / mb;
		double const eb = std::abs(growth.energy_before);
		result.energy_residual = std::abs(static_cast<double>(
			static_cast<long double>(growth.energy_after) - growth.energy_before -
			growth.inserted_energy - energy_volume_change)) / eb;
		result.energy_residual_naive = std::abs(
			growth.energy_after - growth.energy_before - growth.inserted_energy) / eb;
		root_require(result.mass_residual <= 1e-10,
			"mass bookkeeping residual above 1e-10");
		// The recompute from primitives changes the total energy by the
		// pre-growth mismatch sum (rho (e + v^2/2) V - E) of the old cells (the
		// dual-energy primitive against the total-energy extensive): measured
		// up to 8.7e-4 of the total in the serial run, so only a loose bound.
		root_require(result.energy_residual <= 1e-2,
			"energy bookkeeping residual above the loose bound 1e-2");
		root_require(std::isfinite(result.mass_residual_naive) &&
			std::isfinite(result.mass_residual_box) &&
			std::isfinite(result.energy_residual_naive),
			"non-finite growth residual");
	}
	double scalars[6] = {result.volume_closure_after, result.volume_closure_before,
		result.mass_residual, result.mass_residual_naive, result.mass_residual_box,
		result.energy_residual};
	double more[2] = {result.energy_residual_naive, result.max_changed_density_deviation};
	unsigned long long changed = result.changed_old_cells;
#ifdef RICH_MPI
	MPI_Bcast(scalars, 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(more, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&changed, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
#endif
	result.volume_closure_after = scalars[0];
	result.volume_closure_before = scalars[1];
	result.mass_residual = scalars[2];
	result.mass_residual_naive = scalars[3];
	result.mass_residual_box = scalars[4];
	result.energy_residual = scalars[5];
	result.energy_residual_naive = more[0];
	result.max_changed_density_deviation = more[1];
	result.changed_old_cells = changed;
	root_ok = BroadcastFromRoot(root_ok);
	if(!root_ok)
	{
		result.ok = false;
		for(std::string const& failure : root_failures)
			result.failures.push_back(failure);
		if(g_rank != 0)
			result.failures.push_back("per-cell growth check failed on rank 0");
	}
	return result;
}

// The conservative force with an explicit record of how each first half
// kick was applied: from the acceleration cache (no geometry) or through
// ApplyIndividual on a first-half mesh.  Every cached kick records its
// cell's stable ID and the acceleration it used; for the event after a growth
// the test gathers them and compares each, bit for bit, with the refreshed
// cache recorded right after the growth (cells may change rank in between).
class RecordingConservativeForce3D : public ConservativeForce3D
{
public:
	using ConservativeForce3D::ConservativeForce3D;

	void ApplyIndividual(const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& fluxes,
		const vector<Vector3D>& point_velocities,
		double time,
		const IndividualStepContext& context,
		IndividualSourcePhase phase,
		vector<Conserved3D>& extensives) const override
	{
		if(phase == IndividualSourcePhase::FirstHalf)
			++geometry_first_half_calls;
		ConservativeForce3D::ApplyIndividual(tess, cells, fluxes, point_velocities,
			time, context, phase, extensives);
	}

	void ApplyIndividualFirstHalfFromCache(
		const vector<ComputationalCell3D>& cells,
		const vector<Vector3D>& point_velocities,
		double time,
		const IndividualStepContext& context,
		vector<Conserved3D>& extensives) const override
	{
		++cache_first_half_calls;
		for(size_t const index : context.active_indices)
		{
			++cache_active;
			if(index >= context.gravity_half_kick_pending.size() ||
			   index >= context.cached_accelerations.size() || index >= cells.size())
			{
				++cache_pending_violations;
				continue;
			}
			if(context.gravity_half_kick_pending[index] == 0)
				++cache_pending_violations;
			Vector3D const& used = context.cached_accelerations[index];
			kicked.push_back({{static_cast<double>(cells[index].ID), used.x, used.y, used.z}});
		}
		ConservativeForce3D::ApplyIndividualFirstHalfFromCache(cells, point_velocities,
			time, context, extensives);
	}

	void Reset(void)
	{
		geometry_first_half_calls = 0;
		cache_first_half_calls = 0;
		cache_active = 0;
		cache_pending_violations = 0;
		kicked.clear();
	}

	mutable unsigned long long geometry_first_half_calls = 0;
	mutable unsigned long long cache_first_half_calls = 0;
	mutable unsigned long long cache_active = 0;
	mutable unsigned long long cache_pending_violations = 0;
	// Stable ID and acceleration of every cached kick since Reset.
	mutable std::vector<std::array<double, 4> > kicked;
};
}

int main(void)
{
#ifdef RICH_MPI
	MPI_Init(nullptr, nullptr);
	MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &g_size);
#endif
	int const rank = g_rank;
	bool const mpi_mode = g_size > 1;
	auto const wall_start = std::chrono::steady_clock::now();
	try
	{
		Vector3D const ll(-1, -1, -1), ur(1, 1, 1);
		size_t const point_count = EnvironmentSize("RICH_TEST_POINT_COUNT",
			mpi_mode ? static_cast<size_t>(2e5) : static_cast<size_t>(2e4));
		// Serial (2e4 cells): two growths at t ~ 0.16-0.17 (fast interior
		// cells set the 5 maxR shift, 0.37) and ~57 events after them, ~3 min.
		// MPI (2e5 cells, 16 ranks): the finer mesh first grows at the legacy
		// floor shift (0.3) at t ~ 0.25 and again at t ~ 0.26 (then at ~0.50).
		double const final_time = EnvironmentDouble("RICH_TEST_FINAL_TIME",
			mpi_mode ? 0.35 : 0.3);
		double const min_velocity = EnvironmentDouble("RICH_TEST_MIN_VELOCITY", 0.1);
		double const volume_fraction = EnvironmentDouble("RICH_TEST_VOLUME_FRACTION",
			mpi_mode ? 1e-5 : 1e-4);
		size_t const max_events = EnvironmentSize("RICH_TEST_MAX_EVENTS", 20000);
		size_t const min_growths = EnvironmentSize("RICH_TEST_MIN_GROWTHS", 2);
		size_t const min_events_after_growth = 20;
		unsigned const initial_bin = static_cast<unsigned>(
			EnvironmentSize("RICH_TEST_INITIAL_BIN", 30));
		unsigned const maximum_bin = static_cast<unsigned>(
			EnvironmentSize("RICH_TEST_MAXIMUM_BIN", 40));
		// RICH_TEST_FORCE_SYNCHRONIZED=1: every cell on one shared bin
		// (IndividualTimeStepOptions::force_synchronized).  Every event is then
		// synchronized, so the request path (BoxGrowthDue between events) is
		// not exercised and not required.
		char const* const forced_env = std::getenv("RICH_TEST_FORCE_SYNCHRONIZED");
		bool const force_synchronized = forced_env != nullptr && forced_env[0] != '\0' &&
			std::string(forced_env) != "0";
		double const blast_radius = 0.15;
		double const blast_energy = 1.0;
		double const ambient_pressure = 1e-5;

		std::vector<Vector3D> points;
		if(rank == 0)
			points = RandRectangular(point_count, ll, ur);
#ifdef RICH_MPI
		points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif
		points = RoundGrid3D(points, ll, ur, 10);
		Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
		tess.BuildParallel(points);
#else
		tess.Build(points);
#endif

		IdealGas eos(5. / 3.);
		ComputationalCell3D ambient;
		ambient.density = 1;
		ambient.velocity = Vector3D(0, 0, 0);
		ambient.internal_energy = eos.dp2e(ambient.density, ambient_pressure,
			ambient.tracers, ComputationalCell3D::tracerNames);
		ambient.pressure = eos.de2p(ambient.density, ambient.internal_energy,
			ambient.tracers, ComputationalCell3D::tracerNames);
		size_t const n_local = tess.GetPointNo();
		double blast_volume = 0;
		for(size_t i = 0; i < n_local; ++i)
			if(abs(tess.GetMeshPoint(i)) < blast_radius)
				blast_volume += tess.GetVolume(i);
		blast_volume = SumAll(blast_volume);
		if(!(blast_volume > 0))
			throw std::runtime_error("no cell inside the blast radius");
		ComputationalCell3D blast = ambient;
		blast.internal_energy = blast_energy / (ambient.density * blast_volume);
		blast.pressure = eos.de2p(blast.density, blast.internal_energy, blast.tracers,
			ComputationalCell3D::tracerNames);
		std::vector<ComputationalCell3D> cells(n_local, ambient);
		for(size_t i = 0; i < n_local; ++i)
			if(abs(tess.GetMeshPoint(i)) < blast_radius)
				cells[i] = blast;

		Hllc3D rs;
		RigidWallGenerator3D ghost;
		LinearGauss3D interp(eos, ghost);
		IsBoundaryFace3D is_boundary;
		IsBulkFace3D is_bulk;
		RigidWallFlux3D rigid_flux(rs);
		RegularFlux3D regular_flux(rs);
		std::vector<std::pair<const ConditionActionFlux1::Condition3D*,
			const ConditionActionFlux1::Action3D*> > sequence;
		sequence.push_back(std::make_pair(&is_boundary, &rigid_flux));
		sequence.push_back(std::make_pair(&is_bulk, &regular_flux));
		ConditionActionFlux1 flux(sequence, interp);
		std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*,
			const ConditionExtensiveUpdater3D::Action3D*> > eu_sequence;
		ConditionExtensiveUpdater3D eu(eu_sequence);
		DefaultCellUpdater cu;
		// RICH_TEST_BOX_GROWTH_GRAVITY=1: self-gravity through a conservative
		// force over the FMM (default order 2, theta 1, leaf 64, as the TDE
		// driver; RICH_TEST_FMM_ORDER / _THETA / _LEAF for the calibration
		// runs), G = RICH_TEST_GRAVITY_G (0.03: the source limit binds for
		// ambient cells near the walls while the infall stays below
		// min_velocity).  RICH_TEST_GRAVITY_STALE_CACHE=1 is the negative
		// control: after each growth the test puts the pre-growth cache back
		// (new cells zero) and the gravity check must then fail.
		char const* const gravity_env = std::getenv("RICH_TEST_BOX_GROWTH_GRAVITY");
		bool const gravity = gravity_env != nullptr && gravity_env[0] != '\0' &&
			std::string(gravity_env) != "0";
		char const* const stale_env = std::getenv("RICH_TEST_GRAVITY_STALE_CACHE");
		bool const stale_cache_control = gravity && stale_env != nullptr &&
			stale_env[0] != '\0' && std::string(stale_env) != "0";
		double const gravity_G = EnvironmentDouble("RICH_TEST_GRAVITY_G", 0.03);
		ZeroForce3D zero_force;
		FmmGravityOptions fmm_options;
		fmm_options.expansionOrder = static_cast<int>(EnvironmentSize("RICH_TEST_FMM_ORDER", 2));
		fmm_options.thetaCritical = EnvironmentDouble("RICH_TEST_FMM_THETA", 1.0);
		fmm_options.leafCapacity = EnvironmentSize("RICH_TEST_FMM_LEAF", 64);
		FastMultipoleAcceleration3D fmm_acceleration(fmm_options, gravity_G);
		RecordingConservativeForce3D gravity_force(fmm_acceleration, false);
		SourceTerm3D const& force = gravity ?
			static_cast<SourceTerm3D const&>(gravity_force) :
			static_cast<SourceTerm3D const&>(zero_force);
		if(gravity)
			// The first-half phase timing shows whether the event after a growth
			// kicks from the refreshed cache (no first-half mesh).
			setenv("RICH_INDIVIDUAL_PERF_TRACE", "1", 0);
		LimitSettings limit_settings;
		limit_settings.gravity = gravity;
		limit_settings.G = gravity_G;
		limit_settings.drift_fraction = EnvironmentDouble("RICH_INDIVIDUAL_MESH_DRIFT_FRACTION", 0.25);
		limit_settings.gravity_targets = EnvironmentSize("RICH_TEST_GRAVITY_TARGETS", 8000);
		// FMM order 2, theta 1 against the direct sum (serial, ~7000 targets
		// per growth): max 0.042, rms 0.011 of the force scale; MPI: max 0.065.
		// The error falls with a higher order and smaller theta (README.md),
		// and the stale pre-growth cache errs by ~1 (a new cell's whole pull).
		limit_settings.gravity_tolerance = EnvironmentDouble("RICH_TEST_GRAVITY_TOLERANCE", 0.1);
		double const cfl_number = 0.3;
		// The growth caps every bin by the hydro CFL limit on the rebuilt mesh,
		// which needs a CourantFriedrichsLewy time-step function.
		double const source_cfl = 1.0;
		limit_settings.source_cfl = source_cfl;
		auto tsf = std::make_shared<CourantFriedrichsLewy>(cfl_number, source_cfl, force);
		Lagrangian3D bpm;
		RoundCells3D pm(bpm, eos);

		Simulation simulation(tess, cells, eos);
		simulation.SetTimeStepFunction(tsf);
		HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos,
			simulation.getTracker(), pm, *tsf, flux, cu, eu, force,
			std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));
		auto hydro_step = std::make_shared<HydroStep>(sim, HydroStep::TIMEADVANCE_2);
		simulation.addPhysics(hydro_step);

		// A safe first interval: a fraction of the smallest CFL limit.
		double min_crossing = std::numeric_limits<double>::infinity();
		for(size_t i = 0; i < n_local; ++i)
		{
			double const c = eos.de2c(cells[i].density, cells[i].internal_energy,
				cells[i].tracers, ComputationalCell3D::tracerNames);
			min_crossing = std::min(min_crossing, tess.GetWidth(i) / c);
		}
		min_crossing = MinAll(min_crossing);
		double const initial_dt = 0.5 * cfl_number * min_crossing;
		simulation.SetTimeStep(initial_dt);

		IndividualTimeStepOptions options;
		options.initial_bin = static_cast<std::uint8_t>(initial_bin);
		options.maximum_bin = static_cast<std::uint8_t>(maximum_bin);
		options.maximum_neighbor_bin_difference = 1;
		options.mesh_build_policy = force_synchronized ?
			IndividualMeshBuildPolicy::FullReference : IndividualMeshBuildPolicy::AutoPartial;
		options.force_synchronized = force_synchronized;
		simulation.EnableIndividualTimeSteps(options);
#ifdef RICH_MPI
		simulation.PresetLoadBalance("hydro");
#endif

		Totals const initial_totals = ComputeTotals(simulation);
		unsigned long long const initial_cells =
			SumAll(static_cast<unsigned long long>(simulation.getCells().size()));
		if(rank == 0)
			std::cout << std::setprecision(12) << "RICH_TEST_SETUP mode="
				<< (mpi_mode ? "mpi" : "serial") << " variant="
				<< (force_synchronized ? "force_synchronized" : "variable")
				<< (gravity ? "+gravity G=" + std::to_string(gravity_G) +
					" fmm_order=" + std::to_string(fmm_options.expansionOrder) +
					" fmm_theta=" + std::to_string(fmm_options.thetaCritical) +
					" fmm_leaf=" + std::to_string(fmm_options.leafCapacity) +
					(stale_cache_control ? " STALE_CACHE_CONTROL" : "") : std::string())
				<< " ranks=" << g_size
				<< " cells=" << initial_cells << " final_time=" << final_time
				<< " min_velocity=" << min_velocity << " volume_fraction=" << volume_fraction
				<< " initial_dt=" << initial_dt << " initial_bin=" << initial_bin
				<< " maximum_bin=" << maximum_bin << " blast_cells_volume=" << blast_volume
				<< " blast_internal_energy=" << blast.internal_energy
				<< " mass=" << initial_totals.mass << " energy=" << initial_totals.energy
				<< std::endl;

		double expected_mass = initial_totals.mass;
		double expected_energy = initial_totals.energy;
		double max_mass_drift = 0;
		double max_energy_drift = 0;
		size_t events = 0;
		size_t events_after_first_growth = 0;
		size_t growths = 0;
		size_t growths_after_request = 0;
		size_t synchronized_states = 0;
		size_t requests = 0;
		size_t requests_without_growth = 0;
		size_t noop_checks = 0;
		size_t noop_checks_not_due = 0;
		size_t noop_changed = 0;
		size_t due_false_between_events = 0;
		size_t due_without_growth = 0;
		size_t growth_without_due = 0;
		bool growth_checks_ok = true;
		bool pending = false;
		double first_growth_time = -1;
		double growth_seconds = 0;
		double max_mass_residual = 0;
		double max_mass_residual_naive = 0;
		double max_mass_residual_box = 0;
		double max_energy_residual = 0;
		double max_energy_residual_naive = 0;
		double max_volume_closure_after = 0;
		double max_volume_closure_before = 0;
		double min_cfl_margin = std::numeric_limits<double>::infinity();
		double min_expected_margin = std::numeric_limits<double>::infinity();
		double max_gravity_scaled_error = 0;
		double max_gravity_rms_relative_error = 0;
		double min_gravity_stale_error = std::numeric_limits<double>::infinity();
		double min_gravity_source_margin = std::numeric_limits<double>::infinity();
		double max_gravity_new_mass_effect = 0;
		size_t stale_control_growths_caught = 0;
		size_t stale_control_other_failures = 0;
		unsigned long long total_drift_tightened = 0;
		// Index of the growth whose next event is still to be reported.
		size_t next_event_after_growth = 0;
		size_t next_events_reported = 0;
		// Stable ID and refreshed acceleration of every owned cell after the
		// last growth (gravity), compared with the next event's kicks.
		std::vector<std::array<double, 4> > expected_cache;
		bool next_events_finite = true;
		bool next_events_cache_ok = true;
		unsigned long long total_added = 0;
		std::pair<Vector3D, Vector3D> const initial_box = tess.GetBoxCoordinates();

		// A growth at the last event gets its next event too.
		while(simulation.GetTime() < final_time || next_event_after_growth > 0)
		{
			if(events >= max_events)
				break;
			if(gravity && next_event_after_growth > 0)
				gravity_force.Reset();
			simulation.step();
			++events;
			if(growths > 0)
				++events_after_first_growth;
			if(next_event_after_growth > 0)
			{
				// The event right after a growth: finite totals, and the first-half
				// source cost (a kick from the refreshed cache builds no mesh).
				Totals const next = ComputeTotals(simulation);
				SourceStepTiming const source_timing = sim.GetLastSourceStepTiming();
				MeshBuildTiming const mesh_timing = sim.GetLastMeshBuildTiming();
				bool const next_finite = next.finite && next.positive &&
					std::isfinite(source_timing.first_seconds);
				next_events_finite = next_events_finite && next_finite;
				++next_events_reported;
				// Gravity: every rank kicked the first half from the cache (no
				// first-half mesh, no geometry-path kick), every kicked cell had a
				// pending kick, and its cached acceleration was the refreshed one.
				unsigned long long counts[7] = {gravity_force.cache_first_half_calls,
					gravity_force.geometry_first_half_calls, gravity_force.cache_active,
					gravity_force.cache_pending_violations, 0, 0, 0};
				unsigned long long minimum_cache_calls = counts[0];
#ifdef RICH_MPI
				MPI_Allreduce(MPI_IN_PLACE, counts, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
					MPI_COMM_WORLD);
				MPI_Allreduce(MPI_IN_PLACE, &minimum_cache_calls, 1, MPI_UNSIGNED_LONG_LONG,
					MPI_MIN, MPI_COMM_WORLD);
#endif
				if(gravity)
				{
					// Every kick against the refreshed cache by stable ID, on rank 0:
					// matched, mismatched, or an ID the growth did not record.
					std::vector<std::array<double, 4> > const kicks =
						GatherToRoot(gravity_force.kicked);
					std::vector<std::array<double, 4> > const refreshed =
						GatherToRoot(expected_cache);
					if(rank == 0)
					{
						std::unordered_map<unsigned long long, Vector3D> by_id;
						by_id.reserve(refreshed.size());
						for(std::array<double, 4> const& entry : refreshed)
							by_id[static_cast<unsigned long long>(entry[0])] =
								Vector3D(entry[1], entry[2], entry[3]);
						for(std::array<double, 4> const& kick : kicks)
						{
							auto const found = by_id.find(static_cast<unsigned long long>(kick[0]));
							if(found == by_id.end())
								++counts[6];
							else if(SameVector(found->second, Vector3D(kick[1], kick[2], kick[3])))
								++counts[4];
							else
								++counts[5];
						}
					}
#ifdef RICH_MPI
					MPI_Bcast(counts + 4, 3, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
#endif
				}
				bool const cache_used = !gravity || (minimum_cache_calls >= 1 &&
					counts[1] == 0 && counts[2] > 0 && counts[3] == 0 &&
					counts[4] == counts[2] && counts[5] == 0 && counts[6] == 0);
				next_events_cache_ok = next_events_cache_ok && cache_used;
				expected_cache.clear();
				if(rank == 0)
					std::cout << std::setprecision(6) << "RICH_TEST_BOX_GROWTH_NEXT_EVENT index="
						<< next_event_after_growth << " event=" << events
						<< " source_first_s_rank0=" << source_timing.first_seconds
						<< " mesh_builds_rank0=" << mesh_timing.builds
						<< " full_builds_rank0=" << mesh_timing.full_builds
						<< " cache_first_half_calls=" << counts[0]
						<< " min_rank_cache_first_half_calls=" << minimum_cache_calls
						<< " geometry_first_half_calls=" << counts[1]
						<< " cached_kicks=" << counts[2]
						<< " pending_violations=" << counts[3]
						<< " cache_matched=" << counts[4]
						<< " cache_mismatched=" << counts[5]
						<< " cache_unmatched=" << counts[6]
						<< " cache_ok=" << (cache_used ? 1 : 0)
						<< " finite=" << (next_finite ? 1 : 0)
						<< " mass=" << std::setprecision(12) << next.mass
						<< " energy=" << next.energy << std::endl;
				next_event_after_growth = 0;
			}
			if(rank == 0 && events % 100 == 0)
				std::cout << "RICH_TEST_PROGRESS event=" << events << " cycle="
					<< simulation.GetCycle() << " time=" << simulation.GetTime()
					<< " growths=" << growths << std::endl;
			// The driver idiom (runs/BaseTDEComptonIndividual): grow only at a
			// synchronized state; request one when growth is due.
			if(simulation.IndividualStateSynchronized())
			{
				++synchronized_states;
				Totals const totals = ComputeTotals(simulation);
				max_mass_drift = std::max(max_mass_drift,
					RelativeDifference(totals.mass, expected_mass));
				max_energy_drift = std::max(max_energy_drift,
					RelativeDifference(totals.energy, expected_energy));
				bool const due_here = BoxGrowthDue(simulation, min_velocity);
				Snapshot const before = TakeSnapshot(simulation, tess);
				bool const was_pending = pending;
				Simulation::DomainGrowthReport growth;
				auto const growth_start = std::chrono::steady_clock::now();
				bool const grew = UpdateBoxSynchronized(tess, simulation, min_velocity,
					volume_fraction, ambient, &growth);
				double const seconds = std::chrono::duration<double>(
					std::chrono::steady_clock::now() - growth_start).count();
				pending = false;
				if(was_pending && !grew)
					++requests_without_growth;
				if(grew)
				{
					++growths;
					if(was_pending)
						++growths_after_request;
					if(!due_here)
						++growth_without_due;
					if(first_growth_time < 0)
						first_growth_time = simulation.GetTime();
					growth_seconds += seconds;
					total_added += growth.added_cells;
					if(stale_cache_control)
					{
						// Negative control: the pre-growth cache by stable ID (new
						// cells zero), pending kicks untouched.
						std::unordered_map<size_t, Vector3D> old_cache;
						for(size_t i = 0; i < before.cells.size(); ++i)
							old_cache[before.cells[i].ID] = before.states[i].cached_acceleration;
						std::vector<CellTimeState>& live =
							simulation.GetIndividualTimeStepScheduler()->states();
						std::vector<ComputationalCell3D> const& live_cells = simulation.getCells();
						for(size_t i = 0; i < live.size() && i < live_cells.size(); ++i)
						{
							auto const found = old_cache.find(live_cells[i].ID);
							live[i].cached_acceleration =
								found == old_cache.end() ? Vector3D() : found->second;
						}
					}
					GrowthResult const result = VerifyGrowth(before, growth, simulation, tess,
						*tsf, eos, ambient, limit_settings);
					if(gravity)
					{
						// The cache the next event's first half must kick with.
						expected_cache.clear();
						std::vector<CellTimeState> const& live =
							simulation.GetIndividualTimeStepScheduler()->states();
						std::vector<ComputationalCell3D> const& live_cells = simulation.getCells();
						for(size_t i = 0; i < live.size() && i < live_cells.size(); ++i)
							expected_cache.push_back({{static_cast<double>(live_cells[i].ID),
								live[i].cached_acceleration.x, live[i].cached_acceleration.y,
								live[i].cached_acceleration.z}});
					}
					total_drift_tightened += result.drift_tightened;
					next_event_after_growth = growths;
					growth_checks_ok = growth_checks_ok && result.ok;
					expected_mass += growth.mass_after - growth.mass_before;
					expected_energy += growth.energy_after - growth.energy_before;
					max_mass_residual = std::max(max_mass_residual, result.mass_residual);
					max_mass_residual_naive = std::max(max_mass_residual_naive,
						result.mass_residual_naive);
					max_mass_residual_box = std::max(max_mass_residual_box,
						result.mass_residual_box);
					max_energy_residual = std::max(max_energy_residual, result.energy_residual);
					max_energy_residual_naive = std::max(max_energy_residual_naive,
						result.energy_residual_naive);
					max_volume_closure_after = std::max(max_volume_closure_after,
						result.volume_closure_after);
					max_volume_closure_before = std::max(max_volume_closure_before,
						result.volume_closure_before);
					min_cfl_margin = std::min(min_cfl_margin, result.cfl_margin);
					min_expected_margin = std::min(min_expected_margin, result.expected_margin);
					max_gravity_scaled_error = std::max(max_gravity_scaled_error,
						result.gravity_max_scaled_error);
					max_gravity_rms_relative_error = std::max(max_gravity_rms_relative_error,
						result.gravity_rms_relative_error);
					if(gravity)
					{
						min_gravity_stale_error = std::min(min_gravity_stale_error,
							result.gravity_stale_max_scaled_error);
						min_gravity_source_margin = std::min(min_gravity_source_margin,
							result.gravity_source_margin);
						max_gravity_new_mass_effect = std::max(max_gravity_new_mass_effect,
							result.gravity_new_mass_max_scaled_effect);
					}
					if(stale_cache_control)
					{
						bool caught = false;
						for(std::string const& failure : result.failures)
						{
							if(failure.rfind("gravity: cached accelerations off", 0) == 0)
								caught = true;
							else
								++stale_control_other_failures;
						}
						if(caught)
							++stale_control_growths_caught;
					}
					std::pair<Vector3D, Vector3D> const box = tess.GetBoxCoordinates();
					if(rank == 0)
					{
						std::cout << std::setprecision(12) << "RICH_TEST_BOX_GROWTH index=" << growths
							<< " event=" << events << " cycle=" << simulation.GetCycle()
							<< " time=" << simulation.GetTime()
							<< " tick=" << simulation.GetIndividualTimeStepScheduler()->currentTick()
							<< " after_request=" << (was_pending ? 1 : 0)
							<< " due=" << (due_here ? 1 : 0)
							<< " old_ll=" << before.box.first.x << "," << before.box.first.y << "," << before.box.first.z
							<< " old_ur=" << before.box.second.x << "," << before.box.second.y << "," << before.box.second.z
							<< " new_ll=" << box.first.x << "," << box.first.y << "," << box.first.z
							<< " new_ur=" << box.second.x << "," << box.second.y << "," << box.second.z
							<< " walls_moved=" << result.walls_moved << " wall_shift=" << result.wall_shift
							<< " cells_before=" << growth.cells_before << " cells_after=" << growth.cells_after
							<< " added_cells=" << growth.added_cells << " reseeded_cells=" << growth.reseeded_cells
							<< " changed_old_cells=" << result.changed_old_cells
							<< " shortened_cells=" << growth.shortened_cells << " seed_bin=" << growth.seed_bin
							<< " cfl_margin=" << result.cfl_margin
							<< " expected_limit_margin=" << result.expected_margin
							<< " drift_tightened=" << result.drift_tightened
							<< " source_tightened=" << result.source_tightened
							<< " smallest_limit=" << growth.smallest_limit
							<< " smallest_expected_limit=" << result.smallest_cfl_limit
							<< " bins=" << result.shared_bin_min << "-" << result.shared_bin_max
							<< " accelerations_refreshed=" << (growth.accelerations_refreshed ? 1 : 0)
							<< " mass_before=" << growth.mass_before << " mass_after=" << growth.mass_after
							<< " inserted_mass=" << growth.inserted_mass
							<< " energy_before=" << growth.energy_before << " energy_after=" << growth.energy_after
							<< " inserted_energy=" << growth.inserted_energy
							<< std::setprecision(4)
							<< " mass_residual=" << result.mass_residual
							<< " mass_residual_naive=" << result.mass_residual_naive
							<< " mass_residual_box=" << result.mass_residual_box
							<< " energy_residual=" << result.energy_residual
							<< " energy_residual_naive=" << result.energy_residual_naive
							<< " changed_density_deviation=" << result.max_changed_density_deviation
							<< " volume_closure=" << result.volume_closure_after
							<< " gravity_targets=" << result.gravity_targets
							<< " gravity_max_scaled_error=" << result.gravity_max_scaled_error
							<< " gravity_rms_scaled_error=" << result.gravity_rms_scaled_error
							<< " gravity_max_relative_error=" << result.gravity_max_relative_error
							<< " gravity_source_margin=" << result.gravity_source_margin
							<< " gravity_rms_relative_error=" << result.gravity_rms_relative_error
							<< " gravity_stale_max_scaled_error=" << result.gravity_stale_max_scaled_error
							<< " gravity_new_mass_max_scaled_effect=" << result.gravity_new_mass_max_scaled_effect
							<< " growth_seconds=" << growth.seconds
							<< " ok=" << (result.ok ? 1 : 0) << std::endl;
						for(std::string const& failure : result.failures)
							std::cout << "RICH_TEST_BOX_GROWTH_FAILURE index=" << growths << " "
								<< failure << std::endl;
					}
				}
				else
				{
					++noop_checks;
					if(due_here)
						++due_without_growth;
					else
						++noop_checks_not_due;
					if(!NothingChanged(before, simulation, tess))
						++noop_changed;
				}
			}
			else if(pending || BoxGrowthDue(simulation, min_velocity))
			{
				if(!pending)
				{
					++requests;
					if(rank == 0)
						std::cout << std::setprecision(12) << "RICH_TEST_BOX_GROWTH_REQUEST event="
							<< events << " cycle=" << simulation.GetCycle() << " time="
							<< simulation.GetTime() << std::endl;
				}
				pending = true;
				simulation.RequestSynchronizedIndividualEvent();
			}
			else
				++due_false_between_events;
		}

		Totals const final_totals = ComputeTotals(simulation);
		double const final_mass_drift = RelativeDifference(final_totals.mass, expected_mass);
		double const final_energy_drift = RelativeDifference(final_totals.energy, expected_energy);
		max_mass_drift = std::max(max_mass_drift, final_mass_drift);
		max_energy_drift = std::max(max_energy_drift, final_energy_drift);
		double const momentum_ratio = std::sqrt(final_totals.momentum_x * final_totals.momentum_x +
			final_totals.momentum_y * final_totals.momentum_y +
			final_totals.momentum_z * final_totals.momentum_z) /
			std::max(final_totals.momentum_scale, 1e-300);
		unsigned long long const final_cells =
			SumAll(static_cast<unsigned long long>(simulation.getCells().size()));
		std::pair<Vector3D, Vector3D> const final_box = tess.GetBoxCoordinates();
		bool const reached_end = simulation.GetTime() >= final_time;
		double const wall_seconds = std::chrono::duration<double>(
			std::chrono::steady_clock::now() - wall_start).count();

		double const mass_drift_limit = 1e-9;
		double const energy_drift_limit = 1e-3;
		bool const pass =
			reached_end && final_totals.finite && final_totals.positive &&
			growths >= std::max<size_t>(1, min_growths) && growth_checks_ok &&
			(force_synchronized || growths_after_request >= 1) &&
			events_after_first_growth >= min_events_after_growth &&
			noop_checks_not_due >= 1 && noop_changed == 0 &&
			(force_synchronized || due_false_between_events >= 1) &&
			final_cells == initial_cells + total_added &&
			max_mass_drift <= mass_drift_limit &&
			total_drift_tightened > 0 && next_events_finite && next_events_cache_ok &&
			next_events_reported == growths && next_event_after_growth == 0 &&
			!stale_cache_control &&
			(gravity || max_energy_drift <= energy_drift_limit);
		bool const all_pass = AllRanks(pass);

		if(rank == 0)
		{
			std::ofstream out("individual_box_growth_metrics.txt");
			out << std::setprecision(12);
			out << "mode " << (mpi_mode ? "mpi" : "serial") << "\n";
			out << "variant " << (force_synchronized ? "force_synchronized" : "variable")
				<< (gravity ? "+gravity" : "") << "\n";
			out << "ranks " << g_size << "\n";
			out << "initial_cells " << initial_cells << "\n";
			out << "final_cells " << final_cells << "\n";
			out << "total_added_cells " << total_added << "\n";
			out << "final_time " << final_time << "\n";
			out << "reached_time " << simulation.GetTime() << "\n";
			out << "reached_end " << (reached_end ? 1 : 0) << "\n";
			out << "events " << events << "\n";
			out << "max_events " << max_events << "\n";
			out << "min_velocity " << min_velocity << "\n";
			out << "volume_fraction " << volume_fraction << "\n";
			out << "growths " << growths << "\n";
			out << "min_growths " << std::max<size_t>(1, min_growths) << "\n";
			out << "growths_after_request " << growths_after_request << "\n";
			out << "growth_without_due " << growth_without_due << "\n";
			out << "first_growth_time " << first_growth_time << "\n";
			out << "events_after_first_growth " << events_after_first_growth << "\n";
			out << "min_events_after_growth " << min_events_after_growth << "\n";
			out << "growth_checks_ok " << (growth_checks_ok ? 1 : 0) << "\n";
			out << "synchronized_states " << synchronized_states << "\n";
			out << "requests " << requests << "\n";
			out << "requests_without_growth " << requests_without_growth << "\n";
			out << "due_false_between_events " << due_false_between_events << "\n";
			out << "noop_checks " << noop_checks << "\n";
			out << "noop_checks_not_due " << noop_checks_not_due << "\n";
			out << "due_without_growth " << due_without_growth << "\n";
			out << "noop_changed " << noop_changed << "\n";
			out << "initial_box " << initial_box.first.x << "," << initial_box.first.y << ","
				<< initial_box.first.z << ":" << initial_box.second.x << "," << initial_box.second.y
				<< "," << initial_box.second.z << "\n";
			out << "final_box " << final_box.first.x << "," << final_box.first.y << ","
				<< final_box.first.z << ":" << final_box.second.x << "," << final_box.second.y
				<< "," << final_box.second.z << "\n";
			out << "max_mass_residual " << max_mass_residual << "\n";
			out << "mass_residual_limit 1e-10\n";
			out << "max_mass_residual_naive " << max_mass_residual_naive << "\n";
			out << "max_mass_residual_box " << max_mass_residual_box << "\n";
			out << "max_energy_residual " << max_energy_residual << "\n";
			out << "energy_residual_limit 1e-2\n";
			out << "max_energy_residual_naive " << max_energy_residual_naive << "\n";
			out << "max_volume_closure_after " << max_volume_closure_after << "\n";
			out << "max_volume_closure_before " << max_volume_closure_before << "\n";
			out << "min_cfl_margin " << min_cfl_margin << "\n";
			out << "min_expected_limit_margin " << min_expected_margin << "\n";
			out << "total_drift_tightened " << total_drift_tightened << "\n";
			out << "gravity " << (gravity ? 1 : 0) << "\n";
			out << "gravity_G " << gravity_G << "\n";
			out << "max_gravity_scaled_error " << max_gravity_scaled_error << "\n";
			out << "gravity_tolerance " << limit_settings.gravity_tolerance << "\n";
			out << "next_events_finite " << (next_events_finite ? 1 : 0) << "\n";
			out << "next_events_reported " << next_events_reported << "\n";
			out << "next_events_cache_ok " << (next_events_cache_ok ? 1 : 0) << "\n";
			out << "fmm_order " << fmm_options.expansionOrder << "\n";
			out << "fmm_theta " << fmm_options.thetaCritical << "\n";
			out << "fmm_leaf " << fmm_options.leafCapacity << "\n";
			out << "stale_cache_control " << (stale_cache_control ? 1 : 0) << "\n";
			out << "stale_control_growths_caught " << stale_control_growths_caught << "\n";
			out << "stale_control_other_failures " << stale_control_other_failures << "\n";
			out << "max_gravity_rms_relative_error " << max_gravity_rms_relative_error << "\n";
			out << "min_gravity_stale_scaled_error " << min_gravity_stale_error << "\n";
			out << "min_gravity_source_margin " << min_gravity_source_margin << "\n";
			out << "max_gravity_new_mass_scaled_effect " << max_gravity_new_mass_effect << "\n";
			out << "max_mass_drift " << max_mass_drift << "\n";
			out << "mass_drift_limit " << mass_drift_limit << "\n";
			out << "max_energy_drift " << max_energy_drift << "\n";
			out << "energy_drift_limit " << energy_drift_limit << "\n";
			out << "energy_drift_checked " << (gravity ? 0 : 1) << "\n";
			out << "final_mass " << final_totals.mass << "\n";
			out << "final_energy " << final_totals.energy << "\n";
			out << "final_momentum_ratio " << momentum_ratio << "\n";
			out << "final_finite " << (final_totals.finite ? 1 : 0) << "\n";
			out << "final_positive " << (final_totals.positive ? 1 : 0) << "\n";
			out << "growth_seconds " << growth_seconds << "\n";
			out << "wall_seconds " << wall_seconds << "\n";
			out << "pass " << (all_pass ? 1 : 0) << "\n";
			out.close();
			std::cout << "RICH_TEST_RESULT pass=" << (all_pass ? 1 : 0) << " growths=" << growths
				<< " events=" << events << " events_after_first_growth=" << events_after_first_growth
				<< " final_cells=" << final_cells << " time=" << simulation.GetTime()
				<< " wall_seconds=" << wall_seconds << std::endl;
		}
#ifdef RICH_MPI
		MPI_Finalize();
#endif
		return all_pass ? 0 : 1;
	}
	catch(UniversalError const& error)
	{
		reportError(error);
		std::cerr << "RICH_TEST_EXCEPTION rank=" << rank << " (UniversalError)" << std::endl;
	}
	catch(std::exception const& error)
	{
		std::cerr << "RICH_TEST_EXCEPTION rank=" << rank << " what=" << error.what() << std::endl;
	}
	if(rank == 0)
	{
		std::ofstream out("individual_box_growth_metrics.txt");
		out << "exception 1\npass 0\n";
	}
#ifdef RICH_MPI
	MPI_Abort(MPI_COMM_WORLD, 2);
#endif
	return 2;
}
