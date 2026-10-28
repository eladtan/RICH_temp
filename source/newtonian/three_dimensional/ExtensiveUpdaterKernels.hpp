/*! \file ExtensiveUpdaterKernels.hpp
  \brief ConditionExtensiveUpdater3D's update of the extensives on flat arrays: the only implementation, used on the
  host, the host's threads and a device. With threads or a device, each cell gathers its faces in increasing face index,
  which is the order the original loop over faces added them; one thread runs that loop over faces itself, with the
  same functions. The arithmetic follows Conserved3D's term by term, so that results are identical bit for bit.
*/
#ifndef EXTENSIVE_UPDATER_KERNELS_HPP
#define EXTENSIVE_UPDATER_KERNELS_HPP 1

#include <cstddef>
#include <cstdint>
#include "utils/PortableFunction.hpp"
#include "computational_cell.hpp"

namespace extensive
{
//! \brief The fields of a Conserved3D, in this order; the energy groups and the tracers follow
enum Field : std::size_t
{
	Mass = 0,
	MomentumX,
	MomentumY,
	MomentumZ,
	Energy,
	InternalEnergy,
	Erad,
	EradDt,
	EradDtDt,
	FirstGroup
};

constexpr std::size_t FirstTracer = FirstGroup + ENERGY_GROUPS_NUM;
constexpr std::size_t fields = FirstTracer + MAX_TRACERS;

//! \brief A cell's check after its update (as the original loop's, in the same order)
enum Status : std::uint8_t
{
	Ok = 0,
	NegativeGroup,
	BadCell
};

//! \brief A face's change of its neighbors' extensives, in place of its flux: delta = flux * dt * area (no internal energy)
RICH_INLINE_FUNCTION void FaceDelta(double *flux, double dt, double area)
{
	for(std::size_t k = 0; k < fields; ++k)
	{
		flux[k] = flux[k] * dt * area;
	}
	flux[InternalEnergy] = 0;
}

/*! \brief One face's update of a cell's extensives e: e -= delta if the cell is the face's first neighbor, e += delta
  if its second, then the internal energy's correction by the cell's velocity v. Adding -delta is subtracting delta
  exactly (IEEE 754 defines x - y as x + (-y)).
*/
RICH_INLINE_FUNCTION void ApplyFace(double *e, const double *delta, double vx, double vy, double vz, bool first)
{
	const double correction = delta[Energy] - (vx * delta[MomentumX] + vy * delta[MomentumY] + vz * delta[MomentumZ]) +
		0.5 * (vx * vx + vy * vy + vz * vz) * delta[Mass];
	const double sign = first ? -1.0 : 1.0;
	for(std::size_t k = 0; k < fields; ++k)
	{
		e[k] += sign * delta[k];
	}
	e[InternalEnergy] += sign * correction;
}

//! \brief The internal energy as the total minus the kinetic
RICH_INLINE_FUNCTION void FixEnergy(double *e)
{
	const double Eknew = 0.5 * (e[MomentumX] * e[MomentumX] + e[MomentumY] * e[MomentumY] + e[MomentumZ] * e[MomentumZ]) /
		e[Mass];
	e[InternalEnergy] = e[Energy] - Eknew;
}

//! \brief Whether an updated cell is physical (with an "Entropy" tracer, its internal energy may be negative)
RICH_INLINE_FUNCTION std::uint8_t CheckCell(const double *e, bool entropy, std::size_t entropyTracer)
{
	for(std::size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
	{
		if(e[FirstGroup + j] < 0 && e[Mass] > 0)
		{
			return NegativeGroup;
		}
	}
	const double momentum = portable::Sqrt(e[MomentumX] * e[MomentumX] + e[MomentumY] * e[MomentumY] +
		e[MomentumZ] * e[MomentumZ]);
	if(!(e[Mass] > 0) || !(e[Energy] > 0) || (!(e[InternalEnergy] > 0) && (!entropy)) || (!portable::IsFinite(momentum)) ||
		(entropy && e[FirstTracer + entropyTracer] < 0))
	{
		return BadCell;
	}
	return Ok;
}

//! \brief A cell's end of the update: fixes its internal energy and (with check) checks it
RICH_INLINE_FUNCTION std::uint8_t FinishCell(double *e, bool check, bool entropy, std::size_t entropyTracer)
{
	FixEnergy(e);
	return check ? CheckCell(e, entropy, entropyTracer) : static_cast<std::uint8_t>(Ok);
}

//! \brief Updates each cell's extensives by its faces' fluxes, then FinishCell
struct CellUpdates
{
	static constexpr bool onDevice = true;
	const std::size_t *faceOffsets = nullptr; // cell c's faces: entries faceOffsets[c] ... faceOffsets[c + 1] - 1
	const std::size_t *faces = nullptr; // per entry: the face
	const std::uint8_t *first = nullptr; // per entry: whether the cell is the face's first neighbor
	const double *deltas = nullptr; // `fields` per face (FaceDelta)
	const double *velocities = nullptr; // 3 per cell
	double *extensives = nullptr; // `fields` per cell
	std::uint8_t *status = nullptr; // per cell
	bool check = true;
	bool entropy = false;
	std::size_t entropyTracer = 0;

	RICH_INLINE_FUNCTION void operator()(std::size_t cell) const
	{
		// the cell's extensives stay local while its faces are added
		double *out = this->extensives + cell * fields;
		double e[fields];
		for(std::size_t k = 0; k < fields; ++k)
		{
			e[k] = out[k];
		}
		const double *v = this->velocities + 3 * cell;
		for(std::size_t k = this->faceOffsets[cell]; k < this->faceOffsets[cell + 1]; ++k)
		{
			const std::size_t face = this->faces[k];
			ApplyFace(e, this->deltas + face * fields, v[0], v[1], v[2], this->first[k] != 0);
		}
		this->status[cell] = FinishCell(e, this->check, this->entropy, this->entropyTracer);
		for(std::size_t k = 0; k < fields; ++k)
		{
			out[k] = e[k];
		}
	}
};
} // namespace extensive

#endif // EXTENSIVE_UPDATER_KERNELS_HPP
