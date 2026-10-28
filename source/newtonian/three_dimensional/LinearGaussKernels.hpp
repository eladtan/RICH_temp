/*! \file LinearGaussKernels.hpp
  \brief LinearGauss3D's slopes, limiter and face values on flat arrays. The only implementation: serial, host-thread
  and device runs all call these functions. The arithmetic follows the original ComputationalCell3D code term by
  term (including which helper it used: ComputationalCellAddMult or ComputationalCellAddMult3), so that host results
  are identical bit for bit.
*/
#ifndef LINEAR_GAUSS_KERNELS_HPP
#define LINEAR_GAUSS_KERNELS_HPP 1

#include <cstddef>
#include <cstdint>
#include "utils/PortableFunction.hpp"
#include "../two_dimensional/computational_cell_2d.hpp"

namespace gauss
{
//! \brief The fields that have slopes, in this order; the tracers follow
enum Field : std::size_t
{
	Density = 0,
	Pressure,
	InternalEnergy,
	VelocityX,
	VelocityY,
	VelocityZ,
	SoundSpeed,
	TimeStep,
	FirstTracer
};

constexpr std::size_t maxFields = FirstTracer + MAX_TRACERS;

//! \brief The first failure of a cell's slope or a face side's value (Ok if none), possibly with the ManyNeighbors warning bit
enum Status : std::uint8_t
{
	Ok = 0,
	UnresolvedNeighbor,
	BadNeighborDensity,
	SingularMatrix,
	BadSlope,
	BadEos,
	BadLimitedSlope,
	BadFace,
	NoSide, // a face side that is not computed: neither of the face's points is a cell
	ManyNeighbors = 0x80
};

RICH_INLINE_FUNCTION std::uint8_t Failure(std::uint8_t status)
{
	return static_cast<std::uint8_t>(status & ~ManyNeighbors);
}

//! \brief The mesh: CSR faces of each cell, and points that are cells (the first cellsNum) or ghosts
struct Mesh
{
	const std::size_t *cellFaceOffsets = nullptr; // cell c's faces: cellFaces[cellFaceOffsets[c]] ... cellFaces[cellFaceOffsets[c + 1] - 1]
	const std::size_t *cellFaces = nullptr;
	const std::size_t *faceNeighbors = nullptr; // 2 per face
	const double *faceCM = nullptr; // 3 per face
	const double *faceArea = nullptr;
	const std::size_t *wallSlot = nullptr; // per face: 1 + its index among the box faces with a ghost, or 0
	const double *meshPoint = nullptr; // 3 per point
	const double *pointCM = nullptr; // 3 per point
	const double *volume = nullptr; // per cell
	const double *width = nullptr; // per cell
	const std::size_t *stateIndex = nullptr; // per point: whose state it has (itself, or a periodic image's cell)
	std::size_t cellsNum = 0;
	std::size_t pointsNum = 0;
};

//! \brief The states of all points
struct State
{
	const double *values = nullptr; // Config::fields per point
	const std::uint8_t *skip = nullptr; // per point: left out of the limiter's bounds (LinearGauss3D's skip_key)
};

struct Config
{
	std::size_t fields = FirstTracer;
	std::size_t tracers = 0;
	std::uint32_t slopedTracers = 0; // bit t: tracer t keeps its slope
	bool limit = true;
	bool principal = false;
	bool pressureCalc = true;
	bool skip = false;
	double shockRatio = 0;
	double diffuseCoeff = 0;
	double pressureRatio = 0;
	std::size_t energyTracer = MAX_TRACERS; // set to the internal energy at faces (without pressureCalc), if < tracers
};

struct Vec3
{
	double x = 0;
	double y = 0;
	double z = 0;
};

RICH_INLINE_FUNCTION Vec3 Load(const double *p)
{
	return Vec3{p[0], p[1], p[2]};
}

RICH_INLINE_FUNCTION double Dot(const Vec3 &a, const Vec3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

//! \brief value + slope . offset, one derivative after the other (as ComputationalCellAddMult)
RICH_INLINE_FUNCTION double InterpolateSequential(double value, double dx, double dy, double dz, double ox, double oy, double oz)
{
	value += dx * ox;
	value += dy * oy;
	value += dz * oz;
	return value;
}

//! \brief value + slope . offset, the three derivatives at once (as ComputationalCellAddMult3)
RICH_INLINE_FUNCTION double InterpolateFused(double value, double dx, double dy, double dz, double ox, double oy, double oz)
{
	value += dx * ox + dy * oy + dz * oz;
	return value;
}

RICH_INLINE_FUNCTION std::size_t OtherPoint(const Mesh &mesh, std::size_t face, std::size_t point)
{
	return mesh.faceNeighbors[2 * face] == point ? mesh.faceNeighbors[2 * face + 1] : mesh.faceNeighbors[2 * face];
}

// An orthonormal frame whose first vector is v (Duff et al. 2017), or false for a vanishing v
RICH_INLINE_FUNCTION bool BuildPrincipalFrame(const Vec3 &velocity, Vec3 &e1, Vec3 &e2, Vec3 &e3)
{
	const double vmag = portable::Sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
	if(vmag < 1e-14)
	{
		return false;
	}
	e1 = Vec3{velocity.x / vmag, velocity.y / vmag, velocity.z / vmag};
	const double sign_z = portable::CopySign(1.0, e1.z);
	const double a = -1.0 / (sign_z + e1.z);
	const double b = e1.x * e1.y * a;
	e2 = Vec3{1.0 + sign_z * e1.x * e1.x * a, sign_z * b, -sign_z * e1.x};
	e3 = Vec3{b, sign_z + e1.y * e1.y * a, -e1.y};
	return true;
}

RICH_INLINE_FUNCTION void ApplyPrincipalLimit(double &x, double &y, double &z, const Vec3 &e1, const Vec3 &e2,
	const Vec3 &e3, double psi1, double psi2, double psi3)
{
	const Vec3 sv{x, y, z};
	const double c1 = Dot(sv, e1);
	const double c2 = Dot(sv, e2);
	const double c3 = Dot(sv, e3);
	x = e1.x * c1 * psi1 + e2.x * c2 * psi2 + e3.x * c3 * psi3;
	y = e1.y * c1 * psi1 + e2.y * c2 * psi2 + e3.y * c3 * psi3;
	z = e1.z * c1 * psi1 + e2.z * c2 * psi2 + e3.z * c3 * psi3;
}

/* One field's limiter at one face: the blend of its Barth-Jespersen and shock bounds. The original code has three
   variants: density, pressure and energy (absThreshold false, shockSkip 1e-9), velocities (true, 1e-9) and tracers
   (true, 0.001). */
RICH_INLINE_FUNCTION double FieldPsi(double dphi, double centroid, double value, double neighbor, double maxdiff,
	double mindiff, double cmax, double cmin, bool absThreshold, double shockSkip, double diffuseCoeff, double w,
	double w1)
{
	const double skipfactor = 1e-3;
	const double threshold = absThreshold ? portable::Abs(1e-9 * value) : 1e-9 * value;
	double psi_bj = 1.0, psi_sh = 1.0;
	if(portable::Abs(dphi) > skipfactor * portable::Max(portable::Abs(maxdiff), portable::Abs(mindiff)) || centroid * value < 0)
	{
		if(dphi > threshold)
		{
			psi_bj = portable::Max(maxdiff / dphi, 0.0);
		}
		else if(dphi < -threshold)
		{
			psi_bj = portable::Max(mindiff / dphi, 0.0);
		}
	}
	if(portable::Abs(dphi) > shockSkip * portable::Max(portable::Abs(cmax), portable::Abs(cmin)) || centroid * value < 0)
	{
		if(portable::Abs(dphi) > threshold)
		{
			psi_sh = portable::Max(diffuseCoeff * (neighbor - value) / dphi, 0.0);
		}
	}
	return w1 * psi_bj + w * psi_sh;
}

// The shock weight of a cell from its unlimited slope (in [0, 1]; 1 in a strong shock)
RICH_INLINE_FUNCTION double ShockWeight(const Mesh &mesh, const State &state, const Config &config, std::size_t cell,
	const double *slope, double soundSpeed)
{
	const std::size_t fields = config.fields;
	const double *c = state.values + cell * fields;
	const double div_v = (slope[VelocityX] + slope[fields + VelocityY] + slope[2 * fields + VelocityZ]) * mesh.width[cell];
	const double t_div = portable::Max(0.0, portable::Min(1.0, -div_v / (config.shockRatio * soundSpeed)));
	double p_ratio = 1.0;
	const double p = c[Pressure];
	for(std::size_t k = mesh.cellFaceOffsets[cell]; k < mesh.cellFaceOffsets[cell + 1]; ++k)
	{
		const double np = state.values[mesh.stateIndex[OtherPoint(mesh, mesh.cellFaces[k], cell)] * fields + Pressure];
		if(p > np)
		{
			p_ratio = portable::Min(p_ratio, np / p);
		}
		else
		{
			p_ratio = portable::Min(p_ratio, p / np);
		}
	}
	const double t_pres = portable::Max(0.0, portable::Min(1.0, (config.pressureRatio - 0.8 * p_ratio) / (0.4 * config.pressureRatio)));
	return portable::Max(t_div, t_pres);
}

// Limits a cell's slope (x, y and z derivatives, `fields` each) in place
RICH_INLINE_FUNCTION void LimitSlope(const Mesh &mesh, const State &state, const Config &config, std::size_t cell,
	double shockWeight, double *slope)
{
	const std::size_t fields = config.fields;
	const std::size_t tracers = config.tracers;
	const double *c = state.values + cell * fields;
	double *sx = slope;
	double *sy = slope + fields;
	double *sz = slope + 2 * fields;
	double *const derivatives[3] = {sx, sy, sz};
	const std::size_t scalars[3] = {Density, Pressure, InternalEnergy};
	const std::size_t begin = mesh.cellFaceOffsets[cell];
	const std::size_t end = mesh.cellFaceOffsets[cell + 1];

	// The bounds: the cell and its neighbors (but those skip_key skips)
	double cmax[maxFields], cmin[maxFields];
	for(std::size_t f = 0; f < fields; ++f)
	{
		cmax[f] = c[f];
		cmin[f] = c[f];
	}
	const Vec3 cellVelocity{c[VelocityX], c[VelocityY], c[VelocityZ]};
	Vec3 e1, e2, e3;
	const bool hasFrame = config.principal && BuildPrincipalFrame(cellVelocity, e1, e2, e3);
	const double cell_v1 = hasFrame ? Dot(cellVelocity, e1) : 0.0;
	const double cell_v2 = hasFrame ? Dot(cellVelocity, e2) : 0.0;
	const double cell_v3 = hasFrame ? Dot(cellVelocity, e3) : 0.0;
	double v1_max = cell_v1, v1_min = cell_v1;
	double v2_max = cell_v2, v2_min = cell_v2;
	double v3_max = cell_v3, v3_min = cell_v3;
	for(std::size_t k = begin; k < end; ++k)
	{
		const std::size_t neighbor = mesh.stateIndex[OtherPoint(mesh, mesh.cellFaces[k], cell)];
		if(config.skip && state.skip[neighbor])
		{
			continue;
		}
		const double *n = state.values + neighbor * fields;
		for(const std::size_t f : scalars)
		{
			cmax[f] = portable::Max(cmax[f], n[f]);
			cmin[f] = portable::Min(cmin[f], n[f]);
		}
		if(hasFrame)
		{
			const Vec3 nv{n[VelocityX], n[VelocityY], n[VelocityZ]};
			const double nv1 = Dot(nv, e1);
			const double nv2 = Dot(nv, e2);
			const double nv3 = Dot(nv, e3);
			v1_max = portable::Max(v1_max, nv1);
			v1_min = portable::Min(v1_min, nv1);
			v2_max = portable::Max(v2_max, nv2);
			v2_min = portable::Min(v2_min, nv2);
			v3_max = portable::Max(v3_max, nv3);
			v3_min = portable::Min(v3_min, nv3);
		}
		else
		{
			for(std::size_t f = VelocityX; f <= VelocityZ; ++f)
			{
				cmax[f] = portable::Max(cmax[f], n[f]);
				cmin[f] = portable::Min(cmin[f], n[f]);
			}
		}
		for(std::size_t j = 0; j < tracers; ++j)
		{
			cmax[FirstTracer + j] = portable::Max(cmax[FirstTracer + j], n[FirstTracer + j]);
			cmin[FirstTracer + j] = portable::Min(cmin[FirstTracer + j], n[FirstTracer + j]);
		}
	}
	double maxdiff[maxFields], mindiff[maxFields];
	for(std::size_t f = 0; f < fields; ++f)
	{
		maxdiff[f] = cmax[f] - c[f];
		mindiff[f] = cmin[f] - c[f];
	}
	const double v1_maxdiff = v1_max - cell_v1, v1_mindiff = v1_min - cell_v1;
	const double v2_maxdiff = v2_max - cell_v2, v2_mindiff = v2_min - cell_v2;
	const double v3_maxdiff = v3_max - cell_v3, v3_mindiff = v3_min - cell_v3;

	// psi: density, pressure, velocity x, y, z, internal energy, then the tracers
	double psi[6 + MAX_TRACERS];
	for(double &p : psi)
	{
		p = 1;
	}
	double psi_v1 = 1.0, psi_v2 = 1.0, psi_v3 = 1.0;
	const double w = shockWeight;
	const double w1 = 1.0 - w;
	const double sf = 1e-9;
	const Vec3 cm = Load(mesh.pointCM + 3 * cell);
	double centroid[maxFields], dphi[maxFields];
	for(std::size_t k = begin; k < end; ++k)
	{
		const std::size_t face = mesh.cellFaces[k];
		const double *n = state.values + mesh.stateIndex[OtherPoint(mesh, face, cell)] * fields;
		const Vec3 target = Load(mesh.faceCM + 3 * face);
		const double ox = target.x - cm.x, oy = target.y - cm.y, oz = target.z - cm.z;
		for(std::size_t f = 0; f < fields; ++f)
		{
			// the first face as interp, the others as interp23Dsimple
			centroid[f] = k == begin ? InterpolateSequential(c[f], sx[f], sy[f], sz[f], ox, oy, oz) :
				InterpolateFused(c[f], sx[f], sy[f], sz[f], ox, oy, oz);
			dphi[f] = centroid[f] - c[f];
		}
		psi[0] = portable::Min(psi[0], FieldPsi(dphi[Density], centroid[Density], c[Density], n[Density], maxdiff[Density],
			mindiff[Density], cmax[Density], cmin[Density], false, sf, config.diffuseCoeff, w, w1));
		psi[1] = portable::Min(psi[1], FieldPsi(dphi[Pressure], centroid[Pressure], c[Pressure], n[Pressure], maxdiff[Pressure],
			mindiff[Pressure], cmax[Pressure], cmin[Pressure], false, sf, config.diffuseCoeff, w, w1));
		psi[5] = portable::Min(psi[5], FieldPsi(dphi[InternalEnergy], centroid[InternalEnergy], c[InternalEnergy],
			n[InternalEnergy], maxdiff[InternalEnergy], mindiff[InternalEnergy], cmax[InternalEnergy], cmin[InternalEnergy],
			false, sf, config.diffuseCoeff, w, w1));
		if(hasFrame)
		{
			const Vec3 iv{centroid[VelocityX], centroid[VelocityY], centroid[VelocityZ]};
			const Vec3 nv{n[VelocityX], n[VelocityY], n[VelocityZ]};
			const double iv1 = Dot(iv, e1), iv2 = Dot(iv, e2), iv3 = Dot(iv, e3);
			psi_v1 = portable::Min(psi_v1, FieldPsi(iv1 - cell_v1, iv1, cell_v1, Dot(nv, e1), v1_maxdiff, v1_mindiff, v1_max,
				v1_min, true, sf, config.diffuseCoeff, w, w1));
			psi_v2 = portable::Min(psi_v2, FieldPsi(iv2 - cell_v2, iv2, cell_v2, Dot(nv, e2), v2_maxdiff, v2_mindiff, v2_max,
				v2_min, true, sf, config.diffuseCoeff, w, w1));
			psi_v3 = portable::Min(psi_v3, FieldPsi(iv3 - cell_v3, iv3, cell_v3, Dot(nv, e3), v3_maxdiff, v3_mindiff, v3_max,
				v3_min, true, sf, config.diffuseCoeff, w, w1));
		}
		else
		{
			for(std::size_t f = VelocityX; f <= VelocityZ; ++f)
			{
				psi[2 + f - VelocityX] = portable::Min(psi[2 + f - VelocityX], FieldPsi(dphi[f], centroid[f], c[f], n[f],
					maxdiff[f], mindiff[f], cmax[f], cmin[f], true, sf, config.diffuseCoeff, w, w1));
			}
		}
		for(std::size_t j = 0; j < tracers; ++j)
		{
			const std::size_t f = FirstTracer + j;
			psi[6 + j] = portable::Min(psi[6 + j], FieldPsi(dphi[f], centroid[f], c[f], n[f], maxdiff[f], mindiff[f],
				cmax[f], cmin[f], true, 0.001, config.diffuseCoeff, w, w1));
		}
	}
	psi[1] = portable::Min(psi[1], psi[5]);
	psi[5] = psi[1];
	if(shockWeight > 0.85)
	{
		const double psi_scalar_min = portable::Min(psi[1], psi[5]);
		psi[0] = portable::Min(psi[0], psi_scalar_min);
		psi[2] = portable::Min(psi[2], psi_scalar_min);
		psi[3] = portable::Min(psi[3], psi_scalar_min);
		psi[4] = portable::Min(psi[4], psi_scalar_min);
		if(hasFrame)
		{
			psi_v1 = portable::Min(psi_v1, psi_scalar_min);
			psi_v2 = portable::Min(psi_v2, psi_scalar_min);
			psi_v3 = portable::Min(psi_v3, psi_scalar_min);
		}
	}
	for(double *s : derivatives)
	{
		s[Density] *= psi[0];
		s[Pressure] *= psi[1];
		if(hasFrame)
		{
			ApplyPrincipalLimit(s[VelocityX], s[VelocityY], s[VelocityZ], e1, e2, e3, psi_v1, psi_v2, psi_v3);
		}
		else
		{
			s[VelocityX] *= psi[2];
			s[VelocityY] *= psi[3];
			s[VelocityZ] *= psi[4];
		}
		s[InternalEnergy] *= psi[5];
		for(std::size_t j = 0; j < tracers; ++j)
		{
			s[FirstTracer + j] *= psi[6 + j];
		}
	}
	// In strong shocks, caps the velocity's variation across the cell at 10 times its velocity
	if(shockWeight > 0.9)
	{
		const Vec3 dvx{sx[VelocityX], sx[VelocityY], sx[VelocityZ]};
		const Vec3 dvy{sy[VelocityX], sy[VelocityY], sy[VelocityZ]};
		const Vec3 dvz{sz[VelocityX], sz[VelocityY], sz[VelocityZ]};
		double maxDv = Dot(dvx, dvx) + Dot(dvy, dvy) + Dot(dvz, dvz);
		maxDv *= mesh.width[cell] * mesh.width[cell];
		if(maxDv > 100 * Dot(cellVelocity, cellVelocity))
		{
			const double sfactor = portable::Sqrt(100 * Dot(cellVelocity, cellVelocity) / maxDv);
			for(double *s : derivatives)
			{
				s[VelocityX] *= sfactor;
				s[VelocityY] *= sfactor;
				s[VelocityZ] *= sfactor;
			}
		}
	}
}

/*! \brief A cell's slope: the Green-Gauss gradient from its centroid, then (with Config::limit) the limiter
  \param slope Out: the x, y and z derivatives, Config::fields each
  \param naive Out: the same before the limiter (and before unsloped tracers are zeroed)
  \param soundSpeed The cell's sound speed (for the limiter), and whether the equation of state failed to give it
  \param matrix Out, if not null: the normalized 3x3 matrix and its determinant (for error reports)
*/
RICH_INLINE_FUNCTION std::uint8_t CellSlope(const Mesh &mesh, const State &state, const Config &config, std::size_t cell,
	double soundSpeed, bool soundSpeedFailed, double *slope, double *naive, double *matrix = nullptr)
{
	const std::size_t fields = config.fields;
	const std::size_t begin = mesh.cellFaceOffsets[cell];
	const std::size_t end = mesh.cellFaceOffsets[cell + 1];
	for(std::size_t k = begin; k < end; ++k)
	{
		const std::size_t neighbor = mesh.stateIndex[OtherPoint(mesh, mesh.cellFaces[k], cell)];
		if(neighbor >= mesh.pointsNum)
		{
			return UnresolvedNeighbor;
		}
		if(not portable::IsFinite(state.values[neighbor * fields + Density]))
		{
			return BadNeighborDensity;
		}
	}
	const std::uint8_t warning = end - begin > 100 ? ManyNeighbors : Ok;

	// The Green-Gauss gradient, corrected for the offset of the cell's centroid
	const double *c = state.values + cell * fields;
	const Vec3 center = Load(mesh.meshPoint + 3 * cell);
	const Vec3 cellCM = Load(mesh.pointCM + 3 * cell);
	const double volume = mesh.volume[cell];
	double m[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	double t[3][maxFields];
	for(std::size_t k = begin; k < end; ++k)
	{
		const std::size_t face = mesh.cellFaces[k];
		const std::size_t other = OtherPoint(mesh, face, cell);
		const double *n = state.values + mesh.stateIndex[other] * fields;
		const Vec3 neighborCM = Load(mesh.pointCM + 3 * other);
		const Vec3 faceCM = Load(mesh.faceCM + 3 * face);
		Vec3 c_ij = neighborCM;
		c_ij.x += cellCM.x;
		c_ij.y += cellCM.y;
		c_ij.z += cellCM.z;
		c_ij.x *= -0.5;
		c_ij.y *= -0.5;
		c_ij.z *= -0.5;
		c_ij.x += faceCM.x;
		c_ij.y += faceCM.y;
		c_ij.z += faceCM.z;
		const Vec3 neighborCenter = Load(mesh.meshPoint + 3 * other);
		const Vec3 d{neighborCenter.x - center.x, neighborCenter.y - center.y, neighborCenter.z - center.z};
		const double l = portable::Sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
		const Vec3 r_ij{d.x / l, d.y / l, d.z / l};
		const double A = mesh.faceArea[face];
		m[0] -= c_ij.x * r_ij.x * A;
		m[1] -= c_ij.y * r_ij.x * A;
		m[2] -= c_ij.z * r_ij.x * A;
		m[3] -= c_ij.x * r_ij.y * A;
		m[4] -= c_ij.y * r_ij.y * A;
		m[5] -= c_ij.z * r_ij.y * A;
		m[6] -= c_ij.x * r_ij.z * A;
		m[7] -= c_ij.y * r_ij.z * A;
		m[8] -= c_ij.z * r_ij.z * A;
		const double s[3] = {r_ij.x * A, r_ij.y * A, r_ij.z * A};
		for(std::size_t dim = 0; dim < 3; ++dim)
		{
			for(std::size_t f = 0; f < fields; ++f)
			{
				// the first neighbor scales as ComputationalCell3D::operator*=, which leaves the time step as is
				if(k == begin)
				{
					t[dim][f] = f == TimeStep ? n[f] : n[f] * s[dim];
				}
				else
				{
					t[dim][f] += n[f] * s[dim];
				}
				t[dim][f] += c[f] * s[dim];
			}
		}
	}
	const double v_inv = 1.0 / volume;
	for(double &entry : m)
	{
		entry *= v_inv;
	}
	m[0] += 1;
	m[4] += 1;
	m[8] += 1;
	const double det = -m[2] * m[4] * m[6] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7] - m[0] * m[5] * m[7] -
		m[1] * m[3] * m[8] + m[0] * m[4] * m[8];
	if(matrix != nullptr)
	{
		for(std::size_t i = 0; i < 9; ++i)
		{
			matrix[i] = m[i];
		}
		matrix[9] = det;
	}
	if(portable::Abs(det) < 1e-10)
	{
		return static_cast<std::uint8_t>(SingularMatrix | warning);
	}
	double m_inv[9];
	m_inv[0] = m[4] * m[8] - m[5] * m[7];
	m_inv[1] = m[2] * m[7] - m[1] * m[8];
	m_inv[2] = m[1] * m[5] - m[2] * m[4];
	m_inv[3] = m[5] * m[6] - m[3] * m[8];
	m_inv[4] = m[0] * m[8] - m[2] * m[6];
	m_inv[5] = m[2] * m[3] - m[5] * m[0];
	m_inv[6] = m[3] * m[7] - m[6] * m[4];
	m_inv[7] = m[6] * m[1] - m[0] * m[7];
	m_inv[8] = m[4] * m[0] - m[1] * m[3];
	for(double &entry : m_inv)
	{
		entry /= (2 * volume * det);
	}
	for(std::size_t dim = 0; dim < 3; ++dim)
	{
		double *out = slope + dim * fields;
		for(std::size_t f = 0; f < fields; ++f)
		{
			out[f] = f == TimeStep ? t[0][f] : t[0][f] * m_inv[3 * dim];
			out[f] += t[1][f] * m_inv[3 * dim + 1];
			out[f] += t[2][f] * m_inv[3 * dim + 2];
		}
	}
	if(not portable::IsFinite(slope[Density]))
	{
		return static_cast<std::uint8_t>(BadSlope | warning);
	}
	for(std::size_t i = 0; i < 3 * fields; ++i)
	{
		naive[i] = slope[i];
	}
	for(std::size_t j = 0; j < config.tracers; ++j)
	{
		if(not (config.slopedTracers & (std::uint32_t(1) << j)))
		{
			for(std::size_t dim = 0; dim < 3; ++dim)
			{
				slope[dim * fields + FirstTracer + j] = 0;
			}
		}
	}
	if(config.limit)
	{
		if(soundSpeedFailed)
		{
			return static_cast<std::uint8_t>(BadEos | warning);
		}
		LimitSlope(mesh, state, config, cell, ShockWeight(mesh, state, config, cell, slope, soundSpeed), slope);
		if(not portable::IsFinite(slope[Density]))
		{
			return static_cast<std::uint8_t>(BadLimitedSlope | warning);
		}
	}
	return warning;
}

/*! \brief The value of one side of a face: its point's state and slope at the face's centroid (the equation of state
  comes after). A ghost has the slope of the point whose state it has, or (at the box) the wall's slope (zero if null).
  \param out The side's fields
*/
RICH_INLINE_FUNCTION std::uint8_t FaceSide(const Mesh &mesh, const State &state, const Config &config,
	const double *slopes, const double *wallSlopes, std::size_t face, std::size_t side, double *out)
{
	const std::size_t fields = config.fields;
	const std::size_t point = mesh.faceNeighbors[2 * face + side];
	if(point >= mesh.cellsNum and mesh.faceNeighbors[2 * face + 1 - side] >= mesh.cellsNum)
	{
		return NoSide;
	}
	const std::size_t source = point < mesh.cellsNum ? point : mesh.stateIndex[point];
	const double *base = state.values + source * fields;
	const double *slope = slopes + source * 3 * fields;
	if(point >= mesh.cellsNum and mesh.wallSlot[face] != 0)
	{
		slope = wallSlopes == nullptr ? nullptr : wallSlopes + (mesh.wallSlot[face] - 1) * 3 * fields;
	}
	const Vec3 cm = Load(mesh.pointCM + 3 * point);
	const Vec3 target = Load(mesh.faceCM + 3 * face);
	const double ox = target.x - cm.x, oy = target.y - cm.y, oz = target.z - cm.z;
	for(std::size_t f = 0; f < fields; ++f)
	{
		const double dx = slope == nullptr ? 0.0 : slope[f];
		const double dy = slope == nullptr ? 0.0 : slope[fields + f];
		const double dz = slope == nullptr ? 0.0 : slope[2 * fields + f];
		out[f] = InterpolateFused(base[f], dx, dy, dz, ox, oy, oz);
	}
	return Ok;
}

/*! \brief A face side after the equation of state: the energy tracer (without Config::pressureCalc), then the
  check that the value is physical
*/
RICH_INLINE_FUNCTION std::uint8_t CheckFaceSide(const Config &config, double *out)
{
	if(not config.pressureCalc and config.energyTracer < config.tracers)
	{
		out[FirstTracer + config.energyTracer] = out[InternalEnergy];
	}
	if((!(out[Density] > 0)) || (!(out[InternalEnergy] > 0)) || (!portable::IsFinite(out[VelocityX])) ||
		(!portable::IsFinite(out[VelocityY])) || (!portable::IsFinite(out[VelocityZ])))
	{
		return BadFace;
	}
	return Ok;
}

//! \brief Computes each cell's slope
struct CellSlopes
{
	static constexpr bool onDevice = true;
	Mesh mesh;
	State state;
	Config config;
	const double *soundSpeeds = nullptr; // per cell (with Config::limit)
	const std::uint8_t *soundSpeedStatus = nullptr; // per cell (with Config::limit): not 0 where the equation of state failed
	double *slopes = nullptr; // 3 * fields per point (the cells' are written)
	double *naive = nullptr; // 3 * fields per cell
	std::uint8_t *status = nullptr; // per cell

	RICH_INLINE_FUNCTION void operator()(std::size_t cell) const
	{
		const std::size_t size = 3 * this->config.fields;
		const double soundSpeed = this->config.limit ? this->soundSpeeds[cell] : 0.0;
		const bool soundSpeedFailed = this->config.limit and this->soundSpeedStatus[cell] != 0;
		this->status[cell] = CellSlope(this->mesh, this->state, this->config, cell, soundSpeed, soundSpeedFailed,
			this->slopes + cell * size, this->naive + cell * size);
	}
};

//! \brief Interpolates both sides of each face
struct FaceSides
{
	static constexpr bool onDevice = true;
	Mesh mesh;
	State state;
	Config config;
	const double *slopes = nullptr; // 3 * fields per point, ghosts included
	const double *wallSlopes = nullptr; // 3 * fields per box face (Mesh::wallSlot), or null for zero
	double *values = nullptr; // fields per face side
	std::uint8_t *status = nullptr; // per face side

	RICH_INLINE_FUNCTION void operator()(std::size_t face) const
	{
		for(std::size_t side = 0; side < 2; ++side)
		{
			this->status[2 * face + side] = FaceSide(this->mesh, this->state, this->config, this->slopes, this->wallSlopes,
				face, side, this->values + (2 * face + side) * this->config.fields);
		}
	}
};

//! \brief Checks each face side that the equation of state did not fail
struct FaceChecks
{
	static constexpr bool onDevice = true;
	Config config;
	double *values = nullptr; // fields per face side
	std::uint8_t *status = nullptr; // per face side

	RICH_INLINE_FUNCTION void operator()(std::size_t side) const
	{
		if(this->status[side] == Ok)
		{
			this->status[side] = CheckFaceSide(this->config, this->values + side * this->config.fields);
		}
	}
};
} // namespace gauss

#endif // LINEAR_GAUSS_KERNELS_HPP
