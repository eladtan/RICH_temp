#include "ConditionActionFlux1.hpp"
#include "Hllc3D.hpp"
#include "HllcKernels.hpp"
#include "LinearGauss3D.hpp"
#include "LinearGaussArrays.hpp"
#include "utils/Portable.hpp"

// The flat arrays of the flux kernel, kept between calls
struct ConditionActionFluxArrays
{
	portable::Buffer<double> normals, faceVelocities, soundSpeeds, fluxes;
	portable::Buffer<std::uint8_t> action, aux, sideStatus, status;
	std::vector<std::uint8_t> actionKinds; // per action in the sequence: what the kernel does with it (FaceKind)
};

ConditionActionFlux1::ConditionActionFlux1(const vector<pair<const Condition3D*, const Action3D*> >& sequence,
	SpatialReconstruction3D const& interp) :
	sequence_(sequence),interp_(interp) {}

ConditionActionFlux1::~ConditionActionFlux1(void)
{}

namespace
{
	void choose_action(size_t face, const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const EquationOfState& eos, const Vector3D& face_velocity,
		const vector<pair<const ConditionActionFlux1::Condition3D*, const ConditionActionFlux1::Action3D*> >& sequence,
		Conserved3D &res, double time, std::pair<ComputationalCell3D, ComputationalCell3D>
		const& face_values)
	{
		for (size_t i = 0; i < sequence.size(); ++i)
		{
			const pair<bool, bool> flag_aux = (*sequence[i].first)
				(face, tess, cells);
#ifdef RICH_DEBUG
			try
			{
#endif
				if (flag_aux.first)
					return (*sequence[i].second)
					(face, tess, face_velocity, cells, eos, flag_aux.second, res, time, face_values);
#ifdef RICH_DEBUG
			}
			catch (UniversalError &eo)
			{
				eo.addEntry("Face number", face);
				size_t n0 = tess.GetFaceNeighbors(face).first;
				size_t n1 = tess.GetFaceNeighbors(face).second;
				eo.addEntry("Left cell density", cells[n0].density);
				eo.addEntry("Left cell pressure", cells[n0].pressure);
				eo.addEntry("Left cell Vx", cells[n0].velocity.x);
				eo.addEntry("Left cell Vy", cells[n0].velocity.y);
				eo.addEntry("Left cell Vz", cells[n0].velocity.z);
				eo.addEntry("Left cell internal energy", cells[n0].internal_energy);
				eo.addEntry("Left cell ID", cells[n0].ID);
				eo.addEntry("Right cell density", cells[n1].density);
				eo.addEntry("Right cell pressure", cells[n1].pressure);
				eo.addEntry("Right cell Vx", cells[n1].velocity.x);
				eo.addEntry("Right cell Vy", cells[n1].velocity.y);
				eo.addEntry("Right cell Vz", cells[n1].velocity.z);
				eo.addEntry("Right cell internal energy", cells[n1].internal_energy);
				eo.addEntry("Right cell ID", cells[n1].ID);
				throw;
			}
#endif
		}
		throw UniversalError("Error in ConditionActionFlux1");
	}
}

namespace
{
	// What the kernel does with a face's action: nothing (the face is computed on the host), RegularFlux3D or RigidWallFlux3D
	enum FaceKind : std::uint8_t
	{
		HostFace = 0,
		BulkFace,
		WallFace
	};

	constexpr std::size_t maxKernelActions = 8;
	constexpr std::uint8_t noAction = 255;

	// Each face's flux by HLLC, as RegularFlux3D and RigidWallFlux3D with Hllc3D compute it
	struct FaceFluxes
	{
		static constexpr bool onDevice = true;
		const double *values = nullptr; // LinearGauss3D's face sides, `fields` each
		std::size_t fields = 0;
		const std::uint8_t *sideStatus = nullptr; // per side: not 0 if the side or its sound speed is missing
		const double *soundSpeeds = nullptr; // per side
		const double *normals = nullptr; // 3 per face
		const double *faceVelocities = nullptr; // 3 per face
		const std::uint8_t *action = nullptr; // per face: its index in the sequence
		const std::uint8_t *aux = nullptr; // per face: the condition's auxiliary flag
		std::uint8_t kinds[maxKernelActions] = {};
		double gammas[maxKernelActions] = {};
		double *fluxes = nullptr; // 5 per face: mass, momentum, energy
		std::uint8_t *status = nullptr; // per face: 0 if computed here

		RICH_INLINE_FUNCTION hllc::Primitive Side(std::size_t face, std::size_t side) const
		{
			const double *v = this->values + (2 * face + side) * this->fields;
			return hllc::Primitive{v[gauss::Density], v[gauss::Pressure], v[gauss::InternalEnergy], v[gauss::VelocityX],
				v[gauss::VelocityY], v[gauss::VelocityZ]};
		}

		RICH_INLINE_FUNCTION void operator()(std::size_t face) const
		{
			this->status[face] = 1;
			const std::uint8_t a = this->action[face];
			if(a >= maxKernelActions or this->kinds[a] == HostFace)
			{
				return;
			}
			const double nx = this->normals[3 * face], ny = this->normals[3 * face + 1], nz = this->normals[3 * face + 2];
			hllc::Primitive left, right;
			double cl = 0, cr = 0;
			if(this->kinds[a] == WallFace)
			{
				// the face's own side on both sides of the wall, mirrored on the ghost's
				const std::size_t own = this->aux[face] ? 0 : 1;
				if(this->sideStatus[2 * face + own] != 0)
				{
					return;
				}
				hllc::Primitive mirror = this->Side(face, own);
				hllc::Reflect(mirror.vx, mirror.vy, mirror.vz, nx, ny, nz);
				left = own == 0 ? this->Side(face, 0) : mirror;
				right = own == 0 ? mirror : this->Side(face, 1);
				cl = this->soundSpeeds[2 * face + own];
				cr = cl;
			}
			else
			{
				if(this->sideStatus[2 * face] != 0 or this->sideStatus[2 * face + 1] != 0)
				{
					return;
				}
				left = this->Side(face, 0);
				right = this->Side(face, 1);
				cl = this->soundSpeeds[2 * face];
				cr = this->soundSpeeds[2 * face + 1];
			}
			const double velocity = hllc::NormalVelocity(nx, ny, nz, this->faceVelocities[3 * face],
				this->faceVelocities[3 * face + 1], this->faceVelocities[3 * face + 2]);
			hllc::Flux flux;
			hllc::Primitive local_left, local_right;
			hllc::WaveSpeeds ws;
			double frame_velocity = 0;
			if(hllc::Solve(left, right, velocity, nx, ny, nz, cl, cr, this->gammas[a], flux, local_left, local_right, ws,
				frame_velocity) != hllc::Ok)
			{
				return;
			}
			double *out = this->fluxes + 5 * face;
			out[0] = flux.mass;
			out[1] = flux.mx;
			out[2] = flux.my;
			out[3] = flux.mz;
			out[4] = flux.energy;
			this->status[face] = 0;
		}
	};
}

const ConditionActionFluxArrays *ConditionActionFlux1::KernelFluxes(const Tessellation3D& tess,
	const vector<Vector3D>& face_velocities, const vector<ComputationalCell3D>& cells, const EquationOfState& eos) const
{
	const LinearGauss3D *gauss = dynamic_cast<const LinearGauss3D*>(&interp_);
	LinearGaussArrays *sides = gauss != nullptr ? gauss->GetArrays() : nullptr;
	const size_t faces = tess.GetTotalFacesNumber();
	if(sides == nullptr or sides->sr or sides->facesNum != faces or face_velocities.size() < faces)
	{
		return nullptr;
	}
	// The actions the kernel computes
	FaceFluxes kernel;
	bool any = false;
	for(size_t i = 0; i < sequence_.size() and i < maxKernelActions; ++i)
	{
		const RiemannSolver3D *rs = nullptr;
		if(const RegularFlux3D *regular = dynamic_cast<const RegularFlux3D*>(sequence_[i].second))
		{
			rs = &regular->GetRiemannSolver();
			kernel.kinds[i] = BulkFace;
		}
		else if(const RigidWallFlux3D *wall = dynamic_cast<const RigidWallFlux3D*>(sequence_[i].second))
		{
			rs = &wall->GetRiemannSolver();
			kernel.kinds[i] = WallFace;
		}
		const Hllc3D *hllc = dynamic_cast<const Hllc3D*>(rs);
		if(hllc == nullptr)
		{
			kernel.kinds[i] = HostFace;
			continue;
		}
		kernel.gammas[i] = hllc->GetIterationFactor();
		any = true;
	}
	if(not any)
	{
		return nullptr;
	}
	if(not arrays_)
	{
		arrays_ = std::make_shared<ConditionActionFluxArrays>();
	}
	ConditionActionFluxArrays &data = *arrays_;

	data.actionKinds.assign(kernel.kinds, kernel.kinds + maxKernelActions);

	// Each face's action, as choose_action finds it; on the host's threads if the conditions only read the
	// tessellation (IsBoundaryFace3D, IsBulkFace3D), otherwise one face after another
	vector<std::uint8_t> &action = data.action.Host();
	vector<std::uint8_t> &aux = data.aux.Host();
	action.assign(faces, noAction);
	aux.assign(faces, 0);
	const auto findAction = [&](size_t face)
	{
		for(size_t i = 0; i < sequence_.size(); ++i)
		{
			const pair<bool, bool> flag_aux = (*sequence_[i].first)(face, tess, cells);
			if(flag_aux.first)
			{
				action[face] = i < maxKernelActions ? static_cast<std::uint8_t>(i) : noAction;
				aux[face] = flag_aux.second ? 1 : 0;
				break;
			}
		}
	};
	bool parallelConditions = true;
	for(const pair<const Condition3D*, const Action3D*> &step : sequence_)
	{
		if(dynamic_cast<const IsBoundaryFace3D*>(step.first) == nullptr and dynamic_cast<const IsBulkFace3D*>(step.first) == nullptr)
		{
			parallelConditions = false;
		}
	}
	if(parallelConditions)
	{
		portable::HostParallelFor(faces, findAction);
	}
	else
	{
		for(size_t face = 0; face < faces; ++face)
		{
			findAction(face);
		}
	}
	vector<double> &normals = data.normals.Host();
	vector<double> &velocities = data.faceVelocities.Host();
	normals.resize(3 * faces);
	velocities.resize(3 * faces);
	portable::HostParallelFor(faces, [&](size_t face)
	{
		const Vector3D normal = normalize(tess.Normal(face));
		normals[3 * face] = normal.x;
		normals[3 * face + 1] = normal.y;
		normals[3 * face + 2] = normal.z;
		velocities[3 * face] = face_velocities[face].x;
		velocities[3 * face + 1] = face_velocities[face].y;
		velocities[3 * face + 2] = face_velocities[face].z;
	});

	// The sides' sound speeds (a side LinearGauss3D did not compute stays missing)
	const portable::Where where = sides->faceValuesWhere == portable::Where::Device and eos.RunsOnDevice() ?
		portable::Where::Device : portable::Where::Host;
	data.sideStatus.Host() = sides->faceStatus.Host();
	data.soundSpeeds.Host().resize(2 * faces);
	data.fluxes.Host().resize(5 * faces);
	data.status.Host().resize(faces);
	if(where == portable::Where::Device)
	{
		data.action.ToDevice();
		data.aux.ToDevice();
		data.normals.ToDevice();
		data.faceVelocities.ToDevice();
		data.sideStatus.ToDevice();
		data.soundSpeeds.AllocateDevice();
		data.fluxes.AllocateDevice();
		data.status.AllocateDevice();
	}
	const size_t fields = sides->config.fields;
	double *values = sides->faceValues.Data(where);
	EosArrays arrays;
	arrays.n = 2 * faces;
	arrays.stride = fields;
	arrays.density = values + gauss::Density;
	arrays.second = values + gauss::Pressure;
	arrays.tracers = sides->config.tracers > 0 ? values + gauss::FirstTracer : nullptr;
	arrays.tracersNum = sides->config.tracers;
	arrays.tracerNames = &ComputationalCell3D::tracerNames;
	arrays.result = data.soundSpeeds.Data(where);
	arrays.status = data.sideStatus.Data(where);
	eos.dp2cArrays(arrays, where);

	kernel.values = values;
	kernel.fields = fields;
	kernel.sideStatus = data.sideStatus.Data(where);
	kernel.soundSpeeds = data.soundSpeeds.Data(where);
	kernel.normals = data.normals.Data(where);
	kernel.faceVelocities = data.faceVelocities.Data(where);
	kernel.action = data.action.Data(where);
	kernel.aux = data.aux.Data(where);
	kernel.fluxes = data.fluxes.Data(where);
	kernel.status = data.status.Data(where);
	portable::ParallelFor(where, "ConditionActionFlux1::FaceFluxes", faces, kernel);
	if(where == portable::Where::Device)
	{
		data.fluxes.ToHost();
		data.status.ToHost();
	}
	return &data;
}

std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > ConditionActionFlux1::operator()(vector<Conserved3D> &fluxes, const Tessellation3D& tess, const vector<Vector3D>& face_velocities,
													   const vector<ComputationalCell3D>& cells, const vector<Conserved3D>& /*extensives*/, const EquationOfState& eos,
	const double time, const double /*dt*/) const
{
	vector<std::pair<ComputationalCell3D, ComputationalCell3D> > face_values;
	this->Calculate(fluxes, tess, face_velocities, cells, vector<Conserved3D>(), eos, time, 0, face_values);
	return face_values;
}

void ConditionActionFlux1::Calculate(vector<Conserved3D> &fluxes, const Tessellation3D& tess, const vector<Vector3D>& face_velocities,
													   const vector<ComputationalCell3D>& cells, const vector<Conserved3D>& /*extensives*/, const EquationOfState& eos,
	const double time, const double /*dt*/, std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values) const
{
	for (size_t i = 0; i < sequence_.size(); ++i)
		sequence_[i].second->Reset();
	// LinearGauss3D rewrites every face value, so the vector keeps its elements between steps
	if (dynamic_cast<const LinearGauss3D*>(&interp_) == nullptr)
		face_values.clear();
	interp_(tess, cells, time, face_values);
	fluxes.resize(tess.GetTotalFacesNumber());
	size_t Nloop = fluxes.size();
	const ConditionActionFluxArrays *flat = this->KernelFluxes(tess, face_velocities, cells, eos);
	/* The faces the kernel computed get their fluxes here, on the host's threads (they cannot throw). The loop
	   below then goes through the others in order: bad inputs (thrown) and faces left to the actions. */
	enum : std::uint8_t { KernelDone = 0, BadInput, HostAction };
	vector<std::uint8_t> remaining(Nloop);
	portable::HostParallelFor(Nloop, [&](size_t i)
	{
		if (face_values[i].first.density < 0 ||  face_values[i].first.internal_energy < 0
			|| face_values[i].second.density < 0  || face_values[i].second.internal_energy < 0)
		{
			remaining[i] = BadInput;
			return;
		}
		if (flat == nullptr || flat->status.Host()[i] != 0)
		{
			remaining[i] = HostAction;
			return;
		}
		remaining[i] = KernelDone;
		// a wall's two sides advect the face's own side
		const double *f = flat->fluxes.Host().data() + 5 * i;
		Conserved3D &res = fluxes[i];
		res = Conserved3D();
		res.mass = f[0];
		res.momentum = Vector3D(f[1], f[2], f[3]);
		res.energy = f[4];
		res.internal_energy = 0;
		if (flat->actionKinds[flat->action.Host()[i]] == WallFace)
		{
			const ComputationalCell3D &own = flat->aux.Host()[i] ? face_values[i].first : face_values[i].second;
			AddTracers3D(own, own, res);
		}
		else
			AddTracers3D(face_values[i].first, face_values[i].second, res);
	});
	for (size_t i = 0; i < Nloop; ++i)
	{
		if (remaining[i] == KernelDone)
			continue;
		if (remaining[i] == BadInput)
		{
			UniversalError eo("Bad input to flux calculator");
			eo.addEntry("Face", static_cast<double>(i));
			eo.addEntry("Face neigh 0", static_cast<double>(tess.GetFaceNeighbors(i).first));
			eo.addEntry("Face neigh 1", static_cast<double>(tess.GetFaceNeighbors(i).second));
			eo.addEntry("First input Density", face_values[i].first.density);
			eo.addEntry("First input pressure", face_values[i].first.pressure);
			eo.addEntry("First input internal energy", face_values[i].first.internal_energy);
			eo.addEntry("First input vx", face_values[i].first.velocity.x);
			eo.addEntry("First input vy", face_values[i].first.velocity.y);
			eo.addEntry("First input vz", face_values[i].first.velocity.z);
			eo.addEntry("Second input Density", face_values[i].second.density);
			eo.addEntry("Second input pressure", face_values[i].second.pressure);
			eo.addEntry("Second input internal energy", face_values[i].second.internal_energy);
			eo.addEntry("Second input vx", face_values[i].second.velocity.x);
			eo.addEntry("Second input vy", face_values[i].second.velocity.y);
			eo.addEntry("Second input vz", face_values[i].second.velocity.z);
			eo.addEntry("First cell Density", cells[tess.GetFaceNeighbors(i).first].density);
			eo.addEntry("First cell pressure", cells[tess.GetFaceNeighbors(i).first].pressure);
			eo.addEntry("First cell internal energy", cells[tess.GetFaceNeighbors(i).first].internal_energy);
			eo.addEntry("First cell vx", cells[tess.GetFaceNeighbors(i).first].velocity.x);
			eo.addEntry("First cell vy", cells[tess.GetFaceNeighbors(i).first].velocity.y);
			eo.addEntry("First cell vz", cells[tess.GetFaceNeighbors(i).first].velocity.z);
			eo.addEntry("Second cell Density", cells[tess.GetFaceNeighbors(i).second].density);
			eo.addEntry("Second cell pressure", cells[tess.GetFaceNeighbors(i).second].pressure);
			eo.addEntry("Second cell internal energy", cells[tess.GetFaceNeighbors(i).second].internal_energy);
			eo.addEntry("Second cell vx", cells[tess.GetFaceNeighbors(i).second].velocity.x);
			eo.addEntry("Second cell vy", cells[tess.GetFaceNeighbors(i).second].velocity.y);
			eo.addEntry("Second cell vz", cells[tess.GetFaceNeighbors(i).second].velocity.z);
			eo.addEntry("Face vx", face_velocities[i].x);
			eo.addEntry("Face vy", face_velocities[i].y);
			eo.addEntry("Face vz", face_velocities[i].z);
			throw eo;
		}
		choose_action(i, tess, cells, eos, face_velocities[i], sequence_, fluxes[i], time, face_values[i]);
	}
}

ConditionActionFlux1::Condition3D::~Condition3D(void) {}

ConditionActionFlux1::Action3D::~Action3D(void) {}

RegularFlux3D::RegularFlux3D(const RiemannSolver3D& rs) :
	rs_(rs) {}

void RegularFlux3D::operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
	const vector<ComputationalCell3D>& /*cells*/, const EquationOfState& eos, const bool /*aux*/, Conserved3D &res,
	double /*time*/, std::pair<ComputationalCell3D, ComputationalCell3D>
	const& face_values) const
{
	const Vector3D normal = normalize(tess.Normal(face_index));
	RotateSolveBack3D(normal, face_values.first, face_values.second, face_velocity, rs_, res, eos);
}

RigidWallFlux3D::RigidWallFlux3D(const RiemannSolver3D& rs) : rs_(rs) {}

namespace
{
	void reflect(ComputationalCell3D & cell, Vector3D const& normal)
	{
		hllc::Reflect(cell.velocity.x, cell.velocity.y, cell.velocity.z, normal.x, normal.y, normal.z);
	}

	void rigid_wall_states(std::pair<ComputationalCell3D, ComputationalCell3D> &res,
		Vector3D const& normal, const bool aux)
	{
		if (aux)
		{
			res.second = res.first;
			reflect(res.second, normal);
		}
		else
		{
			res.first = res.second;
			reflect(res.first, normal);
		}
	}
}

void RigidWallFlux3D::operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
	const vector<ComputationalCell3D>& /*cells*/, const EquationOfState& eos, const bool aux, Conserved3D &res,
	double /*time*/, std::pair<ComputationalCell3D, ComputationalCell3D>
	const& face_values) const
{
	const Vector3D normal = normalize(tess.Normal(face_index));
	std::pair<ComputationalCell3D, ComputationalCell3D>	rigid_states(face_values);
	rigid_wall_states(rigid_states, normal, aux);
	RotateSolveBack3D(normal, rigid_states.first, rigid_states.second, face_velocity, rs_, res, eos);
}

FreeFlowFlux3D::FreeFlowFlux3D(const RiemannSolver3D& rs) : rs_(rs) {}

void FreeFlowFlux3D::operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
	const vector<ComputationalCell3D>& /*cells*/, const EquationOfState& eos, const bool aux, Conserved3D &res,
	double /*time*/, std::pair<ComputationalCell3D, ComputationalCell3D>
	const& face_values) const
{
	const Vector3D normal = normalize(tess.Normal(face_index));
	std::pair<ComputationalCell3D, ComputationalCell3D>	states(face_values);
	if (aux)
		states.second = states.first;
	else
		states.first = states.second;
	RotateSolveBack3D(normal, states.first, states.second, face_velocity, rs_, res, eos);
}

IsBoundaryFace3D::IsBoundaryFace3D(void) {}

pair<bool, bool> IsBoundaryFace3D::operator()(size_t face_index, const Tessellation3D& tess,
	const vector<ComputationalCell3D>& /*cells*/) const
{
	if (!tess.BoundaryFace(face_index))
		return pair<bool, bool>(false, false);
	if (tess.GetFaceNeighbors(face_index).first >= tess.GetPointNo())
		return pair<bool, bool>(true, false);
	else
		return pair<bool, bool>(true, true);
}

IsBulkFace3D::IsBulkFace3D(void) {}

pair<bool, bool> IsBulkFace3D::operator()(size_t face_index, const Tessellation3D& tess,
	const vector<ComputationalCell3D>& /*cells*/)const
{
	if (tess.BoundaryFace(face_index))
		return pair<bool, bool>(false, false);
	else
		return pair<bool, bool>(true, false);
}

RegularSpecialEdge3D::RegularSpecialEdge3D(const string& sticker_name) :
	sticker_name_(sticker_name) {}

pair<bool, bool> RegularSpecialEdge3D::operator()(size_t face_index, const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells)const
{
  if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).first).stickers.begin(), ComputationalCell3D::stickerNames.begin(),
		     ComputationalCell3D::stickerNames.end(), sticker_name_))
	{
	  if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).second).stickers.begin(), ComputationalCell3D::stickerNames.begin(),
			     ComputationalCell3D::stickerNames.end(), sticker_name_))
			return pair<bool, bool>(false, false);
		return pair<bool, bool>(true, false);
	}
  if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).second).stickers.begin(), ComputationalCell3D::stickerNames.begin(),
		     ComputationalCell3D::stickerNames.end(), sticker_name_))
		return pair<bool, bool>(true, true);
	return pair<bool, bool>(false, false);
}

LagrangianFlux3D::LagrangianFlux3D(const LagrangianHLLC3D & rs, const LagrangianHLLC3D & rs2, 
	LagrangianCriteria3D const & criteria):ws_(std::vector<double>()), edge_vel_(std::vector<double>()),
	Lag_calc_(std::vector<bool>()), rs_(rs), rs2_(rs2), criteria_(criteria) {}

void LagrangianFlux3D::operator()(size_t face_index, const Tessellation3D & tess, const Vector3D & face_velocity, 
	const vector<ComputationalCell3D>& cells, const EquationOfState & eos, const bool aux, Conserved3D & res, double time, 
	 std::pair<ComputationalCell3D, ComputationalCell3D> const & face_values) const
{
	size_t N = tess.GetTotalFacesNumber();
	ws_.resize(N, 0.0);
	edge_vel_.resize(N, 0.0);
	Lag_calc_.resize(N, false);
	const Vector3D normal = normalize(tess.Normal(face_index));
	if (criteria_(face_index, tess, face_velocity, cells, eos, aux, face_values, time))
	{
		RotateSolveBack3D(normal, face_values.first, face_values.second, face_velocity, rs_, res, eos);
		ws_[face_index] = rs_.ws;
		Lag_calc_[face_index] = true;
	}
	else
	{
		RotateSolveBack3D(normal, face_values.first, face_values.second, face_velocity, rs2_, res, eos);
		ws_[face_index] = 0;
		Lag_calc_[face_index] = false;
	}
	edge_vel_[face_index] = ScalarProd(normal, face_velocity);
}

void LagrangianFlux3D::Reset(void) const
{
	ws_.assign(ws_.size(), 0);
	edge_vel_.assign(edge_vel_.size(), 0);
	Lag_calc_.assign(Lag_calc_.size(), false);
}

LagrangianFlux3D::LagrangianCriteria3D::~LagrangianCriteria3D(){}

BothSpecialEdge3D::BothSpecialEdge3D(const string& sticker_name):sticker_name_(sticker_name)
{}

pair<bool, bool> BothSpecialEdge3D::operator()(size_t face_index, const Tessellation3D& tess, const vector<ComputationalCell3D>& cells) const
{
	if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).first).stickers.begin(), 
		ComputationalCell3D::stickerNames.begin(),
		ComputationalCell3D::stickerNames.end(), sticker_name_) &&
		*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).second).stickers.begin(), 
			ComputationalCell3D::stickerNames.begin(),
			ComputationalCell3D::stickerNames.end(), sticker_name_))
			return pair<bool, bool>(true, false);
	else
		return pair<bool, bool>(false, false);
}

TwoSpecialEdge3D::TwoSpecialEdge3D(const string& sticker_name1, const string& sticker_name2):
	sticker_name1_(sticker_name1), sticker_name2_(sticker_name2){}

pair<bool, bool> TwoSpecialEdge3D::operator()(size_t face_index, const Tessellation3D& tess, const vector<ComputationalCell3D>& cells) const
{
	if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).first).stickers.begin(), 
			ComputationalCell3D::stickerNames.begin(),
			ComputationalCell3D::stickerNames.end(), sticker_name1_) &&
		*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).second).stickers.begin(), 
			ComputationalCell3D::stickerNames.begin(),
			ComputationalCell3D::stickerNames.end(), sticker_name2_))
		return pair<bool, bool>(true, false);
	else
		if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).first).stickers.begin(), 
				ComputationalCell3D::stickerNames.begin(),
				ComputationalCell3D::stickerNames.end(), sticker_name2_) &&
			*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).second).stickers.begin(), 
				ComputationalCell3D::stickerNames.begin(),
				ComputationalCell3D::stickerNames.end(), sticker_name1_))
			return pair<bool, bool>(true, true);
		else
			return pair<bool, bool>(false, false);
}

void ZeroFlux3D::operator()(size_t /*face_index*/, const Tessellation3D& /*tess*/, 
	const Vector3D& /*face_velocity*/, const vector<ComputationalCell3D>& 
	/*cells*/, const EquationOfState& /*eos*/, const bool /*aux*/, Conserved3D& res, 
	double /*time*/, std::pair<ComputationalCell3D, ComputationalCell3D> const& /*face_values*/) const
{
	res.energy = 0;
	res.internal_energy = 0;
	res.mass = 0;
	res.momentum = Vector3D();
	size_t const Ntracers = ComputationalCell3D::tracerNames.size();
	for (size_t i = 0; i < Ntracers; ++i)
		res.tracers[i] = 0;
}
