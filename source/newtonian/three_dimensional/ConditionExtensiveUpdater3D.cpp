#include "ConditionExtensiveUpdater3D.hpp"
#include "../../misc/utils.hpp"
#include <iostream>
#include <cfloat>
#include <algorithm>
#include "ExtensiveUpdaterKernels.hpp"
#include "utils/Portable.hpp"
#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif

ConditionExtensiveUpdater3D::Condition3D::~Condition3D() {}

ConditionExtensiveUpdater3D::Action3D::~Action3D() {}

ConditionExtensiveUpdater3D::~ConditionExtensiveUpdater3D() {}

ConditionExtensiveUpdater3D::ConditionExtensiveUpdater3D(const vector<pair<const Condition3D*, const Action3D*> >& sequence) :
	sequence_(sequence), onDevice_(false) {}

// The flat arrays of the update (ExtensiveUpdaterKernels.hpp), kept between calls
struct ConditionExtensiveUpdaterArrays
{
	portable::Buffer<std::size_t> faceOffsets, faces;
	portable::Buffer<std::uint8_t> first, status;
	portable::Buffer<double> deltas, velocities, extensives;
};

namespace
{
	void ToFields(Conserved3D const& c, double *out)
	{
		out[extensive::Mass] = c.mass;
		out[extensive::MomentumX] = c.momentum.x;
		out[extensive::MomentumY] = c.momentum.y;
		out[extensive::MomentumZ] = c.momentum.z;
		out[extensive::Energy] = c.energy;
		out[extensive::InternalEnergy] = c.internal_energy;
		out[extensive::Erad] = c.Erad;
		out[extensive::EradDt] = c.Erad_dt;
		out[extensive::EradDtDt] = c.Erad_dt_dt;
		for (size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
			out[extensive::FirstGroup + j] = c.Eg[j];
		for (size_t j = 0; j < MAX_TRACERS; ++j)
			out[extensive::FirstTracer + j] = c.tracers[j];
	}

	void FromFields(const double *in, Conserved3D &c)
	{
		c.mass = in[extensive::Mass];
		c.momentum.x = in[extensive::MomentumX];
		c.momentum.y = in[extensive::MomentumY];
		c.momentum.z = in[extensive::MomentumZ];
		c.energy = in[extensive::Energy];
		c.internal_energy = in[extensive::InternalEnergy];
		c.Erad = in[extensive::Erad];
		c.Erad_dt = in[extensive::EradDt];
		c.Erad_dt_dt = in[extensive::EradDtDt];
		for (size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
			c.Eg[j] = in[extensive::FirstGroup + j];
		for (size_t j = 0; j < MAX_TRACERS; ++j)
			c.tracers[j] = in[extensive::FirstTracer + j];
	}

	// Prints a cell that failed its check, with its faces and neighbors, and throws
	void ThrowBadCell(size_t i, const vector<Conserved3D>& fluxes, const Tessellation3D& tess, const double dt,
		const vector<ComputationalCell3D>& cells, vector<Conserved3D>& extensives, const vector<Vector3D>& edge_velocities,
		const vector<Vector3D>& point_velocities,
		std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const& interp_values)
	{
		Conserved3D delta;
		UniversalError eo("Bad extesnsive update");
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
		std::cout << "Bad cell in ExtensiveUpdater3D, cell " << i << " rank " << rank <<" dt "<<dt<<" ID "<<cells[i].ID<< std::endl;
		std::cout << "mass " << extensives[i].mass << " energy " << extensives[i].energy << " internalE " <<
			extensives[i].internal_energy << " momentum" << abs(extensives[i].momentum) << " volume " << tess.GetVolume(i)
			<< std::endl;
		std::cout << "Old cell, density " << cells[i].density << " pressure " << cells[i].pressure << " vx " <<
			cells[i].velocity.x << " vy " << cells[i].velocity.y << " vz " << cells[i].velocity.z << std::endl;
		std::cout<<"Point "<<tess.GetMeshPoint(i)<<" CM "<<tess.GetCellCM(i)<<" d "<<abs(tess.GetMeshPoint(i) - tess.GetCellCM(i)) / tess.GetWidth(i)<<std::endl;
		for (size_t j = 0; j < ComputationalCell3D::tracerNames.size(); ++j)
		{
		  std::cout << ComputationalCell3D::tracerNames[j] << " old cell " << cells[i].tracers[j] << 
				" extensive "<<extensives[i].tracers[j]<<std::endl;
		}
		face_vec temp = tess.GetCellFaces(i);
		Conserved3D old_ext(extensives[i]);
		// recover old_extensive
		for (size_t j = 0; j < temp.size(); ++j)
		{
			double Area = tess.GetArea(temp[j]) * dt;
			//				size_t N0 = tess.GetFaceNeighbors(temp[j]).first;
			size_t N1 = tess.GetFaceNeighbors(temp[j]).second;
			if (N1 == i)
			{
			  //					double newEk = 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;
				old_ext.mass -= Area * fluxes[temp[j]].mass;
				old_ext.momentum -= Area * fluxes[temp[j]].momentum;
				//old_ext.energy -= Area * fluxes[temp[j]].energy;
				/*					old_ext.internal_energy -= Area * fluxes[temp[j]].energy -
					newEk + 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;*/
			}
			else
			{
			  //					double newEk = 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;
				old_ext.mass += Area * fluxes[temp[j]].mass;
				old_ext.momentum += Area * fluxes[temp[j]].momentum;
				//					old_ext.energy += Area * fluxes[temp[j]].energy;
				/*old_ext.internal_energy += Area * fluxes[temp[j]].energy +
				  newEk - 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;*/
			}
		}
		for (size_t j = 0; j < temp.size(); ++j)
		{
			size_t N0 = tess.GetFaceNeighbors(temp[j]).first;
			size_t N1 = tess.GetFaceNeighbors(temp[j]).second;
			double Area = tess.GetArea(temp[j]) * dt;
			double Ek = 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;
			delta = Area * fluxes[temp[j]];
			double dEtherm1 = 0;
			if (N1 == i)
			{
				old_ext += delta;
				double Eknew1 = 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;
				dEtherm1 = delta.energy - (Eknew1 - Ek);
				//					old_ext.internal_energy += delta.energy - (Eknew1 - Ek);
			}
			else
			{
				old_ext -= delta;
				double Eknew1 = 0.5 * ScalarProd(old_ext.momentum, old_ext.momentum) / old_ext.mass;
				dEtherm1 = -delta.energy - (Eknew1 - Ek);
				//					old_ext.internal_energy += dEtherm1;
			}
			Vector3D normalf = normalize(tess.Normal(temp[j]));
			std::cout << "Face " << temp[j] << " neigh " << N0 << "," << N1 << " mass=" << fluxes[temp[j]].mass * Area <<
				" energy= " << fluxes[temp[j]].energy * Area << " Etherm= " << dEtherm1 << " momentum= " << abs(fluxes[temp[j]].momentum) * Area <<
				" Area*dt " << Area << " normal " << normalf.x << "," << normalf.y << "," << normalf.z <<
				" face velocity "<<edge_velocities[temp[j]].x<<","<< edge_velocities[temp[j]].y<<","<< edge_velocities[temp[j]].z<<
				" point0 "<<tess.GetMeshPoint(N0)<<" point1 "<<tess.GetMeshPoint(N1)<<
				" point_vel0 "<<(N0 < point_velocities.size() ? point_velocities[N0] : Vector3D())<<
				" point_vel1 "<<(N1 < point_velocities.size() ? point_velocities[N1] : Vector3D())<<" Face CM "<<tess.FaceCM(temp[j])<< std::endl;
			eo.addEntry("Face", static_cast<double>(temp[j]));
			eo.addEntry("Face neigh 0", static_cast<double>(tess.GetFaceNeighbors(temp[j]).first));
			eo.addEntry("Face neigh 1", static_cast<double>(tess.GetFaceNeighbors(temp[j]).second));
			eo.addEntry("First input Density", interp_values[temp[j]].first.density);
			eo.addEntry("First input pressure", interp_values[temp[j]].first.pressure);
			eo.addEntry("First input internal energy", interp_values[temp[j]].first.internal_energy);
			eo.addEntry("First input vx", interp_values[temp[j]].first.velocity.x);
			eo.addEntry("First input vy", interp_values[temp[j]].first.velocity.y);
			eo.addEntry("First input vz", interp_values[temp[j]].first.velocity.z);
			eo.addEntry("Second input Density", interp_values[temp[j]].second.density);
			eo.addEntry("Second input pressure", interp_values[temp[j]].second.pressure);
			eo.addEntry("Second input internal energy", interp_values[temp[j]].second.internal_energy);
			eo.addEntry("Second input vx", interp_values[temp[j]].second.velocity.x);
			eo.addEntry("Second input vy", interp_values[temp[j]].second.velocity.y);
			eo.addEntry("Second input vz", interp_values[temp[j]].second.velocity.z);
			for (size_t k = 0; k < ComputationalCell3D::tracerNames.size(); ++k)
			{
			  std::cout << ComputationalCell3D::tracerNames[k] << " flux is " << fluxes[temp[j]].tracers[k] * Area << std::endl;
			}
		}
		for (size_t j = 0; j < temp.size(); ++j)
		{
			size_t N0 = tess.GetFaceNeighbors(temp[j]).first;
			size_t N1 = tess.GetFaceNeighbors(temp[j]).second;
			size_t Nother = N1 == i ? N0 : N1;
			std::cout << "Neigh cell " << Nother << ", density " << cells[Nother].density << " pressure " <<
				cells[Nother].pressure << " vx " << cells[Nother].velocity.x << " vy " << cells[Nother].velocity.y << " vz "
				<< cells[Nother].velocity.z << std::endl;
			for (size_t k = 0; k < ComputationalCell3D::tracerNames.size(); ++k)
			{
			  std::cout << ComputationalCell3D::tracerNames[k] << " of other " << cells[Nother].tracers[k] << std::endl;
			}
		}
		throw eo;
	}
}

void ConditionExtensiveUpdater3D::RunOnDevice(bool onDevice)
{
	onDevice_ = onDevice;
}

void ConditionExtensiveUpdater3D::operator()(const vector<Conserved3D>& fluxes, const Tessellation3D& tess,
	const double dt, const vector<ComputationalCell3D>& cells, vector<Conserved3D>& extensives, double time,
	 const vector<Vector3D>& edge_velocities,
	const vector<Vector3D>& point_velocities,
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const& interp_values) const
{
	size_t N = tess.GetPointNo();
	bool entropy = !(std::find(ComputationalCell3D::tracerNames.begin(), ComputationalCell3D::tracerNames.end(), std::string("Entropy")) ==
			 ComputationalCell3D::tracerNames.end());
	size_t entropy_index = static_cast<size_t>(std::find(ComputationalCell3D::tracerNames.begin(),
							     ComputationalCell3D::tracerNames.end(), std::string("Entropy")) - ComputationalCell3D::tracerNames.begin());
	size_t Nfluxes = fluxes.size();
	const portable::Where where = onDevice_ ? portable::Where::Device : portable::Where::Host;
	if (not arrays_)
		arrays_ = std::make_shared<ConditionExtensiveUpdaterArrays>();
	ConditionExtensiveUpdaterArrays &data = *arrays_;
	const size_t K = extensive::fields;

	vector<double> &velocities = data.velocities.Host();
	vector<double> &flat_extensives = data.extensives.Host();
	vector<std::uint8_t> &statuses = data.status.Host();
	velocities.resize(3 * N);
	flat_extensives.resize(N * K);
	statuses.resize(N);
	portable::HostParallelFor(N, [&](size_t i)
	{
		velocities[3 * i] = cells[i].velocity.x;
		velocities[3 * i + 1] = cells[i].velocity.y;
		velocities[3 * i + 2] = cells[i].velocity.z;
		ToFields(extensives[i], flat_extensives.data() + i * K);
	});
	// with conditions and actions, the cells are checked after them, below
	const bool check = sequence_.empty();

	if (where == portable::Where::Host and portable::HostThreads() == 1)
	{
		// One thread: the original loop over faces, each updating its two neighbors
		for (size_t i = 0; i < Nfluxes; ++i)
		{
			double delta[extensive::fields];
			ToFields(fluxes[i], delta);
			extensive::FaceDelta(delta, dt, tess.GetArea(i));
			size_t n0 = tess.GetFaceNeighbors(i).first;
			size_t n1 = tess.GetFaceNeighbors(i).second;
			if (n0 < N)
				extensive::ApplyFace(flat_extensives.data() + n0 * K, delta, velocities[3 * n0], velocities[3 * n0 + 1],
					velocities[3 * n0 + 2], true);
			if (n1 < N)
				extensive::ApplyFace(flat_extensives.data() + n1 * K, delta, velocities[3 * n1], velocities[3 * n1 + 1],
					velocities[3 * n1 + 2], false);
		}
		for (size_t i = 0; i < N; ++i)
			statuses[i] = extensive::FinishCell(flat_extensives.data() + i * K, check, entropy, entropy_index);
	}
	else
	{
		// Each cell's faces in increasing face index, the order the faces' fluxes used to be added in
		vector<size_t> &offsets = data.faceOffsets.Host();
		offsets.resize(N + 1);
		offsets[0] = 0;
		portable::HostParallelFor(N, [&](size_t i)
		{
			face_vec const& cellFaces = tess.GetCellFaces(i);
			offsets[i + 1] = static_cast<size_t>(std::count_if(cellFaces.begin(), cellFaces.end(),
				[Nfluxes](size_t face) { return face < Nfluxes; }));
		});
		for (size_t i = 0; i < N; ++i)
			offsets[i + 1] += offsets[i];
		vector<size_t> &faces = data.faces.Host();
		vector<std::uint8_t> &first = data.first.Host();
		faces.resize(offsets[N]);
		first.resize(offsets[N]);
		portable::HostParallelFor(N, [&](size_t i)
		{
			face_vec const& cellFaces = tess.GetCellFaces(i);
			std::copy_if(cellFaces.begin(), cellFaces.end(), faces.begin() + static_cast<std::ptrdiff_t>(offsets[i]),
				[Nfluxes](size_t face) { return face < Nfluxes; });
			std::sort(faces.begin() + static_cast<std::ptrdiff_t>(offsets[i]), faces.begin() + static_cast<std::ptrdiff_t>(offsets[i + 1]));
			for (size_t k = offsets[i]; k < offsets[i + 1]; ++k)
				first[k] = tess.GetFaceNeighbors(faces[k]).first == i ? 1 : 0;
		});

		vector<double> &deltas = data.deltas.Host();
		deltas.resize(Nfluxes * K);
		portable::HostParallelFor(Nfluxes, [&](size_t i)
		{
			ToFields(fluxes[i], deltas.data() + i * K);
			extensive::FaceDelta(deltas.data() + i * K, dt, tess.GetArea(i));
		});
		if (where == portable::Where::Device)
		{
			data.faceOffsets.ToDevice();
			data.faces.ToDevice();
			data.first.ToDevice();
			data.deltas.ToDevice();
			data.velocities.ToDevice();
			data.extensives.ToDevice();
			data.status.AllocateDevice();
		}
		extensive::CellUpdates kernel;
		kernel.faceOffsets = data.faceOffsets.Data(where);
		kernel.faces = data.faces.Data(where);
		kernel.first = data.first.Data(where);
		kernel.deltas = data.deltas.Data(where);
		kernel.velocities = data.velocities.Data(where);
		kernel.extensives = data.extensives.Data(where);
		kernel.status = data.status.Data(where);
		kernel.check = check;
		kernel.entropy = entropy;
		kernel.entropyTracer = entropy_index;
		portable::ParallelFor(where, "ConditionExtensiveUpdater3D::CellUpdates", N, kernel);
		if (where == portable::Where::Device)
		{
			data.extensives.ToHost();
			data.status.ToHost();
		}
	}
	portable::HostParallelFor(N, [&](size_t i)
	{
		FromFields(flat_extensives.data() + i * K, extensives[i]);
	});

	for (size_t i = 0; i < N; ++i)
	{
		std::uint8_t status = statuses[i];
		if (not sequence_.empty())
		{
			for (size_t j = 0; j < sequence_.size(); ++j)
			{
				if (sequence_[j].first->operator()(i, tess, cells, time))
				{
					sequence_[j].second->operator()(fluxes, tess, dt, cells, extensives, i, time);
					break;
				}
			}
			double e[extensive::fields];
			ToFields(extensives[i], e);
			status = extensive::CheckCell(e, entropy, entropy_index);
		}
		if (status == extensive::NegativeGroup)
		{
			for(size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
			{
				if(extensives[i].Eg[j] < 0 && extensives[i].mass > 0)
				{

					UniversalError eo("Negative energy group");
					eo.addEntry("ID",cells[i].ID);
					eo.addEntry("group", j);
					eo.addEntry("group energy", extensives[i].Eg[j]);
					eo.addEntry("approximate Eg", cells[i].Eg[j] * extensives[i].mass);
					eo.addEntry("mass", extensives[i].mass);
					throw eo;
				}
			}
		}
		if (status == extensive::BadCell)
			ThrowBadCell(i, fluxes, tess, dt, cells, extensives, edge_velocities, point_velocities, interp_values);
	}
	extensives.resize(tess.GetPointNo());
}

void RegularExtensiveUpdate3D::operator()(const vector<Conserved3D>& /*fluxes*/, const Tessellation3D& /*tess*/, const double /*dt*/,
	const vector<ComputationalCell3D>& /*cells*/, vector<Conserved3D>& /*extensives*/,
	size_t /*index*/, double /*time*/)const
{
	//assert(extensives[index].internal_energy > 0);
	return;
}

void NoExtensiveUpdate3D::operator()(const vector<Conserved3D>& /*fluxes*/, 
				     const Tessellation3D& tess, const double /*dt*/, const vector<ComputationalCell3D>& cells, 
				     vector<Conserved3D>& extensives, size_t index, double /*time*/) const
{
	PrimitiveToConserved(cells[index], tess.GetVolume(index), extensives[index]);
}
