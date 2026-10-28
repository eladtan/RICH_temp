#include "LinearGauss3D.hpp"
#include "Hllc3D.hpp"
#include "../../misc/utils.hpp"
#include <array>
#include <iostream>
#include <cstring>  // for memcpy in fused operations
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include "LinearGaussKernels.hpp"
#include "LinearGaussArrays.hpp"
#include "utils/Portable.hpp"
#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif

using portable::Buffer;
using portable::Where;

namespace
{
	// new_cells = cells, then default cells up to `total` (as assign and resize), copied in parallel into the kept vector
	void CopyCells(vector<ComputationalCell3D> &new_cells, vector<ComputationalCell3D> const& cells, size_t total)
	{
		new_cells.resize(total);
		const size_t copied = std::min(cells.size(), total);
		portable::HostParallelFor(total, [&](size_t i)
		{
			if(i < copied)
			{
				new_cells[i] = cells[i];
			}
			else
			{
				new_cells[i] = ComputationalCell3D();
			}
		});
	}

	// The arrays kept between calls, made on first use
	LinearGaussArrays &KeptArrays(std::shared_ptr<LinearGaussArrays> &arrays)
	{
		if(not arrays)
		{
			arrays = std::make_shared<LinearGaussArrays>();
		}
		arrays->inputsOnDevice = false;
		arrays->slopesOnDevice = 0;
		return *arrays;
	}

	vector<bool> BuildMPIGhostMask(Tessellation3D const& tess)
	{
		if(!tess.HasPeriodic())
		{
			return vector<bool>();
		}
		vector<bool> result(tess.GetTotalPointNumber(), false);
#ifdef RICH_MPI
		const vector<vector<size_t> > &ghostIndices = tess.GetGhostIndeces();
		for(const vector<size_t> &indices : ghostIndices)
		{
			for(size_t index : indices)
			{
				if(index < result.size())
				{
					result[index] = true;
				}
			}
		}
#endif
		return result;
	}

	size_t ResolveStateIndex(Tessellation3D const& tess, size_t meshIndex,
		size_t cellNumber, const vector<bool> &mpiGhostMask)
	{
		if(meshIndex < cellNumber)
		{
			return meshIndex;
		}
		if(meshIndex < mpiGhostMask.size() && mpiGhostMask[meshIndex])
		{
			return meshIndex;
		}
		if(!tess.IsPeriodicImage(meshIndex))
		{
			return meshIndex;
		}
		const size_t physical = tess.ResolvePeriodicImageIndex(meshIndex);
		return physical < cellNumber ? physical : meshIndex;
	}

	void FillLocalPeriodicImagePrimitives(Tessellation3D const& tess, size_t cellNumber,
		const vector<bool> &mpiGhostMask, vector<ComputationalCell3D> &allCells)
	{
		if(!tess.HasPeriodic())
		{
			return;
		}
		const size_t total = tess.GetTotalPointNumber();
		for(size_t index = cellNumber; index < total; ++index)
		{
			if(!tess.IsPeriodicImage(index) ||
				(index < mpiGhostMask.size() && mpiGhostMask[index]))
			{
				continue;
			}
			const size_t physical = ResolveStateIndex(tess, index, cellNumber, mpiGhostMask);
			if(physical < cellNumber)
			{
				ReplaceComputationalCell(allCells[index], allCells[physical]);
			}
		}
	}

	ComputationalCell3D interp(ComputationalCell3D const& cell, Slope3D const& slope,
		Vector3D const& target, Vector3D const& cm, EquationOfState const& eos, 
		bool pressure_calc)
	{
		ComputationalCell3D res(cell);
		const double ox = target.x - cm.x, oy = target.y - cm.y, oz = target.z - cm.z;
		const auto add = [&](double &value, double dx, double dy, double dz)
		{
			value = gauss::InterpolateSequential(value, dx, dy, dz, ox, oy, oz);
		};
		const ComputationalCell3D &sx = slope.xderivative, &sy = slope.yderivative, &sz = slope.zderivative;
		add(res.density, sx.density, sy.density, sz.density);
		add(res.pressure, sx.pressure, sy.pressure, sz.pressure);
		add(res.internal_energy, sx.internal_energy, sy.internal_energy, sz.internal_energy);
		add(res.temperature, sx.temperature, sy.temperature, sz.temperature);
		add(res.Erad, sx.Erad, sy.Erad, sz.Erad);
		add(res.Erad_dt, sx.Erad_dt, sy.Erad_dt, sz.Erad_dt);
		add(res.Erad_dt_dt, sx.Erad_dt_dt, sy.Erad_dt_dt, sz.Erad_dt_dt);
		add(res.cs, sx.cs, sy.cs, sz.cs);
		add(res.dt, sx.dt, sy.dt, sz.dt);
		add(res.velocity.x, sx.velocity.x, sy.velocity.x, sz.velocity.x);
		add(res.velocity.y, sx.velocity.y, sy.velocity.y, sz.velocity.y);
		add(res.velocity.z, sx.velocity.z, sy.velocity.z, sz.velocity.z);
		for (size_t j = 0; j < MAX_TRACERS; ++j)
			add(res.tracers[j], sx.tracers[j], sy.tracers[j], sz.tracers[j]);
		for (size_t j = 0; j < res.Eg.size(); ++j)
			add(res.Eg[j], sx.Eg[j], sy.Eg[j], sz.Eg[j]);
		if (pressure_calc)
			try
		{
			//res.pressure = eos.de2p(res.density, res.internal_energy, res.tracers, tsn.tracer_names);
		  res.internal_energy = eos.dp2e(res.density, res.pressure, res.tracers, ComputationalCell3D::tracerNames);
		}
		catch (UniversalError &eo)
		{
			eo.addEntry("density", res.density);
			eo.addEntry("internal energy", res.internal_energy);
			throw eo;
		}
		return res;
	}

	// The kernels run on the device only if the equation of state, which runs between them, does too
	Where WhereToRun(bool onDevice, EquationOfState const& eos)
	{
		return onDevice and eos.RunsOnDevice() ? Where::Device : Where::Host;
	}

	/* The equation of state's arrays for the items of a flat array with `fields` values each: density and `second`
	   (a field) are inputs, `result` (a field, or its own array with resultStride 1) the output */
	EosArrays FieldArrays(size_t n, size_t fields, size_t tracers, const double *values, size_t second,
		double *result, size_t resultStride, std::uint8_t *status, std::uint8_t failure)
	{
		EosArrays arrays;
		arrays.n = n;
		arrays.stride = fields;
		arrays.density = values + gauss::Density;
		arrays.second = values + second;
		arrays.tracers = tracers > 0 ? values + gauss::FirstTracer : nullptr;
		arrays.tracersNum = tracers;
		arrays.tracerNames = &ComputationalCell3D::tracerNames;
		arrays.result = result;
		arrays.resultStride = resultStride;
		arrays.status = status;
		arrays.failure = failure;
		return arrays;
	}

	// Writes the fields that have slopes into a cell; its other fields are left as they are
	void SetFields(ComputationalCell3D &cell, const double *values, size_t tracers)
	{
		cell.density = values[gauss::Density];
		cell.pressure = values[gauss::Pressure];
		cell.internal_energy = values[gauss::InternalEnergy];
		cell.velocity.x = values[gauss::VelocityX];
		cell.velocity.y = values[gauss::VelocityY];
		cell.velocity.z = values[gauss::VelocityZ];
		cell.cs = values[gauss::SoundSpeed];
		cell.dt = values[gauss::TimeStep];
		for(size_t j = 0; j < tracers; ++j)
		{
			cell.tracers[j] = values[gauss::FirstTracer + j];
		}
	}

	// A slope from its flat x, y and z derivatives; the fields without slopes have zero slopes
	void SetSlope(Slope3D &slope, const double *values, size_t fields, size_t tracers)
	{
		ComputationalCell3D *derivatives[3] = {&slope.xderivative, &slope.yderivative, &slope.zderivative};
		for(size_t dim = 0; dim < 3; ++dim)
		{
			ComputationalCell3D &derivative = *derivatives[dim];
			derivative.temperature = 0;
			derivative.Erad = 0;
			derivative.Erad_dt = 0;
			derivative.Erad_dt_dt = 0;
			derivative.Eg.assign(ENERGY_GROUPS_NUM, 0);
			derivative.tracers.fill(0);
			SetFields(derivative, values + dim * fields, tracers);
		}
	}

	// The flat fields of a slope
	void GetSlope(const Slope3D &slope, double *values, size_t fields, size_t tracers)
	{
		const ComputationalCell3D *derivatives[3] = {&slope.xderivative, &slope.yderivative, &slope.zderivative};
		for(size_t dim = 0; dim < 3; ++dim)
		{
			const ComputationalCell3D &derivative = *derivatives[dim];
			double *out = values + dim * fields;
			out[gauss::Density] = derivative.density;
			out[gauss::Pressure] = derivative.pressure;
			out[gauss::InternalEnergy] = derivative.internal_energy;
			out[gauss::VelocityX] = derivative.velocity.x;
			out[gauss::VelocityY] = derivative.velocity.y;
			out[gauss::VelocityZ] = derivative.velocity.z;
			out[gauss::SoundSpeed] = derivative.cs;
			out[gauss::TimeStep] = derivative.dt;
			for(size_t j = 0; j < tracers; ++j)
			{
				out[gauss::FirstTracer + j] = derivative.tracers[j];
			}
		}
	}

	gauss::Config MakeConfig(const vector<string> &calc_tracers, bool slf, double shockratio, double diffusecoeff,
		double pressure_ratio, const string &skip_key, bool pressure_calc, bool apply_principal_limit)
	{
		const vector<string> &tracerNames = ComputationalCell3D::tracerNames;
		gauss::Config config;
		config.tracers = tracerNames.size();
		config.fields = gauss::FirstTracer + config.tracers;
		for(size_t i = 0; i < config.tracers; ++i)
		{
			if(std::find(calc_tracers.begin(), calc_tracers.end(), tracerNames[i]) != calc_tracers.end())
			{
				config.slopedTracers |= std::uint32_t(1) << i;
			}
		}
		config.limit = slf;
		config.principal = apply_principal_limit;
		config.pressureCalc = pressure_calc;
		config.skip = !skip_key.empty();
		config.shockRatio = shockratio;
		config.diffuseCoeff = diffusecoeff;
		config.pressureRatio = pressure_ratio;
		const vector<string>::const_iterator energy = binary_find(tracerNames.begin(), tracerNames.end(), string("Energy"));
		config.energyTracer = energy != tracerNames.end() ? static_cast<size_t>(energy - tracerNames.begin()) : MAX_TRACERS;
		return config;
	}

	// The flat mesh, and the states of all points (cells, then ghosts) from new_cells
	void PackMeshAndStates(LinearGaussArrays &data, Tessellation3D const& tess, vector<ComputationalCell3D> const& new_cells,
		const vector<bool> &mpiGhostMask, const string &skip_key)
	{
		const size_t CellNumber = tess.GetPointNo();
		const size_t total = new_cells.size();
		const size_t faces = tess.GetTotalFacesNumber();
		data.cellsNum = CellNumber;
		data.pointsNum = total;
		data.facesNum = faces;
		vector<size_t> &offsets = data.cellFaceOffsets.Host();
		offsets.resize(CellNumber + 1);
		offsets[0] = 0;
		for(size_t i = 0; i < CellNumber; ++i)
		{
			offsets[i + 1] = offsets[i] + tess.GetCellFaces(i).size();
		}
		vector<size_t> &cellFaces = data.cellFaces.Host();
		cellFaces.resize(offsets[CellNumber]);
		portable::HostParallelFor(CellNumber, [&](size_t i)
		{
			std::copy(tess.GetCellFaces(i).begin(), tess.GetCellFaces(i).end(), cellFaces.begin() + static_cast<std::ptrdiff_t>(offsets[i]));
		});
		vector<size_t> &faceNeighbors = data.faceNeighbors.Host();
		vector<double> &faceCM = data.faceCM.Host();
		vector<double> &faceArea = data.faceArea.Host();
		vector<size_t> &wallSlot = data.wallSlot.Host();
		faceNeighbors.resize(2 * faces);
		faceCM.resize(3 * faces);
		faceArea.resize(faces);
		wallSlot.resize(faces);
		portable::HostParallelFor(faces, [&](size_t f)
		{
			const std::pair<size_t, size_t> &neighbors = tess.GetFaceNeighbors(f);
			faceNeighbors[2 * f] = neighbors.first;
			faceNeighbors[2 * f + 1] = neighbors.second;
			const Vector3D &cm = tess.FaceCM(f);
			faceCM[3 * f] = cm.x;
			faceCM[3 * f + 1] = cm.y;
			faceCM[3 * f + 2] = cm.z;
			faceArea[f] = tess.GetArea(f);
			const bool ghost = neighbors.first >= CellNumber or neighbors.second >= CellNumber;
			wallSlot[f] = ghost and tess.BoundaryFace(f) ? 1 : 0;
		});
		data.wallsNum = 0;
		for(size_t f = 0; f < faces; ++f)
		{
			if(wallSlot[f] != 0)
			{
				wallSlot[f] = ++data.wallsNum;
			}
		}
		vector<double> &meshPoint = data.meshPoint.Host();
		vector<double> &pointCM = data.pointCM.Host();
		vector<size_t> &stateIndex = data.stateIndex.Host();
		meshPoint.resize(3 * total);
		pointCM.resize(3 * total);
		stateIndex.resize(total);
		portable::HostParallelFor(total, [&](size_t p)
		{
			const Vector3D &point = tess.GetMeshPoint(p);
			const Vector3D &cm = tess.GetCellCM(p);
			meshPoint[3 * p] = point.x;
			meshPoint[3 * p + 1] = point.y;
			meshPoint[3 * p + 2] = point.z;
			pointCM[3 * p] = cm.x;
			pointCM[3 * p + 1] = cm.y;
			pointCM[3 * p + 2] = cm.z;
		});
		for(size_t p = 0; p < total; ++p)
		{
			stateIndex[p] = ResolveStateIndex(tess, p, CellNumber, mpiGhostMask);
		}
		vector<double> &volume = data.volume.Host();
		vector<double> &width = data.width.Host();
		volume.resize(CellNumber);
		width.resize(CellNumber);
		portable::HostParallelFor(CellNumber, [&](size_t i)
		{
			volume[i] = tess.GetVolume(i);
			width[i] = tess.GetWidth(i);
		});

		const size_t fields = data.config.fields;
		vector<double> &states = data.states.Host();
		states.resize(total * fields);
		portable::HostParallelFor(total, [&](size_t p)
		{
			const ComputationalCell3D &cell = new_cells[p];
			double *values = states.data() + p * fields;
			values[gauss::Density] = cell.density;
			values[gauss::Pressure] = cell.pressure;
			values[gauss::InternalEnergy] = cell.internal_energy;
			values[gauss::VelocityX] = cell.velocity.x;
			values[gauss::VelocityY] = cell.velocity.y;
			values[gauss::VelocityZ] = cell.velocity.z;
			values[gauss::SoundSpeed] = cell.cs;
			values[gauss::TimeStep] = cell.dt;
			for(size_t j = 0; j < data.config.tracers; ++j)
			{
				values[gauss::FirstTracer + j] = cell.tracers[j];
			}
		});
		vector<std::uint8_t> &skip = data.skip.Host();
		skip.assign(data.config.skip ? total : 0, 0);
		if(data.config.skip)
		{
			const vector<string> &stickerNames = ComputationalCell3D::stickerNames;
			const vector<string>::const_iterator sticker = binary_find(stickerNames.begin(), stickerNames.end(), skip_key);
			assert(sticker != stickerNames.end());
			if(sticker != stickerNames.end())
			{
				const size_t index = static_cast<size_t>(sticker - stickerNames.begin());
				for(size_t p = 0; p < total; ++p)
				{
					skip[p] = new_cells[p].stickers[index] ? 1 : 0;
				}
			}
		}
	}

	bool HasBoundaryFace(Tessellation3D const& tess, size_t cell)
	{
		for(size_t face : tess.GetCellFaces(cell))
		{
			if(tess.BoundaryFace(face))
			{
				return true;
			}
		}
		return false;
	}

	// Each cell's slope and unlimited slope, in the flat arrays; throws the first failure, in the cells' order
	void ComputeSlopes(LinearGaussArrays &data, Tessellation3D const& tess, EquationOfState const& eos, Where where,
		vector<ComputationalCell3D> const& new_cells)
	{
		const gauss::Config &config = data.config;
		const size_t CellNumber = data.cellsNum;
		const size_t size = 3 * config.fields;
		data.slopes.Host().resize(data.pointsNum * size);
		data.naive.Host().resize(CellNumber * size);
		data.cellStatus.Host().resize(CellNumber);
		data.soundSpeeds.Host().resize(config.limit ? CellNumber : 0);
		data.soundSpeedStatus.Host().assign(config.limit ? CellNumber : 0, 0);
		if(where == Where::Device)
		{
			data.InputsToDevice();
			data.slopes.AllocateDevice();
			data.naive.AllocateDevice();
			data.cellStatus.AllocateDevice();
			data.soundSpeeds.AllocateDevice();
			data.soundSpeedStatus.ToDevice();
		}
		// The cells' sound speeds, for the limiter
		if(config.limit)
		{
			eos.de2cArrays(FieldArrays(CellNumber, config.fields, config.tracers, data.states.Data(where), gauss::InternalEnergy,
				data.soundSpeeds.Data(where), 1, data.soundSpeedStatus.Data(where), 1), where);
		}
		gauss::CellSlopes kernel;
		kernel.mesh = data.MeshView(where);
		kernel.state = data.StateView(where);
		kernel.config = config;
		kernel.soundSpeeds = data.soundSpeeds.Data(where);
		kernel.soundSpeedStatus = data.soundSpeedStatus.Data(where);
		kernel.slopes = data.slopes.Data(where);
		kernel.naive = data.naive.Data(where);
		kernel.status = data.cellStatus.Data(where);
		portable::ParallelFor(where, "LinearGauss3D::CellSlopes", CellNumber, kernel);
		if(where == Where::Device)
		{
			data.slopes.ToHost(0, CellNumber * size);
			data.slopesOnDevice = CellNumber;
			data.cellStatus.ToHost();
		}
		data.naiveOnDevice = where == Where::Device;

		const vector<std::uint8_t> &status = data.cellStatus.Host();
		for(size_t i = 0; i < CellNumber; ++i)
		{
			if(status[i] == gauss::Ok)
			{
				continue;
			}
			const Vector3D &center = tess.GetMeshPoint(i);
			if(status[i] & gauss::ManyNeighbors)
			{
				std::cout << "Cell has too many neighbors in calc naive slope, Cell x cor " << center.x <<
					" Cell y cor " << center.y << " Cell z cor " << center.z << std::endl;
			}
			const ComputationalCell3D &cell = new_cells[i];
			switch(gauss::Failure(status[i]))
			{
				case gauss::Ok:
					break;
				case gauss::UnresolvedNeighbor:
					for(size_t face : tess.GetCellFaces(i))
					{
						const size_t neighbor = tess.GetFaceNeighbors(face).first == i ? tess.GetFaceNeighbors(face).second :
							tess.GetFaceNeighbors(face).first;
						const size_t other_cell = data.stateIndex.Host()[neighbor];
						if(other_cell >= new_cells.size())
						{
							UniversalError error("Unable to resolve periodic neighbor in LinearGauss3D");
							error.addEntry("Cell", static_cast<double>(i));
							error.addEntry("Face", static_cast<double>(face));
							error.addEntry("Resolved neighbor", static_cast<double>(other_cell));
							error.addEntry("Cell array size", static_cast<double>(new_cells.size()));
							throw error;
						}
					}
					throw UniversalError("Unable to resolve periodic neighbor in LinearGauss3D");
				case gauss::BadNeighborDensity:
					throw UniversalError("Bad density getneighborcell");
				case gauss::SingularMatrix:
				{
					// The matrix, again, for the report
					std::vector<double> slope(size), naive(size);
					std::array<double, 10> matrix{};
					gauss::CellSlope(data.MeshView(Where::Host), data.StateView(Where::Host), config, i, 0.0, false,
						slope.data(), naive.data(), matrix.data());
					const Vector3D &cell_cm = tess.GetCellCM(i);
					UniversalError eo("Singular matrix");
					eo.addEntry("Cell x cor", center.x);
					eo.addEntry("Cell y cor", center.y);
					eo.addEntry("Cell z cor", center.z);
					eo.addEntry("Cell CMx cor", cell_cm.x);
					eo.addEntry("Cell CMy cor", cell_cm.y);
					eo.addEntry("Cell CMz cor", cell_cm.z);
					eo.addEntry("Cell volume", tess.GetVolume(i));
					eo.addEntry("Det was", matrix[9]);
					for(size_t face : tess.GetCellFaces(i))
					{
						const size_t other = tess.GetFaceNeighbors(face).first == i ? tess.GetFaceNeighbors(face).second :
							tess.GetFaceNeighbors(face).first;
						const Vector3D c_ij = tess.FaceCM(face) - 0.5 * (tess.GetCellCM(other) + cell_cm);
						eo.addEntry("Neighbor x", tess.GetMeshPoint(other).x);
						eo.addEntry("Neighbor y", tess.GetMeshPoint(other).y);
						eo.addEntry("Neighbor z", tess.GetMeshPoint(other).z);
						eo.addEntry("Face", static_cast<double>(face));
						eo.addEntry("Neighbor Cx", c_ij.x);
						eo.addEntry("Neighbor Cy", c_ij.y);
						eo.addEntry("Neighbor Cz", c_ij.z);
						eo.addEntry("Face Cx", tess.FaceCM(face).x);
						eo.addEntry("Face Cy", tess.FaceCM(face).y);
						eo.addEntry("Face Cz", tess.FaceCM(face).z);
						eo.addEntry("Face area", tess.GetArea(face));
					}
					for (size_t j = 0; j < 9; ++j)
						eo.addEntry("M", matrix[j]);
					throw eo;
				}
				case gauss::BadSlope:
				{
					UniversalError eo("Bad slope");
					eo.addEntry("boundary_slope", HasBoundaryFace(tess, i));
					throw eo;
				}
				case gauss::BadEos:
#ifdef RICH_DEBUG
					try
					{
#endif
						eos.de2c(cell.density, cell.internal_energy, cell.tracers, ComputationalCell3D::tracerNames);
						throw UniversalError("LinearGauss3D: the equation of state failed at a cell");
#ifdef RICH_DEBUG
					}
					catch (UniversalError &eo)
					{
						eo.addEntry("Error LinearGauss3D", 0);
						eo.addEntry("Cell number", i);
						throw eo;
					}
#endif
				case gauss::BadLimitedSlope:
				{
					UniversalError eo("Bad slope limited");
					eo.addEntry("boundary_slope", HasBoundaryFace(tess, i));
#ifdef RICH_DEBUG
					eo.addEntry("Error LinearGauss3D", 0);
					eo.addEntry("Cell number", i);
#endif
					throw eo;
				}
				default:
					throw UniversalError("LinearGauss3D: unknown slope status");
			}
		}
	}

	// Copies the cells' flat slopes into rslopes (sized for the cells); the unlimited ones wait for GetSlopesUnlimited
	void UnpackSlopes(LinearGaussArrays &data, vector<Slope3D> &rslopes)
	{
		const size_t size = 3 * data.config.fields;
		portable::HostParallelFor(data.cellsNum, [&](size_t i)
		{
			SetSlope(rslopes[i], data.slopes.Host().data() + i * size, data.config.fields, data.config.tracers);
		});
		data.naivePending = true;
	}

	// Copies the cells' unlimited slopes into naive_rslopes (sized for the cells)
	void UnpackNaiveSlopes(LinearGaussArrays &data, vector<Slope3D> &naive_rslopes)
	{
		if(data.naiveOnDevice)
		{
			data.naive.ToHost(0, data.naive.Host().size());
			data.naiveOnDevice = false;
		}
		const size_t size = 3 * data.config.fields;
		portable::HostParallelFor(data.cellsNum, [&](size_t i)
		{
			SetSlope(naive_rslopes[i], data.naive.Host().data() + i * size, data.config.fields, data.config.tracers);
		});
		data.naivePending = false;
	}

	// The flat slopes of the points from `first` on, from rslopes (which has the exchanged ghosts)
	void PackSlopes(LinearGaussArrays &data, vector<Slope3D> const& rslopes, size_t first)
	{
		const size_t size = 3 * data.config.fields;
		data.slopes.Host().resize(data.pointsNum * size, 0.0);
		const size_t end = std::max(first, std::min(data.pointsNum, rslopes.size()));
		portable::HostParallelFor(end - first, [&](size_t k)
		{
			GetSlope(rslopes[first + k], data.slopes.Host().data() + (first + k) * size, data.config.fields, data.config.tracers);
		});
		std::fill(data.slopes.Host().begin() + static_cast<std::ptrdiff_t>(end * size), data.slopes.Host().end(), 0.0);
	}

	// Both sides of each face (where it has a cell) in the flat arrays, with the equation of state applied
	void ComputeFaceValues(LinearGaussArrays &data, Tessellation3D const& tess, EquationOfState const& eos, Ghost3D const& ghost,
		vector<ComputationalCell3D> const& cells, vector<Slope3D> const& rslopes, double time, Where where)
	{
		const gauss::Config &config = data.config;
		const size_t CellNumber = data.cellsNum;
		const size_t faces = data.facesNum;
		const size_t size = 3 * config.fields;

		// The ghosts' slopes at the box
		vector<double> &wallSlopes = data.wallSlopes.Host();
		wallSlopes.resize(data.wallsNum * size);
		for(size_t f = 0; f < faces; ++f)
		{
			const size_t slot = data.wallSlot.Host()[f];
			if(slot == 0)
			{
				continue;
			}
			const std::pair<size_t, size_t> &neighbors = tess.GetFaceNeighbors(f);
			const size_t N0 = neighbors.first >= CellNumber ? neighbors.first : neighbors.second;
			GetSlope(ghost.GetGhostGradient(tess, cells, rslopes, N0, time, f), wallSlopes.data() + (slot - 1) * size,
				config.fields, config.tracers);
		}

		data.faceValues.Host().resize(2 * faces * config.fields);
		data.faceStatus.Host().resize(2 * faces);
		if(where == Where::Device)
		{
			if(not data.inputsOnDevice)
			{
				data.InputsToDevice();
			}
			data.slopes.ToDevice(data.slopesOnDevice * size, (data.pointsNum - data.slopesOnDevice) * size);
			data.slopesOnDevice = data.pointsNum;
			data.wallSlopes.ToDevice();
			data.faceValues.AllocateDevice();
			data.faceStatus.AllocateDevice();
		}
		gauss::FaceSides sides;
		sides.mesh = data.MeshView(where);
		sides.state = data.StateView(where);
		sides.config = config;
		sides.slopes = data.slopes.Data(where);
		sides.wallSlopes = data.wallSlopes.Data(where);
		sides.values = data.faceValues.Data(where);
		sides.status = data.faceStatus.Data(where);
		portable::ParallelFor(where, "LinearGauss3D::FaceSides", faces, sides);
		// The equation of state of the sides, then their checks
		if(config.pressureCalc)
		{
			eos.dp2eArrays(FieldArrays(2 * faces, config.fields, config.tracers, sides.values, gauss::Pressure,
				sides.values + gauss::InternalEnergy, config.fields, sides.status, gauss::BadEos), where);
		}
		else
		{
			eos.de2pArrays(FieldArrays(2 * faces, config.fields, config.tracers, sides.values, gauss::InternalEnergy,
				sides.values + gauss::Pressure, config.fields, sides.status, gauss::BadEos), where);
		}
		gauss::FaceChecks checks;
		checks.config = config;
		checks.values = sides.values;
		checks.status = sides.status;
		portable::ParallelFor(where, "LinearGauss3D::FaceChecks", 2 * faces, checks);
		data.faceValuesWhere = where;
		if(where == Where::Device)
		{
			data.faceValues.ToHost();
			data.faceStatus.ToHost();
		}
	}

	// A face side as a cell: its point's state, with the interpolated fields
	void SetSide(LinearGaussArrays &data, vector<ComputationalCell3D> const& new_cells, size_t face, size_t side,
		ComputationalCell3D &value)
	{
		const size_t point = data.faceNeighbors.Host()[2 * face + side];
		ReplaceComputationalCell(value, new_cells[point < data.cellsNum ? point : data.stateIndex.Host()[point]]);
		SetFields(value, data.faceValues.Host().data() + (2 * face + side) * data.config.fields, data.config.tracers);
	}

	ComputationalCell3D SideCell(LinearGaussArrays &data, vector<ComputationalCell3D> const& new_cells, size_t face, size_t side)
	{
		ComputationalCell3D value;
		SetSide(data, new_cells, face, side, value);
		return value;
	}

	/* The error of an equation of state that failed at a face: calls it again, on the host, with the face's values
	   (before the equation of state), and returns what it throws */
	UniversalError FaceEosError(EquationOfState const& eos, bool pressure_calc, ComputationalCell3D const& value)
	{
		try
		{
			if(pressure_calc)
			{
				eos.dp2e(value.density, value.pressure, value.tracers, ComputationalCell3D::tracerNames);
			}
			else
			{
				eos.de2p(value.density, value.internal_energy, value.tracers, ComputationalCell3D::tracerNames);
			}
		}
		catch(UniversalError &eo)
		{
			eo.addEntry("density", value.density);
			eo.addEntry("internal energy", value.internal_energy);
			return eo;
		}
		UniversalError eo("LinearGauss3D: the equation of state failed at a face");
		eo.addEntry("density", value.density);
		eo.addEntry("pressure", value.pressure);
		eo.addEntry("internal energy", value.internal_energy);
		return eo;
	}

	// Throws the first failed face side, in the order of the original loops: each cell's own faces, then the ghosts' sides
	void ReportFaceErrors(LinearGaussArrays &data, Tessellation3D const& tess, EquationOfState const& eos, bool pressure_calc,
		vector<ComputationalCell3D> const& new_cells, vector<Slope3D> const& rslopes)
	{
		const vector<std::uint8_t> &status = data.faceStatus.Host();
		const auto failed = [](std::uint8_t s) { return s == gauss::BadEos or s == gauss::BadFace; };
		if(std::none_of(status.begin(), status.end(), failed))
		{
			return;
		}
		const size_t CellNumber = data.cellsNum;
		for(size_t i = 0; i < CellNumber; ++i)
		{
			for(size_t face : tess.GetCellFaces(i))
			{
				const size_t side = tess.GetFaceNeighbors(face).first == i ? 0 : 1;
				if(not failed(status[2 * face + side]))
				{
					continue;
				}
				const ComputationalCell3D &source_cell = new_cells[i];
				const Slope3D &cell_slope = rslopes[i];
				const ComputationalCell3D value = SideCell(data, new_cells, face, side);
				UniversalError eo = status[2 * face + side] == gauss::BadEos ? FaceEosError(eos, pressure_calc, value) :
					UniversalError("Bad cell after interpolation in LinearGauss3D");
				eo.addEntry(side == 0 ? "dslope_x" : "dslope_x1",  cell_slope.xderivative.density);
				eo.addEntry(side == 0 ? "dslope_y" : "dslope_y1",  cell_slope.yderivative.density);
				eo.addEntry(side == 0 ? "dslope_z" : "dslope_z1",  cell_slope.zderivative.density);
				eo.addEntry("Old density", source_cell.density);
				eo.addEntry("Old internal energy", source_cell.internal_energy);
				eo.addEntry("Face", static_cast<double>(face));
				eo.addEntry("Cell", static_cast<double>(i));
				eo.addEntry("Vx", source_cell.velocity.x);
				eo.addEntry("Vy", source_cell.velocity.y);
				eo.addEntry("Vz", source_cell.velocity.z);
				eo.addEntry("Cell id", static_cast<double>(source_cell.ID));
				eo.addEntry("Interpolated density",value.density);
				eo.addEntry("Interpolated pressure",value.pressure);
				eo.addEntry("Interpolated internal energy",value.internal_energy);
				eo.addEntry("Interpolated Vx",value.velocity.x);
				eo.addEntry("Interpolated Vy",value.velocity.y);
				eo.addEntry("Interpolated Vz",value.velocity.z);
				throw eo;
			}
		}
		for(size_t i = 0; i < CellNumber; ++i)
		{
			for(size_t face : tess.GetCellFaces(i))
			{
				const size_t side = tess.GetFaceNeighbors(face).first == i ? 1 : 0;
				const size_t N0 = side == 0 ? tess.GetFaceNeighbors(face).first : tess.GetFaceNeighbors(face).second;
				if(N0 < CellNumber or not failed(status[2 * face + side]))
				{
					continue;
				}
				const size_t stateIndex = data.stateIndex.Host()[N0];
				const ComputationalCell3D value = SideCell(data, new_cells, face, side);
				UniversalError eo = status[2 * face + side] == gauss::BadEos ? FaceEosError(eos, pressure_calc, value) :
					UniversalError("Bad cell after interpolation in LinearGauss3D");
				eo.addEntry("old density", new_cells[stateIndex].density);
				eo.addEntry("old internal energy", new_cells[stateIndex].internal_energy);
				eo.addEntry("Boundary Face", static_cast<double>(face));
				eo.addEntry("Cell", static_cast<double>(N0));
				eo.addEntry("Vx", new_cells[stateIndex].velocity.x);
				eo.addEntry("Vy", new_cells[stateIndex].velocity.y);
				eo.addEntry("Vz", new_cells[stateIndex].velocity.z);
				eo.addEntry("Cell id", static_cast<double>(new_cells[stateIndex].ID));
				eo.addEntry("Interpolated density",value.density);
				eo.addEntry("Interpolated pressure",value.pressure);
				eo.addEntry("Interpolated internal energy",value.internal_energy);
				eo.addEntry("Interpolated Vx",value.velocity.x);
				eo.addEntry("Interpolated Vy",value.velocity.y);
				eo.addEntry("Interpolated Vz",value.velocity.z);
				eo.addEntry("Face CMx", tess.FaceCM(face).x);
				eo.addEntry("Face CMy", tess.FaceCM(face).y);
				eo.addEntry("Face CMz", tess.FaceCM(face).z);
				eo.addEntry("Cell CMx", tess.GetCellCM(N0).x);
				eo.addEntry("Cell CMy", tess.GetCellCM(N0).y);
				eo.addEntry("Cell CMz", tess.GetCellCM(N0).z);
				const size_t N1 = side == 0 ? tess.GetFaceNeighbors(face).second : tess.GetFaceNeighbors(face).first;
				eo.addEntry("Other cell ID", static_cast<double>(new_cells[N1].ID));
				eo.addEntry("Other Cell CMx", tess.GetCellCM(N1).x);
				eo.addEntry("Other Cell CMy", tess.GetCellCM(N1).y);
				eo.addEntry("Other Cell CMz", tess.GetCellCM(N1).z);
				eo.addEntry("Other Cell density", new_cells[N1].density);
				eo.addEntry("Other Cell pressure", new_cells[N1].pressure);
				if(side == 0)
				{
					eo.addEntry("Slopex", rslopes[stateIndex].xderivative.density);
					eo.addEntry("Slopey", rslopes[stateIndex].yderivative.density);
					eo.addEntry("Slopez", rslopes[stateIndex].zderivative.density);
				}
#ifdef RICH_MPI
				int rank = 0;
				MPI_Comm_rank(MPI_COMM_WORLD, &rank);
				eo.addEntry("Rank", static_cast<double>(rank));
				for (size_t j = 0; j < tess.GetGhostIndeces().size(); ++j)
					for (size_t k = 0; k < tess.GetGhostIndeces()[j].size(); ++k)
						if (tess.GetGhostIndeces()[j][k] == N0)
						{
							eo.addEntry("Point recv from proc", static_cast<double>(tess.GetDuplicatedProcs()[j]));
							eo.addEntry("Point recv index", static_cast<double>(k));
							eo.addEntry("Point proc index", static_cast<double>(j));
						}
#endif
				throw eo;
			}
		}
	}

#ifdef RICH_MPI

	void exchange_ghost_slopes(Tessellation3D const& tess, vector<Slope3D> & slopes)
	{
		Slope3D sdummy;
		MPI_exchange_data(tess, slopes, true);
	}
#endif//RICH_MPI
}

void LinearGauss3D::Interp(ComputationalCell3D &res, ComputationalCell3D const& cell, size_t cell_index, Vector3D const& cm,
	Vector3D const& target, EquationOfState const& eos)const
{
	try
	{
		res = interp(cell, rslopes_[cell_index], target, cm, eos, true);
	}
	catch (UniversalError &eo)
	{
		eo.addEntry("Cell density", cell.density);
		eo.addEntry("Cell internal energy", cell.internal_energy);
		eo.addEntry("cell index", static_cast<double>(cell_index));
		eo.addEntry("CMx", cm.x);
		eo.addEntry("CMy", cm.y);
		eo.addEntry("CMz", cm.z);
		eo.addEntry("Targetx", target.x);
		eo.addEntry("Targety", target.y);
		eo.addEntry("Targetz", target.z);
		throw eo;
	}
}

LinearGauss3D::LinearGauss3D(EquationOfState const& eos, Ghost3D const& ghost, bool slf, double delta_v, double theta,
	double delta_P, bool SR, const vector<string>& calc_tracers, const string& skip_key,
	bool pressure_calc, bool apply_principal_limit) : eos_(eos), ghost_(ghost), rslopes_(),
	naive_rslopes_(), slf_(slf), shockratio_(delta_v), diffusecoeff_(theta), pressure_ratio_(delta_P), SR_(SR),
	calc_tracers_(calc_tracers), skip_key_(skip_key), to_skip_(), pressure_calc_(pressure_calc),
	apply_principal_limit_(apply_principal_limit), onDevice_(false) {}

void LinearGauss3D::RunOnDevice(bool onDevice)
{
	onDevice_ = onDevice;
}

void LinearGauss3D::BuildSlopes(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, double time) 
{
	const size_t CellNumber = tess.GetPointNo();
	// Get ghost points
	boost::container::flat_map<size_t, ComputationalCell3D> ghost_cells;
	ghost_.operator()(tess, cells, time, ghost_cells);
	// Reuse persistent buffer instead of allocating a new vector each call
	CopyCells(new_cells_, cells, tess.GetTotalPointNumber());
	for (boost::container::flat_map<size_t, ComputationalCell3D>::const_iterator it = ghost_cells.begin(); it !=
		ghost_cells.end(); ++it)
		new_cells_[it->first] = it->second;
	const vector<bool> mpiGhostMask = BuildMPIGhostMask(tess);
	FillLocalPeriodicImagePrimitives(tess, CellNumber, mpiGhostMask, new_cells_);
	if (SR_)
	{
		size_t Nall = new_cells_.size();
		for (size_t j = 0; j < Nall; ++j)
		{
			double gamma = 1.0 / std::sqrt(1 - ScalarProd(new_cells_[j].velocity, new_cells_[j].velocity));
			new_cells_[j].velocity *= gamma;
		}
	}
	if (CellNumber > 0)
		rslopes_.resize(CellNumber, Slope3D(cells[0], cells[0], cells[0]));
	else
		rslopes_.clear();
	conditional_shrink(rslopes_);
	naive_rslopes_.resize(CellNumber);
	conditional_shrink(naive_rslopes_);
	LinearGaussArrays &data = KeptArrays(arrays_);
	data.config = MakeConfig(calc_tracers_, slf_, shockratio_, diffusecoeff_, pressure_ratio_, skip_key_, pressure_calc_,
		apply_principal_limit_);
	PackMeshAndStates(data, tess, new_cells_, mpiGhostMask, skip_key_);
	ComputeSlopes(data, tess, eos_, WhereToRun(onDevice_, eos_), new_cells_);
	UnpackSlopes(data, rslopes_);
#ifdef RICH_MPI
	// communicate ghost slopes
	exchange_ghost_slopes(tess, rslopes_);
#endif //RICH_MPI
}

void LinearGauss3D::operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, double time,
	vector<pair<ComputationalCell3D, ComputationalCell3D> > &res) const
{
	const size_t CellNumber = tess.GetPointNo();
	// Get ghost points
	boost::container::flat_map<size_t, ComputationalCell3D> ghost_cells;
	ghost_.operator()(tess, cells, time, ghost_cells);
	// Reuse persistent buffer instead of allocating a new vector each call
	CopyCells(new_cells_, cells, tess.GetTotalPointNumber());
	for (boost::container::flat_map<size_t, ComputationalCell3D>::const_iterator it = ghost_cells.begin(); it !=
		ghost_cells.end(); ++it)
		new_cells_[it->first] = it->second;
	const vector<bool> mpiGhostMask = BuildMPIGhostMask(tess);
	FillLocalPeriodicImagePrimitives(tess, CellNumber, mpiGhostMask, new_cells_);
	if (SR_)
	{
		size_t Nall = new_cells_.size();
		for (size_t j = 0; j < Nall; ++j)
		{
			double gamma = 1.0 / std::sqrt(1 - ScalarProd(new_cells_[j].velocity, new_cells_[j].velocity));
			new_cells_[j].velocity *= gamma;
		}
	}
	if (CellNumber > 0)
		rslopes_.resize(CellNumber, Slope3D(cells[0], cells[0], cells[0]));
	else
		rslopes_.clear();
	conditional_shrink(rslopes_);
	naive_rslopes_.resize(CellNumber);
	conditional_shrink(naive_rslopes_);
	const Where where = WhereToRun(onDevice_, eos_);
	LinearGaussArrays &data = KeptArrays(arrays_);
	data.config = MakeConfig(calc_tracers_, slf_, shockratio_, diffusecoeff_, pressure_ratio_, skip_key_, pressure_calc_,
		apply_principal_limit_);
	PackMeshAndStates(data, tess, new_cells_, mpiGhostMask, skip_key_);
	ComputeSlopes(data, tess, eos_, where, new_cells_);
	UnpackSlopes(data, rslopes_);
#ifdef RICH_MPI
	// communicate ghost slopes
	exchange_ghost_slopes(tess, rslopes_);
#endif //RICH_MPI
	PackSlopes(data, rslopes_, CellNumber);
	ComputeFaceValues(data, tess, eos_, ghost_, cells, rslopes_, time, where);
	data.sr = SR_;
	ReportFaceErrors(data, tess, eos_, pressure_calc_, new_cells_, rslopes_);

	if (CellNumber > 0)
		res.resize(tess.GetTotalFacesNumber(), pair<ComputationalCell3D, ComputationalCell3D>(cells[0], cells[0]));
	else
		res.clear();
	// Every side is written, those without a value as a new element (res may hold the last call's face values)
	portable::HostParallelFor(data.facesNum, [&](size_t face)
	{
		for (size_t side = 0; side < 2; ++side)
		{
			ComputationalCell3D &value = side == 0 ? res[face].first : res[face].second;
			if (data.faceStatus.Host()[2 * face + side] != gauss::NoSide)
				SetSide(data, new_cells_, face, side, value);
			else
				value = cells[0];
		}
	});
	//In SR convert back to velocities
	if (SR_)
	{
		size_t N = res.size();
		for (size_t i = 0; i < N; ++i)
		{
			double factor = 1.0 / std::sqrt(1 + ScalarProd(res[i].first.velocity, res[i].first.velocity));
			res[i].first.velocity *= factor;
			factor = 1.0 / std::sqrt(1 + ScalarProd(res[i].second.velocity, res[i].second.velocity));
			res[i].second.velocity *= factor;
		}
	}
}


vector<Slope3D>& LinearGauss3D::GetSlopes(void)
{
	return rslopes_;
}

vector<Slope3D>& LinearGauss3D::GetSlopesUnlimited(void)const
{
	if(arrays_ and arrays_->naivePending)
		UnpackNaiveSlopes(*arrays_, naive_rslopes_);
	return naive_rslopes_;
}

std::vector<double> LinearGauss3D::CalcDissipationStreamingFromPreparedSlopes(
	const Tessellation3D& tess,
	const std::vector<ComputationalCell3D>& cells,
	double time,
	Hllc3D const& rs,
	const EquationOfState& eos) const
{
	const size_t N = tess.GetPointNo();
	const size_t Nfaces = tess.GetTotalFacesNumber();
	std::vector<double> result(N, 0.0);
	const vector<bool> mpiGhostMask = BuildMPIGhostMask(tess);

	LinearGaussArrays &data = KeptArrays(arrays_);
	data.config = MakeConfig(calc_tracers_, slf_, shockratio_, diffusecoeff_, pressure_ratio_, skip_key_, pressure_calc_,
		apply_principal_limit_);
	PackMeshAndStates(data, tess, new_cells_, mpiGhostMask, skip_key_);
	PackSlopes(data, rslopes_, 0);
	ComputeFaceValues(data, tess, eos_, ghost_, cells, rslopes_, time, WhereToRun(onDevice_, eos_));

	auto reconstruct_side = [&](ComputationalCell3D& out, size_t face_idx, size_t cell_idx)
	{
		const size_t side = data.faceNeighbors.Host()[2 * face_idx] == cell_idx ? 0 : 1;
		out = SideCell(data, new_cells_, face_idx, side);
		if(data.faceStatus.Host()[2 * face_idx + side] == gauss::BadEos)
			throw FaceEosError(eos_, pressure_calc_, out);
	};

	ComputationalCell3D left, right;

	for(size_t i = 0; i < Nfaces; ++i)
	{
		auto neigh = tess.GetFaceNeighbors(i);
		bool is_first = neigh.first < N;
		bool is_second = neigh.second < N;
		if(!is_first && !is_second)
			continue;

		reconstruct_side(left, i, neigh.first);
		reconstruct_side(right, i, neigh.second);

		if(SR_)
		{
			double factor = 1.0 / std::sqrt(1 + ScalarProd(left.velocity, left.velocity));
			left.velocity *= factor;
			factor = 1.0 / std::sqrt(1 + ScalarProd(right.velocity, right.velocity));
			right.velocity *= factor;
		}

		const Vector3D normal = normalize(tess.Normal(i));
		const double A = tess.GetArea(i);
		auto UstarPstar = rs.GetUstarPstar(left, right, eos, normal);

		if(is_first)
			result[neigh.first] -= A * ((UstarPstar.first - ScalarProd(normal, cells[neigh.first].velocity)) * UstarPstar.second
				- cells[neigh.first].pressure * UstarPstar.first);
		if(is_second)
			result[neigh.second] += A * ((UstarPstar.first - ScalarProd(normal, cells[neigh.second].velocity)) * UstarPstar.second
				- cells[neigh.second].pressure * UstarPstar.first);
	}

	for(size_t i = 0; i < N; ++i)
		result[i] /= tess.GetVolume(i);

	return result;
}

