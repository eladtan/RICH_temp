#include "default_cell_updater.hpp"
#include "../../misc/utils.hpp"
#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif

namespace
{
	void EntropyFix(EquationOfState const& eos, ComputationalCell3D &res, size_t entropy_index, double &energy,
		Conserved3D &extensive, double Et_min)
	{
		if(!(res.density > 0) || !(res.tracers[entropy_index] > 0))
		{
			UniversalError eo("Negative EntropyFix");
			eo.addEntry("entropy", res.tracers[entropy_index]);
			eo.addEntry("density", res.density);
			throw eo;
		}
		double de = 0;
		try
		{
	  		double new_pressure = eos.sd2p(res.tracers[entropy_index], res.density, res.tracers, ComputationalCell3D::tracerNames);
			res.pressure = new_pressure;
			double new_energy = std::max(Et_min, eos.dp2e(res.density, new_pressure));
			if(new_energy == Et_min)
			{
				res.pressure = eos.de2p(res.density, new_energy);
				double new_entropy = eos.dp2s(res.density, res.pressure, res.tracers, ComputationalCell3D::tracerNames);
				res.tracers[entropy_index] = new_entropy;
				extensive.tracers[entropy_index] = new_entropy * extensive.mass;
			}
			de = new_energy - energy;
			energy += de;
			res.internal_energy = energy;
			extensive.energy += de * extensive.mass;
			extensive.internal_energy += de * extensive.mass;
		}
		catch(UniversalError const& eo)
		{
			de = Et_min - energy;
			energy = Et_min;
			res.pressure = eos.de2p(res.density, energy);
			res.internal_energy = energy;
			extensive.energy += de * extensive.mass;
			extensive.internal_energy += de * extensive.mass;
		}
		
		if(!(energy > 0))
		{
			UniversalError eo("negative thermal energu in entropy fix");
			eo.addEntry("de", de);
			eo.addEntry("energy", energy);
			eo.addEntry("entropy", res.tracers[entropy_index]);
			eo.addEntry("density", res.density);
			throw eo;
		}
	}

	void EntropyFixSR(EquationOfState const& eos, ComputationalCell3D &res, size_t entropy_index, Conserved3D &extensive, double vol)
	{
	  double new_pressure = eos.sd2p(res.tracers[entropy_index], res.density, res.tracers, ComputationalCell3D::tracerNames);
		res.pressure = new_pressure;
		double energy = eos.dp2e(res.density, res.pressure);
		double gamma = std::sqrt(1.0 / (1 - ScalarProd(res.velocity, res.velocity)));
		extensive.energy = extensive.mass*(energy*gamma + gamma - 1) - res.pressure*vol;
	}

	bool HighRelativeKineticEnergy(Tessellation3D const& tess, size_t index, vector<Conserved3D> const& cells,
		Conserved3D const& cell)
	{
		std::vector<size_t> neigh;
		tess.GetNeighbors(index, neigh);
		size_t N = neigh.size();
		double maxDV = 0;
		size_t Norg = tess.GetPointNo();
		Vector3D Vcell = cell.momentum / cell.mass;
		double Et = cell.energy / cell.mass - 0.5*ScalarProd(Vcell, Vcell);
		double div_V = 0;
		Vector3D const r_i = tess.GetMeshPoint(index);
		auto faces = tess.GetCellFaces(index);
		for (size_t i = 0; i < N; ++i)
		{
			size_t neighbor_j = neigh[i];
			if (neighbor_j < Norg || !tess.IsPointOutsideBox(neighbor_j))
			{
				auto const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
				double const A_ij = tess.GetArea(faces[i]);
				Vector3D velocity_j = cells.at(neighbor_j).momentum / cells[neighbor_j].mass;
				div_V -= 0.5*ScalarProd(Vcell+velocity_j, r_ij) * A_ij;
				maxDV = std::max(maxDV, abs(Vcell - velocity_j));
			}
		}
		div_V /= tess.GetVolume(index);
		if(div_V < 0 && (div_V * tess.GetWidth(index) < fastabs(Vcell) * 0.05))
			return false;
		return 0.005*maxDV*maxDV > Et;
	}

	void regular_update(std::vector<ComputationalCell3D> &res, std::vector<Conserved3D> & extensives,
		Tessellation3D const& tess, size_t entropy_index,
		EquationOfState const& eos, bool const includes_temperature, const RadiationDriver* diffusion, double const min_temperature)
	{
		size_t Nloop = tess.GetPointNo();
		size_t print_id = -1;
		size_t Ntracers = ComputationalCell3D::tracerNames.size();
		for (size_t i = 0; i < Nloop; ++i)
		{
			try
			{
				Conserved3D& extensive = extensives[i];
				const double vol = tess.GetVolume(i);
				res[i].density = extensive.mass / vol;
				double d_factor = 1;
				if(res[i].density > 0 && res[i].density < 2e-21)
				{
					d_factor = 2e-21 / res[i].density;
					res[i].density = 2e-21;
					extensive.mass *= d_factor;
					extensive.momentum *= d_factor;
					extensive.internal_energy *= d_factor;
					for (size_t j = 0; j < Ntracers; ++j)
						extensive.tracers[j] *= d_factor;
				}
				double const old_etherm = res[i].internal_energy;
				res[i].velocity = extensive.momentum / extensive.mass;
				double energy = extensive.internal_energy / extensive.mass;
				// extensive.energy = extensive.mass*(energy + 0.5*ScalarProd(res[i].velocity, res[i].velocity));
				for (size_t j = 0; j < Ntracers; ++j)
					res[i].tracers[j] = extensive.tracers[j] / extensive.mass;
				// Entropy fix if needed
				if (entropy_index < ComputationalCell3D::tracerNames.size())
				{
					double Et_min = 0;
					try
					{
						if(min_temperature > 0)
						{
							Et_min = eos.dT2e(res[i].density, min_temperature, res[i].tracers, ComputationalCell3D::tracerNames);
							if(energy < Et_min)
							{
								res[i].temperature = min_temperature;
								res[i].internal_energy = Et_min;
								extensive.energy += (res[i].internal_energy - energy) * extensive.mass;
								extensive.internal_energy += (res[i].internal_energy - energy) * extensive.mass;
								energy = Et_min;
								res[i].pressure = eos.de2p(res[i].density, res[i].internal_energy, res[i].tracers, ComputationalCell3D::tracerNames);
								double new_entropy = eos.dp2s(res[i].density, res[i].pressure, res[i].tracers, ComputationalCell3D::tracerNames);
								res[i].tracers[entropy_index] = new_entropy;
								extensive.tracers[entropy_index] = new_entropy * extensive.mass;
							}
						}
						// Do we have a negative thermal energy?
						if (energy < 0)
						{
							if(print_id == res[i].ID)
								std::cout<<"Entered entropy fix1"<<std::endl;
							EntropyFix(eos, res[i], entropy_index, energy, extensive, Et_min);
						}
						else
						{
							// Is the kinetic energy small?
							if ((energy*extensive.mass < 0.005*extensive.energy) &&
								HighRelativeKineticEnergy(tess, i, extensives, extensive))
							{
								if(print_id == res[i].ID)
									std::cout<<"Entered entropy fix2"<<std::endl;
								EntropyFix(eos, res[i], entropy_index, energy, extensive, Et_min);
							}
							else
							{
								double new_pressure = eos.de2p(res[i].density, energy, res[i].tracers, ComputationalCell3D::tracerNames);
							  	double new_entropy = eos.dp2s(res[i].density, new_pressure, res[i].tracers, ComputationalCell3D::tracerNames);
								// We don't need the entropy fix, update entropy
								res[i].internal_energy = energy;
								res[i].pressure = new_pressure;
								res[i].tracers[entropy_index] = new_entropy;
								extensive.tracers[entropy_index] = new_entropy * extensive.mass;
							}
						}
					}
					catch (UniversalError &eo)
					{
						eo.addEntry("Cell index", static_cast<double>(i));
						eo.addEntry("Cell mass", extensives[i].mass);
						eo.addEntry("Cell x momentum", extensives[i].momentum.x);
						eo.addEntry("Cell y momentum", extensives[i].momentum.y);
						eo.addEntry("Cell z momentum", extensives[i].momentum.z);
						eo.addEntry("Cell x location", tess.GetMeshPoint(i).x);
						eo.addEntry("Cell y location", tess.GetMeshPoint(i).y);
						eo.addEntry("Cell z location", tess.GetMeshPoint(i).z);
						eo.addEntry("cell Temperature", res[i].temperature);
						eo.addEntry("Cell volume", vol);
						eo.addEntry("Cell energy", extensives[i].energy);
						eo.addEntry("Cell thermal energy per unit mass", energy);
						eo.addEntry("Et_min", Et_min);
						eo.addEntry("old_etherm", old_etherm);
						eo.addEntry("Cell id", static_cast<double>(res[i].ID));
						for(size_t j = 0; j <ComputationalCell3D::tracerNames.size(); ++j)
							eo.addEntry(ComputationalCell3D::tracerNames[j], extensives[i].tracers[j]);
						throw eo;
					}
				}
				else
				{
				  	res[i].pressure = eos.de2p(res[i].density, energy, res[i].tracers, ComputationalCell3D::tracerNames);
					res[i].internal_energy = energy;
				}
				if(includes_temperature)
				{
					if(diffusion != nullptr)
					{
						res[i].Erad = extensive.Erad / extensive.mass;
						if(!(res[i].Erad > 0))
						{
							UniversalError eo("Negative quantity in cell update");
							eo.addEntry("Cell index", static_cast<double>(i));
							eo.addEntry("extensive.Erad", extensive.Erad);
							eo.addEntry("Cell mass", extensives[i].mass);
							eo.addEntry("Cell x momentum", extensives[i].momentum.x);
							eo.addEntry("Cell y momentum", extensives[i].momentum.y);
							eo.addEntry("Cell z momentum", extensives[i].momentum.z);
							eo.addEntry("Cell x location", tess.GetMeshPoint(i).x);
							eo.addEntry("Cell y location", tess.GetMeshPoint(i).y);
							eo.addEntry("Cell z location", tess.GetMeshPoint(i).z);
							eo.addEntry("Cell volume", vol);
							eo.addEntry("Cell energy", extensives[i].energy);
							eo.addEntry("Cell thermal energy per unit mass", energy);
							eo.addEntry("Cell id", static_cast<double>(res[i].ID));
							throw eo;
						}
						

						// update group energies
						for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
							res[i].Eg[g] = extensive.Eg[g] / extensive.mass;
						}
					}
					res[i].temperature = eos.de2T(res[i].density, res[i].internal_energy, res[i].tracers, ComputationalCell3D::tracerNames);
					if(min_temperature > 0 && res[i].temperature < min_temperature)
					{
						res[i].temperature = min_temperature;
						double old_e = res[i].internal_energy;
                        res[i].internal_energy = eos.dT2e(res[i].density, res[i].temperature, res[i].tracers, ComputationalCell3D::tracerNames);
						extensive.energy += (res[i].internal_energy - old_e) * extensive.mass;
						extensive.internal_energy += (res[i].internal_energy - old_e) * extensive.mass;
                        res[i].pressure = eos.de2p(res[i].density, res[i].internal_energy, res[i].tracers, ComputationalCell3D::tracerNames);
                        if (entropy_index < ComputationalCell3D::tracerNames.size())
						{
							double new_entropy = eos.dp2s(res[i].density, res[i].pressure, res[i].tracers, ComputationalCell3D::tracerNames);
							res[i].tracers[entropy_index] = new_entropy;
							extensive.tracers[entropy_index] = new_entropy * extensive.mass;
						}
					}
				}
				if (!(res[i].density > 0) || !(res[i].pressure > 0) || (!std::isfinite(fastabs(extensives[i].momentum))))
				{
					UniversalError eo("Negative quantity in cell update");
					eo.addEntry("Cell index", static_cast<double>(i));
					eo.addEntry("Cell mass", extensives[i].mass);
					eo.addEntry("Cell x momentum", extensives[i].momentum.x);
					eo.addEntry("Cell y momentum", extensives[i].momentum.y);
					eo.addEntry("Cell z momentum", extensives[i].momentum.z);
					eo.addEntry("Cell x location", tess.GetMeshPoint(i).x);
					eo.addEntry("Cell y location", tess.GetMeshPoint(i).y);
					eo.addEntry("Cell z location", tess.GetMeshPoint(i).z);
					eo.addEntry("Cell volume", vol);
					eo.addEntry("Cell energy", extensives[i].energy);
					eo.addEntry("Cell thermal energy per unit mass", energy);
					eo.addEntry("Cell id", static_cast<double>(res[i].ID));
					throw eo;
				}
			}
			catch (UniversalError &eo)
			{
				eo.addEntry("Cell index", static_cast<double>(i));
				eo.addEntry("Cell mass", extensives[i].mass);
				eo.addEntry("Cell x momentum", extensives[i].momentum.x);
				eo.addEntry("Cell y momentum", extensives[i].momentum.y);
				eo.addEntry("Cell z momentum", extensives[i].momentum.z);
				eo.addEntry("Cell energy", extensives[i].energy);
				throw eo;
			}
		}
	}

	void regular_updateSR(std::vector<ComputationalCell3D> &res, std::vector<Conserved3D> & extensives,
		Tessellation3D const& tess, size_t entropy_index,
		EquationOfState const& eos, double G)
	{
		size_t Nloop = tess.GetPointNo();
		size_t Ntracers = ComputationalCell3D::tracerNames.size();
		for (size_t i = 0; i < Nloop; ++i)
		{
			try
			{
				double v = GetVelocity(extensives[i], G);
				const double volume = 1.0 / tess.GetVolume(i);
				if (fastabs(extensives[i].momentum)*1e8 < extensives[i].mass)
				{
					res[i].velocity = extensives[i].momentum / extensives[i].mass;
					//v = abs(res[i].velocity);
				}
				else
					res[i].velocity = v * extensives[i].momentum / abs(extensives[i].momentum);
				double gamma_1 = std::sqrt(1 - ScalarProd(res[i].velocity, res[i].velocity));
				res[i].density = extensives[i].mass *gamma_1*volume;
				//				double r = fastabs(tess.GetMeshPoint(i));
				if (res[i].density < 0)
					throw UniversalError("Negative density");
				for (size_t j = 0; j < extensives[i].tracers.size(); ++j)
					res[i].tracers[j] = extensives[i].tracers[j] / extensives[i].mass;
				if (fastabs(res[i].velocity) < 1e-5)
					res[i].pressure = (G - 1)*((extensives[i].energy - ScalarProd(extensives[i].momentum, res[i].velocity))*volume
						+ (0.5*ScalarProd(res[i].velocity, res[i].velocity))*res[i].density);
				else
					res[i].pressure = (G - 1)*(extensives[i].energy*volume - ScalarProd(extensives[i].momentum, res[i].velocity)*volume
						+ (1.0 / gamma_1 - 1)*res[i].density);
				// Entropy fix if needed
				if (entropy_index < Ntracers)
				{
					if (!(extensives[i].tracers[entropy_index] > 0))
					{
						UniversalError eo("Negative entropy");
						eo.addEntry("Extensive Entropy", extensives[i].tracers[entropy_index]);
						throw eo;
					}
					// Do we have a negative thermal energy?
					if (res[i].pressure < 0)
					{
						EntropyFixSR(eos, res[i], entropy_index, extensives[i], 1.0 / volume);
					}
					else
					{
					  double new_entropy = eos.dp2s(res[i].density, res[i].pressure, res[i].tracers, ComputationalCell3D::tracerNames);
						// We don't need the entropy fix, update entropy
						res[i].tracers[entropy_index] = new_entropy;
						extensives[i].tracers[entropy_index] = new_entropy * extensives[i].mass;
					}
				}
				if (!(res[i].density > 0) || !(res[i].pressure > 0)||(!std::isfinite(fastabs(extensives[i].momentum))))
				{
					UniversalError eo("Negative quantity in cell update");
					throw eo;
				}
				res[i].internal_energy = eos.dp2e(res[i].density, res[i].pressure,res[i].tracers,ComputationalCell3D::tracerNames);
			}
			catch (UniversalError &eo)
			{
			  eo.addEntry("Cell ID", static_cast<double>(res[i].ID));
				eo.addEntry("Cell volume", tess.GetVolume(i));
				eo.addEntry("Cell index", static_cast<double>(i));
				eo.addEntry("Cell mass", extensives[i].mass);
				eo.addEntry("Cell x momentum", extensives[i].momentum.x);
				eo.addEntry("Cell y momentum", extensives[i].momentum.y);
				eo.addEntry("Cell z momentum", extensives[i].momentum.z);
				eo.addEntry("Cell x location", tess.GetMeshPoint(i).x);
				eo.addEntry("Cell y location", tess.GetMeshPoint(i).y);
				eo.addEntry("Cell z location", tess.GetMeshPoint(i).z);
				eo.addEntry("Cell energy", extensives[i].energy);
				eo.addEntry("Number of tracers", static_cast<double>(Ntracers));
				for (size_t j = 0; j < Ntracers; ++j)
					eo.addEntry("Tracer extensive value", extensives[i].tracers[j]);
				eo.addEntry("Entropy index", static_cast<double>(entropy_index));
				throw;
			}
		}
	}

}

DefaultCellUpdater::DefaultCellUpdater(bool SR, double G, bool const includes_temperature, double const min_temperature, const RadiationDriver* diffusion) :SR_(SR), G_(G), includes_temperature_(includes_temperature), min_temperature_(min_temperature), diffusion_(diffusion), entropy_index_(9999999) {}

void DefaultCellUpdater::operator()(vector<ComputationalCell3D> &res, EquationOfState const& eos,
	const Tessellation3D& tess, vector<Conserved3D>& extensives) const
{
  entropy_index_ = ComputationalCell3D::tracerNames.size();
  vector<string>::const_iterator it = binary_find(ComputationalCell3D::tracerNames.begin(),
						  ComputationalCell3D::tracerNames.end(), string("Entropy"));
  if (it != ComputationalCell3D::tracerNames.end())
    entropy_index_ = static_cast<size_t>(it - ComputationalCell3D::tracerNames.begin());
#ifdef RICH_MPI
  if (entropy_index_ < ComputationalCell3D::tracerNames.size())
	{
		MPI_exchange_data(tess, extensives, true);
	}
#endif
	if (!SR_)
		regular_update(res, extensives, tess, entropy_index_, eos, includes_temperature_, diffusion_, min_temperature_);
	else
		regular_updateSR(res, extensives, tess, entropy_index_, eos, G_);
#ifdef RICH_MPI
	if (entropy_index_ < ComputationalCell3D::tracerNames.size())
		extensives.resize(tess.GetPointNo());
#endif

}
