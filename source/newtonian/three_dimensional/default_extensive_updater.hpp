/*! \file default_extensive_updater.hpp
  \brief Generates extensive conserved variables
  \author Almog Yalinewich
 */

#ifndef DEFAULT_EXTENSIVE_UPDATER_HPP
#define DEFAULT_EXTENSIVE_UPDATER_HPP 1

#include "conserved_3d.hpp"
#include "extensive_updater3d.hpp"
#include <cstdint>
#include <string>

// Bits: Erad=1, radiation derivatives=2/4, groups=8, mass=16,
// nonfinite total energy=32, tracked thermal energy=64, momentum=128.
// Negative finite thermal energy remains eligible for entropy recovery.
std::uint64_t IndividualHydroInvalidComponentMask(Conserved3D const& state);

void PersistIndividualHydroDiagnosticRecord(const std::string& record_id,
	const std::string& record);

//! \brief Generates a list of conserved variables
class DefaultExtensiveUpdater: public ExtensiveUpdater3D
{
public:

  /*! \brief Class constructor
   */
	DefaultExtensiveUpdater(void);

	void operator()(const vector<Conserved3D>& fluxes, const Tessellation3D& tess,
		const double dt, const vector<ComputationalCell3D>& cells, vector<Conserved3D>& extensives, double time, const vector<Vector3D>& edge_velocities,
		const vector<Vector3D>& point_velocities,
		std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const& interp_values) const override;

	bool SupportsIndividualTimeSteps(void) const override{return true;}

	void UpdateIndividual(const vector<Conserved3D>& fluxes,
		const Tessellation3D& tess,
		const IndividualStepContext& context,
		const vector<ComputationalCell3D>& cells,
		vector<Conserved3D>& extensives,
		double time,
		const vector<Vector3D>& edge_velocities,
		const vector<Vector3D>& point_velocities,
		const std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >& interp_values,
		const vector<ComputationalCell3D>* canonical_cells = nullptr,
		vector<Conserved3D>* canonical_extensives = nullptr) const override;
private:
	mutable std::vector<double> oldEk_, oldEtherm_, oldE_;
};

#endif // DEFAULT_EXTENSIVE_UPDATER_HPP
