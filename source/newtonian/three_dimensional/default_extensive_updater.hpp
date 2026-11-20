/*! \file default_extensive_updater.hpp
  \brief Generates extensive conserved variables
  \author Almog Yalinewich
 */

#ifndef DEFAULT_EXTENSIVE_UPDATER_HPP
#define DEFAULT_EXTENSIVE_UPDATER_HPP 1

#include "conserved_3d.hpp"
#include "extensive_updater3d.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <typeinfo>
#include <vector>

// Bits: Erad=1, radiation derivatives=2/4, groups=8, mass=16,
// nonfinite total energy=32, tracked thermal energy=64, momentum=128.
// Negative finite thermal energy remains eligible for entropy recovery.
std::uint64_t IndividualHydroInvalidComponentMask(Conserved3D const& state);

void PersistIndividualHydroDiagnosticRecord(const std::string& record_id,
	const std::string& record);

/*! \brief RICH_INDIVIDUAL_EXTENSIVE_LOG_FORWARD (B1; unset, 0, 1 or 2; default 0).
  \details 0: the individual hydro step replaces its owned-size pre-flux snapshot by the updater's canonical write log
  only for an updater of exact type DefaultExtensiveUpdater, and ConditionExtensiveUpdater3D builds a fresh inner
  updater per call and runs its own validation scan.  1: any updater offering a complete log
  (IndividualCanonicalWriteLog) replaces the snapshot; ConditionExtensiveUpdater3D keeps its inner updater and, when
  no rank has a condition/action pair, skips its duplicate validation scan and reduction.  2: verify; the snapshot
  and the validation run as in 0, and the forwarded log is compared with the snapshot bit for bit.  Read once and
  agreed across ranks: collective on first use.
*/
int IndividualExtensiveLogForwardMode(void);

/*! \brief RICH_D2_TIMERS (unset, 0 or 1; default 0) as read by the hydro step: one RICH_D2_HYDRO line per individual
  event.  Read once and agreed across ranks: collective on first use.
*/
bool HydroD2TimersEnabled(void);

/*! \brief RICH_INDIVIDUAL_ALL_ACTIVE_IDENTITY (B9; unset, 0, 1 or 2; default 0): on an event where every rank is
  all-active on a full mesh whose targets are the canonical cells in order (agreed by the hydro step), 0 runs the s48
  face-delta transaction; 1 looks cells up through the shared canonical index instead of a per-event map of the
  targets and exchanges deltas and radiation scales with the duplicated-point peers only, keeping ascending source
  rank order; 2 runs both, compares the results bit for bit and checks cross-rank face conservation.  Read once and
  agreed across ranks: collective on first use.
*/
int IndividualAllActiveIdentityMode(void);

//! \brief A canonical cell written by an individual extensive update, with the conserved values it held before the write.
struct IndividualCanonicalWrite
{
    std::size_t CanonicalIndex;
    double Mass;
    double Energy;
    double InternalEnergy;
};

/*! \brief An extensive updater that logs the canonical cells its UpdateIndividual call writes.
  \details The individual hydro step finds it with dynamic_cast, so any updater can offer its log.
*/
class IndividualCanonicalWriteLog
{
public:
    virtual ~IndividualCanonicalWriteLog(void) = default;

    /*! \brief True when a completed UpdateIndividual call with canonical arrays logs every canonical cell that it, or
      anything it calls, may change, so that a canonical cell absent from the log kept its value bit for bit.  The
      same on every rank; may be collective on its first call, so every rank must call it.
    */
    virtual bool ForwardsCanonicalWrites(void) const = 0;

    //! \brief The log of the last UpdateIndividual call, or nullptr when that call did not complete with a full log.
    virtual std::vector<IndividualCanonicalWrite> const* LastForwardedCanonicalWrites(void) const = 0;

    //! \brief Seconds of the last call's own validation scan and its reduction when measured (RICH_D2_TIMERS), else 0.
    virtual double LastValidationSeconds(void) const {return 0;}

    /*! \brief B9 hint for the next UpdateIndividual calls: 0 the s48 transaction, 1 the all-active identity
      transaction, 2 that transaction with the cross-rank face conservation check.  Set to the same value on every
      rank, and only on an event the hydro step agreed is all-active with identity targets; false when unsupported.
    */
    virtual bool SetAllActiveIdentity(int /*Mode*/) const {return false;}
};

//! \brief Generates a list of conserved variables
class DefaultExtensiveUpdater: public ExtensiveUpdater3D, public IndividualCanonicalWriteLog
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

	//! \brief A canonical cell written by UpdateIndividual, with the conserved values it held before the write.
	using CanonicalWrite = IndividualCanonicalWrite;

	/*! \brief Every canonical cell the last UpdateIndividual call wrote, once each, in write order.
	  \details Meaningful only when LastCanonicalWritesComplete() is true.  A canonical cell absent from the list
	  kept its value bitwise.  Before writing anything the call throws (on every rank) if a canonical cell outside
	  the owned targets, or the work copy of an owned target, has a non-finite mass, energy or thermal energy; the
	  individual hydro step gathers those work copies from the canonical array, so then every canonical cell
	  entered the call finite.
	*/
	std::vector<CanonicalWrite> const& GetLastCanonicalWrites(void) const {return LastCanonicalWrites_;}

	//! \brief True when the last UpdateIndividual call received canonical arrays and returned normally.
	bool LastCanonicalWritesComplete(void) const {return LastCanonicalWritesComplete_;}

    //! \brief Only for this exact type: a derived class may write canonical cells outside the log.
    bool ForwardsCanonicalWrites(void) const override {return typeid(*this) == typeid(DefaultExtensiveUpdater);}

    std::vector<IndividualCanonicalWrite> const* LastForwardedCanonicalWrites(void) const override
    {
        return LastCanonicalWritesComplete_ ? &LastCanonicalWrites_ : nullptr;
    }

    bool SetAllActiveIdentity(int Mode) const override
    {
        AllActiveIdentity_ = Mode;
        return true;
    }
private:
    mutable int AllActiveIdentity_ = 0;
	mutable std::vector<double> oldEk_, oldEtherm_, oldE_;
	mutable std::vector<CanonicalWrite> LastCanonicalWrites_;
	mutable bool LastCanonicalWritesComplete_ = false;
};

#endif // DEFAULT_EXTENSIVE_UPDATER_HPP
