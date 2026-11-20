#ifndef CONDITION_EXTENSIVE_UPDATER3D_HPP
#define CONDITION_EXTENSIVE_UPDATER3D_HPP 1

#include "extensive_updater3d.hpp"
#include "default_extensive_updater.hpp"
#include "LinearGauss3D.hpp"
#include "../common/equation_of_state.hpp"
#include "misc/utils.hpp"

//! \brief Updates the extensives based on a series of conditions and actions. Does a normal update of all cells before going into the conditions check
class ConditionExtensiveUpdater3D : public ExtensiveUpdater3D, public IndividualCanonicalWriteLog
{
public:

	//! \brief Determines the kind of cell
	class Condition3D
	{
	public:

		/*! \brief Checks if an interface satisfies a condition
		\param index The index of the cell
		\param tess Tessellation
		\param cells Computational cells
		\param time The sim time
		\param tracerstickernames The names of the tracers and stickers
		\return Whether the cell satisfies a condition.
		*/
		virtual bool operator()(size_t index,const Tessellation3D& tess,const vector<ComputationalCell3D>& cells,
			double time) const = 0;

		virtual ~Condition3D(void);
	};

	//! \brief Action taken to update extensive
	class Action3D
	{
	public:
		/*!
		\param fluxes Fluxes
		\param tess Tessellation
		\param dt Time step
		\param cells Computational cells
		\param extensives Extensive variables, input is after the addition of hydro fluxes
		\param index The index of the cell
		\param time The time
		\param tracerstickernames The names of the tracers and stickers
		*/
		virtual void operator()	(const vector<Conserved3D>& fluxes,const Tessellation3D& tess,const double dt,
			const vector<ComputationalCell3D>& cells,vector<Conserved3D> &extensives,size_t index,double time) const = 0;

		virtual ~Action3D(void);
	};

	/*! \brief Class constructor
	\param sequence Series of condition and action action pairs. Both have to be dynamically allocated pointers, and will be explicitly destructed upon descruction of the class
	*/
	explicit ConditionExtensiveUpdater3D(const vector<pair<const Condition3D*, const Action3D*> >& sequence);

	~ConditionExtensiveUpdater3D(void) override;

	void operator()(const vector<Conserved3D>& fluxes,const Tessellation3D& tess,const double dt,
		const vector<ComputationalCell3D>& cells,vector<Conserved3D>& extensives,double time, const vector<Vector3D>& edge_velocities,
		const vector<Vector3D>& point_velocities,
		std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const& interp_values) const override;

	bool SupportsIndividualTimeSteps(void) const override { return true; }

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

    /*! \brief With RICH_INDIVIDUAL_EXTENSIVE_LOG_FORWARD at 1 or 2, for this exact type, when no rank has a
      condition/action pair: then nothing but the inner DefaultExtensiveUpdater writes canonical cells, and its log is
      complete.  Collective on its first call in those modes.
    */
    bool ForwardsCanonicalWrites(void) const override;

    std::vector<IndividualCanonicalWrite> const* LastForwardedCanonicalWrites(void) const override;

    double LastValidationSeconds(void) const override {return LastValidationSeconds_;}

    //! \brief Passed to the inner DefaultExtensiveUpdater of every following UpdateIndividual call.
    bool SetAllActiveIdentity(int Mode) const override
    {
        AllActiveIdentity_ = Mode;
        return true;
    }

private:
    // True when no rank has a condition/action pair.  Agreed in one reduction on its first call (collective).
    bool SequenceEmptyOnEveryRank(void) const;

	const vector<pair<const Condition3D*, const Action3D*> > sequence_;
    // Inner updater kept across calls under log forwarding (modes 1 and 2); mode 0 builds a fresh one per call.
    DefaultExtensiveUpdater RegularUpdate_;
    // -1 not agreed yet, 0 some rank has a condition/action pair, 1 no rank has one.
    mutable int SequenceEmptyAgreed_ = -1;
    // The last UpdateIndividual call completed with a complete inner log and no condition/action pair anywhere.
    mutable bool LastCallForwarded_ = false;
    mutable double LastValidationSeconds_ = 0;
    mutable int AllActiveIdentity_ = 0;
};

//! \brief Class that returns true for all cells
class ChooseAll : public ConditionExtensiveUpdater3D::Condition3D
{
public:

	bool operator()(size_t /*index*/, const Tessellation3D& /*tess*/, const vector<ComputationalCell3D>& /*cells*/,
		double /*time*/) const override
	{
		return true;
	}
};

class StickerChoose : public ConditionExtensiveUpdater3D::Condition3D
{
private:
	std::string const sticker_name_;
public:
	StickerChoose(std::string const sticker_name) : sticker_name_(sticker_name) {}

	bool operator()(size_t index, const Tessellation3D& /*tess*/, const vector<ComputationalCell3D>& cells,
		double /*time*/) const override
	{
		return *safe_retrieve(cells[index].stickers.begin(), ComputationalCell3D::stickerNames.begin(),
			ComputationalCell3D::stickerNames.end(), sticker_name_);
	}
};

//! \brief Default extensive updater
class RegularExtensiveUpdate3D : public ConditionExtensiveUpdater3D::Action3D
{
public:
	void operator()	(const vector<Conserved3D>& fluxes, const Tessellation3D& tess, const double dt,
		const vector<ComputationalCell3D>& cells, vector<Conserved3D> &extensives, size_t index, double time)const override;
};

//! \brief Default extensive updater
class NoExtensiveUpdate3D : public ConditionExtensiveUpdater3D::Action3D
{
public:
	void operator()	(const vector<Conserved3D>& fluxes, const Tessellation3D& tess, const double dt,
		const vector<ComputationalCell3D>& cells, vector<Conserved3D>& extensives, size_t index, double time)const override;
};

#endif // CONDITION_EXTENSIVE_UPDATER3D_HPP
