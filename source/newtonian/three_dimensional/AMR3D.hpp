/*! \file AMR3D.hpp
\brief Abstract class for amr in 3D
\author Elad Steinberg
*/

#ifndef AMR3D_HPP
#define AMR3D_HPP 1

#include "computational_cell.hpp"
#include "conserved_3d.hpp"
#include "../common/equation_of_state.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include <boost/scoped_ptr.hpp>
#include "3D/tessellation/utils/PolyClip.hpp"
#include "LinearGauss3D.hpp"
#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include <string>
#include <utility>
#include <vector>

//! \brief Abstract class for cell update scheme in amr
class AMRCellUpdater3D
{
public:

	/*! \brief Calculates the computational cell
	\param extensive Extensive conserved variables
	\param eos Equation of state
	\param volume Cell volume
	\param old_cell Old computational cell
	\param tracerstickernames The names of the tracers and stickers
	\return Computational cell
	*/
	virtual ComputationalCell3D ConvertExtensiveToPrimitve3D(Conserved3D& extensive, const EquationOfState& eos,
		double volume, ComputationalCell3D const& old_cell) const = 0;

	//! \brief Class destructor
	virtual ~AMRCellUpdater3D(void);

  /*! \brief Null constructor
   */
  AMRCellUpdater3D(void);

  /*! \brief Copy constructor
   */
  AMRCellUpdater3D(const AMRCellUpdater3D&);

  /*! \brief Copy constructor
    \return Reference to new object
   */
  AMRCellUpdater3D& operator=(const AMRCellUpdater3D&);
};

//! \brief Abstract class for extensive update scheme in amr
class AMRExtensiveUpdater3D
{
public:

	/*! \brief Calculates the computational cell
	\param cell Computational cell
	\param eos Equation of state
	\param volume Cell volume
	\param tracerstickernames The names of the tracers and stickers
	\param slope Gradients
	\param CMold Old centre of mass
	\param CMnew New centre of mass
	\return Extensive
	*/
	virtual Conserved3D ConvertPrimitveToExtensive3D(const ComputationalCell3D& cell, const EquationOfState& eos,
		double volume, Slope3D const& slope,Vector3D const& CMold,Vector3D const& CMnew) const = 0;

	//! \brief Class destructor
	virtual ~AMRExtensiveUpdater3D(void);

  /*! \brief Null constructor
   */
  AMRExtensiveUpdater3D(void);

  /*! \brief Copy constructor
   */
  AMRExtensiveUpdater3D(const AMRExtensiveUpdater3D&);

  /*! \brief Copy assignment
    \return Reference to new object
   */
  AMRExtensiveUpdater3D& operator=(const AMRExtensiveUpdater3D&);
};

//! \brief Simple class for extensive update scheme in amr
class SimpleAMRExtensiveUpdater3D : public AMRExtensiveUpdater3D
{
public:
	SimpleAMRExtensiveUpdater3D(void);

	Conserved3D ConvertPrimitveToExtensive3D(const ComputationalCell3D& cell, const EquationOfState& eos,
		double volume, Slope3D const& slope, Vector3D const& CMold, Vector3D const& CMnew) const override;
};

//! \brief Simple class for cell update scheme in amr
class SimpleAMRCellUpdater3D : public AMRCellUpdater3D
{
private:
	const vector<string>  toskip_;
public:
	/*!
	\brief class constructor
	\param toskip A list of sticker names to skip their cell update
	*/
	SimpleAMRCellUpdater3D(const vector<string>& toskip = vector<string>());

	ComputationalCell3D ConvertExtensiveToPrimitve3D(Conserved3D& extensive, const EquationOfState& eos,
		double volume, ComputationalCell3D const& old_cell) const override;
};

//! \brief Simple class for extensive update scheme in amr for SR
class SimpleAMRExtensiveUpdaterSR3D : public AMRExtensiveUpdater3D
{
public:
  /*! 
    \param cell Computational cell
    \param eos Equation of state
    \param volume Volume
    \param tracerstickernames Tracers and stickers names
    \param slope Gradients
    \param CMold Old centre of mass
    \param CMnew New centre of mass
    \return Conserved variables
   */
  Conserved3D ConvertPrimitveToExtensive3D(const ComputationalCell3D& cell, const EquationOfState& eos,
		double volume, Slope3D const& slope, Vector3D const& CMold, Vector3D const& CMnew) const override;
};

//! \brief Simple class for cell update scheme in amr for SR
class SimpleAMRCellUpdaterSR3D : public AMRCellUpdater3D
{
private:
	const double G_;
	const vector<string>  toskip_;
public:
	/*!
	\brief class constructor
	\param G The adiabatic index
	\param toskip A list of sticker names to skip their cell update
	*/
  SimpleAMRCellUpdaterSR3D(double G, const vector<string>& toskip);

	ComputationalCell3D ConvertExtensiveToPrimitve3D(Conserved3D& extensive, const EquationOfState& eos,
							 double volume, const ComputationalCell3D& old_cell) const override;
};

//! \brief Chooses which cells should be remove
class CellsToRemove3D
{
public:
	/*!
	\brief Finds the cells to remove
	\param tess The tesselation
	\param cells The computational cells
	\param time The sim time
	\param tracerstickernames The names of the tracers and stickers
	\return The indeces of cells to remove with a corresponding merit which decides if there are neighboring cells which one to choose to remove
	*/
	virtual std::pair<vector<size_t>, vector<double> > ToRemove(Tessellation3D const& tess,
		vector<ComputationalCell3D> const& cells, double time)const = 0;

	//! \brief Virtual destructor
	virtual ~CellsToRemove3D(void);
};

//! \brief Chooses which cells should be refined
class CellsToRefine3D
{
public:
	/*!
	\brief Finds the cells to refine
	\param tess The tesselation
	\param cells The computational cells
	\param time The sim time
	\param tracerstickernames The names of the tracers and stickers
	\return The indeces of cells to remove and the direction to split (can be given empty)
	*/
	virtual std::pair<vector<size_t>,vector<Vector3D> > ToRefine(Tessellation3D const& tess,
		vector<ComputationalCell3D> const& cells, double time)const = 0;

	//! \brief Virtual destructor
	virtual ~CellsToRefine3D(void);
};

//! \brief Base class for amr
class AMR3D
{
private:
	EquationOfState const& eos_;
	CellsToRefine3D const& refine_;
	CellsToRemove3D  const& remove_;
	SimpleAMRCellUpdater3D scu_;
	SimpleAMRExtensiveUpdater3D seu_;
	SpatialReconstruction3D &interp_;
	AMRCellUpdater3D* cu_;
	AMRExtensiveUpdater3D* eu_;
	bool distribute_clips_;
	IndividualAMRChangeSet last_change_set_;
    // RICH_D2_TIMERS=1: per-rank stage seconds and cell counts of the passes since the last ReportD2Pass, and the
    // replicated pass facts (number of passes and of rebuilding passes, the last pass's kind, cycle and time).
    std::vector<double> D2StageSeconds;
    std::vector<double> D2Counts;
    // Removal-selection sub-stage seconds and the number of volume-limit rounds (replicated) since the last report.
    std::vector<double> D2SelectSeconds;
    int D2SelectRounds = 0;
    int D2Passes = 0;
    int D2Rebuilds = 0;
    bool D2LastIndividual = false;
    size_t D2LastCycle = 0;
    double D2LastTime = 0;
	AMR3D(AMR3D const& amr);
	AMR3D& operator=(AMR3D const&);
	
public:
	/*!
	\brief Runs the AMR
	\param sim The sim object
	*/
	void operator() (Simulation &sim);

	//! Topology changes accepted by the most recent AMR call on this rank.
	IndividualAMRChangeSet const& GetLastChangeSet(void) const
	{return last_change_set_;}

	/*! \brief Run AMR for cells active in one accepted individual-timestep event.
	 *
	 * Returns the stable-ID topology map used by the individual scheduler.
	 */
	IndividualAMRChangeSet ApplyIndividual(Simulation &sim,
		IndividualStepContext const& context);

	/*! \brief Class constructor
	\param refine Refinement scheme
	\param remove Removal scheme
	\param cu Cell updater
	\param eu Extensive updater
	\param eos Equation of state
	\param interp Interpolation scheme
	\param distribute_clips Whether to distribute clip work across idle MPI ranks
	*/
	AMR3D(EquationOfState const& eos, CellsToRefine3D const& refine, CellsToRemove3D const& remove,SpatialReconstruction3D &interp, AMRCellUpdater3D* cu = nullptr,
		AMRExtensiveUpdater3D* eu = nullptr, bool distribute_clips = true);

	//! \brief Enable or disable distributed clip offloading
	void SetDistributeClips(bool v) { distribute_clips_ = v; }

    /*! \brief RICH_D2_TIMERS (unset, 0 or 1): AMR pass stage timers.  Parsed once and agreed over MPI_COMM_WORLD on
     * the first call, which must be one every rank makes together (every AMR pass calls it on entry).
     */
    static bool D2TimersEnabled(void);

    //! True when RICH_D2_TIMERS=1 and an AMR pass ran since the last ReportD2Pass; the same on every rank.
    bool D2PassPending(void) const { return D2Passes > 0; }

    /*! \brief Prints one RICH_D2_AMR line on rank 0 for the passes since the last call (normally one), with the
     * rank maximum, mean and rank of the maximum of every stage and the summed cell counts, then resets the
     * accumulators.  ExtraStages adds per-rank seconds the caller measured around the pass (the same names in the
     * same order on every rank).  Collective: call it on every rank, and only when D2PassPending() is true.
     */
    void ReportD2Pass(std::vector<std::pair<std::string, double> > const& ExtraStages);

	//! Class destructor
	~AMR3D();

private:
	IndividualAMRChangeSet Apply(Simulation &sim,
		vector<size_t> const* active_indices,
		vector<Vector3D> const* canonical_points);

    //! Sizes the RICH_D2_TIMERS accumulators on first use.
    void D2PrepareAccumulators(void);

    //! Adds one finished pass (its cell counts, kind and whether it rebuilt the mesh) to the accumulators.
    void D2FinishPass(Simulation const& Sim, std::vector<double> const& PassCounts, bool Individual, bool Rebuilt);

    //! Adds the removal-selection sub-stage seconds, local counts and round count of one pass to the accumulators.
    void D2AddSelectTiming(double Anchors, double Partners, double Targets, double Rounds, double Final,
        double FastTotal, double AnchorCount, double ProposedCount, double FirstRoundClaims, int RoundCount);
};

#endif // AMR3D_HPP
