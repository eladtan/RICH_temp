/*! \file LinearGauss3D.hpp
\brief Linear interpolation that guarantees compliance with the equation of state and calcualtes the GG gradient from the CM of the cell
\author Elad Steinberg
*/

#ifndef LINEAR_GAUSS3D_HPP
#define LINEAR_GAUSS3D_HPP 1

#include "../common/equation_of_state.hpp"
#include "SpatialReconstruction3D.hpp"
#include <cmath>
#include "../../misc/universal_error.hpp"
#include "Ghost3D.hpp"

/*! \brief Paths of one individual-event interpolation (LinearGauss3D::InterpolateIndividualWithOptions).  All false is
  the path of InterpolateIndividual.
*/
struct IndividualInterpolationOptions {
    //! RICH_INDIVIDUAL_FLUX_PASSES: keep the face states of faces without an active endpoint instead of resetting them
    bool KeepInactiveFaceStates = false;
    //! RICH_INDIVIDUAL_FLUX_PASSES: read the input cells in place instead of a per-mesh-point copy (non-SR only)
    bool ReadCellsInPlace = false;
    //! RICH_INDIVIDUAL_GHOST_SLOPE_REQUEST: compute and send only the ghost slopes that the peers request
    bool RequestGhostSlopes = false;
};

//! \brief Counts of the last individual-event interpolation on this rank
struct IndividualInterpolationCounts {
    //! Owned cells whose slopes were computed
    size_t Reconstructed = 0;
    //! Owned cells duplicated to a peer, summed over peers
    size_t GhostSources = 0;
    //! Entries of GhostSources whose slopes were computed and sent
    size_t SentGhostSources = 0;
};

//! \brief Linear gauss interpolation
class LinearGauss3D : public SpatialReconstruction3D
{
public:

	/*! \brief Class constructor
	\param eos Equation of state
	\param slf Slope limiter flag
	\param delta_v The GradV*L/Cs ratio needed for slope limiter
	\param theta The theta from tess in slope limiter.
	\param delta_P The pressure ratio for shock detection
	\param ghost The ghost point generator
	\param SR Flag for SR
	\param calc_tracers Names of tracers for which to calc the slope for
	\param skip_key The sticker name to skip cells for taking them into account for the slope limit
	\param tsn The names of the stickers and tracers
	\param pressure_calc Determine whether the pressure should be recalculated
	\param apply_principal_limit Enable principal-frame velocity limiting
	*/
	LinearGauss3D(EquationOfState const& eos, Ghost3D const& ghost,bool slf = true,double delta_v = 0.2,
		double theta = 0.5,double delta_P = 0.7,bool SR=false,const vector<string>& calc_tracers = vector<string>(),
		const string& skip_key = string(),bool pressure_calc=true, bool apply_principal_limit=false);

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, double time,
		vector<pair<ComputationalCell3D, ComputationalCell3D> > &res) const override;

	void InterpolateIndividual(
		const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells,
		double time,
		const vector<unsigned char>& active_mask,
		vector<pair<ComputationalCell3D, ComputationalCell3D> >& res) const override;

    /*! \brief InterpolateIndividual with the paths that Options selects.  With RequestGhostSlopes every rank must call
        it with the same options: it adds one exchange with the duplicated-point peers.
    */
    void InterpolateIndividualWithOptions(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
        double time, const vector<unsigned char>& active_mask,
        vector<pair<ComputationalCell3D, ComputationalCell3D> >& res, IndividualInterpolationOptions const& Options) const;

    //! \brief Counts of the last individual-event interpolation on this rank
    IndividualInterpolationCounts const& GetLastIndividualCounts(void) const {return LastIndividualCounts_;}

	/*! \brief Interpolates a cell
	\param cell The primitives of the cell
	\param cell_index The index of the cell
	\param cm The cell's center of mass
	\param target The location of the interpolation
	\param res The interpolated value
	\param tsn The names of the stickers and tracers
	\param eos The equation of state
	*/
	void Interp(ComputationalCell3D &res,ComputationalCell3D const& cell, size_t cell_index, Vector3D const& cm, Vector3D const& target,
		EquationOfState const& eos)const;

	/*!
	\brief Returns the gradients
	\return The gradients
	*/
	vector<Slope3D>& GetSlopes(void) override;

	const vector<Slope3D>* GetSlopesForTimePrediction(void) const override
	{return &rslopes_;}

	/*!
	\brief Returns the unsloped limtied gradients
	\return The gradients
	*/
	vector<Slope3D>& GetSlopesUnlimited(void)const;

	void BuildSlopes(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, double time) override;

	/*! \brief Compute per-cell dissipation without materializing face pairs.
	    Precondition: BuildSlopes(tess, cells, time) must have been called
	    for the same tessellation/cells/time before this method.
	    Reuses new_cells_ and rslopes_ prepared by that call.
	*/
	std::vector<double> CalcDissipationStreamingFromPreparedSlopes(
		const Tessellation3D& tess,
		const std::vector<ComputationalCell3D>& cells,
		double time,
		class Hllc3D const& rs,
		const EquationOfState& eos) const;

private:
	void InterpolateImpl(
		const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells,
		double time,
		const vector<unsigned char>* active_mask,
		vector<pair<ComputationalCell3D, ComputationalCell3D> >& res,
		IndividualInterpolationOptions const& Options) const;

	EquationOfState const& eos_;
	Ghost3D const& ghost_;
	mutable vector<Slope3D> rslopes_;
	mutable vector<Slope3D> naive_rslopes_;
	mutable vector<ComputationalCell3D> new_cells_;
	const bool slf_;
	const double shockratio_;
	const double diffusecoeff_;
	const double pressure_ratio_;
	const bool SR_;
	const vector<string> calc_tracers_;
	const string skip_key_;
	mutable vector<size_t> to_skip_;
	const bool pressure_calc_;
	const bool apply_principal_limit_;
    mutable IndividualInterpolationCounts LastIndividualCounts_;

	LinearGauss3D(const LinearGauss3D& origin);

	LinearGauss3D& operator=(const LinearGauss3D& origin);
};

/*! \brief RICH_INDIVIDUAL_FLUX_PASSES (unset, 0, 1 or 2; default 0), individual events only.  0 today's passes.  1 the
  face states of faces without an active endpoint are not reset (no reader of the event reads them), the cells are read
  in place instead of copied per mesh point, and the flux array, which the hydro step zeroes just before, is not
  assigned again.  2 both, compared bit for bit (RICH_B10_FLUX_VERIFY).  Read once and agreed across ranks: collective
  on first use.
*/
int IndividualFluxPassesMode(void);

/*! \brief RICH_INDIVIDUAL_GHOST_SLOPE_REQUEST (unset, 0, 1 or 2; default 0), individual events only.  0 the slopes of
  every owned cell duplicated to a peer are computed and sent.  1 the peers name the ghosts whose slopes they read in
  one exchange per event, and only those slopes are computed and sent.  2 both, compared bit for bit
  (RICH_B10_SLOPE_VERIFY).  Read once and agreed across ranks: collective on first use.
*/
int IndividualGhostSlopeRequestMode(void);

#endif //LINEAR_GAUSS3D_HPP
