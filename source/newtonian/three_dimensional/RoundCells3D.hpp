/*! \file RoundCells3D.hpp
\author Elad Steinberg
\brief Correction to point velocities that keeps cells round
*/

#ifndef ROUND_CELLS3D_HPP
#define ROUND_CELLS3D_HPP 1

#include "point_motion_3d.hpp"
#include "../common/equation_of_state.hpp"
#include "../two_dimensional/computational_cell_2d.hpp"

//! \brief Correction to point velocities that keeps cells round
//! \details Based on equation 63 in the Arepo paper
class RoundCells3D : public PointMotion3D
{
public:

	/*! \brief Class constructor
	\param pm Base point motion
	\param eos Equation of state
	\param chi chi parameter in equation 63
	\param eta eta parameter in equation 63
	\param cold Switch for cold flows
	\param min_dw The minimum magnitude for the cell' velocity scale
	\param dt_speed The speed in units of radius/dt
	\param no_move List of stickers not to move the cells for
	\param max_velocity The maximum velocity that dw can have
	*/
	RoundCells3D(const PointMotion3D& pm, const EquationOfState& eos,
		double chi = 1.25, double eta = 0.02, bool cold = false,double min_dw=0,double dt_speed=0.01,
		const vector<std::string>& no_move=vector<std::string>(), double max_velocity=1e100);

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		double time, vector<Vector3D> &res) const override;

	void ApplyFix(Tessellation3D const& tess, vector<ComputationalCell3D> const& cells, double time,
		double dt, vector<Vector3D> &velocities)const override;

	void ApplyFixIndividual(Tessellation3D const& tess,
		vector<ComputationalCell3D> const& cells,
		vector<ComputationalCell3D> const& all_cells, double time, double dt,
		vector<Vector3D>& velocities,
		vector<Vector3D>& all_velocities) const override;

    /*! \brief RICH_INDIVIDUAL_HYDRO_TARGET_PASSES (B10): the active targets (mesh-local indices, ascending) of the
      individual event whose next ApplyFix or ApplyFixIndividual call follows; nullptr (the default) fixes every
      target.  With a list, mode 1 runs the RoundCells kick, SlowDown and overshoot passes only where they can reach
      an active target's velocity on this rank or, through the velocity sync, on a peer; mode 2 runs both and compares
      the active targets' velocities bit for bit.  Set on every rank or on none.
    */
    void SetIndividualActiveTargets(std::vector<size_t> const* ActiveTargets) const {ActiveTargets_ = ActiveTargets;}

    //! \brief Per-rank sizes of the last fix with an active list (D2 timers): targets, active, kicked, slowed, sent.
    struct TargetPassCounts {
        size_t Targets = 0;
        size_t Active = 0;
        size_t Kicked = 0;
        size_t Slowed = 0;
        size_t Sent = 0;
    };

    TargetPassCounts const& LastTargetPassCounts(void) const {return LastTargetPassCounts_;}
private:
    // The s48 body of ApplyFix(Individual); with Active, the passes are restricted as SetIndividualActiveTargets says.
    void ApplyFixPasses(Tessellation3D const& tess, vector<ComputationalCell3D> const& cells,
        vector<ComputationalCell3D> const* all_cells, double time, double dt, vector<Vector3D>& velocities,
        vector<Vector3D>* all_velocities, std::vector<size_t> const* Active) const;

	void ApplyFixImpl(Tessellation3D const& tess,
		vector<ComputationalCell3D> const& cells,
		vector<ComputationalCell3D> const* all_cells, double time, double dt,
		vector<Vector3D>& velocities,
		vector<Vector3D>* all_velocities) const;

	void calc_dw(Vector3D &velocty, size_t i, const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const vector<Vector3D> & velocities, vector<char> const& nomove) const;

	void calc_dw(Vector3D &velocty, size_t i, const Tessellation3D& tess, double dt, vector<ComputationalCell3D> const& cells,
		vector<Vector3D> & velocities, vector<char> const& nomove)const;

    // True when no rank configures a no-move sticker (RICH_ROUNDCELLS_SKIP_EMPTY_STICKERS).  Agreed in one reduction
    // on its first call, which must be collective; the sticker configuration is fixed at setup.
    bool NoMoveStickerOnNoRank(size_t LocalStickerCount) const;

	const PointMotion3D& pm_;
	const EquationOfState& eos_;
	const double chi_;
	const double eta_;
	const bool cold_;
	const double min_dw_;
	const double dt_speed_;
	const vector<std::string> no_move_;
	const double max_velocity_;
	mutable vector<size_t> slowdown_neigh_buf_;
	mutable vector<size_t> calc_dw_neigh_buf_;
	mutable vector<size_t> calc_dw2_neigh_buf_;
    // NoMoveStickerOnNoRank: -1 not agreed yet, 0 some rank has a no-move sticker, 1 no rank has one; and this
    // rank's sticker count when it was agreed.
    mutable int NoStickerAgreed_ = -1;
    mutable size_t AgreedLocalStickerCount_ = 0;
    mutable std::vector<size_t> const* ActiveTargets_ = nullptr;
    mutable TargetPassCounts LastTargetPassCounts_;
};

/*! \brief RICH_INDIVIDUAL_HYDRO_TARGET_PASSES (B10; unset, 0, 1 or 2; default 0): 0 the s48 passes over every target;
  1 restricted passes (see RoundCells3D::SetIndividualActiveTargets); 2 both, compared bit for bit.  Read once and
  agreed across ranks: collective on first use.
*/
int IndividualHydroTargetPassesMode(void);

#endif // ROUND_CELLS3D_HPP
