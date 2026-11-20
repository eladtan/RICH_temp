/*! \file default_cell_updater.hpp
  \brief Default cell update scheme
  \author Almog Yalinewich
 */

#ifndef DEFAULT_CELL_UPDATER_HPP
#define DEFAULT_CELL_UPDATER_HPP 1

#include "cell_updater_3d.hpp"
#include "../../Radiation/RadiationDriver.hpp"

 //! \brief Default scheme for cell update
class DefaultCellUpdater : public CellUpdater3D
{
public:

	//! \brief Class constructor
  //! \param SR Special relativity flag
  //! \param G Correction to adiabatic index
  //! \param includes_temperature Flag if to compute the temperature as well or not
	DefaultCellUpdater(bool SR = false,double G=0, bool const includes_temperature = false, double const min_temperature = 0, const RadiationDriver* diffusion = nullptr);

	void operator()(vector<ComputationalCell3D> &res, EquationOfState const& eos,
			const Tessellation3D& tess, vector<Conserved3D>& extensives) const override;

	void UpdateIndividual(vector<ComputationalCell3D> &res,
		EquationOfState const& eos,
		const Tessellation3D& tess,
		vector<Conserved3D>& extensives,
		const IndividualStepContext& context) const override;
private:
	void UpdateCells(vector<ComputationalCell3D> &res,
		EquationOfState const& eos,
		const Tessellation3D& tess,
		vector<Conserved3D>& extensives,
		bool synchronize_entropy_ghosts) const;

    /*! \brief RICH_INDIVIDUAL_CELL_UPDATE_SPARSE: recovers the active targets (owned, strictly ascending) in one ordered
        sweep, in place, with the results of the full copy-sweep-select path.
    */
    void UpdateActiveCells(vector<ComputationalCell3D>& Cells, EquationOfState const& Eos, const Tessellation3D& Tess,
        vector<Conserved3D>& Extensives, std::vector<size_t> const& ActiveIndices) const;

	const bool SR_;
	const double G_;
	const bool includes_temperature_;
	double const min_temperature_;
	mutable size_t entropy_index_;
	const RadiationDriver* diffusion_;
};

/*! \brief RICH_INDIVIDUAL_CELL_UPDATE_SPARSE (unset, 0, 1 or 2; default 0), DefaultCellUpdater::UpdateIndividual only.
  0 every owned target is recovered on copies and the active ones are kept.  1 only the active targets are recovered, in
  one ordered sweep in place, with identical results.  2 both, compared bit for bit (RICH_B10_CELL_VERIFY).  Read once
  and agreed across ranks: collective on first use.
*/
int IndividualCellUpdateSparseMode(void);

#endif // DEFAULT_CELL_UPDATER_HPP
