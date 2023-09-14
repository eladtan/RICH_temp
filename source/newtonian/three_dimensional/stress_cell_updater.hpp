/*! \file stress_cell_updater.hpp
  \brief Stress cell update scheme
  \author Almog Yalinewich
 */

#ifndef STRESS_CELL_UPDATER_HPP
#define STRESS_CELL_UPDATER_HPP 1

#include "cell_updater_3d.hpp"
#include "StrengthModel.hpp"

 //! \brief Stress scheme for cell update
class StressCellUpdater : public CellUpdater3D
{
public:

	//! \brief Class constructor
  //! \param SR Special relativity flag
  //! \param G Correction to adiabatic index
  //! \param includes_temperature Flag if to compute the temperature as well or not
	StressCellUpdater(vector<bool> is_strength_arr, vector<StrengthModel*> strength_arr, vector<double> dens0_arr, bool SR = false,double G=0, bool const includes_temperature = false);

	void operator()(vector<ComputationalCell3D> &res, EquationOfState const& eos,
		const Tessellation3D& tess, vector<Conserved3D>& extensives) const override;
	const vector<StrengthModel*> strength_arr_;
private:
	const bool SR_;
	const double G_;
	const bool includes_temperature_;
	mutable size_t entropy_index_;
	const vector<bool> is_strength_arr_;
	const vector<double> dens0_arr_;
};

#endif // STRESS_CELL_UPDATER_HPP
