/*! \file ideal_gas.hpp
  \brief Ideal gas equation of state
  \author Almog Yalinewich
 */

#ifndef IDEAL_GAS_HPP
#define IDEAL_GAS_HPP 1

#include "equation_of_state.hpp"
#include "../../misc/utils.hpp"

//! \brief The ideal gas formulas, shared by IdealGas's scalar and array forms. Each returns false where IdealGas throws.
namespace ideal_gas
{
	RICH_INLINE_FUNCTION bool dp2e(double g, double d, double p, double &e)
	{
#ifdef RICH_DEBUG
		if(!(g > 0) || !(p > 0) || !(d > 0))
		{
			return false;
		}
#endif // RICH_DEBUG
		e = p / d / (g - 1);
		return true;
	}

	RICH_INLINE_FUNCTION bool de2p(double g, double d, double e, double &p)
	{
#ifdef RICH_DEBUG
		if(!(g > 0) || !(e > 0) || !(d > 0))
		{
			return false;
		}
#endif // RICH_DEBUG
		if(e < 0)
		{
			return false;
		}
		p = (g - 1) * e * d;
		return true;
	}

	RICH_INLINE_FUNCTION bool dp2c(double g, double d, double p, double &c)
	{
#ifdef RICH_DEBUG
		if(!(g > 0) || !(p > 0) || !(d > 0))
		{
			return false;
		}
#endif // RICH_DEBUG
		c = portable::Sqrt(g * p / d);
		return true;
	}

	RICH_INLINE_FUNCTION bool de2c(double g, double d, double e, double &c)
	{
		double p = 0;
		return de2p(g, d, e, p) and dp2c(g, d, p, c);
	}
} // namespace ideal_gas

//! \brief Ideal gas equation of state
class IdealGas: public EquationOfState
{
private:

  double const g_, f_, beta_, mu_;

public:

  /*! \brief Class constructor
    \param AdiabaticIndex Adiabatic index
   */
  explicit IdealGas(double const AdiabaticIndex);

  /*! \brief Class constructor for ideal gas with temperature equation as well, e=f*T^beta*rho^{-mu}
    \param AdiabaticIndex Adiabatic index
    \param f The constant prefactor
    \param beta The power law of the temperature
    \param mu The inverse of the power law of the density
   */
  IdealGas(double const AdiabaticIndex, double const f, double const beta, double const mu);

  /*! \brief Returns the adiabatic index
    \return Adiabatic index
   */
  double getAdiabaticIndex(void) const;

  double dp2e(double d, double p, tvector const& tracers, vector<string> const& tracernames) const override;

  double de2p(double d, double e, tvector const& tracers, vector<string> const& tracernames) const override;

  double dp2c(double d, double p, tvector const& tracers, vector<string> const& tracernames) const override;

  double de2c(double d, double e, tvector const& tracers, vector<string> const& tracernames) const override;

  double dp2s(double d, double p, tvector const& tracers, vector<string> const& tracernames) const override;

  double sd2p(double s, double d, tvector const& tracers, vector<string> const& tracernames) const override;

  double dT2cv(double const d, double const T, tvector const& tracers, vector<string> const& tracernames) const override;

  double de2T(double const d, double const T, tvector const& tracers, vector<string> const& tracernames) const override;

  double dT2e(double const d, double const T, tvector const& tracers, vector<string> const& tracernames) const override;

  void dp2eArrays(EosArrays const& arrays, portable::Where where) const override;

  void de2pArrays(EosArrays const& arrays, portable::Where where) const override;

  void dp2cArrays(EosArrays const& arrays, portable::Where where) const override;

  void de2cArrays(EosArrays const& arrays, portable::Where where) const override;

  bool RunsOnDevice(void) const override;

  bool ThreadSafe(void) const override;
};

#endif // IDEAL_GAS_HPP
