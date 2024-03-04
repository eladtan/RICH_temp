
#ifndef MIEGRUN2_HPP
#define MIEGRUN2_HPP 1

#include <string>
#include <array>
#include "../two_dimensional/computational_cell_2d.hpp"
#include "./equation_of_state.hpp"

/** typedef for tracer vector */
//typedef std::array<double,MAX_TRACERS> tvector;
/** typedef for string vector */
//typedef std::array<bool, MAX_STICKERS> svector;
using std::string;
using std::vector;

//! \brief Base class for equation of state
class MieGrun2 : public EquationOfState
{
private:
  double const rho0_, gamma_, a0_, s_;
public:
  explicit MieGrun2(double const rho0, double const a0, double const gamma, double const s);

  double dp2e(double d, double p, tvector const& tracers = tvector(),vector<string> const& tracernames = vector<string>())
	  const override;

  double de2p(double d, double e,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

  double de2c(double d, double e,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

  double dp2c(double d, double p,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

  double dp2s(double d, double p,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

  double sd2p(double s, double d,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

  double dT2cv(double const d, double const T,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

  double de2T(double const d, double const e,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

  double dT2e(double const d, double const T,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;
};

#endif
