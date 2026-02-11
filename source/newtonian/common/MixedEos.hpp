#ifndef MIXED_EOS_HPP
#define MIXED_EOS_HPP

#include "equation_of_state.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

class MixedEOS : public EquationOfState
{
public:
  MixedEOS(const std::vector<EquationOfState*> &eosList);

  double dp2e(double d, double p, tvector const& tracers,vector<string> const& tracernames) const override;
  
  double de2p(double d, double e, tvector const& tracers, vector<string> const& tracernames) const override;
  
  double de2c(double d, double e, tvector const& tracers, vector<string> const& tracernames) const override;
  
  double dp2c(double d, double p, tvector const& tracers, vector<string> const& tracernames) const override;
  
  double dp2s(double d, double p, tvector const& tracers, vector<string> const& tracernames) const override;

  double sd2p(double s, double d, tvector const& tracers, vector<string> const& tracernames) const override;

  double dT2cv(double const d, double const T, tvector const& tracers, vector<string> const& tracernames) const override;

  double de2T(double const d, double const e, tvector const& tracers, vector<string> const& tracernames) const override;

  double dT2e(double const d, double const T, tvector const& tracers, vector<string> const& tracernames) const override;

private:
    std::vector<EquationOfState*> eosList; // List of EOS objects to use for calculations
};

#endif // MIXED_EOS_HPP