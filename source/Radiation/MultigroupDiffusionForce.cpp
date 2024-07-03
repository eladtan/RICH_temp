#include "MultigroupDiffusionForce.hpp"
#include "Diffusion.hpp" // CalcSingleFluxLimiter
#include "boost/math/special_functions/pow.hpp"

using boost::math::pow;

MultigroupDiffusionForce::MultigroupDiffusionForce(MultigroupDiffusion const& multigroup_diffusion,
                                                   EquationOfState const& eos,
                                                   bool const momentum_limit) : 
                                                                multigroup_diffusion_(multigroup_diffusion),
                                                                next_dt_(1e-6 * std::numeric_limits<double>::max()),
                                                                eos_(eos),
                                                                momentum_limit_(momentum_limit){}

