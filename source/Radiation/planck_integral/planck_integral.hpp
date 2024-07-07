#ifndef PLANK_INTEGRAL_HPP
#define PLANK_INTEGRAL_HPP

#include <boost/math/special_functions/pow.hpp>
#include <cmath>
#include <numbers>
#include "../conj_grad_solve.hpp"


inline double Clark_Taylor(double const x){
    double const x2 = x*x;
    double const x3 = x2*x;

    return x3*(1./3. + x*(-1./8. + x*(1./60.+x2*(-1./5040.+x2*(1./272160.+x2*(-1./13305600+x2/622702080.))))));
}

static int constexpr N_clark=5;
static double constexpr x_clark = 2.;

inline double Clark_series(double const x){
    double const x2 = x*x;
    double const x3 = x2*x;
    double sum=0;

    for(int n=1; n <= N_clark; ++n){
        double const in = 1. / static_cast<double>(n);
        sum += in*(x3+in*(3.*x2+6.*in*(x+in)))*std::exp(-(x*n));
    }

    return -sum;
}

inline double planck_integral(double const a, double const b){
    assert(a<b);
    using boost::math::pow;

    static double constexpr coeff = 15./pow<4>(M_PI);

    if(a > x_clark) return coeff*(Clark_series(b) - Clark_series(a));
    if(b < x_clark) return coeff*(Clark_Taylor(b) - Clark_Taylor(a));
    
    return 1.0 + coeff*(Clark_series(b) - Clark_Taylor(a));
}

inline double planck_energy_density_group_integral(double const E_low, double const E_high, double const T) {
    using boost::math::pow;
    assert(E_low < E_high);

    if(T < 1e50*std::numeric_limits<double>::min()){
        return 0.0;
    }

    double const kT = CG::boltzmann_constant * T;
    
    double const a = E_low / kT;
    double const b = E_high / kT;

    return CG::radiation_constant * pow<4>(T) * planck_integral(a, b);
}

#endif //PLANK_INTEGRAL_HPP