#include "multi_eos.hpp"
#include "source/misc/universal_error.hpp"
#include <cmath>
#include <limits>

double MultiEOS::dp2e(double d, double p, tvector const& tracers, vector<string> const& tracernames) const
{
    for(size_t i = 0; i < tracernames.size(); ++i)
        if(tracers[i] > 0.5)
            return eos_vec_[i]->dp2e(d, p, tracers, tracernames);
    throw UniversalError("Mixed Material in MultiEOS");
    return 0;    
}

double MultiEOS::de2p(double d, double e, tvector const& tracers, vector<string> const& tracernames) const
{
    for(size_t i = 0; i < tracernames.size(); ++i)
        if(tracers[i] > 0.5)
            return eos_vec_[i]->de2p(d, e, tracers, tracernames);
    throw UniversalError("Mixed Material in MultiEOS");
    return 0;   
}

double MultiEOS::dp2c(double d, double p, tvector const& tracers, vector<string> const& tracernames) const
{
    for(size_t i = 0; i < tracernames.size(); ++i)
        if(tracers[i] > 0.5)
            return eos_vec_[i]->dp2c(d, p, tracers, tracernames);
    throw UniversalError("Mixed Material in MultiEOS");
    return 0;   
}

double MultiEOS::de2c(double d, double e, tvector const& tracers, vector<string> const& tracernames) const
{
    for(size_t i = 0; i < tracernames.size(); ++i)
        if(tracers[i] > 0.5)
            return eos_vec_[i]->de2c(d, e, tracers, tracernames);
    throw UniversalError("Mixed Material in MultiEOS");
    return 0;   
}


double MultiEOS::dp2s(double d, double p, tvector const& tracers, vector<string> const& tracernames) const
{
    throw UniversalError("dp2s not implemented in MultiEOS");
    return 0;
}

double MultiEOS::sd2p(double s, double d, tvector const& tracers, vector<string> const& tracernames) const
{
    throw UniversalError("sd2p not implemented in MultiEOS");
    return 0;
}

double MultiEOS::dT2cv(double d, double T, tvector const& tracers, vector<string> const& tracernames) const
{
    throw UniversalError("dT2cv not implemented in MultiEOS");
    return 0;
}

double MultiEOS::de2T(double d, double e, tvector const& tracers, vector<string> const& tracernames) const
{
    throw UniversalError("de2T not implemented in MultiEOS");
    return 0;
}

double MultiEOS::dT2e(double d, double T, tvector const& tracers, vector<string> const& tracernames) const
{
    throw UniversalError("dT2e not implemented in MultiEOS");
    return 0;
}