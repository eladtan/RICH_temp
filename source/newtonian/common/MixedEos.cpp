#include "MixedEos.hpp"

MixedEOS::MixedEOS(const std::vector<EquationOfState*> &eosList) : eosList(eosList)
{}


double MixedEOS::dp2e(double d, double p, tvector const& tracers, vector<string> const& tracernames) const
{
    size_t numTracers = tracers.size();
    for(size_t i = 0; i < numTracers; i++)
    {
        if(tracers[i] > 0.5)
        {
            return eosList[i]->dp2e(d, p, tracers, tracernames);
        }       
    }
    UniversalError eo("MixedEOS::dp2e: No suitable EOS found for given tracers");
    eo.addEntry("Tracers", tracers);
    eo.addEntry("Tracer Names", tracernames);
    throw eo;
}

double MixedEOS::de2p(double d, double e, tvector const& tracers, vector<string> const& tracernames) const
{
    size_t numTracers = tracers.size();
    for(size_t i = 0; i < numTracers; i++)
    {
        if(tracers[i] > 0.5)
        {
            return eosList[i]->de2p(d, e, tracers, tracernames);
        }       
    }
    UniversalError eo("MixedEOS::de2p: No suitable EOS found for given tracers");
    eo.addEntry("Tracers", tracers);
    eo.addEntry("Tracer Names", tracernames);
    throw eo;
}

double MixedEOS::de2c(double d, double e, tvector const& tracers, vector<string> const& tracernames) const
{
    size_t numTracers = tracers.size();
    for(size_t i = 0; i < numTracers; i++)
    {
        if(tracers[i] > 0.5)
        {
            return eosList[i]->de2c(d, e, tracers, tracernames);
        }       
    }
    UniversalError eo("MixedEOS::de2c: No suitable EOS found for given tracers");
    eo.addEntry("Tracers", tracers);
    eo.addEntry("Tracer Names", tracernames);
    throw eo;
}

double MixedEOS::dp2c(double d, double p, tvector const& tracers, vector<string> const& tracernames) const
{
    size_t numTracers = tracers.size();
    for(size_t i = 0; i < numTracers; i++)
    {
        if(tracers[i] > 0.5)
        {
            return eosList[i]->dp2c(d, p, tracers, tracernames);
        }       
    }
    UniversalError eo("MixedEOS::dp2c: No suitable EOS found for given tracers");
    eo.addEntry("Tracers", tracers);
    eo.addEntry("Tracer Names", tracernames);
    throw eo;
}

double MixedEOS::dp2s(double d, double p, tvector const& tracers, vector<string> const& tracernames) const
{
    size_t numTracers = tracers.size();
    for(size_t i = 0; i < numTracers; i++)
    {
        if(tracers[i] > 0.5)
        {
            return eosList[i]->dp2s(d, p, tracers, tracernames);
        }       
    }
    UniversalError eo("MixedEOS::dp2s: No suitable EOS found for given tracers");
    eo.addEntry("Tracers", tracers);
    eo.addEntry("Tracer Names", tracernames);
    throw eo;
}

double MixedEOS::sd2p(double s, double d, tvector const& tracers, vector<string> const& tracernames) const
{
    size_t numTracers = tracers.size();
    for(size_t i = 0; i < numTracers; i++)
    {
        if(tracers[i] > 0.5)
        {
            return eosList[i]->sd2p(s, d, tracers, tracernames);
        }       
    }
    UniversalError eo("MixedEOS::sd2p: No suitable EOS found for given tracers");
    eo.addEntry("Tracers", tracers);
    eo.addEntry("Tracer Names", tracernames);
    throw eo;
}

double MixedEOS::dT2cv(double const d, double const T, tvector const& tracers, vector<string> const& tracernames) const
{
    size_t numTracers = tracers.size();
    for(size_t i = 0; i < numTracers; i++)
    {
        if(tracers[i] > 0.5)
        {
            return eosList[i]->dT2cv(d, T, tracers, tracernames);
        }       
    }
    UniversalError eo("MixedEOS::dT2cv: No suitable EOS found for given tracers");
    eo.addEntry("Tracers", tracers);
    eo.addEntry("Tracer Names", tracernames);
    throw eo;
}

double MixedEOS::de2T(double const d, double const e, tvector const& tracers, vector<string> const& tracernames) const
{
    size_t numTracers = tracers.size();
    for(size_t i = 0; i < numTracers; i++)
    {
        if(tracers[i] > 0.5)
        {
            return eosList[i]->de2T(d, e, tracers, tracernames);
        }       
    }
    UniversalError eo("MixedEOS::de2T: No suitable EOS found for given tracers");
    eo.addEntry("Tracers", tracers);
    eo.addEntry("Tracer Names", tracernames);
    throw eo;
}

double MixedEOS::dT2e(double const d, double const T, tvector const& tracers, vector<string> const& tracernames) const
{
    size_t numTracers = tracers.size();
    for(size_t i = 0; i < numTracers; i++)
    {
        if(tracers[i] > 0.5)
        {
            return eosList[i]->dT2e(d, T, tracers, tracernames);
        }       
    }
    UniversalError eo("MixedEOS::dT2e: No suitable EOS found for given tracers");
    eo.addEntry("Tracers", tracers);
    eo.addEntry("Tracer Names", tracernames);
    throw eo;
}