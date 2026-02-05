#include "OndrejEOS.hpp"
#include <limits>
#include "../../misc/universal_error.hpp"
#include "../../misc/simple_io.hpp"
#include <cmath>
#include <algorithm>

namespace
{
    typedef std::vector<double>::const_iterator cit;

    // Optimized cubic Hermite interpolation with pre-computed common subexpressions
    double cubichermite(cit x_begin, cit x_end, cit y_begin, cit y_end, double xi)
    {
        size_t index = static_cast<size_t>(std::upper_bound(x_begin, x_end, xi) - x_begin);
        
        // Boundary checks - cache pointer base for efficiency
        double const* __restrict xp = &(*x_begin);
        double const* __restrict yp = &(*y_begin);
        size_t const y_size = static_cast<size_t>(y_end - y_begin);
        
        if ((index < 2 && xi < xp[1]) || index >= y_size - 1)
        {
            if(index == 1 && std::abs(xp[index] - xi) < 1e-13)
                ++index;
            else
            {
                if(index == 3 && std::abs(xp[index - 1] - xi) < 1e-13)
                    --index;
                else
                {
                    UniversalError eo("Bad interpolation");
                    eo.addEntry("xi", xi);
                    eo.addEntry("index", index);
                    eo.addEntry("Closest value", xp[index]);
                    throw eo;
                }
            }
        }
        
        // Load the 4 stencil points into local variables (cache-friendly)
        size_t const base = index - 2;
        double const x0 = xp[base];
        double const x1 = xp[base + 1];
        double const x2 = xp[base + 2];
        double const x3 = xp[base + 3];
        double const y0 = yp[base];
        double const y1 = yp[base + 1];
        double const y2 = yp[base + 2];
        double const y3 = yp[base + 3];
        
        // Pre-compute spacing values
        double const dx = x2 - x1;           // spacing between x[1] and x[2]
        double const dxL = x1 - x0;          // spacing on left (x[0] to x[1])
        double const dxR = x3 - x2;          // spacing on right (x[2] to x[3])
        double const dxL_plus_dx = dxL + dx;
        double const dxR_plus_dx = dxR + dx;
        
        // Pre-compute inverse of common denominators
        double const inv_dxL_dxLplusdx = 1.0 / (dxL * dxL_plus_dx);
        double const inv_dx_dxL = 1.0 / (dx * dxL);
        double const inv_dx_dxLplusdx = 1.0 / (dx * dxL_plus_dx);
        double const inv_dxR_dxRplusdx = 1.0 / (dxR * dxR_plus_dx);
        double const inv_dx_dxR = 1.0 / (dx * dxR);
        double const inv_dx_dxL_dxLplusdx = 1.0 / (dx * dxL * dxL_plus_dx);
        double const inv_dx_dxR_dxRplusdx = 1.0 / (dx * dxR * dxR_plus_dx);
        
        // Compute derivatives at x[1] and x[2]
        double const m0 = -dx * y0 * inv_dxL_dxLplusdx 
                        + (dx - dxL) * y1 * inv_dx_dxL 
                        + dxL * y2 * inv_dx_dxLplusdx;
        double const m00 = 2.0 * (y0 * dx - dxL_plus_dx * y1 + dxL * y2) * inv_dx_dxL_dxLplusdx;
        
        double const inv_dx_dxRplusdx = 1.0 / (dx * dxR_plus_dx);
        double const m1 = -dxR * y1 * inv_dx_dxRplusdx 
                        - (dx - dxR) * y2 * inv_dx_dxR 
                        + dx * y3 * inv_dxR_dxRplusdx;
        double const m11 = 2.0 * (y1 * dxR - dxR_plus_dx * y2 + dx * y3) * inv_dx_dxR_dxRplusdx;
        
        // Compute interpolation position
        double const dx0 = xi - x1;
        double const dx1 = xi - x2;
        
        // Pre-compute powers of (dx0 - dx1) = dx (the spacing)
        double const ddx = dx0 - dx1;  // = dx = x2 - x1
        double const ddx2 = ddx * ddx;
        double const ddx3 = ddx2 * ddx;
        double const ddx4 = ddx3 * ddx;
        double const ddx5 = ddx4 * ddx;
        double const inv_ddx3 = 1.0 / ddx3;
        double const inv_ddx4 = 1.0 / ddx4;
        double const inv_ddx5 = 1.0 / ddx5;
        
        // Pre-compute powers of dx0
        double const dx0_2 = dx0 * dx0;
        double const dx0_3 = dx0_2 * dx0;
        
        // Pre-compute powers of dx1
        double const dx1_2 = dx1 * dx1;
        
        // Compute interpolation terms
        double const temp0 = y1 + m0 * dx0 + 0.5 * m00 * dx0_2;
        double const temp1 = (y2 - y1 - m0 * ddx - 0.5 * m00 * ddx2) * dx0_3 * inv_ddx3;
        double const temp2 = (3.0 * y1 - 3.0 * y2 + (2.0 * m0 + m1) * ddx + 0.5 * m00 * ddx2) * dx0_3 * dx1 * inv_ddx4;
        double const temp3 = (6.0 * y2 - 6.0 * y1 - 3.0 * (m0 + m1) * ddx - 0.5 * (m11 - m00) * ddx2) * dx0_3 * dx1_2 * inv_ddx5;
        
        return temp0 + temp1 + temp2 + temp3;
    }
}

OndrejEOS::OndrejEOS(std::string const& density_file, std::string const &Pfile, std::string const &csfile,
                     std::string const &Sfile, std::string const &Ufile, std::string const &Tfile, std::string const &CVfile, double lscale, double mscale, double tscale)
    : lscale_(lscale), mscale_(mscale), tscale_(tscale)
{
    auto density = read_vector(density_file);
    mind_ = density[0];
    maxd_ = density.back();
    dd_ = density[1] - density[0];
    P_ = read_vector(Pfile);
    cs_ = read_vector(csfile);
    S_ = read_vector(Sfile);
    U_ = read_vector(Ufile);
    T_ = read_vector(Tfile);
    CV_ = read_vector(CVfile);
    Nt_ = (T_.back() - T_[0]) / (T_[1] - T_[0]) + 1;
    size_t const Nd = (maxd_ - mind_) / dd_ + 1;
    for (size_t i = 0; i < Nd; ++i)
    {
        for (size_t j = 0; j < Nt_; ++j)
        {
            *(S_.begin() + Nt_ * i + j) -= mind_ + dd_ * i;
            *(U_.begin() + Nt_ * i + j) -= mind_ + dd_ * i;
        }
    }
    dvec_.resize(4);
    tvec_.resize(4);
}

double OndrejEOS::InterpData2(double density, double other, std::vector<double> const &otherv,
                              std::vector<double> const &data) const
{
    const size_t d_index = (density - mind_) / dd_;
    for (size_t i = 0; i < 4; ++i)
    {
        try
        {
            tvec_[i] = cubichermite(otherv.begin() + Nt_ * (d_index - 1 + i),
                                    otherv.begin() + Nt_ * (d_index + i), data.begin() + Nt_ * (d_index - 1 + i),
                                    data.begin() + Nt_ * (d_index + i), other);
        }
        catch (UniversalError &eo)
        {
            eo.addEntry("Density", density);
            eo.addEntry("d_index", d_index);
            eo.addEntry("thrown in first loop", 0);
            eo.addEntry("other", other);
            eo.addEntry("First other", otherv[0]);
            eo.addEntry("First data", data[0]);
            eo.addEntry("Data begin",*(data.begin() + Nt_ * (d_index + i)));
            eo.addEntry("other begin",*(otherv.begin() + Nt_ * (d_index + i)));
            throw eo;
        }
    }
    for (size_t i = 0; i < 4; ++i)
        dvec_[i] = mind_ + dd_ * (d_index - 1 + i);
    try
    {
        return cubichermite(dvec_.begin(), dvec_.end(), tvec_.begin(), tvec_.end(), density);
    }
    catch (UniversalError &eo)
    {
        eo.addEntry("Density", density);
        eo.addEntry("d_index", d_index);
        eo.addEntry("Other", other);
        eo.addEntry("thrown in second loop", 0);
        for (size_t j = 0; j < 4; ++j)
            eo.addEntry("tvec", tvec_[j]);
        for (size_t j = 0; j < 4; ++j)
            eo.addEntry("dvec", dvec_[j]);
        eo.addEntry("First data", data[0]);
        throw eo;
    }
}

double OndrejEOS::dp2e(double d, double p, tvector const &tracers, vector<string> const &tracernames) const
{
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    double p_cgs = p * mscale_ / (tscale_ * tscale_ * lscale_);
    if(p_cgs < d_cgs * 1e8)
        p_cgs = d_cgs * 1e8;
    if (p_cgs > 1e16 * d_cgs)
        return tscale_ * tscale_ * 1.5 * p_cgs / (d_cgs * lscale_ * lscale_);
    else
    {
        double value = std::numeric_limits<double>::max();
        try
        {
            value = InterpData2(std::log(d_cgs), std::log(p_cgs), P_, U_);
        }
        catch(UniversalError &eo)
        {
            eo.addEntry("located in dp2e", 0);
            throw eo;
        }        
        return tscale_ * tscale_ * std::exp(value) / (lscale_ * lscale_);
    }
}

double OndrejEOS::dp2T(double d, double p, tvector const &tracers, vector<string> const &tracernames) const
{
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    double p_cgs = p * mscale_ / (tscale_ * tscale_ * lscale_);
    if (p_cgs > 1e16 * d_cgs)
        return p_cgs * 7.452e-9 / d_cgs;
    else
    {
        double value = std::numeric_limits<double>::max();
        try
        {
            value = InterpData2(std::log(d_cgs), std::log(p_cgs), P_, T_);
        }
        catch(UniversalError &eo)
        {
            eo.addEntry("located in dp2T", 0);
            throw eo;
        }
        return std::exp(value);
    }
}

double OndrejEOS::de2T(double const d, double const e, tvector const& tracers, vector<string> const& tracernames) const
{
    double e_cgs = e * lscale_ * lscale_ / (tscale_ * tscale_);
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    if (e_cgs > 1e16)
        return e_cgs / (1.5 * 1.3419e8);
    else
    {
        double value = std::numeric_limits<double>::max();
        try
        {
            value = InterpData2(std::log(d_cgs), std::log(e_cgs), U_, T_);
        }
        catch(UniversalError &eo)
        {
            eo.addEntry("located in de2T", 0);
            throw eo;
        }
        return std::exp(value);
    }
}

double OndrejEOS::dT2p(double d, double T, tvector const &tracers, vector<string> const &tracernames) const
{
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    if (T > 5e7)
        return tscale_ * tscale_ * lscale_ * T * d_cgs * 1.3419e8 / mscale_;
    else
    {
        double value = std::numeric_limits<double>::max();
        try
        {
            value = InterpData2(std::log(d_cgs), std::log(T), T_, P_);
        }
        catch(UniversalError &eo)
        {
            eo.addEntry("located in dT2p", 0);
            throw eo;
        }
        return tscale_ * tscale_ * lscale_ * std::exp(value) / mscale_;
    }
}

double OndrejEOS::dT2e(double d, double T, tvector const &tracers, vector<string> const &tracernames) const
{
    if (T > 8e7)
        return tscale_ * tscale_ * T * 1.5 * 1.3419e8 / (lscale_ * lscale_);
    else
    {
        double value = std::numeric_limits<double>::max();
        try
        {
            value = InterpData2(std::log(d * mscale_ / (lscale_ * lscale_ * lscale_)), std::log(T), T_, U_);
        }
        catch(UniversalError &eo)
        {
            eo.addEntry("located in dT2e", 0);
            throw eo;
        }
        return tscale_ * tscale_ * std::exp(value) / (lscale_ * lscale_);
    }
}

double OndrejEOS::de2p(double d, double e, tvector const &tracers, vector<string> const &tracernames) const
{
    double e_cgs = e * lscale_ * lscale_ / (tscale_ * tscale_);
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    if (e_cgs > 1e16)
        return tscale_ * tscale_ * lscale_ * e_cgs * d_cgs * 0.66666666666 / mscale_;
    else
    {
        double value = std::numeric_limits<double>::max();
        try
        {
            value = InterpData2(std::log(d_cgs), std::log(e_cgs), U_, P_);
        }
        catch(UniversalError &eo)
        {
            eo.addEntry("located in de2p", 0);
            throw eo;
        }
        double newp = tscale_ * tscale_ * lscale_ * std::exp(value) / mscale_;
        return newp;
    }
}

double OndrejEOS::dp2c(double d, double p, tvector const &tracers, vector<string> const &tracernames) const
{
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    double p_cgs = p * mscale_ / (tscale_ * tscale_ * lscale_);
    if(p_cgs < d_cgs * 1e8)
        p_cgs = d_cgs * 1e8;
    if (p_cgs > 1e16 * d_cgs)
        return tscale_ * std::sqrt(5 * p_cgs / (3 * d_cgs)) / lscale_;
    else
    {
        double value = std::numeric_limits<double>::max();
        try
        {
            value = InterpData2(std::log(d_cgs), std::log(p_cgs), P_, cs_);
        }
        catch(UniversalError &eo)
        {
            eo.addEntry("located in dp2c", 0);
            throw eo;
        }
        return tscale_ * std::sqrt(value) / lscale_;
    }
}

double OndrejEOS::dp2cv(double d, double p, tvector const &tracers, vector<string> const &tracernames) const
{
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    double p_cgs = p * mscale_ / (tscale_ * tscale_ * lscale_);
    double const cv_factor = lscale_ * tscale_ * tscale_ / mscale_;
    if (p_cgs > 1e15 * d_cgs)
        return 2.0128e8 * d_cgs * cv_factor;
    else
    {
        double value = std::numeric_limits<double>::max();
        try
        {
            value = InterpData2(std::log(d_cgs), std::log(p_cgs), P_, CV_);
        }
        catch(UniversalError &eo)
        {
            eo.addEntry("located in dp2cv", 0);
            throw eo;
        }
        return std::exp(value) * cv_factor;
    }
}

double OndrejEOS::dT2cv(double const d, double const T, tvector const& tracers, vector<string> const& tracernames) const
{
    double const d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    double const cv_factor = lscale_ * tscale_ * tscale_ / mscale_;
    if(T > 1e6 || d_cgs > 10)
        return  2.0128e8 * d_cgs * cv_factor;
    else
    {
        double value = std::numeric_limits<double>::max();
        try
        {
            value = InterpData2(std::log(d_cgs), std::log(T), T_, CV_);
        }
        catch(UniversalError &eo)
        {
            eo.addEntry("located in dT2cv", 0);
            throw eo;
        }
        return std::exp(value) * cv_factor;
    }
}

double OndrejEOS::de2c(double d, double e, tvector const &tracers, vector<string> const &tracernames) const
{
    double p = de2p(d, e);
    return dp2c(d, p);
}

double OndrejEOS::dp2s(double d, double p, tvector const &tracers, vector<string> const &tracernames) const
{
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    double p_cgs = p * mscale_ / (tscale_ * tscale_ * lscale_);
    if(p_cgs < 0)
        throw UniversalError("Negative Pressure in dp2s");
    if (p_cgs > 1e16 * d_cgs)
        return tscale_ * tscale_ * std::pow(10.0, (8.128 + std::log10(-38.43 + std::log(std::pow(p_cgs, 1.5) * std::pow(d_cgs, -2.5))))) / (lscale_ * lscale_);
    double value = std::numeric_limits<double>::max();
    try
    {
        value = InterpData2(std::log(d_cgs), std::log(p_cgs), P_, S_);
    }
    catch(UniversalError &eo)
    {
        eo.addEntry("located in dp2s", 0);
        throw eo;
    }
    return tscale_ * tscale_ * std::exp(value) / (lscale_ * lscale_);
}

double OndrejEOS::sd2p(double s, double d, tvector const &tracers, vector<string> const &tracernames) const
{
    double smax = dp2s(d, 5.2 * d, tracers, tracernames);
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    double s_cgs = s * lscale_ * lscale_ / (tscale_ * tscale_);
    if(s_cgs < 0)
        throw UniversalError("Negative Entropy in sd2p");
    if (s > smax)
        return tscale_ * tscale_ * lscale_ * std::pow(std::pow(d_cgs, 2.5) * std::exp(s_cgs * std::pow(10.0, -8.128) + 38.43), 0.666666666) / mscale_;
    else
    {
        double value = std::numeric_limits<double>::max();
        try
        {
            value = InterpData2(std::log(d_cgs), std::log(s_cgs), S_, P_);
        }
        catch(UniversalError &eo)
        {
            eo.addEntry("located in sd2p", 0);
            throw eo;
        }
        return tscale_ * tscale_ * lscale_ * std::exp(value) / mscale_;
    }
}

