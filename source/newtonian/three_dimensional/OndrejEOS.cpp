#include "OndrejEOS.hpp"
#include "../../misc/universal_error.hpp"
#include "../../misc/simple_io.hpp"
#include <cmath>

namespace
{
    typedef std::vector<double>::const_iterator cit;

    double cubichermite(cit x_begin, cit x_end, cit y_begin, cit y_end, double xi)
    {
        size_t index = static_cast<size_t>(std::upper_bound(x_begin, x_end, xi) - x_begin);
        if ((index < 2 && xi < *(x_begin + 1)) || index >= static_cast<size_t>(y_end - y_begin - 1))
        {
            if(index == 1 && std::abs(*(x_begin + index) -  xi) < 1e-13)
                ++index;
            else
            {
                if(index == 3 && std::abs(*(x_begin + index) -  xi) < 1e-13)
                    --index;
                else
                {
                    UniversalError eo("Bad interpolation");
                    eo.addEntry("xi", xi);
                    eo.addEntry("index", index);
                    eo.addEntry("Closest value", *(x_begin + index));
                    throw eo;
                }
            }
        }
        double m0 = (*(y_begin + index) - *(y_begin + index - 2)) / (*(x_begin + index) - *(x_begin + index - 2));
        double m1 = (*(y_begin + index + 1) - *(y_begin + index - 1)) / (*(x_begin + index + 1) - *(x_begin + index - 1));
        double dx = *(x_begin + index) - *(x_begin + index - 1);

        std::array<double, 4> x, y;
        for (size_t i = 0; i < 4; ++i)
        {
            x[i] = *(x_begin + (index + i - 2));
            y[i] = *(y_begin + (index + i - 2));
        }
        double dxother = *(x_begin + index - 1) - *(x_begin + index - 2);
        m0 = -dx * (*(y_begin + index - 2)) / (dxother * (dxother + dx)) + (dx - dxother) * (*(y_begin + index - 1)) / (dx * dxother) + dxother * (*(y_begin + index)) / (dx * (dxother + dx));
        double m00 = 2 * (y[0] * dx - (dxother + dx) * y[1] + dxother * y[2]) / (dx * dxother * (dxother + dx));
        dxother = *(x_begin + index + 1) - *(x_begin + index);
        m1 = -dxother * (*(y_begin + index - 1)) / (dx * (dxother + dx)) - (dx - dxother) * (*(y_begin + index)) / (dx * dxother) + dx * (*(y_begin + index + 1)) / (dxother * (dxother + dx));
        double m11 = 2 * (y[1] * dxother - (dxother + dx) * y[2] + dx * y[3]) / (dx * dxother * (dxother + dx));
        double dx0 = xi - x[1];
        double dx1 = xi - x[2];

        double temp0 = y[1] + m0 * dx0 + 0.5 * m00 * dx0 * dx0;
        double temp1 = (y[2] - y[1] - m0 * (dx0 - dx1) - m00 * 0.5 * (dx0 - dx1) * (dx0 - dx1)) * dx0 * dx0 * dx0 / ((dx0 - dx1) * (dx0 - dx1) * (dx0 - dx1));
        double temp2 = (3 * y[1] - 3 * y[2] + (2 * m0 + m1) * (dx0 - dx1) + 0.5 * m00 * (dx0 - dx1) * (dx0 - dx1)) * dx0 * dx0 * dx0 * dx1 / ((dx0 - dx1) * (dx0 - dx1) * (dx0 - dx1) * (dx0 - dx1));
        double temp3 = (6 * y[2] - 6 * y[1] - 3 * (m0 + m1) * (dx0 - dx1) + 0.5 * (m11 - m00) * (dx0 - dx1) * (dx1 - dx0)) * dx0 * dx0 * dx0 * dx1 * dx1 / ((dx0 - dx1) * (dx0 - dx1) * (dx0 - dx1) * (dx0 - dx1) * (dx0 - dx1));
        return temp0 + temp1 + temp2 + temp3;
    }
}

OndrejEOS::OndrejEOS(double mind, double maxd, double dd, std::string const &Pfile, std::string const &csfile,
                     std::string const &Sfile, std::string const &Ufile, std::string const &Tfile, std::string const &CVfile, double lscale, double mscale, double tscale)
    : mind_(mind), maxd_(maxd), dd_(dd), lscale_(lscale), mscale_(mscale), tscale_(tscale)
{
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
        return tscale_ * tscale_ * std::exp(InterpData2(std::log(d_cgs), std::log(p_cgs), P_, U_)) / (lscale_ * lscale_);
}

double OndrejEOS::dp2T(double d, double p, tvector const &tracers, vector<string> const &tracernames) const
{
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    double p_cgs = p * mscale_ / (tscale_ * tscale_ * lscale_);
    if (p_cgs > 1e16 * d_cgs)
        return p_cgs * 7.452e-9 / d_cgs;
    else
        return std::exp(InterpData2(std::log(d_cgs), std::log(p_cgs), P_, T_));
}

double OndrejEOS::de2T(double const d, double const e, tvector const& tracers, vector<string> const& tracernames) const
{
    double e_cgs = e * lscale_ * lscale_ / (tscale_ * tscale_);
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    if (e_cgs > 1e16)
        return e_cgs / (1.5 * 1.3419e8);
    else
        return std::exp(InterpData2(std::log(d_cgs), std::log(e_cgs), U_, T_));
}

double OndrejEOS::dT2p(double d, double T, tvector const &tracers, vector<string> const &tracernames) const
{
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    if (T > 5e7)
        return tscale_ * tscale_ * lscale_ * T * d_cgs * 1.3419e8 / mscale_;
    else
        return tscale_ * tscale_ * lscale_ * std::exp(InterpData2(std::log(d_cgs), std::log(T), T_, P_)) / mscale_;
}

double OndrejEOS::dT2e(double d, double T, tvector const &tracers, vector<string> const &tracernames) const
{
    if (T > 8e7)
        return tscale_ * tscale_ * T * 1.5 * 1.3419e8 / (lscale_ * lscale_);
    else
        return tscale_ * tscale_ * std::exp(InterpData2(std::log(d * mscale_ / (lscale_ * lscale_ * lscale_)), std::log(T), T_, U_)) / (lscale_ * lscale_);
}

double OndrejEOS::de2p(double d, double e, tvector const &tracers, vector<string> const &tracernames) const
{
    double e_cgs = e * lscale_ * lscale_ / (tscale_ * tscale_);
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    if (e_cgs > 1e16)
        return tscale_ * tscale_ * lscale_ * e_cgs * d_cgs * 0.66666666666 / mscale_;
    else
    {
        double newp = tscale_ * tscale_ * lscale_ * std::exp(InterpData2(std::log(d_cgs), std::log(e_cgs), U_, P_)) / mscale_;
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
        return tscale_ * std::sqrt(InterpData2(std::log(d_cgs), std::log(p_cgs), P_, cs_)) / lscale_;
}

double OndrejEOS::dp2cv(double d, double p, tvector const &tracers, vector<string> const &tracernames) const
{
    double d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    double p_cgs = p * mscale_ / (tscale_ * tscale_ * lscale_);
    double const cv_factor = lscale_ * tscale_ * tscale_ / mscale_;
    if (p_cgs > 1e15 * d_cgs)
        return 2.0128e8 * d_cgs * cv_factor;
    else
        return std::exp(InterpData2(std::log(d_cgs), std::log(p_cgs), P_, CV_)) * cv_factor;
}

double OndrejEOS::dT2cv(double const d, double const T, tvector const& tracers, vector<string> const& tracernames) const
{
    double const d_cgs = d * mscale_ / (lscale_ * lscale_ * lscale_);
    double const cv_factor = lscale_ * tscale_ * tscale_ / mscale_;
    if(T > 1e6 || d_cgs > 10)
        return  2.0128e8 * d_cgs * cv_factor;
    else
        return std::exp(InterpData2(std::log(d_cgs), std::log(T), T_, CV_)) * cv_factor;
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
    return tscale_ * tscale_ * std::exp(InterpData2(std::log(d_cgs), std::log(p_cgs), P_, S_)) / (lscale_ * lscale_);
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
        return tscale_ * tscale_ * lscale_ * std::exp(InterpData2(std::log(d_cgs), std::log(s_cgs), S_, P_)) / mscale_;
}
