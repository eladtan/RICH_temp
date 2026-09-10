#ifndef RMTV_REFERENCE_HPP
#define RMTV_REFERENCE_HPP
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rmtv
{
constexpr double gamma = 1.25;
constexpr double alpha = 9.0 / 13.0;
constexpr double kappa = -19.0 / 9.0;
constexpr double beta0 = 7.197534e7;
inline double zeta()
{
    return std::pow(std::pow(0.5 * beta0, 1.0 / 12.0) / alpha, alpha);
}
struct Scales
{
    double length = 1.0, density = 1.0, time = 0.0256, temperature = 937.5;
    double velocity() const
    {
        return length / time;
    }
    double energy() const
    {
        return velocity() * velocity();
    }
    double cv() const
    {
        return energy() / ((gamma - 1) * temperature);
    }
    // chi_phys = chi0_phys rho_phys^-2 T_phys^6.5
    double chi0() const
    {
        return std::pow(density, 3) * std::pow(length, 4) /
               (std::pow(time, 3) * std::pow(temperature, 7.5));
    }
    double age(double front) const
    {
        return time * std::pow(front / (2 * zeta()), 1 / alpha);
    }
    double front(double age) const
    {
        return 2 * zeta() * std::pow(age / time, alpha);
    }
};
struct State
{
    double rho, v, T;
};
class Reference
{
    using Row = std::array<double, 4>;
    std::vector<Row> inside_, outside_;
    static Row interpolate(const std::vector<Row> &rows, double x)
    {
        std::vector<Row>::const_iterator it = std::upper_bound(rows.begin(), rows.end(), x,
                                                               [](double v, const Row &r)
                                                               { return v < r[0]; });
        if(it == rows.begin())
        {
            return *it;
        }
        if(it == rows.end())
        {
            return rows.back();
        }
        const Row &a = *(it - 1);
        const Row &b = *it;
        double f = (x - a[0]) / (b[0] - a[0]);
        Row out{};
        for(int j = 0; j < 4; ++j)
        {
            out[j] = a[j] + f * (b[j] - a[j]);
        }
        return out;
    }

  public:
    explicit Reference(const std::string &path)
    {
        std::ifstream in(path);
        if(!in)
        {
            throw std::runtime_error("Cannot open reference: " + path);
        }
        std::string line;
        bool outer = false;
        double previous = -1;
        while(std::getline(in, line))
        {
            if(line.empty() || line[0] == '#')
            {
                continue;
            }
            std::istringstream stream(line);
            Row r{};
            if(!(stream >> r[0] >> r[1] >> r[2] >> r[3]))
            {
                throw std::runtime_error("Malformed reference row");
            }
            for(double v : r)
            {
                if(!std::isfinite(v))
                {
                    throw std::runtime_error("Nonfinite reference");
                }
            }
            if(r[0] == 1 && previous == 1)
            {
                outer = true;
            }
            if(r[0] < previous || r[1] <= 0 || r[3] <= 0)
            {
                throw std::runtime_error("Invalid reference shape");
            }
            (outer ? outside_ : inside_).push_back(r);
            previous = r[0];
        }
        if(inside_.size() < 3 || outside_.size() < 3 || inside_.front()[0] != 0 ||
            inside_.back()[0] != 1 || outside_.front()[0] != 1 || outside_.back()[0] < 1.999)
            throw std::runtime_error(
                "Reference must contain separate inner and outer shock branches");
    }
    State operator()(double radius, double age, const Scales &s) const
    {
        if(radius < 0 || !(age > 0))
        {
            throw std::runtime_error("Invalid reference radius/age");
        }
        const double R = 0.5 * s.front(age), xi = radius / (s.length * R);
        if(xi >= 2)
        {
            return {s.density * std::pow(radius / s.length, kappa), 0, 0};
        }
        Row a = interpolate(xi <= 1 ? inside_ : outside_, xi);
        // Resolve the last tiny interval to T=0 at the exact heat front.
        if(xi > outside_.back()[0])
        {
            const double f = (2 - xi) / (2 - outside_.back()[0]);
            a[1] = std::pow(2.0, kappa) + f * (a[1] - std::pow(2.0, kappa));
            a[2] *= f;
            a[3] *= std::pow(f, 1.0 / 6.5);
        }
        const double vscale = alpha * R / (age / s.time);
        return {s.density * std::pow(R, kappa) * a[1], s.velocity() * vscale * a[2],
                s.temperature * vscale * vscale * a[3]};
    }
    // Midpoint tensor quadrature over a Cartesian cell, conserving total gas
    // energy, including unresolved kinetic energy. Even q avoids r=0 samples.
    std::array<double, 6> average(double x, double y, double z, double dx, double age,
                                  const Scales &s, int q, double floor, int depth = 2) const
    {
        std::array<double, 6> a{}; // rho, momentum xyz, total energy density, floor energy
        // Refine cells cut by either spherical front. Uniform sampling alone
        // aliases the large density jump even at quite high quadrature orders.
        const double h = 0.5 * dx;
        double near2 = 0, far2 = 0;
        for(double v : {x, y, z})
        {
            near2 += std::pow(std::max(0.0, std::abs(v) - h), 2);
            far2 += std::pow(std::abs(v) + h, 2);
        }
        const double rf = s.length * s.front(age), rs = 0.5 * rf;
        // Tangency has zero measure. Roundoff in serial/MPI cell centers must
        // not select different quadrature rules at a tangent face or corner.
        const auto crosses = [&](double radius)
        {
            return near2 < radius * radius * (1 - 1e-12) &&
                   far2 > radius * radius * (1 + 1e-12);
        };
        if(depth > 0 && (crosses(rs) || crosses(rf)))
        {
            for(int i : {-1, 1})
            {
                for(int j : {-1, 1})
                {
                    for(int k : {-1, 1})
                    {
                        std::array<double, 6> child = average(x + i * h / 2, y + j * h / 2,
                                                               z + k * h / 2, h, age, s, q, floor,
                                                               depth - 1);
                        for(int n = 0; n < 6; ++n)
                        {
                            a[n] += child[n] / 8;
                        }
                    }
                }
            }
            return a;
        }
        for(int i = 0; i < q; ++i)
        {
            for(int j = 0; j < q; ++j)
            {
                for(int k = 0; k < q; ++k)
                {
                    const double xx = x + dx * ((i + 0.5) / q - 0.5),
                                 yy = y + dx * ((j + 0.5) / q - 0.5),
                                 zz = z + dx * ((k + 0.5) / q - 0.5),
                                 r = std::sqrt(xx * xx + yy * yy + zz * zz);
                    State p = (*this)(r, age, s);
                    const double T = std::max(p.T, floor);
                    a[0] += p.rho;
                    if(r > 0)
                    {
                        a[1] += p.rho * p.v * xx / r;
                        a[2] += p.rho * p.v * yy / r;
                        a[3] += p.rho * p.v * zz / r;
                    }
                    a[4] += p.rho * (s.cv() * T + 0.5 * p.v * p.v);
                    a[5] += p.rho * s.cv() * (T - p.T);
                }
            }
        }
        for(double &v : a)
        {
            v /= double(q) * q * q;
        }
        return a;
    }
};
} // namespace rmtv
#endif
