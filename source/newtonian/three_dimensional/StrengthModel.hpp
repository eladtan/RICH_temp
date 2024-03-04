/* \file StrengthModel.hpp
\brief Abstract class for different constitute models
*/

#ifndef STRENGTHMODEL_HPP
#define STRENGTHMODEL_HPP 1
#include "computational_cell.hpp"

class StrengthModel
{
    public:
    virtual double getG(ComputationalCell3D const& cell) const = 0;
    virtual double getY(ComputationalCell3D const& cell) const = 0;
};

class ConstStrength : public StrengthModel
{
public:
    ConstStrength(double G, double Y) : G_(G), Y_(Y) {};
    inline double getG(ComputationalCell3D const& cell) const override
    {
        return G_;
    }
    inline double getY(ComputationalCell3D const& cell) const override
    {
        return Y_;
    }
private:
    const double G_;
    const double Y_;
};

#endif