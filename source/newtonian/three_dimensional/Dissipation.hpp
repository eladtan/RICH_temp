#include "hdsim_3d.hpp"
#include "Hllc3D.hpp"

#ifndef DISSIPATION_HPP
#define DISSIPATION_HPP 1

class Dissipation
{
    private:
        Hllc3D const& rs_;
        EquationOfState const& eos_;
    public:
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > face_values;

    Dissipation(Hllc3D const& rs, EquationOfState const& eos): rs_(rs), eos_(eos){}

    std::vector<double> CalcDissipation(HDSim3D const& sim) const
    {
        Tessellation3D const& tess = sim.getTesselation();
        size_t const N = tess.GetPointNo();
        size_t const Nfaces = tess.GetTotalFacesNumber();
        if(Nfaces != face_values.size())
        {
            UniversalError eo("Wrong face number");
            eo.addEntry("Nfaces", Nfaces);
            eo.addEntry("face_values", face_values.size());
            throw eo;
        }
        std::vector<ComputationalCell3D> const& cells = sim.getCells();
        std::vector<double> result(N, 0);
        for(size_t i = 0; i < Nfaces; ++i)
        {
            auto neigh = tess.GetFaceNeighbors(i);
            bool is_first = neigh.first < N;
            bool is_second = neigh.second < N;
            if(is_first || is_second)
            {
                double const A = tess.GetArea(i);
                const Vector3D normal = normalize(tess.Normal(i));
                auto UstarPstar = rs_.GetUstarPstar(face_values[i].first, face_values[i].second, eos_, normal);
                if(is_first)
                {
                    result[neigh.first] += A *((UstarPstar.first - ScalarProd(normal, cells[neigh.first].velocity)) * UstarPstar.second
                        + cells[neigh.first].pressure * UstarPstar.first);
                }
                if(is_second)
                {
                    result[neigh.second] -= A *((UstarPstar.first - ScalarProd(normal, cells[neigh.second].velocity)) * UstarPstar.second
                        + cells[neigh.second].pressure * UstarPstar.first);
                }
            }
        }
        for(size_t i = 0; i < N; ++i)
            result[i] /= tess.GetVolume(i);
        return result;
    }
};
#endif // DISSIPATION_HPP