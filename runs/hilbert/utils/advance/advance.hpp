#ifndef _ADVANCE_HPP
#define _ADVANCE_HPP

#include <numeric>
#include "3D/elementary/Vector3D.hpp"

void changePoints(std::vector<Vector3D> &points, const std::vector<size_t> &indicesToBuild, const Vector3D &ll, const Vector3D &ur)
{
    const double EPS1 = 0.001;
    const double EPS2 = 0.001;
    for(const size_t &pointIdx : indicesToBuild)
    {
        if(pointIdx >= points.size())
        {
            continue;
        }
        Vector3D &point = points[pointIdx];
        if(rand() % 14 == 0)
        {
            for(int j = 0; j < 3; j++)
            {
                if(rand() % 2 == 0)
                {
                    double val = point[j] + EPS1;
                    if((val > ll[j]) and (val < ur[j]))
                    {
                        point[j] = val;
                    }
                }
                else
                {
                    double val = point[j] - EPS2;
                    if((val > ll[j]) and (val < ur[j]))
                    {
                        point[j] = val;
                    }
                }
            }
        }
    }
}

void changePoints(std::vector<Vector3D> &points, const Vector3D &ll, const Vector3D &ur)
{
    std::vector<size_t> indicesToBuild(points.size());
    std::iota(indicesToBuild.begin(), indicesToBuild.end(), 0);
    changePoints(points, indicesToBuild, ll, ur);
}

#endif // _ADVANCE_HPP