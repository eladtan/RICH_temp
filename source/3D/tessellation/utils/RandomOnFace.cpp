#include "RandomOnFace.hpp"
#include <algorithm>

Vector3D RandomPointOnFace(const Tessellation3D &voronoi, size_t faceIndex)
{
    static std::mt19937 re(0);
    static std::uniform_real_distribution<double> dist(0.0, 1.0);

    static size_t cachedFace = SIZE_MAX;
    static const Tessellation3D *cachedGrid = nullptr;
    static size_t cachedBuildGeneration = SIZE_MAX;
    static std::vector<double> cumAreas;
    static std::vector<std::array<size_t, 3>> tris;
    static double totalArea = 0;

    if(faceIndex != cachedFace or &voronoi != cachedGrid
       or voronoi.GetBuildGeneration() != cachedBuildGeneration)
    {
        cachedFace = faceIndex;
        cachedGrid = &voronoi;
        cachedBuildGeneration = voronoi.GetBuildGeneration();

        cumAreas.clear();
        tris.clear();
        totalArea = 0;

        const auto &fv = voronoi.GetPointsInFace(faceIndex);
        const std::vector<Vector3D> &verts = voronoi.GetFacePoints();
        for(size_t i = 1; i + 1 < fv.size(); i++)
        {
            double area = abs(CrossProduct(verts[fv[i]] - verts[fv[0]], verts[fv[i + 1]] - verts[fv[0]]));
            totalArea += area;
            cumAreas.push_back(totalArea);
            tris.push_back({fv[0], fv[i], fv[i + 1]});
        }
    }

    const std::vector<Vector3D> &verts = voronoi.GetFacePoints();

    double r = dist(re) * totalArea;
    size_t idx = static_cast<size_t>(std::lower_bound(cumAreas.begin(), cumAreas.end(), r) - cumAreas.begin());
    if(idx >= tris.size()) idx = tris.size() - 1;

    double r1 = dist(re), r2 = dist(re);
    if(r1 + r2 > 1)
    {
        r1 = 1 - r1;
        r2 = 1 - r2;
    }

    const auto &tv = tris[idx];
    return (1 - r1 - r2) * verts[tv[0]] + r1 * verts[tv[1]] + r2 * verts[tv[2]];
}
