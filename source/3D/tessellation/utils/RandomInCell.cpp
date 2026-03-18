#include "RandomInCell.hpp"
#include <algorithm>

Vector3D RandomPointInCell(const Tessellation3D &voronoi, size_t cellIndex)
{
    static std::mt19937 re(0);
    static std::uniform_real_distribution<double> dist(0.0, 1.0);

    static size_t cachedCell = SIZE_MAX;
    static const Tessellation3D *cachedGrid = nullptr;
    static size_t cachedBuildGeneration = SIZE_MAX;
    static std::vector<double> cumVolumes;
    static std::vector<std::array<size_t, 3>> tris;
    static double totalVolume = 0;
    static Vector3D center;

    if(cellIndex != cachedCell or &voronoi != cachedGrid
       or voronoi.GetBuildGeneration() != cachedBuildGeneration)
    {
        cachedCell = cellIndex;
        cachedGrid = &voronoi;
        cachedBuildGeneration = voronoi.GetBuildGeneration();
        center = voronoi.GetMeshPoint(cellIndex);

        cumVolumes.clear();
        tris.clear();
        totalVolume = 0;

        for(const size_t &faceIdx : voronoi.GetCellFaces(cellIndex))
        {
            const auto &fv = voronoi.GetPointsInFace(faceIdx);
            if(fv.size() < 3) continue;
            const std::vector<Vector3D> &verts = voronoi.GetFacePoints();
            for(size_t i = 1; i + 1 < fv.size(); i++)
            {
                Vector3D a = verts[fv[0]] - center;
                Vector3D b = verts[fv[i]] - center;
                Vector3D c = verts[fv[i + 1]] - center;
                double vol = std::abs(ScalarProd(a, CrossProduct(b, c)));
                totalVolume += vol;
                cumVolumes.push_back(totalVolume);
                tris.push_back({fv[0], fv[i], fv[i + 1]});
            }
        }
    }

    const std::vector<Vector3D> &verts = voronoi.GetFacePoints();

    double r = dist(re) * totalVolume;
    size_t idx = static_cast<size_t>(std::lower_bound(cumVolumes.begin(), cumVolumes.end(), r) - cumVolumes.begin());
    if(idx >= tris.size()) idx = tris.size() - 1;

    double s = dist(re), t = dist(re), u = dist(re);
    if(s > t) std::swap(s, t);
    if(t > u) std::swap(t, u);
    if(s > t) std::swap(s, t);

    const auto &tv = tris[idx];
    return s * verts[tv[0]] + (t - s) * verts[tv[1]] + (u - t) * verts[tv[2]] + (1 - u) * center;
}
