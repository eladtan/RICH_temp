#include "RandomInCell.hpp"

Vector3D RandomPointInCell(const Tessellation3D &voronoi, size_t cellIndex)
{
    static std::mt19937 re(0); // use a seed for reproducibility
    static std::uniform_real_distribution<double> dist(EPSILON, 1 - EPSILON);
    const std::vector<Vector3D> &voronoiVertices = voronoi.GetFacePoints();

    Vector3D point;
    double SumOfAlpha = 0;
    for(const size_t &faceIdx : voronoi.GetCellFaces(cellIndex))
    {
        const auto &faceVertices = voronoi.GetPointsInFace(faceIdx);
        for(const size_t &vertexIdx : faceVertices)
        {
            double uniform = dist(re);
            double alpha = -std::log(uniform);
            SumOfAlpha += alpha;
            point += alpha * voronoiVertices[vertexIdx];
        }
    }
    point = point * (1 / SumOfAlpha);
    return point;
}
