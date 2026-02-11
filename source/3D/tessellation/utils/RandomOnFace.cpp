#include "RandomOnFace.hpp"

Vector3D RandomPointOnFace(const Tessellation3D &voronoi, size_t faceIndex)
{
    static std::mt19937 re(0); // use a seed for reproducibility
    static std::uniform_real_distribution<double> dist(EPSILON, 1 - EPSILON);
    const std::vector<Vector3D> &voronoiVertices = voronoi.GetFacePoints();

    Vector3D point;
    double SumOfAlpha = 0;
    const auto &faceVertices = voronoi.GetPointsInFace(faceIndex);
    for(const size_t &vertexIdx : faceVertices)
    {
        double uniform = dist(re);
        double alpha = -std::log(uniform);
        SumOfAlpha += alpha;
        point += alpha * voronoiVertices[vertexIdx];
    }
    point = point * (1 / SumOfAlpha);
    return point;
}
