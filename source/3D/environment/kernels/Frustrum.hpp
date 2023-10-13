#ifndef FRUSTRUM_KERNEL_HPP
#define FRUSTRUM_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "3D/elementary/Face.hpp"
#include "3D/elementary/Mat44.hpp"
#include "IndexingKernel3D.hpp"

#define NUM_FACES 6
#define FACE_EDGES_NUMBER 4
#define VERTICES_NUMBER 4

// see here: https://math.stackexchange.com/questions/2265255/mapping-a-3d-point-inside-a-hexahedron-to-a-unit-cube

namespace
{
    inline Vector3D GetNormal(const Face &face)
    {
        return normalize(CrossProduct(face.vertices[1] - face.vertices[0], face.vertices[2] - face.vertices[1]));
    }

    inline Vector3D GetIntersection(const Vector3D &point1, const Vector3D &vec1, const Vector3D &point2, const Vector3D &vec2)
    {
        Vector3D point = point1 - point2;
        Vector3D vec = vec1 - vec2;

        // `point1` + t`vec1` and `point2`+s`vec2` are two lines
        double div = (vec2[0] * vec1[1]) - (vec2[1] * vec1[0]);
        if(div == 0)
        {
            throw UniversalError("Parallel to each other");
        }
        double t = (vec2[1] * (point1[0] - point2[0]) - vec2[0] * (point1[1] - point2[1])) / div;
        double s = ((point1[0] - point2[0]) + (vec1[0] * t)) / vec2[0];
        if((point1[2] - point2[2]) != (s - t) * (vec2[2] - vec1[2]))
        {
            throw UniversalError("Vectors are not on the same plane");
        }
        return point1 + t * vec1;
    }
}
class Frustrum : public IndexingKernel3D
{
public:
    inline Frustrum(const std::vector<Face> &faces, const IndexingKernel3D *indexing = nullptr);

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = (this->indexing == nullptr)? vector : (*this->indexing)(vector);

        // matrix multiplication
        double almostX = (this->P(0, 0) * vec[0]) + (this->P(0, 1) * vec[1]) + (this->P(0, 2) * vec[2]) + this->P(0, 3);
        double almostY = (this->P(1, 0) * vec[0]) + (this->P(1, 1) * vec[1]) + (this->P(1, 2) * vec[2]) + this->P(1, 3);
        double almostZ = (this->P(2, 0) * vec[0]) + (this->P(2, 1) * vec[1]) + (this->P(2, 2) * vec[2]) + this->P(2, 3);
        double factor = 1/((this->P(3, 0) * vec[0]) + (this->P(3, 1) * vec[1]) + (this->P(3, 2) * vec[2]) + this->P(3, 3));
        return (Vector3D(almostX, almostY, almostZ) * factor);
    };

private:
    Mat44<double> P;
    const IndexingKernel3D *indexing;

    Vector3D find_S(const std::vector<Face> &faces) const;
};

Vector3D Frustrum::find_S(const std::vector<Face> &faces) const
{    
    // first find the parallel faces
    std::vector<Vector3D> normals;
    for(const Face &face : faces)
    {
        if(face.vertices.size() != VERTICES_NUMBER)
        {
            throw UniversalError("Can not use 'Frustrum' kernelization when there's a face with " + std::to_string(face.vertices.size()) + " vertices (expected " + std::to_string(VERTICES_NUMBER) + ")");
        }
        normals.emplace_back(GetNormal(face));
    }
    
    std::pair<size_t, size_t> parallelIdx;
    for(size_t faceIdx = 0; faceIdx < faces.size(); faceIdx++)
    {
        for(size_t faceIdx2 = 0; faceIdx2 < faces.size(); faceIdx2++)
        {
            if(faceIdx == faceIdx2)
            {
                continue;
            }
            if((normals[faceIdx] == normals[faceIdx2]) or (normals[faceIdx] == -1 * normals[faceIdx2]))
            {
                parallelIdx.first = faceIdx;
                parallelIdx.second = faceIdx2;
            }
        }
    }

    std::vector<Vector3D> intersections;

    for(size_t idx1 = 0; idx1 < VERTICES_NUMBER; idx1++)
    {
        for(size_t idx2 = 0; idx2 < VERTICES_NUMBER; idx2++)
        {
            Vector3D vec1 = faces[parallelIdx.first].vertices[idx1] - faces[parallelIdx.second].vertices[idx2];
            for(size_t idx3 = 0; idx3 < VERTICES_NUMBER; idx3++)
            {
                for(size_t idx4 = 0; idx4 < VERTICES_NUMBER; idx4++)
                {
                    Vector3D vec2 = faces[parallelIdx.first].vertices[idx3] - faces[parallelIdx.second].vertices[idx4];
                    try
                    {
                        Vector3D intersection = GetIntersection(faces[parallelIdx.first].vertices[idx1], vec1, faces[parallelIdx.first].vertices[idx3], vec2);
                        if(std::find(intersections.begin(), intersections.end(), intersection) != intersections.end())
                        {
                            return intersection;
                        }
                        intersections.emplace_back(intersection);
                    }
                    catch(const std::exception& e)
                    {
                        std::cerr << e.what() << '\n';
                    }
                    
                }
            }
        }
    }
    throw UniversalError("Body is not a frustrum");
}

Frustrum::Frustrum(const std::vector<Face> &faces, const IndexingKernel3D *indexing)
{
    this->indexing = indexing;
    if(faces.size() != NUM_FACES)
    {
        throw UniversalError("Can not use 'Frustrum' kernelization when there are not " + std::to_string(NUM_FACES) + " faces (given " + std::to_string(faces.size()) + ")");
    }

    Mat44<double> C(0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0);

    Vector3D point1 = faces[0].vertices[0], point2 = faces[0].vertices[1], point3 = faces[0].vertices[2];
    Vector3D point4 = this->find_S(faces); // S
    Mat44<double> F(point1.x, point2.x, point3.x, point4.x, point1.y, point2.y, point3.y, point4.y, point1.z, point2.z, point3.z, point4.z, 1, 1, 1, 1);
    this->P = C * F.inverse();
}

#endif // FRUSTRUM_KERNEL_HPP