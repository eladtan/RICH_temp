#include "Frustrum.hpp"

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
    
    bool found = false;
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
                found = true;
                break;
            }
        }
        if(found)
        {
            break;
        }
    }

    if(!found)
    {
        throw UniversalError("Can not use 'Frustrum' kernelization when there are no parallel faces");
    }

    std::vector<Vector3D> intersections;

    for(size_t idx1 = 0; idx1 < VERTICES_NUMBER; idx1++)
    {
        for(size_t idx3 = 0; idx3 < VERTICES_NUMBER; idx3++)
        {
            if(idx1 == idx3)
            {
                continue;
            }
            for(size_t shift = 0; shift < VERTICES_NUMBER; shift++)
            {
                size_t idx2 = (idx1 + shift) % VERTICES_NUMBER;
                size_t idx4 = (idx3 + shift) % VERTICES_NUMBER;
                Vector3D vec1 = faces[parallelIdx.first].vertices[idx1] - faces[parallelIdx.second].vertices[idx2];
                Vector3D vec2 = faces[parallelIdx.first].vertices[idx3] - faces[parallelIdx.second].vertices[idx4];
                try
                {
                    Vector3D intersection = GetIntersection(faces[parallelIdx.first].vertices[idx1], vec1, faces[parallelIdx.first].vertices[idx3], vec2);
                    bool abovePlane1 = ScalarProd(intersection - faces[parallelIdx.first].vertices[idx1], normals[parallelIdx.first]) > 0;
                    bool abovePlane2 = ScalarProd(intersection - faces[parallelIdx.second].vertices[idx2], normals[parallelIdx.second]) > 0;
                    // the 'head' should be above or below both of the two bases
                    if((abovePlane1 and abovePlane2) or (!abovePlane1 and !abovePlane2))
                    {
                        if(std::find(intersections.begin(), intersections.end(), intersection) != intersections.end())
                        {
                            return intersection;
                        }
                        intersections.emplace_back(intersection);
                    }
                }
                catch(const UniversalError& e)
                {
                    // no intersection
                }
            }
        }
    }
    throw UniversalError("Body is not a frustrum");
}

Frustrum::Frustrum(const std::vector<Face> &faces, const IndexingKernel3D *indexing)
{
    if(faces.size() != NUM_FACES)
    {
        throw UniversalError("Can not use 'Frustrum' kernelization when there are not " + std::to_string(NUM_FACES) + " faces (given " + std::to_string(faces.size()) + ")");
    }
    Vector3D move_factor = faces[0].vertices[0];
    this->indexing = new Move(move_factor, indexing);

    const Mat44<double> C(0, 1, 1, 0,
                         0, 0, 1, 0,
                         0, 0, 0, 1,
                         1, 1, 1, 0);
    Vector3D point1 = (*this->indexing)(faces[0].vertices[0]), point2 = (*this->indexing)(faces[0].vertices[1]), point3 = (*this->indexing)(faces[0].vertices[2]);
    Vector3D point4 = (*this->indexing)(this->find_S(faces)); // head (S)
    Mat44<double> F(point1.x, point2.x, point3.x, point4.x, point1.y, point2.y, point3.y, point4.y, point1.z, point2.z, point3.z, point4.z, 1, 1, 1, 1);
    this->P = C * F.inverse();

    std::vector<Vector3D> allVertices;
    for(const Face &face : faces)
    {
        for(const Vector3D &vertex : face.vertices)
        {
            allVertices.push_back(this->beforeScaling(vertex));
        }
    }
    this->scaleToBox = Rectangle(allVertices);
}

