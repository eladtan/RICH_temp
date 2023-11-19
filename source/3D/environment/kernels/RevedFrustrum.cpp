#include "RevedFrustrum.hpp"

namespace
{
    inline Vector3D GetNormal(const Face &face)
    {
        return normalize(CrossProduct(face.vertices[1] - face.vertices[0], face.vertices[2] - face.vertices[1]));
    }

    inline Vector3D GetFacesIntersection(const Face &face1, const Face &face2, const Face &face3)
    {
        Vector3D normal1 = GetNormal(face1);
        Vector3D normal2 = GetNormal(face2);
        Vector3D normal3 = GetNormal(face3);
        Vector3D D(ScalarProd(normal1, face1.vertices[0]), ScalarProd(normal2, face2.vertices[0]), ScalarProd(normal3, face3.vertices[0]));
        Mat33<double> mat = Mat33<double>(normal1.x, normal1.y, normal1.z, normal2.x, normal2.y, normal2.z, normal3.x, normal3.y, normal3.z);
        if(std::abs(mat.determinant()) < EPSILON)
        {
            throw UniversalError("One or more of the side faces of the body are parallel (all the side faces intersect in one point) (in 'Frustum')");
        }
        Mat33<double> inverse = mat.inverse();
        return inverse * D;
    }
}

Vector3D RevedFrustrum::find_S(const std::vector<Face> &faces) const
{    
    // first find the parallel faces
    std::vector<Vector3D> normals;
    for(const Face &face : faces)
    {
        if(face.vertices.size() != VERTICES_NUMBER)
        {
            throw UniversalError("Can not use 'RevedFrustrum' kernelization when there's a face with " + std::to_string(face.vertices.size()) + " vertices (expected " + std::to_string(VERTICES_NUMBER) + ")");
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
        throw UniversalError("Can not use 'RevedFrustrum' kernelization when there are no parallel faces");
    }

    std::vector<size_t> nonParallelIdx;
    for(size_t idx = 0; idx < 6; idx++)
    {
        if(idx != parallelIdx.first and idx != parallelIdx.second)
        {
            nonParallelIdx.push_back(idx);
        }
    }

    Vector3D intersection1 = GetFacesIntersection(faces[nonParallelIdx[0]], faces[nonParallelIdx[1]], faces[nonParallelIdx[2]]);
    Vector3D intersection2 = GetFacesIntersection(faces[nonParallelIdx[1]], faces[nonParallelIdx[2]], faces[nonParallelIdx[3]]);

    if(intersection1 != intersection2)
    {
        std::stringstream intersection1Str, intersection2Str;
        intersection1Str << intersection1;
        intersection2Str << intersection2;
        throw UniversalError("Body is not a frustrum (two distinct points of intersection: " + intersection1Str.str() + ", " + intersection2Str.str() + ")");
    }
    return intersection1;
}

RevedFrustrum::RevedFrustrum(const std::vector<Face> &faces, const IndexingKernel3D *beforeIndexing, const IndexingKernel3D *afterIndexing)
{
    if(faces.size() != NUM_FACES)
    {
        throw UniversalError("Can not use 'Frustrum' kernelization when there are not " + std::to_string(NUM_FACES) + " faces (given " + std::to_string(faces.size()) + ")");
    }

    std::vector<Vector3D> allVertices;
    for(const Face &face : faces)
    {
        for(const Vector3D &vertex : face.vertices)
        {
            allVertices.push_back(vertex);
        }
    }
    this->beforeIndexing = new Identity(beforeIndexing);
    
    std::vector<Face> kerneledFaces;
    for(const Face &face : faces)
    {
        Face newFace;
        for(const Vector3D &vertex : face.vertices)
        {
            newFace.vertices.push_back((*this->beforeIndexing)(vertex));
        }
        kerneledFaces.emplace_back(newFace);
    }
    this->S = this->find_S(kerneledFaces); // head (S)
    
    Vector3D normalBase1 = GetNormal(kerneledFaces[0]);
    Vector3D normalBase2 = GetNormal(kerneledFaces[1]);

    if((normalBase1 != Vector3D(0, 0, 1) and normalBase1 != Vector3D(0, 0, -1)) or (normalBase2 != Vector3D(0, 0, 1) and normalBase2 != Vector3D(0, 0, -1)))
    {
        // this message is thrown in order to calculate whether the summit is "above" or "below" the frustrum (the terms "above" and "below" are not clear otherwise)
        // necessary also for calculating the height of the pyramid (although it's easier to change)
        throw UniversalError("Currently, frustrum kernel is supported only when the bases are parallel to the XY plane");
    }

    this->h = std::abs(this->S.z - kerneledFaces[0].vertices[0].z); // height of the pyramid

    allVertices.clear();
    for(const Face &face : faces)
    {
        for(const Vector3D &vertex : face.vertices)
        {
            allVertices.push_back(this->beforeTransformation(vertex));
        }
    }

    this->afterIndexing = nullptr;
}

