#include "RevedFrustrum.hpp"

Kernelization3D::RevedFrustrum::~RevedFrustrum()
{
    delete this->beforeIndexing;
    delete this->afterIndexing;
}

Vector3D Kernelization3D::RevedFrustrum::operator()(const Vector3D &vector) const
{
    Vector3D vec = this->beforeTransformation(vector);
    return (this->afterIndexing == nullptr) ? vec : (*this->afterIndexing)(vec);
}

Vector3D Kernelization3D::RevedFrustrum::beforeTransformation(const Vector3D &vector) const
{
    Vector3D vec = (this->beforeIndexing == nullptr) ? vector : (*this->beforeIndexing)(vector);
    double slope = this->h / (vec.z - this->S.z);
    double new_x = this->S.x + slope * (vec.x - this->S.x);
    double new_y = this->S.y + slope * (vec.y - this->S.y);
    double new_z = vec.z * this->ratio;
    return Vector3D(new_x, new_y, new_z);
}

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

Vector3D Kernelization3D::RevedFrustrum::find_S(const std::vector<Face> &faces) const
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
        UniversalError eo("Body is not a frustrum (two distinct intersections and heads)");
		eo.addEntry("head 1", intersection1);
		eo.addEntry("head 2", intersection2);
		throw eo;
    }
    return intersection1;
}

Kernelization3D::RevedFrustrum::RevedFrustrum(const std::vector<Face> &faces, const IndexingKernel3D *beforeIndexing, const IndexingKernel3D *afterIndexing)
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

    double base1Area = faces[0].GetArea();
    double base2Area = faces[1].GetArea();

    if(base1Area < EPSILON)
    {
        UniversalError eo("Can not use 'RevedFrustrum' kernelization when the first base has an area of 0");
        eo.addEntry("base 1 vertices", faces[0].vertices);
        throw eo;
    }
    if(base2Area < EPSILON)
    {
        UniversalError eo("Can not use 'RevedFrustrum' kernelization when the second base has an area of 0");
        eo.addEntry("base 2 vertices", faces[1].vertices);
        throw eo;
    }

    this->h = std::abs(this->S.z - kerneledFaces[0].vertices[0].z); // height of the pyramid
    this->ratio = sqrt(std::min(base1Area, base2Area)) /* / this->h */;

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

std::string Kernelization3D::RevedFrustrum::getTypeName() const { return "RevedFrustrum"; }
