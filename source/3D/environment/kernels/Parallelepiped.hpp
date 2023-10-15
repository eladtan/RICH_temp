#ifndef PARALLELEPIPED_KERNEL_HPP
#define PARALLELEPIPED_KERNEL_HPP

#include <vector>
#include <algorithm>

#include "3D/elementary/Face.hpp"
#include "3D/elementary/Mat33.hpp"
#include "IndexingKernel3D.hpp"

#define NUM_FACES 6 // a parallelepiped has 6 faces
#define FACE_VERTICES_NUM 4 // each face should have 4 vertices

class Parallelepiped : public IndexingKernel3D
{
public:
    inline Parallelepiped(const Vector3D &u, const Vector3D &v, const Vector3D &w, const IndexingKernel3D *indexing = nullptr): indexing(indexing)
    {
        this->calculateTransformation(u, v, w);
    }
    
    Parallelepiped(const std::vector<Face> &faces, const IndexingKernel3D *indexing = nullptr);

    inline Vector3D operator()(const Vector3D &vector) const override
    {
        Vector3D vec = (this->indexing == nullptr)? vector : (*this->indexing)(vector);
        return this->transformation * vec;
    };

private:
    void calculateTransformation(const Vector3D &u, const Vector3D &v, const Vector3D &w);

    Mat33<typename Vector3D::coord_type> transformation;
    const IndexingKernel3D *indexing;
};

Parallelepiped::Parallelepiped(const std::vector<Face> &faces, const IndexingKernel3D *indexing)
{
    this->indexing = indexing;
    if(faces.size() != NUM_FACES)
    {
        throw UniversalError("Can not use 'Parallelepiped' kernelization when there are not " + std::to_string(NUM_FACES) + " faces (given " + std::to_string(faces.size()) + ")");
    }

    std::vector<size_t> contrastFaces;
    std::vector<Vector3D> allEdges;
    std::vector<std::pair<Vector3D, Vector3D> edgesVectors;
    this->edgesVectors.resize(NUM_FACES);
    this->contrastFaces.reserve(NUM_FACES);

    for(size_t faceIdx = 0; faceIdx < faces.size(); faceIdx++)
    {
        const point_vec_v &vertices = faces[faceIdx].vertices;
        if(vertices.size() != FACE_VERTICES_NUM)
        {
            throw UniversalError("Can not use 'Parallelepiped' kernelization when there's a face with " + std::to_string(vertices.size()) + " vertices (expected " + std::to_string(FACE_VERTICES_NUM) + ")");
        }
        const Vector3D &a = vertices[0];
        const Vector3D &b = vertices[1];
        const Vector3D &c = vertices[2];
        const Vector3D &d = vertices[3];

        Vector3D edge1 = (b-a), edge2 = (d-c), edge3 = (c-b), edge4 = (d-a);
        if((edge1 != edge2) or (edge3 != edge4))
        {
            throw UniversalError("One pair or more of edges of face number " + std::to_string(faceIdx) + " (in 'Parallelepiped' kernelization) are not parallel");
        }
        edgesVectors.push_back({edge1, edge3});
        allEdges.push_back(edge1);
        allEdges.push_back(edge2);
    }

    for(size_t faceIdx = 0; faceIdx < faces.size(); faceIdx++)
    {
        const point_vec_v &vertices = faces[faceIdx].vertices;
        const Vector3D &a = vertices[0];
        const Vector3D &b = vertices[1];
        const Vector3D &c = vertices[2];
        const Vector3D &d = vertices[3];
        Vector3D edge1 = (b-a), edge2 = (d-c), edge3 = (c-b), edge4 = (d-a);

        // find a face that has two edges directions same as us
        size_t face2Idx;
        for(face2Idx = 0; face2Idx < faces.size(); face2Idx++)
        {
            if(face2Idx == faceIdx)
            {
                continue;
            }
            const Vector3D &face2edge1 = edgesVectors[face2Idx].first;
            const Vector3D &face2edge2 = edgesVectors[face2Idx].second;

            if(((edge1 == face2edge1) and (edge2 == face2edge2)) or ((edge1 == face2edge2) and (edge2 == face2edge1)))
            {
                contrastFaces[faceIdx] = face2Idx;
                break;
            }
        }

        if(face2Idx == faces.size())
        {
            throw UniversalError("The face of index " + std::to_string(faceIdx) + " does not have a matching parallel face (in 'Parallelepiped' kernelization)");
        }
    }

    // determine u, v, w
    auto it = std::unique(allEdges.begin(), allEdges.end());
    if(std::distance(allEdges.begin(), it) != 3)
    {
        throw UniversalError("The given shape is not a parallelepiped (in 'Parallelepiped' kernelization)");
    }
    this->calculateTransformation(allEdges[0], allEdges[1], allEdges[2]);
}

void Parallelepiped::calculateTransformation(const Vector3D &u, const Vector3D &v, const Vector3D &w)
{
    Mat33<typename Vector3D::coord_type> inverseTransformation;
    for(int i = 0; i < 3; i++)
    {
        inverseTransformation[i][0] = u[i];
        inverseTransformation[i][1] = v[i];
        inverseTransformation[i][2] = w[i];
    }
    this->transformation = inverseTransformation.inverse();
}

#endif // PARALLELEPIPED_KERNEL_HPP