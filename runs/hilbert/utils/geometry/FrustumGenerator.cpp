#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "3D/elementary/Face.hpp"
#include "FrustumGenerator.h"

std::vector<Face> GenerateFrustum(const Face &base1, const Face &base2)
{
    std::vector<Face> faces;
    const Vector3D &A = base1.vertices[0];
    const Vector3D &B = base1.vertices[1];
    const Vector3D &C = base1.vertices[2];
    const Vector3D &D = base1.vertices[3];
    const Vector3D &E = base2.vertices[0];
    const Vector3D &F = base2.vertices[1];
    const Vector3D &G = base2.vertices[2];
    const Vector3D &H = base2.vertices[3];

    faces.push_back(Face({A, D, C, B}, 0, 0));
    faces.push_back(Face({E, F, G, H}, 0, 0));
    faces.push_back(Face({A, E, H, D}, 0, 0));
    faces.push_back(Face({G, F, B, C}, 0, 0));
    faces.push_back(Face({A, B, F, E}, 0, 0));
    faces.push_back(Face({G, C, D, H}, 0, 0));
    
    return faces;
}