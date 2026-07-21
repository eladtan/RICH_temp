#ifndef POLYCLIP_HPP
#define POLYCLIP_HPP

#include <vector>
#include "3D/tessellation/Tessellation3D.hpp"
#include "misc/universal_error.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include "newtonian/three_dimensional/computational_cell.hpp"

struct Plane
    #ifdef RICH_MPI
        : public Serializable
    #endif // RICH_MPI
{
    Vector3D normal;
    Vector3D point;

    Plane(const Vector3D &n = Vector3D(), const Vector3D &p = Vector3D()): normal(n), point(p) {}

    double signedDistance(const Vector3D &v) const
    {
        return ScalarProd(v - point, normal);
    }

    Vector3D intersect(const Vector3D &a, const Vector3D &b) const
    {
        Vector3D ab = b - a;
        double da = signedDistance(a);
        double db = signedDistance(b);
        double t = da / (da - db + std::numeric_limits<double>::min() * 1e-10);
        return a + ab * t;
    }

#ifdef RICH_MPI
    size_t dump(Serializer *serializer) const override
    {
        size_t bytes = 0;
        bytes += serializer->insert(this->normal);
        bytes += serializer->insert(this->point);
        return bytes;
    }
    
    size_t load(const Serializer *serializer, size_t byteOffset) override
    {
        size_t bytes = 0;
        bytes += serializer->extract(this->normal, byteOffset + bytes);
        bytes += serializer->extract(this->point, byteOffset + bytes);
        return bytes;
    }
#endif // RICH_MPI


    friend std::ostream &operator<<(std::ostream &stream, const Plane &plane)
    {
        stream << "normal=" << plane.normal << " point=" << plane.point;
        return stream;
    }
};

Vector3D computeCenter(const std::vector<Face> &faces);

std::pair<Face, Face> clipFace(const Face &face, const Plane &plane, bool print = false);

Face ConvexHullFace(const Face &face);

double computeVolume(const std::vector<Face> &faces);

double polygonArea(const Face &face);

Vector3D faceCenter(const Face &face);

std::vector<Face> clipPolyhedron(const std::vector<Face> &faces, const Plane &plane, bool print = false);

std::tuple<double, double, Vector3D> clipCells(const Tessellation3D &tess, size_t check_index, const std::vector<Face> &polyhedron, const Plane *vof = 0, bool print = false);

std::tuple<double, double, Vector3D> clipCells(const std::vector<Face> &polyhedron, const std::vector<Plane> &other_poly, const Plane *vof = 0, bool print = false);

std::vector<Plane> CreatePolyPlanes(const Tessellation3D &tess, size_t cell_index);

void CreatePolyPlanes(const Tessellation3D &tess, size_t cell_index, std::vector<Plane> &poly);

std::vector<Face> CreatePolyFaces(const Tessellation3D &tess, size_t cell_index);

void CreatePolyFaces(const Tessellation3D &tess, size_t cell_index, std::vector<Face> &poly);

#endif // POLYCLIP_HPP