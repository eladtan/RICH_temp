#ifndef POLYCLIP_HPP
#define POLYCLIP_HPP

#include <vector>
#include "3D/tessellation/Tessellation3D.hpp"
#include "misc/universal_error.hpp"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <tuple>
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

struct ClipBounds
{
    Vector3D lower;
    Vector3D upper;
    bool valid;

    ClipBounds(void): lower(Vector3D::max()), upper(Vector3D::min()), valid(false) {}
};

struct ClipWorkspace
{
    std::vector<Face> buf_a;
    std::vector<Face> buf_b;
    std::vector<Plane> planes;
    Face bottom;
};

Vector3D computeCenter(const std::vector<Face> &faces);

std::pair<Face, Face> clipFace(const Face &face, const Plane &plane, bool print = false);

Face ConvexHullFace(const Face &face);

double computeVolume(const std::vector<Face> &faces);

double polygonArea(const Face &face);

Vector3D faceCenter(const Face &face);

ClipBounds computeBounds(const std::vector<Face> &faces);

ClipBounds CreatePolyBounds(const Tessellation3D &tess, size_t cell_index);

std::vector<Face> clipPolyhedron(const std::vector<Face> &faces, const Plane &plane, bool print = false);

void clipPolyhedron(const std::vector<Face> &faces, const Plane &plane, std::vector<Face> &result, bool print = false);

void clipPolyhedron(const std::vector<Face> &faces, const Plane &plane, std::vector<Face> &result, ClipWorkspace &workspace, bool print = false);

std::tuple<double, double, Vector3D> clipCells(const Tessellation3D &tess, size_t check_index, const std::vector<Face> &polyhedron, const Plane *vof = 0, bool print = false);

std::tuple<double, double, Vector3D> clipCells(const std::vector<Face> &polyhedron, const std::vector<Plane> &other_poly, const Plane *vof = 0, bool print = false);

std::tuple<double, double, Vector3D> clipCells(const Tessellation3D &tess, size_t check_index, const std::vector<Face> &polyhedron,
    ClipWorkspace &workspace, const ClipBounds *source_bounds = 0, const ClipBounds *target_bounds = 0, const Plane *vof = 0, bool print = false);

std::tuple<double, double, Vector3D> clipCells(const std::vector<Face> &polyhedron, const std::vector<Plane> &other_poly,
    ClipWorkspace &workspace, const ClipBounds *source_bounds = 0, const ClipBounds *target_bounds = 0, const Plane *vof = 0, bool print = false);

std::vector<Plane> CreatePolyPlanes(const Tessellation3D &tess, size_t cell_index);

void CreatePolyPlanes(const Tessellation3D &tess, size_t cell_index, std::vector<Plane> &poly);

std::vector<Face> CreatePolyFaces(const Tessellation3D &tess, size_t cell_index);

void CreatePolyFaces(const Tessellation3D &tess, size_t cell_index, std::vector<Face> &poly);

#endif // POLYCLIP_HPP
