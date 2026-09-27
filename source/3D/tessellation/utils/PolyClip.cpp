#include "PolyClip.hpp"

std::pair<double, Vector3D> computeCM(const std::vector<Face> &faces);

namespace
{
    void ExtendBounds(ClipBounds &bounds, const Vector3D &point)
    {
        bounds.lower.x = std::min(bounds.lower.x, point.x);
        bounds.lower.y = std::min(bounds.lower.y, point.y);
        bounds.lower.z = std::min(bounds.lower.z, point.z);
        bounds.upper.x = std::max(bounds.upper.x, point.x);
        bounds.upper.y = std::max(bounds.upper.y, point.y);
        bounds.upper.z = std::max(bounds.upper.z, point.z);
        bounds.valid = true;
    }

    double BoundsScale(const ClipBounds *bounds)
    {
        if(bounds == 0 || !bounds->valid)
        {
            return 1.0;
        }
        return std::max(1.0, fastabs(bounds->upper - bounds->lower));
    }

    bool BoundsDisjoint(const ClipBounds &a, const ClipBounds &b)
    {
        if(!a.valid || !b.valid)
        {
            return false;
        }
        const double tol = 1e-12 * std::max(BoundsScale(&a), BoundsScale(&b));
        return a.upper.x < b.lower.x - tol || b.upper.x < a.lower.x - tol ||
               a.upper.y < b.lower.y - tol || b.upper.y < a.lower.y - tol ||
               a.upper.z < b.lower.z - tol || b.upper.z < a.lower.z - tol;
    }

    std::tuple<double, double, Vector3D> clipCellsFull(const std::vector<Face> &polyhedron, const std::vector<Plane> &other_poly,
        ClipWorkspace &workspace, const Plane *vof, bool print);

    bool TryClipFastPath(const std::vector<Face> &polyhedron, const std::vector<Plane> &other_poly, const ClipBounds *source_bounds,
        const ClipBounds *target_bounds, const Plane *vof, std::tuple<double, double, Vector3D> &result)
    {
        if(source_bounds != 0 && target_bounds != 0 && BoundsDisjoint(*source_bounds, *target_bounds))
        {
            result = std::make_tuple(0.0, 0.0, Vector3D(0, 0, 0));
            return true;
        }

        if(polyhedron.empty())
        {
            return false;
        }

        const double tol = 1e-12 * BoundsScale(source_bounds);
        bool all_vertices_inside_all_planes = true;
        for(const Plane &plane : other_poly)
        {
            bool all_vertices_outside_this_plane = true;
            bool all_vertices_inside_this_plane = true;
            const double offset = ScalarProd(plane.normal, plane.point);
            for(const Face &face : polyhedron)
            {
                for(const Vector3D &vertex : face.vertices)
                {
                    // The distance clipFace uses.
                    const double d = ScalarProd(vertex, plane.normal) - offset;
                    if(std::abs(d) <= tol)
                    {
                        return false;
                    }
                    if(d > tol)
                    {
                        all_vertices_outside_this_plane = false;
                    }
                    else
                    {
                        all_vertices_inside_this_plane = false;
                    }
                }
            }
            if(all_vertices_outside_this_plane)
            {
                result = std::make_tuple(0.0, 0.0, Vector3D(0, 0, 0));
                return true;
            }
            all_vertices_inside_all_planes = all_vertices_inside_all_planes && all_vertices_inside_this_plane;
        }

        if(all_vertices_inside_all_planes && vof == 0)
        {
            auto [volume, CM] = computeCM(polyhedron);
            result = std::make_tuple(volume, 0.0, CM);
            return true;
        }
        return false;
    }
}

double polygonArea(const Face &face)
{
    Vector3D ref = face.vertices[0];
    Vector3D area;
    for(size_t i = 1; i + 1 < face.vertices.size(); i++)
    {
        area += CrossProduct(face.vertices[i] - ref, face.vertices[i + 1] - ref);
    }
    return 0.5 * abs(area);
}

Vector3D faceCenter(const Face &face)
{
    Vector3D center;
    int n = 0;
    for(const Vector3D &v : face.vertices)
    {
        center += v;
        ++n;
    }
    return center * (1.0 / n);
}

double tetraVolume(const Vector3D &p0, const Vector3D &p1, const Vector3D &p2, const Vector3D &p3)
{
    Vector3D e1 = p1 - p0;
    Vector3D e2 = p2 - p0;
    Vector3D e3 = p3 - p0;
    return std::abs(ScalarProd(e1, CrossProduct(e2, e3))) / 6.0;
}

Vector3D tetraCM(const Vector3D &p0, const Vector3D &p1, const Vector3D &p2, const Vector3D &p3)
{
    return (p0 + p1 + p2 + p3) * 0.25;
}

Vector3D computeCenter(const std::vector<Face> &faces)
{
    Vector3D center(0, 0, 0);
    size_t n = 0;
    for(const Face &face : faces)
    {
        for(const Vector3D &v : face.vertices)
        {
            center += v;
            ++n;
        }
    }
    return center * (1.0 / n);
}

std::pair<double, Vector3D> computeCM(const std::vector<Face> &faces)
{
    if(faces.empty())
    {
        return {0, Vector3D(0, 0, 0)};
    }
    Vector3D reference = computeCenter(faces);
    double volume = 0;
    Vector3D CM;
    for(const Face &face : faces)
    {
        for(size_t i = 1; i + 1 < face.vertices.size(); i++)
        {
            double v = tetraVolume(reference, face.vertices[0], face.vertices[i], face.vertices[i + 1]);
            Vector3D center = tetraCM(reference, face.vertices[0], face.vertices[i], face.vertices[i + 1]);
            CM += v * center;
            volume += v;
        }
    }

    CM *= 1.0 / (volume + std::numeric_limits<double>::min() * 1e10);
    return std::make_pair(volume, CM);
}

ClipBounds computeBounds(const std::vector<Face> &faces)
{
    ClipBounds bounds;
    for(const Face &face : faces)
    {
        for(const Vector3D &vertex : face.vertices)
        {
            ExtendBounds(bounds, vertex);
        }
    }
    return bounds;
}

namespace
{
    bool LexicographicallyLess(const Vector3D &a, const Vector3D &b)
    {
        if(a.x != b.x)
        {
            return a.x < b.x;
        }
        if(a.y != b.y)
        {
            return a.y < b.y;
        }
        return a.z < b.z;
    }

    // The on-plane band E of a clip: a bound on the rounding of n.v - n.p,
    // from the magnitudes sum_k |n_k v_k| and sum_k |n_k p_k| of both dot
    // products (it scales with the absolute coordinates, not with the face:
    // far from the origin, copies of one vertex held by different faces differ
    // by an ulp of the coordinates), or 1e-12 of the edge length.
    //
    // Accuracy contract of clipCells/clipPolyhedron (normalized planes, a
    // convex, consistently connected input polyhedron P whose faces are
    // resolved by the face cleanup, i.e. no feature of a face is shorter than
    // CleanFace's threshold, 1e-12 of the face's longest edge or 100 ulps of
    // its coordinates; RICH's Voronoi cells satisfy this): a vertex whose
    // computed distance is within E is on the plane.  With e the evaluation
    // error of the computed distance (|e| <= 64 eps * the same magnitudes, so
    // e <= E), the result differs from the exact P cap H only within the slab
    // |n.x - n.p| <= W = E + e <= 2E, and
    //   |V_clip - V_exact| <= Vol(P cap slab) + eps_V <= 2 W max_{|t|<=W} A(t) + eps_V
    // with A(t) the cross-section of P at offset t and eps_V the rounding of
    // the volume sums; geometry thinner than W may be assigned wholly to either
    // side.  Complementary clips (n, p) and (-n, p) conserve V(P) to the same
    // bound.  Identical vertex coordinates always classify alike; the copies
    // of a vertex in P's faces are identical when P comes from a tessellation
    // (or exact MPI copies of it) and every cut point is formed canonically
    // (below), so all faces agree on each vertex.  Two representations of one
    // plane agree within their bands.
    double OnPlaneBand(double max_edge, double error_scale)
    {
        return std::max(max_edge * 1e-12, 64 * std::numeric_limits<double>::epsilon() * error_scale);
    }

    double DotMagnitude(const Vector3D &v, const Vector3D &n)
    {
        return std::abs(v.x * n.x) + std::abs(v.y * n.y) + std::abs(v.z * n.z);
    }

    // The tolerance of one clip, the same for every face of the polyhedron so
    // that a vertex held by several faces is classified alike in all of them:
    // the band over the whole polyhedron, or 0 (raw signs) when every vertex
    // lies within it.  Such a polyhedron is a slab thinner than the rounding
    // around the plane; the band would put all of its faces on the plane and
    // collapse it into one cap.
    double ClipTolerance(const std::vector<Face> &faces, const Plane &plane)
    {
        const double offset = ScalarProd(plane.normal, plane.point);
        double max_edge = 0;
        double error_scale = DotMagnitude(plane.point, plane.normal);
        double max_distance = 0;
        for(const Face &face : faces)
        {
            const size_t n = face.vertices.size();
            if(n < 3)
            {
                continue;
            }
            for(size_t i = 0; i < n; i++)
            {
                const Vector3D &curr = face.vertices[i];
                max_edge = std::max(max_edge, fastabs(face.vertices[(i + 1) % n] - curr));
                max_distance = std::max(max_distance, std::abs(ScalarProd(curr, plane.normal) - offset));
                error_scale = std::max(error_scale, DotMagnitude(curr, plane.normal));
            }
        }
        const double band = OnPlaneBand(max_edge, error_scale);
        return max_distance <= band ? 0.0 : band;
    }

std::pair<Face, Face> clipFaceImpl(const Face &face, const Plane &plane, double tolerance, bool print)
{
    Face out, clip_points;
    const size_t n = face.vertices.size();
    if(n < 3)
    {
        return {out, clip_points};
    }
    // Signed distances n.v - n.p with n.p formed once, so that every point of
    // the same plane gives the same distances.  A vertex within the tolerance
    // (ClipTolerance, one per polyhedron and plane) is on the plane.  Only strict sign changes are cut, at a parameter clamped to
    // the edge; on-plane vertices are kept and join the cap.
    const double offset = ScalarProd(plane.normal, plane.point);
    constexpr size_t local_size = 32;
    double local_d[local_size];
    std::vector<double> heap_d;
    double *d = local_d;
    if(n > local_size)
    {
        heap_d.resize(n);
        d = heap_d.data();
    }
    double maxR = 0;
    for(size_t i = 0; i < n; i++)
    {
        const Vector3D &curr = face.vertices[i];
        const Vector3D &next = face.vertices[(i + 1) % n];
        maxR = std::max(maxR, fastabs(next - curr));
        d[i] = ScalarProd(curr, plane.normal) - offset;
    }
    bool any_inside = false;
    bool all_on = true;
    for(size_t i = 0; i < n; i++)
    {
        any_inside = any_inside || d[i] > tolerance;
        all_on = all_on && std::abs(d[i]) <= tolerance;
    }
    if(print)
    {
        std::cout << "clipFace tolerance " << tolerance << " maxR " << maxR << " all_on " << all_on
                  << " any_inside " << any_inside << std::endl;
        for(size_t i = 0; i < n; i++)
        {
            std::cout << "  d[" << i << "] = " << d[i] << std::endl;
        }
    }
    // A face on the plane belongs to the cap.  A face with no vertex strictly
    // inside contributes no face, only its on-plane vertices to the cap.
    if(all_on)
    {
        clip_points = face;
        return {out, clip_points};
    }
    if(not any_inside)
    {
        for(size_t i = 0; i < n; i++)
        {
            if(std::abs(d[i]) <= tolerance)
            {
                clip_points.vertices.push_back(face.vertices[i]);
            }
        }
        return {out, clip_points};
    }
    for(size_t i = 0; i < n; i++)
    {
        const size_t j = (i + 1) % n;
        const Vector3D &curr = face.vertices[i];
        const Vector3D &next = face.vertices[j];
        const int side_curr = d[i] > tolerance ? 1 : (d[i] < -tolerance ? -1 : 0);
        const int side_next = d[j] > tolerance ? 1 : (d[j] < -tolerance ? -1 : 0);
        if(side_curr >= 0)
        {
            out.vertices.push_back(curr);
        }
        if(side_curr == 0)
        {
            clip_points.vertices.push_back(curr);
        }
        if(side_curr * side_next < 0)
        {
            // From the endpoints in a canonical order, so that the two faces
            // sharing the edge create the identical point (from each face's own
            // direction the two points differ by ulps).
            const bool swap = LexicographicallyLess(next, curr);
            const Vector3D &p = swap ? next : curr;
            const Vector3D &q = swap ? curr : next;
            const double dp = swap ? d[j] : d[i];
            const double dq = swap ? d[i] : d[j];
            const double t = std::min(1.0, std::max(0.0, dp / (dp - dq)));
            const Vector3D intersection = p + (q - p) * t;
            out.vertices.push_back(intersection);
            clip_points.vertices.push_back(intersection);
        }
    }
    Face out2;
    if(out.vertices.size() > 2)
    {
        out2.vertices.push_back(out.vertices[0]);
        const size_t Nvert = out.vertices.size();
        for(size_t i = 1; i < Nvert; i++)
        {
            if(fastabs(out.vertices[i] - out2.vertices.back()) > maxR * 1e-12)
            {
                out2.vertices.push_back(out.vertices[i]);
            }
        }
    }
    return {out2, clip_points};
}
}

std::pair<Face, Face> clipFace(const Face &face, const Plane &plane, bool print)
{
    return clipFaceImpl(face, plane, ClipTolerance(std::vector<Face>(1, face), plane), print);
}

Face ConvexHullFace(const Face &face)
{
    Face result;
    if(face.vertices.size() < 3)
    {
        return result;
    }
    Vector3D center = faceCenter(face);
    Vector3D X = face.vertices[0] - center;
    double size = abs(X);
    if(!(std::isfinite(size)) || size <= std::numeric_limits<double>::min())
        return result;
    X *= 1.0 / size;
    Vector3D edge;
    Vector3D raw_normal;
    const double edge_eps = 1e-10 * size;
    for(size_t i = 1; i < face.vertices.size(); ++i)
    {
        Vector3D candidate = face.vertices[i] - face.vertices[0];
        if(fastabs(candidate) <= edge_eps)
            continue;
        Vector3D candidate_normal = CrossProduct(X, candidate);
        if(abs(candidate_normal) <= std::numeric_limits<double>::min())
            continue;
        edge = candidate;
        raw_normal = candidate_normal;
        break;
    }
    double normal_size = abs(raw_normal);
    if(!(std::isfinite(normal_size)) || normal_size <= std::numeric_limits<double>::min())
    {
        return result;
    }
    Vector3D N = normalize(raw_normal);
    Vector3D Y = CrossProduct(N, X);

    struct projected
    {
        double angle;
        Vector3D original;
    };

    std::vector<projected> projected_points;

    for(const Vector3D &p : face.vertices)
    {
        Vector3D v = p - center;
        double x = ScalarProd(v, X);
        double y = ScalarProd(v, Y);
        double angle = std::atan2(y, x);
        if(angle < 0)
        {
            angle += 2 * M_PI;
        }
        projected_points.push_back({angle, p});
    }

    std::sort(projected_points.begin(), projected_points.end(), [](const projected &a, const projected &b){return a.angle < b.angle;});

    for(const projected &p : projected_points)
    {
        result.vertices.push_back(p.original);
    }
    return result;
}

Face CleanFace(const Face &face)
{
    size_t Nvert = face.vertices.size();
    Face result;
    if(Nvert < 3)
    {
        return result;
    }
    double maxR = std::numeric_limits<double>::epsilon();
    double close_eps  = 0;
    for(size_t i = 0; i < Nvert; i++)
    {
        maxR = std::max(maxR, fastabs(face.vertices[(i + 1) % Nvert] - face.vertices[i]));
        double R = fastabs(face.vertices[i]);
        close_eps = std::max(close_eps, 100 * std::abs(std::nextafter(R,  std::numeric_limits<double>::infinity()) - R));
    }
    close_eps = std::max(close_eps, 1e-12 * maxR);
   
    result.vertices.push_back(face.vertices[0]);
    for(size_t i = 0; i < Nvert - 2; i++)
    {
        if(fastabs(face.vertices[i + 1] - face.vertices[i]) > close_eps)
        {
            result.vertices.push_back(face.vertices[i + 1]);
        }
    }
    if(fastabs(face.vertices.back() - face.vertices[Nvert - 2]) > close_eps && fastabs(face.vertices.back() - face.vertices[0]) > close_eps)
    {
        result.vertices.push_back(face.vertices.back());
    }
    if(result.vertices.size() < 3)
    {
        result.vertices.clear();
        return result;
    }
    return ConvexHullFace(result);
}

std::vector<Face> clipPolyhedron(const std::vector<Face> &faces, const Plane &plane, bool print)
{
    const double tolerance = ClipTolerance(faces, plane);
    std::vector<Face> result;
    Face bottom;
    if(print)
    {
        std::cout << "Clipping plane " << plane << std::endl;
    }
    for(const Face &face : faces)
    {
        if(face.vertices.size() < 3)
        {
            continue;
        }
        if(print)
        {
            std::cout << "Clipping face " << face << std::endl;
        }
        auto clipped = clipFaceImpl(face, plane, tolerance, print);
        if(print)
        {
            std::cout << "Clip result: " << clipped.first << ", " << clipped.second << std::endl;
        }
        if(clipped.first.vertices.size() >= 3)
        {
            Face clean = CleanFace(clipped.first);
            clean = ConvexHullFace(clean);
            clean = CleanFace(clean);
            if(clean.vertices.size() > 2)
            {
                result.push_back(clean);
            }
        }
        if(not clipped.second.vertices.empty())
        {
            bottom.vertices.insert(bottom.vertices.end(), clipped.second.vertices.begin(), clipped.second.vertices.end());
        }
    }
    if(bottom.vertices.size() > 2)
    {
        Face bottom2 = CleanFace(bottom);
        bottom2 = ConvexHullFace(bottom2);
        bottom2 = CleanFace(bottom2);
        if(bottom2.vertices.size() > 2)
        {
            result.push_back(bottom2);
        }
    }
    // std::cout << "Final result: " << std::endl;
    // for(Face &face : result)
    // {
    //     face = ConvexHullFace(face);
    //     std::cout << face << std::endl;
    // }
    return result;
}

void clipPolyhedron(const std::vector<Face> &faces, const Plane &plane, std::vector<Face> &result, bool print)
{
    ClipWorkspace workspace;
    clipPolyhedron(faces, plane, result, workspace, print);
}

void clipPolyhedron(const std::vector<Face> &faces, const Plane &plane, std::vector<Face> &result, ClipWorkspace &workspace, bool print)
{
    const double tolerance = ClipTolerance(faces, plane);
    result.clear();
    result.reserve(faces.size() + 1);
    Face &bottom = workspace.bottom;
    bottom.vertices.clear();
    if(print)
    {
        std::cout << "Clipping plane " << plane << std::endl;
    }
    for(const Face &face : faces)
    {
        if(face.vertices.size() < 3)
        {
            continue;
        }
        if(print)
        {
            std::cout << "Clipping face " << face << std::endl;
        }
        auto clipped = clipFaceImpl(face, plane, tolerance, print);
        if(print)
        {
            std::cout << "Clip result: " << clipped.first << ", " << clipped.second << std::endl;
        }
        if(clipped.first.vertices.size() >= 3)
        {
            Face clean = CleanFace(clipped.first);
            clean = ConvexHullFace(clean);
            clean = CleanFace(clean);
            if(clean.vertices.size() > 2)
            {
                result.push_back(clean);
            }
        }
        if(not clipped.second.vertices.empty())
        {
            bottom.vertices.insert(bottom.vertices.end(), clipped.second.vertices.begin(), clipped.second.vertices.end());
        }
    }
    if(bottom.vertices.size() > 2)
    {
        Face bottom2 = CleanFace(bottom);
        bottom2 = ConvexHullFace(bottom2);
        bottom2 = CleanFace(bottom2);
        if(bottom2.vertices.size() > 2)
        {
            result.push_back(bottom2);
        }
    }
}

double computeVolume(const std::vector<Face> &faces)
{
    double volume = 0.0;
    if(faces.size() < 4)
    {
        return 0;
    }
    Vector3D center = computeCenter(faces);
    size_t Nfaces = faces.size();
    for(size_t j = 0; j < Nfaces; j++)
    {
        const Face &face = faces[j];
        Vector3D ref = face.vertices[0];
        size_t N = face.vertices.size();
        double volume_face = 0;
        for(size_t i = 1; i + 1 < N; i++)
        {
            Vector3D a = face.vertices[i] - ref;
            Vector3D b = face.vertices[i + 1] - ref;
            volume_face += (ScalarProd(ref - center, CrossProduct(b, a)));
        }
        volume += std::abs(volume_face);
    }
    return volume / 6.0;
}

void CreatePolyFaces(const Tessellation3D &tess, size_t cell_index, std::vector<Face> &poly)
{
    const auto &face_indeces = tess.GetCellFaces(cell_index);
    const size_t Nfaces = face_indeces.size();
    poly.clear();
    poly.resize(Nfaces);
    const auto &face_points = tess.GetFacePoints();
    for(size_t i = 0; i < Nfaces; i++)
    {
        const point_vec &points = tess.GetPointsInFace(face_indeces[i]);
        for(size_t j = 0; j < points.size(); j ++)
        {
            poly[i].vertices.push_back(face_points[points[j]]);
        }
    }
}

ClipBounds CreatePolyBounds(const Tessellation3D &tess, size_t cell_index)
{
    ClipBounds bounds;
    const auto &face_indeces = tess.GetCellFaces(cell_index);
    const auto &face_points = tess.GetFacePoints();
    for(size_t face_index : face_indeces)
    {
        const point_vec &points = tess.GetPointsInFace(face_index);
        for(size_t point_index : points)
        {
            ExtendBounds(bounds, face_points[point_index]);
        }
    }
    return bounds;
}

std::vector<Face> CreatePolyFaces(const Tessellation3D &tess, size_t cell_index)
{
    std::vector<Face> poly;
    CreatePolyFaces(tess, cell_index, poly);
    return poly;
}

void CreatePolyPlanes(const Tessellation3D &tess, size_t cell_index, std::vector<Plane> &faces)
{
    const auto &face_indeces = tess.GetCellFaces(cell_index);
    const size_t Nfaces = face_indeces.size();
    faces.clear();
    faces.resize(Nfaces);
    const auto &face_points = tess.GetFacePoints();
    Face face;
    for(size_t i = 0; i < Nfaces; i++)
    {
        const point_vec &points = tess.GetPointsInFace(face_indeces[i]);
        for(size_t j = 0; j < points.size(); j++)
        {
            face.vertices.push_back(face_points[points[j]]);
        }
        Vector3D face_CM = faceCenter(face);
        face.vertices.clear();
        faces[i].point = face_CM;
        size_t otherIndex = tess.GetFaceNeighbors(face_indeces[i]).first == cell_index ? tess.GetFaceNeighbors(face_indeces[i]).second : tess.GetFaceNeighbors(face_indeces[i]).first;
        Vector3D delta = tess.GetMeshPoint(cell_index) - tess.GetMeshPoint(otherIndex);
        faces[i].normal = normalize(delta);
    }
}

std::vector<Plane> CreatePolyPlanes(const Tessellation3D &tess, size_t cell_index)
{
    std::vector<Plane> planes;
    CreatePolyPlanes(tess, cell_index, planes);
    return planes;
}

std::tuple<double, double, Vector3D> clipCells(const Tessellation3D &tess, size_t check_index, const std::vector<Face> &polyhedron, const Plane *vof, bool print)
{
    ClipWorkspace workspace;
    return clipCells(tess, check_index, polyhedron, workspace, 0, 0, vof, print);
}

std::tuple<double, double, Vector3D> clipCells(const std::vector<Face> &polyhedron, const std::vector<Plane> &other_poly, const Plane *vof, bool print)
{
    ClipWorkspace workspace;
    return clipCells(polyhedron, other_poly, workspace, 0, 0, vof, print);
}

std::tuple<double, double, Vector3D> clipCells(const Tessellation3D &tess, size_t check_index, const std::vector<Face> &polyhedron,
    ClipWorkspace &workspace, const ClipBounds *source_bounds, const ClipBounds *target_bounds, const Plane *vof, bool print)
{
    CreatePolyPlanes(tess, check_index, workspace.planes);
    return clipCells(polyhedron, workspace.planes, workspace, source_bounds, target_bounds, vof, print);
}

std::tuple<double, double, Vector3D> clipCells(const std::vector<Face> &polyhedron, const std::vector<Plane> &other_poly,
    ClipWorkspace &workspace, const ClipBounds *source_bounds, const ClipBounds *target_bounds, const Plane *vof, bool print)
{
    std::tuple<double, double, Vector3D> result;
    if(TryClipFastPath(polyhedron, other_poly, source_bounds, target_bounds, vof, result))
    {
        return result;
    }
    return clipCellsFull(polyhedron, other_poly, workspace, vof, print);
}

namespace
{
std::tuple<double, double, Vector3D> clipCellsFull(const std::vector<Face> &polyhedron, const std::vector<Plane> &other_poly,
    ClipWorkspace &workspace, const Plane *vof, bool print)
{
    workspace.buf_a = polyhedron;
    workspace.buf_b.clear();
    std::vector<Face> *src = &workspace.buf_a, *dst = &workspace.buf_b;
    if(print)
    {
        auto [volume, CM] = computeCM(polyhedron);
        double volume1 = computeVolume(*src);
        std::cout << "Starting cell clip volume0 " << volume << " volume1 " << volume1 << std::endl;
    }
    const size_t Nplanes = other_poly.size();
    for(size_t i = 0; i < Nplanes; i++)
    {
        clipPolyhedron(*src, other_poly[i], *dst, workspace, print);
        std::swap(src, dst);
        if(print)
        {
            auto [volume, CM] = computeCM(*src);
            std::cout << "Volume " << volume << " CM " << CM << std::endl;
            std::cout << "Clipped poly: " << std::endl;
            for(const Face &face : *src)
            {
                std::cout << face << std::endl;
            }
        }
    }
    auto [volume, CM] = computeCM(*src);
    double vof_volume = 0;
    if(vof != 0)
    {
        if(print)
        {
            std::cout << "Starting vof clip" << std::endl;
        }
        clipPolyhedron(*src, *vof, *dst, workspace, print);
        std::swap(src, dst);
        if(print)
        {
            std::cout << "Clipped poly: " << std::endl;
            for(const Face &face : *src)
            {
                std::cout << face << std::endl;
            }
        }
        auto [volume2, CM2] = computeCM(*src);
        vof_volume = std::min(volume, volume2);
    }
    return {volume, vof_volume, CM};
}
}
