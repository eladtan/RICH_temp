#include "PolyClip.hpp"

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

std::pair<Face, Face> clipFace(const Face &face, const Plane &plane, bool print)
{
    Face out, clip_points;
    int n = face.vertices.size();
    double old_d = plane.signedDistance(face.vertices[0]);
    double maxR = 0;
    for(int i = 0; i < n; i++)
    {
        const Vector3D &curr = face.vertices[i];
        const Vector3D &next = face.vertices[(i + 1) % n];
        double R = fastabs(next - curr);
        maxR = std::max(maxR, R);
    }
    double maxD = std::abs(old_d);
    for(int i = 0; i < n; i++)
    {
        const Vector3D &curr = face.vertices[i];
        const Vector3D &next = face.vertices[(i + 1) % n];
        double R = fastabs(next - curr);
        double d1 = old_d;
        double d2 = plane.signedDistance(next);
        maxD = std::max(maxD, std::abs(d2));
        old_d = d2;
        bool in1 = d1 >= -maxR * 1e-12;
        bool in2 = d2 >= -maxR * 1e-12;
        if(print)
        {
            std::cout << "i=" << i << " d1 = " << d1 << " d2 = " << d2 << " in1 = " << in1 << " in2 = " << in2 << " maxR = " << maxR << " R " << R << std::endl;
        }
        if(in1)
        {
            out.vertices.push_back(curr);
        }
        if(in1 != in2)
        {
            if(R * 1e-12 < std::max(std::abs(d1), std::abs(d2)))
            {
                if(R * 1e-10 < std::max(std::abs(d1), std::abs(d2)))
                {
                    Vector3D intersection = plane.intersect(curr, next);
                    out.vertices.push_back(intersection);
                    clip_points.vertices.push_back(intersection);
                }
                else
                {
                    out.vertices.push_back(next);
                    clip_points.vertices.push_back(next);
                }
            }
        }
    }
    Face out2;
    if(maxD < maxR * 1e-10)
    {
        // out2 = face;
        // clip_points.vertices.clear();
        clip_points = face;
    }
    else
    {
        if(out.vertices.size() > 2)
        {
            out2.vertices.push_back(out.vertices[0]);
            size_t Nvert = out.vertices.size();
            for(size_t i = 1; i < Nvert; i++)
            {
                if(fastabs(out.vertices[i] - out2.vertices.back()) > maxR * 1e-12)
                {
                    out2.vertices.push_back(out.vertices[i]);
                }
            }
        }
    }

    return {out2, clip_points};
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
    std::vector<Face> result;
    Face bottom;
    if(print)
    {
        std::cout << "Clipping plane " << plane << std::endl;
    }
    for(const Face &face : faces)
    {
        if(print)
        {
            std::cout << "Clipping face " << face << std::endl;
        }
        auto clipped = clipFace(face, plane, print);
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
            result.push_back(CleanFace(bottom2));
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
    std::vector<Plane> other_poly = CreatePolyPlanes(tess, check_index);
    return clipCells(polyhedron, other_poly, vof, print);
}

std::tuple<double, double, Vector3D> clipCells(const std::vector<Face> &polyhedron, const std::vector<Plane> &other_poly, const Plane *vof, bool print)
{
    std::vector<Face> clipped_poly(polyhedron);
    if(print)
    {
        auto [volume, CM] = computeCM(polyhedron);
        double volume1 = computeVolume(clipped_poly);
        std::cout << "Starting cell clip volume0 " << volume << " volume1 " << volume1 << std::endl;
    }
    const size_t Nplanes = other_poly.size();
    for(size_t i = 0; i < Nplanes; i++)
    {
        clipped_poly = clipPolyhedron(clipped_poly, other_poly[i], print);
        if(print)
        {
            auto [volume, CM] = computeCM(clipped_poly);
            std::cout << "Volume " << volume << " CM " << CM << std::endl;
            std::cout << "Clipped poly: " << std::endl;
            for(const Face &face : clipped_poly)
            {
                std::cout << face << std::endl;
            }
        }
    }
    auto [volume, CM] = computeCM(clipped_poly);
    double vof_volume = 0;
    if(vof != 0)
    {
        if(print)
        {
            std::cout << "Starting vof clip" << std::endl;
        }
        clipped_poly = clipPolyhedron(clipped_poly, *vof, print);
        if(print)
        {
            std::cout << "Clipped poly: " << std::endl;
            for(const Face &face : clipped_poly)
            {
                std::cout << face << std::endl;
            }
        }
        auto [volume2, CM2] = computeCM(clipped_poly);
        vof_volume = std::min(volume, volume2);
    }
    return {volume, vof_volume, CM};
}