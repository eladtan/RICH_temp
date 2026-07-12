#include "SphericalObserver.hpp"
#include "misc/mesh_generator3D.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "misc/universal_error.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace {

struct Triangle { size_t a, b, c; };

void addScalarStat(SphericalObserver::RunningScalarStats& stat, double value)
{
    stat.sum += value;
    stat.sumSq += value * value;
}

double scalarMean(SphericalObserver::RunningScalarStats const& stat, size_t samples)
{
    return samples > 0 ? stat.sum / static_cast<double>(samples) : 0.0;
}

double scalarStderr(SphericalObserver::RunningScalarStats const& stat, size_t samples)
{
    if (samples <= 1)
        return 0.0;
    double const n = static_cast<double>(samples);
    double const var = std::max(0.0, (stat.sumSq - stat.sum * stat.sum / n) / (n - 1.0));
    return std::sqrt(var / n);
}

void initVectorStat(SphericalObserver::RunningVectorStats& stat, size_t n)
{
    if (stat.sum.size() != n) {
        stat.sum.assign(n, 0.0);
        stat.sumSq.assign(n, 0.0);
    }
}

void addVectorStat(SphericalObserver::RunningVectorStats& stat, std::vector<double> const& values)
{
    initVectorStat(stat, values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        stat.sum[i] += values[i];
        stat.sumSq[i] += values[i] * values[i];
    }
}

std::vector<double> vectorMean(SphericalObserver::RunningVectorStats const& stat, size_t samples)
{
    std::vector<double> result(stat.sum.size(), 0.0);
    if (samples == 0)
        return result;
    double const inv = 1.0 / static_cast<double>(samples);
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = stat.sum[i] * inv;
    return result;
}

std::vector<double> vectorStderr(SphericalObserver::RunningVectorStats const& stat, size_t samples)
{
    std::vector<double> result(stat.sum.size(), 0.0);
    if (samples <= 1)
        return result;
    double const n = static_cast<double>(samples);
    for (size_t i = 0; i < result.size(); ++i) {
        double const var = std::max(0.0, (stat.sumSq[i] - stat.sum[i] * stat.sum[i] / n) / (n - 1.0));
        result[i] = std::sqrt(var / n);
    }
    return result;
}

std::vector<double> relativeError(std::vector<double> const& mean,
                                  std::vector<double> const& stderr)
{
    std::vector<double> result(mean.size(), 0.0);
    for (size_t i = 0; i < mean.size(); ++i)
        result[i] = (mean[i] != 0.0) ? stderr[i] / std::abs(mean[i]) : 0.0;
    return result;
}

double relativeError(double mean, double stderr)
{
    return (mean != 0.0) ? stderr / std::abs(mean) : 0.0;
}

void writeVtkScalar(std::ofstream& file,
                    std::string const& name,
                    std::vector<double> const& values,
                    size_t expectedSize)
{
    if (values.empty())
        return;
    if (values.size() != expectedSize)
        throw UniversalError("SphericalObserver::writeVTK: scalar size mismatch for " + name);
    file << "SCALARS " << name << " double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (double value : values)
        file << value << "\n";
}

void writeVtkScalar(std::ofstream& file,
                    std::string const& name,
                    std::vector<int> const& values,
                    size_t expectedSize)
{
    if (values.empty())
        return;
    if (values.size() != expectedSize)
        throw UniversalError("SphericalObserver::writeVTK: scalar size mismatch for " + name);
    file << "SCALARS " << name << " double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int value : values)
        file << static_cast<double>(value) << "\n";
}

std::vector<double> groupColumn(std::vector<std::vector<double>> const& values,
                                size_t group)
{
    std::vector<double> column(values.size(), 0.0);
    for (size_t i = 0; i < values.size(); ++i)
        if (group < values[i].size())
            column[i] = values[i][group];
    return column;
}

std::vector<int> groupColumn(std::vector<std::vector<int>> const& values,
                             size_t group)
{
    std::vector<int> column(values.size(), 0);
    for (size_t i = 0; i < values.size(); ++i)
        if (group < values[i].size())
            column[i] = values[i][group];
    return column;
}

void initMatrixStat(SphericalObserver::RunningMatrixStats& stat, size_t n, size_t m)
{
    if (stat.sum.size() != n || (!stat.sum.empty() && stat.sum[0].size() != m)) {
        stat.sum.assign(n, std::vector<double>(m, 0.0));
        stat.sumSq.assign(n, std::vector<double>(m, 0.0));
    }
}

void addMatrixStat(SphericalObserver::RunningMatrixStats& stat,
                   std::vector<std::vector<double>> const& values)
{
    size_t const n = values.size();
    size_t const m = n > 0 ? values[0].size() : 0;
    initMatrixStat(stat, n, m);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            stat.sum[i][j] += values[i][j];
            stat.sumSq[i][j] += values[i][j] * values[i][j];
        }
    }
}

std::vector<std::vector<double>> matrixMean(SphericalObserver::RunningMatrixStats const& stat,
                                            size_t samples)
{
    std::vector<std::vector<double>> result = stat.sum;
    if (samples == 0)
        return result;
    double const inv = 1.0 / static_cast<double>(samples);
    for (auto& row : result)
        for (auto& value : row)
            value *= inv;
    return result;
}

std::vector<std::vector<double>> matrixStderr(SphericalObserver::RunningMatrixStats const& stat,
                                              size_t samples)
{
    std::vector<std::vector<double>> result = stat.sum;
    for (auto& row : result)
        std::fill(row.begin(), row.end(), 0.0);
    if (samples <= 1)
        return result;
    double const n = static_cast<double>(samples);
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result[i].size(); ++j) {
            double const var = std::max(
                0.0, (stat.sumSq[i][j] - stat.sum[i][j] * stat.sum[i][j] / n) / (n - 1.0));
            result[i][j] = std::sqrt(var / n);
        }
    }
    return result;
}

std::vector<std::vector<double>> relativeError(
    std::vector<std::vector<double>> const& mean,
    std::vector<std::vector<double>> const& stderr)
{
    std::vector<std::vector<double>> result = mean;
    for (size_t i = 0; i < result.size(); ++i)
        for (size_t j = 0; j < result[i].size(); ++j)
            result[i][j] = (mean[i][j] != 0.0) ? stderr[i][j] / std::abs(mean[i][j]) : 0.0;
    return result;
}

std::vector<double> scaleVector(std::vector<double> values, double factor)
{
    for (auto& v : values)
        v *= factor;
    return values;
}

std::vector<std::vector<double>> scaleMatrix(std::vector<std::vector<double>> values, double factor)
{
    for (auto& row : values)
        for (auto& v : row)
            v *= factor;
    return values;
}

std::vector<double> multiplyByObserverFactor(
    std::vector<double> const& values,
    std::vector<double> const& factor)
{
    std::vector<double> result(values.size(), 0.0);
    for (size_t i = 0; i < values.size(); ++i)
        result[i] = values[i] * factor[i];
    return result;
}

std::vector<double> log10Luminosity(std::vector<double> const& lum)
{
    std::vector<double> result(lum.size(), -99.0);
    for (size_t i = 0; i < lum.size(); ++i)
        if (lum[i] > 0.0)
            result[i] = std::log10(lum[i]);
    return result;
}

std::vector<double> log10Stderr(std::vector<double> const& lum,
                                std::vector<double> const& lumStderr)
{
    std::vector<double> result(lum.size(), 0.0);
    double const invLn10 = 1.0 / std::log(10.0);
    for (size_t i = 0; i < lum.size(); ++i)
        if (lum[i] > 0.0)
            result[i] = lumStderr[i] * invLn10 / lum[i];
    return result;
}

std::vector<double> packetStderr(std::vector<double> const& weightSqSum, size_t samples,
                                 double factor = 1.0)
{
    std::vector<double> result(weightSqSum.size(), 0.0);
    if (samples == 0)
        return result;
    double const invSamples = 1.0 / static_cast<double>(samples);
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = std::sqrt(std::max(0.0, weightSqSum[i])) * invSamples * factor;
    return result;
}

std::vector<double> packetNeff(std::vector<double> const& mean,
                               std::vector<double> const& stderr)
{
    std::vector<double> result(mean.size(), 0.0);
    for (size_t i = 0; i < mean.size(); ++i) {
        double const var = stderr[i] * stderr[i];
        result[i] = (var > 0.0) ? mean[i] * mean[i] / var : 0.0;
    }
    return result;
}

std::vector<std::vector<double>> packetStderr(
    std::vector<std::vector<double>> const& weightSqSum,
    size_t samples,
    double factor = 1.0)
{
    std::vector<std::vector<double>> result = weightSqSum;
    if (samples == 0) {
        for (auto& row : result)
            std::fill(row.begin(), row.end(), 0.0);
        return result;
    }
    double const invSamples = 1.0 / static_cast<double>(samples);
    for (auto& row : result)
        for (auto& v : row)
            v = std::sqrt(std::max(0.0, v)) * invSamples * factor;
    return result;
}

std::vector<std::vector<double>> packetNeff(
    std::vector<std::vector<double>> const& mean,
    std::vector<std::vector<double>> const& stderr)
{
    std::vector<std::vector<double>> result = mean;
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result[i].size(); ++j) {
            double const var = stderr[i][j] * stderr[i][j];
            result[i][j] = (var > 0.0) ? mean[i][j] * mean[i][j] / var : 0.0;
        }
    }
    return result;
}

void writeVectorStatErrors(HDF5Writer& writer, std::string const& path,
                           SphericalObserver::RunningVectorStats const& stat,
                           std::vector<double> const& mean,
                           size_t samples)
{
    auto stderr = vectorStderr(stat, samples);
    writer.WriteElement(path + "_stderr_gen", stderr);
    writer.WriteElement(path + "_relerr_gen", relativeError(mean, stderr));
}

void writeMatrixStatErrors(HDF5Writer& writer, std::string const& path,
                           SphericalObserver::RunningMatrixStats const& stat,
                           std::vector<std::vector<double>> const& mean,
                           size_t samples)
{
    auto stderr = matrixStderr(stat, samples);
    writer.WriteElement(path + "_stderr_gen", stderr);
    writer.WriteElement(path + "_relerr_gen", relativeError(mean, stderr));
}

void writeScalarStatErrors(HDF5Writer& writer, std::string const& path,
                           SphericalObserver::RunningScalarStats const& stat,
                           double mean,
                           size_t samples)
{
    double const stderr = scalarStderr(stat, samples);
    writer.WriteElement(path + "_stderr_gen", stderr);
    writer.WriteElement(path + "_relerr_gen", relativeError(mean, stderr));
}

void writeVectorPacketErrors(HDF5Writer& writer, std::string const& path,
                             std::vector<double> const& mean,
                             std::vector<double> const& weightSqSum,
                             size_t samples,
                             double factor = 1.0)
{
    auto stderr = packetStderr(weightSqSum, samples, factor);
    writer.WriteElement(path + "_stderr_packet", stderr);
    writer.WriteElement(path + "_relerr_packet", relativeError(mean, stderr));
    writer.WriteElement(path + "_neff", packetNeff(mean, stderr));
}

void writeMatrixPacketErrors(HDF5Writer& writer, std::string const& path,
                             std::vector<std::vector<double>> const& mean,
                             std::vector<std::vector<double>> const& weightSqSum,
                             size_t samples,
                             double factor = 1.0)
{
    auto stderr = packetStderr(weightSqSum, samples, factor);
    writer.WriteElement(path + "_stderr_packet", stderr);
    writer.WriteElement(path + "_relerr_packet", relativeError(mean, stderr));
    writer.WriteElement(path + "_neff", packetNeff(mean, stderr));
}

void writeScalarVtk(std::ofstream& file, std::string const& name,
                    std::vector<double> const& values)
{
    file << "SCALARS " << name << " double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (double v : values)
        file << v << "\n";
}

#ifdef MONTECARLO_POLARIZATION
std::string groupScalarName(std::string const& prefix, size_t group)
{
    std::ostringstream name;
    name << prefix << std::setw(3) << std::setfill('0') << group;
    return name.str();
}
#endif

std::vector<Triangle> convexHullTriangulation(const std::vector<Vector3D>& pts)
{
    size_t N = pts.size();
    if (N < 4) return {};

    auto outwardNormal = [&](Triangle const& tri) -> Vector3D {
        return CrossProduct(pts[tri.b] - pts[tri.a], pts[tri.c] - pts[tri.a]);
    };

    size_t i0 = 0, i1 = 1, i2 = 2;
    for (size_t i = 2; i < N; ++i) {
        Vector3D c = CrossProduct(pts[i1] - pts[i0], pts[i] - pts[i0]);
        if (abs(c) > 1e-30 * abs(pts[i0])) { i2 = i; break; }
    }

    Vector3D n012 = CrossProduct(pts[i1] - pts[i0], pts[i2] - pts[i0]);
    size_t i3 = i2;
    for (size_t i = 0; i < N; ++i) {
        if (i == i0 || i == i1 || i == i2) continue;
        if (std::abs(ScalarProd(pts[i] - pts[i0], n012)) > std::abs(ScalarProd(pts[i3] - pts[i0], n012)) || i3 == i2)
            i3 = i;
    }

    if (ScalarProd(pts[i3] - pts[i0], n012) > 0)
        std::swap(i1, i2);

    Vector3D centroid = (pts[i0] + pts[i1] + pts[i2] + pts[i3]) * 0.25;

    std::vector<Triangle> triangles;
    triangles.push_back({i0, i1, i2});
    triangles.push_back({i0, i2, i3});
    triangles.push_back({i1, i3, i2});
    triangles.push_back({i0, i3, i1});

    for (auto& tri : triangles) {
        Vector3D n = outwardNormal(tri);
        Vector3D faceMid = (pts[tri.a] + pts[tri.b] + pts[tri.c]) * (1.0/3.0);
        if (ScalarProd(n, faceMid - centroid) < 0)
            std::swap(tri.b, tri.c);
    }

    auto edgeKey = [](size_t a, size_t b) -> uint64_t {
        return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    };

    std::vector<bool> alive(4, true);

    for (size_t pi = 0; pi < N; ++pi) {
        if (pi == i0 || pi == i1 || pi == i2 || pi == i3) continue;
        Vector3D const& p = pts[pi];

        std::vector<bool> visible(triangles.size(), false);
        bool anyVisible = false;
        for (size_t t = 0; t < triangles.size(); ++t) {
            if (!alive[t]) continue;
            Vector3D n = outwardNormal(triangles[t]);
            if (ScalarProd(p - pts[triangles[t].a], n) > 0) {
                visible[t] = true;
                anyVisible = true;
            }
        }
        if (!anyVisible) continue;

        std::map<uint64_t, size_t> edgeFace;
        for (size_t t = 0; t < triangles.size(); ++t) {
            if (!alive[t]) continue;
            size_t v[3] = {triangles[t].a, triangles[t].b, triangles[t].c};
            for (int e = 0; e < 3; ++e)
                edgeFace[edgeKey(v[e], v[(e+1)%3])] = t;
        }

        std::vector<std::pair<size_t,size_t>> horizon;
        for (size_t t = 0; t < triangles.size(); ++t) {
            if (!alive[t] || !visible[t]) continue;
            size_t v[3] = {triangles[t].a, triangles[t].b, triangles[t].c};
            for (int e = 0; e < 3; ++e) {
                size_t ea = v[e], eb = v[(e+1)%3];
                auto it = edgeFace.find(edgeKey(eb, ea));
                if (it != edgeFace.end() && alive[it->second] && !visible[it->second])
                    horizon.push_back({ea, eb});
            }
        }

        for (size_t t = 0; t < triangles.size(); ++t)
            if (visible[t]) alive[t] = false;

        for (auto const& he : horizon) {
            triangles.push_back({he.first, he.second, pi});
            alive.push_back(true);
        }
    }

    std::vector<Triangle> result;
    result.reserve(2 * (N - 2));
    for (size_t t = 0; t < triangles.size(); ++t)
        if (alive[t]) result.push_back(triangles[t]);
    return result;
}

double sphericalTriangleArea(Vector3D const& a,
                             Vector3D const& b,
                             Vector3D const& c)
{
    double const numerator = std::abs(ScalarProd(a, CrossProduct(b, c)));
    double const denominator = 1.0 + ScalarProd(a, b)
                                   + ScalarProd(b, c)
                                   + ScalarProd(c, a);
    return 2.0 * std::atan2(numerator, denominator);
}

bool sphericalCircumcenter(Vector3D const& a,
                           Vector3D const& b,
                           Vector3D const& c,
                           Vector3D& center)
{
    center = CrossProduct(b - a, c - a);
    double const norm = abs(center);
    if (!(norm > 0.0) || !std::isfinite(norm))
        return false;
    center *= 1.0 / norm;

    // The two antipodal normals are both equidistant from a, b, and c.
    // The outward hull normal is the spherical Delaunay circumcenter.
    if (ScalarProd(center, a + b + c) < 0.0)
        center *= -1.0;
    return std::isfinite(center.x) && std::isfinite(center.y)
        && std::isfinite(center.z);
}

std::vector<double> computePerObserverSolidAngles(
    const std::vector<Vector3D>& dirs, size_t N)
{
    std::vector<double> solidAngles(N, 0.0);
    double const fourPi = 4.0 * M_PI;
    double const uniform = fourPi / static_cast<double>(N);
    auto useUniform = [&]() {
        std::fill(solidAngles.begin(), solidAngles.end(), uniform);
        return solidAngles;
    };

    if (N < 4)
        return useUniform();

    std::vector<Triangle> const tris = convexHullTriangulation(dirs);
    size_t const expectedFaces = 2 * (N - 2);
    if (tris.size() != expectedFaces)
        return useUniform();

    // Observer assignment uses the nearest direction on the unit sphere, so
    // each bin is a spherical Voronoi cell. The convex-hull faces are the
    // spherical Delaunay triangles; their circumcenters are Voronoi vertices.
    std::vector<std::vector<Vector3D>> voronoiVertices(N);
    for (auto const& tri : tris) {
        Vector3D center;
        if (!sphericalCircumcenter(
                dirs[tri.a], dirs[tri.b], dirs[tri.c], center))
            return useUniform();
        voronoiVertices[tri.a].push_back(center);
        voronoiVertices[tri.b].push_back(center);
        voronoiVertices[tri.c].push_back(center);
    }

    for (size_t i = 0; i < N; ++i) {
        auto& vertices = voronoiVertices[i];
        if (vertices.size() < 3)
            return useUniform();

        Vector3D const& site = dirs[i];
        Vector3D const reference = std::abs(site.z) < 0.9
            ? Vector3D(0.0, 0.0, 1.0)
            : Vector3D(1.0, 0.0, 0.0);
        Vector3D tangentX = CrossProduct(reference, site);
        double const tangentNorm = abs(tangentX);
        if (!(tangentNorm > 0.0) || !std::isfinite(tangentNorm))
            return useUniform();
        tangentX *= 1.0 / tangentNorm;
        Vector3D const tangentY = CrossProduct(site, tangentX);

        std::sort(vertices.begin(), vertices.end(),
                  [&](Vector3D const& lhs, Vector3D const& rhs) {
            double const lhsAngle = std::atan2(
                ScalarProd(lhs, tangentY), ScalarProd(lhs, tangentX));
            double const rhsAngle = std::atan2(
                ScalarProd(rhs, tangentY), ScalarProd(rhs, tangentX));
            return lhsAngle < rhsAngle;
        });

        double area = 0.0;
        for (size_t j = 0; j < vertices.size(); ++j) {
            area += sphericalTriangleArea(
                site, vertices[j], vertices[(j + 1) % vertices.size()]);
        }
        if (!(area > 0.0) || !std::isfinite(area))
            return useUniform();
        solidAngles[i] = area;
    }

    double const total = std::accumulate(
        solidAngles.begin(), solidAngles.end(), 0.0);
    if (!(total > 0.0) || !std::isfinite(total))
        return useUniform();

    // Remove only accumulated roundoff; the relative Voronoi areas are kept.
    double const scale = fourPi / total;
    for (double& omega : solidAngles)
        omega *= scale;
    return solidAngles;
}

#ifdef MONTECARLO_POLARIZATION
inline double PolarizationDegree(double I, double Q, double U)
{
    if(!(I > 0.0) || !std::isfinite(I))
        return 0.0;
    return std::sqrt(Q*Q + U*U) / I;
}

inline double PolarizationAngle(double Q, double U)
{
    return 0.5 * std::atan2(U, Q);
}
#endif

} // anonymous namespace

SphericalObserver::SphericalObserver(Vector3D center, double radius,
                                     size_t numObservers,
                                     std::vector<double> groupBoundaries)
    : center_(center), radius_(radius), radiusSquared_(radius * radius),
      numObservers_(numObservers), groupBoundaries_(std::move(groupBoundaries))
{
    if (radius_ <= 0.0)
        throw UniversalError("SphericalObserver: radius must be positive");
    if (numObservers_ == 0)
        throw UniversalError("SphericalObserver: numObservers must be > 0");

    if (groupBoundaries_.empty()) {
        numGroups_ = 1;
    } else {
        if (groupBoundaries_.size() < 2)
            throw UniversalError("SphericalObserver: groupBoundaries must have at least 2 entries");
        for (size_t i = 1; i < groupBoundaries_.size(); ++i) {
            if (groupBoundaries_[i] <= groupBoundaries_[i - 1])
                throw UniversalError("SphericalObserver: groupBoundaries must be strictly increasing");
        }
        numGroups_ = groupBoundaries_.size() - 1;
    }

    directions_ = fibonacci_sphere_directions(numObservers_);
    for (auto& d : directions_) {
        double norm = abs(d);
        if (norm > 0.0)
            d = d * (1.0 / norm);
    }

    observerEnergy_.assign(numObservers_, 0.0);
    observerEnergyWeightSq_.assign(numObservers_, 0.0);
    observerMaxPacketEnergy_.assign(numObservers_, 0.0);
    observerCrossingCount_.assign(numObservers_, 0);
    observerSolidAngle_ = computePerObserverSolidAngles(directions_, numObservers_);
    groupEnergy_.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
    groupEnergyWeightSq_.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
    groupCrossingCount_.assign(numObservers_, std::vector<size_t>(numGroups_, 0));
#ifdef MONTECARLO_POLARIZATION
    observerStokesQ_.assign(numObservers_, 0.0);
    observerStokesU_.assign(numObservers_, 0.0);
    observerSumWeightSq_.assign(numObservers_, 0.0);
    observerSumWQ2_.assign(numObservers_, 0.0);
    observerSumWU2_.assign(numObservers_, 0.0);
    mismatchWeightedSum_.assign(numObservers_, 0.0);
    mismatchWeighted2Sum_.assign(numObservers_, 0.0);
    mismatchMax_.assign(numObservers_, 0.0);
    groupStokesQ_.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
    groupStokesU_.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
    groupSumWQ2_.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
    groupSumWU2_.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
    buildSkyBases();
#endif
}

SphericalObserver::Crossing
SphericalObserver::nextOutwardCrossing(Vector3D const& position,
                                       Vector3D const& velocity,
                                       double maxTime) const
{
    Crossing result;
    if (maxTime <= 0.0)
        return result;

    double a = ScalarProd(velocity, velocity);
    if (a <= 0.0)
        return result;

    Vector3D oc = position - center_;
    double b = 2.0 * ScalarProd(oc, velocity);
    double c = ScalarProd(oc, oc) - radiusSquared_;
    double disc = b * b - 4.0 * a * c;
    if (disc < 0.0)
        return result;

    double sqrtDisc = std::sqrt(disc);
    double inv2a = 1.0 / (2.0 * a);
    double t0 = (-b - sqrtDisc) * inv2a;
    double t1 = (-b + sqrtDisc) * inv2a;

    double eps = 1e-12 * std::max(1.0, maxTime);

    double roots[2] = {t0, t1};
    for (double t : roots) {
        if (t <= eps || t > maxTime)
            continue;
        Vector3D point = position + velocity * t;
        Vector3D radial = point - center_;
        double radialNorm = abs(radial);
        if (radialNorm <= 0.0)
            continue;
        radial = radial * (1.0 / radialNorm);
        if (ScalarProd(velocity, radial) > 0.0) {
            result.hit = true;
            result.time = t;
            result.point = point;
            return result;
        }
    }
    return result;
}

void SphericalObserver::recordCrossing(Vector3D const& crossingPoint,
                                       double weight, double frequency)
{
#ifdef MONTECARLO_POLARIZATION
    recordCrossing(crossingPoint, weight, frequency, 0.0, 0.0);
#else
    if (weight == 0.0 || !std::isfinite(weight))
        return;
    size_t obs = findNearestObserver(crossingPoint);
    observerEnergy_[obs] += weight;
    observerEnergyWeightSq_[obs] += weight * weight;
    if (weight > observerMaxPacketEnergy_[obs])
        observerMaxPacketEnergy_[obs] = weight;
    ++observerCrossingCount_[obs];
    if (numGroups_ > 1) {
        size_t g = findGroup(frequency);
        groupEnergy_[obs][g] += weight;
        groupEnergyWeightSq_[obs][g] += weight * weight;
        ++groupCrossingCount_[obs][g];
    }
#endif
}

void SphericalObserver::recordCrossing(ObserverCrossingRecord const& rec)
{
    if (rec.weight == 0.0 || !std::isfinite(rec.weight))
        return;

    Vector3D rhat = rec.crossingPoint - center_;
    double const rnorm = abs(rhat);
    if (rnorm <= 0.0)
        return;
    rhat = rhat * (1.0 / rnorm);

    double const dirNorm = abs(rec.direction);
    if (dirNorm <= 0.0 || !std::isfinite(dirNorm))
        return;
    Vector3D const khat = rec.direction * (1.0 / dirNorm);

    size_t obs = findNearestObserverDirection(khat);
    observerEnergy_[obs] += rec.weight;
    observerEnergyWeightSq_[obs] += rec.weight * rec.weight;
    recordGenerationSourceCellEscape(obs, rec.sourceCellID, rec.weight);
    if (rec.weight > observerMaxPacketEnergy_[obs])
        observerMaxPacketEnergy_[obs] = rec.weight;
    ++observerCrossingCount_[obs];

    if (numGroups_ > 1) {
        size_t g = findGroup(rec.frequency);
        groupEnergy_[obs][g] += rec.weight;
        groupEnergyWeightSq_[obs][g] += rec.weight * rec.weight;
        ++groupCrossingCount_[obs][g];
        if (generationSourceCellGroupStatsEnabled_)
            recordGenerationSourceCellGroupEscape(obs, g, rec.sourceCellID, rec.weight);
    }

#ifdef MONTECARLO_POLARIZATION
    if (polarizationOutputEnabled_) {
        rotateAndAccumulate(rec, obs);
        accumulateMismatch(rec, obs, rhat);
    }
#endif
}

#ifdef MONTECARLO_POLARIZATION
void SphericalObserver::recordCrossing(Vector3D const& crossingPoint,
                                       double weight,
                                       double frequency,
                                       double qObserver,
                                       double uObserver)
{
    if (weight == 0.0 || !std::isfinite(weight))
        return;

    if(!std::isfinite(qObserver))
        qObserver = 0.0;
    if(!std::isfinite(uObserver))
        uObserver = 0.0;

    double const p2 = qObserver*qObserver + uObserver*uObserver;
    if(p2 > 1.0)
    {
        double const invP = 1.0 / std::sqrt(p2);
        qObserver *= invP;
        uObserver *= invP;
    }

    size_t obs = findNearestObserver(crossingPoint);
    observerEnergy_[obs] += weight;
    observerEnergyWeightSq_[obs] += weight * weight;
    // No source cell is available on this legacy overload.
    if (weight > observerMaxPacketEnergy_[obs])
        observerMaxPacketEnergy_[obs] = weight;
    observerStokesQ_[obs] += weight * qObserver;
    observerStokesU_[obs] += weight * uObserver;
    observerSumWeightSq_[obs] += weight * weight;
    observerSumWQ2_[obs] += weight * qObserver * qObserver;
    observerSumWU2_[obs] += weight * uObserver * uObserver;
    ++observerCrossingCount_[obs];

    if (numGroups_ > 1) {
        size_t g = findGroup(frequency);
        groupEnergy_[obs][g] += weight;
        groupEnergyWeightSq_[obs][g] += weight * weight;
        groupStokesQ_[obs][g] += weight * qObserver;
        groupStokesU_[obs][g] += weight * uObserver;
        groupSumWQ2_[obs][g] += weight * qObserver * qObserver;
        groupSumWU2_[obs][g] += weight * uObserver * uObserver;
        ++groupCrossingCount_[obs][g];
    }
}

void SphericalObserver::setPolarizationMetadata(bool enabled,
                                                int manualScatteringsAfterAcceleration,
                                                double depolarizationScatterings,
                                                std::string acceleratedClosure)
{
    polarizationOutputEnabled_ = enabled;
    polarizationManualScatteringsAfterAcceleration_ = manualScatteringsAfterAcceleration;
    polarizationDepolarizationScatterings_ = depolarizationScatterings;
    polarizationAcceleratedClosure_ = std::move(acceleratedClosure);
}

void SphericalObserver::setPolarizationConfig(ObserverPolarizationConfig const& config)
{
    polConfig_ = config;
    double const refNorm = abs(polConfig_.referenceAxis);
    if (refNorm < 1e-14)
        throw UniversalError("ObserverPolarizationConfig: referenceAxis is zero");
    polConfig_.referenceAxis = polConfig_.referenceAxis * (1.0 / refNorm);

    double const fbNorm = abs(polConfig_.fallbackAxis);
    if (fbNorm < 1e-14)
        throw UniversalError("ObserverPolarizationConfig: fallbackAxis is zero");
    polConfig_.fallbackAxis = polConfig_.fallbackAxis * (1.0 / fbNorm);

    polConfig_.poleTolerance = std::clamp(polConfig_.poleTolerance, 0.0, 1.0);
    buildSkyBases();
}

void SphericalObserver::buildSkyBases()
{
    skyE1_.resize(numObservers_);
    for (size_t i = 0; i < numObservers_; ++i) {
        Vector3D const& n = directions_[i];
        Vector3D ref = polConfig_.referenceAxis;
        if (std::abs(ScalarProd(ref, n)) > polConfig_.poleTolerance)
            ref = polConfig_.fallbackAxis;

        Vector3D e1 = ref - ScalarProd(ref, n) * n;
        double const norm1 = abs(e1);
        if (norm1 > 1e-14) {
            e1 = e1 * (1.0 / norm1);
        } else {
            Vector3D fallbackCross = CrossProduct(n, polConfig_.fallbackAxis);
            double const fcNorm = abs(fallbackCross);
            if (fcNorm > 1e-14)
                e1 = fallbackCross * (1.0 / fcNorm);
            else
                e1 = (std::abs(n.z) < 0.9)
                    ? normalize(CrossProduct(n, Vector3D(0, 0, 1)))
                    : normalize(CrossProduct(n, Vector3D(0, 1, 0)));
        }
        skyE1_[i] = e1;
    }
}

void SphericalObserver::rotateAndAccumulate(ObserverCrossingRecord const& rec,
                                            size_t obs)
{
    if (!rec.polarizationInitialized) {
        ++uninitializedPolarizationCount_;
        return;
    }

    double const dirNorm = abs(rec.direction);
    if (dirNorm < 1e-14 || !std::isfinite(dirNorm))
        return;
    Vector3D const khat = rec.direction * (1.0 / dirNorm);
    Vector3D const& skyE1 = skyE1_[obs];

    Vector3D e1_target = skyE1 - ScalarProd(skyE1, khat) * khat;
    double const norm = abs(e1_target);
    if (norm < 1e-14) {
        Vector3D helper = (std::abs(khat.z) < 0.9)
            ? Vector3D(0.0, 0.0, 1.0)
            : Vector3D(0.0, 1.0, 0.0);
        e1_target = helper - ScalarProd(helper, khat) * khat;
        e1_target = normalize(e1_target);
    } else {
        e1_target = e1_target * (1.0 / norm);
    }

    Vector3D polE1 = rec.polBasis - ScalarProd(rec.polBasis, khat) * khat;
    double const polNorm = abs(polE1);
    if (polNorm < 1e-14)
        return;
    polE1 = polE1 * (1.0 / polNorm);

    // Rotation angle psi from packet basis to observer basis around khat.
    // Uses double-angle identities: cos(2psi) = cos^2(psi) - sin^2(psi),
    // sin(2psi) = 2*cos(psi)*sin(psi).
    // This is mathematically identical to IMCPolarization::ProjectToBasis.
    double const cosPsi = std::clamp(ScalarProd(polE1, e1_target), -1.0, 1.0);
    double const sinPsi = ScalarProd(khat, CrossProduct(polE1, e1_target));
    double const cos2psi = cosPsi * cosPsi - sinPsi * sinPsi;
    double const sin2psi = 2.0 * cosPsi * sinPsi;

    double const qRot = rec.stokesQ * cos2psi + rec.stokesU * sin2psi;
    double const uRot = -rec.stokesQ * sin2psi + rec.stokesU * cos2psi;

    observerStokesQ_[obs] += rec.weight * qRot;
    observerStokesU_[obs] += rec.weight * uRot;
    observerSumWeightSq_[obs] += rec.weight * rec.weight;
    observerSumWQ2_[obs] += rec.weight * qRot * qRot;
    observerSumWU2_[obs] += rec.weight * uRot * uRot;

    if (numGroups_ > 1) {
        size_t g = findGroup(rec.frequency);
        groupStokesQ_[obs][g] += rec.weight * qRot;
        groupStokesU_[obs][g] += rec.weight * uRot;
        groupSumWQ2_[obs][g] += rec.weight * qRot * qRot;
        groupSumWU2_[obs][g] += rec.weight * uRot * uRot;
    }
}

void SphericalObserver::accumulateMismatch(ObserverCrossingRecord const& rec,
                                           size_t obs,
                                           Vector3D const& rhat)
{
    double const dirNorm = abs(rec.direction);
    if (dirNorm < 1e-14 || !std::isfinite(dirNorm))
        return;
    Vector3D const khat = rec.direction * (1.0 / dirNorm);
    double const mu = std::clamp(ScalarProd(rhat, khat), -1.0, 1.0);
    double const mismatchAngle = std::acos(mu);

    mismatchWeightedSum_[obs] += rec.weight * mismatchAngle;
    mismatchWeighted2Sum_[obs] += rec.weight * mismatchAngle * mismatchAngle;
    if (mismatchAngle > mismatchMax_[obs])
        mismatchMax_[obs] = mismatchAngle;

    if (polConfig_.warnMismatchAngle > 0.0 &&
        mismatchAngle > polConfig_.warnMismatchAngle)
        ++mismatchWarningCount_;

    if (polConfig_.failMismatchAngle > 0.0 &&
        mismatchAngle > polConfig_.failMismatchAngle) {
        UniversalError eo("Extraction-sphere observer approximation failed: "
                          "khat-rhat mismatch exceeds threshold");
        eo.addEntry("mismatch_angle_rad", mismatchAngle);
        eo.addEntry("threshold_rad", polConfig_.failMismatchAngle);
        eo.addEntry("observer_index", static_cast<double>(obs));
        throw eo;
    }
}
#endif

void SphericalObserver::addEmittedEnergy(double energy) { emittedEnergy_ += energy; }
void SphericalObserver::addAbsorbedEnergy(double energy) { absorbedEnergy_ += energy; }
void SphericalObserver::addBoxEscapeEnergy(double energy) { boxEscapeEnergy_ += energy; }
void SphericalObserver::addTimedOutEnergy(double energy) { timedOutEnergy_ += energy; }
void SphericalObserver::addCutoffEnergy(double energy) { cutoffEnergy_ += energy; }

void SphericalObserver::setPhotosphereData(PhotosphereData data)
{
    photosphereData_ = std::move(data);
}

SphericalObserver::PhotosphereData const& SphericalObserver::getPhotosphereData() const
{
    return photosphereData_;
}

bool SphericalObserver::hasPhotosphereData() const
{
    return photosphereData_.hasAny();
}

void SphericalObserver::resetGenerationSourceCellEscapeStats()
{
    generationSourceCellEscape_.clear();
}

std::vector<SphericalObserver::SourceCellEscapeStat>
SphericalObserver::getGenerationSourceCellEscapeStats() const
{
    std::vector<SourceCellEscapeStat> result;
    size_t totalStats = 0;
    for (auto const& byObserver : generationSourceCellEscape_)
        totalStats += byObserver.second.size();
    result.reserve(totalStats);
    for (auto const& byObserver : generationSourceCellEscape_) {
        for (auto const& kv : byObserver.second)
            result.push_back(kv.second);
    }
    return result;
}

SphericalObserver::ObserverQualitySnapshot
SphericalObserver::getObserverQualitySnapshot() const
{
    ObserverQualitySnapshot snap;
    snap.energy = observerEnergy_;
    snap.energyWeightSq = observerEnergyWeightSq_;
    snap.crossingCount.resize(observerCrossingCount_.size());
    for (size_t i = 0; i < observerCrossingCount_.size(); ++i)
        snap.crossingCount[i] = static_cast<unsigned long long>(observerCrossingCount_[i]);

    size_t const n = observerEnergy_.size();
    snap.stokesQ.assign(n, 0.0);
    snap.stokesU.assign(n, 0.0);
    snap.polarizationWeightSq.assign(n, 0.0);
    snap.sumWQ2.assign(n, 0.0);
    snap.sumWU2.assign(n, 0.0);
#ifdef MONTECARLO_POLARIZATION
    snap.polarizationEnabled = polarizationOutputEnabled_;
    if (polarizationOutputEnabled_) {
        snap.stokesQ = observerStokesQ_;
        snap.stokesU = observerStokesU_;
        snap.polarizationWeightSq = observerSumWeightSq_;
        snap.sumWQ2 = observerSumWQ2_;
        snap.sumWU2 = observerSumWU2_;
    }
#endif
    return snap;
}

void SphericalObserver::resetGenerationSourceCellGroupEscapeStats()
{
    generationSourceCellGroupEscape_.clear();
}

std::vector<SphericalObserver::SourceCellGroupEscapeStat>
SphericalObserver::getGenerationSourceCellGroupEscapeStats() const
{
    std::vector<SourceCellGroupEscapeStat> result;
    result.reserve(generationSourceCellGroupEscape_.size());
    for (auto const& kv : generationSourceCellGroupEscape_)
        result.push_back(kv.second);
    return result;
}

void SphericalObserver::setGenerationSourceCellGroupStatsEnabled(bool enabled)
{
    generationSourceCellGroupStatsEnabled_ = enabled;
    if (!enabled)
        generationSourceCellGroupEscape_.clear();
}

SphericalObserver::ObserverGroupQualitySnapshot
SphericalObserver::getObserverGroupQualitySnapshot() const
{
    ObserverGroupQualitySnapshot snap;
    snap.observerCount = numObservers_;
    snap.groupCount = numGroups_;
    snap.energy = groupEnergy_;
    snap.energyWeightSq = groupEnergyWeightSq_;
    snap.crossingCount = groupCrossingCount_;
    snap.polarizationEnabled = false;
    snap.stokesQ.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
    snap.stokesU.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
    snap.sumWQ2.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
    snap.sumWU2.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
#ifdef MONTECARLO_POLARIZATION
    if (polarizationOutputEnabled_) {
        snap.polarizationEnabled = true;
        snap.stokesQ = groupStokesQ_;
        snap.stokesU = groupStokesU_;
        snap.sumWQ2 = groupSumWQ2_;
        snap.sumWU2 = groupSumWU2_;
    }
#endif
    return snap;
}

void SphericalObserver::resetTallies()
{
    std::fill(observerEnergy_.begin(), observerEnergy_.end(), 0.0);
    std::fill(observerEnergyWeightSq_.begin(), observerEnergyWeightSq_.end(), 0.0);
    std::fill(observerMaxPacketEnergy_.begin(), observerMaxPacketEnergy_.end(), 0.0);
    std::fill(observerCrossingCount_.begin(), observerCrossingCount_.end(), 0);
    for (auto& row : groupEnergy_)
        std::fill(row.begin(), row.end(), 0.0);
    for (auto& row : groupEnergyWeightSq_)
        std::fill(row.begin(), row.end(), 0.0);
    for (auto& row : groupCrossingCount_)
        std::fill(row.begin(), row.end(), 0);
#ifdef MONTECARLO_POLARIZATION
    std::fill(observerStokesQ_.begin(), observerStokesQ_.end(), 0.0);
    std::fill(observerStokesU_.begin(), observerStokesU_.end(), 0.0);
    std::fill(observerSumWeightSq_.begin(), observerSumWeightSq_.end(), 0.0);
    std::fill(observerSumWQ2_.begin(), observerSumWQ2_.end(), 0.0);
    std::fill(observerSumWU2_.begin(), observerSumWU2_.end(), 0.0);
    std::fill(mismatchWeightedSum_.begin(), mismatchWeightedSum_.end(), 0.0);
    std::fill(mismatchWeighted2Sum_.begin(), mismatchWeighted2Sum_.end(), 0.0);
    std::fill(mismatchMax_.begin(), mismatchMax_.end(), 0.0);
    for (auto& row : groupStokesQ_)
        std::fill(row.begin(), row.end(), 0.0);
    for (auto& row : groupStokesU_)
        std::fill(row.begin(), row.end(), 0.0);
    for (auto& row : groupSumWQ2_)
        std::fill(row.begin(), row.end(), 0.0);
    for (auto& row : groupSumWU2_)
        std::fill(row.begin(), row.end(), 0.0);
#endif
    emittedEnergy_ = 0.0;
    absorbedEnergy_ = 0.0;
    boxEscapeEnergy_ = 0.0;
    timedOutEnergy_ = 0.0;
    cutoffEnergy_ = 0.0;
}

void SphericalObserver::clearGenerationStatistics()
{
    generationStats_ = GenerationStatistics{};
}

void SphericalObserver::accumulateCurrentTalliesForStatistics(double sourceDt)
{
    double const invDt = (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0;
    double const fourPi = 4.0 * M_PI;
    std::vector<double> lum = scaleVector(observerEnergy_, invDt);
    std::vector<double> iso(numObservers_, 0.0);
    std::vector<double> flux(numObservers_, 0.0);
    for (size_t i = 0; i < numObservers_; ++i) {
        iso[i] = (observerSolidAngle_[i] > 0.0)
            ? lum[i] * fourPi / observerSolidAngle_[i] : 0.0;
        double const patchArea = observerSolidAngle_[i] * radiusSquared_;
        flux[i] = (patchArea > 0.0) ? lum[i] / patchArea : 0.0;
    }
    addVectorStat(generationStats_.energy, observerEnergy_);
    addVectorStat(generationStats_.luminosity, lum);
    addVectorStat(generationStats_.isoLuminosity, iso);
    addVectorStat(generationStats_.flux, flux);
    addMatrixStat(generationStats_.groupEnergy, groupEnergy_);
    addMatrixStat(generationStats_.groupLuminosity, scaleMatrix(groupEnergy_, invDt));

#ifdef MONTECARLO_POLARIZATION
    if (polarizationOutputEnabled_) {
        std::vector<double> q(numObservers_, 0.0), u(numObservers_, 0.0);
        std::vector<double> qLum(numObservers_, 0.0), uLum(numObservers_, 0.0);
        std::vector<double> polDegree(numObservers_, 0.0), polAngle(numObservers_, 0.0);
        for (size_t i = 0; i < numObservers_; ++i) {
            double const invI = (observerEnergy_[i] > 0.0) ? 1.0 / observerEnergy_[i] : 0.0;
            q[i] = observerStokesQ_[i] * invI;
            u[i] = observerStokesU_[i] * invI;
            qLum[i] = observerStokesQ_[i] * invDt;
            uLum[i] = observerStokesU_[i] * invDt;
            polDegree[i] = PolarizationDegree(observerEnergy_[i], observerStokesQ_[i], observerStokesU_[i]);
            polAngle[i] = PolarizationAngle(observerStokesQ_[i], observerStokesU_[i]);
        }
        addVectorStat(generationStats_.stokesQ, observerStokesQ_);
        addVectorStat(generationStats_.stokesU, observerStokesU_);
        addVectorStat(generationStats_.q, q);
        addVectorStat(generationStats_.u, u);
        addVectorStat(generationStats_.stokesQLuminosity, qLum);
        addVectorStat(generationStats_.stokesULuminosity, uLum);
        addVectorStat(generationStats_.polarizationDegree, polDegree);
        addVectorStat(generationStats_.polarizationAngle, polAngle);

        std::vector<std::vector<double>> gQ(numObservers_, std::vector<double>(numGroups_, 0.0));
        std::vector<std::vector<double>> gU(numObservers_, std::vector<double>(numGroups_, 0.0));
        std::vector<std::vector<double>> gQLum = scaleMatrix(groupStokesQ_, invDt);
        std::vector<std::vector<double>> gULum = scaleMatrix(groupStokesU_, invDt);
        std::vector<std::vector<double>> gDegree(numObservers_, std::vector<double>(numGroups_, 0.0));
        std::vector<std::vector<double>> gAngle(numObservers_, std::vector<double>(numGroups_, 0.0));
        for (size_t i = 0; i < numObservers_; ++i) {
            for (size_t g = 0; g < numGroups_; ++g) {
                double const invI = (groupEnergy_[i][g] > 0.0) ? 1.0 / groupEnergy_[i][g] : 0.0;
                gQ[i][g] = groupStokesQ_[i][g] * invI;
                gU[i][g] = groupStokesU_[i][g] * invI;
                gDegree[i][g] = PolarizationDegree(groupEnergy_[i][g], groupStokesQ_[i][g], groupStokesU_[i][g]);
                gAngle[i][g] = PolarizationAngle(groupStokesQ_[i][g], groupStokesU_[i][g]);
            }
        }
        addMatrixStat(generationStats_.groupStokesQ, groupStokesQ_);
        addMatrixStat(generationStats_.groupStokesU, groupStokesU_);
        addMatrixStat(generationStats_.groupQ, gQ);
        addMatrixStat(generationStats_.groupU, gU);
        addMatrixStat(generationStats_.groupQLuminosity, gQLum);
        addMatrixStat(generationStats_.groupULuminosity, gULum);
        addMatrixStat(generationStats_.groupPolarizationDegree, gDegree);
        addMatrixStat(generationStats_.groupPolarizationAngle, gAngle);
    }
#endif

    double const totalEnergy = getTotalCrossingEnergy();
    addScalarStat(generationStats_.totalEnergy, totalEnergy);
    addScalarStat(generationStats_.totalLuminosity, totalEnergy * invDt);
    addScalarStat(generationStats_.emittedEnergy, emittedEnergy_);
    addScalarStat(generationStats_.absorbedEnergy, absorbedEnergy_);
    addScalarStat(generationStats_.boxEscapeEnergy, boxEscapeEnergy_);
    addScalarStat(generationStats_.timedOutEnergy, timedOutEnergy_);
    addScalarStat(generationStats_.cutoffEnergy, cutoffEnergy_);
    double const transportSinkResidual = emittedEnergy_ - absorbedEnergy_
        - boxEscapeEnergy_ - timedOutEnergy_ - cutoffEnergy_;
    double const timedOutFraction = (emittedEnergy_ > 0.0)
        ? timedOutEnergy_ / emittedEnergy_ : 0.0;
    addScalarStat(generationStats_.transportSinkResidual, transportSinkResidual);
    addScalarStat(generationStats_.timedOutFraction, timedOutFraction);

    if (generationStats_.samples == 0) {
        generationStats_.energyWeightSqSum.assign(numObservers_, 0.0);
        generationStats_.groupEnergyWeightSqSum.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
        generationStats_.observerCrossingCountSum.assign(numObservers_, 0);
        generationStats_.groupCrossingCountSum.assign(numObservers_, std::vector<size_t>(numGroups_, 0));
        generationStats_.observerMaxPacketEnergyMax.assign(numObservers_, 0.0);
#ifdef MONTECARLO_POLARIZATION
        generationStats_.observerSumWeightSqSum.assign(numObservers_, 0.0);
        generationStats_.observerSumWQ2Sum.assign(numObservers_, 0.0);
        generationStats_.observerSumWU2Sum.assign(numObservers_, 0.0);
        generationStats_.groupSumWQ2Sum.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
        generationStats_.groupSumWU2Sum.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
        generationStats_.mismatchWeightedSumSum.assign(numObservers_, 0.0);
        generationStats_.mismatchWeighted2SumSum.assign(numObservers_, 0.0);
        generationStats_.mismatchMaxMax.assign(numObservers_, 0.0);
#endif
    }
    for (size_t i = 0; i < numObservers_; ++i) {
        generationStats_.energyWeightSqSum[i] += observerEnergyWeightSq_[i];
        generationStats_.observerCrossingCountSum[i] += observerCrossingCount_[i];
        generationStats_.observerMaxPacketEnergyMax[i] =
            std::max(generationStats_.observerMaxPacketEnergyMax[i], observerMaxPacketEnergy_[i]);
        for (size_t g = 0; g < numGroups_; ++g) {
            generationStats_.groupEnergyWeightSqSum[i][g] += groupEnergyWeightSq_[i][g];
            generationStats_.groupCrossingCountSum[i][g] += groupCrossingCount_[i][g];
        }
#ifdef MONTECARLO_POLARIZATION
        generationStats_.observerSumWeightSqSum[i] += observerSumWeightSq_[i];
        generationStats_.observerSumWQ2Sum[i] += observerSumWQ2_[i];
        generationStats_.observerSumWU2Sum[i] += observerSumWU2_[i];
        for (size_t g = 0; g < numGroups_; ++g) {
            generationStats_.groupSumWQ2Sum[i][g] += groupSumWQ2_[i][g];
            generationStats_.groupSumWU2Sum[i][g] += groupSumWU2_[i][g];
        }
        generationStats_.mismatchWeightedSumSum[i] += mismatchWeightedSum_[i];
        generationStats_.mismatchWeighted2SumSum[i] += mismatchWeighted2Sum_[i];
        generationStats_.mismatchMaxMax[i] =
            std::max(generationStats_.mismatchMaxMax[i], mismatchMax_[i]);
#endif
    }
    generationStats_.totalEnergyWeightSqSum +=
        std::accumulate(observerEnergyWeightSq_.begin(), observerEnergyWeightSq_.end(), 0.0);
    generationStats_.samples += 1;
}

void SphericalObserver::loadStatisticalMeanTallies()
{
    size_t const samples = generationStats_.samples;
    if (samples == 0)
        return;
    observerEnergy_ = vectorMean(generationStats_.energy, samples);
    groupEnergy_ = matrixMean(generationStats_.groupEnergy, samples);
    emittedEnergy_ = scalarMean(generationStats_.emittedEnergy, samples);
    absorbedEnergy_ = scalarMean(generationStats_.absorbedEnergy, samples);
    boxEscapeEnergy_ = scalarMean(generationStats_.boxEscapeEnergy, samples);
    timedOutEnergy_ = scalarMean(generationStats_.timedOutEnergy, samples);
    cutoffEnergy_ = scalarMean(generationStats_.cutoffEnergy, samples);
    observerEnergyWeightSq_ = generationStats_.energyWeightSqSum;
    groupEnergyWeightSq_ = generationStats_.groupEnergyWeightSqSum;
    observerCrossingCount_ = generationStats_.observerCrossingCountSum;
    groupCrossingCount_ = generationStats_.groupCrossingCountSum;
    observerMaxPacketEnergy_ = generationStats_.observerMaxPacketEnergyMax;
#ifdef MONTECARLO_POLARIZATION
    if (polarizationOutputEnabled_) {
        observerStokesQ_ = vectorMean(generationStats_.stokesQ, samples);
        observerStokesU_ = vectorMean(generationStats_.stokesU, samples);
        groupStokesQ_ = matrixMean(generationStats_.groupStokesQ, samples);
        groupStokesU_ = matrixMean(generationStats_.groupStokesU, samples);
        double const invSamples = 1.0 / static_cast<double>(samples);
        double const invSamplesSq = invSamples * invSamples;
        observerSumWeightSq_ = generationStats_.observerSumWeightSqSum;
        observerSumWQ2_ = generationStats_.observerSumWQ2Sum;
        observerSumWU2_ = generationStats_.observerSumWU2Sum;
        groupSumWQ2_ = generationStats_.groupSumWQ2Sum;
        groupSumWU2_ = generationStats_.groupSumWU2Sum;
        mismatchWeightedSum_ = generationStats_.mismatchWeightedSumSum;
        mismatchWeighted2Sum_ = generationStats_.mismatchWeighted2SumSum;
        mismatchMax_ = generationStats_.mismatchMaxMax;
        for (auto& e : observerSumWeightSq_) e *= invSamplesSq;
        for (auto& row : groupEnergyWeightSq_)
            for (auto& e : row)
                e *= invSamplesSq;
        for (auto& e : observerSumWQ2_) e *= invSamples;
        for (auto& e : observerSumWU2_) e *= invSamples;
        for (auto& row : groupSumWQ2_)
            for (auto& e : row)
                e *= invSamples;
        for (auto& row : groupSumWU2_)
            for (auto& e : row)
                e *= invSamples;
        for (auto& e : mismatchWeightedSum_) e *= invSamples;
        for (auto& e : mismatchWeighted2Sum_) e *= invSamples;
    }
#endif
}

size_t SphericalObserver::getStatisticsSamples() const
{
    return generationStats_.samples;
}

double SphericalObserver::getTotalLuminosityStderrGen(double sourceDt) const
{
    (void)sourceDt;
    return scalarStderr(generationStats_.totalLuminosity, generationStats_.samples);
}

double SphericalObserver::getTotalLuminosityRelErrGen(double sourceDt) const
{
    double const mean = (sourceDt > 0.0) ? getTotalCrossingEnergy() / sourceDt : 0.0;
    return relativeError(mean, getTotalLuminosityStderrGen(sourceDt));
}

void SphericalObserver::recordGenerationSourceCellEscape(size_t observerIndex, size_t cellID, double energy)
{
    if (observerIndex == std::numeric_limits<size_t>::max() ||
        cellID == std::numeric_limits<size_t>::max() ||
        !(energy > 0.0) || !std::isfinite(energy))
        return;
    auto& stat = generationSourceCellEscape_[observerIndex][cellID];
    stat.cellID = cellID;
    stat.observerIndex = observerIndex;
    stat.energy += energy;
    stat.weightSq += energy * energy;
    stat.maxWeight = std::max(stat.maxWeight, energy);
    ++stat.count;
}

void SphericalObserver::recordGenerationSourceCellGroupEscape(size_t observerIndex, size_t groupIndex, size_t cellID, double weight)
{
    if (observerIndex == std::numeric_limits<size_t>::max() ||
        groupIndex == std::numeric_limits<size_t>::max() ||
        cellID == std::numeric_limits<size_t>::max() ||
        !(weight > 0.0) || !std::isfinite(weight))
        return;
    SourceObserverGroupCellKey key{observerIndex, groupIndex, cellID};
    auto& stat = generationSourceCellGroupEscape_[key];
    stat.cellID = cellID;
    stat.observerIndex = observerIndex;
    stat.groupIndex = groupIndex;
    stat.energy += weight;
    stat.weightSq += weight * weight;
    stat.maxWeight = std::max(stat.maxWeight, weight);
    ++stat.count;
}

size_t SphericalObserver::findNearestObserverDirection(Vector3D const& direction) const
{
    Vector3D u = direction;
    double norm = abs(u);
    if (norm <= 0.0 || !std::isfinite(norm))
        return 0;
    u = u * (1.0 / norm);

    size_t best = 0;
    double bestDot = ScalarProd(u, directions_[0]);
    for (size_t i = 1; i < numObservers_; ++i) {
        double d = ScalarProd(u, directions_[i]);
        if (d > bestDot) {
            bestDot = d;
            best = i;
        }
    }
    return best;
}

size_t SphericalObserver::findNearestObserver(Vector3D const& crossingPoint) const
{
    return findNearestObserverDirection(crossingPoint - center_);
}

size_t SphericalObserver::findGroup(double frequency) const
{
    if (numGroups_ == 1)
        return 0;
    auto it = std::upper_bound(groupBoundaries_.begin(), groupBoundaries_.end(), frequency);
    if (it == groupBoundaries_.begin())
        return 0;
    size_t idx = static_cast<size_t>(std::distance(groupBoundaries_.begin(), it)) - 1;
    return std::min(idx, numGroups_ - 1);
}

std::vector<double> SphericalObserver::getLuminosity(double sourceDt) const
{
    std::vector<double> lum(numObservers_);
    double inv = (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0;
    for (size_t i = 0; i < numObservers_; ++i)
        lum[i] = observerEnergy_[i] * inv;
    return lum;
}

std::vector<std::vector<double>> SphericalObserver::getGroupLuminosity(double sourceDt) const
{
    std::vector<std::vector<double>> lum(numObservers_, std::vector<double>(numGroups_));
    double inv = (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0;
    for (size_t i = 0; i < numObservers_; ++i)
        for (size_t g = 0; g < numGroups_; ++g)
            lum[i][g] = groupEnergy_[i][g] * inv;
    return lum;
}

double SphericalObserver::getTotalCrossingEnergy() const
{
    return std::accumulate(observerEnergy_.begin(), observerEnergy_.end(), 0.0);
}

double SphericalObserver::getEmittedEnergy() const { return emittedEnergy_; }
double SphericalObserver::getAbsorbedEnergy() const { return absorbedEnergy_; }
double SphericalObserver::getBoxEscapeEnergy() const { return boxEscapeEnergy_; }
double SphericalObserver::getTimedOutEnergy() const { return timedOutEnergy_; }
double SphericalObserver::getCutoffEnergy() const { return cutoffEnergy_; }

std::vector<Vector3D> const& SphericalObserver::getDirections() const { return directions_; }
Vector3D SphericalObserver::getCenter() const { return center_; }
double SphericalObserver::getRadius() const { return radius_; }
size_t SphericalObserver::getNumObservers() const { return numObservers_; }
size_t SphericalObserver::getNumGroups() const { return numGroups_; }

double SphericalObserver::getSolidAngle() const
{
    return 4.0 * M_PI / static_cast<double>(numObservers_);
}

double SphericalObserver::getPatchArea() const
{
    return 4.0 * M_PI * radiusSquared_ / static_cast<double>(numObservers_);
}

std::vector<double> const& SphericalObserver::getObserverSolidAngles() const
{
    return observerSolidAngle_;
}

std::vector<double> const& SphericalObserver::getMaxPacketEnergy() const
{
    return observerMaxPacketEnergy_;
}

void SphericalObserver::scale(double factor)
{
    for (auto& e : observerEnergy_) e *= factor;
    for (auto& e : observerEnergyWeightSq_) e *= factor * factor;
    for (auto& gv : groupEnergy_)
        for (auto& e : gv) e *= factor;
    for (auto& gv : groupEnergyWeightSq_)
        for (auto& e : gv) e *= factor * factor;
#ifdef MONTECARLO_POLARIZATION
    for (auto& e : observerStokesQ_) e *= factor;
    for (auto& e : observerStokesU_) e *= factor;
    for (auto& e : observerSumWeightSq_) e *= factor * factor;
    for (auto& e : observerSumWQ2_) e *= factor;
    for (auto& e : observerSumWU2_) e *= factor;
    for (auto& e : mismatchWeightedSum_) e *= factor;
    for (auto& e : mismatchWeighted2Sum_) e *= factor;
    for (auto& gv : groupStokesQ_)
        for (auto& e : gv) e *= factor;
    for (auto& gv : groupStokesU_)
        for (auto& e : gv) e *= factor;
    for (auto& gv : groupSumWQ2_)
        for (auto& e : gv) e *= factor;
    for (auto& gv : groupSumWU2_)
        for (auto& e : gv) e *= factor;
#endif
    emittedEnergy_ *= factor;
    absorbedEnergy_ *= factor;
    boxEscapeEnergy_ *= factor;
    timedOutEnergy_ *= factor;
    cutoffEnergy_ *= factor;
}

void SphericalObserver::mpiReduceToRank0()
{
#ifdef RICH_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (observerEnergy_.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw UniversalError("SphericalObserver MPI reduction too large");

    int count = static_cast<int>(observerEnergy_.size());
    std::vector<double> recvBuf(static_cast<size_t>(count), 0.0);

    MPI_Reduce(observerEnergy_.data(), recvBuf.data(), count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        observerEnergy_ = recvBuf;

    MPI_Reduce(observerEnergyWeightSq_.data(), recvBuf.data(), count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        observerEnergyWeightSq_ = recvBuf;

    std::vector<double> maxRecvBuf(static_cast<size_t>(count), 0.0);
    MPI_Reduce(observerMaxPacketEnergy_.data(), maxRecvBuf.data(), count,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        observerMaxPacketEnergy_ = maxRecvBuf;

#ifdef MONTECARLO_POLARIZATION
    MPI_Reduce(observerStokesQ_.data(), recvBuf.data(), count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        observerStokesQ_ = recvBuf;

    MPI_Reduce(observerStokesU_.data(), recvBuf.data(), count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        observerStokesU_ = recvBuf;

    MPI_Reduce(observerSumWeightSq_.data(), recvBuf.data(), count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        observerSumWeightSq_ = recvBuf;

    MPI_Reduce(observerSumWQ2_.data(), recvBuf.data(), count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        observerSumWQ2_ = recvBuf;

    MPI_Reduce(observerSumWU2_.data(), recvBuf.data(), count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        observerSumWU2_ = recvBuf;

    MPI_Reduce(mismatchWeightedSum_.data(), recvBuf.data(), count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        mismatchWeightedSum_ = recvBuf;

    MPI_Reduce(mismatchWeighted2Sum_.data(), recvBuf.data(), count,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        mismatchWeighted2Sum_ = recvBuf;

    std::vector<double> maxMismatchRecv(static_cast<size_t>(count), 0.0);
    MPI_Reduce(mismatchMax_.data(), maxMismatchRecv.data(), count,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        mismatchMax_ = maxMismatchRecv;
#endif

    std::vector<size_t> countRecvBuf(static_cast<size_t>(count), 0);
    MPI_Reduce(observerCrossingCount_.data(), countRecvBuf.data(), count,
               MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        observerCrossingCount_ = countRecvBuf;

    size_t flatSize = numObservers_ * numGroups_;
    if (flatSize > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw UniversalError("SphericalObserver MPI group reduction too large");

    std::vector<double> flatSend(flatSize, 0.0);
    for (size_t i = 0; i < numObservers_; ++i)
        for (size_t g = 0; g < numGroups_; ++g)
            flatSend[i * numGroups_ + g] = groupEnergy_[i][g];

    std::vector<double> flatRecv(flatSize, 0.0);
    MPI_Reduce(flatSend.data(), flatRecv.data(), static_cast<int>(flatSize),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (size_t i = 0; i < numObservers_; ++i)
            for (size_t g = 0; g < numGroups_; ++g)
                groupEnergy_[i][g] = flatRecv[i * numGroups_ + g];
    }

    for (size_t i = 0; i < numObservers_; ++i)
        for (size_t g = 0; g < numGroups_; ++g)
            flatSend[i * numGroups_ + g] = groupEnergyWeightSq_[i][g];
    std::fill(flatRecv.begin(), flatRecv.end(), 0.0);
    MPI_Reduce(flatSend.data(), flatRecv.data(), static_cast<int>(flatSize),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (size_t i = 0; i < numObservers_; ++i)
            for (size_t g = 0; g < numGroups_; ++g)
                groupEnergyWeightSq_[i][g] = flatRecv[i * numGroups_ + g];
    }

#ifdef MONTECARLO_POLARIZATION
    for (size_t i = 0; i < numObservers_; ++i)
        for (size_t g = 0; g < numGroups_; ++g)
            flatSend[i * numGroups_ + g] = groupStokesQ_[i][g];
    std::fill(flatRecv.begin(), flatRecv.end(), 0.0);
    MPI_Reduce(flatSend.data(), flatRecv.data(), static_cast<int>(flatSize),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (size_t i = 0; i < numObservers_; ++i)
            for (size_t g = 0; g < numGroups_; ++g)
                groupStokesQ_[i][g] = flatRecv[i * numGroups_ + g];
    }

    for (size_t i = 0; i < numObservers_; ++i)
        for (size_t g = 0; g < numGroups_; ++g)
            flatSend[i * numGroups_ + g] = groupStokesU_[i][g];
    std::fill(flatRecv.begin(), flatRecv.end(), 0.0);
    MPI_Reduce(flatSend.data(), flatRecv.data(), static_cast<int>(flatSize),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (size_t i = 0; i < numObservers_; ++i)
            for (size_t g = 0; g < numGroups_; ++g)
                groupStokesU_[i][g] = flatRecv[i * numGroups_ + g];
    }

    for (size_t i = 0; i < numObservers_; ++i)
        for (size_t g = 0; g < numGroups_; ++g)
            flatSend[i * numGroups_ + g] = groupSumWQ2_[i][g];
    std::fill(flatRecv.begin(), flatRecv.end(), 0.0);
    MPI_Reduce(flatSend.data(), flatRecv.data(), static_cast<int>(flatSize),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (size_t i = 0; i < numObservers_; ++i)
            for (size_t g = 0; g < numGroups_; ++g)
                groupSumWQ2_[i][g] = flatRecv[i * numGroups_ + g];
    }

    for (size_t i = 0; i < numObservers_; ++i)
        for (size_t g = 0; g < numGroups_; ++g)
            flatSend[i * numGroups_ + g] = groupSumWU2_[i][g];
    std::fill(flatRecv.begin(), flatRecv.end(), 0.0);
    MPI_Reduce(flatSend.data(), flatRecv.data(), static_cast<int>(flatSize),
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (size_t i = 0; i < numObservers_; ++i)
            for (size_t g = 0; g < numGroups_; ++g)
                groupSumWU2_[i][g] = flatRecv[i * numGroups_ + g];
    }
#endif

    std::vector<size_t> flatCountSend(flatSize, 0);
    for (size_t i = 0; i < numObservers_; ++i)
        for (size_t g = 0; g < numGroups_; ++g)
            flatCountSend[i * numGroups_ + g] = groupCrossingCount_[i][g];

    std::vector<size_t> flatCountRecv(flatSize, 0);
    MPI_Reduce(flatCountSend.data(), flatCountRecv.data(), static_cast<int>(flatSize),
               MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (size_t i = 0; i < numObservers_; ++i)
            for (size_t g = 0; g < numGroups_; ++g)
                groupCrossingCount_[i][g] = flatCountRecv[i * numGroups_ + g];
    }

    double scalars[5] = {emittedEnergy_, absorbedEnergy_, boxEscapeEnergy_,
                         timedOutEnergy_, cutoffEnergy_};
    double scalarRecv[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Reduce(scalars, scalarRecv, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        emittedEnergy_ = scalarRecv[0];
        absorbedEnergy_ = scalarRecv[1];
        boxEscapeEnergy_ = scalarRecv[2];
        timedOutEnergy_ = scalarRecv[3];
        cutoffEnergy_ = scalarRecv[4];
    }
#endif
}

void SphericalObserver::writeHDF5(std::string const& filename,
                                  Diagnostics const& diagnostics) const
{
    HDF5Writer writer(filename);

    writer.WriteElement("/observer/radius", radius_);
    std::vector<double> centerVec = {center_.x, center_.y, center_.z};
    writer.WriteElement("/observer/center", centerVec);
    writer.WriteElement("/observer/num_observers", numObservers_);

    std::vector<double> dirX(numObservers_), dirY(numObservers_), dirZ(numObservers_);
    for (size_t i = 0; i < numObservers_; ++i) {
        dirX[i] = directions_[i].x;
        dirY[i] = directions_[i].y;
        dirZ[i] = directions_[i].z;
    }
    writer.WriteElement("/observer/direction_x", dirX);
    writer.WriteElement("/observer/direction_y", dirY);
    writer.WriteElement("/observer/direction_z", dirZ);
    writer.WriteElement("/observer/mean_solid_angle", getSolidAngle());
    writer.WriteElement("/observer/mean_patch_area", getPatchArea());
    writer.WriteElement("/observer/solid_angle", observerSolidAngle_);

    double sourceDt = diagnostics.sourceDt;
    std::vector<double> lum = getLuminosity(sourceDt);
    double totalEnergy = getTotalCrossingEnergy();
    double totalLum = (sourceDt > 0.0) ? totalEnergy / sourceDt : 0.0;

    writer.WriteElement("/tally/energy", observerEnergy_);
    writer.WriteElement("/tally/luminosity", lum);
    writer.WriteElement("/tally/total_energy", totalEnergy);
    writer.WriteElement("/tally/total_luminosity", totalLum);
#ifdef MONTECARLO_POLARIZATION
    double invDt = (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0;
    if(polarizationOutputEnabled_)
    {
        std::vector<double> polDegree(numObservers_), polAngle(numObservers_);
        std::vector<double> qLum(numObservers_), uLum(numObservers_);
        std::vector<double> qNorm(numObservers_), uNorm(numObservers_);
        for (size_t i = 0; i < numObservers_; ++i) {
            polDegree[i] = PolarizationDegree(observerEnergy_[i], observerStokesQ_[i], observerStokesU_[i]);
            polAngle[i] = PolarizationAngle(observerStokesQ_[i], observerStokesU_[i]);
            qLum[i] = observerStokesQ_[i] * invDt;
            uLum[i] = observerStokesU_[i] * invDt;
            double const invI = (observerEnergy_[i] > 0.0) ? 1.0 / observerEnergy_[i] : 0.0;
            qNorm[i] = observerStokesQ_[i] * invI;
            uNorm[i] = observerStokesU_[i] * invI;
        }
        writer.WriteElement("/tally/observer_stokes_Q", observerStokesQ_);
        writer.WriteElement("/tally/observer_stokes_U", observerStokesU_);
        writer.WriteElement("/tally/observer_q", qNorm);
        writer.WriteElement("/tally/observer_u", uNorm);
        writer.WriteElement("/tally/observer_Q_luminosity", qLum);
        writer.WriteElement("/tally/observer_U_luminosity", uLum);
        writer.WriteElement("/tally/observer_polarization_degree", polDegree);
        writer.WriteElement("/tally/observer_polarization_angle", polAngle);

        std::vector<double> meanMismatch(numObservers_, 0.0);
        std::vector<double> rmsMismatch(numObservers_, 0.0);
        for (size_t i = 0; i < numObservers_; ++i) {
            if (observerEnergy_[i] > 0.0) {
                double const invW = 1.0 / observerEnergy_[i];
                meanMismatch[i] = mismatchWeightedSum_[i] * invW;
                double const var = mismatchWeighted2Sum_[i] * invW
                                 - meanMismatch[i] * meanMismatch[i];
                rmsMismatch[i] = std::sqrt(std::max(0.0, var));
            }
        }
        writer.WriteElement("/tally/observer_mean_direction_mismatch", meanMismatch);
        writer.WriteElement("/tally/observer_rms_direction_mismatch", rmsMismatch);
        writer.WriteElement("/tally/observer_max_direction_mismatch", mismatchMax_);

        std::vector<double> crossingCountDbl(numObservers_);
        for (size_t i = 0; i < numObservers_; ++i)
            crossingCountDbl[i] = static_cast<double>(observerCrossingCount_[i]);
        writer.WriteElement("/tally/observer_packet_count", crossingCountDbl);

        std::vector<double> theta(numObservers_), phi(numObservers_);
        for (size_t i = 0; i < numObservers_; ++i) {
            theta[i] = std::acos(std::clamp(directions_[i].z, -1.0, 1.0));
            phi[i] = std::atan2(directions_[i].y, directions_[i].x);
        }
        writer.WriteElement("/observer/theta", theta);
        writer.WriteElement("/observer/phi", phi);
    }
#endif

    double fourPi = 4.0 * M_PI;
    std::vector<double> isoEquiv(numObservers_);
    for (size_t i = 0; i < numObservers_; ++i)
        isoEquiv[i] = (observerSolidAngle_[i] > 0.0)
            ? lum[i] * fourPi / observerSolidAngle_[i] : 0.0;
    writer.WriteElement("/tally/isotropic_equivalent_luminosity", isoEquiv);

    std::vector<double> flux(numObservers_);
    for (size_t i = 0; i < numObservers_; ++i) {
        double patchArea_i = observerSolidAngle_[i] * radiusSquared_;
        flux[i] = (patchArea_i > 0.0) ? lum[i] / patchArea_i : 0.0;
    }
    writer.WriteElement("/tally/flux", flux);
    writer.WriteElement("/tally/log10_luminosity", log10Luminosity(lum));

    if (photosphereData_.hasAny()) {
        writer.WriteElement("/photosphere/tau_threshold", photosphereData_.tauThreshold);
        writer.WriteElement("/photosphere/tau_total_threshold", photosphereData_.tauThreshold);
        writer.WriteElement("/photosphere/thermalization_tau_threshold",
                            photosphereData_.thermalizationTauThreshold);
        if (photosphereData_.hasMG()) {
            if (!photosphereData_.mgGroupRadiusTauTotal.empty()) {
                writer.WriteElement("/photosphere/mg/group_radius_tau_total",
                                    photosphereData_.mgGroupRadiusTauTotal);
                writer.WriteElement("/photosphere/mg/group_radius_thermalization",
                                    photosphereData_.mgGroupRadiusThermalization);
                writer.WriteElement("/photosphere/mg/group_valid_tau_total",
                                    photosphereData_.mgGroupValidTauTotal);
                writer.WriteElement("/photosphere/mg/group_valid_thermalization",
                                    photosphereData_.mgGroupValidThermalization);
            }
            writer.WriteElement("/photosphere/mg/integrated_radius_tau_total",
                                photosphereData_.mgIntegratedRadiusTauTotal);
            writer.WriteElement("/photosphere/mg/integrated_radius_thermalization",
                                photosphereData_.mgIntegratedRadiusThermalization);
            writer.WriteElement("/photosphere/mg/integrated_valid_tau_total",
                                photosphereData_.mgIntegratedValidTauTotal);
            writer.WriteElement("/photosphere/mg/integrated_valid_thermalization",
                                photosphereData_.mgIntegratedValidThermalization);
        }
        if (photosphereData_.hasGrey()) {
            writer.WriteElement("/photosphere/grey/radius_tau_total",
                                photosphereData_.greyRadiusTauTotal);
            writer.WriteElement("/photosphere/grey/radius_thermalization",
                                photosphereData_.greyRadiusThermalization);
            writer.WriteElement("/photosphere/grey/valid_tau_total",
                                photosphereData_.greyValidTauTotal);
            writer.WriteElement("/photosphere/grey/valid_thermalization",
                                photosphereData_.greyValidThermalization);
        }
    }

    if (numGroups_ > 1) {
        writer.WriteElement("/tally/multigroup/group_boundaries", groupBoundaries_);

        std::vector<std::vector<double>> gLum = getGroupLuminosity(sourceDt);
        writer.WriteElement("/tally/multigroup/group_energy", groupEnergy_);
        writer.WriteElement("/tally/multigroup/group_luminosity", gLum);
        writer.WriteElement("/tally/multigroup/group_crossing_count", groupCrossingCount_);
#ifdef MONTECARLO_POLARIZATION
        if(polarizationOutputEnabled_)
        {
            std::vector<std::vector<double>> gDegree(numObservers_, std::vector<double>(numGroups_, 0.0));
            std::vector<std::vector<double>> gAngle(numObservers_, std::vector<double>(numGroups_, 0.0));
            std::vector<std::vector<double>> gQNorm(numObservers_, std::vector<double>(numGroups_, 0.0));
            std::vector<std::vector<double>> gUNorm(numObservers_, std::vector<double>(numGroups_, 0.0));
            std::vector<std::vector<double>> gQLum(numObservers_, std::vector<double>(numGroups_, 0.0));
            std::vector<std::vector<double>> gULum(numObservers_, std::vector<double>(numGroups_, 0.0));
            for (size_t i = 0; i < numObservers_; ++i) {
                for (size_t g = 0; g < numGroups_; ++g) {
                    gDegree[i][g] = PolarizationDegree(groupEnergy_[i][g], groupStokesQ_[i][g], groupStokesU_[i][g]);
                    gAngle[i][g] = PolarizationAngle(groupStokesQ_[i][g], groupStokesU_[i][g]);
                    double const invI = (groupEnergy_[i][g] > 0.0) ? 1.0 / groupEnergy_[i][g] : 0.0;
                    gQNorm[i][g] = groupStokesQ_[i][g] * invI;
                    gUNorm[i][g] = groupStokesU_[i][g] * invI;
                    gQLum[i][g] = groupStokesQ_[i][g] * invDt;
                    gULum[i][g] = groupStokesU_[i][g] * invDt;
                }
            }
            writer.WriteElement("/tally/multigroup/group_stokes_Q", groupStokesQ_);
            writer.WriteElement("/tally/multigroup/group_stokes_U", groupStokesU_);
            writer.WriteElement("/tally/multigroup/group_q", gQNorm);
            writer.WriteElement("/tally/multigroup/group_u", gUNorm);
            writer.WriteElement("/tally/multigroup/group_Q_luminosity", gQLum);
            writer.WriteElement("/tally/multigroup/group_U_luminosity", gULum);
            writer.WriteElement("/tally/multigroup/group_polarization_degree", gDegree);
            writer.WriteElement("/tally/multigroup/group_polarization_angle", gAngle);
        }
#endif
    }

    writer.WriteElement("/diagnostics/source_dt", diagnostics.sourceDt);
    writer.WriteElement("/diagnostics/transport_time", diagnostics.transportTime);
    writer.WriteElement("/diagnostics/mpi_ranks", diagnostics.mpiRanks);
    writer.WriteElement("/diagnostics/compton_enabled", diagnostics.comptonEnabled);
    writer.WriteElement("/diagnostics/emitted_energy", diagnostics.emittedEnergy);
    writer.WriteElement("/diagnostics/sphere_crossing_energy", totalEnergy);
    writer.WriteElement("/diagnostics/absorbed_energy", diagnostics.absorbedEnergy);
    writer.WriteElement("/diagnostics/box_escape_energy", diagnostics.boxEscapeEnergy);
    writer.WriteElement("/diagnostics/timed_out_energy", diagnostics.timedOutEnergy);
    writer.WriteElement("/diagnostics/cutoff_energy", diagnostics.cutoffEnergy);

    double residual = diagnostics.emittedEnergy - diagnostics.absorbedEnergy
                    - diagnostics.boxEscapeEnergy - diagnostics.timedOutEnergy
                    - diagnostics.cutoffEnergy;
    writer.WriteElement("/diagnostics/transport_sink_residual", residual);

    double timedOutFraction = (diagnostics.emittedEnergy > 0.0)
        ? diagnostics.timedOutEnergy / diagnostics.emittedEnergy : 0.0;
    writer.WriteElement("/diagnostics/timed_out_fraction", timedOutFraction);

    writer.WriteElement("/diagnostics/snapshot_time", diagnostics.snapshotTime);
    writer.WriteElement("/diagnostics/snapshot_cycle", diagnostics.snapshotCycle);
    writer.WriteElement("/diagnostics/n_generations", diagnostics.nGenerations);
    writer.WriteElement("/diagnostics/included_final_generations", diagnostics.includedFinalGenerations);
    writer.WriteElement("/diagnostics/discarded_burnin_generations", diagnostics.discardedBurninGenerations);
    writer.WriteElement("/diagnostics/adaptive_only_final_output", diagnostics.adaptiveOnlyFinalOutput);
    writer.WriteElement("/adaptive/config/group_quality_enabled", diagnostics.adaptiveGroupQualityEnabled);
    writer.WriteElement("/adaptive/config/group_source_cells_enabled", diagnostics.adaptiveGroupSourceCellsEnabled);
    writer.WriteElement("/adaptive/config/group_frequency_sampling_enabled", diagnostics.adaptiveGroupFrequencySamplingEnabled);
    writer.WriteElement("/adaptive/config/group_history_enabled", diagnostics.adaptiveGroupHistoryEnabled);
    writer.WriteElement("/adaptive/config/group_luminosity_normalization", diagnostics.adaptiveGroupLuminosityNormalization);
    writer.WriteElement("/adaptive/config/group_target_neff", diagnostics.adaptiveGroupTargetNeff);
    writer.WriteElement("/adaptive/config/group_target_pol_snr", diagnostics.adaptiveGroupTargetPolSnr);
    writer.WriteElement("/adaptive/config/group_deficit_max", diagnostics.adaptiveGroupDeficitMax);
    writer.WriteElement("/adaptive/config/group_min_crossings", diagnostics.adaptiveGroupMinCrossings);
    writer.WriteElement("/adaptive/config/group_min_luminosity", diagnostics.adaptiveGroupMinLuminosity);
    writer.WriteElement("/adaptive/config/group_min_luminosity_frac_of_group_max", diagnostics.adaptiveGroupMinLuminosityFracOfGroupMax);
    writer.WriteElement("/adaptive/config/group_latest_weight", diagnostics.adaptiveGroupLatestWeight);
    writer.WriteElement("/adaptive/config/group_cumulative_weight", diagnostics.adaptiveGroupCumulativeWeight);
    writer.WriteElement("/adaptive/config/group_ema_weight", diagnostics.adaptiveGroupEmaWeight);
    writer.WriteElement("/adaptive/group_sampling/enabled", diagnostics.adaptiveGroupFrequencySamplingEnabled);
    writer.WriteElement("/adaptive/group_sampling/strength", diagnostics.adaptiveGroupSamplingStrength);
    writer.WriteElement("/adaptive/group_sampling/pdf_floor", diagnostics.adaptiveGroupSamplingPdfFloor);
    writer.WriteElement("/adaptive/group_sampling/max_bias", diagnostics.adaptiveGroupSamplingMaxBias);
    writer.WriteElement("/adaptive/group_sampling/max_weight_correction", diagnostics.adaptiveGroupSamplingMaxWeightCorrection);
    writer.WriteElement("/adaptive/group_sampling/total_sampled", diagnostics.adaptiveGroupSamplingTotalSampled);
    writer.WriteElement("/adaptive/group_sampling/weight_correction_min", diagnostics.adaptiveGroupWeightCorrectionMin);
    writer.WriteElement("/adaptive/group_sampling/weight_correction_mean", diagnostics.adaptiveGroupWeightCorrectionMean);
    writer.WriteElement("/adaptive/group_sampling/weight_correction_max", diagnostics.adaptiveGroupWeightCorrectionMax);
    writer.WriteElement("/adaptive/group_sampling/weight_correction_capped_fraction", diagnostics.adaptiveGroupWeightCorrectionCappedFraction);
    writer.WriteElement("/adaptive/group_sampling/weight_correction_fallback_count", diagnostics.adaptiveGroupWeightCorrectionFallbackCount);
    writer.WriteElement("/adaptive/group_sampling/invalid_pdf_fallback_count", diagnostics.adaptiveGroupInvalidPdfFallbackCount);
    writer.WriteElement("/adaptive/group_sampling/invalid_pdf_fallback_packet_count", diagnostics.adaptiveGroupInvalidPdfFallbackPacketCount);
    writer.WriteElement("/adaptive/group_sampling/capped_energy_fraction", diagnostics.adaptiveGroupCappedEnergyFraction);
    writer.WriteElement("/adaptive/group_sampling/estimator_potentially_biased", diagnostics.adaptiveGroupEstimatorPotentiallyBiased);
    writer.WriteElement("/adaptive/source/fallback_to_integrated_path", diagnostics.adaptiveGroupFallbackToIntegratedPath);
    writer.WriteElement("/adaptive/source/fallback_reason", diagnostics.adaptiveGroupFallbackReason);
    writer.WriteElement("/adaptive/source/local_stats_after_prune", diagnostics.adaptiveGroupSourceLocalStatsAfterPrune);
    writer.WriteElement("/adaptive/source/local_stats_dropped", diagnostics.adaptiveGroupSourceLocalStatsDropped);
    writer.WriteElement("/adaptive/source/mpi_stats_exchanged", diagnostics.adaptiveGroupSourceMpiStatsExchanged);
    writer.WriteElement("/adaptive/source/mpi_packed_bytes", diagnostics.adaptiveGroupSourceMpiPackedBytes);
    if (generationStats_.samples > 0) {
        size_t const samples = generationStats_.samples;
        writer.WriteElement("/diagnostics/statistics/included_generations",
                            static_cast<double>(samples));
        writeVectorStatErrors(writer, "/tally/energy",
                              generationStats_.energy, observerEnergy_, samples);
        writeVectorStatErrors(writer, "/tally/luminosity",
                              generationStats_.luminosity, lum, samples);
        writeVectorStatErrors(writer, "/tally/isotropic_equivalent_luminosity",
                              generationStats_.isoLuminosity, isoEquiv, samples);
        writeVectorStatErrors(writer, "/tally/flux",
                              generationStats_.flux, flux, samples);
        auto lumGenStderr = vectorStderr(generationStats_.luminosity, samples);
        auto logLum = log10Luminosity(lum);
        auto logLumGenStderr = log10Stderr(lum, lumGenStderr);
        writer.WriteElement("/tally/log10_luminosity_stderr_gen", logLumGenStderr);
        writer.WriteElement("/tally/log10_luminosity_relerr_gen",
                            relativeError(logLum, logLumGenStderr));

        writeVectorPacketErrors(writer, "/tally/energy",
                                observerEnergy_, generationStats_.energyWeightSqSum, samples);
        writeVectorPacketErrors(writer, "/tally/luminosity",
                                lum, generationStats_.energyWeightSqSum, samples,
                                (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0);
        auto energyPacketStderr = packetStderr(generationStats_.energyWeightSqSum, samples);
        auto lumPacketStderr = packetStderr(generationStats_.energyWeightSqSum, samples,
                                            (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0);
        std::vector<double> isoFactor(numObservers_, 0.0);
        std::vector<double> fluxFactor(numObservers_, 0.0);
        for (size_t i = 0; i < numObservers_; ++i) {
            isoFactor[i] = (observerSolidAngle_[i] > 0.0)
                ? ((sourceDt > 0.0) ? 1.0 / sourceDt : 0.0) * fourPi / observerSolidAngle_[i] : 0.0;
            double const patchArea = observerSolidAngle_[i] * radiusSquared_;
            fluxFactor[i] = (patchArea > 0.0)
                ? ((sourceDt > 0.0) ? 1.0 / sourceDt : 0.0) / patchArea : 0.0;
        }
        auto isoPacketStderr = multiplyByObserverFactor(energyPacketStderr, isoFactor);
        auto fluxPacketStderr = multiplyByObserverFactor(energyPacketStderr, fluxFactor);
        writer.WriteElement("/tally/isotropic_equivalent_luminosity_stderr_packet", isoPacketStderr);
        writer.WriteElement("/tally/isotropic_equivalent_luminosity_relerr_packet",
                            relativeError(isoEquiv, isoPacketStderr));
        writer.WriteElement("/tally/isotropic_equivalent_luminosity_neff",
                            packetNeff(isoEquiv, isoPacketStderr));
        writer.WriteElement("/tally/flux_stderr_packet", fluxPacketStderr);
        writer.WriteElement("/tally/flux_relerr_packet", relativeError(flux, fluxPacketStderr));
        writer.WriteElement("/tally/flux_neff", packetNeff(flux, fluxPacketStderr));
        auto logLumPacketStderr = log10Stderr(lum, lumPacketStderr);
        writer.WriteElement("/tally/log10_luminosity_stderr_packet", logLumPacketStderr);
        writer.WriteElement("/tally/log10_luminosity_relerr_packet",
                            relativeError(logLum, logLumPacketStderr));
        writer.WriteElement("/tally/log10_luminosity_neff",
                            packetNeff(logLum, logLumPacketStderr));

        writeScalarStatErrors(writer, "/tally/total_energy",
                              generationStats_.totalEnergy, totalEnergy, samples);
        writeScalarStatErrors(writer, "/tally/total_luminosity",
                              generationStats_.totalLuminosity, totalLum, samples);
        writeScalarStatErrors(writer, "/diagnostics/sphere_crossing_energy",
                              generationStats_.totalEnergy, totalEnergy, samples);
        writeScalarStatErrors(writer, "/diagnostics/transport_sink_residual",
                              generationStats_.transportSinkResidual, residual, samples);
        writeScalarStatErrors(writer, "/diagnostics/timed_out_fraction",
                              generationStats_.timedOutFraction, timedOutFraction, samples);
        double const totalEnergyPacketStderr =
            std::sqrt(std::max(0.0, generationStats_.totalEnergyWeightSqSum)) /
            static_cast<double>(samples);
        double const totalLumPacketStderr = totalEnergyPacketStderr *
            ((sourceDt > 0.0) ? 1.0 / sourceDt : 0.0);
        writer.WriteElement("/tally/total_energy_stderr_packet", totalEnergyPacketStderr);
        writer.WriteElement("/tally/total_energy_relerr_packet",
                            relativeError(totalEnergy, totalEnergyPacketStderr));
        writer.WriteElement("/tally/total_energy_neff",
                            (totalEnergyPacketStderr > 0.0)
                                ? totalEnergy * totalEnergy /
                                  (totalEnergyPacketStderr * totalEnergyPacketStderr)
                                : 0.0);
        writer.WriteElement("/diagnostics/sphere_crossing_energy_stderr_packet",
                            totalEnergyPacketStderr);
        writer.WriteElement("/diagnostics/sphere_crossing_energy_relerr_packet",
                            relativeError(totalEnergy, totalEnergyPacketStderr));
        writer.WriteElement("/diagnostics/sphere_crossing_energy_neff",
                            (totalEnergyPacketStderr > 0.0)
                                ? totalEnergy * totalEnergy /
                                  (totalEnergyPacketStderr * totalEnergyPacketStderr)
                                : 0.0);
        writer.WriteElement("/tally/total_luminosity_stderr_packet", totalLumPacketStderr);
        writer.WriteElement("/tally/total_luminosity_relerr_packet",
                            relativeError(totalLum, totalLumPacketStderr));
        writer.WriteElement("/tally/total_luminosity_neff",
                            (totalLumPacketStderr > 0.0)
                                ? totalLum * totalLum /
                                  (totalLumPacketStderr * totalLumPacketStderr)
                                : 0.0);

        if (numGroups_ > 1) {
            auto gLum = getGroupLuminosity(sourceDt);
            writeMatrixStatErrors(writer, "/tally/multigroup/group_energy",
                                  generationStats_.groupEnergy, groupEnergy_, samples);
            writeMatrixStatErrors(writer, "/tally/multigroup/group_luminosity",
                                  generationStats_.groupLuminosity, gLum, samples);
            writeMatrixPacketErrors(writer, "/tally/multigroup/group_energy",
                                    groupEnergy_, generationStats_.groupEnergyWeightSqSum, samples);
            writeMatrixPacketErrors(writer, "/tally/multigroup/group_luminosity",
                                    gLum, generationStats_.groupEnergyWeightSqSum, samples,
                                    (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0);
        }

#ifdef MONTECARLO_POLARIZATION
        if (polarizationOutputEnabled_) {
            std::vector<double> qNorm(numObservers_, 0.0), uNorm(numObservers_, 0.0);
            std::vector<double> qLum(numObservers_, 0.0), uLum(numObservers_, 0.0);
            std::vector<double> polDegree(numObservers_, 0.0), polAngle(numObservers_, 0.0);
            double const invDt = (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0;
            for (size_t i = 0; i < numObservers_; ++i) {
                double const invI = (observerEnergy_[i] > 0.0) ? 1.0 / observerEnergy_[i] : 0.0;
                qNorm[i] = observerStokesQ_[i] * invI;
                uNorm[i] = observerStokesU_[i] * invI;
                qLum[i] = observerStokesQ_[i] * invDt;
                uLum[i] = observerStokesU_[i] * invDt;
                polDegree[i] = PolarizationDegree(observerEnergy_[i], observerStokesQ_[i], observerStokesU_[i]);
                polAngle[i] = PolarizationAngle(observerStokesQ_[i], observerStokesU_[i]);
            }
            writeVectorStatErrors(writer, "/tally/observer_stokes_Q",
                                  generationStats_.stokesQ, observerStokesQ_, samples);
            writeVectorStatErrors(writer, "/tally/observer_stokes_U",
                                  generationStats_.stokesU, observerStokesU_, samples);
            writeVectorStatErrors(writer, "/tally/observer_q",
                                  generationStats_.q, qNorm, samples);
            writeVectorStatErrors(writer, "/tally/observer_u",
                                  generationStats_.u, uNorm, samples);
            writeVectorStatErrors(writer, "/tally/observer_Q_luminosity",
                                  generationStats_.stokesQLuminosity, qLum, samples);
            writeVectorStatErrors(writer, "/tally/observer_U_luminosity",
                                  generationStats_.stokesULuminosity, uLum, samples);
            writeVectorStatErrors(writer, "/tally/observer_polarization_degree",
                                  generationStats_.polarizationDegree, polDegree, samples);
            writeVectorStatErrors(writer, "/tally/observer_polarization_angle",
                                  generationStats_.polarizationAngle, polAngle, samples);
            if (numGroups_ > 1) {
                std::vector<std::vector<double>> gQ(numObservers_, std::vector<double>(numGroups_, 0.0));
                std::vector<std::vector<double>> gU(numObservers_, std::vector<double>(numGroups_, 0.0));
                std::vector<std::vector<double>> gQLum = scaleMatrix(groupStokesQ_, invDt);
                std::vector<std::vector<double>> gULum = scaleMatrix(groupStokesU_, invDt);
                std::vector<std::vector<double>> gDegree(numObservers_, std::vector<double>(numGroups_, 0.0));
                std::vector<std::vector<double>> gAngle(numObservers_, std::vector<double>(numGroups_, 0.0));
                for (size_t i = 0; i < numObservers_; ++i) {
                    for (size_t g = 0; g < numGroups_; ++g) {
                        double const invI = (groupEnergy_[i][g] > 0.0) ? 1.0 / groupEnergy_[i][g] : 0.0;
                        gQ[i][g] = groupStokesQ_[i][g] * invI;
                        gU[i][g] = groupStokesU_[i][g] * invI;
                        gDegree[i][g] = PolarizationDegree(groupEnergy_[i][g], groupStokesQ_[i][g], groupStokesU_[i][g]);
                        gAngle[i][g] = PolarizationAngle(groupStokesQ_[i][g], groupStokesU_[i][g]);
                    }
                }
                writeMatrixStatErrors(writer, "/tally/multigroup/group_stokes_Q",
                                      generationStats_.groupStokesQ, groupStokesQ_, samples);
                writeMatrixStatErrors(writer, "/tally/multigroup/group_stokes_U",
                                      generationStats_.groupStokesU, groupStokesU_, samples);
                writeMatrixStatErrors(writer, "/tally/multigroup/group_q",
                                      generationStats_.groupQ, gQ, samples);
                writeMatrixStatErrors(writer, "/tally/multigroup/group_u",
                                      generationStats_.groupU, gU, samples);
                writeMatrixStatErrors(writer, "/tally/multigroup/group_Q_luminosity",
                                      generationStats_.groupQLuminosity, gQLum, samples);
                writeMatrixStatErrors(writer, "/tally/multigroup/group_U_luminosity",
                                      generationStats_.groupULuminosity, gULum, samples);
                writeMatrixStatErrors(writer, "/tally/multigroup/group_polarization_degree",
                                      generationStats_.groupPolarizationDegree, gDegree, samples);
                writeMatrixStatErrors(writer, "/tally/multigroup/group_polarization_angle",
                                      generationStats_.groupPolarizationAngle, gAngle, samples);
            }
        }
#endif

        writeScalarStatErrors(writer, "/diagnostics/emitted_energy",
                              generationStats_.emittedEnergy, diagnostics.emittedEnergy, samples);
        writeScalarStatErrors(writer, "/diagnostics/absorbed_energy",
                              generationStats_.absorbedEnergy, diagnostics.absorbedEnergy, samples);
        writeScalarStatErrors(writer, "/diagnostics/box_escape_energy",
                              generationStats_.boxEscapeEnergy, diagnostics.boxEscapeEnergy, samples);
        writeScalarStatErrors(writer, "/diagnostics/timed_out_energy",
                              generationStats_.timedOutEnergy, diagnostics.timedOutEnergy, samples);
        writeScalarStatErrors(writer, "/diagnostics/cutoff_energy",
                              generationStats_.cutoffEnergy, diagnostics.cutoffEnergy, samples);
    }
#ifdef MONTECARLO_POLARIZATION
    writer.WriteElement("/diagnostics/polarization_compiled", 1);
    writer.WriteElement("/diagnostics/polarization_enabled", polarizationOutputEnabled_ ? 1 : 0);
    writer.WriteElement("/diagnostics/polarization_basis",
                        std::string("observer_basis_1_global_z_projected_perpendicular_to_observer_direction"));
    writer.WriteElement("/diagnostics/polarization_accelerated_closure",
                        polarizationAcceleratedClosure_);
    writer.WriteElement("/diagnostics/polarization_manual_scatterings_after_acceleration",
                        polarizationManualScatteringsAfterAcceleration_);
    writer.WriteElement("/diagnostics/polarization_depolarization_scatterings",
                        polarizationDepolarizationScatterings_);
    writer.WriteElement("/diagnostics/polarization_basis_boost",
                        std::string("project_after_boost_low_velocity_approximation"));
    writer.WriteElement("/diagnostics/polarization_transport_model",
                        std::string("polarized_thomson_explicit_scalar_acceleration_closure"));
    writer.WriteElement("/diagnostics/polarization_mismatch_warning_count",
                        static_cast<double>(mismatchWarningCount_));
    writer.WriteElement("/diagnostics/polarization_uninitialized_count",
                        static_cast<double>(uninitializedPolarizationCount_));
    writer.WriteElement("/diagnostics/polarization_positive_Q_convention",
                        std::string("aligned_with_projected_reference_axis"));
#endif
}

void SphericalObserver::writeVTK(std::string const& filename, double sourceDt) const
{
    std::ofstream file(filename);
    if (!file.is_open())
        throw UniversalError("SphericalObserver::writeVTK: cannot open " + filename);

    file << std::scientific << std::setprecision(12);

    size_t N = numObservers_;
    double invDt = (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0;
    double fourPi = 4.0 * M_PI;

    file << "# vtk DataFile Version 3.0\n";
    file << "SphericalObserver luminosity map\n";
    file << "ASCII\n";
    file << "DATASET POLYDATA\n";

    file << "POINTS " << N << " double\n";
    for (size_t i = 0; i < N; ++i) {
        Vector3D pt = center_ + directions_[i] * radius_;
        file << pt.x << " " << pt.y << " " << pt.z << "\n";
    }

    std::vector<Triangle> tris = convexHullTriangulation(directions_);
    size_t expectedMinFaces = (N >= 4) ? 2 * (N - 2) : 0;
    if (tris.size() >= expectedMinFaces && expectedMinFaces > 0) {
        file << "POLYGONS " << tris.size() << " " << 4 * tris.size() << "\n";
        for (auto const& t : tris)
            file << "3 " << t.a << " " << t.b << " " << t.c << "\n";
    } else {
        file << "VERTICES " << N << " " << 2 * N << "\n";
        for (size_t i = 0; i < N; ++i)
            file << "1 " << i << "\n";
    }

    file << "POINT_DATA " << N << "\n";

    file << "SCALARS luminosity double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < N; ++i)
        file << observerEnergy_[i] * invDt << "\n";

    file << "SCALARS energy double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < N; ++i)
        file << observerEnergy_[i] << "\n";

    file << "SCALARS isotropic_equivalent_luminosity double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < N; ++i) {
        double isoEquiv = (observerSolidAngle_[i] > 0.0)
            ? observerEnergy_[i] * invDt * fourPi / observerSolidAngle_[i]
            : 0.0;
        file << isoEquiv << "\n";
    }

    file << "SCALARS flux double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < N; ++i) {
        double patchArea_i = observerSolidAngle_[i] * radiusSquared_;
        double f = (patchArea_i > 0.0) ? observerEnergy_[i] * invDt / patchArea_i : 0.0;
        file << f << "\n";
    }

    file << "SCALARS log10_luminosity double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < N; ++i) {
        double lum = observerEnergy_[i] * invDt;
        file << ((lum > 0.0) ? std::log10(lum) : -99.0) << "\n";
    }

    file << "SCALARS crossing_count double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < N; ++i)
        file << static_cast<double>(observerCrossingCount_[i]) << "\n";

    file << "SCALARS max_packet_energy double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < N; ++i)
        file << observerMaxPacketEnergy_[i] << "\n";

    file << "SCALARS solid_angle double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < N; ++i)
        file << observerSolidAngle_[i] << "\n";

    if (photosphereData_.hasMG()) {
        writeVtkScalar(file, "photosphere_integrated_radius_tau_total",
                       photosphereData_.mgIntegratedRadiusTauTotal, N);
        writeVtkScalar(file, "photosphere_integrated_radius_thermalization",
                       photosphereData_.mgIntegratedRadiusThermalization, N);
        writeVtkScalar(file, "photosphere_integrated_valid_tau_total",
                       photosphereData_.mgIntegratedValidTauTotal, N);
        writeVtkScalar(file, "photosphere_integrated_valid_thermalization",
                       photosphereData_.mgIntegratedValidThermalization, N);

        size_t groupCount = 0;
        if (!photosphereData_.mgGroupRadiusTauTotal.empty())
            groupCount = photosphereData_.mgGroupRadiusTauTotal.front().size();
        for (size_t g = 0; g < groupCount; ++g) {
            writeVtkScalar(file, "photosphere_g" + std::to_string(g) + "_radius_tau_total",
                           groupColumn(photosphereData_.mgGroupRadiusTauTotal, g), N);
            writeVtkScalar(file, "photosphere_g" + std::to_string(g) + "_radius_thermalization",
                           groupColumn(photosphereData_.mgGroupRadiusThermalization, g), N);
            writeVtkScalar(file, "photosphere_g" + std::to_string(g) + "_valid_tau_total",
                           groupColumn(photosphereData_.mgGroupValidTauTotal, g), N);
            writeVtkScalar(file, "photosphere_g" + std::to_string(g) + "_valid_thermalization",
                           groupColumn(photosphereData_.mgGroupValidThermalization, g), N);
        }
    }

    if (photosphereData_.hasGrey()) {
        writeVtkScalar(file, "photosphere_grey_radius_tau_total",
                       photosphereData_.greyRadiusTauTotal, N);
        writeVtkScalar(file, "photosphere_grey_radius_thermalization",
                       photosphereData_.greyRadiusThermalization, N);
        writeVtkScalar(file, "photosphere_grey_valid_tau_total",
                       photosphereData_.greyValidTauTotal, N);
        writeVtkScalar(file, "photosphere_grey_valid_thermalization",
                       photosphereData_.greyValidThermalization, N);
    }

#ifdef MONTECARLO_POLARIZATION
    if(polarizationOutputEnabled_)
    {
        file << "SCALARS stokes_Q double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i)
            file << observerStokesQ_[i] << "\n";

        file << "SCALARS stokes_U double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i)
            file << observerStokesU_[i] << "\n";

        file << "SCALARS polarization_degree double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i)
            file << PolarizationDegree(observerEnergy_[i], observerStokesQ_[i], observerStokesU_[i]) << "\n";

        file << "SCALARS polarization_angle double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i)
            file << PolarizationAngle(observerStokesQ_[i], observerStokesU_[i]) << "\n";

        file << "SCALARS polarization_effective_packets double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i) {
            double neff = (observerSumWeightSq_[i] > 0.0)
                ? (observerEnergy_[i] * observerEnergy_[i]) / observerSumWeightSq_[i] : 0.0;
            file << neff << "\n";
        }

        file << "SCALARS polarization_sigma_q double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i) {
            double Iv = observerEnergy_[i];
            double W2 = observerSumWeightSq_[i];
            if (Iv <= 0.0 || W2 <= 0.0) { file << 0.0 << "\n"; continue; }
            double q = observerStokesQ_[i] / Iv;
            double varQ = observerSumWQ2_[i] / Iv - q * q;
            if (varQ < 0.0) varQ = 0.0;
            double neff = Iv * Iv / W2;
            file << std::sqrt(varQ / neff) << "\n";
        }

        file << "SCALARS polarization_sigma_u double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i) {
            double Iv = observerEnergy_[i];
            double W2 = observerSumWeightSq_[i];
            if (Iv <= 0.0 || W2 <= 0.0) { file << 0.0 << "\n"; continue; }
            double u = observerStokesU_[i] / Iv;
            double varU = observerSumWU2_[i] / Iv - u * u;
            if (varU < 0.0) varU = 0.0;
            double neff = Iv * Iv / W2;
            file << std::sqrt(varU / neff) << "\n";
        }

        file << "SCALARS polarization_sigma_p double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i) {
            double Iv = observerEnergy_[i];
            double W2 = observerSumWeightSq_[i];
            if (Iv <= 0.0 || W2 <= 0.0) { file << 0.0 << "\n"; continue; }
            double q = observerStokesQ_[i] / Iv;
            double u = observerStokesU_[i] / Iv;
            double varQ = observerSumWQ2_[i] / Iv - q * q;
            double varU = observerSumWU2_[i] / Iv - u * u;
            if (varQ < 0.0) varQ = 0.0;
            if (varU < 0.0) varU = 0.0;
            double neff = Iv * Iv / W2;
            double sigQ = std::sqrt(varQ / neff);
            double sigU = std::sqrt(varU / neff);
            file << std::sqrt(sigQ * sigQ + sigU * sigU) << "\n";
        }

        file << "SCALARS polarization_snr double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i) {
            double Iv = observerEnergy_[i];
            double W2 = observerSumWeightSq_[i];
            if (Iv <= 0.0 || W2 <= 0.0) { file << 0.0 << "\n"; continue; }
            double q = observerStokesQ_[i] / Iv;
            double u = observerStokesU_[i] / Iv;
            double p = std::sqrt(q * q + u * u);
            double varQ = observerSumWQ2_[i] / Iv - q * q;
            double varU = observerSumWU2_[i] / Iv - u * u;
            if (varQ < 0.0) varQ = 0.0;
            if (varU < 0.0) varU = 0.0;
            double neff = Iv * Iv / W2;
            double sigQ = std::sqrt(varQ / neff);
            double sigU = std::sqrt(varU / neff);
            double sigP = std::sqrt(sigQ * sigQ + sigU * sigU);
            file << ((sigP > 0.0) ? p / sigP : 0.0) << "\n";
        }

        file << "SCALARS mean_direction_mismatch double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i) {
            double mean = (observerEnergy_[i] > 0.0)
                ? mismatchWeightedSum_[i] / observerEnergy_[i] : 0.0;
            file << mean << "\n";
        }

        file << "SCALARS max_direction_mismatch double 1\n";
        file << "LOOKUP_TABLE default\n";
        for (size_t i = 0; i < N; ++i)
            file << mismatchMax_[i] << "\n";
    }
#endif

    if (numGroups_ > 1) {
        for (size_t g = 0; g < numGroups_; ++g) {
            file << "SCALARS group_" << g << "_luminosity double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (size_t i = 0; i < N; ++i)
                file << groupEnergy_[i][g] * invDt << "\n";
        }

        for (size_t g = 0; g < numGroups_; ++g) {
            file << "SCALARS group_" << g << "_crossing_count double 1\n";
            file << "LOOKUP_TABLE default\n";
            for (size_t i = 0; i < N; ++i)
                file << static_cast<double>(groupCrossingCount_[i][g]) << "\n";
        }

#ifdef MONTECARLO_POLARIZATION
        if (polarizationOutputEnabled_) {
            for (size_t g = 0; g < numGroups_; ++g) {
                std::vector<double> values(N, 0.0);
                for (size_t i = 0; i < N; ++i)
                    values[i] = PolarizationDegree(groupEnergy_[i][g],
                                                   groupStokesQ_[i][g],
                                                   groupStokesU_[i][g]);
                writeScalarVtk(file, groupScalarName("polarization_degree_group_", g), values);
            }

            for (size_t g = 0; g < numGroups_; ++g) {
                std::vector<double> values(N, 0.0);
                for (size_t i = 0; i < N; ++i) {
                    double const Iv = groupEnergy_[i][g];
                    double const W2 = groupEnergyWeightSq_[i][g];
                    if (Iv <= 0.0 || W2 <= 0.0)
                        continue;
                    double const q = groupStokesQ_[i][g] / Iv;
                    double const u = groupStokesU_[i][g] / Iv;
                    double const p = std::sqrt(q * q + u * u);
                    double varQ = groupSumWQ2_[i][g] / Iv - q * q;
                    double varU = groupSumWU2_[i][g] / Iv - u * u;
                    if (varQ < 0.0) varQ = 0.0;
                    if (varU < 0.0) varU = 0.0;
                    double const neff = Iv * Iv / W2;
                    double const sigP = std::sqrt(varQ / neff + varU / neff);
                    values[i] = (sigP > 0.0) ? p / sigP : 0.0;
                }
                writeScalarVtk(file, groupScalarName("polarization_snr_group_", g), values);
            }
        }
#endif
    }

    if (generationStats_.samples > 0) {
        size_t const samples = generationStats_.samples;
        std::vector<double> lum = scaleVector(observerEnergy_, invDt);
        std::vector<double> iso(N, 0.0), flux(N, 0.0);
        for (size_t i = 0; i < N; ++i) {
            iso[i] = (observerSolidAngle_[i] > 0.0)
                ? lum[i] * fourPi / observerSolidAngle_[i] : 0.0;
            double const patchArea = observerSolidAngle_[i] * radiusSquared_;
            flux[i] = (patchArea > 0.0) ? lum[i] / patchArea : 0.0;
        }
        auto lumStderr = vectorStderr(generationStats_.luminosity, samples);
        auto energyStderr = vectorStderr(generationStats_.energy, samples);
        auto isoStderr = vectorStderr(generationStats_.isoLuminosity, samples);
        auto fluxStderr = vectorStderr(generationStats_.flux, samples);
        auto logLum = log10Luminosity(lum);
        auto logLumStderr = log10Stderr(lum, lumStderr);
        writeScalarVtk(file, "luminosity_stderr_gen", lumStderr);
        writeScalarVtk(file, "luminosity_relerr_gen", relativeError(lum, lumStderr));
        writeScalarVtk(file, "energy_stderr_gen", energyStderr);
        writeScalarVtk(file, "energy_relerr_gen", relativeError(observerEnergy_, energyStderr));
        writeScalarVtk(file, "isotropic_equivalent_luminosity_stderr_gen", isoStderr);
        writeScalarVtk(file, "isotropic_equivalent_luminosity_relerr_gen", relativeError(iso, isoStderr));
        writeScalarVtk(file, "flux_stderr_gen", fluxStderr);
        writeScalarVtk(file, "flux_relerr_gen", relativeError(flux, fluxStderr));
        writeScalarVtk(file, "log10_luminosity_stderr_gen", logLumStderr);
        writeScalarVtk(file, "log10_luminosity_relerr_gen", relativeError(logLum, logLumStderr));

        auto energyPacket = packetStderr(generationStats_.energyWeightSqSum, samples);
        auto lumPacket = packetStderr(generationStats_.energyWeightSqSum, samples, invDt);
        writeScalarVtk(file, "energy_stderr_packet", energyPacket);
        writeScalarVtk(file, "energy_relerr_packet", relativeError(observerEnergy_, energyPacket));
        writeScalarVtk(file, "energy_neff", packetNeff(observerEnergy_, energyPacket));
        writeScalarVtk(file, "luminosity_stderr_packet", lumPacket);
        writeScalarVtk(file, "luminosity_relerr_packet", relativeError(lum, lumPacket));
        writeScalarVtk(file, "luminosity_neff", packetNeff(lum, lumPacket));

        std::vector<double> isoFactor(N, 0.0), fluxFactor(N, 0.0);
        for (size_t i = 0; i < N; ++i) {
            isoFactor[i] = (observerSolidAngle_[i] > 0.0)
                ? invDt * fourPi / observerSolidAngle_[i] : 0.0;
            double const patchArea = observerSolidAngle_[i] * radiusSquared_;
            fluxFactor[i] = (patchArea > 0.0) ? invDt / patchArea : 0.0;
        }
        auto isoPacket = multiplyByObserverFactor(energyPacket, isoFactor);
        auto fluxPacket = multiplyByObserverFactor(energyPacket, fluxFactor);
        writeScalarVtk(file, "isotropic_equivalent_luminosity_stderr_packet", isoPacket);
        writeScalarVtk(file, "isotropic_equivalent_luminosity_relerr_packet", relativeError(iso, isoPacket));
        writeScalarVtk(file, "isotropic_equivalent_luminosity_neff", packetNeff(iso, isoPacket));
        writeScalarVtk(file, "flux_stderr_packet", fluxPacket);
        writeScalarVtk(file, "flux_relerr_packet", relativeError(flux, fluxPacket));
        writeScalarVtk(file, "flux_neff", packetNeff(flux, fluxPacket));
        auto logPacket = log10Stderr(lum, lumPacket);
        writeScalarVtk(file, "log10_luminosity_stderr_packet", logPacket);
        writeScalarVtk(file, "log10_luminosity_relerr_packet", relativeError(logLum, logPacket));
        writeScalarVtk(file, "log10_luminosity_neff", packetNeff(logLum, logPacket));

#ifdef MONTECARLO_POLARIZATION
        if (polarizationOutputEnabled_) {
            std::vector<double> polDegree(N, 0.0), polAngle(N, 0.0);
            for (size_t i = 0; i < N; ++i) {
                polDegree[i] = PolarizationDegree(observerEnergy_[i], observerStokesQ_[i], observerStokesU_[i]);
                polAngle[i] = PolarizationAngle(observerStokesQ_[i], observerStokesU_[i]);
            }
            auto stokesQStderr = vectorStderr(generationStats_.stokesQ, samples);
            auto stokesUStderr = vectorStderr(generationStats_.stokesU, samples);
            auto polDegreeStderr = vectorStderr(generationStats_.polarizationDegree, samples);
            auto polAngleStderr = vectorStderr(generationStats_.polarizationAngle, samples);
            writeScalarVtk(file, "stokes_Q_stderr_gen", stokesQStderr);
            writeScalarVtk(file, "stokes_Q_relerr_gen", relativeError(observerStokesQ_, stokesQStderr));
            writeScalarVtk(file, "stokes_U_stderr_gen", stokesUStderr);
            writeScalarVtk(file, "stokes_U_relerr_gen", relativeError(observerStokesU_, stokesUStderr));
            writeScalarVtk(file, "polarization_degree_stderr_gen", polDegreeStderr);
            writeScalarVtk(file, "polarization_degree_relerr_gen", relativeError(polDegree, polDegreeStderr));
            writeScalarVtk(file, "polarization_angle_stderr_gen", polAngleStderr);
            writeScalarVtk(file, "polarization_angle_relerr_gen", relativeError(polAngle, polAngleStderr));
        }
#endif

        if (numGroups_ > 1) {
            auto groupLum = getGroupLuminosity(sourceDt);
            auto groupLumStderr = matrixStderr(generationStats_.groupLuminosity, samples);
            auto groupLumPacket = packetStderr(generationStats_.groupEnergyWeightSqSum, samples, invDt);
            for (size_t g = 0; g < numGroups_; ++g) {
                std::vector<double> meanG(N), stderrG(N), relG(N), packetG(N), relPacketG(N), neffG(N);
                for (size_t i = 0; i < N; ++i) {
                    meanG[i] = groupLum[i][g];
                    stderrG[i] = groupLumStderr[i][g];
                    packetG[i] = groupLumPacket[i][g];
                }
                relG = relativeError(meanG, stderrG);
                relPacketG = relativeError(meanG, packetG);
                neffG = packetNeff(meanG, packetG);
                writeScalarVtk(file, "group_" + std::to_string(g) + "_luminosity_stderr_gen", stderrG);
                writeScalarVtk(file, "group_" + std::to_string(g) + "_luminosity_relerr_gen", relG);
                writeScalarVtk(file, "group_" + std::to_string(g) + "_luminosity_stderr_packet", packetG);
                writeScalarVtk(file, "group_" + std::to_string(g) + "_luminosity_relerr_packet", relPacketG);
                writeScalarVtk(file, "group_" + std::to_string(g) + "_luminosity_neff", neffG);
            }
        }
    }
}

void SphericalObserver::writeTXT(std::string const& filename, double sourceDt) const
{
    std::ofstream file(filename);
    if (!file.is_open())
        throw UniversalError("SphericalObserver::writeTXT: cannot open " + filename);

    file << std::scientific << std::setprecision(8);

    file << "# obs_index theta_obs phi_obs solid_angle N_packets energy luminosity";
#ifdef MONTECARLO_POLARIZATION
    if (polarizationOutputEnabled_)
        file << " I Q U q u P polarization_angle_rad"
             << " mean_k_minus_r_angle_rad rms_k_minus_r_angle_rad max_k_minus_r_angle_rad";
#endif
    file << "\n";

    double const invDt = (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0;
    for (size_t i = 0; i < numObservers_; ++i) {
        double const theta = std::acos(std::clamp(directions_[i].z, -1.0, 1.0));
        double const phi = std::atan2(directions_[i].y, directions_[i].x);
        double const lum = observerEnergy_[i] * invDt;

        file << i << " " << theta << " " << phi << " "
             << observerSolidAngle_[i] << " "
             << observerCrossingCount_[i] << " "
             << observerEnergy_[i] << " " << lum;

#ifdef MONTECARLO_POLARIZATION
        if (polarizationOutputEnabled_) {
            double const Iv = observerEnergy_[i];
            double const invI = (Iv > 0.0) ? 1.0 / Iv : 0.0;
            double const q = observerStokesQ_[i] * invI;
            double const u = observerStokesU_[i] * invI;
            double const P = std::sqrt(q * q + u * u);
            double const polAngle = 0.5 * std::atan2(observerStokesU_[i], observerStokesQ_[i]);

            double meanMismatch = 0.0, rmsMismatch = 0.0;
            if (Iv > 0.0) {
                meanMismatch = mismatchWeightedSum_[i] / Iv;
                double const var = mismatchWeighted2Sum_[i] / Iv
                                 - meanMismatch * meanMismatch;
                rmsMismatch = std::sqrt(std::max(0.0, var));
            }

            file << " " << Iv << " " << observerStokesQ_[i]
                 << " " << observerStokesU_[i]
                 << " " << q << " " << u << " " << P << " " << polAngle
                 << " " << meanMismatch << " " << rmsMismatch << " " << mismatchMax_[i];
        }
#endif
        file << "\n";
    }
}
