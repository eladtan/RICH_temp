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
#include <stdexcept>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace {

struct Triangle { size_t a, b, c; };

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

std::vector<double> computePerObserverSolidAngles(
    const std::vector<Vector3D>& dirs, size_t N)
{
    std::vector<double> solidAngles(N, 0.0);
    double uniform = 4.0 * M_PI / static_cast<double>(N);

    if (N < 4) {
        for (auto& s : solidAngles) s = uniform;
        return solidAngles;
    }

    std::vector<Triangle> tris = convexHullTriangulation(dirs);
    size_t expectedMinFaces = 2 * (N - 2);
    if (tris.size() < expectedMinFaces) {
        for (auto& s : solidAngles) s = uniform;
        return solidAngles;
    }

    for (auto const& tri : tris) {
        Vector3D cross = CrossProduct(dirs[tri.b] - dirs[tri.a],
                                      dirs[tri.c] - dirs[tri.a]);
        double area = 0.5 * abs(cross);
        solidAngles[tri.a] += area / 3.0;
        solidAngles[tri.b] += area / 3.0;
        solidAngles[tri.c] += area / 3.0;
    }

    for (size_t i = 0; i < N; ++i)
        if (solidAngles[i] <= 0.0) solidAngles[i] = uniform;

    return solidAngles;
}

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
    observerCrossingCount_.assign(numObservers_, 0);
    observerSolidAngle_ = computePerObserverSolidAngles(directions_, numObservers_);
    groupEnergy_.assign(numObservers_, std::vector<double>(numGroups_, 0.0));
    groupCrossingCount_.assign(numObservers_, std::vector<size_t>(numGroups_, 0));
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
    if (weight == 0.0 || !std::isfinite(weight))
        return;
    size_t obs = findNearestObserver(crossingPoint);
    observerEnergy_[obs] += weight;
    ++observerCrossingCount_[obs];
    if (numGroups_ > 1) {
        size_t g = findGroup(frequency);
        groupEnergy_[obs][g] += weight;
        ++groupCrossingCount_[obs][g];
    }
}

void SphericalObserver::addEmittedEnergy(double energy) { emittedEnergy_ += energy; }
void SphericalObserver::addAbsorbedEnergy(double energy) { absorbedEnergy_ += energy; }
void SphericalObserver::addBoxEscapeEnergy(double energy) { boxEscapeEnergy_ += energy; }
void SphericalObserver::addTimedOutEnergy(double energy) { timedOutEnergy_ += energy; }
void SphericalObserver::addCutoffEnergy(double energy) { cutoffEnergy_ += energy; }

size_t SphericalObserver::findNearestObserver(Vector3D const& crossingPoint) const
{
    Vector3D u = crossingPoint - center_;
    double norm = abs(u);
    if (norm <= 0.0)
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

void SphericalObserver::scale(double factor)
{
    for (auto& e : observerEnergy_) e *= factor;
    for (auto& gv : groupEnergy_)
        for (auto& e : gv) e *= factor;
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

    if (numGroups_ > 1) {
        writer.WriteElement("/tally/multigroup/group_boundaries", groupBoundaries_);

        std::vector<std::vector<double>> gLum = getGroupLuminosity(sourceDt);
        writer.WriteElement("/tally/multigroup/group_energy", groupEnergy_);
        writer.WriteElement("/tally/multigroup/group_luminosity", gLum);
        writer.WriteElement("/tally/multigroup/group_crossing_count", groupCrossingCount_);
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

    file << "SCALARS solid_angle double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < N; ++i)
        file << observerSolidAngle_[i] << "\n";

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
    }
}
