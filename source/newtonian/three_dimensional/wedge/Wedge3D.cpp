#include "Wedge3D.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include "newtonian/three_dimensional/flux_calculator_3d.hpp"
#include "misc/universal_error.hpp"
#include "misc/utils.hpp"
#ifdef RICH_MPI
#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#endif

namespace {
    // Orders the vertices so that (v1-v0)x(v2-v0) points towards Interior (MadVoro convention)
    Face OrientedFace(std::vector<Vector3D> const& Vertices, Vector3D const& Interior) {
        Face Res;
        Res.vertices.assign(Vertices.begin(), Vertices.end());
        Vector3D const Normal = CrossProduct(Vertices[1] - Vertices[0], Vertices[2] - Vertices[0]);
        if (ScalarProd(Normal, Interior - Vertices[0]) < 0)
            std::reverse(Res.vertices.begin(), Res.vertices.end());
        return Res;
    }

    // The neighbor of a face that is not a local cell (a box ghost or an MPI ghost), or the second neighbor
    size_t OutsideNeighbor(Tessellation3D const& Tess, size_t FaceIndex) {
        std::pair<size_t, size_t> const& Neighbors = Tess.GetFaceNeighbors(FaceIndex);
        return Neighbors.first >= Tess.GetPointNo() ? Neighbors.first : Neighbors.second;
    }
}

WedgeGeometry3D::WedgeGeometry3D(double XMin, double XMax, double YMax, double TanAlpha)
    : XMin_(XMin), XMax_(XMax), YMax_(YMax), TanAlpha_(TanAlpha), CosAlpha_(1.0 / std::sqrt(1.0 + TanAlpha * TanAlpha)) {
    if (!(XMax > XMin) || !(YMax > 0) || !(TanAlpha > 0) || TanAlpha > 1)
        throw UniversalError("WedgeGeometry3D: need XMax > XMin, YMax > 0 and 0 < tan(alpha) <= 1");
}

std::vector<Face> WedgeGeometry3D::BuildBoxFaces(void) const {
    double const ZMax = YMax_ * TanAlpha_;
    Vector3D const A0(XMin_, 0, 0), A1(XMax_, 0, 0);
    Vector3D const B0Up(XMin_, YMax_, ZMax), B0Down(XMin_, YMax_, -ZMax);
    Vector3D const B1Up(XMax_, YMax_, ZMax), B1Down(XMax_, YMax_, -ZMax);
    Vector3D const Interior(0.5 * (XMin_ + XMax_), 0.5 * YMax_, 0);
    std::vector<Face> Res;
    Res.push_back(OrientedFace({A0, B0Down, B0Up}, Interior));
    Res.push_back(OrientedFace({A1, B1Up, B1Down}, Interior));
    Res.push_back(OrientedFace({B0Up, B0Down, B1Down, B1Up}, Interior));
    // Faces 3 and 4 are the walls (WallFaceIndices)
    Res.push_back(OrientedFace({A0, B0Up, B1Up, A1}, Interior));
    Res.push_back(OrientedFace({A0, A1, B1Down, B0Down}, Interior));
    // Check the orientation with the same predicate MadVoro uses for inside/outside tests
    double const Eps = 1e-6 * std::min(XMax_ - XMin_, YMax_);
    std::vector<Vector3D> Inside = {Interior, Vector3D(XMin_ + Eps, 0.5 * YMax_, 0), Vector3D(XMax_ - Eps, 0.5 * YMax_, 0),
        Vector3D(Interior.x, YMax_ - Eps, 0), Vector3D(Interior.x, 0.5 * YMax_, 0.5 * YMax_ * TanAlpha_ - Eps)};
    std::vector<Vector3D> Outside = {Vector3D(XMin_ - Eps, 0.5 * YMax_, 0), Vector3D(XMax_ + Eps, 0.5 * YMax_, 0),
        Vector3D(Interior.x, YMax_ + Eps, 0), Vector3D(Interior.x, 0.5 * YMax_, 0.5 * YMax_ * TanAlpha_ + Eps),
        Vector3D(Interior.x, 0.5 * YMax_, -0.5 * YMax_ * TanAlpha_ - Eps), Vector3D(Interior.x, -Eps, 0)};
    for (Vector3D const& Point : Inside)
        if (!PointInPoly(Res, Point))
            throw UniversalError("WedgeGeometry3D: interior test point classified as outside, bad face orientation");
    for (Vector3D const& Point : Outside)
        if (PointInPoly(Res, Point))
            throw UniversalError("WedgeGeometry3D: exterior test point classified as inside, bad face orientation");
    return Res;
}

Vector3D WedgeGeometry3D::GetLowerLeft(void) const {
    return Vector3D(XMin_, 0, -YMax_ * TanAlpha_);
}

Vector3D WedgeGeometry3D::GetUpperRight(void) const {
    return Vector3D(XMax_, YMax_, YMax_ * TanAlpha_);
}

bool WedgeGeometry3D::IsWallMirror(Vector3D const& Point) const {
    // A wall mirror of (x, y, 0) is (x, y cos(2a), +-y sin(2a)), so |z| = y tan(2a) > 2 y tan(a).
    // Mirrors across x = XMin, x = XMax and y = YMax, and MPI ghosts, stay on z = 0.
    return std::abs(Point.z) > 1.5 * TanAlpha_ * std::abs(Point.y) && Point.z != 0;
}

bool WedgeGeometry3D::IsWallFace(Tessellation3D const& Tess, size_t FaceIndex) const {
    size_t const Outside = OutsideNeighbor(Tess, FaceIndex);
    return Outside >= Tess.GetPointNo() && IsWallMirror(Tess.GetMeshPoint(Outside));
}

double WedgeGeometry3D::InPlaneArea(Tessellation3D const& Tess, size_t Cell, double* MaxEdge) const {
    double WallArea = 0;
    size_t WallFace = std::numeric_limits<size_t>::max();
    for (size_t const FaceIndex : Tess.GetCellFaces(Cell)) {
        if (IsWallFace(Tess, FaceIndex) && Tess.GetArea(FaceIndex) > WallArea) {
            WallArea = Tess.GetArea(FaceIndex);
            WallFace = FaceIndex;
        }
    }
    if (WallArea <= 0) {
        // Every cell spans the full wedge thickness, so this is only reached if a wall face was lost
        if (MaxEdge != nullptr)
            *MaxEdge = 0;
        return std::max(Tess.GetVolume(Cell), 0.0) / (2 * std::max(Tess.GetCellCM(Cell).y, std::numeric_limits<double>::min()) * TanAlpha_);
    }
    if (MaxEdge != nullptr) {
        // The wall face projects onto z = 0 as the cell's 2D polygon; its in-plane edges are the 2D edges
        *MaxEdge = 0;
        point_vec const& Vertices = Tess.GetPointsInFace(WallFace);
        std::vector<Vector3D> const& Points = Tess.GetFacePoints();
        for (size_t k = 0; k < Vertices.size(); ++k) {
            Vector3D const& P0 = Points[Vertices[k]];
            Vector3D const& P1 = Points[Vertices[(k + 1) % Vertices.size()]];
            *MaxEdge = std::max(*MaxEdge, std::sqrt((P1.x - P0.x) * (P1.x - P0.x) + (P1.y - P0.y) * (P1.y - P0.y)));
        }
    }
    return WallArea * CosAlpha_;
}

double WedgeGeometry3D::InPlaneWidth(Tessellation3D const& Tess, size_t Cell) const {
    double MaxEdge = 0;
    double const Area = InPlaneArea(Tess, Cell, &MaxEdge);
    double const Round = std::sqrt(Area / M_PI);
    return MaxEdge > 0 ? std::min(Round, Area / MaxEdge) : Round;
}

double WedgeGeometry3D::ApproximateInPlaneWidth(Tessellation3D const& Tess, size_t Index) const {
    double const Y = std::max(Tess.GetCellCM(Index).y, std::numeric_limits<double>::min());
    return std::sqrt(std::max(Tess.GetVolume(Index), 0.0) / (2 * Y * TanAlpha_ * M_PI));
}

double WedgeGeometry3D::FullRevolutionFactor(void) const {
    return M_PI / TanAlpha_;
}

WedgeVoronoi3D::WedgeVoronoi3D(WedgeGeometry3D const& Geometry, bool InPlaneGhostSearch)
    : WedgeVoronoi3D(Geometry, InPlaneGhostSearch ? WedgeMesh::InPlaneSearch : WedgeMesh::Generic) {}

WedgeVoronoi3D::WedgeVoronoi3D(WedgeGeometry3D const& Geometry, WedgeMesh Mesh)
    : Voronoi3D(Geometry.GetLowerLeft(), Geometry.GetUpperRight()), Geometry_(Geometry) {
    // Constructing from the bounding box keeps MadVoro's MPI Hilbert manager on a non-degenerate box; the wedge
    // faces replace the six box faces and are pushed to the engine at the next build.
    ModifyBoxFaces() = Geometry.BuildBoxFaces();
    if (Mesh != WedgeMesh::Generic) {
        // Every generator is on z = 0 and is bounded by exactly its own two wall mirrors and its in-plane neighbors
        engine().SetGeneratorPlane(Vector3D(0, 0, 1), 0);
        engine().SetAlwaysMirroredFaces(Geometry.WallFaceIndices());
    }
    engine().SetPlaneFaces(Mesh == WedgeMesh::PlaneFaces);
}

double WedgeVoronoi3D::GetWidth(size_t Index) const {
    double const Width = Geometry_.ApproximateInPlaneWidth(*this, Index);
    return Width > 0 ? Width : Voronoi3D::GetWidth(Index);
}

std::pair<bool, bool> IsWedgeWallFace3D::operator()(size_t FaceIndex, const Tessellation3D& Tess, const vector<ComputationalCell3D>& /*Cells*/) const {
    if (!Geometry_.IsWallFace(Tess, FaceIndex))
        return std::pair<bool, bool>(false, false);
    // Same aux convention as IsBoundaryFace3D: true when the real cell is the first neighbor
    return std::pair<bool, bool>(true, Tess.GetFaceNeighbors(FaceIndex).first < Tess.GetPointNo());
}

void WedgeWallFlux3D::operator()(size_t FaceIndex, const Tessellation3D& Tess, const Vector3D& FaceVelocity, const vector<ComputationalCell3D>& /*Cells*/,
    const EquationOfState& Eos, const bool Aux, Conserved3D& Res, double /*Time*/, std::pair<ComputationalCell3D, ComputationalCell3D> const& FaceValues) const {
    Vector3D const Normal = normalize(Tess.Normal(FaceIndex));
    std::pair<ComputationalCell3D, ComputationalCell3D> States(FaceValues);
    ComputationalCell3D& Inner = Aux ? States.first : States.second;
    Inner.velocity -= ScalarProd(Inner.velocity, Normal) * Normal;
    if (Aux)
        States.second = States.first;
    else
        States.first = States.second;
    RotateSolveBack3D(Normal, States.first, States.second, FaceVelocity, Rs_, Res, Eos);
}

size_t WedgeGhostCriteria3D::GhostChoose(Tessellation3D const& Tess, size_t Index) const {
    return Geometry_.IsWallMirror(Tess.GetMeshPoint(Index)) ? 0 : 1;
}

void WedgePointMotion3D::operator()(const Tessellation3D& Tess, const vector<ComputationalCell3D>& Cells, double Time, vector<Vector3D>& Res) const {
    Base_(Tess, Cells, Time, Res);
    for (Vector3D& Velocity : Res)
        Velocity.z = 0;
}

void WedgePointMotion3D::ApplyFix(Tessellation3D const& Tess, vector<ComputationalCell3D> const& Cells, double Time, double Dt, vector<Vector3D>& Velocities) const {
    Base_.ApplyFix(Tess, Cells, Time, Dt, Velocities);
    for (Vector3D& Velocity : Velocities)
        Velocity.z = 0;
    size_t const N = Tess.GetPointNo();
    // The base fix returns local velocities only (RoundCells3D resizes to N); MPI ghosts need theirs for pairs that
    // straddle a rank boundary
    std::vector<Vector3D> Original(Velocities);
    Original.resize(N);
#ifdef RICH_MPI
    MPI_exchange_data(Tess, Original, true);
#endif
    std::vector<size_t> Neighbors;
    std::pair<Vector3D, Vector3D> const Box = Tess.GetBoxCoordinates();
    for (size_t i = 0; i < N; ++i) {
        Vector3D const& Point = Tess.GetMeshPoint(i);
        // In-plane slow-down against the nearest in-plane neighbor (box ghosts excluded)
        Tess.GetNeighbors(i, Neighbors);
        double Nearest = std::numeric_limits<double>::max();
        size_t NearestIndex = i;
        for (size_t const Neighbor : Neighbors) {
            if (Neighbor >= N && Tess.IsPointOutsideBox(Neighbor))
                continue;
            double const Distance = fastabs(Point - Tess.GetMeshPoint(Neighbor));
            if (Distance < Nearest) {
                Nearest = Distance;
                NearestIndex = Neighbor;
            }
        }
        if (NearestIndex != i && NearestIndex < Original.size() && Nearest > 0
            && Nearest < 0.4 * std::min(Tess.GetWidth(i), Tess.GetWidth(NearestIndex))) {
            Vector3D Direction = Point - Tess.GetMeshPoint(NearestIndex);
            Direction.z = 0;
            Direction *= 1.0 / fastabs(Direction);
            double const Approach = ScalarProd(Original[i] - Original[NearestIndex], Direction);
            if (Approach < 0)
                Velocities[i] -= 0.5 * Approach * Direction;
        }
        // Axis guard. Dt is the first CFL pass; hydro recomputes dt after this fix, so keep a factor 4 margin.
        if (Dt > 0 && Velocities[i].y * Dt < -0.25 * Point.y)
            Velocities[i].y = -0.25 * Point.y / Dt;
        // Outer-face guard (x = XMin, x = XMax, y = YMax). RoundCells3D only keeps a generator from covering more than 30%
        // of the way to its box mirror per step, so in a steady outflow it creeps onto the face geometrically: in the smoke
        // run one reached 1e-9 of x = XMax, and the near-coincident mirror pair left a 6e-11 face between another cell and
        // a wall mirror. Keep generators 0.2 in-plane widths inside, covering at most a quarter of the remaining room per
        // first-pass dt (same margin as the axis guard); the gas keeps leaving through the outflow flux.
        if (Dt > 0) {
            double const Keep = 0.2 * Tess.GetWidth(i);
            double const RoomXMax = std::max(0.0, Box.second.x - Point.x - Keep), RoomXMin = std::max(0.0, Point.x - Box.first.x - Keep);
            double const RoomYMax = std::max(0.0, Box.second.y - Point.y - Keep);
            if (Velocities[i].x * Dt > 0.25 * RoomXMax)
                Velocities[i].x = 0.25 * RoomXMax / Dt;
            if (-Velocities[i].x * Dt > 0.25 * RoomXMin)
                Velocities[i].x = -0.25 * RoomXMin / Dt;
            if (Velocities[i].y * Dt > 0.25 * RoomYMax)
                Velocities[i].y = 0.25 * RoomYMax / Dt;
        }
    }
}

void WedgeCellUpdater3D::operator()(vector<ComputationalCell3D>& Res, EquationOfState const& Eos, const Tessellation3D& Tess, vector<Conserved3D>& Extensives) const {
    size_t const N = Tess.GetPointNo();
    for (size_t i = 0; i < N; ++i) {
        Conserved3D& Extensive = Extensives[i];
        if (Extensive.mass > 0)
            Extensive.internal_energy += 0.5 * Extensive.momentum.z * Extensive.momentum.z / Extensive.mass;
        Extensive.momentum.z = 0;
    }
    Base_(Res, Eos, Tess, Extensives);
    for (size_t i = 0; i < N; ++i)
        Res[i].velocity.z = 0;
}

WedgeCFL3D::WedgeCFL3D(WedgeGeometry3D const& Geometry, double Cfl, double SourceCfl, SourceTerm3D const& Source)
    : Geometry_(Geometry), Cfl_(Cfl), SourceCfl_(SourceCfl), Source_(Source), FirstTry_(true), Requested_(-1), Dt_(-1), SuggestedDt_(-1),
      LastTime_(-std::numeric_limits<double>::max()) {
    if (!(Cfl < 1))
        throw UniversalError("WedgeCFL3D: cfl number must be smaller than 1");
}

double WedgeCFL3D::operator()(const Tessellation3D& Tess, const vector<ComputationalCell3D>& Cells, const EquationOfState& Eos,
    const vector<Vector3D>& FaceVelocities, const double Time) {
    double Res = 0.001 * std::numeric_limits<double>::max();
    size_t const N = Tess.GetPointNo();
    for (size_t i = 0; i < N; ++i) {
        ComputationalCell3D const& Cell = Cells[i];
        double const SoundSpeed = Eos.de2c(Cell.density, Cell.internal_energy, Cell.tracers, ComputationalCell3D::tracerNames);
        double SignalSpeed = SoundSpeed;
        for (size_t const FaceIndex : Tess.GetCellFaces(i)) {
            if (Geometry_.IsWallFace(Tess, FaceIndex))
                continue;
            Vector3D const Normal = normalize(Tess.Normal(FaceIndex));
            SignalSpeed = std::max(SignalSpeed, SoundSpeed + std::abs(ScalarProd(Normal, Cell.velocity - FaceVelocities[FaceIndex])));
        }
        Res = std::min(Res, Geometry_.InPlaneWidth(Tess, i) / SignalSpeed);
    }
    Res *= Cfl_;
    Res = 1.0 / std::max(Source_.SuggestInverseTimeStep() / SourceCfl_, 1.0 / Res);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &Res, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
    SuggestedDt_ = Res;
    // The requested step caps the first call and the repeated call at the same time, then is cleared
    if (Requested_ > 0 && (FirstTry_ || LastTime_ == Time)) {
        Res = std::min(Res, Requested_);
        FirstTry_ = false;
        if (LastTime_ == Time)
            Requested_ = -1;
    }
    LastTime_ = Time;
    Dt_ = Res;
    return Res;
}

void WedgeCFL3D::SetTimeStep(double Dt) {
    Requested_ = Dt;
    FirstTry_ = true;
}

double WedgeCFL3D::GetTimeStep(void) const {
    return FirstTry_ ? Requested_ : Dt_;
}

double WedgeCFL3D::SuggestTimeStep(void) const {
    return SuggestedDt_;
}

void DiffusionWedgeBoundary::SetBoundaryValues(Tessellation3D const& Tess, size_t const Index, size_t const OutsidePoint, double const Dt,
    std::vector<ComputationalCell3D> const& Cells, double const Area, double& A, double& B, size_t const FaceIndex) const {
    if (Geometry_.IsWallMirror(Tess.GetMeshPoint(OutsidePoint)))
        Closed_.SetBoundaryValues(Tess, Index, OutsidePoint, Dt, Cells, Area, A, B, FaceIndex);
    else
        Open_.SetBoundaryValues(Tess, Index, OutsidePoint, Dt, Cells, Area, A, B, FaceIndex);
}

void DiffusionWedgeBoundary::GetOutSideValues(Tessellation3D const& Tess, std::vector<ComputationalCell3D> const& Cells, size_t const Index,
    size_t const OutsidePoint, std::vector<double> const& NewE, double& EOutside, Vector3D& VOutside) const {
    if (Geometry_.IsWallMirror(Tess.GetMeshPoint(OutsidePoint)))
        Closed_.GetOutSideValues(Tess, Cells, Index, OutsidePoint, NewE, EOutside, VOutside);
    else
        Open_.GetOutSideValues(Tess, Cells, Index, OutsidePoint, NewE, EOutside, VOutside);
}

void DiffusionWedgeBoundary::SetMomentumTermBoundary(Tessellation3D const& Tess, size_t const Index, size_t const OutsidePoint, double const Dt,
    ComputationalCell3D const& Cell, double const Area, double& A, double& B, size_t const FaceIndex, double const FleckFactor,
    double const FluxLimiter, double const D, double const SigmaPlanck) const {
    if (!Geometry_.IsWallMirror(Tess.GetMeshPoint(OutsidePoint))) {
        Open_.SetMomentumTermBoundary(Tess, Index, OutsidePoint, Dt, Cell, Area, A, B, FaceIndex, FleckFactor, FluxLimiter, D, SigmaPlanck);
        return;
    }
    // Interior term of Diffusion::BuildMatrix (diagonal and neighbor entries) with the neighbor being the cell itself.
    // Dt, Area and Cell are already in cgs here.
    Vector3D const Direction = normalize(Tess.GetMeshPoint(Index) - Tess.GetMeshPoint(OutsidePoint));
    double const VelocityRatio = std::min(1.0, 0.05 * CG::speed_of_light / (fastabs(Cell.velocity) + 1e-2));
    double const Term = -0.5 * Dt * FluxLimiter * Area * (VelocityRatio * FleckFactor * 2 * 3 * SigmaPlanck * D / CG::speed_of_light - 1)
        * ScalarProd(Cell.velocity, Direction) / 3;
    A += 2 * Term;
}
