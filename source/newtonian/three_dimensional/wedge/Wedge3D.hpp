/*! \file Wedge3D.hpp
  \brief Thin 3D wedge that reproduces 2D axisymmetric (cylindrical) flow with the 3D code.

  The symmetry axis is the x axis and y >= 0 is the cylindrical radius. All mesh generating points stay
  on the plane z = 0 and the domain is closed by the two planes z = +-y*tan(alpha), which meet on the
  axis. Every Voronoi cell is then the 2D (x,y) Voronoi polygon extruded between the two planes, so its
  volume and the area of its side faces are proportional to y, exactly as in a 2D cylindrical code.
  The pressure on the two tilted walls supplies the p/y hoop term, and the Gauss sums of the radiation
  solver pick up the same term through the wall faces.

  Wall faces are reflecting. The outer faces x = XMin, x = XMax and y = YMax are outflow.
  Masses and volumes in the wedge are full-revolution values divided by FullRevolutionFactor().
  A face is a wall face when its outside neighbor is the mirror image of a generator across a wall; every
  class here uses that one test, so hydro, CFL, ghosts and radiation always agree on which faces are walls.
*/
#ifndef WEDGE3D_HPP
#define WEDGE3D_HPP 1

#include <vector>
#include "3D/tessellation/Voronoi3D.hpp"
#include "newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "newtonian/three_dimensional/Ghost3D.hpp"
#include "newtonian/three_dimensional/point_motion_3d.hpp"
#include "newtonian/three_dimensional/cell_updater_3d.hpp"
#include "newtonian/three_dimensional/time_step_function3D.hpp"
#include "newtonian/three_dimensional/SourceTerm3D.hpp"
#include "Radiation/Diffusion.hpp"

//! \brief Geometry of the wedge: XMin <= x <= XMax, 0 <= y <= YMax, |z| <= y*TanAlpha
class WedgeGeometry3D {
public:
    WedgeGeometry3D(double XMin, double XMax, double YMax, double TanAlpha);

    /*! \brief Box faces in the MadVoro convention: (v1-v0)x(v2-v0) points into the domain.
      Throws if the orientation check against MadVoro's PointInPoly fails.
    */
    std::vector<Face> BuildBoxFaces(void) const;

    //! \brief Indices of the two walls (z = +-y tan(alpha)) among BuildBoxFaces()
    std::vector<size_t> WallFaceIndices(void) const { return {3, 4}; }

    //! \brief Lower corner of the bounding box (use it to construct Voronoi3D, then replace the faces)
    Vector3D GetLowerLeft(void) const;

    //! \brief Upper corner of the bounding box
    Vector3D GetUpperRight(void) const;

    //! \brief True for a ghost point that is the mirror image of a generator across one of the two walls
    bool IsWallMirror(Vector3D const& Point) const;

    //! \brief True for a face whose outside neighbor is a wall mirror
    bool IsWallFace(Tessellation3D const& Tess, size_t FaceIndex) const;

    //! \brief Meridional (x,y) area of a local cell, from its wall face; MaxEdge is its longest in-plane edge
    double InPlaneArea(Tessellation3D const& Tess, size_t Cell, double* MaxEdge = nullptr) const;

    //! \brief Meridional length scale of a local cell: min(sqrt(A/pi), A/longest edge), the 2D analog of the 3D CFL width
    double InPlaneWidth(Tessellation3D const& Tess, size_t Cell) const;

    //! \brief sqrt(A/pi) with A = V/(2 tan(alpha) y_cm); needs only volume and centroid, so it also works for MPI ghosts
    double ApproximateInPlaneWidth(Tessellation3D const& Tess, size_t Index) const;

    //! \brief Full-revolution value / wedge value for volumes, masses and energies (pi/tan(alpha))
    double FullRevolutionFactor(void) const;

    double GetXMin(void) const { return XMin_; }
    double GetXMax(void) const { return XMax_; }
    double GetYMax(void) const { return YMax_; }
    double GetTanAlpha(void) const { return TanAlpha_; }

private:
    double XMin_, XMax_, YMax_, TanAlpha_, CosAlpha_;
};

/*! \brief Voronoi3D on the wedge. Construct it, then build. GetWidth returns the 2D (meridional) width for local cells
  and MPI ghosts (box ghosts have no volume and return 0; every caller in RICH skips them), so the
  heuristics that scale with the cell size (RoundCells3D, the LinearGauss3D shock detector, the cell updater's
  entropy switch, the radiation flux-limiter floor, the gravity time step) see the 2D cell size instead of the
  3D one, which in the wedge differs from it by (1.5 tan(alpha) y / r)^(1/3).
  With InPlaneGhostSearch (default) MadVoro mirrors every local generator across both walls up front and looks for MPI
  ghosts with the circles where the tetrahedra's circumspheres meet z = 0. Without it the search spheres have radius
  ~ y tan(alpha) and the ghost search grows with the wedge height (34 s per step for 2D_h1 at tan(alpha) = 0.05).
  WedgeMesh::PlaneFaces also builds the faces from each cell's in-plane polygon extended between the walls, which is
  the exact cell; the faces from the tetrahedra have slivers towards other cells' mirrors and an error that grows as
  tan(alpha) shrinks.
*/
enum class WedgeMesh {
    Generic,        //!< MadVoro's generic 3D build
    InPlaneSearch,  //!< walls always mirrored, ghost search with the circles on z = 0
    PlaneFaces      //!< InPlaneSearch, with the faces from the in-plane polygons
};

class WedgeVoronoi3D : public Voronoi3D {
public:
    explicit WedgeVoronoi3D(WedgeGeometry3D const& Geometry, bool InPlaneGhostSearch = true);

    WedgeVoronoi3D(WedgeGeometry3D const& Geometry, WedgeMesh Mesh);

    double GetWidth(size_t Index) const override;

    Tessellation3D* clone(void) const override { return new WedgeVoronoi3D(*this); }

private:
    WedgeGeometry3D const& Geometry_;
};

//! \brief Flux condition: a boundary face on one of the two walls. Put it before IsBoundaryFace3D.
class IsWedgeWallFace3D : public ConditionActionFlux1::Condition3D {
public:
    explicit IsWedgeWallFace3D(WedgeGeometry3D const& Geometry) : Geometry_(Geometry) {}

    std::pair<bool, bool> operator()(size_t FaceIndex, const Tessellation3D& Tess, const vector<ComputationalCell3D>& Cells) const override;

private:
    WedgeGeometry3D const& Geometry_;
};

/*! \brief Wall flux for the wedge: the face state is made tangent to the wall before the rigid reflection,
  so the Riemann solver returns exactly the face pressure and no mass or energy flux.
*/
class WedgeWallFlux3D : public ConditionActionFlux1::Action3D {
public:
    explicit WedgeWallFlux3D(RiemannSolver3D const& Rs) : Rs_(Rs) {}

    void operator()(size_t FaceIndex, const Tessellation3D& Tess, const Vector3D& FaceVelocity, const vector<ComputationalCell3D>& Cells,
        const EquationOfState& Eos, const bool Aux, Conserved3D& Res, double Time,
        std::pair<ComputationalCell3D, ComputationalCell3D> const& FaceValues) const override;

private:
    RiemannSolver3D const& Rs_;
};

//! \brief Ghost selector for SeveralGhostGenerator3D: index 0 for wall mirrors, index 1 for the outer faces
class WedgeGhostCriteria3D : public SeveralGhostGenerator3D::GhostCriteria3D {
public:
    explicit WedgeGhostCriteria3D(WedgeGeometry3D const& Geometry) : Geometry_(Geometry) {}

    size_t GhostChoose(Tessellation3D const& Tess, size_t Index) const override;

private:
    WedgeGeometry3D const& Geometry_;
};

/*! \brief Point motion decorator for the wedge.
  * Every generator keeps z = 0 (z velocity set to zero after the base motion and its fix).
  * In-plane slow-down: RoundCells3D's slow-down looks at the nearest neighbor, which near the axis is the cell's own
    wall mirror, so it never acts there. Here, when a generator is closer than 0.4 of the smaller 2D width to its
    nearest in-plane neighbor, each of the pair removes half of their approach speed.
  * Axis guard: a generator moves at most a quarter of its distance to the axis towards it per first-pass dt.
  * Outer-face guard: a generator stays at least 0.2 in-plane widths inside x = XMin, x = XMax and y = YMax (outflow
    would otherwise push it geometrically onto the face, where the near-coincident mirror makes the geometry ill-conditioned).
*/
class WedgePointMotion3D : public PointMotion3D {
public:
    explicit WedgePointMotion3D(PointMotion3D const& Base) : Base_(Base) {}

    void operator()(const Tessellation3D& Tess, const vector<ComputationalCell3D>& Cells, double Time, vector<Vector3D>& Res) const override;

    void ApplyFix(Tessellation3D const& Tess, vector<ComputationalCell3D> const& Cells, double Time, double Dt, vector<Vector3D>& Velocities) const override;

    bool MovedPoints() const override { return Base_.MovedPoints(); }

private:
    PointMotion3D const& Base_;
};

/*! \brief Cell updater decorator that removes z momentum before the primitive update.
  The removed kinetic energy is added to the internal energy, so total energy is unchanged.
  By symmetry z momentum is round-off; without this the explicit update with the 2D time step
  would let it grow (the thickness mode across the wedge).
*/
class WedgeCellUpdater3D : public CellUpdater3D {
public:
    explicit WedgeCellUpdater3D(CellUpdater3D const& Base) : Base_(Base) {}

    void operator()(vector<ComputationalCell3D>& Res, EquationOfState const& Eos, const Tessellation3D& Tess, vector<Conserved3D>& Extensives) const override;

private:
    CellUpdater3D const& Base_;
};

/*! \brief CFL time step for the wedge. The length scale is WedgeGeometry3D::InPlaneWidth and the wall faces are
  skipped, because no wave crosses them. The requested-step protocol (SetTimeStep caps the first and the repeated
  call at the same time, then is cleared) is the same as CourantFriedrichsLewy's.
*/
class WedgeCFL3D : public TimeStepFunction3D {
public:
    WedgeCFL3D(WedgeGeometry3D const& Geometry, double Cfl, double SourceCfl, SourceTerm3D const& Source);

    double operator()(const Tessellation3D& Tess, const vector<ComputationalCell3D>& Cells, const EquationOfState& Eos,
        const vector<Vector3D>& FaceVelocities, const double Time) override;

    void SetTimeStep(double Dt) override;

    double GetTimeStep(void) const override;

    double SuggestTimeStep(void) const override;

private:
    WedgeGeometry3D const& Geometry_;
    double const Cfl_, SourceCfl_;
    SourceTerm3D const& Source_;
    bool FirstTry_;
    double Requested_, Dt_, SuggestedDt_, LastTime_;
};

/*! \brief Gray FLD boundary: reflecting walls, open (vacuum) outer faces.
  On a wall the mirror cell has the cell's own E, so the diffusive flux is zero (DiffusionClosedBox), and the
  radiation momentum term uses the interior formula of Diffusion::BuildMatrix with E_mirror = E_cell. The walls'
  share of that term is the geometric v_y/y part of the compression work, so it must match the interior faces.
*/
class DiffusionWedgeBoundary : public DiffusionBoundaryCalculator {
public:
    explicit DiffusionWedgeBoundary(WedgeGeometry3D const& Geometry) : Geometry_(Geometry) {}

    void SetBoundaryValues(Tessellation3D const& Tess, size_t const Index, size_t const OutsidePoint, double const Dt,
        std::vector<ComputationalCell3D> const& Cells, double const Area, double& A, double& B, size_t const FaceIndex) const override;

    void GetOutSideValues(Tessellation3D const& Tess, std::vector<ComputationalCell3D> const& Cells, size_t const Index, size_t const OutsidePoint,
        std::vector<double> const& NewE, double& EOutside, Vector3D& VOutside) const override;

    void SetMomentumTermBoundary(Tessellation3D const& Tess, size_t const Index, size_t const OutsidePoint, double const Dt,
        ComputationalCell3D const& Cell, double const Area, double& A, double& B, size_t const FaceIndex, double const FleckFactor,
        double const FluxLimiter, double const D, double const SigmaPlanck) const override;

private:
    WedgeGeometry3D const& Geometry_;
    DiffusionClosedBox Closed_;
    DiffusionOpenBoundary Open_;
};

#endif // WEDGE3D_HPP
