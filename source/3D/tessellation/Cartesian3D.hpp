/*! \file Cartesian3D.hpp
  \brief RICH adapter: Cartesian3D wraps MadCart::CartesianMesh3D<Vector3D> as a Tessellation3D
*/

#ifndef RICH_CARTESIAN3D_HPP
#define RICH_CARTESIAN3D_HPP 1

#include <vector>
#include "Tessellation3D.hpp"
#include "cartesian/CartesianMesh3D.hpp"

inline Face toFace(const MadCart::Face3D<Vector3D> &f3d)
{
    Face f;
    f.vertices.assign(f3d.vertices.begin(), f3d.vertices.end());
    f.neighbors = f3d.neighbors;
    return f;
}

inline MadCart::Face3D<Vector3D> toCartFace3D(const Face &f)
{
    MadCart::Face3D<Vector3D> f3d;
    f3d.vertices.assign(f.vertices.begin(), f.vertices.end());
    f3d.neighbors = f.neighbors;
    return f3d;
}

class Cartesian3D : public Tessellation3D
{
public:
    using Face_T = Face;
    using Point_T = Vector3D;

    Cartesian3D(const Vector3D &ll, const Vector3D &ur,
                std::size_t nx, std::size_t ny, std::size_t nz)
        : engine_(ll, ur, nx, ny, nz) {}

    Cartesian3D(Cartesian3D const &other) = default;

    void BuildPartially(const std::vector<Vector3D> &allPoints,
                        const std::vector<size_t> &indicesToBuild) override
    {
        (void)allPoints;
        (void)indicesToBuild;
    }

    void Build(vector<Vector3D> const &points) override
    {
        (void)points;
    }

#ifdef RICH_MPI
    bool PointInMyDomain(const Vector3D &point) const override { return engine_.PointInMyDomain(point); }
    int GetOwner(const Vector3D &point) const override { return engine_.GetOwner(point); }
    void SetImbalanceTolerance(double tolerance) override { engine_.SetImbalanceTolerance(tolerance); }
    void SetLoadBalancer(std::shared_ptr<LoadBalancer<Vector3D>> lb) override { engine_.SetLoadBalancer(lb); }
    void PresetLoadBalancer(std::shared_ptr<LoadBalancer<Vector3D>> lb) override { engine_.PresetLoadBalancer(lb); }
    void Rebalance(const std::vector<double> &weights) override { engine_.Rebalance(weights); }
    bool ShouldRebalance(const std::vector<double> &weights) const override { return engine_.ShouldRebalance(weights); }
    bool ShouldRebalance(void) const override { return engine_.ShouldRebalance(); }
    std::shared_ptr<LoadBalancer<Vector3D>> GetLoadBalancer(void) override { return engine_.GetLoadBalancer(); }
    const std::shared_ptr<LoadBalancer<Vector3D>> GetLoadBalancer(void) const override { return engine_.GetLoadBalancer(); }

    void BuildPartiallyParallel(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToBuild, bool suppressRebalancing = false, bool suppressExchange = false) override
    {
        engine_.BuildPartiallyParallel(allPoints, allWeights, indicesToBuild, suppressRebalancing, suppressExchange);
        box_faces_dirty_ = true;
    }

    bool DidRebalance(void) const override { return engine_.DidRebalance(); }
    const std::vector<double> &GetPointsBuildWeights() const override { return engine_.GetBuildWeights(); }

    const std::shared_ptr<EnvironmentAgent<Vector3D>> GetEnvironmentAgent() const override { return engine_.GetEnvironmentAgent(); }

    void PreparePoints(const std::vector<Vector3D> &points, const std::vector<size_t> &mask) override { engine_.PreparePoints(points, mask); }

    vector<vector<size_t>> &GetDuplicatedPoints(void) override { return engine_.GetDuplicatedPoints(); }
    vector<vector<size_t>> const &GetDuplicatedPoints(void) const override { return engine_.GetDuplicatedPoints(); }
    vector<int> GetDuplicatedProcs(void) const override { return engine_.GetDuplicatedProcs(); }
    vector<int> GetSentProcs(void) const override { return engine_.GetSentProcs(); }
    vector<vector<size_t>> const &GetSentPoints(void) const override { return engine_.GetSentPoints(); }
    vector<size_t> const &GetSelfIndex(void) const override { return engine_.GetSelfIndex(); }
    vector<int> &GetSentProcs(void) override { return engine_.GetSentProcs(); }
    vector<vector<size_t>> &GetSentPoints(void) override { return engine_.GetSentPoints(); }
    vector<size_t> &GetSelfIndex(void) override { return engine_.GetSelfIndex(); }
    vector<vector<size_t>> const &GetGhostIndeces(void) const override { return engine_.GetGhostIndeces(); }
    vector<vector<size_t>> &GetGhostIndeces(void) override { return engine_.GetGhostIndeces(); }
#endif // RICH_MPI

    size_t GetContainingCell(const Vector3D &point) const override { return engine_.GetContainingCell(point); }
    size_t GetPointNo(void) const override { return engine_.GetPointNo(); }
    size_t GetAllPointsNo(void) const override { return engine_.GetAllPointsNo(); }
    size_t &GetPointNo(void) override { return engine_.GetPointNo(); }
    const Vector3D &GetMeshPoint(size_t index) const override { return engine_.GetMeshPoint(index); }
    double GetArea(size_t index) const override { return engine_.GetArea(index); }
    vector<double> &GetAllArea(void) override { return engine_.GetAllArea(); }
    Vector3D const &GetCellCM(size_t index) const override { return engine_.GetCellCM(index); }
    size_t GetTotalFacesNumber(void) const override { return engine_.GetTotalFacesNumber(); }
    double GetWidth(size_t index) const override { return engine_.GetWidth(index); }
    double GetVolume(size_t index) const override { return engine_.GetVolume(index); }

    face_vec const &GetCellFaces(size_t index) const override { return engine_.GetCellFaces(index); }
    vector<face_vec> &GetAllCellFaces(void) override { return engine_.GetAllCellFaces(); }
    vector<face_vec> const &GetAllCellFaces(void) const override { return engine_.GetAllCellFaces(); }

    vector<Vector3D> &accessMeshPoints(void) override { return engine_.accessMeshPoints(); }
    const vector<Vector3D> &getMeshPoints(void) const override { return engine_.getMeshPoints(); }

    const Tessellation3D::AllPointsMap &GetIndicesInAllPoints(void) const override
    {
        return engine_.GetIndicesInAllPoints();
    }

    const std::vector<Vector3D> &getAllPoints(void) const override { return engine_.getAllPoints(); }
    std::vector<Vector3D> &getAllPoints(void) override { return engine_.getAllPoints(); }

    vector<Vector3D> &GetFacePoints(void) override { return engine_.GetFacePoints(); }
    vector<Vector3D> const &GetFacePoints(void) const override { return engine_.GetFacePoints(); }

    point_vec const &GetPointsInFace(size_t index) const override { return engine_.GetPointsInFace(index); }
    vector<point_vec> &GetAllPointsInFace(void) override { return engine_.GetAllPointsInFace(); }
    vector<point_vec> const &GetAllPointsInFace(void) const override { return engine_.GetAllPointsInFace(); }

    vector<size_t> GetNeighbors(size_t index) const override { return engine_.GetNeighbors(index); }
    void GetNeighbors(size_t index, vector<size_t> &res) const override { engine_.GetNeighbors(index, res); }

    Tessellation3D *clone(void) const override
    {
        return static_cast<Tessellation3D *>(new Cartesian3D(*this));
    }

    bool NearBoundary(size_t index) const override { return engine_.NearBoundary(index); }
    bool BoundaryFace(size_t index) const override { return engine_.BoundaryFace(index); }

    bool IsPointInCell(const Vector3D &point, size_t cellIndex, bool verbose = false) const override
    {
        return engine_.IsPointInCell(point, cellIndex, verbose);
    }

    size_t GetTotalPointNumber(void) const override { return engine_.GetTotalPointNumber(); }

    vector<Vector3D> &GetAllCM(void) override { return engine_.GetAllCM(); }
    vector<Vector3D> GetAllCM(void) const override { return engine_.GetAllCM(); }

    vector<double> &GetAllVolumes(void) override { return engine_.GetAllVolumes(); }
    vector<double> GetAllVolumes(void) const override { return engine_.GetAllVolumes(); }

    void GetNeighborNeighbors(vector<size_t> &result, size_t point) const override
    {
        engine_.GetNeighborNeighbors(result, point);
    }

    const std::pair<size_t, size_t> &GetFaceNeighbors(size_t face_index) const override
    {
        return engine_.GetFaceNeighbors(face_index);
    }

    std::vector<std::pair<size_t, size_t>> &GetAllFaceNeighbors(void) override
    {
        return engine_.GetAllFaceNeighbors();
    }

    const std::vector<std::pair<size_t, size_t>> &GetAllFaceNeighbors(void) const override
    {
        return engine_.GetAllFaceNeighbors();
    }

    Vector3D Normal(size_t faceindex) const override { return engine_.Normal(faceindex); }

    bool IsGhostPoint(size_t index) const override { return engine_.IsGhostPoint(index); }

    Vector3D CalcFaceVelocity(size_t index, Vector3D const &v0, Vector3D const &v1) const override
    {
        return engine_.CalcFaceVelocity(index, v0, v1);
    }

    vector<Vector3D> &GetAllFaceCM(void) override { return engine_.GetAllFaceCM(); }
    const Vector3D &FaceCM(size_t index) const override { return engine_.FaceCM(index); }

    std::pair<Vector3D, Vector3D> GetBoxCoordinates(void) const override { return engine_.GetBoxCoordinates(); }

    void BuildNoBox(vector<Vector3D> const &points, vector<vector<Vector3D>> const &ghosts,
                    vector<size_t> toduplicate) override
    {
        engine_.BuildNoBox(points, ghosts, std::move(toduplicate));
    }

    bool IsPointOutsideBox(size_t index) const override { return engine_.IsPointOutsideBox(index); }
    bool IsPointOutsideBox(const Vector3D &point) const override { return engine_.IsPointOutsideBox(point); }

    void output(std::string const &filename) const override { engine_.output(filename); }

    void SetBox(Vector3D const &ll, Vector3D const &ur) override
    {
        syncBoxFacesToEngine();
        engine_.SetBox(ll, ur);
        box_faces_dirty_ = true;
    }

    std::vector<Face> &ModifyBoxFaces(void) override
    {
        syncBoxFacesFromEngine();
        box_faces_engine_stale_ = true;
        return box_faces_cache_;
    }

    const std::vector<Face> &GetBoxFaces(void) const override
    {
        syncBoxFacesFromEngine();
        return box_faces_cache_;
    }

    std::vector<Face> GetBoxFaces(void) override
    {
        syncBoxFacesFromEngine();
        return box_faces_cache_;
    }

    size_t GetBuildGeneration(void) const override { return engine_.GetBuildGeneration(); }

    // ---- Cartesian-specific accessors ----
    std::size_t nx() const { return engine_.nx(); }
    std::size_t ny() const { return engine_.ny(); }
    std::size_t nz() const { return engine_.nz(); }
    double dx() const { return engine_.dx(); }
    double dy() const { return engine_.dy(); }
    double dz() const { return engine_.dz(); }

    MadCart::CartesianMesh3D<Vector3D> &engine() { return engine_; }
    const MadCart::CartesianMesh3D<Vector3D> &engine() const { return engine_; }

private:
    MadCart::CartesianMesh3D<Vector3D> engine_;
    mutable std::vector<Face> box_faces_cache_;
    mutable bool box_faces_dirty_ = true;
    bool box_faces_engine_stale_ = false;


    void syncBoxFacesFromEngine() const
    {
        if (!box_faces_dirty_)
            return;
        const auto &engine_faces = engine_.GetBoxFaces();
        box_faces_cache_.resize(engine_faces.size());
        for (size_t i = 0; i < engine_faces.size(); ++i)
            box_faces_cache_[i] = toFace(engine_faces[i]);
        box_faces_dirty_ = false;
    }

    void syncBoxFacesToEngine()
    {
        if (!box_faces_engine_stale_)
            return;
        auto &engine_faces = engine_.ModifyBoxFaces();
        engine_faces.resize(box_faces_cache_.size());
        for (size_t i = 0; i < box_faces_cache_.size(); ++i)
            engine_faces[i] = toCartFace3D(box_faces_cache_[i]);
        box_faces_engine_stale_ = false;
        box_faces_dirty_ = false;
    }
};

#endif // RICH_CARTESIAN3D_HPP
