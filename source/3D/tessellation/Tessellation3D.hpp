/*! \file Tessellation3D.hpp
  \brief Abstract class for the tessellation in 3D
  \author Elad Steinberg
*/

#ifndef TESSELLATION3D_HPP
#define TESSELLATION3D_HPP 1

#include <array>
#include <cassert>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <limits>
#include <utility>
#include <vector>
#include <boost/container/small_vector.hpp>
#include "../elementary/Face.hpp"
#ifdef RICH_MPI
#include <MeshDecomposer3D/environment/EnvironmentAgent.hpp>
#include <MeshDecomposer3D/load_balancing/LoadBalancer.hpp>
#include "mpi/mpi_exchange_commands.hpp"
#endif // RICH_MPI

//! \brief Container for points defining a face
typedef boost::container::small_vector<size_t, 24> face_vec;
//! \brief Container for neighbouring points
typedef boost::container::small_vector<size_t, 8> point_vec;
//! \brief Container for neighbouring tetrahedra
typedef boost::container::small_vector<size_t, 40> tetra_vec;

using std::vector;

/*! \brief Abstract class for tessellation in 3D
  \author Elad Steinberg
*/
class Tessellation3D
{
public:
  using AllPointsMap = boost::container::flat_map<size_t, size_t>;
  using Face_T = Face;
  using Point_T = Vector3D;
  
  virtual void BuildPartially(const std::vector<Vector3D> &allPoints, const std::vector<size_t> &indicesToBuild) = 0;

  /*! \brief Builds the tessellation
    \param points Initial position of mesh generating points
  */
  virtual void Build(vector<Vector3D> const& points)
  {
    std::vector<size_t> indicesToBuild(points.size());
    std::iota(indicesToBuild.begin(), indicesToBuild.end(), 0);
    this->BuildPartially(points, indicesToBuild);
  }

  #ifdef RICH_MPI
  /*! \brief Returns true iff the given point lies inside my domain (under my responsibility).
  */
  virtual bool PointInMyDomain(const Vector3D &point) const = 0;

  virtual int GetOwner(const Vector3D &point) const = 0;

  virtual void SetImbalanceTolerance(double tolerance) = 0;

  virtual void SetLoadBalancer(std::shared_ptr<LoadBalancer<Vector3D>> loadBalancer) = 0;

  virtual void PresetLoadBalancer(std::shared_ptr<LoadBalancer<Vector3D>> loadBalancer) = 0;

  virtual void Rebalance(const std::vector<double> &weights) = 0;

  virtual bool ShouldRebalance(const std::vector<double> &weights) const = 0;

  virtual bool ShouldRebalance(void) const = 0;

  virtual std::shared_ptr<LoadBalancer<Vector3D>> GetLoadBalancer(void) = 0;

  virtual const std::shared_ptr<LoadBalancer<Vector3D>> GetLoadBalancer(void) const = 0;

  virtual void BuildPartiallyParallel(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToBuild, bool suppressRebalancing = false, bool suppressExchange = false) = 0;

  inline void BuildPartiallyParallel(const std::vector<Vector3D> &allPoints, const std::vector<size_t> &indicesToBuild, bool suppressRebalancing = false, bool suppressExchange = false)
  {
    this->BuildPartiallyParallel(allPoints, std::vector<double>(allPoints.size(), 1.0), indicesToBuild, suppressRebalancing, suppressExchange);
  }

  virtual void BuildParallel(const std::vector<Vector3D> &points, const std::vector<double> &weights, bool suppressRebalancing = false, bool suppressExchange = false)
  {
      std::vector<size_t> indicesToBuild(points.size());
      std::iota(indicesToBuild.begin(), indicesToBuild.end(), 0);
      this->BuildPartiallyParallel(points, weights, indicesToBuild, suppressRebalancing, suppressExchange);
  }

  virtual bool DidRebalance(void) const = 0;

  inline void BuildParallel(const std::vector<Vector3D> &points, bool suppressRebalancing = false, bool suppressExchange = false)
  {
      this->BuildParallel(points, std::vector<double>(points.size(), 1.0), suppressRebalancing, suppressExchange);
  }

  virtual const std::vector<double> &GetPointsBuildWeights() const = 0;
  
  virtual const std::shared_ptr<EnvironmentAgent<Vector3D>> GetEnvironmentAgent() const = 0;

  virtual void PreparePoints(const std::vector<Vector3D> &points, const std::vector<size_t> &mask) = 0;

  #endif // RICH_MPI

  virtual size_t GetContainingCell(const Vector3D &point) const = 0;
  
  /*! \brief Get Total number of mesh generating points
    \return Number of mesh generating points
  */
  virtual size_t GetPointNo(void) const = 0;

  /*! \brief Returns the number of all the points, even those which are not participating in the build
    \return Number of all the points
  */
  virtual size_t GetAllPointsNo(void) const = 0;


  /*! \brief Get number of points
    \return Number of all points
   */
  virtual size_t& GetPointNo(void) = 0;

  /*! \brief Returns Position of mesh generating point
    \param index Mesh generating point index
    \return Position of mesh generating point
  */
  virtual const Vector3D &GetMeshPoint(size_t index) const = 0;

  /*! \brief Returns Area of face
    \param index The index of the face
    \return The area of the face
  */
  virtual double GetArea(size_t index) const = 0;

  /*! \brief Get areas of all faces
    \return List of areas of all faces
   */
  virtual vector<double>& GetAllArea(void) = 0;


  /*! \brief Returns Position of Cell's Center of Mass
    \param index Mesh generating point index (the cell's index)
    \return Position of CM
  */
  virtual Vector3D const& GetCellCM(size_t index) const = 0;

  /*! \brief Returns the total number of faces
    \return Total number of faces
  */
  virtual size_t GetTotalFacesNumber(void) const = 0;

  /*! \brief Returns the effective width of a cell
    \param index Cell index
    \return Effective cell width
  */
  virtual double GetWidth(size_t index) const = 0;

  /*! \brief Returns the volume of a cell
    \param index Cell index
    \return Cell volume
  */
  virtual double GetVolume(size_t index) const = 0;

  /*! \brief Returns the indeces of a cell's Faces
    \param index Cell index
    \return Cell edges
  */
  virtual face_vec const& GetCellFaces(size_t index) const = 0;

  /*! \brief Get all cell faces
    \return List of all cell faces
   */
  virtual vector<face_vec >& GetAllCellFaces(void) = 0;

  
  /*! \brief Get all cell faces
    \return List of all cell faces
   */
  virtual vector<face_vec> const &GetAllCellFaces(void) const = 0;

  /*!
    \brief Returns a reference to the point vector
    \returns The reference
  */
  virtual vector<Vector3D>& accessMeshPoints(void) = 0;

  /*! \brief Get all mesh points
    \return List of all mesh points
   */
  virtual const vector<Vector3D>& getMeshPoints(void) const = 0;

  virtual const Tessellation3D::AllPointsMap &GetIndicesInAllPoints(void) const = 0;

  /*! \brief Returns all the points, even those which are not participating in the build
    \return List of all the points
  */
  virtual const std::vector<Vector3D> &getAllPoints(void) const = 0;

  /*! \brief Returns all the points, even those which are not participating in the build
    \return List of all the points
  */
  virtual std::vector<Vector3D> &getAllPoints(void) = 0;

  /*!
    \brief Returns a reference to the points composing the faces vector
    \returns The reference
  */
  virtual vector<Vector3D>& GetFacePoints(void) = 0;

  /*!
    \brief Returns a reference to the points composing the faces vector
    \returns The reference
  */
  virtual vector<Vector3D>const& GetFacePoints(void) const = 0;

  /*!
    \brief Returns a reference to the indeces of the points composing a face. Points are order in a right hand fashion, normal pointing towards the first neighbor
    \param index The index of the face
    \returns The reference
  */
  virtual point_vec const& GetPointsInFace(size_t index) const = 0;

  /*! \brief Get a list of all points in face
    \return List to all points in face
   */
  virtual vector<point_vec > & GetAllPointsInFace(void) = 0;

    /*! \brief Get a list of all points in face
    \return List to all points in face
   */
  virtual vector<point_vec > const& GetAllPointsInFace(void) const = 0;

  /*!
    \brief Returns a list of the neighbors of a cell
    \param index The cell to check
    \return The neighbors
  */
  virtual vector<size_t> GetNeighbors(size_t index)const = 0;

  /*!
    \brief Returns a list of the neighbors of a cell
    \param index The cell to check
    \param res The neighbors, returned
  */
  virtual void GetNeighbors(size_t index,vector<size_t> &res)const = 0;

  /*!
    \brief Cloning function
    \return Pointer to new tessellation
  */
  virtual Tessellation3D* clone(void) const = 0;

  //! \brief Virtual destructor
  virtual ~Tessellation3D(void);

  /*! 
    \brief Returns if the cell is adjacent to a boundary
    \param index The cell to check
    \return If near boundary
  */
  virtual bool NearBoundary(size_t index) const = 0;

  /*! 
    \brief Returns if the face is a boundary one
    \param index The face to check
    \return True if boundary false otherwise
  */
  virtual bool BoundaryFace(size_t index) const = 0;

  virtual bool IsPointInCell(const Vector3D &point, size_t cellIndex, bool verbose = false) const = 0;

  /*! \brief Returns whether a point is inside a cell and, if not, the first neighboring cell across a violated face.
   *  The default is a safe unsupported-operation result for non-Voronoi tessellations.
   */
  virtual std::pair<bool, size_t> FindViolatedFaceNeighbor(const Vector3D &point, size_t cellIndex) const
  {
    (void)point;
    (void)cellIndex;
    return {false, std::numeric_limits<size_t>::max()};
  }

  /*! \brief Enables periodicity along each axis of the domain box.
   *  The default leaves the tessellation non-periodic.
   */
  virtual void SetPeriodic(bool periodicX, bool periodicY, bool periodicZ)
  {
    (void)periodicX;
    (void)periodicY;
    (void)periodicZ;
  }

  /*! \brief Returns periodicity flags for x, y, and z. */
  virtual std::array<bool, 3> GetPeriodic() const
  {
    return {false, false, false};
  }

  /*! \brief Returns true if at least one axis is periodic. */
  virtual bool HasPeriodic() const
  {
    return false;
  }

  /*! \brief Wraps a point into the primary domain of a periodic tessellation.
   *  The default leaves the point untouched, which is what every non periodic tessellation needs.
   */
  virtual void WrapPeriodicPoint(Vector3D &point) const
  {
    (void)point;
  }

  virtual bool IsPeriodicImage(size_t meshIndex) const
  {
    (void)meshIndex;
    return false;
  }

  virtual Vector3D GetPeriodicImageTranslation(size_t meshIndex) const
  {
    (void)meshIndex;
    return Vector3D();
  }

  virtual size_t ResolvePeriodicImageIndex(size_t meshIndex) const
  {
    const size_t cellNumber = GetPointNo();
    if(meshIndex < cellNumber)
    {
      return meshIndex;
    }
    if(HasPeriodic())
    {
      Vector3D periodicPoint = GetMeshPoint(meshIndex);
      WrapPeriodicPoint(periodicPoint);
      const size_t containing = GetContainingCell(periodicPoint);
      if(containing < cellNumber)
      {
        return containing;
      }
    }
    return meshIndex;
  }

  #ifdef RICH_MPI
    /*!
      \brief Returns the indeces of the points that were sent to other processors as ghost points 
      \return The sent points, outer vector is the index of the cpu and inner vector are the points sent through the face
    */
    virtual vector<vector<size_t> >& GetDuplicatedPoints(void) = 0;
    /*!
      \brief Returns the indeces of the points that were sent to other processors as ghost points
      \return The sent points, outer vector is the index of the cpu and inner vector are the points sent through the face
    */
    virtual vector<vector<size_t> >const& GetDuplicatedPoints(void)const = 0;

    /*!
      \brief Returns the indeces of the points that were sent to other processors as ghost points
      \return The sent points, outer vector is the index of the cpu and inner vector are the points sent through the face
    */
    virtual vector<int> GetDuplicatedProcs(void)const = 0;

    /*! \brief Gets the list of parallel process to which points have been sent
      \return List of process indices
    */
    virtual vector<int> GetSentProcs(void)const = 0;

    /*! \brief Get Indices of points sent to other parallel processes
      \return List of indices of cells sent to other processes, partitioned by process
    */
    virtual vector<vector<size_t> > const& GetSentPoints(void)const = 0;

    /*! \brief Get real index of points
      \return List of real indices
    */
    virtual vector<size_t> const& GetSelfIndex(void) const = 0;

    /*! \brief Gets the list of parallel process to which points have been sent
      \return List of process indices
    */
    virtual vector<int>& GetSentProcs(void) = 0;

    /*! \brief Get Indices of points sent to other parallel processes
      \return List of indices of cells sent to other processes, partitioned by process
    */
    virtual vector<vector<size_t> > & GetSentPoints(void) = 0;

    /*! \brief Get self inidices of points
      \return List of all indices
    */
    virtual vector<size_t> & GetSelfIndex(void) = 0;
  #endif // RICH_MPI

  /*!
    \brief Returns the total number of points (including ghost)
    \return The total number of points
  */
  virtual size_t GetTotalPointNumber(void)const = 0;

  /*!
    \brief Returns the center of masses of the cells
    \return The CM's
  */
  virtual vector<Vector3D>& GetAllCM(void) = 0;

  /*!
    \brief Returns the center of masses of the cells
    \return The CM's
  */
  virtual vector<Vector3D> GetAllCM(void) const = 0;

  /*!
    \brief Returns the volumes of the cells
    \return The volumes
  */
  virtual vector<double>& GetAllVolumes(void) = 0;

  /*!
    \brief Returns the volumes of the cells
    \return The volumes
  */
  virtual vector<double> GetAllVolumes(void)const = 0;

  /*!
    \brief Returns the neighbors and neighbors of the neighbors of a cell
    \param point The index of the cell to calculate for
    \param result The neighbors and their neighbors indeces
  */
  virtual void GetNeighborNeighbors(vector<size_t> &result,size_t point) const = 0;

  /*! \brief Get the indices of neighbours of a face
    \param face_index Index of the face
    \return Pair of indices of cells on the two sides of the face
   */
  virtual const std::pair<size_t,size_t> &GetFaceNeighbors(size_t face_index) const = 0;

  /*! \brief Retrieve all neighbouring points who share a face
    \return List of pairs of indices of all neighbouring points
   */
  virtual std::vector<std::pair<size_t, size_t>> &GetAllFaceNeighbors(void) = 0;

  /*! \brief Retrieve all neighbouring points who share a face
    \return List of pairs of indices of all neighbouring points
   */
  virtual const std::vector<std::pair<size_t, size_t>> &GetAllFaceNeighbors(void) const = 0;

  /*!
    \brief Returns a vector normal to the face whose magnitude is the seperation between the neighboring points
    \param faceindex The index of the face
    \return The vector normal to the face whose magnitude is the seperation between the neighboring points pointing from the first neighbor to the second
  */
  virtual Vector3D Normal(size_t faceindex)const=0;

  /*!
    \brief Checks if a point is a ghost point or not
    \param index Point index
    \return True if is a ghost point, false otherwise
  */
  virtual bool IsGhostPoint(size_t index)const=0;

  /*!
    \brief Calculates the velocity of a face
    \param index The face index
    \param v0 The velocity of the first neighbor
    \param v1 The velocity of the second neighbor
    \return The velocity of the face
  */
  virtual Vector3D CalcFaceVelocity(size_t index,Vector3D const& v0,Vector3D const& v1)const=0;

  /*! \brief Calculates the centres of mass of all faces
    \return List of all centres of mass
   */
  virtual vector<Vector3D>& GetAllFaceCM(void) = 0;

  /*! \brief Return the centre of mass of a face
    \param index Face index
    \return Position of the face centre of mass
   */
  virtual const Vector3D &FaceCM(size_t index)const=0;

  #ifdef RICH_MPI
    /*! \brief Get indices of ghost points
      \return List of list of ghost point indices
    */
    virtual vector<vector<size_t> > const& GetGhostIndeces(void) const = 0;

    /*! \brief Get indices of ghost points
      \return List of indices of ghost points
    */
    virtual vector<vector<size_t> > & GetGhostIndeces(void) = 0;
  #endif // RICH_MPI
  
  /*! \brief Get the coordinate of opposite corners of the boundary
    \return Pair of coordiantes of opposite corners
   */
  virtual std::pair<Vector3D, Vector3D> GetBoxCoordinates(void)const = 0;

  /*! \brief Build tessellatoin without a box
    \param points Mesh generating points
    \param ghosts Ghost points
    \param toduplicate List of duplicate points
   */
  virtual void BuildNoBox(vector<Vector3D> const& points, vector<vector<Vector3D> > const& ghosts, vector<size_t> toduplicate) = 0;

  /*! \brief Checks if a point is inside the box
    \param index Point index
    \return True if point is inside the box
   */
  virtual bool IsPointOutsideBox(size_t index) const = 0;

  virtual bool IsPointOutsideBox(const Vector3D &point) const = 0;

  /*! \brief Write tessellation to file
    \param filename Name of output file
   */
  virtual void output(std::string const& filename)const=0;

  /*! \brief Adjust the boundary
    \param ll Lower left
    \param ur Upper right
   */
  virtual void SetBox(Vector3D const& ll, Vector3D const& ur) = 0;
/*!
\brief Access method to box faces
\return The box faces
*/
  virtual std::vector<Face>& ModifyBoxFaces(void) = 0;

  /*!
  \brief Access method to box faces
  \return The box faces
  */
  virtual const std::vector<Face> &GetBoxFaces(void) const = 0;

  /*!
  \brief Access method to box faces
  \return The box faces
  */
  virtual std::vector<Face> GetBoxFaces(void) = 0;

  virtual size_t GetBuildGeneration(void) const = 0;

  /**
   * Suppose `partialBuildData` contains correct local data for partial build, and that `allBuildData` have correct values for all the points (non active local).
   * This collective method updates `partialBuildData` to have the data of both active/non active local and global points.    
  */
  template<typename T>
  void SyncPartialBuildData(std::vector<T> &partialBuildData, std::vector<T> &allBuildData) const;
};

/*! \brief Create a subset of a vector of points
  \param v Source list of points
  \param index List of indices
  \return Points selected according to list of indices
 */
point_vec_v VectorValues(std::vector<Vector3D> const&v, point_vec const &index);

template<typename T>
inline void Tessellation3D::SyncPartialBuildData(std::vector<T> &partialBuildData, std::vector<T> &allBuildData) const
{
  size_t Norg = this->GetPointNo();
  if(partialBuildData.size() < Norg)
  {
    UniversalError eo("Tessellation3D::SyncPartialBuildData: Partial build data has lower size than the number of points");
    eo.addEntry("Partial build data size", partialBuildData.size());
    eo.addEntry("Number of points", Norg);
    throw eo;
  }
  const Tessellation3D::AllPointsMap &indicesInAllMyPoints = this->GetIndicesInAllPoints();
  
  allBuildData.resize(this->GetAllPointsNo());

  // update CMs of active local points in all points CM vector
  for(size_t i = 0; i < Norg; i++)
  {
      size_t pointIdx = indicesInAllMyPoints.at(i);
      allBuildData[pointIdx] = partialBuildData[i];
  }

  // update the CM of local non active points
  size_t sizeOfMeshPoints = this->getMeshPoints().size();
  partialBuildData.resize(sizeOfMeshPoints);
  for(size_t i = Norg; i < sizeOfMeshPoints; i++)
  {
      // check if the point is mine. i.e, appears in `indicesInAllMyPoints`. Just copy the CM from there.
      bool pointIsMine = (indicesInAllMyPoints.find(i) != indicesInAllMyPoints.cend());
      if(pointIsMine)
      {
          size_t pointIdx = indicesInAllMyPoints.at(i);
          partialBuildData[i] = allBuildData[pointIdx];
      }
  }

  #ifdef RICH_MPI
      // std::chrono::_V2::system_clock::time_point start, end;
      // update the CM of active and not active, but non local points

      // start = std::chrono::system_clock::now();
      // MPI_Exchanger exchanger(this->GetDuplicatedProcs());
      // std::vector<std::vector<T>> incoming = exchanger.exchange_indices_seperated<T, size_t>(allBuildData, this->GetDuplicatedProcs(), this->GetDuplicatedPoints());
      const std::vector<int> &dupProcs = this->GetDuplicatedProcs();
      std::vector<std::vector<T>> incoming = MPI_exchange_data_indexed(dupProcs, allBuildData, this->GetDuplicatedPoints());
      size_t incomingSize = incoming.size();
      const std::vector<std::vector<size_t>> &Nghost = this->GetGhostIndeces();
      assert(dupProcs.size() == Nghost.size());
      assert(incomingSize == Nghost.size());
      for (size_t i = 0; i < incomingSize; ++i)
      {
          size_t _size = incoming[i].size();
          assert(_size == Nghost[i].size());
          for (size_t j = 0; j < _size; ++j)
          {
              partialBuildData[Nghost.at(i).at(j)] = incoming[i][j];
          }
      }
  #endif // RICH_MPI
}

#endif // TESSELLATION3D_HPP
