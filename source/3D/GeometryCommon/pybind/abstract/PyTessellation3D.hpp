#ifndef PY_PYBIND_TESSELLATION3D_HPP
#define PY_PYBIND_TESSELLATION3D_HPP 1

#include "<pybind11/pybind11.h>"
#include "../../Tessellation3D.hpp"

using std::vector;

class PyTessellation3D : public Tessellation3D
{
public:
  #ifdef RICH_MPI
    vector<Vector3D> UpdateMPIPoints(Tessellation3D const& vproc, int rank,
      vector<Vector3D> const& points, vector<std::size_t>& selfindex, vector<int>& sentproc, vector<vector<std::size_t> >& sentpoints) override 
    {
      PYBIND11_OVERRIDE_PURE
      (
          vector<Vector3D>, // return value
          Tessellation3D, // class
          "update_mpi_points", // name of the function in python
          UpdateMPIPoints, // name of the function in C++
          vproc, rank, points, selfindex, sentproc, sentpoints // arguments
      );
    }
  #endif

  void Build(vector<Vector3D> const& points) override
  {
    PYBIND11_OVERRIDE_PURE_NAME
    (
      void, // return value
      Tessellation3D, // class
      "build", // name of the function in python
      Build, // name of the function in C++
      points // arguments
    );
  }

  #ifdef RICH_MPI
    void Build(vector<Vector3D> const& points, Tessellation3D const& tproc) override
    {
      PYBIND11_OVERRIDE_PURE
      (
        void, // return value
        Tessellation3D, // class
        "build", // name of the function in python
        Build, // name of the function in C++
        points, tproc // arguments
      );
    }
  #endif

  size_t GetPointNo(void) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      size_t, // return value
      Tessellation3D, // class
      "get_point_no", // name of the function in python
      GetPointNo, // name of the function in C++
    );
  }

  size_t& GetPointNo(void) override
  {
    PYBIND11_OVERRIDE_PURE
    (
      size_t&, // return value
      Tessellation3D, // class
      "get_point_no", // name of the function in python
      GetPointNo, // name of the function in C++
    );
  }

  Vector3D GetMeshPoint(size_t index) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      Vector3D, // return value
      Tessellation3D, // class
      "get_mesh_point", // name of the function in python
      GetMeshPoint, // name of the function in C++
      index // arguments
    );
  }

  double GetArea(size_t index) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      double, // return value
      Tessellation3D, // class
      "get_area", // name of the function in python
      GetArea, // name of the function in C++
      index // arguments
    );
  }

  vector<double>& GetAllArea(void) override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<double>&, // return value
      Tessellation3D, // class
      "get_all_area", // name of the function in python
      GetAllArea, // name of the function in C++
    );
  }

  Vector3D const& GetCellCM(size_t index) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      Vector3D const&, // return value
      Tessellation3D, // class
      "get_cell_cm", // name of the function in python
      GetCellCM, // name of the function in C++
      index // arguments
    );
  }

  size_t GetTotalFacesNumber(void) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      size_t, // return value
      Tessellation3D, // class
      "get_total_faces_number", // name of the function in python
      GetTotalFacesNumber, // name of the function in C++
    );
  }

  double GetWidth(size_t index) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      Vector3D const&, // return value
      Tessellation3D, // class
      "get_width", // name of the function in python
      GetWidth, // name of the function in C++
      index // arguments
    );
  }

  double GetVolume(size_t index) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      Vector3D const&, // return value
      Tessellation3D, // class
      "get_volume", // name of the function in python
      GetVolume, // name of the function in C++
      index // arguments
    );
  }

  face_vec const& GetCellFaces(size_t index) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      face_vec const&, // return value
      Tessellation3D, // class
      "get_cell_faces", // name of the function in python
      GetCellFaces, // name of the function in C++
      index // arguments
    );
  }

  vector<face_vec >& GetAllCellFaces(void) override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<face_vec>&, // return value
      Tessellation3D, // class
      "get_all_cell_faces", // name of the function in python
      GetAllCellFaces, // name of the function in C++
    );
  }

  vector<Vector3D>& accessMeshPoints(void) override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<Vector3D>&, // return value
      Tessellation3D, // class
      "access_mesh_points", // name of the function in python
      accessMeshPoints, // name of the function in C++
    );
  }

  const vector<Vector3D>& getMeshPoints(void) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      const vector<Vector3D>&, // return value
      Tessellation3D, // class
      "get_mesh_points", // name of the function in python
      getMeshPoints, // name of the function in C++
    );
  }
  
  vector<Vector3D>& GetFacePoints(void) override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<Vector3D>&, // return value
      Tessellation3D, // class
      "get_face_points", // name of the function in python
      GetFacePoints, // name of the function in C++
    );
  }

  vector<Vector3D>const& GetFacePoints(void) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<Vector3D>const&, // return value
      Tessellation3D, // class
      "get_face_points", // name of the function in python
      GetFacePoints, // name of the function in C++
    );
  }

  point_vec const& GetPointsInFace(size_t index) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      point_vec const&, // return value
      Tessellation3D, // class
      "get_points_in_face", // name of the function in python
      GetPointsInFace, // name of the function in C++
      index // arguments
    );
  }

  vector<point_vec>& GetAllPointsInFace(void) override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<point_vec>&, // return value
      Tessellation3D, // class
      "get_all_points_in_face", // name of the function in python
      GetAllPointsInFace, // name of the function in C++
    );
  }

  vector<size_t> GetNeighbors(size_t index) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<size_t>, // return value
      Tessellation3D, // class
      "get_neighbors", // name of the function in python
      GetNeighbors, // name of the function in C++
      index // arguments
    );
  }

  void GetNeighbors(size_t index, vector<size_t> &res) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      void, // return value
      Tessellation3D, // class
      "get_neighbors", // name of the function in python
      GetNeighbors, // name of the function in C++
      index, res // arguments
    );
  }

  Tessellation3D* clone(void) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      Tessellation3D*, // return value
      Tessellation3D, // class
      "clone", // name of the function in python
      clone, // name of the function in C++
    );
  }

  ~Tessellation3D(void) = default;

  bool NearBoundary(size_t index) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      bool, // return value
      Tessellation3D, // class
      "near_boundary", // name of the function in python
      NearBoundary, // name of the function in C++
      index // arguments
    );
  }

  bool BoundaryFace(size_t index) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      bool, // return value
      Tessellation3D, // class
      "boundary_face", // name of the function in python
      BoundaryFace, // name of the function in C++
      index // arguments
    );
  }

  vector<vector<size_t> >& GetDuplicatedPoints(void) override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<vector<size_t> >&, // return value
      Tessellation3D, // class
      "get_duplicated_points", // name of the function in python
      GetDuplicatedPoints, // name of the function in C++
    );
  }

  vector<vector<size_t> >const& GetDuplicatedPoints(void) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<vector<size_t> >const&, // return value
      Tessellation3D, // class
      "get_duplicated_points", // name of the function in python
      GetDuplicatedPoints, // name of the function in C++
    );
  };

  vector<int> GetDuplicatedProcs(void) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<int>, // return value
      Tessellation3D, // class
      "get_duplicated_procs", // name of the function in python
      GetDuplicatedProcs, // name of the function in C++
    );
  };

  vector<int>& GetSentProcs(void) override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<int>&, // return value
      Tessellation3D, // class
      "get_sent_procs", // name of the function in python
      GetSentProcs, // name of the function in C++
    );
  };

  vector<int> GetSentProcs(void) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<int>, // return value
      Tessellation3D, // class
      "get_sent_procs", // name of the function in python
      GetSentProcs, // name of the function in C++
    );
  };
  
  vector<vector<size_t>>& GetSentPoints(void) override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<vector<size_t>>&, // return value
      Tessellation3D, // class
      "get_send_points", // name of the function in python
      GetSentPoints, // name of the function in C++
    );
  }

  vector<vector<size_t>> const& GetSentPoints(void) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<vector<size_t>> const&, // return value
      Tessellation3D, // class
      "get_send_points", // name of the function in python
      GetSentPoints, // name of the function in C++
    );
  }

  vector<size_t>& GetSelfIndex(void) 
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<size_t>&, // return value
      Tessellation3D, // class
      "get_self_index", // name of the function in python
      GetSelfIndex, // name of the function in C++
    );
  }

  vector<size_t> const& GetSelfIndex(void) const override
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<size_t> const&, // return value
      Tessellation3D, // class
      "get_self_index", // name of the function in python
      GetSelfIndex, // name of the function in C++
    );
  }

  size_t GetTotalPointNumber(void) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      size_t, // return value
      Tessellation3D, // class
      "get_total_point_number", // name of the function in python
      GetTotalPointNumber, // name of the function in C++
    );
  }

  vector<Vector3D>& GetAllCM(void) override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<Vector3D>&, // return value
      Tessellation3D, // class
      "get_all_cm", // name of the function in python
      GetAllCM, // name of the function in C++
    );
  }

  vector<Vector3D> GetAllCM(void) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<Vector3D>, // return value
      Tessellation3D, // class
      "get_all_cm", // name of the function in python
      GetTotalPointNumber, // name of the function in C++
    );
  }

  virtual vector<double>& GetAllVolumes(void) override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<double>&, // return value
      Tessellation3D, // class
      "get_all_volumes", // name of the function in python
      GetAllVolumes, // name of the function in C++
    );
  }

  virtual vector<double> GetAllVolumes(void) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<double>, // return value
      Tessellation3D, // class
      "get_all_volumes", // name of the function in python
      GetAllVolumes, // name of the function in C++
    );
  }

  void GetNeighborNeighbors(vector<size_t> &result, size_t point) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      void, // return value
      Tessellation3D, // class
      "get_neighbor_neighbors", // name of the function in python
      GetNeighborNeighbors, // name of the function in C++
      result, point // arguments
    );
  }

  std::pair<size_t, size_t> GetFaceNeighbors(size_t face_index) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      std::pair<size_t, size_t>, // return value
      Tessellation3D, // class
      "get_face_neighbors", // name of the function in python
      GetFaceNeighbors, // name of the function in C++
      face_index // arguments
    );
  }

  std::vector<std::pair<size_t, size_t>>& GetAllFaceNeighbors(void) override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      std::vector<std::pair<size_t, size_t>>&, // return value
      Tessellation3D, // class
      "get_all_face_neighbors", // name of the function in python
      GetAllFaceNeighbors, // name of the function in C++
    );
  }

  Vector3D Normal(size_t faceindex) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      Vector3D, // return value
      Tessellation3D, // class
      "normal", // name of the function in python
      Normal, // name of the function in C++
      faceindex // arguments
    );
  }

  bool IsGhostPoint(size_t index) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      bool, // return value
      Tessellation3D, // class
      "is_ghost_point", // name of the function in python
      IsGhostPoint, // name of the function in C++
      index // arguments
    );
  }

  Vector3D CalcFaceVelocity(size_t index, Vector3D const& v0, Vector3D const& v1) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      Vector3D, // return value
      Tessellation3D, // class
      "calc_face_velocity", // name of the function in python
      CalcFaceVelocity, // name of the function in C++
      index, v0, v1 // arguments
    );
  }

  vector<Vector3D>& GetAllFaceCM(void) override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<Vector3D>&, // return value
      Tessellation3D, // class
      "get_all_face_cm", // name of the function in python
      GetAllFaceCM, // name of the function in C++
    );
  }

  Vector3D FaceCM(size_t index) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      Vector3D, // return value
      Tessellation3D, // class
      "face_cm", // name of the function in python
      FaceCM, // name of the function in C++
      index // arguments
    );
  }

  vector<vector<size_t>>& GetGhostIndeces(void) override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<vector<size_t>>&, // return value
      Tessellation3D, // class
      "get_ghost_indeces", // name of the function in python
      GetGhostIndeces, // name of the function in C++
    );
  }

  vector<vector<size_t>> const& GetGhostIndeces(void) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      vector<vector<size_t>> const&, // return value
      Tessellation3D, // class
      "get_ghost_indeces", // name of the function in python
      GetGhostIndeces, // name of the function in C++
    );
  }

  std::pair<Vector3D, Vector3D> GetBoxCoordinates(void) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      std::pair<Vector3D, Vector3D>, // return value
      Tessellation3D, // class
      "get_box_coordinates", // name of the function in python
      GetBoxCoordinates, // name of the function in C++
    );
  }

  void BuildNoBox(vector<Vector3D> const& points, vector<vector<Vector3D>> const& ghosts, vector<size_t> toduplicate) override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      void, // return value
      Tessellation3D, // class
      "build_no_box", // name of the function in python
      BuildNoBox, // name of the function in C++
      points, ghosts, toduplicate // arguments
    );
  }

  bool IsPointOutsideBox(size_t index) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      bool, // return value
      Tessellation3D, // class
      "is_point_outside_box", // name of the function in python
      IsPointOutsideBox, // name of the function in C++
      index // arguments
    );
  }

  void output(std::string const& filename) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      void, // return value
      Tessellation3D, // class
      "output", // name of the function in python
      output, // name of the function in C++
      filename // arguments
    );
  }

  void SetBox(Vector3D const& ll, Vector3D const& ur) override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      void, // return value
      Tessellation3D, // class
      "set_box", // name of the function in python
      SetBox, // name of the function in C++
      ll, ur // arguments
    );
  }

  std::vector<Face>& ModifyBoxFaces(void) override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      std::vector<Face>&, // return value
      Tessellation3D, // class
      "modify_box_faces", // name of the function in python
      ModifyBoxFaces, // name of the function in C++
    );
  }

  std::vector<Face> GetBoxFaces(void) const override 
  {
    PYBIND11_OVERRIDE_PURE
    (
      std::vector<Face>, // return value
      Tessellation3D, // class
      "get_box_faces", // name of the function in python
      GetBoxFaces, // name of the function in C++
    );
  }
};

#endif // PY_PYBIND_TESSELLATION3D_HPP
