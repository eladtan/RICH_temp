#include "../../../../pybind/include.h"
#include "abstract/PyTessellation3D.hpp"

#define MODULE_NAME tessellation3D
#define CLASS_NAME "Tessellation3D"
#define MODULE_VERSION "1.0"

static void __exportTessellation3D(py::module &module)
{
    py::class_<Tessellation3D, PyTessellation3D /* a trampoline */>(module, CLASS_NAME, "Abstract class for tessellation in 3D")
        #ifdef RICH_MPI
        .def("update_mpi_points", &Tessellation3D::UpdateMPIPoints, "Update meta tessellation points")
        .def("build", py::overload_cast<vector<Vector3D> const&, Tessellation3D const&>(&Tessellation3D::Build), "Builds the tessellation",
            py::arg("points"), py::args("tproc"))
        #endif
        .def("build", py::overload_cast<vector<Vector3D> const&>(&Tessellation3D::Build), "Builds the tessellation", py::arg("points"))
        .def("get_point_no", py::overload_cast<void>(&Tessellation3D::GetPointNo), "Get Total number of mesh generating points")
        .def("get_point_no", py::overload_cast<void>(&Tessellation3D::GetPointNo, py::const_), "Get Total number of mesh generating points - const version")
        .def("get_mesh_point", &Tessellation3D::GetMeshPoint, "Returns Position of mesh generating point", py::arg("index"))
        .def("get_area", &Tessellation3D::GetArea, "Returns Area of face", py::arg("index"))
        .def("get_all_area", &Tessellation3D::GetAllArea, "Get areas of all faces")
        .def("get_cell_cm", &Tessellation3D::GetCellCM, "Returns Position of Cell's Center of Mass", py::arg("index"))
        .def("get_total_faces_number", &Tessellation3D::GetTotalFacesNumber, "Returns the total number of faces")
        .def("get_width", &Tessellation3D::GetWidth, "Returns the effective width of a cell", py::arg("index"))
        .def("get_volume", &Tessellation3D::GetVolume, "Returns the volume of a cell", py::arg("index"))
        .def("get_cell_faces", &Tessellation3D::GetCellFaces, "Returns the indeces of a cell's Faces", py::arg("index"))
        .def("get_all_cell_faces", &Tessellation3D::GetAllCellFaces, "Get all cell faces")
        .def("access_mesh_points", &Tessellation3D::accessMeshPoints, "Returns a reference to the point vector")
        .def("get_mesh_points", &Tessellation3D::getMeshPoints, "Get all mesh points")
        .def("get_face_points", py::overload_cast<void>(&Tessellation3D::GetFacePoints), "Returns a reference to the points composing the faces vector")
        .def("get_face_points", py::overload_cast<void>(&Tessellation3D::GetFacePoints, py::const_),
                "Returns a reference to the points composing the faces vector - const version")
        .def("get_points_in_face", &Tessellation3D::GetPointsInFace,
                "Returns a reference to the indeces of the points composing a face. Points are order in a right hand fashion, normal pointing towards the first neighbor", py::arg("index"))
        .def("get_neighbors", py::overload_cast<size_t>(&Tessellation3D::GetNeighbors), "Returns a list of the neighbors of a cell", py::arg("index"))
        .def("get_neighbors", py::overload_cast<size_t, vector<size_t>&>(&Tessellation3D::GetNeighbors), "Returns a list of the neighbors of a cell", py::arg("index"), py::arg("res"))
        .def("clone", &Tessellation3D::clone, "Cloning function")
        .def("near_boundary", &Tessellation3D::NearBoundary, "Returns if the cell is adjacent to a boundary", py::arg("index"))
        .def("boundary_face", &Tessellation3D::BoundaryFace, "Returns if the face is a boundary one", py::arg("index"))
        .def("get_duplicated_points", py::overload_cast<void>(&Tessellation3D::GetDuplicatedPoints),
            "Returns the indeces of the points that were sent to other processors as ghost points")
        .def("get_duplicated_points", py::overload_cast<void>(&Tessellation3D::GetDuplicatedPoints, py::const_),
            "Returns the indeces of the points that were sent to other processors as ghost points - const version")
        .def("get_sent_procs", py::overload_cast<void>(&Tessellation3D::GetSentProcs),
            "Gets the list of parallel process to which points have been sent")
        .def("get_sent_procs", py::overload_cast<void>(&Tessellation3D::GetSentProcs, py::const_),
            "Gets the list of parallel process to which points have been sent - const version")
        .def("get_sent_points", py::overload_cast<void>(&Tessellation3D::GetSentPoints),
            "Get Indices of points sent to other parallel processes")
        .def("get_sent_points", py::overload_cast<void>(&Tessellation3D::GetSentPoints, py::const_),
            "Get Indices of points sent to other parallel processes - const version")
        .def("get_self_index", py::overload_cast<void>(&Tessellation3D::GetSelfIndex), "Get real index of points")
        .def("get_self_index", py::overload_cast<void>(&Tessellation3D::GetSelfIndex, py::const_), "Get real index of points - const version")
        .def("get_total_point_number", &Tessellation3D::GetTotalPointNumber, "Returns the total number of points (including ghost)")
        .def("get_all_cm", py::overload_cast<void>(&Tessellation3D::GetAllCM), "Returns the center of masses of the cells")
        .def("get_all_cm", py::overload_cast<void>(&Tessellation3D::GetAllCM, py::const_), "Returns the center of masses of the cells - const version")
        .def("get_all_volumes", py::overload_cast<void>(&Tessellation3D::GetAllVolumes), "Returns the volumes of the cells")
        .def("get_all_volumes", py::overload_cast<void>(&Tessellation3D::GetAllVolumes, py::const_), "Returns the volumes of the cells - const version")
        .def("get_neighbor_neighbors", &Tessellation3D::GetNeighborNeighbors,
            "Returns the neighbors and neighbors of the neighbors of a cell", py::arg("result"), py::arg("point"))
        .def("get_face_neighbors", &Tessellation3D::GetFaceNeighbors, "Get the indices of neighbours of a face", py::arg("face_index"))
        .def("get_all_face_neighbors", &Tessellation3D::GetAllFaceNeighbors, "Retrieve all neighbouring points who share a face")
        .def("normal", &Tessellation3D::Normal,
            "Returns a vector normal to the face whose magnitude is the seperation between the neighboring points", py::arg("faceindex"))
        .def("is_ghost_point", &Tessellation3D::IsGhostPoint, "Checks if a point is a ghost point or not", py::arg("index"))
        .def("calc_face_velocity", &Tessellation3D::CalcFaceVelocity, "Calculates the velocity of a face",
            py::arg("index"), py::arg("v0"), py::arg("v1"))
        .def("get_all_face_cm", &Tessellation3D::GetAllFaceCM, "Calculates the centres of mass of all faces", py::arg("index"))
        .def("face_cm", &Tessellation3D::FaceCM, "Return the centre of mass of a face", py::arg("index"))
        .def("get_ghost_indeces", py::overload_cast<void>(&Tessellation3D::GetGhostIndeces), "Get indices of ghost points")
        .def("get_ghost_indeces", py::overload_cast<void>(&Tessellation3D::GetGhostIndeces, py::const_),
            "Get indices of ghost points - const version")
        .def("get_box_coordinates", &Tessellation3D::GetBoxCoordinates, "Get the coordinate of opposite corners of the boundary")
        .def("build_no_box", &Tessellation3D::BuildNoBox, "Build tessellatoin without a box", py::arg("points"), py::arg("ghosts"), py::arg("toduplicate"))
        .def("is_point_outside_box", &Tessellation3D::IsPointOutsideBox, "Checks if a point is inside the box", py::arg("index"))
        .def("output", &Tessellation3D::output, "Write tessellation to file", py::arg("filename"))
        .def("set_box", &Tessellation3D::SetBox, "Adjust the boundary", py::arg("ll"), py::arg("ur"))
        .def("modify_box_faces", &Tessellation3D::ModifyBoxFaces, "Access method to box faces")
        .def("get_box_faces", &Tessellation3D::GetBoxFaces, "Access method to box faces");
}

PYBIND11_MODULE(MODULE_NAME, module)
{
    module.doc() = std::string("This module contains the class '") + CLASS_NAME + std::string("'.");
    // todo bind vector3D
    __exportTessellation3D(module);
    module.def("vector_values", &VectorValues, "Creates a subset of a vector of points", py::arg("v"), py::arg("index"));
    module.attr(PYTHON_VERSION_ATTR) = (PYBIND_DEVELOPING == 1)? PYBIND_DEVELOPING_VERSION : std::string(MODULE_VERSION);
}