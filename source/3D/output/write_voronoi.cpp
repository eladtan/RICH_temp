#include "write3D.hpp"

struct VTU_Output
{
    std::vector<std::vector<double>> vtu_cell_variables;
    std::vector<std::string> vtu_cell_variable_names;
    std::vector<std::string> vtu_cell_vectors_names;
    std::vector<std::vector<Vector3D>> vtu_cell_vectors;
};

void writeVTU(const std::string &filename, const Tessellation3D &tri, const VTU_Output &data)
{
    std::filesystem::path vtu_name(filename);
    vtu_name.replace_extension("vtu");
    write_vtu3d::write_vtu_3d(vtu_name, data.vtu_cell_variable_names, data.vtu_cell_variables, data.vtu_cell_vectors_names, data.vtu_cell_vectors, tri);
}

VTU_Output WriteVoronoiHelper(H5File &file, Group &writegroup, const std::string &filename, const Voronoi3D &tri, const std::vector<std::vector<double>> &data, const std::vector<std::string> &names, bool write_vtu)
{
    VTU_Output vtu;
    std::vector<std::vector<double>> &vtu_cell_variables = vtu.vtu_cell_variables;
    std::vector<std::string> &vtu_cell_variable_names = vtu.vtu_cell_variable_names;
    std::vector<std::string> &vtu_cell_vectors_names = vtu.vtu_cell_vectors_names;
    std::vector<std::vector<Vector3D>> &vtu_cell_vectors = vtu.vtu_cell_vectors;

    for(size_t i = 0; i < data.size(); ++i)
    {
        write_std_vector_to_hdf5(writegroup, data[i], names[i]);
        vtu_cell_variables.push_back(data[i]);
        vtu_cell_variable_names.push_back(names[i]);
    }

    vector<double> x, y, z, vx, vy, vz;
    vector<size_t> Nfaces;
    vector<size_t> Nvert;
    vector<size_t> FacesInCell;
    vector<size_t> VerticesInFace;
    size_t Npoints = tri.GetPointNo();

    if(write_vtu)
    {
        vtu_cell_vectors_names.push_back("Coordinates");
        std::vector<Vector3D> vel(Npoints);
        for(size_t i = 0; i < Npoints; ++i)
            vel[i] = tri.GetMeshPoint(i);
        vtu_cell_vectors.push_back(vel);
    }

    for(size_t i = 0; i < Npoints; ++i)
    {
        const Vector3D &point = tri.GetMeshPoint(i);
        x.push_back(point.x);
        y.push_back(point.y);
        z.push_back(point.z);
    }

    write_std_vector_to_hdf5(writegroup, x, "mesh_point_x");
    write_std_vector_to_hdf5(writegroup, y, "mesh_point_y");
    write_std_vector_to_hdf5(writegroup, z, "mesh_point_z");
    x.clear();
    y.clear();
    z.clear();

    for(size_t i = 0; i < tri.GetTotalPointNumber(); ++i)
    {
        const Vector3D &point = tri.GetMeshPoint(i);
        x.push_back(point.x);
        y.push_back(point.y);
        z.push_back(point.z);
    }
    write_std_vector_to_hdf5(writegroup, x, "all_mesh_point_x");
    write_std_vector_to_hdf5(writegroup, y, "all_mesh_point_y");
    write_std_vector_to_hdf5(writegroup, z, "all_mesh_point_z");

    if(write_vtu)
    {
        std::vector<double> temp(Npoints);

        for(size_t i = 0; i < Npoints; ++i)
        {
            temp[i] = i;
        }
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("Point Index");
    }

    for(size_t i = 0; i < Npoints; ++i)
    {
        const face_vec &face = tri.GetCellFaces(i);
        Nfaces.push_back(face.size());
        for(size_t j = 0; j < Nfaces.back(); ++j)
        {
            FacesInCell.push_back(face[j]);
        }
    }
    IntType datatype(PredType::NATIVE_ULLONG);
    datatype.setOrder(H5T_ORDER_LE);
    write_std_vector_to_hdf5(writegroup, Nfaces, "Number_of_faces_in_cell", datatype);
    write_std_vector_to_hdf5(writegroup, FacesInCell, "Faces_in_cell", datatype);
    Npoints = tri.GetFacePoints().size();
    for(size_t i = 0; i < Npoints; ++i)
    {
        vx.push_back(tri.GetFacePoints()[i].x);
        vy.push_back(tri.GetFacePoints()[i].y);
        vz.push_back(tri.GetFacePoints()[i].z);
    }
    write_std_vector_to_hdf5(writegroup, vx, "vertice_x");
    write_std_vector_to_hdf5(writegroup, vy, "vertice_y");
    write_std_vector_to_hdf5(writegroup, vz, "vertice_z");
    Npoints = tri.GetTotalFacesNumber();
    for(size_t i = 0; i < Npoints; ++i)
    {
        Nvert.push_back(tri.GetPointsInFace(i).size());
        for(size_t j = 0; j < Nvert.back(); ++j)
        {
            VerticesInFace.push_back(tri.GetPointsInFace(i)[j]);
        }
    }
    write_std_vector_to_hdf5(writegroup, Nvert, "Number_of_vertices_in_face", datatype);
    write_std_vector_to_hdf5(writegroup, VerticesInFace, "Vertices_in_face", datatype);

    return vtu;
}

#if RICH_MPI
    void WriteVoronoiParallel(Voronoi3D const &tri, std::string const &filename, const std::vector<std::vector<double>> &data, const std::vector<std::string> &names, bool write_vtu)
    {
        int rank = 0;
        int ws = 0;
        H5File file;

        fs::path path = fs::absolute(filename).parent_path();
        std::string myFilePath;

        fs::path ranks_files_path = path / fs::path(filename).filename().replace_extension();
        if(not fs::exists(ranks_files_path))
        {
            fs::create_directory(ranks_files_path);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
        myFilePath = (ranks_files_path / std::to_string(rank)).string() + ".h5";

        // truncate my file and open it
        H5File file2(H5std_string(myFilePath), H5F_ACC_TRUNC);
        file2.close();
        file.openFile(H5std_string(myFilePath), H5F_ACC_RDWR);

        Group writegroup = file.openGroup("/");

        if(rank == 0)
        {
            std::vector<double> box(6);
            box[0] = tri.GetBoxCoordinates().first.x;
            box[1] = tri.GetBoxCoordinates().first.y;
            box[2] = tri.GetBoxCoordinates().first.z;
            box[3] = tri.GetBoxCoordinates().second.x;
            box[4] = tri.GetBoxCoordinates().second.y;
            box[5] = tri.GetBoxCoordinates().second.z;
            write_std_vector_to_hdf5(file, box, "Box");
        }

        VTU_Output vtu = WriteVoronoiHelper(file, writegroup, filename, tri, data, names, write_vtu);

        writegroup.close();
        file.close();
        writeVTU(filename, tri, vtu);

        MPI_Barrier(MPI_COMM_WORLD);
        // only rank 0 makes the shared file
        if(rank == 0)
        {
            file2 = H5File(H5std_string(filename), H5F_ACC_TRUNC);
            file2.close();
            hid_t shared_file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            for(int _rank = 0; _rank < ws; _rank++)
            {
                // merge `_rank`'s file
                std::string rankFile((ranks_files_path / std::to_string(_rank)).string() + ".h5");
                std::string rankGroupName("/rank" + std::to_string(_rank));
                //Group rankGroup = sharedFile.createGroup(rankGroupName);
                H5Lcreate_external(rankFile.c_str(),
                                  "/",
                                  shared_file_id,
                                  rankGroupName.c_str(),
                                  H5P_DEFAULT,
                                  H5P_DEFAULT);
                //rankGroup.close();
            }
            H5Fclose(shared_file_id);
        }
    }
#endif // RICH_MPI

void WriteVoronoi(Voronoi3D const &tri, std::string const &filename, const std::vector<std::vector<double>> &data, const std::vector<std::string> &names, bool write_vtu)
{
    #ifdef RICH_MPI
        int rank = 0;
        int ws = 0; // MPI_COMM_WORLD size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
    #endif

    H5File file;

    #ifdef RICH_MPI
        if(rank == 0)
        {
    #endif // RICH_MPI
            H5File file2(H5std_string(filename), H5F_ACC_TRUNC);
            file2.close();
            file.openFile(H5std_string(filename), H5F_ACC_RDWR);
    #ifdef RICH_MPI
        }
    #endif // RICH_MPI

    Group writegroup;
    #ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        int dummy = 0;
        if(rank > 0)
        {
            MPI_Recv(&dummy, 1, MPI_INT, rank - 1, HDF5_WRITE_BLOCK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            file.openFile(H5std_string(filename), H5F_ACC_RDWR);
        }
        file.createGroup("/rank" + std::to_string(rank));
        writegroup = file.openGroup("/rank" + std::to_string(rank));
    #else
        writegroup = file.openGroup("/");
    #endif

    #ifdef RICH_MPI
        if(rank == 0)
        {
    #endif // RICH_MPI
        std::vector<double> box(6);
        box[0] = tri.GetBoxCoordinates().first.x;
        box[1] = tri.GetBoxCoordinates().first.y;
        box[2] = tri.GetBoxCoordinates().first.z;
        box[3] = tri.GetBoxCoordinates().second.x;
        box[4] = tri.GetBoxCoordinates().second.y;
        box[5] = tri.GetBoxCoordinates().second.z;
        write_std_vector_to_hdf5(file, box, "Box");
    #ifdef RICH_MPI
        }
    #endif // RICH_MPI

    VTU_Output vtu = WriteVoronoiHelper(file, writegroup, filename, tri, data, names, write_vtu);

    #ifdef RICH_MPI
        writegroup.close();
        file.close();
        if(rank < (ws - 1))
        {
            int dummy = 0;
            MPI_Send(&dummy, 1, MPI_INT, rank + 1, HDF5_WRITE_BLOCK_TAG, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    #else
        file.close();
    #endif
    writeVTU(filename, tri, vtu);
}

void WriteVoronoiSerial(Voronoi3D const &tri, std::string const &filename, const std::vector<std::vector<double>> &data, const std::vector<std::string> &names, bool writevtu)
{
    H5File file(H5std_string(filename), H5F_ACC_TRUNC);
    std::vector<double> box(6);
    box[0] = tri.GetBoxCoordinates().first.x;
    box[1] = tri.GetBoxCoordinates().first.y;
    box[2] = tri.GetBoxCoordinates().first.z;
    box[3] = tri.GetBoxCoordinates().second.x;
    box[4] = tri.GetBoxCoordinates().second.y;
    box[5] = tri.GetBoxCoordinates().second.z;
    write_std_vector_to_hdf5(file, box, "Box");

    Group writegroup = file.openGroup("/");
    VTU_Output vtu = WriteVoronoiHelper(file, writegroup, filename, tri, data, names, writevtu);
    writeVTU(filename, tri, vtu);
    writegroup.close();
    file.close();
}
