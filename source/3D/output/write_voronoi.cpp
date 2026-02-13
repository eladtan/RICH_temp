#include "write3D.hpp"

struct Voronoi_VTU_Output
{
    std::vector<std::vector<double>> vtu_cell_variables;
    std::vector<std::string> vtu_cell_variable_names;
    std::vector<std::vector<std::string>> vtu_cell_strings;
    std::vector<std::string> vtu_cell_strings_names;
    std::vector<std::string> vtu_cell_vectors_names;
    std::vector<std::vector<Vector3D>> vtu_cell_vectors;
    std::vector<std::pair<std::string, double>> vtu_scalar_values;
};

void writeVTU(const std::string &filename, const Tessellation3D &tri, const Voronoi_VTU_Output &data)
{
    std::filesystem::path vtu_name(filename);
    vtu_name.replace_extension("vtu");
    write_vtu3d::write_vtu_3d(vtu_name, data.vtu_cell_variable_names, data.vtu_cell_variables, data.vtu_cell_strings_names, data.vtu_cell_strings, data.vtu_cell_vectors_names, data.vtu_cell_vectors, data.vtu_scalar_values, tri);
}

Voronoi_VTU_Output WriteVoronoiHelper(HDF5Writer &writer, const std::string &prefix, const Voronoi3D &tri, const std::vector<std::vector<double>> &data, const std::vector<std::string> &names, const std::vector<std::vector<std::string>> &dataStr, const std::vector<std::string> &namesStr, bool write_vtu)
{
    Voronoi_VTU_Output vtu;
    std::vector<std::vector<double>> &vtu_cell_variables = vtu.vtu_cell_variables;
    std::vector<std::string> &vtu_cell_variable_names = vtu.vtu_cell_variable_names;
    std::vector<std::vector<std::string>> &vtu_cell_strings = vtu.vtu_cell_strings;
    std::vector<std::string> &vtu_cell_strings_names = vtu.vtu_cell_strings_names;
    std::vector<std::string> &vtu_cell_vectors_names = vtu.vtu_cell_vectors_names;
    std::vector<std::vector<Vector3D>> &vtu_cell_vectors = vtu.vtu_cell_vectors;

    assert(data.size() == names.size());
    for(size_t i = 0; i < data.size(); ++i)
    {
        assert(data[i].size() == names[i].size());
        writer.WriteElement(prefix + "/" + names[i], data[i]);
        vtu_cell_variables.push_back(data[i]);
        vtu_cell_variable_names.push_back(names[i]);
    }

    assert(dataStr.size() == namesStr.size());
    for(size_t i = 0; i < dataStr.size(); ++i)
    {
        assert(dataStr[i].size() == namesStr[i].size());
        // write_std_vector_to_hdf5(writegroup, data[i], names[i]); TODO: !!!
        vtu_cell_strings.push_back(dataStr[i]);
        vtu_cell_strings_names.push_back(namesStr[i]);
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

    writer.WriteElement(prefix + "/mesh_point_x", x);
    writer.WriteElement(prefix + "/mesh_point_y", y);
    writer.WriteElement(prefix + "/mesh_point_z", z);
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
    writer.WriteElement(prefix + "/all_mesh_point_x", x);
    writer.WriteElement(prefix + "/all_mesh_point_y", y);
    writer.WriteElement(prefix + "/all_mesh_point_z", z);

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
    writer.WriteElement(prefix + "/Number_of_faces_in_cell", Nfaces);
    writer.WriteElement(prefix + "/Faces_in_cell", FacesInCell);
    Npoints = tri.GetFacePoints().size();
    for(size_t i = 0; i < Npoints; ++i)
    {
        vx.push_back(tri.GetFacePoints()[i].x);
        vy.push_back(tri.GetFacePoints()[i].y);
        vz.push_back(tri.GetFacePoints()[i].z);
    }
    writer.WriteElement(prefix + "/vertice_x", vx);
    writer.WriteElement(prefix + "/vertice_y", vy);
    writer.WriteElement(prefix + "/vertice_z", vz);
    Npoints = tri.GetTotalFacesNumber();
    for(size_t i = 0; i < Npoints; ++i)
    {
        Nvert.push_back(tri.GetPointsInFace(i).size());
        for(size_t j = 0; j < Nvert.back(); ++j)
        {
            VerticesInFace.push_back(tri.GetPointsInFace(i)[j]);
        }
    }
    writer.WriteElement(prefix + "/Number_of_vertices_in_face", Nvert);
    writer.WriteElement(prefix + "/Vertices_in_face", VerticesInFace);

    return vtu;
}

#if RICH_MPI
    void WriteVoronoiParallel(const Voronoi3D &tri, const std::string &filename,
                            const std::vector<std::vector<double>> &data, const std::vector<std::string>& names,
                            const std::vector<std::vector<std::string>> &dataStr, const std::vector<std::string>& namesStr,
                            const std::vector<std::pair<std::string, double>> &scalar_values, bool write_vtu)
    {
        int rank = 0;
        int ws = 0;

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
        HDF5Writer filewriter(myFilePath);
        std::shared_ptr<HDF5Writer> globalFileWriter = (rank == 0) ? std::make_shared<HDF5Writer>(filename) : nullptr;

        if(rank == 0)
        {
            std::vector<double> box(6);
            box[0] = tri.GetBoxCoordinates().first.x;
            box[1] = tri.GetBoxCoordinates().first.y;
            box[2] = tri.GetBoxCoordinates().first.z;
            box[3] = tri.GetBoxCoordinates().second.x;
            box[4] = tri.GetBoxCoordinates().second.y;
            box[5] = tri.GetBoxCoordinates().second.z;
            globalFileWriter->WriteElement("/Box", box);
        }

        Voronoi_VTU_Output vtu = WriteVoronoiHelper(filewriter, "", tri, data, names, dataStr, namesStr, write_vtu);
        vtu.vtu_scalar_values = scalar_values;

        writeVTU(filename, tri, vtu);

        MPI_Barrier(MPI_COMM_WORLD);
        // only rank 0 makes the shared file
        if(rank == 0)
        {
            for(int _rank = 0; _rank < ws; _rank++)
            {
                // merge `_rank`'s file
                std::string rankFile((ranks_files_path / std::to_string(_rank)).string() + ".h5");
                globalFileWriter->AddExternalLink(rankFile, "/", "/rank" + std::to_string(_rank));
            }
        }
    }
#endif // RICH_MPI

void WriteVoronoi(const Voronoi3D &tri, const std::string &filename,
                        const std::vector<std::vector<double>> &data, const std::vector<std::string>& names,
                        const std::vector<std::vector<std::string>> &dataStr, const std::vector<std::string>& namesStr,
                        const std::vector<std::pair<std::string, double>> &scalar_values, bool write_vtu)
{
    #ifdef RICH_MPI
        int rank = 0;
        int ws = 0; // MPI_COMM_WORLD size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
    #endif

    std::shared_ptr<HDF5Writer> filewriter = nullptr;

    #ifdef RICH_MPI
        if(rank == 0)
        {
        #endif // RICH_MPI
            filewriter = std::make_shared<HDF5Writer>(filename);
        #ifdef RICH_MPI
        }
    #endif // RICH_MPI

    std::string prefix = "";

    #ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        int dummy = 0;
        if(rank > 0)
        {
            MPI_Recv(&dummy, 1, MPI_INT, rank - 1, HDF5_WRITE_BLOCK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            filewriter = std::make_shared<HDF5Writer>(filename, false /* don't truncate */);
        }
        prefix = "/rank" + std::to_string(rank);
    #else
        prefix = "";
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
        filewriter->WriteElement("/Box", box);
    #ifdef RICH_MPI
        }
    #endif // RICH_MPI

    Voronoi_VTU_Output vtu = WriteVoronoiHelper(*filewriter, prefix, tri, data, names, dataStr, namesStr, write_vtu);
    vtu.vtu_scalar_values = scalar_values;

    filewriter->Close();
    #ifdef RICH_MPI
        if(rank < (ws - 1))
        {
            int dummy = 0;
            MPI_Send(&dummy, 1, MPI_INT, rank + 1, HDF5_WRITE_BLOCK_TAG, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    #endif
    writeVTU(filename, tri, vtu);
}

void WriteVoronoiVTKOnly(const Voronoi3D &tri, const std::string &filename,
                          const std::vector<std::vector<double>> &data, const std::vector<std::string>& names,
                          const std::vector<std::vector<std::string>> &dataStr, const std::vector<std::string>& namesStr,
                          const std::vector<std::pair<std::string, double>> &scalar_values)
{
    Voronoi_VTU_Output vtu;
    std::vector<std::vector<double>> &vtu_cell_variables = vtu.vtu_cell_variables;
    std::vector<std::string> &vtu_cell_variable_names = vtu.vtu_cell_variable_names;
    std::vector<std::vector<std::string>> &vtu_cell_strings = vtu.vtu_cell_strings;
    std::vector<std::string> &vtu_cell_strings_names = vtu.vtu_cell_strings_names;
    std::vector<std::string> &vtu_cell_vectors_names = vtu.vtu_cell_vectors_names;
    std::vector<std::vector<Vector3D>> &vtu_cell_vectors = vtu.vtu_cell_vectors;
    vtu.vtu_scalar_values = scalar_values;

    assert(data.size() == names.size());
    for(size_t i = 0; i < data.size(); ++i)
    {
        vtu_cell_variables.push_back(data[i]);
        vtu_cell_variable_names.push_back(names[i]);
    }

    assert(dataStr.size() == namesStr.size());
    for(size_t i = 0; i < dataStr.size(); ++i)
    {
        vtu_cell_strings.push_back(dataStr[i]);
        vtu_cell_strings_names.push_back(namesStr[i]);
    }

    vector<double> x, y, z, vx, vy, vz;
    vector<size_t> Nfaces;
    vector<size_t> Nvert;
    vector<size_t> FacesInCell;
    vector<size_t> VerticesInFace;
    size_t Npoints = tri.GetPointNo();

    vtu_cell_vectors_names.push_back("Coordinates");
    std::vector<Vector3D> vel(Npoints);
    for(size_t i = 0; i < Npoints; ++i)
        vel[i] = tri.GetMeshPoint(i);
    vtu_cell_vectors.push_back(vel);

    std::vector<double> temp(Npoints);

    for(size_t i = 0; i < Npoints; ++i)
    {
        temp[i] = i;
    }
    vtu_cell_variables.push_back(temp);
    vtu_cell_variable_names.push_back("Point Index");

    writeVTU(filename, tri, vtu);
}

void WriteVoronoiSerial(const Voronoi3D &tri, const std::string &filename,
                        const std::vector<std::vector<double>> &data, const std::vector<std::string>& names,
                        const std::vector<std::vector<std::string>> &dataStr, const std::vector<std::string>& namesStr,
                        const std::vector<std::pair<std::string, double>> &scalar_values, bool write_vtu)
{
    HDF5Writer filewriter(filename, true /* truncate */);
    std::vector<double> box(6);
    box[0] = tri.GetBoxCoordinates().first.x;
    box[1] = tri.GetBoxCoordinates().first.y;
    box[2] = tri.GetBoxCoordinates().first.z;
    box[3] = tri.GetBoxCoordinates().second.x;
    box[4] = tri.GetBoxCoordinates().second.y;
    box[5] = tri.GetBoxCoordinates().second.z;
    filewriter.WriteElement("/Box", box);

    Voronoi_VTU_Output vtu = WriteVoronoiHelper(filewriter, "", tri, data, names, dataStr, namesStr, write_vtu);
    vtu.vtu_scalar_values = scalar_values;

    writeVTU(filename, tri, vtu);
}
