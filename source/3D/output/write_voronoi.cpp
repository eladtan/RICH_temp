#include "write3D.hpp"

#ifdef RICH_MPI
    #include "utils/hdf5/HDF5WriterParallel.hpp"
#endif

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
namespace
{
    void writeBoxGlobal(HDF5Writer &writer, const Voronoi3D &tri)
    {
        std::vector<double> box(6);
        box[0] = tri.GetBoxCoordinates().first.x;
        box[1] = tri.GetBoxCoordinates().first.y;
        box[2] = tri.GetBoxCoordinates().first.z;
        box[3] = tri.GetBoxCoordinates().second.x;
        box[4] = tri.GetBoxCoordinates().second.y;
        box[5] = tri.GetBoxCoordinates().second.z;
        writer.WriteElement("/Box", box);
    }
}

void WriteVoronoiParallel(const Voronoi3D &tri, const std::string &filename,
                        const std::vector<std::vector<double>> &data, const std::vector<std::string>& names,
                        const std::vector<std::vector<std::string>> &dataStr, const std::vector<std::string>& namesStr,
                        const std::vector<std::pair<std::string, double>> &scalar_values, bool write_vtu)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    HDF5WriterParallel pwriter(filename, MPI_COMM_WORLD);
    HDF5Writer writer(pwriter.GetFileId());

    if(rank == 0)
    {
        writeBoxGlobal(writer, tri);
    }

    std::string rankPrefix = pwriter.GetPrefix();
    Voronoi_VTU_Output vtu = WriteVoronoiHelper(writer, rankPrefix, tri, data, names, dataStr, namesStr, write_vtu);
    vtu.vtu_scalar_values = scalar_values;

    writeVTU(filename, tri, vtu);
}
#endif // RICH_MPI

void WriteVoronoi(const Voronoi3D &tri, const std::string &filename,
                        const std::vector<std::vector<double>> &data, const std::vector<std::string>& names,
                        const std::vector<std::vector<std::string>> &dataStr, const std::vector<std::string>& namesStr,
                        const std::vector<std::pair<std::string, double>> &scalar_values, bool write_vtu)
{
    #ifdef RICH_MPI
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        HDF5WriterParallel pwriter(filename, MPI_COMM_WORLD);
        HDF5Writer writer(pwriter.GetFileId());

        if(rank == 0)
        {
            std::vector<double> box(6);
            box[0] = tri.GetBoxCoordinates().first.x;
            box[1] = tri.GetBoxCoordinates().first.y;
            box[2] = tri.GetBoxCoordinates().first.z;
            box[3] = tri.GetBoxCoordinates().second.x;
            box[4] = tri.GetBoxCoordinates().second.y;
            box[5] = tri.GetBoxCoordinates().second.z;
            writer.WriteElement("/Box", box);
        }

        std::string prefix = pwriter.GetPrefix();
        Voronoi_VTU_Output vtu = WriteVoronoiHelper(writer, prefix, tri, data, names, dataStr, namesStr, write_vtu);
        vtu.vtu_scalar_values = scalar_values;
    #else
        HDF5Writer writer(filename);
        std::vector<double> box(6);
        box[0] = tri.GetBoxCoordinates().first.x;
        box[1] = tri.GetBoxCoordinates().first.y;
        box[2] = tri.GetBoxCoordinates().first.z;
        box[3] = tri.GetBoxCoordinates().second.x;
        box[4] = tri.GetBoxCoordinates().second.y;
        box[5] = tri.GetBoxCoordinates().second.z;
        writer.WriteElement("/Box", box);

        Voronoi_VTU_Output vtu = WriteVoronoiHelper(writer, "", tri, data, names, dataStr, namesStr, write_vtu);
        vtu.vtu_scalar_values = scalar_values;
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
