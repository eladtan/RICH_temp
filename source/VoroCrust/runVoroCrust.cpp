#include "runVoroCrust.hpp"
#include "vorocrust_vtk.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <array>
#include <fenv.h>
#include <omp.h>

using Path = std::filesystem::path;

namespace
{
    std::vector<Vector3D> read_vertices(std::string filename)
    {
        std::fstream myfile(filename, std::ios_base::in);

        auto number_of_lines = std::count(std::istreambuf_iterator<char>(myfile), std::istreambuf_iterator<char>(), '\n');
        // std::cout << "Reading vertices data from file: " << filename << ", number of lines (vertices): " << number_of_lines << std::endl;
        myfile.clear();
        myfile.seekg(0);
        std::vector<Vector3D> vertices_from_file = std::vector<Vector3D>(number_of_lines, Vector3D());

        double x, y, z;
        std::size_t i = 0;
        while (myfile >> x)
        {
            myfile >> y;
            myfile >> z;

            vertices_from_file[i].Set(x, y, z);
            i++;
        }

        return vertices_from_file;
    }

    std::vector<std::vector<std::size_t>> read_faces(std::string filename)
    {
        std::fstream myfile(filename, std::ios_base::in);

        auto number_of_lines = std::count(std::istreambuf_iterator<char>(myfile), std::istreambuf_iterator<char>(), '\n');
        // std::cout << "Reading vertices data from file: " << filename << ", number of lines (vertices): " << number_of_lines << "\n";

        myfile.clear();
        myfile.seekg(0);

        std::vector<std::vector<std::size_t>> faces_from_file = std::vector<std::vector<std::size_t>>(number_of_lines, std::vector<std::size_t>({0, 0, 0}));

        double v1, v2, v3;

        std::size_t i = 0;
        while (myfile >> v1)
        {

            myfile >> v2;
            myfile >> v3;

            faces_from_file[i][0] = static_cast<std::size_t>(v1);
            faces_from_file[i][1] = static_cast<std::size_t>(v2);
            faces_from_file[i][2] = static_cast<std::size_t>(v3);
            i++;
        }

        return faces_from_file;
    }

    void write_points(std::string filename, std::vector<Vector3D> const &points)
    {
        std::ofstream myfile_x(filename + "_x.txt");
        std::ofstream myfile_y(filename + "_y.txt");
        std::ofstream myfile_z(filename + "_z.txt");

        for (auto const &p : points)
        {
            myfile_x << std::setprecision(15) << p.x << "\n";
            myfile_y << std::setprecision(15) << p.y << "\n";
            myfile_z << std::setprecision(15) << p.z << "\n";
        }

        myfile_x.close();
        myfile_y.close();
        myfile_z.close();
    }

    void write_seeds(std::string filename, std::vector<Seed> const &seeds)
    {
        std::ofstream myfile_x(filename + "_x.txt");
        std::ofstream myfile_y(filename + "_y.txt");
        std::ofstream myfile_z(filename + "_z.txt");
        std::ofstream myfile_r(filename + "_r.txt");

        for (auto const &seed : seeds)
        {
            myfile_x << std::setprecision(15) << seed.p.x << "\n";
            myfile_y << std::setprecision(15) << seed.p.y << "\n";
            myfile_z << std::setprecision(15) << seed.p.z << "\n";
            myfile_r << std::setprecision(15) << seed.radius << "\n";
        }

        myfile_x.close();
        myfile_y.close();
        myfile_z.close();
        myfile_r.close();
    }

    void from_data_create_boundary_seeds(Path const &dirpath, Path const &output_path, double const theta, double const maxRadius, double const L_lip, double const alpha, std::size_t const max_num_iter)
    {
        std::cout << "Read Data" << std::endl;

        Path vertices_file_path = dirpath / "vertices.txt";

        std::cout << "Read vertices from file: " << vertices_file_path << std::endl;

        auto vertices_from_file = read_vertices(vertices_file_path);

        Path faces_file_path = dirpath / "faces.txt";

        std::cout << "Read faces from file: " << faces_file_path << std::endl;

        auto faces_from_file = read_faces(faces_file_path);

        PL_Complex plc(vertices_from_file);
        for (auto const &indices : faces_from_file)
        {
            plc.addFace(indices);
        }

        VoroCrustAlgorithm alg(plc, theta, theta, maxRadius, L_lip, alpha, max_num_iter, static_cast<std::size_t>(1e5), static_cast<std::size_t>(1e6));

        std::filesystem::create_directories(output_path);
        vorocrust_vtk::write_vtu_PL_Complex(output_path / "all.vtu", *alg.plc);

        alg.run();
        alg.dump(output_path / "dump");

        std::cout << "Finished Creating Balls" << std::endl;
    }

    void load_data_from_dump(Path const &dirdata, Path const &dirdump, Path const &output_path, double const theta, double const maxRadius, double const L_lip, double const alpha, std::size_t const max_num_iter)
    {
        std::cout << "Load from Dump: " << dirdump << std::endl;

        Path vertices_file_path = dirdata / "vertices.txt";

        std::cout << "Read vertices from file: " << vertices_file_path << std::endl;

        auto vertices_from_file = read_vertices(vertices_file_path);

        Path faces_file_path = dirdata / "faces.txt";

        std::cout << "Read faces from file: " << faces_file_path << std::endl;

        auto faces_from_file = read_faces(faces_file_path);

        PL_Complex plc(vertices_from_file);
        for (auto const &indices : faces_from_file)
        {
            plc.addFace(indices);
        }

        VoroCrustAlgorithm alg(plc, theta, theta, maxRadius, L_lip, alpha, max_num_iter, static_cast<std::size_t>(1e5), static_cast<std::size_t>(1e6));

        alg.load_dump(dirdump);

        try
        {
            vorocrust_vtk::write_ballTree(output_path / "corner_sampling_all.vtp", alg.trees.ball_kd_vertices);
            vorocrust_vtk::write_ballTree(output_path / "edges_sampling_all.vtp", alg.trees.ball_kd_edges);
            vorocrust_vtk::write_ballTree(output_path / "faces_sampling_all.vtp", alg.trees.ball_kd_faces);
        }
        catch (std::bad_alloc &exception)
        {
            std::cout << "bad alloc caught: " << exception.what() << std::endl;
        }

        auto const &seeds = alg.getSeeds();
        dumpSeeds(dirdump / "all_seeds", seeds);

        std::vector<Vector3D> seeds_points;
        seeds_points.reserve(seeds.size() + 1);

        for (auto const &seed : seeds)
        {
            seeds_points.push_back(seed.p);
        }

        try
        {
            vorocrust_vtk::write_points(output_path / "seeds_all.vtu", seeds_points);
            vorocrust_vtk::write_vtu_trees(output_path / "trees_all.vtu", alg.trees);
        }
        catch (std::bad_alloc &exception)
        {
            std::cout << "bad alloc caught: " << exception.what() << std::endl;
        }
    }

    void from_data_determine(Path const &dirdata, Path const &dirdump, Path const &output_path, double const theta, double const maxRadius, double const L_lip, double const alpha, std::size_t const max_num_iter)
    {
        std::cout << "Determine the location of the seeds" << std::endl;

        std::vector<PL_Complex> zones_plcs;

        Path vertices_file_path = dirdata / "vertices.txt";

        std::cout << "Read vertices from file: " << vertices_file_path << std::endl;

        auto vertices_from_file = read_vertices(vertices_file_path);

        Path faces_file_path = dirdata / "faces.txt";

        std::cout << "Read faces from file: " << faces_file_path << std::endl;

        auto faces_from_file = read_faces(faces_file_path);

        PL_Complex plc(vertices_from_file);
        for (auto const &indices : faces_from_file)
        {
            plc.addFace(indices);
        }

        plc.detectFeatures(theta, theta);
        vorocrust_vtk::write_vtu_PL_Complex(output_path / "determine.vtu", plc);

        zones_plcs.push_back(plc);

        auto seeds = load_dumpSeeds(dirdump / "all_seeds");

        auto zone_seeds = determineZoneOfSeeds(seeds, zones_plcs);

        dumpSeeds(dirdump / "zone_in_seeds", zone_seeds[0]);
        dumpSeeds(dirdump / "zone_out_seeds", zone_seeds[1]);
    }

    std::pair<std::vector<std::vector<Vector3D> >, std::vector<std::vector<Vector3D> >>  from_data_load_seeds_and_create_volume_seeds(Path const &dirdata, Path const &dirdump, Path const &output_path, double const theta, double const maxRadius, double const L_lip, 
        double const alpha, std::size_t const max_num_iter, VoroCrustAlgorithm *& alg, std::vector<PL_Complex> const& zones_plcs)
    {
        std::vector<std::vector<Seed>> zones_seeds;
        zones_seeds.push_back(load_dumpSeeds(dirdump / "zone_in_seeds"));
        zones_seeds.push_back(load_dumpSeeds(dirdump / "zone_out_seeds"));

        alg = new VoroCrustAlgorithm(zones_plcs[0], theta, theta, maxRadius, L_lip, alpha, max_num_iter, static_cast<std::size_t>(1e5), static_cast<std::size_t>(1e6));
        std::cout<<"Done VoroCrustAlgorithm build"<<std::endl;
        alg->load_dump(dirdump);
        std::cout<<"Done VoroCrustAlgorithm load"<<std::endl;
        auto const &zones_volume_seeds = alg->randomSampleSeeds(zones_plcs, zones_seeds, maxRadius);
        std::cout<<"starting dump"<<std::endl;
        if(zones_volume_seeds.size() > 0)
            dumpSeeds(dirdump / "zone_in_volume_seeds", zones_volume_seeds[0]);
        std::cout<<"finished dump"<<std::endl;
        std::vector<std::vector<Vector3D>> volume_seeds(zones_volume_seeds.size());
        for(size_t i = 0; i < zones_volume_seeds.size(); ++i)
        {
            volume_seeds[i].resize(zones_volume_seeds[i].size());
            for(size_t j = 0; j < zones_volume_seeds[i].size(); ++j)
                volume_seeds[i][j] = zones_volume_seeds[i][j].p;
        }
        std::vector<std::vector<Vector3D>> boundary_seeds(zones_seeds.size());
        for(size_t i = 0; i < boundary_seeds.size(); ++i)
        {
            boundary_seeds[i].resize(zones_seeds[i].size());
            for(size_t j = 0; j < boundary_seeds[i].size(); ++j)
                boundary_seeds[i][j] = zones_seeds[i][j].p;
        }
        return std::pair<std::vector<std::vector<Vector3D> >, std::vector<std::vector<Vector3D> >> (volume_seeds, boundary_seeds);
    }
}

std::pair<std::vector<std::vector<Vector3D> >, std::vector<std::vector<Vector3D> >>  runVoroCrust(std::filesystem::path const& data_directory_path, std::filesystem::path const& output_directory, VoroCrustAlgorithm *& alg, std::vector<PL_Complex> &zones_plcs, bool restart)
{
    std::cout << "RUNNING runVoroCrust!!" << std::endl;

    Path dirpath = data_directory_path;
    Path dir_output = output_directory;
    Path dirdump = dir_output / "dump";
    double theta = M_PI * 45. / 180.;
    double maxRadius = 1.0;
    double L_lip = 0.3;
    double alpha = 0.13;
    std::size_t max_num_iter = 1;

    if(not restart)
    {
        from_data_create_boundary_seeds(dirpath, dir_output, theta, maxRadius, L_lip, alpha, max_num_iter);
        load_data_from_dump(dirpath, dirdump, dir_output, theta, maxRadius, L_lip, alpha, max_num_iter);
        from_data_determine(dirpath, dirdump, dir_output, theta, maxRadius, L_lip, alpha, max_num_iter);
    }

    Path vertices_file_path = dirpath / "vertices.txt";

    std::cout << "Read vertices from file: " << vertices_file_path << std::endl;

    auto vertices_from_file = read_vertices(vertices_file_path);

    Path faces_file_path = dirpath / "faces.txt";

    std::cout << "Read faces from file: " << faces_file_path << std::endl;

    auto faces_from_file = read_faces(faces_file_path);

    PL_Complex plc(vertices_from_file);
    for (auto const &indices : faces_from_file)
    {
        plc.addFace(indices);
    }
    plc.detectFeatures(theta, theta);
    zones_plcs.clear();
    zones_plcs.push_back(plc);

    return from_data_load_seeds_and_create_volume_seeds(dirpath, dirdump, dir_output, theta, maxRadius, L_lip, alpha, max_num_iter, alg, zones_plcs);
}
