#include <iostream>
#include <random>
#include <filesystem>
#include "3D/range/finders/OctTree.hpp"
#include "3D/tessellation/voronoi/Voronoi3D.hpp"
#include "3D/output/write3D.hpp"
#include "misc/simple_io.hpp"
#include "misc/mesh_generator3D.hpp"
#include "utils/io/read.h"
#include "utils/io/out.h"
#include "utils/debug/vtune.h"
#include "utils/advance/advance.hpp"
#include "utils/dup/duplications.hpp"
#include "utils/validation/validate.hpp"
#include "utils/ghost/printGhostPoints.hpp"
#include "mpi/serialize/mpi_commands.hpp"

#ifndef RICH_MPI
using rank_t = int;
#endif // RICH_MPI

#ifdef RICH_MPI
    #include "utils/debug/mpi_debug.h"
    #include <mpi.h>
#endif // RICH_MPI

#define SPACE_EPS 1e-4

namespace fs = std::filesystem;

std::string getPrefixName(const std::string &run_directory, const std::string &projectName)
{
    const std::filesystem::path run_path{run_directory};
    // directory_iterator can be iterated using a range-for loop
    int maxNum = 0;
	for (auto const& dir_entry : std::filesystem::directory_iterator{run_path}) 
    {
		std::string fileName = dir_entry.path().string();
		if(fileName.rfind(run_directory + projectName, 0) == 0)
		{
			std::string strNum = std::string(fileName.cbegin() + run_directory.size() + projectName.size(), fileName.cend());
			int num = strNum.size() == 0? 0 : std::stoi(std::string(fileName.cbegin() + run_directory.size() + projectName.size(), fileName.cend()));
			maxNum = std::max<int>(maxNum, num);
		}
    }
	std::string const run_name = projectName + std::to_string(maxNum + 1);
    std::string new_dir_name = run_directory + run_name + "/";
    #ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
    #endif // RICH_MPI
	fs::create_directories(new_dir_name.c_str());
	return new_dir_name + "snap_";
}

std::pair<Vector3D, Vector3D> getBordersOfSpace(const std::vector<Vector3D> &points)
{
    Vector3D ll(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    Vector3D ur(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());

    for(const Vector3D &point : points)
    {
        ll.x = std::min<double>(ll.x, point.x);
        ll.y = std::min<double>(ll.y, point.y);
        ll.z = std::min<double>(ll.z, point.z);
        ur.x = std::max<double>(ur.x, point.x);
        ur.y = std::max<double>(ur.y, point.y);
        ur.z = std::max<double>(ur.z, point.z);
    }

    double ll_x = ll.x, ll_y = ll.y, ll_z = ll.z, ur_x = ur.x, ur_y = ur.y, ur_z = ur.z;
    #ifdef RICH_MPI
        std::vector<MPI_Request> requests(6, MPI_REQUEST_NULL);
        MPI_Iallreduce(MPI_IN_PLACE, &ll_x, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, &requests[0]);
        MPI_Iallreduce(MPI_IN_PLACE, &ll_y, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, &requests[1]);
        MPI_Iallreduce(MPI_IN_PLACE, &ll_z, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD, &requests[2]);
        MPI_Iallreduce(MPI_IN_PLACE, &ur_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, &requests[3]);
        MPI_Iallreduce(MPI_IN_PLACE, &ur_y, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, &requests[4]);
        MPI_Iallreduce(MPI_IN_PLACE, &ur_z, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD, &requests[5]);
        MPI_Waitall(6, requests.data(), MPI_STATUSES_IGNORE);
    #endif // RICH_MPI
    
    return std::make_pair(Vector3D(ll_x - SPACE_EPS, ll_y - SPACE_EPS, ll_z - SPACE_EPS), Vector3D(ur_x + SPACE_EPS, ur_y + SPACE_EPS, ur_z + SPACE_EPS));   
}

int main(int argc, char *argv[])
{
    // vtune_pause();
    if(argc != 3 and argc != 4 and argc != 5 and argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <path> <iterations> [weight assignment] [print] [read all?]" << std::endl;
        return EXIT_FAILURE;
    }


    bool weightAssignment = (argc >= 4)? std::stoi(argv[3]) : false;
    bool print = (argc >= 5)? std::stoi(argv[4]) : false;
    bool read_all = (argc >= 6)? std::stoi(argv[5]) : false;
    
    rank_t rank = 0, size = 1;
    #ifdef RICH_MPI
        MPI_Init(&argc, &argv);
        // int provided;
        // MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
        // if(provided != MPI_THREAD_MULTIPLE)
        // {
        //     throw UniversalError("MPI does not support MPI_THREAD_MULTIPLE");
        // }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif // RICH_MPI

    char hostname[1024];
    gethostname(hostname, sizeof(hostname));
    std::cout << "I am rank " << rank << " on host " << hostname << ", pid " << getpid() << std::endl;
    std::cout << std::setprecision(12);

    // sleep(30);

    /*
    Vector3D ll(0, 0, 0), ur(1, 1, 1);
    const int N = 20000;
    boost::mt19937_64 gen(rank);
    Voronoi3D voronoi(ll, ur);
    */
   int x = 4;    
   std::cout << "x = " << x << std::endl;
   
    std::vector<Vector3D> points;
    #ifdef RICH_MPI
        if(read_all)
        {
            if(rank == 0)
            {
                auto dirIter = std::filesystem::directory_iterator(std::string(argv[1]));
                for(auto &entry : dirIter)
                {
                    std::string reading_path = entry.path().string();
                    std::vector<Vector3D> tmp = readFromFile(reading_path);
                    points.insert(points.end(), tmp.begin(), tmp.end());
                }
            }
            points = MPI_Spread(points, 0, MPI_COMM_WORLD);
        }
        else
        {
            std::string reading_path = std::string(argv[1]) + "/" + std::to_string(rank);
            points = readFromFile(reading_path);
        }
    #else // RICH_MPI
        std::string reading_path = std::string(argv[1]);
        points = readFromFile(reading_path);
    #endif // RICH_MPI

    // vorocrust read
    /*
    if(rank == 0)
    {
        std::cout << "reading points..." << std::endl;
        points = readVorocrust("/home/maorm/shared/Vorocrust_input/otis");
        std::cout << "After reading" << std::endl;
    }
    */

    // determine ll and ur
    std::pair<Vector3D, Vector3D> ll_ur = getBordersOfSpace(points);
    if(rank == 0)
    {
        std::cout << "ll_ur is " << ll_ur << std::endl;
    }
    Vector3D ll = ll_ur.first, ur = ll_ur.second;
    //Vector3D ll(-2, -2, -2), ur(2, 2, 2);
    //Vector3D ll(0, 0, 0), ur(1, 1, 1);
    Voronoi3D voronoi(ll, ur);

    // output vtk's
    // std::string prefix = getPrefixName("/data/shared/maorm/", "Fox");

    // checkNearestNeighbor(Vector3D(0.1, 0.2, 0.3), 0.41, Vector3D(0.05, 0.2, 0.1), points, ll, ur);
    try
    {
        std::chrono::_V2::system_clock::time_point start, end;
                        
        std::vector<double> weights(points.size(), 1.0);

        double time = 0, first_build_time = 0;

        int iters = std::stoi(std::string(argv[2]));
        for(int i = 0; i < iters; i++)
        {
            // points = RandRectangular(N, ll, ur, gen);
            
            if(rank == 0)
            {
                std::cout << "Building, iteration " << i << std::endl;
            }
            // vtune_resume();
            #ifdef RICH_MPI
                // if(i == 1 and rank == 0)
                // {
                //     size_t k = weights.size(); // / 2;
                //     double x = 16.0;
                //     // assign a weight of x to a random of k elements
                //     for(size_t j = 0; j < k; j++)
                //     {
                //         // std::cout << "here!!" << std::endl;
                //         weights[j] = x;
                //     }
                //     std::srand(rank);
                //     std::random_shuffle(weights.begin(), weights.end());
                // }                
                if(i >= 1 and weightAssignment)
                {
                    for(size_t j = 0; j < points.size(); j++)
                    {
                        weights[j] = voronoi.GetMaxRadius(j) / voronoi.GetMinRadius(j);
                        assert(weights[j] > 0);
                    }
                }
            #endif // RICH_MPI
            start = std::chrono::system_clock::now();
            #ifdef RICH_MPI
                // std::cout << "rank " << rank << " weights are: " << weights << std::endl;
                voronoi.BuildParallel(points, weights);
            #else // RICH_MPI
                voronoi.Build(points);
            #endif // RICH_MPI
            // vtune_pause();
            end = std::chrono::system_clock::now();

            points = voronoi.getMeshPoints();
            points.resize(voronoi.GetPointNo());
            #ifdef RICH_MPI
                weights = voronoi.GetPointsBuildWeights();
            #endif // RICH_MPI
            if(points.size() != weights.size())
            {
                UniversalError eo("points and weights sizes do not match");
                eo.addEntry("points size", points.size());
                eo.addEntry("weights size", weights.size());
                throw eo;
            }
            // if(i == 0)
            // {
            //     std::cout << "Rank " << rank << " holds " << points.size() << " points" << std::endl;
            // }

            double elapsed_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
            if(i > 0)
            {
                time += elapsed_sec;
            }
            else // i == 0
            {
                first_build_time = elapsed_sec;
            }

            if(rank == 0)
            {
                std::cout << "finished, time is " << elapsed_sec << std::endl;
                std::cout << "**************************************************************************************" << std::endl;
            }

            if(i == 1 and print)
            {
                #ifdef RICH_MPI
                    WriteVoronoi(voronoi, "hilbert_voronoi_p.h5");
                #else // RICH_MPI
                    WriteVoronoiSerial(voronoi, "hilbert_voronoi_s.h5");
                #endif // RICH_MPI

                // // print the file
                // std::vector<std::vector<double>> data;
                // std::vector<std::string> dataNames;
                // for(int _rank = 0; _rank < size; _rank++)
                // {
                //     data.emplace_back(GetWhetherGhostPoint(voronoi, _rank));
                //     dataNames.push_back("IsGhost" + std::to_string(_rank));
                // }
                // WriteVoronoiParallel(voronoi, "hilbert_voronoi.h5", data, dataNames);
            }

            // changePoints(points, ll, ur);
            
            //neighborsCheck(voronoi, ll, ur, 1);
            volumeCheck(ll, ur, voronoi);
        }

        if(iters > 1 and rank == 0)
        {
            std::cout << "First build time is " << first_build_time << std::endl;
            double avg_time = time / (iters - 1);
            std::cout << "Average time is " << avg_time << std::endl;
            double time_per_point = avg_time / points.size();
            std::cout << "Average time per point is " << time_per_point << ", which is " << 1.0 / time_per_point << " points per second" << std::endl;
        }

    }
    catch(const UniversalError& e)
    {
        reportError(e);
    }


    #ifdef RICH_MPI
        MPI_Finalize();
    #endif // RICH_MPI

    return EXIT_SUCCESS;
}
