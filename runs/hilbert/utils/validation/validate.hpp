#ifndef _VALIDATE_H
#define _VALIDATE_H

#include <iostream>
#ifdef RICH_MPI
    #include <mpi.h>
#endif // RICH_MPI

#include "3D/tessellation/Voronoi3D.hpp"

void checkNearestNeighbor(const Vector3D &center, double radius, const Vector3D &point, std::vector<Vector3D> &points, const Vector3D &ll, const Vector3D &ur)
{
    OctTreeFinder finder(points.begin(), points.end(), ll, ur);
    std::vector<size_t> result = finder.closestPointInSphere(center, radius, point, RangeFinder::_set<size_t>());
    if(!result.empty())
    {
        std::cout << "oct tree closest: " << points[result[0]] << std::endl;
    }
    Vector3D closestPoint(0, 0, 0);
    double closestDistance = std::numeric_limits<double>::max();

    for(const Vector3D &_point : points)
    {
        double distToCenter = (center[0] - _point[0]) * (center[0] - _point[0]) + (center[1] - _point[1]) * (center[1] - _point[1]) + (center[2] - _point[2]) * (center[2] - _point[2]);
        if(distToCenter > (radius * radius))
        {
            continue;
        }

        double distToPoint = (point[0] - _point[0]) * (point[0] - _point[0]) + (point[1] - _point[1]) * (point[1] - _point[1]) + (point[2] - _point[2]) * (point[2] - _point[2]);

        if(distToPoint < closestDistance)
        {
            closestDistance = distToPoint;
            closestPoint = _point;
        }
    }
    std::cout << "real closest: " << closestPoint << std::endl;
}

void checkNearestNeighbors2(const Vector3D &point, std::vector<Vector3D> &points, const Vector3D &ll, const Vector3D &ur)
{

    OctTreeFinder finder(points.begin(), points.end(), ll, ur);
    Vector3D center = (ur + ll) / 2;
    double radius = 50;
    RangeFinder::_set<size_t> ignore;
    for(size_t i = 0; i < points.size(); i++)
    {
        std::vector<size_t> result = finder.closestPointInSphere(center, radius, point, ignore);
        if(!result.empty())
        {
            Vector3D _point = points[result[0]];
            // double distToPoint = (point[0] - _point[0]) * (point[0] - _point[0]) + (point[1] - _point[1]) * (point[1] - _point[1]) + (point[2] - _point[2]) * (point[2] - _point[2]);
            // std::cout << "iteration " << i << " oct tree closest: " << _point << " distance is " << distToPoint << std::endl;
            ignore.insert(result[0]);
        }
        else
        {
            std::cout << "error!" << std::endl;
            exit(1);
        }
    }
    if(!finder.closestPointInSphere(center, radius, point, ignore).empty())
    {
        std::cout << "error!" << std::endl;
        exit(1);
    }
    // Vector3D closestPoint(0, 0, 0);
    // double closestDistance = std::numeric_limits<double>::max();

    // for(const Vector3D &_point : points)
    // {
    //     double distToCenter = (center[0] - _point[0]) * (center[0] - _point[0]) + (center[1] - _point[1]) * (center[1] - _point[1]) + (center[2] - _point[2]) * (center[2] - _point[2]);
    //     if(distToCenter > (radius * radius))
    //     {
    //         continue;
    //     }

    //     double distToPoint = (point[0] - _point[0]) * (point[0] - _point[0]) + (point[1] - _point[1]) * (point[1] - _point[1]) + (point[2] - _point[2]) * (point[2] - _point[2]);

    //     if(distToPoint < closestDistance)
    //     {
    //         closestDistance = distToPoint;
    //         closestPoint = _point;
    //     }
    // }
    // std::cout << "real closest: " << closestPoint << std::endl;
}

bool volumeCheck(const Vector3D &ll, const Vector3D &ur, const Voronoi3D &voronoi)
{
    int rank = 0;
    #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif // RICH_MPI

    if(rank == 0)
    {
        std::cout << "Running volume check" << std::endl;
    }
    // volume check
    double volume = 0;
    for(size_t i = 0; i < voronoi.GetPointNo(); i++)
    {
        volume += voronoi.GetVolume(i);
    }
    #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif // RICH_MPI
    double expectedVolume = (ur[0] - ll[0]) * (ur[1] - ll[1]) * (ur[2] - ll[2]);
    if(rank == 0)
    {
        std::cout << "volume is " << volume << " (ll=" << ll << ", ur=" << ur << "), expected-volume is " << (expectedVolume-volume) << std::endl;

        /*
        if(std::abs(expectedVolume-totalVolume) > TOLERANCE)
        {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        */
    }
    return true;
}

bool neighborsCheck(const Voronoi3D &voronoi, const Vector3D &ll, const Vector3D &ur, int iterations)
{
    int rank = 0;
    #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif // RICH_MPI

    for(int i = 0; i < iterations; i++)
    {
        if(rank == 0)
        {
            std::cout << "Running validation, iteration " << i << std::endl;
        }
        
        std::vector<Vector3D> validation = readFromFile("/data/shared/maorm/input/validation_1000_1024/" + std::to_string(i));
        for(size_t i = 0; i < validation.size(); i++)
        {
            const Vector3D &stamPoint = validation[i];
            struct
            {
                double min;
                int proc;
            } in, out;
            out.proc = rank;

            in.min = std::numeric_limits<double>::max();
            in.proc = rank;
            size_t idx = -1;

            size_t N = voronoi.GetPointNo();
            for(size_t i = 0; i < N; i++)
            {
                Vector3D const diff = voronoi.GetMeshPoint(i) - stamPoint;
                double distance_2 = ScalarProd(diff, diff);
                if(distance_2 < in.min)
                {
                    in.min = distance_2;
                    idx = i;
                }
            }

            #ifdef RICH_MPI
                MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
            #endif // RICH_MPI

            if(rank == out.proc)
            {
                if(PointInPoly(voronoi, stamPoint, idx) != true and !(stamPoint == ll or stamPoint == ur))
                {
                    std::cerr << "error in " << i << "th point. The point " << stamPoint << " should be in polygon in rank " << rank << "(center of idx of mine is " << voronoi.GetMeshPoint(idx) << ")" << std::endl;
                    #ifdef RICH_MPI
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    #else // RICH_MPI
                        exit(1);
                    #endif // RICH_MPI
                }
            }


        }
    }
    if(rank == 0)
    {
        std::cout << "All is OK" << std::endl;
    }
    return true;
}

#endif // _VALIDATE_H