#ifndef HYDRO_ADVANCE_UTILS_HPP
#define HYDRO_ADVANCE_UTILS_HPP

#include <chrono>
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"

void MovePoints(Tessellation3D& tess, std::vector<Vector3D> const& point_vel, double const dt);

void UpdateTessellation(Tessellation3D& tess, const std::vector<size_t> &participatingIndices, const vector<Vector3D>& point_vel, double dt, std::vector<Vector3D> const* orgpoints = nullptr);

inline void UpdateTessellation(Tessellation3D& tess, const vector<Vector3D>& point_vel, double dt, std::vector<Vector3D> const* orgpoints = nullptr)
{
	std::vector<size_t> indices(tess.GetPointNo());
	std::iota(indices.begin(), indices.end(), 0);
	UpdateTessellation(tess, indices, point_vel, dt, orgpoints);
}

void ExtensiveAvg(vector<Conserved3D>& res, vector<Conserved3D> const& other);

template<class T>
void DisplayTime(T const& t1, T const& t2, std::string const& msg)
{
	#ifdef RICH_MPI
		int rank = -1;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if(rank == 0)
			std::cout<<msg<<" "<<t2 - t1<<" seconds"<<std::endl;
	#else
		std::cout<<msg<< std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()<<" seconds"<<std::endl;
	#endif
}

#ifdef RICH_MPI
	double get_time();
#else
	std::chrono::time_point<std::chrono::high_resolution_clock> get_time();
#endif

#endif // HYDRO_ADVANCE_UTILS_HPP