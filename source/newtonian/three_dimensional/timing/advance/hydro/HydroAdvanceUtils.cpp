#include "HydroTimeAdvance.hpp"

#include "3D/tessellation/Tessellation3D.hpp"

void MovePoints(Tessellation3D& tess, std::vector<Vector3D> const& point_vel, double const dt)
{
	std::vector<Vector3D>& points = tess.getAllPoints();
	size_t N = points.size();
	if(N > point_vel.size())
	{
		UniversalError eo("MovePoint: Too many points (N) in all points compared to point_vel size");
		eo.addEntry("point_vel.size()", point_vel.size());
		eo.addEntry("N", N);
		throw eo;
	}
	for(size_t i = 0; i < N; ++i)
		points[i] += point_vel[i] * dt;
}

void UpdateTessellation(Tessellation3D& tess, const std::vector<size_t> &participatingIndices, const vector<Vector3D>& point_vel, double dt, std::vector<Vector3D> const* orgpoints)
{
	vector<Vector3D> points;
	if (orgpoints == nullptr)
		points = tess.getAllPoints();
	else
		points = *orgpoints;
	points.resize(tess.GetPointNo());
	if(orgpoints != nullptr)
	{
		size_t const N = points.size();
		for (size_t i = 0; i < N; ++i)
			points[i] += point_vel[i] * dt;
	}

	#ifdef RICH_MPI
		tess.BuildPartiallyParallel(points, participatingIndices);
	#else // RICH_MPI
		tess.BuildPartially(points, participatingIndices);
	#endif // RICH_MPI
}

void ExtensiveAvg(vector<Conserved3D>& res, vector<Conserved3D> const& other)
{
	assert(res.size() == other.size());
	size_t N = res.size();
	for (size_t i = 0; i < N; ++i)
	{
		res[i] += other[i];
		res[i] *= 0.5;
	}
}

#ifdef RICH_MPI
double get_time()
{
	return MPI_Wtime();
}
#else
std::chrono::time_point<std::chrono::high_resolution_clock> get_time()
{
	return std::chrono::high_resolution_clock::now();
}
#endif