#include "RoundGrid3D.hpp"

#include "../../misc/simple_io.hpp"
#include "../../misc/int2str.hpp"
#include <fstream>
#include <limits>
#ifdef RICH_MPI
#include <mpi.h>
#endif

vector<Vector3D> RoundGrid3D(vector<Vector3D> const& points, Vector3D const& ll, Vector3D const& ur,
	size_t NumberIt, Tessellation3D *tess)
{
	Voronoi3D default_tess(ll, ur);
	if (tess == nullptr)
		tess = &default_tess;
	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	tess->BuildParallel(points);
#else
	tess->Build(points);
#endif
	double eta_ = 0.02, chi_ = 1;
	//	size_t N = tess->GetPointNo();
	vector<Vector3D> res(points);
	for (size_t j = 0; j < NumberIt; ++j)
	{
		if(rank == 0)
			std::cout<<"Round Grid Iteration: "<<j<<std::endl;
	  size_t N = tess->GetPointNo();
#ifdef RICH_MPI
		res = tess->getMeshPoints();
		res.resize(static_cast<size_t>(N));
#endif
		for (size_t i = 0; i < N; ++i)
		{
			double R = tess->GetWidth(i);
			Vector3D s = tess->GetCellCM(i);
			Vector3D r = tess->GetMeshPoint(i);
			double d = fastabs(s - r);
			Vector3D dw;
			if (d / eta_ / R < 0.95)
				dw = 0 * s;
			else
				dw = chi_*0.5*(s - r);
			res[i] = tess->GetMeshPoint(i) + dw;
		}
#ifdef RICH_MPI
		tess->BuildParallel(res);
#else
		tess->Build(res);
#endif
	}
#ifdef RICH_MPI
	size_t N = tess->GetPointNo();
	res = tess->getMeshPoints();
	res.resize(N);
#endif
	return res;
}

#ifdef RICH_MPI
vector<Vector3D> RoundGrid3DSingle(vector<Vector3D> const& points, Vector3D const& ll, Vector3D const& ur,
	size_t NumberIt)
{
	Voronoi3D tess(ll, ur);
	double eta_ = 0.02, chi_ = 1;
	tess.Build(points);
	size_t N = tess.GetPointNo(); //build tess first
	vector<Vector3D> res(points);

	for (size_t j = 0; j < NumberIt; ++j)
	{
		for (size_t i = 0; i < N; ++i)
		{
			double R = tess.GetWidth(i);
			Vector3D s = tess.GetCellCM(i);
			Vector3D r = tess.GetMeshPoint(i);
			double d = fastabs(s - r);
			Vector3D dw;
			if (d / eta_ / R < 0.95)
				dw = 0 * s;
			else
				dw = chi_*0.5*(s - r);
			res[i] = tess.GetMeshPoint(i) + dw;
		}
		if(j<(NumberIt-1))
			tess.Build(res); 
	}
	return res;
}
#endif

std::vector<Vector3D> RoundGridSphere3D(std::vector<Vector3D> const& points, Vector3D const& ll, Vector3D const& ur, Vector3D const center,
	size_t NumberIt, Tessellation3D *tess, std::function<bool(Vector3D)> const& criteria, bool verbose)
{
	Voronoi3D default_tess(ll, ur);
	if (tess == nullptr)
		tess = &default_tess;
	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	tess->BuildParallel(points);
#else
	tess->Build(points);
#endif
	double eta_ = 1e-7, chi_ = 1;
	std::vector<Vector3D> res(points);
	for (size_t j = 0; j < NumberIt; ++j)
	{
		if(rank == 0 && verbose)
			std::cout<<"Round Grid Sphere Iteration: "<<j<<std::endl;
	  size_t N = tess->GetPointNo();
#ifdef RICH_MPI
		res = tess->getMeshPoints();
		res.resize(static_cast<size_t>(N));
#endif
		double totalCriteriaVolume = 0;
		size_t criteriaCount = 0;
		double V_min = std::numeric_limits<double>::max();
		double V_max = 0;
		for (size_t i = 0; i < N; ++i)
		{
			Vector3D r = tess->GetMeshPoint(i);
			if (not criteria(r))
				continue;
			double vol = tess->GetVolume(i);
			totalCriteriaVolume += vol;
			if (verbose)
			{
				V_min = std::min(V_min, vol);
				V_max = std::max(V_max, vol);
			}
			++criteriaCount;
		}
#ifdef RICH_MPI
		double globalTotalVolume = 0;
		unsigned long long globalCount = 0;
		unsigned long long localCount = static_cast<unsigned long long>(criteriaCount);
		MPI_Allreduce(&totalCriteriaVolume, &globalTotalVolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		double const V_avg = (globalCount > 0) ? globalTotalVolume / static_cast<double>(globalCount) : 1.0;
		if (verbose)
		{
			double globalVmin = 0, globalVmax = 0;
			MPI_Allreduce(&V_min, &globalVmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
			MPI_Allreduce(&V_max, &globalVmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			V_min = globalVmin;
			V_max = globalVmax;
			if (rank == 0)
				std::cout << "  V_min: " << V_min << "  V_max: " << V_max << "  V_avg: " << V_avg << "  ratio: " << V_max / V_min << std::endl;
		}
#else
		double const V_avg = (criteriaCount > 0) ? totalCriteriaVolume / static_cast<double>(criteriaCount) : 1.0;
		if (verbose && rank == 0)
			std::cout << "  V_min: " << V_min << "  V_max: " << V_max << "  V_avg: " << V_avg << "  ratio: " << V_max / V_min << std::endl;
#endif
		double const pressure_strength = 3;

		for (size_t i = 0; i < N; ++i)
		{
			double R = tess->GetWidth(i);
            Vector3D s = tess->GetCellCM(i);
            Vector3D r = tess->GetMeshPoint(i);
			if(not criteria(r))
				continue;
			double d = fastabs(s - r);
			Vector3D dw;
			if (d / eta_ / R < 0.95)
				dw = 0 * s;
			else
				dw = chi_*0.25*(s - r);
			double V_i = tess->GetVolume(i);
			Vector3D dw_pressure(0, 0, 0);
			face_vec const& faces = tess->GetCellFaces(i);
			double total_area = 0;
			for (size_t f = 0; f < faces.size(); ++f)
			{
				auto const& nbr_pair = tess->GetFaceNeighbors(faces[f]);
				size_t nbr = (nbr_pair.first == i) ? nbr_pair.second : nbr_pair.first;
				if (nbr >= N && tess->IsPointOutsideBox(nbr))
					continue;
				double A_f = tess->GetArea(faces[f]);
				double V_j = tess->GetVolume(nbr);
				Vector3D r_j = tess->GetMeshPoint(nbr);
				double dist_ij = fastabs(r_j - r);
				if (dist_ij > 0)
				{
					dw_pressure = dw_pressure + A_f * (V_j - V_i) / V_avg * (r_j - r) / dist_ij;
					total_area += A_f;
				}
			}
			if (total_area > 0)
			{
				double strength_factor = std::min(0.1,  pressure_strength * total_area / (abs(dw_pressure) + 1e-100));
				dw_pressure = dw_pressure * (strength_factor * R / (abs(dw_pressure) + 1e-100));
			}
			dw = dw + dw_pressure;
			double const oldR = abs(r - center);
			res[i] = tess->GetMeshPoint(i) + dw;
			Vector3D const dir = normalize(res[i] - center);
			res[i] = center + dir * oldR;
		}
#ifdef RICH_MPI
		tess->BuildParallel(res);
#else
		tess->Build(res);
#endif
	}

	size_t const extraIt = std::max(static_cast<size_t>(2), NumberIt / 2);
	{
		size_t N = tess->GetPointNo();
#ifdef RICH_MPI
		res = tess->getMeshPoints();
		res.resize(static_cast<size_t>(N));
#endif
		double totalCriteriaVolume = 0;
		double V_min = std::numeric_limits<double>::max();
		double V_max = 0;
		size_t criteriaCount = 0;
		for (size_t i = 0; i < N; ++i)
		{
			Vector3D r = tess->GetMeshPoint(i);
			if (not criteria(r))
				continue;
			double vol = tess->GetVolume(i);
			totalCriteriaVolume += vol;
			V_min = std::min(V_min, vol);
			V_max = std::max(V_max, vol);
			++criteriaCount;
		}
#ifdef RICH_MPI
		double globalTotalVolume = 0;
		double globalVmin = 0, globalVmax = 0;
		unsigned long long globalCount = 0;
		unsigned long long localCount = static_cast<unsigned long long>(criteriaCount);
		MPI_Allreduce(&totalCriteriaVolume, &globalTotalVolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&V_min, &globalVmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(&V_max, &globalVmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		V_min = globalVmin;
		V_max = globalVmax;
#endif
		if (rank == 0 && verbose)
			std::cout << "Before no-pressure iterations: V_min: " << V_min << "  V_max: " << V_max
			          << "  ratio: " << V_max / V_min << "  starting " << extraIt << " iterations without pressure" << std::endl;
	}

	for (size_t j = 0; j < extraIt; ++j)
	{
		if (rank == 0 && verbose)
			std::cout << "Round Grid Sphere (no pressure) Iteration: " << j << std::endl;
		size_t N = tess->GetPointNo();
#ifdef RICH_MPI
		res = tess->getMeshPoints();
		res.resize(static_cast<size_t>(N));
#endif
		for (size_t i = 0; i < N; ++i)
		{
			double R = tess->GetWidth(i);
			Vector3D s = tess->GetCellCM(i);
			Vector3D r = tess->GetMeshPoint(i);
			if (not criteria(r))
				continue;
			double d = fastabs(s - r);
			Vector3D dw;
			if (d / eta_ / R < 0.95)
				dw = 0 * s;
			else
				dw = chi_ * 0.25 * (s - r);
			double const oldR = abs(r - center);
			res[i] = tess->GetMeshPoint(i) + dw;
			Vector3D const dir = normalize(res[i] - center);
			res[i] = center + dir * oldR;
		}
#ifdef RICH_MPI
		tess->BuildParallel(res);
#else
		tess->Build(res);
#endif
	}

#ifdef RICH_MPI
	size_t Nfinal = tess->GetPointNo();
	res = tess->getMeshPoints();
	res.resize(Nfinal);
#endif
	return res;
}