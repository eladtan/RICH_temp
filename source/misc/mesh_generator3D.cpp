#include "mesh_generator3D.hpp"
#include <boost/random/normal_distribution.hpp>
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include <array>
#include <limits>
#ifdef RICH_MPI
#include <mpi.h>
#endif

using std::array;

 vector<Vector3D> CartesianMesh(std::size_t nx, std::size_t ny, std::size_t nz, Vector3D const& lower_left, Vector3D const& upper_right)
{
	assert(upper_right.x > lower_left.x);
	assert(upper_right.y > lower_left.y);
	assert(upper_right.z > lower_left.z);

	vector<Vector3D> res;
	const double dx = (upper_right.x - lower_left.x) /
		static_cast<double>(nx);
	const double dy = (upper_right.y - lower_left.y) /
		static_cast<double>(ny);
	const double dz = (upper_right.z - lower_left.z) /
		static_cast<double>(nz);
	for(size_t i = 0; i < nx; ++i)
		for(size_t j = 0; j < ny; ++j)
			for(size_t k = 0; k < nz; ++k)
			{

				Vector3D new_point = Vector3D(lower_left.x+0.5*dx+i*dx, lower_left.y+0.5*dy+j*dy, lower_left.z+0.5*dz+k*dz);
				res.push_back(new_point);
			}
	return res;
}

vector<Vector3D> RandRectangular(std::size_t PointNum, Vector3D const& ll, Vector3D const& ur)
{
	typedef boost::mt19937_64 base_generator_type;
	double ran[3];
	Vector3D diff = ur - ll;
	vector<Vector3D> res;
	Vector3D point;
	base_generator_type generator;
	boost::random::uniform_real_distribution<> dist;

	for (size_t i = 0; i < PointNum; ++i)
	{
		ran[0] = dist(generator);
		ran[1] = dist(generator);
		ran[2] = dist(generator);
		point.x = ran[0] * diff.x + ll.x;
		point.y = ran[1] * diff.y + ll.y;
		point.z = ran[2] * diff.z + ll.z;
		res.push_back(point);
	}
	return res;
}

vector<Vector3D> RandRectangular(std::size_t PointNum, Vector3D const& ll, Vector3D const& ur, boost::mt19937_64 &generator)
{
	double ran[3];
	Vector3D diff = ur - ll;
	vector<Vector3D> res;
	Vector3D point;
	boost::random::uniform_real_distribution<> dist;
	for (size_t i = 0; i < PointNum; ++i)
	{
		ran[0] = dist(generator);
		ran[1] = dist(generator);
		ran[2] = dist(generator);
		point.x = ran[0] * diff.x + ll.x;
		point.y = ran[1] * diff.y + ll.y;
		point.z = ran[2] * diff.z + ll.z;
		res.push_back(point);
	}
	return res;
}


vector<Vector3D> RandSphereR2(std::size_t PointNum, Vector3D const& ll, Vector3D const& ur, double Rmin, double Rmax, const Vector3D& center)
{
	typedef boost::mt19937_64 base_generator_type;
	base_generator_type generator;
	boost::random::uniform_real_distribution<> dist;
	vector<Vector3D> res;
	res.reserve(PointNum);
	while (res.size() < PointNum)
	{
		double r = dist(generator)*(Rmax - Rmin) + Rmin;
		double phi = 2 * M_PI*dist(generator);
		double t = acos(2 * dist(generator) - 1);
		Vector3D point(r*sin(t)*cos(phi), r*sin(t)*sin(phi), r*cos(t));
		point += center;
		if (point.x<ur.x&&point.x>ll.x&&point.y > ll.y&&point.y<ur.y&&point.z>ll.z&&point.z < ur.z)
			res.push_back(point);
	}
	return res;
}

vector<Vector3D> RandSphereR(std::size_t PointNum, Vector3D const& ll, Vector3D const& ur, double Rmin, double Rmax,
	const Vector3D& center)
{
	typedef boost::mt19937_64 base_generator_type;
	base_generator_type generator;
	boost::random::uniform_real_distribution<> dist;
	vector<Vector3D> res;

	res.reserve(PointNum);
	while (res.size() < PointNum)
	{
		double r = std::pow(dist(generator)*(Rmax*Rmax*Rmax - Rmin * Rmin*Rmin) + Rmin * Rmin*Rmin, 0.333333333);
		double phi = 2 * M_PI*dist(generator);
		double t = acos(2 * dist(generator) - 1);
		Vector3D point(r*sin(t)*cos(phi), r*sin(t)*sin(phi), r*cos(t));
		point += center;
		if (point.x<ur.x&&point.x>ll.x&&point.y > ll.y&&point.y<ur.y&&point.z>ll.z&&point.z < ur.z)
			res.push_back(point);
	}
	return res;
}

vector<Vector3D> RandSphereR1(std::size_t PointNum, Vector3D const& ll, Vector3D const& ur, double Rmin, double Rmax, const Vector3D& center)
{
	typedef boost::mt19937_64 base_generator_type;
	base_generator_type generator;
	boost::random::uniform_real_distribution<> dist;
	vector<Vector3D> res;
	
	res.reserve(PointNum);
	while (res.size() < PointNum)
	{
		double r = std::sqrt(dist(generator)*(Rmax - Rmin)*(Rmax - Rmin)) + Rmin;
		double phi = 2 * M_PI*dist(generator);
		double t = acos(2 * dist(generator) - 1);
		Vector3D point(r*sin(t)*cos(phi), r*sin(t)*sin(phi), r*cos(t));
		if (point.x<ur.x&&point.x>ll.x&&point.y > ll.y&&point.y<ur.y&&point.z>ll.z&&point.z < ur.z)
			res.push_back(point);
	}
	return res;
}

vector<Vector3D> RandSphereRa(std::size_t PointNum, Vector3D const & ll, Vector3D const & ur, double Rmin, double Rmax, double a, Vector3D const& center)
{
	typedef boost::mt19937_64 base_generator_type;
	base_generator_type generator;
	boost::random::uniform_real_distribution<> dist;
	vector<Vector3D> res;
	double Rmx = std::pow(Rmax, a);
	double Rmn = std::pow(Rmin, a);
	double a_1 = 1.0 / a;

	res.reserve(PointNum);
	while (res.size() < PointNum)
	{
		double r = std::pow(dist(generator)*(Rmx - Rmn) + Rmn, a_1);
		double phi = 2 * M_PI*dist(generator);
		double t = acos(2 * dist(generator) - 1);
		Vector3D point(r*sin(t)*cos(phi), r*sin(t)*sin(phi), r*cos(t));
		point += center;
		if (point.x<ur.x&&point.x>ll.x&&point.y > ll.y&&point.y<ur.y&&point.z>ll.z&&point.z < ur.z)
			res.push_back(point);
	}

	return res;
}

std::vector<Vector3D> RandSphereSurface(double const Radius, size_t const PointNum, Vector3D const center)
{
	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	std::vector<Vector3D> points;
	points.reserve(PointNum);
	typedef boost::mt19937_64 base_generator_type;
	base_generator_type generator(rank);
	boost::random::normal_distribution<> dist;
	double ran[3];
	for(size_t i = 0; i < PointNum; ++i)
	{
		ran[0] = dist(generator);
		ran[1] = dist(generator);
		ran[2] = dist(generator);
		double norm = std::sqrt(ran[0] * ran[0] + ran[1] * ran[1] + ran[2] * ran[2]);
		Vector3D point = Vector3D(ran[0] / norm * Radius, ran[1] / norm * Radius, ran[2] / norm * Radius);
		point += center;
		points.push_back(point);
	}
	return points;
}

std::vector<Vector3D> CubedSphereSurface(double const Radius, size_t const N_per_edge,
	Vector3D const center, size_t const Niterations)
{
	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	size_t const Nface = N_per_edge * N_per_edge;
	size_t const Ntotal = 6 * Nface;
	double const dR = Radius * std::sqrt(4.0 * M_PI / static_cast<double>(Ntotal));
	double const new_R = Radius + 7 * dR;
	Vector3D const ll = Vector3D(-new_R, -new_R, -new_R) + center;
	Vector3D const ur = Vector3D(new_R, new_R, new_R) + center;

	std::vector<Vector3D> spherePoints;
	if (rank == 0)
	{
		spherePoints.reserve(Ntotal);
		double const inv_N = 1.0 / static_cast<double>(N_per_edge);
		for (size_t face = 0; face < 6; ++face)
		{
			for (size_t i = 0; i < N_per_edge; ++i)
			{
				double u = -1.0 + (2.0 * static_cast<double>(i) + 1.0) * inv_N;
				for (size_t j = 0; j < N_per_edge; ++j)
				{
					double v = -1.0 + (2.0 * static_cast<double>(j) + 1.0) * inv_N;
					Vector3D p;
					switch (face)
					{
						case 0: p = Vector3D( 1, u, v); break;
						case 1: p = Vector3D(-1, u, v); break;
						case 2: p = Vector3D(u,  1, v); break;
						case 3: p = Vector3D(u, -1, v); break;
						case 4: p = Vector3D(u, v,  1); break;
						default: p = Vector3D(u, v, -1); break;
					}
					double r = abs(p);
					spherePoints.push_back(center + p * (Radius / r));
				}
			}
		}
	}

	std::function<bool(Vector3D const&)> criteriaR =
		[center, Radius, dR](Vector3D const& point) {
			return std::abs(Radius - abs(point - center)) < 0.25 * dR;
		};

	size_t const fluffCount = static_cast<size_t>(
		std::max(10, static_cast<int>(0.01 * static_cast<double>(Ntotal) * Radius / dR)));

	Voronoi3D tess(ll, ur);

	for (size_t iter = 0; iter < Niterations; ++iter)
	{
		if (rank == 0)
			std::cout << "CubedSphereSurface iteration: " << iter << std::endl;

		std::vector<Vector3D> allPoints;
		if (rank == 0)
		{
			allPoints.reserve(3 * spherePoints.size() + 2 * fluffCount);
			allPoints.insert(allPoints.end(), spherePoints.begin(), spherePoints.end());

			for (auto const& p : spherePoints) {
				Vector3D dir = normalize(p - center);
				allPoints.push_back(center + dir * (Radius - dR));
			}
			for (auto const& p : spherePoints) {
				Vector3D dir = normalize(p - center);
				allPoints.push_back(center + dir * (Radius + dR));
			}

			auto fill = RandSphereR(fluffCount, ll, ur, 0, Radius - 4 * dR, center);
			allPoints.insert(allPoints.end(), fill.begin(), fill.end());
			fill = RandSphereR(fluffCount, ll, ur, Radius + 4 * dR, new_R * 1.4, center);
			allPoints.insert(allPoints.end(), fill.begin(), fill.end());
		}

		allPoints = RoundGridSphere3D(allPoints, ll, ur, center, 1, &tess, criteriaR, false);

		{
			size_t Np = tess.GetPointNo();
			double totalCriteriaVolume = 0;
			double Vmin = std::numeric_limits<double>::max();
			double Vmax = 0;
			size_t count = 0;
			for (size_t i = 0; i < Np; ++i)
			{
				if (!criteriaR(tess.GetMeshPoint(i)))
					continue;
				double vol = tess.GetVolume(i);
				totalCriteriaVolume += vol;
				Vmin = std::min(Vmin, vol);
				Vmax = std::max(Vmax, vol);
				++count;
			}
#ifdef RICH_MPI
			double globalTotalVolume = 0;
			double globalVmin = 0, globalVmax = 0;
			unsigned long long globalCount = 0;
			unsigned long long localCount = static_cast<unsigned long long>(count);
			MPI_Allreduce(&totalCriteriaVolume, &globalTotalVolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(&Vmin, &globalVmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
			MPI_Allreduce(&Vmax, &globalVmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			double const Vavg = (globalCount > 0) ? globalTotalVolume / static_cast<double>(globalCount) : 1.0;
			Vmin = globalVmin;
			Vmax = globalVmax;
			count = static_cast<size_t>(globalCount);
#else
			double const Vavg = (count > 0) ? totalCriteriaVolume / static_cast<double>(count) : 1.0;
#endif
			if (rank == 0)
				std::cout << "  iter " << iter << ": N=" << count
					<< "  V_min: " << Vmin << "  V_max: " << Vmax
					<< "  V_avg: " << Vavg << "  ratio: " << (Vmin > 0 ? Vmax / Vmin : 0) << std::endl;
		}

#ifdef RICH_MPI
		allPoints = MPI_Gatherv_serializable(allPoints, 0, MPI_COMM_WORLD);
#endif
		spherePoints.clear();
		for (auto const& p : allPoints) {
			if (criteriaR(p))
				spherePoints.push_back(p);
		}
	}

	if (rank == 0)
		std::cout << "CubedSphereSurface: " << spherePoints.size() << " points from "
			<< N_per_edge << "x" << N_per_edge << " x 6 faces" << std::endl;
	return spherePoints;
}

std::vector<Vector3D> RandSphereSurfaceRounded(double const Radius, size_t const PointNum, Vector3D const center, size_t const Niterations)
{
	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	double const dR = Radius / std::sqrt(PointNum);
	double const new_R = Radius + 7 * dR;
	Vector3D const ll = Vector3D(-new_R, -new_R, -new_R) + center;
	Vector3D const ur = Vector3D(new_R, new_R, new_R) + center;

	std::vector<Vector3D> spherePoints;
	if (rank == 0)
		spherePoints = RandSphereSurface(Radius, PointNum, center);

	std::function<bool(Vector3D const&)> criteriaR =
		[center, Radius, dR](Vector3D const& point) {
			return std::abs(Radius - abs(point - center)) < 0.25 * dR;
		};

	size_t const fluffCount = static_cast<size_t>(
		std::max(10, static_cast<int>(0.01 * PointNum * Radius / dR)));

	Voronoi3D tess(ll, ur);

	for (size_t iter = 0; iter < Niterations; ++iter)
	{
		if (rank == 0)
			std::cout << "RandSphereSurfaceRounded iteration: " << iter << std::endl;

		std::vector<Vector3D> allPoints;
		if (rank == 0)
		{
			allPoints.reserve(3 * spherePoints.size() + 2 * fluffCount);
			allPoints.insert(allPoints.end(), spherePoints.begin(), spherePoints.end());

			for (auto const& p : spherePoints) {
				Vector3D dir = normalize(p - center);
				allPoints.push_back(center + dir * (Radius - dR));
			}
			for (auto const& p : spherePoints) {
				Vector3D dir = normalize(p - center);
				allPoints.push_back(center + dir * (Radius + dR));
			}

			auto fill = RandSphereR(fluffCount, ll, ur, 0, Radius - 4 * dR, center);
			allPoints.insert(allPoints.end(), fill.begin(), fill.end());
			fill = RandSphereR(fluffCount, ll, ur, Radius + 4 * dR, new_R * 1.4, center);
			allPoints.insert(allPoints.end(), fill.begin(), fill.end());
		}

		allPoints = RoundGridSphere3D(allPoints, ll, ur, center, 1, &tess, criteriaR, true);

#ifdef RICH_MPI
		allPoints = MPI_Gatherv_serializable(allPoints, 0, MPI_COMM_WORLD);
#endif
		spherePoints.clear();
		for (auto const& p : allPoints) {
			if (criteriaR(p))
				spherePoints.push_back(p);
		}
	}

	return spherePoints;
}
