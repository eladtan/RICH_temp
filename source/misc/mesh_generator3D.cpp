#include "mesh_generator3D.hpp"
#include <boost/random/normal_distribution.hpp>
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include <array>
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

std::vector<Vector3D> RandSphereSurfaceRounded(double const Radius, size_t const PointNum, Vector3D const& center, size_t const Niterations)
{
	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	std::vector<Vector3D> points1;
	double const dR = Radius / std::sqrt(PointNum);
	double const new_R = Radius + 7 * dR;
	Vector3D const ll = Vector3D(-new_R, -new_R, -new_R) + center;
	Vector3D const ur = Vector3D(new_R, new_R, new_R) + center;
	if(rank == 0)
	{
		points1 = RandSphereSurface(Radius, PointNum, center);
		std::vector<Vector3D> points2 = RandSphereSurface(Radius - dR, PointNum, center);
		points1.insert(points1.end(), points2.begin(), points2.end());
		points2 = RandSphereSurface(Radius + dR, PointNum, center);
		points1.insert(points1.end(), points2.begin(), points2.end());
		points2 = RandSphereR(std::max(10, static_cast<int>(0.01 * PointNum * Radius / dR)), ll, ur, 0, Radius - 4 * dR, center);
		points1.insert(points1.end(), points2.begin(), points2.end());
		points2 = RandSphereR(std::max(10, static_cast<int>(0.01 * PointNum * Radius / dR)), ll, ur, Radius + 4 * dR, new_R * 1.4, center);
		points1.insert(points1.end(), points2.begin(), points2.end());
		std::cout << "Total points before rounding: " << points1.size() << std::endl;
	}

	std::function<bool(Vector3D const&)> criteriaR;
	criteriaR = [center, Radius, dR](Vector3D const& point) {
        return std::abs(Radius - abs(point - center)) < 0.25 * dR;
    };

	points1 = RoundGridSphere3D(points1, ll, ur, center, Niterations, nullptr, criteriaR);
#ifdef RICH_MPI
	points1 = MPI_Gatherv_serializable(points1, 0, MPI_COMM_WORLD);
#endif
	std::vector<Vector3D> points;
	for(auto const& point : points1)
	{
		if(std::abs(Radius - abs(point - center)) < 0.25 * dR)
			points.push_back(point);
	}
	return points;
}
