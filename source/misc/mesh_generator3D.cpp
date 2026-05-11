#include "mesh_generator3D.hpp"
#include <boost/random/normal_distribution.hpp>
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include <array>
#include <limits>
#include <numeric>
#ifdef RICH_MPI
#include <mpi.h>
#endif

using std::array;

namespace
{
	bool inside_box(Vector3D const& p, Vector3D const& ll, Vector3D const& ur)
	{
		return p.x > ll.x && p.x < ur.x && p.y > ll.y && p.y < ur.y && p.z > ll.z && p.z < ur.z;
	}

	double shell_spacing_fraction(SphericalShellMeshOptions const& options, size_t n_angular)
	{
		if (options.radial_spacing_fraction > 0.0)
			return options.radial_spacing_fraction;
		return std::sqrt(4.0 * M_PI / static_cast<double>(n_angular));
	}

	std::vector<double> spherical_shell_radii(SphericalShellMeshOptions const& options,
		double spacing_fraction)
	{
		std::vector<double> radii;
		double R = options.outer_radius;
		while (R > options.inner_radius) {
			radii.push_back(R);
			R *= (1.0 - spacing_fraction);
		}
		return radii;
	}

	std::vector<Vector3D> fibonacci_sphere_directions(size_t n)
	{
		std::vector<Vector3D> dirs;
		dirs.reserve(n);
		double const golden_angle = M_PI * (3.0 - std::sqrt(5.0));
		for (size_t i = 0; i < n; ++i) {
			double const z = 1.0 - (2.0 * static_cast<double>(i) + 1.0) / static_cast<double>(n);
			double const r = std::sqrt(std::max(0.0, 1.0 - z * z));
			double const phi = golden_angle * static_cast<double>(i);
			dirs.push_back(Vector3D(r * std::cos(phi), r * std::sin(phi), z));
		}
		return dirs;
	}

	void append_shell(std::vector<Vector3D>& points, std::vector<Vector3D> const& dirs,
		double radius, Vector3D const& center, Vector3D const& ll, Vector3D const& ur)
	{
		for (auto const& d : dirs) {
			Vector3D p = center + d * radius;
			if (inside_box(p, ll, ur))
				points.push_back(p);
		}
	}

	void accumulate_shell_stats(Tessellation3D const& tess, std::vector<double> const& edges,
		Vector3D const& center,
		double& cv_max, double& ratio_max, size_t& measured_bins)
	{
		if (edges.size() < 2)
			return;
		size_t const nbins = edges.size() - 1;
		std::vector<unsigned long long> counts(nbins, 0);
		std::vector<double> sum(nbins, 0.0);
		std::vector<double> sum2(nbins, 0.0);
		std::vector<double> vmin(nbins, std::numeric_limits<double>::max());
		std::vector<double> vmax(nbins, 0.0);
		size_t const N = tess.GetPointNo();
		for (size_t i = 0; i < N; ++i) {
			double const r = abs(tess.GetCellCM(i) - center);
			if (r < edges.front() || r >= edges.back())
				continue;
			auto it = std::upper_bound(edges.begin(), edges.end(), r);
			if (it == edges.begin())
				continue;
			size_t const b = static_cast<size_t>(it - edges.begin()) - 1;
			if (b >= nbins)
				continue;
			double const vol = tess.GetVolume(i);
			++counts[b];
			sum[b] += vol;
			sum2[b] += vol * vol;
			vmin[b] = std::min(vmin[b], vol);
			vmax[b] = std::max(vmax[b], vol);
		}
#ifdef RICH_MPI
		{
			int const n = static_cast<int>(nbins);
			std::vector<unsigned long long> gcounts(nbins, 0);
			std::vector<double> g(nbins, 0.0);
			MPI_Allreduce(counts.data(), gcounts.data(), n, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
			counts = gcounts;
			MPI_Allreduce(sum.data(), g.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			sum = g;
			MPI_Allreduce(sum2.data(), g.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			sum2 = g;
			MPI_Allreduce(vmin.data(), g.data(), n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
			vmin = g;
			MPI_Allreduce(vmax.data(), g.data(), n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			vmax = g;
		}
#endif
		for (size_t b = 0; b < nbins; ++b) {
			if (counts[b] < 6 || sum[b] <= 0.0 || vmin[b] <= 0.0)
				continue;
			double const inv_count = 1.0 / static_cast<double>(counts[b]);
			double const mean = sum[b] * inv_count;
			double const var = std::max(0.0, sum2[b] * inv_count - mean * mean);
			cv_max = std::max(cv_max, std::sqrt(var) / mean);
			ratio_max = std::max(ratio_max, vmax[b] / vmin[b]);
			++measured_bins;
		}
	}
}

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

std::vector<double> SphericalShellMeshActiveBinEdges(SphericalShellMeshOptions const& options)
{
	size_t const n_angular = 6 * options.angular_edge_count * options.angular_edge_count;
	double const dR_over_R = shell_spacing_fraction(options, n_angular);
	std::vector<double> edges;
	double R = options.outer_radius;
	while (R > options.inner_radius) {
		edges.push_back(R);
		R *= (1.0 - dR_over_R);
	}
	edges.push_back(R);
	std::reverse(edges.begin(), edges.end());
	return edges;
}

std::vector<Vector3D> GenerateSphericalShellMesh3D(Vector3D const& ll, Vector3D const& ur,
	SphericalShellMeshOptions const& options,
	SphericalShellMeshDiagnostics* diagnostics)
{
	size_t const n_angular = 6 * options.angular_edge_count * options.angular_edge_count;
	std::vector<Vector3D> unit_dirs = fibonacci_sphere_directions(n_angular);

	double const dR_over_R = shell_spacing_fraction(options, unit_dirs.size());
	std::vector<double> active_radii = spherical_shell_radii(options, dR_over_R);

	std::vector<Vector3D> points;
	points.reserve(active_radii.size() * unit_dirs.size());

	for (double const r : active_radii)
		append_shell(points, unit_dirs, r, options.center, ll, ur);

	for (size_t k = 1; k <= options.guard_shell_count; ++k) {
		double const r = options.inner_radius * std::pow(1.0 - dR_over_R, static_cast<double>(k));
		if (r > 1e-12)
			append_shell(points, unit_dirs, r, options.center, ll, ur);
	}

	for (size_t k = 1; k <= options.guard_shell_count; ++k) {
		double const r = options.outer_radius * std::pow(1.0 + dR_over_R, static_cast<double>(k));
		append_shell(points, unit_dirs, r, options.center, ll, ur);
	}

	double const inner_guard_radius = options.inner_radius *
		std::pow(1.0 - dR_over_R, static_cast<double>(options.guard_shell_count));
	if (options.fill_inner_core && inner_guard_radius > 0.0) {
		double const h = std::max(options.inner_radius * dR_over_R, inner_guard_radius / 8.0);
		for (double x = options.center.x - inner_guard_radius + 0.5 * h;
			 x < options.center.x + inner_guard_radius; x += h) {
			for (double y = options.center.y - inner_guard_radius + 0.5 * h;
				 y < options.center.y + inner_guard_radius; y += h) {
				for (double z = options.center.z - inner_guard_radius + 0.5 * h;
					 z < options.center.z + inner_guard_radius; z += h) {
					Vector3D const p(x, y, z);
					if (abs(p - options.center) < inner_guard_radius && inside_box(p, ll, ur))
						points.push_back(p);
				}
			}
		}
		points.push_back(options.center);
	}

	double outer_fill_radius = options.outer_radius *
		std::pow(1.0 + dR_over_R, static_cast<double>(options.guard_shell_count));
	if (options.fill_outer_box) {
		double const h = std::max(options.outer_radius * dR_over_R * options.exterior_spacing_factor,
			options.outer_radius * dR_over_R);
		for (double x = ll.x + 0.5 * h; x < ur.x; x += h) {
			for (double y = ll.y + 0.5 * h; y < ur.y; y += h) {
				for (double z = ll.z + 0.5 * h; z < ur.z; z += h) {
					Vector3D const p(x, y, z);
					if (abs(p - options.center) > outer_fill_radius && inside_box(p, ll, ur))
						points.push_back(p);
				}
			}
		}
	}

	if (diagnostics != nullptr) {
		Voronoi3D tess(ll, ur);
		tess.Build(points);
		*diagnostics = MeasureSphericalShellMesh3D(tess, options);
	}

	return points;
}

SphericalShellMeshDiagnostics MeasureSphericalShellMesh3D(Tessellation3D const& tess,
	SphericalShellMeshOptions const& options)
{
	SphericalShellMeshDiagnostics res;
	std::vector<double> active_edges = SphericalShellMeshActiveBinEdges(options);
	accumulate_shell_stats(tess, active_edges, options.center,
		res.active_shell_volume_cv_max, res.active_shell_volume_ratio_max,
		res.active_shell_count);

	size_t const n_angular = 6 * options.angular_edge_count * options.angular_edge_count;
	double const dR_over_R = shell_spacing_fraction(options, n_angular);
	double guard_cv = 0.0;
	std::vector<double> inner_edges;
	for (size_t k = options.guard_shell_count; k > 0; --k)
		inner_edges.push_back(options.inner_radius * std::pow(1.0 - dR_over_R, static_cast<double>(k)));
	inner_edges.push_back(options.inner_radius);
	accumulate_shell_stats(tess, inner_edges, options.center, guard_cv,
		res.guard_shell_volume_ratio_max, res.guard_shell_count);

	std::vector<double> outer_edges;
	outer_edges.push_back(options.outer_radius);
	for (size_t k = 1; k <= options.guard_shell_count; ++k)
		outer_edges.push_back(options.outer_radius * std::pow(1.0 + dR_over_R, static_cast<double>(k)));
	accumulate_shell_stats(tess, outer_edges, options.center, guard_cv,
		res.guard_shell_volume_ratio_max, res.guard_shell_count);
	return res;
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
