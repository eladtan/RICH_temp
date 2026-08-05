#ifndef SPHERICAL_SHELL_GEOMETRY_3D_HPP
#define SPHERICAL_SHELL_GEOMETRY_3D_HPP

#include <cstddef>
#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "3D/tessellation/Tessellation3D.hpp"

/*! \brief Tracks complete spherical generator layers across mesh motion and AMR.
 *
 * The constructor radii seed the initial shell population. Update then finds
 * radius clusters with a comparable global population. This admits whole-shell
 * refinement/coarsening while leaving isolated, nonspherical AMR cells outside
 * the spherical background.
 */
class SphericalShellGeometry3D
{
public:
	SphericalShellGeometry3D(Vector3D center,
		std::vector<double> shell_radii,
		double relative_radius_tolerance = 1e-8,
		double absolute_radius_tolerance = 1e-12);

	bool Update(Tessellation3D const& tess) const;
	int FindShell(double radius) const;

	Vector3D const& GetCenter() const;
	std::vector<double> const& GetShellRadii() const;

private:
	struct RadiusCluster
	{
		double radius = 0;
		double population = 0;
	};

	double RadiusTolerance(double left, double right) const;
	void NormalizeShellRadii();
	void InitializeReferencePopulation(Tessellation3D const& tess) const;
	std::vector<RadiusCluster> BuildGlobalRadiusClusters(
		Tessellation3D const& tess) const;
	bool MeshChanged(Tessellation3D const& tess) const;

	Vector3D center_;
	mutable std::vector<double> shell_radii_;
	double relative_radius_tolerance_;
	double absolute_radius_tolerance_;
	mutable double reference_shell_population_ = 0;
	mutable bool reference_population_initialized_ = false;
	mutable bool mesh_signature_initialized_ = false;
	mutable unsigned long long mesh_point_count_ = 0;
	mutable unsigned long long mesh_hash_sum_ = 0;
	mutable unsigned long long mesh_hash_square_sum_ = 0;
	mutable unsigned long long mesh_hash_xor_ = 0;
};

#endif
