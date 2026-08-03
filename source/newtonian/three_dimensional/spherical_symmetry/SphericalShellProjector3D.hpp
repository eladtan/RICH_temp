#ifndef SPHERICAL_SHELL_PROJECTOR_3D_HPP
#define SPHERICAL_SHELL_PROJECTOR_3D_HPP

#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"

class SphericalShellProjector3D
{
public:
	SphericalShellProjector3D(Vector3D center,
		std::vector<double> shell_radii,
		double relative_radius_tolerance = 1e-9,
		double absolute_radius_tolerance = 1e-12);

	void ProjectExtensives(Tessellation3D const& tess,
		std::vector<Conserved3D> const& input,
		std::vector<Conserved3D>& output) const;

	void ProjectRates(Tessellation3D const& tess,
		std::vector<Conserved3D> const& input,
		std::vector<Conserved3D>& output) const;

	Vector3D const& GetCenter() const;
	std::vector<double> const& GetShellRadii() const;

private:
	void ProjectLinear(Tessellation3D const& tess,
		std::vector<Conserved3D> const& input,
		std::vector<Conserved3D>& output) const;
	int FindShell(double radius) const;

	Vector3D center_;
	std::vector<double> shell_radii_;
	double relative_radius_tolerance_;
	double absolute_radius_tolerance_;
};

#endif
