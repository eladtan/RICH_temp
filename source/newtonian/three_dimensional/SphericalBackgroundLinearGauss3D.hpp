#ifndef SPHERICAL_BACKGROUND_LINEAR_GAUSS_3D_HPP
#define SPHERICAL_BACKGROUND_LINEAR_GAUSS_3D_HPP

#include <array>
#include <memory>
#include <vector>

#include "LinearGauss3D.hpp"
#include "spherical_symmetry/SphericalShellGeometry3D.hpp"

class SphericalBackgroundLinearGauss3D : public SpatialReconstruction3D
{
public:
	SphericalBackgroundLinearGauss3D(
		EquationOfState const& eos,
		Ghost3D const& ghost,
		Vector3D center,
		std::vector<double> shell_radii,
		bool slope_limiter = true,
		double delta_v = 0.2,
		double theta = 0.5,
		double delta_pressure = 0.7);
	SphericalBackgroundLinearGauss3D(
		EquationOfState const& eos,
		Ghost3D const& ghost,
		std::shared_ptr<SphericalShellGeometry3D> shell_geometry,
		bool slope_limiter = true,
		double delta_v = 0.2,
		double theta = 0.5,
		double delta_pressure = 0.7);

	void operator()(Tessellation3D const& tess,
		std::vector<ComputationalCell3D> const& cells,
		double time,
		std::vector<std::pair<ComputationalCell3D,
			ComputationalCell3D>>& result) const override;

	void BuildSlopes(Tessellation3D const& tess,
		std::vector<ComputationalCell3D> const& cells,
		double time) override;

	std::vector<Slope3D>& GetSlopes() override;

	void Interp(ComputationalCell3D& result,
		ComputationalCell3D const& cell,
		size_t cell_index,
		Vector3D const& cell_cm,
		Vector3D const& target,
		EquationOfState const& eos) const;

private:
	struct ShellPrimitive
	{
		double volume = 0;
		double density = 0;
		double pressure = 0;
		double radial_velocity = 0;
		std::array<double, MAX_TRACERS> tracers{};
	};

	void PrepareBackground(Tessellation3D const& tess,
		std::vector<ComputationalCell3D> const& cells) const;
	int FindShell(double radius) const;
	ShellPrimitive RadialState(int shell, double target_radius) const;
	ComputationalCell3D Combine(ComputationalCell3D const& full,
		ComputationalCell3D const& cartesian_background,
		int shell,
		Vector3D const& direction,
		double target_radius) const;

	EquationOfState const& eos_;
	Vector3D center_;
	mutable std::vector<double> shell_radii_;
	std::shared_ptr<SphericalShellGeometry3D> shell_geometry_;
	LinearGauss3D full_reconstruction_;
	LinearGauss3D background_reconstruction_;
	mutable std::vector<ShellPrimitive> shell_states_;
	mutable std::vector<ShellPrimitive> shell_slopes_;
	mutable std::vector<int> shell_ids_;
	mutable std::vector<ComputationalCell3D> background_cells_;
};

#endif
