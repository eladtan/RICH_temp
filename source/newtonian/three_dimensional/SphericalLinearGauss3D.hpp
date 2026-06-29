/*! \file SphericalLinearGauss3D.hpp
\brief Linear interpolation using weighted least-squares gradient in spherical
       coordinate space (d/dr, d/dtheta, d/dphi). Each cell/neighbor velocity is
       projected onto its own local spherical frame; the interpolated velocity at a
       face centroid is converted back to Cartesian using the basis at that point.
\author Elad Steinberg
*/

#ifndef SPHERICAL_LINEAR_GAUSS3D_HPP
#define SPHERICAL_LINEAR_GAUSS3D_HPP 1

#include "../common/equation_of_state.hpp"
#include "SpatialReconstruction3D.hpp"
#include <cmath>
#include "../../misc/universal_error.hpp"
#include "Ghost3D.hpp"

//! \brief Linear LSQ interpolation in spherical coordinates (d/dr, d/dtheta, d/dphi)
class SphericalLinearGauss3D : public SpatialReconstruction3D
{
public:
	enum class FaceRadiusPolicy
	{
		PhysicalFaceCM,
		SphericalShellGeneratorAverage
	};

	/*! \brief Class constructor
	\param eos Equation of state
	\param ghost The ghost point generator
	\param origin Centre of the spherical coordinate system
	\param slf Slope limiter flag
	\param delta_v The GradV*L/Cs ratio needed for slope limiter
	\param theta The theta from tess in slope limiter
	\param delta_P The pressure ratio for shock detection
	\param SR Flag for SR
	\param calc_tracers Names of tracers for which to calc the slope
	\param skip_key The sticker name to skip cells for slope limit
	\param pressure_calc Determine whether the pressure should be recalculated
	\param apply_principal_limit Enable principal-frame velocity limiting
	\param velocity_radial_extrapolation When true, velocity is extrapolated to the average radius between generators rather than face CM radius, improving spherical symmetry preservation
	\param face_radius_policy Effective face-radius policy for spherical reconstruction
	\param shell_radius_abs_tol Absolute same-shell radius tolerance
	\param shell_radius_rel_tol Relative same-shell radius tolerance
	*/
	SphericalLinearGauss3D(EquationOfState const& eos, Ghost3D const& ghost,
		Vector3D const& origin = Vector3D(),
		bool slf = true, double delta_v = 0.2,
		double theta = 0.5, double delta_P = 0.7, bool SR = false,
		const vector<string>& calc_tracers = vector<string>(),
		const string& skip_key = string(), bool pressure_calc = true,
		bool apply_principal_limit = false,
		bool velocity_radial_extrapolation = false,
		FaceRadiusPolicy face_radius_policy = FaceRadiusPolicy::PhysicalFaceCM,
		double shell_radius_abs_tol = 1e-12,
		double shell_radius_rel_tol = 1e-7);

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, double time,
		vector<pair<ComputationalCell3D, ComputationalCell3D> > &res) const override;

	/*! \brief Interpolates a cell in spherical coordinates
	\param res The interpolated value (output, Cartesian velocity)
	\param cell The primitives of the cell (Cartesian velocity)
	\param cell_index The index of the cell
	\param cm The cell's center of mass
	\param target The location of the interpolation
	\param eos The equation of state
	*/
	void Interp(ComputationalCell3D &res, ComputationalCell3D const& cell, size_t cell_index,
		Vector3D const& cm, Vector3D const& target, EquationOfState const& eos) const;

	vector<Slope3D>& GetSlopes(void) override;

	vector<Slope3D>& GetSlopesUnlimited(void) const;

	void BuildSlopes(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, double time) override;

	/*! \brief Returns the per-cell spherical basis vectors
	\param er Radial basis vectors (output)
	\param etheta Polar basis vectors (output)
	\param ephi Azimuthal basis vectors (output)
	*/
	void GetBasis(vector<Vector3D>& er, vector<Vector3D>& etheta, vector<Vector3D>& ephi) const;

	const vector<Slope3D>& GetPrevSlopes() const { return prev_rslopes_; }
	const vector<Slope3D>& GetPrevSlopesUnlimited() const { return prev_naive_rslopes_; }

private:
	EquationOfState const& eos_;
	Ghost3D const& ghost_;
	const Vector3D origin_;
	mutable vector<Slope3D> rslopes_;
	mutable vector<Slope3D> naive_rslopes_;
	mutable vector<Slope3D> prev_rslopes_;
	mutable vector<Slope3D> prev_naive_rslopes_;
	mutable vector<Vector3D> er_;
	mutable vector<Vector3D> etheta_;
	mutable vector<Vector3D> ephi_;
	const bool slf_;
	const double shockratio_;
	const double diffusecoeff_;
	const double pressure_ratio_;
	const bool SR_;
	const vector<string> calc_tracers_;
	const string skip_key_;
	const bool pressure_calc_;
	const bool apply_principal_limit_;
	const bool velocity_radial_extrapolation_;
	const FaceRadiusPolicy face_radius_policy_;
	const double shell_radius_abs_tol_;
	const double shell_radius_rel_tol_;
	mutable std::vector<bool> is_pole_cell_;

	SphericalLinearGauss3D(const SphericalLinearGauss3D& origin);
	SphericalLinearGauss3D& operator=(const SphericalLinearGauss3D& origin);
};

#endif //SPHERICAL_LINEAR_GAUSS3D_HPP
