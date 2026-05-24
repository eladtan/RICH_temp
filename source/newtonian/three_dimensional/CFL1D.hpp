#ifndef CFL1D_HPP
#define CFL1D_HPP 1

#include "time_step_function3D.hpp"
#include "SourceTerm3D.hpp"

class CFL1D : public TimeStepFunction3D
{
public:
	explicit CFL1D(double cfl, double SourceCFL, SourceTerm3D const& source,
		std::vector<std::string> no_calc = std::vector<std::string>(), bool debug = false);

	double operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, const EquationOfState& eos,
		const vector<Vector3D>& face_velocities, const double time);

	void SetTimeStep(double dt) override;
	double GetTimeStep(void) const override;
	double SuggestTimeStep(void) const override;

	void SetPointVelocities(const vector<Vector3D>* pv) { point_velocities_ = pv; }

private:
	const double cfl_, sourcecfl_;
	SourceTerm3D const& source_;
	std::vector<std::string> const no_calc_;
	const bool debug_;
	bool first_try_;
	double dt_;
	double dt_first_;
	double dt_suggest_;
	double last_time_;
	const vector<Vector3D>* point_velocities_ = nullptr;
};

#endif // CFL1D_HPP
