#ifndef CONSERVED_3D_HPP
#define CONSERVED_3D_HPP 1

#include <array>
#include "3D/elementary/Vector3D.hpp"
#include "../common/equation_of_state.hpp"
#include "computational_cell.hpp"
#ifdef RICH_MPI
	#include <mpi.h>
	#include <mpi_utils/serialize/Serializer.hpp>
#endif // RICH_MPI

//! \brief Conserved variables for a 3D computational cell
class Conserved3D
	#ifdef RICH_MPI
		: public Serializable
	#endif // RICH_MPI
{
public:
	//! \brief Mass
	double mass;

	//! \brief Momentum
	Vector3D momentum;

	//! \brief Energy
	double energy;

	//! \brief Internal energy
	double internal_energy;

	//! \brief Radiation energy
	double Erad;

	//! \brief The radiation energy in a given energy group
	boost::container::small_vector<double, ENERGY_GROUPS_NUM> Eg;

	double Erad_dt;

	double Erad_dt_dt;

	//! \brief Tracers
	std::array<double,MAX_TRACERS> tracers;

	//! \brief Class constructor (sets everything to zero)
	Conserved3D(void);

	/*! \brief Class constructor (does not initialize tracers)
	  \param mass_i Mass
	  \param momentum_i Momentum
	  \param energy_i Energy
	  \param internal_energy_i Internal energy
	 */
	Conserved3D(double mass_i,
		const Vector3D& momentum_i,
		double energy_i,
		double internal_energy_i);

	/*! \brief Class constructor
	  \param mass_i Mass
	  \param momentum_i Momentum
	  \param energy_i Energy
	  \param internal_energy_i Internal energy
	  \param tracers_i Tracers
	 */
	Conserved3D(double mass_i,
		const Vector3D& momentum_i,
		double energy_i,
		double internal_energy_i,
		const std::array<double,MAX_TRACERS>& tracers_i);

	/*! \brief Reduction operator
	  \param diff Difference
	  \return Reference to self
	 */
	Conserved3D& operator-=(const Conserved3D& diff);

	/*! \brief Addition operator
	  \param diff Difference
	  \return Reference to self
	 */
	Conserved3D& operator+=(const Conserved3D& diff);

	/*! \brief Self multiplication operator
	\param s The scalar to multiply
	\return Reference to self
	*/
	Conserved3D& operator*=(double s);

	#ifdef RICH_MPI
		size_t dump(Serializer *serializer) const override;

		size_t load(const Serializer *serializer, std::size_t byteOffset) override;
	#endif // RICH_MPI

	friend std::ostream &operator<<(std::ostream &stream, const Conserved3D &cell);
};

/*! \brief Scalar product operator
  \param s Scalar
  \param c Conserved variable
  \return Product of scalar with conserved
 */
Conserved3D operator*(double s, const Conserved3D& c);

/*! \brief Scalar product operator
\param s Scalar
\param c Conserved variable
\return Product of scalar with conserved
*/
Conserved3D operator*(const Conserved3D& c, double s);

/*! \brief Scalar division operator
  \param c Conserved variable
  \param s Scalar
  \return Ratio
 */
Conserved3D operator/(const Conserved3D& c, double s);

Conserved3D operator+(Conserved3D const& p1, Conserved3D const& p2);

Conserved3D operator-(Conserved3D const& p1, Conserved3D const& p2);


void PrimitiveToConserved(ComputationalCell3D const& cell, double vol, Conserved3D &res);

void PrimitiveToConservedSR(ComputationalCell3D const& cell, double vol, Conserved3D &res,EquationOfState const& eos);

#endif // CONSERVED_3D_HPP
