/*! \file computational_cell.hpp
  \brief A container for the hydrodynamic variables
  \author Almog Yalinewich
 */

#ifndef COMPUTATIONAL_CELL3D_HPP
#define COMPUTATIONAL_CELL3D_HPP 1

#include <array>
#include "3D/elementary/Vector3D.hpp"
#include "../two_dimensional/computational_cell_2d.hpp"
#include <boost/container/small_vector.hpp>
#ifdef RICH_MPI
	#include <mpi_utils/serialize/Serializer.hpp>
	#include "mpi/mpi_commands.hpp"
#endif // RICH_MPI

#ifndef ENERGY_GROUPS_NUM
#define ENERGY_GROUPS_NUM 1
#endif

 //! \brief Container for the hydrodynamic variables
class ComputationalCell3D
				#ifdef RICH_MPI
						: public Serializable
				#endif // RICH_MPI
{
public:
	//! \brief Density
	double density;

	//! \brief Pressure
	double pressure;

	//! \brief Internal energy
	double internal_energy;

	//! \brief Temperature
	double temperature;

	//! \brief Unique ID
	size_t ID;

	//! \brief Velocity
	Vector3D velocity;

	double dt;
	
	//! \brief Radiation enregy per unit mass
	double Erad;

	//! \brief The radiation energy per unit mass in a given energy group
	boost::container::small_vector<double, ENERGY_GROUPS_NUM> Eg;

	double Erad_dt;

	double Erad_dt_dt;

	double cs;
	
	static vector<string> tracerNames;
	static vector<string> stickerNames;
	static std::array<double, ENERGY_GROUPS_NUM + 1> energyBoundaries;

	//! \brief Tracers
	std::array<double,MAX_TRACERS> tracers;

	//! \brief Stickers
	std::array<bool,MAX_STICKERS> stickers;

	//! \brief Null constructor
	ComputationalCell3D(void);

	/*! \brief Class constructor
	  \param density_i Density
	  \param pressure_i Pressure
	  \param velocity_i Velocity
	  \param internal_energy_i Internal energy per unit mass
	  \param ID_i The ID
	  */
	ComputationalCell3D(double density_i,
		double pressure_i,double internal_energy_i, size_t ID_i,
		const Vector3D& velocity_i);

	/*! \brief Class constructor
	  \param density_i Density
	  \param pressure_i Pressure
	  \param internal_energy_i Internal energy per unit mass
	  \param velocity_i Velocity
	  \param tracers_i Tracers
	  \param stickers_i Stickers
	  \param ID_i The ID
	  */
	ComputationalCell3D(double density_i,
		double pressure_i,
		double internal_energy_i,
		size_t ID_i,
		const Vector3D& velocity_i,
		const std::array<double, MAX_TRACERS>& tracers_i,
		const std::array<bool, MAX_STICKERS>& stickers_i);

  /*! \brief Copy constructor
    \param other Source
   */
  ComputationalCell3D(const ComputationalCell3D& other);
	
	/*! \brief Self increment operator
	\param other Addition
	\return Reference to self
	*/
	ComputationalCell3D& operator+=(ComputationalCell3D const& other);
	/*! \brief Self reduction operator
	\param other Reduction
	\return Reference to self
	*/
	ComputationalCell3D& operator-=(ComputationalCell3D const& other);

	/*! \brief Self multiplication operator
	\param s The scalar to multiply
	\return Reference to self
	*/
	ComputationalCell3D& operator*=(double s);

	/*! \brief Self decrement operator
	\param other difference
	\return Reference to self
	*/
	ComputationalCell3D& operator=(ComputationalCell3D const& other);

/**
 * \brief Overloaded output stream operator for ComputationalCell3D class.
 *
 * This operator allows for convenient printing of ComputationalCell3D objects to an output stream.
 * The output format is a space-separated list of the cell's properties: density, pressure, internal energy,
 * temperature, ID, velocity components (x, y, z), Erad, Eg (energy groups), 
 * tracer values, and sticker values.
 *
 * \param stream The output stream to which the cell's data will be written.
 * \param cell The ComputationalCell3D object to be printed.
 *
 * \return The output stream with the cell's data appended.
 */
	friend std::ostream &operator<<(std::ostream &stream, const ComputationalCell3D &cell);

	#ifdef RICH_MPI
		size_t dump(Serializer *serializer) const override;

		size_t load(const Serializer *serializer, std::size_t byteOffset) override;
	#endif//RICH_MPI

};

/*! \brief Term by term addition
\param p1 Computational Cell
\param p2 Computational Cell
\return Computational Cell
*/
ComputationalCell3D operator+(ComputationalCell3D const& p1, ComputationalCell3D const& p2);

/*! \brief Term by term subtraction
\param p1 Computational Cell
\param p2 Computational Cell
\return Computational Cell
*/
ComputationalCell3D operator-(ComputationalCell3D const& p1, ComputationalCell3D const& p2);

/*! \brief Scalar division
\param p Computational Cell
\param s Scalar
\return Computational Cell
*/
ComputationalCell3D operator/(ComputationalCell3D const& p, double s);

/*! \brief Scalar multiplication on the right
\param p Computational Cell
\param s Scalar
\return Computational Cell
*/
ComputationalCell3D operator*(ComputationalCell3D const& p, double s);

/*! \brief Scalar multiplication on the left
\param s Scalar
\param p Computational Cell
\return Computational Cell
*/
ComputationalCell3D operator*(double s, ComputationalCell3D const& p);

/*! \brief Perform addition and scalar multiplication
  \param res Result
  \param other Addition
  \param scalar Scalar
 */
void ComputationalCellAddMult(ComputationalCell3D &res, ComputationalCell3D const& other, double scalar);

/*! \brief Vectorized add-mult for 3 derivatives at once
  \param res Result
  \param dx X derivative
  \param dy Y derivative
  \param dz Z derivative
  \param sx X scalar
  \param sy Y scalar
  \param sz Z scalar
 */
void ComputationalCellAddMult3(ComputationalCell3D &res, 
	ComputationalCell3D const& dx, ComputationalCell3D const& dy, ComputationalCell3D const& dz,
	double sx, double sy, double sz);

/*! \brief Swap computation cell
  \param cell Result
  \param other Other cell
 */
void ReplaceComputationalCell(ComputationalCell3D &cell, ComputationalCell3D const& other);

//! \brief Class for 3D spatial interpolations
class Slope3D
#ifdef RICH_MPI
	: public Serializable
#endif // RICH_MPI
{
public:
	//! \brief Slope in the x direction
	ComputationalCell3D xderivative;

	//! \brief Slope in the y direction
	ComputationalCell3D yderivative;

	//! \brief Slope in the z direction
	ComputationalCell3D zderivative;
	/*!
	\brief Class constructor
	\param x The x derivative
	\param y The y derivative
	\param z The z derivative
	*/
	Slope3D(ComputationalCell3D const& x, ComputationalCell3D const& y, ComputationalCell3D const& z);
	//! \brief Default constructor
	Slope3D(void);

	#ifdef RICH_MPI
		size_t dump(Serializer *serializer) const override;

		size_t load(const Serializer *serializer, std::size_t byteOffset) override;
	#endif//RICH_MPI

};


#endif // COMPUTATIONAL_CELL3D_HPP
