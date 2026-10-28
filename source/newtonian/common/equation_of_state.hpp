/*! \brief Base class for equation of state
  \file equation_of_state.hpp
  \author Almog Yalinewich
 */

#ifndef EQUATION_OF_STATE_HPP
#define EQUATION_OF_STATE_HPP 1

#include <string>
#include <array>
#include <cstddef>
#include <cstdint>
#include "../two_dimensional/computational_cell_2d.hpp"
#include "utils/PortableFunction.hpp"

/** typedef for tracer vector */
//typedef std::array<double,MAX_TRACERS> tvector;
/** typedef for string vector */
//typedef std::array<bool, MAX_STICKERS> svector;
using std::string;
using std::vector;

/*! \brief The arrays an equation of state works on all at once. Item i's density is density[i * stride], its
  second argument second[i * stride], its tracers tracers[i * stride] onwards, and its result goes to
  result[i * resultStride]. The arrays are where the call runs: on the host, or on the device.
*/
struct EosArrays
{
	//! \brief The number of items
	std::size_t n = 0;
	//! \brief The distance between two items' inputs, in doubles
	std::size_t stride = 1;
	const double *density = nullptr;
	//! \brief The call's second argument: pressure or internal energy
	const double *second = nullptr;
	//! \brief Each item's first tracer, followed by its others (null if there are none)
	const double *tracers = nullptr;
	std::size_t tracersNum = 0;
	//! \brief The tracers' names, for the scalar forms (null for none)
	const vector<string> *tracerNames = nullptr;
	double *result = nullptr;
	//! \brief The distance between two items' results, in doubles
	std::size_t resultStride = 1;
	//! \brief Per item, or null: an item whose status is not 0 is skipped; one where the scalar form throws is set to `failure`
	std::uint8_t *status = nullptr;
	std::uint8_t failure = 1;
};

//! \brief Base class for equation of state
class EquationOfState
{
public:

  /*! \brief Calculates the thermal energy per unit mass
    \param d Density
    \param p Pressure
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Thermal energy per unit mass
   */
  virtual double dp2e(double d, double p, tvector const& tracers = tvector(),vector<string> const& tracernames = vector<string>())
	  const = 0;

  /*! \brief Calculates the pressure
    \param d Density
    \param e Specific thermal energy
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Presusre
   */
  virtual double de2p(double d, double e,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const = 0;

  /*! \brief Calculates the speed of sound
    \param d Density
    \param e Specific thermal energy
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Speed of sound
  */
  virtual double de2c(double d, double e,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const = 0;

  /*! \brief Calculates the speed of sound
    \param d Density
    \param p Pressure
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Speed of sound
   */
  virtual double dp2c(double d, double p,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const = 0;

  /*! \brief Calculates the entropy per unit mass
    \param d Density
    \param p Pressure
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Entropy
  */
  virtual double dp2s(double d, double p,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const = 0;

  /*! \brief Calculates the pressure from the netropy
    \param s Entropy
    \param d Density
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Entropy
  */
  virtual double sd2p(double s, double d,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const = 0;

/*! \brief Calculates the heat capacity (Cv) per unit volume
    \param d Density
    \param T Temperature
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Heat capacity per unit volume
  */
  virtual double dT2cv(double const d, double const T,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const;

/*! \brief Calculates the temperature
    \param d Density
    \param e Thermal energy per unit mass
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return The temperature
  */
  virtual double de2T(double const d, double const e,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const;

/*! \brief Calculates the internal energy
    \param d Density
    \param T temperature
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Thermal energy per unit mass
  */
  virtual double dT2e(double const d, double const T,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const;

  /*! \brief dp2e of every item: the thermal energies from densities and pressures. By default, calls dp2e for each
      item in turn, on the host.
    \param arrays The items
    \param where Where the call runs (the device only if RunsOnDevice())
  */
  virtual void dp2eArrays(EosArrays const& arrays, portable::Where where) const;

  //! \brief de2p of every item: the pressures from densities and thermal energies (as dp2eArrays)
  virtual void de2pArrays(EosArrays const& arrays, portable::Where where) const;

  //! \brief dp2c of every item: the sound speeds from densities and pressures (as dp2eArrays)
  virtual void dp2cArrays(EosArrays const& arrays, portable::Where where) const;

  //! \brief de2c of every item: the sound speeds from densities and thermal energies (as dp2eArrays)
  virtual void de2cArrays(EosArrays const& arrays, portable::Where where) const;

  /*! \brief Whether the array forms can run on the device
    \return False unless the equation of state overrides them for the device
  */
  virtual bool RunsOnDevice(void) const
  {
	  return false;
  }

  /*! \brief Whether the methods may be called from several threads at once (no state changes in a call)
    \return False unless the equation of state says so
  */
  virtual bool ThreadSafe(void) const
  {
	  return false;
  }

  virtual ~EquationOfState(void);
};

#endif
