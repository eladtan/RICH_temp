/*! \brief Base class for equation of state
  \file equation_of_state.hpp
  \author Almog Yalinewich
 */

#ifndef MULTI_EOS_HPP
#define MULTI_EOS_HPP 1

#include <string>
#include <array>
#include "../two_dimensional/computational_cell_2d.hpp"
#include "equation_of_state.hpp"

/** typedef for tracer vector */
//typedef std::array<double,MAX_TRACERS> tvector;
/** typedef for string vector */
//typedef std::array<bool, MAX_STICKERS> svector;
using std::string;
using std::vector;

//! \brief Base class for equation of state
class MultiEOS : public EquationOfState
{
private:
  const vector<EquationOfState*> eos_vec_;

public:

  MultiEOS(const vector<EquationOfState*> eos_vec) : eos_vec_(eos_vec) {}

  /*! \brief Calculates the thermal energy per unit mass
    \param d Density
    \param p Pressure
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Thermal energy per unit mass
   */
  virtual double dp2e(double d, double p, tvector const& tracers = tvector(),vector<string> const& tracernames = vector<string>())
	  const override;

  /*! \brief Calculates the pressure
    \param d Density
    \param e Specific thermal energy
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Presusre
   */
  virtual double de2p(double d, double e,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

  /*! \brief Calculates the speed of sound
    \param d Density
    \param e Specific thermal energy
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Speed of sound
  */
  virtual double de2c(double d, double e,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

  /*! \brief Calculates the speed of sound
    \param d Density
    \param p Pressure
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Speed of sound
   */
  virtual double dp2c(double d, double p,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

  /*! \brief Calculates the entropy per unit mass
    \param d Density
    \param p Pressure
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Entropy
  */
  virtual double dp2s(double d, double p,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

  /*! \brief Calculates the pressure from the netropy
    \param s Entropy
    \param d Density
    \param tracers Tracers
	\param tracernames The names of the tracers
    \return Entropy
  */
  virtual double sd2p(double s, double d,
	  tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const override;

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

};

#endif
