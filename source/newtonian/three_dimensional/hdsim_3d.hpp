#ifndef HDSIM_3D_HPP
#define HDSIM_3D_HPP 1

#include <cassert>
#include <chrono>
#include <MeshDecomposer3D/hilbert/HilbertOrder3D.hpp>
#include "misc/utils.hpp"
#include "computational_cell.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "conserved_3d.hpp"
#include "../common/equation_of_state.hpp"
#include "point_motion_3d.hpp"
#include "time_step_function3D.hpp"
#include "flux_calculator_3d.hpp"
#include "cell_updater_3d.hpp"
#include "extensive_updater3d.hpp"
#include "SourceTerm3D.hpp"
#include "newtonian/three_dimensional/simulation/ProgressTracker.hpp"
#include "CostCalculator3D.hpp"

#ifdef RICH_MPI
  #include <mpi.h>
  #include "mpi/mpi_commands.hpp"
  #include "mpi/ExchangeChain.hpp"
#endif

//! \brief Three dimensional simulation
class HDSim3D
{
public:
  /*! \brief Class constructor
    \param tess Tessellation
    \param cells Initial computational cells
    \param eos Equation of state
    \param pm Point motion scheme
    \param tsc Time step calculator
    \param fc Flux calculator
    \param cu Cell updater
    \param eu Extensive updater
    \param source Source term
    \param tsn The names of the tracers and stickers, first is the tracers and second is stickers
    \param SR Special relativity flag
  */
  HDSim3D(Tessellation3D& tess,
	  vector<ComputationalCell3D>& cells,
    vector<Conserved3D>& extensives,
	  const EquationOfState& eos,
    ProgressTracker &pt,
	  const PointMotion3D& pm,
	  TimeStepFunction3D& tsc,
	  const FluxCalculator3D& fc,
	  const CellUpdater3D& cu,
	  const ExtensiveUpdater3D& eu,
	  const	SourceTerm3D& source,
	  const pair<vector<string>, vector<string> >& tsn,
	  bool SR=false
    #ifdef RICH_MPI
      , std::shared_ptr<CostCalculator3D> cost_calc = std::make_shared<CostCalculator3D>()
    #endif // RICH_MPI  
  );

  //! \brief Advances the simulation in time (first order)
  void timeAdvance();
  //! \brief Advances the simulation in time (second order)
  void timeAdvance2();

  /*! \brief Third order time advance
   */
  void timeAdvance3();

  /*! \brief Third order time advance
   */
  void timeAdvance32();

  /*! \brief Third order time advance
   */
  void timeAdvance33();

  /*! \brief Fourth order time advance
   */
  void timeAdvance4();

  /*! \brief Access to tessellation
    \return Tessellation
   */
  const Tessellation3D& getTessellation(void) const;

  /*! \brief Access to computational cells
    \return Computational cells
   */
  const vector<ComputationalCell3D>& getCells(void) const;
  /*! \brief Access to extensive cells
  \return Extensive cells
  */
  const vector<Conserved3D>& getExtensives(void) const;
  /*! \brief Access to tessellation
  \return Tessellation
  */
  Tessellation3D& getTessellation(void);

  /*! \brief Access to computational cells
  \return Computational cells
  */
  vector<ComputationalCell3D>& getCells(void);
  /*! \brief Access to extensive cells
  \return Extensive cells
  */
  vector<Conserved3D>& getExtensives(void);

  double getTime(void) const;

  size_t getCycle(void) const;

  double getTimeStep(void) const {return this->tsc_.GetTimeStep();}

  double suggestTimeStep(void) const {return this->tsc_.SuggestTimeStep();}

  void SetTimeStep(double dt) {this->tsc_.SetTimeStep(dt);}

  #ifdef RICH_MPI
    const ExchangeChain &GetExchangeChain(void) const {return this->exchange_chain_;}
  #endif // RICH_MPI

  #ifdef RICH_MPI
    std::shared_ptr<CostCalculator3D> cost_calc_;
  #endif // RICH_MPI
  
private:
  Tessellation3D& tess_;
  const EquationOfState& eos_;
  vector<ComputationalCell3D> &cells_;
  vector<Conserved3D> &extensive_;
  const PointMotion3D& pm_;
  TimeStepFunction3D& tsc_;
  const FluxCalculator3D& fc_;
  const CellUpdater3D& cu_;
  const ExtensiveUpdater3D& eu_;
  const	SourceTerm3D &source_;
  const ProgressTracker &pt_;
  #ifdef RICH_MPI
    ExchangeChain exchange_chain_;
  #endif // RICH_MPI
};

#endif // HDSIM_3D_HPP
