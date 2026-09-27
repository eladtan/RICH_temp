#ifndef INDIVIDUAL_CHANGE_WAKE_ACCOUNTING_HPP
#define INDIVIDUAL_CHANGE_WAKE_ACCOUNTING_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

/*! \brief Conserved-change accounting for one canonical cell since its last
  activation.  Filled from the flux update of every event the cell is touched
  in, read when the cell activates (mass-loss timestep limit) and while it is
  passive (conserved-change wake), reset at activation.
*/
struct IndividualConservedChange
{
  double mass_at_activation = 0;
  double energy_at_activation = 0;
  double mass_loss = 0;
  double thermal_loss = 0;
  double mass_abs_change = 0;
  double energy_abs_change = 0;
  // Set by a conserved-change wake; at the cell's next activation the
  // accumulated ratio is sampled (the wake's accuracy check) and cleared.
  bool change_woken = false;
  // The absolute changes when the wake was first issued, so the sample can
  // separate the change detected at issue from the change taken afterwards.
  double mass_abs_change_at_issue = 0;
  double energy_abs_change_at_issue = 0;
};

//! Relative mass or energy change accumulated since activation, the larger of the two.
inline double IndividualConservedChangeRatio(IndividualConservedChange const& change)
{
  double ratio = 0;
  if(change.mass_at_activation > 0)
    ratio = std::max(ratio, change.mass_abs_change / change.mass_at_activation);
  if(std::abs(change.energy_at_activation) > 0)
    ratio = std::max(ratio, change.energy_abs_change / std::abs(change.energy_at_activation));
  return ratio;
}

//! Change taken after the wake was first issued, relative to the activation values.
inline double IndividualConservedChangeRatioAfterIssue(IndividualConservedChange const& change)
{
  double ratio = 0;
  if(change.mass_at_activation > 0)
    ratio = std::max(ratio, (change.mass_abs_change - change.mass_abs_change_at_issue) /
      change.mass_at_activation);
  if(std::abs(change.energy_at_activation) > 0)
    ratio = std::max(ratio, (change.energy_abs_change - change.energy_abs_change_at_issue) /
      std::abs(change.energy_at_activation));
  return ratio;
}

/*! \brief Accuracy accounting of the conserved-change wake.  Every woken cell
  is issued once and leaves exactly once: sampled at its next activation, or
  censored when its accumulators are discarded first (AMR, load balance, mode
  entry, re-seed).  A censored cell's ratio so far is a lower bound on its
  ratio at activation, so a censored ratio above the limit is still a
  violation.  The ratio at issue (what the guard detected) and the change
  after issue (what the scheduling response let through) are kept apart.
  Rank-local; the caller reduces and reports the tally.
*/
class IndividualChangeWakeAccounting
{
public:
  struct Tally
  {
    unsigned long long issued = 0;
    unsigned long long sampled = 0;
    unsigned long long sampled_above = 0;
    unsigned long long censored = 0;
    unsigned long long censored_above = 0;
    double sampled_largest = 0;
    double censored_largest = 0;
    std::size_t sampled_largest_id = 0;
    // Largest ratio at issue, and largest change after issue at activation.
    double issued_largest = 0;
    double sampled_after_issue_largest = 0;
    unsigned long long sampled_after_issue_above = 0;
  };

  void Issue(IndividualConservedChange& change)
  {
    if(!change.change_woken)
    {
      ++tally_.issued;
      tally_.issued_largest = std::max(tally_.issued_largest, IndividualConservedChangeRatio(change));
      change.mass_abs_change_at_issue = change.mass_abs_change;
      change.energy_abs_change_at_issue = change.energy_abs_change;
    }
    change.change_woken = true;
  }

  void SampleAtActivation(IndividualConservedChange& change, std::size_t id, double limit)
  {
    if(!change.change_woken)
      return;
    double const ratio = IndividualConservedChangeRatio(change);
    double const after_issue = IndividualConservedChangeRatioAfterIssue(change);
    ++tally_.sampled;
    tally_.sampled_above += ratio > limit ? 1 : 0;
    if(ratio > tally_.sampled_largest)
    {
      tally_.sampled_largest = ratio;
      tally_.sampled_largest_id = id;
    }
    tally_.sampled_after_issue_largest = std::max(tally_.sampled_after_issue_largest, after_issue);
    tally_.sampled_after_issue_above += after_issue > limit ? 1 : 0;
    change.change_woken = false;
  }

  //! Woken cells still waiting for activation (issued - sampled - censored over the run).
  static unsigned long long Outstanding(std::vector<IndividualConservedChange> const& changes)
  {
    unsigned long long outstanding = 0;
    for(IndividualConservedChange const& change : changes)
      outstanding += change.change_woken ? 1 : 0;
    return outstanding;
  }

  void Drain(std::vector<IndividualConservedChange>& changes, double limit)
  {
    for(IndividualConservedChange& change : changes)
    {
      if(!change.change_woken)
        continue;
      double const ratio = IndividualConservedChangeRatio(change);
      ++tally_.censored;
      tally_.censored_above += ratio > limit ? 1 : 0;
      tally_.censored_largest = std::max(tally_.censored_largest, ratio);
      change.change_woken = false;
    }
  }

  Tally const& GetTally(void) const {return tally_;}

  void ClearTally(void) {tally_ = Tally();}

private:
  Tally tally_;
};

#endif // INDIVIDUAL_CHANGE_WAKE_ACCOUNTING_HPP
