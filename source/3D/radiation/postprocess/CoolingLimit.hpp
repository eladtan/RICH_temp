#ifndef IMC_POSTPROCESS_TDE_COOLING_LIMIT_HPP
#define IMC_POSTPROCESS_TDE_COOLING_LIMIT_HPP

#include <cstddef>
#include <functional>
#include <string>

#include "PostProcessConfig.hpp"
#include "PostProcessRuntime.hpp"

namespace imc_postprocess_tde {

// Volume emission holds every outside cell at its snapshot temperature with
// Fleck factor 1. A cell whose radiative cooling time is far shorter than the
// hydro time step is not at a physical temperature in the snapshot: the
// hydro's Fleck step removes at most (aT^4 - E) V / beta of its energy per step
// (about a quarter of its thermal energy when T >> T_rad), so the cooling layer
// behind a shock spans t_hydro / t_cool times too much gas. Emitting
// c kappa_P (aT^4 - E) V from that temperature overcounts by the same factor.
// The cooling limit caps the cell's net emission at the hydro's per-step rate
// and lowers the cell temperature to the value that gives it; transport is
// unchanged (Fleck factor 1, real absorption).

enum class CoolingLimitKind { Unchanged, Hot, Cold };

struct CoolingLimitOutcome
{
    CoolingLimitKind kind = CoolingLimitKind::Unchanged;
    double temperature = 0.0;  // K; the input temperature when unchanged
    double netBefore = 0.0;    // erg/s, c kappa_P (aT^4 - E) V at the input temperature
    double netAfter = 0.0;     // erg/s, the same at the returned temperature
};

// Material of one cell (cgs).
struct CoolingLimitMaterial
{
    // Planck absorption coefficient [1/cm] at a temperature.
    std::function<double(double)> planckOpacity;
    // Internal energy density [erg/cm^3] at a temperature (cells colder than
    // the radiation only).
    std::function<double(double)> internalEnergyDensity;
    // Heat capacity per volume [erg/(cm^3 K)] at the snapshot temperature.
    double heatCapacity = 0.0;
    // Internal energy density [erg/cm^3] of the snapshot state.
    double snapshotInternalEnergyDensity = 0.0;
};

// One cell with temperature T, radiation energy density E [erg/cm^3] and
// volume V [cm^3]:
//  - aT^4 > E: the net emission is capped at (aT^4 - E) V / (beta t_hydro),
//    beta = max(1, Cv T / (2 u)) 4 a T^3 / Cv as in the hydro (Diffusion.cpp);
//  - aT^4 < E: the net absorption is capped at the energy that heats the cell
//    to the radiation temperature, (u(T_rad) - u(T)) V / t_hydro (the Fleck
//    linearisation would overshoot T_rad here);
//  - otherwise (|aT^4 - E| <= 1e-12 max(aT^4, E)), or when the cap is not
//    reached, the cell is unchanged.
// A capped cell gets the temperature between T and T_rad at which
// c kappa_P(T') (aT'^4 - E) V equals the cap. Throws on an invalid state.
CoolingLimitOutcome LimitCellCooling(
    double temperature,
    double radiationEnergyDensity,
    double volume,
    double hydroTime,
    CoolingLimitMaterial const& material);

// Applies LimitCellCooling to every owned cell at or beyond the flux-source
// surface, with the Planck opacity the multigroup transport emits with (the
// grey table with Planck scaling, the Planck mean of the unscaled group
// opacities without scaling, alpha(T') times it with Rosseland scaling) and
// the scenario EOS, and records the summary in runtime.coolingLimit. t_hydro
// is cfg.volumeEmissionCoolingLimitDt when positive, else the RICH Courant
// step (CFL cfg.volumeEmissionCoolingLimitCfl, face velocities from the gas).
// When that opacity is not the grey table and
// cfg.volumeEmissionCoolingLimitPerPass is set, the snapshot temperatures of
// the cells it changes are kept for ApplyGreyPassCoolingLimit. Call once,
// after InitializeFluxSourceSurface and before the first
// ConfigureFluxSourceForCurrentDecomposition. Collective.
void ApplyVolumeEmissionCoolingLimit(
    Config const& cfg,
    PostprocessRuntime& runtime);

// The grey pass emits with the grey table's Planck mean. When the multigroup
// limit used another one (and cfg.volumeEmissionCoolingLimitPerPass), restores
// the snapshot temperatures of the cells it changed and limits every outside
// cell again with the grey table and the same t_hydro: temperatures and, in
// emission mode, runtime.coolingLimitEmissionCaps are replaced; the summary
// goes to runtime.greyCoolingLimit (rank 0 prints
// VOLUME_EMISSION_COOLING_LIMIT pass=grey). No-op otherwise. Call once, after
// the multigroup pass and before the grey pass's first source setup.
// Collective.
void ApplyGreyPassCoolingLimit(
    Config const& cfg,
    PostprocessRuntime& runtime);

// Fills runtime.escapeOriginHotCells from the current (limited) temperatures
// of the cells at or beyond the flux-source surface. Collective.
void BuildEscapeOriginHotCells(
    Config const& cfg,
    PostprocessRuntime& runtime);

// Emission scale of a cell under the emission-mode cooling limit
// (runtime.coolingLimitEmissionCaps): the factor on its gross emission
// `gross` (erg/s) at which its net emission, gross x scale - absorbed, equals
// its cap, where `absorbed` is its absorption of the snapshot radiation with
// the same Planck mean. At most 1; 1 for a cell without a cap.
double CoolingLimitEmissionScale(
    PostprocessRuntime const& runtime,
    size_t cellID,
    double gross,
    double absorbed);

// Multigroup source-spectrum diagnostic of the volume emission, over the cells
// at or beyond the flux-source surface at their current temperatures: per
// group the gross emission c kappa_g u_g(T) V, the absorption of the
// snapshot's own group radiation c kappa_g rho E_g V and their difference
// (SOURCE_GROUP_VOLUME), and per temperature decade the cells, their gross
// emission and the part of it above 0.3 keV (SOURCE_GROUP_VOLUME_T). `stage`
// labels the lines (before or after the cooling limit). No-op without volume
// emission or with a single group. Collective; rank 0 prints.
void ReportVolumeSourceGroups(
    Config const& cfg,
    PostprocessRuntime& runtime,
    OpacityCalculator const& opacity,
    std::string const& stage);

// The DDMC comoving-frame shift compresses (or expands) a packet for as long
// as it stays in the frozen flow, so it can drift up across the top of its
// cell's diffusive band and escape in a transparent band, with its weight
// multiplied by the same factor. transport.ddmc-compression-max-dynamical-times
// bounds each packet's shift by a clock that advances at
// max(|div v|, |v| / max(r, cell size)) on every DDMC and IMC step.
// Rank 0 prints COMPRESSION with the counters of the last transport:
// band-top conversions and their energy, DDMC steps in which a packet's clock
// ran out (once per packet that reaches the limit in DDMC), and
// the work gained (compression) and lost (expansion). Collective; no-op
// without DDMC.
void PrintCompressionReport(
    Config const& cfg,
    RadiationIMC const& physics,
    std::string const& label,
    size_t generation,
    int rank);

} // namespace imc_postprocess_tde
#endif
