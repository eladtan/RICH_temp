# Post-process volume emission: the cooling limit

The post-process volume emission (`volume-emission.enabled`) emits from every
cell outside the flux-source surface at its snapshot temperature with Fleck
factor 1. That is right for a cell whose temperature is physical. It is wrong
for a cell whose radiative cooling time `t_cool` is far shorter than the hydro
time step.

## Why such a cell is too hot in the snapshot

In the hydro, the Fleck step linearises `aT^4` about the old temperature. When
`x = beta c kappa_P dt >> 1` one step removes at most

    (aT^4 - E) V / beta,   beta = max(1, Cv T / (2 u)) 4 a T^3 / Cv

(`source/Radiation/Diffusion.cpp`), about a quarter of the excess thermal
energy when `T >> T_rad`. A shocked cell that would physically cool in
microseconds therefore takes several steps, and the cooling layer behind a
shock spans about `t_hydro / t_cool` times too much hot gas. The radiative
equilibrium temperature is not changed by the Fleck factor; the relaxation
towards it is slowed.

Emitting `c kappa_P (aT^4 - E) V` from the snapshot temperature for the whole
source interval then overcounts by about `t_hydro / t_cool`. On snap_109 of the
TDE run this made the grey Monte Carlo luminosity 3e44 erg/s against 1e42 from
FLD: about ten hot (5-6e5 K), low-density (1e-14 to 1e-13 g/cm^3) cells with
cooling times of 1e-6 s emitted millions of times their thermal energy per
source interval.

## The limit (`volume-emission.cooling-limit`, default on)

Once per run, after the flux-source surface, FLD luminosity and photospheres are
built from the snapshot temperatures and before any source setup, every owned
cell at or beyond the surface is limited (`postprocess/CoolingLimit.cpp`):

- `aT^4 > E`: the net emission is capped at the hydro's per-step exchange per
  hydro time, `(aT^4 - E) V / (beta t_hydro)`.
- `aT^4 < E`: the net absorption is capped at the energy that heats the cell to
  the radiation temperature, `(u(T_rad) - u(T)) V / t_hydro` (the linearisation
  overshoots `T_rad` for cold gas).
- Cells in equilibrium to round-off, and cells below the cap, are unchanged.

A capped cell gets the temperature between `T` and `T_rad` at which
`c kappa_P(T') (aT'^4 - E) V` equals the cap (bisection on `aT'^4 - E`,
scenario EOS). `kappa_P` is the Planck mean the multigroup transport emits
with: the grey table's with `--opacity-scaling.mode planck` (the scaled group
opacities reproduce it), and the Planck mean of the unscaled group opacities
with `none`, the setting for post-processing a multigroup hydro run (the log
line says `planck_opacity=grey|multigroup`).

The grey pass emits with the grey table's Planck mean. When the multigroup
pass's limit used another one (`none`, or Rosseland scaling), the grey pass
redoes the limit (`--volume-emission.cooling-limit-per-pass`, default on):
at its start it restores the snapshot temperatures of the cells the first
limit changed and limits every outside cell again with the grey table and the
same `t_hydro`, which prints a second line, `VOLUME_EMISSION_COOLING_LIMIT
pass=grey ... planck_opacity=grey`. The two Planck means can differ by orders
of magnitude (in snap_429 the grey one is about 550 times the unscaled
multigroup one on average). Without the per-pass limit the grey pass
inherited temperatures solved with the multigroup mean: a cold cell, heated
only until `c kappa_P,MG (E - aT'^4) V` reached its cap, absorbed about
`kappa_P,grey / kappa_P,MG` times the cap in the grey pass and emitted too
little. In snap_429 that cut the grey luminosity from 6.4e41 to 3.5e41 erg/s
(jobs 28361416 and 28366918, which differ only in the limit's opacity); the
grey pass's net emission against the snapshot field was -6.2e45 erg/s
instead of about +2e40. `false` reproduces the old grey pass. With Planck
scaling (grey snapshots) the first limit already uses the grey table and the
grey pass keeps it.

With `--volume-emission.cooling-limit-mode emission` a hot cell is not cooled:
it keeps its snapshot temperature, opacity and spectrum, and its emission is
scaled by `s = (cap + A) / G` (at most 1), where `G` is its gross emission and
`A = c kappa_P E V` its absorption of the snapshot radiation, both with the
pass's own Planck mean; its net emission `s G - A` is then the cap. This is how
the hydro's Fleck step treats such a cell, emitting from the real temperature
at a reduced rate, and it keeps the hard tail of hot cells that cooling would
remove (in snap_429 the temperature mode cools the hottest outside cell from
1.1e6 K to 2.8e5 K, cuts the outside gas's 0.3-1 keV gross emission by a
factor of five and removes its >1 keV emission). Cold cells are still heated
as above. `emission` is the default; `temperature` is the mode described
above, in which opacity, emission and multigroup spectrum all follow from the
limited temperature. Transport keeps Fleck factor 1 and real absorption in
both. The post-process never recomputes temperatures from internal energy,
and load-balance repartitions migrate the cells, so the limit is applied once
per pass (the grey pass finds the changed cells by ID).

The multigroup opacity scale factor alpha (`opacity-scaling.mode`) is a function
of each cell's temperature. It is computed at startup from the snapshot
temperatures (the surface and FLD use that), rebuilt once right after the
limit when any cell changed (log line `... scale after cooling limit`), and
rebuilt after every measured load-balance repartition as before.

`t_hydro` is `volume-emission.cooling-limit-dt` when positive, else the RICH
Courant step (`CourantFriedrichsLewy`, CFL `volume-emission.cooling-limit-cfl`,
default 0.25). The snapshot has no mesh velocities; each face gets the mean
velocity of its two cells, as on the Lagrangian mesh. The limited cells'
contribution scales as `1 / t_hydro`; the snapshot does not store the hydro
step, so pass the step from the hydro log when it is known.

The run prints one line,

    VOLUME_EMISSION_COOLING_LIMIT hydro_time=... source=courant cfl=0.25
      courant_cell_id=... courant_cell_r=... courant_cell_rho=... courant_cell_T=...
      outside_cells=... hot_limited=... cold_limited=...
      net_emission_before=... net_emission_after=... erg/s max_relative_dT=...

with the cell that sets the Courant step. All the cooling-limit options are in
the restart must-match list; a result written before
`volume-emission.cooling-limit-per-pass` existed can be continued only with
`--volume-emission.cooling-limit-per-pass false` (for a grey snapshot that is
the same run). `volume-emission.cooling-limit false` reproduces the behaviour
without any limit exactly.

## Test

`regression_tests/cases/imc_postprocess_cooling_limit`: single-cell caps to
1e-10; an equilibrium envelope is bit-identical with the limit on and off and
its `t_hydro` equals a direct `CourantFriedrichsLewy` call; with hot and cold
cells planted in a thin envelope exactly those cells are limited, the net
emission matches the formula, and the grey escaping luminosity returns to the
equilibrium value.
