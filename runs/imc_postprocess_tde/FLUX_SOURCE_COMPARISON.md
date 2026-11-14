# Grey-RHD flux-source comparison

This optional `imc_postprocess_tde` mode compares grey and multigroup
post-processing while holding the bolometric source fixed.  The source is the
grey FLD flux reconstructed from the imported RHD snapshot, not an independent
`c * kappa_P * a * T^4` volume source in every cell.

The workflow is:

1. Launch inward rays from the observer sphere.
2. Locate the irregular grey thermalization surface at
   `tau_eff = --flux-source-tau`, using
   `sqrt(3*kappa_a*(kappa_a+kappa_s))`.
3. Leave angular directions open when the requested optical depth is never
   reached.  No neighboring-ray radius is invented for an optically thin hole.
   With `--flux-source.smoothing-floor f` (default 0.5) a resolved direction
   whose radius is below `f` times the median radius of its 8 nearest
   resolved directions is raised to that value; the run prints a
   `FLUX_SOURCE_SMOOTHING WARNING` whenever this moves the surface, since the
   raised surface is shallower than the target `tau_eff`.
   Each ray also records every channel's photosphere along the same ray
   (total depth `kappa_a+kappa_s` reaching 2/3, as in the observer
   photosphere; `FLUX_SOURCE_CHANNEL ... photosphere_p05/med/p95`). With
   `--flux-source.smoothing-photosphere-cap true` (default) smoothing never
   raises a direction above its own deepest photosphere, and does not raise
   a direction that has none.
   A cell takes the radius of its nearest construction ray. With
   `--flux-source.mapping-neighbours k` (default 6) each ray's radius is first
   lowered to the smallest among itself and its `k` nearest resolved rays, so
   a cell between rays is inside only if it is below all of them and never
   inherits a neighbour's larger radius (`FLUX_SOURCE_MAPPING`); 0 restores
   the nearest-ray-only mapping. The mapping never lowers a ray below the
   radius where its own `tau_eff` reaches
   `--flux-source.mapping-max-depth-factor` (default 3) times the target:
   on a jagged, opaque boundary the deepest neighbour can lie a whole cell
   inside, and exposing such hot, thermalized cells costs millions of
   packets (or packets far heavier than the escaping light) and buys nothing.
4. Reconstruct the snapshot grey FLD flux from `Erad`.  With
   `--flux-source.face-flux hydro` (default) each CER face carries the
   hydro's own FLD flux across that face (`Diffusion.cpp`): the two-point
   gradient between the cells' centres of mass, both cells' diffusion
   coefficients at the face temperature `((T_1^4 + T_2^4)/2)^(1/4)`, their
   harmonic mean, and the flux limiter on the face gradient.
   `cell-average` keeps the original formula, the mean of the two cells'
   cell-centred fluxes; in snap_109 it gives 0.85 of the hydro's positive
   face luminosity (3.86e42 against 4.56e42 erg/s; net 1.41e42 against
   1.61e42).
   `hydro-multigroup` uses the hydro's multigroup FLD flux instead
   (`MultigroupDiffusion.cpp`): per group, the snapshot's group energy
   `rho E_g`, both cells' group diffusion coefficients `c / (3 sigma_Ross,g)`
   at the face temperature (no Compton term), their harmonic mean and the
   flux limiter. Each group's outward flux is emitted in that group, so the
   face luminosity is the sum of the outward group fluxes and the spectrum is
   the hydro's (inward group fluxes are reported, not emitted). Within a group
   the frequency is uniform in log energy. The grey pass emits the same face
   luminosities without the spectrum. The per-face group fluxes are computed
   once, at the first setup, and kept for the whole surface. Packets returning
   through the CER are still re-emitted with the Planck spectrum below.
   With every mode, the first multigroup setup prints `FLUX_SOURCE_GROUPS`:
   the face luminosity split by a Planck spectrum at the spectrum temperature
   and by the outer cells' snapshot group energies, and the hydro's multigroup
   face flux per group (net and outward). `SOURCE_GROUP_VOLUME` prints the
   volume emission of the outside cells per group, and their absorption of
   the snapshot's group radiation, before and after the cooling limit.
5. Identify every Voronoi face separating the thermalized interior from the
   transported exterior.  Positive outward FLD flux supplies source
   luminosity; inward flux is reported separately rather than hidden.
   Outside cells that do not connect, through other outside cells, to the
   exterior beyond the outermost surface radius are made inside
   (`--flux-source.remove-enclosed-outside`, default true).  Such a region is
   walled in by thermalizing faces on every side, so a packet born or
   arriving there is returned into it at every wall and can only time out;
   in snap_109 these were single hot (1e7-1e8 K) cells in narrow polar cones
   where the keV group's surface dipped to ~2e11 cm.  The run prints
   `FLUX_SOURCE_ENCLOSED removed_cells=... wall_luminosity_removed=...`.
   The surface as built (per-direction and per-group radii, smoothed
   directions, removed cells) is written to `<stem>.source-surface.h5` at
   setup.
   The first setup checks every boundary face against the photosphere of the
   construction ray its outer cell is mapped to and prints
   `FLUX_SOURCE_PHOTOSPHERE_CHECK`: for the deepest photosphere, the grey one
   and each group's, the number of faces (and their share of the face
   luminosity) lying above it, and separately the faces on rays where that
   channel never reaches total depth 2/3 (transparent).
   The per-direction surface, the per-ray photospheres and this check are
   also written to the result file under `/comparison/source_surface/`, and
   the surface along each observer direction under
   `/photosphere/emission_surface/` (VTK `photosphere_emission_surface_*`), next to the
   observer photosphere and the `tau_eff = 1` diagnostic radius.
6. Treat the complete face set as an internal, thermalizing, diffuse boundary.
   A packet that returns through the CER is re-emitted into the transported
   region with conserved comoving energy, a Planck spectrum, and zero Stokes
   Q/U.
7. Run the existing multigroup and grey postprocessors from the same target
   face luminosity.

The multigroup CER spectrum is Planck weighted, at the radiation temperature
`(E_rad/a)^(1/4)` of the face's outer cell
(`--flux-source.spectrum-temperature radiation`, default) or at its gas
temperature (`gas`, the original behaviour).  The face carries the flux of the
thermalized radiation field; in snap_109 a quarter of the face luminosity sat
next to hot (1e7-1e8 K), optically thin gas under T_rad ~1e5 K, and painting
the gas temperature on those packets injected keV photons that escaped
directly.  The run prints `FLUX_SOURCE_SPECTRUM` with the face-luminosity
share of such faces and the weighted radiation and gas temperatures.  It deliberately does not use
the ordinary IMC volume-emission distribution `kappa_nu B_nu`.  Absorption and
LTE re-emission in the exterior are represented as conservative IMC effective
scattering (`Fleck=0`); those exterior re-emission events retain the normal
`kappa_nu B_nu` thermal sampling.

The current adaptive source-cell, observer-equity, adaptive group-frequency,
measured-load-balance, and polarization machinery is retained.  Adaptive
sampling changes packet counts and proposal PDFs only; it does not alter the
physical face luminosities.  Even in the learned-only phases, every positive
source cell retains a one-packet floor so no part of the fixed source is
dropped.  Source topology and luminosity are checked after every MPI
repartition.

Random-walk acceleration remains disabled in this mode because it does not
expose individual CER-face encounters.  DDMC is supported through a native
one-sided thermalizing CER boundary.  A CER-adjacent exterior cell remains
DDMC-eligible only when its diffusion optical depth from the cell generator to
every attached CER face is at least
`--flux-source-ddmc-face-tau` (default 5).  Cells failing that local face test
fall back to explicit IMC; no fixed multi-cell buffer is imposed.

A DDMC packet that reaches the CER preserves its comoving energy, loses its
polarization memory, and is redistributed with the same Planck boundary law as
the explicit calculation.  In the MG run, re-emission inside the local PGRW
DDMC band stays DDMC, while re-emission above the local cutoff becomes an
outward Lambertian IMC packet on the same CER face.  Passing `--no-ddmc` still
provides a fully explicit reference calculation.  Ordinary
`imc_postprocess_tde` runs are unchanged.

## Settings that follow the snapshot

The snapshot's radiation decides the opacity treatment and the face flux
(`SnapshotRadiation.cpp`; the run prints `SNAPSHOT_RADIATION` after reading
the snapshot). `--opacity-scaling.mode` and `--flux-source.face-flux` default
to `auto`:

| snapshot | `auto` scaling | `auto` face flux | explicit settings |
|---|---|---|---|
| multigroup (`Eg_k` for every table group) | `none` | `hydro-multigroup` | scaling must be `none`; any face flux |
| grey (`Erad` only) | `planck` | `hydro` | `none` and `hydro-multigroup` refused |
| other group count | `planck` | `hydro` | scaling free; `hydro-multigroup` refused |
| no radiation | `planck` | `hydro` | free |

Scaling the group opacities to the grey Planck mean is right only for a grey
hydro run: a multigroup run used its group opacities unscaled, and scaling
them (snap_429: alpha 527 on average) closes the transparent windows. A
snapshot without radiation gets LTE radiation first:
`--input.missing-radiation conserve-energy` (the default) shares each cell's
internal energy between gas and radiation at one temperature,
`rho e = rho e_gas(T) + a T^4`; `gas-temperature` sets `T_rad = T_gas`. The
snapshot does not store its group edges, so only the group count is checked.

## Build

```bash
./build_rich.sh gnuReleaseMPI \
  --test_name=imc_postprocess_tde \
  --montecarlo-polarization
```

## Example

```bash
mpirun -np 64 ./rich \
  --input SNAPSHOT.h5 \
  --output flux_compare.h5 \
  --opacity-dir /path/to/STA/MG/ \
  --grey-opacity-dir /path/to/STA/ \
  --eos-dir /path/to/EOS/ \
  --radius 5e14 \
  --n-observers 256 \
  --source-dt 1 \
  --transport-time 2e6 \
  --n-generations 500 \
  --photons-per-cell 100 \
  --flux-source-compare \
  --flux-source-tau 5 \
  --flux-source-ddmc-face-tau 5 \
  --polarization \
  --adaptive-source-cells \
  --adaptive-group-quality \
  --adaptive-group-source-cells \
  --adaptive-group-frequency-sampling
```

The mode requires `ENERGY_GROUPS_NUM > 1`, a positive snapshot radiation-energy
field, and non-Compton transport.

Outputs are the normal MG HDF5/VTK products, the normal grey VTK product, and
`*_flux_compare.tsv`.  The TSV records the common target source luminosity,
actual emitted and observer-crossing luminosities, Monte Carlo generation
error, target-normalized ratios, timed-out fraction, surface completeness,
positive/net/inward FLD diagnostics, and a luminosity-weighted polarization
degree for both methods.
