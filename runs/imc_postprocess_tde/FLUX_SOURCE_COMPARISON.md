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
4. Reconstruct the snapshot grey FLD flux from `Erad`.
5. Identify every Voronoi face separating the thermalized interior from the
   transported exterior.  Positive outward FLD flux supplies source
   luminosity; inward flux is reported separately rather than hidden.
6. Treat the complete face set as an internal, thermalizing, diffuse boundary.
   A packet that returns through the CER is re-emitted into the transported
   region with conserved comoving energy, a Planck spectrum, and zero Stokes
   Q/U.
7. Run the existing multigroup and grey postprocessors from the same target
   face luminosity.

The multigroup CER spectrum is Planck weighted.  It deliberately does not use
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

Random-walk and DDMC acceleration are intentionally disabled in this mode.
Their current closures do not enforce the new internal CER boundary, and using
them would make the grey/MG comparison physically inconsistent.  Ordinary
`imc_postprocess_tde` runs are unchanged.

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
