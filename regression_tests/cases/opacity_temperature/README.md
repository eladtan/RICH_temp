This serial RICH regression checks explicit-temperature opacity evaluation:

- Power-law and free-free laws use the supplied temperature.
- Synthetic log-linear STA tables exercise the real readers and interpolation.
- The STORM adapter forwards gray and frequency-dependent requests correctly.
- Temperature-aware analytic callbacks receive the original cell by reference.
- Legacy callbacks retain ordinary evaluation but reject a different temperature.
- Evaluation preserves the cell's temperature, density, tracers and group storage.

Run through the existing THUNDER regression framework from the RICH root:

```bash
./regression_tests/run_all.sh --mode serial --test opacity_temperature
```

The executable prints `OPACITY_TEMPERATURE_PASS` only after all checks pass.
This case does not use CTest or STORM's removed standalone test directory.
