# E3D paper reference data

`e3d_brantley_reference.csv` holds the Brantley--Novellino Figure 2 curves that
`analyze_e3d.py` overlays on the RICH ensemble.  It is generated, not
hand-entered, by `extract_paper_figure2.py`:

```bash
pip install pymupdf
python3 extract_paper_figure2.py ../E3D_paper.pdf -o e3d_brantley_reference.csv
```

Figure 2 is a vector (matplotlib) figure, so the plotted curves survive in the
PDF as polylines whose vertices are the data points.  The script reads those
polylines, calibrates page coordinates against the panels' major tick marks, and
emits the long-format CSV.  Every number therefore traces back to a vertex in
the PDF content stream rather than to a pixel read off an image.

Five curves are extracted per panel: `Benchmark`, `Atomic Mix`, `CLS LP`,
`CLS LRP`, and `LRP Ramp+Exp`.  Use `--reference-model Benchmark` (the default)
for the RICH-vs-paper metrics; the approximate models are useful context because
they bracket the benchmark.

## Accuracy

Two independent checks are run automatically and the script fails if either
breaks:

1. The relative errors of the approximate models at the final time, quoted in
   Section 3 of the paper, are reproduced to better than one percentage point.
2. Transmission plus reflection sums to unity to within `1e-4`.  This is a
   strong check because the two panels are calibrated separately from their own
   tick marks and use different axis limits (0.30 versus 1.0) and different
   time ranges.

The one soft spot is `CLS LRP` reflection, which digitizes to -3.9% against the
-3% quoted in the text; the other seven quoted percentages agree to within
0.5 points, so this is most likely rounding in the paper.

## Caveats

* Panel (a) plots transmission and panel (b) plots reflection on *different*
  time grids.  Each curve is resampled onto the union of its own two grids by
  linear interpolation in `log(time)`, which is exactly the interpolation the
  figure performs between vertices.
* The curves are clipped to the plotted time window, `1e-11 s` to `5e-9 s`.
  Vertices that matplotlib emitted outside the axes are drawing artefacts of its
  path clipping and are replaced by the interpolated value at the axis limit.
* No uncertainty columns are written.  The paper's `+/- 2 sigma` band is plotted
  but the relative standard deviation of the ensemble mean is 0.03% at the final
  time, which is far narrower than the line width, so digitizing it would record
  stroke geometry rather than data.

## Schema

The accepted long-format schema is:

```text
model,time_s,transmission,reflection,transmission_error,reflection_error
Benchmark,1.0e-12,0.0,0.0,,
```

`model` and the two error columns are optional.  If `model` is omitted, rows are
treated as the `Benchmark` curve.  Error columns may contain standard deviations
or plotted error bars; the analysis script uses them only for the reported
normalized residual, so record their interpretation in the file's provenance or
accompanying notes.
