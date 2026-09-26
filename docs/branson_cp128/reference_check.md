# Reference agreement check

**Scope correction:** this check used the small performance configuration, not
the validated configuration in `IMC_paper`. The paper run does agree closely
with the reference. The large delay also exists in frozen pre-optimization runs.
See [the configuration and regression investigation](../cp_paper_regression/findings.md).

McClarren–Urbatsch is the separate cylindrical hohlraum benchmark in section IV.2
of https://arxiv.org/html/2108.13453v2 . Its Figure 5 line-out is at r=0.05 cm,
t=10 ns. The saved standalone STORM profile at
`source/monte/examples/hohlraum_parallel/hohlraum_profile.txt` is at t=3.00064 ns.
The current driver stops at about 3 ns (`main.cpp:510`), and its regression
checker (`source/monte/regression_tests/lib/regression_checks.sh:209`) checks only
fatal log markers. These artifacts do not demonstrate agreement with the 10-ns
reference. No mismatched-time RMS was computed or interpreted as validation.

The timed Crooked Pipe case has a different reference: Gentile (2001), digitized
from Steinberg–Heizler Figure 8(a), in `fig8_gentile.csv`. This configuration
fails the supplied reference check; it must not be treated as accuracy validated.
At 81.5719 ns, using the mean of the three STORM runs in jobs 10176483/10176484,
and interpolation linear in log(time) of the digitized reference:

| Probe | STORM (keV) | Gentile reference (keV) |
|---|---:|---:|
| 1 | 0.455939 | 0.469932 |
| 2 | 0.283745 | 0.319526 |
| 3 | 0.124691 | 0.218670 |
| 4 | 0.056345 | 0.176134 |
| 5 | 0.046467 | 0.080618 |

Probe 3 reaches 0.08 keV at 38.90–39.15 ns in STORM versus 11.09 ns in the
digitized reference. Probe 4 does not reach that threshold by the end of the
STORM segment; the reference reaches it at 27.88 ns. Probe 5 has insufficient
post-heating overlap in this truncated segment, so the automated insufficient-
overlap result must not be presented as a full-run physical failure.

The references are plot digitizations, not exact solution tables. The downstream
differences are nevertheless substantially larger than the approximate 0.005–
0.01 keV digitization uncertainty documented in the Crooked Pipe README.
