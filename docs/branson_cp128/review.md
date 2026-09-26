# Branson Crooked Pipe comparison review

This is a port of STORM's cylindrical Crooked Pipe material geometry to Branson's
native Cartesian mesh. The upstream Branson checkout has no Crooked Pipe deck.
The user explicitly chose STORM Voronoi versus Branson's own mesh.

## Scope and isolation

- STORM's accepted executable and original Voronoi case are unchanged.
- Branson is built from a clean archive of commit recorded in `upstream_commit.txt`.
  The pre-existing instrumented `/home/maorm/branson` working tree is untouched.
- The archived-source patch is `branson_case.patch`. It sets strict CPU `-O2`,
  x86-64-v3, the 0.001 weight cutoff used by STORM, the inlet source mask, and
  five temperature probes. Branson's transport, source allocation, and MPI
  communication algorithms are unchanged.
- The source mask restricts the existing x-minus blackbody source to region 1
  (the thin material); all other x-minus source temperatures become zero.
  A SOURCE face absorbs exiting photons, matching STORM's default inlet.
- Probes use the same radius-0.1 cm disk in axial/radial coordinates and thin
  material volume weighting as STORM. Branson exposes its cell centers as float;
  probe selection therefore has this additional, small coordinate rounding.

## Matched settings

128 MPI ranks, eight exclusive nodes, 16 ranks/node, one CPU core/rank;
GCC 15.1, OpenMPI 4.1.6; O2, x86-64-v3, floating-point contraction disabled;
no fast math; gray IMC; stationary material; zero physical scattering;
absorption coefficients 2000/0.2 cm^-1; volumetric heat capacities
1e16/1e13 erg/(keV cm^3); material temperature 0.05 keV; inlet temperature
0.5 keV; vacuum outer faces; cutoff fraction 0.001; 120 time steps beginning
at 1e-11 s, growing by 1.1 to a 1e-9 s cap, ending at
8.157189571633594e-8 s. Both use full MPI invocation wall time with executables
and output on node-local storage. Staging/archival are excluded.

## Deliberate differences and interpretation limits

- STORM has 280606 refined Voronoi cells, including 114284 thin-channel cells.
  Branson has 112x50x50 = 280000 uniform Cartesian cells, including 37568
  thin-channel cells. Matching total cells does not match interface resolution.
- Branson's thin volume is 15.0272 cm^3 (analytic 14.9225651); its inlet area
  is 0.768 cm^2 (analytic 0.7853982). Its source temperature is not rescaled
  to compensate for its staircase boundary.
- STORM retains random walk, per-cell census population control, and dynamic
  mesh balancing. Branson uses history/SoA transport and static METIS partitioning.
  Its particle-passing driver does not call its combing function; the input
  explicitly disables the unused combing setting.
- Branson's nominal photon budget 1858414 is the rounded mean of STORM's
  measured generated histories per step (223009712/120 in accepted log1).
  Branson distributes this among initial radiation, thermal emission, and
  boundary source, with per-cell floors. STORM uses separate emission and
  boundary counts and per-cell caps. Actual generated and transported counts
  must be reported; the input budget alone does not establish equal work.
- Branson samples an initial equilibrium radiation field at 0.05 keV. STORM's
  existing case initializes cell Erad to that value, but its transport log starts
  with zero initial census particles. This is a remaining initialization
  difference, not repaired in the accepted STORM binary for this comparison.
- Conservation checks verify internal accounting; they do not establish equal
  discretization accuracy or equal statistical error between the two codes.

## Checks completed before submission

Compilation succeeded. Every Branson compile command was checked for O2 and
absence of O3/fast-math. XML checks verified all 12500 division mappings are
unique and complete, both volumetric heat capacities after unit conversion,
and exactly 120 steps using Branson's endpoint tolerance and time-step update.
No RDMA implementation or reallocation protocol was changed by this comparison.

## Network verification

Separate two-node MPI_Init/barrier/finalize probes, outside the timed allocations,
selected the UCX PML at priority 51 and inter-node `rc_mlx5/mlx5_0:1` with the
same `OMPI_MCA_btl=^openib` setting. Logs are `build/branson_cp128/mpi_probe.err`
and `ucx_probe.out`. The first diagnostic attempted a direct srun MPI launch,
which this OpenMPI installation does not support; the successful checks used
mpirun inside a Slurm allocation, as the actual benchmark jobs do.

## Review correction

The initial review missed the enabled synthetic intensive-scatter workload. See
`discrepancy.md` for the correction and diagnostic runs. It executes on effective
IMC scatter events even though physical scattering opacity is zero.
