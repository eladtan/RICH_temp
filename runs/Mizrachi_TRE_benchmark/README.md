# TRE-1 with RICH/STORM

This directory implements the one-dimensional Thermal Radiative Eigenmode
(TRE-1) proposal in `TRE_Benchmark_Proposal.pdf` using STORM's gray IMC
engine.  The domain is a periodic x interval `[0, 2*pi]`, extruded through a
periodic unit y/z cross-section.  The physical scale is `L0=1 cm`, with
`sigma_a=1/cm`, `beta=1`, `U0=1`, and material perturbation `A=0.1`.

The initial packet distribution is sampled from the exact positive angular
eigenfunction.  After initialization there is no external source: STORM
handles the thermal emission, transport, absorption, and material update.
`analyze_tre_1d.py` compares the material mode at timestep end and STORM's
track-length radiation mode at timestep midpoint with the semi-analytic
dispersion relation.

## Build and run locally

```bash
cmake -S runs/Mizrachi_TRE_benchmark \
  -B runs/Mizrachi_TRE_benchmark/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTORM_MADCART_DIR="$PWD/source/3D/tessellation/cartesian" \
  -DSTORM_MADVORO_DIR="$PWD/source/3D/tessellation/voronoi" \
  -DSTORM_MESH_DECOMPOSER_DIR="$PWD/source/3D/tessellation/MeshDecomposer3D" \
  -DSTORM_WITH_COMPTON=OFF
cmake --build runs/Mizrachi_TRE_benchmark/build --target tre_1d --parallel 4
runs/Mizrachi_TRE_benchmark/build/tre_1d \
  --nx 96 --initial-particles 300 --new-photons 20 \
  --population 350 --dtau 0.02 --final-tau 8 \
  --output runs/Mizrachi_TRE_benchmark/tre_1d_history.csv
python3 runs/Mizrachi_TRE_benchmark/analyze_tre_1d.py \
  --input runs/Mizrachi_TRE_benchmark/tre_1d_history.csv
```

The checked-in `submit.sh` uses the `bigrun` partition.  STORM's serial
manager is intentionally used for this 1D validation run, so one compute core
is sufficient; packet count, timestep, and mesh resolution control the
accuracy/cost tradeoff.
