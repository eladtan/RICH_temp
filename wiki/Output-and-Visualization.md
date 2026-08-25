# Output and Visualization

## HDF5 Snapshots

RICH writes simulation state to HDF5 files using `WriteSnapshot3D()`. Each snapshot contains the full mesh and field data needed to visualize or restart the simulation.

### Writing Snapshots

```cpp
#include "source/3D/output/write3D.hpp"

// Write a snapshot
WriteSnapshot3D(sim, "snap_100.h5");
```

### HDF5 File Structure

**Serial snapshots** write all data to the root group `/`.

**MPI snapshots** organize data by rank: `/rank0/`, `/rank1/`, etc., with global metadata at the root.

| Dataset | Type | Description |
|---------|------|-------------|
| `X`, `Y`, `Z` | double[] | Voronoi generator (mesh point) coordinates |
| `CMx`, `CMy`, `CMz` | double[] | Cell center-of-mass coordinates |
| `Density` | double[] | Mass density |
| `Pressure` | double[] | Pressure |
| `InternalEnergy` | double[] | Specific internal energy |
| `Temperature` | double[] | Temperature |
| `Vx`, `Vy`, `Vz` | double[] | Velocity components |
| `Erad` | double[] | Radiation energy density per mass |
| `Eg_0`, `Eg_1`, ... | double[] | Per-group radiation energy (multigroup) |
| `Volume` | double[] | Cell volume |
| `ID` | size_t[] | Cell ID |
| `Box` | double[6] | Domain bounds [ll.x, ll.y, ll.z, ur.x, ur.y, ur.z] |
| `Time` | double | Simulation time |
| `Cycle` | int | Simulation cycle number |
| `/tracers/<name>` | double[] | Passive tracer fields |
| `/stickers/<name>` | bool[] | Boolean marker fields |

### Reading Snapshots in Python

```python
import h5py
import numpy as np

with h5py.File("sedov_final.h5", "r") as f:
    x = f["X"][:]
    y = f["Y"][:]
    z = f["Z"][:]
    density = f["Density"][:]
    pressure = f["Pressure"][:]
    time = f["Time"][()]
    print(f"Time: {time}, Cells: {len(x)}")
```

For MPI snapshots:

```python
import h5py
import numpy as np

with h5py.File("sedov_final.h5", "r") as f:
    all_density = []
    all_x = []
    rank = 0
    while f"rank{rank}" in f:
        grp = f[f"rank{rank}"]
        all_x.append(grp["X"][:])
        all_density.append(grp["Density"][:])
        rank += 1
    x = np.concatenate(all_x)
    density = np.concatenate(all_density)
```

### Reading Snapshots in C++

```cpp
#include "source/3D/output/Snapshot3D.hpp"

Snapshot3D snap = ReadSnapshot3D("sedov_final.h5");
// snap.mesh_points -- vector of Vector3D
// snap.cells       -- vector of ComputationalCell3D
// snap.volumes     -- vector of double
// snap.time        -- simulation time
// snap.cycle       -- cycle number
```

### Diagnostic Appendices

Custom diagnostic fields can be added to snapshots by implementing `DiagnosticAppendix3D`:

```cpp
class MyDiagnostic : public DiagnosticAppendix3D {
public:
    std::string getName() const override { return "MyField"; }
    std::vector<double> operator()(const HDSim3D& sim) const override {
        // compute your diagnostic field
    }
};
```

Pass appendices to `WriteSnapshot3D`:

```cpp
std::vector<DiagnosticAppendix3D*> appendices = {new MyDiagnostic()};
WriteSnapshot3D(sim, "snap.h5", appendices, true);
```

Diagnostic appendices can store tensor gradients, velocity divergence, and numerical dissipation. See [Simulation Setup](Simulation-Setup) for implementation details.

### Tracers and Stickers in Snapshots

Tracer fields (per-cell scalar values) are written under the `/tracers/` group. Common uses:

| Tracer | Purpose |
|--------|---------|
| `Entropy` | Thermodynamic entropy |
| `Star` | Material tagging (1 = stellar material, 0 = ambient) |
| `WasRemoved` | Mass removed by center draining |

Tracer names are set at simulation start:

```cpp
ComputationalCell3D::tracerNames.push_back("Entropy");
ComputationalCell3D::tracerNames.push_back("Star");
ComputationalCell3D::tracerNames.push_back("WasRemoved");
```

### Reading Multigroup Radiation Data in Python

```python
import h5py
import numpy as np

with h5py.File("snap.h5", "r") as f:
    # Total radiation energy per mass
    Erad = f["Erad"][:]

    # Per-group radiation energy
    n_groups = 0
    while f"Eg_{n_groups}" in f:
        n_groups += 1
    Eg = np.array([f[f"Eg_{g}"][:] for g in range(n_groups)])
    # Eg shape: (n_groups, n_cells)
```

## VTK Output

If VTK is available, `WriteSnapshot3D` can also produce VTU files for visualization in ParaView or VisIt:

```cpp
WriteSnapshot3D(sim, "snap.h5", {}, true, true);  // last arg: write_vtu=true
```

This produces an unstructured grid file that can be opened directly in ParaView.

## Visualization Scripts

RICH includes visualization helpers under `visualisation/`:

### 2D Visualization (Python)

| Script | Description |
|--------|-------------|
| `visualisation/two_dimensional/python/hdf5_voronoi_plot.py` | Voronoi cell plot from HDF5 |
| `visualisation/two_dimensional/python/plot_unstructured.py` | Delaunay-based contour plots |
| `visualisation/two_dimensional/python/multiplot.py` | Multi-file snapshot plotting |
| `visualisation/two_dimensional/python/patch_plot.py` | PolyCollection Voronoi display |

### 1D Visualization (Python)

| Script | Description |
|--------|-------------|
| `visualisation/one_dimensional/python/plotit.py` | Simple 1D profile plotting |

### 3D Visualization (MATLAB)

| Script | Description |
|--------|-------------|
| `visualisation/three_dimensional/matlab/readHDF5Data.m` | 3D HDF5 reader for MATLAB |

### 2D Visualization (MATLAB)

| Script | Description |
|--------|-------------|
| `visualisation/two_dimensional/matlab/read_hdf.m` | Read serial HDF5 |
| `visualisation/two_dimensional/matlab/read_hdfMPI.m` | Read MPI HDF5 |

## Regression Test Plots

The regression framework includes a dedicated plotting tool:

```bash
python3 regression_tests/plot_results.py
```

This generates comparison plots of simulation profiles against analytical solutions. Plots are saved to `regression_tests/plots/`.

Options:

```bash
# Plot all tests with available data
python3 regression_tests/plot_results.py --all

# Custom output directory
python3 regression_tests/plot_results.py --output-dir /tmp/plots
```

## Text Output

Some simulations write lightweight text profiles instead of HDF5 snapshots (common in regression tests for speed):

```cpp
// Example: write a 1D profile
std::ofstream f("profile.txt");
for (size_t i = 0; i < cells.size(); ++i)
    f << x[i] << " " << cells[i].density << " " << cells[i].pressure << "\n";
```

These text files are used by the regression checkers for validation against analytical solutions.

## ParaView Workflow

For 3D visualization with VTK output:

1. Build with VTK support (VTK is a required dependency).
2. Enable VTU output in `WriteSnapshot3D`.
3. Open the `.vtu` files in ParaView.
4. Use ParaView's "Clip", "Slice", and "Threshold" filters to explore 3D data.
5. Color cells by any field (Density, Pressure, Temperature, etc.).
