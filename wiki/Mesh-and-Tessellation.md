# Mesh and Tessellation

RICH uses a 3D Voronoi tessellation as its computational mesh. The mesh can move with the fluid (Lagrangian), stay fixed (Eulerian), or use a hybrid approach (RoundCells).

## Voronoi Tessellation

### Overview

A Voronoi tessellation partitions space into cells, where each cell contains all points closer to its generator (mesh point) than to any other generator. This produces a natural, unstructured mesh that adapts to the point distribution.

RICH constructs the Voronoi tessellation via the dual Delaunay triangulation:

1. Compute the 3D Delaunay triangulation of the mesh points
2. Derive the Voronoi diagram as its geometric dual

### Voronoi3D

The main tessellation class:

```cpp
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"

Vector3D ll(-1, -1, -1), ur(1, 1, 1);  // domain box
Voronoi3D tess(ll, ur);

// Serial build
tess.Build(points);

// Parallel build (MPI)
tess.BuildParallel(points);
```

### Tessellation3D Interface

All tessellation operations go through the abstract `Tessellation3D` interface:

| Method | Description |
|--------|-------------|
| `GetPointNo()` | Number of local mesh points |
| `GetMeshPoint(i)` | Position of mesh point i |
| `GetVolume(i)` | Volume of cell i |
| `GetCellCM(i)` | Center of mass of cell i |
| `GetFaceNeighbors(face)` | Pair of cells sharing a face |
| `GetTotalFacesNumber()` | Number of faces |
| `GetArea(face)` | Area of a face |

## Delaunay Triangulation

The 3D Delaunay triangulation (`Delaunay3D`) underlies the Voronoi construction. It uses exact geometric predicates (`Predicates3D`) for robustness.

```
source/3D/tessellation/delaunay/Delaunay3D.hpp
source/3D/tessellation/utils/Predicates3D.hpp
```

## Point Motion Strategies

The mesh points can move according to different strategies, set by the `PointMotion3D` interface:

### Lagrangian3D

Points move with the local fluid velocity. This minimizes advection errors and naturally concentrates resolution in compressed regions.

```cpp
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
Lagrangian3D pm;
```

Advantages: minimal numerical diffusion, natural shock tracking. Disadvantage: cells can become distorted.

### Eulerian3D

Points stay fixed (static mesh). The flow advects through the mesh.

```cpp
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
Eulerian3D pm;
```

Advantages: no mesh tangling. Disadvantage: more numerical diffusion, no resolution adaptation.

### RoundCells3D

Hybrid approach: Lagrangian motion plus a correction that pushes mesh points toward their cell centroids, producing rounder cells.

```cpp
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"

Lagrangian3D base_motion;
RoundCells3D pm(base_motion, eos);
```

This is the recommended approach for most simulations. It provides:
- Low advection errors (nearly Lagrangian)
- Good cell quality (avoids degeneracy)
- Automatic resolution adaptation

## Mesh Generation

RICH provides several point generation functions:

| Function | Header | Description |
|----------|--------|-------------|
| `RandRectangular(N, ll, ur)` | `mesh_generator3D.hpp` | Random points in a box |
| `RandSphereR(N, ll, ur, Rmin, Rmax, center)` | `mesh_generator3D.hpp` | Random in spherical shell |
| `RandSphereR2(N, ll, ur, Rmin, Rmax, center)` | `mesh_generator3D.hpp` | r^2-biased spherical distribution |
| `linspace(xmin, xmax, N)` | `mesh_generator.hpp` | Uniform 1D grid |

### Mesh Smoothing: RoundGrid3D

After generating initial points, smooth them with Lloyd iteration:

```cpp
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"

// 10 iterations of Lloyd relaxation
points = RoundGrid3D(points, ll, ur, 10);
```

This produces a more uniform Voronoi mesh. More iterations yield rounder cells but take longer.

## MPI Mesh Distribution

In parallel builds, the mesh is distributed across MPI ranks:

1. **Generate points on rank 0**: Only rank 0 creates the initial point set
2. **Spread points**: `MPI_Spread()` distributes points to all ranks
3. **Build parallel**: `tess.BuildParallel(points)` handles domain decomposition

```cpp
std::vector<Vector3D> points;
if (rank == 0)
    points = RandRectangular(Np, ll, ur);

#ifdef RICH_MPI
points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
tess.BuildParallel(points);
#else
tess.Build(points);
#endif
```

## Load Balancing

RICH supports two load balancing strategies for MPI:

### Hilbert Curve

The default approach. Points are ordered along a space-filling Hilbert curve and divided into equal-sized chunks per rank.

```
source/3D/hilbert/
source/3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp
```

### ParMETIS

Graph-based partitioning using the ParMETIS library (optional):

```
source/3D/tessellation/loadBalancing/ParMetisLoadBalancer.hpp
```

ParMETIS can produce better partitions for irregular geometries but adds a library dependency.

## Spatial Queries

RICH includes octree and KD-tree data structures for efficient spatial queries (nearest neighbor, range search):

```
source/3D/range/OctTree3D.hpp
source/3D/range/KDTree.hpp
source/ds/OctTree.hpp
source/ds/DistributedOctTree.hpp
```

These are used internally by the gravity solver and other components that need spatial lookups.
