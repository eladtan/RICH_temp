# MPI Parallelism

RICH supports distributed-memory parallelism via MPI. This document describes the parallel architecture, domain decomposition, communication patterns, and load balancing.

## Overview

When built with an MPI configuration (any config name containing `MPI`), RICH defines the `RICH_MPI` preprocessor macro. All MPI-specific code is guarded by `#ifdef RICH_MPI` blocks, allowing the same source code to compile in both serial and parallel modes.

## Domain Decomposition

RICH uses spatial domain decomposition: each MPI rank owns a subset of the mesh points and their associated cells. The decomposition is performed during the Voronoi tessellation build:

```cpp
Voronoi3D tess(ll, ur);
tess.BuildParallel(points);  // distributes points across ranks
```

### Initial Point Distribution

Points are generated on rank 0 and distributed:

```cpp
std::vector<Vector3D> points;
if (rank == 0)
    points = RandRectangular(Np, ll, ur);

points = MPI_Spread(points, 0, MPI_COMM_WORLD);
```

### Ghost Cells

At process boundaries, ghost cells from neighboring ranks are maintained. The parallel tessellation build automatically:

1. Determines which cells are near rank boundaries
2. Exchanges ghost point positions between neighboring ranks
3. Includes ghost cells in the local Voronoi tessellation
4. Ghost cells participate in gradient estimation and flux computation

## Load Balancing

Good load balance is critical for parallel efficiency. RICH provides two strategies:

### Hilbert Curve (Default)

Points are ordered along a 3D Hilbert space-filling curve, which maps the 3D domain to a 1D sequence while preserving spatial locality. The sequence is then divided into equal-sized segments, one per rank.

```
source/3D/hilbert/HilbertOrder.hpp
source/3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp
source/3D/tessellation/voronoi/HilbertPointsManager.hpp
```

Advantages:
- Fast to compute
- Good spatial locality (ranks own contiguous regions)
- No external library dependency

### ParMETIS (Optional)

Graph-based partitioning using the ParMETIS library. Constructs a graph from the cell adjacency structure and partitions it to minimize communication while balancing load.

```
source/3D/tessellation/loadBalancing/ParMetisLoadBalancer.hpp
```

Advantages:
- Can produce better partitions for irregular geometries
- Minimizes surface area between ranks (reduces communication)

Disadvantages:
- Requires ParMETIS, METIS, and GKlib libraries
- More expensive to compute than Hilbert

## Communication Patterns

### ExchangeChain

RICH uses a ring exchange pattern (`ExchangeChain`) for point-to-point communication between neighboring ranks:

```
source/mpi/ExchangeChain.hpp
```

In a ring exchange, each rank sends data to rank+1 and receives from rank-1 (modulo the number of ranks), cycling through all neighbors. This avoids deadlocks and scales well.

### MPI Commands

Common MPI operations are wrapped in `mpi_commands.hpp`:

```
source/mpi/mpi_commands.hpp
```

Key operations:
- `MPI_Spread`: Distribute data from one rank to all
- Broadcast, gather, scatter wrappers
- Custom datatypes for `Vector3D`, `ComputationalCell3D`, etc.

### Serialization

Complex data structures are serialized for MPI communication:

```
source/mpi/serialize/
```

This allows `ComputationalCell3D`, `Conserved3D`, and other composite types to be efficiently packed and unpacked for MPI send/receive.

## Parallel I/O

### HDF5 Snapshots

In MPI mode, `WriteSnapshot3D` writes data in a structured format:

- Global metadata (`Box`, `Time`, `Cycle`) at the root
- Per-rank data in groups `/rank0/`, `/rank1/`, ...

Writing is sequential (rank-by-rank) to avoid HDF5 contention:

```
rank 0 creates file, writes global data and rank 0 data
rank 1 writes rank 1 data
...
rank N-1 writes rank N-1 data
```

### Reading MPI Snapshots

When reading an MPI snapshot for restart, if the current MPI size differs from the snapshot's rank count, data from extra ranks is merged into available ranks.

## Parallel Gravity

The distributed gravity tree (`DistributedGravityTree`) extends the serial tree for MPI:

```
source/3D/gravity/DistributedGravityTree.hpp
```

Each rank builds a local tree, then trees are exchanged and merged to compute gravitational forces including contributions from remote cells.

## Parallel Tessellation

The parallel Voronoi build involves:

1. **Local Delaunay**: Each rank builds a Delaunay triangulation of its local points
2. **Boundary detection**: Identify cells near rank boundaries
3. **Ghost exchange**: Send boundary cell positions to neighboring ranks
4. **Extended Delaunay**: Rebuild including ghost points
5. **Voronoi extraction**: Compute Voronoi cells from the extended Delaunay

This is repeated during mesh motion when point positions change.

## Writing MPI-Compatible Simulations

The key pattern for MPI compatibility:

```cpp
int rank = 0;
#ifdef RICH_MPI
MPI_Init(NULL, NULL);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

// Generate points on rank 0 only
std::vector<Vector3D> points;
if (rank == 0)
    points = RandRectangular(Np, ll, ur);

#ifdef RICH_MPI
points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

// Build mesh
Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
tess.BuildParallel(points);
#else
tess.Build(points);
#endif

// ... simulation setup and loop ...

#ifdef RICH_MPI
MPI_Finalize();
#endif
```

## Scaling Considerations

- **Strong scaling**: Fixed problem size, increasing ranks. Communication overhead limits speedup.
- **Weak scaling**: Problem size grows with ranks. RICH scales well in weak scaling for large problems.
- **Load imbalance**: AMR and non-uniform point distributions can cause imbalance. Re-balance periodically.
- **Communication**: Ghost exchange and gravity tree communication are the main parallel overheads.

## Running MPI Simulations

MPI jobs must be submitted through SLURM:

```bash
sbatch --wait --exclusive --partition=bigrun --ntasks=128 \
  --wrap "mpirun -np 128 ./build/gnuReleaseMPI/rich"
```

See [Running Simulations](Running-Simulations) for details.
