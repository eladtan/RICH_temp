RICH particle MPI datatype regression
====================================

Build the `rich_particle_mpi_dtype_test` and
`rich_particle_mpi_dtype_extended_test` targets in a configured MPI RICH build,
then run `ctest -R '^rich_particle_mpi_dtype' --output-on-failure` there.
Both tests use two MPI ranks. The extended variant enables debug state,
polarization, and three tracing-history entries.

The tests exercise RICH's actual `MPI_has_complex_dtype<Particle3D>`
specialization, used by the mesh-movement sparse all-to-all exchanges.
Three-particle arrays verify the MPI extent as well as logical state, using
the independent serializer for comparison. Receive buffers are tested both
fresh and reused with deliberately unrelated RNG state. Particles carry
colliding/sentinel IDs but distinct keys and nonzero 64-bit counters; sixteen
subsequent RNG values must match uninterrupted streams.

Missing RNG fields in the former datatype allowed migration to leave the
receiver's default or stale RNG state in place. This regression also covers
source-cell identity, radiation transport state, and optional polarization.
