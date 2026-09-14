#include "mpi/MPI_Particle3D_dtype.hpp"
#include "monte/utils/CounterRNG.hpp"
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

Particle3D makeParticle(int rank, int slot)
{
    Particle3D p;
    const std::uint64_t tag = 1000 * (rank + 1) + slot;
    p.rank = rank;
    // Deliberately collide IDs: only the carried key/counter define the stream.
    p.id = slot == 0 ? std::numeric_limits<std::uint64_t>::max() : 7;
    p.cellID = tag;
    p.sourceCellID = tag + 100;
    p.cellIndex = slot;
    p.location = Vector3D(tag, tag + 1, tag + 2);
    p.velocity = Vector3D(1, 2, 3);
    p.timeLeft = 0.25;
    p.frequency = 2;
    p.weight = 3;
    p.initialWeight = 4;
    p.rngKey = STORM::CounterRNG::makeKey(1234, rank, tag);
    p.rngCounter = (std::uint64_t(1) << 40) + tag;
    p.radiationState.flags = 3;
    p.radiationState.pendingFlux = Vector3D(tag + 3, tag + 4, tag + 5);
    p.radiationState.bypassCellID = tag + 200;
    p.steps = 17;
    p.on_track = p.sent = 1;
#ifdef MONTECARLO_POLARIZATION
    p.stokesQ = 0.125;
    p.stokesU = -0.25;
    p.polarizationBasis = Vector3D(0, 1, 0);
    p.polarizationInitialized = 1;
    p.radiationState.pendingMeanScatterings = 3.5;
#endif
#ifdef STORM_DEBUG
    p.checkedHere = 1;
    p.ghostIndex = tag;
    p.newCellValue = Vector3D(4, 5, 6);
    p.nextRank = rank;
    p.removedFromRank = 1;
    p.sentByRank = rank;
    p.lastSeen = tag;
    p.lastSeenRank = p.lastSeenRankBuf = rank;
    p.lastSeenIndex = tag;
#endif
#ifdef STORM_WITH_TRACING_HISTORY
    p.recordHistory(tag, rank, 2);
#endif
    return p;
}

void check(const Particle3D &actual, const Particle3D &expected)
{
    // The independent serializer covers every logical field (not vtables or padding).
    Serializer a, b;
    actual.dump(&a);
    expected.dump(&b);
    if(a.size() != b.size() || std::memcmp(a.getData(), b.getData(), a.size()))
        throw std::runtime_error("MPI datatype lost particle state");
    for(std::uint64_t draw = 0; draw < 16; ++draw)
        if(STORM::CounterRNG::next(actual.rngKey, actual.rngCounter + draw) !=
           STORM::CounterRNG::next(expected.rngKey, expected.rngCounter + draw))
            throw std::runtime_error("Migration changed RNG stream continuation");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try
    {
        if(size != 2) throw std::runtime_error("Run this regression with two MPI ranks");
        const auto dtype = MPI_has_complex_dtype<Particle3D>::getDatatype();
        MPI_Aint lower, extent;
        MPI_Type_get_extent(dtype, &lower, &extent);
        if(lower != 0 || extent != sizeof(Particle3D))
            throw std::runtime_error("Incorrect particle-array MPI stride");
        std::array<Particle3D, 3> send, receive;
        for(int i = 0; i < 3; ++i) send[i] = makeParticle(rank, i);
        for(int pass = 0; pass < 2; ++pass)
        {
            // First fresh buffers, then reuse buffers containing unrelated RNG state.
            if(pass) for(int i = 0; i < 3; ++i) receive[i] = makeParticle(rank + 10, i);
            MPI_Sendrecv(send.data(), 3, dtype, 1-rank, pass,
                         receive.data(), 3, dtype, 1-rank, pass, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(int i = 0; i < 3; ++i) check(receive[i], makeParticle(1-rank, i));
        }
        if(rank == 0) std::cout << "Particle MPI state and RNG continuation preserved in fresh/reused buffers\n";
    }
    catch(const std::exception &e)
    {
        std::cerr << "Rank " << rank << ": " << e.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
}
