#include "3D/output/read3D.hpp"

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }
    std::string filename = argv[1];
    Snapshot3D snapshot = ReadSnapshot3DParallel_AOS(filename);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank == 0)
    {
        std::cout << "LL UR are " << snapshot.ll << " " << snapshot.ur << std::endl;
        std::cout << "Tracer names: " << snapshot.tracerstickernames.first << std::endl;
        std::cout << "Sticker names: " << snapshot.tracerstickernames.second << std::endl;
    }
    if(rank % 4 == 0)
    {
        std::cout << "Rank " << rank << " has " << snapshot.cells.size() << " cells" << std::endl;
        std::cout << "Rank " << rank << " has " << snapshot.mesh_points.size() << " mesh points" << std::endl;
        std::cout << "Rank " << rank << " has " << snapshot.volumes.size() << " volumes" << std::endl;
        std::cout << "Rank " << rank << ", first cells' tracers " << snapshot.cells[0].tracers << std::endl;
    }

    MPI_Finalize();
    return 0;
}