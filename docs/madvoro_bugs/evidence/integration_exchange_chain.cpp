#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "source/mpi/ExchangeChain.hpp"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const std::string mode = argc > 1 ? argv[1] : "permutation";
    ExchangeChain chain;
    std::vector<int> data;
    if(mode == "permutation")
    {
        chain.Reset(2);
        chain.Exchange({}, {}, {1, 0});
        data = {100 + 10 * rank, 101 + 10 * rank};
        std::cout << "rank=" << rank << " targets=" << chain.GetTarget(0).second
                  << "," << chain.GetTarget(1).second << std::endl;
        MPI_exchange_data(chain, data);
        const bool ok = data == std::vector<int>({101 + 10 * rank, 100 + 10 * rank});
        std::cout << "rank=" << rank << " expected=" << 101 + 10 * rank << ","
                  << 100 + 10 * rank << " actual=" << data.at(0) << ","
                  << data.at(1) << " pass=" << ok << std::endl;
        MPI_Finalize();
        return ok ? 0 : 3;
    }
    if(mode == "empty" && size == 2)
    {
        chain.Reset(rank == 0 ? 1 : 0);
        if(rank == 0)
        {
            chain.Exchange({1}, {{0}}, {});
            data = {1234};
        }
        else
        {
            chain.Exchange({0}, {{}}, {});
        }
        std::cout << "rank=" << rank << " original_count=" << chain.GetNorg()
                  << " final_count=" << chain.GetReversedTranslationMap().size()
                  << " entering_field_transfer" << std::endl;
        MPI_exchange_data(chain, data);
        std::cout << "rank=" << rank << " returned_field_transfer size=" << data.size() << std::endl;
        MPI_Finalize();
        return 0;
    }
    std::cerr << "Usage: permutation (any rank count), empty (2 ranks)" << std::endl;
    MPI_Finalize();
    return 2;
}
