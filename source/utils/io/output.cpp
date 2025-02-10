#include "output.h"

void printArray(int rank, std::vector<int> &data)
{
    std::cout << "Final Process " << rank << ": " << data.size() << " Elements, as follows: " << std::endl;
    std::for_each(data.begin(), data.end(), [](int &value){std::cout << value << " ";});
    std::cout << std::endl;
}

inline void printArraySize(int rank, std::vector<int> &data)
{
    std::cout << "Final Process " << rank << ": " << data.size() << " Elements." << std::endl;
}

void printToFile(std::string fileName, std::vector<int> &vector)
{
	std::ofstream file;
    file.open(fileName, std::ios::out | std::ios::trunc);
	if(!file.good())
    {
		std::cerr << "Error in writing to " << fileName << std::endl;
		exit(EXIT_FAILURE);
	}
	else
    {
		for(int &value : vector)
        {
            file << value << std::endl;
        }
	}
	file.close();
}

#ifdef RICH_MPI

    void printAllArrays(std::vector<int> &myData)
    {
        rank_t rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto do_func = [&]() { printArray(rank, myData); };
        MPI_Sync(MPI_COMM_WORLD, do_func);
    }

    void printAllArraysSizes(std::vector<int> &myData)
    {
        rank_t rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        auto do_func = [&]() { printArraySize(rank, myData); };
        MPI_Sync(MPI_COMM_WORLD, do_func);
    }

    void printPointsDirectory(std::string dir_path, const std::vector<Vector3D> &points)
    {
        std::ofstream file;
        rank_t rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        std::string my_file = dir_path + "/" + std::to_string(rank);
        file.open(my_file, std::ios::out | std::ios::trunc);
        
        if(!file.good())
        {
            std::cerr << "Error in writing to " << my_file << std::endl;
            exit(EXIT_FAILURE);
        }
        else
        {
            file << std::setprecision(15);
            std::cout << "rank " << rank << " prints " << points.size() << " points" << std::endl;
            for(const Vector3D &point : points)
            {
                file << point << std::endl;
            }
        }
    }

#endif // RICH_MPI
