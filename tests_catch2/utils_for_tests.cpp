#include "utils_for_tests.hpp"
#include <catch2/catch_test_macros.hpp>

namespace utils_for_tests {

std::tuple<
    named_vector<double>,
    named_vector<double>,
    named_vector<double>
> extract_data_from_cells(
    std::string const& name,
    std::vector<ComputationalCell3D> const& cells,
    Vector3D ComputationalCell3D::* const ptr_to_cell_data
){
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;

    x.reserve(cells.size());
    y.reserve(cells.size());
    z.reserve(cells.size());

    for(std::size_t i=0; i<cells.size(); ++i){
        auto const& vec = cells[i].*ptr_to_cell_data;

        x.push_back(vec.x);
        y.push_back(vec.y);
        z.push_back(vec.z);
    }

    return {
        make_named_vector(name+"_x", x),
        make_named_vector(name+"_y", y),
        make_named_vector(name+"_z", z)
    };
}

std::tuple<
    named_vector<double>,
    named_vector<double>,
    named_vector<double>
>
extract_center_of_mass(Tessellation3D const& tess){
    std::size_t Ncells = tess.GetPointNo();
    
    std::vector<double> CMx;
    std::vector<double> CMy;
    std::vector<double> CMz;
    
    CMx.reserve(Ncells);
    CMy.reserve(Ncells);
    CMz.reserve(Ncells);

    for(std::size_t cell=0; cell < Ncells; ++cell){
        Vector3D const CM = tess.GetCellCM(cell);
        CMx.push_back(CM.x);
        CMy.push_back(CM.y);
        CMz.push_back(CM.z);
    }

    return {
        make_named_vector("CM_x", CMx),
        make_named_vector("CM_y", CMy),
        make_named_vector("CM_z", CMz)
    };
}

} // namespace utils_for_tests

namespace utils_for_tests::mpi {
    int get_mpi_rank() {
        int rank = 0;
        
        #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        #endif

        return rank;
    }

    int get_mpi_world_size(){
        int ws = 1;

        #ifdef RICH_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
        #endif

        return ws;
    }

    void mpi_barrier(){
        #ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        #endif
    }

    RichBasicTestFixture::RichBasicTestFixture() 
    :   rank{get_mpi_rank()},
        comm_size{get_mpi_world_size()} 
    {}

    RichNoMpiTestFixture::RichNoMpiTestFixture() : RichBasicTestFixture() {
        if(rank != rank_root){
            SKIP("Test has no MPI, running only in root");
        }
    }


    RichMpiFixture::RichMpiFixture() : RichBasicTestFixture(){}
} // namespace utils_for_tests::mpi