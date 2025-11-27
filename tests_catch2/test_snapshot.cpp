#include <catch2/catch_test_macros.hpp>
#include "snapshot.hpp"

using namespace snapshot;
using utils_for_tests::make_named_vector;

TEST_CASE_METHOD(mpi::RichMpiFixture, "Gather single",  "[gather][1][mpi]"){
    SECTION("Gather int vector"){
        std::vector<int> vec = {rank};

        Gather_single(vec);

        INFO("Rank " + std::to_string(rank));
        
        if(rank == mpi::rank_root){
            CHECK(vec.size() == comm_size);

            for(std::size_t i=0; i<comm_size; ++i){
                CHECK(vec[i] == i);
            }
        } 
    }
}

TEST_CASE_METHOD(mpi::RichMpiFixture, "Gather", "[gather][2][mpi]"){
    INFO("Rank " + std::to_string(rank));

    SECTION("Gather two vectors"){

        auto vec_int = make_named_vector("INT", {rank});
        auto vec_double = make_named_vector("DOUBLE", {static_cast<double>(rank)});

        Gather(vec_int, vec_double);

        if(rank == mpi::rank_root){
            CHECK(vec_int.vec.size() == comm_size);
            CHECK(vec_double.vec.size() == comm_size);

            for(std::size_t i=0; i<comm_size; ++i){
                CHECK(vec_int.vec[i] == i);
                CHECK(vec_double.vec[i] == static_cast<double>(i));
            }
        }
    }

    SECTION("Gather 3 vectors"){
        auto vec_int = make_named_vector("INT", {rank});
        auto vec_double = make_named_vector("DOUBLE", {static_cast<double>(rank)});
        auto vec_float = make_named_vector("FLOAT", {static_cast<float>(rank)});

        Gather(vec_int, vec_float, vec_double);

        if(rank == mpi::rank_root){
            CHECK(vec_int.vec.size() == comm_size);
            CHECK(vec_float.vec.size() == comm_size);
            CHECK(vec_double.vec.size() == comm_size);

            for(std::size_t i=0; i<comm_size; ++i){
                CHECK(vec_int.vec[i] == i);
                CHECK(vec_double.vec[i] == static_cast<double>(i));
                CHECK(vec_float.vec[i] == static_cast<float>(i));
            }
        }
    }
    
}

TEST_CASE_METHOD(mpi::RichNoMpiTestFixture, "close_enough", "[close_enough]"){
    SECTION("Exactly equal values are always close enough") {
        REQUIRE(close_enough(1.0, 1.0, 1e-6));
        REQUIRE(close_enough(0.0, 0.0, 1e-6));
        REQUIRE(close_enough(42, 42, 1e-6));
    }

    SECTION("Values within relative tolerance are reported as close") {
        // doubles
        double v1 = 10.0;
        double v2 = 10.0009;
        double rel_tol = 1e-3; // 0.1% relative tolerance

        REQUIRE(close_enough(v1, v2, rel_tol));

        // integers (tests the template with non-floating types)
        int i1 = 10;
        int i2 = 11;
        double int_rel_tol = 0.2; // 20% relative tolerance

        REQUIRE(close_enough(i1, i2, int_rel_tol));
    }

    SECTION("Values outside relative tolerance are not close") {
        double v1 = 10.0;
        double v2 = 10.0009;
        double small_tol = 1e-5; // much stricter tolerance

        REQUIRE_FALSE(close_enough(v1, v2, small_tol));

        int i1 = 10;
        int i2 = 15;
        double int_small_tol = 0.1; // 10% relative tolerance

        REQUIRE_FALSE(close_enough(i1, i2, int_small_tol));
    }
}

TEST_CASE_METHOD(mpi::RichNoMpiTestFixture, "compare_vectors_check", "[compare_vectors_check]"){
    double const rel_tol = 1e-3;

    SECTION("Identical vectors pass the check") {
        std::vector<double> v1{1.0, -2.5, 3.14, 0.0};
        std::vector<double> v2{1.0, -2.5, 3.14, 0.0};

        // Should not throw and all elements are "close enough"
        REQUIRE(compare_vectors_check("identical vectors", v1, v2, rel_tol));
    }

    SECTION("Vectors with small perturbations pass within tolerance") {
        std::vector<double> v1{10.0, 20.0, 30.0};
        std::vector<double> v2{
            10.0005,  // within 0.01% relative difference
            19.998,   // small relative difference
            30.0001
        };

        // With a reasonably loose rel_tol, they should be close enough
        double loose_tol = 1e-2; // 1%
        REQUIRE(compare_vectors_check("perturbed vectors", v1, v2, loose_tol));
    }

    SECTION("Vectors with different sizes throw runtime_error") {
        std::vector<double> v1{1.0, 2.0, 3.0};
        std::vector<double> v2{1.0, 2.0};  // shorter

        REQUIRE_THROWS_AS(
            compare_vectors_check("size mismatch", v1, v2, rel_tol),
            std::runtime_error
        );
    }
}

TEST_CASE_METHOD(mpi::RichMpiFixture, "Test_SnapShot", "[snapshot]"){
    SECTION("Test test_name"){
        snapshot::SnapShot snap{};
        REQUIRE(snap.test_name == "Test_SnapShot");
    }

    SECTION("Test CompareOrSaveGather"){
        snapshot::SnapShot const snap{"test_compare_or_save_gather"};
        
        auto const rank_size_t = static_cast<std::size_t>(rank);
        auto const rank_double = static_cast<double>(rank);
        auto const rank_float = static_cast<float>(rank);

        std::vector<std::size_t> id{rank_size_t*3, rank_size_t*3+1, rank_size_t*3+2};
        std::vector<double> values{3.0*rank_double, 3.0*rank_double+1.0, 3.0*rank_double+2.0};
        std::vector<float> values_float{3.0f*rank_float, 3.0f*rank_float+1.0f, 3.0f*rank_float+2.0f};

        auto const compare_success = snap.CompareOrSaveGather(
                "snap_shot_test_comm_size_" + std::to_string(rank),
                make_named_vector("ID", id),
                make_named_vector("values", values),
                make_named_vector("values_float", values_float)
            );

        REQUIRE(compare_success);
    }
}

