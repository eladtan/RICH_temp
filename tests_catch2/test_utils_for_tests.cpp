#include <catch2/catch_test_macros.hpp>
#include "utils_for_tests.hpp"

TEST_CASE("Test_Zip") {
    std::vector<int> const v_int{1,2,3};
    std::vector<double> const v_double{11.0, 12.0, 13.0};
    SECTION("zip int and double"){
        auto const v_zip = utils_for_tests::zip(v_int, v_double);

        REQUIRE(v_zip.size() == v_int.size());
        REQUIRE(v_zip.size() == v_double.size());

        for(std::size_t i=0; i < v_zip.size(); ++i){
            REQUIRE(v_zip[i].first == v_int[i]);
            REQUIRE(v_zip[i].second == v_double[i]);
        }
    }

    SECTION("zip then unzip"){
        auto const [v_int_after, v_double_after] = utils_for_tests::unzip(utils_for_tests::zip(v_int, v_double));

        REQUIRE(v_int == v_int_after);
        REQUIRE(v_double == v_double_after);
    }

    SECTION("zip mismatched sizes"){
        REQUIRE_THROWS_AS(utils_for_tests::zip(std::vector<int>{1,2,3}, std::vector<int>{4,5}), std::runtime_error);
    }
}

TEST_CASE("Test unzip"){
    std::vector<std::pair<int, double>> const vec_zip{{1, 2.0}, {3, 4.0}, {5, 6.0}};
    
    SECTION("<int, double> vector unzip"){
        auto const [v1, v2] = utils_for_tests::unzip(vec_zip);

        REQUIRE(v1.size() == vec_zip.size());
        REQUIRE(v2.size() == vec_zip.size());

        for(std::size_t i=0; i<vec_zip.size(); ++i){
            REQUIRE(vec_zip[i].first == v1[i]);
            REQUIRE(vec_zip[i].second == v2[i]);
        }
    }

    SECTION("unzip then zip"){
        auto const [v1, v2] = utils_for_tests::unzip(vec_zip);
        auto const vec_zip_again = utils_for_tests::zip(v1, v2);

        REQUIRE(vec_zip_again == vec_zip);
    }
}