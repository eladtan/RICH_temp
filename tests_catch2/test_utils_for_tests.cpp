#include <catch2/catch_test_macros.hpp>
#include "utils_for_tests.hpp"
#include <filesystem>

TEST_CASE("Test_Zip") {
    std::vector<int> const v_int{1,2,3};
    std::vector<double> const v_double{11.0, 12.0, 13.0};
    SECTION("zip int and double"){
        auto const v_zip = utils_for_tests::zip(v_int, v_double);

        REQUIRE(v_zip.size() == v_int.size());
        REQUIRE(v_zip.size() == v_double.size());

        for(std::size_t i=0; i < v_zip.size(); ++i){
            auto const [val_int, val_double] = v_zip[i]; 

            REQUIRE(val_int == v_int[i]);
            REQUIRE(val_double == v_double[i]);
        }
    }

    std::vector<std::string> v_string{"ay", "ayay", "ayayay"};
    SECTION("zip string double int"){
        auto const v_zip = utils_for_tests::zip(v_string, v_double, v_int);

        REQUIRE(v_zip.size() == v_int.size());
        REQUIRE(v_zip.size() == v_double.size());
        REQUIRE(v_zip.size() == v_string.size());

        for(std::size_t i=0; i < v_zip.size(); ++i){
            auto const [val_string, val_double, val_int] = v_zip[i]; 

            REQUIRE(val_string == v_string[i]);
            REQUIRE(val_double == v_double[i]);
            REQUIRE(val_int == v_int[i]);
        }
    }

    SECTION("zip then unzip"){
        auto const [v_int_after, v_double_after] = utils_for_tests::unzip(utils_for_tests::zip(v_int, v_double));

        REQUIRE(v_int == v_int_after);
        REQUIRE(v_double == v_double_after);
    }

    SECTION("zip mismatched sizes"){
        REQUIRE_THROWS_AS(utils_for_tests::zip(std::vector<int>{1,2,3}, std::vector<int>{4,5}), std::runtime_error);
        REQUIRE_THROWS_AS(utils_for_tests::zip(std::vector<int>{1,2,3}, std::vector<std::string>{"ayy", "ayyy", "ayyyy", "ayy"}), std::runtime_error);
    }
}

TEST_CASE("Test unzip"){
    std::vector<std::tuple<int, double>> const vec_zip{{1, 2.0}, {3, 4.0}, {5, 6.0}};
    
    SECTION("<int, double> vector unzip"){
        auto const [v1, v2] = utils_for_tests::unzip(vec_zip);

        REQUIRE(v1.size() == vec_zip.size());
        REQUIRE(v2.size() == vec_zip.size());

        for(std::size_t i=0; i<vec_zip.size(); ++i){
            auto const [val1, val2] = vec_zip[i];
            REQUIRE(val1 == v1[i]);
            REQUIRE(val2 == v2[i]);
        }
    }

    SECTION("unzip then zip"){
        auto const [v1, v2] = utils_for_tests::unzip(vec_zip);
        auto const vec_zip_again = utils_for_tests::zip(v1, v2);

        REQUIRE(vec_zip_again == vec_zip);
    }
}

TEST_CASE("sort_vectors_by_index<0> sorts by first vector (two vectors)", "[sort_vectors_by_index]")
{
    std::vector<int>    ids    {3, 1, 2};
    std::vector<double> values {30.0, 10.0, 20.0};

    auto [sorted_ids, sorted_values] = utils_for_tests::sort_vectors_by_index<0>(ids, values);

    REQUIRE(sorted_ids   == std::vector<int>   {1, 2, 3});
    REQUIRE(sorted_values == std::vector<double>{10.0, 20.0, 30.0});
}

TEST_CASE("sort_vectors_by_index<1> sorts by second vector (two vectors)", "[sort_vectors_by_index]")
{
    std::vector<int>    ids    {3, 1, 2};
    std::vector<double> values {30.0, 10.0, 20.0};

    auto [sorted_ids, sorted_values] = utils_for_tests::sort_vectors_by_index<1>(ids, values);

    // Sorted by values: 10.0, 20.0, 30.0
    REQUIRE(sorted_values == std::vector<double>{10.0, 20.0, 30.0});
    REQUIRE(sorted_ids    == std::vector<int>   {1, 2, 3});  // ids follow the sort
}

TEST_CASE("sort_vectors_by_index works with three vectors and different sort indices", "[sort_vectors_by_index]")
{
    std::vector<int>         ids    {3, 1, 2};
    std::vector<double>      scores {9.5, 7.0, 8.0};
    std::vector<std::string> names  {"charlie", "alice", "bob"};

    SECTION("sort by ids (Index = 0)")
    {
        auto [s_ids, s_scores, s_names] = utils_for_tests::sort_vectors_by_index<0>(ids, scores, names);

        REQUIRE(s_ids    == std::vector<int>         {1, 2, 3});
        REQUIRE(s_scores == std::vector<double>      {7.0, 8.0, 9.5});
        REQUIRE(s_names  == std::vector<std::string> {"alice", "bob", "charlie"});
    }

    SECTION("sort by scores (Index = 1)")
    {
        auto [s_ids, s_scores, s_names] = utils_for_tests::sort_vectors_by_index<1>(ids, scores, names);

        REQUIRE(s_scores == std::vector<double>      {7.0, 8.0, 9.5});
        REQUIRE(s_ids    == std::vector<int>         {1, 2, 3});
        REQUIRE(s_names  == std::vector<std::string> {"alice", "bob", "charlie"});
    }

    SECTION("sort by names (Index = 2)")
    {
        auto [s_ids, s_scores, s_names] = utils_for_tests::sort_vectors_by_index<2>(ids, scores, names);

        // Alphabetical: "alice", "bob", "charlie"
        REQUIRE(s_names  == std::vector<std::string> {"alice", "bob", "charlie"});
        REQUIRE(s_ids    == std::vector<int>         {1, 2, 3});
        REQUIRE(s_scores == std::vector<double>      {7.0, 8.0, 9.5});
    }
}

TEST_CASE("sort_vectors_by_index on empty vectors returns empty vectors", "[sort_vectors_by_index]")
{
    std::vector<int>         ids;
    std::vector<double>      scores;
    std::vector<std::string> names;

    auto [s_ids, s_scores, s_names] = utils_for_tests::sort_vectors_by_index<0>(ids, scores, names);

    REQUIRE(s_ids.empty());
    REQUIRE(s_scores.empty());
    REQUIRE(s_names.empty());
}

TEST_CASE("sort_vectors_by_index throws when vector sizes differ", "[sort_vectors_by_index]")
{
    std::vector<int>    ids    {1, 2, 3};
    std::vector<double> values {10.0, 20.0};  // shorter

    // Assuming sort_vectors_by_index uses zip() which throws std::runtime_error
    REQUIRE_THROWS_AS(
        (utils_for_tests::sort_vectors_by_index<0>(ids, values)),
        std::runtime_error
    );
}

TEST_CASE("json utilities"){
    using namespace utils_for_tests::json;
    std::filesystem::path tmp_dir = std::filesystem::path{__FILE__}.parent_path() / "tmp"; 
    
    if(utils_for_tests::get_mpi_rank() == 0){
        SECTION("json test sucess"){
            auto const path_to_json = tmp_dir / "test_save_data_to_json_file_section_1.json";

            auto const a_int_vec = make_named_vector<int>("a_vec", {1, 2, 3});
            auto const b_double_vec = make_named_vector<double>("b_vec", {4.0, 5.0, 6.0});
            auto const c_string_vec = make_named_vector<std::string>("c_vec", {"yo", "yoyo", "yoyoyoyo", "yoyoyoyoyo"});
            
            save_data_to_json_file(path_to_json, a_int_vec, b_double_vec, c_string_vec);

            auto const& [a_load, b_load, c_load] = load_data_from_json<int, double, std::string>(path_to_json, {"a_vec", "b_vec", "c_vec"});
            
            REQUIRE(a_load.name == a_int_vec.name);
            REQUIRE(a_load.vec == a_int_vec.vec);
            
            REQUIRE(b_load.name == b_double_vec.name);
            REQUIRE(b_load.vec == b_double_vec.vec);
            
            REQUIRE(c_load.name == c_string_vec.name);
            REQUIRE(c_load.vec == c_string_vec.vec);
        }

        SECTION("json different number of parameters"){
            auto const a_int_vec = make_named_vector<int>("a_vec", {1, 2, 3});
            auto const path_to_json = tmp_dir / "test_save_data_to_json_file_data_section_2.json";

            save_data_to_json_file(path_to_json, a_int_vec);
            REQUIRE_THROWS(load_data_from_json<int, double, std::string>(path_to_json, {"a_vec", "b_vec", "c_vec"}));
        }
    }
}