#ifndef RUNVOROCRUST_HPP
#define RUNVOROCRUST_HPP 1
#include <filesystem>
#include <vector>
#include "source/3D/elementary/Vector3D.hpp"
#include "PLC/PL_Complex.hpp"
#include "VoroCrustAlgorithm.hpp"

std::pair<std::vector<std::vector<Vector3D> >, std::vector<std::vector<Vector3D> >> 
    runVoroCrust(std::filesystem::path const& data_directory_path, std::filesystem::path const& output_directory, VoroCrustAlgorithm *& alg, std::vector<PL_Complex> &zones_plcs, bool restart = false);

#endif // RUNVOROCRUST_HPP