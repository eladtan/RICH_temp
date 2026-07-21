#ifndef UTILS_READ_H
#define UTILS_READ_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include "3D/elementary/Vector3D.hpp"
#include "3D/hilbert/hilbertTypes.h"

std::vector<Vector3D> readFromFile(const std::string &fileName);

std::vector<Vector3D> readAllFilesInDirectory(const std::string &dirName);

std::vector<Vector3D> readVorocrust(const std::string &path);

#endif // UTILS_READ_H