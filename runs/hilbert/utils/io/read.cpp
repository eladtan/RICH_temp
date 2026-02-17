#include "read.h"

std::vector<Vector3D> readFromFile(const std::string &fileName)
{
    std::vector<Vector3D> points;
    std::ifstream file(fileName, std::ios::in);

    if(!file.good())
    {
        std::cerr << "An error in the given file ('" << fileName << "')." << std::endl;
        throw std::runtime_error("Error reading file.");
    }
    int lineCounter = 1;
    std::string line;

    while(std::getline(file, line))
    {
        if(*(line.begin()) != '(' or *(line.end() - 1) != ')')
        {
            std::cerr << "Error in line " << lineCounter << ": " << line << std::endl;
            file.close();
            throw std::runtime_error("Error in file format.");
        }
        auto it = std::find(line.begin(), line.end(), ',');
        if(it == line.end())
        {
            std::cerr << "No comma separation was given between x and y in line " << lineCounter << ": " << line << std::endl;
            file.close();
            throw std::runtime_error("Error in file format.");
        }
        coord_t _x = std::stod(std::string(line.begin() + 1, it));
        auto it2 = std::find(it + 1, line.end(), ',');
        if(it2 == line.end())
        {
            std::cerr << "No comma separation was given between y and z in line " << lineCounter << ": " << line << std::endl;
            file.close();
            throw std::runtime_error("Error in file format.");
        }
        coord_t _y = std::stod(std::string(it + 1, it2));
        coord_t _z = std::stod(std::string(it2 + 1, line.end() - 1));

        Vector3D point(_x, _y, _z);
        points.push_back(point);
    }

    file.close();
    return points;
}

std::vector<Vector3D> readAllFilesInDirectory(const std::string &dirName)
{
    std::vector<Vector3D> finalResult;
    for(const auto &entry : std::filesystem::directory_iterator(dirName))
    {
        std::vector<Vector3D> points = readFromFile(entry.path());
        finalResult.insert(finalResult.end(), points.begin(), points.end());
    }
    return finalResult;
}

/*
std::vector<Vector3D> readVorocrust(const std::string &path)
{
    // READ HERE POINTS
    std::vector<double> in_seeds_x = read_vector(path + "/dump/zone_in_volume_seeds/x.txt");
    std::vector<double> in_seeds_y = read_vector(path + "/dump/zone_in_volume_seeds/y.txt");
    std::vector<double> in_seeds_z = read_vector(path + "/dump/zone_in_volume_seeds/z.txt");

    std::vector<double> out_seeds_x = read_vector(path + "/dump/zone_out_seeds/x.txt");
    std::vector<double> out_seeds_y = read_vector(path + "/dump/zone_out_seeds/y.txt");
    std::vector<double> out_seeds_z = read_vector(path + "/dump/zone_out_seeds/z.txt");

    size_t const out_size = out_seeds_x.size();
    size_t const in_size = in_seeds_x.size();
    size_t const tot_size = out_size + in_size;

    std::vector<Vector3D> points(tot_size, Vector3D()); // fill this vector
    

    // set in seeds
    for(size_t i=0; i<in_size; ++i){
        // std::cout << i << std::setprecision(16) << ", " << in_seeds_x[i] << ", " << in_seeds_y[i] << ", " << in_seeds_z[i] << std::endl;
        points[i].Set(in_seeds_x[i], in_seeds_y[i], in_seeds_z[i]);
    }

    // set out seeds
    for(size_t i=in_size; i<tot_size; ++i){
        // std::cout << i << std::setprecision(16) << ", " << out_seeds_x[i-in_size] << ", " << out_seeds_y[i-in_size] << ", " << out_seeds_z[i-in_size] << std::endl;

        points[i].Set(out_seeds_x[i-in_size], out_seeds_y[i-in_size], out_seeds_z[i-in_size]);
    }

    return points;
}
*/