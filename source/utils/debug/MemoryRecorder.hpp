#ifndef MEMORY_RECORDER_HPP
#define MEMORY_RECORDER_HPP

#include <boost/container/flat_set.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "misc/universal_error.hpp"

namespace MemoryRecorder
{
    struct Location
    {
        std::string location;
        size_t rss;

        friend std::ostream &operator<<(std::ostream &os, const Location &location);
    };

    extern std::vector<Location> locations;

    size_t GetRSS(void);

    size_t Record(const std::string &location);

    void PrintRecords(std::ostream &os);

    void Clear(void);
}

#endif // MEMORY_RECORDER_HPP