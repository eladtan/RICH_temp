#include "MemoryRecorder.hpp"
#include "misc/universal_error.hpp"

std::vector<MemoryRecorder::Location> MemoryRecorder::locations = std::vector<MemoryRecorder::Location>();

std::ostream &MemoryRecorder::operator<<(std::ostream &os, const Location &loc)
{
    os << loc.location << " (RSS: " << loc.rss << " KB)";
    return os;
}

void MemoryRecorder::Clear(void)
{
    locations.clear();
}

size_t MemoryRecorder::GetRSS(void)
{
    std::string line;
    std::ifstream status_file("/proc/" + std::to_string(getpid()) + "/status");
    if(!status_file.is_open())
    {
        UniversalError eo("Can't calculate getRSS: couldn't find proc ID");
        eo.addEntry("PID", getpid());
        throw eo;
    }

    while(std::getline(status_file, line))
    {
        if(line.compare(0, 6, "VmRSS:") == 0)
        {
            size_t firstDigit = line.find_first_of("0123456789");
            size_t lastDigit = line.find_last_of("0123456789", firstDigit);
            if(firstDigit != std::string::npos)
            {
                return std::stol(line.substr(firstDigit, lastDigit));
            }
        }
    }
    UniversalError eo("Can't calculate getRSS: shouldn't reach here");
    throw eo;
}

size_t MemoryRecorder::Record(const std::string &location)
{
    Location &loc = locations.emplace_back();
    loc.location = location;
    try
    {
        loc.rss = GetRSS();
    }
    catch(UniversalError &eo)
    {
        eo.addEntry("Location", location);
        throw eo;
    }
    return loc.rss;
}

void MemoryRecorder::PrintRecords(std::ostream &os)
{
    if(locations.empty())
    {
        os << "No Records Found" << std::endl;
        return;
    }
    os << locations[0] << std::endl;
    for(size_t i = 1; i < locations.size(); i++)
    {
        int64_t numericChange = static_cast<int64_t>(locations[i].rss) - static_cast<int64_t>(locations[i - 1].rss);
        std::string change = std::string(((numericChange > 0)? "+": "")) + std::to_string(numericChange) + " KB";
        os << "  " << locations[i] << " (change: " << change << ")" << std::endl;
    }
}