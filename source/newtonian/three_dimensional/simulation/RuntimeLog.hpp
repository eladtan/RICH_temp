#ifndef RUNTIME_LOG_HPP
#define RUNTIME_LOG_HPP

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>

enum class RuntimeLogLevel
{
    Summary = 0,
    Detailed = 1
};

enum class RuntimeColorMode
{
    Auto = 0,
    Always = 1,
    Never = 2
};

struct RuntimeLogConfiguration
{
    RuntimeLogConfiguration(
        RuntimeLogLevel const configured_level = RuntimeLogLevel::Summary,
        bool const configured_valid = true) :
        level(configured_level), valid(configured_valid)
    {}

    RuntimeLogLevel level;
    bool valid;
};

struct RuntimeColorConfiguration
{
    RuntimeColorConfiguration(
        RuntimeColorMode const configured_mode = RuntimeColorMode::Auto,
        bool const configured_valid = true) :
        mode(configured_mode), valid(configured_valid)
    {}

    RuntimeColorMode mode;
    bool valid;
};

inline RuntimeLogConfiguration GetRuntimeLogConfiguration(void)
{
    char const* const value = std::getenv("RICH_RUNTIME_LOG");
    if(value == nullptr || value[0] == '\0' || std::strcmp(value, "summary") == 0)
        return RuntimeLogConfiguration();
    if(std::strcmp(value, "detailed") == 0)
        return RuntimeLogConfiguration{RuntimeLogLevel::Detailed, true};
    return RuntimeLogConfiguration{RuntimeLogLevel::Summary, false};
}

inline bool RuntimeLogDetailed(void)
{
    RuntimeLogConfiguration const configuration = GetRuntimeLogConfiguration();
    return configuration.valid && configuration.level == RuntimeLogLevel::Detailed;
}

inline RuntimeColorConfiguration GetRuntimeColorConfiguration(void)
{
    char const* const value = std::getenv("RICH_RUNTIME_COLOR");
    if(value == nullptr || value[0] == '\0' || std::strcmp(value, "auto") == 0)
        return RuntimeColorConfiguration();
    if(std::strcmp(value, "always") == 0)
        return RuntimeColorConfiguration{RuntimeColorMode::Always, true};
    if(std::strcmp(value, "never") == 0)
        return RuntimeColorConfiguration{RuntimeColorMode::Never, true};
    return RuntimeColorConfiguration{RuntimeColorMode::Auto, false};
}

inline bool RuntimeColorEnabled(void)
{
    RuntimeColorConfiguration const configuration =
        GetRuntimeColorConfiguration();
    if(!configuration.valid || configuration.mode == RuntimeColorMode::Never)
        return false;
    if(configuration.mode == RuntimeColorMode::Always)
        return true;
    if(std::getenv("NO_COLOR") != nullptr)
        return false;
    return isatty(STDOUT_FILENO) != 0;
}

inline std::ostream& RuntimeTraceStream(void)
{
    return RuntimeLogDetailed() ? std::cout : std::clog;
}

#endif // RUNTIME_LOG_HPP
