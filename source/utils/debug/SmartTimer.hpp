#ifndef SMART_TIMER_HPP
#define SMART_TIMER_HPP

#include <vector>
#include <chrono>
#include <algorithm>
#include <map>
#include <execinfo.h>
#include <cxxabi.h>
#include <memory>
#include <string>
#include <iostream>
#include <cassert>
#include <exception>
#define BOOST_STACKTRACE_USE_ADDR2LINE
#include <boost/stacktrace.hpp>
#ifdef RICH_MPI
    #include <mpi.h>
#endif // RICH_MPI

#ifndef ALLOW_TIMING

#define START_TIMER(name)
#define START_TIMER_PREEMPTIVE(name)
#define START_TIMER_DISTINCT(name)
#define PRINT_TIMES()
#define SILENCE_TIMERS()
#define UNSILENCE_TIMERS()
#define CLEAR_TIMES()
#define DISABLE_TIMERS()

#else // ALLOW_TIMING

#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)
#define UNIQUE_NAME(base) CONCAT(base, CONCAT(__LINE__, __COUNTER__))
#define START_TIMER(name) SmartTimer::TimerCreator UNIQUE_NAME(timer_creator_)(name);
#define START_TIMER_PREEMPTIVE(name) SmartTimer::TimerCreator UNIQUE_NAME(timer_creator_)(name, false, true);
#define START_TIMER_DISTINCT(name) SmartTimer::TimerCreator UNIQUE_NAME(timer_creator_)(name, true);
#define PRINT_TIMES() SmartTimer::PrintTimes();
#define SILENCE_TIMERS() SmartTimer::TimerCreator::globalSilent = true;
#define UNSILENCE_TIMERS() SmartTimer::TimerCreator::globalSilent = false;
#define CLEAR_TIMES() SmartTimer::Node::root = SmartTimer::Node::MakeRoot();
#define DISABLE_TIMERS() SmartTimer::TimerCreator::disable = true;

namespace SmartTimer
{
    class Timer
    {
    public:
        Timer();

        void Destroy(void);

        bool done;
        std::chrono::high_resolution_clock::time_point start;    
        double time;
        double totalTime;
    };

    class TimerCreator;

    class Node
    {
    public:
        ~Node();

        enum Type
        {
            FUNCTION,
            BLOCK
        };

        Type type;
        std::string name;
        std::shared_ptr<Timer> timer;
        std::vector<std::shared_ptr<Node>> children;
        std::map<std::string, std::shared_ptr<Node>> children_function_map;
        std::map<std::string, std::shared_ptr<Node>> children_block_map;
        TimerCreator *creator = nullptr;
        bool destroyed = false;

        void PrintHelper(std::ostream &stream, const std::vector<std::pair<size_t, std::string>> &indentations) const;
        
        static std::shared_ptr<Node> root;
        static std::shared_ptr<Node> MakeRoot(void);
    };

    class TimerCreator
    {
    public:
        TimerCreator(const std::string &name, bool distinct = false, bool preempt = false);

        ~TimerCreator();

        void Destroy(void);

        bool destroyed;
        bool silent;
        std::weak_ptr<Node> node;
        static bool globalSilent;
        static bool disable;
    };

    void PrintTimes(void);
}

#endif // ALLOW_TIMING

#endif // SMART_TIMER_HPP