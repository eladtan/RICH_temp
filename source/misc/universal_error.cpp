#include <iostream>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <array>

#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>

#ifdef RICH_MPI
  #include <mpi.h>
#endif // RICH_MPI
#include "universal_error.hpp"

using namespace std;

UniversalError::UniversalError(const string& err_msg):
  err_msg_(err_msg),
  fields_(),
  stack_frames_{},
  stack_depth_(backtrace(stack_frames_, MAX_STACK_FRAMES))
{}

void UniversalError::Append2ErrorMessage(string const& msg)
{
  err_msg_ += msg;
}

const string& UniversalError::getErrorMessage(void) const
{
  return err_msg_;
}

UniversalError::~UniversalError(void) {}

UniversalError::UniversalError(const UniversalError& eo):
  err_msg_(eo.getErrorMessage()),
  fields_(eo.fields_),
  stack_depth_(eo.stack_depth_)
{
  std::memcpy(stack_frames_, eo.stack_frames_, sizeof(void*) * stack_depth_);
}

namespace
{

std::string demangle(const char *mangled)
{
    int status = 0;
    std::unique_ptr<char, void(*)(void*)> demangled(
        abi::__cxa_demangle(mangled, nullptr, nullptr, &status),
        std::free);
    return (status == 0 && demangled) ? demangled.get() : mangled;
}

std::string getExecutablePath()
{
    std::array<char, 4096> buf{};
    ssize_t len = readlink("/proc/self/exe", buf.data(), buf.size() - 1);
    if(len > 0)
    {
        buf[static_cast<size_t>(len)] = '\0';
        return std::string(buf.data());
    }
    return "";
}

struct SourceLocation
{
    std::string function;
    std::string file;
    std::string line;
};

SourceLocation resolveAddress(const std::string &exe, void *addr)
{
    SourceLocation loc;

    Dl_info info;
    if(dladdr(addr, &info) && info.dli_sname)
    {
        loc.function = demangle(info.dli_sname);
    }

    if(exe.empty())
        return loc;

    std::ostringstream cmd;
    cmd << "addr2line -C -f -p -e " << exe << " " << addr << " 2>/dev/null";

    FILE *pipe = popen(cmd.str().c_str(), "r");
    if(!pipe)
        return loc;

    std::array<char, 512> line_buf{};
    std::string output;
    while(fgets(line_buf.data(), static_cast<int>(line_buf.size()), pipe))
        output += line_buf.data();
    pclose(pipe);

    if(output.empty() || output.find("??") == 0)
        return loc;

    // addr2line -C -f -p output: "function at file:line"
    // or "function at file:line (discriminator N)"
    auto at_pos = output.find(" at ");
    if(at_pos != std::string::npos)
    {
        std::string func = output.substr(0, at_pos);
        if(!func.empty() && func != "??")
            loc.function = func;

        std::string rest = output.substr(at_pos + 4);
        while(!rest.empty() && (rest.back() == '\n' || rest.back() == '\r'))
            rest.pop_back();

        auto paren = rest.find(" (discriminator");
        if(paren != std::string::npos)
            rest = rest.substr(0, paren);

        auto colon = rest.rfind(':');
        if(colon != std::string::npos)
        {
            loc.file = rest.substr(0, colon);
            loc.line = rest.substr(colon + 1);
        }
    }

    return loc;
}

void printStackTrace(void * const *frames, int depth, std::ostream &os, const std::string &prefix)
{
    if(depth <= 0)
        return;

    std::string exe = getExecutablePath();

    os << prefix << "Stack trace (" << depth << " frames):" << std::endl;

    // skip frames 0-1 (UniversalError ctor + backtrace)
    int start = 2;
    for(int i = start; i < depth; ++i)
    {
        SourceLocation loc = resolveAddress(exe, frames[i]);

        os << prefix << "  #" << (i - start) << "  ";

        if(!loc.function.empty())
            os << loc.function;
        else
            os << frames[i];

        if(!loc.file.empty() && loc.file != "??")
        {
            os << "  (" << loc.file;
            if(!loc.line.empty() && loc.line != "0" && loc.line != "?")
                os << ":" << loc.line;
            os << ")";
        }

        os << std::endl;
    }
}

} // anonymous namespace

void reportError(UniversalError const& eo, std::ostream& os)
{
  std::string prefix = "";
  #ifdef RICH_MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    prefix = "============" + to_string(rank) + "============ ";
  #endif // RICH_MPI
  os.precision(14);
  os << prefix << eo.getErrorMessage() << std::endl;
  for_each(eo.fields_.begin(), eo.fields_.end(),
          [&os, &prefix](const pair<string, UniversalError::PrintableAny>& f) {os << prefix << f.first << " " << f.second << endl;});

  printStackTrace(eo.stack_frames_, eo.stack_depth_, os, prefix);
}