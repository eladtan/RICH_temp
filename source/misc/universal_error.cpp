#include <iostream>
#include <algorithm>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <array>
#include <cstdint>
#include <iomanip>

#include <execinfo.h>
#include <cxxabi.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <sys/file.h>
#include <unistd.h>

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

std::string shellQuote(const std::string &value)
{
    std::string quoted = "'";
    for(char c : value)
    {
        if(c == '\'')
            quoted += "'\\''";
        else
            quoted += c;
    }
    quoted += "'";
    return quoted;
}

std::string formatAddress(std::uintptr_t addr)
{
    std::ostringstream os;
    os << "0x" << std::hex << addr;
    return os.str();
}

bool applyAddr2Line(const std::string &objectPath, std::uintptr_t address, SourceLocation &loc)
{
    if(objectPath.empty())
        return false;

    std::ostringstream cmd;
    cmd << "addr2line -C -f -p -e " << shellQuote(objectPath)
        << " " << formatAddress(address) << " 2>/dev/null";

    FILE *pipe = popen(cmd.str().c_str(), "r");
    if(!pipe)
        return false;

    std::array<char, 512> line_buf{};
    std::string output;
    while(fgets(line_buf.data(), static_cast<int>(line_buf.size()), pipe))
        output += line_buf.data();
    pclose(pipe);

    if(output.empty() || output.find("??") == 0)
        return false;

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

    return !loc.file.empty();
}

SourceLocation resolveAddress(const std::string &exe, void *addr)
{
    SourceLocation loc;
    std::string objectPath = exe;
    std::uintptr_t lookupAddress = reinterpret_cast<std::uintptr_t>(addr);
    std::uintptr_t rawAddress = lookupAddress;

    Dl_info info;
    if(dladdr(addr, &info))
    {
        if(info.dli_sname)
            loc.function = demangle(info.dli_sname);
        if(info.dli_fname)
            objectPath = info.dli_fname;
        if(info.dli_fbase)
        {
            std::uintptr_t baseAddress = reinterpret_cast<std::uintptr_t>(info.dli_fbase);
            if(lookupAddress >= baseAddress)
                lookupAddress -= baseAddress;
        }
    }

    bool const isMainExecutable = (objectPath == exe);
    if(isMainExecutable)
    {
        if(!applyAddr2Line(objectPath, rawAddress, loc) && rawAddress != lookupAddress)
            applyAddr2Line(objectPath, lookupAddress, loc);
    }
    else if(!applyAddr2Line(objectPath, lookupAddress, loc) && rawAddress != lookupAddress)
    {
        applyAddr2Line(objectPath, rawAddress, loc);
    }

    return loc;
}

// The frame's object file and the address addr2line needs for it ("object@
// address"), and the dynamic symbol name when there is one.  No subprocess.
// A position-dependent executable is looked up by its absolute address, a
// shared object or PIE by the offset from its load address.
SourceLocation describeAddress(const std::string &exe, void *addr)
{
    SourceLocation loc;
    std::string objectPath = exe;
    std::uintptr_t lookup = reinterpret_cast<std::uintptr_t>(addr);
    Dl_info info;
    if(dladdr(addr, &info))
    {
        if(info.dli_sname)
            loc.function = demangle(info.dli_sname);
        if(info.dli_fname)
            objectPath = info.dli_fname;
        if(info.dli_fbase)
        {
            const ElfW(Ehdr) *header =
                static_cast<const ElfW(Ehdr) *>(info.dli_fbase);
            const std::uintptr_t base =
                reinterpret_cast<std::uintptr_t>(info.dli_fbase);
            if(header->e_type != ET_EXEC && lookup >= base)
                lookup -= base;
        }
    }
    loc.file = objectPath + "@" + formatAddress(lookup);
    return loc;
}

// addr2line can need ~2 GB on a large -g3 binary (binutils 2.42), and a
// collective error is reported by every rank at once: 192 concurrent
// symbolizers would exhaust a node.  At most one process per node therefore
// symbolizes at a time, chosen by a non-blocking lock on a node-local file;
// the others print object@address at once.  A lone failing rank always gets
// the lock.  RICH_SYMBOLIZE_STACK=0 disables symbolization.  Without a lock
// file, an MPI run symbolizes on rank 0 only.
class SymbolizerLock
{
public:
    SymbolizerLock(): fd_(-1), held_(false), disabled_(false)
    {
        const char *setting = std::getenv("RICH_SYMBOLIZE_STACK");
        if(setting != nullptr && setting[0] == '0' && setting[1] == '\0')
        {
            disabled_ = true;
            return;
        }
        const std::string path = "/dev/shm/rich_symbolize_" +
            std::to_string(static_cast<unsigned long>(getuid())) + ".lock";
        fd_ = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if(fd_ >= 0)
        {
            held_ = flock(fd_, LOCK_EX | LOCK_NB) == 0;
            return;
        }
        held_ = true;
#ifdef RICH_MPI
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        if(initialized)
            MPI_Finalized(&finalized);
        if(initialized && !finalized)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            held_ = rank == 0;
        }
#endif // RICH_MPI
    }

    ~SymbolizerLock()
    {
        if(fd_ >= 0)
        {
            if(held_)
                flock(fd_, LOCK_UN);
            close(fd_);
        }
    }

    SymbolizerLock(const SymbolizerLock&) = delete;
    SymbolizerLock& operator=(const SymbolizerLock&) = delete;

    bool held() const { return held_; }
    bool disabled() const { return disabled_; }

private:
    int fd_;
    bool held_;
    bool disabled_;
};

void printStackTrace(void * const *frames, int depth, std::ostream &os, const std::string &prefix)
{
    if(depth <= 0)
        return;

    std::string exe = getExecutablePath();
    const SymbolizerLock symbolizer;
    const bool symbolize = symbolizer.held() && !symbolizer.disabled();

    os << prefix << "Stack trace (" << depth << " frames)";
    if(symbolizer.disabled())
        os << ", not symbolized (RICH_SYMBOLIZE_STACK=0)";
    else if(!symbolize)
        os << ", not symbolized (another process on this node is symbolizing)";
    os << ":" << std::endl;

    // skip frames 0-1 (UniversalError ctor + backtrace)
    int start = 2;
    for(int i = start; i < depth; ++i)
    {
        SourceLocation loc = symbolize ? resolveAddress(exe, frames[i]) :
            describeAddress(exe, frames[i]);

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
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if(initialized)
      MPI_Finalized(&finalized);
    if(initialized && !finalized)
    {
      int rank = 0;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      prefix = "============" + to_string(rank) + "============ ";
    }
  #endif // RICH_MPI
  os.precision(14);
  os << prefix << eo.getErrorMessage() << std::endl;
  for_each(eo.fields_.begin(), eo.fields_.end(),
          [&os, &prefix](const pair<string, UniversalError::PrintableAny>& f) {os << prefix << f.first << " " << f.second << endl;});

  printStackTrace(eo.stack_frames_, eo.stack_depth_, os, prefix);
}
