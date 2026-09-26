#include <atomic>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sys/time.h>
#include <ucontext.h>
#include <unistd.h>
namespace {
constexpr unsigned capacity=4*1024*1024;
uintptr_t* samples=nullptr;
std::atomic<unsigned> count{0};
static_assert(std::atomic<unsigned>::is_always_lock_free);
void sample(int,siginfo_t*,void* context){
 unsigned i=count.fetch_add(1,std::memory_order_relaxed);
 if(i<capacity) samples[i]=static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_RIP];
}
char path[512];
}
__attribute__((constructor)) static void start_sampling(){
 const char* rank=std::getenv("OMPI_COMM_WORLD_RANK");
 if(!rank) return;
 const char* dir=std::getenv("STORM_SAMPLE_DIR"); if(!dir) return;
 const char* job=std::getenv("SLURM_JOB_ID");
 std::snprintf(path,sizeof(path),"%s/rank_%s",dir,rank);
 samples=new uintptr_t[capacity];
 struct sigaction action{};action.sa_sigaction=sample;action.sa_flags=SA_SIGINFO|SA_RESTART;sigemptyset(&action.sa_mask);sigaction(SIGPROF,&action,nullptr);
 itimerval timer{};timer.it_interval.tv_usec=10000;timer.it_value=timer.it_interval;setitimer(ITIMER_PROF,&timer,nullptr);
}
__attribute__((destructor)) static void stop_sampling(){
 if(!samples)return;
 itimerval timer{};setitimer(ITIMER_PROF,&timer,nullptr);
 std::ifstream maps("/proc/self/maps");std::ofstream mapout(std::string(path)+".maps");mapout<<maps.rdbuf();
 std::ofstream out(std::string(path)+".bin",std::ios::binary);
 unsigned n=std::min(count.load(),capacity);out.write(reinterpret_cast<char*>(samples),n*sizeof(uintptr_t));
}
