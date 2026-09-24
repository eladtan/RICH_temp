#include <mpi.h>
#ifdef FORCE_SYNCHRONOUS
#define MPI_Isend MPI_Issend
#endif
#include "gpu/DeviceParticle.hpp"
#include "manager/communication/SerialCommunicationEngine.hpp"
#include <iostream>
struct Engine:STORM::SerialCommunicationEngine<STORM::gpu::DeviceVec3> {
 STORM::rank_t Rank() const override { int r; MPI_Comm_rank(MPI_COMM_WORLD,&r);return r; }
 STORM::rank_t Size() const override {return 2;}
 MPI_Comm Communicator() const override {return MPI_COMM_WORLD;}
};
int main(int argc,char**argv) {
 MPI_Init(&argc,&argv);
 { Engine e; unsigned long long values[6]={1,2,3,4,5,6};
   if(e.Rank()==0)e.PublishCounters(values,6,5);
   MPI_Barrier(MPI_COMM_WORLD);
   if(e.Rank()==1)e.PublishCounters(values,6,5);
   MPI_Barrier(MPI_COMM_WORLD);
   if(argc>1 && e.Rank()==0) {unsigned long long receive[6]; MPI_Recv(receive,6,MPI_UNSIGNED_LONG_LONG,1,9941,MPI_COMM_WORLD,MPI_STATUS_IGNORE);}
   std::cerr<<"rank "<<e.Rank()<<" entering FinishCounters\n";
   e.FinishCounters();
   std::cerr<<"rank "<<e.Rank()<<" finished\n";
   MPI_Barrier(MPI_COMM_WORLD);
   if(e.Rank()==0 && argc==1) {MPI_Status s; MPI_Probe(1,9941,MPI_COMM_WORLD,&s); std::cout<<"unreceived snapshot remains after FinishCounters\n";}
 }
 MPI_Finalize();
}
