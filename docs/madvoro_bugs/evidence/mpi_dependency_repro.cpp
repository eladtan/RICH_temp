// Bounded tests of the production dependency header; no production code edits.
// The compaction mode injects one API-permitted completion-index permutation.
// Other modes use real MPI operations and the unchanged production methods.
#include <bits/stdc++.h>
#include <mpi.h>
#include <boost/container/flat_map.hpp>
#include <boost/container/flat_set.hpp>
#include <mpi_utils/serialize/Serializer.hpp>
#define private public
#include <mpi_utils/BuffersManager.hpp>
#undef private
#include <mpi_utils/queryAgent/BuffersManagerQueryAgent.hpp>

static bool inject_compaction = false;
extern "C" int MPI_Testsome(int n, MPI_Request *req, int *out,
                            int *idx, MPI_Status *status) {
  if (inject_compaction) {
    inject_compaction = false;
    if (n != 3) std::abort();
    *out = 2; idx[0] = 2; idx[1] = 0;
    req[2] = req[0] = MPI_REQUEST_NULL;
    status[0] = {}; status[1] = {};
    return MPI_SUCCESS;
  }
  return PMPI_Testsome(n, req, out, idx, status);
}

struct Query : Serializable {
  std::uint64_t value = 123;
  size_t dump(Serializer *s) const override { return s->insert(value); }
  size_t load(const Serializer *s, size_t off) override { return s->extract(value, off); }
};
struct Talk : TalkAgent<Query> {
  RanksSet getTalkList(const Query &) const override { return {0}; }
};
struct Answer : AnswerAgent<Query, std::uint32_t> {
  std::vector<std::uint32_t> answer(const Query &q, int) override {
    return {static_cast<std::uint32_t>(q.value)};
  }
};
struct Blob : Serializable {
  std::vector<char> bytes;
  size_t dump(Serializer *s) const override { return s->insert(bytes); }
  size_t load(const Serializer *s, size_t off) override { return s->extract(bytes, off); }
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
  const std::string mode = argc > 1 ? argv[1] : "";
  int failed = 0;
  if (mode == "compaction" && size == 1) {
    BuffersManager<std::uint32_t> b(MPI_COMM_WORLD, [](auto, auto, auto){}, 303, 1024, 64, 2, 1);
    // Synthetic post-MPI bookkeeping state; no fabricated handle is passed to MPI.
    b.sendRequests.assign(3, MPI_REQUEST_NULL);
    b.sendBuffersByRequests = {{0,0}, {1,1}, {2,2}};
    b.activeSendRequests = 3;
    inject_compaction = true;
    try { b.CleanSendRequests(); }
    catch (const std::exception &e) {
      failed = 1;
      std::cout << "COMP_ACTION_DEFECT completion_indices=[2,0] exception=" << e.what()
                << " remaining_slots=" << b.sendRequests.size() << std::endl;
    }
    b.sendRequests.clear(); b.sendBuffersByRequests.clear(); b.activeSendRequests=0;
    b.Destroy();
  } else if (mode == "scalar8" && size == 2) {
    size_t callbacks = 0;
    BuffersManager<std::uint64_t> b(MPI_COMM_WORLD,
      [&](const auto *, size_t n, int) { callbacks += n; }, 304, 1024, 64, 2, 1);
    if (rank == 0) b.Add(1, std::uint64_t(123));
    for (int i = 0; i < 10000; ++i) b.HandleIncomingOutcoming();
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "SCALAR8 rank=" << rank << " callbacks=" << callbacks
              << " sent_messages=" << b.GetSentCounter()
              << " pending_buffers=" << b.GetPendingNumber() << std::endl;
    if (rank == 0) failed = b.GetSentCounter() == 0 && b.GetPendingNumber() == 1;
    if (rank == 1) failed = callbacks == 0;
    b.Destroy();
  } else if (mode == "self" && size == 1) {
    Talk talk; Answer answer;
    BuffersManagerQueryAgent<Query,std::uint32_t> agent(&talk, &answer, true, MPI_COMM_WORLD);
    const auto batch = agent.runBatch({Query{}});
    std::cout << "SELF per_query=" << batch.queriesAnswers.at(0).finalResults.size()
              << " by_rank=" << batch.dataByRanks.at(0).size()
              << " flattened=" << batch.result.size()
              << " recv_peers=" << agent.getRecvProc().size() << std::endl;
    failed = batch.queriesAnswers.at(0).finalResults.size() == 1 && batch.result.empty();
  } else if ((mode == "oversize" || mode == "overflowfit") && size == 2) {
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    size_t callbacks = 0, receivedBytes = 0;
    BuffersManager<Blob> b(MPI_COMM_WORLD,
      [&](const Blob *v, size_t n, int) {
        callbacks += n;
        for(size_t i=0;i<n;++i) receivedBytes += v[i].bytes.size();
      }, 305, 64, mode == "overflowfit" ? 64 : 32, 100, 1);
    const size_t expected = mode == "overflowfit" ? 80 : 200;
    if (rank == 0) {
      Blob blob; blob.bytes.assign(mode == "overflowfit" ? 40 : 200, 'x');
      b.Add(1, blob);
      if(mode == "overflowfit") b.Add(1, blob);
    }
    for (int i = 0; i < 10000; ++i) b.HandleIncomingOutcoming();
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "BUFFER_BOUNDS mode=" << mode << " rank=" << rank << " callbacks=" << callbacks
              << " received_payload_bytes=" << receivedBytes
              << " sent_payload_bytes=" << (rank == 0 ? expected : 0) << std::endl;
    if(rank == 1) failed = callbacks > 0 && receivedBytes != expected;
    b.Destroy();
  } else {
    if (rank == 0) std::cerr << "mode/rank mismatch\n";
    failed = 2;
  }
  int any = 0; MPI_Allreduce(&failed, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  // Nonzero means a defect was detected, not that the reproduction failed to run.
  return any;
}
