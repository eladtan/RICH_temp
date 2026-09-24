# Review for the sub-55-second investigation

The stable starting point is the strict O2 binary recorded in docs/rdma128_o2. No fast-math, reassociation, or floating-point contraction has been enabled.

## Fixed-epoch progress shortcut

Only RDMACommunicationEngine changes in the communication stack. During fixedTransportActive, Progress calls provider progress and returns; UsesAsyncReallocation returns false and ProgressReallocations returns without polling resize requests.

Safety argument:

- BeginTransport rejects pending outgoing buffers and asynchronous reallocations; its two collective barriers close the previous prepare/resize phase and finish metadata snapshots before producers may post.
- RankHandler2 prohibits all local, remote, reset and allocation-changing operations during this epoch. Every participant uses the same fixed-queue configuration. No peer can legitimately request a mid-epoch resize.
- Provider completion progress remains active. Periodic outer-loop AmountManager progress and full completion verification remain unchanged.
- TransferParticlePrefix retains payload writes, QuiesceTarget, flushed tail publication, and synchronous completion before source-buffer consumption or deregistration.
- EndTransport checks pending state and synchronizes every rank before clearing the allocation lock. The ordinary resizing/polling path resumes after the flag is cleared. Legacy/non-fixed transport takes its previous path.
- No RMA backend, registration, remote key, receive-credit, wraparound, or queue-growth code changed in this investigation.

Validation: Slurm 10176482 includes new release and STORM_DEBUG 4-rank/24-cycle runs over both OFI and MPI RMA, with 7-entry initial rings, a 64-entry limit, four-event slices, and 1-microsecond send age. Native protocol unit tests from the preceding round remain applicable because RankHandler2 and the RMA backend source are unchanged.

## Numerical experiment: direct log

The optional STORM_CPU_DIRECT_LOG trial replaces -log1p(u-1) with -log(u) in the shared CPU IMC event kernel. It does not alter RNG consumption. CounterRNG::unitOpen returns (2*m+1)*2^-53, for 0 <= m < 2^52. Both the subtraction u-1 and its inverse are exact on this lattice. Thus both expressions specify the same exponential variate mathematically. The two libm entry points can return different last bits, so this is not a bitwise-preserving optimization.

On the login host, 1,000,008 RNG-lattice checks including endpoints found a maximum difference of one ULP (2.22044e-16 relative). A 300,000-event ASan/UBSan differential test preserved event choices and RNG counters; maximum normalized floating state/tally difference was 4.44089e-16. The native job reruns both tests with the compute-node math library. Full-history bitwise identity is not claimed.

## Experiments reverted

- The SoA face arrays, vector random-walk radius, and cached table logs from rdma128_next produced only a 0.74% median reduction with overlapping ranges. Restored their exact prior O2 source; archived the experiment.
- Early random-walk rejection preserved 300,000 complete random-walk attempts exactly; another 300,000 attempts included points on/near/outside a face. Both passed ASan/UBSan. It regressed the production trial and was reverted.
- Measured per-cell physics-cost sampling and partition weights were kept separate from the RDMA protocol. Review corrected zero-cell collective participation and per-step sample reset before the native cost trial. More frequent repartitioning increased mesh overhead; this experiment was archived and removed from the working source.

These checks cover the tested CPU case and backends. GPU and other native RDM providers have not been validated in this investigation.
