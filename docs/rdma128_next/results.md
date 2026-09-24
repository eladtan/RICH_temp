# O2 follow-up and P2P comparison

Job 10176477, 8 exclusive nodes × 16 ranks, same 120-cycle Crooked Pipe deck and strict O2 build. Total elapsed seconds include mpirun startup and teardown.

| Variant | Runs (s) | Median (s) |
|---|---|---|
| Prior optimized O2 RDMA | 68.376, 68.737, 67.894 | 68.376 |
| Additional geometry/RW-cache RDMA experiment | 67.871, 67.415, 68.341 | 67.871 |
| Same experimental CPU build, P2P | 78.327, 78.196, 77.802 | 78.196 |

The extra CPU changes reduced the median by only 0.74%, with overlapping ranges; reverted to the prior O2 source because this does not justify the added face arrays (24 bytes per directed face) and complexity. Experimental source is archived under build/rdma128_next/candidate_source and changes.patch. RDMA was 13.2% below P2P with the identical experimental binary. Both managers used fair scheduling.

Validation passed: 800,000 intersection/radius comparisons, 300,000 exact full random-walk attempts against the prior scalar implementation under ASan/UBSan, native OFI and MPI resize protocol tests (reused protocol_final binary), and 4-rank 24-cycle energy ledgers for both managers in release and STORM_DEBUG builds (jobs 10176477/10176478). These CPU changes did not modify the communication or reallocation protocol.
