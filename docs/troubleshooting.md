# Troubleshooting

## Build Issues

### Error: "Required config (-DCONFIG=...)"

**Cause:** The `CONFIG` variable was not passed to CMake.

**Solution:** Always use `build_rich.sh` instead of calling CMake directly:

```bash
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d
```

### Error: "Could not find Boost"

**Cause:** Boost is not in `LD_LIBRARY_PATH` or `BOOST_ROOT`.

**Solution:** Load the Boost module:

```bash
ml boost/1.78.0
```

Or set the path manually:

```bash
export BOOST_ROOT=/path/to/boost
```

### Error: "Could not find HDF5"

**Cause:** HDF5 libraries not found.

**Solution:** Load the HDF5 module and ensure it is in `LD_LIBRARY_PATH`:

```bash
ml hdf5/1.14.2/gcc/12.3.0_cxx
echo $LD_LIBRARY_PATH  # verify HDF5 path is present
```

### Error: "Could not find VTK"

**Cause:** VTK libraries not found.

**Solution:** Load the VTK module:

```bash
ml vtk/9.3.0/gcc/12.3.0/with_mesa
```

### Linker error: undefined reference to MPI symbols

**Cause:** Building with an MPI config but MPI is not loaded.

**Solution:** Load MPI before building:

```bash
ml openmpi/4.1.6/gcc/12.3.0
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d
```

### Build fails after changing source files

**Cause:** CMake cache may be stale.

**Solution:** The script normally detects this, but you can force a clean rebuild:

```bash
rm -rf build/gnuReleaseMPI/
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d
```

### "No rule to make target" errors

**Cause:** Source files were added or removed but CMake was not re-run.

**Solution:** `build_rich.sh` should detect this automatically. If not, delete the build directory and rebuild.

## Runtime Issues

### Segmentation fault / core dumped

**Possible causes:**
- Array out-of-bounds access (often from degenerate mesh cells)
- Invalid mesh geometry (overlapping points, points outside domain)
- Memory exhaustion

**Debugging steps:**
1. Build with Debug config: `./build_rich.sh gnuDebug --test_name=my_run`
2. Build with AddressSanitizer: `./build_rich.sh gnuDebug --test_name=my_run --with_asan`
3. Run under GDB: `gdb ./build/gnuDebug/rich`
4. Check if `UniversalError` was thrown (look for the error report in stdout)

### UniversalError exception

RICH throws `UniversalError` with diagnostic context when an error is detected. The error report includes:

- The error message
- Variable names and values at the point of failure
- Cell indices, coordinates, and field values

Check stdout for `reportError` output to identify the failing cell and condition.

### NaN or Inf values in output

**Possible causes:**
- Negative pressure or density (unphysical state)
- Division by zero in EOS calculations
- Timestep too large
- Degenerate Voronoi cells (zero volume)

**Solutions:**
1. Reduce the CFL number (e.g., from 0.3 to 0.1)
2. Check initial conditions for physical consistency
3. Enable Debug assertions to catch the problem early
4. Verify the EOS is appropriate for your pressure/density range

### Simulation runs very slowly

**Possible causes:**
- Debug configuration (use Release for production)
- Too many cells per MPI rank (increase ranks)
- Too few cells per MPI rank (communication overhead)
- AMR producing too many cells
- Small timestep forced by a few problematic cells

**Solutions:**
1. Use Release config: `gnuReleaseMPI` or `intelReleaseMPI`
2. Profile with a `Prof` config: `gnuReleaseMPIProf`
3. Check the timestep: if dt is very small, look for cells with extreme conditions

### Simulation hangs (no output, no crash)

**Possible causes:**
- MPI deadlock (usually from mismatched send/receive)
- SLURM job waiting in queue
- I/O hang on NFS

**Solutions:**
1. Check SLURM queue: `squeue -u $USER`
2. Check if the process is alive: `sstat <jobid>` or `top`
3. Kill and restart: `scancel <jobid>`

## MPI Issues

### "MPI_ABORT was invoked on rank 0"

**Cause:** One rank encountered a fatal error and called `MPI_Abort`.

**Solution:** Check stderr for the error message from the failing rank. Common causes include file I/O errors and assertion failures.

### Different results in serial vs MPI

Small differences (< 1e-10 relative) are expected due to floating-point operation ordering. Larger differences may indicate:

- Load balancing issues affecting mesh construction order
- Race conditions in parallel I/O
- Uninitialized variables that happen to have different values per rank

The `amr_random` regression test specifically checks for conservation drift in MPI mode (threshold: 1e-6).

### SLURM job fails immediately

**Check:**
1. Is the partition correct? `sinfo -p bigrun`
2. Are enough nodes available? `sinfo -N`
3. Is the binary executable? `ls -la build/gnuReleaseMPI/rich`
4. Are modules loaded? Check `$LD_LIBRARY_PATH` in the SLURM script

## Regression Test Issues

### "missing or stale sod_profile.txt"

**Cause:** The simulation did not produce the expected output file, or the file timestamp is before the suite start time (NFS cache issue).

**Solutions:**
1. Run with `--verbose` to see simulation output
2. Check `run.stderr.log` for errors
3. For NFS issues, the checker retries 3 times with 2-second pauses

### "fatal marker found in stdout/stderr"

**Cause:** The simulation crashed or encountered an error. The checker scans for keywords like "Segmentation fault", "UniversalError", "Aborted", etc.

**Solution:** Read the full stdout/stderr logs to find the actual error.

### Python checker fails: "ModuleNotFoundError: numpy"

**Cause:** Python packages not installed.

**Solution:**

```bash
pip3 install numpy matplotlib scipy h5py
```

### All tests pass but artifacts are deleted

**By design:** Artifacts are cleaned when all tests pass. Use `--keep-artifacts` to retain them:

```bash
./regression_tests/run_all.sh --keep-artifacts
```
