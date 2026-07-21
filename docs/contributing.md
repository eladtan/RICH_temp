# Contributing to RICH

## Code Style

### C++

- **Standard:** C++17
- **Naming:** CamelCase for classes (`HDSim3D`, `Voronoi3D`), snake_case or camelCase for functions and variables
- **Headers:** Include what you use; avoid heavy includes in headers
- **Const correctness:** Use `const` for references that should not be modified
- **Error handling:** Use `UniversalError` with diagnostic context for failures
- **Performance:** Avoid unnecessary allocations and copies in hot loops; pass large objects by const reference

### General Principles

- Validate inputs at function boundaries with actionable error messages
- Prefer deterministic behavior in numerical code
- Avoid silent fallbacks that mask data issues
- Keep changes minimal and localized

```cpp
// DO: explicit precondition with clear error
if (cells.empty()) {
    throw std::runtime_error("Cells list is empty in remap step");
}

// DON'T: silent fallback hiding bugs
if (cells.empty()) {
    return;
}
```

```cpp
// DO: pass large objects by const reference
double computeMass(const std::vector<Cell>& cells);

// DON'T: pass large containers by value
double computeMass(std::vector<Cell> cells);
```

### Bash Scripts

- Quote variable expansions: `"${VAR}"` not `$VAR`
- Validate arguments and fail with clear stderr messages
- Check required files/directories before use
- Use `set -u` (or `set -eu`) for strict mode

## Repository Structure

- **`source/`** -- Main C++ source code
- **`runs/`** -- Per-simulation entry points (`main.cpp` or `test.cpp`)
- **`regression_tests/`** -- Regression testing framework
- **`config/`** -- CMake configuration modules
- **`analytic/`** -- Analytical solutions (Python)
- **`data/`** -- Reference data (EOS tables, Lane-Emden, opacity)
- **`visualisation/`** -- Visualization scripts (Python, MATLAB)

## Adding a New Simulation

1. Create a directory under `runs/your_simulation/`
2. Write `main.cpp` following the pattern in [Simulation Setup](user-guide/simulation-setup.md)
3. Build and test:
   ```bash
   ./build_rich.sh gnuReleaseMPI --test_name=your_simulation
   ```

## Adding a New Regression Test

1. Create `regression_tests/cases/your_test/test.cpp`
2. Create `regression_tests/tests/your_test.sh` with test metadata
3. Add a `check_your_test_case` function in `regression_tests/lib/regression_checks.sh`
4. Run and verify:
   ```bash
   ./regression_tests/run_all.sh --test your_test --config gnuRelease --verbose
   ```

See [Regression Tests Overview](regression-tests/overview.md) for detailed instructions.

## Adding a New Physics Module

1. Identify the appropriate abstract interface:
   - `EquationOfState` for a new EOS
   - `SourceTerm3D` / `Acceleration3D` for external forces
   - `PointMotion3D` for mesh motion strategies
   - `CellUpdater3D` for primitive variable updates
   - `RadiationDriver` for radiation transport methods

2. Implement the interface in a new `.hpp`/`.cpp` file under the appropriate `source/` subdirectory

3. Add a regression test that validates the new physics

4. Update documentation in `docs/user-guide/`

## Build and Test Workflow

Before submitting changes:

1. **Build successfully** with at least one config:
   ```bash
   ./build_rich.sh gnuReleaseMPI --test_name=sedov_3d
   ```

2. **Run relevant regression tests:**
   ```bash
   ./regression_tests/run_all.sh --mode serial --verbose
   ```

3. **For physics changes**, run the full regression suite:
   ```bash
   ./regression_tests/run_all.sh --mode serial_then_mpi --verbose
   ```

4. **Report** which tests were run and their results

## Documentation

When changing user-facing behavior (build flags, test options, run workflows):

- Update `README.md`
- Update relevant docs under `docs/`
- Keep commands copy-pasteable and aligned with `build_rich.sh` and `run_all.sh`
- Prefer concrete examples over abstract descriptions
- If behavior differs by environment (GNU vs Intel, serial vs MPI), state the distinction

## Git Workflow

- Work on a feature branch: `git checkout -b feature/my-feature`
- Keep commits focused and well-described
- Test before pushing
- Create a merge request on GitLab for review
