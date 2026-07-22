# RICH regression tests

RICH uses the THUNDER submodule as its regression runner:

```bash
./regression_tests/run_all.sh --list-tests
./regression_tests/run_all.sh --mode serial
./regression_tests/run_all.sh --with-mpi
./regression_tests/run_all.sh --test sod_1d
./regression_tests/run_all.sh --dry-run
```

Tests are discovered automatically. Any directory below the roots in
[`config.json`](config.json) containing a `REGRESSION_INFO` file is a test.
RICH cases live below `regression_tests/cases`. STORM tests are discovered
recursively through the subproject configuration referenced by
[`config.json`](config.json), so RICH does not duplicate STORM's test roots,
build commands, or check-library settings.

The metadata file is the only test registration required. Project-specific
build mappings, profiles, test roots, and check libraries are defined in
[`config.json`](config.json). The common runner implementation and its
configuration schema live in the
[`THUNDER`](THUNDER) submodule.

