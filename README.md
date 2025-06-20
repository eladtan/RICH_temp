# huji-rich
RICH is an compressible hydrodynamic simulation on a moving mesh written in c++.
We've recently published papers explaining the [serial](http://iopscience.iop.org/0067-0049/216/2/35/) and 
[parallel](http://adsabs.harvard.edu/abs/2015ApJS..216...14S) versions of the code.

# Prerequisites
- A C++ compiler - GNU or Intel (Supporting C++ 17)
- Boost >= 1.74.0
- HDF5
- VTK
- Recommended: MPI - OpenMPI and IntelMPI were checked. \
Parallel HDF5 and VTK (built with the matching MPI library) are necessary.
- Recommended: UCX and UCC for performance boost on parallel applications.

# Installation

## Clone
Clone the latest version of RICH:
```shell
git clone --recursive https://gitlab.com/eladtan/RICH.git
```

Make sure the submodules are installed:
```shell
cd RICH
git submodule
```

# Compilation

## Setting up GNU compiler environment

### Modules
The list of recommended modules to run RICH:
```
boost/1.78.0
hdf5/1.14.2/gcc/12.3.0_cxx
vtk/9.3.0/gcc/12.3.0/with_mesa
openmpi/4.1.6/gcc/12.3.0
gcc/15.1.0
```

### Save Configuration
It is advised to save these module configuration via 

```shell
ml save rich
```

and then reload it on a new shell via:

```shell
ml restore rich
```

Saved `module` configurations can be find in:

```shell
ls ~/.lmod.d
```

## Setting up Intel compiler environment

### Gates 1, 2 (Linux 4.18.0)
Similarly, for `intel` compilation, use `ml purge` and then load these modules:
```
Intel/OneApi/2024.2.1
boost/1.78.0
hdf5/1.14.2/Intel/OneApi-2023.2.0_cxx
gcc/15.1.0
vtk/9.3.0/Intel/OneApi/2024.2.1/with_X
```

It is advised to save these module configuration via 

```shell
ml save rich_intel
```

and then reload it on a new shell via:

```shell
ml restore rich_intel
```

## Compiling a specific run (`main.cpp`) file

Once these modules which defines the correct compilation envoinronment are loaded, the code is compiled via the command:

```shell
./build_rich gnuReleaseMPI --test_name=sedov2d_test
```

where `sedov2d_test` represents the subdirectory `runs/sedov2d_test` which contains a `main.cpp` file which defines a specific simulation. Other runs can be compiled by adding another directory with a `main.cpp` file to `runs/`.

For other compilation configurations, replace `gnuReleaseMPI` with one of:
```shell
gnuReleaseMPI
gnuReleaseMPIProf
gnuReleaseProf
gnuRelease
gnuDebugMPI
gnuDebugMPIProf
gnuDebugProf
gnuDebug

intelReleaseMPI
intelReleaseMPIProf
intelRelease
intelReleaseProf
intelDebugMPI
intelDebugMPIProf
intelDebugProf
intelDebug
```


## Profiling

To run the `gprof` profiler (for compilation configs with `Prof`), after a simulation run is finished, a `gmon.out` file will be generated in the run directory. This file contains profiling information and can be processed into a nice PDF (`gprof.pdf`) via:

```shell
gprof RICH_EXE_PATH gmon.out | gprof2dot -s -w --show-samples | dot -Tpdf -o gprof.pdf
```

where `RICH_EXE_PATH` is a path to the `rich` binary that was used in this simulation.