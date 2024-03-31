# huji-rich
RICH is an compressible hydrodynamic simulation on a moving mesh written in c++.
We've recently published papers explaining the [serial](http://iopscience.iop.org/0067-0049/216/2/35/) and 
[parallel](http://adsabs.harvard.edu/abs/2015ApJS..216...14S) versions of the code.

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

# compilation on ICPL (as of 31/03/2024)

## Setting up GNU compiler environment

### Gates 1, 2 (Linux 4.18.0)
For `gnu` compilation, use `ml purge` and then load these modules:
```
  1) git/2.33.0     4) cmake/3.20.2     7) openmpi/4.1.4/gcc/12.2.0
  2) python/3.9.7   5) pybind11/2.6.1   8) hdf5/1.12.2/gcc/12.2.0_cxx
  3) boost/1.74.0   6) gcc/12.2.0       9) vtk/9.2.0/gcc/12.2.0/with_X
```

### Gates 3, 4 (Linux 5.14.0)
For `gnu` compilation, use `ml purge` and then load these modules:
```
  1) gcc/12.3.0     4) hdf5/1.14.2/gcc/12.3.0_cxx     7) pybind11/2.11.1
  2) python3/3.9.18   5) openmpi/4.1.4/gcc/12.2.0   
  3) boost/1.78.0   6) vtk/9.3.0/gcc/12.3.0/with_X      
```

### Save Configuration
It is advised to save these module configuration via 

```shell
ml save rich_gcc12
```

and then reload it on a new shell via:

```shell
ml restore rich_gcc12
```

Saved `module` configurations can be find in:

```shell
ls ~/.lmod.d
```

## Setting up intel compiler environment

### Gates 1, 2 (Linux 4.18.0)
Similarly, for `intel` compilation, use `ml purge` and then load these modules:
```
  1) git/2.33.0             8) gcc/12.2.0
  2) python/3.9.7           9) Intel/OneApi/2022.3.0/compiler/2022.2.0
  3) boost/1.74.0          10) Intel/OneApi-2022.3.0
  4) cmake/3.20.2          11) openmpi/4.1.4/Intel/OneApi-2022.3.0
  5) pybind11/2.6.1        12) hdf5/1.12.2/Intel/OneApi-2022.3.0_cxx
  6) tbb/2021.7.0          13) vtk/9.2.0/gcc/12.2.0/with_X
```

It is advised to save these module configuration via 

```shell
ml save rich_intel2022_3
```

and then reload it on a new shell via:

```shell
ml restore rich_intel2022_3
```

## Compiling a specific run (`main.cpp`) file

Once these modules which defines the correct compilation envoinronment are loaded, the code is compiled via the command:

```shell
python3 -m compilation gnuReleaseMPI --test_name=sedov2d_test
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