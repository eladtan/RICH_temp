import logging
import os
import subprocess
import pathlib
import sys

# importing modules from this package
from .buildutils import run_make

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("build_program.main")

root_dir = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.append(root_dir)


def _run_cmake(*, build_dir, exe_name, config, SysLibsDict, test_dir, args=None, definitionOfReal=8):
    warning_flags = " -Wextra -Wshadow -Wunused-value -Wunused-variable -Wunused-function -Wunused-macros"
    common_cxx_flags = f" -std=c++17 {warning_flags} -fno-common -fstack-protector-all -rdynamic -g -DENERGY_GROUPS_NUM={int(args.energy_groups_num)}"
    common_cxx_flags_debug = " -DDEBUG -O0 -g3 -gdwarf-3 "
    common_cxx_flags_release = " -DNDEBUG -O3 -DOMPI_SKIP_MPICXX "
    
    hdf5_lib_dir = SysLibsDict["hdf5_lib_dir"]
    hdf5_include_dir = SysLibsDict["hdf5_include"]
    vtk_dir = SysLibsDict["vtk"]

    if config.startswith("gnu"):
        fortran_compiler = SysLibsDict["gfortran"]
        # removed -Werror from:
        cmake_fortran_flags = f" -fall-intrinsics -std=f2018 -fdec-static -finit-local-zero -finit-integer=-2147483647 -finit-real=snan -finit-logical=True -finit-derived -ffpe-trap=invalid,zero,overflow -ffree-line-length-none -cpp -fdefault-real-{definitionOfReal} {'-fdefault-double-8' if definitionOfReal==8 else ''} -fbacktrace -g -Wall -Wextra -Wsurprising  -Wpedantic -Wno-uninitialized "
        cmake_fortran_flags_debug = " -O0 -fcheck=all -Wno-maybe-uninitialized -Wno-tabs -Wno-conversion "
        cmake_fortran_flags_release = " -O2 "
        cmake_fortran_flags += "  -mcmodel=medium -shared-libgcc "

        c_compiler = SysLibsDict["gcc"]
        cxx_compiler = SysLibsDict["g++"]

        cmake_cxx_standard = "17"
        cmake_cxx_flags = " -Wdouble-promotion -fstrict-aliasing -Wno-deprecated-copy -Wno-double-promotion -Wno-shadow "
        cmake_cxx_flags_debug = " -D_GLIBCXX_DEBUG "
        cmake_cxx_flags_release = " "
    elif config.startswith("intel"):
        fortran_compiler = SysLibsDict["ifort"]
        cmake_fortran_flags = f" -init=arrays,zero,minus_huge,snan -fp-speculation=safe -r{definitionOfReal} -assume no2underscores -lstdc++ -lrt -traceback -fpe0 -gen-interfaces -warn all -warn errors "
        cmake_fortran_flags_debug = " -O0 -fp-model precise -fp-model source -fimf-arch-consistency=true -fp-stack-check -debug -check bounds -check format -check output_conversion -check pointers -check uninit -check stack -check shape "
        cmake_fortran_flags_release = " -O2 "
        cmake_fortran_flags += " -mcmodel=medium -shared-intel "

        c_compiler = SysLibsDict["icc"]
        cxx_compiler = SysLibsDict["icpc"]

        common_cxx_flags += " -diag-remark=13397,13401,15552,2196 -pedantic-errors -Wall "
        cmake_cxx_standard = "17"
        cmake_cxx_flags = " -ansi-alias -fimf-arch-consistency=true "
        cmake_cxx_flags_debug = " -fp-model consistent -diag-disable=openmp -Wno-unknown-pragmas "
        cmake_cxx_flags_release = " -fp-model precise -march=core-avx2 -qopenmp "
    else:
        logger.error(f"no known compiler was given at head of {config}")
    
    if "MPI" in config:
        mpi = "on"
        common_cxx_flags += " -DRICH_MPI "
        if "gnu" in config:
            c_compiler = SysLibsDict["mpicc_gcc"]
            cxx_compiler = SysLibsDict["mpic++_gcc"]
        else:
            c_compiler = SysLibsDict["mpicc_intel"]
            cxx_compiler = SysLibsDict["mpic++_intel"]
    else:
        mpi = "off"
    
    if "Release" in config:
        build_type = "Release"
    if "Debug" in config:
        build_type = "Debug"
    
    prof = "on" if "Prof" in config else "off"
    
    vcl_include_dir = SysLibsDict["vcl_include"] if "vcl_include" in SysLibsDict else os.path.join(root_dir, "source/opt/vcl")
    r3d_include_dir = SysLibsDict["r3d_include"] if "r3d_include" in SysLibsDict else os.path.join(root_dir, "source/opt/r3d/src")
    clipper_include_dir = SysLibsDict["clipper_include"] if "clipper_include" in SysLibsDict else os.path.join(root_dir, "source/opt/clipper")
    boost_dir = SysLibsDict["boost_dir"] if "boost_dir" in SysLibsDict else (os.environ["BOOST_DIR"] if "BOOST_DIR" in os.environ else "")

    cmd = ['cmake',
            f'-DMPI={mpi}',
            f'-DPROF={prof}',
            f'-DCMAKE_Fortran_COMPILER={fortran_compiler}',
            f'-DCMAKE_C_COMPILER={c_compiler}',
            f'-DCMAKE_CXX_COMPILER={cxx_compiler}',
            f'-DCMAKE_BUILD_TYPE={build_type}',
            f'-DCMAKE_Fortran_FLAGS={cmake_fortran_flags}',
            f'-DCMAKE_Fortran_FLAGS_DEBUG={cmake_fortran_flags_debug}',
            f'-DCMAKE_Fortran_FLAGS_RELEASE={cmake_fortran_flags_release}',
            f'-DCMAKE_CXX_STANDARD={cmake_cxx_standard}',
            f'-DCMAKE_CXX_FLAGS={common_cxx_flags + cmake_cxx_flags}',
            f'-DCMAKE_CXX_FLAGS_DEBUG={common_cxx_flags_debug + cmake_cxx_flags_debug}',
            f'-DCMAKE_CXX_FLAGS_RELEASE={common_cxx_flags_release + cmake_cxx_flags_release}',
            f'-DEXE_NAME={exe_name}',
            f'-DHDF5_LIB_DIRECTORY={hdf5_lib_dir}',
            f'-DHDF5_INCLUDE={hdf5_include_dir}',
            f'-DVTK_DIRECTORY={vtk_dir}',
            f'-DBOOST_DIR={boost_dir}' if boost_dir else "",
            f'-DJSONCPP_INCLUDE={SysLibsDict["jsoncpp_include"]}' if "jsoncpp_include" in SysLibsDict else "",
            f'-DJSONCPP_LIB_DIRECTORY={SysLibsDict["jsoncpp_lib_dir"]}' if "jsoncpp_lib_dir" in SysLibsDict else "",
            f'-DVTUNE_INCLUDE={SysLibsDict["vtune_include"]}' if "vtune_include" in SysLibsDict else "",
            f'-DVTUNE_LIB_DIRECTORY={SysLibsDict["vtune_lib_dir"]}' if "vtune_lib_dir" in SysLibsDict else "",
            f'-DVCL_INCLUDE={vcl_include_dir}' if vcl_include_dir else "",
            f'-DCLIPPER_INCLUDE={clipper_include_dir}' if clipper_include_dir else "",
            f'-DR3D_INCLUDE={r3d_include_dir}' if r3d_include_dir else "",
            f'-DCGAL_INCLUDE={SysLibsDict["cgal_include"] if "cgal_include" in SysLibsDict else ""}',
            f'-DPYBIND11={SysLibsDict["pybind11"]}' if "pybind11" in SysLibsDict else "",
            f'-DPYTHON_EXECUTABLE={SysLibsDict["python3"]}' if "python3" in SysLibsDict else "",
            f'-DTEST_DIR={test_dir}',
            '-DCMAKE_VERBOSE_MAKEFILE=on',
            f'-DPROJECT_ROOT_DIR={root_dir}',
            f'{build_dir}',
            f'{root_dir}/source']
    print(cmd)
    cmake_result = subprocess.run(cmd,
                                  stdout=open(os.path.join(build_dir, config+'_cmake.out'), 'w'),
                                  stderr=open(os.path.join(build_dir, config+'_cmake.err'), 'w'),
                                  #stdin=open('/dev/null'),
                                  cwd=os.path.join(build_dir, config))

    return cmake_result

def remove_unnecessary_files(src_dir):
    VCL_PATH = "opt/vcl"
    files = {os.path.join(src_dir, VCL_PATH, "dispatch_example1.cpp"), os.path.join(src_dir, VCL_PATH, "dispatch_example2.cpp")}

    for file_path in files:
        if os.path.exists(file_path):
            os.remove(file_path)
    
def build_program(*, configs, make_dir, src_dir, test_dir):
    """Build the program with the desired configurations."""
    exe_name = "rich"
    logger.debug(f"args:\nconfigs = {configs}\nroot_dir = {root_dir}\nmake_dir = {make_dir}\nsrc_dir = {src_dir}\ntest_dir = {test_dir}")
    assert os.path.isdir(os.path.join(root_dir, "source")), f"Directory {root_dir} does not contain a directory named source"

    with open(os.path.join(root_dir, "compilation", "SystemLibsLinks.py"), "r") as f:
        SysLibsDict = eval(f.read())
 

# Fix issue in r3d
    with open(os.path.join(root_dir,"source/opt/r3d/config/r3d-config.h.in"), "r") as fin:
        with open(os.path.join(root_dir,"source/opt/r3d/src/r3d-config.h"), "w") as fout:
            for line in fin:
                fout.write(line.replace('@R3D_MAX_VERTS@', '256'))
    
    for config in configs:
        logger.info(f"Building {config}")
        build_dir = os.path.join(make_dir, "build")
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        config_dir = os.path.join(build_dir, config)
        if not os.path.isdir(config_dir):
            os.makedirs(config_dir)
        
        #run cmake for the specific config
        logger.info("Running cmake")
        cmake = _run_cmake(build_dir=build_dir,
                            exe_name=exe_name,
                            config=config,
                            SysLibsDict=SysLibsDict,
                            test_dir=test_dir,
                            args=args)
        assert cmake.returncode == 0, f"Running cmake for {config} failed"

        short_exe_path = os.path.join(config_dir, exe_name)
        if os.path.islink(short_exe_path):
            os.remove(short_exe_path)   

        print(f"ENERGY_GROUPS_NUM = {args.energy_groups_num}")

        #run make   
        logger.info("Running make")
        make = run_make.main(make_dir, config)
        assert make.returncode == 0, f"Running make for {config} failed"
        
        exe_suffix = "_"
        exe_suffix += config

        exe_path_with_suffix = os.path.join(config_dir, exe_name + exe_suffix)
        os.rename(short_exe_path, exe_path_with_suffix)
        os.symlink(exe_path_with_suffix, short_exe_path)
    
