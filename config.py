#!/usr/bin/env python3
"""
RICH Build Configuration Script

Usage:
    ./config.py --problem=<name> [options]

Examples:
    ./config.py --problem=sedov2d_test
    ./config.py --problem=sedov2d_test --compiler=intel --build=debug
    ./config.py --problem=sedov2d_test --mpi --profile

Then: make -j
"""

import argparse
import sys
import subprocess
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).parent.absolute()
BUILD_DIR = ROOT_DIR / "build"

COMPILERS = {
    'gnu': {'c': 'gcc', 'cxx': 'g++', 'fortran': 'gfortran'},
    'intel': {'c': 'icx-cc', 'cxx': 'icx', 'fortran': 'ifx'}
}

def main():
    parser = argparse.ArgumentParser(description='Configure RICH build')
    parser.add_argument('--problem', required=True, help='Problem name (e.g., sedov2d_test, Marshak)')
    parser.add_argument('--compiler', choices=['gnu', 'intel'], default='gnu')
    parser.add_argument('--build', choices=['release', 'debug'], default='release')
    parser.add_argument('--mpi', action='store_true', help='Enable MPI')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--energy-groups', type=int, default=1, help='Number of energy groups for multigroup radiation')
    args = parser.parse_args()

    # Check problem directory exists
    problem_dir = ROOT_DIR / 'runs' / args.problem
    if not problem_dir.exists():
        print(f"Error: Problem directory 'runs/{args.problem}' not found")
        print(f"Available: {', '.join(sorted([d.name for d in (ROOT_DIR/'runs').iterdir() if d.is_dir() and not d.name.startswith('.')]))}")
        sys.exit(1)

    # Check compiler
    compiler = COMPILERS[args.compiler]
    if not shutil.which(compiler['cxx']):
        print(f"Error: Compiler '{compiler['cxx']}' not found in PATH")
        sys.exit(1)

    # Build CMake command
    cmake_cmd = [
        'cmake', '-S', str(ROOT_DIR), '-B', str(BUILD_DIR),
        f'-DCMAKE_BUILD_TYPE={args.build.capitalize()}',
        f'-DPROBLEM_NAME={args.problem}',
        f'-DCMAKE_C_COMPILER={compiler["c"]}',
        f'-DCMAKE_CXX_COMPILER={compiler["cxx"]}',
        f'-DCMAKE_Fortran_COMPILER={compiler["fortran"]}',
        f'-DENABLE_MPI={"ON" if args.mpi else "OFF"}',
        f'-DENABLE_PROFILING={"ON" if args.profile else "OFF"}',
        f'-DENERGY_GROUPS_NUM={args.energy_groups}',
    ]

    # Run CMake
    print(f"Configuring: {args.problem} ({args.compiler} {args.build})")
    try:
        subprocess.run(cmake_cmd, check=True)
    except subprocess.CalledProcessError:
        print("\nConfiguration failed!")
        sys.exit(1)

    # Create Makefile
    (ROOT_DIR / 'Makefile').write_text(f"""# Auto-generated - run ./config.py to reconfigure
all:
\t@$(MAKE) -C {BUILD_DIR}
clean:
\t@$(MAKE) -C {BUILD_DIR} clean
""")

    print(f"\nDone! Run: make -j")
    print(f"Executable: {BUILD_DIR}/rich")

if __name__ == '__main__':
    main()
