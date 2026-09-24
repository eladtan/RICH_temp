#!/bin/bash
set -euo pipefail
cd /home/maorm/RICH
python3 docs/branson_cp128/prepare.py
cmake -S build/branson_cp128/source/src -B build/branson_cp128/branson_o2 \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_C_COMPILER=mpicc \
 -DMETIS_ROOT_DIR=/home/maorm/branson/metis-install -DN_GROUPS=1 \
 -DUSE_GPU=OFF -DUSE_OPENMP=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/branson_cp128/branson_o2 --target BRANSON -j 4
