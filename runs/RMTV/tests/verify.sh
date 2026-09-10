#!/usr/bin/env bash
set -euo pipefail
rmtv_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rmtv_root=$(cd -- "$rmtv_dir/../.." && pwd)
rmtv_bin=${RMTV_BIN:-$rmtv_root/build/gnuRelease/RMTV/rich}
rmtv_work=$(mktemp -d "${TMPDIR:-/tmp}/rmtv_verify.XXXXXXXX")
cd "$rmtv_dir"
python3 reference/check.py
${CXX:-g++} -std=c++17 -O2 -x c++ tests/reference_test.cxx.in -o "$rmtv_work/reference_test"
"$rmtv_work/reference_test" reference/shape.dat > "$rmtv_work/quadrature.txt"
"$rmtv_bin" --n 8 --quadrature 16 --init-only --initial-photons 4 --output "$rmtv_work/initial" > "$rmtv_work/initial.log" 2>&1
python3 analyze.py "$rmtv_work/initial" --label initial --smoke
"$rmtv_bin" --n 8 --quadrature 16 --hydro-only --max-steps 4 --initial-photons 4 --output "$rmtv_work/hydro" > "$rmtv_work/hydro.log" 2>&1
python3 analyze.py "$rmtv_work/hydro" --smoke
for rmtv_method in imc ddmc; do
    rmtv_options=()
    if [[ "$rmtv_method" == imc ]]; then rmtv_options+=(--imc); fi
    "$rmtv_bin" --n 8 --quadrature 16 --max-steps 4 --dt-fraction 0.00001 \
      --initial-photons 8 --photons 8 --max-photons 16 --opacity-cap 1e8 \
      --output "$rmtv_work/$rmtv_method" "${rmtv_options[@]}" > "$rmtv_work/$rmtv_method.log" 2>&1
    python3 analyze.py "$rmtv_work/$rmtv_method" --smoke
done
python3 - "$rmtv_work" <<'PY'
from pathlib import Path
import sys,json,numpy as np
p=Path(sys.argv[1])
initial=np.genfromtxt(p/'initial/initial_rank0.csv',delimiter=',',names=True,skip_header=1)
for n in ['rho','T','p','e','vx','vy','vz']:
    assert np.allclose(initial[n],initial[n+'_ref'],rtol=1e-12,atol=1e-12),n
imc=np.genfromtxt(p/'imc/final_rank0.csv',delimiter=',',names=True,skip_header=1)
ddmc=np.genfromtxt(p/'ddmc/final_rank0.csv',delimiter=',',names=True,skip_header=1)
relative=np.sum(imc['volume']*abs(imc['T']-ddmc['T']))/np.sum(imc['volume']*imc['T'])
assert np.isfinite(relative)
dd=np.atleast_1d(np.genfromtxt(p/'ddmc/diagnostics.csv',delimiter=',',names=True))
assert np.sum(dd['ddmc_steps'])>0,'DDMC comparison never used acceleration'
print(json.dumps({'IMC_DDMC_short_interval_temperature_L1':float(relative)}))
PY
printf 'Verification artifacts: %s\n' "$rmtv_work"
