#!/usr/bin/env bash
set -euo pipefail
rmtv_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rmtv_root=$(cd -- "$rmtv_dir/../.." && pwd)
rmtv_serial=${RMTV_BIN:-$rmtv_root/build/gnuRelease/RMTV/rich}
rmtv_mpi=${RMTV_MPI_BIN:-$rmtv_root/build/gnuReleaseMPI/RMTV/rich}
rmtv_work=$(mktemp -d "${TMPDIR:-/tmp}/rmtv_mpi_verify.XXXXXXXX")
cd "$rmtv_dir"
"$rmtv_serial" --octant --n 8 --quadrature 8 --initial-photons 4 --hydro-only \
    --max-steps 4 --output "$rmtv_work/serial" > "$rmtv_work/serial.log" 2>&1
${MPIEXEC:-mpirun} -np 2 "$rmtv_mpi" --octant --n 8 --quadrature 8 --initial-photons 4 \
    --hydro-only --max-steps 4 --output "$rmtv_work/mpi" > "$rmtv_work/mpi.log" 2>&1
${MPIEXEC:-mpirun} -np 2 "$rmtv_mpi" --octant --n 8 --quadrature 8 --initial-photons 4 \
    --photons 4 --max-photons 8 --max-steps 4 --output "$rmtv_work/coupled" > "$rmtv_work/coupled.log" 2>&1
python3 analyze.py "$rmtv_work/coupled" --smoke
python3 - "$rmtv_work" <<'PY'
from pathlib import Path
import sys,json,numpy as np
root=Path(sys.argv[1])
def read(name,label):
    arrays=[np.atleast_1d(np.genfromtxt(p,delimiter=',',names=True,skip_header=1))
            for p in sorted((root/name).glob(label+'_rank*.csv'))]
    a=np.concatenate(arrays)
    return {tuple(np.round([row[k] for k in ['x','y','z']],12)):row for row in a}
results={}
for label in ['initial','final']:
    a=read('serial',label);b=read('mpi',label);assert a.keys()==b.keys()
    errors={}
    for f in ['rho','T','p','e','vx','vy','vz']:
        aa=np.array([a[k][f] for k in a]);bb=np.array([b[k][f] for k in a])
        errors[f]=float(max(abs(aa-bb))/max(max(abs(aa)),1e-300))
    assert max(errors.values())<1e-10,errors
    results[label]=errors
print(json.dumps(results,indent=2))
(root/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
PY
printf 'MPI verification artifacts: %s\n' "$rmtv_work"
