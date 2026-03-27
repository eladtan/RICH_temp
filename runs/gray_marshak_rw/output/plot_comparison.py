import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re, glob, sys
from pathlib import Path

plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#e6edf3',
    'text.color': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#30363d',
    'grid.alpha': 0.5,
    'font.family': 'monospace',
    'font.size': 10,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
})

RW_COLOR = '#58a6ff'
NORW_COLOR = '#f78166'
ACCENT = '#3fb950'
outdir = Path(__file__).parent

def load_profile(fname):
    x, T = [], []
    with open(outdir / fname) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            x.append(float(parts[0]))
            T.append(float(parts[1]))
    return np.array(x), np.array(T)

def parse_log(logfile):
    rw_elapsed, norw_elapsed = [], []
    rw_steps, norw_steps = [], []
    rw_rwsteps = []
    in_rw = True
    wall_time_count = 0
    with open(logfile) as f:
        for line in f:
            if '=== RW OFF ===' in line:
                in_rw = False
            if 'Total wall time:' in line:
                wall_time_count += 1
                if wall_time_count == 1:
                    in_rw = False
            m = re.search(r'Elapsed: ([0-9.e+-]+) seconds', line)
            if m:
                val = float(m.group(1))
                if in_rw:
                    rw_elapsed.append(val)
                else:
                    norw_elapsed.append(val)
            m = re.search(r'Total steps: (\d+)', line)
            if m:
                val = int(m.group(1))
                if in_rw:
                    rw_steps.append(val)
                else:
                    norw_steps.append(val)
            m = re.search(r'RW steps: (\d+)', line)
            if m:
                rw_rwsteps.append(int(m.group(1)))
    return (np.array(rw_elapsed), np.array(norw_elapsed),
            np.array(rw_steps, dtype=float), np.array(norw_steps, dtype=float),
            np.array(rw_rwsteps, dtype=float))

keV_K = 11604.5

logfiles = list(outdir.parent.glob('gray_marshak_*.out'))
if logfiles:
    logfile = max(logfiles, key=lambda p: p.stat().st_mtime)
else:
    logfile = outdir.parent / 'rw_debug.out'
    if not logfile.exists():
        print("No log file found"); sys.exit(1)
print(f"Using log: {logfile}")

rw_elapsed, norw_elapsed, rw_steps, norw_steps, rw_rwsteps = parse_log(logfile)

rw_final = 'gray_rw_final.txt'
norw_final = 'gray_norw_final.txt'
if not (outdir / rw_final).exists() or not (outdir / norw_final).exists():
    print("Final output files not found"); sys.exit(1)

rw_snaps = list(outdir.glob('gray_rw_0*.txt'))
norw_snaps = list(outdir.glob('gray_norw_0*.txt'))

snap_pairs = []
if rw_snaps and norw_snaps:
    latest_rw = max(rw_snaps, key=lambda p: p.stat().st_mtime)
    latest_norw = max(norw_snaps, key=lambda p: p.stat().st_mtime)
    snap_pairs.append((latest_rw.name, latest_norw.name, 'latest snapshot'))
snap_pairs.append((rw_final, norw_final, 'final'))

fig = plt.figure(figsize=(16, 10))
fig.suptitle('Gray Marshak Wave — RW vs Standard IMC', fontsize=15, fontweight='bold', y=0.98)

if len(rw_elapsed) > 1 and len(norw_elapsed) > 1:
    avg_rw = rw_elapsed[1:].mean()
    avg_norw = norw_elapsed[1:].mean()
    speedup = avg_norw / avg_rw if avg_rw > 0 else 0
    info = (f'RW avg cycle: {avg_rw:.3f}s   |   No-RW avg cycle: {avg_norw:.2f}s   |   '
            f'Speedup: {speedup:.1f}x   |   RW total: {rw_elapsed.sum():.1f}s   |   No-RW total: {norw_elapsed.sum():.1f}s')
    fig.text(0.5, 0.945, info, ha='center', fontsize=10, color='#8b949e')

nplots = len(snap_pairs) + 3
ncols = 2
nrows = (nplots + 1) // 2

for idx, (rw_file, norw_file, title) in enumerate(snap_pairs):
    ax = fig.add_subplot(nrows, ncols, idx + 1)
    xr, Tr = load_profile(rw_file)
    xn, Tn = load_profile(norw_file)
    ax.semilogy(xr, Tr / keV_K, color=RW_COLOR, lw=1.8, label='RW')
    ax.semilogy(xn, Tn / keV_K, color=NORW_COLOR, lw=1.8, ls='--', label='No RW')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('T (keV)')
    ax.set_title(f'Temperature — {title}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True)

base = len(snap_pairs)
ax_diff = fig.add_subplot(nrows, ncols, base + 1)
xr, Tr = load_profile(rw_final)
xn, Tn = load_profile(norw_final)
rel_diff = np.where(Tn > 0, (Tr - Tn) / Tn * 100, 0)
ax_diff.plot(xr, rel_diff, color=ACCENT, lw=1.2)
ax_diff.axhline(0, color='#8b949e', ls=':', lw=0.8)
ax_diff.set_xlabel('x (cm)')
ax_diff.set_ylabel('Relative diff (%)')
ax_diff.set_title('Temperature Difference (final)', fontsize=11, fontweight='bold')
ax_diff.set_ylim(-50, 50)
ax_diff.grid(True)

if len(rw_elapsed) > 0 and len(norw_elapsed) > 0:
    ax_time = fig.add_subplot(nrows, ncols, base + 2)
    ax_time.semilogy(np.arange(1, len(rw_elapsed)+1), rw_elapsed, color=RW_COLOR, lw=1.5, label='RW')
    ax_time.semilogy(np.arange(1, len(norw_elapsed)+1), norw_elapsed, color=NORW_COLOR, lw=1.5, label='No RW')
    ax_time.set_xlabel('Cycle')
    ax_time.set_ylabel('Elapsed (s)')
    ax_time.set_title('Per-Cycle Elapsed Time', fontsize=11, fontweight='bold')
    ax_time.legend(fontsize=9)
    ax_time.grid(True)

if len(rw_steps) > 0 and len(norw_steps) > 0:
    ax_steps = fig.add_subplot(nrows, ncols, base + 3)
    ax_steps.plot(np.arange(1, len(rw_steps)+1), rw_steps / 1e6, color=RW_COLOR, lw=1.5, label='RW')
    ax_steps.plot(np.arange(1, len(norw_steps)+1), norw_steps / 1e6, color=NORW_COLOR, lw=1.5, label='No RW')
    ax_steps.set_xlabel('Cycle')
    ax_steps.set_ylabel('Total Steps (M)')
    ax_steps.set_title('Per-Cycle Total Steps', fontsize=11, fontweight='bold')
    ax_steps.legend(fontsize=9)
    ax_steps.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = outdir / 'rw_comparison.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved {out_path}')
print(f'RW cycles: {len(rw_elapsed)}, No-RW cycles: {len(norw_elapsed)}')
if len(rel_diff) > 0:
    print(f'Max |rel_diff|: {np.max(np.abs(rel_diff)):.1f}%')
