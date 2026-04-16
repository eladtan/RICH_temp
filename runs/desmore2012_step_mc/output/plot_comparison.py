import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
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
    """Parse elapsed times, total steps, and RW steps from the log file."""
    rw_elapsed, norw_elapsed = [], []
    rw_steps, norw_steps = [], []
    rw_rwsteps = []
    in_rw = True
    with open(logfile) as f:
        for line in f:
            if '=== RW OFF ===' in line:
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

logfile = outdir.parent / 'desmore_mc_9804763.out'
rw_elapsed, norw_elapsed, rw_steps, norw_steps, rw_rwsteps = parse_log(logfile)

snap_pairs = [
    ('desmore_step_mc_rw_00020.txt', 'desmore_step_mc_norw_00020.txt', 't = 0.1 ns'),
    ('desmore_step_mc_rw_00100.txt', 'desmore_step_mc_norw_00100.txt', 't = 0.5 ns'),
    ('desmore_step_mc_rw_final.txt',  'desmore_step_mc_norw_final.txt',  't = 1.0 ns (final)'),
]

fig = plt.figure(figsize=(16, 14))
fig.suptitle('Random Walk (fixed series) vs Standard IMC  —  Densmore 2012',
             fontsize=15, fontweight='bold', y=0.98)

avg_rw = rw_elapsed[1:].mean()
avg_norw = norw_elapsed[1:].mean()
speedup = avg_norw / avg_rw
total_rw = rw_elapsed.sum()
total_norw = norw_elapsed.sum()

info = (f'RW avg cycle: {avg_rw:.3f}s   |   No-RW avg cycle: {avg_norw:.2f}s   |   '
        f'Speedup: {speedup:.1f}x   |   RW total: {total_rw:.1f}s   |   No-RW total: {total_norw:.1f}s')
fig.text(0.5, 0.945, info, ha='center', fontsize=10, color='#8b949e')

for idx, (rw_file, norw_file, title) in enumerate(snap_pairs):
    ax = fig.add_subplot(3, 2, idx + 1)
    xr, Tr = load_profile(rw_file)
    xn, Tn = load_profile(norw_file)
    ax.semilogy(xr, Tr / keV_K, color=RW_COLOR, lw=1.8, label='RW')
    ax.semilogy(xn, Tn / keV_K, color=NORW_COLOR, lw=1.8, ls='--', label='No RW')
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('T (keV)')
    ax.set_title(f'Temperature — {title}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True)

ax4 = fig.add_subplot(3, 2, 4)
xr, Tr = load_profile('desmore_step_mc_rw_final.txt')
xn, Tn = load_profile('desmore_step_mc_norw_final.txt')
rel_diff = np.where(Tn > 0, (Tr - Tn) / Tn * 100, 0)
ax4.plot(xr, rel_diff, color=ACCENT, lw=1.2)
ax4.axhline(0, color='#8b949e', ls=':', lw=0.8)
ax4.set_xlabel('x (cm)')
ax4.set_ylabel('Relative diff (%)')
ax4.set_title('Temperature Difference (RW − NoRW) at t = 1.0 ns', fontsize=11, fontweight='bold')
ax4.set_ylim(-50, 50)
ax4.grid(True)

ax5 = fig.add_subplot(3, 2, 5)
ax5.semilogy(np.arange(1, len(rw_elapsed)+1), rw_elapsed, color=RW_COLOR, lw=1.5, label='RW')
ax5.semilogy(np.arange(1, len(norw_elapsed)+1), norw_elapsed, color=NORW_COLOR, lw=1.5, label='No RW')
ax5.set_xlabel('Cycle')
ax5.set_ylabel('Elapsed (s)')
ax5.set_title('Per-Cycle Elapsed Time', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True)

ax6 = fig.add_subplot(3, 2, 6)
ax6.plot(np.arange(1, len(rw_steps)+1), rw_steps / 1e6, color=RW_COLOR, lw=1.5, label='RW')
ax6.plot(np.arange(1, len(norw_steps)+1), norw_steps / 1e6, color=NORW_COLOR, lw=1.5, label='No RW')
ax6.set_xlabel('Cycle')
ax6.set_ylabel('Total Steps (M)')
ax6.set_title('Per-Cycle Total Steps', fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_path = outdir / 'rw_comparison.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved {out_path}')
print(f'RW cycles: {len(rw_elapsed)}, No-RW cycles: {len(norw_elapsed)}')
print(f'Max |rel_diff|: {np.max(np.abs(rel_diff)):.1f}%')
print(f'Mean |rel_diff| (x>2): {np.mean(np.abs(rel_diff[xr>2])):.1f}%')
