#!/usr/bin/env python3
"""Analyze Mach2 radiative shock simulation results."""
import numpy as np

def load_file(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 5:
                try:
                    x = float(parts[0])
                    rho = float(parts[1])
                    T_gas = float(parts[2])
                    T_rad = float(parts[3])
                    v_x = float(parts[4])
                    data.append((x, rho, T_gas, T_rad, v_x))
                except ValueError:
                    pass
    return np.array(data)

def analyze_file(data, label):
    if len(data) < 6:
        print(f"Not enough data in {label}")
        return
    
    # A. Downstream (leftmost 3 cells - most negative x)
    downstream = data[0:3]
    # B. Upstream (rightmost 3 cells - most positive x)
    upstream = data[-3:]
    
    # C. Shock position - steepest density gradient
    drho = np.abs(np.diff(data[:, 1]))
    imax = np.argmax(drho)
    x_left = data[imax, 0]
    x_right = data[imax+1, 0]
    rho_left = data[imax, 1]
    rho_right = data[imax+1, 1]
    x_shock = (x_left + x_right) / 2
    
    # D. At x≈0 - find nearest cells
    dist = np.abs(data[:, 0])
    idx_near = np.argmin(dist)
    cell_near = data[idx_near]
    # Also get the two cells bracketing 0 if different
    idx_neg = np.where(data[:, 0] <= 0)[0]
    idx_pos = np.where(data[:, 0] >= 0)[0]
    if len(idx_neg) > 0 and len(idx_pos) > 0:
        i_neg = idx_neg[-1]
        i_pos = idx_pos[0]
        if i_neg != i_pos:
            cell_left = data[i_neg]
            cell_right = data[i_pos]
    
    print(f"\n{'='*80}")
    print(f"  {label}")
    print('='*80)
    print("\nA. DOWNSTREAM (leftmost 3 cells):")
    print("   x(cm)        rho(g/cc)  T_gas(keV)  T_rad(keV)  v_x(cm/s)")
    for row in downstream:
        print(f"   {row[0]:10.6f}  {row[1]:10.5f}  {row[2]:10.5f}  {row[3]:10.5f}  {row[4]:.4e}")
    
    print("\nB. UPSTREAM (rightmost 3 cells):")
    print("   x(cm)        rho(g/cc)  T_gas(keV)  T_rad(keV)  v_x(cm/s)")
    for row in upstream:
        print(f"   {row[0]:10.6f}  {row[1]:10.5f}  {row[2]:10.5f}  {row[3]:10.5f}  {row[4]:.4e}")
    
    print("\nC. SHOCK POSITION (steepest density gradient):")
    print(f"   x ≈ {x_shock:.6f} cm")
    print(f"   rho downstream (left):  {rho_left:.5f} g/cc")
    print(f"   rho upstream (right):   {rho_right:.5f} g/cc")
    print(f"   |Δρ| = {abs(rho_right - rho_left):.5f} g/cc")
    
    print("\nD. AT x≈0 (nearest cell(s)):")
    print(f"   Nearest cell: x={cell_near[0]:.6f} cm")
    print(f"   rho={cell_near[1]:.5f}, T_gas={cell_near[2]:.5f}, T_rad={cell_near[3]:.5f}, v_x={cell_near[4]:.4e} cm/s")
    if len(idx_neg) > 0 and len(idx_pos) > 0 and i_neg != i_pos:
        print(f"   Cell left of 0:  x={cell_left[0]:.6f}  rho={cell_left[1]:.5f}  T_gas={cell_left[2]:.5f}  T_rad={cell_left[3]:.5f}  v_x={cell_left[4]:.4e}")
        print(f"   Cell right of 0: x={cell_right[0]:.6f}  rho={cell_right[1]:.5f}  T_gas={cell_right[2]:.5f}  T_rad={cell_right[3]:.5f}  v_x={cell_right[4]:.4e}")

def main():
    base = "/home/maorm/RICH/runs/Elad_paper_mach2"
    files = [
        ("mach2_mc_init.txt", "INITIAL (t=0 ns)"),
        ("mach2_mc_00001.txt", "FIRST DUMP (~0.6 ns)"),
        ("mach2_mc_final.txt", "FINAL (t=5 ns)"),
    ]
    for fname, label in files:
        path = f"{base}/{fname}"
        try:
            data = load_file(path)
            analyze_file(data, label)
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    print("\n" + "="*80)
    print("  REFERENCE: Rankine-Hugoniot expected values")
    print("="*80)
    print("  Downstream: rho=2.29, T=0.253 keV, v=-1.95e7 cm/s")
    print("  Upstream:   rho=1.0,  T=0.122 keV, v=0")

if __name__ == "__main__":
    main()
