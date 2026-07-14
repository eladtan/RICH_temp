#!/usr/bin/env python3
"""Mollweide projection of polarization degree from a Fibonacci-sphere VTK file.

Barycentric interpolation on the spherical triangulation — identical to
what ParaView does with Gouraud shading, evaluated on a fine grid.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

vtk_path = "luminosity.vtk"

with open(vtk_path, 'r') as f:
    lines = f.readlines()

# --- Parse points ---
i = 0
while not lines[i].startswith("POINTS"):
    i += 1
n_points = int(lines[i].split()[1])
i += 1
points = np.zeros((n_points, 3))
for j in range(n_points):
    points[j] = [float(x) for x in lines[i + j].split()]
i += n_points

# --- Parse triangles ---
while not lines[i].startswith("POLYGONS"):
    i += 1
n_polys = int(lines[i].split()[1])
i += 1
triangles = np.zeros((n_polys, 3), dtype=int)
for j in range(n_polys):
    parts = [int(x) for x in lines[i + j].split()]
    triangles[j] = parts[1:4]

# --- Parse polarization_degree ---
i = 0
while i < len(lines):
    if lines[i].startswith("SCALARS") and "polarization_degree" in lines[i]:
        break
    i += 1
i += 2
pol_deg = np.array([float(lines[i + j].strip()) for j in range(n_points)])

# --- Unit direction vectors ---
r = np.linalg.norm(points, axis=1, keepdims=True)
dirs = points / r

# --- Build adjacency: vertex -> list of triangle indices ---
vert_tris = [[] for _ in range(n_points)]
for ti, t in enumerate(triangles):
    for vi in t:
        vert_tris[vi].append(ti)

# --- Fine lon-lat grid ---
n_lon, n_lat = 720, 360
lon_grid = np.linspace(-np.pi, np.pi, n_lon)
lat_grid = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
LON, LAT = np.meshgrid(lon_grid, lat_grid)

# Grid points as unit vectors
X = np.cos(LAT) * np.cos(LON)
Y = np.cos(LAT) * np.sin(LON)
Zs = np.sin(LAT)
grid_xyz = np.column_stack([X.ravel(), Y.ravel(), Zs.ravel()])
n_grid = grid_xyz.shape[0]

# --- KD-tree for fast nearest-vertex lookup ---
tree = cKDTree(dirs)
_, nearest = tree.query(grid_xyz)

# --- Barycentric interpolation on spherical triangles ---
# For each grid point, find the containing triangle among those adjacent
# to the nearest vertex, then interpolate.

def spherical_bary(q, v0, v1, v2):
    """Barycentric coordinates of q w.r.t. spherical triangle (v0,v1,v2).
    Uses the scalar triple product method."""
    c0 = np.cross(v1, v2)
    c1 = np.cross(v2, v0)
    c2 = np.cross(v0, v1)
    w0 = np.dot(q, c0)
    w1 = np.dot(q, c1)
    w2 = np.dot(q, c2)
    return w0, w1, w2

result = np.empty(n_grid)

for gi in range(n_grid):
    q = grid_xyz[gi]
    nv = nearest[gi]
    best_val = pol_deg[nv]
    found = False
    for ti in vert_tris[nv]:
        t = triangles[ti]
        w0, w1, w2 = spherical_bary(q, dirs[t[0]], dirs[t[1]], dirs[t[2]])
        if w0 >= -1e-10 and w1 >= -1e-10 and w2 >= -1e-10:
            s = w0 + w1 + w2
            if s > 0:
                best_val = (w0 * pol_deg[t[0]] + w1 * pol_deg[t[1]] + w2 * pol_deg[t[2]]) / s
            found = True
            break
    if not found:
        # Fallback: check k-nearest neighbors' triangles
        _, knn = tree.query(q, k=6)
        for nv2 in knn:
            for ti in vert_tris[nv2]:
                t = triangles[ti]
                w0, w1, w2 = spherical_bary(q, dirs[t[0]], dirs[t[1]], dirs[t[2]])
                if w0 >= -1e-10 and w1 >= -1e-10 and w2 >= -1e-10:
                    s = w0 + w1 + w2
                    if s > 0:
                        best_val = (w0 * pol_deg[t[0]] + w1 * pol_deg[t[1]] + w2 * pol_deg[t[2]]) / s
                    found = True
                    break
            if found:
                break
    result[gi] = best_val

Z = result.reshape(LON.shape)

# --- Plot ---
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection='mollweide')

pcm = ax.pcolormesh(LON, LAT, Z, cmap='inferno', shading='auto', rasterized=True)
cb = fig.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7)
cb.set_label('Polarization Degree', fontsize=14)

ax.set_title('Mollweide Projection \u2014 Polarization Degree\n(Fibonacci sphere, N=%d)' % n_points,
             fontsize=16, pad=20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "polarization_mollweide.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"Saved to {out_path}")
