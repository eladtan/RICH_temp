#!/usr/bin/env python3
"""
Visualize a single Voronoi cell and its Monte Carlo particles.

Usage:
    python3 draw.py <vtk_file> <hdf5_file> <rank> <cell_index>

- vtk_file: VTU/PVTU file written by write_voronoi.cpp (cell geometry)
- hdf5_file: HDF5 file written by write_simulation.cpp (particle data)
- rank: MPI rank whose data to read
- cell_index: local cell index within that rank
"""
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import vtk
except ImportError:
    sys.exit("Error: python vtk package is required. Install with: pip install vtk")


def read_cell_from_vtk(vtk_file, cell_index):
    """Extract face polygons for a single cell from a VTU/PVTU file."""
    if vtk_file.endswith(".pvtu"):
        reader = vtk.vtkXMLPUnstructuredGridReader()
    else:
        reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    grid = reader.GetOutput()

    n_cells = grid.GetNumberOfCells()
    if cell_index < 0 or cell_index >= n_cells:
        sys.exit(f"Cell index {cell_index} out of range [0, {n_cells})")

    cell = grid.GetCell(cell_index)
    faces = []
    for i in range(cell.GetNumberOfFaces()):
        face = cell.GetFace(i)
        pts = []
        for j in range(face.GetNumberOfPoints()):
            pts.append(grid.GetPoint(face.GetPointId(j)))
        faces.append(np.array(pts))
    return faces


def find_particles_dataset(group):
    """Recursively search for a 'particles' dataset inside an HDF5 group."""
    if "particles" in group:
        item = group["particles"]
        if isinstance(item, h5py.Dataset):
            return item
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            result = find_particles_dataset(item)
            if result is not None:
                return result
    return None


def read_particles(hdf5_file, rank, cell_index):
    """Read particle locations for a specific cell from the HDF5 file."""
    with h5py.File(hdf5_file, "r") as f:
        rank_key = f"rank{rank}"
        if rank_key not in f:
            available = [k for k in f.keys()]
            sys.exit(
                f"Group '{rank_key}' not found in {hdf5_file}.\n"
                f"Available top-level keys: {available}"
            )

        rank_group = f[rank_key]
        ds = find_particles_dataset(rank_group)
        if ds is None:
            def list_tree(g, prefix=""):
                lines = []
                for k in g:
                    path = f"{prefix}/{k}"
                    lines.append(path)
                    if isinstance(g[k], h5py.Group):
                        lines.extend(list_tree(g[k], path))
                return lines

            tree = "\n  ".join(list_tree(rank_group))
            sys.exit(
                f"No 'particles' dataset found under '{rank_key}'.\n"
                f"Contents:\n  {tree}"
            )

        data = ds[:]
        mask = data["cellIndex"] == cell_index
        filtered = data[mask]

        loc = filtered["location"]
        return loc["x"], loc["y"], loc["z"]


def main():
    if len(sys.argv) != 5:
        print(__doc__.strip())
        sys.exit(1)

    vtk_file = sys.argv[1]
    hdf5_file = sys.argv[2]
    rank = int(sys.argv[3])
    cell_index = int(sys.argv[4])

    print(f"Reading cell {cell_index} geometry from {vtk_file} ...")
    faces = read_cell_from_vtk(vtk_file, cell_index)
    print(f"  {len(faces)} faces")

    print(f"Reading particles for rank {rank}, cell {cell_index} from {hdf5_file} ...")
    px, py, pz = read_particles(hdf5_file, rank, cell_index)
    print(f"  {len(px)} particles")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    poly = Poly3DCollection(
        faces, alpha=0.2, facecolor="cyan", edgecolor="none",
    )
    ax.add_collection3d(poly)

    for face in faces:
        closed = np.vstack([face, face[0:1]])
        ax.plot(closed[:, 0], closed[:, 1], closed[:, 2],
                color="navy", linewidth=0.8)

    if len(px) > 0:
        ax.scatter(px, py, pz, c="red", s=10, depthshade=True,
                   label=f"Particles ({len(px)})")
        ax.legend()

    all_pts = np.concatenate(faces)
    if len(px) > 0:
        all_pts = np.vstack([all_pts, np.column_stack([px, py, pz])])

    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2
    half_range = (maxs - mins).max() / 2 * 1.05

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Cell {cell_index}  (rank {rank})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
