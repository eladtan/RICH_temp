"""Read a cycle's VTU pieces and reduce them to per-shell radial averages.

The converging-Marshak meshes are built from concentric shells of identical
radius, so averaging per shell (rather than into uniform radial bins) avoids the
aliasing that uniform bins produce when a bin straddles a varying number of shells.
"""

import os
import sys
import xml.etree.ElementTree as ElementTree
import numpy as np
import vtk
from vtk.util import numpy_support as vtk_np
from concurrent.futures import ProcessPoolExecutor


def read_piece(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    cell_data = reader.GetOutput().GetCellData()
    coords = vtk_np.vtk_to_numpy(cell_data.GetArray("Coordinates"))
    radius = np.linalg.norm(coords, axis=1)
    temperature = vtk_np.vtk_to_numpy(cell_data.GetArray("temperature"))
    density = vtk_np.vtk_to_numpy(cell_data.GetArray("density"))
    erad = np.maximum(vtk_np.vtk_to_numpy(cell_data.GetArray("Erad_time_avg")), 0.0)
    return radius, temperature, density, erad


def pieces_from_manifest(pvtu_path):
    """Piece list declared by the .pvtu.

    Globbing the cycle directory is not safe: a run using fewer ranks than a
    previous one overwrites only the leading pieces and leaves the rest behind,
    so a glob silently mixes two different simulations.
    """
    root = os.path.dirname(os.path.abspath(pvtu_path))
    tree = ElementTree.parse(pvtu_path)
    return [os.path.join(root, piece.get("Source")) for piece in tree.iter("Piece")]


def load(pvtu_path, workers=24):
    files = pieces_from_manifest(pvtu_path)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(read_piece, files, chunksize=4))
    return len(files), tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def shells(radius, decimals):
    _, inverse = np.unique(np.round(radius, decimals), return_inverse=True)
    count = np.bincount(inverse).astype(float)
    return inverse, count, np.bincount(inverse, weights=radius) / count


if __name__ == "__main__":
    pvtu_path, out_path, radius_limit, decimals = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
    npieces, (radius, temperature, density, erad) = load(pvtu_path)
    keep = radius < radius_limit
    radius, temperature, density, erad = radius[keep], temperature[keep], density[keep], erad[keep]
    inverse, count, shell_radius = shells(radius, decimals)
    mean_temperature = np.bincount(inverse, weights=temperature) / count
    mean_square = np.bincount(inverse, weights=temperature ** 2) / count
    np.savez(out_path,
             shell_radius=shell_radius,
             count=count,
             temperature=mean_temperature,
             temperature_std=np.sqrt(np.maximum(mean_square - mean_temperature ** 2, 0.0)),
             density=np.bincount(inverse, weights=density) / count,
             erad=np.bincount(inverse, weights=erad) / count)
    print(f"{pvtu_path}: {npieces} pieces, {radius.size} cells, {shell_radius.size} shells "
          f"({int(count.min())}-{int(count.max())} cells/shell) -> {out_path}", flush=True)
