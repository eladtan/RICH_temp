"""Extract Fig. 9(a) DIMC vector paths from arXiv:2108.13453v2.

Usage: python extract_fig9a.py paper.pdf output.csv
Requires PyMuPDF and NumPy. No smoothing or imposed physical values.
"""
import sys
import fitz
import numpy as np

page = fitz.open(sys.argv[1])[11]
drawings = page.get_drawings()
axes = drawings[0]["rect"]
curves = []
for drawing in drawings:
    color = drawing.get("color")
    rect = drawing["rect"]
    if (color and np.allclose(color, (0, 1, 0))
            and len(drawing["items"]) > 100 and rect.x1 < 300):
        items = drawing["items"]
        assert all(item[0] == "l" for item in items)
        points = np.array([tuple(items[0][1])] + [tuple(item[2]) for item in items])
        curves.append(points)
assert len(curves) == 4
assert all(np.array_equal(curves[0][:, 0], c[:, 0]) for c in curves)
# Figure bounds: x = [-0.19, -0.15], ordinate = [0, 4.2].
# Drawing order: material temperature, radiation temperature, density, -velocity.
x = -0.19 + (curves[0][:, 0] - axes.x0) / axes.width * 0.04
y = [(axes.y1 - c[:, 1]) / axes.height * 4.2 for c in curves]
data = np.column_stack((x, y[0] / 15, y[1] / 15, y[2], -2e7 * y[3]))
assert np.all(np.diff(x) > 0)
assert abs(data[0, 4]) < 1e4
assert -2.01e7 < data[-1, 4] < -1.9e7
np.savetxt(sys.argv[2], data, delimiter=",", fmt="%.10g", header=(
    "DIMC vector paths extracted from arXiv:2108.13453v2, Figure 9(a), page 12.\n"
    "x(cm), T_gas(keV), T_rad(keV), rho(g/cc), vx(cm/s)\n"
    "Reproduce with extract_fig9a.py; no smoothing, fitting, or imposed values."))
print(f"Extracted {len(data)} vertices per curve; upstream/downstream vx: {data[0,4]:.6g}, {data[-1,4]:.6g} cm/s")
