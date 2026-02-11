import matplotlib.pyplot as plt
import sys

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
assert len(sys.argv) >= 2, "Usage: python graph.py <results files>"

plt.figure()

for i, filename in enumerate(sys.argv[1:]):
    x_vals = []
    y_vals = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                x_str, y_str = line.split(",")
                x = float(x_str.strip())
                y = float(y_str.strip())
                x_vals.append(x)
                y_vals.append(y)
            except ValueError:
                continue  # skip malformed lines
    print(f"{filename}, {x_vals[:5]} ({len(x_vals)} values) , {y_vals[:5]} ({len(y_vals)} values)")
    plt.plot(x_vals, y_vals, color=COLORS[i % len(COLORS)], label=filename)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Graph of Points from results.txt')
plt.grid(True)
plt.show()