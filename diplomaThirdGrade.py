import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x, y = sp.symbols("x y")

signal_points = [(2, 2), (10, 2), (6, 6), (2, 6), (4, 2)]

def dist(M, Q):
    return (M[0] - Q[0]) ** 2 + (M[1] - Q[1]) ** 2


def plotPointsLines(points):
    for i, p in enumerate(points):
        ax.scatter(p[0], p[1], color="green", s=40)
        ax.text(
            p[0] + 0.15,
            p[1] - 0.15,
            f"$P_{{{i+1}}}$",
            fontweight="bold",
            fontsize=8,
        )


def getValidPairs(res):
    distances = np.zeros((res, res, len(signal_points)))
    for idx, obj in enumerate(signal_points):
        distances[:, :, idx] = dist((X, Y), obj)

    closest_three = np.sort(np.argsort(distances, axis=2)[:, :, :3], axis=2)
    unique_rows = np.unique(closest_three.reshape(-1, 3), axis=0)
    valid_pairs = [tuple(row) for row in unique_rows]

    return valid_pairs


def getBoundaryEquations(data, validPairs):
    equations = {}
    n = len(data)
    for pair in validPairs:
        i, j, m = pair
        boundaryEquations = []
        for k in range(n):
            if k == i or k == j or k == m:
                continue
            boundaryEquations.append(dist((x, y), data[i]) - dist((x, y), data[k]))
            boundaryEquations.append(dist((x, y), data[j]) - dist((x, y), data[k]))
            boundaryEquations.append(dist((x, y), data[m]) - dist((x, y), data[k]))
        equations[(i, j, m)] = boundaryEquations
    return equations


def plotDiagram(equations):
    areas = []
    keys = list(equations.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(keys)))

    for i, key in enumerate(keys):
        mask = np.ones_like(X, dtype=bool)
        for expr in equations[key]:
            f = sp.lambdify((x, y), expr, "numpy")
            Z = f(X, Y)
            mask &= Z <= 0

        if np.any(mask):
            ax.contourf(
                X, Y, mask.astype(int), levels=[0.5, 1.5], colors=[colors[i]], alpha=0.3
            )
            center_x = np.mean(X[mask])
            center_y = np.mean(Y[mask])
            ax.text(
                center_x,
                center_y,
                f"({key[0]+1}, {key[1]+1}, {key[2]+1})",
                fontsize=6,
                ha="center",
                va="center",
            )
            areas.append(mask.sum())


fig, ax = plt.subplots()
ax.set_aspect("equal")
plotPointsLines(signal_points)

xs = []
ys = []

for obj in signal_points:
    xs.append(obj[0])
    ys.append(obj[1])

pad = 3
xmax = max(xs)
ymax = max(ys)
xmin = min(xs)
ymin = min(ys)
xMin, xMax = min(xmin, ymin) - 2, xmax + pad
yMin, yMax = min(xmin, ymin) - 2, ymax + pad
res = 500

xs = np.linspace(xMin, xMax, res)
ys = np.linspace(yMin, yMax, res)
X, Y = np.meshgrid(xs, ys)

validPairs = getValidPairs(res)
equations = getBoundaryEquations(signal_points, validPairs)
plotDiagram(equations)

plt.title("Երրորդ կարգի Վորոնովի դիագրամ", fontsize=12)
plt.show()
