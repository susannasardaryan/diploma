import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import solve, exp, Eq

x, y = sp.symbols("x y")

sources = [
    (2, 2, 1.2),
    (10, 2, 1.0),
    (6, 6, 1.5),
    (2, 6, 0.9),
    (9, 5, 1.3),
    (4, 1, 0.8),
    (8, 7, 1.1),
    (1, 4, 1.0),
]


def getLineParams(line):
    (x1, y1), (x2, y2) = line
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c


def isPoint(p):
    return isinstance(p[0], (int, float))


def isSegment(s):
    return isinstance(s[0], tuple)


def dist(M, Q):
    if isSegment(Q):
        x0, y0 = M
        a, b, c = getLineParams(Q)
        return (a * x0 + b * y0 + c) ** 2 / (a**2 + b**2)
    elif isSegment(M):
        x0, y0 = Q
        a, b, c = getLineParams(M)
        return (a * x0 + b * y0 + c) ** 2 / (a**2 + b**2)
    else:
        return (M[0] - Q[0]) ** 2 + (M[1] - Q[1]) ** 2


def plotPointsLines(points):
    for p in points:
        if isPoint(p):
            ax.scatter(p[0], p[1], color="green", s=50)
        else:
            ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color="green", linewidth=2)


def getBoundaryEquations(data):
    equations = {}
    n = len(data)
    for i in range(n):
        for j in range(i + 1, n):
            boundaryEquations = []
            for k in range(n):
                if k == i or k == j:
                    continue
                boundaryEquations.append(
                    sp.simplify(dist((x, y), data[i]) - dist((x, y), data[k]))
                )
                boundaryEquations.append(
                    sp.simplify(dist((x, y), data[j]) - dist((x, y), data[k]))
                )
            equations[(i, j)] = boundaryEquations
    return equations

fig, ax = plt.subplots()
ax.set_aspect("equal")

equations = getBoundaryEquations(sources)
cmap = plt.get_cmap('tab20c')
colors = cmap(np.linspace(0, 1, len(equations)))

areas = []
masks = []
keys = list(equations.keys())
xs = []
ys = []

for obj in sources:
    if isPoint(obj):
        xs.append(obj[0])
        ys.append(obj[1])
    else:
        xs.extend([obj[0][0], obj[1][0]])
        ys.extend([obj[0][1], obj[1][1]])


pad = 3
xMin, xMax = 0, max(xs)+pad
yMin, yMax = 0, max(ys)+pad
res = 600

xs = np.linspace(xMin, xMax, res)
ys = np.linspace(yMin, yMax, res)
X, Y = np.meshgrid(xs, ys)


for i, key in enumerate(keys):
    mask = np.ones_like(X, dtype=bool)
    for expr in equations[key]:
        f = sp.lambdify((x, y), expr, "numpy")
        Z = f(X, Y)
        mask &= Z <= 0

    ax.contourf(
        X, Y, mask.astype(int), levels=[0.5, 1.5], colors=[colors[i]], alpha=0.6
    )
    areas.append(mask.sum())

plotPointsLines(sources)

plt.show()
