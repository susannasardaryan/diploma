import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x, y = sp.symbols("x y")

RSSI = [(2, 2), (10, 2), (6, 6), (2, 6), ((5, 0), (5, 5))]


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
    else:
        return (M[0] - Q[0]) ** 2 + (M[1] - Q[1]) ** 2


def plotPointsLines(points):
    for i, p in enumerate(points):
        if isPoint(p):
            ax.scatter(p[0], p[1], color="green", s=40)
            ax.text(p[0] + 0.05, p[1] + 0.05, f"P{i+1}", fontsize=10)
        else:
            ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color="green", linewidth=2)


def getNearestPoints(M):
    n = len(RSSI)
    near_points = []
    for j in range(n):
        d_ij = dist(M, RSSI[j])
        near_points.append((d_ij, j))

    near_points.sort(key=lambda x: x[0])
    return [j for _, j in near_points[:2]]


X, Y = np.meshgrid(np.linspace(0, 12, 600), np.linspace(0, 8, 400))

Nx, Ny = X.shape
regions = np.empty((Nx, Ny), dtype=object)

for i in range(Nx):
    for j in range(Ny):
        M = (float(X[i, j]), float(Y[i, j]))
        index_i, index_j = getNearestPoints(M)

        pi = RSSI[index_i]
        pj = RSSI[index_j]

        regions[i, j] = tuple(sorted((index_i, index_j)))

boundary = np.zeros((Nx, Ny), dtype=int)

for i in range(Nx - 1):
    for j in range(Ny - 1):
        if regions[i, j] != regions[i + 1, j] or regions[i, j] != regions[i, j + 1]:
            boundary[i, j] = 1

fig, ax = plt.subplots()
ax.set_aspect("equal")

cmap = plt.get_cmap("tab20c")
colors = cmap(np.linspace(0, 1, 20))
xs = []
ys = []
pad = 3

for obj in RSSI:
    if isPoint(obj):
        xs.append(obj[0])
        ys.append(obj[1])
    else:
        xs.extend([obj[0][0], obj[1][0]])
        ys.extend([obj[0][1], obj[1][1]])

xMin, xMax = min(xs) - 2, max(xs) + pad
yMin, yMax = min(ys), max(ys) + pad
ax.set_xlim(xMin, xMax)
ax.set_ylim(yMin, yMax)

plt.imshow(boundary, cmap, colors, extent=(0, 12, 0, 8), origin="lower")

plotPointsLines(RSSI)
plt.show()
