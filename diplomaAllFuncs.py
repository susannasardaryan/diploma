import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from sympy import solve, exp, Eq

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
    elif isSegment(M):
        x0, y0 = Q
        a, b, c = getLineParams(M)
        return (a * x0 + b * y0 + c) ** 2 / (a**2 + b**2)
    else:
        return (M[0] - Q[0]) ** 2 + (M[1] - Q[1]) ** 2


def getNearestPoints(i):
    M = RSSI[i]
    n = len(RSSI)
    near_points = []
    for j in range(n):
        if i != j:
            d_ij = dist(M, RSSI[j])
            near_points.append((d_ij, j))

    near_points.sort(key=lambda x: x[0])
    return [j for _, j in near_points[:2]]


fig, ax = plt.subplots()
ax.set_aspect("equal")
equations = {}
solutions = {}
n = len(RSSI)
for i in range(n):
    for j in range(i+1, n):
        nearPointsEquations = []
        for k in range(n):
            if k == i or k == j:
                continue
            nearPointsEquations.append(sp.simplify(dist((x, y), RSSI[i]) - dist((x, y), RSSI[k])))
            nearPointsEquations.append(sp.simplify(dist((x, y), RSSI[j]) - dist((x, y), RSSI[k])))
        equations[(i, j)] = nearPointsEquations
        # solution = solve(nearPointsEquations, (x, y))
        # solutions[i] = solution


for p in RSSI:
    if isPoint(p):
        ax.scatter(p[0], p[1], color="green", s=40)
    else:
        ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color="green", linewidth=2)


# def plotRealPoints(ax, p, **args):
#     try:
#         px = float(sp.N(p[0]))
#         py = float(sp.N(p[1]))
#         ax.scatter(px, py, **args)
#     except Exception:
#         pass


# for i in range(len(RSSI)):
#     for item in solutions.get(i, []):
#         plotRealPoints(ax, item, color="blue", s=40)


# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# print(eqs)
# for expr in eqs:
#     f = sp.lambdify((x, y), expr, 'numpy')
#     Z = f(X, Y)

#     ax.contourf(
#     X, Y,
#     Z <= 0
# )



colors = plt.cm.tab10(np.linspace(0, 1, len(equations)))

areas = []
masks = []
keys = list(equations.keys())
xMin, xMax = 0, 12
yMin, yMax = 0, 8
res = 400

xs = np.linspace(xMin, xMax, res)
ys = np.linspace(yMin, yMax, res)
X, Y = np.meshgrid(xs, ys)

for idx, key in enumerate(keys):
    mask = np.ones_like(X, dtype=bool)
    for expr in equations[key]:
        f = sp.lambdify((x, y), expr, "numpy")
        Z = f(X, Y)
        mask &= (Z <= 0)

    masks.append(mask)
    areas.append(mask.sum())

imax = int(np.argmax(areas))

ax.contourf(X, Y, masks[imax].astype(int), levels=[0.5, 1.5], alpha=0.7)

for idx, mask in enumerate(masks):
    ax.contourf(X, Y, mask.astype(int), levels=[0.5, 1.5], colors=[colors[idx]], alpha=0.25)

plt.show()
