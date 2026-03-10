import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x, y = sp.symbols("x y")

RSSI = [(2, 2), (10, 2), (6, 6), (2, 6), ((5, 0), (5, 5)), (4, 2)]


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
        (x1, y1), (x2, y2) = Q
    elif isSegment(M):
        x0, y0 = Q
        (x1, y1), (x2, y2) = M
    else:
        return (M[0] - Q[0]) ** 2 + (M[1] - Q[1]) ** 2

    dx = x2 - x1
    dy = y2 - y1

    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)

    if isinstance(t, np.ndarray):
        t = np.clip(t, 0, 1)
    elif isinstance(t, sp.Expr):
        t = sp.Max(0, sp.Min(1, t))
    else:
        t = max(0, min(1, t))

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return (x0 - proj_x) ** 2 + (y0 - proj_y) ** 2


def plotPointsLines(points):
    for i, p in enumerate(points):
        if isPoint(p):
            ax.scatter(p[0], p[1], color="green", s=40)
            ax.text(
                p[0] + 0.15,
                p[1] - 0.15,
                f"$P_{{{i+1}}}$",
                fontweight="bold",
                fontsize=8,
            )
        else:
            ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color="green", linewidth=2)
            ax.text(
                p[0][0] + 0.15,
                p[1][0] + 0.15,
                f"$P_{{{i+1}}}$",
                fontweight="bold",
                fontsize=8,
            )

def getValidPairs(res):
    validPairsSet = set()

    for i in range(res):
        for j in range(res):
            distances = []
            for idx, obj in enumerate(RSSI):
                d = dist((X[i, j], Y[i, j]), obj)
                distances.append((d, idx))

            distances.sort()
            pair = tuple(sorted([distances[0][1], distances[1][1]]))
            validPairsSet.add(pair)

    return list(validPairsSet)

def getBoundaryEquations(data, validPairs):
    equations = {}
    n = len(data)
    for pair in validPairs:
        i, j = pair
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

def plotDiagram(equations):
    areas = []
    keys = list(equations.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(equations)))

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
                f"({key[0]+1}, {key[1]+1})",
                fontsize=6,
                ha="center",
                va="center",
            )
            areas.append(mask.sum())


fig, ax = plt.subplots()
ax.set_aspect("equal")
plotPointsLines(RSSI)

xs = []
ys = []

for obj in RSSI:
    if isPoint(obj):
        xs.append(obj[0])
        ys.append(obj[1])
    else:
        xs.extend([obj[0][0], obj[1][0]])
        ys.extend([obj[0][1], obj[1][1]])

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
equations = getBoundaryEquations(RSSI, validPairs)
plotDiagram(equations)

plt.title("Երկրորդ կարգի Վորոնովի դիագրամ", fontsize=12, fontweight="bold")
plt.show()
