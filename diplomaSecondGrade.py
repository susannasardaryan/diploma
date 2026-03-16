import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import streamlit as st

x, y = sp.symbols("x y")

signal_points = [(2, 2), (10, 2), (6, 6), (2, 6), ((5, 0), (5, 5))]

def is_point(p):
    return isinstance(p[0], (int, float))


def is_segment(s):
    return isinstance(s[0], tuple)


def dist(M, Q):
    if is_segment(Q):
        x0, y0 = M
        (x1, y1), (x2, y2) = Q
    elif is_segment(M):
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


def plot_points_and_lines(points):
    for i, p in enumerate(points):
        if is_point(p):
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

def get_valid_pairs(res):
    valid_pairs_set = set()

    for i in range(res):
        for j in range(res):
            distances = []
            for idx, obj in enumerate(signal_points):
                d = dist((X[i, j], Y[i, j]), obj)
                distances.append((d, idx))

            distances.sort()
            pair = tuple(sorted([distances[0][1], distances[1][1]]))
            valid_pairs_set.add(pair)

    return list(valid_pairs_set)


def get_boundary_equations(data, valid_pairs):
    equations = {}
    n = len(data)
    for pair in valid_pairs:
        i, j = pair
        boundary_equations = []
        for k in range(n):
            if k == i or k == j:
                continue
            boundary_equations.append(dist((x, y), data[i]) - dist((x, y), data[k]))
            boundary_equations.append(dist((x, y), data[j]) - dist((x, y), data[k]))
        equations[(i, j)] = boundary_equations
    return equations


def plot_diagram(equations):
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
plot_points_and_lines(signal_points)

xs = []
ys = []

for obj in signal_points:
    if is_point(obj):
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
xMin, xMax = min(xmin, ymin), xmax + pad
yMin, yMax = min(xmin, ymin), ymax + pad
res = 800

xs = np.linspace(xMin, xMax, res)
ys = np.linspace(yMin, yMax, res)
X, Y = np.meshgrid(xs, ys)

validPairs = get_valid_pairs(res)
equations = get_boundary_equations(signal_points, validPairs)
plot_diagram(equations)

plt.title("Երկրորդ կարգի Վորոնովի դիագրամ", fontsize=12, fontweight="bold")
st.pyplot(fig)