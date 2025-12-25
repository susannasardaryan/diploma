import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

x, y = sp.symbols('x y')

RSSI = [(2, 2), (10, 2), (6, 6), (2, 6), ((5, 0), (5, 5))]

def getLineParams(line):
    (x1, y1), (x2, y2) = line
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    return a, b, c

def isPoint(p):
    return isinstance(p[0], (int, float))

def isSegment(s):
    return isinstance(s[0], tuple)

def dist(M, Q):
    if isSegment(Q):
        x0, y0 = M
        a, b, c = getLineParams(Q)
        return (a*x0 + b*y0 + c)**2 / (a**2 + b**2)
    else:
        return (M[0] - Q[0])**2 + (M[1] - Q[1])**2

def getNearestPoints(M):
    n = len(RSSI)
    near_points = []
    for j in range(n):
        d_ij = dist(M, RSSI[j])
        near_points.append((d_ij, j))

    near_points.sort(key=lambda x: x[0])
    return [j for _, j in near_points[:2]]


# eqs = []
# n = len(RSSI)
# for i in range(n):
#     near_points = get_nearest_indices(i)
#     for j in near_points:
#         eqs.append(dist((x, y), RSSI[i]) - dist((x, y), RSSI[j]))

X, Y = np.meshgrid(
    np.linspace(0, 12, 600),
    np.linspace(0, 8, 400)
)

Nx, Ny = X.shape
regions = np.empty((Nx, Ny), dtype=object)

for i in range(Nx):
    for j in range(Ny):
        M = (float(X[i, j]), float(Y[i, j]))
        index_i, index_j = getNearestPoints(M)

        pi = RSSI[index_i]
        pj = RSSI[index_j]

        regions[i, j] = tuple(sorted((index_i, index_j)))

print(regions)
boundary = np.zeros((Nx, Ny), dtype=int)

for i in range(Nx - 1):
    for j in range(Ny - 1):
        if regions[i, j] != regions[i+1, j] \
        or regions[i, j] != regions[i, j+1]:
            boundary[i, j] = 1

print(boundary)
fig, ax = plt.subplots()
ax.set_aspect('equal')
# print(eqs)
# for expr in eqs:
#     f = sp.lambdify((x, y), expr, 'numpy')
#     Z = f(X, Y)

#     ax.contourf(
#     X, Y,
#     Z <= 0
# )


for p in RSSI:
    if isPoint(p):
        ax.scatter(p[0], p[1], color='green', s=40)
    else:
        ax.plot(
            [p[0][0], p[1][0]],
            [p[0][1], p[1][1]],
            color='green',
            linewidth=2
        )

ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
plt.imshow(boundary, cmap='gray',
           extent=(0,12,0,8),
           origin='lower')
plt.show()
