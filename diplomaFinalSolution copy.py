import matplotlib.pyplot as plt
import numpy as np

RSSI = [
    (2, 2), (10, 2), (6, 6), (2, 6), ((5, 0), (5, 5)),
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

def dist_fast(X, Y, Q):
    if isSegment(Q):
        a, b, c = getLineParams(Q)
        return (a * X + b * Y + c) ** 2 / (a**2 + b**2)
    return (X - Q[0]) ** 2 + (Y - Q[1]) ** 2

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_aspect("equal")

xMin, xMax, yMin, yMax, res = 0, 12, 0, 8, 600
xx = np.linspace(xMin, xMax, res)
yy = np.linspace(yMin, yMax, res)
X, Y = np.meshgrid(xx, yy)

grid = np.zeros((res, res, len(RSSI)))

for key, point in enumerate(RSSI):
    grid[:, :, key] = dist_fast(X, Y, point)

near2 = np.sort(np.argsort(grid, axis=2)[:, :, :2], axis=2)
print(near2)
pair = near2[:, :, 0] +' '+ near2[:, :, 1]
print(pair)
unique_hashes = np.unique(pair)

Z_map = np.zeros((res, res))
for i, h in enumerate(unique_hashes):
    Z_map[pair == h] = i

cmap = plt.get_cmap('tab20c')
ax.imshow(Z_map, origin='lower', extent=[xMin, xMax, yMin, yMax],
          cmap=cmap, alpha=0.6, aspect='auto')


for i in range(len(unique_hashes)):
    ax.contour(X, Y, Z_map == i, levels=[0.5], colors="black", linewidths=1.2)

for i, p in enumerate(RSSI):
    if isPoint(p):
        ax.scatter(p[0], p[1], color="green", s=60, zorder=11)
        ax.text(p[0]+0.15, p[1]+0.15, f"P{i}", fontweight='bold')
    else:
        ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color="green", linewidth=4, zorder=11)

plt.xlim(xMin, xMax)
plt.ylim(yMin, yMax)
plt.title("Fast 2nd Order Voronoi Diagram (Pure NumPy)")
plt.show()