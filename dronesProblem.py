import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import numpy as np
import sympy as sp
import math

x, y = sp.symbols("x y")
random_points = set()
dt = 0.1

while len(random_points) < 10:
    x_point = np.random.randint(0, 2000)
    y_point = np.random.randint(0, 2000)
    random_points.add((x_point, y_point))

class Drone:
    def __init__(self, id, pos):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.k = 0.3
        self.found = False

drones = list(random_points)
drone_objects = [Drone(i, p) for i, p in enumerate(drones)]

person_pos = np.array([np.random.randint(300, 1800), np.random.randint(300, 1800)])

def dist(M, Q):
    return (M[0] - Q[0]) ** 2 + (M[1] - Q[1]) ** 2


def plotPoints(points, ax, color="black"):
    for i, p in enumerate(points):
        ax.scatter(p[0], p[1], color=color, s=40, marker=(4, 1, 45))
        ax.text(p[0] + 15, p[1] - 15, f"$P_{{{i+1}}}$", fontweight="bold", fontsize=8)


def getValidPairs(res, points, X, Y):
    distances = np.zeros((res, res, len(points)))
    for idx, obj in enumerate(points):
        distances[:, :, idx] = dist((X, Y), obj)
    closest_three = np.sort(np.argsort(distances, axis=2)[:, :, :3], axis=2)
    unique_rows = np.unique(closest_three.reshape(-1, 3), axis=0)
    return [tuple(row) for row in unique_rows]


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


def plotDiagram(equations, ax, X, Y):
    keys = list(equations.keys())
    cell_data = {}
    colors = sns.color_palette("husl", len(keys))
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
            cx, cy = np.mean(X[mask]), np.mean(Y[mask])
            cell_data[key] = (cx, cy)
            ax.text(
                cx, cy, f"({key[0]+1},{key[1]+1},{key[2]+1})", fontsize=6, ha="center"
            )
    return cell_data


def searchPerson(drone_objects):
    for d in drone_objects:
        if np.linalg.norm(d.pos - person_pos) < 300:
            indices = np.argsort(
                [np.linalg.norm(do.pos - person_pos) for do in drone_objects]
            )[:3]

            for idx, drone in enumerate(drone_objects):
                if idx in indices:
                    drone.k = 3
                    drone.found = True
                else:
                    drone.k = 0.001
                    drone.found = False
            break


res = 200
xs = np.linspace(0, 2000, res)
ys = np.linspace(0, 2000, res)
X, Y = np.meshgrid(xs, ys)

fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.set_aspect("equal")
ax1.set_title("Initial State")
plotPoints(drones, ax1, "green")
ax1.scatter(person_pos[0], person_pos[1], color="red", marker="X", s=100)
validPairs1 = getValidPairs(res, drones, X, Y)
eqs1 = getBoundaryEquations(drones, validPairs1)
cell_results = plotDiagram(eqs1, ax1, X, Y)

Ti_sum = np.zeros((len(drones), 2))
counts = np.zeros(len(drones))
for key, center in cell_results.items():
    for idx in key:
        Ti_sum[idx] += center
        counts[idx] += 1

for i in range(len(drones)):
    if counts[i] > 0:
        Ti = Ti_sum[i] / counts[i]
        k = drone_objects[i].k
        drone_objects[i].pos = Ti + (drone_objects[i].pos - Ti) * math.exp(k * dt)

fig2, ax2 = plt.subplots(figsize=(8, 8))


def updateDiagram(frame):
    ax2.clear()
    ax2.set_xlim(0, 2000)
    ax2.set_ylim(0, 2000)
    ax2.set_aspect("equal")
    ax2.set_title(f"Քայլ {frame + 1}")

    curr_positions = [tuple(d.pos) for d in drone_objects]

    v_pairs = getValidPairs(res, curr_positions, X, Y)
    eqs = getBoundaryEquations(curr_positions, v_pairs)

    current_cell_centers = plotDiagram(eqs, ax2, X, Y)

    Ti_sum = np.zeros((len(drone_objects), 2))
    counts = np.zeros(len(drone_objects))

    for key, center in current_cell_centers.items():
        for idx in key:
            Ti_sum[idx] += center
            counts[idx] += 1

    searchPerson(drone_objects)

    arrived = 0
    for d in drone_objects:
        if d.found and np.linalg.norm(d.pos - person_pos) < 10:
            arrived += 1

    if arrived == 1:
        ax2.set_title("Մարդը գտնված է!")
        plotPoints([tuple(d.pos) for d in drone_objects], ax2, "blue")
        ax2.scatter(person_pos[0], person_pos[1], color="red", marker="X", s=100)
        animation.event_source.stop()
        return

    for i in range(len(drone_objects)):
        if counts[i] > 0:
            if drone_objects[i].found:
                vector_to_signal = person_pos - drone_objects[i].pos
                distance = np.linalg.norm(vector_to_signal)

                if distance > 0:
                    direction_vector = vector_to_signal / distance
                    step_size = drone_objects[i].k * dt * 50
                    drone_objects[i].pos += direction_vector * step_size
            else:
                Ti = Ti_sum[i] / counts[i]
                direction = Ti - drone_objects[i].pos
                k = drone_objects[i].k
                drone_objects[i].pos += direction * math.exp(-k * dt)

    plotPoints([tuple(d.pos) for d in drone_objects], ax2, "blue")
    ax2.scatter(person_pos[0], person_pos[1], color="red", marker="X", s=100)


animation = FuncAnimation(fig2, updateDiagram, frames=20, interval=200, repeat=False)
plt.show()
