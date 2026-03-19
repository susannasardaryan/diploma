import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import numpy as np
import sympy as sp
import math

x, y = sp.symbols("x y")
dt = 0.2
PERSON_IS_FOUND = False

class Drone:
    def __init__(self, id, pos):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.k = 0.3
        self.found = False

def dist(M, Q):
    return (M[0] - Q[0]) ** 2 + (M[1] - Q[1]) ** 2

def get_points(count, min_distance, min_val, max_val, compare_points=None): 
    if compare_points is None:
        compare_points = []

    random_points = set()
    while len(random_points) < count:
        x_point = np.random.randint(min_val, max_val)
        y_point = np.random.randint(min_val, max_val)

        valid_position = True
        
        for existing_x, existing_y in random_points:
            distance = math.sqrt(dist((existing_x, existing_y), (x_point, y_point)))
            if distance < min_distance:
                valid_position = False
                break

        if valid_position and len(compare_points):
            for existing_x, existing_y in compare_points:
                distance = math.sqrt(dist((existing_x, existing_y), (x_point, y_point)))
                if distance < min_distance:
                    valid_position = False
                    break

        if valid_position:
            random_points.add((x_point, y_point))

    return random_points

def plot_points(points, ax, color="black"):
    for i, p in enumerate(points):
        ax.scatter(p[0], p[1], color=color, s=40, marker=(4, 1, 45))
        ax.text(p[0] + 15, p[1] - 15, f"$P_{{{i+1}}}$", fontweight="bold", fontsize=8)


def get_valid_pairs(res, points, X, Y):
    distances = np.zeros((res, res, len(points)))
    for i, point in enumerate(points):
        distances[:, :, i] = dist((X, Y), point)
    closest_three = np.sort(np.argsort(distances, axis=2)[:, :, :3], axis=2)
    unique_rows = np.unique(closest_three.reshape(-1, 3), axis=0)
    return [tuple(row) for row in unique_rows]


def get_boundary_equations(data, valid_pairs):
    equations = {}
    n = len(data)
    for pair in valid_pairs:
        i, j, m = pair
        boundary_equations = []
        for k in range(n):
            if k == i or k == j or k == m:
                continue
            boundary_equations.append(dist((x, y), data[i]) - dist((x, y), data[k]))
            boundary_equations.append(dist((x, y), data[j]) - dist((x, y), data[k]))
            boundary_equations.append(dist((x, y), data[m]) - dist((x, y), data[k]))
        equations[(i, j, m)] = boundary_equations
    return equations


def plot_diagram(equations, ax, X, Y):
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


def search_person(drone_objects):
    global PERSON_IS_FOUND
    
    if not PERSON_IS_FOUND:
        for d in drone_objects:
            if np.linalg.norm(d.pos - person_pos) <= 300:
                PERSON_IS_FOUND = True
                break

    if PERSON_IS_FOUND:
        distances = [np.linalg.norm(do.pos - person_pos) for do in drone_objects]
        indices = np.argsort(distances)[:3]

        for idx, drone in enumerate(drone_objects):
            if idx in indices:
                drone.k = 5.0    
                drone.found = True
            else:
                drone.k = 0.5    
                drone.found = False
        return True

    else:
        for drone in drone_objects:
            drone.k = 0.5
            drone.found = False
        return False

random_points = get_points(count=10, min_distance=400, min_val=0, max_val=2000)
drones = list(random_points)
drone_objects = [Drone(i, p) for i, p in enumerate(drones)]
person_points = get_points(count=1, min_distance=300, min_val=300, max_val=1700, compare_points=drones)
person_pos = list(person_points)[0]

res_low = 100
xs_low = np.linspace(0, 2000, res_low)
ys_low = np.linspace(0, 2000, res_low)
X_low, Y_low = np.meshgrid(xs_low, ys_low)

res = 500
xs = np.linspace(0, 2000, res)
ys = np.linspace(0, 2000, res)
X, Y = np.meshgrid(xs, ys)

fig1, ax1 = plt.subplots(figsize=(8, 8))
ax1.set_aspect("equal")
ax1.set_title("Սկզբնական դիրքը")
plot_points(drones, ax1, "blue")
ax1.scatter(person_pos[0], person_pos[1], color="red", marker="X", s=100)
valid_pairs1 = get_valid_pairs(res_low, drones, X_low, Y_low)
eqs1 = get_boundary_equations(drones, valid_pairs1)
cell_results = plot_diagram(eqs1, ax1, X, Y)

res = 350
xs = np.linspace(0, 2000, res)
ys = np.linspace(0, 2000, res)
X, Y = np.meshgrid(xs, ys)
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
        drone_objects[i].pos = Ti + (drone_objects[i].pos - Ti) * math.exp(-k * dt)

fig2, ax2 = plt.subplots(figsize=(8, 8))


def update_diagram(frame):
    ax2.clear()
    ax2.set_xlim(0, 2000)
    ax2.set_ylim(0, 2000)
    ax2.set_aspect("equal")
    ax2.set_title(f"Քայլ {frame + 1}")

    curr_positions = [tuple(d.pos) for d in drone_objects]

    v_pairs = get_valid_pairs(res_low, curr_positions, X_low, Y_low)
    eqs = get_boundary_equations(curr_positions, v_pairs)

    current_cell_centers = plot_diagram(eqs, ax2, X, Y)

    Ti_sum = np.zeros((len(drone_objects), 2))
    counts = np.zeros(len(drone_objects))

    for key, center in current_cell_centers.items():
        for idx in key:
            Ti_sum[idx] += center
            counts[idx] += 1

    person_found = search_person(drone_objects)

    arrived = 0
    for i in range(len(drone_objects)):
        d = drone_objects[i]
        if counts[i] > 0:
            Ti = Ti_sum[i] / counts[i]
            if not person_found and frame > 35:
                search_radius = 300
                offset_x = math.sin(frame * 0.08 + d.id) * search_radius
                offset_y = math.cos(frame * 0.05 + d.id) * search_radius

                target = np.clip(Ti + np.array([offset_x, offset_y]), 100, 1900)
            else:
                target = Ti

            if d.found and np.linalg.norm(d.pos - target) < 10:
                arrived += 1

            d.pos = target + (d.pos - target) * math.exp(-d.k * dt)

    if arrived >= 3:
        active_drones = [d.id + 1 for d in drone_objects if d.found]
        drones_names = ", ".join(map(str, active_drones))
        ax2.set_title(f"Մարդը գտնված է {drones_names} դրոնների կողմից!")
        plot_points([tuple(d.pos) for d in drone_objects], ax2, "green")
        ax2.scatter(person_pos[0], person_pos[1], color="red", marker="X", s=100)
        animation.event_source.stop()
        return
    
    plot_points([tuple(d.pos) for d in drone_objects], ax2, "black")
    ax2.scatter(person_pos[0], person_pos[1], color="red", marker="X", s=100)
    # plt.savefig(f"frames/frame_{frame:03d}.png", dpi=400, bbox_inches='tight')


animation = FuncAnimation(fig2, update_diagram, frames=150, interval=200, repeat=False)
# animation.save('frames/drone_search.gif', writer='pillow', fps=50)
plt.show()
