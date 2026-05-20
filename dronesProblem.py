import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import numpy as np
import sympy as sp
import math

COLORS = {
    "drone_normal": "#1a2544",
    "drone_found": "#ca4300",
    "drone_passive": "#000000",
    "person": "#af2621",
}

x, y = sp.symbols("x y")
dt = 0.2
PERSON_IS_FOUND = False


class Drone:
    def __init__(self, id, pos):
        self.id = id
        self.pos = np.array(pos, dtype=float)
        self.k = 0.5
        self.found = False


def dist(M, Q):
    return (M[0] - Q[0]) ** 2 + (M[1] - Q[1]) ** 2


def get_points(count, min_distance, min_val, max_val, compare_points=None):
    if compare_points is None:
        compare_points = []

    random_points = set()
    attempts = 0
    while len(random_points) < count and attempts < 1000:
        attempts += 1
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

        if valid_position or attempts >= 1000:
            random_points.add((x_point, y_point))
            attempts = 0

    return random_points


def plot_points(points, ax, drones_list=None):
    for i, p in enumerate(points):
        if drones_list is not None:
            d = drones_list[i]
            if d.found:
                color, size = COLORS["drone_found"], 80
            elif PERSON_IS_FOUND and not d.found:
                color, size = COLORS["drone_passive"], 55
            else:
                color, size = COLORS["drone_normal"], 75
        else:
            color, size = COLORS["drone_normal"], 75

        ax.scatter(
            p[0],
            p[1],
            color=color,
            s=size,
            marker=(4, 1, 45),
            linewidths=0.8,
        )
        ax.text(
            p[0] + 15,
            p[1] - 15,
            f"$D_{{{i+1}}}$",
            fontweight="bold",
            fontsize=8,
            color=color,
        )


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
                cx,
                cy,
                f"({key[0]+1},{key[1]+1},{key[2]+1})",
                fontsize=6,
                ha="center",
                color="#333333",
            )
    return cell_data


def search_person(drone_objects):
    global PERSON_IS_FOUND

    if not PERSON_IS_FOUND:
        for d in drone_objects:
            if np.linalg.norm(d.pos - person_pos) <= 320:
                PERSON_IS_FOUND = True
                break

    if PERSON_IS_FOUND:
        distances = [np.linalg.norm(do.pos - person_pos) for do in drone_objects]
        indices = np.argsort(distances)[:3]

        for idx, drone in enumerate(drone_objects):
            if idx in indices:
                drone.k = 1.7
                drone.found = True
            else:
                drone.k = 0.001
                drone.found = False
        return True

    else:
        for drone in drone_objects:
            drone.k = 0.5
            drone.found = False
        return False


random_points = get_points(count=10, min_distance=400, min_val=50, max_val=1950)
drones = list(random_points)
drone_objects = [Drone(i, p) for i, p in enumerate(drones)]
person_points = get_points(
    count=1, min_distance=350, min_val=700, max_val=1450, compare_points=drones
)
person_pos = list(person_points)[0]
res_low = 190
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
valid_pairs1 = get_valid_pairs(res_low, drones, X_low, Y_low)
eqs1 = get_boundary_equations(drones, valid_pairs1)
cell_results = plot_diagram(eqs1, ax1, X, Y)
plot_points(drones, ax1)
ax1.scatter(
    person_pos[0],
    person_pos[1],
    color=COLORS["person"],
    marker="X",
    s=130,
    linewidths=1,
)

res = 400
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

    search_person(drone_objects)

    arrived = 0
    for i in range(len(drone_objects)):
        d = drone_objects[i]
        if counts[i] > 0:
            Ti = Ti_sum[i] / counts[i]
            if not PERSON_IS_FOUND and frame > 35:
                search_radius = 100

                offset_x = math.sin(frame * 0.04) * search_radius
                offset_y = math.cos(frame * 0.05) * search_radius

                target = np.clip(Ti + np.array([offset_x, offset_y]), 100, 1900)
            else:
                target = Ti

            if d.found and np.linalg.norm(d.pos - target) < 30:
                arrived += 1

            d.pos = target + (d.pos - target) * math.exp(-d.k * dt)

    if arrived >= 3:
        active_drones = [d.id + 1 for d in drone_objects if d.found]
        drones_names = ", ".join(map(str, active_drones))
        ax2.set_title(
            f"Մարդը գտնված է {drones_names} դրոնների կողմից!",
            color=COLORS["drone_found"],
        )

        plot_points(
            [tuple(d.pos) for d in drone_objects], ax2, drones_list=drone_objects
        )
        ax2.scatter(
            person_pos[0],
            person_pos[1],
            color=COLORS["person"],
            marker="X",
            s=130,
            linewidths=0.8,
        )
        animation.event_source.stop()
        return

    plot_points([tuple(d.pos) for d in drone_objects], ax2, drones_list=drone_objects)
    ax2.scatter(
        person_pos[0],
        person_pos[1],
        color=COLORS["person"],
        marker="X",
        s=130,
        linewidths=0.8,
    )


animation = FuncAnimation(fig2, update_diagram, frames=400, interval=50, repeat=False)

plt.show()
