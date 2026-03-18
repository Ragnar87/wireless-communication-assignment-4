import math
import numpy as np

WALL = 1
WINDOW = 2
DOOR = 3

def cells_on_line(tx, target):
    """
    Returns grid cells crossed by the line from tx to target.
    tx and target are (x, y).
    """
    x0, y0 = tx
    x1, y1 = target

    x = x0 + 0.5
    y = y0 + 0.5
    end_x = x1 + 0.5
    end_y = y1 + 0.5

    dx = end_x - x
    dy = end_y - y

    step_x = 0 if dx == 0 else (1 if dx > 0 else -1)
    step_y = 0 if dy == 0 else (1 if dy > 0 else -1)

    if dx != 0:
        t_delta_x = abs(1.0 / dx)
        next_vert_grid = math.floor(x) + 1 if step_x > 0 else math.floor(x)
        t_max_x = abs((next_vert_grid - x) / dx)
    else:
        t_delta_x = float("inf")
        t_max_x = float("inf")

    if dy != 0:
        t_delta_y = abs(1.0 / dy)
        next_horiz_grid = math.floor(y) + 1 if step_y > 0 else math.floor(y)
        t_max_y = abs((next_horiz_grid - y) / dy)
    else:
        t_delta_y = float("inf")
        t_max_y = float("inf")

    cx, cy = x0, y0
    cells = [(cx, cy)]

    while (cx, cy) != (x1, y1):
        if t_max_x < t_max_y:
            cx += step_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_x:
            cy += step_y
            t_max_y += t_delta_y
        else:
            cx += step_x
            cy += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y

        cells.append((cx, cy))

    return cells


def count_obstacles_to_point(grid, tx, target):
    """
    Count crossings into contiguous obstacle regions instead of counting every cell.

    Returns:
        (n_walls, n_windows, n_doors)
    """
    h, w = grid.shape
    path = cells_on_line(tx, target)

    n_walls = 0
    n_windows = 0
    n_doors = 0

    prev_material = 0
    same_material_count = 0

    for x, y in path[1:]:  # skip tx cell
        if not (0 <= x < w and 0 <= y < h):
            continue

        material = grid[y, x]

        if material != prev_material:
            if material == WALL:
                n_walls += 1
            elif material == WINDOW:
                n_windows += 1
            elif material == DOOR:
                n_doors += 1

        prev_material = material

    return n_walls, n_windows, n_doors


def count_obstacles_grid(grid, tx):
    h, w = grid.shape

    wall_counts = np.zeros((h, w), dtype=int)
    window_counts = np.zeros((h, w), dtype=int)
    door_counts = np.zeros((h, w), dtype=int)

    for y in range(h):
        for x in range(w):
            n_walls, n_windows, n_doors = count_obstacles_to_point(grid, tx, (x, y))
            wall_counts[y, x] = n_walls
            window_counts[y, x] = n_windows
            door_counts[y, x] = n_doors

    return wall_counts, window_counts, door_counts
