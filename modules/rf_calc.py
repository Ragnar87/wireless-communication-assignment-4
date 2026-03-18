
from dataclasses import dataclass

import numpy as np
import math

from modules.path_tracing import count_obstacles_to_point

EXPONENT_N = 2.622
PL_d0 = 34.93 # path loss at reference distance d0 in dB
PL_DOOR = 1 # additional loss per door in dB
PL_WINDOW = 15.64 # additional loss per window in dB
PL_WALL = 19.54 # additional loss per wall in dB

def path_loss(distance, exponent_n=EXPONENT_N, PL_d0=PL_d0):
    distance = np.asarray(distance, dtype=float)

    result = np.zeros_like(distance)

    mask = distance > 0
    result[mask] = PL_d0 + 10 * exponent_n * np.log10(distance[mask])

    return result

def other_losses(n_walls, n_windows, n_doors, p_wall=PL_WALL, p_window=PL_WINDOW, p_door=PL_DOOR):
    return n_walls * p_wall + n_windows * p_window + n_doors * p_door

def pr_min(interference):
    gamma_min = 7
    B = 1000000 #bandwidth in MHz
    NF = 6 #Noise figure
    N = -174 + 10 * math.log10(B) + NF
    return interference + N + gamma_min

def euclidean_distance_between_points(a, b, cell_size=0.25):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx**2 + dy**2) * cell_size

def calculate_pl_between_nodes(grid, tx_node, rx_node, cell_size=0.25):
    tx = tx_node.coordinates
    rx = rx_node.coordinates

    distance = euclidean_distance_between_points(tx, rx, cell_size=cell_size)
    n_walls, n_windows, n_doors = count_obstacles_to_point(grid, tx, rx)

    return path_loss(distance) + other_losses(n_walls, n_windows, n_doors)

def calculate_pl_matrix(grid, nodes, cell_size=0.25):
    n = len(nodes)
    pl_matrix = np.zeros((n, n), dtype=float)

    for i, tx_node in enumerate(nodes):
        for j, rx_node in enumerate(nodes):
            if i == j:
                np.nan
            else:
                pl_matrix[i, j] = calculate_pl_between_nodes(
                    grid,
                    tx_node,
                    rx_node,
                    cell_size=cell_size
                )

    return pl_matrix

def calculate_pt_req_materix(pl_matrix, interference=0):
    pr_min_value = pr_min(interference)
    pt_req_matrix = pr_min_value + pl_matrix # transmit antenna gain and receive antenna gain are assumed to be 0
    pt_req_matrix[pt_req_matrix < 0] = 0 # Assuming minimum transmit power is 0 dBm
    return pt_req_matrix

def mark_infeasible_links(pt_req_matrix, max_transmit_power=10):
    infeasible_mask = pt_req_matrix > max_transmit_power
    pt_req_matrix[infeasible_mask] = np.nan
    return pt_req_matrix

def convert_pt_db_to_mw(pt_db_matrix):
    return 10 ** (pt_db_matrix / 10)

def energy_per_meassage(pt_mW_matrix, packet_duration, E_proc=0.005):
    return pt_mW_matrix * packet_duration + E_proc
