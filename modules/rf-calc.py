
import numpy as np

def manhattan_distance_grid(Z, tx):
    h, w = Z.shape
    tx_x, tx_y = tx

    y, x = np.indices((h, w))  # coordinate grids

    return np.abs(x - tx_x) + np.abs(y - tx_y)

def euclidean_distance(shape, tx, cell_size=0.25):
    h, w = shape
    tx_x, tx_y = tx

    x = np.arange(w)
    y = np.arange(h)

    dx = x - tx_x
    dy = y[:, None] - tx_y

    distance_cells = np.sqrt(dx**2 + dy**2)

    return distance_cells * cell_size

def path_loss(distance, exponent_n, PL_d0):
    distance = np.asarray(distance, dtype=float)

    result = np.zeros_like(distance)

    mask = distance > 0
    result[mask] = PL_d0 + 10 * exponent_n * np.log10(distance[mask])

    return result

def other_losses(n_walls, n_windows, n_doors, p_wall, p_window, p_door):
    return n_walls * p_wall + n_windows * p_window + n_doors * p_door

def power_received(distance, n_walls, n_windows, n_doors, exponent_n, PL_d0, p_wall, p_window, p_door):
    Pt = 0 #transmit power
    Gt = 0 #transmit antenna gain
    Gr = 0 #receive antenna gain
    return Pt + Gt + Gr - path_loss(distance, exponent_n, PL_d0) - other_losses(n_walls, n_windows, n_doors, p_wall, p_window, p_door)

def snr(p_r):
    B = 1000000 #bandwidth in MHz
    NF = 6 #Noise figure
    N = -174 + 10 * math.log10(B) + NF
    return p_r - N
