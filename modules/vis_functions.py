import json

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from modules.models import Node

def visualize_graph_on_floorplan(
    graph,
    floorplan_path,
    grid_shape,
    title="Graph on floorplan",
    flip_floorplan=True,
    show_node_labels=True,
    node_label_attr="id",
    show_edge_labels=False,
    edge_label_attr="cost_mw",
    node_marker="*",
    node_size=200,
    edge_linewidth=1.5,
):
    floorplan = mpimg.imread(floorplan_path)
    if flip_floorplan:
        floorplan = np.flipud(floorplan)

    h, w = grid_shape

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use the exact same extent convention as your working point plot
    extent = (0, w, 0, h)

    ax.imshow(
        floorplan,
        origin="lower",
        extent=extent
    )

    # Collect node positions
    pos = {}
    for node_id, attrs in graph.nodes(data=True):
        if "coordinates" not in attrs or attrs["coordinates"] is None:
            raise ValueError(f"Node {node_id} is missing 'coordinates'")
        x, y = attrs["coordinates"]
        pos[node_id] = (x + 0.5, y + 0.5)

    # Draw edges
    for u, v, attrs in graph.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        ax.plot([x1, x2], [y1, y2], linewidth=edge_linewidth)

        if graph.is_directed():
            dx = x2 - x1
            dy = y2 - y1
            ax.arrow(
                x1, y1,
                dx * 0.85, dy * 0.85,
                head_width=2,
                head_length=3,
                length_includes_head=True,
                linewidth=0
            )

    # Draw nodes
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, s=node_size, marker=node_marker, edgecolors="black")

    # Node labels
    if show_node_labels:
        for node_id, attrs in graph.nodes(data=True):
            x, y = pos[node_id]

            if node_label_attr == "id":
                label = str(node_id)
            else:
                label = str(attrs.get(node_label_attr, node_id))

            ax.text(x, y, label, fontsize=9, ha="left", va="bottom")

    # Edge labels
    if show_edge_labels:
        for u, v, attrs in graph.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]

            xm = (x1 + x2) / 2
            ym = (y1 + y2) / 2

            label = attrs.get(edge_label_attr, "")
            if isinstance(label, float):
                label = f"{label:.2f}"

            ax.text(xm, ym, str(label), fontsize=8, ha="center", va="center")

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()
    plt.savefig(f"./figures/{title.replace(' ', '_')}.png")
    plt.close()

#helper to place the nodes on the floorplan, not used in the final program
def select_nodes_on_floorplan(floorplan_path, grid_shape, flip_floorplan=True):
    """
    Click nodes on the floorplan.
    Press ENTER when finished.
    """
    floorplan = mpimg.imread(floorplan_path)

    if flip_floorplan:
        floorplan = np.flipud(floorplan)

    h, w = grid_shape

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(floorplan, origin="lower", extent=(0, w, 0, h))
    ax.set_title("Click node locations. Press ENTER when finished.")
    ax.set_aspect("equal")

    points = plt.ginput(n=-1, timeout=0)
    plt.close(fig)

    nodes = []

    for i, (x, y) in enumerate(points):
        cell = (int(x), int(y))
        room = input(f"Room label for node {i}: ")

        nodes.append(Node(i, room, cell))

    return nodes

def save_nodes(nodes, filename):
    data = []

    for node in nodes:
        data.append({
            "id": node.id,
            "room_label": node.room_label,
            "coordinates": node.coordinates
        })

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(nodes)} nodes to {filename}")

#helper to place the nodes on the floorplan, not used in the final program
def select_and_save_nodes(grid):
    nodes = select_nodes_on_floorplan("entire_floorplan.png", grid.shape)
    save_nodes(nodes, "extra-nodes.json")


def show_points_on_floorplan(
    floorplan_path,
    points,
    grid_shape=None,
    title="Points on floorplan",
    flip_floorplan=True,
    marker="*",
    marker_size=200,
    show_labels=False,
    label_mode="index",
    gateway_id=None,
    gateway_marker="X",
    gateway_size=300
):
    floorplan = mpimg.imread(floorplan_path)
    if flip_floorplan:
        floorplan = np.flipud(floorplan)

    fig, ax = plt.subplots(figsize=(10, 8))

    if grid_shape is None:
        h, w = floorplan.shape[:2]
        extent = (0, w, 0, h)
        offset = 0.0
    else:
        h, w = grid_shape
        extent = (0, w, 0, h)
        offset = 0.5

    ax.imshow(floorplan, origin="lower", extent=extent)

    coords = []
    labels = []
    ids = []

    for i, p in enumerate(points):
        if hasattr(p, "coordinates"):
            x, y = p.coordinates
            node_id = getattr(p, "id", i)

            if label_mode == "id":
                labels.append(str(node_id))
            elif label_mode == "room_label" and hasattr(p, "room_label"):
                labels.append(str(p.room_label))
            else:
                labels.append(str(i))

            ids.append(node_id)
        else:
            x, y = p
            labels.append(str(i))
            ids.append(i)

        coords.append((x, y))

    coords = np.asarray(coords, dtype=float)

    xs = coords[:, 0] + offset
    ys = coords[:, 1] + offset

    # draw normal nodes
    for x, y, node_id in zip(xs, ys, ids):
        if node_id == gateway_id:
            ax.scatter(x, y, s=gateway_size, marker=gateway_marker,
                       edgecolors="black", color="red", zorder=3)
        else:
            ax.scatter(x, y, s=marker_size, marker=marker,
                       edgecolors="black", color="blue", zorder=2)

    if show_labels:
        for label, x, y in zip(labels, xs, ys):
            ax.text(x, y+4, label, fontsize=9, ha="center", va="bottom", zorder=4)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()
    plt.savefig(f"./figures/{title.replace(' ', '_')}.png")
    plt.close()