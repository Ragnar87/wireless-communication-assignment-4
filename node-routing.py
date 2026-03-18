from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import networkx as nx
import json

from modules import rf_calc as rf
from modules import path_tracing as pt
from modules import vis_functions as vis
from modules.models import Node

#Objectives
# T1: Define node positions
# 1. Identify all romms included in your floor plan.
# 2. Pace one sensor node at the center of each room.
# 3. Mark the fixed gateway position.
# 3. PRovide a table of node IDs, room labels, and coordinates.

# T2: Compute pairwise losses and required transmit powers
# 1. Compute the pairwise path loss PL_ij between all node pairs using your model from Assignment 2.
# 2. For each interference scenario, compute P_r,min using Eq. 4
# 3. Compute the required transmit power Preq_t,ij for all canidate links using Eq. 5
# 4. Mark links as infeasible if Prec_t, ij > pt,max

# T3: Compute energy per message
# 1. Compute the packet duration.
# 2. Convert the required transmit powers to linear units.
# 3. Compute the message energy for each feasible link.
# 4. Present the resulting link-energy matrix or edge list for at least one interference scenario.

# T4: Build and visualize the weighted graph
# 1. Construct the weighted connectivity graph for each interference scenario.
# 2. Plot the floor plan with node positions and gateway overlaid.
# 3. Draw the feasible links between nodes.
# 4. Briefly comment on how the graph changes as the interference level increases.

# T5: Determine minium-energy routes
# 1. Determine a route from every sensor node to the gateway that minimizes the total network energy in Eq. 8.
# 2. Identify which nodes act as relays in the resulting solution.
# 3. Compute the total network energy, average hop count, and maximum hop count.

# T6: Compare interference scenarios
# For each of the three interference scenarios, report:
# - number of sensors connected directly to the gateway
# - number of nodes acting as relays
# - total network energy per message collection round
# - average hop count
# - maximum hop count
# Briefly interpret the results. In particular, discuss wether:
# - low interference leads to more direct or more star like communication
# - high interference forces shorter links and more relaying
# - the minimum-energy solution is always the minimum hop solution.

# T7 (optional): Comparison with unit hop count
# As an optional extension, compare the energy-aware routing solution with a simpler graph where
# every feasible link has unit cost 1. Discuss how the resulting topology differs from the energy-based
# solution.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def load_nodes(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    nodes = []

    for item in data:
        nodes.append(
            Node(
                item["id"],
                item["room_label"],
                tuple(item["coordinates"])
            )
        )

    print(f"Loaded {len(nodes)} nodes from {filename}")
    print_nodes_table(nodes)
    return nodes

def print_nodes_table(nodes):
    print("Nodes:")
    for node in nodes:
        print(f" {node.id} & {node.room_label} & {node.coordinates} \\\\")

def build_graph_from_link_matrix(
    nodes,
    link_costs
):
    link_costs = np.asarray(link_costs, dtype=float)

    n = len(nodes)
    if link_costs.shape != (n, n):
        raise ValueError(
            f"link_costs must have shape ({n}, {n}), got {link_costs.shape}"
        )

    G = nx.Graph()

    def node_key(i):
        return nodes[i].id

    # Add nodes
    for i, node in enumerate(nodes):
        G.add_node(
            node_key(i),
            object=node,
            room_label=getattr(node, "room_label", None),
            coordinates=getattr(node, "coordinates", None),
        )

    # Add edges
    for i in range(n):
        for j in range(i + 1, n):
            c_ij = link_costs[i, j]
            c_ji = link_costs[j, i]

            if np.isnan(c_ij) and np.isnan(c_ji):
                continue

            # If both directions exist, use the average.
            # If only one exists, use that one.
            if not np.isnan(c_ij) and not np.isnan(c_ji):
                cost = (c_ij + c_ji) / 2.0
            elif not np.isnan(c_ij):
                cost = c_ij
            else:
                cost = c_ji

            G.add_edge(
                node_key(i),
                node_key(j),
                weight=float(cost),
                cost_mw=float(cost),
            )

    return G



def shortest_path_energy(gateway_node_id, graph):
    nodes = graph.nodes()
    path_edges = set()
    paths = []
    for node in nodes:
        if node == gateway_node_id:
            continue
        path = nx.shortest_path(graph, source=node, target=gateway_node_id, weight="weight")
        edges = set(zip(path[:-1], path[1:]))
        edges2 = set(zip(path[1:], path[:-1]))
        path_edges.update(edges)
        path_edges.update(edges2)
        paths.append(path)

    all_edges = set(graph.edges())
    edges_not_in_paths = all_edges - path_edges
    graph.remove_edges_from(edges_not_in_paths)
    return graph, paths

def T1_create_and_print_nodes(grid):
    #select_and_save_nodes(grid)

    nodes = load_nodes("nodes-all.json")
    vis.show_points_on_floorplan(
        "entire_floorplan.png", points=nodes, grid_shape=grid.shape, title="Node locations", show_labels=True, label_mode="id",gateway_id=0
    )
    return nodes

def T2_compute_pr_matrix(grid, nodes):
    interferences = [0, 5, 10] #dBm
    pr_matrix = rf.calculate_pl_matrix(grid, nodes, cell_size=0.25)
    interference_pt_req = {}
    for interference in interferences:
        interference_pt_req[interference] = rf.calculate_pt_req_materix(pr_matrix, interference=interference)
        interference_pt_req[interference][interference_pt_req[interference] > 10] = np.nan #mark infeasible links with NaN
        print("number of infeasible links with interference ", interference, " dBm: ", np.sum(np.isnan(interference_pt_req[interference])))
    return interference_pt_req

def T3_compute_energy_matrix(interference_pt_req):
    packet_size_bytes = 25
    bit_rate = 250 # bits per second
    packet_duration = (packet_size_bytes * 8) / bit_rate # seconds
    interference_energy_matrix = {}
    for interference, pt_req_matrix in interference_pt_req.items():
        pt_mW_matrix = rf.convert_pt_db_to_mw(pt_req_matrix)
        interference_energy_matrix[interference] = rf.energy_per_meassage(pt_mW_matrix, packet_duration)
    return interference_energy_matrix

def T4_build_and_visualize_graph(nodes, interference_energy_matrix, floorplan_path, grid_shape):
    interferences_graphs = {}
    for interference, energy_matrix in interference_energy_matrix.items():
        G = build_graph_from_link_matrix(nodes, energy_matrix)
        interferences_graphs[interference] = G
        vis.visualize_graph_on_floorplan(
            G,
            floorplan_path,
            grid_shape=grid_shape,
            title=f"Connectivity graph with interference {interference} dB",
            show_node_labels=True,
            node_label_attr="id",
            show_edge_labels=True,
            edge_label_attr="cost_mw",
            node_marker="*",
            node_size=200,
            edge_linewidth=1.5,
        )
        for node_id, attrs in G.nodes(data=True):
            print(node_id, attrs["coordinates"])
            break
    path = nx.shortest_path(interferences_graphs[0], source=0, target=1, weight="weight")
    print("Shortest path from node 0 to node 1 with interference 0 dBm: ", path)
    return interferences_graphs

def T5_compute_min_energy_routes(interference_graphs, grid_shape):
    interference_paths = {}
    for interference, graph in interference_graphs.items():
        path_graph, paths = shortest_path_energy(gateway_node_id=0, graph=graph)
        vis.visualize_graph_on_floorplan(
            path_graph,
            "entire_floorplan.png",
            grid_shape=grid_shape,
            title=f"Minimum-energy routes with interference {interference} dB",
            show_node_labels=True,
            node_label_attr="id",
            show_edge_labels=True,
            edge_label_attr="cost_mw",
            node_marker="o",
            node_size=100,
            edge_linewidth=0.8,
        )
        interference_paths[interference] = path_graph, paths
    return interference_paths
    
def calculate_statistics(interference_paths):
    for interference, (graph, paths) in interference_paths.items():
        num_direct = sum(1 for path in paths if len(path) == 2)
        num_relays = set(node for path in paths if len(path) > 2 for node in path[1:-1])
        total_energy = sum(sum(graph.edges[edge]["cost_mw"] for edge in zip(path[:-1], path[1:])) for path in paths)
        hop_counts = [len(path) - 1 for path in paths]
        avg_hop_count = np.mean(hop_counts)
        max_hop_count = np.max(hop_counts)

        print(f"Interference {interference} dB:")
        print(f"  Number of direct connections to gateway: {num_direct}")
        print(f"  Number of nodes acting as relays: {len(num_relays)}")
        print(f"  Total network energy per message collection round: {total_energy:.2f} mW")
        print(f"  Average hop count: {avg_hop_count:.2f}")
        print(f"  Maximum hop count: {max_hop_count}")

def print_paths_with_costs(interference_paths):
    for interference, (graph, paths) in interference_paths.items():
        print(f"Interference {interference} dB:")
        for path in paths:
            path_cost = sum(graph.edges[edge]["cost_mw"] for edge in zip(path[:-1], path[1:]))
            print(f"  Path: {path}, Cost: {path_cost:.2f} mW")

def main():
    grid = np.load("entire_floorplan_grid.npy")

    nodes = T1_create_and_print_nodes(grid)
    interference_pt_req = T2_compute_pr_matrix(grid, nodes)
    interference_energy_matrix = T3_compute_energy_matrix(interference_pt_req)
    interference_graphs = T4_build_and_visualize_graph(nodes, interference_energy_matrix, "entire_floorplan.png", grid.shape)
    interference_paths = T5_compute_min_energy_routes(interference_graphs, grid.shape)
    calculate_statistics(interference_paths)
    print_paths_with_costs(interference_paths)

if __name__ == "__main__":
    main()