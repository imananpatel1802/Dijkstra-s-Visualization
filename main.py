####
# Europe Rail Pathfinder – Python Project
# Author: Your Name
# Description: This project demonstrates Dijkstra's shortest path algorithm on a sample European rail network.
# It visualizes the full graph with Graphviz and animates the pathfinding process using NetworkX and Matplotlib.
####

import os
import csv
import heapq
from graphviz import Graph
import networkx as nx
import matplotlib.pyplot as plt
import time

# Fix Graphviz PATH for Windows (if needed)
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# ---------------------------
# Step 1: Load graph from CSV
# ---------------------------
def load_graph(stations_file, edges_file):
    graph = {}
    train_types = {}  # kept for compatibility, but not used in labels anymore

    with open(stations_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            graph[row['Station']] = {}

    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start, end = row['From'], row['To']
            time_cost = float(row['Time'])
            t_type = row['TrainType']
            graph[start][end] = time_cost
            train_types[(start, end)] = t_type

    return graph, train_types

# ---------------------------
# Step 2: Dijkstra algorithm
# ---------------------------
def dijkstra_steps(graph, start, end):

    """
    Dijkstra's algorithm implementation.
    Returns:
        distance: total distance to destination
        path: list of stations in shortest path
        steps: list of visited nodes and path at each step (for animation)
    """

    queue = [(0, start, [])]  # distance, node, path
    visited = set()
    steps = []

    while queue:
        dist, current, path = heapq.heappop(queue)
        if current in visited:
            continue
        path = path + [current]
        visited.add(current)
        steps.append((current, path.copy()))

        if current == end:
            return dist, path, steps

        for neighbor, weight in graph.get(current, {}).items():
            if neighbor not in visited:
                heapq.heappush(queue, (dist + weight, neighbor, path))

    return float('inf'), [], steps

# ---------------------------
# Step 3: Graphviz visualization (full graph) – unchanged
# ---------------------------
def visualize_graph(graph, train_types, shortest_path=[]):

    """
    Creates a PDF of the full rail network using Graphviz.
    Highlights the shortest path.
    """

    g = Graph('Rail_Network', filename='rail_network.gv', engine='neato')
    g.attr(overlap='false')
    g.attr(splines='true')
    g.attr(fontsize='10')

    for node in graph:
        g.node(node, label=node,
               color='red' if node in shortest_path else 'lightblue',
               style='filled',
               fontcolor='white' if node in shortest_path else 'black',
               fontsize='10')

    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            label = f"{weight}h ({train_types[(node, neighbor)]})"
            if node in shortest_path and neighbor in shortest_path and abs(shortest_path.index(node) - shortest_path.index(neighbor)) == 1:
                g.edge(node, neighbor, label=label, color='red', penwidth='2', fontsize='10', arrowhead='normal')
            else:
                g.edge(node, neighbor, label=label, color='gray', penwidth='1', fontsize='8',arrowhead='normal')

    g.render(view=True)

# ---------------------------
# Step 4: NetworkX animation – only time shown on edges
# ---------------------------
def animate_dijkstra(graph, shortest_path, steps):

    """
    Animate Dijkstra's algorithm on the rail network using NetworkX.
    Highlights visited nodes and final shortest path.
    """

    G = nx.Graph()
    for node, edges in graph.items():
        for neighbor, weight in edges.items():
            if not G.has_edge(node, neighbor):
                G.add_edge(node, neighbor, weight=weight)

    H = G

    # Better spacing – adjust k if still too crowded (higher = more spread out)
    pos = nx.spring_layout(H, k=0.75, iterations=120, seed=42)

    plt.ion()
    fig, ax = plt.subplots(figsize=(20, 15))  # even larger for crowded graphs

    # Initial draw – everything visible
    nx.draw_networkx_nodes(H, pos, node_color='lightblue', node_size=700, ax=ax)
    nx.draw_networkx_edges(H, pos, alpha=0.35, ax=ax)

    # Node labels always visible
    labels = nx.draw_networkx_labels(H, pos, font_size=10, font_color='black', ax=ax)

    # Edge labels: only time (e.g. "2.0h")
    edge_labels = {(u, v): f"{d['weight']:.1f}h" for u, v, d in H.edges(data=True)}
    nx.draw_networkx_edge_labels(
        H, pos,
        edge_labels=edge_labels,
        font_color='gray',
        font_size=9,
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='none', alpha=0.75),
        ax=ax
    )

    # Animation: highlight visited nodes
    for node, path in steps:
        if node in H:
            nx.draw_networkx_nodes(H, pos, nodelist=[node], node_color='yellow', node_size=900, ax=ax)
            fig.canvas.draw()
            plt.pause(0.4)

    # Final shortest path highlight
    path_edges = [(shortest_path[i], shortest_path[i+1]) for i in range(len(shortest_path)-1)]
    nx.draw_networkx_nodes(H, pos, nodelist=shortest_path, node_color='red', node_size=1300, ax=ax)
    nx.draw_networkx_edges(H, pos, edgelist=path_edges, edge_color='red', width=3.5, ax=ax)

    # Change color of path node labels to white (no new labels drawn)
    for node in shortest_path:
        if node in labels:
            labels[node].set_color('black')
            labels[node].set_fontweight('bold')
            labels[node].set_zorder(10)

    # Highlight path edge labels in red
    path_edge_labels = {(u, v): f"{d['weight']:.1f}h" for u, v, d in H.edges(data=True) if (u, v) in path_edges or (v, u) in path_edges}
    nx.draw_networkx_edge_labels(
        H, pos,
        edge_labels=path_edge_labels,
        font_color='red',
        font_size=10,
        font_weight='bold',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='none', alpha=0.85),
        ax=ax
    )

    plt.axis('off')
    fig.canvas.draw()
    plt.ioff()
    plt.show()

# ---------------------------
# Step 5: Main program
# ---------------------------
if __name__ == "__main__":
    stations_file = 'stations.csv'
    edges_file = 'edges.csv'

    graph, train_types = load_graph(stations_file, edges_file)

    print("Welcome to the Europe Rail Pathfinder!\n")
    print("Available stations:", ', '.join(sorted(graph.keys())))

    start = input("Enter the starting station: ").strip()
    while start not in graph:
        start = input("Station not found. Enter a valid starting station: ").strip()

    end = input("Enter the destination station: ").strip()
    while end not in graph:
        end = input("Station not found. Enter a valid destination station: ").strip()

    distance, path, steps = dijkstra_steps(graph, start, end)

    if path:
        print(f"\nShortest path from {start} to {end}: {' → '.join(path)}")
        print(f"Total time: {distance:.1f} hours")
        visualize_graph(graph, train_types, shortest_path=path)
        animate_dijkstra(graph, path, steps)
    else:
        print(f"No path found from {start} to {end}.")