from dataclasses import dataclass
import random
import json
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import networkx as nx



def distance(
    node1: Tuple[float, float],
    node2: Tuple[float, float]
) -> float:
    return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def generate_graph(
    nodes_amount: int,
    size: Tuple[int, int],
    demand_density: float,
    demand_range: Tuple[int, int],
    seed: int | None = None
):

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    width, height = size

    nodes = {i: (random.uniform(0, width), random.uniform(0, height)) for i in range(nodes_amount)}

    demand_matrix = np.zeros((nodes_amount, nodes_amount), dtype=int)
    for i in range(nodes_amount):
        for j in range(nodes_amount):
            if i != j and random.random() < demand_density:
                demand_matrix[i][j] = random.randint(*demand_range)

    return nodes, demand_matrix


@dataclass()
class Palette:
    node: str = "#FAD7A0"
    depot: str = "#AED6F1"

    edge_colors = colormaps["Set1"].colors # type: ignore

    @property
    def edge(self):
        while True:
            for color in self.edge_colors:
                yield color 


palette = Palette()

def plot_individ(
    nodes: dict[int, Tuple[float, float]],
    routes: list[list[int]]
):

    depot = 0

    graph = nx.Graph()

    graph.add_nodes_from([(node, {"position": nodes[node]}) for node in nodes])

    for route in routes:
        for i in range(len(route) - 1):
            graph.add_edge(route[i], route[i + 1])

    fig, ax = plt.subplots()

    positions = nx.get_node_attributes(graph, "position")

    nx.draw_networkx_nodes(graph, positions, nodelist=[node for node in nodes if node != depot], node_color=palette.node, ax=ax)
    nx.draw_networkx_nodes(graph, positions, nodelist=[0], node_color=palette.depot, ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=8, font_color="black", ax=ax)

    edge_color = palette.edge

    for route in routes:
        edgelist = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        current_edge_color = next(edge_color)
        nx.draw_networkx_edges(graph, positions, edgelist=edgelist, edge_color=current_edge_color, width=2, ax=ax)
        depot_edge = (depot, route[0]) if distance(nodes[depot], nodes[route[0]]) < distance(nodes[depot], nodes[route[-1]]) else (route[-1], depot)
        nx.draw_networkx_edges(graph, positions, edgelist=[depot_edge], edge_color=current_edge_color, style="dashed", width=2, ax=ax)

    return fig, ax


def plot_structure(
    nodes: dict[int, Tuple[float, float]],
    demand_matrix: list[list[int]],
):

    palette = Palette()
    depot = 0

    graph = nx.DiGraph()

    graph.add_nodes_from([(node, {"position": nodes[node]}) for node in nodes])

    for i in nodes:
        for j in nodes:
            if i != j and demand_matrix[i][j] != 0:
                graph.add_edge(i, j, demand=demand_matrix[i][j])

    fig, ax = plt.subplots()

    positions = nx.get_node_attributes(graph, "position")

    nx.draw_networkx_nodes(graph, positions, nodelist=[node for node in nodes if node != depot], node_color=palette.node, ax=ax)
    nx.draw_networkx_nodes(graph, positions, nodelist=[0], node_color=palette.depot, ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=8, font_color="black", ax=ax)
    nx.draw_networkx_edges(graph, positions, edgelist=graph.edges(), width=2, ax=ax)
    nx.draw_networkx_edge_labels(graph, positions, edge_labels={(i, j): graph[i][j]["demand"] for i, j in graph.edges()}, label_pos=0.3, ax=ax)

    return fig, ax



def draw_graph(nodes, routes):

    palette = [""]

    graph = nx.Graph()

    graph.add_nodes_from([(node, {"position": nodes[node]}) for node in nodes])

    fig, ax = plt.subplots()

    for i, route in enumerate(routes):
        for i in range(len(route) - 1):

            if graph.has_edge(route[i], route[i + 1]):
                graph[route[i]][route[i + 1]]["routes"].append(i)
                continue

            graph.add_edge(route[i], route[i + 1], distance=distance(nodes[route[i]], nodes[route[i + 1]]), routes=[i])

    positions = nx.get_node_attributes(graph, "position")
    nx.draw_networkx_nodes(graph, positions, nodelist=[node for node in nodes if node != 0], node_size=300, node_color="#FAD7A0", ax=ax)
    nx.draw_networkx_nodes(graph, positions, nodelist=[0], node_size=300, node_color="#AED6F1", ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=8, font_color="black", ax=ax)

    for route in routes:
        edgelist = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        nx.draw_networkx_edges(graph, positions, edgelist=edgelist, width=2, ax=ax)

    return fig, ax

# nodes = json.loads('{"0": [51.55498775040024, 967.4335428783833], "1": [347.6984907944436, 987.353245108907], "2": [271.8471834037725, 548.1822363536991], "3": [89.23777876010553, 855.3852616396583], "4": [85.77826676906086, 318.4399161674757], "5": [396.2543880507927, 298.4215434891787], "6": [876.3037049018883, 694.3852418103171], "7": [567.089011467881, 224.50190378051727], "8": [263.25758865425576, 64.1888999983925], "9": [332.1760012372496, 220.4573272100673], "10": [502.95804951070556, 389.64750227517743], "11": [441.108224708265, 526.6692930813657], "12": [579.1955016771377, 571.5996332967328], "13": [791.7641031292524, 932.58545721392], "14": [300.7656085609078, 859.1629906941639], "15": [899.6235801381167, 478.442662790898], "16": [364.85686912646844, 975.5029273406615], "17": [254.64764996313193, 334.7358963265287], "18": [28.117958978541303, 62.70269219363478], "19": [88.2886973475886, 63.23109615034783]}')
# nodes = {int(k): v for k, v in nodes.items()}
# draw_graph(nodes, [[3], [14], [1, 16, 13, 6, 15], [2, 11, 12, 10, 7, 5, 1, 16, 9, 17, 4, 18, 19, 8]])
# plot_individ(nodes, [[3], [14], [1, 16, 13, 6, 15], [2, 11, 12, 10, 7, 5, 16, 13, 9, 17, 4, 18, 19, 8]])
nodes = json.loads('{"0": [866.1416693499021, 598.9165393877729], "1": [719.7162240245636, 187.8474049220439], "2": [715.9002954417289, 408.3280900224735], "3": [546.7861113847151, 64.67257777821334], "4": [27.454399598703883, 869.5196025795204], "5": [609.5005724182835, 36.06036415513203]}')
nodes = {int(node): nodes[node] for node in nodes}
demand_matrix = [
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 40],
    [0, 0, 0, 0, 0, 150],
    [0, 0, 0, 20, 60, 0],
]
plot_structure(nodes, demand_matrix)

plt.show()
