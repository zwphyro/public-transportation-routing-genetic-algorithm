from dataclasses import dataclass
from typing import Tuple
import math

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import colors
import networkx as nx

colors.to_hex

def distance(
    node1: Tuple[float, float],
    node2: Tuple[float, float]
) -> float:
    return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


@dataclass()
class Palette:
    node: str = "#FAD7A0"
    depot: str = "#AED6F1"

    edge_colors = [colors.to_hex(color) for color in colormaps["Set1"].colors] # type: ignore

    @property
    def edge(self):
        while True:
            for color in self.edge_colors:
                yield color 


def plot_individ(
    nodes: dict[int, Tuple[float, float]],
    routes: list[list[int]]
):

    depot = 0
    palette = Palette()

    graph = nx.Graph()

    graph.add_nodes_from([(node, {"position": nodes[node]}) for node in nodes])
    fig, ax = plt.subplots()

    positions = nx.get_node_attributes(graph, "position")

    nx.draw_networkx_nodes(graph, positions, nodelist=[node for node in nodes if node != depot], node_color=palette.node, ax=ax)
    nx.draw_networkx_nodes(graph, positions, nodelist=[0], node_color=palette.depot, ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=8, font_color="black", ax=ax)

    color_generator = palette.edge

    initial_width = 2.5
    result_width = 1 
    width_step = (initial_width - result_width) / len(routes)

    current_width = initial_width

    for route in routes:
        edgelist = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        current_edge_color = next(color_generator)
        nx.draw_networkx_edges(graph, positions, edgelist=edgelist, edge_color=current_edge_color, width=current_width, ax=ax)
        depot_edge = (depot, route[0]) if distance(nodes[depot], nodes[route[0]]) < distance(nodes[depot], nodes[route[-1]]) else (route[-1], depot)
        nx.draw_networkx_edges(graph, positions, edgelist=[depot_edge], edge_color=current_edge_color, style="dashed", width=current_width, ax=ax)
        current_width -= width_step

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
    nx.draw_networkx_edge_labels(graph, positions, edge_labels={(i, j): graph[i][j]["demand"] for i, j in graph.edges()}, label_pos=0.7, ax=ax)

    return fig, ax
