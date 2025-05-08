from dataclasses import dataclass
from typing import Tuple
import math

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import colors
import networkx as nx


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
    graph: nx.DiGraph,
    routes: list[list[int]],
    *,
    node_size: int=300,
    font_size: int=12
):

    depot = [node for node in graph if graph.nodes[node]["is_depot"]][0]
    palette = Palette()

    routes_graph = nx.Graph()

    routes_graph.add_nodes_from([(node, {"position": graph.nodes[node]["position"]}) for node in graph])
    fig, ax = plt.subplots()

    positions = nx.get_node_attributes(routes_graph, "position")

    nx.draw_networkx_nodes(
        routes_graph,
        positions,
        nodelist=[node for node in routes_graph if node != depot],
        node_size=node_size,
        node_color=palette.node,
        ax=ax
    )
    nx.draw_networkx_nodes(
        routes_graph,
        positions,
        nodelist=[depot],
        node_size=node_size,
        node_color=palette.depot,
        ax=ax
    )
    nx.draw_networkx_labels(
        routes_graph,
        positions,
        font_size=font_size,
        ax=ax
    )

    color_generator = palette.edge

    initial_width = 2.5
    result_width = 1 
    width_step = (initial_width - result_width) / len(routes)

    current_width = initial_width

    for route in routes:
        edgelist = [(route[i], route[i + 1]) for i in range(len(route) - 1)]
        current_edge_color = next(color_generator)
        nx.draw_networkx_edges(
            routes_graph,
            positions,
            edgelist=edgelist,
            edge_color=current_edge_color,
            width=current_width,
            node_size=node_size,
            ax=ax
        )

        direct_depot_distance = graph.edges[depot, route[0]]["distance"] + graph.edges[route[0], depot]["distance"]
        reverse_depot_distance = graph.edges[depot, route[-1]]["distance"] + graph.edges[route[-1], depot]["distance"]

        depot_edge = (depot, route[0]) if direct_depot_distance < reverse_depot_distance else (route[-1], depot)

        nx.draw_networkx_edges(
            routes_graph,
            positions,
            edgelist=[depot_edge],
            edge_color=current_edge_color,
            style="dashed",
            width=current_width,
            node_size=node_size,
            ax=ax
        )
        current_width -= width_step

    return fig, ax


def plot_structure(
    graph: nx.DiGraph,
    *,
    node_size: int=300,
    font_size: int=12,
    display_connections: bool=False
):

    palette = Palette()

    fig, ax = plt.subplots()

    positions = nx.get_node_attributes(graph, "position")

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[node for node in graph if not graph.nodes[node]["is_depot"]],
        node_size=node_size,
        node_color=palette.node,
        ax=ax
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=[node for node in graph if graph.nodes[node]["is_depot"]],
        node_size=node_size,
        node_color=palette.depot,
        ax=ax
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=font_size,
        ax=ax
    )
    if display_connections:
        distance_edges = [edge for edge in graph.edges() if graph.edges[edge]["distance"] is not None] 
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=distance_edges,
            edge_color="#CCCCCC05",
            node_size=node_size,
            width=2,
            ax=ax
        )
    demand_edges = [edge for edge in graph.edges() if graph.edges[edge]["demand"] != 0]
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=demand_edges,
        node_size=node_size,
        width=2,
        ax=ax
    )
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels={edge: graph.edges[edge]["demand"] for edge in demand_edges},
        label_pos=0.7,
        font_size=font_size,
        ax=ax
    )


    return fig, ax
