import random
from typing import Tuple, TypeAlias

import networkx as nx
from deap import (
    base,
    creator,
    tools,
)

from .utils import (
    individual_to_routes
)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
Individual: TypeAlias = list[int | None]


def _generate_individual(
    graph: nx.DiGraph,
    routes_amount: int
) -> Individual:

    nodes_list = [node for node in graph if not graph.nodes[node]["is_depot"]]
    random.shuffle(nodes_list)
    
    split_points = sorted(random.sample(range(1, len(nodes_list)), routes_amount - 1))

    individual = []
    last_index = 0
    for split in split_points:
        individual.extend(nodes_list[last_index:split])
        individual.append(None)
        last_index = split
    individual.extend(nodes_list[last_index:])

    return individual


def _route_distance(
    graph: nx.DiGraph,
    route: list[int]
) -> float:

    depot = [node for node in graph if graph.nodes[node]["is_depot"]][0]

    if len(route) == 0:
        return float("inf")

    distance = 0

    for i in range(len(route) - 1):

        distance += graph.edges[route[i], route[i + 1]]["distance"] + \
                    graph.edges[route[i + 1], route[i]]["distance"]

    direct_depot_distance = graph.edges[depot, route[0]]["distance"] + graph.edges[route[0], depot]["distance"]
    reverse_depot_distance = graph.edges[depot, route[-1]]["distance"] + graph.edges[route[-1], depot]["distance"]
    
    depot_distance = min(direct_depot_distance, reverse_depot_distance)

    return distance + depot_distance

def _check_penalty(
    graph: nx.DiGraph,
    route: list[int]
) -> bool:

    depot = [node for node in graph if graph.nodes[node]["is_depot"]][0]

    if len(route) == 0:
        return True 

    for i in range(len(route) - 1):

        if not graph.edges[route[i], route[i + 1]]["connected"] or \
            not graph.edges[route[i + 1], route[i]]["connected"]:
            return True

    direct_depot_distance = graph.edges[depot, route[0]]["distance"] + graph.edges[route[0], depot]["distance"]
    reverse_depot_distance = graph.edges[depot, route[-1]]["distance"] + graph.edges[route[-1], depot]["distance"]

    if direct_depot_distance <= reverse_depot_distance:
        if not graph.edges[depot, route[0]]["connected"] or \
            not graph.edges[route[0], depot]["distance"]:
            return True
    else:
        if not graph.edges[depot, route[-1]]["connected"] or \
            not graph.edges[route[-1], depot]["distance"]:
            return True
    
    return False


def _path_distance(
    graph: nx.DiGraph,
    path: list[int]
) -> float:

    distance = 0

    for i in range(len(path) - 1):
        distance += graph.edges[path[i], path[i + 1]]["distance"]

    return distance


def _coverage(
    graph: nx.DiGraph,
    routes: list[list[int]]
) -> float:

    depot = [node for node in graph if graph.nodes[node]["is_depot"]][0]

    coverage = 0

    routes_graph = nx.DiGraph()

    for node in graph:

        if graph.nodes[node]["is_depot"]:
            continue

        routes_graph.add_node(node)

    for route in routes:
        if len(route) == 0:
            continue

        for i in range(len(route) - 1):
            routes_graph.add_edge(route[i], route[i + 1], distance=graph.edges[route[i], route[i + 1]]["distance"])
            routes_graph.add_edge(route[i + 1], route[i], distance=graph.edges[route[i + 1], route[i]]["distance"])

    shortest_paths = dict(nx.all_pairs_all_shortest_paths(routes_graph, weight="distance"))

    for source, target in [edge for edge in graph.edges if graph.edges[edge]["demand"] != 0]:
        if source == target or not source in shortest_paths or not target in shortest_paths[source] or graph.edges[source, target]["demand"] == 0:
            continue
        coverage += graph.edges[source, target]["demand"] / _path_distance(graph, shortest_paths[source][target][0]) * len(shortest_paths[source][target])

    return coverage


def _fitness(
    individual: Individual,
    *,
    graph: nx.DiGraph,
    total_distance_weight: float=1,
    demand_coverage_weight: float=1
) -> Tuple[float]:

    routes = individual_to_routes(individual)

    total_distance = sum(_route_distance(graph, route) for route in routes)
    penalty = any(_check_penalty(graph, route) for route in routes)

    if penalty:
        return total_distance_weight * total_distance,

    coverage = _coverage(graph, routes)

    fitness = total_distance_weight * total_distance - demand_coverage_weight * coverage

    return fitness,


def _repair_missing_values(
    graph: nx.DiGraph,
    routes_amount: int,
    individual: Individual
) -> Individual:

    routes_found = 1

    entries = {node: 0 for node in graph if not graph.nodes[node]["is_depot"]}

    for node in individual:
        if node is None:
            routes_found += 1
            continue

        entries[node] += 1

    missing = set(node for node in graph 
                  if not graph.nodes[node]["is_depot"] and entries[node] == 0)

    for i in range(len(individual)):

        if len(missing) == 0 and routes_found == routes_amount:
            break

        if len(missing) != 0 and individual[i] is not None and entries[individual[i]] > 1: # type: ignore

            entries[individual[i]] -= 1 # type: ignore
            individual[i] = missing.pop()
            entries[individual[i]] += 1 # type: ignore

            continue

        if len(missing) != 0 and individual[i] is None and routes_found > routes_amount:

            individual[i] = missing.pop()
            routes_found -= 1

            continue

        if routes_found < routes_amount and individual[i] is not None and entries[individual[i]] > 1: # type: ignore

            entries[individual[i]] -= 1 # type: ignore
            individual[i] = None
            routes_found += 1

            continue

    return individual 


def _repair_cycles(individual: Individual) -> Individual:

    i = 0
    while i < len(individual) - 1:
        if individual[i] == individual[i + 1] and individual[i] is not None:
            individual.pop(i)
            continue
        i += 1

    return individual


def _repair(
    individual: Individual,
    *,
    graph: nx.DiGraph,
    routes_amount: int
) -> Individual:

    individual = _repair_missing_values(graph, routes_amount, individual)
    individual = _repair_cycles(individual)

    return individual 


def _get_unique(population: list, k: int):
    population_hash = [hash(tuple(individual)) for individual in population]

    selected = set()
    result = []

    for individual, hash_value in zip(population, population_hash):
        if hash_value in selected:
            continue

        selected.add(hash_value)
        result.append(individual)

        if len(result) == k:
            break
        
    if len(result) < k:
        result.extend(population[:k - len(result)])

    return result


def _select_population(population: list, offsprings: list):
    population_size = len(population)

    new_population = sorted(population + offsprings, key=lambda x: x.fitness.values)

    return _get_unique(new_population, population_size)


def _mutate_insert(
    individual: Individual,
    *,
    graph: nx.DiGraph,
    indpb: float
) -> Individual:

    used_nodes = [node for node in graph if not graph.nodes[node]["is_depot"]]

    i = 0
    while i < len(individual):

        if random.random() < indpb:
            new_node = random.choice(used_nodes)
            individual.insert(i, new_node)

        i += 1

    return individual


def _individual_distance(individual: Individual, *, graph: nx.DiGraph) -> float:

    routes = individual_to_routes(individual)

    return sum(_route_distance(graph, route) for route in routes)

def _individual_penalty(individual: Individual, *, graph: nx.DiGraph) -> bool:

    routes = individual_to_routes(individual)

    return any(_check_penalty(graph, route) for route in routes)

def _individual_coverage(individual: Individual, *, graph: nx.DiGraph) -> float:

    routes = individual_to_routes(individual)

    return _coverage(graph, routes)


def get_toolbox(
    graph: nx.DiGraph,
    routes_amount: int,
    total_distance_weight: float,
    demand_coverage_weight: float
) -> base.Toolbox:

    toolbox = base.Toolbox()

    toolbox.register("generate_individual", _generate_individual, graph, routes_amount)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        _fitness,
        graph=graph,
        total_distance_weight=total_distance_weight,
        demand_coverage_weight=demand_coverage_weight
    )
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutInversion)
    toolbox.register(
        "mutate_insert",
        _mutate_insert,
        graph=graph,
        indpb=0.1
    )
    toolbox.register("select", tools.selRoulette)
    toolbox.register(
        "repair",
        _repair,
        graph=graph,
        routes_amount=routes_amount
    )
    toolbox.register("select_population", _select_population)

    toolbox.register("individual_distance", _individual_distance, graph=graph)
    toolbox.register("individual_penalty", _individual_penalty, graph=graph)
    toolbox.register("individual_coverage", _individual_coverage, graph=graph)

    return toolbox
