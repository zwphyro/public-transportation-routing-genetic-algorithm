import random
from typing import Tuple

import networkx as nx
from deap import (
    base,
    creator,
    tools,
)

from .utils import (
    distance as calculate_distance,
    individual_to_routes,
)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def _generate_individual(
    nodes: dict[int, Tuple[float, float]],
    routes_amount: int
) -> list[int | None]:

    depot = 0

    nodes_list = [node for node in nodes if node != depot]
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
    nodes: dict[int, Tuple[float, float]],
    route: list[int]
) -> float:

    depot = 0

    if len(route) == 0:
        return float("inf")

    distance = 0

    for i in range(len(route) - 1):

        distance += calculate_distance(nodes[route[i]], nodes[route[i + 1]])

    return distance + min(calculate_distance(nodes[route[0]], nodes[depot]), calculate_distance(nodes[route[-1]], nodes[depot]))


def _coverage(
    nodes: dict[int, Tuple[float, float]],
    demand_matrix: list[list[int]],
    routes: list[list[int]]
) -> float:

    depot = 0

    coverage = 0

    graph = nx.Graph()

    for node in nodes:
        graph.add_node(node)

    for route in routes:
        if len(route) == 0:
            continue

        for i in range(len(route) - 1):
            graph.add_edge(route[i], route[i + 1], distance=calculate_distance(nodes[route[i]], nodes[route[i + 1]]))

        if calculate_distance(nodes[route[0]], nodes[depot]) < calculate_distance(nodes[route[-1]], nodes[depot]):
            graph.add_edge(route[0], depot, distance=calculate_distance(nodes[route[0]], nodes[depot]))
        else:
            graph.add_edge(route[-1], depot, distance=calculate_distance(nodes[route[-1]], nodes[depot]))


    shortest_paths = dict(nx.all_pairs_all_shortest_paths(graph, weight="distance"))

    for source in nodes:
        for target in nodes:

            if demand_matrix[source][target] == 0 or len(shortest_paths[source][target]) == 0:
                continue

            coverage += demand_matrix[source][target] / _route_distance(nodes, shortest_paths[source][target][0])

    return coverage


def _fitness(
    individual: list[int | None],
    *,
    nodes: dict[int, Tuple[float, float]],
    demand_matrix: list[list[int]],
    total_distance_weight: float=1,
    demand_coverage_weight: float=1
) -> Tuple[float]:

    routes = individual_to_routes(individual)

    total_distance = sum(_route_distance(nodes, route) for route in routes)
    coverage = _coverage(nodes, demand_matrix, routes)

    fitness = total_distance_weight * total_distance - demand_coverage_weight * coverage

    return fitness,


def _repair_missing_values(
    nodes: dict[int, Tuple[float, float]],
    routes_amount: int,
    individual: list[int | None]
) -> list[int | None]:

    depot = 0
    routes_found = 1

    entries = {node: 0 for node in nodes if node != depot}

    for node in individual:
        if node is None:
            routes_found += 1
            continue

        entries[node] += 1

    missing = set(node for node in nodes
                  if node != depot and entries[node] == 0)

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


def _repair_cycles(individual: list[int | None]) -> list[int | None]:

    i = 0
    while i < len(individual) - 1:
        if individual[i] == individual[i + 1] and individual[i] is not None:
            individual.pop(i)
            continue
        i += 1

    return individual


def _repair(
    individual: list[int | None],
    *,
    nodes: dict[int, Tuple[float, float]],
    routes_amount: int
) -> list[int | None]:

    individual = _repair_missing_values(nodes, routes_amount, individual)
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
    individual: list[int | None],
    *,
    nodes: dict[int, Tuple[float, float]],
    indpb: float
):

    depot = 0

    used_nodes = [node for node in nodes if node != depot]

    i = 0
    while i < len(individual):

        if random.random() < indpb:
            new_node = random.choice(used_nodes)
            individual.insert(i, new_node)

        i += 1

    return individual


def get_toolbox(
    nodes: dict[int, Tuple[float, float]],
    demand_matrix: list[list[int]],
    routes_amount: int,
    total_distance_weight: float,
    demand_coverage_weight: float
):

    toolbox = base.Toolbox()

    toolbox.register("generate_individual", _generate_individual, nodes, routes_amount)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        _fitness,
        nodes=nodes,
        demand_matrix=demand_matrix,
        total_distance_weight=total_distance_weight,
        demand_coverage_weight=demand_coverage_weight
    )
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutInversion)
    toolbox.register(
        "mutate_insert",
        _mutate_insert,
        nodes=nodes,
        indpb=0.1
    )
    toolbox.register("select", tools.selRoulette)
    toolbox.register(
        "repair",
        _repair,
        nodes=nodes,
        routes_amount=routes_amount
    )
    toolbox.register("select_population", _select_population)

    return toolbox
