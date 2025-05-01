import random
from typing import Tuple

import networkx as nx
import numpy as np
from deap import (
    base,
    creator,
    tools,
)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def _routes(individual: list[int | None]) -> list[list[int]]:
    routes = [[]]

    for node in individual:
        if node is None:
            routes.append([])
        else:
            routes[-1].append(node)

    return routes


def _individual(routes: list[list[int]]) -> list[int | None]:
    individual = []

    for route in routes:
        individual.extend(route)
        individual.append(None)

    return individual


def _distance(
    node1: Tuple[float, float],
    node2: Tuple[float, float]
) -> float:
    return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


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

        distance += _distance(nodes[route[i]], nodes[route[i + 1]])

    return distance + min(_distance(nodes[route[0]], nodes[depot]), _distance(nodes[route[-1]], nodes[depot]))


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
            graph.add_edge(route[i], route[i + 1], distance=_distance(nodes[route[i]], nodes[route[i + 1]]))

        if _distance(nodes[route[0]], nodes[depot]) < _distance(nodes[route[-1]], nodes[depot]):
            graph.add_edge(route[0], depot, distance=_distance(nodes[route[0]], nodes[depot]))
        else:
            graph.add_edge(route[-1], depot, distance=_distance(nodes[route[-1]], nodes[depot]))


    shortest_paths = dict(nx.all_pairs_all_shortest_paths(graph, weight="distance"))

    for source in nodes:
        for target in nodes:

            if demand_matrix[source][target] == 0 or len(shortest_paths[source][target]) == 0:
                continue

            coverage += demand_matrix[source][target] / _route_distance(nodes, shortest_paths[source][target][0])

    return coverage


def _fitness(
    nodes: dict[int, Tuple[float, float]],
    demand_matrix: list[list[int]],
    individual: list[int | None],
    full_length_weight: float=1,
    demand_coverage_weight: float=1
) -> Tuple[float]:

    routes = _routes(individual)

    distance = sum(_route_distance(nodes, route) for route in routes)

    coverage = _coverage(nodes, demand_matrix, routes)

    fitness = full_length_weight * distance - demand_coverage_weight * coverage

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


# INFO: not quite working...
def _repair_empty_routes(individual: list[int | None]) -> list[int | None]:

    if len(individual) <= 1:
        return individual

    first_non_none = 0
    while first_non_none < len(individual) and individual[first_non_none] is None:
        first_non_none += 1
    if first_non_none != 0:
        individual[0], individual[first_non_none] = individual[first_non_none], individual[0]

    last_non_none = len(individual) - 1
    while last_non_none >= 0 and individual[last_non_none] is None:
        last_non_none -= 1
    if last_non_none != len(individual) - 1:
        individual[-1], individual[last_non_none] = individual[last_non_none], individual[-1]

    i = 1
    while i < len(individual) - 1:

        if individual[i] is not None or individual[i + 1] is not None:
            i += 1
            continue

        j = i + 2
        found = False

        while j < len(individual) - 1:
            if individual[j] is None:
                j += 1
                continue

            individual[i + 1], individual[j] = individual[j], individual[i + 1]
            found = True
            break

        if not found:
            j = i - 1
            while j > 0:
                if individual[j] is None:
                    j -= 1
                    continue

                individual[i], individual[j] = individual[j], individual[i]
                found = True
                break

        if found:
            if individual[-1] is None:
                last = len(individual) - 2
                while last >= 0 and individual[last] is None:
                    last -= 1
                if last >= 0:
                    individual[-1], individual[last] = individual[last], individual[-1]
            i += 1

        i += 1

    return individual


def _repair(
    nodes: dict[int, Tuple[float, float]],
    routes_amount: int, individual: list[int | None]
) -> list[int | None]:

    individual = _repair_missing_values(nodes, routes_amount, individual)
    individual = _repair_cycles(individual)
    # individual = _repair_empty_routes(individual)

    return individual 


def _get_unique(population, k):
    population_hash = [hash(tuple(individ)) for individ in population]

    selected = set()
    result = []

    for individ, hash_value in zip(population, population_hash):
        if hash_value in selected:
            continue

        selected.add(hash_value)
        result.append(individ)

        if len(result) == k:
            break
        
    if len(result) < k:
        result.extend(population[:k - len(result)])

    return result


def _select_population(population, offsprings):
    population_size = len(population)

    new_population = sorted(population + offsprings, key=lambda x: x.fitness.values)

    return _get_unique(new_population, population_size)


def _mutate_insert(individ: list[int | None], nodes: dict[int, Tuple[float, float]], indpb: float):

    depot = 0

    used_nodes = [node for node in nodes if node != depot]

    i = 0
    while i < len(individ):

        if random.random() < indpb:
            new_node = random.choice(used_nodes)
            individ.insert(i, new_node)

        i += 1

    return individ


def get_toolbox(
    nodes: dict[int, Tuple[float, float]],
    demand_matrix: list[list[int]],
    routes_amount: int,
    full_length_weight: float,
    demand_coverage_weight: float
):

    toolbox = base.Toolbox()

    toolbox.register("generate_individual", _generate_individual, nodes, routes_amount)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.generate_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", _fitness, nodes, demand_matrix, full_length_weight=full_length_weight, demand_coverage_weight=demand_coverage_weight)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutInversion)
    toolbox.register("mutate_insert", _mutate_insert, nodes=nodes, indpb=0.1)
    # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    # toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("select", tools.selRoulette)
    toolbox.register("repair", _repair, nodes, routes_amount)
    toolbox.register("select_population", _select_population)

    return toolbox
