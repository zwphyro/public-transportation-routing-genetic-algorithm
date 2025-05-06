import math
from typing import Tuple


def individual_to_routes(individual: list[int | None]) -> list[list[int]]:
    routes = [[]]

    for node in individual:
        if node is None:
            routes.append([])
        else:
            routes[-1].append(node)

    return routes


def routes_to_individual(routes: list[list[int]]) -> list[int | None]:
    individual = []

    for route in routes:
        individual.extend(route)
        individual.append(None)

    return individual


def distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
