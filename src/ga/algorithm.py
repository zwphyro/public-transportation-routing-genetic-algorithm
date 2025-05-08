from typing import Callable, Iterable
import random

from deap import (
    base,
    tools,
)
import networkx as nx

from settings import Settings # type: ignore

from .stats import get_stats
from .toolbox import get_toolbox


def _get_fittest_individual(population: list):
    return tools.selBest(population, 1)[0]


def algorithm(
    population: list,
    toolbox: base.Toolbox,
    cxpb: float,
    mutpb: float,
    stats: tools.Statistics,
) -> Iterable[tuple[list, tools.Logbook]]:

    logbook = tools.Logbook()
    logbook.header = ["generation_index", "fittest_individual"] + stats.fields

    fitnesses = list(map(toolbox.evaluate, population))
    for individ, fitness in zip(population, fitnesses):
        individ.fitness.values = fitness

    generation_index = 0

    fittest_individual = _get_fittest_individual(population)
    record = stats.compile(population)
    logbook.record(generation_index=generation_index, fittest_individual=fittest_individual, **record)

    yield population, logbook

    while True:
        generation_index += 1

        offsprings = toolbox.select(population, len(population))
        offsprings = list(map(toolbox.clone, offsprings))

        for child1, child2 in zip(offsprings[::2], offsprings[1::2]):

            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                toolbox.repair(child1)
                toolbox.repair(child2)

                del child1.fitness.values
                del child2.fitness.values

        for mutant in offsprings:

            if random.random() < mutpb:
                toolbox.mutate(mutant)
                toolbox.repair(mutant)

                del mutant.fitness.values

            if random.random() < mutpb:
                toolbox.mutate_insert(mutant)
                toolbox.repair(mutant)

                del mutant.fitness.values

        invalid_individs = [individ for individ in offsprings if not individ.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_individs)
        for individ, fitness in zip(invalid_individs, fitnesses):
            individ.fitness.values = fitness

        new_population = toolbox.select_population(population, offsprings)
        population[:] = new_population

        fittest_individual = _get_fittest_individual(population)
        record = stats.compile(population)
        logbook.record(generation_index=generation_index, fittest_individual=fittest_individual, **record)

        yield population, logbook


def run_algorithm(
    graph: nx.DiGraph,
    settings: Settings,
    iteration_callback: Callable[[list, tools.Logbook], None]
):


    def termitation_criteria(stats: dict):
        return (settings.is_fittnes_value_limited and stats["min"] <= settings.min_fittness_value) or \
            (settings.is_generations_amount_limited and stats["generation_index"] >= settings.max_generations_amount) or \
            (False)


    if settings.is_seed_fixed:
        random.seed(settings.seed)

    toolbox = get_toolbox(
        graph,
        settings.routes_amount,
        settings.full_length_weight,
        settings.demand_coverage_weight
    )

    stats = get_stats()

    for population, logbook in algorithm(
        population=toolbox.population(n=settings.population_size),
        toolbox=toolbox,
        cxpb=settings.crossover_probability,
        mutpb=settings.mutation_probability,
        stats=stats,
    ):

        iteration_callback(population, logbook)

        if termitation_criteria(logbook[-1]):
            break
