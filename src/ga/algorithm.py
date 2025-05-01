from typing import Iterable
import random

from deap import (
    base,
    tools,
)


def _get_fittest_individ(population):
    return tools.selBest(population, 1)[0]


def algorithm(
    population: list,
    toolbox: base.Toolbox,
    cxpb: float,
    mutpb: float,
    stats: tools.Statistics,
) -> Iterable[tuple[list, tools.Logbook]]:

    logbook = tools.Logbook()
    logbook.header = ["generation_index", "fittest_individ"] + stats.fields

    fitnesses = list(map(toolbox.evaluate, population))
    for individ, fitness in zip(population, fitnesses):
        individ.fitness.values = fitness

    generation_index = 0

    fittest_individ = _get_fittest_individ(population)
    record = stats.compile(population)
    logbook.record(generation_index=generation_index, fittest_individ=fittest_individ, **record)

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

        fittest_individ = _get_fittest_individ(population)
        record = stats.compile(population)
        logbook.record(generation_index=generation_index, fittest_individ=fittest_individ, **record)

        yield population, logbook
