from deap import (
    tools,
)
import numpy as np


def get_stats() -> tools.Statistics:

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)

    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("mean", np.mean)
    stats.register("std", np.std)

    return stats

