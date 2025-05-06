import random
from typing import Callable

from deap.tools import Logbook
import streamlit as st

from ga.toolbox import get_toolbox
from ga.stats import get_stats 
from ga.algorithm import algorithm
from settings import Settings
from ui import (
    ui_settings,
    ui_main,
)


st.set_page_config(layout="wide")

settings_layout = st.sidebar
data_layout = st.container()

settings = ui_settings(settings_layout)


def run_algorithm(
    nodes,
    demand_matrix,
    settings: Settings,
    iteration_callback: Callable[[list, Logbook], None]
):


    def termitation_criteria(stats: dict):
        return (settings.is_fittnes_value_limited and stats["min"] <= settings.min_fittness_value) or \
            (settings.is_generations_amount_limited and stats["generation_index"] >= settings.max_generations_amount) or \
            (False)


    if settings.is_seed_fixed:
        random.seed(settings.seed)

    toolbox = get_toolbox(
        nodes,
        demand_matrix,
        settings.routes_amount,
        settings.full_length_weight,
        settings.demand_coverage_weight
    )

    stats = get_stats()

    for population, logbook in algorithm(
        population=toolbox.population(n=settings.population_size), # type: ignore
        toolbox=toolbox,
        cxpb=settings.crossover_probability,
        mutpb=settings.mutation_probability,
        stats=stats,
    ):

        iteration_callback(population, logbook)

        if termitation_criteria(logbook[-1]):
            break


ui_main(data_layout, settings, run_algorithm)
