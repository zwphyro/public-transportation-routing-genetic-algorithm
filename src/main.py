import random
import json

import streamlit as st

from settings import default_settings as settings
from plotting import plot_individ, plot_structure
from ga.toolbox import get_toolbox, _routes
from ga.stats import stats
from ga.algorithm import algorithm


st.set_page_config(layout="wide")

if "nodes" not in st.session_state:
    st.session_state.nodes = None

if "demand_matrix" not in st.session_state:
    st.session_state.demand_matrix = None

if "logbook" not in st.session_state:
    st.session_state.logbook = None

settings_layout = st.sidebar
data_layout = st

settings_layout.markdown("# Настройки")


settings_layout.markdown("## Начальная популяция")

population_size = settings_layout.number_input("Размер популяции", 1, value=settings.population_size)


settings_layout.markdown("## Выполнение")

is_seed_fixed = settings_layout.checkbox("Фиксировать начальное случайное значение", value=settings.is_seed_fixed)
seed = settings_layout.number_input("Начальное случайное значение", 1, value=settings.seed, disabled=not is_seed_fixed)

xover_probability = settings_layout.slider("Вероятность кроссовера", min_value=0.0, max_value=1.0, value=settings.xover_probability, step=0.01)
mutation_probability = settings_layout.slider("Вероятность мутации", min_value=0.0, max_value=1.0, value=settings.mutation_probability, step=0.01)


settings_layout.markdown("## Решение")

routes_amount = settings_layout.number_input("Количество путей", 1, value=3)


settings_layout.markdown("## Функция приспособленности")

full_length_weight = settings_layout.number_input("Вес длины всех мершрутов", 0.0, value=1.0, step=0.01)
demand_coverage_weight = settings_layout.number_input("Вес покрытия спроса", 0.0, value=1.0, step=0.01)


settings_layout.markdown("## Критерий остановки")

is_generations_amount_limited = settings_layout.checkbox("Ограничить максимальное количество поколений", value=settings.is_generations_amount_limited)
generations_amount = settings_layout.number_input("Количество поколений", 1, value=settings.generations_amount, disabled=not is_generations_amount_limited)

is_fittnes_value_limited = settings_layout.checkbox("Ограничить минимальное значение функции приспособленности", value=settings.is_fittnes_value_limited)
min_fittness_value = settings_layout.number_input("Минимальное значение функции приспособленности", 0, value=settings.min_fittness_value, disabled=not is_fittnes_value_limited)


settings_layout.markdown("## Построение графика")
plotting_generations = settings_layout.number_input("Частота обновления графика в поколениях", 1, value=1)


def termitation_criteria(stats):
    return (is_fittnes_value_limited and stats["min"] <= min_fittness_value) or \
           (is_generations_amount_limited and stats["generation_index"] >= generations_amount) or \
           (False)


data_layout.markdown("# Выполнение алгоритма")

graph_file = data_layout.file_uploader("Загрузить датасет", type=["json"])


if graph_file is not None:

    graph = json.loads(graph_file.getvalue().decode("utf-8"))

    st.session_state.nodes = {int(node): position for node, position in graph["nodes"].items()}
    st.session_state.demand_matrix = graph["demand_matrix"]

    with st.popover("Просмотреть датасет"):
        fig, ax = plot_structure(st.session_state.nodes, st.session_state.demand_matrix)
        st.pyplot(fig)

    start_execution = data_layout.button("Начать выполнение")

    metrics_column, individ_column = st.columns(2)
    generation_container = metrics_column.empty()
    min_metric_layout, max_metric_layout = metrics_column.columns(2)
    mean_metric_layout, std_metric_layout = metrics_column.columns(2)

    individ_container = individ_column.empty()

    min_metric_container = min_metric_layout.empty()
    max_metric_container = max_metric_layout.empty()
    mean_metric_container = mean_metric_layout.empty()
    std_metric_container = std_metric_layout.empty()


    live_chart_container = metrics_column.empty()

    chart_container = metrics_column.empty()
    dataframe_container = data_layout.empty()


    def update_chart(chart, stats):
        chart.add_rows({
            "min": [stats["min"]],
            "max": [stats["max"]],
            "mean": [stats["mean"]],
            "std": [stats["std"]],
        })


    def update_metrics(
        generation_container,
        min_metric_container,
        max_metric_container,
        mean_metric_container,
        std_metric_container,
        stats
    ):

        generation_container.metric("Поколение", stats['generation_index'])
        min_metric_container.metric("min", f"{stats['min']:.2f}")
        max_metric_container.metric("max", f"{stats['max']:.2f}")
        mean_metric_container.metric("mean", f"{stats['mean']:.2f}")
        std_metric_container.metric("std", f"{stats['std']:.2f}")


    if start_execution:

        chart_container.empty()
        dataframe_container.empty()
        
        chart = live_chart_container.line_chart()

        if is_seed_fixed:
            random.seed(seed)

        toolbox = get_toolbox(
            st.session_state.nodes,
            st.session_state.demand_matrix,
            routes_amount,
            full_length_weight,
            demand_coverage_weight
        )

        for population, logbook in algorithm(
            population=toolbox.population(n=population_size), # type: ignore
            toolbox=toolbox,
            cxpb=xover_probability,
            mutpb=mutation_probability,
            stats=stats,
        ):

            st.session_state.logbook = logbook

            if logbook[-1]["generation_index"] % plotting_generations == 0:
                update_chart(chart, logbook[-1])

            update_metrics(
                generation_container,
                min_metric_container,
                max_metric_container,
                mean_metric_container,
                std_metric_container,
                logbook[-1]
            )

            if termitation_criteria(logbook[-1]):
                break

        live_chart_container.empty()


    if st.session_state.logbook is not None:
        update_metrics(
            generation_container,
            min_metric_container,
            max_metric_container,
            mean_metric_container,
            std_metric_container,
            st.session_state.logbook[-1]
        )
        chart_container.line_chart(st.session_state.logbook, x="generation_index", y=["min", "max", "mean", "std"])
        dataframe_container.dataframe(st.session_state.logbook)

        individ = st.session_state.logbook[-1]["fittest_individ"]
        fig, ax = plot_individ(st.session_state.nodes, _routes(individ))
        individ_container.pyplot(fig)
