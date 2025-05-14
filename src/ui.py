import json
import time

from deap.tools import Logbook
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import networkx as nx

from settings import (
    Settings,
    default_settings,
)
from plotting import (
    plot_individ,
    plot_structure,
)

from ga.utils import individual_to_routes
from ga.algorithm import run_algorithm


def ui_settings(layout: DeltaGenerator) -> Settings:
    layout.markdown("# Настройки")


    layout.markdown("## Начальная популяция")

    population_size = layout.number_input(
        "Размер популяции",
        min_value=1,
        value=default_settings.population_size
    )


    layout.markdown("## Выполнение алгоритма")

    is_seed_fixed = layout.checkbox(
        "Фиксировать начальное случайное значение",
        value=default_settings.is_seed_fixed
    )
    seed = layout.number_input(
        "Начальное случайное значение",
        min_value=1,
        value=default_settings.seed,
        disabled=not is_seed_fixed
    )

    crossover_probability = layout.slider(
        "Вероятность кроссовера",
        min_value=0.0,
        max_value=1.0,
        value=default_settings.crossover_probability,
        step=0.01
    )
    mutation_probability = layout.slider(
        "Вероятность мутации",
        min_value=0.0,
        max_value=1.0,
        value=default_settings.mutation_probability,
        step=0.01
    )


    layout.markdown("## Решение")

    routes_amount = layout.number_input(
        "Количество путей",
        min_value=1,
        value=default_settings.routes_amount
    )


    layout.markdown("## Функция приспособленности")

    full_length_weight = layout.number_input(
        "Вес длины всех мершрутов",
        min_value=0.0,
        value=default_settings.full_length_weight,
        step=0.01
    )
    demand_coverage_weight = layout.number_input(
        "Вес покрытия спроса",
        min_value=0.0,
        value=default_settings.demand_coverage_weight,
        step=0.01
    )


    layout.markdown("## Критерий остановки")

    is_generations_amount_limited = layout.checkbox(
        "Ограничить максимальное количество поколений",
        value=default_settings.is_generations_amount_limited
    )
    max_generations_amount = layout.number_input(
        "Количество поколений",
        min_value=1,
        value=default_settings.max_generations_amount,
        disabled=not is_generations_amount_limited
    )

    is_fittnes_value_limited = layout.checkbox(
        "Ограничить минимальное значение функции приспособленности",
        value=default_settings.is_fittnes_value_limited
    )
    min_fittness_value = layout.number_input(
        "Минимальное значение функции приспособленности",
        min_value=0,
        value=default_settings.min_fittness_value,
        disabled=not is_fittnes_value_limited
    )


    layout.markdown("## Построение графика")
    plot_updating_frequency = layout.number_input(
        "Частота обновления графика в поколениях",
        min_value=default_settings.plot_updating_frequency,
        value=1
    )

    return Settings(
        population_size=population_size,

        is_seed_fixed=is_seed_fixed,
        seed=seed,

        crossover_probability=crossover_probability,
        mutation_probability=mutation_probability,

        routes_amount=routes_amount,

        full_length_weight=full_length_weight,
        demand_coverage_weight=demand_coverage_weight,

        is_generations_amount_limited=is_generations_amount_limited,
        max_generations_amount=max_generations_amount,

        is_fittnes_value_limited=is_fittnes_value_limited,
        min_fittness_value=min_fittness_value,

        plot_updating_frequency=plot_updating_frequency,
    )


def ui_main(
    layout: DeltaGenerator,
    settings: Settings
) -> None:

    if "graph" not in st.session_state:
        st.session_state.graph = None

    if "logbook" not in st.session_state:
        st.session_state.logbook = None

    if "time_diff" not in st.session_state:
        st.session_state.time_diff = None

    layout.markdown("# Выполнение алгоритма")

    graph_file = layout.file_uploader("Загрузить датасет", type=["json"])


    if graph_file is None:
        st.session_state.graph = None
        return

    try:
        graph = nx.node_link_graph(json.loads(graph_file.getvalue().decode("utf-8")), edges="edges")

        max_distance = max(graph.edges[edge]["distance"] for edge in graph.edges if graph.edges[edge]["distance"] is not None)

        for edge in graph.edges:

            if graph.edges[edge]["distance"] is not None:
                graph.edges[edge]["connected"] = True
                continue

            graph.edges[edge]["distance"] = max_distance * graph.number_of_nodes()
            graph.edges[edge]["connected"] = False

    except:
        st.session_state.graph = None
        layout.error("Неверный формат датасета")
        return

    st.session_state.graph = graph

    with layout.popover("Просмотреть датасет"):
        fig, _ = plot_structure(st.session_state.graph, node_size=100, font_size=5)
        st.pyplot(fig)

    start_execution = layout.button("Начать выполнение")

    metrics_column, individ_column = layout.columns(2)
    generation_layout, time_layout = metrics_column.columns(2)
    min_metric_layout, max_metric_layout = metrics_column.columns(2)
    mean_metric_layout, std_metric_layout = metrics_column.columns(2)

    individ_container = individ_column.empty()

    generation_container = generation_layout.empty()
    time_container = time_layout.empty()
    min_metric_container = min_metric_layout.empty()
    max_metric_container = max_metric_layout.empty()
    mean_metric_container = mean_metric_layout.empty()
    std_metric_container = std_metric_layout.empty()


    def update_metrics(stats):
        generation_container.metric("Поколение", stats['generation_index'])
        min_metric_container.metric("min", f"{stats['min']:.2f}")
        max_metric_container.metric("max", f"{stats['max']:.2f}")
        mean_metric_container.metric("mean", f"{stats['mean']:.2f}")
        std_metric_container.metric("std", f"{stats['std']:.2f}")


    live_chart_container = metrics_column.empty()

    chart_container = metrics_column.empty()
    dataframe_container = layout.empty()


    def update_chart(chart, stats):
        chart.add_rows({
            "min": [stats["min"]],
            "max": [stats["max"]],
            "mean": [stats["mean"]],
            "std": [stats["std"]],
        })


    if start_execution:

        chart_container.empty()
        time_container.empty()
        dataframe_container.empty()

        chart = live_chart_container.line_chart()

        def iteration_callback(population: list, logbook: Logbook):

            st.session_state.logbook = logbook
            current_stats = logbook[-1]

            if logbook[-1]["generation_index"] % settings.plot_updating_frequency == 0:
                update_chart(chart, current_stats)

            update_metrics(current_stats)

        start_time = time.time()

        run_algorithm(
            st.session_state.graph,
            settings,
            iteration_callback
        )
        
        end_time = time.time()

        st.session_state.time_diff = end_time - start_time

        live_chart_container.empty()


    if st.session_state.logbook is None:
        return


    update_metrics(st.session_state.logbook[-1])
    chart_container.line_chart(st.session_state.logbook, x="generation_index", y=["min", "max", "mean", "std"])
    dataframe_container.dataframe(st.session_state.logbook)

    individual = st.session_state.logbook[-1]["fittest_individual"]
    fig, _ = plot_individ(st.session_state.graph, individual_to_routes(individual), node_size=100, font_size=5)
    individ_container.pyplot(fig)

    if st.session_state.time_diff is not None:
        time_container.metric("Время выполнения", f"{st.session_state.time_diff:.2f} с")
