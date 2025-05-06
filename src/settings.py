from pydantic import (
    BaseModel,
    Field,
)


class Settings(BaseModel):
    population_size: int = Field(default=100)

    is_seed_fixed: bool = Field(default=True)
    seed: int = Field(default=1337)

    crossover_probability: float = Field(default=0.5)
    mutation_probability: float = Field(default=0.2)

    routes_amount: int = Field(default=3)

    full_length_weight: float = Field(default=1.0)
    demand_coverage_weight: float = Field(default=1.0)

    is_generations_amount_limited: bool = Field(default=True)
    max_generations_amount: int = Field(default=200)

    is_fittnes_value_limited: bool = Field(default=True)
    min_fittness_value: int = Field(default=0)

    plot_updating_frequency: int = Field(default=1)


default_settings = Settings()
