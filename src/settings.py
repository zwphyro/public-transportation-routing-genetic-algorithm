from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    is_seed_fixed: bool = Field(default=True)
    seed: int = Field(default=1337)
    population_size: int = Field(default=100)
    generations_amount: int = Field(default=200)
    is_generations_amount_limited: bool = Field(default=True)
    xover_probability: float = Field(default=0.5)
    mutation_probability: float = Field(default=0.2)
    is_fittnes_value_limited: bool = Field(default=True)
    min_fittness_value: int = Field(default=0)


default_settings = Settings()
