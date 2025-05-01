FROM python:3.13.3-slim

WORKDIR /code

RUN apt-get update && apt-get install -y \
    build-essential

ENV POETRY_VIRTUALENVS_CREATE=false

COPY ./pyproject.toml ./poetry.lock /code/
RUN pip install poetry && \
    poetry install

COPY . .

CMD ["streamlit", "run", "/code/src/main.py"]
