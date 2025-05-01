FROM python:3.13.3-slim

WORKDIR /code

# RUN apk add --no-cache \
#     build-base \
#     cmake \
#     boost-dev \
#     zlib-dev \
#     curl \
#     bash \
#     git \
#     openssl-dev \
#     py3-pip \
#     py3-numpy \
#     py3-wheel
#
# RUN apk add --no-cache \
#     apache-arrow-dev \
#     py3-setuptools 
#
# RUN apk add --no-cache \
#     python3-dev \
#     musl-dev \
#     linux-headers
#
# RUN git clone --no-checkout https://github.com/apache/arrow.git /arrow \
#     && cd /arrow \
#     && git checkout tags/apache-arrow-19.0.1 \
#     && cd cpp \
#     && mkdir build \
#     && cd build \
#     && cmake -DARROW_CSV=ON -DARROW_JSON=ON -DARROW_FILESYSTEM=ON .. \
#     && make -j$(nproc) \
#     && make install
#
# ENV CMAKE_PREFIX_PATH=/usr/local/lib/cmake/Arrow

RUN apt-get update && apt-get install -y \
    build-essential

ENV POETRY_VIRTUALENVS_CREATE=false

COPY ./pyproject.toml ./poetry.lock /code/
RUN pip install poetry && \
    poetry install

COPY . .

CMD ["streamlit", "run", "/code/src/main.py"]
