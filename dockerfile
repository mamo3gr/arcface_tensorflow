FROM nvcr.io/nvidia/tensorflow:21.05-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y make build-essential libssl-dev zlib1g-dev \
     libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
     libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
     git \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock ./

ENV POETRY_VERSION 1.0.10
RUN pip install poetry==${POETRY_VERSION} \
  && poetry config virtualenvs.create false \
  && poetry install --no-dev --no-interaction --no-ansi

COPY . .
