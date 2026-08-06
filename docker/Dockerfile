# Image principale du banc d'essai. CPU uniquement, par construction.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/cache/huggingface \
    IVR_BENCH_WEIGHTS_DIR=/cache/weights

# git est requis pour la dependance Needle epinglee par commit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv sync --all-extras --no-dev

COPY . .

ENTRYPOINT ["uv", "run", "ivr-bench"]
CMD ["--help"]
