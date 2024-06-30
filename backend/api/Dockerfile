FROM python:3.11-slim-buster as builder

RUN pip install poetry==1.5.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml .
RUN touch README.md

RUN --mount=type=cache,target=$POETRY_CACHE_DIR poetry install --no-root

FROM python:3.11-slim

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=builder /app/pyproject.toml ./

# Install curl
RUN apt-get update && apt-get install -y curl libpq-dev libsndfile1

# Copy python files
COPY app ./app

# Copy scripts (wait-for-db.sh)
# COPY api/scripts ./scripts
# RUN chmod +x ./scripts/wait-for-db.sh

# Activate poetry virtual environment
RUN . .venv/bin/activate

# CMD ["bash", "-c", "./scripts/wait-for-db.sh && uvicorn app.main:api --host 0.0.0.0 --port 80"]
CMD ["bash", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 80"]