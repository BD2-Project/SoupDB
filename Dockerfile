# syntax=docker/dockerfile:1

# La imagen oficial de uv con Python ya incluye shell: la variante `uv:<versión>`
# a secas es scratch con el binario y nada más, así que ningún RUN funciona ahí.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SOUPDB_DATA=/app/data
COPY --from=build /app/.venv ./.venv
COPY engine ./engine
EXPOSE 55432
# Arranca el gestor de verdad: antes el CMD solo importaba el paquete e imprimía
# un mensaje, así que el contenedor no servía nada en su puerto.
CMD ["python", "-m", "engine"]
