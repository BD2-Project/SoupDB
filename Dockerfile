# syntax=docker/dockerfile:1

# La imagen oficial de uv con Python ya incluye shell: la variante `uv:<versión>`
# a secas es scratch con el binario y nada más, así que ningún RUN funciona ahí.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY plugins ./plugins
RUN uv sync --frozen --no-dev --no-install-project --no-install-workspace

FROM python:3.12-slim AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SOUPDB_DATA=/app/data \
    SOUPDB_PLUGINS_DIR=/opt/soupdb/plugins \
    SOUPDB_PLUGIN_SITE=/opt/soupdb/plugin-site \
    PYTHONPATH=/opt/soupdb/plugin-site
COPY --from=build /app/.venv ./.venv
COPY engine ./engine
COPY docker/entrypoint.sh /usr/local/bin/soupdb-entrypoint
RUN chmod +x /usr/local/bin/soupdb-entrypoint
EXPOSE 55432
ENTRYPOINT ["soupdb-entrypoint"]
