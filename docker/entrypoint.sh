#!/bin/sh
# Mount plugin wheels under $SOUPDB_PLUGINS_DIR and set SOUPDB_PLUGINS=1 to install them.
set -eu

plugins_dir="${SOUPDB_PLUGINS_DIR:-/opt/soupdb/plugins}"
plugin_site="${SOUPDB_PLUGIN_SITE:-/opt/soupdb/plugin-site}"

if [ "${SOUPDB_PLUGINS:-0}" = "1" ] && [ -d "$plugins_dir" ]; then
    for wheel in "$plugins_dir"/*.whl; do
        [ -e "$wheel" ] || continue
        echo "soupdb: installing plugin $wheel"
        python -m pip install --no-deps --target "$plugin_site" "$wheel"
    done
fi

exec python -m engine
