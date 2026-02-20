#!/usr/bin/env sh
set -eu

PORT="${PORT:-8080}"

case "$PORT" in
  ''|*[!0-9]*)
    echo "ERROR: PORT must be a numeric value, got '$PORT'" >&2
    exit 1
    ;;
esac

exec streamlit run streamlit_app_erm.py \
  --server.address=0.0.0.0 \
  --server.port="$PORT" \
  --server.headless=true
