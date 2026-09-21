#!/usr/bin/env bash
#
# Run the Discord bot that answers questions posted in a channel.
#
# systemd starts this with a nearly empty environment, so the bot token and the
# LLM key are read from .env. The token is deliberately not in
# configs/locations.yaml: discord_bot.token is a ${DISCORD_BOT_TOKEN:} placeholder
# that config_loader fills in from the environment.
#
# This one process needs no camera and no GPU; it only reads the tracking
# database, which the tracker writes with WAL enabled, so both can run at once.
#
# Usage:
#   ./scripts/run_discord_bot.sh                # connect and answer messages
#   ./scripts/run_discord_bot.sh --check-only   # validate the config, do not connect
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
BOT="$PROJECT_ROOT/src/execute_discord_bot.py"

if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    echo "run_discord_bot: $PROJECT_ROOT/.env is missing." >&2
    echo "run_discord_bot: it must define DISCORD_BOT_TOKEN and LLM_API_KEY." >&2
    exit 1
fi

# config_loader substitutes ${VAR:} at import time, so the values have to be in the
# environment before python starts - exporting them inside the program is too late.
set -a
# shellcheck disable=SC1091
source "$PROJECT_ROOT/.env"
set +a

exec "$PYTHON" "$BOT" "$@"
