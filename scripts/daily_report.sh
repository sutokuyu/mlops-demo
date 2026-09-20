#!/usr/bin/env bash
#
# Send the daily cat-location summary.
#
# cron runs with a nearly empty environment, so the LLM key and the Discord
# webhook are read from .env instead of relying on the interactive shell. Make
# sure .env exists and is not committed (.gitignore already covers it).
#
# Usage:
#   ./scripts/daily_report.sh                # summarize yesterday and send it
#   ./scripts/daily_report.sh --print-only   # print without sending
#   ./scripts/daily_report.sh --days-ago 3   # summarize three days ago
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

PYTHON="$PROJECT_ROOT/.venv/bin/python"
REPORT="$PROJECT_ROOT/src/execute_location_report.py"

if [[ $# -gt 0 ]]; then
    exec "$PYTHON" "$REPORT" "$@"
fi

# A cron job just after midnight wants the day that just ended.
exec "$PYTHON" "$REPORT" --days-ago 1
