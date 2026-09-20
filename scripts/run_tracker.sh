#!/usr/bin/env bash
#
# Run the location tracker in the foreground until it is stopped.
#
# systemd starts this with a nearly empty environment, so everything the tracker
# needs is read from .env. The camera URLs are deliberately not in
# configs/config.yaml: the placeholders there are empty and the real values,
# credentials included, are substituted from the environment by config_loader.
#
# Usage:
#   ./scripts/run_tracker.sh                 # every camera in locations.yaml
#   ./scripts/run_tracker.sh --camera sofa   # a subset, for debugging
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
TRACKER="$PROJECT_ROOT/src/execute_location_tracker.py"

if [[ ! -f "$PROJECT_ROOT/.env" ]]; then
    echo "run_tracker: $PROJECT_ROOT/.env is missing." >&2
    echo "run_tracker: it must define LLM_API_KEY, LOCATION_REPORT_WEBHOOK_URL" >&2
    echo "run_tracker: and one *_RTSP_URL per camera in configs/config.yaml." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1091
source "$PROJECT_ROOT/.env"
set +a

# config_loader substitutes ${VAR:} with an empty string, so a camera whose URL
# was never exported fails much later with a bare "could not open stream". The
# check goes through the project's own config loader rather than a hardcoded
# list, so renaming a camera cannot leave this behind.
PROJECT_ROOT="$PROJECT_ROOT" "$PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["PROJECT_ROOT"])

from src.monitoring.location_config import camera_rtsp_url, configured_cameras

missing = [name for name in configured_cameras() if not camera_rtsp_url(name)]
if missing:
    print(f"run_tracker: no RTSP URL for camera(s): {', '.join(missing)}", file=sys.stderr)
    print("run_tracker: add the matching *_RTSP_URL line to .env and retry.", file=sys.stderr)
    raise SystemExit(1)

for name in configured_cameras():
    # Only the scheme, host and port: these lines end up in the journal.
    url = camera_rtsp_url(name)
    tail = url.rsplit("@", 1)[-1]
    print(f"run_tracker: {name} -> rtsp://...@{tail}", file=sys.stderr)
PY

# The installer runs the checks above without wanting to start the tracker.
if [[ -n "${TRACKER_CHECK_ONLY:-}" ]]; then
    exit 0
fi

# -u because stdout is a pipe under systemd: block buffering would hold
# "connected" and "lost stream; reconnecting" in memory, so the journal only
# shows them in bursts and a dead camera looks like a quiet camera.
exec "$PYTHON" -u "$TRACKER" "$@"
