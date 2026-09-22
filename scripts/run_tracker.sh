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

# The cameras hang behind a relay that hands them a new address every so often, and a
# stale URL is invisible until a camera quietly stops being recorded. Fix the file
# before the preflight below, so the URLs that get checked are the ones that will be
# used. Skipped while the installer is only probing (TRACKER_CHECK_ONLY), and never
# fatal: a camera that is simply unplugged must not stop the other two from running.
if [[ -z "${TRACKER_CHECK_ONLY:-}" && -z "${SKIP_CAMERA_SYNC:-}" ]]; then
    if ! "$PYTHON" -u "$PROJECT_ROOT/scripts/sync_camera_ips.py" --apply; then
        echo "run_tracker: camera address sync did not come back clean; continuing" >&2
    fi
    # The sync may have rewritten .env, so re-read it before anything validates it.
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

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
