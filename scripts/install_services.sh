#!/usr/bin/env bash
#
# Install the tracker service and the daily report timer as systemd user units.
#
# The units are symlinked from deploy/systemd rather than copied, so editing the
# repo and re-running this script is the whole upgrade path.
#
# Usage:
#   ./scripts/install_services.sh
#   ./scripts/install_services.sh --uninstall
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SOURCE="$PROJECT_ROOT/deploy/systemd"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNITS=(cat-tracker.service cat-report.service cat-report.timer)

if [[ "${1:-}" == "--uninstall" ]]; then
    systemctl --user disable --now cat-tracker.service cat-report.timer || true
    for unit in "${UNITS[@]}"; do
        rm -f "$UNIT_DIR/$unit"
    done
    systemctl --user daemon-reload
    echo "Removed the cat tracker service and the report timer."
    exit 0
fi

mkdir -p "$UNIT_DIR"
for unit in "${UNITS[@]}"; do
    ln -sfn "$UNIT_SOURCE/$unit" "$UNIT_DIR/$unit"
done
systemctl --user daemon-reload

# Linger is what keeps the user manager, and therefore the tracker, alive after
# the last terminal closes. Without it the tracker dies with the WSL session.
if [[ "$(loginctl show-user "$USER" -p Linger --value)" != "yes" ]]; then
    if loginctl enable-linger "$USER" 2>/dev/null; then
        echo "Enabled lingering so the tracker outlives the login session."
    else
        echo "Could not enable lingering. Run this yourself, it needs no password:" >&2
        echo "    loginctl enable-linger $USER" >&2
    fi
fi

systemctl --user enable --now cat-report.timer

# Starting the tracker when a camera URL is missing only produces a failed unit
# and burns the restart budget, so ask the launcher's own preflight first.
if TRACKER_CHECK_ONLY=1 "$PROJECT_ROOT/scripts/run_tracker.sh" >/dev/null 2>&1; then
    systemctl --user enable --now cat-tracker.service
    tracker_state="started"
else
    systemctl --user enable cat-tracker.service
    tracker_state="enabled, NOT started"
fi

cat <<EOF

Installed. The tracker is $tracker_state.

Useful commands:

    systemctl --user status cat-tracker.service     # is it running
    journalctl --user -u cat-tracker -f             # live tracker log
    systemctl --user list-timers cat-report.timer   # when the next report runs
    journalctl --user -u cat-report -n 50           # what the last report did
    systemctl --user start cat-report.service       # send one now, to test

The tracker is only restarted 10 times per 5 minutes. If startup is broken it
gives up in a failed state; after fixing the cause, reset it with:

    systemctl --user reset-failed cat-tracker && systemctl --user start cat-tracker

If the project moves, update the paths in $UNIT_SOURCE and re-run this script.
EOF

if [[ "$tracker_state" != "started" ]]; then
    echo
    echo "The tracker did not start because its preflight failed. Run it directly"
    echo "to see why, then start the service:"
    echo
    echo "    ./scripts/run_tracker.sh"
    echo "    systemctl --user start cat-tracker.service"
fi
