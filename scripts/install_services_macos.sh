#!/usr/bin/env bash
#
# Install the tracker, the daily report, the Discord bot and the camera lookup as
# launchd user agents - the macOS counterpart of scripts/install_services.sh, which
# only knows systemd.
#
# The plists are generated from deploy/launchd/*.plist into ~/Library/LaunchAgents:
# launchd cannot expand %h or $HOME itself, so __PROJECT_ROOT__/__LOG_DIR__ placeholders
# are substituted here. Re-running this script after editing the repo is the whole
# upgrade path, same as the systemd installer.
#
# Usage:
#   ./scripts/install_services_macos.sh
#   ./scripts/install_services_macos.sh --uninstall
#
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "install_services_macos: this installs launchd agents and only runs on macOS." >&2
    echo "install_services_macos: use ./scripts/install_services.sh on Linux." >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_SOURCE="$PROJECT_ROOT/deploy/launchd"
PLIST_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$HOME/Library/Logs/mlops-demo"
DOMAIN="gui/$(id -u)"
LABELS=(
    com.sutokuyu.mlops-demo.cat-tracker
    com.sutokuyu.mlops-demo.cat-report
    com.sutokuyu.mlops-demo.cat-discord
    com.sutokuyu.mlops-demo.cat-camera-ip
)

if [[ "${1:-}" == "--uninstall" ]]; then
    for label in "${LABELS[@]}"; do
        launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
        rm -f "$PLIST_DIR/$label.plist"
    done
    echo "Removed the cat tracker agent, the report agent, the Discord bot agent and the camera lookup agent."
    exit 0
fi

mkdir -p "$PLIST_DIR" "$LOG_DIR"
for label in "${LABELS[@]}"; do
    sed \
        -e "s#__PROJECT_ROOT__#$PROJECT_ROOT#g" \
        -e "s#__LOG_DIR__#$LOG_DIR#g" \
        "$PLIST_SOURCE/$label.plist" >"$PLIST_DIR/$label.plist"
    # bootout first: bootstrap fails loudly if the label is already loaded, and
    # re-running this script to pick up an edited plist is the expected upgrade path.
    launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$PLIST_DIR/$label.plist"
done

# Starting an agent whose preflight already fails only burns its restart budget
# and leaves a confusing log, so each launcher is asked first, same as the systemd
# installer. The reason is kept rather than discarded: "loaded, NOT started" on its
# own is the least useful message this script could print.
if tracker_preflight="$(TRACKER_CHECK_ONLY=1 "$PROJECT_ROOT/scripts/run_tracker.sh" 2>&1)"; then
    launchctl kickstart -k "$DOMAIN/com.sutokuyu.mlops-demo.cat-tracker"
    tracker_state="started"
else
    tracker_state="loaded, NOT started"
fi

if bot_preflight="$("$PROJECT_ROOT/scripts/run_discord_bot.sh" --check-only 2>&1)"; then
    launchctl kickstart -k "$DOMAIN/com.sutokuyu.mlops-demo.cat-discord"
    bot_state="started"
else
    bot_state="loaded, NOT started"
fi

cat <<EOF

Installed. The tracker is $tracker_state. The bot is $bot_state.

Useful commands:

    launchctl print $DOMAIN/com.sutokuyu.mlops-demo.cat-tracker    # is it running
    tail -f $LOG_DIR/cat-tracker.log                               # live tracker log
    launchctl print $DOMAIN/com.sutokuyu.mlops-demo.cat-report     # when it last ran
    tail -n 50 $LOG_DIR/cat-report.log                             # what the last report did
    launchctl kickstart -k $DOMAIN/com.sutokuyu.mlops-demo.cat-report   # send one now, to test
    launchctl print $DOMAIN/com.sutokuyu.mlops-demo.cat-discord    # is the bot connected
    tail -f $LOG_DIR/cat-discord.log                               # live bot log
    tail -f $LOG_DIR/cat-camera-ip.log                             # camera lookup history
    ./scripts/sync_camera_ips.py --explain                         # where the cameras are right now

launchd restarts a crashing agent every ~10s and does not give up on its own; if
startup is broken, fix the cause and then:

    launchctl kickstart -k $DOMAIN/com.sutokuyu.mlops-demo.cat-tracker

If the project moves, update the paths in $PLIST_SOURCE and re-run this script.

A Mac mini only runs these agents while someone is logged in (LaunchAgents, not
LaunchDaemons) unless "Login Window" automatic login is set up in System Settings
> Users & Groups, and the Mac is set to not sleep (System Settings > Energy, or
\`sudo pmset -a sleep 0\` for a headless box that is always plugged in).
EOF

if [[ "$tracker_state" != "started" ]]; then
    echo
    echo "The tracker did not start because its preflight failed:"
    echo
    printf '%s\n' "$tracker_preflight"
    echo
    echo "Fix that, then start the agent:"
    echo
    echo "    ./scripts/run_tracker.sh"
    echo "    launchctl kickstart -k $DOMAIN/com.sutokuyu.mlops-demo.cat-tracker"
fi

if [[ "$bot_state" != "started" ]]; then
    echo
    echo "The bot did not start because its preflight failed:"
    echo
    printf '%s\n' "$bot_preflight"
    echo
    echo "Fix that, then start the agent:"
    echo
    echo "    ./scripts/run_discord_bot.sh --check-only"
    echo "    launchctl kickstart -k $DOMAIN/com.sutokuyu.mlops-demo.cat-discord"
fi
