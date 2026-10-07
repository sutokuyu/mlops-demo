#!/usr/bin/env python
"""Write the camera addresses that were found on the network back into .env.

The cameras sit behind a Wi-Fi relay that hands them a new address every so often,
and a stale ``*_RTSP_URL`` is invisible until a camera quietly stops being recorded.
This is the offline half of the job ``location_tracker`` does live: the tracker
relocates itself in place - and rewrites .env itself when it does - while this script
covers the case where nothing is recording, and gives the installer a preflight that
checks the addresses that will actually be used.

It deliberately stands down while the tracker is running. A lookup opens a stream on
every camera it has to identify, and that competes with the recording for the same
relay link: measured on this network, two lookups running alongside the tracker
stalled a 2560x1440 handshake past 30 seconds and the tracker dropped its stream on
another camera. Two lookups at once are also refused by a lock, for the same reason.

How a camera is identified, and why it is not by MAC address, is documented in
``src/monitoring/camera_discovery.py``.

Usage:
    ./scripts/sync_camera_ips.py             # report only, change nothing
    ./scripts/sync_camera_ips.py --apply     # rewrite .env (keeps .env.bak)
    ./scripts/sync_camera_ips.py --explain --force

Exit codes:
    0  nothing to do: every located camera is already correct, or the lookup stood
       down because the tracker is running
    2  something was written, or would be, but at least one camera is still missing -
       the owner is told separately, and the tracker keeps retrying
    1  a hard error: unreadable config, missing .env, unparsable YAML
"""

import argparse
import fcntl
import platform
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = PROJECT_ROOT / ".env"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
LOCK_PATH = PROJECT_ROOT / "data" / "camera-lookup.lock"
TRACKER_UNIT = "cat-tracker.service"
# launchd has no per-unit "is-active": a job can be loaded (bootstrapped) without a
# live PID, e.g. between KeepAlive restarts. See deploy/launchd/ for the plist.
TRACKER_LAUNCHD_LABEL = "com.sutokuyu.mlops-demo.cat-tracker"

# .env at module level, before anything can read the config: the config substitutes
# ${VAR} from the environment at import time, so loading it inside main() would be too
# late for the imports main() itself does. config_loader is light (no numpy or cv2), so
# importing it here does not slow --help down; camera_discovery stays inside main().
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_env_file

load_env_file()


def tracker_is_running(
    unit: str = TRACKER_UNIT, launchd_label: str = TRACKER_LAUNCHD_LABEL
) -> bool:
    """True while the recorder is active, so the lookup can stand down.

    Anything that is not a definite "running" counts as not running: with no
    systemd/launchd there is nothing to compete with, and refusing to look would be
    worse than looking.
    """
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["launchctl", "list", launchd_label],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        if result.returncode != 0:
            return False
        # A loaded-but-not-running job (e.g. between KeepAlive restarts) omits the
        # "PID" key entirely rather than printing a placeholder, unlike systemd.
        return bool(re.search(r'"PID"\s*=\s*\d+;', result.stdout))

    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", unit],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.stdout.strip() == "active"


@contextmanager
def lookup_lock(path: Path = LOCK_PATH):
    """Yield True when this process is the only one looking.

    The installer's preflight and the timer can fire at the same time - they did, on
    the first run of this feature - and two sweeps at once means two streams per
    camera. The later caller gives up instead of adding to the load.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def preview_hosts(env_path: Path, hosts: dict[str, str]) -> dict[str, tuple[str, str]]:
    """What writing ``hosts`` would change, without touching the file."""
    from src.monitoring.camera_discovery import rewrite_env_text

    _, changes = rewrite_env_text(env_path.read_text(encoding="utf-8"), hosts)
    return changes


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the cameras on the network and fix their hosts in .env.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite .env. Without it this only reports what it would change.",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV,
        help="Environment file to update.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="camera_names",
        help="Restrict the search to these cameras; repeat for several.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print what every reachable address answered for every camera.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Look even while the tracker is recording.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    env_path = args.env if args.env.is_absolute() else PROJECT_ROOT / args.env
    # Already loaded at import time when it is the default file; this call is what makes
    # --env work, and it is a no-op for the default path.
    if not load_env_file(env_path):
        print(f"error: {env_path} does not exist", file=sys.stderr)
        return 1
    if not DEFAULT_CONFIG.is_file():
        print(f"error: {DEFAULT_CONFIG} does not exist", file=sys.stderr)
        return 1

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.monitoring.camera_discovery import (
        apply_hosts_to_env,
        camera_env_vars,
        describe,
        discover,
        discovery_settings,
    )

    try:
        variables = camera_env_vars(DEFAULT_CONFIG)
    except Exception as error:  # noqa: BLE001 - an unreadable config is a hard error
        print(f"error: cannot read {DEFAULT_CONFIG}: {error}", file=sys.stderr)
        return 1
    if not variables:
        print(f"error: no camera in {DEFAULT_CONFIG} names a ${{VAR}} rtsp_url", file=sys.stderr)
        return 1

    wanted = [name for name in (args.camera_names or list(variables)) if name in variables]
    for name in [name for name in (args.camera_names or []) if name not in variables]:
        print(f"warning: '{name}' is not a camera in {DEFAULT_CONFIG}", file=sys.stderr)

    if not args.force and tracker_is_running():
        print(
            "the tracker is recording, so this lookup stands down: it would open a "
            "second stream on cameras already using the relay, and the tracker keeps "
            ".env current itself when a camera moves. Use --force to look anyway."
        )
        return 0

    with lookup_lock() as acquired:
        if not acquired:
            print("another camera lookup is already running; nothing to do")
            return 0
        settings = discovery_settings()
        print(f"looking for {', '.join(wanted)} on {', '.join(settings.subnets)}")
        result = discover(cameras=wanted, settings=settings)

    for line in describe(result):
        print(f"  {line}")
    if args.explain:
        for evidence in result.evidence:
            other = evidence.best_other()
            print(
                f"    {evidence.host:<16} as {evidence.camera:<12} "
                f"password={'ok' if evidence.credential_ok else 'no':<3} "
                f"matches={evidence.own_matches:<4} "
                f"next={other[0] + ':' + str(other[1]) if other else '-'} "
                f"size={evidence.resolution or '-'}"
            )

    hosts: dict[str, str] = {}
    for camera, found in result.found.items():
        variable = variables.get(camera)
        if variable is None:
            print(f"  {camera}: no ${{VAR}} placeholder to rewrite", file=sys.stderr)
            continue
        hosts[variable] = found.host

    if not hosts:
        print("nothing was located, so .env is left alone")
        return 2

    changes = apply_hosts_to_env(env_path, hosts) if args.apply else preview_hosts(env_path, hosts)
    if not changes:
        print(".env already names every located camera correctly")
        return 2 if result.missing else 0
    for variable, (old_host, new_host) in sorted(changes.items()):
        print(f"  {variable}: {old_host} -> {new_host}")
    if args.apply:
        print(f"updated {env_path} ({env_path.name}.bak holds the previous version)")
        if result.missing:
            print(f"still missing: {', '.join(result.missing)}")
    else:
        print("dry run; pass --apply to write it")
    return 2 if result.missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
