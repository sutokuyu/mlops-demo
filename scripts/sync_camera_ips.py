#!/usr/bin/env python
"""Write the camera addresses that were found on the network back into .env.

The cameras sit behind a Wi-Fi relay that hands them a new address every so often,
and a stale ``*_RTSP_URL`` is invisible until a camera quietly stops being recorded.
This is the offline half of the job ``location_tracker`` does live: the tracker
relocates itself in place without a restart, while this keeps the file that every
*other* entry point reads - ``run_tracker.sh``, ``realtime_view``, the browser
preview - pointing at the right place, and it heals a tracker that is not running at
all.

How a camera is identified, and why it is not by MAC address, is documented in
``src/monitoring/camera_discovery.py``.

Usage:
    ./scripts/sync_camera_ips.py             # report only, change nothing
    ./scripts/sync_camera_ips.py --apply     # rewrite .env (keeps .env.bak)

Exit codes:
    0  every camera was found and .env is correct (or was corrected)
    2  something was written, or would be, but at least one camera is still missing
       - the owner has been told separately, and the tracker keeps retrying
    1  a hard error: unreadable config, missing .env, unparsable YAML
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = PROJECT_ROOT / ".env"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
PLACEHOLDER_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::[^}]*)?\}")


def load_env_file(path: Path) -> bool:
    """Put .env into os.environ before anything reads the configuration.

    ``config_loader`` substitutes ``${VARS}`` from ``os.environ`` and
    ``location_config`` freezes the substituted config at import time, so this has to
    happen before those imports - which is why they sit inside ``main``. Same reason
    ``scripts/run_tracker.sh`` sources .env: systemd starts a unit with almost no
    environment, so a script that only worked in an interactive shell would work
    everywhere except where it matters.

    .env wins over the surrounding environment: it is the documented single source of
    truth for the camera URLs.
    """
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("'").strip('"')
    return True


def camera_env_vars(config_path: Path) -> dict[str, str]:
    """Map each configured camera to the environment variable holding its URL.

    Read straight from the YAML, because the ``${VAR}`` placeholder is exactly what is
    being looked for and ``load_config`` would have already replaced it with the
    current - possibly wrong - value. Going through the config instead of guessing at
    ``NAME.upper() + "_RTSP_URL"`` means a renamed variable cannot silently leave this
    script writing to a key nothing reads.
    """
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    mapping: dict[str, str] = {}
    for camera in (raw.get("identity_collection") or {}).get("cameras") or []:
        name = camera.get("name")
        match = PLACEHOLDER_PATTERN.search(str(camera.get("rtsp_url") or ""))
        if name and match:
            mapping[str(name)] = match.group("name")
    return mapping


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
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    env_path = args.env if args.env.is_absolute() else PROJECT_ROOT / args.env
    config_path = DEFAULT_CONFIG
    if not load_env_file(env_path):
        print(f"error: {env_path} does not exist", file=sys.stderr)
        return 1
    if not config_path.is_file():
        print(f"error: {config_path} does not exist", file=sys.stderr)
        return 1

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.monitoring.camera_discovery import (
        describe,
        discover,
        discovery_settings,
        rewrite_env_text,
    )

    try:
        variables = camera_env_vars(config_path)
    except Exception as error:  # noqa: BLE001 - a broken YAML is a hard error here
        print(f"error: cannot read {config_path}: {error}", file=sys.stderr)
        return 1
    if not variables:
        print(f"error: no camera in {config_path} names a ${{VAR}} rtsp_url", file=sys.stderr)
        return 1

    wanted = [name for name in (args.camera_names or list(variables)) if name in variables]
    unknown = [name for name in (args.camera_names or []) if name not in variables]
    for name in unknown:
        print(f"warning: '{name}' is not a camera in {config_path}", file=sys.stderr)

    settings = discovery_settings()
    print(f"looking for {', '.join(wanted)} on {', '.join(settings.subnets)}")
    result = discover(cameras=wanted, settings=settings)

    for line in describe(result):
        print(f"  {line}")
    if args.explain:
        for evidence in result.evidence:
            print(
                f"    {evidence.host:<16} as {evidence.camera:<12} "
                f"password={'ok' if evidence.credential_ok else 'no':<3} "
                f"matches={evidence.matches:<4} size={evidence.resolution or '-'}"
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

    updated, changes = rewrite_env_text(env_path.read_text(encoding="utf-8"), hosts)
    if not changes:
        print(".env already names every located camera correctly")
        return 2 if result.missing else 0

    for variable, (old_host, new_host) in sorted(changes.items()):
        print(f"  {variable}: {old_host} -> {new_host}")
    if not args.apply:
        print("dry run; pass --apply to write it")
        return 2 if result.missing else 0

    backup = env_path.with_name(env_path.name + ".bak")
    shutil.copy2(env_path, backup)
    env_path.write_text(updated, encoding="utf-8")
    print(f"updated {env_path} ({backup.name} holds the previous version)")
    if result.missing:
        print(f"still missing: {', '.join(result.missing)}")
    return 2 if result.missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
