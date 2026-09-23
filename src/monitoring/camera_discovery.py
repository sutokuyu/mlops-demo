"""Find the cameras again after the network hands them a different address.

Why this is not done by MAC address
-----------------------------------
The cameras hang behind a Wi-Fi relay that rewrites the source MAC, so every device
behind it answers ARP with the same hardware address. Measured on 2026-09-23 in the
owner's network: twelve addresses in ``192.168.3.0/24`` - including all three
cameras - reported the identical MAC ``D4-2C-46-8D-41-D9``. A MAC table therefore
only says "this device is behind the relay", and a DHCP reservation on the router
cannot separate the cameras either, because the router sees that same one address.

Two further facts shaped this module: this machine is WSL2 in NAT mode, so its own
ARP table never contains a camera at all (it ARPs for the Hyper-V gateway and
routes everything else), and the subnet scan still works from inside WSL because
Windows forwards it. So the search is ordinary TCP, plus asking each candidate the
one question only a camera can answer.

What the identity is instead
----------------------------
* **The credentials.** Every camera has its own RTSP password, and the others are
  refused with 401. That answer is exact, instant and needs no picture.
* **The picture.** One frame grabbed from the host, matched with the same ORB
  matcher the zone alignment uses, against the reference frame the zones were drawn
  on. This is what actually decides which *room* a host is, and it is the only
  signal that still works if two cameras ever share a password.

Neither signal is trusted alone. A host is accepted for a camera when the
credentials work *and* the picture does not contradict them; a picture that matches
a different camera's reference far better than the claimed one is treated as a
contradiction and the camera is left alone. A wrong host is worse than an offline
camera: offline loses samples, a mixed-up camera files one room's cat under another
room's zone.

Measured on the real outage this was written for: ``192.168.3.59`` answered
``feeder``'s password, its frame matched ``feeder``'s reference with 79 feature
pairs against 10 for the runner-up, and its resolution (1920x1080) matched feeder's
while ``living_room`` and ``sofa`` are both 2560x1440.
"""

import ipaddress
import os
import re
import shutil
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.monitoring.alignment import estimate_alignment
from src.monitoring.location_config import (
    LOCATION_CONFIG,
    PROJECT_ROOT,
    camera_rtsp_url,
    configured_cameras,
)
from src.monitoring.location_zones import load_calibrations
from src.monitoring.recalibration import reference_features_for

# The cameras are the only RTSP speakers on this network, so 554 open is the
# cheapest useful signal. 8000 is Hikvision's SDK port: it is not enough to grab a
# frame from, but a host that answers it is worth mentioning when a camera cannot
# be found, because it says "there is a camera here whose RTSP is not answering".
RTSP_PORT = 554
HINT_PORTS = (8000,)

# Long enough for a LAN reply, short enough that a full /24 of dead addresses still
# finishes in about a second with the worker count below.
HOST_TIMEOUT_SECONDS = 0.6
MAX_WORKERS = 128

# Every address that did not answer is asked again, once. A device whose ARP entry
# has gone stale drops the first packet while the address is being resolved, and that
# resolution outlives a short connect timeout. Measured on this network: 192.168.3.59
# answered 20/20 probes once its entry was warm, but a scan that arrived just after
# the entry expired missed the camera completely. Being missed is the expensive kind
# of wrong here - it reads as "your camera is gone" - so the second pass is cheap
# insurance, and it also gives the resolution the time it needs.
SCAN_ATTEMPTS = 2

# Opening an RTSP stream costs a DESCRIBE round trip plus a keyframe wait, so it is
# only ever attempted on hosts that already answered a TCP connect.
FRAME_TIMEOUT_SECONDS = 8.0
FRAMES_TO_TRY = 10

# A frame from a different room still shares a few corners, so this is set above the
# noise level rather than at it: the real feeder match scored 79 and the runner-up
# 10. It is a veto on top of the credentials, not an identity test by itself.
MIN_FINGERPRINT_MATCHES = 12

# How long a camera may stay unreachable before the tracker goes looking, and how
# often it looks again while the camera is still missing.
MISSING_AFTER_SECONDS = 120.0
RETRY_SECONDS = 300.0

# Matches ``rtsp://user:pass@host:port/path`` and lets only the host be replaced, so
# credentials, port and stream path survive byte for byte. Written as a pattern
# rather than urlsplit/urlunsplit because a re-encoded password (``%40`` for ``@``)
# must not be rewritten.
URL_PATTERN = re.compile(
    r"(?P<prefix>[a-zA-Z][\w+.-]*://(?:[^/\s]*@)?)(?P<host>[^/:\s@]+)(?P<rest>(?::\d+)?(?:/\S*)?)"
)

# ``${VAR:default}`` in configs/config.yaml, which is where a camera's address comes
# from and how this module finds the .env key that has to be rewritten.
PLACEHOLDER_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::[^}]*)?\}")


@dataclass(frozen=True)
class DiscoverySettings:
    """Everything the search needs, defaulted for this deployment."""

    enabled: bool = True
    subnets: tuple[str, ...] = ("192.168.3.0/24",)
    rtsp_port: int = RTSP_PORT
    hint_ports: tuple[int, ...] = HINT_PORTS
    host_timeout_seconds: float = HOST_TIMEOUT_SECONDS
    max_workers: int = MAX_WORKERS
    scan_attempts: int = SCAN_ATTEMPTS
    frame_timeout_seconds: float = FRAME_TIMEOUT_SECONDS
    frames_to_try: int = FRAMES_TO_TRY
    min_fingerprint_matches: int = MIN_FINGERPRINT_MATCHES
    work_width: int = 640
    missing_after_seconds: float = MISSING_AFTER_SECONDS
    retry_seconds: float = RETRY_SECONDS


def discovery_settings(config: dict | None = None) -> DiscoverySettings:
    """Read the ``discovery`` block, falling back to the defaults above."""
    raw = (config if config is not None else LOCATION_CONFIG).get("discovery") or {}
    defaults = DiscoverySettings()
    subnets = raw.get("subnets") or defaults.subnets
    hint_ports = raw.get("hint_ports")
    return DiscoverySettings(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        subnets=tuple(str(subnet) for subnet in subnets),
        rtsp_port=int(raw.get("rtsp_port", defaults.rtsp_port)),
        hint_ports=(
            defaults.hint_ports if hint_ports is None else tuple(int(port) for port in hint_ports)
        ),
        host_timeout_seconds=float(raw.get("host_timeout_seconds", defaults.host_timeout_seconds)),
        max_workers=int(raw.get("max_workers", defaults.max_workers)),
        scan_attempts=int(raw.get("scan_attempts", defaults.scan_attempts)),
        frame_timeout_seconds=float(
            raw.get("frame_timeout_seconds", defaults.frame_timeout_seconds)
        ),
        frames_to_try=int(raw.get("frames_to_try", defaults.frames_to_try)),
        min_fingerprint_matches=int(
            raw.get("min_fingerprint_matches", defaults.min_fingerprint_matches)
        ),
        work_width=int(raw.get("work_width", defaults.work_width)),
        missing_after_seconds=float(
            raw.get("missing_after_seconds", defaults.missing_after_seconds)
        ),
        retry_seconds=float(raw.get("retry_seconds", defaults.retry_seconds)),
    )


def safe_url(url: str) -> str:
    """The URL with its credentials masked, for anything that gets logged."""
    return re.sub(r"://[^/@\s]*@", "://***@", url or "")


def swap_host(url: str, host: str) -> str:
    """Point an RTSP URL at another host, keeping everything else identical."""
    return URL_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{host}{match.group('rest')}", url, count=1
    )


def host_of(url: str) -> str:
    match = URL_PATTERN.search(url or "")
    return match.group("host") if match else ""


def hosts_in_subnet(subnet: str) -> list[str]:
    """Every usable address in a CIDR block, e.g. 192.168.3.1 .. 192.168.3.254."""
    return [str(host) for host in ipaddress.ip_network(subnet, strict=False).hosts()]


def tcp_open(host: str, port: int, timeout: float) -> bool:
    with socket.socket() as probe:
        probe.settimeout(timeout)
        try:
            probe.connect((host, port))
        except OSError:
            return False
    return True


def open_hosts(
    subnets: tuple[str, ...],
    ports: tuple[int, ...],
    timeout: float,
    workers: int,
    connector=tcp_open,
    attempts: int = SCAN_ATTEMPTS,
) -> dict[str, tuple[int, ...]]:
    """Map each reachable address to the ports that answered.

    Scanned in parallel because a /24 is 254 addresses and the timeout is per
    address; a serial scan would take minutes and this can run inside the tracker.
    ``attempts`` is documented on ``SCAN_ATTEMPTS``: an address is given up on only
    after every attempt missed it, so a camera behind a cold ARP entry is not
    reported as absent.
    """
    addresses = [address for subnet in subnets for address in hosts_in_subnet(subnet)]

    def probe(address: str) -> tuple[str, tuple[int, ...]]:
        return address, tuple(port for port in ports if connector(address, port, timeout))

    found: dict[str, tuple[int, ...]] = {}
    pending = addresses
    for _ in range(max(1, attempts)):
        if not pending:
            break
        missed: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for address, answered in pool.map(probe, pending):
                if answered:
                    found[address] = answered
                else:
                    missed.append(address)
        pending = missed
    return found


def frame_from_url(
    url: str,
    timeout_seconds: float = FRAME_TIMEOUT_SECONDS,
    frames_to_try: int = FRAMES_TO_TRY,
) -> np.ndarray | None:
    """One frame from a stream, or None when it cannot be read at all.

    A refused connection and a wrong password both end up here: nobody needs to tell
    them apart, because both mean "not this camera at this address".

    The timeout is passed to the *constructor*, which is the only place it has an
    effect on opening - ``set(CAP_PROP_OPEN_TIMEOUT_MSEC)`` after the fact is read
    too late, and the backend falls back to its own 30 seconds. That was measured the
    hard way: a stalled handshake held a lookup for half a minute per attempt, and
    with two lookups running at once the whole tracker appeared to be broken.
    """
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    milliseconds = max(1000, int(timeout_seconds * 1000))
    try:
        capture = cv2.VideoCapture(
            url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                milliseconds,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                milliseconds,
            ],
        )
    except TypeError:
        # Older OpenCV builds have no parameter form, so fall back to the constructor
        # plus a best-effort set(). Worse, but not worse than not working at all.
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, milliseconds)
    try:
        capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, milliseconds)
        if not capture.isOpened():
            return None
        for _ in range(max(1, frames_to_try)):
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
        return None
    finally:
        capture.release()


def fingerprint_matches(frame: np.ndarray, reference, work_width: int = 640) -> int:
    """Feature pairs between a live frame and a camera's reference frame.

    ``min_inliers`` is 1 on purpose: the gates in ``estimate_alignment`` answer "is
    this a usable transform", and the question here is only "how many points line
    up". A count that a gate would have rejected is still exactly the signal that
    separates a real match (79) from a coincidence (10).
    """
    if reference is None or frame is None:
        return 0
    return int(estimate_alignment(reference, frame, work_width=work_width, min_inliers=1).matches)


@dataclass
class HostEvidence:
    """What one host answered for one camera, kept so the log can show its work.

    ``matches`` holds the picture comparison against *every* camera's reference frame,
    not just the one whose password was tried. It costs nothing extra - the frame is
    already in hand - and it is what lets the picture veto a claim even when only one
    camera is being looked for.
    """

    host: str
    camera: str
    credential_ok: bool = False
    resolution: str = ""
    matches: dict[str, int] = field(default_factory=dict)

    @property
    def own_matches(self) -> int:
        """How well the frame matches the camera whose password opened this host."""
        return self.matches.get(self.camera, 0)

    def best_other(self) -> tuple[str, int] | None:
        """The camera - other than this one - whose reference fits the frame best."""
        others = {name: score for name, score in self.matches.items() if name != self.camera}
        if not others:
            return None
        name, score = max(others.items(), key=lambda item: item[1])
        return name, score


@dataclass
class DiscoveredCamera:
    camera: str
    host: str
    url: str
    evidence: HostEvidence
    previous_host: str = ""

    @property
    def moved(self) -> bool:
        return bool(self.previous_host) and self.previous_host != self.host


@dataclass
class DiscoveryResult:
    found: dict[str, DiscoveredCamera] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    candidates: dict[str, tuple[int, ...]] = field(default_factory=dict)
    evidence: list[HostEvidence] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing


def _resolution_of(frame: np.ndarray | None) -> str:
    return f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else ""


def discover(
    cameras: list[str] | None = None,
    settings: DiscoverySettings | None = None,
    *,
    urls: dict[str, str] | None = None,
    references: dict | None = None,
    candidates: dict[str, tuple[int, ...]] | None = None,
    skip_hosts: tuple[str, ...] = (),
    connector=tcp_open,
    opener=frame_from_url,
    fingerprint=fingerprint_matches,
    log=print,
) -> DiscoveryResult:
    """Look for every requested camera and report where each one answers.

    ``skip_hosts`` are addresses not to touch at all. The tracker passes the hosts its
    *connected* cameras are using: those cameras are fine where they are, and opening
    a second stream on one costs real bandwidth on a camera that is already being
    recorded through a Wi-Fi relay - measured: two lookups running alongside the
    tracker stalled a 2560x1440 handshake past 30 seconds and the tracker's own stream
    dropped.

    The expensive bits - the subnet scan, opening a stream, matching a frame - are
    injectable so the rules can be tested without a network or a GPU, the same way
    the rest of this codebase fakes a reader or a calibration.
    """
    settings = settings or discovery_settings()
    names = list(cameras if cameras is not None else configured_cameras())
    urls = urls or {name: camera_rtsp_url(name) for name in names}
    if references is None:
        # Every calibrated camera, not only the ones being looked for: the picture check
        # needs the other rooms' reference frames to say "this frame is not the sofa".
        calibrations = load_calibrations()
        references = {
            name: reference_features_for(calibration, settings.work_width)
            for name, calibration in calibrations.items()
        }

    result = DiscoveryResult()
    if not settings.enabled:
        result.missing = list(names)
        result.notes.append("camera discovery is disabled (discovery.enabled: false)")
        return result

    if candidates is None:
        candidates = open_hosts(
            settings.subnets,
            (settings.rtsp_port, *settings.hint_ports),
            settings.host_timeout_seconds,
            settings.max_workers,
            connector=connector,
            attempts=settings.scan_attempts,
        )
    result.candidates = dict(candidates)
    answered = ", ".join(
        f"{host} ({'/'.join(str(port) for port in ports)})"
        for host, ports in sorted(candidates.items())
        if host not in skip_hosts
    )
    skipped = len([host for host in candidates if host in skip_hosts])
    log(
        f"[discovery] {len(candidates)} host(s) answered on "
        f"{', '.join(settings.subnets)}: {answered or 'none'}"
        + (f"; {skipped} left alone (in use by a camera that is recording)" if skipped else "")
    )

    # Every camera being looked for is asked of every reachable host, so a host can be
    # recognised even when the credentials that were supposed to open it are stale.
    # The picture is compared against every reference, which is local work on a frame
    # that has already been grabbed.
    for name in names:
        for host in sorted(candidates):
            if host in skip_hosts or settings.rtsp_port not in candidates[host]:
                # Nothing to open a stream against. Hosts with no RTSP are still worth
                # knowing about, and _explain_missing names them.
                continue
            frame = opener(
                swap_host(urls[name], host),
                settings.frame_timeout_seconds,
                settings.frames_to_try,
            )
            result.evidence.append(
                HostEvidence(
                    host=host,
                    camera=name,
                    credential_ok=frame is not None,
                    resolution=_resolution_of(frame),
                    matches={
                        other: fingerprint(frame, reference, settings.work_width)
                        for other, reference in references.items()
                    }
                    if frame is not None
                    else {},
                )
            )

    _decide(names, urls, settings, result, log)
    return result


def _decide(
    names: list[str],
    urls: dict[str, str],
    settings: DiscoverySettings,
    result: DiscoveryResult,
    log,
) -> None:
    """Turn the (host, camera) evidence into one host per camera, or into a refusal."""
    claims: dict[str, list[HostEvidence]] = {}
    for evidence in result.evidence:
        if evidence.credential_ok:
            claims.setdefault(evidence.camera, []).append(evidence)

    taken: dict[str, str] = {}
    for name in names:
        options = claims.get(name, [])
        if not options:
            result.missing.append(name)
            _explain_missing(name, settings, result, log)
            continue
        if len(options) > 1:
            # Two addresses answered the same credentials. The picture decides,
            # and a tie is left to a human rather than guessed at.
            ranked = sorted(options, key=lambda item: item.own_matches, reverse=True)
            if ranked[0].own_matches <= ranked[1].own_matches:
                result.missing.append(name)
                result.notes.append(
                    f"{name}: ambiguous - {len(options)} addresses accept its password "
                    f"({', '.join(item.host for item in ranked)}) and the pictures do not "
                    "separate them; left alone"
                )
                log(
                    f"[discovery] {name}: ambiguous, {len(options)} hosts claim it; not touching it"
                )
                continue
            chosen = ranked[0]
            result.notes.append(
                f"{name}: {len(options)} addresses accept its password, "
                f"picked {chosen.host} on the picture ({chosen.own_matches} feature pairs)"
            )
        else:
            chosen = options[0]

        if not _picture_agrees(name, chosen, settings, result, log):
            result.missing.append(name)
            continue
        if chosen.host in taken:
            result.missing.append(name)
            result.notes.append(
                f"{name}: {chosen.host} was already claimed by {taken[chosen.host]}"
            )
            continue

        taken[chosen.host] = name
        result.found[name] = DiscoveredCamera(
            camera=name,
            host=chosen.host,
            url=swap_host(urls[name], chosen.host),
            evidence=chosen,
            previous_host=host_of(urls[name]),
        )


def _picture_agrees(
    name: str, chosen: HostEvidence, settings: DiscoverySettings, result: DiscoveryResult, log
) -> bool:
    """Refuse a host whose picture clearly belongs to a different camera.

    A score below the floor is not enough on its own - a night frame on IR can score low
    against a daylight reference while still being the right room - so it only vetoes
    when another camera's reference matches the same frame *better than the floor*,
    which is what a shared password would look like.
    """
    if chosen.own_matches >= settings.min_fingerprint_matches:
        return True
    better = chosen.best_other()
    if better is not None and better[1] >= settings.min_fingerprint_matches:
        result.notes.append(
            f"{name}: {chosen.host} accepts its password but the picture matches "
            f"{better[0]} far better ({better[1]} vs {chosen.own_matches}); left alone"
        )
        log(
            f"[discovery] {name}: refused {chosen.host} - its frame looks like "
            f"{better[0]} ({better[1]} vs {chosen.own_matches} feature pairs)"
        )
        return False
    result.notes.append(
        f"{name}: {chosen.host} accepted by credentials, picture inconclusive "
        f"({chosen.own_matches} feature pairs)"
    )
    return True


def _explain_missing(name: str, settings: DiscoverySettings, result: DiscoveryResult, log) -> None:
    """Say why a camera was not found, with the hint that fits the evidence."""
    refusals = sorted(
        {
            evidence.host
            for evidence in result.evidence
            if evidence.camera == name
            and not evidence.credential_ok
            and settings.rtsp_port in result.candidates.get(evidence.host, ())
        }
    )
    hints = sorted(
        {
            host
            for host, ports in result.candidates.items()
            if settings.rtsp_port not in ports and set(ports) & set(settings.hint_ports)
        }
    )
    reason = "no address answered"
    if refusals:
        reason = f"RTSP answered at {', '.join(refusals)} but none accepted its password"
    if hints:
        reason += (
            f"; {', '.join(hints)} answer an SDK port but not RTSP, which is what a "
            "camera with RTSP switched off looks like"
        )
    result.notes.append(f"{name}: not found ({reason})")
    log(f"[discovery] {name}: not found on {', '.join(settings.subnets)} - {reason}")


def describe(result: DiscoveryResult) -> list[str]:
    """One line per camera: where it is, and the evidence that put it there."""
    lines = []
    for name, found in sorted(result.found.items()):
        moved = f" (was {found.previous_host})" if found.moved else ""
        lines.append(
            f"{name:<11} -> {found.host}{moved}  "
            f"[password ok, {found.evidence.own_matches} feature pairs, "
            f"{found.evidence.resolution}]"
        )
    for name in sorted(result.missing):
        lines.append(f"{name:<11} -> NOT FOUND")
    return lines


def camera_env_vars(config_path: Path | None = None) -> dict[str, str]:
    """Map each configured camera to the environment variable holding its URL.

    Read straight from the YAML, because the ``${VAR}`` placeholder is exactly what is
    being looked for and ``load_config`` would have already replaced it with the
    current - possibly wrong - value. Going through the config instead of guessing at
    ``NAME.upper() + "_RTSP_URL"`` means a renamed variable cannot silently leave a
    caller writing to a key nothing reads.
    """
    path = (
        Path(config_path) if config_path is not None else PROJECT_ROOT / "configs" / "config.yaml"
    )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mapping: dict[str, str] = {}
    for camera in (raw.get("identity_collection") or {}).get("cameras") or []:
        name = camera.get("name")
        match = PLACEHOLDER_PATTERN.search(str(camera.get("rtsp_url") or ""))
        if name and match:
            mapping[str(name)] = match.group("name")
    return mapping


def apply_hosts_to_env(
    env_path: Path, hosts: dict[str, str], backup: bool = True
) -> dict[str, tuple[str, str]]:
    """Point the named keys in .env at new hosts, keeping a ``.bak`` of the old file.

    Only the host changes; see ``rewrite_env_text``. Written by whichever component
    learned the new address - the tracker when it relocates a camera, the sync script
    when it runs on its own - so both go through this one path.
    """
    env_path = Path(env_path)
    updated, changes = rewrite_env_text(env_path.read_text(encoding="utf-8"), hosts)
    if not changes:
        return {}
    if backup:
        shutil.copy2(env_path, env_path.with_name(env_path.name + ".bak"))
    env_path.write_text(updated, encoding="utf-8")
    return changes


# ``KEY=value`` with optional ``export``, and a value that is one bare word optionally
# wrapped in quotes - which is what an RTSP URL line in this project's .env is.
ENV_LINE_PATTERN = re.compile(
    r"^(?P<lead>\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*)"
    r"(?P<value>.*?)(?P<trail>\s*)$"
)
ENV_VALUE_PATTERN = re.compile(r"^(?P<quote>['\"]?)(?P<url>\S*)(?P=quote)$")


def rewrite_env_text(text: str, hosts: dict[str, str]) -> tuple[str, dict[str, tuple[str, str]]]:
    """Point the given ``*_RTSP_URL`` keys at new hosts, touching nothing else.

    ``hosts`` maps an environment key to the host it should name. Only the host is
    replaced: the variable name, the quoting, the credentials, the port, the stream
    path and any trailing whitespace are carried over byte for byte, so a rewrite can
    never quietly normalise something the camera needs. Lines that are not one of
    those keys - comments, secrets, blank lines - are returned untouched.

    Returns the new text and, per key, ``(old_host, new_host)`` so the caller can say
    what it did. A key with no matching line, an unparsable value, or a host that is
    already right is simply not reported as a change.
    """
    changes: dict[str, tuple[str, str]] = {}
    rewritten: list[str] = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n")
        newline = line[len(body) :]
        match = ENV_LINE_PATTERN.match(body)
        if match is None or match.group("key") not in hosts:
            rewritten.append(line)
            continue
        value_match = ENV_VALUE_PATTERN.match(match.group("value"))
        if value_match is None:
            rewritten.append(line)
            continue
        url = value_match.group("url")
        old_host = host_of(url)
        new_host = hosts[match.group("key")]
        if not old_host or old_host == new_host:
            rewritten.append(line)
            continue
        quote = value_match.group("quote")
        rewritten.append(
            f"{match.group('lead')}{quote}{swap_host(url, new_host)}{quote}"
            f"{match.group('trail')}{newline}"
        )
        changes[match.group("key")] = (old_host, new_host)
    return "".join(rewritten), changes
