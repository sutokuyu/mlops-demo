"""Tests for finding the cameras again after the network moves them.

Nothing here touches a network or a camera: the scan, the stream open and the picture
match are all injected, the same way the tracker tests fake a reader. The rules under
test are the ones that were expensive to get right - which signal decides identity,
and what to do when the evidence disagrees.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.camera_discovery import (
    DiscoverySettings,
    discover,
    discovery_settings,
    host_of,
    hosts_in_subnet,
    open_hosts,
    rewrite_env_text,
    swap_host,
)

CAMERAS = ("living_room", "sofa", "feeder")
REFERENCES = {name: f"reference:{name}" for name in CAMERAS}


def url_for(camera: str, host: str = "192.168.3.9") -> str:
    """A URL whose password names the camera, so a fake opener can tell them apart."""
    return f"rtsp://admin:{camera}@{host}:554/h264/ch1/main/av_stream"


class FakeFrame:
    """Stands in for a grabbed frame, carrying what a fingerprint would measure."""

    def __init__(self, host: str, camera: str, shape=(1080, 1920, 3)) -> None:
        self.host = host
        self.camera = camera
        self.shape = shape


class FakeNetwork:
    """A network with chosen cameras at chosen addresses."""

    def __init__(self, cameras_at, matches=None) -> None:
        # cameras_at: {"living_room": "192.168.3.57"} or {..: [".45", ".57"]} when two
        # addresses accept the same password, which is what the ambiguity rule is for.
        self.cameras_at = cameras_at
        # matches: {(host, camera_whose_reference, camera_that_answered): count}
        self.matches = matches or {}
        self.opened: list[tuple[str, str]] = []

    def opener(self, url, timeout_seconds, frames_to_try):
        host = host_of(url)
        password = url.split("//", 1)[1].split("@", 1)[0].split(":", 1)[1]
        self.opened.append((host, password))
        if host in addresses_of(self.cameras_at.get(password)):
            return FakeFrame(host, password)
        return None

    def fingerprint(self, frame, reference, work_width=640):
        if frame is None:
            return 0
        if reference is None:
            return 0
        asked = reference.split(":", 1)[1]
        return self.matches.get((frame.host, asked, frame.camera), 0)


def addresses_of(value) -> set:
    """One address or several, always as a set."""
    if value is None:
        return set()
    return set(value) if isinstance(value, (list, tuple, set)) else {value}


def settings(**overrides) -> DiscoverySettings:
    values = {"subnets": ("192.168.3.0/24",), "min_fingerprint_matches": 12}
    values.update(overrides)
    return DiscoverySettings(**values)


def run(
    network, candidates, *, names=CAMERAS, log=None, configured_host="192.168.3.9", **overrides
):
    return discover(
        cameras=list(names),
        settings=settings(**overrides),
        urls={name: url_for(name, configured_host) for name in names},
        references=dict(REFERENCES),
        candidates=candidates,
        opener=network.opener,
        fingerprint=network.fingerprint,
        log=log or (lambda *args, **kwargs: None),
    )


# --- turning an address into a different address -----------------------------------


def test_swap_host_changes_only_the_address() -> None:
    """Credentials, port and stream path survive byte for byte.

    A rewrite that re-encoded a password or dropped a sub-stream path would produce a
    URL that looks plausible and cannot be opened, which is worse than not rewriting at
    all: the file would be wrong in a way nobody reads.
    """
    original = "rtsp://admin:p%40ss@192.168.3.13:8554/h264/ch1/sub/av_stream"
    assert swap_host(original, "192.168.3.59") == (
        "rtsp://admin:p%40ss@192.168.3.59:8554/h264/ch1/sub/av_stream"
    )
    assert swap_host("rtsp://192.168.3.13:554/h264/ch1/main/av_stream", "192.168.3.59") == (
        "rtsp://192.168.3.59:554/h264/ch1/main/av_stream"
    )


def test_host_of_is_not_fooled_by_an_at_sign_in_the_password() -> None:
    # An unencoded '@' is what ffmpeg treats as the end of the credentials, so it has to
    # be what this reads as the end of them too.
    assert host_of("rtsp://admin:we@ird@192.168.3.13:554/h264/ch1/main/av_stream") == "192.168.3.13"


def test_hosts_in_subnet_skips_the_network_and_broadcast_addresses() -> None:
    addresses = hosts_in_subnet("192.168.3.0/24")
    assert len(addresses) == 254
    assert addresses[0] == "192.168.3.1"
    assert addresses[-1] == "192.168.3.254"


# --- the scan --------------------------------------------------------------------


class FlakyConnector:
    """A network where a device drops the first packet while its ARP entry is cold."""

    def __init__(self, present, misses_before_answering: int) -> None:
        self.present = set(present)
        self.misses_before_answering = misses_before_answering
        self.attempts: dict[str, int] = {}

    def __call__(self, host, port, timeout):
        if host not in self.present:
            return False
        self.attempts[host] = self.attempts.get(host, 0) + 1
        return self.attempts[host] > self.misses_before_answering


def test_the_scan_asks_an_address_again_before_giving_up_on_it() -> None:
    """Measured behaviour, not a guess.

    192.168.3.59 answered 20/20 probes once its neighbour entry was warm, but the scan
    that ran just after the entry expired missed the camera completely - the first
    packet is dropped while the address is resolved, and that takes longer than a short
    connect timeout. Missing a camera reads as "your camera is gone", so the retry is
    the cheap side of this trade.
    """
    connector = FlakyConnector({"192.168.3.59"}, misses_before_answering=1)
    found = open_hosts(("192.168.3.0/24",), (554,), 0.01, 64, connector=connector, attempts=2)
    assert "192.168.3.59" in found

    connector = FlakyConnector({"192.168.3.59"}, misses_before_answering=1)
    found = open_hosts(("192.168.3.0/24",), (554,), 0.01, 64, connector=connector, attempts=1)
    assert "192.168.3.59" not in found


def test_only_the_ports_that_answered_are_reported() -> None:
    def connector(host, port, timeout):
        return host == "192.168.3.45" and port == 8000

    found = open_hosts(("192.168.3.0/24",), (554, 8000), 0.01, 64, connector=connector)
    assert found == {"192.168.3.45": (8000,)}


# --- who is who ------------------------------------------------------------------


def test_a_camera_is_found_where_its_own_password_is_accepted() -> None:
    network = FakeNetwork({"living_room": "192.168.3.57", "feeder": "192.168.3.59"})
    result = run(
        network, {"192.168.3.45": (554, 8000), "192.168.3.57": (554,), "192.168.3.59": (554,)}
    )

    assert sorted(result.found) == ["feeder", "living_room"]
    assert result.found["feeder"].host == "192.168.3.59"
    assert result.found["feeder"].moved, "it used to be somewhere else"
    assert result.missing == ["sofa"]
    assert not result.complete


def test_a_camera_already_in_the_right_place_is_reported_as_unmoved() -> None:
    network = FakeNetwork({"living_room": "192.168.3.57"})
    result = run(
        network, {"192.168.3.57": (554,)}, names=["living_room"], configured_host="192.168.3.57"
    )
    assert result.found["living_room"].host == "192.168.3.57"
    assert not result.found["living_room"].moved


def test_a_picture_that_belongs_to_another_camera_vetoes_the_credentials() -> None:
    """Both passwords work at one address; only the picture can say whose it is.

    Accepting on credentials alone would file one room's cat under another room's
    zones, which is worse than leaving the camera offline: offline loses samples, a
    mixed-up camera silently corrupts the day's history.
    """
    network = FakeNetwork(
        {"living_room": "192.168.3.45", "sofa": "192.168.3.45"},
        matches={("192.168.3.45", "sofa", "living_room"): 40, ("192.168.3.45", "sofa", "sofa"): 40},
    )
    result = run(network, {"192.168.3.45": (554,)})

    assert "living_room" in result.missing
    assert result.found["sofa"].host == "192.168.3.45"
    assert any("matches sofa far better" in note for note in result.notes)


def test_a_low_score_on_its_own_does_not_veto_anything() -> None:
    """A night frame on IR can score low against a daylight reference and still be right.

    The floor is a veto only when another camera's reference beats it, so this camera
    is accepted with a note rather than refused.
    """
    network = FakeNetwork(
        {"living_room": "192.168.3.45"},
        matches={
            ("192.168.3.45", "living_room", "living_room"): 4,
            ("192.168.3.45", "sofa", "living_room"): 3,
        },
    )
    result = run(network, {"192.168.3.45": (554,)}, names=["living_room"])
    assert result.found["living_room"].host == "192.168.3.45"
    assert any("inconclusive" in note for note in result.notes)


def test_two_addresses_claiming_one_camera_with_equal_pictures_are_left_alone() -> None:
    """A tie is a question for a human, not a coin flip for a robot."""
    network = FakeNetwork(
        {"living_room": ["192.168.3.45", "192.168.3.57"]},
        matches={
            ("192.168.3.45", "living_room", "living_room"): 30,
            ("192.168.3.57", "living_room", "living_room"): 30,
        },
    )
    result = run(network, {"192.168.3.45": (554,), "192.168.3.57": (554,)}, names=["living_room"])
    assert result.missing == ["living_room"]
    assert any("ambiguous" in note for note in result.notes)


def test_the_same_address_cannot_be_two_cameras() -> None:
    network = FakeNetwork(
        {"living_room": "192.168.3.57", "sofa": "192.168.3.57"},
        matches={
            ("192.168.3.57", "living_room", "living_room"): 40,
            ("192.168.3.57", "sofa", "sofa"): 40,
        },
    )
    result = run(network, {"192.168.3.57": (554,)}, names=["living_room", "sofa"])
    assert len(result.found) == 1, "the first camera to claim the address keeps it"
    assert len(result.missing) == 1


def test_a_missing_camera_says_which_hosts_refused_its_password() -> None:
    """\"Not found\" on its own sends the owner looking for a power cable.

    Knowing that an address answered RTSP and refused this camera's password is the
    difference between a camera that is unplugged and one whose password or stream path
    changed - which is exactly the case found on this network at 192.168.3.45.
    """
    network = FakeNetwork({})
    result = run(network, {"192.168.3.45": (554, 8000), "192.168.3.57": (554,)}, names=["sofa"])
    assert result.missing == ["sofa"]
    assert any(
        "none accepted its password" in note and "192.168.3.45" in note for note in result.notes
    )


def test_a_host_that_only_answers_the_sdk_port_is_named_as_a_hint() -> None:
    network = FakeNetwork({})
    result = run(network, {"192.168.3.44": (8000,)}, names=["sofa"])
    assert any("SDK port" in note for note in result.notes)


def test_disabled_discovery_looks_for_nothing() -> None:
    network = FakeNetwork({"living_room": "192.168.3.57"})
    result = run(network, {"192.168.3.57": (554,)}, enabled=False)
    assert result.found == {}
    assert result.missing == list(CAMERAS)
    assert network.opened == []


# --- reading the settings --------------------------------------------------------


def test_discovery_settings_read_the_config_block() -> None:
    parsed = discovery_settings(
        {
            "discovery": {
                "enabled": False,
                "subnets": ["10.0.0.0/24"],
                "rtsp_port": 8554,
                "hint_ports": [8001],
                "min_fingerprint_matches": 20,
                "missing_after_seconds": 30,
            }
        }
    )
    assert parsed.enabled is False
    assert parsed.subnets == ("10.0.0.0/24",)
    assert parsed.rtsp_port == 8554
    assert parsed.hint_ports == (8001,)
    assert parsed.min_fingerprint_matches == 20
    assert parsed.missing_after_seconds == 30.0
    # Untouched keys keep the deployment defaults.
    assert parsed.retry_seconds == DiscoverySettings().retry_seconds


def test_discovery_settings_fall_back_completely_on_an_empty_config() -> None:
    assert discovery_settings({}) == DiscoverySettings()


# --- writing .env ------------------------------------------------------------------


ENV = """\
# Camera URLs. Credentials live here, not in the YAML.
LLM_API_KEY=sk-secret

LIVING_ROOM_RTSP_URL='rtsp://admin:pw@192.168.3.57:554/h264/ch1/main/av_stream'
FEEDER_RTSP_URL="rtsp://admin:pw@192.168.3.13:554/h264/ch1/main/av_stream"
SOFA_RTSP_URL=rtsp://admin:pw@192.168.3.48:554/h264/ch1/main/av_stream
"""


def test_rewrite_env_text_changes_the_address_and_nothing_else() -> None:
    updated, changes = rewrite_env_text(ENV, {"FEEDER_RTSP_URL": "192.168.3.59"})

    assert changes == {"FEEDER_RTSP_URL": ("192.168.3.13", "192.168.3.59")}
    assert updated.replace("192.168.3.59", "192.168.3.13") == ENV, (
        "every other byte, including the quoting and the comments, is untouched"
    )
    assert "'rtsp://admin:pw@192.168.3.57:554/h264/ch1/main/av_stream'" in updated


def test_rewrite_env_text_keeps_each_line_s_own_quoting() -> None:
    updated, _ = rewrite_env_text(
        ENV, {"FEEDER_RTSP_URL": "192.168.3.59", "SOFA_RTSP_URL": "192.168.3.60"}
    )
    assert 'FEEDER_RTSP_URL="rtsp://admin:pw@192.168.3.59:554/h264/ch1/main/av_stream"' in updated
    assert "SOFA_RTSP_URL=rtsp://admin:pw@192.168.3.60:554/h264/ch1/main/av_stream" in updated


def test_rewrite_env_text_reports_nothing_when_the_address_is_already_right() -> None:
    updated, changes = rewrite_env_text(ENV, {"FEEDER_RTSP_URL": "192.168.3.13"})
    assert changes == {}
    assert updated == ENV


def test_rewrite_env_text_ignores_keys_it_was_not_asked_about() -> None:
    updated, changes = rewrite_env_text(ENV, {"SOMETHING_ELSE": "10.0.0.1", "LLM_API_KEY": "x"})
    assert changes == {}
    assert updated == ENV


def test_rewrite_env_text_leaves_a_line_it_cannot_parse() -> None:
    text = "FEEDER_RTSP_URL=rtsp://admin:pw@192.168.3.13:554/x  # trailing note\n"
    updated, changes = rewrite_env_text(text, {"FEEDER_RTSP_URL": "192.168.3.59"})
    assert changes == {}
    assert updated == text
