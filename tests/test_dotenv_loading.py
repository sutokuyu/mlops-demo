"""Every entry point reads .env itself, and only one helper does the reading.

The bug this guards against is silent. ``config_loader`` substitutes ``${VAR}`` into
the config at **import** time, so an entry point that forgets to load .env first does
not fail loudly - it quietly uses whatever the shell happened to export. That is how a
terminal from days ago kept a stale camera address winning over the .env the tracker
keeps up to date: of nineteen runnable scripts, exactly one read the file.

Two things are checked here: the helper's own behaviour, and the ordering, by running
a fresh interpreter rather than by reading the source.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import ENV_PATH_VARIABLE, load_env_file

ENTRY_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts")
MAIN_GUARD = 'if __name__ == "__main__"'
# Import statements that read the configuration, i.e. the ones that must come after the
# load. Matched at the start of a stripped line.
CONFIG_READERS = (
    "from src.monitoring",
    "from src.data",
    "from src.training",
    "from src.notification",
)


def runnable_scripts() -> list[Path]:
    return [
        path
        for root in ENTRY_ROOTS
        for path in sorted(root.rglob("*.py"))
        if MAIN_GUARD in path.read_text(encoding="utf-8")
    ]


def test_every_runnable_script_loads_dotenv_before_it_reads_the_config() -> None:
    """A new entry point that forgets this would look like a stale camera address.

    Line-based on purpose: a docstring or a comment that happens to mention one of
    these imports is not an import.
    """
    offenders = []
    for path in runnable_scripts():
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        reader = next(
            (index for index, line in enumerate(lines) if line.startswith(CONFIG_READERS)),
            None,
        )
        if reader is None:
            # A tool that never touches the config (the dataset zippers) has nothing to
            # read from .env.
            continue
        load = next((index for index, line in enumerate(lines) if "load_env_file(" in line), None)
        if load is None or load > reader:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == [], (
        "these read the config before loading .env, so ${VAR} will come from the shell: "
        + ", ".join(offenders)
    )


def test_the_entry_point_list_is_not_empty() -> None:
    """Otherwise the guard above passes by finding nothing to check."""
    assert len(runnable_scripts()) > 15


def test_the_config_sees_the_file_in_a_fresh_interpreter(tmp_path) -> None:
    """The real thing, end to end: load .env, import the config, read a camera URL."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LIVING_ROOM_RTSP_URL='rtsp://admin:pw@10.9.9.9:554/h264/ch1/main/av_stream'\n",
        encoding="utf-8",
    )
    script = (
        "from src.config_loader import load_env_file\n"
        "load_env_file()\n"
        "from src.monitoring.location_config import camera_rtsp_url\n"
        "print(camera_rtsp_url('living_room'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env={**os.environ, ENV_PATH_VARIABLE: str(env_file)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "rtsp://admin:pw@10.9.9.9:554/h264/ch1/main/av_stream"


def test_a_stale_export_loses_to_the_file_in_a_fresh_interpreter(tmp_path) -> None:
    """The reported bug, pinned: an old exported value must not beat .env.

    The owner's terminal had exported the camera URLs days earlier, so the preview
    kept dialling an address the relay had abandoned - while .env, which the tracker
    had been correcting all along, said something else.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("LIVING_ROOM_RTSP_URL=rtsp://admin:pw@10.9.9.9:554/new\n", encoding="utf-8")
    script = (
        "from src.config_loader import load_env_file\n"
        "load_env_file()\n"
        "import os\n"
        "print(os.environ['LIVING_ROOM_RTSP_URL'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env={
            **os.environ,
            ENV_PATH_VARIABLE: str(env_file),
            "LIVING_ROOM_RTSP_URL": "rtsp://admin:pw@10.9.9.9:554/old",
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "rtsp://admin:pw@10.9.9.9:554/new"


# --- the helper itself --------------------------------------------------------------


def write_env(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_values_are_read_with_quoting_stripped_and_comments_skipped(tmp_path, monkeypatch):
    env_file = write_env(
        tmp_path / ".env",
        "# a comment\n\nPLAIN=one\nSINGLE='two words'\nDOUBLE=\"three words\"\nNOT_A_LINE\n",
    )
    for name in ("PLAIN", "SINGLE", "DOUBLE"):
        monkeypatch.delenv(name, raising=False)

    assert load_env_file(env_file) is True

    assert os.environ["PLAIN"] == "one"
    assert os.environ["SINGLE"] == "two words"
    assert os.environ["DOUBLE"] == "three words"


def test_an_equals_sign_inside_a_value_survives(tmp_path, monkeypatch) -> None:
    env_file = write_env(tmp_path / ".env", "WEBHOOK=https://example.invalid/a=b\n")
    monkeypatch.delenv("WEBHOOK", raising=False)
    load_env_file(env_file)
    assert os.environ["WEBHOOK"] == "https://example.invalid/a=b"


def test_a_missing_file_is_reported_and_changes_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PLAIN", raising=False)
    assert load_env_file(tmp_path / "nowhere" / ".env") is False
    assert "PLAIN" not in os.environ


def test_the_same_file_is_only_read_once(tmp_path, monkeypatch) -> None:
    """A second call must not undo an override a caller made in between."""
    env_file = write_env(tmp_path / ".env", "PLAIN=from-the-file\n")
    monkeypatch.setenv("PLAIN", "from-the-file")
    load_env_file(env_file)

    monkeypatch.setenv("PLAIN", "changed-in-process")
    load_env_file(env_file)

    assert os.environ["PLAIN"] == "changed-in-process"


def test_the_default_path_can_be_redirected_by_the_environment(tmp_path, monkeypatch) -> None:
    env_file = write_env(tmp_path / "somewhere-else.env", "PLAIN=redirected\n")
    monkeypatch.setenv(ENV_PATH_VARIABLE, str(env_file))
    monkeypatch.delenv("PLAIN", raising=False)

    assert load_env_file() is True

    assert os.environ["PLAIN"] == "redirected"


def test_an_explicit_path_ignores_the_redirect(tmp_path, monkeypatch) -> None:
    redirected = write_env(tmp_path / "redirected.env", "PLAIN=wrong-file\n")
    explicit = write_env(tmp_path / "explicit.env", "PLAIN=right-file\n")
    monkeypatch.setenv(ENV_PATH_VARIABLE, str(redirected))
    monkeypatch.delenv("PLAIN", raising=False)

    load_env_file(explicit)

    assert os.environ["PLAIN"] == "right-file"
