import os
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
# Set this to read a different file, which is how the camera-free smoke test runs
# without editing the real .env.
ENV_PATH_VARIABLE = "MLOPS_ENV_FILE"
_loaded_env_paths: set[Path] = set()


def load_env_file(path: str | Path | None = None, *, override: bool = True) -> bool:
    """Apply .env to ``os.environ``, and report whether a file was read at all.

    Every runnable entry point calls this *before* importing anything that reads the
    configuration, because :func:`load_config` substitutes ``${VAR}`` from the
    environment at **import** time. A value set after that import never reaches the
    camera URLs, which is how a terminal that had exported them days ago kept winning
    over the file the tracker keeps up to date: every entry point except that one
    expected the shell to have sourced .env first.

    ``.env`` wins over the surrounding environment, because it is the documented single
    source of truth and the alternative - a stale export shadowing it - is exactly the
    failure this function exists to stop. Override deliberately by pointing
    ``MLOPS_ENV_FILE`` at another file.

    Reading the same path twice is a no-op, so a second call cannot undo something a
    caller set in between.
    """
    resolved = (
        Path(path)
        if path is not None
        else Path(os.environ.get(ENV_PATH_VARIABLE) or DEFAULT_ENV_PATH)
    )
    if resolved in _loaded_env_paths:
        return True
    if not resolved.is_file():
        return False
    for line in resolved.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not override and key in os.environ:
            continue
        os.environ[key] = value.strip().strip("'").strip('"')
    _loaded_env_paths.add(resolved)
    return True


def _substitute_env(value: Any) -> Any:
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

        def replace(match: re.Match[str]) -> str:
            env_name = match.group(1)
            default = match.group(2)
            return os.environ.get(env_name, default if default is not None else "")

        return pattern.sub(replace, value)

    if isinstance(value, list):
        return [_substitute_env(item) for item in value]

    if isinstance(value, dict):
        return {key: _substitute_env(item) for key, item in value.items()}

    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else CONFIG_PATH
    with open(config_path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _substitute_env(data)


def resolve_config_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
