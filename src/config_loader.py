import os
import re
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


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
