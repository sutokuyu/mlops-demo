import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "configs").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Could not locate project root")


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring.toilet_monitor import main as run_monitor, register_notification_callback
from src.notification.notification_controller import send_notification


def main():
    print("Starting toilet monitor...")
    register_notification_callback(send_notification)
    run_monitor()


if __name__ == "__main__":
    main()
