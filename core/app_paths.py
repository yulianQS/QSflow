"""Per-user settings and diagnostics paths for distributable QSflow builds."""

from pathlib import Path
import shutil

from PySide6.QtCore import QSettings, QStandardPaths


APP_ROOT = Path(__file__).resolve().parents[1]


def user_data_dir() -> Path:
    location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    path = Path(location) if location else Path.home() / ".qsflow"
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path() -> Path:
    """Use private per-user settings; migrate only a valid local legacy session."""
    target = user_data_dir() / "user_settings.ini"
    legacy = APP_ROOT / "config" / "user_settings.ini"
    if not target.exists() and legacy.exists():
        old = QSettings(str(legacy), QSettings.Format.IniFormat)
        remembered_project = old.value("session/project", "", str)
        if remembered_project and Path(remembered_project).is_dir():
            shutil.copy2(legacy, target)
    return target


def logs_dir() -> Path:
    path = user_data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
