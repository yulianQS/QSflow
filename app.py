from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QSplashScreen

from ui.main_window import MainWindow
from core.app_paths import logs_dir
from core.version import APP_VERSION


def configure_diagnostics() -> Path:
    log_path = logs_dir() / "qsflow.log"
    handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)

    def report_unhandled(exc_type, exc_value, exc_traceback):
        logging.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
        detail = "".join(traceback.format_exception_only(exc_type, exc_value)).strip()
        QMessageBox.critical(None, "QSflow 发生异常", f"{detail}\n\n诊断日志已保存到：\n{log_path}")

    sys.excepthook = report_unhandled
    return log_path


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("QSflow")
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("K&C")
    app.setOrganizationDomain("local.qsflow")
    app.setStyle("Fusion")
    log_path = configure_diagnostics()
    logging.info("QSflow starting; Python=%s; log=%s", sys.version.split()[0], log_path)
    splash_path = Path(__file__).resolve().parent / "assets" / "qsflow_splash.png"
    splash = QSplashScreen(QPixmap(str(splash_path))) if splash_path.exists() else None
    if splash: splash.show(); app.processEvents()
    window = MainWindow()
    project = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if project and project.exists():
        window.open_project(project)
    else:
        window.restore_last_project()
    window.show()
    if splash: QTimer.singleShot(1200, splash.close)
    result = app.exec()
    logging.info("QSflow exited with code %s", result)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
