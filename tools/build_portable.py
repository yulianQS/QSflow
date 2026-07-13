"""Build a clean QSflow source distribution without personal state."""

from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DIST_ROOT = ROOT / "dist"
TARGET = DIST_ROOT / "QSflow-portable"


def copy_tree(name: str) -> None:
    source = ROOT / name
    destination = TARGET / name
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )


def main() -> int:
    resolved_dist = DIST_ROOT.resolve()
    resolved_target = TARGET.resolve()
    if resolved_dist not in resolved_target.parents:
        raise RuntimeError("Refusing to clean a target outside dist")
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)

    for folder in ("ai", "core", "ui", "voc"):
        copy_tree(folder)

    (TARGET / "assets").mkdir()
    shutil.copy2(ROOT / "assets" / "qsflow_splash.png", TARGET / "assets" / "qsflow_splash.png")
    (TARGET / "config").mkdir()
    shutil.copy2(ROOT / "config" / "settings.example.json", TARGET / "config" / "settings.example.json")

    for name in (
        "app.py", "requirements.txt", "pyproject.toml", "README.md", "LICENSE", "NOTICE",
        "分享版使用说明.txt",
        "启动_QSflow.bat", "安装_GPU_训练环境.bat", "重装运行环境.bat",
    ):
        shutil.copy2(ROOT / name, TARGET / name)

    base_model = ROOT / "yolo11n.pt"
    if base_model.exists():
        shutil.copy2(base_model, TARGET / base_model.name)

    file_count = sum(1 for path in TARGET.rglob("*") if path.is_file())
    size_mb = sum(path.stat().st_size for path in TARGET.rglob("*") if path.is_file()) / 1024 / 1024
    print(f"Portable build ready: {TARGET}")
    print(f"Files: {file_count}; size: {size_mb:.1f} MB")
    print("You may compress the QSflow-portable folder and share it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
