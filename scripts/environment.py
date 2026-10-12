"""Shared Windows/Linux installer for the repository-local course environment."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE_INDEX = "https://mirrors.ustc.edu.cn/pypi/simple"
DEFAULT_TORCH_INDEX = "https://mirrors.nju.edu.cn/pytorch/whl/cpu"


def check_environment() -> None:
    if sys.version_info[:2] != (3, 10) or sys.maxsize <= 2**32:
        raise RuntimeError("64-bit Python 3.10.x is required; recreate .venv.")
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError("Use the repository .venv via install.ps1/install.sh or run.ps1/run.sh.")
    settings = {
        key.strip(): value.strip().lower()
        for line in (ROOT / ".venv" / "pyvenv.cfg").read_text().splitlines() if "=" in line
        for key, value in [line.split("=", 1)]
    }
    if settings.get("include-system-site-packages") != "false":
        raise RuntimeError(".venv must isolate system packages; recreate it without --system-site-packages.")


def run_python(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def install(package_index: str, torch_index: str) -> None:
    print(f"Installing into: {sys.executable}", flush=True)
    print(f"PyPI index: {package_index}\nPyTorch index: {torch_index}", flush=True)
    run_python("-m", "ensurepip", "--upgrade")
    run_python("-m", "pip", "config", "--site", "set", "global.index-url", package_index)
    run_python("-m", "pip", "install", "-r", str(ROOT / "requirements.lock.txt"), "--index-url", package_index)
    run_python("-m", "pip", "install", "torch==2.7.0+cpu", "--index-url", torch_index)
    run_python(
        "-m", "pip", "install", "--no-deps", "--index-url", package_index,
        str(ROOT / "vendor" / "motrix_env_core"), str(ROOT / "vendor" / "motrix_env_motrixsim"),
    )
    run_python("-m", "pip", "check")
    run_python("-X", "utf8", str(ROOT / "course" / "cli.py"), "doctor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    setup = commands.add_parser("install")
    setup.add_argument("--package-index", default=DEFAULT_PACKAGE_INDEX)
    setup.add_argument("--torch-index", default=DEFAULT_TORCH_INDEX)
    args = parser.parse_args()
    try:
        check_environment()
        if args.command == "install":
            install(args.package_index, args.torch_index)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Environment error: {error}\n")


if __name__ == "__main__":
    main()
