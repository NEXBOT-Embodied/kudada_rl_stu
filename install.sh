#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
course_python="${1:-python3.10}"
"$course_python" -c 'import sys; assert sys.version_info[:2] == (3,10), sys.version'
"$course_python" -m venv "$repo/.venv"
course_venv_python="$repo/.venv/bin/python"
"$course_venv_python" -m pip install -r "$repo/requirements.txt"
"$course_venv_python" -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
"$course_venv_python" -m pip install --no-deps "$repo/vendor/motrix_env_core" "$repo/vendor/motrix_env_motrixsim"
"$course_venv_python" -m pip check
"$course_venv_python" -X utf8 "$repo/course/cli.py" doctor
