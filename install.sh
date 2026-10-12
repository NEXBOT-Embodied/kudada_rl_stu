#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
course_python="${1:-}"
course_package_index="${2:-https://mirrors.ustc.edu.cn/pypi/simple}"
course_torch_index="${3:-https://mirrors.nju.edu.cn/pytorch/whl/cpu}"
course_venv_python="$repo/.venv/bin/python"

if [[ ! -e "$repo/.venv" ]]; then
    if [[ -n "$course_python" ]]; then
        "$course_python" -c 'import sys; assert sys.version_info[:2] == (3, 10) and sys.maxsize > 2**32, "64-bit Python 3.10.x is required"'
        "$course_python" -m venv "$repo/.venv"
    elif command -v uv >/dev/null 2>&1; then
        UV_PYTHON_INSTALL_MIRROR="${UV_PYTHON_INSTALL_MIRROR:-https://mirrors.nju.edu.cn/github-release/astral-sh/python-build-standalone}" \
            uv venv --seed --python 3.10.21 --index-url "$course_package_index" "$repo/.venv"
    else
        echo 'Install uv, or run: bash install.sh /path/to/python3.10' >&2
        exit 1
    fi
else
    echo "Reusing: $repo/.venv"
fi
if [[ ! -x "$course_venv_python" ]]; then
    echo 'Incomplete .venv. Rename it to keep a backup, then rerun install.sh.' >&2
    exit 1
fi
"$course_venv_python" -X utf8 "$repo/scripts/environment.py" install --package-index "$course_package_index" --torch-index "$course_torch_index"
echo 'Ready. Run: bash run.sh doctor'
