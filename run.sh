#!/usr/bin/env bash
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
course_venv_python="$repo/.venv/bin/python"
if [[ ! -x "$course_venv_python" ]]; then
    echo 'Repository .venv Python not found. Run: bash install.sh' >&2
    exit 1
fi
"$course_venv_python" -X utf8 "$repo/scripts/environment.py" check
cd -- "$repo"
exec "$course_venv_python" -X utf8 "$repo/course/cli.py" "$@"
