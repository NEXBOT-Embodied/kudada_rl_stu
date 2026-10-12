# Copyright Motphys Technology Co., Ltd. 2025, 2026
# SPDX-License-Identifier: Apache-2.0

"""In-process and on-disk caches for manager fused kernels and terms."""

import logging
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_KERNEL_CACHE: dict[str, Any] = {}
_TERM_CACHE: dict[Callable[..., Any], Any] = {}


def invalidate_term_cache() -> None:
    for dispatcher in _TERM_CACHE.values():
        try:
            dispatcher._cache.flush()
        except (AttributeError, OSError):
            logger.debug("Unable to remove invalid manager term cache", exc_info=True)
    _TERM_CACHE.clear()


def generated_cache_dir() -> str:
    cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    if not cache_dir:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "motrixlab", "numba-manager")
    return cache_dir


def generated_source_path(plan_key: str) -> str:
    return str(Path(generated_cache_dir()) / "generated" / f"{plan_key}.py")


def materialize_source(source: str, plan_key: str) -> str:
    cache_dir = generated_cache_dir()
    path = Path(cache_dir) / "generated" / f"{plan_key}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != source:
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
            ) as temporary:
                temporary.write(source)
                temporary_name = temporary.name
            os.replace(temporary_name, path)
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
    return str(path)


def generated_cache_paths(plan_key: str) -> tuple[Path, ...]:
    """Return compiled-kernel cache files for one plan key.

    Numba stores the cached specializations of the generated module next to
    its cache locator: under ``generated_<hash>/`` when ``NUMBA_CACHE_DIR``
    is set, and under ``generated/__pycache__/`` (beside the generated
    source) when it is not. Both layouts are covered so probe and
    invalidation agree with where the files actually live.
    """
    cache_dir = Path(generated_cache_dir())
    paths: list[Path] = []
    for pattern in (f"generated_*/*{plan_key}*", f"generated/__pycache__/*{plan_key}*"):
        try:
            paths.extend(cache_dir.glob(pattern))
        except OSError:
            logger.debug("Unable to scan generated kernel cache: %s", cache_dir, exc_info=True)
    return tuple(paths)


def has_generated_kernel_cache(plan_key: str) -> bool:
    """Return whether compiled-kernel cache files exist for one plan key."""
    return any(path.suffix in {".nbi", ".nbc"} for path in generated_cache_paths(plan_key))


def invalidate_generated_cache(plan_key: str) -> None:
    cache_dir = generated_cache_dir()
    source_path = Path(cache_dir) / "generated" / f"{plan_key}.py"
    try:
        source_path.unlink(missing_ok=True)
    except OSError:
        logger.debug("Unable to remove invalid generated source: %s", source_path, exc_info=True)
    for cache_path in generated_cache_paths(plan_key):
        try:
            cache_path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Unable to remove invalid generated cache: %s", cache_path, exc_info=True)


__all__ = [
    "_KERNEL_CACHE",
    "_TERM_CACHE",
    "generated_cache_dir",
    "generated_cache_paths",
    "generated_source_path",
    "has_generated_kernel_cache",
    "invalidate_generated_cache",
    "invalidate_term_cache",
    "materialize_source",
]
