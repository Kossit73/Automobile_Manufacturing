"""Runtime shim that defers to real pandas when available."""

from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Iterable, List


def _remove_shim_paths() -> List[str]:
    """Temporarily remove paths that point at the bundled shim package."""

    package_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(package_dir)
    removed: List[str] = []
    for path in list(sys.path):
        abs_path = os.path.abspath(path or ".")
        if abs_path in {package_dir, project_root}:
            removed.append(path)
            sys.path.remove(path)
    return removed


def _restore_paths(paths: Iterable[str]) -> None:
    for path in reversed(list(paths)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _import_real_pandas() -> ModuleType | None:
    """Attempt to import the system-installed pandas distribution."""

    shim_name = __name__
    existing = sys.modules.get(shim_name)
    sys.modules.pop(shim_name, None)
    removed_paths = _remove_shim_paths()
    try:
        return importlib.import_module(shim_name)
    except ModuleNotFoundError:
        if existing is not None:
            sys.modules[shim_name] = existing
        return None
    finally:
        _restore_paths(removed_paths)


_real_pandas = None
if not os.environ.get("PANDAS_SHIM_ONLY"):
    _real_pandas = _import_real_pandas()

if _real_pandas is not None:
    globals().update(_real_pandas.__dict__)
    sys.modules[__name__] = _real_pandas
    USING_PANDAS_SHIM = False
else:
    from ._shim import *  # type: ignore  # noqa: F401,F403
    from ._shim import USING_PANDAS_SHIM  # type: ignore  # noqa: F401

