"""
Root-level conftest.py — environment safety shims for the DecisionLENS test suite.

Some optional packages (shap, numba, llvmlite) can fail at import time with
OSError (missing shared library) on certain Python / platform combinations,
particularly Python 3.14 where llvmlite wheels may not yet be fully supported.

src/models.py guards these with ``try: import shap / except ImportError``, but
OSError is not caught there.  Pre-stubbing the packages here before any test
module is imported ensures:
  • The OSError is never raised during test collection.
  • src.models._SHAP_AVAILABLE is set to False (graceful degradation).
  • Tests that exercise SHAP can be skipped explicitly with
    ``pytest.importorskip("shap")``.
"""

from __future__ import annotations

import sys
import types


def _stub_if_broken(package: str) -> None:
    """
    If ``package`` is not yet imported AND its import would raise an error
    that is NOT ``ImportError`` (e.g. OSError from a missing shared library),
    pre-insert a dummy ``None`` entry in sys.modules so that subsequent
    ``import package`` statements raise ``ImportError`` instead of ``OSError``.

    A ``None`` value in sys.modules causes Python to raise:
        ImportError: import of <package> halted; use of sys.modules[...] ...
    which is exactly what the ``except ImportError`` guards in src/* expect.
    """
    if package in sys.modules:
        return  # already imported (or already stubbed)
    try:
        __import__(package)
    except ImportError:
        pass  # normal failure — already results in ImportError, no shim needed
    except Exception:
        # Non-ImportError failure (OSError, etc.) — pre-stub with None so that
        # any subsequent 'import package' raises ImportError instead
        sys.modules[package] = None  # type: ignore[assignment]


# Packages that are optional and may fail with OSError on some platforms
_OPTIONAL_PACKAGES = ["shap", "numba", "llvmlite"]
for _pkg in _OPTIONAL_PACKAGES:
    _stub_if_broken(_pkg)
