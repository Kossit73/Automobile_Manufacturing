"""Sparse array placeholders for the pandas shim.

The Streamlit/pyarrow bridge checks for pandas sparse array types during
conversion.  The minimal shim does not implement sparse data structures, but
we expose lightweight stand-ins so ``isinstance`` checks succeed and the
compatibility layer treats shimmed objects as dense.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class SparseDtype:
    """Placeholder sparse dtype descriptor."""

    subtype: Optional[str] = None
    fill_value: Any = None


class SparseArray:
    """Fallback sparse array implementation.

    The shim never produces sparse values, but certain libraries expect the
    ``pandas.core.arrays.sparse`` module to be importable and to expose a
    ``SparseArray`` type for ``isinstance`` checks.  This minimal class stores
    the provided data without offering sparse semantics.
    """

    def __init__(self, data: Optional[Any] = None, dtype: Optional[SparseDtype] = None, fill_value: Any = None):
        self.data = data
        self.dtype = dtype or SparseDtype()
        self.fill_value = fill_value

    def to_dense(self) -> Any:
        """Return the stored data as a dense representation."""

        return self.data


__all__ = ["SparseArray", "SparseDtype"]
