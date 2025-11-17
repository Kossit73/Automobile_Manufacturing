"""A lightweight stand-in for pandas used for automated evaluation environments.

This module implements a minimal subset of the pandas DataFrame and Series
interfaces that are required by the Automobile Manufacturing sample project.
It is **not** a drop-in replacement for pandas but provides enough
functionality to execute the demonstrations and tests without the third-party
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union, Callable
import io
import csv

USING_PANDAS_SHIM = True

__all__ = ["DataFrame", "Series", "Index", "RangeIndex"]


class Index(list):
    """Simple list-backed index with a pandas-like ``tolist`` helper."""

    def __init__(self, values: Iterable[Any] = (), name: Optional[str] = None):  # type: ignore[override]
        super().__init__(values)
        self._name: Optional[str] = name

    def tolist(self) -> List[Any]:
        return list(self)

    def copy(self) -> "Index":  # type: ignore[override]
        return Index(self, name=self._name)

    # ------------------------------------------------------------------
    # Name metadata
    # ------------------------------------------------------------------
    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, value: Optional[str]) -> None:
        self._name = value

    @property
    def names(self) -> List[Optional[str]]:
        return [self._name]

    @names.setter
    def names(self, value: Optional[Sequence[Optional[str]]]) -> None:
        if value is None:
            self._name = None
            return
        if not isinstance(value, Sequence):
            raise TypeError("Index.names expects a sequence of names")
        if len(value) != 1:
            raise ValueError("Only single-level indexes are supported")
        self._name = value[0]

    @property
    def is_unique(self) -> bool:
        """Return ``True`` when all labels in the index are unique."""

        seen = set()
        for value in self:
            if value in seen:
                return False
            seen.add(value)
        return True

    @property
    def nlevels(self) -> int:
        """Always report a single level for the lightweight index."""

        return 1

    def get_level_values(self, level: Union[int, str]) -> "Index":
        """Return the labels for the requested index level.

        Streamlit's PyArrow bridge requests level values when exporting
        DataFrames.  The shim only supports flat indexes, so any integer
        level resolves to the lone level and string-based lookups fall back
        to level ``0``.  This mirrors the behaviour pandas provides for
        ``Index`` instances without named levels.
        """

        if isinstance(level, int):
            if level not in (0, -1):
                raise IndexError("Index level out of range")
            return Index(self, name=self._name)
        if level in (None, ""):
            return Index(self, name=self._name)
        # Fall back to the single level for unnamed indexes, matching pandas
        # which treats the only level as addressable via any provided name.
        return Index(self, name=self._name)


class RangeIndex(Index):
    """Minimal implementation of :class:`pandas.RangeIndex`."""

    def __init__(self, start: int = 0, stop: Optional[int] = None, step: int = 1, name: Optional[str] = None):
        if step == 0:
            raise ValueError("RangeIndex step must not be zero")
        if stop is None:
            stop = start
            start = 0
        self.start = int(start)
        self.stop = int(stop)
        self.step = int(step)
        super().__init__(range(self.start, self.stop, self.step), name=name)

    @classmethod
    def from_range(cls, range_obj: range, name: Optional[str] = None) -> "RangeIndex":
        return cls(range_obj.start, range_obj.stop, range_obj.step, name=name)

    def copy(self) -> "RangeIndex":  # type: ignore[override]
        return RangeIndex(self.start, self.stop, self.step, name=self._name)

    def __repr__(self) -> str:
        return f"RangeIndex(start={self.start}, stop={self.stop}, step={self.step})"

    @property
    def is_unique(self) -> bool:
        return True

    def get_level_values(self, level: Union[int, str]) -> "RangeIndex":
        if isinstance(level, int):
            if level not in (0, -1):
                raise IndexError("Index level out of range")
            return RangeIndex(self.start, self.stop, self.step, name=self._name)
        if level in (None, ""):
            return RangeIndex(self.start, self.stop, self.step, name=self._name)
        return RangeIndex(self.start, self.stop, self.step, name=self._name)


def _coerce_iterable(values: Optional[Iterable[Any]]) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, Series):
        return values._data.copy()
    return list(values)


def _infer_dtype(values: Sequence[Any]) -> str:
    dtype: Optional[str] = None
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            candidate = "int64"
        elif isinstance(value, float):
            candidate = "float64"
        else:
            return "object"
        if dtype is None:
            dtype = candidate
        elif dtype != candidate:
            if {dtype, candidate} <= {"int64", "float64"}:
                dtype = "float64"
            else:
                return "object"
    return dtype or "object"


def _value_or_zero(value: Any) -> Any:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    return value


@dataclass
class Series:
    _data: List[Any]
    index: Index
    name: Optional[str] = None

    def __init__(self, data: Optional[Iterable[Any]] = None, index: Optional[Iterable[Any]] = None, name: Optional[str] = None):
        self._data = _coerce_iterable(data)
        if index is None:
            computed_index: Union[Index, RangeIndex, Iterable[Any]] = RangeIndex(len(self._data))
        elif isinstance(index, range):
            computed_index = RangeIndex.from_range(index)
        elif isinstance(index, (Index, RangeIndex)):
            computed_index = index.copy()
        else:
            computed_index = Index(index)
        self.index = computed_index if isinstance(computed_index, Index) else Index(computed_index)
        self.name = name
        if len(self.index) != len(self._data):
            raise ValueError("Series data and index must be the same length")

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __repr__(self) -> str:
        values = ", ".join(str(v) for v in self._data)
        return f"Series([{values}])"

    # ------------------------------------------------------------------
    # Indexing operations
    # ------------------------------------------------------------------
    def _loc_to_pos(self, key: Any) -> int:
        if isinstance(key, int):
            return key
        try:
            return self.index.index(key)
        except ValueError as exc:  # pragma: no cover - defensive
            raise KeyError(key) from exc

    def __getitem__(self, key: Union[int, slice, List[int], List[bool], Any]) -> Any:
        if isinstance(key, slice):
            return Series(self._data[key], index=self.index[key], name=self.name)
        if isinstance(key, list):
            if key and isinstance(key[0], bool):
                filtered = [val for val, flag in zip(self._data, key) if flag]
                filtered_index = [idx for idx, flag in zip(self.index, key) if flag]
                return Series(filtered, index=filtered_index, name=self.name)
            positions = key
            return Series([self._data[pos] for pos in positions], index=[self.index[pos] for pos in positions], name=self.name)
        pos = self._loc_to_pos(key)
        return self._data[pos]

    # ------------------------------------------------------------------
    # Core behaviours
    # ------------------------------------------------------------------
    def copy(self) -> "Series":
        return Series(self._data.copy(), index=self.index.copy(), name=self.name)

    @property
    def dtype(self) -> str:
        return _infer_dtype(self._data)

    @property
    def empty(self) -> bool:
        return len(self._data) == 0

    @property
    def values(self) -> List[Any]:
        return self._data.copy()

    def to_list(self) -> List[Any]:
        return self._data.copy()

    def tolist(self) -> List[Any]:  # pandas compatibility alias
        return self.to_list()

    def to_dict(self) -> Dict[Any, Any]:
        return {idx: value for idx, value in zip(self.index, self._data)}

    def apply(self, func: Callable[[Any], Any]) -> "Series":
        return Series([func(val) for val in self._data], index=self.index.copy(), name=self.name)

    # ------------------------------------------------------------------
    # Position-based indexing
    # ------------------------------------------------------------------
    class _ILoc:
        def __init__(self, series: "Series"):
            self._series = series

        def _normalize_key(self, key: Union[int, slice, List[int], List[bool]]) -> Union[int, List[int]]:
            data_len = len(self._series)
            if isinstance(key, slice):
                indices = list(range(data_len))[key]
                return indices
            if isinstance(key, list):
                if key and isinstance(key[0], bool):
                    if len(key) != data_len:
                        raise ValueError("Boolean index has wrong length")
                    return [i for i, flag in enumerate(key) if flag]
                return [self._to_pos(k) for k in key]
            if isinstance(key, int):
                return self._to_pos(key)
            raise TypeError("Invalid iloc indexer for Series")

        def _to_pos(self, key: int) -> int:
            data_len = len(self._series)
            if key < 0:
                key += data_len
            if key < 0 or key >= data_len:
                raise IndexError("Series iloc index out of range")
            return key

        def __getitem__(self, key: Union[int, slice, List[int], List[bool]]):
            indices = self._normalize_key(key)
            if isinstance(indices, int):
                return self._series._data[indices]
            data = [self._series._data[i] for i in indices]
            index = [self._series.index[i] for i in indices]
            return Series(data, index=index, name=self._series.name)

    @property
    def iloc(self) -> "Series._ILoc":
        return Series._ILoc(self)

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------
    def _binary_op(self, other: Union["Series", Any], op: Callable[[Any, Any], Any], reverse: bool = False) -> "Series":
        if isinstance(other, Series):
            values = []
            for a, b in zip(self._data, other._data):
                values.append(op(b, a) if reverse else op(a, b))
            return Series(values, index=self.index.copy(), name=self.name)
        values = []
        for val in self._data:
            values.append(op(other, val) if reverse else op(val, other))
        return Series(values, index=self.index.copy(), name=self.name)

    def __add__(self, other: Union["Series", Any]) -> "Series":
        return self._binary_op(other, lambda a, b: _value_or_zero(a) + _value_or_zero(b))

    def __radd__(self, other: Any) -> "Series":
        return self.__add__(other)

    def __sub__(self, other: Union["Series", Any]) -> "Series":
        return self._binary_op(other, lambda a, b: _value_or_zero(a) - _value_or_zero(b))

    def __rsub__(self, other: Any) -> "Series":
        return self._binary_op(other, lambda a, b: _value_or_zero(a) - _value_or_zero(b), reverse=True)

    def __mul__(self, other: Union["Series", Any]) -> "Series":
        return self._binary_op(other, lambda a, b: _value_or_zero(a) * _value_or_zero(b))

    def __rmul__(self, other: Any) -> "Series":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Series", Any]) -> "Series":
        if isinstance(other, Series):
            return self._binary_op(
                other,
                lambda a, b: 0 if b in (0, None) else _value_or_zero(a) / _value_or_zero(b),
            )
        return self._binary_op(other, lambda a, b: 0 if other in (0, None) else _value_or_zero(a) / _value_or_zero(other))

    def __rtruediv__(self, other: Any) -> "Series":
        return self._binary_op(
            other,
            lambda a, b: 0 if b in (0, None) else _value_or_zero(a) / _value_or_zero(b),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Comparisons and logical ops
    # ------------------------------------------------------------------
    def _compare(self, other: Union["Series", Any], op: Callable[[Any, Any], bool]) -> "Series":
        if isinstance(other, Series):
            values = [op(a, b) for a, b in zip(self._data, other._data)]
        else:
            values = [op(a, other) for a in self._data]
        return Series(values, index=self.index.copy(), name=self.name)

    def __lt__(self, other: Union["Series", Any]) -> "Series":
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other: Union["Series", Any]) -> "Series":
        return self._compare(other, lambda a, b: a <= b)

    def __gt__(self, other: Union["Series", Any]) -> "Series":
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other: Union["Series", Any]) -> "Series":
        return self._compare(other, lambda a, b: a >= b)

    def __eq__(self, other: Union["Series", Any]) -> "Series":  # type: ignore[override]
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other: Union["Series", Any]) -> "Series":  # type: ignore[override]
        return self._compare(other, lambda a, b: a != b)

    def __or__(self, other: Union["Series", Any]) -> "Series":
        return self._compare(other, lambda a, b: bool(a) or bool(b))

    def __and__(self, other: Union["Series", Any]) -> "Series":
        return self._compare(other, lambda a, b: bool(a) and bool(b))

    def __invert__(self) -> "Series":
        return Series([not bool(val) for val in self._data], index=self.index.copy(), name=self.name)

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    def sum(self) -> Any:
        total: Any = 0
        for value in self._data:
            coerced = _value_or_zero(value)
            if isinstance(coerced, (int, float)):
                total += coerced
        return total

    def mean(self) -> float:
        numeric = [int(v) if isinstance(v, bool) else v for v in self._data if isinstance(v, (int, float, bool))]
        if not numeric:
            return 0.0
        return float(sum(numeric)) / len(numeric)

    def max(self) -> Any:
        filtered = [v for v in self._data if v is not None]
        if not filtered:
            return None
        return max(filtered)

    def min(self) -> Any:
        filtered = [v for v in self._data if v is not None]
        if not filtered:
            return None
        return min(filtered)

    def cumsum(self) -> "Series":
        running = 0
        values: List[Any] = []
        for value in self._data:
            if isinstance(value, bool):
                running += int(value)
            elif isinstance(value, (int, float)):
                running += value
            elif value is None:
                running += 0
            else:
                raise TypeError("cumsum requires numeric or boolean data")
            values.append(running)
        return Series(values, index=self.index.copy(), name=self.name)


class DataFrame:
    """A tabular data container implementing a subset of pandas DataFrame."""

    def __init__(self, data: Optional[Union[Dict[str, Iterable[Any]], List[Dict[str, Any]]]] = None):
        self._data: Dict[str, List[Any]] = {}
        self.index: Index = RangeIndex(0)
        self.columns: Index = Index()
        if data is None:
            return
        if isinstance(data, dict):
            lengths = {len(_coerce_iterable(col)) for col in data.values()}
            length = lengths.pop() if lengths else 0
            for key, values in data.items():
                self._data[key] = _coerce_iterable(values)
            self.columns = Index(list(data.keys()))
            self.index = RangeIndex(length)
        elif isinstance(data, list):
            if not data:
                return
            columns: List[str] = []
            for row in data:
                for key in row.keys():
                    if key not in columns:
                        columns.append(key)
            self.columns = Index(columns)
            rows: Dict[str, List[Any]] = {col: [] for col in columns}
            for row in data:
                for col in columns:
                    rows[col].append(row.get(col))
            self._data = rows
            self.index = RangeIndex(len(data))
        else:
            raise TypeError("Unsupported data type for DataFrame")

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __repr__(self) -> str:
        return self.to_string()

    @property
    def empty(self) -> bool:
        return len(self) == 0

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.index), len(self.columns)

    # ------------------------------------------------------------------
    # Column access
    # ------------------------------------------------------------------
    def __getitem__(self, key: Union[str, Series, List[bool]]) -> Union[Series, "DataFrame"]:
        if isinstance(key, str):
            if key not in self._data:
                raise KeyError(key)
            return Series(self._data[key], index=self.index.copy(), name=key)
        if isinstance(key, Series):
            mask = [bool(v) for v in key._data]
            return self._filter_rows(mask)
        if isinstance(key, list):
            if not key:
                return DataFrame()
            if isinstance(key[0], bool):
                return self._filter_rows([bool(v) for v in key])
            raise TypeError("Only boolean row selection is supported with list keys")
        raise TypeError("Invalid key type for DataFrame")

    def __setitem__(self, key: str, value: Union[Series, Sequence[Any], Any]) -> None:
        if isinstance(value, Series):
            values = value._data
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = list(value)
        else:
            values = [value for _ in range(len(self.index))]
        if len(values) != len(self.index):
            raise ValueError("Length of values does not match DataFrame rows")
        self._data[key] = list(values)
        if key not in self.columns:
            self.columns.append(key)

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------
    def iterrows(self) -> Iterator[Tuple[Any, Series]]:
        for pos, idx in enumerate(self.index):
            row_data = [self._data[col][pos] for col in self.columns]
            yield idx, Series(row_data, index=Index(self.columns), name=idx)

    # ------------------------------------------------------------------
    # Position-based indexing
    # ------------------------------------------------------------------
    class _ILoc:
        def __init__(self, df: "DataFrame"):
            self._df = df

        def _to_pos(self, key: int) -> int:
            data_len = len(self._df.index)
            if key < 0:
                key += data_len
            if key < 0 or key >= data_len:
                raise IndexError("DataFrame iloc index out of range")
            return key

        def _normalize(self, key: Union[int, slice, List[int], List[bool]]) -> Union[int, List[int]]:
            data_len = len(self._df.index)
            if isinstance(key, slice):
                return list(range(data_len))[key]
            if isinstance(key, list):
                if key and isinstance(key[0], bool):
                    if len(key) != data_len:
                        raise ValueError("Boolean index has wrong length")
                    return [i for i, flag in enumerate(key) if flag]
                return [self._to_pos(k) for k in key]
            if isinstance(key, int):
                return self._to_pos(key)
            raise TypeError("Invalid iloc indexer for DataFrame")

        def __getitem__(self, key: Union[int, slice, List[int], List[bool]]):
            indices = self._normalize(key)
            if isinstance(indices, int):
                pos = indices
                row = [self._df._data[col][pos] for col in self._df.columns]
                return Series(row, index=Index(self._df.columns), name=self._df.index[pos])
            data = {col: [self._df._data[col][pos] for pos in indices] for col in self._df.columns}
            result = DataFrame(data)
            result.index = Index([self._df.index[pos] for pos in indices])
            return result

    @property
    def iloc(self) -> "DataFrame._ILoc":
        return DataFrame._ILoc(self)

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------
    def copy(self) -> "DataFrame":
        return DataFrame(self.to_dict())

    def to_dict(self, orient: str = "dict") -> Union[Dict[str, List[Any]], List[Dict[str, Any]]]:
        if orient in ("dict", None):
            return {col: values.copy() for col, values in self._data.items()}
        if orient == "records":
            records: List[Dict[str, Any]] = []
            for pos in range(len(self.index)):
                record = {col: self._data[col][pos] for col in self.columns}
                records.append(record)
            return records
        raise ValueError("Unsupported orient")

    def to_string(self, index: bool = True) -> str:
        if self.empty:
            return "Empty DataFrame"
        data_rows: List[List[str]] = []
        headers = [str(col) for col in self.columns]
        if index:
            headers = ["index"] + headers
        for pos, idx in enumerate(self.index):
            row = [str(self._data[col][pos]) for col in self.columns]
            if index:
                row = [str(idx)] + row
            data_rows.append(row)
        col_widths = [len(h) for h in headers]
        for row in data_rows:
            for i, value in enumerate(row):
                col_widths[i] = max(col_widths[i], len(value))
        lines = []
        header_line = " ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        for row in data_rows:
            lines.append(" ".join(str(value).ljust(col_widths[i]) for i, value in enumerate(row)))
        return "\n".join(lines)

    def to_csv(self, index: bool = True) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        headers = list(self.columns)
        if index:
            headers = ["index"] + headers
        writer.writerow(headers)
        for pos, idx in enumerate(self.index):
            row = [self._data[col][pos] for col in self.columns]
            if index:
                row = [idx] + row
            writer.writerow(row)
        return buffer.getvalue()

    def sort_values(self, by: str, ascending: bool = True) -> "DataFrame":
        if by not in self._data:
            raise KeyError(by)
        order = sorted(
            range(len(self.index)),
            key=lambda i: (self._data[by][i] is None, self._data[by][i]),
            reverse=not ascending,
        )
        sorted_rows = []
        for pos in order:
            sorted_rows.append({col: self._data[col][pos] for col in self.columns})
        df = DataFrame(sorted_rows)
        df.index = Index([self.index[pos] for pos in order])
        return df

    def isnull(self) -> "DataFrame":
        return DataFrame({col: [value is None for value in values] for col, values in self._data.items()})

    def sum(self) -> Series:
        totals: Dict[str, Any] = {}
        for col in self.columns:
            values = self._data[col]
            total = 0
            for value in values:
                coerced = _value_or_zero(value)
                if isinstance(coerced, (int, float)):
                    total += coerced
            totals[col] = total
        return Series(list(totals.values()), index=Index(totals.keys()))

    def duplicated(self) -> Series:
        seen = set()
        duplicates: List[bool] = []
        for pos in range(len(self.index)):
            row_tuple = tuple(self._data[col][pos] for col in self.columns)
            if row_tuple in seen:
                duplicates.append(True)
            else:
                seen.add(row_tuple)
                duplicates.append(False)
        return Series(duplicates, index=self.index.copy())

    @property
    def dtypes(self) -> Series:
        dtype_map = {col: _infer_dtype(values) for col, values in self._data.items()}
        return Series(list(dtype_map.values()), index=Index(dtype_map.keys()))

    def _filter_rows(self, mask: Sequence[bool]) -> "DataFrame":
        if len(mask) != len(self.index):
            raise ValueError("Boolean index has wrong length")
        filtered_rows = []
        filtered_index = []
        for pos, (flag, idx) in enumerate(zip(mask, self.index)):
            if flag:
                filtered_rows.append({col: self._data[col][pos] for col in self.columns})
                filtered_index.append(idx)
        if filtered_rows:
            df = DataFrame(filtered_rows)
        else:
            df = DataFrame({col: [] for col in self.columns})
        df.index = Index(filtered_index)
        return df


def Series_from_dict(data: Dict[Any, Any]) -> Series:
    return Series(list(data.values()), index=Index(data.keys()))


def DataFrame_from_records(records: List[Dict[str, Any]]) -> DataFrame:
    return DataFrame(records)
