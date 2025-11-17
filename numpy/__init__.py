"""Runtime shim that defers to real numpy when available."""
from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Iterable, Sequence, Union


def _remove_shim_paths() -> list[str]:
    """Temporarily remove shim paths so the real package can be imported."""

    package_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(package_dir)
    removed: list[str] = []
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


def _import_real_numpy() -> ModuleType | None:
    """Attempt to import the system-installed numpy distribution."""

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


_real_numpy = None
if not os.environ.get("NUMPY_SHIM_ONLY"):
    _real_numpy = _import_real_numpy()

if _real_numpy is not None:
    globals().update(_real_numpy.__dict__)
    sys.modules[__name__] = _real_numpy
else:
    import math
    import random as _random
    import statistics

    Number = Union[int, float]

    # ---------------- Array helper -----------------
    class SimpleArray:
        def __init__(self, data):
            if isinstance(data, SimpleArray):
                self.data = data.data
            else:
                self.data = _to_nested_list(data)

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

        @property
        def T(self):
            if not self.data:
                return SimpleArray([])
            if not isinstance(self.data[0], (list, tuple, SimpleArray)):
                return SimpleArray([[v] for v in self.data])
            rows = [[row[i] for row in self.data] for i in range(len(self.data[0]))]
            return SimpleArray(rows)

        def _binary_op(self, other, op):
            other_data = other.data if isinstance(other, SimpleArray) else other
            return SimpleArray(_apply_elementwise(self.data, other_data, op))

        def __add__(self, other):
            return self._binary_op(other, lambda a, b: a + b)

        def __sub__(self, other):
            return self._binary_op(other, lambda a, b: a - b)

        def __mul__(self, other):
            return self._binary_op(other, lambda a, b: a * b)

        __rmul__ = __mul__

        def __truediv__(self, other):
            return self._binary_op(other, lambda a, b: a / b)

        def __pow__(self, power, modulo=None):
            return self._binary_op(power, lambda a, b: a ** b)

        def __matmul__(self, other):
            other_data = other.data if isinstance(other, SimpleArray) else other
            return SimpleArray(_matmul(self.data, other_data))

        def to_list(self):
            return self.data

        def __repr__(self):
            return f"SimpleArray({self.data!r})"


# --------------- core helpers ------------------
def _to_nested_list(data):
    if isinstance(data, SimpleArray):
        return data.data
    if isinstance(data, (list, tuple)):
        return [_to_nested_list(d) for d in data]
    return data


def _apply_elementwise(a, b, op):
    if isinstance(a, (list, tuple)):
        if isinstance(b, (list, tuple)):
            return [op(x, y) for x, y in zip(a, b)]
        return [op(x, b) for x in a]
    if isinstance(b, (list, tuple)):
        return [op(a, y) for y in b]
    return op(a, b)


def _flatten(values):
    if isinstance(values, SimpleArray):
        values = values.data
    for v in values:
        if isinstance(v, (list, tuple, SimpleArray)):
            yield from _flatten(v)
        else:
            yield v


# ---------------- public API -------------------
def array(data):
    return SimpleArray(data)


def ones(length: int):
    return SimpleArray([1.0 for _ in range(length)])


def zeros_like(seq):
    return SimpleArray([0.0 for _ in range(len(seq))])


def zeros(shape):
    if isinstance(shape, int):
        return SimpleArray([0.0 for _ in range(shape)])
    rows, cols = shape
    return SimpleArray([[0.0 for _ in range(cols)] for _ in range(rows)])


def sum(values):
    return builtins_sum(list(_flatten(values)))


def mean(values):
    vals = list(_flatten(values))
    return statistics.mean(vals) if vals else 0.0


def median(values):
    vals = list(_flatten(values))
    return statistics.median(vals) if vals else 0.0


def std(values):
    vals = list(_flatten(values))
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def var(values):
    vals = list(_flatten(values))
    return statistics.pvariance(vals) if len(vals) > 1 else 0.0


def min(values):
    return builtins_min(list(_flatten(values)))


def max(values):
    return builtins_max(list(_flatten(values)))


def percentile(values, pct):
    vals = sorted(list(_flatten(values)))
    if not vals:
        return 0.0
    k = (len(vals) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    return vals[f] * (c - k) + vals[c] * (k - f)


def diff(values):
    vals = list(values)
    return SimpleArray([vals[i + 1] - vals[i] for i in range(len(vals) - 1)])


def clip(values, min_value, max_value):
    return SimpleArray([max(min_value, min(max_value, v)) for v in values])


def linspace(start, stop, num):
    if num == 1:
        return SimpleArray([float(start)])
    step = (stop - start) / (num - 1)
    return SimpleArray([start + i * step for i in range(num)])


def convolve(values, kernel, mode="same"):
    vals = list(values)
    kern = list(kernel)
    output = []
    pad = len(kern) - 1
    padded = [0] * pad + vals + [0] * pad
    for i in range(len(vals)):
        window = padded[i : i + len(kern)]
        output.append(sum(a * b for a, b in zip(window, reversed(kern))))
    return SimpleArray(output)


def cov(matrix):
    data = [[float(x) for x in row] for row in matrix]
    n = len(data[0]) if data else 0
    result = [[0.0 for _ in range(n)] for _ in range(n)]
    means = [statistics.mean(col) for col in zip(*data)] if data else []
    for i in range(n):
        for j in range(n):
            denom = (len(data) - 1) if len(data) > 1 else 1
            result[i][j] = sum((row[i] - means[i]) * (row[j] - means[j]) for row in data) / denom
    return SimpleArray(result)


def corrcoef(x, y=None):
    xs = list(_flatten(x))
    ys = list(_flatten(y if y is not None else x))
    if len(xs) != len(ys) or len(xs) < 2:
        return SimpleArray([[1.0, 0.0], [0.0, 1.0]])
    mean_x = mean(xs)
    mean_y = mean(ys)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys))
    denom = math.sqrt(sum((a - mean_x) ** 2 for a in xs) * sum((b - mean_y) ** 2 for b in ys)) or 1
    r = num / denom
    return SimpleArray([[1.0, r], [r, 1.0]])


def column_stack(arrays: Sequence[Iterable]):
    cols = [list(a) for a in arrays]
    rows = list(zip(*cols))
    return SimpleArray([list(r) for r in rows])


def sqrt(value):
    if isinstance(value, SimpleArray):
        return SimpleArray([math.sqrt(v) for v in value])
    return math.sqrt(value)


def percentile_array(values, percent):
    return percentile(values, percent)


def array_sum(values):
    return sum(values)


# ------------- linear algebra ------------------
def _matmul(a, b):
    a_list = a.data if isinstance(a, SimpleArray) else a
    b_list = b.data if isinstance(b, SimpleArray) else b
    if not isinstance(a_list[0], (list, tuple)):
        b_rows = b_list if isinstance(b_list[0], (list, tuple)) else [[x] for x in b_list]
        return [sum(x * y for x, y in zip(a_list, col)) for col in zip(*b_rows)]
    result = []
    for row in a_list:
        res_row = []
        for col in zip(*b_list):
            res_row.append(sum(x * y for x, y in zip(row, col)))
        result.append(res_row)
    return result


class _Linalg:
    @staticmethod
    def inv(matrix):
        m = [[float(x) for x in row] for row in (matrix.data if isinstance(matrix, SimpleArray) else matrix)]
        n = len(m)
        identity = [[float(i == j) for j in range(n)] for i in range(n)]
        for i in range(n):
            pivot = m[i][i]
            if pivot == 0:
                for j in range(i + 1, n):
                    if m[j][i] != 0:
                        m[i], m[j] = m[j], m[i]
                        identity[i], identity[j] = identity[j], identity[i]
                        pivot = m[i][i]
                        break
            if pivot == 0:
                raise ValueError("Singular matrix")
            m[i] = [x / pivot for x in m[i]]
            identity[i] = [x / pivot for x in identity[i]]
            for j in range(n):
                if j == i:
                    continue
                factor = m[j][i]
                m[j] = [a - factor * b for a, b in zip(m[j], m[i])]
                identity[j] = [a - factor * b for a, b in zip(identity[j], identity[i])]
        return SimpleArray(identity)


linalg = _Linalg()


# ---------------- random -----------------------
class _Random:
    def seed(self, value):
        _random.seed(value)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return [_random.gauss(loc, scale) for _ in range(size or 1)]

    def uniform(self, low=0.0, high=1.0, size=None):
        return [_random.uniform(low, high) for _ in range(size or 1)]

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        return [_random.lognormvariate(mean, sigma) for _ in range(size or 1)]

    def triangular(self, left, mode, right, size=None):
        return [_random.triangular(left, right, mode) for _ in range(size or 1)]


random = _Random()


# --------------- polynomial fit ----------------
def polyfit(x_values, y_values, degree):
    if degree != 1:
        raise NotImplementedError("Only degree=1 supported in shim")
    n = len(x_values)
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values) or 1
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return [slope, intercept]


# aliases
ndarray = SimpleArray
builtins_sum = __builtins__["sum"]
builtins_min = __builtins__["min"]
builtins_max = __builtins__["max"]
