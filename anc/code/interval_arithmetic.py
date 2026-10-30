from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Small outward-rounded interval arithmetic for bounded float64 inputs.\n\nEvery input float is interpreted as its exact binary value. Operations assume\nfinite IEEE-754 round-to-nearest results, checked explicitly at public outputs.\n'
from dataclasses import dataclass
import numpy as np
NEG = -np.inf
POS = np.inf
EPS = np.finfo(float).eps
TINY = np.nextafter(0.0, 1.0)

def down(x):
    return np.nextafter(x, NEG)

def up(x):
    return np.nextafter(x, POS)

@dataclass
class I:
    lo: object
    hi: object = None

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=float)
        self.hi = self.lo.copy() if self.hi is None else np.asarray(self.hi, dtype=float)

    def __add__(self, other):
        b = as_i(other)
        return I(down(self.lo + b.lo), up(self.hi + b.hi))
    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -as_i(other)

    def __rsub__(self, other):
        return as_i(other) + -self

    def __mul__(self, other):
        b = as_i(other)
        products = np.stack(np.broadcast_arrays(self.lo * b.lo, self.lo * b.hi, self.hi * b.lo, self.hi * b.hi))
        return I(down(np.min(products, axis=0)), up(np.max(products, axis=0)))
    __rmul__ = __mul__

    def __truediv__(self, other):
        b = as_i(other)
        if np.any((b.lo <= 0) & (b.hi >= 0)):
            raise ValueError('Interval division through zero')
        return self * I(down(1 / b.hi), up(1 / b.lo))

    def __getitem__(self, index):
        return I(self.lo[index], self.hi[index])

    def square(self):
        lo = np.minimum(self.lo * self.lo, self.hi * self.hi)
        lo = np.where((self.lo <= 0) & (self.hi >= 0), 0, down(lo))
        return I(np.maximum(0, lo), up(np.maximum(self.lo * self.lo, self.hi * self.hi)))

    def sum(self, axis=None):
        n = self.lo.size if axis is None else self.lo.shape[axis]
        gamma = up(n * EPS / (1 - n * EPS))
        mag = up(np.sum(np.maximum(np.abs(self.lo), np.abs(self.hi)), axis=axis) / (1 - gamma))
        error = up(gamma * mag + n * TINY)
        return I(down(np.sum(self.lo, axis=axis) - error), up(np.sum(self.hi, axis=axis) + error))

def as_i(value):
    return value if isinstance(value, I) else I(value)

def matvec(matrix, vector):
    a, b = (as_i(matrix), as_i(vector))
    out = I(np.zeros(a.lo.shape[:-1]))
    for j in range(a.lo.shape[-1]):
        out = out + a[..., j] * b[j]
    return out

def dot(a, b):
    return (as_i(a) * as_i(b)).sum()

def quadratic_max_upper(a, b, c):
    """Upper for max_{|z|<=1} a*z²+b*z+c with interval coefficients."""
    a, b, c = (as_i(a), as_i(b), as_i(c))
    aa = a.hi
    bb = np.maximum(np.abs(b.lo), np.abs(b.hi))
    endpoint = up(aa + bb)
    interior = (aa < 0) & (bb < -2 * aa)
    denom = np.where(interior, -4 * aa, 1.0)
    vertex = up(up(bb * bb) / denom)
    return up(c.hi + np.where(interior, vertex, endpoint))

def quadratic_range(a, b, c):
    return I(-quadratic_max_upper(-as_i(a), -as_i(b), -as_i(c)), quadratic_max_upper(a, b, c))
