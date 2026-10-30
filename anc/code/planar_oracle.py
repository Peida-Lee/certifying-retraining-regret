from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Scoped certified planar replacement for v74 row oracles. No old module mutation.'
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, types
from fractions import Fraction as Q
import numpy as np
BASE = Path(__file__).resolve().parent
from functools import cmp_to_key

def exact_row_max(a0, b0, generators, gamma, with_witness=False):
    """Exact planar sweep; reconstruct the best box witness once, after search.

    O(k log k) rational comparisons and O(k) remaining arithmetic operations;
    Fraction numerator/denominator bit complexity is not a unit-cost guarantee.
    """
    gamma = Q(gamma)
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    a0 = Q(a0)
    b0 = Q(b0)
    generators = [(Q(x), Q(y)) for x, y in generators]
    oriented = []
    for j, (x, y) in enumerate(generators):
        if x == y == 0:
            continue
        o = 1 if y > 0 or (y == 0 and x > 0) else -1
        oriented.append((o * x, o * y, j, o))

    def compare(a, b):
        c = a[0] * b[1] - a[1] * b[0]
        return -1 if c > 0 else 1 if c < 0 else 0
    oriented.sort(key=cmp_to_key(compare))
    a = a0 - sum((x for x, y, j, o in oriented), Q(0))
    b = b0 - sum((y for x, y, j, o in oriented), Q(0))
    best = a * a - gamma * b * b
    best_step = None
    best_t = Q(0)
    m = len(oriented)
    for step in range(2 * m):
        direction = 1 if step < m else -1
        x, y, j, o = oriented[step % m]
        da = 2 * direction * x
        db = 2 * direction * y
        aa = da * da - gamma * db * db
        bb = 2 * (a * da - gamma * b * db)
        cc = a * a - gamma * b * b
        choices = [(cc, Q(0)), (aa + bb + cc, Q(1))]
        if aa < 0 and 0 < bb < -2 * aa:
            choices.append((cc - bb * bb / (4 * aa), -bb / (2 * aa)))
        value, t = max(choices, key=lambda z: z[0])
        if value > best:
            best = value
            best_step = step
            best_t = t
        a += da
        b += db
    if not with_witness:
        return best
    signs = [Q(-1)] * len(generators)
    for x, y, j, o in oriented:
        signs[j] = Q(-o)
    if best_step is not None:
        for step in range(best_step):
            x, y, j, o = oriented[step % m]
            signs[j] = Q(o if step < m else -o)
        direction = 1 if best_step < m else -1
        x, y, j, o = oriented[best_step % m]
        signs[j] = o * direction * (2 * best_t - 1)
    av = a0 + sum((x * z for (x, y), z in zip(generators, signs)), Q(0))
    bv = b0 + sum((y * z for (x, y), z in zip(generators, signs)), Q(0))
    assert av * av - gamma * bv * bv == best and all((-1 <= z <= 1 for z in signs))
    return (best, signs)
from interval_arithmetic import I
from rank_certificate import float_down, float_up

def row_max(a0, b0, ag, bg, gamma, with_witness=False):
    """Enclose maximum for every coefficient in input intervals, plus feasible float box witness.

    Exact midpoint optimum/witness is computed with Fraction. A uniform perturbation
    bound certifies interval coefficients; returned lower uses the float witness.
    Rational witnesses are available directly from exact_row_max(...,True).
    """
    if not isinstance(gamma, I):
        gamma = I(gamma)
    if float(gamma.lo) < 0:
        raise ValueError('gamma must be nonnegative')
    signs = np.zeros(ag.lo.shape)
    lower = []
    upper = []

    def middle(a, i):
        return (Q(float(a.lo[i])) + Q(float(a.hi[i]))) / 2

    def radius(a, i):
        return (Q(float(a.hi[i])) - Q(float(a.lo[i]))) / 2
    for i in range(len(a0.lo)):
        ac = middle(a0, i)
        bc = middle(b0, i)
        gc = (Q(float(gamma.lo)) + Q(float(gamma.hi))) / 2
        gg = [(middle(ag, (i, j)), middle(bg, (i, j))) for j in range(ag.lo.shape[1])]
        if gc == 0:
            candidates = []
            for s in (-1, 1):
                zs = [Q(s if x >= 0 else -s) for x, y in gg]
                av = ac + sum((x * z for (x, y), z in zip(gg, zs)), Q(0))
                candidates.append((av * av, zs))
            val, z = max(candidates, key=lambda z: z[0])
        else:
            val, z = exact_row_max(ac, bc, gg, gc, True)
        signs[i] = [float(v) for v in z]
        A = abs(ac) + sum((abs(x) for x, y in gg), Q(0))
        B = abs(bc) + sum((abs(y) for x, y in gg), Q(0))
        da = radius(a0, i) + sum((radius(ag, (i, j)) for j in range(len(gg))), Q(0))
        db = radius(b0, i) + sum((radius(bg, (i, j)) for j in range(len(gg))), Q(0))
        dg = (Q(float(gamma.hi)) - Q(float(gamma.lo))) / 2
        err = 2 * A * da + da * da + abs(gc) * (2 * B * db + db * db) + dg * (B + db) ** 2
        upper.append(float_up(val + err))
    aw = a0 + (ag * I(signs)).sum(axis=1)
    bw = b0 + (bg * I(signs)).sum(axis=1)
    lower = (aw.square() - gamma * bw.square()).lo
    if np.any(lower > np.array(upper)):
        raise ArithmeticError('Planar certified lower exceeds upper')
    return (I(lower, np.array(upper)), signs)

def _scoped_modules():
    anchor = types.ModuleType('planar_observed_scoped')
    anchor.__file__ = str(MODULE_DIR / 'observed_anchor.py')
    source = Path(anchor.__file__).read_text(encoding='utf-8')
    import ast
    class PlanarGuard(ast.NodeTransformer):
        def visit_If(self, node):
            if (isinstance(node.test, ast.Compare) and ast.unparse(node.test) == 'self.k > 6'
                and 'Pilot row reference limited to six uncertain cells' in ast.unparse(node)):
                return None
            return self.generic_visit(node)
    tree = PlanarGuard().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree)
    exec(compile(source, anchor.__file__, 'exec'), anchor.__dict__)
    anchor.row_max = row_max
    moment = types.ModuleType('planar_moment_scoped')
    moment.__dict__.update(anchor.__dict__)
    source = (MODULE_DIR / 'moment_tail.py').read_text(encoding='utf-8').replace('from observed_anchor import *', '')
    exec(compile(source, 'planar_moment_scoped', 'exec'), moment.__dict__)
    return (anchor, moment)
_anchor, _moment = _scoped_modules()
PlanarObservedAnchor = _anchor.ObservedAnchor
PlanarAnchorSearch = _anchor.AnchorSearch
PlanarMomentSearch = _moment.MomentSearch
