from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Exact planar maximum of a²-gamma*b² from a center and generators.'
from fractions import Fraction as Q
from functools import cmp_to_key

def maximize(a0, b0, generators, gamma, with_witness=False):
    gamma = Q(gamma)
    assert gamma > 0
    oriented = []
    signs = [Q(-1)] * len(generators)
    for j, (x, y) in enumerate(generators):
        if x == y == 0:
            continue
        orientation = 1 if y > 0 or (y == 0 and x > 0) else -1
        oriented.append((orientation * x, orientation * y, j, orientation))
        signs[j] = -orientation

    def compare(a, b):
        cross = a[0] * b[1] - a[1] * b[0]
        return -1 if cross > 0 else 1 if cross < 0 else 0
    oriented.sort(key=cmp_to_key(compare))
    a = a0 - sum((g[0] for g in oriented), Q(0))
    b = b0 - sum((g[1] for g in oriented), Q(0))
    best = a * a - gamma * b * b
    best_signs = signs[:]
    for direction in (1, -1):
        for gx, gy, j, o in oriented:
            da = 2 * direction * gx
            db = 2 * direction * gy
            aa = da * da - gamma * db * db
            bb = 2 * (a * da - gamma * b * db)
            cc = a * a - gamma * b * b
            choices = [(cc, Q(0)), (aa + bb + cc, Q(1))]
            if aa < 0 and 0 < bb < -2 * aa:
                choices.append((cc - bb * bb / (4 * aa), -bb / (2 * aa)))
            value, t = max(choices, key=lambda z: z[0])
            if value > best:
                best = value
                best_signs = signs[:]
                best_signs[j] = o * direction * (2 * t - 1)
            a += da
            b += db
            signs[j] = o * direction
    if with_witness:
        av = a0 + sum((g[0] * z for g, z in zip(generators, best_signs)), Q(0))
        bv = b0 + sum((g[1] * z for g, z in zip(generators, best_signs)), Q(0))
        assert av * av - gamma * bv * bv == best and all((-1 <= z <= 1 for z in best_signs))
        return (best, best_signs)
    return best
