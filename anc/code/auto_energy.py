from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Generic rank-one energy verifier from actual input intervals and proposal.'
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import sys
from pathlib import Path
from fractions import Fraction as F
from math import isqrt
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
import numpy as np
from observed_anchor import I, mm, matvec, transpose, interval_q, qdot, float_up, float_down, AnchorSearch
from moment_tail import MomentSearch

def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))

def sqrange(a, b, c):
    vals = [a - b + c, a + b + c]
    if a and -1 < -b / (2 * a) < 1:
        vals.append(c - b * b / (4 * a))
    return (min(vals), max(vals))

def maxabs(interval):
    return max((abs(v) for v in interval))

def minabs(interval):
    l, h = interval
    return F(0) if l <= 0 <= h else min(abs(l), abs(h))

def sqrt_upper(x, bits=40):
    x = F(x)
    if x < 0:
        raise ValueError('Negative squared norm')
    units = x * 2 ** (2 * bits)
    q = isqrt(units.numerator // units.denominator)
    if F(q * q) < units:
        q += 1
    ans = F(q, 2 ** bits)
    assert ans * ans >= x
    return ans

def derive_constants(a, R):
    lo = [[F(v) for v in row] for row in a.lo]
    hi = [[F(v) for v in row] for row in a.hi]
    beta = [F(v) for v in a.beta]
    y = [F(v) for v in a.y]
    n, p = (a.n, a.p)
    lam = F(a.lam)
    numeric = [F(v) for v in np.asarray(R)[:, 0]]
    r = [dot([F(v) for v in row], numeric) for row in a.t]
    h = dot(r, r)
    if h <= 0:
        raise ValueError('Zero physical proposal')
    centers = [[(x + z) / 2 for x, z in zip(l, u)] for l, u in zip(lo, hi)]
    widths = [[(z - x) / 2 for x, z in zip(l, u)] for l, u in zip(lo, hi)]
    rows = []
    for i, (c, w, yy) in enumerate(zip(centers, widths, y)):
        js = [j for j, v in enumerate(w) if v]
        if len(js) > 1:
            raise ValueError('Generic pilot verifier requires at most one uncertain cell per row')
        k = js[0] if js else 0
        s = w[k] if js else F(0)
        residual = dot(c, beta) - yy
        resgen = s * beta[k]
        proj = dot(c, r)
        projgen = s * r[k]
        residual_range = (residual - abs(resgen), residual + abs(resgen))
        projrange = (proj - abs(projgen), proj + abs(projgen))

        def discarded_norm(x):
            return dot(x, x) - dot(x, r) ** 2 / h
        endpoints = []
        for sign in (-1, 1):
            x = c.copy()
            x[k] += sign * s
            endpoints.append(discarded_norm(x))
        assert min(endpoints) >= 0
        rows.append(dict(index=i, score=maxabs(residual_range), residual_center=residual, residual_generator=resgen, projection_center=proj, projection_generator=projgen, projection_range=projrange, residual_range=residual_range, discarded_norm_squared_upper=max(endpoints)))
    threshold = max((row['score'] for row in rows)) / 2
    signal = [v for v in rows if v['score'] >= threshold]
    nuisance = [v for v in rows if v['score'] < threshold]
    retained_norm2 = lambda group: sum((maxabs(v['projection_range']) ** 2 / (n * h) for v in group), F(0))
    residual_norm2 = lambda group: sum((maxabs(v['residual_range']) ** 2 / n for v in group), F(0))
    a2 = retained_norm2(nuisance)
    sigma2 = retained_norm2(signal)
    eps2 = residual_norm2(nuisance)
    E2 = residual_norm2(signal)
    b2 = sum((v['discarded_norm_squared_upper'] / n for v in signal), F(0))
    d2 = dot(beta, beta) - dot(beta, r) ** 2 / h
    kappa = lam + sum((minabs(v['projection_range']) ** 2 / (n * h) for v in signal), F(0))
    gl = gu = lam * dot(beta, r)
    for v in signal:
        aa = v['projection_generator'] * v['residual_generator']
        bb = v['projection_center'] * v['residual_generator'] + v['projection_generator'] * v['residual_center']
        cc = v['projection_center'] * v['residual_center']
        l, u = sqrange(aa, bb, cc)
        gl += l / n
        gu += u / n
    G2 = max(abs(gl), abs(gu)) ** 2 / h
    au, eps, sigma, E, b, d, G = [sqrt_upper(z) for z in (a2, eps2, sigma2, E2, b2, d2, G2)]
    sqrtlam = sqrt_upper(lam)
    omega = sqrt_upper(b * b / (lam + b * b))
    kb = lam * (kappa + b * b) / (lam + b * b)
    physical_radius = (G + au * eps + omega * (au * E + sigma * eps) / 2 + omega * omega * sigma * E + (au * sqrtlam + sigma * b) * d) / kb
    eta = (eps + au * physical_radius + omega * (E + sigma * physical_radius) + sqrtlam * d) ** 2
    rawroot = sqrt_upper(physical_radius * physical_radius / h)
    units = rawroot * 2 ** 20
    root = F((units.numerator + units.denominator - 1) // units.denominator, 2 ** 20)
    assert h * root * root >= physical_radius * physical_radius
    fractions = dict(h=h, a_squared=a2, epsilon_squared=eps2, sigma_squared=sigma2, E_squared=E2, b_squared=b2, d_squared=d2, G_squared=G2, a=au, epsilon=eps, sigma=sigma, E=E, b=b, d=d, G=G, kappa=kappa, omega=omega, sqrt_lambda=sqrtlam, kappa_b=kb, physical_radius=physical_radius, eta=eta, coordinate_radius=root, partition_threshold=threshold, signal_gradient_r_lower=gl, signal_gradient_r_upper=gu)
    return dict(status='PASS', verifier='generic_rank1_original_row_leakage_energy', one_uncertain_cell_per_row_verified=True, partition_rule='score=max_box_abs_actual_candidate_residual; signal iff score>=half_max_score', signal_rows=[v['index'] for v in signal], nuisance_rows=[v['index'] for v in nuisance], row_scores=[str(v['score']) for v in rows], physical_direction_r=[str(v) for v in r], numerical_basis=[str(v) for v in numeric], implicit_unit_Q='r/sqrt(h); Pi=rr.T/h; no rounded physical span substitution', fractions={k: str(v) for k, v in fractions.items()}, candidate_component_parallel_squared=str(dot(beta, r) ** 2 / h), automatic_direction=True, fixed_direction_override=False)

class AutoEnergySearch(MomentSearch):

    def __init__(self, a, R, receipt):
        self.a = a
        self.rank = 1
        self.mode = 'adaptive'
        self.kind = 'automatic_verified_leakage_energy'
        self.r = np.asarray(R, float).reshape(a.p, 1)
        self.structural_receipt = receipt
        self.qH = [[dot([F(v) for v in self.r[:, 0]], [F(v) for v in self.r[:, 0]])]]
        self.H = interval_q(self.qH)
        self.hmu = float(self.H.lo[0, 0])
        if self.hmu <= 0:
            raise ValueError('Actual retained Gram not verified positive')
        self.rho = 0.0
        self.qgamma = F(1)
        self.gamma = I(1.0)
        self.qtail = F(receipt['fractions']['eta'])
        self.anchor_error = F(0)
        self.K = mm(I(self.r.T), mm(a.metric, I(self.r)))
        qb = [F(v) for v in a.beta]
        g0 = [qdot(row, qb) - v for row, v in zip(a.qG, a.qr)]
        self.tr = mm(I(a.t), I(self.r))
        self.bproj = mm(a.ct, I(self.r))
        self.linear = matvec(transpose(self.tr), I([float_down(v) for v in g0], [float_up(v) for v in g0]))
        pr = (I(a.width) * I(np.maximum(abs(self.tr[:, 0].lo), abs(self.tr[:, 0].hi))[a.cells])).sum(axis=1)
        bmax = np.maximum(abs(self.bproj[:, 0].lo), abs(self.bproj[:, 0].hi))
        self.hdiag = np.array([float((self.K[0, 0] + (I(bmax) + pr).square().sum() / a.n).hi)])
        self.radius = float(F(receipt['fractions']['coordinate_radius']))
        self.initial_upper = float_up(F(a.gradient_upper) + self.qtail)
        self.best_search = -np.inf
        self.best_original = 0.0
        self.best_witness = None
        self.cache = {}
        self.points = []
        self.tree = []

    def run(self, **kwargs):
        result = AnchorSearch.run(self, **kwargs)
        result.update(method='automatic_leakage_energy', basis_kind=self.kind, structural_receipt=self.structural_receipt, search_values_include_additive_tail=True)
        if 'certificate' in result:
            result['certificate'].update(method='automatic_leakage_energy', basis_kind=self.kind, structural_receipt=self.structural_receipt, hdiag=self.hdiag.tolist(), tail_error=str(self.qtail), anchor_error='0', coordinate_radius=self.radius, search_bound=self.mode, initial_upper=self.initial_upper, original_gradient_upper=self.a.gradient_upper, search_values_include_additive_tail=True)
        return result
