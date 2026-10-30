from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Rational verifier for observed-anchor energy contraction, arbitrary rank.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import sys, itertools
from fractions import Fraction as F
from unittest.mock import patch
import numpy as np
B = Path(__file__).resolve().parent
ROOT = B.parent
sys.dont_write_bytecode = True
from auto_energy import sqrt_upper, dot
from observed_anchor import ObservedAnchor, qarray, exact_inverse, eigen_gershgorin, interval_q, float_up
import moment_tail
from moment_tail import MomentSearch

def qform(z, H):
    return dot(z, [dot(row, z) for row in H])

def proposal(a, rank):
    qb = [F(v) for v in a.beta]
    g0 = [dot(row, qb) - v for row, v in zip(a.qG, a.qr)]
    qh = [dot(col, g0) for col in zip(*qarray(a.t))]
    h = np.array(qh, float)
    vectors = moment_tail.basis_proposal(a, 'gradient')
    if np.linalg.norm(h) > 1e-20:
        vectors = np.column_stack([h / np.linalg.norm(h), vectors])
    R = np.linalg.qr(vectors)[0][:, :rank]
    return np.round(R * 2 ** 24) / 2 ** 24

def constants(a, R):
    assert a.anchor_type == 'observed' and a.preconditioner_type == 'anchor'
    n, p = (a.n, a.p)
    r = R.shape[1]
    qr = qarray(R)
    cols = list(map(list, zip(*qr)))
    H = [[dot(x, y) for y in cols] for x in cols]
    Hi = exact_inverse(H)
    hmu = eigen_gershgorin(interval_q(H))
    assert hmu > 0
    assert all((sum((H[i][k] * Hi[k][j] for k in range(r))) == int(i == j) for i in range(r) for j in range(r)))
    qb = [F(v) for v in a.beta]
    g0 = [dot(row, qb) - v for row, v in zip(a.qG, a.qr)]
    qt = qarray(a.t)
    h = [dot(col, g0) for col in zip(*qt)]
    hr = [dot(col, h) for col in cols]
    G2 = qform(hr, Hi)
    d2 = dot(h, h) - G2
    assert G2 >= 0 and d2 >= 0
    U = [[dot(row, col) for col in cols] for row in qt]
    ucols = list(map(list, zip(*U)))
    a2 = F(0)
    eps2 = F(0)
    rows = []
    for i in a.remaining:
        c = [F(v) for v in a.c[i]]
        w = [F(v) for v in a.rad[i]]
        js = [j for j in range(p) if w[j]]
        ec = dot(c, qb) - F(a.y[i])
        eg = [w[j] * qb[j] for j in js]
        eu = abs(ec) + sum(map(abs, eg), F(0))
        pc = [dot(c, u) for u in ucols]
        pg = [[w[j] * u[j] for j in js] for u in ucols]
        vertex_values = [qform([pc[k] + sum((pg[k][l] * s for l, s in enumerate(signs)), F(0)) for k in range(r)], Hi) for signs in itertools.product((-1, 1), repeat=len(js))]
        ai = max(vertex_values)
        assert ai >= 0
        a2 += ai / n
        eps2 += eu ** 2 / n
        rows.append(dict(row=int(i), residual_abs_upper=str(eu), retained_squared_upper=str(ai), projection_vertex_values=[str(v) for v in vertex_values]))
    mu = F(a.mu)
    nu = F(a.nu)
    assert mu > 0 and nu >= mu
    c = (nu - mu) / 2
    invs = sqrt_upper(1 / mu)
    aa, eps, G, d = [sqrt_upper(v) for v in (a2, eps2, G2, d2)]
    R0 = (G + aa * eps + (c / mu + aa * invs / 2) * d + c * eps * invs / 2) / mu
    terms = dict(residual=eps, retained_coupling=aa * R0, observed_gradient_leakage=d * invs, metric_leakage=c * R0 * invs)
    eta = sum(terms.values(), F(0)) ** 2
    raw = sqrt_upper(R0 ** 2 / F(hmu))
    units = raw * 2 ** 20
    radius = F((units.numerator + units.denominator - 1) // units.denominator, 2 ** 20)
    fractions = dict(mu=mu, nu=nu, c=c, a2=a2, epsilon2=eps2, G2=G2, d2=d2, a=aa, epsilon=eps, G=G, d=d, inverse_sqrt_mu=invs, retained_radius=R0, coordinate_radius=radius, eta=eta, gram_minimum=F(hmu))
    return dict(status='PASS', verifier='observed_anchor_energy_exact_rational', actual_dyadic_transform=True, rank=r, observed_rows=int(sum(a.anchor_rows)), uncertain_rows=len(rows), fractions={k: str(v) for k, v in fractions.items()}, gram=[[str(v) for v in row] for row in H], h=[str(v) for v in h], rows=rows, components={k: float(v) for k, v in terms.items()}, tail=float_up(eta), tail_target_met=eta <= F(1, 400000), tail_target=2.5e-06)

class AnchoredEnergySearch(MomentSearch):

    def __init__(self, a, R, receipt, mode='energy'):
        with patch.object(moment_tail, 'basis_proposal', return_value=R):
            super().__init__(a, R.shape[1], kind='gradient')
        self.receipt = receipt
        self.energy_mode = mode
        self.original_moment_tail = float(self.qtail)
        self.original_moment_radius = self.radius
        er = float(F(receipt['fractions']['coordinate_radius']))
        em = F(receipt['fractions']['eta'])
        gn = F(self.tail_receipt['gradient_norm'])
        cn = F(self.tail_receipt['coupling_norm'])
        shared = (gn + cn * F(er)) ** 2 / F(a.mu)
        self.shared_radius_moment_tail = float_up(shared)
        if mode == 'energy':
            self.qtail = em
            self.radius = er
        elif mode == 'hybrid':
            self.qtail = min(em, shared)
            self.radius = er
        elif mode == 'moment_shared':
            self.qtail = shared
            self.radius = er
        elif mode != 'moment':
            raise ValueError(mode)
        self.initial_upper = float_up(F(a.gradient_upper) + self.qtail)

    def run(self, **kwargs):
        out = super().run(**kwargs)
        out.update(method='anchored_' + self.energy_mode, energy_receipt=self.receipt, original_moment_tail=self.original_moment_tail, original_moment_radius=self.original_moment_radius, shared_radius_moment_tail=self.shared_radius_moment_tail, retained_radius=self.radius)
        if 'certificate' in out:
            out['certificate'].update(method='anchored_' + self.energy_mode, energy_receipt=self.receipt, tail_error=str(self.qtail), coordinate_radius=self.radius, initial_upper=self.initial_upper)
        return out
