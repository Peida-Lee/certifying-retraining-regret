from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Certified observed-anchor energy with signal support and retained shift.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import sys, itertools, time, heapq
from fractions import Fraction as F
from unittest.mock import patch
import numpy as np
from math import comb
B = Path(__file__).resolve().parent
ROOT = B.parent
sys.dont_write_bytecode = True
from anchor_energy import dot, qform, sqrt_upper, ObservedAnchor, qarray, exact_inverse, interval_q, float_up
from observed_anchor import I, mm, transpose, matvec, row_max, abs_upper, float_down, qdot, dot as interval_dot, down, up
import moment_tail
from moment_tail import MomentSearch

def proposal(a, meta):
    """Preserve the two signal generators exactly, then add anchor correction."""
    si = meta['signal_row']
    sj = meta['signal_column']
    qt = qarray(a.t)
    qs = [F(v) for v in a.c[si]]
    first = [dot(qs, col) for col in zip(*qt)]
    second = qt[sj]

    def scale(v):
        norm = np.linalg.norm(np.array(v, float))
        assert norm > 0
        factor = 2.0 ** (-int(np.floor(np.log2(norm))))
        ans = np.array([float(q * F(factor)) for q in v])
        assert all((F(x) == q * F(factor) for x, q in zip(ans, v))), 'Signal generator must remain exact'
        return ans
    initial = np.column_stack([scale(first), scale(second)])
    QB = np.linalg.qr(initial)[0]
    g0 = [dot(row, [F(v) for v in a.beta]) - v for row, v in zip(a.qG, a.qr)]
    h = np.array([dot(col, g0) for col in zip(*qt)], float)
    metric = (a.metric.lo + a.metric.hi) / 2
    correction = np.linalg.solve(metric, h)
    v = correction - QB @ (QB.T @ correction)
    if np.linalg.norm(v) < 1e-16:
        options = moment_tail.basis_proposal(a, 'gradient')
        opts = options - QB @ (QB.T @ options)
        v = opts[:, np.argmax(np.sum(opts * opts, axis=0))]
    v = np.round(v / np.linalg.norm(v) * 2 ** 24) / 2 ** 24
    return np.column_stack([initial, v])

def univariate_product(pc, pg, ec, eg):
    aa = pg * eg
    bb = pc * eg + pg * ec
    cc = pc * ec
    vv = [aa - bb + cc, aa + bb + cc]
    if aa and -1 < -bb / (2 * aa) < 1:
        vv.append(cc - bb * bb / (4 * aa))
    return (min(vv), max(vv))

def product_box(pc, pg, qc, qg):
    k = len(pg)
    if k == 0:
        return (pc * qc, pc * qc)
    values = []
    for free in range(k):
        others = [j for j in range(k) if j != free]
        for bits in itertools.product((-1, 1), repeat=k - 1):
            a = pc + sum((pg[j] * s for j, s in zip(others, bits)), F(0))
            b = qc + sum((qg[j] * s for j, s in zip(others, bits)), F(0))
            values.extend(univariate_product(a, pg[free], b, qg[free]))
    return (min(values), max(values))

def quartic_upper(power, depth=5):
    cp = [sum((power[k] * comb(k, j) * (-1) ** (k - j) * 2 ** j for k in range(j, 5)), F(0)) for j in range(5)]
    bern = [sum((cp[k] * F(comb(j, k), comb(4, k)) for k in range(j + 1)), F(0)) for j in range(5)]
    pieces = [bern]
    for _ in range(depth):
        children = []
        for coeff in pieces:
            triangle = [coeff]
            for k in range(4):
                triangle.append([(x + y) / 2 for x, y in zip(triangle[-1], triangle[-1][1:])])
            children.append([triangle[k][0] for k in range(5)])
            children.append([triangle[4 - k][k] for k in range(5)])
        pieces = children
    return max((max(piece) for piece in pieces))

def exact_geometry(a, R):
    qr = qarray(R)
    cols = list(map(list, zip(*qr)))
    qt = qarray(a.t)
    H = [[dot(x, y) for y in cols] for x in cols]
    Hi = exact_inverse(H)
    diag_min = min((H[i][i] - sum((abs(H[i][j]) for j in range(len(H)) if i != j)) for i in range(len(H))))
    hmu = max(diag_min, 1 / sum((Hi[i][i] for i in range(len(H)))))
    assert hmu > 0 and all((sum((H[i][k] * Hi[k][j] for k in range(len(H)))) == int(i == j) for i in range(len(H)) for j in range(len(H))))
    U = [[dot(row, col) for col in cols] for row in qt]
    ucols = list(map(list, zip(*U)))
    return (qr, cols, H, Hi, hmu, U, ucols, qt)

def constants(a, R, meta, reference_only=False):
    n, p = (a.n, a.p)
    r = R.shape[1]
    qr, cols, H, Hi, hmu, U, ucols, qt = exact_geometry(a, R)
    qb = [F(v) for v in a.beta]
    g0 = [dot(row, qb) - v for row, v in zip(a.qG, a.qr)]
    h = [dot(col, g0) for col in zip(*qt)]
    kr = [[dot(u, [dot(row, v) for row in a.qG]) for v in ucols] for u in ucols]
    kri = exact_inverse(kr)
    rh = [dot(col, h) for col in cols]
    shift = [dot(row, rh) for row in kri]
    reference = [b - dot(row, shift) for b, row in zip(qb, U)]
    gref = [dot(row, reference) - v for row, v in zip(a.qG, a.qr)]
    href = [dot(col, gref) for col in zip(*qt)]
    hq = [dot(col, href) for col in cols]
    assert all((v == 0 for v in hq))
    if reference_only:
        return dict(status='PASS', verifier='exact_retained_observed_anchor_reference', fractions=dict(gram_minimum=str(hmu), coordinate_radius='0', eta='0'), shift=[str(v) for v in shift], reference_beta=[str(v) for v in reference])
    d2 = dot(href, href)
    assert d2 >= 0
    si = meta['signal_row']
    sj = meta['signal_column']
    assert sum(a.rad[si] != 0) == 1 and a.rad[si, sj] > 0
    signal_c = [F(v) for v in a.c[si]]
    sigw = F(a.rad[si, sj])
    signal_vectors = [[dot(signal_c, col) for col in zip(*qt)], [sigw * v for v in qt[sj]]]
    support = []
    for v in signal_vectors:
        z = [dot(v, col) for col in cols]
        discarded = dot(v, v) - qform(z, Hi)
        assert discarded == 0
        support.append(str(discarded))
    pc = [dot(signal_c, col) for col in ucols]
    pg = [sigw * col[sj] for col in ucols]
    ec = dot(signal_c, reference) - F(a.y[si])
    eg = sigw * reference[sj]
    gi = [tuple((v / n + hq[j] for v in univariate_product(pc[j], pg[j], ec, eg))) for j in range(r)]
    Gbox = max((qform(list(v), Hi) for v in itertools.product(*gi)))
    poly = [[hq[j] + pc[j] * ec / n for j in range(r)], [(pc[j] * eg + pg[j] * ec) / n for j in range(r)], [pg[j] * eg / n for j in range(r)]]
    power = [sum((dot(poly[i], [dot(row, poly[k - i]) for row in Hi]) for i in range(3) if 0 <= k - i < 3), F(0)) for k in range(5)]
    Gquartic = quartic_upper(power)
    G2 = min(Gbox, Gquartic)
    a2 = F(0)
    eps2 = F(0)
    rows = []
    L = np.round(np.linalg.inv(np.linalg.cholesky(np.array(H, float)).T) * 2 ** 20) / 2 ** 20
    qL = qarray(L)
    lcols = list(map(list, zip(*qL)))
    LH = [[dot(col, [dot(row, other) for row in H]) for other in lcols] for col in lcols]
    lm = min((LH[j][j] - sum((abs(LH[j][k]) for k in range(r) if k != j)) for j in range(r)))
    assert lm > 0
    op_low = [[F(0) for _ in range(r)] for _ in range(r)]
    op_high = [[F(0) for _ in range(r)] for _ in range(r)]
    nuisance = [int(i) for i in a.remaining if i != si]
    for i in nuisance:
        c = [F(v) for v in a.c[i]]
        w = [F(v) for v in a.rad[i]]
        js = [j for j in range(p) if w[j]]
        ec0 = dot(c, reference) - F(a.y[i])
        eu = abs(ec0) + sum((abs(w[j] * reference[j]) for j in js), F(0))
        pc0 = [dot(c, col) for col in ucols]
        pg0 = [[w[j] * col[j] for j in js] for col in ucols]
        vv = [qform([pc0[k] + sum((pg0[k][l] * s for l, s in enumerate(signs)), F(0)) for k in range(r)], Hi) for signs in itertools.product((-1, 1), repeat=len(js))]
        ai = max(vv)
        assert ai >= 0
        a2 += ai / n
        eps2 += eu ** 2 / n
        pl = [dot(pc0, col) for col in lcols]
        pgl = [[dot([pg0[j][t] for j in range(r)], col) for t in range(len(js))] for col in lcols]
        for j in range(r):
            for k in range(r):
                l, u = product_box(pl[j], pgl[j], pl[k], pgl[k])
                op_low[j][k] += l / n
                op_high[j][k] += u / n
        rows.append(dict(row=i, residual_abs_upper=str(eu), retained_squared_upper=str(ai), projection_vertex_values=[str(v) for v in vv]))
    trace_a2 = a2
    op_a2 = max((op_high[j][j] + sum((max(abs(op_low[j][k]), abs(op_high[j][k])) for k in range(r) if k != j)) for j in range(r))) / lm
    a2 = min(trace_a2, op_a2)
    mu = F(a.mu)
    nu = F(a.nu)
    c = (nu - mu) / 2
    assert mu > 0 and c >= 0
    aa, eps, G, d = [sqrt_upper(v) for v in (a2, eps2, G2, d2)]
    invs = sqrt_upper(1 / mu)
    radius = (G + aa * eps + (c / mu + aa * invs / 2) * d + c * eps * invs / 2) / mu
    terms = dict(nuisance_residual=eps, retained_coupling=aa * radius, anchor_gradient_leakage=d * invs, metric_leakage=c * radius * invs)
    eta = sum(terms.values(), F(0)) ** 2
    coord = sqrt_upper(radius ** 2 / hmu)
    units = coord * 2 ** 24
    coord = F((units.numerator + units.denominator - 1) // units.denominator, 2 ** 24)
    ff = dict(mu=mu, nu=nu, c=c, a2=a2, trace_a2=trace_a2, operator_a2=op_a2, epsilon2=eps2, G2=G2, Gbox2=Gbox, Gquartic2=Gquartic, d2=d2, a=aa, epsilon=eps, G=G, d=d, inverse_sqrt_mu=invs, retained_radius=radius, coordinate_radius=coord, eta=eta, gram_minimum=hmu)
    return dict(status='PASS', verifier='centered_signal_observed_anchor_energy_exact_rational', rank=r, signal_rows=[si], nuisance_rows=nuisance, signal_discarded_norms_squared=support, observed_rows=int(sum(a.anchor_rows)), fractions={k: str(v) for k, v in ff.items()}, gram=[[str(v) for v in row] for row in H], shift=[str(v) for v in shift], reference_beta=[str(v) for v in reference], reference_anchor_gradient=[str(v) for v in href], signal_gradient_intervals=[[str(v) for v in g] for g in gi], signal_gradient_squared_power=[str(v) for v in power], bernstein_depth=5, operator_basis=L.tolist(), operator_gram_minimum=str(lm), operator_moment_lower=[[str(v) for v in row] for row in op_low], operator_moment_upper=[[str(v) for v in row] for row in op_high], rows=rows, components={k: float(v) for k, v in terms.items()}, tail=float_up(eta), tail_target_met=eta <= F(1, 400000), tail_target=2.5e-06)

def projected_reference_gradient(a, transform, reference):
    qb = reference
    g0 = [dot(row, qb) - v for row, v in zip(a.qG, a.qr)]
    g0I = I([float_down(v) for v in g0], [float_up(v) for v in g0])
    base = matvec(transpose(transform), g0I)
    proj = mm(I(a.cr), transform)
    ec = [dot([F(v) for v in row], qb) - F(y) for row, y in zip(a.cr, a.yr)]
    ecI = I([float_down(v) for v in ec], [float_up(v) for v in ec])
    betaI = I([float_down(v) for v in qb], [float_up(v) for v in qb])
    ag = I(a.width) * betaI[a.cells]
    output = []
    for j in range(transform.lo.shape[1]):
        gens = I(a.width) * transform[a.cells, j]
        pr = proj[:, j]
        aa = (pr + ecI) / 2
        bb = (pr - ecI) / 2
        ga = (gens + ag) / 2
        gb = (gens - ag) / 2
        high, _ = row_max(aa, bb, ga, gb, 1.0)
        negative, _ = row_max(bb, aa, gb, ga, 1.0)
        output.append(I(-negative.hi, high.hi).sum() / a.n + base[j])
    return I([float(v.lo) for v in output], [float(v.hi) for v in output])

class CenteredSignalSearch(MomentSearch):

    def __init__(self, a, R, receipt, mode='energy'):
        with patch.object(moment_tail, 'basis_proposal', return_value=R):
            super().__init__(a, R.shape[1], kind='gradient')
        self.energy_receipt = receipt
        self.proof_mode = mode
        self.original_moment_tail = float(self.qtail)
        self.original_moment_radius = self.radius
        self.center = list(map(F, receipt['shift']))
        self.energy_coordinate_radius = F(receipt['fractions']['coordinate_radius'])
        energy = F(receipt['fractions']['eta'])
        reference = list(map(F, receipt['reference_beta']))
        mu = F(a.mu)
        hmu = F(receipt['fractions']['gram_minimum'])
        gp = projected_reference_gradient(a, mm(I(a.t), self.tail_receipt['projector']), reference)
        gn = sqrt_upper(F(float(gp.square().sum().hi)))
        allg = projected_reference_gradient(a, I(a.t), reference)
        fullg2 = F(float(allg.square().sum().hi))
        mr = sqrt_upper(fullg2 / (mu * mu * hmu))
        units = mr * 2 ** 24
        mr = F((units.numerator + units.denominator - 1) // units.denominator, 2 ** 24)
        coupling = F(self.tail_receipt['coupling_norm'])
        centered_moment = (gn + coupling * mr) ** 2 / mu
        shared_moment = (gn + coupling * self.energy_coordinate_radius) ** 2 / mu
        self.centered_moment_tail = float_up(centered_moment)
        self.centered_moment_radius = float_up(mr)
        self.shared_moment_tail = float_up(shared_moment)
        self.centered_moment_receipt = dict(gradient_norm=str(gn), full_gradient_squared=str(fullg2), coupling=str(coupling), coordinate_radius=str(mr), tail=str(centered_moment), shared_radius_tail=str(shared_moment), gradient_lo=gp.lo.tolist(), gradient_hi=gp.hi.tolist())
        if mode == 'energy':
            self.qtail = energy
            self.qradius = self.energy_coordinate_radius
        elif mode == 'hybrid':
            self.qtail = min(energy, shared_moment)
            self.qradius = self.energy_coordinate_radius
        elif mode == 'moment_centered':
            self.qtail = centered_moment
            self.qradius = mr
        elif mode == 'moment_shared':
            self.qtail = shared_moment
            self.qradius = self.energy_coordinate_radius
        else:
            raise ValueError(mode)
        self.radius = float_up(self.qradius)
        self.initial_upper = float_up(F(a.gradient_upper) + self.qtail)

    def node_upper(self, lo, hi, cap):
        a = self.a
        c = (np.array(lo) + hi) / 2
        rad = I(np.maximum(abs_upper(I(c) - I(lo)), abs_upper(I(hi) - I(c))))
        point = self.evaluate(c)
        zz = I(c)
        theta = I(a.beta) - matvec(self.tr, zz)
        bg = I(a.width) * theta[a.cells]
        b0 = a.a - matvec(self.bproj, zz)
        br = I(abs_upper(bg)).sum(axis=1)
        bb = I(down(b0.lo - br.hi), up(b0.hi + br.hi))
        pen = 2 * self.linear - 2 * matvec(self.K, zz)
        gradient = []
        for j in range(self.rank):
            pr = (I(a.width) * I(abs_upper(self.tr[:, j])[a.cells])).sum(axis=1)
            xx = I(down(self.bproj[:, j].lo - pr.hi), up(self.bproj[:, j].hi + pr.hi))
            gradient.append(pen[j] + 2 * (bb * xx).sum() / a.n)
        gg = I([float(v.lo) for v in gradient], [float(v.hi) for v in gradient])
        bound = min(cap, float((I(point['upper']) + (I(abs_upper(gg)) * rad).sum()).hi))
        if self.rank <= 3 and self.mode != 'center':
            corners = max((self.evaluate(q)['upper'] for q in itertools.product(*zip(lo, hi))))
            bound = min(bound, float((I(corners) + (I(self.hdiag) * rad.square()).sum()).hi))
        return bound

    def run(self, time_limit=10, absolute_tolerance=1e-05, relative_tolerance=0.0, record=False):
        start = time.monotonic()
        lo = tuple((float_down(v - self.qradius) for v in self.center))
        hi = tuple((float_up(v + self.qradius) for v in self.center))
        root_lo, root_hi = (lo, hi)
        upper = self.node_upper(lo, hi, self.initial_upper)
        heap = [(-upper, 0, lo, hi)]
        counter = expanded = 0
        reason = 'tolerance'
        trace = []
        self.tree = [{'id': 0, 'parent': None, 'lo': lo, 'hi': hi, 'upper': upper, 'children': []}]
        while heap:
            us = max(self.best_search, -heap[0][0])
            u = min(us, self.a.gradient_upper)
            l = max(0.0, self.best_original, float_down(F(self.best_search) - self.qtail))
            trace.append(dict(seconds=time.monotonic() - start, lower=l, upper=u, search_lower=self.best_search, search_upper=us, nodes=expanded))
            if u - l <= max(absolute_tolerance, relative_tolerance * l):
                break
            if time.monotonic() - start >= time_limit:
                reason = 'time_limit'
                break
            if us - self.best_search < 1e-13 and u - l > absolute_tolerance:
                reason = 'tail_or_arithmetic_floor'
                break
            negative, idx, lo, hi = heapq.heappop(heap)
            if -negative <= self.best_search:
                continue
            widths = np.array(hi) - lo
            j = int(np.argmax(self.hdiag * widths ** 2))
            mid = (lo[j] + hi[j]) / 2
            for side in (0, 1):
                cl = list(lo)
                ch = list(hi)
                if side == 0:
                    ch[j] = mid
                else:
                    cl[j] = mid
                cl, ch = (tuple(cl), tuple(ch))
                bound = self.node_upper(cl, ch, -negative)
                counter += 1
                self.tree[idx]['children'].append(counter)
                self.tree.append(dict(id=counter, parent=idx, lo=cl, hi=ch, upper=bound, children=[]))
                if bound > self.best_search:
                    heapq.heappush(heap, (-bound, counter, cl, ch))
            expanded += 1
        us = max(self.best_search, -heap[0][0] if heap else self.best_search)
        lower = max(0.0, self.best_original, float_down(F(self.best_search) - self.qtail))
        upper = min(us, self.a.gradient_upper)
        if lower > upper:
            raise ArithmeticError('Original feasible lower exceeds certified upper')
        out = dict(status='PASS', method='centered_signal_' + self.proof_mode, anchor=self.a.anchor_type, rank=self.rank, features=self.a.p, n=self.a.n, lower=lower, upper=upper, absolute_gap=upper - lower, relative_gap=(upper - lower) / max(lower, 1e-300), search_lower=self.best_search, search_upper=us, search_error=us - self.best_search, tail_error=float_up(self.qtail), anchor_error=0.0, seconds=time.monotonic() - start, nodes=expanded, points=len(self.cache), stop_reason=reason, trace=trace, budget_seconds=time_limit, absolute_tolerance=absolute_tolerance, original_moment_tail=self.original_moment_tail, centered_moment_tail=self.centered_moment_tail, shared_radius_moment_tail=self.shared_moment_tail, original_moment_radius=self.original_moment_radius, centered_moment_radius=self.centered_moment_radius, energy_radius=float_up(self.energy_coordinate_radius))
        if record:
            out['certificate'] = dict(lo=self.a.lo.tolist(), hi=self.a.hi.tolist(), y=self.a.y.tolist(), beta=self.a.beta.tolist(), lambda_value=self.a.lam, T=self.a.t.tolist(), mu=self.a.mu, nu=self.a.nu, R=self.r.tolist(), nodes=self.tree, points=self.points, witness=self.best_witness, method=out['method'], hdiag=self.hdiag.tolist(), original_gradient_upper=self.a.gradient_upper, tail_error=str(self.qtail), anchor_error='0', initial_upper=self.initial_upper, root_lo=root_lo, root_hi=root_hi, center=[str(v) for v in self.center], coordinate_radius=str(self.qradius), energy_receipt=self.energy_receipt, centered_moment_receipt=self.centered_moment_receipt)
        return out
