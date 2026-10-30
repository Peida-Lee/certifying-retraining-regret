from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Verified dense anchors and certificates for the original ridge regret.'
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_k] = '1'
from pathlib import Path
import sys, time, heapq, itertools
from fractions import Fraction as Q
import numpy as np
from interval_arithmetic import I, matvec, dot, quadratic_max_upper, up, down
from automatic_basis import mm, transpose, interval_q, exact_inverse, eigen_gershgorin
from rank_certificate import float_up, float_down, qdot

def abs_upper(a):
    return np.maximum(np.abs(a.lo), np.abs(a.hi))

def qarray(a):
    return [[Q(v) for v in row] for row in np.asarray(a)]

def sqrt_up(value):
    q = Q(value)
    v = float(np.sqrt(float(q)))
    while Q(v) ** 2 < q:
        v = float(np.nextafter(v, np.inf))
    return v

def exact_moments(x, y, den, lam):
    if not len(x):
        return ([[Q(lam) * int(j == k) for k in range(x.shape[1])] for j in range(x.shape[1])], [Q(0)] * x.shape[1])
    d = max((Q(v).denominator for v in x.flat))
    dy = max((Q(v).denominator for v in y))
    xx = np.array([[int(Q(v) * d) for v in row] for row in x], dtype=np.int64)
    yy = np.array([int(Q(v) * dy) for v in y], dtype=np.int64)
    mag = max((abs(int(v)) for v in xx.flat))
    ymag = max((abs(int(v)) for v in yy))
    assert len(x) * max(mag * mag, mag * ymag) < 2 ** 63
    gram = xx.T @ xx
    response = xx.T @ yy
    p = x.shape[1]
    return ([[Q(int(gram[j, k]), den * d * d) + Q(lam) * int(j == k) for k in range(p)] for j in range(p)], [Q(int(v), den * d * dy) for v in response])

def row_max(a0, b0, ag, bg, gamma, with_witness=False):
    """Original box-edge reference, vectorized; intended k<=4 in this pilot."""
    n = len(a0.lo)
    k = ag.lo.shape[1]
    if k == 0:
        v = a0.square() - gamma * b0.square()
        return (v, np.empty((n, 0)))
    upper = np.full(n, -np.inf)
    best = np.full(n, -np.inf)
    signs = np.zeros((n, k))
    amid = (ag.lo + ag.hi) / 2
    bmid = (bg.lo + bg.hi) / 2
    for free in range(k):
        other = [j for j in range(k) if j != free]
        for bits in itertools.product((-1.0, 1.0), repeat=len(other)):
            aa = a0
            bb = b0
            for j, sgn in zip(other, bits):
                aa = aa + ag[:, j] * sgn
                bb = bb + bg[:, j] * sgn
            qa = ag[:, free].square() - gamma * bg[:, free].square()
            qb = 2 * (aa * ag[:, free] - gamma * bb * bg[:, free])
            qc = aa.square() - gamma * bb.square()
            upper = np.maximum(upper, quadratic_max_upper(qa, qb, qc))
            av = (qa.lo + qa.hi) / 2
            bv = (qb.lo + qb.hi) / 2
            z = np.where(bv >= 0, 1.0, -1.0)
            concave = av < 0
            z[concave] = np.clip(-bv[concave] / (2 * av[concave]), -1, 1)
            candidate = np.zeros((n, k))
            candidate[:, free] = z
            for j, sgn in zip(other, bits):
                candidate[:, j] = sgn
            a_num = (a0.lo + a0.hi) / 2 + np.sum(amid * candidate, axis=1)
            b_num = (b0.lo + b0.hi) / 2 + np.sum(bmid * candidate, axis=1)
            gamma_numeric = float((gamma.lo + gamma.hi) / 2) if isinstance(gamma, I) else float(gamma)
            value = a_num * a_num - gamma_numeric * b_num * b_num
            improve = value > best
            best[improve] = value[improve]
            signs[improve] = candidate[improve]
    aw = a0 + (ag * I(signs)).sum(axis=1)
    bw = b0 + (bg * I(signs)).sum(axis=1)
    lower = (aw.square() - gamma * bw.square()).lo
    if np.any(lower > upper):
        raise ArithmeticError('Invalid row interval')
    return (I(lower, upper), signs)

class ObservedAnchor:

    def __init__(self, lo, hi, y, beta, lam, anchor='observed', preconditioner='anchor'):
        start = time.monotonic()
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.y = np.asarray(y, float)
        self.beta = np.asarray(beta, float)
        self.lam = float(lam)
        self.n, self.p = self.lo.shape
        self.c = (self.lo + self.hi) / 2
        self.rad = (self.hi - self.lo) / 2
        assert np.all(self.c - self.rad == self.lo) and np.all(self.c + self.rad == self.hi)
        self.fixed = np.all(self.rad == 0, axis=1)
        self.anchor_type = anchor
        self.anchor_rows = self.fixed if anchor == 'observed' else np.zeros(self.n, dtype=bool)
        self.remaining = np.flatnonzero(~self.anchor_rows)
        self.m = len(self.remaining)
        self.qG, self.qr = exact_moments(self.c[self.anchor_rows], self.y[self.anchor_rows], self.n, lam)
        self.G = np.array(self.qG, float)
        gI = interval_q(self.qG)
        proposal = self.G if preconditioner == 'anchor' else self.c.T @ self.c / self.n + self.lam * np.eye(self.p)
        self.preconditioner_type = preconditioner
        self.t = np.round(np.linalg.inv(np.linalg.cholesky(proposal).T) * 2 ** 20) / 2 ** 20
        self.theta_bar = np.round(np.linalg.solve(self.G, np.array(self.qr, float)) * 2 ** 24) / 2 ** 24
        self.metric = mm(I(self.t.T), mm(gI, I(self.t)))
        self.mu = eigen_gershgorin(self.metric)
        self.nu = eigen_gershgorin(self.metric, upper=True)
        if preconditioner == 'midpoint' or self.mu <= 0:
            _, v = np.linalg.eigh((self.metric.lo + self.metric.hi) / 2)
            test = np.round(v * 2 ** 24) / 2 ** 24
            gg = mm(I(test.T), I(test))
            kk = mm(I(test.T), mm(self.metric, I(test)))
            gm = eigen_gershgorin(gg)
            gx = eigen_gershgorin(gg, True)
            if gm <= 0:
                raise ValueError('Metric test basis not verified invertible')
            self.mu = max(self.mu, float((I(eigen_gershgorin(kk)) / gx).lo))
            self.nu = min(self.nu, float((I(eigen_gershgorin(kk, True)) / gm).hi))
        if self.mu <= 0:
            raise ValueError('Dense anchor metric not verified positive')
        qt = qarray(self.t)
        qbar = list(map(Q, self.theta_bar))
        qbeta = list(map(Q, self.beta))
        residual = [qdot(row, qbar) - ri for row, ri in zip(self.qG, self.qr)]
        self.qb = [qdot(col, residual) for col in zip(*qt)]
        b2 = qdot(self.qb, self.qb)
        self.qtheta = [a - qdot(row, self.qb) / Q(self.mu) for a, row in zip(qbar, qt)]

        def qanchor(b):
            return qdot(b, [qdot(row, b) for row in self.qG]) - 2 * qdot(self.qr, b)
        self.qC = qanchor(qbeta) - qanchor(qbar) + b2 / Q(self.mu)
        cr = self.c[self.remaining]
        rad = self.rad[self.remaining]
        yr = self.y[self.remaining]
        self.dq = [qdot(list(map(Q, row)), self.qtheta) - Q(yy) for row, yy in zip(cr, yr)]
        dqmax = [abs(v) + sum((Q(s) * abs(t) for s, t in zip(r, self.qtheta)), Q(0)) for v, r in zip(self.dq, rad)]
        self.qB = sum((v * v for v in dqmax), Q(0)) / self.n
        self.qanchor_error = (Q(self.nu) - Q(self.mu)) * (2 * b2 / Q(self.mu) ** 2 + self.qB / (2 * Q(self.mu)))
        self.d = interval_q([[v] for v in self.dq])[:, 0]
        self.a = matvec(I(cr), I(self.beta)) - I(yr)
        self.ct = mm(I(cr), I(self.t))
        self.radii = rad
        self.cr = cr
        self.yr = yr
        ks = np.sum(rad != 0, axis=1)
        self.k = int(max(ks, default=0))
        if self.k > 6:
            raise ValueError('Pilot row reference limited to six uncertain cells')
        self.cells = np.zeros((self.m, self.k), dtype=int)
        self.width = np.zeros((self.m, self.k))
        for i, row in enumerate(rad):
            jj = np.flatnonzero(row)
            self.cells[i, :len(jj)] = jj
            self.width[i, :len(jj)] = row[jj]
        self.ag = I(self.width) * I(self.beta[self.cells])
        kcenter = mm(transpose(self.ct), self.ct) / self.n
        self.center_metric = self.metric + kcenter
        ww = (I(ks[:, None]) * I(rad).square()).sum(axis=0) / self.n
        self.kcenter = kcenter / self.mu
        self.envelope = 2 * (kcenter + mm(I(self.t.T), I(ww.lo[:, None], ww.hi[:, None]) * I(self.t))) / self.mu
        self.avg = (kcenter + mm(I(self.t.T), (I(rad).square().sum(axis=0) / self.n)[:, None] * I(self.t))) / self.mu
        self.path, self.basis = self._rank_path(self.envelope)
        self.gradient_upper = self._gradient_upper()
        self.setup_seconds = time.monotonic() - start

    def _rank_path(self, k):
        midpoint = (k.lo + k.hi) / 2
        vals, v = np.linalg.eigh(midpoint)
        s = np.round(v[:, ::-1] * 2 ** 24) / 2 ** 24
        gram = mm(I(s.T), I(s))
        mu = eigen_gershgorin(gram)
        nu = eigen_gershgorin(gram, True)
        if mu <= 0:
            raise ValueError('Numerical eigenbasis not verified invertible')
        err = gram - I(np.eye(self.p))
        e = eigen_gershgorin(I(-abs_upper(err), abs_upper(err)), True)
        transformed = mm(I(s.T), mm(k, I(s)))
        all_nu = eigen_gershgorin(transformed, True)
        kmax = float((I(all_nu) / mu).hi)
        correction = float((I(sqrt_up(kmax)) * sqrt_up(nu) * e / mu).hi)
        avgmid = (self.avg.lo + self.avg.hi) / 2
        av = np.linalg.eigvalsh(avgmid)[::-1]
        rows = []
        for r in range(1, self.p + 1):
            if r == self.p:
                rho = 0.0
            else:
                tail = eigen_gershgorin(transformed[r:, r:], True)
                rho = float(((I(sqrt_up(max(0.0, tail))) + correction).square() / mu).hi)
            qb = Q(rho) / (1 + Q(rho)) * self.qB
            rows.append({'rank': r, 'rho_upper': rho, 'B0_squared': float_up(self.qB), 'tail_error': float_up(qb), 'anchor_error': float_up(self.qanchor_error), 'average_eigenvalue_diagnostic': max(0.0, float(av[r])) if r < self.p else 0.0})
        return (rows, s)

    def _gradient_upper(self):
        qbeta = list(map(Q, self.beta))
        g0 = [qdot(row, qbeta) - ri for row, ri in zip(self.qG, self.qr)]
        base = [qdot(col, g0) for col in zip(*qarray(self.t))]
        out = []
        for j in range(self.p):
            pr = self.ct[:, j]
            gens = I(self.width) * I(self.t[self.cells, j])
            resgens = self.ag
            aa = (pr + self.a) / 2
            bb = (pr - self.a) / 2
            ga = (gens + resgens) / 2
            gb = (gens - resgens) / 2
            high, _ = row_max(aa, bb, ga, gb, 1.0)
            neg, _ = row_max(bb, aa, gb, ga, 1.0)
            interval = I(-neg.hi, high.hi).sum() / self.n + I(float_down(base[j]), float_up(base[j]))
            out.append(interval)
        gradient = I([float(v.lo) for v in out], [float(v.hi) for v in out])
        return float((gradient.square().sum() / self.mu).hi)

    def summary(self):
        return {'anchor': self.anchor_type, 'preconditioner': self.preconditioner_type, 'n': self.n, 'features': self.p, 'anchor_rows': int(np.sum(self.anchor_rows)), 'remaining_rows': self.m, 'uncertain_cells': int(np.sum(self.rad != 0)), 'uncertain_columns': int(np.sum(np.any(self.rad != 0, axis=0))), 'mu': self.mu, 'nu': self.nu, 'B0_squared': float_up(self.qB), 'anchor_error': float_up(self.qanchor_error), 'gradient_upper': self.gradient_upper, 'rank_path': self.path, 'setup_seconds': self.setup_seconds}

class AnchorSearch:

    def __init__(self, anchor, rank, search_bound='adaptive'):
        self.a = anchor
        self.rank = rank
        self.mode = search_bound
        self.r = np.eye(anchor.p) if rank == anchor.p else anchor.basis[:, :rank]
        qr = qarray(self.r)
        self.qH = [[qdot(col1, col2) for col2 in zip(*qr)] for col1 in zip(*qr)]
        self.H = interval_q(self.qH)
        self.hmu = eigen_gershgorin(self.H)
        if self.hmu <= 0:
            raise ValueError('Retained Gram not verified positive')
        self.rho = anchor.path[rank - 1]['rho_upper']
        self.qgamma = 1 / (1 + Q(self.rho))
        self.gamma = I(float_down(self.qgamma), float_up(self.qgamma))
        self.qtail = Q(self.rho) / (1 + Q(self.rho)) * anchor.qB
        self.bproj = mm(anchor.ct, I(self.r))
        self.tr = mm(I(anchor.t), I(self.r))
        self.theta = I([float_down(v) for v in anchor.qtheta], [float_up(v) for v in anchor.qtheta])
        self.hdiag = []
        for j in range(rank):
            proj_radius = (I(anchor.width) * I(abs_upper(self.tr[:, j])[anchor.cells])).sum(axis=1)
            h = anchor.mu * self.H[j, j] + self.gamma * (I(abs_upper(self.bproj[:, j])) + proj_radius).square().sum() / anchor.n
            self.hdiag.append(float(h.hi))
        self.hdiag = np.array(self.hdiag)
        radius2 = self.qgamma * anchor.qB / (4 * Q(anchor.mu) * Q(self.hmu))
        self.radius = np.ceil(np.sqrt(float(radius2)) * 2 ** 20) / 2 ** 20
        while Q(float(self.radius)) ** 2 < radius2:
            self.radius += 2 ** (-20)
        self.constant = I(float_down(anchor.qC), float_up(anchor.qC))
        self.anchor_error = anchor.qanchor_error
        self.initial_upper = float_up(Q(anchor.gradient_upper) + self.qtail + anchor.qanchor_error)
        self.best_search = -np.inf
        self.best_original = 0.0
        self.best_witness = None
        self.cache = {}
        self.points = []
        self.tree = []

    def evaluate(self, z):
        key = tuple(z)
        if key in self.cache:
            return self.cache[key]
        a = self.a
        zz = I(z)
        b0 = a.d + matvec(self.bproj, zz)
        theta = self.theta + matvec(self.tr, zz)
        bg = I(a.width) * theta[a.cells]
        rows, signs = row_max(a.a, b0, a.ag, bg, self.gamma)
        penalty = a.mu * dot(zz, matvec(self.H, zz))
        value = self.constant - penalty + rows.sum() / a.n
        point = {'z': list(z), 'upper': float(value.hi), 'lower': float(value.lo)}
        self.points.append(point)
        if float(value.lo) > self.best_search:
            self.best_search = float(value.lo)
            x = a.c.copy()
            for i, row in enumerate(a.remaining):
                for j, s, sgn in zip(a.cells[i], a.width[i], signs[i]):
                    if s:
                        x[row, j] = np.clip(x[row, j] + s * sgn, a.lo[row, j], a.hi[row, j])
            fit = np.linalg.solve(x.T @ x / a.n + a.lam * np.eye(a.p), x.T @ a.y / a.n)

            def loss(b):
                return (matvec(I(x), I(b)) - I(a.y)).square().sum() / a.n + a.lam * dot(I(b), I(b))
            lower = float((loss(a.beta) - loss(fit)).lo)
            if lower > self.best_original:
                self.best_original = lower
                self.best_witness = {'x': x.tolist(), 'theta': fit.tolist(), 'regret_lower': lower, 'z': list(z)}
        self.cache[key] = point
        return point

    def node_upper(self, lo, hi, cap):
        a = self.a
        c = (np.array(lo) + hi) / 2
        radius = (I(hi) - I(lo)) / 2
        center = self.evaluate(c)
        zz = I(c)
        theta = self.theta + matvec(self.tr, zz)
        bg = I(a.width) * theta[a.cells]
        bcenter = a.d + matvec(self.bproj, zz)
        bradius = I(abs_upper(bg)).sum(axis=1)
        b = I(down(bcenter.lo - bradius.hi), up(bcenter.hi + bradius.hi))
        gradient = []
        pen = -2 * a.mu * matvec(self.H, zz)
        for j in range(self.rank):
            pradius = (I(a.width) * I(abs_upper(self.tr[:, j])[a.cells])).sum(axis=1)
            xx = I(down(self.bproj[:, j].lo - pradius.hi), up(self.bproj[:, j].hi + pradius.hi))
            gradient.append(pen[j] - 2 * self.gamma * (b * xx).sum() / a.n)
        grad = I([float(v.lo) for v in gradient], [float(v.hi) for v in gradient])
        lipschitz = float((I(center['upper']) + (I(abs_upper(grad)) * radius).sum()).hi)
        bound = min(cap, lipschitz)
        if self.rank <= 3 and self.mode != 'center':
            corners = max((self.evaluate(q)['upper'] for q in itertools.product(*zip(lo, hi))))
            bound = min(bound, float((I(corners) + (I(self.hdiag) * radius.square()).sum()).hi))
        return bound

    def run(self, time_limit=10, absolute_tolerance=0.0001, relative_tolerance=0.0, record=False):
        start = time.monotonic()
        lo = tuple([-self.radius] * self.rank)
        hi = tuple([self.radius] * self.rank)
        upper = self.node_upper(lo, hi, self.initial_upper)
        heap = [(-upper, 0, lo, hi)]
        counter = expanded = 0
        reason = 'tolerance'
        trace = []
        self.tree = [{'id': 0, 'parent': None, 'lo': lo, 'hi': hi, 'upper': upper, 'children': []}]
        while heap:
            us = max(self.best_search, -heap[0][0])
            u = min(us, self.a.gradient_upper)
            l = max(0.0, self.best_original, float_down(Q(self.best_search) - self.qtail - self.anchor_error))
            trace.append({'seconds': time.monotonic() - start, 'lower': l, 'upper': u, 'search_lower': self.best_search, 'search_upper': us, 'nodes': expanded})
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
                self.tree.append({'id': counter, 'parent': idx, 'lo': cl, 'hi': ch, 'upper': bound, 'children': []})
                if bound > self.best_search:
                    heapq.heappush(heap, (-bound, counter, cl, ch))
            expanded += 1
        us = max(self.best_search, -heap[0][0] if heap else self.best_search)
        lower = max(0.0, self.best_original, float_down(Q(self.best_search) - self.qtail - self.anchor_error))
        upper = min(us, self.a.gradient_upper)
        if lower > upper:
            raise ArithmeticError('Original feasible lower exceeds certified upper')
        out = {'status': 'PASS', 'anchor': self.a.anchor_type, 'rank': self.rank, 'features': self.a.p, 'n': self.a.n, 'lower': lower, 'upper': upper, 'absolute_gap': upper - lower, 'relative_gap': (upper - lower) / max(lower, 1e-300), 'search_lower': self.best_search, 'search_upper': us, 'search_error': us - self.best_search, 'tail_error': float_up(self.qtail), 'anchor_error': float_up(self.anchor_error), 'B0_squared': float_up(self.a.qB), 'rho': self.rho, 'seconds': time.monotonic() - start, 'nodes': expanded, 'points': len(self.cache), 'stop_reason': reason, 'trace': trace, 'budget_seconds': time_limit, 'absolute_tolerance': absolute_tolerance}
        if record:
            out['certificate'] = {'lo': self.a.lo.tolist(), 'hi': self.a.hi.tolist(), 'y': self.a.y.tolist(), 'beta': self.a.beta.tolist(), 'lambda': self.a.lam, 'T': self.a.t.tolist(), 'theta_bar': self.a.theta_bar.tolist(), 'theta_c': [str(v) for v in self.a.qtheta], 'C_safe': str(self.a.qC), 'mu': self.a.mu, 'nu': self.a.nu, 'B0_squared': str(self.a.qB), 'anchor_error': str(self.a.qanchor_error), 'R': self.r.tolist(), 'rho': self.rho, 'gamma': str(self.qgamma), 'nodes': self.tree, 'points': self.points, 'witness': self.best_witness}
        return out
