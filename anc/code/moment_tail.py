from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Original-metric retained search with a verified residual-moment tail.'
from observed_anchor import *

def norms(a, axis=1):
    values = I(abs_upper(a)).square().sum(axis=axis).hi
    return np.array([sqrt_up(v) for v in np.asarray(values).flat]).reshape(np.asarray(values).shape)

def basis_proposal(anchor, kind='gradient'):
    a = anchor
    if kind == 'midpoint':
        k = (a.center_metric.lo + a.center_metric.hi) / 2
    elif kind == 'gradient':
        uc = a.cr @ a.t
        res = a.cr @ a.beta - a.yr
        generators = []
        for i in range(a.m):
            for jj, s in zip(a.cells[i], a.width[i]):
                if s:
                    generators.append(s / a.n * (a.beta[jj] * uc[i] + res[i] * a.t[jj]))
            for j, sj in zip(a.cells[i], a.width[i]):
                for l, sl in zip(a.cells[i], a.width[i]):
                    if sj and sl:
                        generators.append(sj * sl * a.beta[l] / a.n * a.t[j])
        gg = np.array(generators)
        k = gg.T @ gg if len(gg) else np.eye(a.p)
    else:
        raise ValueError(kind)
    _, v = np.linalg.eigh(k)
    return np.round(v[:, ::-1] * 2 ** 24) / 2 ** 24

def projected_gradient(anchor, transform):
    a = anchor
    qb = list(map(Q, a.beta))
    qg0 = [qdot(row, qb) - v for row, v in zip(a.qG, a.qr)]
    g0 = I([float_down(v) for v in qg0], [float_up(v) for v in qg0])
    base = matvec(transpose(transform), g0)
    proj = mm(I(a.cr), transform)
    output = []
    for j in range(transform.lo.shape[1]):
        gens = I(a.width) * transform[a.cells, j]
        pr = proj[:, j]
        aa = (pr + a.a) / 2
        bb = (pr - a.a) / 2
        ga = (gens + a.ag) / 2
        gb = (gens - a.ag) / 2
        high, _ = row_max(aa, bb, ga, gb, 1.0)
        negative, _ = row_max(bb, aa, gb, ga, 1.0)
        output.append(I(-negative.hi, high.hi).sum() / a.n + base[j])
    return I([float(v.lo) for v in output], [float(v.hi) for v in output])

def tail_receipt(anchor, r):
    a = anchor
    rank = r.shape[1]
    qr = qarray(r)
    qh = [[qdot(c1, c2) for c2 in zip(*qr)] for c1 in zip(*qr)]
    h = interval_q(qh)
    hmu = eigen_gershgorin(h)
    gram_method = 'gershgorin'
    qhinv = None
    if hmu <= 0:
        # H=R^T R is exactly positive semidefinite. An exact inverse proves
        # full rank, and ||H^{-1}||_2 <= ||H^{-1}||_infinity gives a lower bound.
        qhinv = exact_inverse(qh)
        inverse_norm = max(sum(abs(v) for v in row) for row in qhinv)
        hmu = float_down(1 / inverse_norm)
        gram_method = 'exact_inverse_norm'
    if hmu <= 0:
        raise ValueError('Retained basis metric not verified positive')
    radius2 = Q(a.gradient_upper) / (Q(a.mu) * Q(hmu))
    radius = sqrt_up(radius2)
    if rank == a.p:
        return {'tail_bound': 0.0, 'gradient_norm': 0.0, 'coupling_norm': 0.0, 'coordinate_radius': radius, 'Hmu': hmu, 'gram_lower_method': gram_method, 'projector': I(np.zeros((a.p, a.p))), 'H': h, 'qH': qh, 'full_rank': True}
    invh = interval_q(qhinv if qhinv is not None else exact_inverse(qh))
    projector = I(np.eye(a.p)) - mm(mm(I(r), invh), I(r.T))
    tp = mm(I(a.t), projector)
    g = projected_gradient(a, tp)
    gnorm = sqrt_up(float(g.square().sum().hi))
    cross = mm(projector, mm(a.center_metric, I(r)))
    cross_norm = sqrt_up(float(cross.square().sum().hi))
    center_p = mm(a.ct, projector)
    center_r = mm(a.ct, I(r))
    tr = mm(I(a.t), I(r))
    generator_p = norms(tp)
    generator_r = norms(tr)
    delta_p = (I(a.width) * I(generator_p[a.cells])).sum(axis=1)
    delta_r = (I(a.width) * I(generator_r[a.cells])).sum(axis=1)
    nc_p = norms(center_p)
    nc_r = norms(center_r)
    variation = (delta_p * I(nc_r) + I(nc_p) * delta_r + delta_p * delta_r).sum() / a.n
    coupling = float((I(cross_norm) + variation).hi)
    tail = float(((I(gnorm) + I(coupling) * radius).square() / a.mu).hi)
    return {'tail_bound': tail, 'gradient_norm': gnorm, 'coupling_norm': coupling, 'coordinate_radius': radius, 'center_cross_norm': cross_norm, 'variation_coupling': float(variation.hi), 'Hmu': hmu, 'gram_lower_method': gram_method, 'projector': projector, 'H': h, 'qH': qh, 'gradient_lo': g.lo.tolist(), 'gradient_hi': g.hi.tolist(), 'generator_p_norms': generator_p.tolist(), 'generator_r_norms': generator_r.tolist(), 'center_p_norms': nc_p.tolist(), 'center_r_norms': nc_r.tolist(), 'full_rank': False}

class MomentSearch(AnchorSearch):

    def __init__(self, anchor, rank, kind='gradient', search_bound='adaptive'):
        self.a = anchor
        self.rank = rank
        self.mode = search_bound
        self.kind = kind
        self.r = np.eye(anchor.p) if rank == anchor.p else basis_proposal(anchor, kind)[:, :rank]
        self.tail_receipt = tail_receipt(anchor, self.r)
        receipt = self.tail_receipt
        self.H = receipt['H']
        self.qH = receipt['qH']
        self.hmu = receipt['Hmu']
        self.rho = 0.0
        self.qtail = Q(receipt['tail_bound'])
        self.anchor_error = Q(0)
        self.qgamma = Q(1)
        self.gamma = I(1.0)
        self.K = mm(I(self.r.T), mm(anchor.metric, I(self.r)))
        qb = list(map(Q, anchor.beta))
        g0 = [qdot(row, qb) - v for row, v in zip(anchor.qG, anchor.qr)]
        self.tr = mm(I(anchor.t), I(self.r))
        self.bproj = mm(anchor.ct, I(self.r))
        self.linear = matvec(transpose(self.tr), I([float_down(v) for v in g0], [float_up(v) for v in g0]))
        self.hdiag = []
        for j in range(rank):
            pr = (I(anchor.width) * I(abs_upper(self.tr[:, j])[anchor.cells])).sum(axis=1)
            bound = self.K[j, j] + (I(abs_upper(self.bproj[:, j])) + pr).square().sum() / anchor.n
            self.hdiag.append(float(bound.hi))
        self.hdiag = np.array(self.hdiag)
        self.radius = np.ceil(receipt['coordinate_radius'] * 2 ** 20) / 2 ** 20
        self.initial_upper = float_up(Q(anchor.gradient_upper) + self.qtail)
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
        b0 = a.a - matvec(self.bproj, zz)
        theta = I(a.beta) - matvec(self.tr, zz)
        bg = I(a.width) * theta[a.cells]
        rows, signs = row_max(a.a, b0, a.ag, bg, 1.0)
        anchor = 2 * dot(self.linear, zz) - dot(zz, matvec(self.K, zz))
        value = anchor + rows.sum() / a.n + self.qtail_as_interval()
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

    def qtail_as_interval(self):
        return I(float_down(self.qtail), float_up(self.qtail))

    def node_upper(self, lo, hi, cap):
        a = self.a
        c = (np.array(lo) + hi) / 2
        rad = (I(hi) - I(lo)) / 2
        point = self.evaluate(c)
        zz = I(c)
        theta = I(a.beta) - matvec(self.tr, zz)
        bg = I(a.width) * theta[a.cells]
        b0 = a.a - matvec(self.bproj, zz)
        br = I(abs_upper(bg)).sum(axis=1)
        b = I(down(b0.lo - br.hi), up(b0.hi + br.hi))
        pen = 2 * self.linear - 2 * matvec(self.K, zz)
        gradient = []
        for j in range(self.rank):
            pr = (I(a.width) * I(abs_upper(self.tr[:, j])[a.cells])).sum(axis=1)
            xx = I(down(self.bproj[:, j].lo - pr.hi), up(self.bproj[:, j].hi + pr.hi))
            gradient.append(pen[j] + 2 * (b * xx).sum() / a.n)
        g = I([float(v.lo) for v in gradient], [float(v.hi) for v in gradient])
        bound = min(cap, float((I(point['upper']) + (I(abs_upper(g)) * rad).sum()).hi))
        if self.rank <= 3 and self.mode != 'center':
            corners = max((self.evaluate(q)['upper'] for q in itertools.product(*zip(lo, hi))))
            bound = min(bound, float((I(corners) + (I(self.hdiag) * rad.square()).sum()).hi))
        return bound

    def run(self, **kwargs):
        result = super().run(**kwargs)
        result['method'] = 'residual_moment'
        result['basis_kind'] = self.kind
        result['tail_gradient_norm'] = self.tail_receipt['gradient_norm']
        result['tail_coupling_norm'] = self.tail_receipt['coupling_norm']
        if 'certificate' in result:
            result['certificate']['method'] = 'residual_moment'
            result['certificate']['original_gradient_upper'] = self.a.gradient_upper
            result['certificate']['tail_receipt'] = {k: v for k, v in self.tail_receipt.items() if k not in ('projector', 'H', 'qH')}
            result['certificate']['hdiag'] = self.hdiag.tolist()
            result['certificate']['tail_error'] = str(self.qtail)
        return result
