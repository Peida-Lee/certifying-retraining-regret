from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Full-box regret by a row oracle and model-space curvature interpolation.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from fractions import Fraction as Q
from itertools import product
import heapq, time
import numpy as np
from interval_arithmetic import I, up, down, matvec, dot, quadratic_max_upper, quadratic_range

def reserve_cores():
    import psutil
    proc = psutil.Process()
    choices = [v for v in proc.cpu_affinity() if v >= 4]
    if choices:
        proc.cpu_affinity([choices[0]])

class FullBoxOracle:

    def __init__(self, lower, upper, response, beta, lam):
        self.lo = np.asarray(lower, float)
        self.hi = np.asarray(upper, float)
        self.y = np.asarray(response, float)
        self.beta = np.asarray(beta, float)
        self.lam = float(lam)
        self.n, self.p = self.lo.shape
        if np.any(self.lo > self.hi) or np.any(np.sum(self.lo != self.hi, axis=1) > 1):
            raise ValueError('At most one interval cell per row is required')
        if self.lam <= 0:
            raise ValueError('Reference implementation requires positive ridge')
        self.c = (self.lo + self.hi) / 2
        self.s = np.max((self.hi - self.lo) / 2, axis=1)
        if np.any(self.c - self.s[:, None] * (self.lo != self.hi) != self.lo):
            raise ValueError('Dyadic exact midpoint representation required')
        self.j = np.argmax(self.lo != self.hi, axis=1)
        self.fixed = self.s == 0
        fixed_x = self.c[self.fixed]
        self.mf = fixed_x.T @ fixed_x / self.n + self.lam * np.eye(self.p)
        chol = np.linalg.cholesky(self.mf)
        self.t = np.round(np.linalg.inv(chol.T) * 2 ** 16) / 2 ** 16
        self.r0 = matvec(I(self.c), I(self.beta)) - I(self.y)
        self.ct = I(np.column_stack([matvec(I(self.c), I(self.t[:, j])).lo for j in range(self.p)]), np.column_stack([matvec(I(self.c), I(self.t[:, j])).hi for j in range(self.p)]))
        mf = []
        for j in range(self.p):
            row = []
            for k in range(self.p):
                row.append((self.ct[self.fixed, j] * self.ct[self.fixed, k]).sum() / self.n + self.lam * dot(I(self.t[:, j]), I(self.t[:, k])))
            mf.append(row)
        diagonal = []
        for j in range(self.p):
            off = sum((max(abs(float(mf[j][k].lo)), abs(float(mf[j][k].hi))) for k in range(self.p) if k != j))
            off = float(up(off * (1 + self.p * np.finfo(float).eps)))
            diagonal.append(float(down(mf[j][j].lo - off)))
        self.mu = min(diagonal)
        if self.mu <= 0:
            raise ValueError('Preconditioned fixed-row curvature could not be certified')
        self.hdiag = []
        for j in range(self.p):
            radius = I(self.s) * I(np.abs(self.t[self.j, j]))
            abscenter = np.maximum(np.abs(self.ct[:, j].lo), np.abs(self.ct[:, j].hi))
            bound = (I(abscenter) + radius).square().sum() / self.n + self.lam * dot(I(self.t[:, j]), I(self.t[:, j]))
            self.hdiag.append(float(bound.hi))
        self.hdiag = np.array(self.hdiag)
        self.gradient_box = self._gradient_box()
        self.gradient_upper = float((self.gradient_box.square().sum() / self.mu).hi)
        self.direct_upper = self._direct_upper()
        self.initial_upper = min(self.gradient_upper, self.direct_upper)
        radius = np.sqrt(self.initial_upper / self.mu)
        self.radius = np.ceil(radius * 2 ** 16) / 2 ** 16
        while Q(float(self.radius)) ** 2 * Q(self.mu) < Q(self.initial_upper):
            self.radius += 2 ** (-16)
        self.best_lower = 0.0
        self.best_witness = None
        self.calls = 0
        self.cache = {}

    def _gradient_box(self):
        out = []
        s = I(self.s)
        bj = I(self.beta[self.j])
        for j in range(self.p):
            tj = I(self.t[self.j, j])
            center = self.ct[:, j]
            a = s.square() * tj * bj
            b = s * (tj * self.r0 + center * bj)
            c = center * self.r0
            out.append(quadratic_range(a, b, c).sum() / self.n + self.lam * dot(I(self.t[:, j]), I(self.beta)))
        return I([float(x.lo) for x in out], [float(x.hi) for x in out])

    def loss(self, x, theta, weights_denominator=None):
        den = self.n if weights_denominator is None else weights_denominator
        residual = matvec(I(x), I(theta)) - I(self.y if len(x) == self.n else self.y[self.fixed])
        return residual.square().sum() / den + self.lam * dot(I(theta), I(theta))

    def _direct_upper(self):
        candidate = quadratic_range(I(self.s).square() * I(self.beta[self.j]).square(), 2 * I(self.s) * I(self.beta[self.j]) * self.r0, self.r0.square())
        maximum = I(candidate.hi).sum() / self.n + self.lam * dot(I(self.beta), I(self.beta))
        xf = self.c[self.fixed]
        yf = self.y[self.fixed]
        theta = np.linalg.solve(self.mf, xf.T @ yf / self.n)
        loss = (matvec(I(xf), I(theta)) - I(yf)).square().sum() / self.n + self.lam * dot(I(theta), I(theta))
        residual = matvec(I(xf), I(theta)) - I(yf)
        grad = []
        for j in range(self.p):
            grad.append((I(xf[:, j]) * residual).sum() / self.n + self.lam * I(theta[j]))
        grad_i = I([float(g.lo) for g in grad], [float(g.hi) for g in grad])
        optimum_lower = loss - grad_i.square().sum() / self.lam
        return max(0.0, float((maximum - optimum_lower).hi))

    def evaluate(self, z):
        key = tuple(z)
        if key in self.cache:
            return self.cache[key]
        self.calls += 1
        v = matvec(I(self.t), I(z))
        xv = matvec(I(self.c), v)
        vj = v[self.j]
        bj = I(self.beta[self.j])
        s = I(self.s)
        a = s.square() * (2 * bj * vj - vj.square())
        b = 2 * s * (bj * xv + vj * self.r0 - vj * xv)
        c = 2 * self.r0 * xv - xv.square()
        upper = I(quadratic_max_upper(a, b, c)).sum() / self.n + self.lam * (2 * dot(I(self.beta), v) - v.square().sum())
        value = float(upper.hi)
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite row-oracle bound')
        if value > self.best_lower or self.best_witness is None:
            vf = self.t @ np.asarray(z)
            xvf = self.c @ vf
            r0f = self.c @ self.beta - self.y
            af = self.s ** 2 * (2 * self.beta[self.j] * vf[self.j] - vf[self.j] ** 2)
            bf = 2 * self.s * (self.beta[self.j] * xvf + vf[self.j] * r0f - vf[self.j] * xvf)
            zz = np.where(bf >= 0, 1.0, -1.0)
            mask = af < 0
            zz[mask] = np.clip(-bf[mask] / (2 * af[mask]), -1, 1)
            x = self.c.copy()
            indices = np.arange(self.n)
            x[indices, self.j] = np.clip(x[indices, self.j] + self.s * zz, self.lo[indices, self.j], self.hi[indices, self.j])
            matrix = x.T @ x / self.n + self.lam * np.eye(self.p)
            theta = np.linalg.solve(matrix, x.T @ self.y / self.n)
            lower = float((self.loss(x, self.beta) - self.loss(x, theta)).lo)
            if lower > self.best_lower:
                self.best_lower = lower
                self.best_witness = {'z': list(z), 'completion_coordinate': zz.tolist(), 'completion_values': x[indices, self.j].tolist(), 'theta': theta.tolist(), 'regret_lower': lower}
        self.cache[key] = value
        return value

    def node_upper(self, lo, hi, cap):
        maximum = max((self.evaluate(z) for z in product(*zip(lo, hi))))
        radius = (I(hi) - I(lo)) / 2
        error = (I(self.hdiag) * radius.square()).sum()
        return min(cap, float((I(maximum) + error).hi))

    def run(self, relative_tolerance=0.01, absolute_tolerance=1e-09, max_nodes=12000, time_limit=60, record_certificate=False):
        started = time.monotonic()
        lo = tuple([-self.radius] * self.p)
        hi = tuple([self.radius] * self.p)
        self.evaluate(np.zeros(self.p))
        upper = self.node_upper(lo, hi, self.initial_upper)
        counter = 0
        heap = [(-upper, counter, lo, hi)]
        expanded = 0
        reason = 'tolerance'
        tree = [{'id': 0, 'parent': None, 'lo': lo, 'hi': hi, 'upper': upper, 'children': []}]
        while heap:
            upper = max(self.best_lower, -heap[0][0])
            if upper - self.best_lower <= max(absolute_tolerance, relative_tolerance * self.best_lower):
                break
            if expanded >= max_nodes:
                reason = 'node_limit'
                break
            if time.monotonic() - started >= time_limit:
                reason = 'time_limit'
                break
            negative, node_id, lo, hi = heapq.heappop(heap)
            if -negative <= self.best_lower:
                continue
            widths = np.array(hi) - lo
            j = int(np.argmax(self.hdiag * widths ** 2))
            mid = (lo[j] + hi[j]) / 2
            lowhi = list(hi)
            lowhi[j] = mid
            highlo = list(lo)
            highlo[j] = mid
            for child_lo, child_hi in ((lo, tuple(lowhi)), (tuple(highlo), hi)):
                child_upper = self.node_upper(child_lo, child_hi, -negative)
                counter += 1
                tree[node_id]['children'].append(counter)
                tree.append({'id': counter, 'parent': node_id, 'lo': child_lo, 'hi': child_hi, 'upper': child_upper, 'children': []})
                if child_upper > self.best_lower:
                    heapq.heappush(heap, (-child_upper, counter, child_lo, child_hi))
            expanded += 1
        upper = max(self.best_lower, -heap[0][0] if heap else self.best_lower)
        self.search_tree = tree
        result = {'status': 'PASS', 'stop_reason': reason, 'n': self.n, 'p': self.p, 'uncertain_cells': int(np.sum(self.s > 0)), 'lower': self.best_lower, 'upper': upper, 'relative_gap': (upper - self.best_lower) / max(self.best_lower, 1e-300), 'direct_upper': self.direct_upper, 'gradient_rectangle_upper': self.gradient_upper, 'mu_preconditioned': self.mu, 'root_radius': self.radius, 'nodes_expanded': expanded, 'open_nodes': len(heap), 'row_oracle_calls': self.calls, 'seconds': time.monotonic() - started, 'witness': self.best_witness, 'certificate_arithmetic': 'float64 intervals with outward elementary operations and conservative finite-sum error bounds'}
        if record_certificate:
            result['certificate'] = {'lower_table': self.lo.tolist(), 'upper_table': self.hi.tolist(), 'response': self.y.tolist(), 'beta': self.beta.tolist(), 'lambda': self.lam, 'preconditioner': self.t.tolist(), 'hdiag_upper': self.hdiag.tolist(), 'gradient_box_lower': self.gradient_box.lo.tolist(), 'gradient_box_upper': self.gradient_box.hi.tolist(), 'initial_upper': self.initial_upper, 'nodes': tree, 'points': [{'z': list(z), 'upper': v} for z, v in self.cache.items()]}
        return result
