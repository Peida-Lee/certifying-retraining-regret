from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Verified dyadic subspaces for the sharp ridge spectral-tail theorem.\n\nThe basis is not assumed orthonormal. Its exact Gram controls both the\nprojector and the retained quadratic penalty. All intervals are outward.\n'
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import sys
from pathlib import Path
from fractions import Fraction as Q
import numpy as np
from interval_arithmetic import I, dot, matvec, up, down
from dual_certificate import FullBoxOracle
from rank_certificate import qdot, float_up, float_down
from discounted_planar_oracle import maximize
GRID = 2 ** 80

def transpose(a):
    return I(a.lo.T, a.hi.T)

def mm(a, b):
    out = I(np.zeros((a.lo.shape[0], b.lo.shape[1])))
    for k in range(a.lo.shape[1]):
        out = out + a[:, k][:, None] * b[k, :][None, :]
    return out

def exact_inverse(a):
    n = len(a)
    b = [row[:] + [Q(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if b[i][j]))
        b[j], b[pivot] = (b[pivot], b[j])
        z = b[j][j]
        b[j] = [v / z for v in b[j]]
        for i in range(n):
            if i != j:
                z = b[i][j]
                b[i] = [x - z * y for x, y in zip(b[i], b[j])]
    return [row[n:] for row in b]

def interval_q(a):
    return I([[float_down(v) for v in row] for row in a], [[float_up(v) for v in row] for row in a])

def eigen_gershgorin(a, upper=False):
    bounds = []
    for j in range(a.lo.shape[0]):
        off = I([max(abs(float(a.lo[j, k])), abs(float(a.hi[j, k]))) for k in range(a.lo.shape[0]) if k != j]).sum()
        bounds.append(float((a[j, j] + off).hi) if upper else float((a[j, j] - off).lo))
    return max(bounds) if upper else min(bounds)

def covariance_envelope(center, radii, scale):
    n = len(center)
    c = I(center) * scale
    s = I(radii) * scale
    k = np.sum(radii != 0, axis=1)
    diagonal = (I(k[:, None]) * s.square()).sum(axis=0)
    return (mm(transpose(c), c) + I(np.diag(diagonal.lo), np.diag(diagonal.hi))) * I(2) / n

def basis_receipt(k, r, mandatory=()):
    p = k.lo.shape[0]
    if r == p:
        eye = np.eye(p)
        return {'rank': p, 'R': eye.tolist(), 'rho': 0.0, 'metric_mu': 1.0, 'test_basis': eye.tolist(), 'test_mu': 1.0, 'test_nu': 0.0, 'exact_full_rank': True}
    midpoint = (k.lo + k.hi) / 2
    if mandatory:
        remaining = [j for j in range(p) if j not in mandatory]
        rr = np.zeros((p, r))
        rr[list(mandatory), range(len(mandatory))] = 1
        extra = r - len(mandatory)
        if extra:
            _, v = np.linalg.eigh(midpoint[np.ix_(remaining, remaining)])
            rr[np.ix_(remaining, range(len(mandatory), r))] = np.round(v[:, -extra:] * 2 ** 24) / 2 ** 24
    else:
        _, v = np.linalg.eigh(midpoint)
        rr = np.round(v[:, -r:] * 2 ** 24) / 2 ** 24
    qr = [[Q(x) for x in row] for row in rr]
    h = [[sum((qr[i][j] * qr[i][l] for i in range(p)), Q(0)) for l in range(r)] for j in range(r)]
    hinv = exact_inverse(h)
    hmu = eigen_gershgorin(interval_q(h))
    if hmu <= 0:
        raise ValueError('Retained basis is not verified independent')
    pp = np.eye(p) - rr @ np.linalg.solve(rr.T @ rr, rr.T)
    _, vv = np.linalg.eigh(pp @ midpoint @ pp)
    test = np.round(vv * 2 ** 24) / 2 ** 24
    a = I(test) - mm(mm(I(rr), interval_q(hinv)), mm(I(rr.T), I(test)))
    nu = eigen_gershgorin(mm(transpose(a), mm(k, a)), upper=True)
    mu = eigen_gershgorin(mm(I(test.T), I(test)))
    if mu <= 0:
        raise ValueError('Test basis not verified invertible')
    rho = max(0.0, float((I(nu) / mu).hi))
    return {'rank': r, 'R': rr.tolist(), 'rho': rho, 'metric_mu': hmu, 'test_basis': test.tolist(), 'test_mu': mu, 'test_nu': nu, 'exact_full_rank': False}

class AutomaticRankCertificate(FullBoxOracle):

    def __init__(self, lo, hi, y, beta, lam, tail_budget=2e-05, max_rank=3, protect_uncertainty=False):
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.y = np.asarray(y, float)
        self.beta = np.asarray(beta, float)
        self.lam = float(lam)
        self.n, self.ambient_p = self.lo.shape
        self.tail_budget = Q(tail_budget)
        self.center = (self.lo + self.hi) / 2
        self.radii = (self.hi - self.lo) / 2
        if np.any(self.center - self.radii != self.lo) or np.any(self.center + self.radii != self.hi):
            raise ValueError('Exact dyadic centers needed')
        self.scale = 1 / np.sqrt(self.lam)
        if Q(self.scale) ** 2 * Q(self.lam) != 1:
            raise ValueError('Exact dyadic ridge whitening required')
        self.qy = list(map(Q, self.y))
        self.qbeta = list(map(Q, self.beta))
        self.qlam = Q(self.lam)
        self.qscale = Q(self.scale)
        self.b0 = sum((v * v for v in self.qy), Q(0)) / self.n
        mandatory = np.flatnonzero(np.any(self.radii != 0, axis=0)).tolist() if protect_uncertainty else []
        if protect_uncertainty:
            if len(mandatory) > max_rank:
                raise ValueError('Rank cap below protected uncertainty span')
            c = I(self.center) * self.scale
            self.envelope = mm(transpose(c), c) / self.n
        else:
            self.envelope = covariance_envelope(self.center, self.radii, self.scale)
        self.rank_path = []
        candidates = []
        for r in range(max(1, len(mandatory)), min(self.ambient_p, max_rank) + 1):
            selected = basis_receipt(self.envelope, r, mandatory)
            rho = Q(selected['rho'])
            tail = rho / (1 + rho) * self.b0
            self.rank_path.append({'rank': r, 'rho': selected['rho'], 'tail_error': float_up(tail)})
            candidates.append((selected, rho, tail))
            if tail <= Q(tail_budget):
                break
        if tail > Q(tail_budget):
            best = min((v[2] for v in candidates))
            selected, rho, tail = next((v for v in candidates if float(v[2] - best) <= max(1e-12, 1e-07 * float(best))))
        selected['protected_coordinates'] = mandatory
        selected['envelope_type'] = 'center covariance; uncertainty span retained exactly' if protect_uncertainty else 'Young covariance envelope'
        self.receipt = selected
        self.p = selected['rank']
        self.r = np.array(selected['R'])
        self.qr = [[Q(x) for x in row] for row in self.r]
        self.qh = [[sum((self.qr[i][j] * self.qr[i][k] for i in range(self.ambient_p)), Q(0)) for k in range(self.p)] for j in range(self.p)]
        self.rho = rho
        self.gamma = 1 / (1 + rho)
        self.tail = tail
        self.qc = self.qlam * qdot(self.qbeta, self.qbeta)
        self.qcenter = [[Q(v) for v in row] for row in self.center]
        self.aproj = [qdot(row, self.qbeta) - yy for row, yy in zip(self.qcenter, self.qy)]
        self.bproj = [[self.qscale * sum((row[l] * self.qr[l][j] for l in range(self.ambient_p)), Q(0)) for j in range(self.p)] for row in self.qcenter]
        self.uncertain = [np.flatnonzero(row).tolist() for row in self.radii]
        self.agenerator = [[Q(self.radii[i, j]) * self.qbeta[j] for j in inds] for i, inds in enumerate(self.uncertain)]
        self.bgenerator = [[[self.qscale * Q(self.radii[i, j]) * self.qr[j][k] for k in range(self.p)] for j in inds] for i, inds in enumerate(self.uncertain)]
        self.hdiag = []
        for j in range(self.p):
            total = sum(((abs(self.bproj[i][j]) + sum((abs(g[j]) for g in self.bgenerator[i]), Q(0))) ** 2 for i in range(self.n)), Q(0)) / self.n
            self.hdiag.append(float_up(self.qh[j][j] + self.gamma * total))
        self.hdiag = np.array(self.hdiag)
        self.mu = selected['metric_mu']
        radius_squared = self.gamma * self.b0 / (4 * Q(self.mu))
        self.radius = np.ceil(np.sqrt(float(radius_squared)) * 2 ** 20) / 2 ** 20
        while Q(float(self.radius)) ** 2 < radius_squared:
            self.radius += 2 ** (-20)
        candidate = self.qc + sum(((abs(a) + sum((abs(g) for g in gg), Q(0))) ** 2 for a, gg in zip(self.aproj, self.agenerator)), Q(0)) / self.n
        self.initial_upper = float_up(candidate)
        self.gradient_upper = self.initial_upper
        self.direct_upper = self.initial_upper
        self.best_lower = -np.inf
        self.best_original_lower = 0.0
        self.best_witness = None
        self.best_original_witness = None
        self.calls = 0
        self.cache = {}
        self.s = np.max(self.radii, axis=1)
        self.fixed = self.s == 0

    def evaluate(self, z):
        key = tuple(z)
        if key in self.cache:
            return self.cache[key]
        self.calls += 1
        qz = list(map(Q, z))
        high = low = 0
        row_signs = []
        for i in range(self.n):
            b0 = qdot(self.bproj[i], qz) - self.qy[i]
            generators = [(aa, qdot(bb, qz)) for aa, bb in zip(self.agenerator[i], self.bgenerator[i])]
            value, signs = maximize(self.aproj[i], b0, generators, self.gamma, True)
            high += (value.numerator * GRID + value.denominator - 1) // value.denominator
            low += value.numerator * GRID // value.denominator
            row_signs.append(signs)
        anchor = self.qc - qdot(qz, [qdot(row, qz) for row in self.qh])
        upper = float_up(Q(high, GRID * self.n) + anchor)
        lower = float_down(Q(low, GRID * self.n) + anchor)
        if lower > self.best_lower:
            self.best_lower = lower
            x = self.center.copy()
            for i, inds in enumerate(self.uncertain):
                for j, sgn in zip(inds, row_signs[i]):
                    x[i, j] = np.clip(float(self.qcenter[i][j] + Q(self.radii[i, j]) * sgn), self.lo[i, j], self.hi[i, j])
            theta = np.linalg.solve(x.T @ x / self.n + self.lam * np.eye(self.ambient_p), x.T @ self.y / self.n)
            original = float((self.loss(x, self.beta) - self.loss(x, theta)).lo)
            if original > self.best_original_lower:
                self.best_original_lower = original
                self.best_original_witness = {'x': x.tolist(), 'theta': theta.tolist()}
            self.best_witness = {'z': list(z), 'upper_problem_lower': lower, 'full_model_feasible_lower': original}
        self.cache[key] = upper
        return upper

    def run(self, record=False, **kwargs):
        result = super().run(**kwargs)
        search_lower = result['lower']
        lower = max(0.0, self.best_original_lower, float_down(Q(search_lower) - self.tail))
        result.update({'ambient_features': self.ambient_p, 'retained_dimension': self.p, 'lower': lower, 'relative_gap': (result['upper'] - lower) / max(lower, 1e-300), 'search_error': result['upper'] - search_lower, 'upper_problem_lower': search_lower, 'tail_error_bound': float_up(self.tail), 'rho': float_up(self.rho), 'rank_path': self.rank_path, 'rank_budget_met': self.tail <= self.tail_budget, 'basis_verified': True, 'basis_type': 'automatic dyadic eigenbasis with actual nonorthogonal Gram', 'uncertain_cells': int(np.sum(self.radii > 0))})
        if record:
            result['certificate'] = {'lo': self.lo.tolist(), 'hi': self.hi.tolist(), 'y': self.y.tolist(), 'beta': self.beta.tolist(), 'lambda': self.lam, 'scale': self.scale, 'basis': self.receipt, 'envelope_lo': self.envelope.lo.tolist(), 'envelope_hi': self.envelope.hi.tolist(), 'hdiag': self.hdiag.tolist(), 'radius': self.radius, 'nodes': self.search_tree, 'points': [{'z': list(z), 'upper': v} for z, v in self.cache.items()], 'witness': self.best_original_witness}
        return result
