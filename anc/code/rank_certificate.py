from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Controlled full-ambient regression certified in two retained directions.'
from fractions import Fraction as Q
import numpy as np
from dual_certificate import FullBoxOracle
from discounted_planar_oracle import maximize
from interval_arithmetic import I, matvec, dot
GRID = 2 ** 80

def qdot(a, b):
    return sum((x * y for x, y in zip(a, b)), Q(0))

def float_up(value):
    result = float(value)
    return float(np.nextafter(result, np.inf)) if Q(result) < value else result

def float_down(value):
    result = float(value)
    return float(np.nextafter(result, -np.inf)) if Q(result) > value else result

class RankTwoCertificate(FullBoxOracle):

    def __init__(self, lower, upper, response, beta, lam, rho):
        self.lo = np.asarray(lower, float)
        self.hi = np.asarray(upper, float)
        self.y = np.asarray(response, float)
        self.beta = np.asarray(beta, float)
        self.lam = float(lam)
        self.n, self.ambient_p = self.lo.shape
        self.p = 2
        assert Q(self.lam) == Q(1, 64)
        assert np.all(self.lo[:, 2:] == self.hi[:, 2:])
        self.center = (self.lo + self.hi) / 2
        self.radii = (self.hi - self.lo) / 2
        assert np.all(self.radii[:, :2] == 1 / 16)
        self.s = np.max(self.radii, axis=1)
        self.fixed = self.s == 0
        self.rho = Q(rho)
        self.gamma = 1 / (1 + self.rho)
        self.qbeta = list(map(Q, self.beta))
        self.qy = list(map(Q, self.y))
        self.qcenter = [[Q(v) for v in row] for row in self.center]
        self.residual = [qdot(row, self.qbeta) - y for row, y in zip(self.qcenter, self.qy)]
        self.qc = Q(1, 64) * qdot(self.qbeta, self.qbeta)
        self.b0 = sum((v * v for v in self.qy), Q(0)) / self.n
        self.tail = self.rho / (1 + self.rho) * self.b0
        candidate = self.qc + sum(((abs(r) + (abs(self.qbeta[0]) + abs(self.qbeta[1])) / 16) ** 2 for r in self.residual), Q(0)) / self.n
        self.initial_upper = float_up(candidate)
        self.gradient_upper = self.initial_upper
        self.direct_upper = self.initial_upper
        self.mu = 1.0
        self.t = np.eye(2)
        self.hdiag = np.array([float_up(1 + self.gamma * 64 * sum(((abs(row[j]) + Q(1, 16)) ** 2 for row in self.qcenter), Q(0)) / self.n) for j in range(2)])
        radius_squared = self.gamma * self.b0 / 4
        self.radius = np.ceil(np.sqrt(float(radius_squared)) * 2 ** 20) / 2 ** 20
        while Q(float(self.radius)) ** 2 < radius_squared:
            self.radius += 2 ** (-20)
        self.best_lower = -np.inf
        self.best_original_lower = 0.0
        self.best_original_witness = None
        self.best_witness = None
        self.calls = 0
        self.cache = {}

    def evaluate(self, z):
        key = tuple(z)
        if key in self.cache:
            return self.cache[key]
        self.calls += 1
        qz = list(map(Q, z))
        theta = [-8 * s for s in qz]
        row_values = []
        for i, row in enumerate(self.qcenter):
            b0 = qdot(row[:2], theta) - self.qy[i]
            generators = [(self.qbeta[j] / 16, theta[j] / 16) for j in range(2)]
            row_values.append(maximize(self.residual[i], b0, generators, self.gamma))
        high = sum(((v.numerator * GRID + v.denominator - 1) // v.denominator for v in row_values))
        low = sum((v.numerator * GRID // v.denominator for v in row_values))
        anchor = self.qc - qdot(qz, qz)
        upper = float_up(Q(high, GRID * self.n) + anchor)
        lower = float_down(Q(low, GRID * self.n) + anchor)
        if lower > self.best_lower:
            self.best_lower = lower
            x = self.center.copy()
            for i, row in enumerate(self.qcenter):
                b0 = qdot(row[:2], theta) - self.qy[i]
                _, signs = maximize(self.residual[i], b0, [(self.qbeta[j] / 16, theta[j] / 16) for j in range(2)], self.gamma, True)
                for j in range(2):
                    x[i, j] = np.clip(float(row[j] + signs[j] / 16), self.lo[i, j], self.hi[i, j])
            optimum = np.linalg.solve(x.T @ x / self.n + self.lam * np.eye(self.ambient_p), x.T @ self.y / self.n)
            original_lower = float((self.loss(x, self.beta) - self.loss(x, optimum)).lo)
            if original_lower > self.best_original_lower:
                self.best_original_lower = original_lower
                self.best_original_witness = {'completed_head': x[:, :2].tolist(), 'theta': optimum.tolist(), 'regret_lower': original_lower}
            self.best_witness = {'z': list(z), 'upper_problem_value_lower': lower, 'original_regret_lower': original_lower, 'completed_head': x[:, :2].tolist(), 'theta': optimum.tolist()}
        self.cache[key] = upper
        return upper

    def run(self, **kwargs):
        record = kwargs.pop('record_certificate', False)
        result = super().run(**kwargs)
        search_lower = result['lower']
        search_upper = result['upper']
        original_lower = max(0.0, self.best_original_lower, float_down(Q(search_lower) - self.tail))
        output = {'status': result['status'], 'stop_reason': result['stop_reason'], 'n': self.n, 'ambient_features': self.ambient_p, 'retained_dimension': 2, 'uncertain_cells': 2 * self.n, 'lower': original_lower, 'upper': search_upper, 'relative_gap': (search_upper - original_lower) / max(original_lower, 1e-300), 'upper_problem_lower': search_lower, 'upper_problem_upper': search_upper, 'search_error': search_upper - search_lower, 'tail_error_bound': float_up(self.tail), 'rho_exact': str(self.rho), 'gamma_exact': str(self.gamma), 'B0_squared_exact': str(self.b0), 'search_plus_tail_error': float_up(Q(search_upper) - Q(search_lower) + self.tail), 'feasible_full_model_lower': self.best_original_lower, 'nodes_expanded': result['nodes_expanded'], 'row_oracle_calls': self.calls, 'seconds': result['seconds'], 'witness': self.best_witness, 'root_radius': self.radius, 'row_oracle': 'exact discounted planar zonotope', 'scope': 'ridge-only exact whitening; retained coordinate basis; supplied tail spectral bound verified by integer construction'}
        if record:
            output['certificate'] = {'lower_table': self.lo.tolist(), 'upper_table': self.hi.tolist(), 'response': self.y.tolist(), 'beta': self.beta.tolist(), 'lambda': self.lam, 'hdiag_upper': self.hdiag.tolist(), 'nodes': self.search_tree, 'points': [{'z': list(z), 'upper': value} for z, value in self.cache.items()], 'initial_upper': self.initial_upper, 'best_original_witness': self.best_original_witness, 'parameterization': 'theta=(-8*s1,-8*s2,0,...,0); ridge-only anchor', 'retained_coordinates': [0, 1]}
        return output
