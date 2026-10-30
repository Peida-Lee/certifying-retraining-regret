from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Fixed-direction nonzero-residual extension of the v75 mechanism.'
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parent
V74 = MODULE_DIR
V75 = MODULE_DIR
from residual_energy import structure_problem
from observed_anchor import ObservedAnchor, AnchorSearch, I, mm, matvec, transpose, float_up, float_down, qdot
from moment_tail import MomentSearch

def epsilon_problem(delta):
    delta = Q(delta)
    if delta < 0 or Q(float(delta)) != delta:
        raise ValueError('delta must be a nonnegative binary64 dyadic')
    lo, hi, y, beta, lam = structure_problem(64, Q(1, 32))
    y[64:96] += float(delta)
    y[96:] -= float(delta)
    return (lo, hi, y, beta, lam)

def analytic(delta):
    d, a = (Q(delta), Q(1, 32))
    r = (1 + 16 * a * d) / 25
    eta = (d + a * r) ** 2 / 2
    h = Q(5) / (4000 + 512 * a * a) + 2 * d * d / 5
    v = Q(1) / (800 + 512 * a * a)
    values = {'delta': d, 'epsilon_squared': d * d / 2, 'a_squared': a * a / 2, 'a_times_epsilon': a * d / 2, 'G': Q(1, 32), 'kappa': Q(25, 32), 'original_projection_radius': r, 'tail_error': eta, 'exact_original_H': h, 'exact_retained_V': v, 'actual_reduction_gap': h - v, 'original_gradient_cap': Q(1, 512) + 2 * d * d}
    return {'fraction': {k: str(vv) for k, vv in values.items()}, 'numeric': {k: float(vv) for k, vv in values.items()}, 'epsilon': float(d) / np.sqrt(2.0), 'epsilon_exact_expression': str(d) + '/sqrt(2)', 'candidate_is_perturbed_midpoint_fit': not bool(d), 'retained_direction_source': 'fixed_original_e0_from_declared_mechanism'}

def verify_inputs(a, delta, retained):
    inputs = epsilon_problem(delta)
    for name, actual, expected in zip(('lo', 'hi', 'y', 'beta'), (a.lo, a.hi, a.y, a.beta), inputs[:4]):
        if not np.array_equal(actual, expected):
            raise ValueError('Declared original-data family mismatch: ' + name)
    if Q(a.lam) != Q(1, 2) or a.anchor_type != 'observed' or np.any(a.anchor_rows):
        raise ValueError('Require ridge-only observed anchor, lambda=1/2')
    scale = Q(float(a.t[0, 0]))
    if scale <= 0 or not np.array_equal(a.t, float(scale) * np.eye(a.p)):
        raise ValueError('Require positive actual scalar dyadic transform')
    required = np.zeros((a.p, 1))
    required[0, 0] = 1.0
    if not np.array_equal(retained, required):
        raise ValueError('Require explicitly fixed Q=e0')
    gradient = [Q(a.lam) * Q(float(b)) for b in a.beta]
    for row, response in zip(a.c, a.y):
        residual = Q(float(row[0])) - Q(float(response))
        for j in np.flatnonzero(row):
            gradient[int(j)] += Q(float(row[j])) * residual / a.n
    expected_gradient = [Q(0)] + [-(1 if j < 32 else -1) * 3 * Q(delta) / 32 for j in range(64)]
    if gradient != expected_gradient:
        raise ArithmeticError('Exact midpoint-gradient identity failed')
    constants = analytic(delta)['fraction']
    physical = Q(constants['original_projection_radius'])
    exact_coordinate = physical / scale
    units = exact_coordinate * 2 ** 20
    rounded = Q((units.numerator + units.denominator - 1) // units.denominator, 2 ** 20)
    return {'certificate_type': 'verified_nonzero_residual_fixed_direction_family', 'm': 64, 'n': 128, 'p': 65, 'alpha': '1/32', 'delta': str(Q(delta)), 'lambda': '1/2', **{key: constants[key] for key in ('G', 'kappa', 'a_squared', 'epsilon_squared', 'a_times_epsilon', 'original_projection_radius', 'tail_error')}, 'epsilon_exact_expression': str(Q(delta)) + '/sqrt(2)', 'scalar_T': str(scale), 'basis_sign': '1', 'exact_coordinate_radius': str(exact_coordinate), 'rounded_coordinate_radius': str(rounded), 'actual_original_root_radius': str(scale * rounded), 'midpoint_gradient': [str(x) for x in gradient], 'candidate_is_perturbed_midpoint_fit': not bool(delta), 'automatic_direction_recovery_claimed': False, 'retained_direction_source': 'fixed_original_e0_from_declared_mechanism', 'signal_rows': [0, 64], 'nuisance_rows': [64, 128], 'row_ranges_are_half_open': True, 'all_input_cells_match_declared_family': True, 'full_column_rank_every_completion': True}

class EpsilonSearch(MomentSearch):
    """Reuse v74 oracle/tree, explicitly retain e0 rather than propose it."""

    def __init__(self, anchor, delta):
        self.a, self.rank, self.mode = (anchor, 1, 'adaptive')
        self.kind = 'fixed_e0'
        self.r = np.zeros((anchor.p, 1))
        self.r[0, 0] = 1.0
        self.structural_receipt = verify_inputs(anchor, delta, self.r)
        self.delta = Q(delta)
        self.H, self.qH, self.hmu = (I([[1.0]]), [[Q(1)]], 1.0)
        self.rho, self.qgamma, self.gamma = (0.0, Q(1), I(1.0))
        self.qtail = Q(self.structural_receipt['tail_error'])
        self.anchor_error = Q(0)
        self.K = mm(I(self.r.T), mm(anchor.metric, I(self.r)))
        qb = list(map(Q, anchor.beta))
        g0 = [qdot(row, qb) - v for row, v in zip(anchor.qG, anchor.qr)]
        self.tr = mm(I(anchor.t), I(self.r))
        self.bproj = mm(anchor.ct, I(self.r))
        self.linear = matvec(transpose(self.tr), I([float_down(v) for v in g0], [float_up(v) for v in g0]))
        pr = (I(anchor.width) * I(np.maximum(abs(self.tr[:, 0].lo), abs(self.tr[:, 0].hi))[anchor.cells])).sum(axis=1)
        bmax = np.maximum(abs(self.bproj[:, 0].lo), abs(self.bproj[:, 0].hi))
        self.hdiag = np.array([float((self.K[0, 0] + (I(bmax) + pr).square().sum() / anchor.n).hi)])
        self.radius = float(Q(self.structural_receipt['rounded_coordinate_radius']))
        self.initial_upper = float_up(Q(anchor.gradient_upper) + self.qtail)
        self.best_search, self.best_original, self.best_witness = (-np.inf, 0.0, None)
        self.cache, self.points, self.tree = ({}, [], [])

    def run(self, **kwargs):
        result = AnchorSearch.run(self, **kwargs)
        result.update({'method': 'nonzero_residual_energy', 'basis_kind': self.kind, 'structural_receipt': self.structural_receipt, 'search_values_include_additive_tail': True, 'analytic': analytic(self.delta)})
        if 'certificate' in result:
            result['certificate'].update({'method': 'nonzero_residual_energy', 'basis_kind': self.kind, 'hdiag': self.hdiag.tolist(), 'tail_error': str(self.qtail), 'anchor_error': '0', 'coordinate_radius': self.radius, 'search_bound': self.mode, 'initial_upper': self.initial_upper, 'original_gradient_upper': self.a.gradient_upper, 'structural_receipt': self.structural_receipt, 'search_values_include_additive_tail': True})
        return result
