from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
from fractions import Fraction as Q
import sys
sys.dont_write_bytecode = True
B = Path(__file__).resolve().parent
import numpy as np
from unittest.mock import patch
import epsilon_energy as old
from observed_anchor import ObservedAnchor, AnchorSearch
from moment_tail import MomentSearch

def problem(delta):
    lo, hi, y, beta, lam = old.structure_problem(64, Q(1, 32))
    lo[:64, 1] = hi[:64, 1] = float(delta)
    return (lo, hi, y, beta, lam)

def constants(delta):
    d = Q(delta)
    alpha = Q(1, 32)
    kb = (25 + 16 * d * d) / (32 * (1 + d * d))
    r = (1 + 20 * alpha * d + 25 * d * d) * (1 + d * d) / (25 + 16 * d * d)
    eta = (alpha * r + Q(5, 4) * d * (1 + r)) ** 2 / 2
    return dict(delta=str(d), alpha=str(alpha), G='1/32', kappa='25/32', a_squared=str(alpha * alpha / 2), signal_operator_squared='25/32', signal_residual_squared='25/32', leakage_squared=str(d * d / 2), epsilon_squared='0', candidate_discarded_norm='0', kappa_b=str(kb), original_projection_radius=str(r), tail_error=str(eta), constant_scope='conservative rational upper of the proved leakage-energy theorem')

def receipt(a, delta, retained):
    for actual, expected in zip((a.lo, a.hi, a.y, a.beta), problem(delta)[:4]):
        assert np.array_equal(actual, expected)
    assert a.lam == 0.5 and (not np.any(a.anchor_rows)) and (a.preconditioner_type == 'anchor')
    scale = Q(float(a.t[0, 0]))
    assert scale > 0 and np.array_equal(a.t, float(scale) * np.eye(a.p))
    required = np.zeros((65, 1))
    required[0, 0] = 1
    assert np.array_equal(retained, required)
    c = constants(delta)
    radius = Q(c['original_projection_radius']) / scale
    units = radius * 2 ** 20
    rounded = Q((units.numerator + units.denominator - 1) // units.denominator, 2 ** 20)
    return dict(certificate_type='verified_signal_leakage_energy_family', m=64, n=128, p=65, **c, scalar_T=str(scale), basis_sign='1', exact_coordinate_radius=str(radius), rounded_coordinate_radius=str(rounded), actual_original_root_radius=str(scale * rounded), candidate_is_midpoint_fit=not bool(delta), automatic_direction_recovery_claimed=False, retained_direction_source='declared_e0', all_input_cells_verified=True)

class LeakageSearch(old.EpsilonSearch):

    def __init__(self, a, delta):
        with patch.object(old, 'verify_inputs', side_effect=receipt):
            super().__init__(a, delta)

    def run(self, **kwargs):
        r = AnchorSearch.run(self, **kwargs)
        r.update(method='signal_leakage_energy', basis_kind='fixed_e0', structural_receipt=self.structural_receipt, search_values_include_additive_tail=True)
        if 'certificate' in r:
            r['certificate'].update(method='signal_leakage_energy', basis_kind='fixed_e0', hdiag=self.hdiag.tolist(), tail_error=str(self.qtail), anchor_error='0', coordinate_radius=self.radius, search_bound=self.mode, initial_upper=self.initial_upper, original_gradient_upper=self.a.gradient_upper, structural_receipt=self.structural_receipt, search_values_include_additive_tail=True)
        return r
