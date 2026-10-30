from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Verified residual-energy refinement for the declared v75 mechanism family.\n\nThe search and original-table witnesses reuse v74. This module verifies a\nspecific original-data sufficient condition; it is not a general selector.\n'
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
from fractions import Fraction
from math import isqrt
from pathlib import Path
import sys
import numpy as np
BASE = Path(__file__).resolve().parent
V74 = MODULE_DIR
from observed_anchor import AnchorSearch, ObservedAnchor, Q, float_up
from moment_tail import MomentSearch

def structure_problem(m=64, alpha=Fraction(1, 32)):
    """Return (lo, hi, y, beta, lambda), with independent original intervals."""
    m = int(m)
    root = isqrt(m)
    alpha = Fraction(alpha)
    if m < 4 or root * root != m or alpha < 0:
        raise ValueError('Require square m >= 4 and alpha >= 0')
    af = float(alpha)
    if Fraction(af) != alpha:
        raise ValueError('alpha must be exactly representable as binary64')
    n, p = (2 * m, m + 1)
    lo = np.zeros((n, p))
    hi = np.zeros_like(lo)
    y = np.empty(n)
    beta = np.zeros(p)
    beta[0] = 1.0
    lo[:m, 0], hi[:m, 0], y[:m] = (0.75, 1.25, 2.0)
    lo[m:, 0] = hi[m:, 0] = af
    y[m:] = af
    for j in range(m):
        lo[m + j, j + 1], hi[m + j, j + 1] = (root, 2 * root)
    return (lo, hi, y, beta, 0.5)

def analytic_structure(m=64, alpha=Fraction(1, 32)):
    """Exact targets and provable separations, without numerical eigensolvers."""
    m, alpha = (int(m), Fraction(alpha))
    exact_h = Fraction(5) / (4000 + 512 * alpha ** 2)
    exact_v = Fraction(1) / (800 + 512 * alpha ** 2)
    eta = alpha ** 2 / 1250
    delta = Fraction(5, 8) - alpha ** 2 / 2
    if delta <= 0:
        raise ValueError('PCA comparison requires alpha^2 < 5/4')
    pca_bound = Fraction(1, 512) * (Fraction(9, 16) * alpha ** 2) / delta ** 2
    b0 = 2 + alpha ** 2 / 2
    qvalues = {'alpha': alpha, 'exact_original_H': exact_h, 'exact_retained_V': exact_v, 'actual_reduction_gap': exact_h - exact_v, 'energy_tail': eta, 'ideal_generic_moment_tail': alpha ** 2 / 128, 'original_gradient_upper': Fraction(1, 512), 'original_projection_radius': Fraction(1, 25), 'signal_gradient_upper': Fraction(1, 32), 'signal_curvature_lower': Fraction(25, 32), 'nuisance_retained_operator_squared': alpha ** 2 / 2, 'endpoint_covariance_lambda_rplus1_lower': Fraction(5, 4), 'ridge_whitened_rho_lower': Fraction(5, 2), 'spectral_B0_squared': b0, 'spectral_tail_budget_lower': Fraction(5, 7) * b0, 'discounted_upper_lower_bound': Fraction(159, 224) - alpha ** 2 / 7, 'midpoint_covariance_pca_restricted_upper': pca_bound, 'pca_original_gap_lower': exact_h - pca_bound}
    return {'m': m, 'n': 2 * m, 'p': m + 1, 'independent_uncertain_cells': 2 * m, 'uncertain_columns': m + 1, 'full_column_rank_every_completion': True, 'all_nuisance_intervals_affect_regret': alpha > 0, 'spectral_rank_range': [1, m - 1], 'pca_rank_range': [1, m], 'fraction': {key: str(value) for key, value in qvalues.items()}, 'numeric': {key: float(value) for key, value in qvalues.items()}, 'discounted_upper_scope': 'Lower bound on the covariance-discounted retained upper itself; independent original-objective caps may tighten final software output.'}

def _serial_generic_receipt(receipt):
    return {key: value for key, value in receipt.items() if key not in ('projector', 'H', 'qH')}

def verify_structure_anchor(anchor, retained):
    """Check every input cell, exact midpoint gradient, and actual T/R geometry.

    Array comparisons here compare actual binary64 dyadic numbers. All
    nonzero input constants are subsequently represented as Fractions.
    """
    a = anchor
    if a.n % 2 or a.p != a.n // 2 + 1:
        raise ValueError('Input dimensions do not match the declared family')
    m = a.n // 2
    alpha = Fraction(float(a.lo[m, 0]))
    expected = structure_problem(m, alpha)
    for name, actual, required in zip(('lo', 'hi', 'y', 'beta'), (a.lo, a.hi, a.y, a.beta), expected[:4]):
        if not np.array_equal(actual, required):
            raise ValueError('Family verification failed for ' + name)
    if Fraction(a.lam) != Fraction(1, 2):
        raise ValueError('The declared ridge penalty is lambda=1/2')
    if a.anchor_type != 'observed' or np.any(a.anchor_rows):
        raise ValueError('Require the all-uncertain-row observed/ridge anchor')
    if a.preconditioner_type != 'anchor':
        raise ValueError('Exact proposal recovery requires ridge-anchor preprocessing')
    scale = Fraction(float(a.t[0, 0]))
    if scale <= 0 or not np.array_equal(a.t, float(scale) * np.eye(a.p)):
        raise ValueError('Actual dyadic preconditioner must be positive scalar')
    if retained.shape != (a.p, 1):
        raise ValueError('This family uses one retained direction')
    sign = Fraction(float(retained[0, 0]))
    expected_r = np.zeros_like(retained)
    expected_r[0, 0] = float(sign)
    if abs(sign) != 1 or not np.array_equal(retained, expected_r):
        raise ValueError('Existing gradient proposal did not recover span(e0) exactly')
    gradient = [Fraction(a.lam) * Fraction(float(b)) for b in a.beta]
    for i in range(a.n):
        residual = Fraction(float(a.c[i, 0])) - Fraction(float(a.y[i]))
        for j in np.flatnonzero(a.c[i]):
            gradient[int(j)] += Fraction(float(a.c[i, j])) * residual / a.n
    if any(gradient):
        raise ArithmeticError('Actual full-feature midpoint ridge gradient is nonzero')
    projection_radius = Fraction(1, 25)
    exact_coordinate_radius = projection_radius / abs(scale * sign)
    grid = 2 ** 20
    scaled = exact_coordinate_radius * grid
    grid_units = (scaled.numerator + scaled.denominator - 1) // scaled.denominator
    radius = Fraction(grid_units, grid)
    if radius < exact_coordinate_radius:
        raise ArithmeticError('Rounded root does not enclose the required projection')
    return {'certificate_type': 'verified_original_row_residual_energy_family', 'm': m, 'n': a.n, 'p': a.p, 'alpha': str(alpha), 'lambda': '1/2', 'G': '1/32', 'kappa': '25/32', 'a_squared': str(alpha ** 2 / 2), 'epsilon': '0', 'original_projection_radius': str(projection_radius), 'tail_error': str(alpha ** 2 / 1250), 'scalar_T': str(scale), 'basis_sign': str(sign), 'exact_coordinate_radius': str(exact_coordinate_radius), 'rounded_coordinate_radius': str(radius), 'actual_original_root_radius': str(abs(scale * sign) * radius), 'midpoint_gradient': [str(v) for v in gradient], 'signal_rows': [0, m], 'nuisance_rows': [m, 2 * m], 'row_ranges_are_half_open': True, 'all_input_cells_match_declared_family': True, 'effective_anchor': 'ridge_only_because_no_rows_are_fully_observed', 'all_feature_columns_independently_uncertain': True, 'full_column_rank_every_completion': True, 'actual_positive_nuisance_sensitivity': alpha > 0, 'gradient_proposal_raw_span_e0': True, 'root_scope': 'Tail is proved at the full optimizer projection, whose original magnitude is <=1/25. The rounded search box may be slightly larger.'}

class ResidualEnergySearch(MomentSearch):
    """v74 original-metric search with a verified family residual-energy tail."""

    def __init__(self, anchor, search_bound='adaptive'):
        super().__init__(anchor, 1, kind='gradient', search_bound=search_bound)
        self.generic_moment_receipt = _serial_generic_receipt(self.tail_receipt)
        self.generic_moment_tail = self.qtail
        self.structural_receipt = verify_structure_anchor(anchor, self.r)
        self.qtail = Fraction(self.structural_receipt['tail_error'])
        self.radius = float(Fraction(self.structural_receipt['rounded_coordinate_radius']))
        self.initial_upper = float_up(Fraction(anchor.gradient_upper) + self.qtail)

    def run(self, **kwargs):
        result = AnchorSearch.run(self, **kwargs)
        result.update({'method': 'residual_energy_structure', 'basis_kind': 'gradient', 'structural_receipt': self.structural_receipt, 'generic_v74_tail_error': float_up(self.generic_moment_tail), 'generic_v74_tail_error_fraction': str(self.generic_moment_tail), 'generic_v74_coordinate_radius': self.generic_moment_receipt['coordinate_radius'], 'coordinate_root_radius': self.radius, 'original_parameter_root_radius_fraction': self.structural_receipt['actual_original_root_radius'], 'search_values_include_additive_tail': True, 'analytic': analytic_structure(self.structural_receipt['m'], Fraction(self.structural_receipt['alpha']))})
        if 'certificate' in result:
            certificate = result['certificate']
            certificate.update({'method': 'residual_energy_structure', 'basis_kind': 'gradient', 'original_gradient_upper': self.a.gradient_upper, 'hdiag': self.hdiag.tolist(), 'tail_error': str(self.qtail), 'anchor_error': '0', 'coordinate_radius': self.radius, 'search_bound': self.mode, 'initial_upper': self.initial_upper, 'structural_receipt': self.structural_receipt, 'generic_moment_tail_receipt': self.generic_moment_receipt, 'search_values_include_additive_tail': True, 'node_bound_rule': 'min(parent_cap, center_gradient, corner_interpolation)', 'point_objective': 'original_retained_displacement_objective_plus_exact_tail'})
        return result
