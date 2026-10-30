'Independent exact audit of the v75 representative residual-energy receipt.\n\nImports no optimization/certificate implementation and runs no search.  All\nproof checks use fractions.Fraction on the actual saved JSON numbers.  The\nanalytic optimum is an additional final check, never a substitute for the\npoint, node-bound, and binary-cover proofs.\n'
from __future__ import annotations
from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
import argparse
from collections import Counter
from fractions import Fraction as Q
import json
from math import isqrt
from pathlib import Path
import traceback
BASE = Path(__file__).resolve().parent
ZERO = Q(0)

class AuditFailure(AssertionError):
    pass

def require(condition, message):
    if not condition:
        raise AuditFailure(message)

def qvector(values):
    return [Q(value) for value in values]

def qmatrix(values):
    return [qvector(row) for row in values]

def dot(left, right):
    require(len(left) == len(right), 'Dot-product dimension mismatch')
    return sum((a * b for a, b in zip(left, right)), ZERO)

def scalar_quadratic_range(a, b, c):
    """Exact extrema of a*z^2+b*z+c on [-1,1]."""
    values = [a - b + c, a + b + c]
    if a:
        stationary = -b / (2 * a)
        if -1 < stationary < 1:
            values.append(c - b * b / (4 * a))
    return (min(values), max(values))

def scalar_row_max(candidate_center, comparator_center, candidate_generator, comparator_generator):
    """Direct original-row residual difference; no planar oracle imported."""
    return scalar_quadratic_range(candidate_generator ** 2 - comparator_generator ** 2, 2 * (candidate_center * candidate_generator - comparator_center * comparator_generator), candidate_center ** 2 - comparator_center ** 2)[1]

def interval_product(left, right):
    values = [a * b for a in left for b in right]
    return (min(values), max(values))

def original_loss(table, responses, theta, ridge):
    require(len(table) == len(responses), 'Loss row count mismatch')
    return sum(((dot(row, theta) - response) ** 2 for row, response in zip(table, responses)), ZERO) / len(responses) + ridge * dot(theta, theta)

def sparse_covariance(center, widths, average_endpoints=False):
    """Compute actual full original covariance using sparse row products."""
    n, p = (len(center), len(center[0]))
    answer = [[ZERO for _ in range(p)] for _ in range(p)]
    for row, radius in zip(center, widths):
        nonzero = [(j, value) for j, value in enumerate(row) if value]
        for j, a in nonzero:
            for k, b in nonzero:
                answer[j][k] += a * b / n
        if average_endpoints:
            for j, value in enumerate(radius):
                answer[j][j] += value * value / n
    return answer

def read_input_conditions(certificate):
    lo = qmatrix(certificate['lo'])
    hi = qmatrix(certificate['hi'])
    y = qvector(certificate['y'])
    beta = qvector(certificate['beta'])
    ridge = Q(certificate['lambda'])
    n, p = (len(y), len(beta))
    require(n > 0 and n % 2 == 0, 'Require even positive row count')
    m = n // 2
    require(p == m + 1 and m >= 4, 'Incorrect family dimensions')
    power = m
    while power % 4 == 0:
        power //= 4
    require(power == 1, 'Declared family requires m=4^k, k>=1')
    root = isqrt(m)
    require(root * root == m, 'sqrt(m) is not an exact integer')
    require(len(lo) == len(hi) == n, 'Input matrix row mismatch')
    require(all((len(row) == p for row in lo + hi)), 'Input width mismatch')
    require(ridge == Q(1, 2), 'Ridge must be 1/2, counted once')
    require(beta == [Q(1)] + [ZERO] * m, 'Candidate is not exactly e0')
    alpha = lo[m][0]
    require(alpha > 0, 'Representative audit requires nonzero alpha')
    require(alpha.denominator & alpha.denominator - 1 == 0, 'alpha is not dyadic')
    uncertain_columns = set()
    uncertain_count = 0
    for i in range(n):
        expected_lo = [ZERO] * p
        expected_hi = [ZERO] * p
        if i < m:
            expected_lo[0], expected_hi[0] = (Q(3, 4), Q(5, 4))
            expected_y = Q(2)
        else:
            j = i - m + 1
            expected_lo[0] = expected_hi[0] = alpha
            expected_lo[j], expected_hi[j] = (Q(root), Q(2 * root))
            expected_y = alpha
        require(lo[i] == expected_lo and hi[i] == expected_hi, 'Input row does not equal declared family: ' + str(i))
        require(y[i] == expected_y, 'Response mismatch at row ' + str(i))
        changing = [j for j in range(p) if lo[i][j] != hi[i][j]]
        require(len(changing) == 1, 'Each row must have one independent cell')
        uncertain_count += len(changing)
        uncertain_columns.update(changing)
    require(uncertain_count == 2 * m and uncertain_columns == set(range(p)), 'Uncertain-cell or uncertain-column condition failed')
    determinant_lower = Q(3, 4) * Q(root) ** m
    require(determinant_lower > 0, 'All-completion full-rank minor failed')
    center = [[(a + b) / 2 for a, b in zip(lrow, hrow)] for lrow, hrow in zip(lo, hi)]
    widths = [[(b - a) / 2 for a, b in zip(lrow, hrow)] for lrow, hrow in zip(lo, hi)]
    gradient = [ridge * b for b in beta]
    for row, response in zip(center, y):
        residual = dot(row, beta) - response
        for j, value in enumerate(row):
            if value:
                gradient[j] += value * residual / n
    require(not any(gradient), 'Exact midpoint full ridge gradient is nonzero')
    return dict(lo=lo, hi=hi, y=y, beta=beta, ridge=ridge, n=n, p=p, m=m, root=root, alpha=alpha, center=center, widths=widths, gradient=gradient, determinant_lower=determinant_lower)

def verify_structure_and_analytic(data, certificate, wrapper, result):
    m, n, p = (data['m'], data['n'], data['p'])
    alpha, ridge = (data['alpha'], data['ridge'])
    rec = certificate['structural_receipt']
    require(certificate['method'] == 'residual_energy_structure', 'Incorrect certificate method')
    require(certificate['basis_kind'] == 'gradient', 'Unexpected basis rule')
    require(Q(certificate['rho']) == 0 and Q(certificate['gamma']) == 1, 'Moment/energy objective must use rho=0 and gamma=1')
    require(Q(certificate['anchor_error']) == 0, 'Actual-metric retained objective must not add surrogate anchor error')
    require(certificate['search_values_include_additive_tail'] is True, 'Point/tail convention is not explicit')
    require(result['search_values_include_additive_tail'] is True, 'Result point/tail convention mismatch')
    for key, value in (('m', m), ('n', n), ('p', p)):
        require(rec[key] == value, 'Receipt dimension mismatch: ' + key)
        if key in wrapper:
            require(wrapper[key] == value, 'Wrapper dimension mismatch: ' + key)
    require(Q(rec['alpha']) == alpha, 'Receipt alpha mismatch')
    if 'alpha' in wrapper:
        require(Q(wrapper['alpha']) == alpha, 'Wrapper alpha mismatch')
    transform = qmatrix(certificate['T'])
    retained = qmatrix(certificate['R'])
    require(len(transform) == p and all((len(row) == p for row in transform)), 'Actual T matrix dimensions are wrong')
    require(len(retained) == p and all((len(row) == 1 for row in retained)), 'Actual retained basis is not p-by-1')
    scale = transform[0][0]
    require(scale > 0, 'Actual scalar preconditioner is not positive')
    require(all((transform[i][j] == scale * int(i == j) for i in range(p) for j in range(p))), 'Actual dense saved T is not the required scalar matrix')
    sign = retained[0][0]
    require(abs(sign) == 1 and all((retained[j][0] == 0 for j in range(1, p))), 'Actual rounded gradient basis is not exactly +/-e0')
    require(Q(rec['scalar_T']) == scale and Q(rec['basis_sign']) == sign, 'Actual T/R disagree with structural receipt')
    direction = [sum((transform[i][j] * retained[j][0] for j in range(p)), ZERO) for i in range(p)]
    require(direction == [scale * sign] + [ZERO] * m, 'Mapped retained direction differs from declared span')
    mu, nu = (Q(certificate['mu']), Q(certificate['nu']))
    require(0 < mu <= ridge * scale ** 2 <= nu, 'Actual rounded anchor metric enclosure failed')
    require(qvector(certificate['theta_bar']) == [ZERO] * p and qvector(certificate['theta_c']) == [ZERO] * p, 'Ridge-only anchor center must be zero')
    require(Q(certificate['C_safe']) == ridge, 'Original ridge anchor constant differs from 1/2')
    gradient_lower = gradient_upper = ridge
    for i in range(m):
        c, s = (data['center'][i][0], data['widths'][i][0])
        lower, upper = scalar_quadratic_range(s * s, 2 * s * (c - 1), c * c - 2 * c)
        gradient_lower += lower / n
        gradient_upper += upper / n
    require(gradient_lower == 0 and gradient_upper == Q(1, 32), 'Uniform retained signal-gradient range failed')
    G = max(abs(gradient_lower), abs(gradient_upper))
    kappa = ridge + sum((data['lo'][i][0] ** 2 / n for i in range(m)), ZERO)
    a_squared = sum((data['lo'][i][0] ** 2 / n for i in range(m, n)), ZERO)
    epsilon_squared = ZERO
    for i in range(m, n):
        c, radius = (data['center'][i], data['widths'][i])
        residual = dot(c, data['beta']) - data['y'][i]
        residual_radius = dot(radius, [abs(v) for v in data['beta']])
        epsilon_squared += (abs(residual) + residual_radius) ** 2 / n
    require(epsilon_squared == 0, 'Nuisance candidate residual is not zero')
    radius_original = G / kappa
    eta = a_squared * radius_original ** 2
    require(G == Q(1, 32) and kappa == Q(25, 32) and (a_squared == alpha ** 2 / 2) and (radius_original == Q(1, 25)) and (eta == alpha ** 2 / 1250), 'Independent structural constants failed')
    expected_receipt = {'lambda': ridge, 'G': G, 'kappa': kappa, 'a_squared': a_squared, 'epsilon': ZERO, 'original_projection_radius': radius_original, 'tail_error': eta}
    for key, expected in expected_receipt.items():
        require(Q(rec[key]) == expected, 'Structural constant mismatch: ' + key)
    require(qvector(rec['midpoint_gradient']) == data['gradient'], 'Saved midpoint gradient is not the exact original gradient')
    require(Q(certificate['tail_error']) == eta, 'Certificate tail mismatch')
    require(Q(result['tail_error']) >= eta, 'Reported tail rounded downward')
    require(Q(result['anchor_error']) == 0, 'Reported anchor error is not zero')
    require(rec['signal_rows'] == [0, m] and rec['nuisance_rows'] == [m, n] and (rec['row_ranges_are_half_open'] is True), 'Signal/nuisance row ranges are incorrect')
    exact_coordinate = radius_original / abs(direction[0])
    grid = 2 ** 20
    units = exact_coordinate * grid
    rounded_coordinate = Q((units.numerator + units.denominator - 1) // units.denominator, grid)
    coordinate_radius = Q(certificate['coordinate_radius'])
    require(Q(rec['exact_coordinate_radius']) == exact_coordinate, 'Exact coordinate radius mismatch')
    require(Q(rec['rounded_coordinate_radius']) == rounded_coordinate and coordinate_radius == rounded_coordinate, 'Actual root is not the declared outward dyadic root')
    actual_physical_radius = abs(direction[0]) * coordinate_radius
    require(actual_physical_radius >= radius_original, 'Physical root fails to cover the full optimizing projection')
    require(Q(rec['actual_original_root_radius']) == actual_physical_radius, 'Physical root receipt mismatch')
    original_cap = Q(certificate['original_gradient_upper'])
    require(original_cap >= scale ** 2 * G ** 2 / mu, 'Original gradient cap is not independently valid')
    require(Q(certificate['initial_upper']) >= original_cap + eta, 'Root parent cap is smaller than its claimed bound')
    require(Q(certificate['B0_squared']) == 2 + alpha ** 2 / 2, 'Original anchor residual energy mismatch')
    midpoint = sparse_covariance(data['center'], data['widths'])
    endpoint = sparse_covariance(data['center'], data['widths'], True)
    cross = Q(3) * alpha / (4 * data['root'])
    for i in range(p):
        for j in range(p):
            if i == j == 0:
                expected_mid = Q(1, 2) + alpha ** 2 / 2
                expected_end = Q(17, 32) + alpha ** 2 / 2
            elif i == j:
                expected_mid, expected_end = (Q(9, 8), Q(5, 4))
            elif i == 0 or j == 0:
                expected_mid = expected_end = cross
            else:
                expected_mid = expected_end = ZERO
            require(midpoint[i][j] == expected_mid and endpoint[i][j] == expected_end, 'Actual covariance identity failed at ' + str((i, j)))
    b_squared = m * cross ** 2
    delta = midpoint[1][1] - midpoint[0][0]
    require(b_squared == Q(9, 16) * alpha ** 2 and delta > 0, 'Midpoint PCA 2-by-2 block or eigenvalue-gap condition failed')
    projection_bound = b_squared / delta ** 2
    x = 4 * b_squared / delta ** 2
    require((1 + x / 2) ** 2 >= 1 + x and x > 0, 'PCA projection inequality algebra failed')
    pca_bound = projection_bound * G ** 2 / ridge
    H = G ** 2 / (kappa + alpha ** 2 / 10)
    V = G ** 2 / (kappa + alpha ** 2 / 2)
    require(H == Q(5) / (4000 + 512 * alpha ** 2) and V == Q(1) / (800 + 512 * alpha ** 2), 'Analytic endpoint extrema failed')
    require(0 < H - V <= eta, 'Nonzero reduction gap or residual-energy tail failed')
    t = Q(2 * data['root'])
    derivative_K = -alpha ** 2 * t / (m * m * (1 + t * t / m) ** 2)
    require(derivative_K < 0 and G > 0, 'Strict nuisance sensitivity failed')
    b0_squared = 2 + alpha ** 2 / 2
    discount_lower = Q(41, 32) - Q(2, 7) * b0_squared
    require(discount_lower == Q(159, 224) - alpha ** 2 / 7, 'Uncapped spectral-discount lower bound algebra failed')
    expected_analytic = {'alpha': alpha, 'exact_original_H': H, 'exact_retained_V': V, 'actual_reduction_gap': H - V, 'energy_tail': eta, 'ideal_generic_moment_tail': alpha ** 2 / 128, 'original_gradient_upper': Q(1, 512), 'original_projection_radius': radius_original, 'signal_gradient_upper': G, 'signal_curvature_lower': kappa, 'nuisance_retained_operator_squared': a_squared, 'endpoint_covariance_lambda_rplus1_lower': Q(5, 4), 'ridge_whitened_rho_lower': Q(5, 2), 'spectral_B0_squared': b0_squared, 'spectral_tail_budget_lower': Q(5, 7) * b0_squared, 'discounted_upper_lower_bound': discount_lower, 'midpoint_covariance_pca_restricted_upper': pca_bound, 'pca_original_gap_lower': H - pca_bound}
    analytic_objects = [result['analytic']]
    if 'analytic' in wrapper:
        analytic_objects.append(wrapper['analytic'])
    for analytic in analytic_objects:
        for key, expected in expected_analytic.items():
            require(Q(analytic['fraction'][key]) == expected, 'Analytic Fraction field mismatch: ' + key)
        require(analytic['spectral_rank_range'] == [1, m - 1] and analytic['pca_rank_range'] == [1, m], 'Spectral/PCA rank ranges are not their proved ranges')
        require(analytic['independent_uncertain_cells'] == 2 * m and analytic['uncertain_columns'] == p, 'Analytic input counts disagree with actual inputs')
    if alpha == Q(1, 32):
        require(H == Q(10, 8001) and V == Q(2, 1601) and (H - V == Q(8, 12809601)) and (eta == Q(1, 1280000)) and (pca_bound == Q(9, 3271682)) and (discount_lower == Q(5087, 7168)), 'Concrete representative rational constants failed')
    return dict(direction=direction, eta=eta, radius=coordinate_radius, original_cap=original_cap, H=H, V=V, pca_bound=pca_bound, discount_lower=discount_lower)

def verify_points_nodes_witness(data, certificate, result, constants):
    n, p = (data['n'], data['p'])
    direction, eta = (constants['direction'], constants['eta'])
    ridge, beta = (data['ridge'], data['beta'])
    coordinate_radius = constants['radius']
    rows = []
    for center, widths, response in zip(data['center'], data['widths'], data['y']):
        index = next((j for j, value in enumerate(widths) if value))
        rows.append((dot(center, beta) - response, widths[index] * beta[index], dot(center, direction), widths[index] * direction[index]))
    linear_ridge = ridge * dot(beta, direction)
    curvature_ridge = ridge * dot(direction, direction)

    def exact_point(z):
        value = 2 * linear_ridge * z - curvature_ridge * z * z
        for a, ga, projection, generator in rows:
            value += scalar_row_max(a, a - z * projection, ga, ga - z * generator) / n
        return value + eta
    points = {}
    exact_values = {}
    for number, point in enumerate(certificate['points']):
        zvalues = qvector(point['z'])
        require(len(zvalues) == 1, 'Saved point has incorrect rank')
        z = zvalues[0]
        require(-coordinate_radius <= z <= coordinate_radius, 'Saved point lies outside its physical search root')
        require(z not in points, 'Duplicate point coordinate in saved cache')
        value = exact_point(z)
        require(Q(point['lower']) <= value <= Q(point['upper']), 'Original scalar-row point enclosure failed: ' + str(number))
        points[z] = point
        exact_values[z] = value
    require(points, 'No saved point values')
    require(result['points'] == len(points), 'Result/cache point count mismatch')
    best_search = max((Q(point['lower']) for point in points.values()))
    require(Q(result['search_lower']) == best_search, 'Reported best search lower does not equal saved point cache')
    hdiag = qvector(certificate['hdiag'])
    require(len(hdiag) == 1, 'Unexpected curvature-diagonal rank')
    exact_curvature = curvature_ridge + sum(((abs(projection) + abs(generator)) ** 2 / n for _, _, projection, generator in rows), ZERO)
    require(hdiag[0] >= exact_curvature > 0, 'Original-objective curvature diagonal is not valid')
    mode = certificate['search_bound']
    require(mode in ('adaptive', 'center'), 'Unknown node-bound mode')
    nodes = certificate['nodes']
    require(nodes, 'Certificate has no search tree')
    incoming = Counter()
    terminal = []
    bound_counts = Counter()
    for index, node in enumerate(nodes):
        require(node['id'] == index, 'Node IDs are not canonical indices')
        lvalues, uvalues = (qvector(node['lo']), qvector(node['hi']))
        require(len(lvalues) == len(uvalues) == 1, 'Node is not rank one')
        lower, upper = (lvalues[0], uvalues[0])
        require(lower < upper, 'Empty/degenerate saved node')
        center, radius = ((lower + upper) / 2, (upper - lower) / 2)
        require(center in points, 'Node center missing from saved point cache')
        if index == 0:
            require(node['parent'] is None and lower == -coordinate_radius and (upper == coordinate_radius), 'Root does not equal saved root')
            parent_cap = Q(certificate['initial_upper'])
        else:
            parent = node['parent']
            require(isinstance(parent, int) and 0 <= parent < index, 'Invalid/cyclic node parent')
            parent_cap = Q(nodes[parent]['upper'])
        gradient_low = gradient_high = 2 * linear_ridge - 2 * curvature_ridge * center
        for a, ga, projection, generator in rows:
            residual_center = a - center * projection
            residual_radius = abs(ga - center * generator)
            product_low, product_high = interval_product((residual_center - residual_radius, residual_center + residual_radius), (projection - abs(generator), projection + abs(generator)))
            gradient_low += 2 * product_low / n
            gradient_high += 2 * product_high / n
        center_bound = Q(points[center]['upper']) + max(abs(gradient_low), abs(gradient_high)) * radius
        valid_bounds = {'parent_cap': parent_cap, 'center_gradient': center_bound}
        if mode != 'center':
            require(lower in points and upper in points, 'Corner interpolation endpoint missing from point cache')
            valid_bounds['corner_interpolation'] = max(Q(points[lower]['upper']), Q(points[upper]['upper'])) + hdiag[0] * radius ** 2
        saved_upper = Q(node['upper'])
        best_bound = min(valid_bounds.values())
        require(saved_upper >= best_bound, 'No verified bound justifies node upper: ' + str(index))
        require(saved_upper <= parent_cap, 'Node upper violates its saved parent cap: ' + str(index))
        for name, value in valid_bounds.items():
            if value == best_bound:
                bound_counts[name] += 1
        children = node['children']
        require(isinstance(children, list) and len(children) in (0, 2), 'Node must have zero or two children')
        if not children:
            terminal.append(saved_upper)
            continue
        left_id, right_id = children
        require(isinstance(left_id, int) and isinstance(right_id, int) and (index < left_id < right_id < len(nodes)), 'Invalid binary child IDs')
        left, right = (nodes[left_id], nodes[right_id])
        require(left['parent'] == right['parent'] == index, 'Child/parent linkage mismatch')
        require(qvector(left['lo']) == [lower] and qvector(left['hi']) == [center] and (qvector(right['lo']) == [center]) and (qvector(right['hi']) == [upper]), 'Binary children do not exactly partition parent interval')
        incoming[left_id] += 1
        incoming[right_id] += 1
    require(incoming.get(0, 0) == 0 and all((incoming[index] == 1 for index in range(1, len(nodes)))), 'Tree has orphan or multiply covered nodes')
    require(terminal and len(nodes) == 2 * result['nodes'] + 1, 'Terminal or expanded-node count mismatch')
    terminal_upper = max(terminal)
    search_upper = Q(result['search_upper'])
    require(search_upper >= max(terminal_upper, best_search), 'Final search upper does not bound every terminal leaf')
    require(Q(result['search_error']) >= search_upper - best_search, 'Reported search-error budget is rounded below its exact value')
    witness = certificate['witness']
    require(isinstance(witness, dict), 'No original full-table witness')
    table, theta = (qmatrix(witness['x']), qvector(witness['theta']))
    require(len(table) == n and len(theta) == p and all((len(row) == p for row in table)), 'Original witness dimensions are wrong')
    for i, (lrow, row, hrow) in enumerate(zip(data['lo'], table, data['hi'])):
        require(all((a <= v <= b for a, v, b in zip(lrow, row, hrow))), 'Original witness violates row interval ' + str(i))
    require(len(witness['z']) == 1 and Q(witness['z'][0]) in points, 'Original witness does not reference a saved retained point')
    witness_regret = original_loss(table, data['y'], beta, ridge) - original_loss(table, data['y'], theta, ridge)
    witness_lower = Q(witness['regret_lower'])
    require(witness_lower <= witness_regret, 'Saved original witness lower exceeds exact original ridge loss')
    feasible_lower = max(ZERO, witness_lower, best_search - eta)
    require(Q(result['lower']) <= feasible_lower, 'Final lower lacks an independently feasible justification')
    proof_upper = min(constants['original_cap'], terminal_upper)
    require(Q(result['upper']) >= proof_upper, 'Final upper is below the independently proved tree/cap upper')
    require(Q(result['lower']) <= constants['H'] <= Q(result['upper']), 'Final certificate excludes independent analytic H')
    return dict(point_count=len(points), node_count=len(nodes), terminal_count=len(terminal), mode=mode, bound_minimizers=dict(bound_counts), exact_terminal_upper=str(terminal_upper), independent_original_upper=str(proof_upper), exact_original_witness_regret=str(witness_regret), feasible_lower_justification=str(feasible_lower), exact_point_original_values_verified=True, every_node_upper_verified=True, complete_binary_coverage_verified=True, all_terminal_uppers_verified=True, original_full_witness_verified=True)

def audit(path):
    path = Path(path)
    wrapper = json.loads(path.read_text(encoding='utf-8'))
    result = wrapper.get('result', wrapper)
    certificate = result['certificate']
    require(result['status'] == 'PASS', 'Saved result is not PASS')
    data = read_input_conditions(certificate)
    constants = verify_structure_and_analytic(data, certificate, wrapper, result)
    tree = verify_points_nodes_witness(data, certificate, result, constants)
    return {'file': str(path.resolve()), 'status': 'PASS', 'audit_type': 'independent_Fraction_original_data_and_complete_tree', 'imports_solver_implementation': False, 'n': data['n'], 'p': data['p'], 'm': data['m'], 'alpha': str(data['alpha']), 'rank': 1, 'full_rank_minor_uniform_lower': str(data['determinant_lower']), 'all_family_input_conditions_verified': True, 'exact_midpoint_ridge_fit_verified': True, 'actual_scalar_T_and_rounded_basis_verified': True, 'original_data_radius_and_energy_tail_verified': True, 'analytic_and_covariance_pca_formulas_verified': True, 'tail_error': str(constants['eta']), 'exact_original_H': str(constants['H']), 'exact_retained_V': str(constants['V']), 'reported_lower': str(Q(result['lower'])), 'reported_upper': str(Q(result['upper'])), **tree}

def configure_affinity(cpu):
    """Pin the audit away from the user's reserved logical CPUs 0--3."""
    require(cpu >= 4, 'Audit CPU must leave logical CPUs 0--3 free')
    import psutil
    process = psutil.Process()
    process.cpu_affinity([cpu])
    require(process.cpu_affinity() == [cpu], 'Could not enforce audit affinity')
    return process.cpu_affinity()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('certificate', nargs='?', type=Path, default=BASE / 'structure_representative_certificate.json')
    parser.add_argument('--output', type=Path, default=BASE / 'independent_structure_audit.json')
    parser.add_argument('--cpu', type=int, default=6)
    args = parser.parse_args()
    try:
        affinity = configure_affinity(args.cpu)
        report = audit(args.certificate)
        report['logical_cpu_affinity'] = affinity
    except Exception as error:
        report = {'file': str(args.certificate), 'status': 'ERROR', 'error': str(error), 'traceback': traceback.format_exc()}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if report['status'] == 'PASS' else 1
if __name__ == '__main__':
    raise SystemExit(main())
