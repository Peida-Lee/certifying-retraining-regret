'Independent exact audit of the v76 nonzero-residual receipts.\n\nImports no optimization/certificate implementation and runs no search.  All\nproof checks use fractions.Fraction on the actual saved JSON numbers.  The\nanalytic optimum is an additional final check, never a substitute for the\npoint, node-bound, and binary-cover proofs.\n'
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
    lo, hi = (qmatrix(certificate['lo']), qmatrix(certificate['hi']))
    y, beta = (qvector(certificate['y']), qvector(certificate['beta']))
    n, p, m, root = (len(y), len(beta), 64, 8)
    ridge = Q(certificate['lambda'])
    require(n == 128 and p == 65 and (len(lo) == len(hi) == n), 'Frozen dimensions failed')
    require(all((len(row) == p for row in lo + hi)), 'Input dimensions failed')
    require(beta == [Q(1)] + [ZERO] * m and ridge == Q(1, 2), 'Candidate or ridge changed')
    alpha = lo[m][0]
    require(alpha == Q(1, 32), 'Coupling is not frozen alpha=1/32')
    delta = y[m] - alpha
    require(delta in (Q(0), Q(1, 2 ** 15), Q(1, 2 ** 14), Q(1, 2 ** 13), Q(1, 2 ** 12)), 'Residual is not one of the frozen levels')
    columns = set()
    for i in range(n):
        el, eh = ([ZERO] * p, [ZERO] * p)
        if i < m:
            el[0], eh[0], ey = (Q(3, 4), Q(5, 4), Q(2))
        else:
            j = i - m
            el[0] = eh[0] = alpha
            el[j + 1], eh[j + 1] = (Q(root), Q(2 * root))
            ey = alpha + (1 if j < 32 else -1) * delta
        require(lo[i] == el and hi[i] == eh and (y[i] == ey), 'Original interval/response mismatch at row ' + str(i))
        changing = [j for j in range(p) if lo[i][j] != hi[i][j]]
        require(len(changing) == 1, 'Each row must have one independent cell')
        columns.update(changing)
    require(columns == set(range(p)), 'Every column must remain uncertain')
    center = [[(a + b) / 2 for a, b in zip(lrow, hrow)] for lrow, hrow in zip(lo, hi)]
    widths = [[(b - a) / 2 for a, b in zip(lrow, hrow)] for lrow, hrow in zip(lo, hi)]
    gradient = [ridge * b for b in beta]
    for row, response in zip(center, y):
        residual = dot(row, beta) - response
        for j, value in enumerate(row):
            if value:
                gradient[j] += value * residual / n
    expected_gradient = [ZERO] + [-(1 if j < 32 else -1) * 3 * delta / 32 for j in range(m)]
    require(gradient == expected_gradient, 'Perturbed midpoint gradient failed')
    require(bool(any(gradient)) == bool(delta), 'Midpoint-fit status failed')
    determinant_lower = Q(3, 4) * Q(root) ** m
    require(determinant_lower > 0, 'Uniform full-rank minor failed')
    return dict(lo=lo, hi=hi, y=y, beta=beta, ridge=ridge, n=n, p=p, m=m, root=root, alpha=alpha, delta=delta, center=center, widths=widths, gradient=gradient, determinant_lower=determinant_lower)

def verify_structure_and_analytic(data, certificate, wrapper, result):
    n, p, m = (data['n'], data['p'], data['m'])
    alpha, delta, ridge = (data['alpha'], data['delta'], data['ridge'])
    rec = certificate['structural_receipt']
    require(certificate['method'] == 'nonzero_residual_energy' and certificate['basis_kind'] == 'fixed_e0', 'Fixed-direction method failed')
    require(Q(certificate['rho']) == 0 and Q(certificate['gamma']) == 1 and (Q(certificate['anchor_error']) == 0), 'Actual-metric objective failed')
    require(certificate['search_values_include_additive_tail'] is True and result['search_values_include_additive_tail'] is True, 'Point/tail convention failed')
    transform, retained = (qmatrix(certificate['T']), qmatrix(certificate['R']))
    require(len(transform) == len(retained) == p and all((len(row) == p for row in transform)) and all((len(row) == 1 for row in retained)), 'Actual T/R dimensions failed')
    scale = transform[0][0]
    require(scale > 0 and all((transform[i][j] == scale * int(i == j) for i in range(p) for j in range(p))), 'Actual scalar T failed')
    require(retained == [[Q(1)]] + [[ZERO] for _ in range(m)], 'Actual fixed Q=e0 failed')
    direction = [scale] + [ZERO] * m
    mu, nu = (Q(certificate['mu']), Q(certificate['nu']))
    require(0 < mu <= ridge * scale * scale <= nu, 'Ridge metric enclosure failed')
    require(qvector(certificate['theta_bar']) == [ZERO] * p and qvector(certificate['theta_c']) == [ZERO] * p and (Q(certificate['C_safe']) == ridge), 'Ridge-only anchor failed')
    low = high = ridge
    for i in range(m):
        c, s = (data['center'][i][0], data['widths'][i][0])
        lower, upper = scalar_quadratic_range(s * s, 2 * s * (c - 1), c * c - 2 * c)
        low += lower / n
        high += upper / n
    require(low == 0 and high == Q(1, 32), 'Uniform signal gradient failed')
    G = max(abs(low), abs(high))
    kappa = ridge + sum((data['lo'][i][0] ** 2 / n for i in range(m)), ZERO)
    a2 = sum((data['lo'][i][0] ** 2 / n for i in range(m, n)), ZERO)
    epsilon2 = sum(((dot(row, data['beta']) - response) ** 2 / n for row, response in zip(data['center'][m:], data['y'][m:])), ZERO)
    require(kappa == Q(25, 32) and a2 == alpha ** 2 / 2 and (epsilon2 == delta ** 2 / 2), 'Original-data structural constants failed')
    ae = alpha * delta / 2
    require(ae ** 2 == a2 * epsilon2, 'Exact a times epsilon identity failed')
    radius_original = (G + ae) / kappa
    eta = (delta + alpha * radius_original) ** 2 / 2
    require(radius_original == (1 + 16 * alpha * delta) / 25, 'Original radius identity failed')
    expected = {'m': m, 'n': n, 'p': p, 'alpha': alpha, 'delta': delta, 'lambda': ridge, 'G': G, 'kappa': kappa, 'a_squared': a2, 'epsilon_squared': epsilon2, 'a_times_epsilon': ae, 'original_projection_radius': radius_original, 'tail_error': eta, 'scalar_T': scale, 'basis_sign': Q(1)}
    for key, value in expected.items():
        require(Q(rec[key]) == value, 'Structural receipt mismatch: ' + key)
    require(qvector(rec['midpoint_gradient']) == data['gradient'], 'Saved midpoint gradient failed')
    require(rec['candidate_is_perturbed_midpoint_fit'] == (delta == 0) and rec['automatic_direction_recovery_claimed'] is False, 'Candidate/direction disclosure failed')
    require(rec['retained_direction_source'] == 'fixed_original_e0_from_declared_mechanism', 'Fixed direction source failed')
    require(Q(certificate['tail_error']) == eta and Q(result['tail_error']) >= eta and (Q(result['anchor_error']) == 0), 'Tail reporting failed')
    require(rec['signal_rows'] == [0, m] and rec['nuisance_rows'] == [m, n] and (rec['row_ranges_are_half_open'] is True), 'Row partition failed')
    exact_coordinate = radius_original / scale
    units = exact_coordinate * 2 ** 20
    rounded = Q((units.numerator + units.denominator - 1) // units.denominator, 2 ** 20)
    require(Q(rec['exact_coordinate_radius']) == exact_coordinate and Q(rec['rounded_coordinate_radius']) == rounded and (Q(certificate['coordinate_radius']) == rounded), 'Exact/outward rounded root failed')
    require(Q(rec['actual_original_root_radius']) == scale * rounded >= radius_original, 'Actual physical root failed')
    original_cap = Q(certificate['original_gradient_upper'])
    require(original_cap >= scale ** 2 * (G * G + delta * delta) / mu, 'Original transformed-gradient cap failed')
    require(Q(certificate['initial_upper']) >= original_cap + eta, 'Initial cap failed')
    require(Q(certificate['B0_squared']) == 2 + (alpha * alpha + delta * delta) / 2, 'Original anchor residual energy failed')
    H = G * G / (kappa + alpha * alpha / 10) + 2 * delta * delta / 5
    V = G * G / (kappa + alpha * alpha / 2)
    require(H == Q(5) / (4000 + 512 * alpha * alpha) + 2 * delta * delta / 5 and V == Q(1) / (800 + 512 * alpha * alpha), 'Analytic endpoint identity failed')
    require(0 < H - V <= eta, 'Independent analytic reduction/tail failed')
    fractions = result['analytic']['fraction']
    analytic_expected = {'delta': delta, 'epsilon_squared': epsilon2, 'a_squared': a2, 'a_times_epsilon': ae, 'G': G, 'kappa': kappa, 'original_projection_radius': radius_original, 'tail_error': eta, 'exact_original_H': H, 'exact_retained_V': V, 'actual_reduction_gap': H - V, 'original_gradient_cap': Q(1, 512) + 2 * delta * delta}
    for key, value in analytic_expected.items():
        require(Q(fractions[key]) == value, 'Analytic field failed: ' + key)
    return dict(direction=direction, eta=eta, radius=rounded, original_cap=original_cap, H=H, V=V, epsilon_squared=epsilon2)

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

def main():
    import hashlib
    import psutil
    psutil.Process().cpu_affinity([4])
    payload = json.loads((BASE / 'epsilon_results.json').read_text(encoding='utf-8'))
    require(payload['status'] == 'completed', 'Queue is not complete')
    require(hashlib.sha256((BASE / 'epsilon_protocol.md').read_bytes()).hexdigest() == payload['protocol_sha256'], 'Frozen protocol changed during execution')
    reports = []
    for entry in payload['results']:
        report = {'condition': entry['condition'], 'delta_fraction': entry['delta_fraction']}
        try:
            require(entry['status'] == 'PASS', 'Queue condition was not PASS')
            wrapper = json.loads((BASE / entry['certificate_path']).read_text(encoding='utf-8'))
            result, certificate = (wrapper['result'], wrapper['result']['certificate'])
            data = read_input_conditions(certificate)
            constants = verify_structure_and_analytic(data, certificate, wrapper, result)
            tree = verify_points_nodes_witness(data, certificate, result, constants)
            report.update({'status': 'PASS', 'audit_type': 'independent_Fraction_original_data_and_complete_tree', 'imports_solver_implementation': False, 'epsilon_squared': str(constants['epsilon_squared']), 'tail_error': str(constants['eta']), 'exact_original_H': str(constants['H']), 'exact_retained_V': str(constants['V']), 'candidate_is_perturbed_midpoint_fit': data['delta'] == 0, 'full_rank_minor_uniform_lower': str(data['determinant_lower']), 'all_original_input_and_structural_conditions_verified': True, **tree})
        except Exception as error:
            report.update({'status': 'ERROR', 'error': str(error), 'traceback': traceback.format_exc()})
        reports.append(report)
    output = {'status': 'PASS' if all((x['status'] == 'PASS' for x in reports)) else 'ERROR', 'imports_solver_implementation': False, 'logical_cpu_affinity': [4], 'reports': reports}
    (BASE / 'epsilon_independent_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if output['status'] == 'PASS' else 1
if __name__ == '__main__':
    raise SystemExit(main())
