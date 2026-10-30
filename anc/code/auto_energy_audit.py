'Independent Fraction audit of the six v80 automatic energy certificates.\n\nNo optimization, proposal, interval, or certificate implementation is imported.\nThe proof audit accepts an arbitrary saved rank-one direction and dyadic T;\nfrozen-family provenance is a separate check.  It assumes no analytic full H.\n'
from __future__ import annotations
from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
import argparse
from collections import Counter
import ctypes
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
BASE = Path(__file__).resolve().parent
ZERO = F(0)
ONE = F(1)
GRID = [(ZERO, ZERO, ZERO), (F(1, 2 ** 18), ZERO, ZERO), (ZERO, F(1, 2 ** 18), ZERO), (F(1, 2 ** 18), F(1, 2 ** 18), ZERO), (F(1, 2 ** 16), F(1, 2 ** 16), ZERO), (F(1, 2 ** 18), F(1, 2 ** 18), F(1, 2 ** 18))]

class AuditFailure(AssertionError):
    pass

def require(condition, message):
    if not condition:
        raise AuditFailure(message)

def qvector(values):
    return [F(value) for value in values]

def qmatrix(values):
    return [qvector(row) for row in values]

def dot(left, right):
    require(len(left) == len(right), 'Dot-product dimension mismatch')
    return sum((a * b for a, b in zip(left, right)), ZERO)

def transpose(matrix):
    return [list(column) for column in zip(*matrix)]

def product(left, right):
    columns = transpose(right)
    return [[dot(row, col) for col in columns] for row in left]

def qrange(a, b, c):
    """Extrema of a*t^2+b*t+c on the original row parameter [-1,1]."""
    values = [a - b + c, a + b + c]
    if a:
        stationary = -b / (2 * a)
        if -1 <= stationary <= 1:
            values.append(a * stationary ** 2 + b * stationary + c)
    return (min(values), max(values))

def maxabs(interval):
    return max(abs(interval[0]), abs(interval[1]))

def minabs(interval):
    lo, hi = interval
    return ZERO if lo <= 0 <= hi else min(abs(lo), abs(hi))

def interval_product(left, right):
    values = [a * b for a in left for b in right]
    return (min(values), max(values))

def affine_range(center, generator):
    return (center - abs(generator), center + abs(generator))

def require_sqrt_ceiling(saved, squared, name, bits=40):
    """Check the certificate's smallest dyadic upper root without computing sqrt."""
    unit = F(1, 2 ** bits)
    require(squared >= 0 and saved >= 0, name + ': negative norm')
    require((saved / unit).denominator == 1, name + ': wrong dyadic grid')
    require(saved * saved >= squared, name + ': downward square root')
    require(saved == 0 if squared == 0 else (saved - unit) ** 2 < squared, name + ': not the smallest dyadic upper root')

def psd_ldl(matrix, name):
    """Exact symmetric LDL proof, including the zero-pivot PSD case."""
    p = len(matrix)
    require(all((len(row) == p for row in matrix)), name + ': nonsquare')
    require(matrix == transpose(matrix), name + ': nonsymmetric')
    lower = [[ZERO] * p for _ in range(p)]
    pivots = []
    for j in range(p):
        lower[j][j] = ONE
        pivot = matrix[j][j] - sum((lower[j][k] ** 2 * pivots[k] for k in range(j)), ZERO)
        require(pivot >= 0, name + ': negative exact LDL pivot ' + str(j))
        pivots.append(pivot)
        for i in range(j + 1, p):
            remainder = matrix[i][j] - sum((lower[i][k] * lower[j][k] * pivots[k] for k in range(j)), ZERO)
            if pivot:
                lower[i][j] = remainder / pivot
            else:
                require(remainder == 0, name + ': nonzero off-diagonal at zero pivot')
    return pivots

def read_original_data(certificate):
    lo, hi = (qmatrix(certificate['lo']), qmatrix(certificate['hi']))
    beta, y = (qvector(certificate['beta']), qvector(certificate['y']))
    n, p, ridge = (len(y), len(beta), F(certificate['lambda']))
    require(n > 0 and p > 0 and (ridge > 0), 'Empty input or nonpositive ridge')
    require(len(lo) == len(hi) == n and all((len(row) == p for row in lo + hi)), 'Original input dimensions failed')
    require(all((l <= h for lrow, hrow in zip(lo, hi) for l, h in zip(lrow, hrow))), 'Invalid original cell interval')
    center = [[(l + h) / 2 for l, h in zip(lrow, hrow)] for lrow, hrow in zip(lo, hi)]
    widths = [[(h - l) / 2 for l, h in zip(lrow, hrow)] for lrow, hrow in zip(lo, hi)]
    rows = []
    fixed = []
    for i, (c, widths_i, response) in enumerate(zip(center, widths, y)):
        changing = [j for j, value in enumerate(widths_i) if value]
        require(len(changing) <= 1, 'More than one uncertain cell in row ' + str(i))
        k = changing[0] if changing else 0
        width = widths_i[k] if changing else ZERO
        if not changing:
            fixed.append(i)
        rows.append(dict(index=i, center=c, index_changing=k, width=width, ec=dot(c, beta) - response, eg=width * beta[k]))
    return dict(lo=lo, hi=hi, center=center, widths=widths, beta=beta, y=y, n=n, p=p, ridge=ridge, rows=rows, fixed_rows=fixed)

def verify_frozen_provenance(data, wrapper, entry):
    """Protocol labels are used here only; the proof calculation does not use them."""
    condition = entry['condition']
    require(isinstance(condition, int) and 0 <= condition < len(GRID), 'Unknown condition')
    delta, ell, gamma = GRID[condition]
    for field, expected in zip(('delta', 'signal_leak', 'candidate_extra'), (delta, ell, gamma)):
        require(F(entry[field]) == F(wrapper[field]) == expected, 'Frozen metadata mismatch: ' + field)
    require(wrapper['condition'] == condition, 'Wrapper condition mismatch')
    n, p, m, alpha = (data['n'], data['p'], 16, F(1, 32))
    require((n, p, data['ridge']) == (32, 17, F(1, 2)), 'Frozen dimensions/ridge failed')
    require(data['beta'] == [ONE, gamma] + [ZERO] * 15, 'Frozen candidate failed')
    for i in range(n):
        lower, upper = ([ZERO] * p, [ZERO] * p)
        if i < m:
            lower[0], upper[0] = (F(3, 4), F(5, 4))
            lower[1] = upper[1] = ell
            response = F(2)
        else:
            lower[0] = upper[0] = alpha
            lower[i - m + 1], upper[i - m + 1] = (F(4), F(8))
            response = alpha + (delta if i < m + 8 else -delta)
        require(data['lo'][i] == lower and data['hi'][i] == upper and (data['y'][i] == response), 'Frozen original row mismatch: ' + str(i))
    return dict(condition=condition, delta=str(delta), signal_leak=str(ell), candidate_extra=str(gamma), original_protocol_rows_verified=n)

def verify_generic_constants(data, certificate, result):
    n, p, ridge, beta = (data['n'], data['p'], data['ridge'], data['beta'])
    receipt = certificate['structural_receipt']
    require(receipt == result['structural_receipt'], 'Duplicate structural receipts disagree')
    require(receipt['status'] == 'PASS' and receipt['verifier'] == 'generic_rank1_original_row_leakage_energy', 'Wrong structural verifier')
    require(receipt['one_uncertain_cell_per_row_verified'] is True, 'Row-scope disclosure failed')
    require(receipt['automatic_direction'] is True and receipt['fixed_direction_override'] is False, 'Direction provenance flags failed')
    require(receipt['partition_rule'] == 'score=max_box_abs_actual_candidate_residual; signal iff score>=half_max_score', 'Partition rule differs from protocol')
    require(certificate['method'] == result['method'] == 'automatic_leakage_energy' and certificate['basis_kind'] == result['basis_kind'] == 'automatic_verified_leakage_energy', 'Wrong method/basis kind')
    require(result['rank'] == 1 and result['n'] == n and (result['features'] == p), 'Reported dimensions failed')
    require(F(certificate['rho']) == 0 and F(certificate['gamma']) == 1 and (F(certificate['anchor_error']) == F(result['anchor_error']) == 0), 'Actual-metric objective convention failed')
    require(certificate['search_values_include_additive_tail'] is True and result['search_values_include_additive_tail'] is True, 'Tail convention failed')
    transform, retained = (qmatrix(certificate['T']), qmatrix(certificate['R']))
    require(len(transform) == len(retained) == p and all((len(row) == p for row in transform)) and all((len(row) == 1 for row in retained)), 'T/R dimensions failed')
    numeric = [row[0] for row in retained]
    direction = [dot(row, numeric) for row in transform]
    h = dot(direction, direction)
    require(h > 0, 'Zero physical retained direction')
    require(qvector(receipt['numerical_basis']) == numeric and qvector(receipt['physical_direction_r']) == direction, 'Actual physical direction differs from receipt')
    rows = []
    for row in data['rows']:
        c, k, width = (row['center'], row['index_changing'], row['width'])
        lc, lg = (dot(c, direction), width * direction[k])
        erange, lrange = (affine_range(row['ec'], row['eg']), affine_range(lc, lg))
        discarded = []
        for sign in (-1, 1):
            endpoint = c.copy()
            endpoint[k] += sign * width
            q = dot(endpoint, endpoint) - dot(endpoint, direction) ** 2 / h
            require(q >= 0, 'Negative discarded projection norm')
            discarded.append(q)
        rows.append(dict(**row, lc=lc, lg=lg, erange=erange, lrange=lrange, score=maxabs(erange), discarded_squared=max(discarded)))
    threshold = max((row['score'] for row in rows)) / 2
    signal = [row for row in rows if row['score'] >= threshold]
    nuisance = [row for row in rows if row['score'] < threshold]
    require(receipt['signal_rows'] == [row['index'] for row in signal] and receipt['nuisance_rows'] == [row['index'] for row in nuisance], 'Automatic residual-score partition failed')
    require(qvector(receipt['row_scores']) == [row['score'] for row in rows], 'Actual candidate row scores failed')
    lnorm = lambda group: sum((maxabs(row['lrange']) ** 2 / (n * h) for row in group), ZERO)
    enorm = lambda group: sum((maxabs(row['erange']) ** 2 / n for row in group), ZERO)
    raw = dict(h=h, a_squared=lnorm(nuisance), epsilon_squared=enorm(nuisance), sigma_squared=lnorm(signal), E_squared=enorm(signal), b_squared=sum((row['discarded_squared'] / n for row in signal), ZERO), d_squared=dot(beta, beta) - dot(beta, direction) ** 2 / h, kappa=ridge + sum((minabs(row['lrange']) ** 2 / (n * h) for row in signal), ZERO), partition_threshold=threshold)
    gl = gu = ridge * dot(beta, direction)
    for row in signal:
        low, high = qrange(row['lg'] * row['eg'], row['lc'] * row['eg'] + row['lg'] * row['ec'], row['lc'] * row['ec'])
        gl += low / n
        gu += high / n
    raw.update(G_squared=max(abs(gl), abs(gu)) ** 2 / h, signal_gradient_r_lower=gl, signal_gradient_r_upper=gu)
    saved = {key: F(value) for key, value in receipt['fractions'].items()}
    for name, value in raw.items():
        require(saved[name] == value, 'Exact original-row constant mismatch: ' + name)
    require(raw['d_squared'] >= 0 and raw['kappa'] >= ridge, 'Projection/curvature sign failed')
    require(F(receipt['candidate_component_parallel_squared']) == dot(beta, direction) ** 2 / h, 'Candidate parallel component failed')
    for norm in ('a', 'epsilon', 'sigma', 'E', 'b', 'd', 'G'):
        require_sqrt_ceiling(saved[norm], raw[norm + '_squared'], norm)
    require_sqrt_ceiling(saved['sqrt_lambda'], ridge, 'sqrt_lambda')
    b = saved['b']
    require_sqrt_ceiling(saved['omega'], b * b / (ridge + b * b), 'omega')
    kb = ridge * (saved['kappa'] + b * b) / (ridge + b * b)
    require(saved['kappa_b'] == kb > 0, 'Positive Schur curvature bound failed')
    a, eps, sigma, E, d, G, omega, sl = (saved[k] for k in ('a', 'epsilon', 'sigma', 'E', 'd', 'G', 'omega', 'sqrt_lambda'))
    physical_radius = (G + a * eps + omega * (a * E + sigma * eps) / 2 + omega ** 2 * sigma * E + (a * sl + sigma * b) * d) / kb
    eta = (eps + a * physical_radius + omega * (E + sigma * physical_radius) + sl * d) ** 2
    require(saved['physical_radius'] == physical_radius and saved['eta'] == eta, 'Leakage-energy radius/tail failed')
    root = saved['coordinate_radius']
    unit = F(1, 2 ** 20)
    require(root >= 0 and (root / unit).denominator == 1 and (h * root * root >= physical_radius ** 2), 'Actual coordinate root does not contain the physical root')
    require(root == 0 if physical_radius == 0 else h * (root - unit) ** 2 < physical_radius ** 2, 'Coordinate root not the minimal outer dyadic grid point')
    require(F(certificate['coordinate_radius']) == root, 'Saved float root changed the exact grid point')
    require(F(certificate['tail_error']) == eta and F(result['tail_error']) >= eta, 'Tail reporting failed')
    anchor = [[ridge * int(i == j) for j in range(p)] for i in range(p)]
    for i in data['fixed_rows']:
        c = data['center'][i]
        for j in range(p):
            for k in range(p):
                anchor[j][k] += c[j] * c[k] / n
    metric = product(transpose(transform), product(anchor, transform))
    mu, nu = (F(certificate['mu']), F(certificate['nu']))
    require(0 < mu <= nu, 'Nonpositive transformed anchor curvature')
    lower_pivots = psd_ldl([[metric[i][j] - mu * int(i == j) for j in range(p)] for i in range(p)], 'T^T G0 T-mu I')
    upper_pivots = psd_ldl([[nu * int(i == j) - metric[i][j] for j in range(p)] for i in range(p)], 'nu I-T^T G0 T')
    gradient_ranges = []
    for column in transpose(transform):
        low = high = ridge * dot(column, beta)
        for row in rows:
            lc, lg = (dot(row['center'], column), row['width'] * column[row['index_changing']])
            ll, hh = qrange(lg * row['eg'], lc * row['eg'] + lg * row['ec'], lc * row['ec'])
            low += ll / n
            high += hh / n
        gradient_ranges.append((low, high))
    exact_cap = sum((maxabs(interval) ** 2 for interval in gradient_ranges), ZERO) / mu
    original_cap = F(certificate['original_gradient_upper'])
    require(original_cap >= exact_cap, 'Original global gradient cap failed')
    require(F(certificate['initial_upper']) >= original_cap + eta, 'Initial search cap failed')
    return dict(direction=direction, rows=rows, eta=eta, radius=root, physical_radius=physical_radius, original_cap=original_cap, exact_gradient_cap=exact_cap, gradient_ranges=gradient_ranges, kappa_b=kb, metric_lower_pivots=lower_pivots, metric_upper_pivots=upper_pivots, signal_count=len(signal), nuisance_count=len(nuisance), constants=saved, exact_anchor_penalty=dot(direction, [dot(row, direction) for row in anchor]), direction_is_axis_e0=all((value == 0 for value in direction[1:])))

def original_loss(table, y, theta, ridge):
    return sum(((dot(row, theta) - response) ** 2 for row, response in zip(table, y)), ZERO) / len(y) + ridge * dot(theta, theta)

def verify_tree(data, certificate, result, constants):
    n, p, ridge, beta = (data['n'], data['p'], data['ridge'], data['beta'])
    direction, eta, root = (constants['direction'], constants['eta'], constants['radius'])
    rows = constants['rows']
    linear_ridge, curvature_ridge = (ridge * dot(beta, direction), ridge * dot(direction, direction))

    def exact_point(z):
        value = 2 * linear_ridge * z - curvature_ridge * z * z
        for row in rows:
            ec, eg, lc, lg = (row[k] for k in ('ec', 'eg', 'lc', 'lg'))
            cc, cg = (ec - z * lc, eg - z * lg)
            value += qrange(eg * eg - cg * cg, 2 * (ec * eg - cc * cg), ec * ec - cc * cc)[1] / n
        return value + eta
    points = {}
    for i, point in enumerate(certificate['points']):
        z = qvector(point['z'])
        require(len(z) == 1 and -root <= z[0] <= root, 'Saved point rank/root failed')
        z = z[0]
        require(z not in points, 'Duplicate cached coordinate')
        value = exact_point(z)
        require(F(point['lower']) <= value <= F(point['upper']), 'Original point enclosure failed: ' + str(i))
        points[z] = point
    require(points and result['points'] == len(points), 'Point count mismatch')
    best_search = max((F(point['lower']) for point in points.values()))
    require(F(result['search_lower']) == best_search, 'Saved best search lower failed')
    hdiag = qvector(certificate['hdiag'])
    exact_curvature = curvature_ridge + sum((maxabs(row['lrange']) ** 2 / n for row in rows), ZERO)
    require(len(hdiag) == 1 and hdiag[0] >= exact_curvature > 0, 'Original curvature diagonal failed')
    mode = certificate['search_bound']
    require(mode in ('adaptive', 'center'), 'Unknown node-bound mode')
    nodes, incoming, terminal, bound_counts = (certificate['nodes'], Counter(), [], Counter())
    require(nodes, 'Empty tree')
    for index, node in enumerate(nodes):
        require(node['id'] == index, 'Noncanonical node ID')
        lower, upper = (qvector(node['lo']), qvector(node['hi']))
        require(len(lower) == len(upper) == 1 and lower[0] < upper[0], 'Node rank/width failed')
        lower, upper = (lower[0], upper[0])
        center, radius = ((lower + upper) / 2, (upper - lower) / 2)
        require(center in points, 'Node center missing from point cache')
        if index == 0:
            require(node['parent'] is None and lower == -root and (upper == root), 'Root mismatch')
            parent_cap = F(certificate['initial_upper'])
        else:
            parent = node['parent']
            require(isinstance(parent, int) and 0 <= parent < index, 'Invalid/cyclic parent')
            parent_cap = F(nodes[parent]['upper'])
        gl = gu = 2 * linear_ridge - 2 * curvature_ridge * center
        for row in rows:
            ec, eg, lc, lg = (row[k] for k in ('ec', 'eg', 'lc', 'lg'))
            ll, hh = interval_product(affine_range(ec - center * lc, eg - center * lg), row['lrange'])
            gl += 2 * ll / n
            gu += 2 * hh / n
        valid = dict(parent_cap=parent_cap, center_gradient=F(points[center]['upper']) + max(abs(gl), abs(gu)) * radius)
        if mode != 'center':
            require(lower in points and upper in points, 'Missing interpolation endpoints')
            valid['corner_interpolation'] = max(F(points[lower]['upper']), F(points[upper]['upper'])) + hdiag[0] * radius ** 2
        saved_upper, least = (F(node['upper']), min(valid.values()))
        require(least <= saved_upper <= parent_cap, 'No justified node upper or cap violation: ' + str(index))
        for name, value in valid.items():
            if value == least:
                bound_counts[name] += 1
        children = node['children']
        require(isinstance(children, list) and len(children) in (0, 2), 'Nonbinary node')
        if not children:
            terminal.append(saved_upper)
            continue
        left_id, right_id = children
        require(isinstance(left_id, int) and isinstance(right_id, int) and (index < left_id < right_id < len(nodes)), 'Invalid child IDs')
        left, right = (nodes[left_id], nodes[right_id])
        require(left['parent'] == right['parent'] == index, 'Child/parent mismatch')
        require(qvector(left['lo']) == [lower] and qvector(left['hi']) == [center] and (qvector(right['lo']) == [center]) and (qvector(right['hi']) == [upper]), 'Children do not exactly partition parent interval')
        incoming[left_id] += 1
        incoming[right_id] += 1
    require(incoming.get(0, 0) == 0 and all((incoming[i] == 1 for i in range(1, len(nodes)))), 'Orphan/multiply covered node')
    require(terminal and len(nodes) == 2 * result['nodes'] + 1, 'Expanded node count mismatch')
    terminal_upper, search_upper = (max(terminal), F(result['search_upper']))
    require(search_upper >= max(terminal_upper, best_search), 'Search upper misses a terminal leaf')
    require(F(result['search_error']) >= search_upper - best_search, 'Search error rounded down')
    witness = certificate['witness']
    require(isinstance(witness, dict), 'No original full-table witness')
    table, theta = (qmatrix(witness['x']), qvector(witness['theta']))
    require(len(table) == n and len(theta) == p and all((len(row) == p for row in table)), 'Witness dimensions failed')
    for i, (lo, row, hi) in enumerate(zip(data['lo'], table, data['hi'])):
        require(all((l <= x <= h for l, x, h in zip(lo, row, hi))), 'Infeasible original witness row ' + str(i))
    require(len(witness['z']) == 1 and F(witness['z'][0]) in points, 'Witness point reference failed')
    regret = original_loss(table, data['y'], beta, ridge) - original_loss(table, data['y'], theta, ridge)
    witness_lower = F(witness['regret_lower'])
    require(witness_lower <= regret, 'Original witness lower exceeds exact ridge loss difference')
    feasible_lower = max(ZERO, witness_lower, best_search - eta)
    proof_upper = min(constants['original_cap'], terminal_upper)
    require(F(result['lower']) <= feasible_lower and F(result['upper']) >= proof_upper, 'Final original interval lacks proof')
    require(0 <= F(result['lower']) <= F(result['upper']), 'Invalid final original interval')
    require(F(result['absolute_gap']) >= F(result['upper']) - F(result['lower']), 'Reported width rounded down')
    return dict(point_count=len(points), node_count=len(nodes), terminal_count=len(terminal), expanded_nodes=result['nodes'], mode=mode, bound_minimizers=dict(bound_counts), exact_curvature_upper=str(exact_curvature), exact_terminal_upper=str(terminal_upper), independent_original_upper=str(proof_upper), exact_original_witness_regret=str(regret), feasible_lower_justification=str(feasible_lower), exact_original_row_point_values_verified=True, every_node_upper_verified=True, complete_binary_coverage_verified=True, original_full_witness_verified=True)

def set_one_cpu():
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        process = kernel.GetCurrentProcess()
        allowed, system = (ctypes.c_size_t(), ctypes.c_size_t())
        require(kernel.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system)), 'Cannot read affinity')
        candidates = [j for j in range(64) if allowed.value & 1 << j and j >= 4]
        require(candidates, 'No CPU outside reserved 0-3')
        cpu = 14 if 14 in candidates else max(candidates)
        require(kernel.SetProcessAffinityMask(process, ctypes.c_size_t(1 << cpu)), 'Cannot set one-CPU affinity')
        return [cpu]
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted((j for j in os.sched_getaffinity(0) if j >= 4))
        require(allowed, 'No CPU outside reserved 0-3')
        cpu = 14 if 14 in allowed else max(allowed)
        os.sched_setaffinity(0, {cpu})
        return [cpu]
    return 'single_thread_stdlib_no_affinity_api'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    base = args.base.resolve()
    affinity, started = (set_one_cpu(), time.monotonic())
    payload = json.loads((base / 'auto_energy_results.json').read_text(encoding='utf-8'))
    require(payload['status'] == 'PASS' and (not payload['errors']), 'Main queue not complete')
    protocol_hash = hashlib.sha256((base / 'auto_energy_protocol.md').read_bytes()).hexdigest()
    require(protocol_hash == payload['protocol_sha256'], 'Frozen protocol digest mismatch')
    entries = [entry for entry in payload['results'] if entry['method'] == 'energy']
    require(sorted((entry['condition'] for entry in entries)) == list(range(6)), 'Exactly six energy conditions required')
    conditions = {condition['condition']: condition for condition in payload['conditions']}
    reports = []
    for entry in entries:
        report = dict(condition=entry['condition'], certificate_file=entry['certificate_file'])
        try:
            require(entry['status'] == 'PASS', 'Energy run failed')
            wrapper = json.loads((base / entry['certificate_file']).read_text(encoding='utf-8'))
            result, certificate = (wrapper['result'], wrapper['result']['certificate'])
            require(result['status'] == 'PASS', 'Certificate status failed')
            data = read_original_data(certificate)
            provenance = verify_frozen_provenance(data, wrapper, entry)
            constants = verify_generic_constants(data, certificate, result)
            tree = verify_tree(data, certificate, result, constants)
            saved_summary = {key: value for key, value in result.items() if key != 'certificate'}
            require(entry['result'] == saved_summary, 'Summary/certificate result mismatch')
            condition = conditions[entry['condition']]
            require(condition['receipt'] == certificate['structural_receipt'], 'Condition receipt mismatch')
            require(qmatrix(condition['numerical_basis']) == qmatrix(certificate['R']), 'Condition direction mismatch')
            require(F(entry['actual_tail']) <= F(result['tail_error']) and float(constants['eta']) == entry['actual_tail'], 'Actual tail summary mismatch')
            require(entry['target_met'] == (F(result['absolute_gap']) <= F(1, 10 ** 6)), 'Final target flag failed')
            require(condition['tail_budget_met'] == (constants['eta'] <= F(1, 10 ** 6)), 'Tail-budget flag failed')
            report.update(**provenance, status='PASS', tail_error=str(constants['eta']), tail_error_float=float(constants['eta']), physical_radius=str(constants['physical_radius']), coordinate_radius=str(constants['radius']), kappa_b=str(constants['kappa_b']), exact_gradient_cap=str(constants['exact_gradient_cap']), transformed_gradient_ranges=[[str(lo), str(hi)] for lo, hi in constants['gradient_ranges']], signal_count=constants['signal_count'], nuisance_count=constants['nuisance_count'], candidate_discarded_squared=str(constants['constants']['d_squared']), nuisance_actual_residual_squared=str(constants['constants']['epsilon_squared']), actual_physical_direction=[str(v) for v in constants['direction']], direction_is_axis_e0=constants['direction_is_axis_e0'], exact_anchor_penalty=str(constants['exact_anchor_penalty']), original_row_constants_verified=True, dyadic_sqrt_minimality_verified=True, actual_span_and_root_verified=True, exact_anchor_metric_psd_verified=True, all_saved_result_fields_match_summary=True, **tree)
        except Exception as error:
            report.update(status='ERROR', error=str(error), traceback=traceback.format_exc())
        reports.append(report)
        print('condition', entry['condition'], report['status'], report.get('error', ''), flush=True)
    passed = [report for report in reports if report['status'] == 'PASS']
    output = dict(status='PASS' if len(passed) == 6 else 'ERROR', logical_cpu_affinity=affinity, numerical_threads=1, seconds=time.monotonic() - started, protocol_sha256=protocol_hash, imports_solver_implementation=False, full_analytic_H_assumed=False, family_labels_used_in_constant_or_tree_proof=False, generic_rank_one_direction_supported=True, proof_scope='independent cells; at most one uncertain original cell per row; fixed uniform weights 1/n', provenance_scope='six frozen input tables checked separately from generic proof', direction_provenance_scope='saved actual proposal audited; proposal-generation execution is established by runner source, not recreated here', empirical_direction_scope='all six actual physical directions span e0; no empirical off-axis recovery claim', total_point_count=sum((report['point_count'] for report in passed)), total_node_count=sum((report['node_count'] for report in passed)), total_terminal_count=sum((report['terminal_count'] for report in passed)), reports=reports)
    (base / 'auto_energy_audit_results.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in output.items() if key != 'reports'}, indent=2))
    return 0 if output['status'] == 'PASS' else 1
if __name__ == '__main__':
    raise SystemExit(main())
