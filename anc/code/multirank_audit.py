from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Independent Fraction checks for multidimensional energy constants and two trees.\n\nImports no proposal, interval solver or energy-verifier implementation.\nUses only the saved original independent-row boxes and actual dyadic matrices.\n'
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import sys, json, itertools, time
B = Path(__file__).resolve().parent
from auto_energy_audit import dot, qvector, qmatrix, product, transpose, qrange, maxabs, interval_product, affine_range, require_sqrt_ceiling, original_loss, read_original_data, set_one_cpu

def inverse(a):
    n = len(a)
    z = [row.copy() + [F(i == j) for j in range(n)] for i, row in enumerate(a)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if z[i][j]))
        z[j], z[pivot] = (z[pivot], z[j])
        v = z[j][j]
        z[j] = [t / v for t in z[j]]
        for i in range(n):
            if i != j:
                v = z[i][j]
                z[i] = [t - v * u for t, u in zip(z[i], z[j])]
    return [row[n:] for row in z]

def quadratic(z, H):
    return sum((z[i] * H[i][j] * z[j] for i in range(len(z)) for j in range(len(z))), F(0))

def constants(data, cert):
    lo, hi, n, p, beta, ridge = (data['lo'], data['hi'], data['n'], data['p'], data['beta'], data['ridge'])
    T = qmatrix(cert['T'])
    R = qmatrix(cert['R'])
    rank = len(R[0])
    U = product(T, R)
    cols = transpose(U)
    H = product(transpose(U), U)
    Hi = inverse(H)
    c = T[0][0]
    assert c > 0 and all((T[i][j] == (c if i == j else 0) for i in range(p) for j in range(p)))
    assert ridge * c * c >= F(cert['mu']) > 0
    rows = []
    for row in data['rows']:
        center = row['center']
        k = row['index_changing']
        w = row['width']
        pc = [dot(center, col) for col in cols]
        pg = [w * col[k] for col in cols]
        ec, eg = (row['ec'], row['eg'])
        endpoints = []
        for sign in (-1, 1):
            x = center.copy()
            x[k] += sign * w
            proj = quadratic([a + sign * b for a, b in zip(pc, pg)], Hi)
            endpoints.append((proj, dot(x, x) - proj))
        assert min((v for pair in endpoints for v in pair)) >= 0
        rows.append(dict(**row, pc=pc, pg=pg, score=abs(ec) + abs(eg), retained=max((x for x, y in endpoints)), discarded=max((y for x, y in endpoints)), residual=(abs(ec) + abs(eg)) ** 2, gr=[qrange(pg[j] * eg, pc[j] * eg + pg[j] * ec, pc[j] * ec) for j in range(rank)]))
    threshold = max((r['score'] for r in rows)) / 2
    sig = [r for r in rows if r['score'] >= threshold]
    nui = [r for r in rows if r['score'] < threshold]
    f = {k: F(v) for k, v in cert['energy_receipt']['fractions'].items()}
    mean = lambda rs, key: sum((r[key] for r in rs), F(0)) / n
    expected = dict(a2=mean(nui, 'retained'), sigma2=mean(sig, 'retained'), epsilon2=mean(nui, 'residual'), E2=mean(sig, 'residual'), b2=mean(sig, 'discarded'))
    zb = [dot(beta, col) for col in cols]
    expected['d2'] = dot(beta, beta) - quadratic(zb, Hi)
    intervals = [(ridge * zb[j] + sum((r['gr'][j][0] / n for r in sig), F(0)), ridge * zb[j] + sum((r['gr'][j][1] / n for r in sig), F(0))) for j in range(rank)]
    expected['G2'] = max((quadratic(list(z), Hi) for z in itertools.product(*intervals)))
    assert all((f[k] == v for k, v in expected.items()))
    low = [[F(0) for _ in range(rank)] for _ in range(rank)]
    high = [[F(0) for _ in range(rank)] for _ in range(rank)]
    for r in sig:
        for j in range(rank):
            for k in range(rank):
                l, u = qrange(r['pg'][j] * r['pg'][k], r['pc'][j] * r['pg'][k] + r['pg'][j] * r['pc'][k], r['pc'][j] * r['pc'][k])
                low[j][k] += l / n
                high[j][k] += u / n
    assert low == qmatrix(cert['energy_receipt']['signal_moment_lower']) and high == qmatrix(cert['energy_receipt']['signal_moment_upper'])
    retained = f['retained_signal_curvature']
    assert retained >= 0
    assert retained == 0 or all((low[j][j] - retained * H[j][j] >= sum((max(abs(low[j][k] - retained * H[j][k]), abs(high[j][k] - retained * H[j][k])) for k in range(rank) if k != j)) for j in range(rank)))
    assert f['kappa'] == ridge + retained
    roots = []
    for key in ('a2', 'epsilon2', 'sigma2', 'E2', 'b2', 'd2', 'G2'):
        q = expected[key]
        unit = F(1, 2 ** 40)
        v = F(0) if q == 0 else None
        if v is None:
            from math import isqrt
            k = isqrt(q.numerator * 2 ** 80 // q.denominator)
            v = F(k, 2 ** 40)
            if v * v < q:
                v += unit
        require_sqrt_ceiling(v, q, key)
        roots.append(v)
    a, epsilon, sigma, E, b, d, G = roots

    def upper_root(q):
        from math import isqrt
        k = isqrt(q.numerator * 2 ** 80 // q.denominator)
        v = F(k, 2 ** 40)
        return v if v * v >= q else v + F(1, 2 ** 40)
    omega = upper_root(b * b / (ridge + b * b))
    sl = upper_root(ridge)
    kb = ridge * (f['kappa'] + b * b) / (ridge + b * b)
    radius = (G + a * epsilon + omega * (a * E + sigma * epsilon) / 2 + omega ** 2 * sigma * E + (a * sl + sigma * b) * d) / kb
    eta = (epsilon + a * radius + omega * (E + sigma * radius) + sl * d) ** 2
    assert kb == f['kappa_b'] and radius == f['radius'] and (eta == f['eta'] == F(cert['tail_error']))
    receipt = cert['energy_receipt']
    terms = dict(residual=epsilon, retained_coupling=a * radius, signal_leakage=omega * (E + sigma * radius), candidate_leakage=sl * d)
    assert receipt['components'] == {k: float(v) for k, v in terms.items()} and receipt['dominant_component'] == max(terms, key=terms.get)
    assert receipt['curvature_ratio'] == float(kb / f['kappa']) and receipt['tail_target_met'] == (eta <= F(1, 100000)) and (receipt['tail_target'] == 1e-05)
    hmu = min((H[j][j] - sum((abs(H[j][k]) for k in range(rank) if k != j)) for j in range(rank)))
    root = F(cert['coordinate_radius'])
    assert hmu > 0 and hmu * root * root >= radius * radius
    gl = [ridge * v for v in beta]
    gu = gl.copy()
    for r in rows:
        for j in range(p):
            gen = r['width'] if j == r['index_changing'] else F(0)
            l, u = qrange(gen * r['eg'], r['center'][j] * r['eg'] + gen * r['ec'], r['center'][j] * r['ec'])
            gl[j] += l / n
            gu[j] += u / n
    cap = c * c / F(cert['mu']) * sum((max(abs(l), abs(u)) ** 2 for l, u in zip(gl, gu)), F(0))
    assert F(cert['original_gradient_upper']) >= cap
    assert F(cert['initial_upper']) >= cap + eta
    return dict(rows=rows, U=U, H=H, zb=zb, eta=eta, root=root, cap=cap, rank=rank, retained_curvature=retained)

def tree(data, cert, result, C):
    rows, H, zb, eta, root, rank = (C['rows'], C['H'], C['zb'], C['eta'], C['root'], C['rank'])
    ridge = data['ridge']
    n = data['n']

    def point(z):
        value = 2 * ridge * dot(zb, z) - ridge * quadratic(z, H)
        for r in rows:
            ec, eg = (r['ec'], r['eg'])
            cc = ec - dot(r['pc'], z)
            cg = eg - dot(r['pg'], z)
            value += qrange(eg * eg - cg * cg, 2 * (ec * eg - cc * cg), ec * ec - cc * cc)[1] / n
        return value + eta
    points = {}
    for q in cert['points']:
        z = tuple(qvector(q['z']))
        assert len(z) == rank and all((abs(v) <= root for v in z))
        v = point(z)
        assert F(q['lower']) <= v <= F(q['upper'])
        assert z not in points
        points[z] = q
    best = max((F(q['lower']) for q in points.values()))
    assert F(result['search_lower']) == best
    hdiag = qvector(cert['hdiag'])
    for j in range(rank):
        assert hdiag[j] >= ridge * H[j][j] + sum((maxabs(affine_range(r['pc'][j], r['pg'][j])) ** 2 / n for r in rows), F(0))
    nodes = cert['nodes']
    incoming = Counter()
    terminal = []
    for i, node in enumerate(nodes):
        assert node['id'] == i
        lo = qvector(node['lo'])
        hi = qvector(node['hi'])
        center = tuple(((l + h) / 2 for l, h in zip(lo, hi)))
        rad = [(h - l) / 2 for l, h in zip(lo, hi)]
        assert center in points
        if i == 0:
            assert node['parent'] is None and lo == [-root] * rank and (hi == [root] * rank)
            cap = F(cert['initial_upper'])
        else:
            assert 0 <= node['parent'] < i
            cap = F(nodes[node['parent']]['upper'])
        grad = []
        for j in range(rank):
            gl = gu = 2 * ridge * (zb[j] - dot(H[j], center))
            for r in rows:
                l, u = interval_product(affine_range(r['ec'] - dot(r['pc'], center), r['eg'] - dot(r['pg'], center)), affine_range(r['pc'][j], r['pg'][j]))
                gl += 2 * l / n
                gu += 2 * u / n
            grad.append(max(abs(gl), abs(gu)))
        candidates = [cap, F(points[center]['upper']) + dot(grad, rad)]
        corners = list(itertools.product(*zip(lo, hi)))
        assert all((z in points for z in corners))
        candidates.append(max((F(points[z]['upper']) for z in corners)) + sum((h * r * r for h, r in zip(hdiag, rad)), F(0)))
        assert min(candidates) <= F(node['upper']) <= cap
        children = node['children']
        assert len(children) in (0, 2)
        if not children:
            terminal.append(F(node['upper']))
            continue
        l, r = [nodes[k] for k in children]
        assert l['parent'] == r['parent'] == i
        ll, lh, rl, rh = [qvector(x) for x in (l['lo'], l['hi'], r['lo'], r['hi'])]
        split = [j for j in range(rank) if lh[j] != hi[j]]
        assert len(split) == 1
        k = split[0]
        assert ll == lo and rh == hi and (lh[k] == rl[k] == center[k])
        assert all((lh[j] == hi[j] and rl[j] == lo[j] for j in range(rank) if j != k))
        incoming.update(children)
    assert all((incoming[i] == 1 for i in range(1, len(nodes)))) and len(nodes) == 2 * result['nodes'] + 1
    assert F(result['search_upper']) >= max(max(terminal), best)
    w = cert['witness']
    x = qmatrix(w['x'])
    theta = qvector(w['theta'])
    assert all((l <= v <= h for lr, xr, hr in zip(data['lo'], x, data['hi']) for l, v, h in zip(lr, xr, hr)))
    regret = original_loss(x, data['y'], data['beta'], ridge) - original_loss(x, data['y'], theta, ridge)
    assert F(w['regret_lower']) <= regret
    lower = max(F(0), F(w['regret_lower']), best - eta)
    upper = min(C['cap'], max(terminal))
    assert F(result['lower']) <= lower and F(result['upper']) >= upper
    return dict(points=len(points), nodes=len(nodes), terminal_leaves=len(terminal), exact_original_row_values=True, node_bounds_and_binary_coverage=True, original_table_witness=True)

def main():
    set_one_cpu()
    start = time.monotonic()
    p = json.loads((B / 'multirank_results.json').read_text())
    assert p['status'] == 'PASS'
    reports = []
    for condition in p['conditions']:
        cid = condition['condition']
        result = json.loads((B / f'multirank_{cid}_energy_certificate.json').read_text())
        cert = result['certificate']
        data = read_original_data(cert)
        C = constants(data, cert)
        row = dict(condition=cid, rank=C['rank'], eta=str(C['eta']), original_constants='PASS')
        if cid in (0, 9):
            row['tree'] = tree(data, cert, result, C)
        reports.append(row)
        print(cid, 'PASS', flush=True)
    out = dict(status='PASS', conditions=len(reports), complete_trees=2, imports_solver_implementation=False, seconds=time.monotonic() - start, reports=reports)
    (B / 'multirank_audit_results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != 'reports'}))
if __name__ == '__main__':
    main()
