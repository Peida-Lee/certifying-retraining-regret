from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Independent rational audit: imports no proposal, verifier or search code.'
from pathlib import Path
from fractions import Fraction as F
from math import comb, isqrt
from collections import Counter
import json, time, itertools, traceback
import numpy as np
import psutil
B = Path(__file__).resolve().parent

def vq(v):
    return list(map(F, v))

def mq(a):
    return [vq(row) for row in a]

def dot(x, y):
    return sum((a * b for a, b in zip(x, y)), F(0))

def tr(a):
    return list(map(list, zip(*a)))

def mm(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]

def inverse(a):
    n = len(a)
    z = [row.copy() + [F(i == j) for j in range(n)] for i, row in enumerate(a)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if z[i][j]))
        z[j], z[pivot] = (z[pivot], z[j])
        d = z[j][j]
        z[j] = [v / d for v in z[j]]
        for i in range(n):
            if i != j:
                d = z[i][j]
                z[i] = [v - d * u for v, u in zip(z[i], z[j])]
    return [row[n:] for row in z]

def quad(v, H):
    return dot(v, [dot(row, v) for row in H])

def qrange(a, b, c):
    vals = [a - b + c, a + b + c]
    if a and -1 < -b / (2 * a) < 1:
        vals.append(c - b * b / (4 * a))
    return (min(vals), max(vals))

def prod_range(a, ag, b, bg):
    k = len(ag)
    if not k:
        return (a * b, a * b)
    vals = []
    for free in range(k):
        others = [j for j in range(k) if j != free]
        for bits in itertools.product((-1, 1), repeat=k - 1):
            ca = a + sum((ag[j] * s for j, s in zip(others, bits)), F(0))
            cb = b + sum((bg[j] * s for j, s in zip(others, bits)), F(0))
            vals.extend(qrange(ag[free] * bg[free], ca * bg[free] + ag[free] * cb, ca * cb))
    return (min(vals), max(vals))

def arange(c, g):
    return (c - sum(map(abs, g), F(0)), c + sum(map(abs, g), F(0)))

def iprod(a, b):
    z = [x * y for x in a for y in b]
    return (min(z), max(z))

def sqrtcheck(v, q):
    unit = F(1, 2 ** 40)
    assert v >= 0 and v * v >= q and (v == 0 or (v - unit) ** 2 < q)

def root(q):
    n = isqrt(q.numerator * 2 ** 80 // q.denominator)
    z = F(n, 2 ** 40)
    return z if z * z >= q else z + F(1, 2 ** 40)

def read(cert):
    lo = mq(cert['lo'])
    hi = mq(cert['hi'])
    y = vq(cert['y'])
    beta = vq(cert['beta'])
    n = len(y)
    p = len(beta)
    ridge = F(cert.get('lambda_value', cert.get('lambda')))
    centers = [[(l + h) / 2 for l, h in zip(a, b)] for a, b in zip(lo, hi)]
    widths = [[(h - l) / 2 for l, h in zip(a, b)] for a, b in zip(lo, hi)]
    fixed = [i for i in range(n) if all((w == 0 for w in widths[i]))]
    unknown = [i for i in range(n) if i not in set(fixed)]
    scale = 1024
    xx = np.array([[int(q * scale) for q in centers[i]] for i in fixed], dtype=np.int64)
    yy = np.array([int(y[i] * scale) for i in fixed], dtype=np.int64)
    assert all((F(int(q * scale), scale) == q for i in fixed for q in centers[i])) and all((F(int(y[i] * scale), scale) == y[i] for i in fixed))
    xmax = max((abs(int(q)) for q in xx.flat))
    ymax = max((abs(int(q)) for q in yy))
    assert len(xx) * max(xmax * xmax, xmax * ymax) < 2 ** 63
    gram = xx.T @ xx
    response = xx.T @ yy
    A = [[F(int(gram[i, j]), n * scale ** 2) + ridge * int(i == j) for j in range(p)] for i in range(p)]
    r = [F(int(q), n * scale ** 2) for q in response]
    return dict(lo=lo, hi=hi, y=y, beta=beta, n=n, p=p, ridge=ridge, c=centers, w=widths, fixed=fixed, unknown=unknown, A=A, r=r)

def audit_constants(D, cert):
    rec = cert['energy_receipt']
    f = {k: F(v) for k, v in rec['fractions'].items()}
    n, p = (D['n'], D['p'])
    T = mq(cert['T'])
    R = mq(cert['R'])
    rank = len(R[0])
    U = mm(T, R)
    ucols = tr(U)
    H = mm(tr(R), R)
    Hi = inverse(H)
    assert H == mq(rec['gram'])
    diag = min((H[j][j] - sum((abs(H[j][k]) for k in range(rank) if k != j), F(0)) for j in range(rank)))
    hmu = max(diag, 1 / sum((Hi[j][j] for j in range(rank)), F(0)))
    assert hmu == f['gram_minimum'] > 0
    BT = mm(D['A'], T)
    K = mm(tr(T), BT)
    mu = f['mu']
    nu = f['nu']
    assert min((K[j][j] - sum((abs(K[j][k]) for k in range(p) if k != j), F(0)) for j in range(p))) >= mu > 0
    assert max((K[j][j] + sum((abs(K[j][k]) for k in range(p) if k != j), F(0)) for j in range(p))) <= nu
    assert f['c'] == (nu - mu) / 2
    g0 = [dot(row, D['beta']) - v for row, v in zip(D['A'], D['r'])]
    h = [dot(col, g0) for col in tr(T)]
    kr = mm(tr(U), mm(D['A'], U))
    rh = [dot(col, g0) for col in ucols]
    shift = vq(rec['shift'])
    assert [dot(row, shift) for row in kr] == rh
    reference = [v - dot(row, shift) for v, row in zip(D['beta'], U)]
    assert reference == vq(rec['reference_beta'])
    gr = [dot(row, reference) - v for row, v in zip(D['A'], D['r'])]
    hr = [dot(col, gr) for col in tr(T)]
    assert hr == vq(rec['reference_anchor_gradient']) and all((dot(col, hr) == 0 for col in tr(R)))
    assert dot(hr, hr) == f['d2']
    si = rec['signal_rows'][0]
    js = [j for j, v in enumerate(D['w'][si]) if v]
    assert len(js) == 1
    sj = js[0]
    signal_vectors = [[dot(D['c'][si], col) for col in tr(T)], [D['w'][si][sj] * v for v in T[sj]]]
    assert all((dot(v, v) == quad([dot(v, col) for col in tr(R)], Hi) for v in signal_vectors))
    pc = [dot(D['c'][si], col) for col in ucols]
    pg = [D['w'][si][sj] * col[sj] for col in ucols]
    ec = dot(D['c'][si], reference) - D['y'][si]
    eg = D['w'][si][sj] * reference[sj]
    intervals = [tuple((q / n for q in qrange(pg[j] * eg, pc[j] * eg + pg[j] * ec, pc[j] * ec))) for j in range(rank)]
    assert intervals == [tuple(vq(v)) for v in rec['signal_gradient_intervals']]
    box = max((quad(list(v), Hi) for v in itertools.product(*intervals)))
    assert box == f['Gbox2']
    vv = [[q * ec / n for q in pc], [(a * eg + b * ec) / n for a, b in zip(pc, pg)], [q * eg / n for q in pg]]
    coeff = [sum((dot(vv[i], [dot(row, vv[k - i]) for row in Hi]) for i in range(3) if 0 <= k - i < 3), F(0)) for k in range(5)]
    assert coeff == vq(rec['signal_gradient_squared_power'])
    upper = F(0)
    count = 2 ** rec['bernstein_depth']
    for block in range(count):
        left = F(2 * block - count, count)
        width = F(2, count)
        cp = [sum((coeff[k] * comb(k, j) * left ** (k - j) * width ** j for k in range(j, 5)), F(0)) for j in range(5)]
        bc = [sum((cp[k] * F(comb(j, k), comb(4, k)) for k in range(j + 1)), F(0)) for j in range(5)]
        upper = max(upper, max(bc))
    assert upper == f['Gquartic2'] and min(box, upper) == f['G2']
    L = mq(rec['operator_basis'])
    LC = tr(L)
    LH = mm(tr(L), mm(H, L))
    lm = min((LH[j][j] - sum((abs(LH[j][k]) for k in range(rank) if j != k), F(0)) for j in range(rank)))
    assert lm == F(rec['operator_gram_minimum']) > 0
    low = [[F(0) for _ in range(rank)] for _ in range(rank)]
    high = [[F(0) for _ in range(rank)] for _ in range(rank)]
    a2 = eps2 = F(0)
    rows = []
    for i in D['unknown']:
        js = [j for j, s in enumerate(D['w'][i]) if s]
        c = D['c'][i]
        w = D['w'][i]
        pc0 = [dot(c, col) for col in ucols]
        pg0 = [[w[j] * col[j] for j in js] for col in ucols]
        e0 = dot(c, D['beta']) - D['y'][i]
        eg0 = [w[j] * D['beta'][j] for j in js]
        rows.append(dict(index=i, pc=pc0, pg=pg0, ec=e0, eg=eg0))
        if i == si:
            continue
        eu = abs(dot(c, reference) - D['y'][i]) + sum((abs(w[j] * reference[j]) for j in js), F(0))
        eps2 += eu ** 2 / n
        vertices = [quad([pc0[k] + sum((pg0[k][t] * s for t, s in enumerate(signs)), F(0)) for k in range(rank)], Hi) for signs in itertools.product((-1, 1), repeat=len(js))]
        a2 += max(vertices) / n
        pl = [dot(pc0, col) for col in LC]
        gl = [[dot([pg0[j][t] for j in range(rank)], col) for t in range(len(js))] for col in LC]
        for j in range(rank):
            for k in range(rank):
                l, u = prod_range(pl[j], gl[j], pl[k], gl[k])
                low[j][k] += l / n
                high[j][k] += u / n
    assert eps2 == f['epsilon2'] and a2 == f['trace_a2']
    assert low == mq(rec['operator_moment_lower']) and high == mq(rec['operator_moment_upper'])
    opa = max((high[j][j] + sum((max(abs(low[j][k]), abs(high[j][k])) for k in range(rank) if j != k), F(0)) for j in range(rank))) / lm
    assert opa == f['operator_a2'] and min(opa, a2) == f['a2']
    for v, q in ((f['a'], f['a2']), (f['epsilon'], eps2), (f['G'], f['G2']), (f['d'], f['d2']), (f['inverse_sqrt_mu'], 1 / mu)):
        sqrtcheck(v, q)
    a, eps, G, d, im = [f[k] for k in ('a', 'epsilon', 'G', 'd', 'inverse_sqrt_mu')]
    cc = f['c']
    rad = (G + a * eps + (cc / mu + a * im / 2) * d + cc * eps * im / 2) / mu
    eta = (eps + a * rad + (d + cc * rad) * im) ** 2
    assert rad == f['retained_radius'] and eta == f['eta'] == F(cert['tail_error']) and (f['coordinate_radius'] ** 2 * hmu >= rad * rad)
    assert rec['tail_target_met'] == (eta <= F(1, 400000))
    gl = h.copy()
    gu = h.copy()
    candidate_uncertainty_loss = F(0)
    for i in D['unknown']:
        c = D['c'][i]
        w = D['w'][i]
        js = [j for j, v in enumerate(w) if v]
        e0 = dot(c, D['beta']) - D['y'][i]
        eg0 = [w[j] * D['beta'][j] for j in js]
        candidate_uncertainty_loss += (abs(e0) + sum(map(abs, eg0), F(0))) ** 2 / n
        for j, col in enumerate(tr(T)):
            l, u = prod_range(dot(c, col), [w[k] * col[k] for k in js], e0, eg0)
            gl[j] += l / n
            gu[j] += u / n
    gc = sum((max(abs(l), abs(u)) ** 2 for l, u in zip(gl, gu)), F(0)) / mu

    def anchor_loss(v):
        return dot(v, [dot(row, v) for row in D['A']]) - 2 * dot(v, D['r'])
    ac = anchor_loss(D['beta']) - anchor_loss(reference) + dot(hr, hr) / mu + candidate_uncertainty_loss
    cap = min(gc, ac)
    assert F(cert['original_gradient_upper']) >= cap and F(cert['initial_upper']) >= F(cert['original_gradient_upper']) + eta
    return dict(rank=rank, U=U, H=H, kr=kr, rh=rh, rows=rows, eta=eta, cap=cap, shift=shift, radius=f['coordinate_radius'])

def exact_original_loss(x, y, beta, ridge):
    return sum(((dot(row, beta) - yy) ** 2 for row, yy in zip(x, y)), F(0)) / len(y) + ridge * dot(beta, beta)

def audit_witness(D, cert, result):
    w = cert['witness']
    x = mq(w['x'])
    theta = vq(w['theta'])
    assert all((l <= v <= h for lo, row, hi in zip(D['lo'], x, D['hi']) for l, v, h in zip(lo, row, hi)))
    v = exact_original_loss(x, D['y'], D['beta'], D['ridge']) - exact_original_loss(x, D['y'], theta, D['ridge'])
    assert F(w['regret_lower']) <= v and F(result['lower']) <= max(v, F(result['search_lower']) - F(cert['tail_error']))
    return v

def audit_tree(D, cert, result, C):
    rank = C['rank']
    n = D['n']
    eta = C['eta']
    rows = C['rows']
    K = C['kr']
    rh = C['rh']

    def point(z):
        val = 2 * dot(rh, z) - quad(z, K)
        for row in rows:
            ec, eg = (row['ec'], row['eg'])
            bc = ec - dot(row['pc'], z)
            bg = [e - dot([g[t] for g in row['pg']], z) for t, e in enumerate(eg)]
            values = []
            k = len(eg)
            if k == 0:
                val += (ec * ec - bc * bc) / n
                continue
            for free in range(k):
                other = [j for j in range(k) if j != free]
                for signs in itertools.product((-1, 1), repeat=k - 1):
                    aa = ec + sum((eg[j] * s for j, s in zip(other, signs)), F(0))
                    bb = bc + sum((bg[j] * s for j, s in zip(other, signs)), F(0))
                    values.append(qrange(eg[free] ** 2 - bg[free] ** 2, 2 * (aa * eg[free] - bb * bg[free]), aa * aa - bb * bb)[1])
            val += max(values) / n
        return val + eta
    rootlo = vq(cert['root_lo'])
    roothi = vq(cert['root_hi'])
    assert all((l <= c - C['radius'] and h >= c + C['radius'] for l, h, c in zip(rootlo, roothi, C['shift'])))
    points = {}
    for q in cert['points']:
        z = tuple(vq(q['z']))
        assert all((l <= v <= h for l, v, h in zip(rootlo, z, roothi)))
        v = point(z)
        assert F(q['lower']) <= v <= F(q['upper'])
        assert z not in points
        points[z] = q
    best = max((F(q['lower']) for q in points.values()))
    assert best == F(result['search_lower'])
    hdiag = vq(cert['hdiag'])
    for j in range(rank):
        assert hdiag[j] >= K[j][j] + sum((max(map(abs, arange(row['pc'][j], row['pg'][j]))) ** 2 / n for row in rows), F(0))
    nodes = cert['nodes']
    incoming = Counter()
    terminal = []
    for idx, node in enumerate(nodes):
        assert node['id'] == idx
        lo = vq(node['lo'])
        hi = vq(node['hi'])
        center = tuple((F(float((l + h) / 2)) for l, h in zip(lo, hi)))
        rad = [max(c - l, h - c) for l, h, c in zip(lo, hi, center)]
        assert center in points
        if idx == 0:
            assert lo == rootlo and hi == roothi and (node['parent'] is None)
            cap = F(cert['initial_upper'])
        else:
            assert 0 <= node['parent'] < idx
            cap = F(nodes[node['parent']]['upper'])
        grad = []
        for j in range(rank):
            gl = gu = 2 * (rh[j] - dot(K[j], center))
            for row in rows:
                br = arange(row['ec'] - dot(row['pc'], center), [e - dot([g[t] for g in row['pg']], center) for t, e in enumerate(row['eg'])])
                pr = arange(row['pc'][j], row['pg'][j])
                l, u = iprod(br, pr)
                gl += 2 * l / n
                gu += 2 * u / n
            grad.append(max(abs(gl), abs(gu)))
        corners = list(itertools.product(*zip(lo, hi)))
        assert all((z in points for z in corners))
        options = [cap, F(points[center]['upper']) + dot(grad, rad), max((F(points[z]['upper']) for z in corners)) + sum((v * r * r for v, r in zip(hdiag, rad)), F(0))]
        assert min(options) <= F(node['upper']) <= cap
        children = node['children']
        assert len(children) in (0, 2)
        if not children:
            terminal.append(F(node['upper']))
            continue
        first, second = [nodes[k] for k in children]
        assert first['parent'] == second['parent'] == idx
        ll, lh, rl, rh2 = map(vq, (first['lo'], first['hi'], second['lo'], second['hi']))
        assert ll == lo and rh2 == hi
        changed = [j for j in range(rank) if lh[j] != hi[j]]
        assert len(changed) == 1
        j = changed[0]
        assert lh[j] == rl[j] == center[j] and all((lh[k] == hi[k] and rl[k] == lo[k] for k in range(rank) if k != j))
        incoming.update(children)
    assert all((incoming[k] == 1 for k in range(1, len(nodes)))) and len(nodes) == 2 * result['nodes'] + 1
    assert F(result['search_upper']) >= max(best, max(terminal)) and F(result['upper']) >= min(C['cap'], max(best, max(terminal)))
    return dict(nodes=len(nodes), points=len(points), leaves=len(terminal), binary_coverage=True, exact_original_row_values=True)

def main():
    psutil.Process().cpu_affinity([12])
    start = time.monotonic()
    source = json.loads((B / 'confirmation_results.json').read_text())
    assert source['status'] == 'PASS'
    reports = []
    out = B / 'independent_audit_results.json'
    for row in source['results']:
        if row['method'] != 'energy':
            continue
        report = dict(dataset=row['dataset'], seed=row['seed'])
        try:
            result = json.loads((B / row['certificate_file']).read_text())
            cert = result['certificate']
            D = read(cert)
            C = audit_constants(D, cert)
            audit_witness(D, cert, result)
            report.update(status='PASS', constants='PASS', feasible_original_witness='PASS', tail=str(C['eta']))
            if row['seed'] == 84111:
                report['complete_tree'] = audit_tree(D, cert, result, C)
        except Exception as e:
            report.update(status='ERROR', error=str(e), traceback=traceback.format_exc())
        reports.append(report)
        out.write_text(json.dumps(dict(status='RUNNING', reports=reports, seconds=time.monotonic() - start), indent=2))
        print(row['dataset'], row['seed'], report['status'], report.get('error', ''), flush=True)
    out.write_text(json.dumps(dict(status='PASS' if len(reports) == 20 and all((r['status'] == 'PASS' for r in reports)) else 'FAILED', conditions=len(reports), full_trees=sum(('complete_tree' in r for r in reports)), imports_solver_implementation=False, reports=reports, seconds=time.monotonic() - start), indent=2))
if __name__ == '__main__':
    main()
