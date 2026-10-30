from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Independent Fraction verification; imports no certificate implementation.'
from pathlib import Path
from fractions import Fraction as Q
from itertools import product
import json, time, sys, os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
BASE = Path(__file__).resolve().parent

def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Q(0))

def tr(a):
    return list(map(list, zip(*a)))

def mm(a, b):
    return [[dot(row, col) for col in tr(b)] for row in a]

def qm(a):
    return [[Q(v) for v in row] for row in a]

def inv(a):
    n = len(a)
    b = [row[:] + [Q(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    for j in range(n):
        v = b[j][j]
        b[j] = [x / v for x in b[j]]
        for i in range(n):
            if i != j:
                v = b[i][j]
                b[i] = [x - v * y for x, y in zip(b[i], b[j])]
    return [r[n:] for r in b]

def gmin(a):
    return min((row[j] - sum((abs(v) for k, v in enumerate(row) if k != j), Q(0)) for j, row in enumerate(a)))

def gmax(a):
    return max((row[j] + sum((abs(v) for k, v in enumerate(row) if k != j), Q(0)) for j, row in enumerate(a)))

def qrange(a, b, c):
    values = [a - b + c, a + b + c]
    if a and -1 < -b / (2 * a) < 1:
        values.append(c - b * b / (4 * a))
    return (min(values), max(values))

def rowmax(a, b, ga, gb):
    if not ga:
        return a * a - b * b
    best = None
    k = len(ga)
    for free in range(k):
        other = [j for j in range(k) if j != free]
        for bits in product((-1, 1), repeat=len(other)):
            aa = a + sum((ga[j] * s for j, s in zip(other, bits)), Q(0))
            bb = b + sum((gb[j] * s for j, s in zip(other, bits)), Q(0))
            value = qrange(ga[free] ** 2 - gb[free] ** 2, 2 * (aa * ga[free] - bb * gb[free]), aa * aa - bb * bb)[1]
            best = value if best is None else max(best, value)
    return best

def ranges(a, b):
    return (min(a, b), max(a, b))

def mul(x, y):
    v = [a * b for a in x for b in y]
    return (min(v), max(v))

def normcheck(value, row):
    assert Q(value) ** 2 >= dot(row, row) and value >= 0

def moments(x, y, n, lam):
    p = len(x[0])
    d = max((v.denominator for row in x for v in row))
    dy = max((v.denominator for v in y))
    xi = np.array([[int(v * d) for v in row] for row in x], dtype=np.int64)
    yi = np.array([int(v * dy) for v in y], dtype=np.int64)
    assert len(x) * max((abs(int(v)) for v in xi.flat)) ** 2 < 2 ** 63
    gram = xi.T @ xi
    resp = xi.T @ yi
    return ([[Q(int(gram[j, k]), n * d * d) + lam * int(j == k) for k in range(p)] for j in range(p)], [Q(int(v), n * d * dy) for v in resp])

def loss(x, y, b, lam):
    return sum(((dot(row, b) - yi) ** 2 for row, yi in zip(x, y)), Q(0)) / len(y) + lam * dot(b, b)

def audit(path, tree=True):
    start = time.monotonic()
    saved = json.loads(path.read_text())
    d = saved['certificate']
    rec = d['tail_receipt']
    lo, hi = (qm(d['lo']), qm(d['hi']))
    y = list(map(Q, d['y']))
    beta = list(map(Q, d['beta']))
    lam = Q(d['lambda'])
    n = len(y)
    p = len(beta)
    c = [[(a + b) / 2 for a, b in zip(l, u)] for l, u in zip(lo, hi)]
    rad = [[(b - a) / 2 for a, b in zip(l, u)] for l, u in zip(lo, hi)]
    fixed = [i for i, row in enumerate(rad) if not any(row)]
    rem = [i for i in range(n) if i not in fixed]
    G, r0 = moments([c[i] for i in fixed], [y[i] for i in fixed], n, lam)
    t = qm(d['T'])
    r = qm(d['R'])
    rank = len(r[0])
    K = mm(tr(t), mm(G, t))
    mu = Q(d['mu'])
    nu = Q(d['nu'])
    _, v = np.linalg.eigh(np.array(K, float))
    s = qm(np.round(v * 2 ** 24) / 2 ** 24)
    kmu = [[v - mu * int(i == j) for j, v in enumerate(row)] for i, row in enumerate(K)]
    knu = [[nu * int(i == j) - v for j, v in enumerate(row)] for i, row in enumerate(K)]
    assert gmin(mm(tr(s), mm(kmu, s))) >= 0 and gmin(mm(tr(s), mm(knu, s))) >= 0
    assert gmin(mm(tr(s), s)) > 0 and mu > 0
    h = mm(tr(r), r)
    hmu = Q(rec['Hmu'])
    assert 0 < hmu <= gmin(h)
    pp = mm(mm(r, inv(h)), tr(r))
    pp = [[Q(int(i == j)) - v for j, v in enumerate(row)] for i, row in enumerate(pp)]
    tp = mm(t, pp)
    trr = mm(t, r)
    ct = mm([c[i] for i in rem], t)
    g0 = [dot(row, beta) - ri for row, ri in zip(G, r0)]

    def gradient(transform):
        base = [dot(col, g0) for col in tr(transform)]
        result = []
        pro = mm([c[i] for i in rem], transform)
        for j in range(len(transform[0])):
            lower = upper = base[j]
            for a, i in enumerate(rem):
                ids = [k for k, v in enumerate(rad[i]) if v]
                pr = pro[a][j]
                res = dot(c[i], beta) - y[i]
                gp = [rad[i][k] * transform[k][j] for k in ids]
                gr = [rad[i][k] * beta[k] for k in ids]
                aa = (pr + res) / 2
                bb = (pr - res) / 2
                ga = [(a + b) / 2 for a, b in zip(gp, gr)]
                gb = [(a - b) / 2 for a, b in zip(gp, gr)]
                lower -= rowmax(bb, aa, gb, ga) / n
                upper += rowmax(aa, bb, ga, gb) / n
            result.append((lower, upper))
        return result
    allg = gradient(t)
    u0 = sum((max(abs(l), abs(u)) ** 2 for l, u in allg), Q(0)) / mu
    u0stored = Q(d['original_gradient_upper'])
    assert u0stored >= u0
    if rank < p:
        pg = gradient(tp)
        for (l, u), sl, su in zip(pg, rec['gradient_lo'], rec['gradient_hi']):
            assert Q(sl) <= l <= u <= Q(su)
        Gnorm = Q(rec['gradient_norm'])
        assert Gnorm ** 2 >= sum((max(abs(l), abs(u)) ** 2 for l, u in pg), Q(0))
        centerG, _ = moments(c, y, n, lam)
        cm = mm(tr(t), mm(centerG, t))
        cross = mm(pp, mm(cm, r))
        flat = [v for row in cross for v in row]
        normcheck(rec['center_cross_norm'], flat)
        for v, row in zip(rec['generator_p_norms'], tp):
            normcheck(v, row)
        for v, row in zip(rec['generator_r_norms'], trr):
            normcheck(v, row)
        cp = mm(ct, pp)
        cr = mm(ct, r)
        for v, row in zip(rec['center_p_norms'], cp):
            normcheck(v, row)
        for v, row in zip(rec['center_r_norms'], cr):
            normcheck(v, row)
        variation = Q(0)
        for a, i in enumerate(rem):
            dp = sum((v * Q(g) for v, g in zip(rad[i], rec['generator_p_norms'])), Q(0))
            dr = sum((v * Q(g) for v, g in zip(rad[i], rec['generator_r_norms'])), Q(0))
            variation += (dp * Q(rec['center_r_norms'][a]) + dr * Q(rec['center_p_norms'][a]) + dp * dr) / n
        assert Q(rec['variation_coupling']) >= variation
        coupling = Q(rec['coupling_norm'])
        assert coupling >= Q(rec['center_cross_norm']) + Q(rec['variation_coupling'])
        radius = Q(rec['coordinate_radius'])
        assert radius ** 2 >= u0stored / (mu * hmu)
        eta = Q(d['tail_error'])
        assert eta >= (Gnorm + coupling * radius) ** 2 / mu
    else:
        eta = Q(d['tail_error'])
        assert eta == 0
    witness = d['witness']
    wx = qm(witness['x'])
    wt = list(map(Q, witness['theta']))
    assert all((a <= v <= b for l, row, u in zip(lo, wx, hi) for a, v, b in zip(l, row, u)))
    wl = loss(wx, y, beta, lam) - loss(wx, y, wt, lam)
    assert wl >= Q(witness['regret_lower'])
    pointcount = nodecount = 0
    if tree:
        kr = mm(tr(r), mm(K, r))
        linear = [dot(col, g0) for col in tr(trr)]
        pro = mm([c[i] for i in rem], trr)
        hd = []
        for j in range(rank):
            hj = kr[j][j] + sum(((abs(pro[a][j]) + sum((v * abs(trr[k][j]) for k, v in enumerate(rad[i])), Q(0))) ** 2 for a, i in enumerate(rem)), Q(0)) / n
            assert Q(d['hdiag'][j]) >= hj
            hd.append(Q(d['hdiag'][j]))
        points = {}
        lower_exact = []
        for point in d['points']:
            z = tuple(map(Q, point['z']))
            theta = [b - dot(row, z) for b, row in zip(beta, trr)]
            val = 2 * dot(linear, z) - dot(z, [dot(row, z) for row in kr]) + eta
            for i in rem:
                ids = [k for k, v in enumerate(rad[i]) if v]
                val += rowmax(dot(c[i], beta) - y[i], dot(c[i], theta) - y[i], [rad[i][k] * beta[k] for k in ids], [rad[i][k] * theta[k] for k in ids]) / n
            assert Q(point['lower']) <= val <= Q(point['upper'])
            points[z] = point
            lower_exact.append(val - eta)
        nodes = d['nodes']
        leaves = []
        for idx, node in enumerate(nodes):
            l = tuple(map(Q, node['lo']))
            u = tuple(map(Q, node['hi']))
            center = tuple(((a + b) / 2 for a, b in zip(l, u)))
            radius = [(b - a) / 2 for a, b in zip(l, u)]
            if idx == 0:
                assert l == tuple((-v for v in u)) and all((v * v >= u0stored / (mu * hmu) for v in u))
            theta = [b - dot(row, center) for b, row in zip(beta, trr)]
            pen = [2 * linear[j] - 2 * dot(kr[j], center) for j in range(rank)]
            gbounds = []
            for j in range(rank):
                gl = gu = pen[j]
                for a, i in enumerate(rem):
                    br = sum((v * abs(b) for v, b in zip(rad[i], theta)), Q(0))
                    bc = dot(c[i], theta) - y[i]
                    xr = sum((v * abs(row[j]) for v, row in zip(rad[i], trr)), Q(0))
                    xc = pro[a][j]
                    low, high = mul((bc - br, bc + br), (xc - xr, xc + xr))
                    gl += 2 * low / n
                    gu += 2 * high / n
                gbounds.append(max(abs(gl), abs(gu)))
            lips = Q(points[center]['upper']) + dot(gbounds, radius)
            cap = u0stored + eta if idx == 0 else Q(nodes[node['parent']]['upper'])
            bounds = [lips, cap]
            if rank <= 3:
                bounds.append(max((Q(points[z]['upper']) for z in product(*zip(l, u)))) + sum((v * q * q for v, q in zip(hd, radius)), Q(0)))
            assert min(bounds) <= Q(node['upper'])
            children = node['children']
            if children:
                assert len(children) == 2
                a, b = [nodes[j] for j in children]
                al, au, bl, bu = [tuple(map(Q, v)) for v in (a['lo'], a['hi'], b['lo'], b['hi'])]
                change = [j for j in range(rank) if au[j] != u[j]]
                assert len(change) == 1
                j = change[0]
                assert al == l and bu == u and (au[j] == bl[j]) and (l[j] < au[j] < u[j])
                assert all((au[k] == u[k] and bl[k] == l[k] for k in range(rank) if k != j))
            else:
                leaves.append(Q(node['upper']))
        assert Q(saved['upper']) >= min(u0stored, max(leaves))
        assert Q(saved['lower']) <= max(Q(0), wl, max(lower_exact))
        pointcount = len(points)
        nodecount = len(nodes)
    return {'file': path.name, 'status': 'PASS', 'p': p, 'rank': rank, 'rows': n, 'anchor_rows': len(fixed), 'exact_points': pointcount, 'covered_nodes': nodecount, 'tail_and_metric_verified': True, 'original_witness_verified': True, 'tree_verified': tree, 'seconds': time.monotonic() - start}

def main():
    import psutil
    psutil.Process().cpu_affinity([6])
    results = []
    for name, tree in (('audit_small_moment_certificate.json', True), ('diabetes_concentrated_moment_certificate.json', True), ('primary_moment_certificate.json', False)):
        try:
            row = audit(BASE / name, tree)
        except Exception as exc:
            import traceback
            row = {'file': name, 'status': 'ERROR', 'error': str(exc), 'traceback': traceback.format_exc()}
        results.append(row)
        print(json.dumps(row), flush=True)
    (BASE / 'independent_moment_audit.json').write_text(json.dumps({'status': 'PASS' if all((v['status'] == 'PASS' for v in results)) else 'COMPLETED_WITH_ERRORS', 'results': results}, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
