from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Exact original-row rank-one energy diagnostics on six fixed public tables.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import sys, time, json, traceback, itertools, csv
from pathlib import Path
from fractions import Fraction as F
from unittest.mock import patch
B = Path(__file__).resolve().parent
ROOT = B.parent
sys.dont_write_bytecode = True
import numpy as np, psutil
from auto_energy import dot, maxabs, minabs, sqrt_upper
from greedy_run import confirmation_problem
from observed_anchor import ObservedAnchor
import moment_tail
from moment_tail import MomentSearch, basis_proposal

def product_range(a, u, b, v):
    k = len(u)
    if not k:
        return (a * b, a * b)
    assert k <= 2
    values = []
    for free in range(k):
        others = [j for j in range(k) if j != free]
        for signs in itertools.product((-1, 1), repeat=len(others)):
            aa = a + sum((u[j] * t for j, t in zip(others, signs)))
            bb = b + sum((v[j] * t for j, t in zip(others, signs)))
            A = u[free] * v[free]
            C = aa * bb
            D = aa * v[free] + bb * u[free]
            values.extend([A - D + C, A + D + C])
            if A and -1 < -D / (2 * A) < 1:
                values.append(C - D * D / (4 * A))
    return (min(values), max(values))

def derive(a, R):
    n, p = (a.n, a.p)
    beta = [F(v) for v in a.beta]
    lam = F(a.lam)
    r = [dot([F(v) for v in row], [F(v) for v in R[:, 0]]) for row in a.t]
    h = dot(r, r)
    rows = []
    assert h > 0
    for l, u, yy in zip(a.lo, a.hi, a.y):
        lo = [F(x) for x in l]
        hi = [F(x) for x in u]
        c = [(x + y) / 2 for x, y in zip(lo, hi)]
        w = [(y - x) / 2 for x, y in zip(lo, hi)]
        js = [j for j, v in enumerate(w) if v]
        assert len(js) <= 2
        ec = dot(c, beta) - F(yy)
        pc = dot(c, r)
        eg = [w[j] * beta[j] for j in js]
        pg = [w[j] * r[j] for j in js]
        er = (ec - sum(map(abs, eg)), ec + sum(map(abs, eg)))
        pr = (pc - sum(map(abs, pg)), pc + sum(map(abs, pg)))
        c2 = dot(c, c)
        dn = []
        for signs in itertools.product((-1, 1), repeat=len(js)):
            norm = c2 + sum((2 * t * w[j] * c[j] + w[j] * w[j] for j, t in zip(js, signs)))
            proj = pc + sum((t * w[j] * r[j] for j, t in zip(js, signs)))
            dn.append(norm - proj * proj / h)
        assert min(dn) >= 0
        rows.append(dict(score=maxabs(er), er=er, pr=pr, d2=max(dn), grange=product_range(pc, pg, ec, eg)))
    threshold = max((x['score'] for x in rows)) / 2
    signal = [x for x in rows if x['score'] >= threshold]
    nuisance = [x for x in rows if x['score'] < threshold]
    retained = lambda rs: sum((maxabs(x['pr']) ** 2 / (n * h) for x in rs), F(0))
    residual = lambda rs: sum((maxabs(x['er']) ** 2 / n for x in rs), F(0))
    a2 = retained(nuisance)
    sigma2 = retained(signal)
    eps2 = residual(nuisance)
    E2 = residual(signal)
    b2 = sum((x['d2'] / n for x in signal), F(0))
    d2 = dot(beta, beta) - dot(beta, r) ** 2 / h
    kappa = lam + sum((minabs(x['pr']) ** 2 / (n * h) for x in signal), F(0))
    gl = lam * dot(beta, r) + sum((x['grange'][0] / n for x in signal), F(0))
    gu = lam * dot(beta, r) + sum((x['grange'][1] / n for x in signal), F(0))
    G2 = max(abs(gl), abs(gu)) ** 2 / h
    au, eps, sigma, E, b, d, G = [sqrt_upper(x) for x in (a2, eps2, sigma2, E2, b2, d2, G2)]
    sl = sqrt_upper(lam)
    omega = sqrt_upper(b * b / (lam + b * b))
    kb = lam * (kappa + b * b) / (lam + b * b)
    radius = (G + au * eps + omega * (au * E + sigma * eps) / 2 + omega ** 2 * sigma * E + (au * sl + sigma * b) * d) / kb
    terms = dict(residual=eps, retained_coupling=au * radius, signal_leakage=omega * (E + sigma * radius), candidate_leakage=sl * d)
    eta = sum(terms.values()) ** 2
    return dict(signal_rows=len(signal), nuisance_rows=len(nuisance), threshold=str(threshold), strict_support=b2 == 0, candidate_in_span=d2 == 0, strict_energy_applicable=b2 == 0 and d2 == 0, fractions={k: str(v) for k, v in dict(h=h, a2=a2, sigma2=sigma2, epsilon2=eps2, E2=E2, b2=b2, d2=d2, G2=G2, kappa=kappa, kappa_b=kb, radius=radius, eta=eta).items()}, components={k: float(v) for k, v in terms.items()}, dominant_component=max(terms, key=terms.get), tail=float(eta), curvature_ratio=float(kb / kappa), tail_target_met=eta <= F(1, 400000))

def main():
    psutil.Process().cpu_affinity([8])
    out = B / 'public_energy_results.json'
    assert not out.exists()
    records = []
    saved = json.loads((MODULE_DIR / 'metric_results.json').read_text())['results']
    start = time.monotonic()
    for name in ('diabetes', 'concrete', 'superconductivity'):
        for spread in ('concentrated', 'dispersed'):
            rec = dict(dataset=name, spread=spread, seed=78011, rank=1)
            try:
                tick = time.monotonic()
                args, meta = confirmation_problem(name, spread, 78011)
                input_s = time.monotonic() - tick
                tick = time.monotonic()
                a = ObservedAnchor(*args, anchor='observed', preconditioner='midpoint')
                anchor_s = time.monotonic() - tick
                tick = time.monotonic()
                R = basis_proposal(a, 'gradient')[:, :1]
                basis_s = time.monotonic() - tick
                tick = time.monotonic()
                bound = derive(a, R)
                energy_s = time.monotonic() - tick
                tick = time.monotonic()
                with patch.object(moment_tail, 'basis_proposal', return_value=R):
                    m = MomentSearch(a, 1, kind='gradient')
                moment_s = time.monotonic() - tick
                rank3 = next((r for r in saved if r['dataset'] == name and r['spread'] == spread and (r['seed'] == 78011) and (r['method'] == 'gradient')))
                rec.update(status='PASS', n=a.n, p=a.p, energy=bound, moment_rank1_tail=float(m.qtail), moment_rank3_tail=rank3['tail'], input_seconds=input_s, anchor_seconds=anchor_s, basis_seconds=basis_s, energy_constants_seconds=energy_s, moment_setup_seconds=moment_s, total_seconds=input_s + anchor_s + basis_s + energy_s + moment_s)
            except Exception as error:
                rec.update(status='ERROR', error=str(error), traceback=traceback.format_exc())
            records.append(rec)
            out.write_text(json.dumps(dict(status='RUNNING', results=records), indent=2))
            print(name, spread, rec['status'], rec.get('energy', {}).get('tail'), flush=True)
    payload = dict(status='PASS' if all((r['status'] == 'PASS' for r in records)) else 'COMPLETED_WITH_ERRORS', results=records, seconds=time.monotonic() - start, search_performed=False)
    out.write_text(json.dumps(payload, indent=2))
    rows = []
    for r in records:
        if r['status'] == 'PASS':
            rows.append(dict(dataset=r['dataset'], spread=r['spread'], n=r['n'], p=r['p'], strict_energy_applicable=r['energy']['strict_energy_applicable'], leakage_energy_tail=r['energy']['tail'], moment_rank1_tail=r['moment_rank1_tail'], moment_rank3_tail=r['moment_rank3_tail'], dominant_component=r['energy']['dominant_component'], curvature_ratio=r['energy']['curvature_ratio'], energy_constants_seconds=r['energy_constants_seconds'], total_seconds=r['total_seconds']))
    with (B / 'public_energy_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
if __name__ == '__main__':
    main()
