from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Rank-three physical-span energy diagnostic from exact row intervals.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import sys, time, json, itertools, traceback, csv
from pathlib import Path
from fractions import Fraction as F
B = Path(__file__).resolve().parent
ROOT = B.parent
sys.dont_write_bytecode = True
from public_energy_run import product_range, confirmation_problem, ObservedAnchor, basis_proposal, dot, maxabs, sqrt_upper, np, psutil

def determinant(a):
    if len(a) == 1:
        return a[0][0]
    return sum(((-1) ** j * a[0][j] * determinant([r[:j] + r[j + 1:] for r in a[1:]]) for j in range(len(a))), F(0))

def psd(a):
    return all((determinant([[a[i][j] for j in ids] for i in ids]) >= 0 for size in range(1, len(a) + 1) for ids in itertools.combinations(range(len(a)), size)))

def inverse(a):
    n = len(a)
    z = [r.copy() + [F(i == j) for j in range(n)] for i, r in enumerate(a)]
    for j in range(n):
        pivot = next((i for i in range(j, n) if z[i][j]))
        z[j], z[pivot] = (z[pivot], z[j])
        v = z[j][j]
        z[j] = [t / v for t in z[j]]
        for i in range(n):
            if i != j:
                v = z[i][j]
                z[i] = [t - v * u for t, u in zip(z[i], z[j])]
    return [r[n:] for r in z]

def qform(z, h):
    return sum((z[i] * h[i][j] * z[j] for i in range(len(z)) for j in range(len(z))), F(0))

def derive(a, R):
    n, p = (a.n, a.p)
    k = R.shape[1]
    beta = [F(v) for v in a.beta]
    lam = F(a.lam)
    u = [[dot([F(v) for v in row], [F(v) for v in R[:, j]]) for j in range(k)] for row in a.t]
    cols = list(map(list, zip(*u)))
    H = [[dot(x, y) for y in cols] for x in cols]
    assert psd(H) and determinant(H) > 0
    Hi = inverse(H)
    assert all((sum((H[i][z] * Hi[z][j] for z in range(k))) == int(i == j) for i in range(k) for j in range(k)))
    rows = []
    for low, high, yy in zip(a.lo, a.hi, a.y):
        lo = [F(v) for v in low]
        hi = [F(v) for v in high]
        c = [(x + y) / 2 for x, y in zip(lo, hi)]
        w = [(y - x) / 2 for x, y in zip(lo, hi)]
        js = [j for j, v in enumerate(w) if v]
        assert len(js) <= 2
        ec = dot(c, beta) - F(yy)
        eg = [w[j] * beta[j] for j in js]
        er = (ec - sum(map(abs, eg)), ec + sum(map(abs, eg)))
        pc = [dot(c, col) for col in cols]
        pg = [[w[j] * col[j] for j in js] for col in cols]
        pr = []
        dn = []
        c2 = dot(c, c)
        for signs in itertools.product((-1, 1), repeat=len(js)):
            z = [pc[l] + sum((pg[l][t] * signs[t] for t in range(len(js)))) for l in range(k)]
            projection = qform(z, Hi)
            norm = c2 + sum((2 * signs[t] * w[j] * c[j] + w[j] ** 2 for t, j in enumerate(js)))
            pr.append(projection)
            dn.append(norm - projection)
        assert min(dn) >= 0 and min(pr) >= 0
        rows.append(dict(score=maxabs(er), residual_squared=maxabs(er) ** 2, retained_squared=max(pr), discarded_squared=max(dn), granges=[product_range(pc[l], pg[l], ec, eg) for l in range(k)], fixed=not js, coordinates=pc))
    threshold = max((r['score'] for r in rows)) / 2
    sig = [r for r in rows if r['score'] >= threshold]
    nui = [r for r in rows if r['score'] < threshold]
    norm = lambda rs, key: sum((r[key] / n for r in rs), F(0))
    a2 = norm(nui, 'retained_squared')
    sigma2 = norm(sig, 'retained_squared')
    eps2 = norm(nui, 'residual_squared')
    E2 = norm(sig, 'residual_squared')
    b2 = norm(sig, 'discarded_squared')
    zb = [dot(beta, col) for col in cols]
    d2 = dot(beta, beta) - qform(zb, Hi)
    assert d2 >= 0
    fixed = [r for r in sig if r['fixed']]
    S = [[sum((r['coordinates'][i] * r['coordinates'][j] / n for r in fixed), F(0)) for j in range(k)] for i in range(k)]
    lower = F(0)
    upper = sum((Hi[i][j] * S[j][i] for i in range(k) for j in range(k)))
    assert upper >= 0 and psd(S)
    for _ in range(40):
        mid = (lower + upper) / 2
        if psd([[S[i][j] - mid * H[i][j] for j in range(k)] for i in range(k)]):
            lower = mid
        else:
            upper = mid
    assert psd([[S[i][j] - lower * H[i][j] for j in range(k)] for i in range(k)])
    kappa = lam + lower
    intervals = [(lam * zb[j] + sum((r['granges'][j][0] / n for r in sig), F(0)), lam * zb[j] + sum((r['granges'][j][1] / n for r in sig), F(0))) for j in range(k)]
    G2 = max((qform(list(z), Hi) for z in itertools.product(*intervals)))
    au, eps, sigma, E, b, d, G = [sqrt_upper(v) for v in (a2, eps2, sigma2, E2, b2, d2, G2)]
    sl = sqrt_upper(lam)
    omega = sqrt_upper(b * b / (lam + b * b))
    kb = lam * (kappa + b * b) / (lam + b * b)
    radius = (G + au * eps + omega * (au * E + sigma * eps) / 2 + omega ** 2 * sigma * E + (au * sl + sigma * b) * d) / kb
    terms = dict(residual=eps, retained_coupling=au * radius, signal_leakage=omega * (E + sigma * radius), candidate_leakage=sl * d)
    eta = sum(terms.values()) ** 2
    return dict(signal_rows=len(sig), nuisance_rows=len(nui), fixed_signal_rows=len(fixed), strict_energy_applicable=b2 == 0 and d2 == 0, strict_support=b2 == 0, candidate_in_span=d2 == 0, tail=float(eta), components={key: float(v) for key, v in terms.items()}, dominant_component=max(terms, key=terms.get), curvature_ratio=float(kb / kappa), tail_target_met=eta <= F(1, 400000), fractions={key: str(v) for key, v in dict(a2=a2, epsilon2=eps2, sigma2=sigma2, E2=E2, b2=b2, d2=d2, G2=G2, kappa=kappa, kappa_b=kb, radius=radius, eta=eta, fixed_signal_curvature=lower).items()}, gram=[[str(v) for v in row] for row in H], retained_curvature_proof='all principal minors of S_fixed-c*H nonnegative, exact40-step bisection')

def main():
    psutil.Process().cpu_affinity([8])
    out = B / 'public_energy_rank3_results.json'
    assert not out.exists()
    rows = []
    saved = json.loads((MODULE_DIR / 'metric_results.json').read_text())['results']
    started = time.monotonic()
    for name in ('diabetes', 'concrete', 'superconductivity'):
        for spread in ('concentrated', 'dispersed'):
            rec = dict(dataset=name, spread=spread, seed=78011, rank=3)
            try:
                t = time.monotonic()
                args, meta = confirmation_problem(name, spread, 78011)
                input_s = time.monotonic() - t
                t = time.monotonic()
                a = ObservedAnchor(*args, anchor='observed', preconditioner='midpoint')
                anchor_s = time.monotonic() - t
                t = time.monotonic()
                R = basis_proposal(a, 'gradient')[:, :3]
                basis_s = time.monotonic() - t
                t = time.monotonic()
                bound = derive(a, R)
                energy_s = time.monotonic() - t
                match = next((r for r in saved if r['dataset'] == name and r['spread'] == spread and (r['seed'] == 78011) and (r['method'] == 'gradient')))
                rec.update(status='PASS', n=a.n, p=a.p, energy=bound, moment_rank3_tail=match['tail'], input_seconds=input_s, anchor_seconds=anchor_s, basis_seconds=basis_s, energy_constants_seconds=energy_s, total_seconds=input_s + anchor_s + basis_s + energy_s)
            except Exception as error:
                rec.update(status='ERROR', error=str(error), traceback=traceback.format_exc())
            rows.append(rec)
            out.write_text(json.dumps(dict(status='RUNNING', results=rows), indent=2))
            print(name, spread, rec['status'], rec.get('energy', {}).get('tail'), flush=True)
    out.write_text(json.dumps(dict(status='PASS' if all((r['status'] == 'PASS' for r in rows)) else 'COMPLETED_WITH_ERRORS', results=rows, seconds=time.monotonic() - started, search_performed=False), indent=2))
    flat = [dict(dataset=r['dataset'], spread=r['spread'], rank=3, strict_energy_applicable=r['energy']['strict_energy_applicable'], leakage_energy_tail=r['energy']['tail'], moment_rank3_tail=r['moment_rank3_tail'], dominant_component=r['energy']['dominant_component'], curvature_ratio=r['energy']['curvature_ratio'], energy_constants_seconds=r['energy_constants_seconds'], total_seconds=r['total_seconds']) for r in rows if r['status'] == 'PASS']
    with (B / 'public_energy_rank3_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
if __name__ == '__main__':
    main()
