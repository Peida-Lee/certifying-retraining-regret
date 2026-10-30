from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Fixed multidimensional interval family and verified energy/moment controls.'
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys, time, json, traceback, itertools, csv
from fractions import Fraction as F
from unittest.mock import patch
B = Path(__file__).resolve().parent
ROOT = B.parent
sys.dont_write_bytecode = True
from public_energy_rank3_run import derive, inverse, dot, sqrt_upper, product_range, np, psutil, ObservedAnchor, basis_proposal
import moment_tail
from moment_tail import MomentSearch
from observed_anchor import float_up
GRID = list(itertools.product((2, 3), (F(0), F(1, 2 ** 10), F(1, 2 ** 6)), (F(0), F(1, 2 ** 12), F(1, 2 ** 4))))

def problem(rank, delta, ell):
    m = 16
    n = 2 * m * rank
    p = rank + m * rank
    lo = np.zeros((n, p))
    hi = lo.copy()
    y = np.zeros(n)
    beta = np.r_[np.ones(rank), np.zeros(m * rank)]
    alpha = F(1, 32)
    for j in range(rank):
        half = F(1, 2 ** (j + 2))
        for i in range(m):
            s = j * m + i
            d = rank + j * m + i
            t = m * rank + s
            lo[s, j] = float(1 - half)
            hi[s, j] = float(1 + half)
            lo[s, d] = hi[s, d] = float(ell)
            y[s] = 2
            lo[t, j] = hi[t, j] = float(alpha)
            lo[t, d] = 4
            hi[t, d] = 8
            y[t] = float(alpha + (1 if i < 8 else -1) * delta)
    return (lo, hi, y, beta, 1 / (2 * rank))

def energy_constants(a, R):
    rec = derive(a, R)
    n, p = (a.n, a.p)
    r = R.shape[1]
    beta = [F(v) for v in a.beta]
    U = [[dot([F(v) for v in row], [F(v) for v in R[:, j]]) for j in range(r)] for row in a.t]
    cols = list(map(list, zip(*U)))
    H = [[dot(x, y) for y in cols] for x in cols]
    scores = []
    rows = []
    for lo, hi, yy in zip(a.lo, a.hi, a.y):
        c = [(F(x) + F(y)) / 2 for x, y in zip(lo, hi)]
        w = [(F(y) - F(x)) / 2 for x, y in zip(lo, hi)]
        js = [j for j, s in enumerate(w) if s]
        scores.append(abs(dot(c, beta) - F(yy)) + sum((abs(w[j] * beta[j]) for j in js)))
        pc = [dot(c, col) for col in cols]
        pg = [[w[j] * col[j] for j in js] for col in cols]
        rows.append((pc, pg))
    threshold = max(scores) / 2
    signal = [i for i, s in enumerate(scores) if s >= threshold]
    low = [[F(0) for _ in range(r)] for _ in range(r)]
    high = [[F(0) for _ in range(r)] for _ in range(r)]
    for i in signal:
        pc, pg = rows[i]
        for j in range(r):
            for k in range(r):
                l, u = product_range(pc[j], pg[j], pc[k], pg[k])
                low[j][k] += l / n
                high[j][k] += u / n

    def valid(c):
        return all((low[j][j] - c * H[j][j] >= sum((max(abs(low[j][k] - c * H[j][k]), abs(high[j][k] - c * H[j][k])) for k in range(r) if k != j)) for j in range(r)))
    left = F(0)
    right = F(1000)
    if valid(F(0)):
        for _ in range(40):
            mid = (left + right) / 2
            if valid(mid):
                left = mid
            else:
                right = mid
        assert valid(left)
    f = {k: F(v) for k, v in rec['fractions'].items()}
    kappa = F(a.lam) + left
    au, eps, sigma, E, b, d, G = [sqrt_upper(f[k]) for k in ('a2', 'epsilon2', 'sigma2', 'E2', 'b2', 'd2', 'G2')]
    lam = F(a.lam)
    sl = sqrt_upper(lam)
    omega = sqrt_upper(b * b / (lam + b * b))
    kb = lam * (kappa + b * b) / (lam + b * b)
    radius = (G + au * eps + omega * (au * E + sigma * eps) / 2 + omega ** 2 * sigma * E + (au * sl + sigma * b) * d) / kb
    eta = (eps + au * radius + omega * (E + sigma * radius) + sl * d) ** 2
    hmu = min((H[j][j] - sum((abs(H[j][k]) for k in range(r) if k != j)) for j in range(r)))
    assert hmu > 0
    root = sqrt_upper(radius ** 2 / hmu)
    units = root * 2 ** 20
    root = F((units.numerator + units.denominator - 1) // units.denominator, 2 ** 20)
    f.update(kappa=kappa, kappa_b=kb, radius=radius, eta=eta, retained_signal_curvature=left, physical_Hmu=hmu, coordinate_radius=root)
    terms = dict(residual=eps, retained_coupling=au * radius, signal_leakage=omega * (E + sigma * radius), candidate_leakage=sl * d)
    rec.update(fractions={k: str(v) for k, v in f.items()}, tail=float(eta), components={k: float(v) for k, v in terms.items()}, dominant_component=max(terms, key=terms.get), curvature_ratio=float(kb / kappa), tail_target_met=eta <= F(1, 100000), tail_target=1e-05, signal_indices=signal, curvature_method='exact interval diagonal-dominance enclosure of retained signal moments', retained_curvature_proof='exact interval diagonal-dominance enclosure of retained signal moments', signal_moment_lower=[[str(v) for v in row] for row in low], signal_moment_upper=[[str(v) for v in row] for row in high])
    return rec

class EnergySearch(MomentSearch):

    def __init__(self, a, R, receipt):
        with patch.object(moment_tail, 'basis_proposal', return_value=R):
            super().__init__(a, R.shape[1], kind='gradient')
        f = receipt['fractions']
        self.qtail = F(f['eta'])
        self.radius = float(F(f['coordinate_radius']))
        self.initial_upper = float_up(F(a.gradient_upper) + self.qtail)
        self.energy_receipt = receipt

    def run(self, **kw):
        out = super().run(**kw)
        out['method'] = 'multirank_leakage_energy'
        if 'certificate' in out:
            out['certificate'].update(method='multirank_leakage_energy', energy_receipt=self.energy_receipt, tail_error=str(self.qtail), coordinate_radius=self.radius, initial_upper=self.initial_upper)
        return out

def exact_control(rank):
    alpha = F(1, 32)
    g = []
    Bv = []
    Kv = []
    for j in range(rank):
        half = F(1, 2 ** (j + 2))
        g.append(half ** 2 / (2 * rank))
        Bv.append((F(1, 2) + (1 - half) ** 2 / 2 + alpha ** 2 / 2) / rank)
        Kv.append((F(1, 2) + (1 - half) ** 2 / 2 + alpha ** 2 / 10) / rank)
    return dict(full=str(sum((v * v / k for v, k in zip(g, Kv)))), active=str(sum((v * v / k for v, k in zip(g, Bv)))), candidate_line=str(sum(g) ** 2 / sum(Bv)), active_displacement=[str(v / k) for v, k in zip(g, Bv)])

def main():
    psutil.Process().cpu_affinity([8])
    out = B / 'multirank_results.json'
    assert not out.exists()
    records = []
    conditions = []
    started = time.monotonic()

    def save():
        out.write_text(json.dumps(dict(status='RUNNING', conditions=conditions, results=records, elapsed_seconds=time.monotonic() - started), indent=2))
    for cid, (rank, delta, ell) in enumerate(GRID):
        meta = dict(condition=cid, active_dimension=rank, delta=str(delta), ell=str(ell))
        try:
            tick = time.monotonic()
            args = problem(rank, delta, ell)
            input_s = time.monotonic() - tick
            tick = time.monotonic()
            a = ObservedAnchor(*args, anchor='observed', preconditioner='anchor')
            anchor_s = time.monotonic() - tick
            tick = time.monotonic()
            R = basis_proposal(a, 'gradient')[:, :rank]
            direction_s = time.monotonic() - tick
            tick = time.monotonic()
            receipt = energy_constants(a, R)
            constants_s = time.monotonic() - tick
            recovered = np.flatnonzero(np.any(R != 0, axis=1)).tolist()
            condition = dict(**meta, n=a.n, p=a.p, recovered_support=recovered, actual_retained_rank=rank, receipt=receipt, input_seconds=input_s, anchor_seconds=anchor_s, direction_seconds=direction_s, constants_seconds=constants_s, exact_control=exact_control(rank) if delta == ell == 0 else None)
            conditions.append(condition)
            save()
        except Exception as exc:
            records.append(dict(**meta, status='ERROR', stage='common', error=str(exc), traceback=traceback.format_exc()))
            save()
            continue
        for method in ('energy', 'moment', 'candidate_line'):
            row = dict(**meta, method=method)
            try:
                tick = time.monotonic()
                basis = R if method != 'candidate_line' else a.beta.reshape(-1, 1)
                if method == 'energy':
                    solver = EnergySearch(a, R, receipt)
                else:
                    with patch.object(moment_tail, 'basis_proposal', return_value=basis):
                        solver = MomentSearch(a, basis.shape[1], kind='gradient')
                setup = time.monotonic() - tick
                tick = time.monotonic()
                result = solver.run(time_limit=2, absolute_tolerance=1e-05, relative_tolerance=0, record=True)
                search = time.monotonic() - tick
                filename = f'multirank_{cid}_{method}_certificate.json'
                (B / filename).write_text(json.dumps(result, indent=2))
                result.pop('certificate', None)
                result.pop('witness', None)
                row.update(status='PASS', retained_rank=solver.rank, tail=float(solver.qtail), width=result['absolute_gap'], target_met=result['absolute_gap'] <= 1e-05, energy_tail_target_met=F(receipt['fractions']['eta']) <= F(1, 100000), lower=result['lower'], upper=result['upper'], setup_seconds=setup, search_seconds=search, total_seconds=input_s + anchor_s + direction_s + constants_s + setup + search, stop_reason=result['stop_reason'], certificate_file=filename)
                if delta == ell == 0:
                    h = F(condition['exact_control']['full'])
                    assert F(result['lower']) <= h <= F(result['upper'])
            except Exception as exc:
                row.update(status='ERROR', error=str(exc), traceback=traceback.format_exc())
            records.append(row)
            save()
            print(cid, method, row['status'], row.get('tail'), row.get('width'), flush=True)
    payload = dict(status='PASS' if len(records) == 54 and all((r['status'] == 'PASS' for r in records)) else 'COMPLETED_WITH_ERRORS', conditions=conditions, results=records, elapsed_seconds=time.monotonic() - started)
    out.write_text(json.dumps(payload, indent=2))
    flat = [{k: v for k, v in row.items() if k not in ('traceback',)} for row in records]
    keys = sorted(set().union(*(r.keys() for r in flat)))
    with (B / 'multirank_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
if __name__ == '__main__':
    main()
