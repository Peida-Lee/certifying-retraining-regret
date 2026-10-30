from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Frozen new-mask and disjoint-pool public confirmation, complete costs.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import sys, time, json, traceback, csv
from fractions import Fraction as F
import numpy as np
import psutil
B = Path(__file__).resolve().parent
from real_world_design import problem
from centered_signal_energy import proposal, constants, CenteredSignalSearch, ObservedAnchor, MomentSearch, I, matvec, dot, float_up

def coarse(a):
    beta = [F(v) for v in a.beta]
    loss = F(0)
    for i in a.remaining:
        e = abs(dot([F(v) for v in a.c[i]], beta) - F(a.y[i])) + sum((abs(F(w) * b) for w, b in zip(a.rad[i], beta)), F(0))
        loss += e ** 2 / a.n
    anchored = a.qC + loss
    original = a.gradient_upper
    a.gradient_upper = min(original, float_up(anchored))
    mid = a.c
    theta = np.linalg.solve(mid.T @ mid / a.n + a.lam * np.eye(a.p), mid.T @ a.y / a.n)

    def objective(b):
        return (matvec(I(mid), I(b)) - I(a.y)).square().sum() / a.n + a.lam * I(b).square().sum()
    lower = max(0.0, float((objective(a.beta) - objective(theta)).lo))
    return (dict(gradient_cap=original, observed_loss_cap=float_up(anchored), common_cap=a.gradient_upper, common_lower=lower, common_gap=a.gradient_upper - lower), dict(x=mid.tolist(), theta=theta.tolist(), regret_lower=lower, z=None))

def main():
    psutil.Process().cpu_affinity([10])
    grid = json.loads((MODULE_DIR / 'experiment_settings.json').read_text())['designs']
    out = B / 'confirmation_results.json'
    assert not out.exists()
    rows = []
    conditions = []
    start = time.monotonic()
    specs = {'concrete': (0, 1 / 16, 84001), 'superconductivity': (0, 1 / 2, 84101), 'airfoil': (0, 1 / 16, 84001), 'powerplant': (0, 1 / 8, 84101)}
    (B / 'confirmation_certificates').mkdir(exist_ok=True)

    def save(status='RUNNING'):
        out.write_text(json.dumps(dict(status=status, conditions=conditions, results=rows, seconds=time.monotonic() - start), indent=2))
    for name, (cid, nw, pool) in specs.items():
        design = next((r for r in grid if r['dataset'] == name and r['condition'] == cid))
        frozen = dict(signal=design['signal_column'], nuisance=design['nuisance_columns'])
        for seed in range(84111, 84116):
            condition = dict(dataset=name, seed=seed, design_id=cid)
            try:
                t = time.monotonic()
                args, meta = problem(name, design['lambda_value'], 4, 16, design['width'], design['spread'], seed=seed, pool_seed=pool, frozen_columns=frozen, eligible_factor=2, iterations=3, nuisance_width=nw)
                input_s = time.monotonic() - t
                t = time.monotonic()
                a = ObservedAnchor(*args, anchor='observed', preconditioner='anchor')
                anchor_s = time.monotonic() - t
                t = time.monotonic()
                caps, witness = coarse(a)
                coarse_s = time.monotonic() - t
                t = time.monotonic()
                R = proposal(a, meta)
                basis_s = time.monotonic() - t
                t = time.monotonic()
                rec = constants(a, R, meta)
                energy_s = time.monotonic() - t
                t = time.monotonic()
                ref = constants(a, R, meta, reference_only=True)
                ref_s = time.monotonic() - t
                condition.update(meta, status='PASS', caps=caps, energy_receipt=rec, input_seconds=input_s, anchor_seconds=anchor_s, coarse_seconds=coarse_s, basis_seconds=basis_s, energy_verification_seconds=energy_s, reference_seconds=ref_s, disjoint_from_development=pool == 84101)
            except Exception as e:
                condition.update(status='ERROR', error=str(e), traceback=traceback.format_exc())
                conditions.append(condition)
                save()
                print(name, seed, 'COMMON ERROR', str(e), flush=True)
                continue
            conditions.append(condition)
            save()
            for method in ('energy', 'moment_centered', 'hybrid', 'full'):
                row = dict(dataset=name, seed=seed, method=method)
                try:
                    pre = input_s + anchor_s + coarse_s + (0 if method == 'full' else basis_s) + (ref_s if method == 'moment_centered' else energy_s if method != 'full' else 0)
                    t = time.monotonic()
                    solver = MomentSearch(a, a.p) if method == 'full' else CenteredSignalSearch(a, R, ref if method == 'moment_centered' else rec, method)
                    setup = time.monotonic() - t
                    solver.best_original = caps['common_lower']
                    solver.best_witness = witness
                    remaining = max(0.001, 10 - pre - setup)
                    result = solver.run(time_limit=remaining, absolute_tolerance=1e-05, relative_tolerance=0, record=True)
                    fn = f'{name}_{seed}_{method}.json'
                    path = B / 'confirmation_certificates' / fn
                    path.write_text(json.dumps(result, indent=2))
                    result.pop('certificate', None)
                    total = pre + setup + result['seconds']
                    row.update(result, status='PASS', method=method, setup_seconds=setup, preprocessing_seconds=pre, total_seconds=total, complete_budget_seconds=10.0, target_met=result['absolute_gap'] <= 1e-05, energy_tail_target_met=rec['tail_target_met'], certificate_file=str(path.relative_to(B)))
                    print(name, seed, method, 'tail', result['tail_error'], 'width', result['absolute_gap'], 'total', total, flush=True)
                except Exception as e:
                    row.update(status='ERROR', error=str(e), traceback=traceback.format_exc())
                    print(name, seed, method, 'ERROR', str(e), flush=True)
                rows.append(row)
                save()
    save('PASS' if len(rows) == 80 and all((r['status'] == 'PASS' for r in rows)) else 'COMPLETED_WITH_ERRORS')
    flat = [{k: v for k, v in r.items() if k not in ('trace', 'traceback')} for r in rows]
    keys = sorted(set().union(*(r.keys() for r in flat)))
    with (B / 'confirmation_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
if __name__ == '__main__':
    main()
