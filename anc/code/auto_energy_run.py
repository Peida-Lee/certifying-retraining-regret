from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import sys, time, json, hashlib, traceback, csv
from pathlib import Path
from unittest.mock import patch
from fractions import Fraction as F
sys.dont_write_bytecode = True
B = Path(__file__).resolve().parent
ROOT = B.parent
import numpy as np, psutil
from residual_energy import structure_problem
from observed_anchor import ObservedAnchor
import moment_tail
from moment_tail import MomentSearch, basis_proposal
from auto_energy import derive_constants, AutoEnergySearch
GRID = [(F(0), F(0), F(0)), (F(1, 2 ** 18), F(0), F(0)), (F(0), F(1, 2 ** 18), F(0)), (F(1, 2 ** 18), F(1, 2 ** 18), F(0)), (F(1, 2 ** 16), F(1, 2 ** 16), F(0)), (F(1, 2 ** 18), F(1, 2 ** 18), F(1, 2 ** 18))]

def problem(delta, ell, candidate_extra):
    lo, hi, y, beta, lam = structure_problem(16, F(1, 32))
    lo[:16, 1] = hi[:16, 1] = float(ell)
    y[16:24] += float(delta)
    y[24:] -= float(delta)
    beta[1] = float(candidate_extra)
    return (lo, hi, y, beta, lam)

def save(p):
    (B / 'auto_energy_results.json').write_text(json.dumps(p, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def main():
    psutil.Process().cpu_affinity([8])
    assert psutil.Process().cpu_affinity() == [8]
    if (B / 'auto_energy_results.json').exists():
        raise FileExistsError('auto_energy_results.json')
    payload = dict(status='RUNNING', cpu=8, threads=1, protocol_sha256=hashlib.sha256((B / 'auto_energy_protocol.md').read_bytes()).hexdigest(), rank=1, search_cap_seconds=2.0, target_width=1e-06, conditions=[], results=[], errors=[])
    for i, (delta, ell, gamma) in enumerate(GRID):
        meta = dict(condition=i, delta=str(delta), signal_leak=str(ell), candidate_extra=str(gamma))
        try:
            start = time.monotonic()
            args = problem(delta, ell, gamma)
            input_s = time.monotonic() - start
            start = time.monotonic()
            a = ObservedAnchor(*args, anchor='observed', preconditioner='anchor')
            anchor_s = time.monotonic() - start
            start = time.monotonic()
            R = basis_proposal(a, 'gradient')[:, :1]
            first = next((v for v in R[:, 0] if v), 1.0)
            if first < 0:
                R = -R
            direction_s = time.monotonic() - start
            start = time.monotonic()
            receipt = derive_constants(a, R)
            verification_s = time.monotonic() - start
            physical = np.array([float(F(v)) for v in receipt['physical_direction_r']])
            h = float(F(receipt['fractions']['h']))
            condition = dict(**meta, n=a.n, p=a.p, input_seconds=input_s, anchor_seconds=anchor_s, direction_seconds=direction_s, constants_partition_seconds=verification_s, numerical_basis=R.tolist(), e0_squared_cosine=float(physical[0] ** 2 / h), receipt=receipt, energy_applicable=receipt['status'] == 'PASS', eta=float(F(receipt['fractions']['eta'])), tail_budget_met=F(receipt['fractions']['eta']) <= F(1, 10 ** 6))
            payload['conditions'].append(condition)
            save(payload)
        except Exception:
            payload['errors'].append(dict(**meta, stage='common_pipeline', traceback=traceback.format_exc()))
            save(payload)
            continue
        common = input_s + anchor_s + direction_s + verification_s
        for method in ('energy', 'moment') if i % 2 == 0 else ('moment', 'energy'):
            row = dict(**meta, method=method, common_seconds=common, direction_seconds=direction_s, constants_partition_seconds=verification_s)
            try:
                tick = time.monotonic()
                if method == 'energy':
                    solver = AutoEnergySearch(a, R, receipt)
                else:
                    with patch.object(moment_tail, 'basis_proposal', return_value=R):
                        solver = MomentSearch(a, 1, kind='automatic_gradient', search_bound='adaptive')
                setup = time.monotonic() - tick
                tick = time.monotonic()
                result = solver.run(time_limit=2.0, absolute_tolerance=1e-06, relative_tolerance=0.0, record=True)
                search = time.monotonic() - tick
                filename = f'auto_energy_condition_{i}_{method}_certificate.json'
                (B / filename).write_text(json.dumps(dict(**meta, result=result), indent=2) + '\n', encoding='utf-8')
                result.pop('certificate')
                row.update(status='PASS', setup_seconds=setup, search_seconds=search, total_seconds=common + setup + search, actual_tail=float(solver.qtail), target_met=result['absolute_gap'] <= 1e-06, certificate_file=filename, result=result)
            except Exception:
                row.update(status='ERROR', traceback=traceback.format_exc())
                payload['errors'].append(row.copy())
            payload['results'].append(row)
            save(payload)
            print(i, method, row['status'], flush=True)
    payload['status'] = 'PASS' if len(payload['results']) == 12 and (not payload['errors']) else 'COMPLETED_WITH_ERRORS'
    save(payload)
    rows = []
    for r in payload['results']:
        rows.append(dict(condition=r['condition'], delta=r['delta'], signal_leak=r['signal_leak'], candidate_extra=r['candidate_extra'], method=r['method'], status=r['status'], direction_seconds=r['direction_seconds'], constants_partition_seconds=r['constants_partition_seconds'], tail=r.get('actual_tail'), width=r.get('result', {}).get('absolute_gap'), target_met=r.get('target_met'), total_seconds=r.get('total_seconds'), stop_reason=r.get('result', {}).get('stop_reason')))
    if rows:
        with (B / 'auto_energy_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
if __name__ == '__main__':
    main()
