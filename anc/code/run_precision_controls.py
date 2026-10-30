from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Fixed-mask candidate precision and random nuisance-row ablation.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import sys, time, json, csv, traceback
import numpy as np, psutil
B = Path(__file__).resolve().parent
OLD = MODULE_DIR
from datasets import dataset
from centered_signal_energy import ObservedAnchor, proposal, constants, CenteredSignalSearch
from run_real_world import coarse

def main():
    psutil.Process().cpu_affinity([10])
    source = json.loads((OLD / 'confirmation_results.json').read_text())
    out = B / 'precision_row_results.json'
    assert not out.exists()
    records = []
    conditions = []
    start = time.monotonic()
    for name in ('concrete', 'superconductivity', 'airfoil', 'powerplant'):
        original = next((c for c in source['conditions'] if c['dataset'] == name and c['seed'] == 84111))
        old = next((r for r in source['results'] if r['dataset'] == name and r['seed'] == 84111 and (r['method'] == 'energy')))
        cert = json.loads((OLD / old['certificate_file']).read_text())['certificate']
        base_lo = np.array(cert['lo'])
        base_hi = np.array(cert['hi'])
        yy = np.array(cert['y'])
        xx, _, _ = dataset(name, original['pool_seed'])
        si = original['signal_row']
        sj = original['signal_column']
        lam = original['lambda_value']
        for rowrule in ('selected', 'random_nuisance'):
            lo = base_lo.copy()
            hi = base_hi.copy()
            meta = original.copy()
            if rowrule == 'random_nuisance':
                lo = xx.copy()
                hi = xx.copy()
                lo[si, sj] = base_lo[si, sj]
                hi[si, sj] = base_hi[si, sj]
                rows = np.random.default_rng(85111).choice(np.setdiff1d(np.arange(len(xx)), [si]), 16, replace=False)
                ns = original['nuisance_columns']
                nw = original['nuisance_width']
                for i in rows:
                    for j in [int(ns[(2 * i + t) % len(ns)]) for t in range(2)]:
                        ll = -1 + np.floor((xx[i, j] + 1) / nw) * nw
                        lo[i, j] = ll
                        hi[i, j] = ll + nw
                meta['row_ids'] = rows.tolist()
            c = (lo + hi) / 2
            for bits in (0, 8, 4):
                condition = dict(dataset=name, row_rule=rowrule, candidate_bits=bits)
                try:
                    t = time.monotonic()
                    beta = np.linalg.solve(c.T @ c / len(c) + lam * np.eye(c.shape[1]), c.T @ yy / len(c))
                    beta = np.round(beta * 2 ** bits) / 2 ** bits if bits else beta
                    fit_s = time.monotonic() - t
                    t = time.monotonic()
                    a = ObservedAnchor(lo, hi, yy, beta, lam, anchor='observed', preconditioner='anchor')
                    anchor_s = time.monotonic() - t
                    t = time.monotonic()
                    caps, witness = coarse(a)
                    caps_s = time.monotonic() - t
                    t = time.monotonic()
                    R = proposal(a, meta)
                    basis_s = time.monotonic() - t
                    t = time.monotonic()
                    rec = constants(a, R, meta)
                    energy_s = time.monotonic() - t
                    t = time.monotonic()
                    ref = constants(a, R, meta, reference_only=True)
                    ref_s = time.monotonic() - t
                    common = original['input_seconds'] + fit_s + anchor_s + caps_s + basis_s
                    condition.update(status='PASS', caps=caps, coarse_already_met=caps['common_gap'] <= 1e-05, energy_tail=rec['tail'], tail_met=rec['tail_target_met'], n=a.n, p=a.p, rank=3, uncertain_cells=int(np.sum(lo != hi)), actual_uncertain_columns=int(sum(np.any(lo != hi, axis=0))), energy_seconds=energy_s, reference_seconds=ref_s, common_seconds=common)
                except Exception as e:
                    condition.update(status='ERROR', error=str(e), traceback=traceback.format_exc())
                    conditions.append(condition)
                    continue
                conditions.append(condition)
                for method in ('energy', 'moment_centered'):
                    row = dict(dataset=name, row_rule=rowrule, candidate_bits=bits, method=method)
                    try:
                        t = time.monotonic()
                        solver = CenteredSignalSearch(a, R, rec if method == 'energy' else ref, method)
                        setup = time.monotonic() - t
                        solver.best_original = caps['common_lower']
                        solver.best_witness = witness
                        pre = common + (energy_s if method == 'energy' else ref_s)
                        result = solver.run(time_limit=max(0.001, 10 - pre - setup), absolute_tolerance=1e-05, relative_tolerance=0, record=True)
                        filename = f'precision_{name}_{rowrule}_{bits}_{method}.json'
                        folder = B / 'precision_certificates'
                        folder.mkdir(exist_ok=True)
                        (folder / filename).write_text(json.dumps(result, indent=2))
                        result.pop('certificate', None)
                        row.update(result, status='PASS', method=method, total_seconds=pre + setup + result['seconds'], coarse_already_met=condition['coarse_already_met'], target_met=result['absolute_gap'] <= 1e-05, certificate_file='precision_certificates/' + filename)
                        print(name, rowrule, bits, method, 'coarse', row['coarse_already_met'], 'tail', row['tail_error'], 'width', row['absolute_gap'], flush=True)
                    except Exception as e:
                        row.update(status='ERROR', error=str(e), traceback=traceback.format_exc())
                        print(name, rowrule, bits, method, 'ERROR', str(e), flush=True)
                    records.append(row)
                    out.write_text(json.dumps(dict(status='RUNNING', conditions=conditions, results=records, seconds=time.monotonic() - start), indent=2))
    out.write_text(json.dumps(dict(status='PASS' if len(records) == 48 and all((r['status'] == 'PASS' for r in records)) else 'COMPLETED_WITH_ERRORS', conditions=conditions, results=records, seconds=time.monotonic() - start), indent=2))
    flat = [{k: v for k, v in r.items() if k not in ('trace', 'traceback')} for r in records]
    keys = sorted(set().union(*(r.keys() for r in flat)))
    with (B / 'precision_row_summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)
if __name__ == '__main__':
    main()
