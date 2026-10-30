from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Independent preprocessing calibration on the complete fixed12-condition grid.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import sys, time, json, hashlib, traceback
from pathlib import Path
from unittest.mock import patch
B = Path(__file__).resolve().parent
ROOT = B.parent
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, psutil
import moment_tail
from moment_tail import MomentSearch
from metric_basis import build_proposal
from observed_anchor import ObservedAnchor
from rep_pool_run import endpoint, fingerprint
METHODS = ('gradient', 'metric_pod', 'metric_greedy', 'full')
CAP = 10.0

def save(path, p):
    path.write_text(json.dumps(p, indent=2, allow_nan=False) + '\n')

def prepare(tab):
    prior = json.loads((MODULE_DIR / 'rep_pool_pools.json').read_text())
    excluded = set(prior['previous_excluded_row_ids'])
    for pool in prior['pools']:
        excluded.update(pool['row_indices'])
    available = np.array(sorted(set(range(len(tab))) - excluded))
    ids = np.sort(np.random.default_rng(81041).choice(available, 512, replace=False))
    start = time.monotonic()
    raw = tab.iloc[ids, :-1].to_numpy(float)
    means = raw.mean(0)
    spans = np.ptp(raw, axis=0)
    spans = np.where(spans > 0, spans, 1.0)
    cost = time.monotonic() - start
    plan = dict(status='FROZEN_BEFORE_TARGET_CERTIFICATES', seed=81041, calibration_records=512, calibration_feature_cells=512 * raw.shape[1], calibration_row_ids=ids.tolist(), excluded_row_ids=sorted(excluded), target_pools=prior['pools'], feature_means=means.tolist(), feature_spans=spans.tolist(), calibration_compute_seconds=cost)
    assert not set(ids) & excluded
    save(B / 'calibration_plan.json', plan)
    return plan

def problem(tab, pool, nr, spread, plan):
    raw = tab.iloc[pool['row_indices'], :-1].to_numpy(float)
    x = np.column_stack([np.ones(len(raw)), np.round((raw - np.array(plan['feature_means'])) / np.array(plan['feature_spans']) * 1024) / 1024])
    y = np.round(tab.iloc[pool['row_indices'], -1].to_numpy(float) / 128 * 2 ** 20) / 2 ** 20
    lo = x.copy()
    hi = x.copy()
    maskrows = np.random.default_rng(78041).permutation(len(x))[:nr]
    cols = [1, 2] if spread == 'concentrated' else list(range(1, 9))
    for k, i in enumerate(maskrows):
        for j in (cols[2 * k % len(cols)], cols[(2 * k + 1) % len(cols)]):
            lo[i, j] = -1 + np.floor((x[i, j] + 1) * 8) / 8
            hi[i, j] = lo[i, j] + 1 / 8
    mid = (lo + hi) / 2
    lam = 1 / 32
    beta = np.linalg.solve(mid.T @ mid / len(x) + lam * np.eye(x.shape[1]), mid.T @ y / len(x))
    cells = np.argwhere(lo != hi)
    cells = cells[np.random.default_rng(78043).permutation(len(cells))]
    gram = x.T @ x / len(x) + lam * np.eye(x.shape[1])
    fit = np.linalg.solve(gram, x.T @ y / len(x))
    d = beta - fit
    actual = float(d @ gram @ d) * 128 ** 2
    return ((lo, hi, y, beta, lam), x, cells, actual)

def main():
    psutil.Process().cpu_affinity([12])
    out = B / 'calibration_results.json'
    assert not out.exists()
    tab = pd.read_csv(DATA_CORE / 'superconductivity/train.csv')
    plan = prepare(tab)
    records = []
    conditions = []
    start = time.monotonic()
    payload = dict(status='RUNNING', protocol_sha256=hashlib.sha256((B / 'calibration_protocol.md').read_bytes()).hexdigest(), plan_file='calibration_plan.json', rank=3, cap_seconds=10, threshold_K2=0.25, cpu=12, threads=1, results=records, conditions=conditions)

    def persist():
        payload['elapsed_seconds'] = time.monotonic() - start
        save(out, payload)
    for pool in plan['target_pools']:
        for nr in (32, 64):
            for spread in ('concentrated', 'dispersed'):
                jid = len(conditions)
                try:
                    tick = time.monotonic()
                    args, true, cells, actual = problem(tab, pool, nr, spread, plan)
                    input_cost = time.monotonic() - tick
                    lo0, hi0, y, beta, lam = args
                    fp = fingerprint(args)
                    conditions.append(dict(job=jid, pool_seed=pool['seed'], uncertain_rows=nr, spread=spread, initial_cells=len(cells), input_sha256=fp, candidate=beta.tolist(), reveal_order_cells=cells.tolist()))
                    group = []
                    reveal = False
                    for cp in (0, 1):
                        if cp and (not reveal):
                            break
                        count = 0 if cp == 0 else len(cells) // 2
                        lo = lo0.copy()
                        hi = hi0.copy()
                        for i, j in cells[:count]:
                            lo[i, j] = hi[i, j] = true[i, j]
                        tick = time.monotonic()
                        a = ObservedAnchor(lo, hi, y, beta, lam, anchor='observed', preconditioner='midpoint')
                        anchor_cost = time.monotonic() - tick
                        common = input_cost + anchor_cost
                        base = dict(job=jid, pool_seed=pool['seed'], uncertain_rows=nr, spread=spread, checkpoint=cp, initial_cells=len(cells), remaining_cells=len(cells) - count, revealed_cells=count, input_sha256=fp, common_seconds=common, input_seconds=input_cost, anchor_seconds=anchor_cost, actual_completion_regret_K2=actual)
                        coarse = dict(seconds=common, lower=0.0, upper=float(a.gradient_upper), stage='shared_anchor')
                        ar = dict(**base, method='anchor', status='PASS', events=[coarse], charged_seconds=common, setup_seconds=0.0, search_seconds=0.0, execution='anchor_only')
                        ar.update(endpoint(ar['events'], 10.0))
                        records.append(ar)
                        group.append(ar)
                        order = list(METHODS)
                        offset = (jid + cp) % 4
                        order = order[offset:] + order[:offset]
                        for method in order:
                            rec = dict(**base, method=method)
                            events = [coarse.copy()]
                            mt = time.monotonic()
                            try:
                                if ar['decision'] == 'STOP':
                                    rec.update(rank=82 if method == 'full' else 3, eta=None, setup_seconds=0.0, search_seconds=0.0, execution='shared_anchor_stop')
                                else:
                                    if method == 'full':
                                        solver = MomentSearch(a, a.p)
                                    else:
                                        proposal, receipt = build_proposal(a, method, pool_seed=78041, pool_size=32, rank=3)
                                        with patch.object(moment_tail, 'basis_proposal', return_value=proposal):
                                            solver = MomentSearch(a, 3, kind=method)
                                    setup = time.monotonic() - mt
                                    offset = common + setup
                                    st = time.monotonic()
                                    if offset < CAP:
                                        result = solver.run(time_limit=CAP - offset, absolute_tolerance=1e-10, record=False)
                                        events.extend((dict(seconds=offset + v['seconds'], lower=float(v['lower']), upper=float(v['upper']), stage='search') for v in result['trace']))
                                        events.append(dict(seconds=common + time.monotonic() - mt, lower=float(result['lower']), upper=float(result['upper']), stage='final'))
                                    rec.update(rank=solver.rank, eta=float(solver.qtail), setup_seconds=setup, search_seconds=time.monotonic() - st, execution='search' if offset < CAP else 'setup_cap')
                                rec.update(status='PASS', events=events, charged_seconds=common + time.monotonic() - mt, checkpoints=[endpoint(events, h) for h in (1.0, 5.0, 10.0)])
                                rec.update(endpoint(events, 10.0))
                            except Exception as error:
                                rec.update(status='ERROR', error=str(error), traceback=traceback.format_exc())
                            records.append(rec)
                            group.append(rec)
                            persist()
                            print(jid, cp, method, rec['status'], rec.get('decision'), flush=True)
                        if cp == 0:
                            reveal = any((r.get('decision') == 'UNRESOLVED' for r in group if r['checkpoint'] == 0 and r['method'] in METHODS))
                    for method in ('anchor',) + METHODS:
                        rs = [r for r in group if r['method'] == method]
                        saved = max([r['remaining_cells'] for r in rs if r.get('decision') == 'STOP'], default=0)
                        for r in rs:
                            r['saved_cells_first_stop'] = saved
                except Exception as error:
                    records.append(dict(job=jid, status='ERROR', error=str(error), traceback=traceback.format_exc()))
                persist()
    payload['status'] = 'PASS' if all((r['status'] == 'PASS' for r in records)) else 'COMPLETED_WITH_ERRORS'
    persist()
    pd.DataFrame([{k: v for k, v in r.items() if k not in ('events', 'checkpoints', 'traceback')} for r in records]).to_csv(B / 'calibration_summary.csv', index=False)
if __name__ == '__main__':
    main()
