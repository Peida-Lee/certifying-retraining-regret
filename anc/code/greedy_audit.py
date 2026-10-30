from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Independent selection replay: never imports greedy_basis or greedy_run.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, hashlib, math, time
from pathlib import Path
import numpy as np, pandas as pd, psutil
from sklearn.datasets import load_diabetes
B = Path(__file__).resolve().parent
ROOT = B.parent

def inputs(name, pattern, seed):
    if name == 'diabetes':
        o = load_diabetes()
        raw = o.data
        yy = o.target
        cols = [3, 4] if pattern == 'concentrated' else [1, 3, 4, 5, 6, 7, 8, 9]
    elif name == 'concrete':
        t = pd.read_excel(DATA_CORE / 'concrete/Concrete_Data.xls')
        raw = t.iloc[:, :-1].to_numpy(float)
        yy = t.iloc[:, -1].to_numpy(float)
        cols = [1, 2] if pattern == 'concentrated' else list(range(1, 9))
    else:
        t = pd.read_csv(DATA_CORE / 'superconductivity/train.csv')
        ids = json.loads((B / 'greedy_pool.json').read_text())['original_row_indices']
        raw = t.iloc[ids, :-1].to_numpy(float)
        yy = t.iloc[ids, -1].to_numpy(float)
        cols = [1, 2] if pattern == 'concentrated' else list(range(1, 9))
    span = np.ptp(raw, axis=0)
    span = np.where(span > 0, span, 1.0)
    x = np.column_stack([np.ones(len(raw)), np.round((raw - raw.mean(0)) / span * 1024) / 1024])
    y = np.round((yy - yy.min()) / np.ptp(yy) * 1024) / 1024
    lo = x.copy()
    hi = x.copy()
    rows = np.random.default_rng(seed).permutation(len(x))[:64]
    for k, i in enumerate(rows):
        for j in (cols[2 * k % len(cols)], cols[(2 * k + 1) % len(cols)]):
            lo[i, j] = -1 + np.floor((x[i, j] + 1) * 8) / 8
            hi[i, j] = lo[i, j] + 1 / 8
    c = (lo + hi) / 2
    lam = 1 / 32
    beta = np.linalg.solve(c.T @ c / len(x) + lam * np.eye(x.shape[1]), c.T @ y / len(x))
    return (lo, hi, y, beta, lam)

def main():
    psutil.Process().cpu_affinity([10])
    start = time.monotonic()
    p = json.loads((B / 'greedy_results.json').read_text())
    assert p['status'] == 'PASS' and len(p['results']) == 54 and (not p['errors'])
    assert p['protocol_sha256'] == hashlib.sha256((B / 'greedy_protocol.md').read_bytes()).hexdigest()
    pool = json.loads((B / 'greedy_pool.json').read_text())
    assert len(pool['original_row_indices']) == 1024 and (not set(pool['original_row_indices']) & set(pool['excluded_rows']))
    n = len(pd.read_csv(DATA_CORE / 'superconductivity/train.csv'))
    allrows = np.arange(n)
    dev = np.sort(np.random.default_rng(74001).choice(allrows, 1024, replace=False))
    old = np.sort(np.random.default_rng(74101).choice(np.setdiff1d(allrows, dev), 1024, replace=False))
    train = np.sort(np.random.default_rng(75001).choice(np.setdiff1d(allrows, np.r_[dev, old]), 1024, replace=False))
    test = np.sort(np.random.default_rng(77031).choice(np.setdiff1d(allrows, np.r_[dev, old, train]), 1024, replace=False))
    excluded = np.unique(np.r_[dev, old, train, test])
    assert np.array_equal(excluded, pool['excluded_rows'])
    chosen = np.sort(np.random.default_rng(78001).choice(np.setdiff1d(allrows, excluded), 1024, replace=False))
    assert np.array_equal(chosen, pool['original_row_indices'])
    replay = []
    maximum_indicator_difference = maximum_span_difference = 0.0
    for idx in range(18):
        rr = [r for r in p['results'] if r['condition'] == idx]
        assert len(rr) == 3
        a = rr[0]
        lo, hi, y, beta, lam = inputs(a['dataset'], a['spread'], a['seed'])
        n, dim = lo.shape
        gen = np.random.default_rng(78041)
        xx = [(lo + hi) / 2] + [lo + (hi - lo) * gen.random(lo.shape) for _ in range(31)]
        ms = [x.T @ x / n + lam * np.eye(dim) for x in xx]
        rhs = [x.T @ y / n for x in xx]
        gs = [m @ beta - b for m, b in zip(ms, rhs)]
        g = next((r for r in rr if r['method'] == 'residual_greedy'))
        record = g['proposal']
        q = np.empty((dim, 0))
        selected = []
        for step in record['steps']:
            vals = []
            for m, gradient in zip(ms, gs):
                z = np.linalg.solve(q.T @ m @ q, q.T @ gradient) if q.shape[1] else np.empty(0)
                residual = gradient - m @ q @ z
                vals.append(float(residual @ residual / lam))
            delta = float(np.max(np.abs(np.array(vals) - step['indicators'])))
            maximum_indicator_difference = max(maximum_indicator_difference, delta)
            assert np.allclose(vals, step['indicators'], rtol=1e-10, atol=1e-18)
            picked = int(np.argmax(vals))
            assert picked == step['selected']
            selected.append(picked)
            d = beta - np.linalg.solve(ms[picked], rhs[picked])
            assert np.allclose(d, step['selected_displacement'], rtol=1e-10, atol=1e-14)
            assert np.linalg.norm(ms[picked] @ (beta - d) - rhs[picked]) < 1e-12
            for _ in range(2):
                d -= q @ (q.T @ d)
            q = np.column_stack([q, d / np.linalg.norm(d)])
        assert selected == record['selected_indices'] and record['fullfit_count'] == 3
        assert np.linalg.norm(q @ q.T - np.array(record['physical_basis']) @ np.array(record['physical_basis']).T) < 1e-08
        c = (lo + hi) / 2
        mid = c.T @ c / n + lam * np.eye(dim)
        t = np.round(np.linalg.inv(np.linalg.cholesky(mid).T) * 2 ** 20) / 2 ** 20
        for r in rr:
            z = r['proposal']
            numeric = np.array(z['numerical_basis'])
            assert np.all(numeric * 2 ** 24 == np.round(numeric * 2 ** 24)) and np.linalg.matrix_rank(numeric) == 3
            assert r['basis_Hmu'] > 0 and r['lower'] <= r['upper']
            assert math.isclose(r['interval_width'], r['upper'] - r['lower'], rel_tol=1e-12, abs_tol=1e-18)
            assert math.isclose(r['tail'], (r['G_r'] + r['C_r'] * r['R_s']) ** 2 / r['mu'], rel_tol=1e-12)
            assert r['total_seconds'] == sum((r[k] for k in ('input_seconds', 'anchor_seconds', 'basis_seconds', 'solver_setup_seconds', 'search_seconds')))
            if r['method'] != 'gradient':
                physical = np.array(z['physical_basis'])
                mapped = t @ numeric
                qr, _ = np.linalg.qr(mapped)
                err = float(np.linalg.norm(physical - qr @ (qr.T @ physical)))
                maximum_span_difference = max(maximum_span_difference, err)
                assert err < 1e-05
                assert np.linalg.norm(physical.T @ physical - np.eye(3)) < 1e-08
                if r['method'] == 'snapshot_pod':
                    ds = np.column_stack([beta - np.linalg.solve(m, b) for m, b in zip(ms, rhs)])
                    u, sv, _ = np.linalg.svd(ds, full_matrices=False)
                    assert np.linalg.norm(u[:, :3] @ u[:, :3].T - physical @ physical.T) < 1e-08 and z['fullfit_count'] == 32
        replay.append(dict(condition=idx, dataset=a['dataset'], spread=a['spread'], seed=a['seed'], selected_indices=selected, status='PASS'))
    out = dict(status='PASS', method_runs=54, conditions=18, selection_steps_replayed=54, max_indicator_absolute_difference=maximum_indicator_difference, max_physical_span_residual=maximum_span_difference, pool_ids_and_exclusions_recreated=True, arithmetic_rows_checked=54, scope='Independent numerical proposal selection, normal equations, spans, metadata and aggregate arithmetic; not an independent outward search-tree audit.', seconds=time.monotonic() - start, replays=replay)
    (B / 'greedy_audit.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
if __name__ == '__main__':
    main()
