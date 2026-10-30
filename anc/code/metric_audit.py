from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Independent numerical metric-proposal replay and deadline arithmetic audit.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import sys, json, hashlib, math, time, csv
from pathlib import Path
import numpy as np, psutil
B = Path(__file__).resolve().parent
ROOT = B.parent
from greedy_audit import inputs

def main():
    psutil.Process().cpu_affinity([10])
    start = time.monotonic()
    p = json.loads((B / 'metric_results.json').read_text())
    assert p['status'] == 'PASS' and len(p['results']) == 54 and (not p['errors'])
    assert p['protocol_sha256'] == hashlib.sha256((B / 'metric_protocol.md').read_bytes()).hexdigest()
    cases = []
    geometry = []
    maxind = maxspan = maxspec = 0.0
    eligible = 0
    for i in range(18):
        rows = [r for r in p['results'] if r['condition'] == i]
        assert len(rows) == 3
        first = rows[0]
        lo, hi, y, beta, lam = inputs(first['dataset'], first['spread'], first['seed'])
        n, dim = lo.shape
        c = (lo + hi) / 2
        t = np.round(np.linalg.inv(np.linalg.cholesky(c.T @ c / n + lam * np.eye(dim)).T) * 2 ** 20) / 2 ** 20
        rng = np.random.default_rng(78041)
        xs = [c] + [lo + (hi - lo) * rng.random(lo.shape) for _ in range(31)]
        ms = [x.T @ x / n + lam * np.eye(dim) for x in xs]
        rhs = [x.T @ y / n for x in xs]
        gs = [m @ beta - b for m, b in zip(ms, rhs)]
        ks = [t.T @ m @ t for m in ms]
        bs = [t.T @ g for g in gs]
        physical = np.column_stack([beta - np.linalg.solve(m, b) for m, b in zip(ms, rhs)])
        solver = np.linalg.solve(t, physical)
        u, sv, _ = np.linalg.svd(solver, full_matrices=False)
        pod = next((r for r in rows if r['method'] == 'metric_pod'))
        pr = pod['proposal']
        q = np.array(pr['unrounded_solver_basis'])
        assert np.allclose(physical, pr['physical_displacements'], rtol=1e-10, atol=1e-14) and np.allclose(solver, pr['solver_displacements'], rtol=1e-10, atol=1e-14)
        assert np.linalg.norm(q @ q.T - u[:, :3] @ u[:, :3].T) < 1e-08 and np.allclose(sv, pr['singular_values'], rtol=1e-10, atol=1e-14)
        spectralerror = abs(float(np.linalg.norm(solver - q @ (q.T @ solver)) ** 2) - float(np.sum(sv[3:] ** 2)))
        maxspec = max(maxspec, spectralerror)
        assert spectralerror < 1e-12
        physical_u, _, _ = np.linalg.svd(physical, full_matrices=False)
        physical_numeric, _ = np.linalg.qr(np.linalg.solve(t, physical_u[:, :3]))
        totalenergy = float(np.linalg.norm(solver) ** 2)
        metric_error = float(np.linalg.norm(solver - q @ (q.T @ solver)) ** 2)
        physical_error = float(np.linalg.norm(solver - physical_numeric @ (physical_numeric.T @ solver)) ** 2)
        cosines = np.linalg.svd(q.T @ physical_numeric, compute_uv=False)
        assert metric_error <= physical_error + 1e-12
        geometry.append(dict(condition=i, dataset=first['dataset'], spread=first['spread'], seed=first['seed'], solver_displacement_energy=totalenergy, metric_POD_rank3_residual_energy=metric_error, physical_POD_rank3_residual_solver_energy=physical_error, metric_capture_fraction=1 - metric_error / totalenergy, physical_capture_fraction_in_solver_metric=1 - physical_error / totalenergy, smallest_principal_angle_cosine=float(cosines[-1])))
        assert pr['fullfit_count'] == 32
        gr = next((r for r in rows if r['method'] == 'metric_greedy'))
        grec = gr['proposal']
        mu = gr['mu']
        q = np.empty((dim, 0))
        selected = []
        fixed = np.all(lo == hi, axis=1)
        g0 = lam * np.eye(dim) + c[fixed].T @ c[fixed] / n
        assert mu > 0 and mu <= np.linalg.eigvalsh(t.T @ g0 @ t)[0] + 1e-12
        for step in grec['steps']:
            vals = []
            for k, b in zip(ks, bs):
                v = q @ np.linalg.solve(q.T @ k @ q, q.T @ b) if q.shape[1] else np.zeros(dim)
                residual = b - k @ v
                vals.append(float(residual @ residual / mu))
            delta = float(np.max(abs(np.array(vals) - step['indicators'])))
            maxind = max(maxind, delta)
            assert np.allclose(vals, step['indicators'], rtol=1e-10, atol=1e-18)
            chosen = int(np.argmax(vals))
            assert chosen == step['selected']
            selected.append(chosen)
            d = beta - np.linalg.solve(ms[chosen], rhs[chosen])
            z = np.linalg.solve(t, d)
            assert np.allclose(d, step['physical_displacement'], rtol=1e-10, atol=1e-14) and np.allclose(z, step['solver_displacement'], rtol=1e-10, atol=1e-14)
            assert np.linalg.norm(ms[chosen] @ (beta - d) - rhs[chosen]) < 1e-12
            for _ in range(2):
                z -= q @ (q.T @ z)
            q = np.column_stack([q, z / np.linalg.norm(z)])
        assert selected == grec['selected_indices'] and grec['fullfit_count'] == 3
        assert np.linalg.norm(q @ q.T - np.array(grec['unrounded_solver_basis']) @ np.array(grec['unrounded_solver_basis']).T) < 1e-08
        common = first['common_seconds']
        anchor = first['events'][0]
        for r in rows:
            assert r['status'] == 'PASS' and r['common_seconds'] == common and (r['events'][0] == anchor)
            assert r['basis_Hmu'] > 0 and math.isclose(r['tail'], (r['G_r'] + r['C_r'] * r['R_s']) ** 2 / r['mu'], rel_tol=1e-12)
            rec = r['proposal']
            numeric = np.array(rec['numerical_basis'])
            assert np.all(numeric * 2 ** 24 == np.round(numeric * 2 ** 24)) and np.linalg.matrix_rank(numeric) == 3
            if r['method'] != 'gradient':
                qq = np.array(rec['unrounded_solver_basis'])
                assert np.linalg.norm(qq.T @ qq - np.eye(3)) < 1e-08
                assert np.array_equal(numeric, np.round(qq * 2 ** 24) / 2 ** 24)
                phy = t @ qq
                actual = t @ numeric
                pp, _ = np.linalg.qr(actual)
                err = float(np.linalg.norm(phy - pp @ (pp.T @ phy)) / np.linalg.norm(phy))
                maxspan = max(maxspan, err)
                assert err < 1e-05
            for cp in r['checkpoints']:
                valid = [e for e in r['events'] if e['seconds'] <= cp['budget_seconds']]
                if valid:
                    lower = max((e['lower'] for e in valid))
                    upper = min((e['upper'] for e in valid))
                    assert lower <= upper
                    assert cp['lower'] == lower and cp['upper'] == upper and (cp['interval_width'] == upper - lower)
                    assert cp['last_event_seconds'] <= cp['budget_seconds']
                    eligible += 1
                else:
                    assert cp['lower'] is None and cp['upper'] is None and (cp['interval_width'] is None)
                assert cp['evidence_events'] == len(valid)
        cases.append(dict(condition=i, dataset=first['dataset'], spread=first['spread'], seed=first['seed'], metric_greedy_selected_indices=selected, status='PASS'))
    out = dict(status='PASS', conditions=18, method_runs=54, checkpoints_checked=162, finite_checkpoints=eligible, selection_steps_replayed=54, max_indicator_absolute_difference=maxind, max_POD_rank3_spectral_error=maxspec, max_physical_span_relative_residual=maxspan, common_anchor_identical_within_condition=True, all_deadlines_reclassified_from_completed_events=True, seconds=time.monotonic() - start, scope='Independent numerical proposal replay, spectrum, metric/physical spans, deadlines and aggregate arithmetic; no independent full outward search-tree audit.', cases=cases)
    (B / 'metric_audit.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(out, indent=2))
    with (B / 'metric_geometry.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(geometry[0]))
        w.writeheader()
        w.writerows(geometry)
if __name__ == '__main__':
    main()
