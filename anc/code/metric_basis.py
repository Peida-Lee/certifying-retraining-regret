from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Metric-matched numerical proposals; uniform validity is checked separately.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import sys, time
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
import numpy as np
from greedy_basis import build_proposal as physical_build, completion_pool

def build_proposal(a, method, pool_seed=78041, pool_size=32, rank=3):
    if method in ('gradient', 'physical_pod', 'physical_greedy'):
        r, z = physical_build(a, {'physical_pod': 'snapshot_pod', 'physical_greedy': 'residual_greedy'}.get(method, method), pool_seed, pool_size, rank)
        z['method'] = method
        return (r, z)
    start = time.monotonic()
    tick = start
    xs = completion_pool(a, pool_seed, pool_size)
    pooltime = time.monotonic() - tick
    tick = time.monotonic()
    ms = [x.T @ x / a.n + a.lam * np.eye(a.p) for x in xs]
    rhs = [x.T @ a.y / a.n for x in xs]
    gs = [m @ a.beta - b for m, b in zip(ms, rhs)]
    ks = [a.t.T @ m @ a.t for m in ms]
    bs = [a.t.T @ g for g in gs]
    momenttime = time.monotonic() - tick
    rec = dict(method=method, rank=rank, pool_seed=pool_seed, pool_size=pool_size, pool_seconds=pooltime, moment_seconds=momenttime, fit_seconds=0.0, indicator_seconds=0.0, coordinate_transform_seconds=0.0, rounding_seconds=0.0, fullfit_count=0, steps=[], selected_indices=[])
    if method == 'metric_pod':
        tick = time.monotonic()
        d = np.column_stack([a.beta - np.linalg.solve(m, b) for m, b in zip(ms, rhs)])
        rec['fit_seconds'] = time.monotonic() - tick
        tick = time.monotonic()
        z = np.linalg.solve(a.t, d)
        rec['coordinate_transform_seconds'] = time.monotonic() - tick
        tick = time.monotonic()
        q, sv, _ = np.linalg.svd(z, full_matrices=False)
        q = q[:, :rank]
        rec['indicator_seconds'] = time.monotonic() - tick
        rec.update(fullfit_count=pool_size, singular_values=sv.tolist(), physical_displacements=d.tolist(), solver_displacements=z.tolist())
    elif method == 'metric_greedy':
        q = np.empty((a.p, 0))
        for i in range(rank):
            tick = time.monotonic()
            vals = []
            for k, b in zip(ks, bs):
                v = q @ np.linalg.solve(q.T @ k @ q, q.T @ b) if q.shape[1] else np.zeros(a.p)
                residual = b - k @ v
                vals.append(float(residual @ residual / a.mu))
            chosen = int(np.argmax(vals))
            rec['indicator_seconds'] += time.monotonic() - tick
            tick = time.monotonic()
            d = a.beta - np.linalg.solve(ms[chosen], rhs[chosen])
            rec['fit_seconds'] += time.monotonic() - tick
            tick = time.monotonic()
            z = np.linalg.solve(a.t, d)
            rec['coordinate_transform_seconds'] += time.monotonic() - tick
            tick = time.monotonic()
            v = z.copy()
            for _ in range(2):
                v -= q @ (q.T @ v)
            norm = float(np.linalg.norm(v))
            if norm < 1e-14:
                raise ValueError('Metric greedy selected dependent displacement')
            q = np.column_stack([q, v / norm])
            rec['indicator_seconds'] += time.monotonic() - tick
            rec['selected_indices'].append(chosen)
            rec['steps'].append(dict(step=i + 1, indicators=vals, selected=chosen, physical_displacement=d.tolist(), solver_displacement=z.tolist(), orthogonalized_norm=norm))
        rec['fullfit_count'] = rank
    else:
        raise ValueError(method)
    tick = time.monotonic()
    r = np.round(q * 2 ** 24) / 2 ** 24
    rec['rounding_seconds'] = time.monotonic() - tick
    assert np.linalg.matrix_rank(r) == rank
    phy = a.t @ q
    actual = a.t @ r
    pq, _ = np.linalg.qr(actual)
    span = float(np.linalg.norm(phy - pq @ (pq.T @ phy)) / max(np.linalg.norm(phy), 1e-300))
    assert span < 1e-05
    rec.update(unrounded_solver_basis=q.tolist(), numerical_basis=r.tolist(), solver_orthogonality_error=float(np.linalg.norm(q.T @ q - np.eye(rank))), physical_span_relative_residual=span, basis_seconds=time.monotonic() - start)
    return (r, rec)
