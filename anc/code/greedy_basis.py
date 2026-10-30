from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Numerical proposals only; validity comes from the unchanged moment verifier.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import sys, time
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
from moment_tail import basis_proposal

def completion_pool(a, seed=78041, size=32):
    rng = np.random.default_rng(seed)
    return [a.c.copy()] + [a.lo + (a.hi - a.lo) * rng.random(a.lo.shape) for _ in range(size - 1)]

def build_proposal(anchor, method, pool_seed=78041, pool_size=32, rank=3):
    a = anchor
    start = time.monotonic()
    receipt = dict(method=method, rank=rank, pool_seed=pool_seed, pool_size=pool_size, pool_seconds=0.0, moment_seconds=0.0, fit_seconds=0.0, indicator_seconds=0.0, mapping_seconds=0.0, fullfit_count=0, selected_indices=[], steps=[])
    if method == 'gradient':
        r = basis_proposal(a, 'gradient')[:, :rank]
        receipt.update(basis_seconds=time.monotonic() - start, numerical_basis=r.tolist())
        return (r, receipt)
    tick = time.monotonic()
    xs = completion_pool(a, pool_seed, pool_size)
    receipt['pool_seconds'] = time.monotonic() - tick
    tick = time.monotonic()
    ms = [x.T @ x / a.n + a.lam * np.eye(a.p) for x in xs]
    rhs = [x.T @ a.y / a.n for x in xs]
    gs = [m @ a.beta - b for m, b in zip(ms, rhs)]
    receipt['moment_seconds'] = time.monotonic() - tick
    if method == 'snapshot_pod':
        tick = time.monotonic()
        d = np.column_stack([a.beta - np.linalg.solve(m, b) for m, b in zip(ms, rhs)])
        receipt['fit_seconds'] = time.monotonic() - tick
        receipt['fullfit_count'] = pool_size
        tick = time.monotonic()
        q, s, _ = np.linalg.svd(d, full_matrices=False)
        q = q[:, :rank]
        receipt['indicator_seconds'] = time.monotonic() - tick
        receipt['snapshot_singular_values'] = s.tolist()
    elif method == 'residual_greedy':
        q = np.empty((a.p, 0))
        for step in range(rank):
            tick = time.monotonic()
            ind = []
            for m, g in zip(ms, gs):
                v = q @ np.linalg.solve(q.T @ m @ q, q.T @ g) if q.shape[1] else np.zeros(a.p)
                res = g - m @ v
                ind.append(float(res @ res / a.lam))
            chosen = int(np.argmax(ind))
            receipt['indicator_seconds'] += time.monotonic() - tick
            tick = time.monotonic()
            displacement = a.beta - np.linalg.solve(ms[chosen], rhs[chosen])
            receipt['fit_seconds'] += time.monotonic() - tick
            tick = time.monotonic()
            v = displacement.copy()
            for _ in range(2):
                v -= q @ (q.T @ v)
            norm = float(np.linalg.norm(v))
            if norm < 1e-14:
                raise ValueError('Greedy selected numerically dependent displacement')
            q = np.column_stack([q, v / norm])
            receipt['indicator_seconds'] += time.monotonic() - tick
            receipt['selected_indices'].append(chosen)
            receipt['steps'].append(dict(step=step + 1, indicators=ind, selected=chosen, selected_displacement=displacement.tolist(), orthogonalized_norm=norm))
        receipt['fullfit_count'] = rank
    else:
        raise ValueError(method)
    tick = time.monotonic()
    mapped = np.linalg.solve(a.t, q)
    mapped, _ = np.linalg.qr(mapped, mode='reduced')
    r = np.round(mapped * 2 ** 24) / 2 ** 24
    physical = a.t @ r
    physq, _ = np.linalg.qr(physical, mode='reduced')
    span = float(np.linalg.norm(q - physq @ (physq.T @ q)))
    assert span < 1e-05 and np.linalg.matrix_rank(r) == rank
    receipt['mapping_seconds'] = time.monotonic() - tick
    receipt.update(physical_basis=q.tolist(), numerical_basis=r.tolist(), physical_span_residual=span, physical_orthogonality_error=float(np.linalg.norm(q.T @ q - np.eye(rank))), basis_seconds=time.monotonic() - start)
    return (r, receipt)
