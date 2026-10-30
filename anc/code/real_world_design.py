from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Real-world interval construction with fixed residual row rules.'
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import sys, time, json, itertools, traceback, csv
import numpy as np
import psutil
B = Path(__file__).resolve().parent
from datasets import dataset

def square_product_range(pc, pg, ec, eg):
    aa = pg * eg
    bb = pc * eg + pg * ec
    cc = pc * ec
    vv = [aa - bb + cc, aa + bb + cc]
    if aa and -1 < -bb / (2 * aa) < 1:
        vv.append(cc - bb * bb / (4 * aa))
    return (min(vv), max(vv))

def problem(name, lam, bits, m, width, spread, seed=84031, pool_seed=84001, frozen_columns=None, eligible_factor=4, iterations=0, nuisance_width=None, candidate_beta=None):
    x, y, names = dataset(name, pool_seed)
    n, p = x.shape
    nw = width if nuisance_width is None else nuisance_width
    fit = np.linalg.solve(x.T @ x / n + lam * np.eye(p), x.T @ y / n)
    if frozen_columns is None:
        order = np.argsort(abs(fit[1:])) + 1
        ns = order[:min(spread, p - 2)]
        sj = int(order[-1])
    else:
        ns = np.array(frozen_columns['nuisance'], int)
        sj = int(frozen_columns['signal'])
    cols = np.r_[ns, sj]
    left = -1 + np.floor((x[:, cols] + 1) / width) * width
    pilot_x = x.copy()
    pilot_x[:, cols] = left + width / 2
    pilot = np.linalg.solve(pilot_x.T @ pilot_x / n + lam * np.eye(p), pilot_x.T @ y / n)
    sr = abs(pilot_x @ pilot - y)
    signal = int(np.argmax(sr))
    eligible = np.argsort(sr)[:min(n - 1, eligible_factor * m)]
    eligible = eligible[eligible != signal]
    rows = np.random.default_rng(seed).choice(eligible, m, replace=False)
    lo = x.copy()
    hi = x.copy()
    lo[signal, sj] = left[signal, -1]
    hi[signal, sj] = left[signal, -1] + width
    for k, i in enumerate(rows):
        cc = [int(ns[(2 * k + t) % len(ns)]) for t in range(min(2, len(ns)))]
        for j in cc:
            l = -1 + np.floor((x[i, j] + 1) / width) * width
            lo[i, j] = l
            hi[i, j] = l + width
    for iteration in range(iterations):
        c = (lo + hi) / 2
        fixed = np.all(lo == hi, axis=1)
        reference = np.linalg.solve(c[fixed].T @ c[fixed] / n + lam * np.eye(p), c[fixed].T @ y[fixed] / n)
        prospective = x.copy()
        for i in range(n):
            for j in [int(ns[(2 * i + t) % len(ns)]) for t in range(min(2, len(ns)))]:
                ll = -1 + np.floor((x[i, j] + 1) / nw) * nw
                prospective[i, j] = ll + nw / 2
        scores = abs(prospective @ reference - y)
        for i in range(n):
            scores[i] += nw / 2 * sum((abs(reference[int(ns[(2 * i + t) % len(ns)])]) for t in range(min(2, len(ns)))))
        scores[signal] = np.inf
        eligible = np.argsort(scores)[:eligible_factor * m]
        rows = np.random.default_rng(seed).choice(eligible, m, replace=False)
        lo = x.copy()
        hi = x.copy()
        lo[signal, sj] = left[signal, -1]
        hi[signal, sj] = left[signal, -1] + width
        for i in rows:
            for j in [int(ns[(2 * i + t) % len(ns)]) for t in range(min(2, len(ns)))]:
                ll = -1 + np.floor((x[i, j] + 1) / nw) * nw
                lo[i, j] = ll
                hi[i, j] = ll + nw
    c = (lo + hi) / 2
    beta = np.linalg.solve(c.T @ c / n + lam * np.eye(p), c.T @ y / n) if candidate_beta is None else np.array(candidate_beta,float).copy()
    if bits and candidate_beta is None:
        beta = np.round(beta * 2 ** bits) / 2 ** bits
    meta = dict(dataset=name, n=n, p=p, lambda_value=lam, candidate_bits=bits, uncertain_nuisance_rows=m, uncertain_rows=m + 1, uncertain_cells=int(sum((lo != hi).flat)), width=width, nuisance_width=nw, spread=spread, seed=seed, pool_seed=pool_seed, signal_row=signal, signal_column=sj, nuisance_columns=ns.tolist(), row_ids=rows.tolist(), column_names=[names[j] for j in cols], pilot_uses_only_allowed_coarsened_inputs=True, eligible_factor=eligible_factor, selection_iterations=iterations)
    return ((lo, hi, y, beta, lam), meta)
