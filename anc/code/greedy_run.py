from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
import os
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
import sys, time, json, hashlib, traceback, csv
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode = True
B = Path(__file__).resolve().parent
ROOT = B.parent
import numpy as np, pandas as pd, psutil
from public_problems import problem
from observed_anchor import ObservedAnchor
import moment_tail
from moment_tail import MomentSearch
from greedy_basis import build_proposal

def confirmation_problem(name, spread, seed):
    if name != 'superconductivity':
        return problem(name=name, spread=spread, seed=seed)
    tab = pd.read_csv(DATA_CORE / 'superconductivity/train.csv')
    manifest = json.loads((B / 'greedy_pool.json').read_text())
    ids = manifest['original_row_indices']
    raw = tab.iloc[ids, :-1].to_numpy(float)
    yy = tab.iloc[ids, -1].to_numpy(float)
    span = np.ptp(raw, axis=0)
    span = np.where(span > 0, span, 1.0)
    x = np.column_stack([np.ones(len(raw)), np.round((raw - raw.mean(0)) / span * 1024) / 1024])
    y = np.round((yy - yy.min()) / np.ptp(yy) * 1024) / 1024
    columns = [1, 2] if spread == 'concentrated' else list(range(1, 9))
    maskrows = np.random.default_rng(seed).permutation(len(x))[:64]
    lo = x.copy()
    hi = x.copy()
    for k, i in enumerate(maskrows):
        for j in (columns[2 * k % len(columns)], columns[(2 * k + 1) % len(columns)]):
            left = -1 + np.floor((x[i, j] + 1) * 8) / 8
            lo[i, j] = left
            hi[i, j] = left + 1 / 8
    c = (lo + hi) / 2
    lam = 1 / 32
    beta = np.linalg.solve(c.T @ c / len(c) + lam * np.eye(c.shape[1]), c.T @ y / len(c))
    meta = dict(dataset=name, n=len(x), features=x.shape[1], uncertain_cells=128, uncertain_rows=64, seed=seed, spread=spread, width=1 / 8, **{'lambda': lam}, pool_seed=78001, original_row_indices=ids)
    return ((lo, hi, y, beta, lam), meta)
