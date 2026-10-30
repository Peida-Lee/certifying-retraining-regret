from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Fixed public full-feature designs; no data-dependent method selection.'
from pathlib import Path
import numpy as np
from sklearn.datasets import load_diabetes
BASE = Path(__file__).resolve().parent

def dataset(name, subset_seed=74001):
    if name == 'diabetes':
        o = load_diabetes()
        xx = o.data
        yy = o.target
        pool = [0, 2, 3, 4, 5, 6, 7, 8, 9]
        names = o.feature_names
    elif name == 'concrete':
        import pandas as pd
        tab = pd.read_excel(DATA_CORE / 'concrete/Concrete_Data.xls')
        xx = tab.iloc[:, :-1].to_numpy(float)
        yy = tab.iloc[:, -1].to_numpy(float)
        pool = list(range(xx.shape[1]))
        names = list(tab.columns[:-1])
    elif name == 'superconductivity':
        import pandas as pd
        tab = pd.read_csv(DATA_CORE / 'superconductivity/train.csv')
        development = np.sort(np.random.default_rng(74001).choice(len(tab), 1024, replace=False))
        if subset_seed == 74001:
            rows = development
        else:
            remaining = np.setdiff1d(np.arange(len(tab)), development)
            rows = np.sort(np.random.default_rng(subset_seed).choice(remaining, 1024, replace=False))
        xx = tab.iloc[rows, :-1].to_numpy(float)
        yy = tab.iloc[rows, -1].to_numpy(float)
        pool = list(range(xx.shape[1]))
        names = list(tab.columns[:-1])
    else:
        raise ValueError(name)
    span = np.ptp(xx, axis=0)
    span = np.where(span > 0, span, 1.0)
    x = np.round((xx - xx.mean(0)) / span * 1024) / 1024
    y = np.round((yy - yy.min()) / np.ptp(yy) * 1024) / 1024
    return (np.column_stack([np.ones(len(x)), x]), y, [j + 1 for j in pool], ['intercept'] + list(names))

def problem(name='diabetes', cells=128, width=1 / 8, seed=74011, spread='concentrated', per_row=2, subset_seed=74001):
    x, y, pool, names = dataset(name, subset_seed)
    if name == 'diabetes':
        concentrated = [3, 4]
    else:
        concentrated = pool[:2]
    columns = concentrated if spread == 'concentrated' else pool[:min(8, len(pool))]
    lo = x.copy()
    hi = x.copy()
    rows = np.random.default_rng(seed).permutation(len(x))[:cells // per_row]
    for k, i in enumerate(rows):
        chosen = [columns[(per_row * k + j) % len(columns)] for j in range(per_row)]
        for j in chosen:
            left = -1 + np.floor((x[i, j] + 1) / width) * width
            lo[i, j] = left
            hi[i, j] = left + width
    c = (lo + hi) / 2
    lam = 1 / 32
    beta = np.linalg.solve(c.T @ c / len(x) + lam * np.eye(x.shape[1]), c.T @ y / len(x))
    meta = {'dataset': name, 'n': len(x), 'features': x.shape[1], 'uncertain_cells': int(np.sum(lo != hi)), 'uncertain_columns': len(columns), 'uncertain_rows': len(rows), 'per_row': per_row, 'width': width, 'seed': seed, 'spread': spread, 'lambda': lam, 'features_names': names, 'subset_seed': subset_seed if name == 'superconductivity' else None}
    return ((lo, hi, y, beta, lam), meta)
