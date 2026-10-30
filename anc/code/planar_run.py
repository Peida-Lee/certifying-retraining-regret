from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Frozen, bounded planar row and full-feature certification experiment.'
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import ctypes
ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
ctypes.windll.kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(), 1 << 8)
from planar_oracle import *
from itertools import product
import json, time, traceback, statistics
import observed_anchor as old
from moment_tail import MomentSearch
from sklearn.datasets import load_diabetes
OUT = BASE / 'planar_results.json'
results = {'protocol': 'planar_protocol.md', 'cpu': 8, 'threads': 1, 'exact_checks': [], 'timings': [], 'certifications': [], 'errors': []}

def save():
    OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')

def exhaustive(ac, bc, gg, gamma):
    best = None
    for j in range(len(gg)):
        others = [l for l in range(len(gg)) if l != j]
        for bits in product((-1, 1), repeat=len(others)):
            a = ac + sum((gg[l][0] * s for l, s in zip(others, bits)), Q(0))
            b = bc + sum((gg[l][1] * s for l, s in zip(others, bits)), Q(0))
            x, y = gg[j]
            aa = x * x - gamma * y * y
            bb = 2 * (a * x - gamma * b * y)
            cc = a * a - gamma * b * b
            vals = [aa - bb + cc, aa + bb + cc]
            if aa < 0 and -1 < -bb / (2 * aa) < 1:
                vals.append(cc - bb * bb / (4 * aa))
            val = max(vals)
            best = val if best is None else max(val, best)
    return best
rng = np.random.default_rng(77103)
for k in (1, 2, 4, 8):
    for case in range(3):
        try:
            ac = Q(int(rng.integers(-8, 9)), 8)
            bc = Q(int(rng.integers(-8, 9)), 8)
            gamma = Q(3, 4)
            gg = [(Q(int(rng.integers(-8, 9)), 8), Q(int(rng.integers(-8, 9)), 8)) for _ in range(k)]
            if case == 1:
                gg = [(Q(j, 8), Q(2 * j, 8)) for j in range(k)]
            if case == 2:
                ac = Q(0)
                bc = Q(1, 4)
                gg = [(Q(0), Q(j + 1, 8)) for j in range(k)]
            t = time.perf_counter()
            v, z = exact_row_max(ac, bc, gg, gamma, True)
            pt = time.perf_counter() - t
            t = time.perf_counter()
            ref = exhaustive(ac, bc, gg, gamma)
            rt = time.perf_counter() - t
            assert v == ref
            results['exact_checks'].append({'k': k, 'case': case, 'exact_equal': True, 'maximum': str(v), 'witness': [str(x) for x in z], 'planar_seconds': pt, 'box_seconds': rt, 'interior': any((-1 < x < 1 for x in z))})
        except Exception:
            results['errors'].append({'stage': 'exact', 'k': k, 'case': case, 'error': traceback.format_exc()})
        save()
for k in (4, 8, 16):
    try:
        gg = [(Q(int(rng.integers(-8, 9)), 8), Q(int(rng.integers(-8, 9)), 8)) for _ in range(k)]
        row = {'k': k, 'coefficient_center': ['1/8', '-1/4'], 'generators': [[str(a), str(b)] for a, b in gg]}
        for name, fn in [('planar', lambda: exact_row_max(Q(1, 8), Q(-1, 4), gg, Q(1), True)), ('box', lambda: exhaustive(Q(1, 8), Q(-1, 4), gg, Q(1)))]:
            if name == 'box' and k > 8:
                continue
            times = []
            for rep in range(3):
                t = time.perf_counter()
                fn()
                times.append(time.perf_counter() - t)
            row[name + '_seconds'] = times
            row[name + '_median'] = statistics.median(times)
        results['timings'].append(row)
    except Exception:
        results['errors'].append({'stage': 'timing', 'k': k, 'error': traceback.format_exc()})
    save()
o = load_diabetes()
xx = o.data
xx = np.round((xx - xx.mean(0)) / np.ptp(xx, axis=0) * 1024) / 1024
x = np.column_stack([np.ones(len(xx)), xx])
y = np.round((o.target - o.target.min()) / np.ptp(o.target) * 1024) / 1024
for k, oracles in [(2, ['box', 'planar']), (10, ['planar'])]:
    lo = x.copy()
    hi = x.copy()
    lo[:4, 1:1 + k] -= 1 / 16
    hi[:4, 1:1 + k] += 1 / 16
    lam = 1 / 32
    c = (lo + hi) / 2
    beta = np.linalg.solve(c.T @ c / len(x) + lam * np.eye(x.shape[1]), c.T @ y / len(x))
    for name in oracles:
        try:
            t = time.perf_counter()
            a = (old.ObservedAnchor if name == 'box' else PlanarObservedAnchor)(lo, hi, y, beta, lam)
            setup = time.perf_counter() - t
            t = time.perf_counter()
            s = (MomentSearch if name == 'box' else PlanarMomentSearch)(a, a.p)
            search_setup = time.perf_counter() - t
            t = time.perf_counter()
            r = s.run(time_limit=2.0, record=True)
            elapsed = time.perf_counter() - t
            cert = BASE / f'planar_diabetes_k{k}_{name}_certificate.json'
            cert.write_text(json.dumps(r, indent=2), encoding='utf-8')
            results['certifications'].append({'dataset': 'sklearn diabetes', 'features': 11, 'rows': 442, 'uncertain_rows': 4, 'k': k, 'oracle': name, 'anchor_setup_seconds': setup, 'search_setup_seconds': search_setup, 'actual_solve_seconds': elapsed, **{key: v for key, v in r.items() if key not in ('certificate', 'trace')}, 'certificate_file': cert.name})
        except Exception:
            results['errors'].append({'stage': 'full', 'k': k, 'oracle': name, 'error': traceback.format_exc()})
        save()
print(json.dumps({'checks': len(results['exact_checks']), 'timings': results['timings'], 'full': results['certifications'], 'errors': results['errors']}, indent=2))
