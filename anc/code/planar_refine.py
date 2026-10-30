from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
from planar_oracle import *
import os, ctypes, time, json, statistics, traceback
ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
ctypes.windll.kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(), 1 << 8)
import observed_anchor as old
from moment_tail import MomentSearch
from sklearn.datasets import load_diabetes
r = {'vectorized_interval_comparison': [], 'tight_certifications': [], 'audits': [], 'errors': []}
source = json.loads((BASE / 'planar_results.json').read_text())
for record in source['timings']:
    k = record['k']
    gg = [[float(Q(v)) for v in g] for g in record['generators']]
    aa = I([0.125])
    bb = I([-0.25])
    ag = I(np.array([[g[0] for g in gg]]))
    bg = I(np.array([[g[1] for g in gg]]))
    row = {'k': k}
    for name, fn in [('planar', row_max), ('v74_vectorized_box', old.row_max)]:
        if k > 8 and name != 'planar':
            continue
        times = []
        for rep in range(3):
            start = time.perf_counter()
            v, z = fn(aa, bb, ag, bg, 1.0)
            times.append(time.perf_counter() - start)
        row[name] = {'seconds': times, 'median': statistics.median(times), 'lower': float(v.lo[0]), 'upper': float(v.hi[0])}
    r['vectorized_interval_comparison'].append(row)
o = load_diabetes()
xx = np.round((o.data - o.data.mean(0)) / np.ptp(o.data, axis=0) * 1024) / 1024
x = np.column_stack([np.ones(len(xx)), xx])
y = np.round((o.target - o.target.min()) / np.ptp(o.target) * 1024) / 1024
for k, names in [(2, ['box', 'planar']), (10, ['planar'])]:
    lo = x.copy()
    hi = x.copy()
    lo[:4, 1:1 + k] -= 1 / 16
    hi[:4, 1:1 + k] += 1 / 16
    lam = 1 / 32
    beta = np.linalg.solve(x.T @ x / len(x) + lam * np.eye(11), x.T @ y / len(x))
    for name in names:
        try:
            start = time.perf_counter()
            a = (old.ObservedAnchor if name == 'box' else PlanarObservedAnchor)(lo, hi, y, beta, lam)
            aset = time.perf_counter() - start
            start = time.perf_counter()
            s = (MomentSearch if name == 'box' else PlanarMomentSearch)(a, a.p)
            sset = time.perf_counter() - start
            start = time.perf_counter()
            out = s.run(time_limit=2.0, absolute_tolerance=1e-08, record=True)
            solve = time.perf_counter() - start
            file = BASE / f'planar_tight_diabetes_k{k}_{name}_certificate.json'
            file.write_text(json.dumps(out, indent=2))
            r['tight_certifications'].append({'k': k, 'oracle': name, 'anchor_setup_seconds': aset, 'search_setup_seconds': sset, 'actual_solve_seconds': solve, 'certificate': file.name, **{key: v for key, v in out.items() if key not in ('certificate', 'trace')}})
        except Exception:
            r['errors'].append({'stage': 'tight', 'k': k, 'oracle': name, 'error': traceback.format_exc()})
        (BASE / 'planar_refinement.json').write_text(json.dumps(r, indent=2))
print(json.dumps(r, indent=2))
