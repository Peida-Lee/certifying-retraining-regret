from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
from planar_oracle import *
import ctypes, json, time, traceback, csv
ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
ctypes.windll.kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
assert ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(), 1 << 8)
r = {'checks': [], 'density': [], 'errors': []}
namespace = {'__file__': str(BASE / 'planar_run.py'), '__name__': 'frozen_helpers'}
src = (BASE / 'planar_run.py').read_text()
snippet = src[src.index('def exhaustive('):src.index('rng=np.random.default_rng')]
exec('from itertools import product\n' + snippet, globals())
rng = np.random.default_rng(77103)
for k in (1, 2, 4, 8):
    for case in range(3):
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
        v, z = exact_row_max(ac, bc, gg, gamma, True)
        assert v == exhaustive(ac, bc, gg, gamma)
        old = json.loads((BASE / 'planar_results.json').read_text())['exact_checks'][len(r['checks'])]
        assert str(v) == old['maximum'] and list(map(str, z)) == old['witness']
        r['checks'].append({'k': k, 'case': case, 'value_and_witness_unchanged': True})
rng = np.random.default_rng(77116)
xx = np.round(rng.normal(size=(128, 16)) / 4 * 1024) / 1024
x = np.column_stack([np.ones(128), xx])
coeff = np.array([1 / (j + 2) for j in range(17)])
noise = np.round(rng.normal(size=128) * 0.1 * 1024) / 1024
y = np.round((x @ coeff + noise) * 1024) / 1024
lam = 1 / 32
beta = np.linalg.solve(x.T @ x / 128 + lam * np.eye(17), x.T @ y / 128)
for k in (4, 8, 16):
    try:
        lo = x.copy()
        hi = x.copy()
        lo[:4, 1:k + 1] -= 1 / 16
        hi[:4, 1:k + 1] += 1 / 16
        start = time.perf_counter()
        a = PlanarObservedAnchor(lo, hi, y, beta, lam)
        aset = time.perf_counter() - start
        start = time.perf_counter()
        s = PlanarMomentSearch(a, 2)
        sset = time.perf_counter() - start
        start = time.perf_counter()
        out = s.run(time_limit=2, absolute_tolerance=1e-08, record=True)
        wall = time.perf_counter() - start
        file = f'planar_density_k{k}_certificate.json'
        (BASE / file).write_text(json.dumps(out, indent=2))
        r['density'].append({'k': k, 'n': 128, 'p': 17, 'uncertain_rows': 4, 'anchor_rows': 124, 'rank': 2, 'anchor_setup_seconds': aset, 'search_setup_seconds': sset, 'actual_solve_seconds': wall, 'certificate_file': file, **{key: v for key, v in out.items() if key not in ('certificate', 'trace')}})
    except Exception:
        r['errors'].append({'k': k, 'error': traceback.format_exc()})
    (BASE / 'planar_density_results.json').write_text(json.dumps(r, indent=2))
ns = {'__file__': str(BASE / 'planar_audit.py'), '__name__': 'upgrade_independent'}
src = (BASE / 'planar_audit.py').read_text()
exec(src[:src.index("r={'method'")], ns)
r['existing_dense_tree_audit'] = ns['check'].audit(BASE / 'planar_tight_diabetes_k10_planar_certificate.json', True)
(BASE / 'planar_density_results.json').write_text(json.dumps(r, indent=2))
keys = ['k', 'n', 'p', 'rank', 'lower', 'upper', 'absolute_gap', 'relative_gap', 'tail_error', 'search_error', 'anchor_setup_seconds', 'search_setup_seconds', 'actual_solve_seconds', 'nodes', 'points', 'stop_reason']
with (BASE / 'planar_density_summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
    w.writeheader()
    w.writerows(r['density'])
print(json.dumps(r, indent=2))
