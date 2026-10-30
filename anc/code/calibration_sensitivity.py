from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Post-hoc primary-calibration sensitivity from saved eligible scalar endpoints.'
from pathlib import Path
import json
import numpy as np, pandas as pd
B = Path(__file__).resolve().parent
ROOT = B.parent
METHODS = ('anchor', 'gradient', 'metric_pod', 'metric_greedy', 'full')

def classify(e, t):
    if e is None:
        return 'UNRESOLVED'
    return 'STOP' if e['upper_K2'] <= t else 'VIOLATING' if e['lower_K2'] > t else 'UNRESOLVED'

def main():
    compact = B / 'calibration_endpoints_v83.json'
    if compact.exists():
        p = json.loads(compact.read_text())
    else:
        source = json.loads((MODULE_DIR / 'calibration_results.json').read_text())
        initial = [r for r in source['results'] if r['checkpoint'] == 0]
        assert len(initial) == 60
        p = dict(conditions=[{k: c[k] for k in ('job', 'pool_seed', 'uncertain_rows', 'spread')} for c in source['conditions']], endpoints=[])
        for r in initial:
            for h in (5.0, 10.0):
                eligible = [e for e in r['events'] if e['seconds'] <= h]
                e = eligible[-1] if eligible else None
                p['endpoints'].append(dict(job=r['job'], method=r['method'], horizon_seconds=h, endpoint=dict(seconds=e['seconds'], lower_K2=e['lower'] * 128 ** 2, upper_K2=e['upper'] * 128 ** 2) if e else None))
        compact.write_text(json.dumps(p, indent=2))
    endpoints = {(r['horizon_seconds'], r['job'], r['method']): r['endpoint'] for r in p['endpoints']}
    assert len(p['conditions']) == 12 and len(endpoints) == 120
    grid = set(np.linspace(0, 1, 501).tolist() + [0.25])
    for e in endpoints.values():
        if e:
            for v in (e['lower_K2'], e['upper_K2']):
                if 0 <= v <= 1:
                    grid.update([v, np.nextafter(v, -np.inf), np.nextafter(v, np.inf)])
    grid = sorted((t for t in grid if 0 <= t <= 1))
    rows = []
    for h in (5.0, 10.0):
        for t in grid:
            for method in METHODS:
                labels = [classify(endpoints[h, j, method], t) for j in range(12)]
                rows.append(dict(horizon_seconds=h, threshold_K2=t, method=method, STOP=labels.count('STOP'), VIOLATING=labels.count('VIOLATING'), UNRESOLVED=labels.count('UNRESOLVED'), definite=sum((s != 'UNRESOLVED' for s in labels))))
    frame = pd.DataFrame(rows)
    frame.to_csv(B / 'calibration_threshold_curve_v83.csv', index=False)
    job = next((c['job'] for c in p['conditions'] if c['pool_seed'] == 79011 and c['uncertain_rows'] == 64 and (c['spread'] == 'concentrated')))
    ints = []
    for h in (5.0, 10.0):
        full = endpoints[h, job, 'full']
        red = [endpoints[h, job, m] for m in METHODS if m not in ('anchor', 'full')]
        lower = max([full['lower_K2']] + [e['upper_K2'] for e in red])
        upper = full['upper_K2']
        ints.append(dict(horizon_seconds=h, lower_inclusive=lower, upper_exclusive=upper, nonempty=lower < upper))
        if lower < upper:
            mid = (lower + upper) / 2
            assert classify(full, mid) == 'UNRESOLVED' and all((classify(e, mid) == 'STOP' for e in red))
    primary = frame[(frame.horizon_seconds == 10) & (frame.threshold_K2 == 0.25)].to_dict('records')
    expected = {'anchor': (9, 0, 3), 'gradient': (10, 1, 1), 'metric_pod': (10, 1, 1), 'metric_greedy': (10, 1, 1), 'full': (9, 1, 2)}
    for r in primary:
        assert (r['STOP'], r['VIOLATING'], r['UNRESOLVED']) == expected[r['method']]
    result = dict(status='PASS', source='v81 independent-calibration primary12', post_hoc=True, searches_rerun=False, conditions=12, grid_points=len(grid), primary_threshold=0.25, primary_counts=primary, critical_case_intervals=ints, policy_note='reclassifies saved primary-threshold runs, including early coarse STOPs; no new policy execution')
    (B / 'calibration_sensitivity_v83.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
if __name__ == '__main__':
    main()
