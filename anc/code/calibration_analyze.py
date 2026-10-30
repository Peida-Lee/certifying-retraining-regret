from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Audit independent calibration provenance, eligible events and policy counts.'
from pathlib import Path
import json
import numpy as np, pandas as pd
B = Path(__file__).resolve().parent
ROOT = B.parent

def label(l, u, t=0.25):
    return 'STOP' if u <= t else 'VIOLATING' if l > t else 'UNRESOLVED'

def main():
    p = json.loads((B / 'calibration_results.json').read_text())
    plan = json.loads((B / 'calibration_plan.json').read_text())
    assert p['status'] == 'PASS'
    ids = set(plan['calibration_row_ids'])
    assert len(ids) == 512 and (not ids & set(plan['excluded_row_ids']))
    targets = set()
    for pool in plan['target_pools']:
        other = set(pool['row_indices'])
        assert len(other) == 1024 and (not targets & other) and (not ids & other)
        targets |= other
    raw = pd.read_csv(DATA_CORE / 'superconductivity/train.csv').iloc[plan['calibration_row_ids'], :-1].to_numpy(float)
    span = np.ptp(raw, axis=0)
    span = np.where(span > 0, span, 1.0)
    assert np.array_equal(raw.mean(0), np.array(plan['feature_means'])) and np.array_equal(span, np.array(plan['feature_spans']))
    assert len(p['conditions']) == 12
    for r in p['results']:
        assert r['status'] == 'PASS'
        anchor = next((q for q in p['results'] if q['job'] == r['job'] and q['checkpoint'] == r['checkpoint'] and (q['method'] == 'anchor')))
        assert r['events'][0] == anchor['events'][0] and r['common_seconds'] == anchor['common_seconds']
        for h in (1.0, 5.0, 10.0):
            eligible = [e for e in r['events'] if e['seconds'] <= h]
            end = eligible[-1] if eligible else None
            if end:
                assert r['actual_completion_regret_K2'] <= end['upper'] * 128 ** 2 + 1e-09
            if h == 10:
                assert r['decision'] == (label(end['lower'] * 128 ** 2, end['upper'] * 128 ** 2) if end else 'UNRESOLVED')
    summary = []
    for method in ('anchor', 'gradient', 'metric_pod', 'metric_greedy', 'full'):
        initial = [r for r in p['results'] if r['method'] == method and r['checkpoint'] == 0]
        assert len(initial) == 12
        summary.append(dict(method=method, conditions=12, STOP=sum((r['decision'] == 'STOP' for r in initial)), VIOLATING=sum((r['decision'] == 'VIOLATING' for r in initial)), UNRESOLVED=sum((r['decision'] == 'UNRESOLVED' for r in initial)), coarse_STOP=sum((next((q for q in p['results'] if q['job'] == r['job'] and q['checkpoint'] == 0 and (q['method'] == 'anchor')))['decision'] == 'STOP' for r in initial)), saved_cells=sum((r['saved_cells_first_stop'] for r in initial))))
    for cond in p['conditions']:
        rs = [r for r in p['results'] if r['job'] == cond['job']]
        reveal = any((r['checkpoint'] == 0 and r['method'] != 'anchor' and (r['decision'] == 'UNRESOLVED') for r in rs))
        assert any((r['checkpoint'] == 1 for r in rs)) == reveal
        for method in ('anchor', 'gradient', 'metric_pod', 'metric_greedy', 'full'):
            ms = [r for r in rs if r['method'] == method]
            s = max([r['remaining_cells'] for r in ms if r['decision'] == 'STOP'], default=0)
            assert all((r['saved_cells_first_stop'] == s for r in ms))
    pd.DataFrame(summary).to_csv(B / 'calibration_counts.csv', index=False)
    out = dict(status='PASS', checks=['512calibration IDs independent of all declared earlier and target pools', 'means/ranges recomputed solely from calibration values', '12conditions and complete five-method comparison', 'identical within-condition shared charges', 'deadline eligibility and original-completion coverage', 'fixed half-reveal saving policy'], elapsed_seconds=p['elapsed_seconds'], calibration_compute_seconds=plan['calibration_compute_seconds'], calibration_feature_cells=plan['calibration_feature_cells'], counts=summary)
    (B / 'calibration_audit.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
if __name__ == '__main__':
    main()
