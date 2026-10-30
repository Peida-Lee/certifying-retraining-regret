from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
from leakage_energy import *
import time, json, traceback, hashlib, csv
import psutil
import moment_tail
psutil.Process().cpu_affinity([8])
out = B / 'leakage_results.json'
assert not out.exists()
payload = dict(status='RUNNING', protocol='leakage_protocol.md', protocol_sha256=hashlib.sha256((B / 'leakage_protocol.md').read_bytes()).hexdigest(), cpu=8, threads=1, results=[])
for idx, delta in enumerate((Q(0), Q(1, 2 ** 18), Q(1, 2 ** 16), Q(1, 2 ** 14), Q(1, 2 ** 12))):
    start = time.monotonic()
    args = problem(delta)
    t = time.monotonic()
    a = ObservedAnchor(*args, anchor='observed', preconditioner='anchor')
    common = time.monotonic() - start
    fixed = np.zeros((a.p, 1))
    fixed[0, 0] = 1
    for method in ('energy', 'moment'):
        row = dict(condition=idx, method=method, common_seconds=common, **constants(delta))
        try:
            t = time.monotonic()
            if method == 'energy':
                s = LeakageSearch(a, delta)
            else:
                with patch.object(moment_tail, 'basis_proposal', return_value=fixed):
                    s = MomentSearch(a, 1)
            setup = time.monotonic() - t
            t = time.monotonic()
            r = s.run(time_limit=2, absolute_tolerance=1e-06, relative_tolerance=0, record=method == 'energy')
            search = time.monotonic() - t
            if 'certificate' in r:
                name = f'leakage_condition_{idx}_certificate.json'
                (B / name).write_text(json.dumps(dict(delta=str(delta), result=r), indent=2))
                r.pop('certificate')
                row['certificate_file'] = name
            row.update(status='PASS', method_seconds=setup, search_seconds=search, total_seconds=common + setup + search, result=r, actual_tail=float(s.qtail), target_met=r['absolute_gap'] <= 1e-06)
        except Exception as e:
            row.update(status='ERROR', error=str(e), traceback=traceback.format_exc())
        payload['results'].append(row)
        out.write_text(json.dumps(payload, indent=2))
        print(idx, method, row['status'], flush=True)
payload['status'] = 'PASS' if all((r['status'] == 'PASS' for r in payload['results'])) else 'COMPLETED_WITH_ERRORS'
out.write_text(json.dumps(payload, indent=2))
rows = [dict(condition=r['condition'], delta=r['delta'], method=r['method'], status=r['status'], tail=r.get('actual_tail'), width=r.get('result', {}).get('absolute_gap'), lower=r.get('result', {}).get('lower'), upper=r.get('result', {}).get('upper'), total_seconds=r.get('total_seconds'), search_seconds=r.get('search_seconds'), target_met=r.get('target_met')) for r in payload['results']]
with (B / 'leakage_summary.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
