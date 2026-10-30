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
import numpy as np, psutil
from greedy_run import confirmation_problem
from observed_anchor import ObservedAnchor
import moment_tail
from moment_tail import MomentSearch
from metric_basis import build_proposal

def save(p):
    (B / 'metric_results.json').write_text(json.dumps(p, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def checkpoints(events):
    out = []
    for budget in (1.0, 5.0, 10.0):
        valid = [e for e in events if e['seconds'] <= budget]
        lo = max([e['lower'] for e in valid], default=None)
        hi = min([e['upper'] for e in valid], default=None)
        out.append(dict(budget_seconds=budget, lower=lo, upper=hi, interval_width=hi - lo if hi is not None else None, evidence_events=len(valid), last_event_seconds=max([e['seconds'] for e in valid], default=None)))
    return out

def main():
    psutil.Process().cpu_affinity([10])
    assert psutil.Process().cpu_affinity() == [10]
    if (B / 'metric_results.json').exists():
        raise FileExistsError('metric_results.json')
    startqueue = time.monotonic()
    p = dict(protocol_sha256=hashlib.sha256((B / 'metric_protocol.md').read_bytes()).hexdigest(), cpu=10, threads=1, total_budget_seconds=10.0, rank=3, conditions_source='../exploration_v78/greedy_results.json (same18conditions, changed metric and budget)', results=[], errors=[], status='RUNNING')
    methods = ['gradient', 'metric_pod', 'metric_greedy']
    for idx, (name, spread, seed) in enumerate(((n, s, se) for n in ('diabetes', 'concrete', 'superconductivity') for s in ('concentrated', 'dispersed') for se in (78011, 78021, 78031))):
        try:
            t = time.monotonic()
            args, meta = confirmation_problem(name, spread, seed)
            input_s = time.monotonic() - t
            t = time.monotonic()
            a = ObservedAnchor(*args, anchor='observed', preconditioner='midpoint')
            anchor_s = time.monotonic() - t
            common = input_s + anchor_s
        except Exception:
            p['errors'].append(dict(condition=idx, dataset=name, spread=spread, seed=seed, stage='common', traceback=traceback.format_exc()))
            save(p)
            continue
        for order, method in enumerate(methods[idx % 3:] + methods[:idx % 3]):
            rec = dict(condition=idx, **meta, method=method, order=order, input_seconds=input_s, anchor_seconds=anchor_s, common_seconds=common, events=[dict(seconds=common, lower=0.0, upper=a.gradient_upper, stage='common_anchor')])
            try:
                begin = time.monotonic()
                r, receipt = build_proposal(a, method)
                t = time.monotonic()
                with patch.object(moment_tail, 'basis_proposal', return_value=r):
                    s = MomentSearch(a, 3, kind=method, search_bound='adaptive')
                setup = time.monotonic() - t
                offset = common + time.monotonic() - begin
                searchstart = time.monotonic()
                if offset < 10:
                    result = s.run(time_limit=10 - offset, absolute_tolerance=1e-10, relative_tolerance=0.0, record=False)
                    rec['events'].extend((dict(seconds=offset + e['seconds'], lower=float(e['lower']), upper=float(e['upper']), stage='search') for e in result['trace']))
                    rec['events'].append(dict(seconds=common + time.monotonic() - begin, lower=float(result['lower']), upper=float(result['upper']), stage='final'))
                else:
                    result = dict(status='SETUP_CAP', lower=0.0, upper=a.gradient_upper, stop_reason='setup_cap')
                search = time.monotonic() - searchstart
                rec.update(status='PASS', solver_status=result['status'], proposal=receipt, basis_seconds=receipt['basis_seconds'], solver_setup_seconds=setup, setup_offset_seconds=offset, search_seconds=search, total_seconds=common + time.monotonic() - begin, final_lower=float(result['lower']), final_upper=float(result['upper']), tail=float(s.qtail), G_r=s.tail_receipt['gradient_norm'], C_r=s.tail_receipt['coupling_norm'], R_s=s.tail_receipt['coordinate_radius'], mu=a.mu, basis_Hmu=s.hmu, stop_reason=result.get('stop_reason'), checkpoints=checkpoints(rec['events']))
            except Exception:
                rec.update(status='ERROR', traceback=traceback.format_exc())
                p['errors'].append({k: rec[k] for k in ('condition', 'dataset', 'spread', 'seed', 'method', 'traceback')})
            p['results'].append(rec)
            save(p)
        print(idx, name, spread, seed, 'DONE', round(time.monotonic() - startqueue, 2), flush=True)
    p.update(status='PASS' if len(p['results']) == 54 and (not p['errors']) else 'COMPLETED_WITH_ERRORS', queue_seconds=time.monotonic() - startqueue)
    save(p)
    rows = []
    for r in p['results']:
        if r['status'] != 'PASS':
            continue
        for cp in r['checkpoints']:
            rows.append(dict(condition=r['condition'], dataset=r['dataset'], spread=r['spread'], seed=r['seed'], method=r['method'], **cp, tail=r['tail'], input_seconds=r['input_seconds'], anchor_seconds=r['anchor_seconds'], basis_seconds=r['basis_seconds'], solver_setup_seconds=r['solver_setup_seconds'], search_seconds=r['search_seconds'], total_seconds=r['total_seconds'], stop_reason=r['stop_reason']))
    with (B / 'metric_summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
if __name__ == '__main__':
    main()
