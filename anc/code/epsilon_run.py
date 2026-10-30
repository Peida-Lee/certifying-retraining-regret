from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Execute only the frozen five-condition v76 epsilon mechanism queue.'
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import sys
sys.dont_write_bytecode = True
from datetime import datetime
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import time
import traceback
BASE = Path(__file__).resolve().parent
DELTAS = (Q(0), Q(1, 2 ** 15), Q(1, 2 ** 14), Q(1, 2 ** 13), Q(1, 2 ** 12))
CAP, TARGET = (10.0, 1e-06)

def save(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def main():
    output = BASE / 'epsilon_results.json'
    if output.exists():
        raise FileExistsError('Frozen timed queue will not overwrite ' + str(output))
    import psutil
    process = psutil.Process()
    process.cpu_affinity([4])
    if process.cpu_affinity() != [4]:
        raise RuntimeError('Timed queue must run on logical CPU4')
    from epsilon_energy import epsilon_problem, ObservedAnchor, EpsilonSearch, analytic
    protocol = BASE / 'epsilon_protocol.md'
    payload = {'study': 'v76_frozen_nonzero_nuisance_response_residual', 'started_at': datetime.now().astimezone().isoformat(), 'timezone': 'Asia/Shanghai', 'protocol': protocol.name, 'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(), 'logical_cpu_affinity': process.cpu_affinity(), 'numerical_threads': 1, 'reserved_logical_cpus': [0, 1, 2, 3], 'search_cap_seconds': CAP, 'absolute_target': TARGET, 'deltas': [str(x) for x in DELTAS], 'results': [], 'status': 'running'}
    save(output, payload)
    started = time.monotonic()
    for index, delta in enumerate(DELTAS):
        entry = {'condition': index, 'delta_fraction': str(delta), 'm': 64, 'n': 128, 'p': 65, 'alpha': '1/32', 'analytic': analytic(delta)}
        try:
            tick = time.monotonic()
            inputs = epsilon_problem(delta)
            construction = time.monotonic() - tick
            tick = time.monotonic()
            anchor = ObservedAnchor(*inputs, anchor='observed', preconditioner='anchor')
            anchor_time = time.monotonic() - tick
            tick = time.monotonic()
            search = EpsilonSearch(anchor, delta)
            method_time = time.monotonic() - tick
            tick = time.monotonic()
            result = search.run(time_limit=CAP, absolute_tolerance=TARGET, relative_tolerance=0.0, record=True)
            search_time = time.monotonic() - tick
            exact_h = Q(entry['analytic']['fraction']['exact_original_H'])
            if not Q(result['lower']) <= exact_h <= Q(result['upper']):
                raise ArithmeticError('Interval excludes analytic H*')
            cert_name = 'epsilon_condition_' + str(index) + '_certificate.json'
            save(BASE / cert_name, {'condition': index, 'delta': str(delta), 'result': result})
            result.pop('certificate')
            entry.update({'status': 'PASS', 'construction_seconds': construction, 'anchor_setup_seconds': anchor_time, 'method_setup_seconds': method_time, 'setup_seconds': construction + anchor_time + method_time, 'search_seconds': search_time, 'total_seconds': construction + anchor_time + method_time + search_time, 'target_attained': result['absolute_gap'] <= TARGET, 'tail_budget_attained': Q(entry['analytic']['fraction']['tail_error']) <= Q(TARGET), 'certificate_path': cert_name, 'result': result})
        except Exception as error:
            entry.update({'status': 'ERROR', 'error': str(error), 'traceback': traceback.format_exc()})
        payload['results'].append(entry)
        payload['elapsed_seconds'] = time.monotonic() - started
        save(output, payload)
        print(json.dumps({k: entry.get(k) for k in ('condition', 'delta_fraction', 'status', 'target_attained', 'tail_budget_attained', 'error')}, ensure_ascii=False), flush=True)
    payload.update({'status': 'completed', 'completed_at': datetime.now().astimezone().isoformat(), 'elapsed_seconds': time.monotonic() - started, 'error_results': sum((x['status'] != 'PASS' for x in payload['results']))})
    save(output, payload)
if __name__ == '__main__':
    main()
