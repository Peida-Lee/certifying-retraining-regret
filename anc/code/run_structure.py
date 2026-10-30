from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Run the frozen twelve-condition residual-structure mechanism queue.\n\nNo queue runs at import. Start only after the other CPU4 queue has finished.\nAll numerical work is single-threaded on logical CPU4; failures are retained\nand remaining conditions continue.\n'
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'
import argparse
from datetime import datetime
from fractions import Fraction
import json
from pathlib import Path
import time
import traceback
BASE = Path(__file__).resolve().parent
CONDITIONS = tuple(((m, alpha) for m in (16, 64, 256) for alpha in (Fraction(0), Fraction(1, 64), Fraction(1, 32), Fraction(1, 16))))
METHODS = ('generic_v74_moment', 'residual_energy')
CAP_SECONDS = 10.0
TARGET = 1e-06

def configure_cpu():
    import psutil
    process = psutil.Process()
    process.cpu_affinity([4])
    actual = process.cpu_affinity()
    if actual != [4]:
        raise RuntimeError('Timed queue requires logical CPU4; actual ' + str(actual))
    return {'logical_cpu_affinity': actual, 'numerical_threads': 1, 'reserved_logical_cpus': [0, 1, 2, 3]}

def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=BASE / 'structure_results.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    execution = configure_cpu()
    from residual_energy import ObservedAnchor, MomentSearch, ResidualEnergySearch, structure_problem, analytic_structure
    started = time.monotonic()
    payload = {'study': 'v75_frozen_original_residual_structure_mechanism', 'timezone': 'Asia/Shanghai', 'started_at': datetime.now().astimezone().isoformat(), 'protocol': str(BASE / 'protocol.md'), 'execution': execution, 'search_cap_seconds': CAP_SECONDS, 'absolute_target': TARGET, 'method_order': list(METHODS), 'conditions': [{'m': m, 'alpha': str(alpha)} for m, alpha in CONDITIONS], 'timing_scope': 'Common input construction and anchor setup are measured once per condition. Each method total attributes that same common cost plus its own setup/search. No code is timed concurrently.', 'results': [], 'status': 'running'}
    save_json(args.output, payload)
    for m, alpha in CONDITIONS:
        case = {'m': m, 'alpha': str(alpha), 'n': 2 * m, 'p': m + 1}
        try:
            tick = time.monotonic()
            inputs = structure_problem(m, alpha)
            construction_seconds = time.monotonic() - tick
            tick = time.monotonic()
            anchor = ObservedAnchor(*inputs, anchor='observed', preconditioner='anchor')
            anchor_seconds = time.monotonic() - tick
            common = {'construction_seconds': construction_seconds, 'anchor_setup_seconds': anchor_seconds, 'anchor_summary': anchor.summary(), 'analytic': analytic_structure(m, alpha)}
        except Exception as exc:
            for method in METHODS:
                payload['results'].append({**case, 'method': method, 'status': 'ERROR', 'phase': 'common_input_or_anchor', 'error': str(exc), 'traceback': traceback.format_exc()})
            save_json(args.output, payload)
            print(json.dumps({**case, 'status': 'ERROR', 'phase': 'common_input_or_anchor'}, ensure_ascii=False), flush=True)
            continue
        for method in METHODS:
            entry = {**case, **common, 'method': method}
            try:
                tick = time.monotonic()
                search = MomentSearch(anchor, 1, kind='gradient') if method == 'generic_v74_moment' else ResidualEnergySearch(anchor)
                method_setup_seconds = time.monotonic() - tick
                representative = method == 'residual_energy' and m == 64 and (alpha == Fraction(1, 32))
                tick = time.monotonic()
                result = search.run(time_limit=CAP_SECONDS, absolute_tolerance=TARGET, relative_tolerance=0.0, record=representative)
                search_wall_seconds = time.monotonic() - tick
                exact_h = Fraction(common['analytic']['fraction']['exact_original_H'])
                if not Fraction(result['lower']) <= exact_h <= Fraction(result['upper']):
                    raise ArithmeticError('Computed interval excludes analytic original optimum')
                entry.update({'status': 'PASS', 'method_setup_seconds': method_setup_seconds, 'search_wall_seconds': search_wall_seconds, 'method_end_to_end_seconds': construction_seconds + anchor_seconds + method_setup_seconds + search_wall_seconds, 'target_attained': result['absolute_gap'] <= TARGET, 'analytic_original_contained': True})
                if representative:
                    certificate_path = BASE / 'structure_representative_certificate.json'
                    receipt = {**case, 'study': payload['study'], 'execution': execution, 'analytic': common['analytic'], 'result': result}
                    save_json(certificate_path, receipt)
                    result.pop('certificate')
                    entry['certificate_path'] = str(certificate_path)
                entry['result'] = result
            except Exception as exc:
                entry.update({'status': 'ERROR', 'error': str(exc), 'traceback': traceback.format_exc()})
            payload['results'].append(entry)
            payload['elapsed_seconds'] = time.monotonic() - started
            save_json(args.output, payload)
            print(json.dumps({**case, 'method': method, 'status': entry['status'], 'width': entry.get('result', {}).get('absolute_gap'), 'stop_reason': entry.get('result', {}).get('stop_reason')}, ensure_ascii=False), flush=True)
    payload.update({'status': 'completed', 'completed_at': datetime.now().astimezone().isoformat(), 'elapsed_seconds': time.monotonic() - started, 'successful_results': sum((r['status'] == 'PASS' for r in payload['results'])), 'error_results': sum((r['status'] == 'ERROR' for r in payload['results']))})
    save_json(args.output, payload)
if __name__ == '__main__':
    main()
