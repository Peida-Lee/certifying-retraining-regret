from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Fixed rank-three controls for basis, bounding rule and dense preconditioner.'
from pathlib import Path
import json, time, traceback
from public_problems import problem
from observed_anchor import ObservedAnchor
from moment_tail import MomentSearch
from dual_certificate import reserve_cores
BASE = Path(__file__).resolve().parent

def main():
    reserve_cores()
    out = BASE / 'additional_components.json'
    if out.exists():
        raise FileExistsError(out)
    results = []
    for name in ('diabetes', 'concrete', 'superconductivity'):
        for spread in ('concentrated', 'dispersed'):
            for precond, kind, bound in (('midpoint', 'gradient', 'center'), ('midpoint', 'midpoint', 'adaptive'), ('anchor', 'gradient', 'adaptive')):
                try:
                    args, meta = problem(name=name, spread=spread)
                    t = time.monotonic()
                    a = ObservedAnchor(*args, preconditioner=precond)
                    r = MomentSearch(a, 3, kind=kind, search_bound=bound).run(time_limit=1, absolute_tolerance=1e-06)
                    row = {**meta, 'preconditioner': precond, 'basis': kind, 'search_bound': bound, 'status': 'PASS', 'result': r, 'total_seconds': time.monotonic() - t}
                except Exception as exc:
                    row = {'dataset': name, 'spread': spread, 'status': 'ERROR', 'error': str(exc), 'traceback': traceback.format_exc()}
                results.append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ('result', 'features_names')}), flush=True)
    out.write_text(json.dumps({'results': results, 'status': 'PASS' if all((r['status'] == 'PASS' for r in results)) else 'COMPLETED_WITH_ERRORS', 'selection': 'all six development tables; fixed rank3,1second,1e-6; three rules enumerated in script before execution'}, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
