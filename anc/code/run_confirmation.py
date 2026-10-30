from project_paths import MODULE_DIR, DATA_CORE, DATA_EXTRA
'Sixty frozen public conditions; bounded searches and continuing queue.'
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json, time, traceback
import numpy as np
from public_problems import problem
from observed_anchor import ObservedAnchor
from moment_tail import MomentSearch, basis_proposal, tail_receipt
from dual_certificate import reserve_cores
BASE = Path(__file__).resolve().parent
SEEDS = (74111, 74123, 74137, 74141, 74153)

def main():
    reserve_cores()
    out = BASE / 'general_confirmation_results.json'
    if out.exists():
        raise FileExistsError(out)
    results = []
    started = time.monotonic()
    for name in ('diabetes', 'concrete', 'superconductivity'):
        for width in (1 / 32, 1 / 8):
            for spread in ('concentrated', 'dispersed'):
                for seed in SEEDS:
                    try:
                        args, meta = problem(name=name, cells=128, width=width, spread=spread, seed=seed, subset_seed=74101)
                        start = time.monotonic()
                        anchor = ObservedAnchor(*args, anchor='observed', preconditioner='midpoint')
                        setup = time.monotonic() - start
                        proposal = basis_proposal(anchor)
                        candidates = []
                        for rank in (1, 2, 3):
                            receipt = tail_receipt(anchor, proposal[:, :rank])
                            candidates.append((rank, receipt['tail_bound']))
                            if receipt['tail_bound'] <= 2.5e-06:
                                break
                        selected = next((rank for rank, tail in candidates if tail <= 2.5e-06), min(candidates, key=lambda q: q[1])[0])
                        rank_selection = time.monotonic() - start - setup
                        t0 = time.monotonic()
                        reduced = MomentSearch(anchor, selected).run(time_limit=1, absolute_tolerance=3e-06)
                        reduced['end_to_end_seconds'] = setup + rank_selection + time.monotonic() - t0
                        t0 = time.monotonic()
                        full = MomentSearch(anchor, anchor.p).run(time_limit=1, absolute_tolerance=3e-06)
                        full['end_to_end_seconds'] = setup + time.monotonic() - t0
                        common = max(reduced['lower'], full['lower'])
                        record = {**meta, 'status': 'PASS', 'rank_candidates': candidates, 'selected_rank': selected, 'tail_budget_met': next((t for r, t in candidates if r == selected)) <= 2.5e-06, 'setup_seconds': setup, 'rank_selection_seconds': rank_selection, 'reduced': reduced, 'full': full, 'common_lower': common, 'upper_reduction': 1 - reduced['upper'] / full['upper'], 'common_width_reduction': 1 - (reduced['upper'] - common) / max(full['upper'] - common, 1e-300), 'elapsed_seconds': time.monotonic() - start}
                    except Exception as exc:
                        record = {'dataset': name, 'width': width, 'spread': spread, 'seed': seed, 'status': 'ERROR', 'error': str(exc), 'traceback': traceback.format_exc()}
                    results.append(record)
                    (BASE / 'confirmation_progress.json').write_text(json.dumps({'status': 'RUNNING', 'time_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'elapsed_seconds': time.monotonic() - started, 'results': results}, indent=2), encoding='utf-8')
                    print(json.dumps({k: v for k, v in record.items() if k not in ('reduced', 'full', 'features_names')} | ({'reduced_gap': record['reduced']['absolute_gap'], 'full_gap': record['full']['absolute_gap']} if 'reduced' in record else {})), flush=True)
    output = {'status': 'PASS' if all((v['status'] == 'PASS' for v in results)) else 'COMPLETED_WITH_ERRORS', 'conditions': len(results), 'results': results, 'elapsed_seconds': time.monotonic() - started, 'protocol': 'confirmation_protocol.md', 'timezone': 'Asia/Shanghai'}
    out.write_text(json.dumps(output, indent=2), encoding='utf-8')
if __name__ == '__main__':
    main()
