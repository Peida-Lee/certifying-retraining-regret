"""Portable 600-second centered-moment runs and aggregate-only checkpoint replay.

Run inputs are the locally regenerated frozen_* receipts.  Aggregate replay
reads only the release's numerical summaries; it never imports a solver or
loads a completed table.  Relative CLI paths are relative to this code folder.
"""
import argparse
from fractions import Fraction as F
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import time
import traceback

B = Path(__file__).resolve().parent
RELEASE = B.parent / 'results' / 'refinement_v92'
FAMILIES = ('controls', 'validation', 'decisions')
CHECKPOINTS = (10, 60, 120, 600)


def path_arg(value):
    path = Path(value)
    return path if path.is_absolute() else B / path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rows_from(path):
    if path.is_dir():
        if (path / 'results.json').exists():
            return rows_from(path / 'results.json')
        files = sorted(path.glob('*_result.json'))
        if not files:
            raise ValueError(f'No *_result.json files in {path}')
        return [load(file) for file in files]
    value = load(path)
    if isinstance(value, list):
        return value
    return value['results']


def checkpoint(row, deadline, moment=False):
    """Credit only bounds completed by the deadline, including setup costs."""
    if row.get('status') != 'PASS':
        return None
    offset = float(row.get('setup_seconds', 0)) if moment else 0.0
    trace = row.get('trace')
    if trace is None:
        trace = [v for v in row.get('checkpoints', {}).values() if v is not None]
        if row.get('checkpoint_time_basis') == 'complete algorithm seconds':
            offset = 0.0
    events = [dict(seconds=float(event['seconds']) + offset,
                   lower=event['lower'], upper=event['upper']) for event in trace]
    if row.get('initial_event') is not None:
        events.append(row['initial_event'])
    end = row.get('complete_seconds', row.get('total_seconds'))
    if end is not None:
        events.append(dict(seconds=float(end), lower=row['lower'], upper=row['upper']))
    eligible = [event for event in events if 0 <= float(event['seconds']) <= deadline]
    for event in eligible:
        if not all(math.isfinite(float(event[k])) for k in ('seconds', 'lower', 'upper')):
            raise ValueError('Nonfinite checkpoint')
        if F(event['lower']) > F(event['upper']):
            raise ValueError('Reversed checkpoint interval')
    return max(eligible, key=lambda event: float(event['seconds'])) if eligible else None


def width_summary(rows, deadline):
    states = [checkpoint(row, deadline, moment=True) for row in rows]
    widths = [float(F(state['upper']) - F(state['lower'])) for state in states if state]
    return dict(deadline_seconds=deadline, available=len(widths),
                target_width='1/100000', target_met=sum(
                    F(state['upper']) - F(state['lower']) <= F(1, 100000)
                    for state in states if state),
                median_width=median(widths) if widths else None,
                maximum_width=max(widths) if widths else None)


def family_summary(family, rows):
    errors = []
    good = []
    seen = set()
    for index, row in enumerate(rows):
        instance = row.get('instance', f'row_{index}')
        try:
            if instance in seen:
                raise ValueError('Duplicate instance')
            seen.add(instance)
            if row.get('status') != 'PASS':
                raise ValueError(row.get('error', 'Run status is not PASS'))
            for deadline in CHECKPOINTS:
                checkpoint(row, deadline, moment=True)
            good.append(row)
        except Exception as exc:
            errors.append(dict(instance=instance, error=str(exc)))
    expected = {row['instance'] for row in rows_from(RELEASE / f'frozen_{family}.json')}
    if seen != expected:
        errors.append(dict(error='Instance keys do not cover the frozen family',
                           missing=sorted(expected - seen), unexpected=sorted(seen - expected)))
    # The matched-control table uses four first-mask primary rows plus the
    # twenty new matched rows; the remaining primary masks are also retained.
    first_seed = load(B / 'contraction_settings.json')['mask_seeds'][0]
    matched = [row for row in good if row['instance'].startswith('matched_') or
               (row['instance'].startswith('primary_') and row['seed'] == first_seed)]
    answer = dict(status='PASS' if not errors and good else 'COMPLETED_WITH_ERRORS',
                  rows=len(rows), valid_rows=len(good), errors=errors,
                  checkpoints={str(t): width_summary(good, t) for t in CHECKPOINTS})
    if family == 'controls':
        answer['matched_control_table'] = dict(rows=len(matched),
            checkpoints={str(t): width_summary(matched, t) for t in CHECKPOINTS})
    return answer


def decision_key(row):
    if 'pool' in row:
        return int(row['pool']), int(row['seed'])
    match = re.match(r'^pool(\d+)_(\d+)_', row['instance'])
    if not match:
        raise ValueError(f"Cannot identify decision instance {row.get('instance')}")
    return tuple(map(int, match.groups()))


def decision_comparison(moment_rows, refined_rows, conditions, settings, deadline):
    expected = {(pool, seed) for pool in range(settings['pools'])
                for seed in settings['mask_seeds']}
    errors = []
    condition_map = {}
    for row in conditions:
        try:
            key = (int(row['pool']), int(row['seed']))
            if key in condition_map:
                errors.append(dict(condition=list(key), error='Duplicate midpoint condition'))
            condition_map[key] = row
        except Exception as exc:
            errors.append(dict(error=f'Invalid midpoint condition: {exc}'))
    if set(condition_map) != expected:
        errors.append(dict(error='Midpoint conditions do not cover the frozen decision grid',
                           missing=sorted(expected - set(condition_map)),
                           unexpected=sorted(set(condition_map) - expected)))
    factor = 1 + F(str(settings['primary_increment_fraction']))
    counts = {method: [0] * settings['pools'] for method in ('R', 'M')}
    reports = []
    for method, rows in (('R', refined_rows), ('M', moment_rows)):
        keyed = {}
        duplicates = set()
        for row in rows:
            try:
                key = decision_key(row)
                if key in keyed:
                    duplicates.add(key)
                    raise ValueError(f'Duplicate {method} decision key {key}')
                keyed[key] = row
                if key not in expected:
                    raise ValueError(f'Unexpected {method} decision key {key}')
            except Exception as exc:
                errors.append(dict(method=method, error=str(exc)))
        for key in sorted(expected):
            report = dict(method=method, pool=key[0], seed=key[1])
            try:
                if key in duplicates:
                    raise ValueError('Duplicate decision key')
                row = keyed[key]
                if row.get('status') != 'PASS':
                    raise ValueError(row.get('error', 'Run status is not PASS'))
                condition = condition_map[key]
                state = checkpoint(row, deadline, moment=method == 'M')
                midlo, midhi = F(condition['midpoint_lower']), F(condition['midpoint_upper'])
                if midlo > midhi:
                    raise ValueError('Reversed midpoint interval')
                label = ('NO_BOUND_IN_BUDGET' if state is None else
                         'STOP' if F(state['upper']) <= factor * midlo else
                         'VIOLATION' if F(state['lower']) > factor * midhi else 'UNRESOLVED')
                stop = label == 'STOP'
                counts[method][key[0]] += int(stop)
                report.update(status='PASS', decision=label, stop_at_120s=stop if deadline == 120 else None,
                              stop_in_budget=stop, seconds=state['seconds'] if state else None,
                              lower=state['lower'] if state else None, upper=state['upper'] if state else None,
                              midpoint_lower=str(midlo), midpoint_upper=str(midhi),
                              exact_stop_threshold=str(factor * midlo), exact_threshold=True)
            except Exception as exc:
                report.update(status='ERROR', error=str(exc))
                errors.append(dict(method=method, pool=key[0], seed=key[1], error=str(exc)))
            reports.append(report)
    return dict(status='PASS' if not errors else 'COMPLETED_WITH_ERRORS', deadline=deadline,
                increment_fraction=str(factor - 1), expected_conditions=len(expected),
                counts=counts, total_stops={k: sum(v) for k, v in counts.items()},
                reports=reports, errors=errors)


def aggregate(args, families):
    root = path_arg(args.aggregate_root)
    report = dict(mode='aggregate_only', reran_timed_search=False, families={})
    loaded = {}
    for family in families:
        try:
            loaded[family] = rows_from(root / f'{family}_baseline_extended.json')
            report['families'][family] = family_summary(family, loaded[family])
        except Exception as exc:
            report['families'][family] = dict(status='ERROR', error=str(exc))
    if 'decisions' in families:
        try:
            condition_value = load(path_arg(args.decision_conditions))
            conditions = condition_value if isinstance(condition_value, list) else condition_value['conditions']
            report['decision_comparison'] = decision_comparison(
                loaded.get('decisions', []), rows_from(path_arg(args.refined_decisions)),
                conditions, load(B / 'decision_settings.json'), args.deadline)
        except Exception as exc:
            report['decision_comparison'] = dict(status='ERROR', error=str(exc))
    checks = list(report['families'].values())
    if 'decision_comparison' in report:
        checks.append(report['decision_comparison'])
    report['status'] = 'PASS' if all(r['status'] == 'PASS' for r in checks) else 'COMPLETED_WITH_ERRORS'
    if args.report and not args.dry_run:
        write(path_arg(args.report), report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report['status'] == 'PASS' else 1


def choose_cpu(requested):
    import psutil
    process = psutil.Process()
    allowed = sorted(process.cpu_affinity())
    candidates = [cpu for cpu in allowed if cpu >= 4]
    chosen = requested if requested is not None else (candidates[-1] if candidates else None)
    if chosen is None or chosen not in allowed:
        raise ValueError(f'Choose --cpu from {allowed}; default reserves logical CPUs 0--3')
    process.cpu_affinity([chosen])
    return chosen


def run_one(file, out, budget, cpu):
    from centered_signal_energy import ObservedAnchor, proposal, constants, CenteredSignalSearch
    from run_real_world import coarse
    meta = load(file)
    instance = file.name[:-len('_meta.json')]
    original = load(file.parent / f'{instance}_adaptive_energy.json')['certificate']
    start = time.monotonic()
    anchor = ObservedAnchor(original['lo'], original['hi'], original['y'], original['beta'], original['lambda_value'])
    caps, witness = coarse(anchor)
    basis = proposal(anchor, meta)
    receipt = constants(anchor, basis, meta, reference_only=True)
    solver = CenteredSignalSearch(anchor, basis, receipt, 'moment_centered')
    solver.best_original, solver.best_witness = caps['common_lower'], witness
    setup = float(meta.get('input_seconds', 0)) + time.monotonic() - start
    result = solver.run(time_limit=max(.001, budget - setup), absolute_tolerance=1e-10,
                        relative_tolerance=0, record=True)
    result.update(dataset=meta['dataset'], seed=meta['seed'], instance=instance,
                  complete_seconds=setup + result['seconds'], setup_seconds=setup,
                  complete_budget_seconds=budget, logical_cpu_affinity=[cpu], numerical_threads=1,
                  trace_time_basis='solver seconds; add setup_seconds for complete checkpoints',
                  input_seconds=float(meta.get('input_seconds', 0)),
                  input_contract='regenerated fixed stored table; new anchor/caps/proposal/constants/search timed',
                  initial_event=dict(seconds=setup, lower=caps['common_lower'], upper=caps['common_cap']))
    result['checkpoints'] = {str(t): checkpoint(result, t, moment=True) for t in CHECKPOINTS}
    result['checkpoint_time_basis'] = 'complete algorithm seconds'
    # Full table-containing evidence stays beside the local run, never in the
    # exported aggregate JSON.
    write(out / f'{instance}_moment.json', result)
    result.pop('certificate')
    return result


def run(args, families):
    if args.dry_run:
        plans = []
        for family in families:
            folder = path_arg(args.frozen_root) / f'frozen_{family}'
            files = sorted(folder.glob('*_meta.json'))
            plans.append(dict(family=family, instances=len(files), input=str(folder),
                              complete_budget_seconds=args.budget))
        print(json.dumps(dict(mode='run_plan', reran_timed_search=False, plans=plans), indent=2))
        return 0
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    cpu = choose_cpu(args.cpu)
    root = path_arg(args.output_root)
    failed = False
    for family in families:
        folder = path_arg(args.frozen_root) / f'frozen_{family}'
        out = root / f'{family}_baseline_extended'
        files = sorted(folder.glob('*_meta.json'))
        rows = []
        if not files:
            rows.append(dict(status='ERROR', error=f'No regenerated *_meta.json in {folder}'))
        for file in files:
            instance = file.name[:-len('_meta.json')]
            saved = out / f'{instance}_moment.json'
            try:
                if saved.exists():
                    row = load(saved)
                    if row.get('status') != 'PASS' or row.get('complete_budget_seconds') != args.budget:
                        raise ValueError('Existing result differs or failed; use a fresh --output-root')
                    row.pop('certificate', None)
                else:
                    row = run_one(file, out, args.budget, cpu)
            except Exception as exc:
                row = dict(status='ERROR', instance=instance, error=str(exc), traceback=traceback.format_exc())
            rows.append(row)
            write(out / 'results.json', rows)
            print(json.dumps({k: row.get(k) for k in ('status', 'instance', 'absolute_gap',
                             'complete_seconds', 'stop_reason', 'error')}), flush=True)
        write(root / f'{family}_baseline_extended.json', rows)
        failed |= any(row.get('status') != 'PASS' for row in rows)
    return int(failed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('all',) + FAMILIES, default='all')
    parser.add_argument('--cpu', type=int, default=os.environ.get('REGRET_CPU'),
                        help='Logical CPU; default uses the last allowed CPU outside reserved CPUs 0--3')
    parser.add_argument('--frozen-root', default='.', help='Parent of locally regenerated frozen_* directories')
    parser.add_argument('--output-root', default='extended_reproduction', help='Local runs and aggregate exports')
    parser.add_argument('--budget', type=float, default=600, help='Complete run cap; published extension uses 600 seconds')
    parser.add_argument('--aggregate-only', action='store_true', help='Replay aggregate checkpoints without solvers or raw data')
    parser.add_argument('--aggregate-root', default=str(RELEASE), help='Directory of *_baseline_extended.json summaries')
    parser.add_argument('--refined-decisions', default=str(RELEASE / 'frozen_decisions.json'),
                        help='Aggregate JSON or regenerated frozen_decisions directory')
    parser.add_argument('--decision-conditions', default=str(B.parent / 'results/energy_decisions/conditions.json'),
                        help='Midpoint conditions JSON or regenerated energy_decisions/results.json')
    parser.add_argument('--deadline', type=float, default=120, help='Common decision checkpoint in complete seconds')
    parser.add_argument('--report', help='Optional aggregate report file; aggregate-only mode')
    parser.add_argument('--dry-run', action='store_true', help='Write no files; aggregate-only mode still computes the full report')
    args = parser.parse_args()
    if args.budget <= 0 or args.deadline <= 0:
        parser.error('Budgets and deadlines must be positive')
    families = FAMILIES if args.family == 'all' else (args.family,)
    return aggregate(args, families) if args.aggregate_only else run(args, families)


if __name__ == '__main__':
    raise SystemExit(main())
