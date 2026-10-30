"""Concrete fixed-point predictor contract, checked without source-data exports."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import argparse,json
from collections import Counter
import numpy as np

def verify(cert):
    beta=np.array(cert['beta']);lo=np.array(cert['lo']);hi=np.array(cert['hi']);x=(lo+hi)/2
    assert np.all(beta*16==np.rint(beta*16))
    assert np.all(x*1024==np.rint(x*1024))
    b=np.rint(beta*16).astype(np.int64);xx=np.rint(x*1024).astype(np.int64)
    assert np.max(b)<=127 and np.min(b)>=-128
    assert np.max(xx)<=32767 and np.min(xx)>=-32768
    # Rowwise l1 bounds also cover every legal encoded completion and every
    # accumulation order, before a 32-bit multiply/accumulate is attempted.
    assert np.max(hi*1024)<=32767 and np.min(lo*1024)>=-32768
    endpoint_codes=np.ceil(np.maximum(abs(lo),abs(hi))*1024).astype(np.int64)
    bound=np.sum(endpoint_codes*abs(b),axis=1)
    assert np.max(bound)<2**31
    stored_inputs=xx.astype(np.int16);stored_coefficients=b.astype(np.int8)
    predicted=stored_inputs.astype(np.int32)@stored_coefficients.astype(np.int32)
    decoded=predicted.astype(float)/2**14
    assert np.array_equal(decoded,x@beta)
    return dict(status='PASS',coefficient_bits=8,coefficient_fraction_bits=4,input_bits=16,
        input_fraction_bits=10,accumulator_bits=32,rows=len(x),features=len(beta),
        largest_coefficient_code=int(abs(b).max()),largest_input_code=int(abs(xx).max()),
        largest_absolute_accumulation_bound=float(bound.max()),exact_decoded_predictions=True)

def condition_key(row):
    return row['dataset'],int(row['seed']),row.get('pool')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-results',default=os.environ.get('REGRET_SOURCE_RESULTS',Path(__file__).parent/'contraction_run/results.json'))
    parser.add_argument('--output',default=os.environ.get('REGRET_FIXED_POINT_OUTPUT'))
    parser.add_argument('--expected-count',type=int,required=True,help='Expected unique conditions: primary 20; decisions 15')
    args=parser.parse_args()
    if args.expected_count<=0:parser.error('--expected-count must be positive')
    sourcefile=Path(args.source_results);source=json.loads(sourcefile.read_text());rows=[];errors=[];keys=[]
    settings=source.get('settings',{})
    if settings.get('designs') and settings.get('mask_seeds'):
        expected={(spec['dataset'],int(seed),None) for spec in settings['designs'] for seed in settings['mask_seeds']}
    elif settings.get('pools') and settings.get('mask_seeds'):
        expected={(settings['dataset'],int(seed),pool) for pool in range(settings['pools']) for seed in settings['mask_seeds']}
    else:
        expected={condition_key(row) for row in source.get('conditions',[])}
    if not expected:errors.append('No expected condition grid in source settings or conditions')
    if len(expected)!=args.expected_count:errors.append(f'Expected grid has {len(expected)} unique conditions, requested {args.expected_count}')
    for row in source['results']:
        if row.get('method') not in ('energy','adaptive_energy'):continue
        report={k:row[k] for k in ('dataset','seed','pool') if k in row}
        try:
            keys.append(condition_key(row))
            if row.get('status')!='PASS':raise ValueError(row.get('error','Source run status is not PASS'))
            cert=json.loads((sourcefile.parent/row['certificate_file']).read_text())['certificate']
            report.update(verify(cert))
        except Exception as exc:report.update(status='ERROR',error=str(exc))
        rows.append(report)
    duplicates=[key for key,count in Counter(keys).items() if count>1]
    missing=sorted(expected-set(keys),key=str);unexpected=sorted(set(keys)-expected,key=str)
    if duplicates:errors.append('Duplicate condition keys')
    if missing:errors.append('Missing expected condition keys')
    if unexpected:errors.append('Unexpected condition keys')
    if len(rows)!=args.expected_count or len(set(keys))!=args.expected_count:errors.append('Selected run count or unique condition count differs from --expected-count')
    passed=sum(row.get('status')=='PASS' for row in rows)
    status='PASS' if not errors and passed==args.expected_count else 'COMPLETED_WITH_ERRORS'
    out=Path(args.output) if args.output else sourcefile.parent/'fixed_point_contract.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(status=status,conditions=len(rows),passed=passed,expected_conditions=args.expected_count,
        unique_conditions=len(set(keys)),missing=missing,unexpected=unexpected,duplicates=duplicates,errors=errors,reports=rows),indent=2))
    print('fixed-point contract',status,passed,'/',args.expected_count,'conditions')
    return 0 if status=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
