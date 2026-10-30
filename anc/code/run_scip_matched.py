"""SCIP on exactly the saved primary instances and midpoint witnesses.

Run after run_contraction_real.py; set REGRET_SOURCE_RESULTS to that results.json.
The input/candidate stage is reused with its recorded cost; anchor, cap and SCIP
construction/solve costs are newly measured. No source row is dropped on error.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,time,traceback
import numpy as np
import psutil
from centered_signal_energy import ObservedAnchor
from run_real_world import coarse
from run_scip import run

def main():
    psutil.Process().cpu_affinity([int(os.environ.get('REGRET_TIMED_CPU','10'))])
    sourcefile=Path(os.environ.get('REGRET_SOURCE_RESULTS',Path(__file__).parent/'contraction_run/results.json'))
    source=json.loads(sourcefile.read_text())
    output=Path(os.environ.get('REGRET_SCIP_OUTPUT',sourcefile.parent/'scip_matched_results.json'))
    assert not output.exists()
    records=[]
    for condition in source['conditions']:
        if os.environ.get('REGRET_ALL_MASKS','1')!='1' and condition['seed']!=source['settings']['mask_seeds'][0]:continue
        if str(condition['seed']) in os.environ.get('REGRET_SKIP_SEEDS','').split(','):continue
        row=dict(dataset=condition['dataset'],seed=condition['seed'])
        try:
            saved=next(r for r in source['results'] if r['dataset']==row['dataset'] and r['seed']==row['seed'] and r['method'] in ('energy','adaptive_energy'))
            cert=json.loads((sourcefile.parent/saved['certificate_file']).read_text())['certificate']
            args=[np.array(cert[k],float) for k in ('lo','hi','y','beta')]+[cert['lambda_value']]
            start=time.monotonic();a=ObservedAnchor(*args,anchor='observed',preconditioner='anchor');anchor_s=time.monotonic()-start
            start=time.monotonic();caps,witness=coarse(a);cap_s=time.monotonic()-start
            common=condition['input_seconds']+anchor_s+cap_s
            row.update(n=a.n,p=a.p,input_seconds=condition['input_seconds'],anchor_seconds=anchor_s,coarse_seconds=cap_s,
                       same_saved_instance=True,midpoint_lower=witness['regret_lower'])
            tolerance=float(os.environ.get('REGRET_SCIP_TOLERANCE','1e-10'))
            row.update(search_tolerance=tolerance)
            row.update(run(a,common,10.,1e6,witness,absolute_tolerance=tolerance))
        except Exception as exc:
            row.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
        records.append(row)
        output.write_text(json.dumps(dict(status='RUNNING',results=records),indent=2))
        print(row['dataset'],row['status'],row.get('numeric_width'),row.get('total_seconds'),flush=True)
    output.write_text(json.dumps(dict(status='PASS' if records and all(r['status']=='PASS' for r in records) else 'COMPLETED_WITH_ERRORS',results=records),indent=2))

if __name__=='__main__':main()
