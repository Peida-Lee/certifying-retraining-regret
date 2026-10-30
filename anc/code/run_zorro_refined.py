"""Portable full-budget ZORRO refinement on primary first-mask instances."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,time,traceback
import numpy as np,psutil
from centered_signal_energy import ObservedAnchor
from run_real_world import coarse
from zorro_refined import run

def main():
    psutil.Process().cpu_affinity([int(os.environ.get('REGRET_TIMED_CPU','10'))])
    sourcefile=Path(os.environ.get('REGRET_SOURCE_RESULTS',Path(__file__).parent/'contraction_run/results.json'))
    source=json.loads(sourcefile.read_text());first=source['settings']['mask_seeds'][0]
    out=Path(os.environ.get('REGRET_ZORRO_REFINED_OUTPUT',sourcefile.parent/'zorro_refined_results.json'))
    assert not out.exists();rows=[]
    for meta in source['conditions']:
        if meta['seed']!=first:continue
        row=dict(dataset=meta['dataset'],seed=meta['seed'])
        try:
            old=next(r for r in source['results'] if r['dataset']==meta['dataset'] and r['seed']==first and r['method'] in ('energy','adaptive_energy'))
            cert=json.loads((sourcefile.parent/old['certificate_file']).read_text())['certificate']
            args=[np.array(cert[k],float) for k in ('lo','hi','y','beta')]+[cert['lambda_value']]
            start=time.monotonic();a=ObservedAnchor(*args,anchor='observed',preconditioner='anchor');anchor=time.monotonic()-start
            start=time.monotonic();caps,wit=coarse(a);cap=time.monotonic()-start
            row.update(run(a,meta['input_seconds']+anchor+cap,10.,wit))
        except Exception as exc:row.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
        rows.append(row);out.write_text(json.dumps(dict(status='RUNNING',results=rows),indent=2))
        print(row['dataset'],row['status'],row.get('absolute_gap'),flush=True)
    out.write_text(json.dumps(dict(status='PASS' if len(rows)==4 and all(r['status']=='PASS' for r in rows) else 'COMPLETED_WITH_ERRORS',results=rows),indent=2))

if __name__=='__main__':main()
