"""Same-instance ZORRO fixed-point model enclosure converted to training regret."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import json,time,traceback
import numpy as np,psutil
from centered_signal_energy import ObservedAnchor
from run_real_world import coarse
from zorro_regret import run

def main():
    psutil.Process().cpu_affinity([int(os.environ.get('REGRET_TIMED_CPU','10'))])
    sourcefile=Path(os.environ.get('REGRET_SOURCE_RESULTS',Path(__file__).parent/'contraction_run/results.json'))
    source=json.loads(sourcefile.read_text())
    output=Path(os.environ.get('REGRET_ZORRO_OUTPUT',sourcefile.parent/'zorro_matched_results.json'))
    assert not output.exists();records=[]
    for condition in source['conditions']:
        if condition['seed']!=source['settings']['mask_seeds'][0]:continue
        row=dict(dataset=condition['dataset'],seed=condition['seed'])
        try:
            saved=next(r for r in source['results'] if r['dataset']==row['dataset'] and r['seed']==row['seed'] and r['method'] in ('energy','adaptive_energy'))
            cert=json.loads((sourcefile.parent/saved['certificate_file']).read_text())['certificate']
            args=[np.array(cert[k],float) for k in ('lo','hi','y','beta')]+[cert['lambda_value']]
            start=time.monotonic();a=ObservedAnchor(*args,anchor='observed',preconditioner='anchor');anchor_s=time.monotonic()-start
            start=time.monotonic();caps,witness=coarse(a);cap_s=time.monotonic()-start
            common=condition['input_seconds']+anchor_s+cap_s
            row.update(n=a.n,p=a.p,same_saved_instance=True,input_seconds=condition['input_seconds'])
            row.update(run(a,common,10.,witness))
        except Exception as exc:row.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
        records.append(row);output.write_text(json.dumps(dict(status='RUNNING',results=records),indent=2))
        print(row['dataset'],row['status'],row.get('fixed_point_cover_complete'),row.get('absolute_gap'),row.get('total_seconds'),row.get('error',''),flush=True)
    output.write_text(json.dumps(dict(status='PASS' if len(records)==4 and all(r['status']=='PASS' for r in records) else 'COMPLETED_WITH_ERRORS',results=records),indent=2))

if __name__=='__main__':main()
