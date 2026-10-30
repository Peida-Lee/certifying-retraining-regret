"""Choose a certificate representation by its cone size, then certify original regret."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import json,time,argparse,traceback
import numpy as np,psutil
from public_problems import problem
from run_general_refinement import GRID,run as moment_run
from run_general_compact_v32 import run as compact_run
from observed_anchor import float_up

PROTOCOL=dict(version='v33',target=3e-6,complete_cap_seconds=9000,conditions=60,
    dispatch='Compare total symmetric cone entries: choose moment if smaller, otherwise compressed rows; depends only on p and row uncertain-cell counts',
    structural='Ranks1/2/3; tail budget2.5e-6; one-second retained search',
    compact='v32 compressed root and observed-column/profile or row-slab cover',
    moment='64 witness starts,8 reference exchanges with32 starts;60s root/40s child proposal;one reference update;33 evaluated nodes',
    inputs='Original60 conditions, original candidates, weights, feature intervals and penalties',
    clock='Dispatch input construction plus every delegated computation and exact bound evaluation; replay reported separately',threads=1)

def dispatch(lo,hi):
    p=len(lo[0]);ks=np.sum(np.asarray(lo)!=np.asarray(hi),axis=1);ks=ks[ks>0]
    order=1+int(sum(ks))+int(sum(k*(k+1)//2 for k in ks))
    moment=order*(order+1)//2
    compact=(p+1)*(p+2)//2+sum((p+int(k)+2)*(p+int(k)+3)//2 for k in ks)
    return dict(engine='moment' if moment<compact else 'compact',features=p,uncertain_rows=len(ks),
        uncertain_cells=int(sum(ks)),moment_order=order,moment_symmetric_entries=int(moment),compact_symmetric_entries=int(compact))

def run(cid,out):
    started=time.monotonic();name,width,spread,seed=GRID[cid]
    args,_=problem(name=name,cells=128,width=width,spread=spread,seed=seed,subset_seed=74101)
    route=dispatch(args[0],args[1]);elapsed=time.monotonic()-started
    (out/'dispatch.json').write_text(json.dumps(dict(**route,seconds=elapsed,protocol=PROTOCOL),indent=2))
    inner=out/'engine';inner.mkdir()
    offset=time.monotonic()-started
    if route['engine']=='compact':r=compact_run(cid,inner)
    else:r=moment_run(cid,inner,cap=max(.01,PROTOCOL['complete_cap_seconds']-offset),target=PROTOCOL['target'])
    history=[dict(h,seconds=offset+h['seconds']) for h in r['history']]
    for h in history:h['gap']=float_up(F(h['upper'])-F(h['lower']))
    keys=('condition','dataset','width','spread','seed','selected_rank','lower','upper','errors')
    result={k:r[k] for k in keys}
    result.update(status='COMPLETED' if not r['errors'] else 'COMPLETED_WITH_ERRORS',protocol=PROTOCOL,dispatch=route,
        absolute_gap=float_up(F(r['upper'])-F(r['lower'])),target=PROTOCOL['target'],
        target_met=F(r['upper'])-F(r['lower'])<=F(PROTOCOL['target']),
        target_met_in_budget=any(h['seconds']<=PROTOCOL['complete_cap_seconds'] and F(h['upper'])-F(h['lower'])<=F(PROTOCOL['target']) for h in history),
        total_seconds=time.monotonic()-started,cap_seconds=PROTOCOL['complete_cap_seconds'],dispatch_seconds=offset,
        engine_seconds=r['total_seconds'],history=history)
    (out/'result.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',type=int,required=True);p.add_argument('--cpu',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();psutil.Process().cpu_affinity([a.cpu]);out=Path(a.output)/f'condition_{a.condition:02d}';out.mkdir(parents=True,exist_ok=False)
    try:r=run(a.condition,out)
    except Exception as exc:r=dict(status='ERROR',condition=a.condition,error=str(exc),traceback=traceback.format_exc());(out/'result.json').write_text(json.dumps(r,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k not in ('history','protocol')}),flush=True)
