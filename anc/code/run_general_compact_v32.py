"""Complete-cost original-grid certification with common compact fallbacks."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import json,sys,time,argparse,traceback
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import numpy as np,psutil
from public_problems import problem
from observed_anchor import ObservedAnchor,float_up
from moment_tail import MomentSearch,basis_proposal,tail_receipt
from run_general_refinement import GRID
from fresh_arrow_v29 import run as root_run
from profile_outer_v31 import run as profile_run
from pilot_arrow_tree_v28 import run as row_run

PROTOCOL=dict(version='v32',target=3e-6,complete_cap_seconds=9000,conditions=60,
    first_stage='Ranks1/2/3, tail budget2.5e-6, one second retained search; original midpoint preconditioner',
    root='64 feasible starts; matrix Young curvature tau1/2;16 reference rounds with32 starts;90s small-cone proposal; exact local/central/row verification',
    profile='At most3 uncertain columns: exact observed-column elimination with rounded verified metric, curvature interpolation,1023 outer boxes',
    rows='More than3 uncertain columns: fixed-reference residual slabs,513 nodes,60s per proposal,10 exchange rounds',
    inputs='Original60 conditions, original candidates, weights, feature intervals and penalties',
    clock='All input/preprocessing, reference and feasible searches, numerical proposals, exact bound evaluations, completed partial children and output; independent replay separately',
    failures='Retain the valid parent and original records; continue other conditions',threads=1)


def run(cid,out):
    start=time.monotonic();elapsed=lambda:time.monotonic()-start;name,width,spread,seed=GRID[cid]
    args,meta=problem(name=name,cells=128,width=width,spread=spread,seed=seed,subset_seed=74101)
    anchor=ObservedAnchor(*args,preconditioner='midpoint');basis=basis_proposal(anchor);candidates=[]
    for rank in (1,2,3):
        tail=tail_receipt(anchor,basis[:,:rank])['tail_bound'];candidates.append((rank,tail))
        if tail<=2.5e-6:break
    rank=next((r for r,t in candidates if t<=2.5e-6),min(candidates,key=lambda x:x[1])[0])
    first=MomentSearch(anchor,rank).run(time_limit=1,absolute_tolerance=PROTOCOL['target'],record=True)
    (out/'first_stage.json').write_text(json.dumps(first));o=dict(lo=anchor.lo.tolist(),hi=anchor.hi.tolist(),y=anchor.y.tolist(),beta=anchor.beta.tolist(),lambda_value=anchor.lam)
    (out/'original_input.json').write_text(json.dumps(o));lower=F(first['lower']);upper=F(first['upper']);target=F(PROTOCOL['target']);history=[];errors=[]
    def record(stage):
        history.append(dict(stage=stage,seconds=elapsed(),lower=float(lower),upper=float_up(upper),gap=float_up(upper-lower)))
        r=dict(status='RUNNING',condition=cid,dataset=name,width=width,spread=spread,seed=seed,selected_rank=rank,rank_candidates=candidates,
            lower=float(lower),upper=float_up(upper),absolute_gap=float_up(upper-lower),target=float(target),target_met=upper-lower<=target,
            target_met_in_budget=any(h['seconds']<=PROTOCOL['complete_cap_seconds'] and F(h['upper'])-F(h['lower'])<=target for h in history),
            total_seconds=elapsed(),cap_seconds=PROTOCOL['complete_cap_seconds'],history=history,errors=errors,protocol=PROTOCOL)
        (out/'result.json').write_text(json.dumps(r,indent=2));return r
    r=record('structural')
    try:
        if upper-lower>target:
            root=out/'root';root.mkdir();rr=root_run(cid,root,iterations=16,use_young=True,root_only=True)
            lower=max(lower,F(rr['lower']));upper=min(upper,F(rr['upper']));r=record('compact_root')
            if upper-lower>target and elapsed()<PROTOCOL['complete_cap_seconds']:
                remaining=PROTOCOL['complete_cap_seconds']-elapsed();count=int(np.sum(np.any(anchor.lo!=anchor.hi,axis=0)))
                if count<=3:
                    child=out/'profile';child.mkdir();pr=profile_run(root,child,cap=remaining,maxnodes=1023);upper=min(upper,F(pr['upper']));r=record('profile_cover')
                else:
                    child=out/'rows';child.mkdir();pr=row_run(root,child,maxnodes=513,cap=remaining,reference_updates=0);upper=min(upper,F(pr['upper']));r=record('row_cover')
    except Exception as exc:
        errors.append(dict(error=str(exc),traceback=traceback.format_exc()));r=record('valid_parent_after_error')
    r['status']='COMPLETED' if not errors else 'COMPLETED_WITH_ERRORS';r['total_seconds']=elapsed()
    (out/'result.json').write_text(json.dumps(r,indent=2));return r


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--cpu',type=int,required=True);a=p.parse_args();psutil.Process().cpu_affinity([a.cpu]);out=Path(a.output)/f'condition_{a.condition:02d}';out.mkdir(parents=True,exist_ok=False)
    try:r=run(a.condition,out)
    except Exception as exc:r=dict(status='ERROR',condition=a.condition,error=str(exc),traceback=traceback.format_exc());(out/'result.json').write_text(json.dumps(r,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k not in ('history','protocol')}),flush=True)
