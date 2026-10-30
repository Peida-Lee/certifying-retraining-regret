"""Unchanged four dense-row tables, original 1e-8 target, joint refinement."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,json,traceback
import numpy as np,psutil
from sklearn.datasets import load_diabetes
import run_general_refinement as general
import planar_oracle as planar
B=Path(__file__).resolve().parent

def problem(name,**unused):
    if name=='dense_diabetes':
        o=load_diabetes();xx=np.round((o.data-o.data.mean(0))/np.ptp(o.data,axis=0)*1024)/1024
        x=np.column_stack([np.ones(len(xx)),xx]);y=np.round((o.target-o.target.min())/np.ptp(o.target)*1024)/1024;k=10
    else:
        k=int(name.rsplit('_',1)[1]);rng=np.random.default_rng(77116)
        xx=np.round(rng.normal(size=(128,16))/4*1024)/1024;x=np.column_stack([np.ones(128),xx])
        coeff=np.array([1/(j+2) for j in range(17)]);noise=np.round(rng.normal(size=128)*.1*1024)/1024
        y=np.round((x@coeff+noise)*1024)/1024
    lam=1/32;beta=np.linalg.solve(x.T@x/len(x)+lam*np.eye(x.shape[1]),x.T@y/len(x))
    lo=x.copy();hi=x.copy();lo[:4,1:k+1]-=1/16;hi[:4,1:k+1]+=1/16
    return (lo,hi,y,beta,lam),dict(dataset=name,k=k,uncertain_rows=4,original_target=1e-8)

def main():
    p=argparse.ArgumentParser();p.add_argument('--condition',type=int,required=True);p.add_argument('--cpu',type=int,default=8)
    p.add_argument('--output',default='dense_refined');a=p.parse_args();psutil.Process().cpu_affinity([a.cpu])
    general.problem=problem;general.ObservedAnchor=planar.PlanarObservedAnchor;general.MomentSearch=planar.PlanarMomentSearch
    general.basis_proposal=planar._moment.basis_proposal;general.tail_receipt=planar._moment.tail_receipt
    general.GRID=[('dense_diabetes',1/8,'all_features',77101)]+[(f'dense_synthetic_{k}',1/8,'first_columns',77116) for k in (4,8,16)]
    out=B/a.output/f'condition_{a.condition:02d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise FileExistsError(out/'result.json')
    try:r=general.run(a.condition,out,cap=1800,target=1e-8,first_rank=11 if a.condition==0 else 2,first_seconds=2)
    except Exception as exc:
        r=dict(status='ERROR',condition=a.condition,error=str(exc),traceback=traceback.format_exc())
        (out/'result.json').write_text(json.dumps(r,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k not in ('nodes','history')}),flush=True)

if __name__=='__main__':main()
