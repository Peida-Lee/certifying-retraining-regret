"""All unchanged 18 multirank instances, exact sparse block elimination."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,traceback
from fractions import Fraction as F
B=Path(__file__).resolve().parent;
from multirank_run import problem,GRID
from arrowhead_certificate import certify
import numpy as np,psutil
psutil.Process().cpu_affinity([6]);out=B/'multirank_exact';out.mkdir(exist_ok=True);records=[]
for cid,(rank,delta,ell) in enumerate(GRID):
 try:
    start=time.monotonic();lo,hi,y,beta,lam=problem(rank,delta,ell);n,p=lo.shape
    # Detect centers from nonzero candidate coordinates, then verify every row
    # belongs to exactly one center/leaf pair before scalar elimination.
    centers=np.flatnonzero(beta);assert np.all(beta[centers]==1);receipts=[];seen=set()
    for center in centers:
        groups=[];leaf_ids=[];row_ids=[]
        for leaf in range(p):
            if leaf in centers:continue
            ids=np.flatnonzero((lo[:,leaf]!=0)|(hi[:,leaf]!=0));ids=ids[(lo[ids,center]!=0)|(hi[ids,center]!=0)]
            # At ell=0 the signal row is paired by its original sparse index.
            nuisance=[i for i in ids if lo[i,leaf]!=hi[i,leaf]]
            if not nuisance:continue
            assert len(nuisance)==1;trow=int(nuisance[0]);srow=trow-n//2
            assert lo[srow,center]!=hi[srow,center] and lo[srow,leaf]==hi[srow,leaf]
            assert y[srow]==2 and lo[trow,center]==hi[trow,center]
            other=[j for j in range(p) if j not in (int(center),leaf)]
            assert not np.any(lo[np.ix_([srow,trow],other)]) and not np.any(hi[np.ix_([srow,trow],other)])
            assert srow not in seen and trow not in seen;seen.update([srow,trow])
            groups.append((lo[srow,leaf],lo[trow,center],y[trow],lo[srow,center],hi[srow,center],lo[trow,leaf],hi[trow,leaf]))
            leaf_ids.append(leaf);row_ids.append([srow,trow])
        rec=certify(groups,n,F(lam),target=F(1,100000*len(centers)));rec.update(center=int(center),leaf_ids=leaf_ids,row_ids=row_ids);receipts.append(rec)
    assert len(seen)==n
    lower=sum((F(r['lower']) for r in receipts),F(0));upper=sum((F(r['upper']) for r in receipts),F(0))
    row=dict(status='PASS',condition=cid,rank=rank,delta=str(delta),ell=str(ell),lower=str(lower),upper=str(upper),absolute_gap=float(upper-lower),
        total_seconds=time.monotonic()-start,target_met=upper-lower<=F(1,100000),blocks=len(receipts),points=sum(len(r['points']) for r in receipts))
    (out/f'condition_{cid}_receipt.json').write_text(json.dumps(dict(result=row,blocks=receipts),indent=2))
 except Exception as exc:row=dict(status='ERROR',condition=cid,error=str(exc),traceback=traceback.format_exc())
 records.append(row);print(json.dumps(row),flush=True);(out/'results.json').write_text(json.dumps(records,indent=2))
