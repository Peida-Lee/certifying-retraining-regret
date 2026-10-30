"""Exact Schur identities and original-table checks for the profile query."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from fractions import Fraction as F
import sys,json,copy
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import numpy as np,psutil
from observed_anchor import ObservedAnchor
from local_residual import LocalResidualSearch
from profile_query_v31 import prepare,audit_input
from compressed_edge_arrow_v27 import prepare_compressed,recover
from audit_compressed_arrow_v27 import verify,inverse,dot


def loss(x,y,b,lam):return lam*dot(b,b)+sum(((dot(row,b)-v)**2 for row,v in zip(x,y)),F(0))/len(y)


def main():
    allowed=psutil.Process().cpu_affinity()
    psutil.Process().cpu_affinity([12] if 12 in allowed else [allowed[-1]])
    rng=np.random.default_rng(93211);count=0
    for k in (1,2,3):
        n,p=12,5;x=rng.integers(-4,5,(n,p))/8;x[:,0]=1;y=rng.integers(-8,9,n)/8;beta=rng.integers(-4,5,p)/16
        lo=x.copy();hi=x.copy();lo[:3,1:k+1]-=1/16;hi[:3,1:k+1]+=1/16;lam=1/8
        a=ObservedAnchor(lo,hi,y,beta,lam);search=LocalResidualSearch(a);search.lower(x,search.fit(x));ids=search.ids.tolist()
        raw=dict(original=dict(lo=lo.tolist(),hi=hi.tolist(),y=y.tolist(),beta=beta.tolist(),lambda_value=lam),
            problem=dict(theta=search.fit(x).tolist(),T=a.t.tolist(),mu=a.mu,active_cells=ids,x0_uncertain=[lo[i,j] for i,j in ids]),
            witness=search.witness,result=dict(lower=search.best,target=3e-6))
        t=rng.integers(-4,5,k)/16;data=prepare(raw,t);audit_input(data)
        rows,const,c,g0,origins,V,oracles,points,error=prepare_compressed(data)
        assert all(len(r['coordinate_indices'])==1 for r in rows)
        proposal=recover([np.eye(1)*1000 for _ in rows],[np.zeros(1) for _ in rows],V,c,const,oracles,[None]*len(rows))
        proposal.update(coordinate_scales=[r['coordinate_scale'].tolist() for r in rows],coordinate_indices=[r['coordinate_indices'].tolist() for r in rows],
            gscale=rows[0]['gscale'],reference_theta=data['problem']['theta'],partition_theta=data['partition_theta'],residual_domains=[None]*len(rows),halfspace_multipliers=[])
        checked=verify(data,proposal);upper=F(checked['exact_upper']);J=data['problem']['profile_columns'];K=[j for j in range(p) if j not in J];yf=list(map(F,y));bf=list(map(F,beta))
        for rep in range(12):
            xx=lo+(hi-lo)*rng.integers(0,17,(n,p))/16;xf=[list(map(F,row)) for row in xx]
            M=[[F(lam)*(i==j)+sum((row[i]*row[j] for row in xf),F(0))/n for j in range(p)] for i in range(p)]
            rhs=[sum((row[j]*yi for row,yi in zip(xf,yf)),F(0))/n for j in range(p)]
            A=[[M[i][j] for j in K] for i in K];Ai=inverse(A);base=list(map(F,data['problem']['theta']))
            residual=[dot(M[i],base)-rhs[i] for i in K];fit=[-dot(row,residual) for row in Ai];full=base.copy()
            for j,v in zip(K,fit):full[j]=v
            direct=loss(xf,yf,bf,F(lam))-loss(xf,yf,full,F(lam))
            profiled=loss(xf,yf,bf,F(lam))-loss(xf,yf,base,F(lam))+dot(residual,[dot(row,residual) for row in Ai])
            assert direct==profiled and direct<=upper;count+=1
        bad=copy.deepcopy(data);bad['problem']['mu']*=2
        try:audit_input(bad)
        except AssertionError:pass
        else:raise AssertionError('Invalid profile curvature accepted')
    report=dict(status='PASS',exact_original_schur_identities=count,original_table_bound_checks=count,invalid_metric_rejections=3,raw_records_opened=False)
    (B/'profile_offline_v32.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
