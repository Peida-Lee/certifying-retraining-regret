"""Profile fully observed columns; certify a fixed retained comparator query."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
from fractions import Fraction as F
from math import lcm
import sys,json,copy,time,argparse,traceback
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import numpy as np,psutil
from observed_anchor import float_down
from audit_sdp_exact import denominator,int_array,dot
from compressed_edge_arrow_v27 import solve
from audit_compressed_arrow_v27 import verify


def exact_metric(original,T):
    lo=np.array(original['lo']);hi=np.array(original['hi']);n,p=lo.shape;J=np.flatnonzero(np.any(lo!=hi,axis=0));K=np.setdiff1d(np.arange(p),J)
    X=lo[:,K];dx=denominator(X.flat);Xi=int_array(X,dx);den=n*dx*dx;Gi=Xi.T@Xi;lam=F(original['lambda_value'])
    assert (lam*den).denominator==1
    for i in range(len(K)):Gi[i,i]+=int(lam*den)
    T=np.asarray(T);assert T.shape==(p,len(K)) and np.all(T[J]==0)
    dt=denominator(T.flat);Ti=int_array(T[K],dt);metric=Ti.T@Gi@Ti;md=den*dt*dt
    mu=min(F(int(metric[i,i]-sum(abs(metric[i,j]) for j in range(len(K)) if j!=i)),md) for i in range(len(K)))
    return mu,J,K


def prepare(data,t=None):
    new=copy.deepcopy(data);o=new['original'];lo=np.array(o['lo']);hi=np.array(o['hi']);n,p=lo.shape
    J=np.flatnonzero(np.any(lo!=hi,axis=0));K=np.setdiff1d(np.arange(p),J);assert 0<len(J)<p
    X=lo[:,K];A=X.T@X/n+o['lambda_value']*np.eye(len(K));T=np.zeros((p,len(K)))
    T[K]=np.round(np.linalg.inv(np.linalg.cholesky(A)).T*2**30)/2**30
    mu,_,_=exact_metric(o,T);assert mu>0
    theta=np.zeros(p);theta[J]=np.array(new['witness']['theta'])[J] if t is None else np.array(t)
    new['problem'].update(T=T.tolist(),mu=float_down(mu),theta=theta.tolist(),profile_columns=J.tolist())
    new['problem'].pop('uniform_curvature_rule',None);new.pop('fixed_gscale',None);new['partition_theta']=theta.tolist()
    return new


def audit_input(data):
    o=data['original'];s=data['problem'];lo=np.array(o['lo']);hi=np.array(o['hi']);n,p=lo.shape
    assert np.all(np.isfinite(lo)) and np.all(lo<=hi);mu,J,K=exact_metric(o,s['T'])
    assert J.tolist()==s['profile_columns'] and 0<F(s['mu'])<=mu and np.all(np.array(s['theta'])[K]==0)
    assert set(map(tuple,s['active_cells']))==set(map(tuple,np.argwhere(lo!=hi)))
    x=np.array(data['witness']['x']);beta=list(map(F,o['beta']));theta=list(map(F,data['witness']['theta']));y=list(map(F,o['y']))
    assert x.shape==lo.shape and np.all(x>=lo) and np.all(x<=hi)
    value=F(o['lambda_value'])*sum((b*b-t*t for b,t in zip(beta,theta)),F(0))
    for xx,yy in zip(x,y):value+=((dot(list(map(F,xx)),beta)-yy)**2-(dot(list(map(F,xx)),theta)-yy)**2)/n
    assert F(data['result']['lower'])<=value
    return dict(status='PASS',original_witness=True,profile_columns=J.tolist(),eliminated_columns=len(K),exact_known_column_metric=True,
                scope='Fixed retained-comparator upper query; outer parameter coverage is required for a global regret bound')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--cpu',type=int,required=True);a=p.parse_args();psutil.Process().cpu_affinity([a.cpu]);source=Path(a.source);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        data=prepare(json.loads((source/'input.json').read_text()));(out/'input.json').write_text(json.dumps(data))
        def emit(best,item):
            (out/'proposal.json').write_text(json.dumps(best));print(json.dumps(item),flush=True)
        ans=solve(data,seconds=120,lower=data['result']['lower'],max_iterations=15,emit=emit)
        checked=verify(data,ans);(out/'exact_audit.json').write_text(json.dumps(checked,indent=2))
        result=dict(status='EXACT_PROFILE_QUERY',upper=checked['upper'],gap_to_global_lower=checked['gap'],seconds=time.monotonic()-start,
                    retained_parameters=len(data['problem']['profile_columns']),scope='One query, not a global certificate')
        (out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:(out/'error.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc()),indent=2));raise
