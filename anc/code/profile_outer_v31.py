"""Complete retained-parameter coverage after profiling observed columns."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import json,sys,time,argparse,traceback,itertools,copy,math
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import numpy as np,psutil
from observed_anchor import float_up,float_down,sqrt_up
from profile_query_v31 import prepare
from compressed_edge_arrow_v27 import solve,prepare_compressed
from audit_compressed_arrow_v27 import verify


def run(source,out,cap=1800,maxnodes=1023):
    start=time.monotonic();raw=json.loads((source/'input.json').read_text());prior=json.loads((source/'exact_audit.json').read_text())
    assert prior['status']=='EXACT_AUDIT_PASS' and raw['problem'].get('profile_columns') is None
    original=raw['original'];lo=np.array(original['lo']);hi=np.array(original['hi']);n,p=lo.shape;J=np.flatnonzero(np.any(lo!=hi,axis=0));assert len(J)<=3
    beta=np.array(original['beta']);lam=F(original['lambda_value']);lower=F(raw['result']['lower']);U0=F(prior['exact_upper']);target=F(raw['result']['target'])
    data=prepare(raw,beta[J]);rows,*_=prepare_compressed(data);data['fixed_gscale']=rows[0]['gscale'];(out/'input.json').write_text(json.dumps(data))
    radius=F(math.ceil(sqrt_up(U0/lam)*2**24),2**24);assert radius*radius*lam>=U0
    rootlo=[float_down(F(beta[j])-radius) for j in J];roothi=[float_up(F(beta[j])+radius) for j in J]
    curvature=[lam+sum((max(F(lo[i,j])**2,F(hi[i,j])**2) for i in range(n)),F(0))/n for j in J]
    queries={};query_records=[];nodes={};leaves=[];history=[];errors=[]
    (out/'protocol.json').write_text(json.dumps(dict(source=str(source),initial_exact_upper=str(U0),lower=str(lower),target=str(target),radius=str(radius),
        rootlo=rootlo,roothi=roothi,curvature=list(map(str,curvature)),profile_columns=J.tolist(),cap=cap,maxnodes=maxnodes,
        query='small arrow dual, exact polynomial and central Schur checks',split='curvature-weighted widest coordinate',cost='Profile stage only; inherited input and full-domain certificate are separate'),indent=2))
    def query(t):
        key=tuple(t)
        if key in queries:return queries[key]
        current=copy.deepcopy(data);theta=np.zeros(p);theta[J]=t;current['problem']['theta']=theta.tolist()
        ans=solve(current,seconds=30,lower=float(lower),max_iterations=10)
        checked=verify(current,ans,check_input=not queries);value=F(checked['exact_upper']);idx=len(queries)
        (out/f'query_{idx:04d}.json').write_text(json.dumps(dict(t=list(t),proposal=ans,audit=checked)))
        queries[key]=value;query_records.append(dict(id=idx,t=list(t),exact_upper=str(value),seconds=time.monotonic()-start))
        return value
    def bound(l,u,limit):
        points=list(itertools.product(*zip(l,u)));best=max(query(t) for t in points)
        correction=sum((h*(F(b)-F(a))**2/4 for h,a,b in zip(curvature,l,u)),F(0))
        return min(limit,best+correction)
    def save(status):
        upper=max((F(nodes[i]['upper']) for i in leaves),default=U0);elapsed=time.monotonic()-start
        report=dict(status=status,lower=float(lower),upper=float_up(upper),gap=float_up(upper-lower),target=float(target),target_met=upper-lower<=target,
            profile_seconds=elapsed,cap_seconds=cap,query_count=len(queries),nodes=list(nodes.values()),active_leaves=leaves,history=history,queries=query_records,errors=errors)
        (out/'result.json').write_text(json.dumps(report,indent=2));return report
    try:
        value=bound(rootlo,roothi,U0);nodes[0]=dict(id=0,parent=None,lo=rootlo,hi=roothi,upper=str(value),seconds=time.monotonic()-start);leaves=[0];save('RUNNING')
        while max(F(nodes[i]['upper']) for i in leaves)-lower>target and time.monotonic()-start<cap and len(nodes)+2<=maxnodes:
            parent=max(leaves,key=lambda i:F(nodes[i]['upper']));node=nodes[parent]
            axis=max(range(len(J)),key=lambda j:curvature[j]*(F(node['hi'][j])-F(node['lo'][j]))**2);cut=(node['lo'][axis]+node['hi'][axis])/2
            assert node['lo'][axis]<cut<node['hi'][axis];children=[]
            for side in (0,1):
                if time.monotonic()-start>=cap:break
                l=node['lo'].copy();u=node['hi'].copy()
                if side==0:u[axis]=cut
                else:l[axis]=cut
                value=bound(l,u,F(node['upper']));idx=len(nodes);nodes[idx]=dict(id=idx,parent=parent,lo=l,hi=u,upper=str(value),seconds=time.monotonic()-start);children.append(idx)
            if len(children)!=2:break
            node.update(children=children,split=dict(axis=axis,cut=cut));leaves.remove(parent);leaves.extend(children)
            upper=max(F(nodes[i]['upper']) for i in leaves);item=dict(nodes=len(nodes),queries=len(queries),gap=float_up(upper-lower),seconds=time.monotonic()-start)
            history.append(item);save('RUNNING');print(json.dumps(item),flush=True)
    except Exception as exc:errors.append(dict(error=str(exc),traceback=traceback.format_exc()))
    return save('VERIFIED_COMPLETE_COVER' if not errors else 'VALID_PARENT_AFTER_ERROR')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--output',required=True);ap.add_argument('--cpu',type=int,required=True);ap.add_argument('--cap',type=float,default=1800);a=ap.parse_args();psutil.Process().cpu_affinity([a.cpu]);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    result=run(Path(a.source),out,a.cap);print(json.dumps({k:v for k,v in result.items() if k not in ('nodes','queries','history')}),flush=True)
