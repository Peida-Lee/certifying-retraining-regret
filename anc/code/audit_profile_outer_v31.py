"""Exact replay of retained-comparator queries and full outer box coverage."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import json,sys,time,argparse,itertools,copy,traceback
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import numpy as np,psutil
from audit_compressed_arrow_v27 import verify
from profile_query_v31 import audit_input


def audit(folder):
    start=time.monotonic();data=json.loads((folder/'input.json').read_text());protocol=json.loads((folder/'protocol.json').read_text());result=json.loads((folder/'result.json').read_text())
    source=Path(protocol['source']);initial=json.loads((source/'input.json').read_text());initial_proposal=json.loads((source/'proposal.json').read_text())
    assert initial['original']==data['original'];initial_audit=verify(initial,initial_proposal)
    U0=F(protocol['initial_exact_upper']);assert U0>=F(initial_audit['exact_upper']);assert initial_audit['status']=='EXACT_AUDIT_PASS'
    audit_input(data);o=data['original'];lo=np.array(o['lo']);hi=np.array(o['hi']);n,p=lo.shape;J=data['problem']['profile_columns'];lam=F(o['lambda_value'])
    lower=F(data['result']['lower']);assert lower==F(protocol['lower']) and F(result['lower'])<=lower
    radius=F(protocol['radius']);assert radius*radius*lam>=U0
    for j,l,u in zip(J,protocol['rootlo'],protocol['roothi']):assert F(l)<=F(o['beta'][j])-radius and F(u)>=F(o['beta'][j])+radius
    curvature=[lam+sum((max(F(lo[i,j])**2,F(hi[i,j])**2) for i in range(n)),F(0))/n for j in J]
    assert curvature==list(map(F,protocol['curvature']))
    query={}
    for rec in result['queries']:
        saved=json.loads((folder/f"query_{rec['id']:04d}.json").read_text());t=saved['t'];assert t==rec['t'];theta=saved['proposal']['reference_theta']
        assert [theta[j] for j in J]==t
        current=copy.deepcopy(data);current['problem']['theta']=theta
        checked=verify(current,saved['proposal'],check_input=False);value=F(checked['exact_upper']);assert F(rec['exact_upper'])>=value
        query[tuple(t)]=F(rec['exact_upper'])
    assert len(query)==result['query_count'];nodes={r['id']:r for r in result['nodes']};assert nodes
    assert nodes[0]['lo']==protocol['rootlo'] and nodes[0]['hi']==protocol['roothi']
    for idx,node in sorted(nodes.items()):
        corners=list(itertools.product(*zip(node['lo'],node['hi'])));corner=max(query[t] for t in corners)
        correction=sum((h*(F(u)-F(l))**2/4 for h,l,u in zip(curvature,node['lo'],node['hi'])),F(0))
        cap=U0 if idx==0 else F(nodes[node['parent']]['upper']);assert F(node['upper'])>=min(cap,corner+correction)
    seen=set();leaves=[];stack=[0]
    while stack:
        idx=stack.pop();assert idx not in seen;seen.add(idx);node=nodes[idx]
        if 'children' not in node:leaves.append(idx);continue
        children=node['children'];assert len(children)==2;axis=node['split']['axis'];cut=node['split']['cut'];assert node['lo'][axis]<cut<node['hi'][axis]
        for side,cid in enumerate(children):
            child=nodes[cid];assert child['parent']==idx;l=node['lo'].copy();u=node['hi'].copy()
            if side==0:u[axis]=cut
            else:l[axis]=cut
            assert child['lo']==l and child['hi']==u
        stack.extend(children)
    assert set(leaves)==set(result['active_leaves']);upper=max(F(nodes[i]['upper']) for i in leaves)
    assert F(result['upper'])>=upper>=lower and F(result['gap'])>=upper-lower
    assert result['target_met']==(upper-lower<=F(result['target']))
    return dict(status='PASS',complete_outer_cover=True,original_witness=True,original_full_regret_root=True,retained_parameters=len(J),
                exact_queries=len(query),cover_nodes=len(seen),unused_checked_nodes=len(nodes)-len(seen),gap=float(upper-lower),target_met=result['target_met'],seconds=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--cpu',type=int,default=12);a=p.parse_args();psutil.Process().cpu_affinity([a.cpu]);folder=Path(a.folder)
    try:r=audit(folder)
    except Exception as exc:r=dict(status='ERROR',error=str(exc),traceback=traceback.format_exc())
    (folder/'independent_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
