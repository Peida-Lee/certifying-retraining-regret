"""Independent replay of every small-cone node and complementary row slab."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import json,sys,argparse,time,traceback
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import psutil
from audit_compressed_arrow_v27 import verify,reconstruct


def audit(folder):
    start=time.monotonic();data=json.loads((folder/'input.json').read_text());result=json.loads((folder/'result.json').read_text())
    assert not data['problem'].get('extra_linear_inequalities')
    nodes={n['id']:n for n in result['nodes']};checked={};bounds={};rootpolys=None
    for idx,node in sorted(nodes.items()):
        receipt=json.loads((folder/f'node_{idx:03d}.json').read_text());proposal=receipt['proposal']
        assert proposal['residual_domains']==node['domains']
        if idx==0:
            assert node['parent'] is None and all(d is None for d in node['domains'])
            _,rootpolys,_=reconstruct(data,proposal)
        audit=verify(data,proposal,check_input=idx==0);value=F(audit['exact_upper'])
        if idx:value=min(value,bounds[node['parent']])
        assert F(node['upper'])>=value
        bounds[idx]=F(node['upper']);checked[idx]=audit['status']
    seen=set();stack=[0];leaves=[]
    while stack:
        idx=stack.pop();assert idx not in seen;seen.add(idx);node=nodes[idx]
        if 'children' not in node:leaves.append(idx);continue
        assert len(node['children'])==2
        split=node['split'];i=split['row'];cut=F(split['cut']);interval=list(map(F,split['interval']));assert interval[0]<cut<interval[1]
        if node['domains'][i] is None:
            residual=rootpolys[i][2];low=residual[0]+sum(min(v,F(0)) for v in residual[1:]);high=residual[0]+sum(max(v,F(0)) for v in residual[1:])
            assert interval[0]<=low<=high<=interval[1]
        else:assert interval==list(map(F,node['domains'][i]))
        for side,child in enumerate(node['children']):
            cn=nodes[child];assert cn['parent']==idx
            expected=[None if d is None else list(map(F,d)) for d in node['domains']]
            expected[i]=[interval[0],cut] if side==0 else [cut,interval[1]]
            actual=[None if d is None else list(map(F,d)) for d in cn['domains']]
            assert actual==expected
        stack.extend(node['children'])
    assert seen==set(nodes) and set(leaves)==set(result['active_leaves'])
    upper=max(bounds[j] for j in leaves);lower=F(data['result']['lower']);assert F(result['upper'])>=upper>=lower
    assert result['target_met']==(upper-lower<=F(result['target']))
    return dict(status='PASS',condition=result['condition'],nodes=len(nodes),active_leaves=len(leaves),complete_cover=True,original_witness=True,
                exact_gap=str(upper-lower),gap=float(upper-lower),target_met=result['target_met'],seconds=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--cpu',type=int,default=12);a=p.parse_args();psutil.Process().cpu_affinity([a.cpu]);folder=Path(a.folder)
    try:r=audit(folder)
    except Exception as exc:r=dict(status='ERROR',error=str(exc),traceback=traceback.format_exc())
    (folder/'independent_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
