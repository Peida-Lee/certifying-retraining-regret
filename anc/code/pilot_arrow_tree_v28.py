"""Uniform residual-slab splitting diagnostic on saved original conditions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import sys,json,time,argparse,traceback,copy
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import numpy as np,psutil
from compressed_edge_arrow_v27 import solve,prepare_compressed
from audit_compressed_arrow_v27 import verify
from observed_anchor import float_up


def run(source,out,maxnodes=33,cap=900,reference_updates=0):
    start=time.monotonic()
    if (source/'input.json').exists():
        data=json.loads((source/'input.json').read_text());result=data['result']
    else:
        receipt=json.loads((source/'source.json').read_text());folder=Path(receipt['folder'])
        original=json.loads((folder/'original_input.json').read_text());saved=json.loads((folder/receipt['receipt']).read_text());result=json.loads((folder/'result.json').read_text())
        data=dict(original=original,problem=saved['problem'],witness=json.loads((folder/'witness.json').read_text()),result=result)
    assert not data['problem'].get('extra_linear_inequalities')
    data['partition_theta']=data.get('partition_theta',data['problem']['theta'])
    lower=F(result['lower']);target=F(result['target']);root=json.loads((source/'proposal.json').read_text());root['halfspace_multipliers']=[]
    rows,_,_,_,_,_,oracles,_,_=prepare_compressed(data);m=len(rows)
    checked=verify(data,root);nodes={0:dict(id=0,parent=None,domains=[None]*m,upper=checked['exact_upper'],proposal=root)};leaves=[0];history=[]
    (out/'input.json').write_text(json.dumps(data));(out/'node_000.json').write_text(json.dumps(dict(proposal=root,audit=checked)))
    def save():
        upper=max(F(nodes[j]['upper']) for j in leaves)
        report=dict(status='EXACT_DIAGNOSTIC',condition=result['condition'],lower=float(lower),upper=float_up(upper),gap=float_up(upper-lower),target=float(target),target_met=upper-lower<=target,
                    elapsed_seconds=time.monotonic()-start,root_proposal_seconds=root['seconds'],source=str(source),timing='Saved root/input; diagnostic only',
                    nodes=[{k:v for k,v in r.items() if k!='proposal'} for r in nodes.values()],active_leaves=leaves,history=history)
        (out/'result.json').write_text(json.dumps(report,indent=2));return report
    while max(F(nodes[i]['upper']) for i in leaves)-lower>target and len(nodes)+2<=maxnodes and time.monotonic()-start<cap:
        parent=max(leaves,key=lambda j:F(nodes[j]['upper']));record=nodes[parent];options=[]
        for c in record['proposal']['branch_candidates']:
            i=c['row'];ii,left,right=oracles[i].intervals(record['domains'][i]);et=oracles[i].et[ii];slope=oracles[i].slope[ii]
            low=float(np.min(np.minimum(et+left*slope,et+right*slope)));high=float(np.max(np.maximum(et+left*slope,et+right*slope)))
            if high-low>1e-10:options.append((c['variance'],i,low,high,c['mean']))
        if not options:break
        _,row,low,high,mean=max(options);cut=float(np.round(np.clip(mean,low+.15*(high-low),high-.15*(high-low))*2**32)/2**32)
        bounds=record['domains'][row]
        if bounds is None:
            pt=np.array(data['partition_theta']);values=rows[row]['x']@pt-rows[row]['y']
            bounds=[float(np.min(values))-1e-12,float(np.max(values))+1e-12]
        children=[]
        for side in (0,1):
            idx=len(nodes);domains=[None if d is None else list(d) for d in record['domains']];domains[row]=[bounds[0],cut] if side==0 else [cut,bounds[1]]
            node_data=copy.deepcopy(data);node_data['problem']['theta']=record['proposal'].get('reference_theta',data['problem']['theta'])
            ans=solve(node_data,seconds=60,domains=domains,initial_points=record['proposal']['points'],lower=float(lower),max_iterations=10)
            for update in range(reference_updates):
                if ans['upper']-float(lower)<=float(target):break
                candidate=.5*np.array(ans['reference_theta'])+.5*np.array(ans['reference_fit'])
                node_data['problem']['theta']=candidate.tolist()
                trial=solve(node_data,seconds=60,domains=domains,initial_points=ans['points'],lower=float(lower),max_iterations=10)
                if trial['upper']<ans['upper']:ans=trial
                else:break
            audit=verify(data,ans,check_input=False);upper=min(F(audit['exact_upper']),F(record['upper']))
            (out/f'node_{idx:03d}.json').write_text(json.dumps(dict(proposal=ans,audit=audit)))
            nodes[idx]=dict(id=idx,parent=parent,domains=domains,upper=str(upper),proposal=ans,seconds=time.monotonic()-start)
            children.append(idx);print(json.dumps(dict(node=idx,parent=parent,gap=float_up(upper-lower),seconds=time.monotonic()-start)),flush=True)
        record.update(children=children,split=dict(row=row,cut=cut,interval=bounds));leaves.remove(parent);leaves.extend(children)
        history.append(dict(seconds=time.monotonic()-start,gap=float_up(max(F(nodes[j]['upper']) for j in leaves)-lower)));save()
    return save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--cpu',type=int,required=True);p.add_argument('--nodes',type=int,default=33);p.add_argument('--reference-updates',type=int,default=0);a=p.parse_args()
    psutil.Process().cpu_affinity([a.cpu]);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    try:r=run(Path(a.source),out,a.nodes,reference_updates=a.reference_updates);print(json.dumps({k:v for k,v in r.items() if k not in ('nodes','history')}),flush=True)
    except Exception as exc:
        (out/'error.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc()),indent=2));raise
