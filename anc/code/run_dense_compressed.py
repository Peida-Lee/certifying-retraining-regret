"""Fixed four-case dense-row refinement with lifted residual coordinates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time,argparse,traceback
import numpy as np,psutil
from run_dense_refinement import problem
from planar_oracle import PlanarObservedAnchor,PlanarMomentSearch
from local_residual import LocalResidualSearch
from refinement_pipeline import multistart
from robust_reference import propose as reference_proposal
from compressed_polynomial import build
from sdp_residual import solve
from aggregate_split import propose as split_proposal,constraints as split_constraints
B=Path(__file__).resolve().parent
NAMES=['dense_diabetes','dense_synthetic_4','dense_synthetic_8','dense_synthetic_16']

def refit_moments(a,source,Y):
    ids=source['active_cells'];keys=list(map(tuple,source['moment_terms']));lookup={k:i for i,k in enumerate(keys)}
    x0=(a.lo+a.hi)/2
    for (i,j),v in zip(ids,source['x0_uncertain']):x0[i,j]=v
    xx=x0[a.remaining];G=a.G+xx.T@xx/a.n;r=np.array(a.qr,float)+xx.T@a.yr/a.n
    d=np.array([(a.hi[i,j]-a.lo[i,j])*(1 if x0[i,j]==a.lo[i,j] else -1) for i,j in ids]);Y=np.asarray(Y)
    for z,(i,j) in enumerate(ids):
        v=d[z]*Y[0,lookup[(z,)]]/a.n;G[j,:]+=v*x0[i];G[:,j]+=v*x0[i];r[j]+=v*a.y[i]
        for w,(rr,k) in enumerate(ids):
            if rr==i:G[j,k]+=d[z]*d[w]*Y[lookup[(z,)],lookup[(w,)]]/a.n
    candidate=np.linalg.solve((G+G.T)/2,r)
    z=np.linalg.solve(a.t,candidate-a.beta);norm=np.linalg.norm(z);radius=np.sqrt(a.gradient_upper/a.mu)
    if not np.isfinite(norm):return np.array(source['theta'])
    if norm>radius:candidate=a.beta+a.t@(z*(radius/norm))
    return candidate

def run(cid,out,cap=3600,target=1e-8):
    t0=time.monotonic();elapsed=lambda:time.monotonic()-t0
    args,meta=problem(NAMES[cid]);a=PlanarObservedAnchor(*args,preconditioner='midpoint')
    first=PlanarMomentSearch(a,11 if cid==0 else 2).run(time_limit=2,absolute_tolerance=target,record=True)
    original=dict(lo=a.lo.tolist(),hi=a.hi.tolist(),y=a.y.tolist(),beta=a.beta.tolist(),lambda_value=a.lam)
    (out/'original_input.json').write_text(json.dumps(original));(out/'first_stage.json').write_text(json.dumps(first))
    lo=first['lower'];up=first['upper'];history=[dict(seconds=elapsed(),lower=lo,upper=up,stage='structural')];nodes={};leaves=[];nextid=0;errors=[]
    def save(status='RUNNING'):
        r=dict(status=status,condition=cid,dataset=NAMES[cid],target=target,lower=lo,upper=up,absolute_gap=up-lo,target_met=up-lo<=target,
               target_met_in_budget=any(h['seconds']<=cap and h['upper']-h['lower']<=target for h in history),cap_seconds=cap,total_seconds=elapsed(),
               first_stage_width=first['absolute_gap'],nodes=[{k:v for k,v in d.items() if k not in ('source','Y')} for d in nodes.values()],active_leaves=leaves,history=history,errors=errors)
        (out/'result.json').write_text(json.dumps(r,indent=2));return r
    save()
    if up-lo<=target:return save('PASS')
    a=PlanarObservedAnchor(*args);s=LocalResidualSearch(a);w=first['certificate']['witness'];s.lower(np.array(w['x']),np.array(w['theta']))
    multistart(s);theta,diag=reference_proposal(s,s.witness);lo=max(lo,s.best);x0=np.array(s.witness['x']);ids=[list(map(int,v)) for v in s.ids]
    x0=np.where(x0>=(a.lo+a.hi)/2,a.hi,a.lo)
    def evaluate(theta,cuts,idx,parent,side=None):
        nonlocal lo
        base=dict(theta=np.asarray(theta).tolist(),T=a.t.tolist(),mu=a.mu,active_cells=ids,x0_uncertain=[float(x0[i,j]) for i,j in ids])
        source=build(original,base);source['domain_inequalities']=cuts;source['extra_linear_inequalities']=source['residual_link_inequalities']+cuts
        ans=solve(source,time_limit=min(90,max(.01,cap-elapsed()-5)),return_moments=True);Y=ans.pop('moment_matrix')
        # Relaxed first moments define an original feasible table after clipping.
        trial=x0.copy()
        for z,(i,j) in enumerate(ids):
            d=(a.hi[i,j]-a.lo[i,j])*(1 if x0[i,j]==a.lo[i,j] else -1)
            trial[i,j]=np.clip(x0[i,j]+d*ans['linear_moments'][z],a.lo[i,j],a.hi[i,j])
        s.lower(trial,s.fit(trial));lo=max(lo,s.best)
        ans.update(dataset=NAMES[cid],condition=cid,seed=77116 if cid else 77101,lower=lo,absolute_gap=ans['upper']-lo)
        file=f'node{idx}_receipt.json';(out/file).write_text(json.dumps(dict(result=ans,problem=source,witness=s.witness)))
        return dict(id=idx,parent=parent,side=side,upper=min(up if parent is None else nodes[parent]['upper'],ans['upper']),source=source,Y=Y,receipt=file,completed_seconds=elapsed())
    try:
        root=evaluate(theta,[],nextid,None);nodes[nextid]=root;leaves=[nextid];nextid+=1;up=min(up,root['upper'])
        history.append(dict(seconds=elapsed(),lower=lo,upper=up,stage='lifted_root'));save()
        while up-lo>target and elapsed()<cap and nextid+2<=49:
            parent=max(leaves,key=lambda j:nodes[j]['upper']);p=nodes[parent]
            axis,cut,sd=split_proposal(p['source'],p['Y']);prop=refit_moments(a,p['source'],p['Y']);candidate=.5*np.array(p['source']['theta'])+.5*prop;children=[]
            for side in (0,1):
                if elapsed()>=cap:break
                cuts=p['source']['domain_inequalities']+split_constraints(axis,cut,side)
                child=evaluate(candidate,cuts,nextid,parent,side);nodes[nextid]=child;children.append(nextid);nextid+=1
            if len(children)!=2:break
            p.update(children=children,split=dict(a=axis.tolist(),cut=cut));leaves.remove(parent);leaves.extend(children)
            up=min(up,max(nodes[j]['upper'] for j in leaves));history.append(dict(seconds=elapsed(),lower=lo,upper=up,stage='lifted_cover'));save()
    except Exception as exc:errors.append(dict(error=str(exc),traceback=traceback.format_exc()))
    return save('PASS' if not errors else 'COMPLETED_WITH_ERRORS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',type=int,required=True);p.add_argument('--cpu',type=int,default=12);p.add_argument('--output',default='dense_compressed');a=p.parse_args();psutil.Process().cpu_affinity([a.cpu])
    out=B/a.output/f'condition_{a.condition:02d}';out.mkdir(parents=True,exist_ok=True);assert not (out/'result.json').exists()
    try:r=run(a.condition,out)
    except Exception as e:r=dict(status='ERROR',error=str(e),traceback=traceback.format_exc());(out/'result.json').write_text(json.dumps(r))
    print(json.dumps({k:v for k,v in r.items() if k not in ('nodes','history')}),flush=True)
