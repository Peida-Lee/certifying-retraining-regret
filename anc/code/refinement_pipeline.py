"""Adaptive residual refinement with completed-evidence timing and covering boxes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,time,heapq,traceback
import numpy as np
from scipy.optimize import minimize
from local_residual import LocalResidualSearch
from robust_reference import propose
from polynomial_majorant import certificate
from sdp_residual import solve


def multistart(s,starts=64,seed=92101):
    a=s.a;ids=s.ids;lookup={int(i):j for j,i in enumerate(a.remaining)}
    ir=np.array([lookup[int(i)] for i,j in ids]);jc=ids[:,1]
    low=a.lo[a.remaining];high=a.hi[a.remaining];ds=(high-low)[ir,jc];qr=np.array(a.qr,float)
    def objective(z):
        x=low.copy();x[ir,jc]=low[ir,jc]+ds*z
        M=a.G+x.T@x/a.n;r=qr+x.T@a.yr/a.n;g=M@a.beta-r;u=np.linalg.solve(M,g);theta=a.beta-u
        eb=x@a.beta-a.yr;et=x@theta-a.yr
        grad=2/a.n*(eb[ir]*a.beta[jc]-et[ir]*theta[jc])*ds
        return -float(g@u),-grad
    rng=np.random.default_rng(seed);best=-np.inf;bestz=None;start=time.monotonic()
    for k in range(starts):
        z=np.full(len(ids),.5) if k==0 else rng.uniform(0,1,len(ids))
        fit=minimize(objective,z,jac=True,bounds=[(0,1)]*len(ids),method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10,'maxiter':200})
        if -fit.fun>best:best=-fit.fun;bestz=fit.x.copy()
    x=a.c.copy();x[ids[:,0],jc]=a.lo[ids[:,0],jc]+ds*np.clip(bestz,0,1)
    s.lower(x,s.fit(x))
    return dict(seconds=time.monotonic()-start,starts=starts,seed=seed)


def refine(a,energy,pre_seconds,outdir,instance,meta,target=1e-5,budget=600,max_nodes=31):
    """pre_seconds includes the complete first-stage method and shared input costs.

    Both children must finish before their parent's bound is replaced.  The
    parent remains the covering certificate at any interrupted split.
    """
    start=time.monotonic();outdir.mkdir(parents=True,exist_ok=True)
    s=LocalResidualSearch(a);w=energy['certificate']['witness']
    s.lower(np.array(w['x']),np.array(w['theta']))
    lower=s.best;energy_upper=float(energy['upper']);upper=energy_upper
    trace=[dict(seconds=pre_seconds,lower=lower,upper=upper,stage='energy',nodes=0)]
    tree=[];errors=[];active={};nodes=0
    elapsed=lambda:pre_seconds+time.monotonic()-start
    def record(stage):
        nonlocal lower,upper
        lower=max(lower,s.best)
        upper=min(energy_upper,max(v['upper'] for v in active.values())) if active else energy_upper
        assert lower<=upper+1e-12
        trace.append(dict(seconds=elapsed(),lower=lower,upper=upper,stage=stage,nodes=nodes))
    def evaluate(lo,hi,node_id,parent):
        nonlocal nodes
        local=LocalResidualSearch(a);x=np.clip(np.array(s.witness['x']),lo,hi);local.lower(x,local.fit(x))
        try:theta,reference=propose(local,local.witness,domain=(lo,hi))
        except Exception as exc:
            theta=local.fit((lo+hi)/2);reference=dict(status='FALLBACK',error=str(exc))
        x=np.array(local.witness['x']);problem=certificate(local,lo,hi,x,theta,build_only=True)
        problem.update(domain_lo=lo.tolist(),domain_hi=hi.tolist())
        result=solve(problem,time_limit=max(.01,budget-elapsed()-1))
        # Moment coordinates also propose a feasible completed table.
        trial=(lo+hi)/2
        for z,((i,j),x0) in enumerate(zip(problem['active_cells'],problem['x0_uncertain'])):
            direction=1 if x0==lo[i,j] else -1
            trial[i,j]=np.clip(x0+direction*(hi[i,j]-lo[i,j])*result['linear_moments'][z],lo[i,j],hi[i,j])
        local.lower(trial,local.fit(trial));s.lower(np.array(local.witness['x']),np.array(local.witness['theta']))
        result.update(dataset=meta['dataset'],seed=meta['seed'],instance=instance,node_id=node_id,parent=parent,
                      lower=local.best,absolute_gap=result['upper']-local.best,completed_seconds=elapsed())
        receipt=dict(result=result,problem=problem,witness=local.witness,reference_proposal=reference)
        filename=f'{instance}_node{node_id}_receipt.json'
        (outdir/filename).write_text(json.dumps(receipt))
        nodes+=1
        return dict(id=node_id,parent=parent,lo=lo,hi=hi,upper=result['upper'],branch=result['branch_index'],
                    cells=problem['active_cells'],receipt=filename,completed_seconds=result['completed_seconds'])
    if upper-lower>target and elapsed()<budget:
        witness_cost=multistart(s);record('feasible_multistart')
        try:
            root=evaluate(a.lo.copy(),a.hi.copy(),0,None);active[0]=root
            tree.append({k:v for k,v in root.items() if k not in ('lo','hi','cells')});record('residual_root')
            counter=0
            while upper-lower>target and elapsed()<budget and nodes+2<=max_nodes:
                idx=max(active,key=lambda k:active[k]['upper']);parent=active[idx]
                i,j=parent['cells'][parent['branch']];cut=(parent['lo'][i,j]+parent['hi'][i,j])/2
                children=[]
                for side in (0,1):
                    if elapsed()>=budget:break
                    lo=parent['lo'].copy();hi=parent['hi'].copy()
                    if side==0:hi[i,j]=cut
                    else:lo[i,j]=cut
                    counter+=1;child=evaluate(lo,hi,counter,idx)
                    child['upper']=min(child['upper'],parent['upper']);children.append(child)
                    tree.append({k:v for k,v in child.items() if k not in ('lo','hi','cells')})
                if len(children)==2:
                    del active[idx]
                    for child in children:active[child['id']]=child
                    next(t for t in tree if t['id']==idx).update(split=[int(i),int(j),float(cut)],children=[c['id'] for c in children])
                    record('residual_split')
                else:record('incomplete_split_parent_retained');break
        except Exception as exc:
            errors.append(dict(error=str(exc),traceback=traceback.format_exc()));record('fallback_after_error')
    else:witness_cost=None
    total=elapsed();eligible=[t for t in trace if t['seconds']<=budget]
    credited=eligible[-1] if eligible else None
    answer=dict(status='PASS',dataset=meta['dataset'],seed=meta['seed'],instance=instance,
        lower=lower,upper=upper,absolute_gap=upper-lower,total_seconds=total,complete_budget_seconds=budget,
        target=target,target_met=upper-lower<=target,target_met_in_budget=bool(credited and credited['upper']-credited['lower']<=target),
        first_target_seconds=next((r['seconds'] for r in trace if r['upper']-r['lower']<=target),None),
        nodes=nodes,trace=trace,tree=tree,active_leaves=list(active),errors=errors,witness_cost=witness_cost,
        checkpoints={str(t):next((r for r in reversed(trace) if r['seconds']<=t),None) for t in (10,60,120,300,600)})
    (outdir/f'{instance}_witness.json').write_text(json.dumps(s.witness))
    (outdir/f'{instance}_result.json').write_text(json.dumps(answer,indent=2))
    return answer
