"""Uniform lifted-residual refinement with valid rowwise box localizers."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,argparse,traceback
import numpy as np,psutil
from scipy.optimize import minimize
from planar_oracle import PlanarObservedAnchor
from local_residual import LocalResidualSearch
from robust_reference import propose as reference_proposal
from compressed_polynomial import build
from localized_moment import solve
from run_dense_compressed import refit_moments
from aggregate_split import propose as split_proposal,constraints as split_constraints
B=Path(__file__).resolve().parent

def witness_search(s,starts=4096,seed=92901):
 a=s.a;ids=s.ids;lookup={int(i):j for j,i in enumerate(a.remaining)};ir=np.array([lookup[int(i)] for i,j in ids]);jc=ids[:,1]
 low=a.lo[a.remaining];ds=(a.hi-a.lo)[ids[:,0],jc];qr=np.array(a.qr,float);rng=np.random.default_rng(seed);best=-np.inf;bestz=None
 def obj(z):
  x=low.copy();x[ir,jc]=low[ir,jc]+ds*z;M=a.G+x.T@x/a.n;r=qr+x.T@a.yr/a.n;g=M@a.beta-r;u=np.linalg.solve(M,g);theta=a.beta-u
  eb=x@a.beta-a.yr;et=x@theta-a.yr;gradient=2/a.n*(eb[ir]*a.beta[jc]-et[ir]*theta[jc])*ds
  return -float(g@u)*1e6,-gradient*1e6
 for k in range(starts):
  z=rng.uniform(0,1,len(ids)) if k%2 else rng.integers(0,2,len(ids)).astype(float)
  fit=minimize(obj,z,jac=True,bounds=[(0,1)]*len(ids),method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-8,'maxiter':300})
  if -fit.fun>best:best=-fit.fun;bestz=fit.x.copy()
 x=a.c.copy();x[ids[:,0],jc]=a.lo[ids[:,0],jc]+ds*np.clip(bestz,0,1);s.lower(x,s.fit(x))

def run(kind,cid,out,cap=1800,proposal_seconds=300,reference_rule='expected',basis_rule='full',solver='CLARABEL'):
 start=time.monotonic();elapsed=lambda:time.monotonic()-start
 if kind=='dense':
  from run_dense_refinement import problem
  names=['dense_diabetes','dense_synthetic_4','dense_synthetic_8','dense_synthetic_16'];args,meta=problem(names[cid]);target=1e-8
 else:
  from public_problems import problem
  from run_general_refinement import GRID
  name,width,spread,seed=GRID[cid];args,meta=problem(name=name,cells=128,width=width,spread=spread,seed=seed,subset_seed=74101);target=3e-6
 a=PlanarObservedAnchor(*args);s=LocalResidualSearch(a);original=dict(lo=a.lo.tolist(),hi=a.hi.tolist(),y=a.y.tolist(),beta=a.beta.tolist(),lambda_value=a.lam)
 (out/'original_input.json').write_text(json.dumps(original));witness_search(s);theta,reference=reference_proposal(s,s.witness)
 x0=np.array(s.witness['x']);x0=np.where(x0>=(a.lo+a.hi)/2,a.hi,a.lo);ids=s.ids.tolist();lo=s.best;up=a.gradient_upper
 nodes={};leaves=[];nextid=0;errors=[];history=[dict(seconds=elapsed(),lower=lo,upper=up,stage='witness_and_reference')]
 def save(status='RUNNING'):
  r=dict(status=status,kind=kind,condition=cid,meta=meta,target=target,lower=lo,upper=up,absolute_gap=up-lo,target_met=up-lo<=target,reference_rule=reference_rule,basis_rule=basis_rule,proposal_solver=solver,
    target_met_in_budget=any(h['seconds']<=cap and h['upper']-h['lower']<=target for h in history),cap_seconds=cap,total_seconds=elapsed(),
    initial_upper=a.gradient_upper,nodes=[{k:v for k,v in d.items() if k not in ('source','Y')} for d in nodes.values()],active_leaves=leaves,history=history,errors=errors)
  (out/'result.json').write_text(json.dumps(r,indent=2));return r
 save()
 def evaluate(theta,cuts,idx,parent,side=None):
  nonlocal lo
  base=dict(theta=np.asarray(theta).tolist(),T=a.t.tolist(),mu=a.mu,active_cells=ids,x0_uncertain=[float(x0[i,j]) for i,j in ids]);src=build(original,base)
  src['domain_inequalities']=cuts;src['extra_linear_inequalities']=src['residual_link_inequalities']+cuts
  src['localizing_solver']=solver
  if basis_rule=='omit':
   src['moment_terms']=[k for k in src['moment_terms'] if not(len(k)==1 and k[0]>=len(ids))]
   src['moment_basis_rule']='omit_linear_residuals'
  ans=solve(src,seconds=min(proposal_seconds,max(1.,cap-elapsed()-2)),return_moments=True);Y=ans.pop('moment_matrix')
  old={tuple(k):(l,u) for k,l,u in zip(src['monomials'],src['coefficient_lower'],src['objective'])};src['monomials']=ans['monomials']
  src['coefficient_lower']=[old.get(tuple(k),(0.,0.))[0] for k in src['monomials']];src['objective']=[old.get(tuple(k),(0.,0.))[1] for k in src['monomials']]
  trial=x0.copy()
  for z,(i,j) in enumerate(ids):
   delta=(a.hi[i,j]-a.lo[i,j])*(1 if x0[i,j]==a.lo[i,j] else -1);trial[i,j]=np.clip(x0[i,j]+delta*ans['linear_moments'][z],a.lo[i,j],a.hi[i,j])
  s.lower(trial,s.fit(trial));lo=max(lo,s.best);ans.update(lower=lo,absolute_gap=ans['upper']-lo,condition=cid)
  file=f'node{idx}_receipt.json';(out/file).write_text(json.dumps(dict(result=ans,problem=src,witness=s.witness)))
  return dict(id=idx,parent=parent,side=side,upper=min(up if parent is None else nodes[parent]['upper'],ans['upper']),source=src,Y=Y,receipt=file,completed_seconds=elapsed())
 try:
  root=evaluate(theta,[],nextid,None);nodes[nextid]=root;leaves=[nextid];nextid+=1;up=min(up,root['upper']);history.append(dict(seconds=elapsed(),lower=lo,upper=up,stage='localized_root'));save()
  while up-lo>target and elapsed()<cap and nextid+2<=17:
   parent=max(leaves,key=lambda j:nodes[j]['upper']);p=nodes[parent];axis,cut,sd=split_proposal(p['source'],p['Y'])
   candidate=np.array(p['source']['theta']) if reference_rule=='parent' else refit_moments(a,p['source'],p['Y']);children=[]
   for side in (0,1):
    if elapsed()>=cap:break
    cuts=p['source']['domain_inequalities']+split_constraints(axis,cut,side);child=evaluate(candidate,cuts,nextid,parent,side);nodes[nextid]=child;children.append(nextid);nextid+=1
   if len(children)!=2:break
   p.update(children=children,split=dict(a=axis.tolist(),cut=cut));leaves.remove(parent);leaves.extend(children)
   up=min(up,max(nodes[j]['upper'] for j in leaves));history.append(dict(seconds=elapsed(),lower=lo,upper=up,stage='localized_cover'));save()
 except Exception as e:errors.append(dict(error=str(e),traceback=traceback.format_exc()))
 return save('PASS' if not errors else 'COMPLETED_WITH_ERRORS')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dense','general'],required=True);p.add_argument('--condition',type=int,required=True);p.add_argument('--cpu',type=int,required=True);p.add_argument('--output',default='localized_refinement');p.add_argument('--reference-rule',choices=['expected','parent'],default='expected');p.add_argument('--basis-rule',choices=['full','omit'],default='full');p.add_argument('--solver',choices=['CLARABEL','SCS'],default='CLARABEL');a=p.parse_args();psutil.Process().cpu_affinity([a.cpu]);out=B/a.output/f'{a.kind}_{a.condition:02d}';out.mkdir(parents=True,exist_ok=True);assert not (out/'result.json').exists()
 try:r=run(a.kind,a.condition,out,reference_rule=a.reference_rule,basis_rule=a.basis_rule,solver=a.solver)
 except Exception as e:r=dict(status='ERROR',condition=a.condition,error=str(e),traceback=traceback.format_exc());(out/'result.json').write_text(json.dumps(r,indent=2))
 print(json.dumps({k:v for k,v in r.items() if k not in ('nodes','history','meta')}),flush=True)
