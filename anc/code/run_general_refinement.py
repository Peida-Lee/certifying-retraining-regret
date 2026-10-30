"""Complete original 60-condition grid with a fixed aggregate-cover fallback."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import sys,json,time,traceback,argparse
B=Path(__file__).resolve().parent
import numpy as np,psutil
from public_problems import problem
from observed_anchor import ObservedAnchor
from moment_tail import MomentSearch,basis_proposal,tail_receipt
from local_residual import LocalResidualSearch
from refinement_pipeline import multistart
from robust_reference import propose as initial_reference
from polynomial_majorant import certificate
from sdp_residual import solve
from aggregate_split import propose as split_propose,constraints as split_constraints
from moment_reference import propose as moment_reference

SEEDS=(74111,74123,74137,74141,74153)
GRID=[(d,w,s,k) for d in ('diabetes','concrete','superconductivity') for w in (1/32,1/8) for s in ('concentrated','dispersed') for k in SEEDS]

def run(cid,out,cap=1800,target=3e-6,first_rank=None,first_seconds=1,root_proposal_seconds=60,child_proposal_seconds=40,max_nodes=33,root_reference_steps=1,latest_root=False,preserve_first_split_reference=False,first_split_proposal_seconds=None):
 name,width,spread,seed=GRID[cid];t0=time.monotonic();elapsed=lambda:time.monotonic()-t0
 args,meta=problem(name=name,cells=128,width=width,spread=spread,seed=seed,subset_seed=74101)
 a=ObservedAnchor(*args,preconditioner='midpoint');proposal=basis_proposal(a);choices=[]
 for rank in ((first_rank,) if first_rank is not None else (1,2,3)):
  tail=tail_receipt(a,proposal[:,:rank])['tail_bound'];choices.append((rank,tail))
  if tail<=2.5e-6:break
 rank=next((r for r,e in choices if e<=2.5e-6),min(choices,key=lambda v:v[1])[0])
 first=MomentSearch(a,rank).run(time_limit=first_seconds,absolute_tolerance=target,record=True)
 (out/'first_stage.json').write_text(json.dumps(first))
 original=dict(lo=a.lo.tolist(),hi=a.hi.tolist(),y=a.y.tolist(),beta=a.beta.tolist(),lambda_value=a.lam)
 (out/'original_input.json').write_text(json.dumps(original))
 lower=first['lower'];upper=first['upper'];history=[dict(seconds=elapsed(),lower=lower,upper=upper,stage='structural_first_stage')]
 nodes={};leaves=[];errors=[];best_witness=first['certificate']['witness'];next_id=0
 def save(status='RUNNING'):
  result=dict(status=status,condition=cid,dataset=name,width=width,spread=spread,seed=seed,target=target,lower=lower,upper=upper,
      absolute_gap=upper-lower,target_met=upper-lower<=target,total_seconds=elapsed(),cap_seconds=cap,
      target_met_in_budget=any(h['seconds']<=cap and h['upper']-h['lower']<=target for h in history),
      first_stage_width=first['absolute_gap'],selected_rank=rank,rank_candidates=choices,history=history,
      root_reference_steps=root_reference_steps,latest_root=latest_root,preserve_first_split_reference=preserve_first_split_reference,first_split_proposal_seconds=first_split_proposal_seconds,
      nodes=[{k:v for k,v in n.items() if k not in ('problem','moments')} for n in nodes.values()],active_leaves=leaves,errors=errors)
  (out/'result.json').write_text(json.dumps(result,indent=2));return result
 save()
 if upper-lower<=target:return save('PASS')
 # Fresh anchor metric for the joint polynomial; its setup is charged.
 a=ObservedAnchor(*args);s=LocalResidualSearch(a);s.lower(np.array(best_witness['x']),np.array(best_witness['theta']))
 multistart(s);theta,reference=initial_reference(s,s.witness);w=s.witness;x0=np.array(w['x']);lower=max(lower,s.best)
 (out/'witness.json').write_text(json.dumps(w))
 def evaluate(theta,cuts,idx,parent,side=None):
  nonlocal lower
  src=certificate(s,a.lo,a.hi,x0,theta,build_only=True);src.update(domain_lo=a.lo.tolist(),domain_hi=a.hi.tolist())
  if cuts:src['extra_linear_inequalities']=cuts
  proposal_seconds=root_proposal_seconds if parent is None else child_proposal_seconds
  if parent is not None and nodes[parent]['parent'] is None and first_split_proposal_seconds is not None:proposal_seconds=first_split_proposal_seconds
  ans=solve(src,time_limit=min(proposal_seconds,max(.01,cap-elapsed()-3)),return_moments=True)
  Y=ans.pop('moment_matrix');lower=max(lower,s.best)
  ans.update(dataset=name,seed=seed,condition=cid,lower=lower,absolute_gap=ans['upper']-lower,target=target)
  file=f'node{idx}_receipt.json';(out/file).write_text(json.dumps(dict(result=ans,problem=src,witness=s.witness)))
  (out/'witness.json').write_text(json.dumps(s.witness))
  return dict(id=idx,parent=parent,side=side,upper=min(upper if parent is None else nodes[parent]['upper'],ans['upper']),
              problem=src,moments=Y,receipt=file,completed_seconds=elapsed())
 try:
  root=evaluate(theta,[],next_id,None);nodes[next_id]=root;leaves=[next_id];next_id+=1;upper=min(upper,root['upper'])
  history.append(dict(seconds=elapsed(),lower=lower,upper=upper,stage='joint_root'));save()
  # A fixed number of relaxed-moment warm-up steps precedes subdivision.
  for warm_step in range(root_reference_steps):
   if upper-lower<=target or elapsed()>=cap:break
   prop,diag=moment_reference(s,root['problem'],root['moments']);candidate=.5*np.array(root['problem']['theta'])+.5*prop
   improved=evaluate(candidate,[],next_id,None);nodes[next_id]=improved
   if latest_root or improved['upper']<root['upper']:leaves=[next_id];root=improved
   next_id+=1;upper=min(upper,root['upper']);history.append(dict(seconds=elapsed(),lower=lower,upper=upper,stage='root_reference_update'));save()
  while upper-lower>target and elapsed()<cap and next_id+2<=max_nodes:
   parent=max(leaves,key=lambda i:nodes[i]['upper']);p=nodes[parent];axis,cut,diag=split_propose(p['problem'],p['moments'])
   if preserve_first_split_reference and p['parent'] is None:candidate=np.array(p['problem']['theta'])
   else:
    prop,rd=moment_reference(s,p['problem'],p['moments']);candidate=.5*np.array(p['problem']['theta'])+.5*prop
   children=[]
   for side in (0,1):
    if elapsed()>=cap:break
    cuts=p['problem'].get('extra_linear_inequalities',[])+split_constraints(axis,cut,side)
    node=evaluate(candidate,cuts,next_id,parent,side);nodes[next_id]=node;children.append(next_id);next_id+=1
   if len(children)!=2:break
   p.update(children=children,split=dict(a=axis.tolist(),cut=cut));leaves.remove(parent);leaves.extend(children)
   upper=min(upper,max(nodes[i]['upper'] for i in leaves));history.append(dict(seconds=elapsed(),lower=lower,upper=upper,stage='aggregate_cover'));save()
 except Exception as exc:errors.append(dict(error=str(exc),traceback=traceback.format_exc()))
 return save('PASS' if not errors else 'COMPLETED_WITH_ERRORS')

def main():
 p=argparse.ArgumentParser();p.add_argument('--condition',type=int,required=True);p.add_argument('--output',default='general_refined');p.add_argument('--cpu',type=int);args=p.parse_args()
 if args.cpu is not None:psutil.Process().cpu_affinity([args.cpu])
 out=B/args.output/f'condition_{args.condition:02d}';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():raise FileExistsError(out/'result.json')
 try:r=run(args.condition,out)
 except Exception as exc:
  r=dict(status='ERROR',condition=args.condition,error=str(exc),traceback=traceback.format_exc());(out/'result.json').write_text(json.dumps(r,indent=2))
 print(json.dumps({k:v for k,v in r.items() if k not in ('nodes','history')}),flush=True)
if __name__=='__main__':main()
