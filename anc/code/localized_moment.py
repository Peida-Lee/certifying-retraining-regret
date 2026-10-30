"""Development: strengthen dense row moment relaxations with box localizers."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,argparse,itertools,warnings
B=Path(__file__).resolve().parent
import numpy as np,cvxpy as cp,psutil
from fractions import Fraction as F
from observed_anchor import float_down,float_up
from scipy.sparse import coo_matrix,vstack,kron,csr_matrix
from sdp_residual import constraints_from_keys
from interval_arithmetic import I
from fast_interval import mm

def solve(src,seconds=120,return_moments=False):
 t=time.monotonic();oldkeys=list(map(tuple,src['monomials']));keys=set(oldkeys);ids=src['active_cells'];nr=len(ids)
 groups={}
 for j,(row,col) in enumerate(ids):groups.setdefault(row,[]).append(j)
 blocks=[dict(basis=src['moment_terms'],polynomial=[[[],1.]],kind='gradient_moment')]
 for ri,(row,group) in enumerate(groups.items()):
  cells=group;group=group+[nr+ri]
  basis=[[]]+[[j] for j in (cells if src.get('moment_basis_rule') in ('omit_linear_residuals','facial_row_residuals') else group)]
  for j in group:blocks.append(dict(basis=basis,polynomial=[[[j],1.],[[j,j],-1.]],kind='box_localizer'))
 if 'custom_moment_blocks' in src:blocks=src['custom_moment_blocks']
 blocks=blocks+src.get('additional_moment_blocks',[])
 for block in blocks:
  for a in block['basis']:
   for b in block['basis']:
    for d,v in block['polynomial']:
     key=tuple(sorted(a+b+d))
     if key:keys.add(key)
 for cut in src.get('extra_linear_inequalities',[]):keys.update(tuple(k) for k,v in cut['terms'] if k)
 for key in list(keys):
  for size in range(1,len(key)):keys.update(itertools.combinations(key,size))
 keys=sorted(keys,key=lambda x:(len(x),x));lookup={k:i for i,k in enumerate(keys)};N=len(keys)
 obj=dict(zip(oldkeys,src['objective']));c=np.array([obj.get(k,0.) for k in keys]);A,rhs,rows=constraints_from_keys(keys)
 extra=src.get('extra_linear_inequalities',[]);rr=[];cc=[];vv=[];bb=[]
 for k,cut in enumerate(extra):
  terms=[(lookup[tuple(key)],float(v)) for key,v in cut['terms']];rows.append(tuple(terms));bb.append(cut['rhs'])
  for j,v in terms:rr.append(k);cc.append(j);vv.append(v)
 if extra:A=vstack([A,coo_matrix((vv,(rr,cc)),shape=(len(extra),N)).tocsr()],format='csr');rhs=rhs+bb
 y=cp.Variable(N);pairs=src.get('proposal_equality_pairs',[]);base_rows=len(rhs)-len(extra)
 if pairs:
  pairs=[(base_rows+i,base_rows+j) for i,j in pairs];removed={j for pair in pairs for j in pair};removed.update(base_rows+j for j in src.get('proposal_omitted_rows',[]));keep=[j for j in range(len(rhs)) if j not in removed]
  E=vstack([(A.getrow(i)-A.getrow(j))*.5 for i,j in pairs],format='csr');be=np.array([(rhs[i]-rhs[j])*.5 for i,j in pairs])
  linear=A[keep]@y<=np.array(rhs)[keep];equality=E@y==be;cons=[linear,equality,y>=0,y<=1]
 elif 'proposal_linear_subset' in src:
  keep=sorted(set(map(int,src['proposal_linear_subset'])))
  assert keep and min(keep)>=0 and max(keep)<len(rhs)
  linear=A[keep]@y<=np.asarray(rhs)[keep];cons=[linear,y>=0,y<=1]
 else:linear=A@y<=rhs;cons=[linear,y>=0,y<=1]
 maps=[];original_maps=[]
 for block in blocks:
  basis=block['basis'];m=len(basis);rr=[];cc=[];vv=[];const=np.zeros(m*m)
  for i,a in enumerate(basis):
   for j,b in enumerate(basis):
    for d,v in block['polynomial']:
     key=tuple(sorted(a+b+d));row=i*m+j
     value=float(F(v))
     if not key:const[row]+=value
     else:rr.append(row);cc.append(lookup[key]);vv.append(value)
  mp=coo_matrix((vv,(rr,cc)),shape=(m*m,N)).tocsr();original_maps.append((mp,const,m))
  if 'basis_transform' in block:
   transform=np.asarray(block['basis_transform']);K=kron(csr_matrix(transform),csr_matrix(transform),format='csr');mp=K@mp;const=K@const;m=len(transform)
  Y=cp.reshape(mp@y+const,(m,m),order='C');cone=Y>>0;cons.append(cone);maps.append((mp,const,cone,m))
 scale=1e6
 if src.get('normalize_objective'):
  scale=2.**int(np.ceil(np.log2(1/max(np.max(np.abs(c)),2.**-32))))
 model=cp.Problem(cp.Maximize(c*scale@y),cons)
 initial_diagnostic=None
 if 'initial_point' in src:
  point=np.array(src['initial_point']);seed=np.array([np.prod(point[list(k)]) for k in keys])
  initial_diagnostic=dict(max_linear_violation=float(np.max(A@seed-np.asarray(rhs))),minimum_moment=float(min(seed)),maximum_moment=float(max(seed)),objective=float(c@seed+src['constant_upper']))
  print(__import__('json').dumps({'initial_feasibility':initial_diagnostic}),flush=True)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',UserWarning)
  if src.get('localizing_solver')=='SCS_WARM':
   import scs
   data,chain,inverse=model.get_problem_data(cp.SCS)
   dims=data['dims'];cone=dict(z=dims.zero,l=dims.nonneg,q=dims.soc,s=dims.psd,ep=dims.exp,p=dims.p3d)
   canonical={k:data[k] for k in ('A','b','c')};assert len(canonical['c'])==len(seed)
   options=dict(eps_abs=1e-9,eps_rel=1e-9,max_iters=300000,time_limit_secs=seconds,verbose=False)
   options.update(src.get('scs_options',{}))
   solver=scs.SCS(canonical,cone,**options)
   dual_start=np.zeros(len(canonical['b']))
   if 'initial_certificate_dual' in src:
    prior=src['initial_certificate_dual']
    assert not pairs and float(prior['scale'])==scale
    assert dims.zero==0 and dims.nonneg==len(rhs)+2*N
    dual_start[:len(rhs)]=np.maximum(prior['dual'],0.)
    offset=dims.nonneg
    for block,(_,_,_,m),factor in zip(blocks,maps,prior['psd_factors']):
     assert 'basis_transform' not in block
     factor=np.asarray(factor);S=factor@factor.T;assert S.shape==(m,m)
     for j in range(m):
      for i in range(j,m):
       dual_start[offset]=S[i,j]*(1. if i==j else np.sqrt(2.));offset+=1
    assert offset==len(dual_start)
    stationarity=canonical['c']+canonical['A'].T@dual_start
    dual_start[len(rhs):len(rhs)+N]=np.maximum(stationarity,0.)
    dual_start[len(rhs)+N:len(rhs)+2*N]=np.maximum(-stationarity,0.)
    defect=canonical['c']+canonical['A'].T@dual_start
    initial_diagnostic['dual_stationarity_max']=float(np.max(np.abs(defect)))
    assert np.max(np.abs(defect))<1e-7
    initial_diagnostic['dual_upper']=float(canonical['b']@dual_start/scale+src['constant_upper'])
   sol=solver.solve(warm_start=True,x=seed,y=dual_start,s=canonical['b']-canonical['A']@seed)
   model.unpack_results(sol,chain,inverse)
  elif src.get('localizing_solver')=='SCS':
   model.solve(solver='SCS',eps=1e-8,max_iters=150000,time_limit_secs=seconds,verbose=False)
  else:
   model.solve(solver='CLARABEL',max_iter=120,time_limit=seconds,tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,verbose=False)
 if linear.dual_value is None:raise RuntimeError(model.status)
 dual_bits=36 if src.get('normalize_objective') else 20
 factor_bits=40 if src.get('normalize_objective') else 24
 proposal=np.maximum(linear.dual_value,0)
 if pairs or 'proposal_linear_subset' in src:
  converted=np.zeros(len(rhs));converted[keep]=proposal
  for value,(i,j) in zip(equality.dual_value if pairs else [],pairs):
   converted[i]=max(value,0);converted[j]=max(-value,0)
  proposal=converted
 lam=np.floor(proposal*2**dual_bits)/2**dual_bits;res=I(c)*scale;rl=res.lo.copy();rh=res.hi.copy();constant=I(0);factors=[]
 for w,row in zip(lam,rows):
  if not w:continue
  for j,v in row:
   val=I(rl[j],rh[j])-I(w)*v;rl[j]=val.lo;rh[j]=val.hi
 for block,(mp,co,cone,m) in zip(blocks,maps):
  dual=(cone.dual_value+cone.dual_value.T)/2;ev,V=np.linalg.eigh(dual);factor=np.round(V*np.sqrt(np.maximum(ev,0))[None,:]*2**factor_bits)/2**factor_bits
  if 'basis_transform' in block:
   factor=np.round(np.asarray(block['basis_transform']).T@factor*2**factor_bits)/2**factor_bits
   factor=np.pad(factor,((0,0),(0,len(block['basis'])-factor.shape[1])))
  factors.append(factor.tolist());psd=mm(I(factor),I(factor.T))
  for i,a in enumerate(block['basis']):
   for j,b in enumerate(block['basis']):
    for d,v in block['polynomial']:
     key=tuple(sorted(a+b+d));q=F(v);val=psd[i,j]*I(float_down(q),float_up(q))
     if not key:constant=constant+val
     else:
      k=lookup[key];value=I(rl[k],rh[k])+val;rl[k]=value.lo;rh[k]=value.hi
 correction=I(np.maximum(rh,0)).sum();upper=(I(lam)*I(rhs)).sum()+constant+correction;upper=upper/scale+src['constant_upper']
 result=dict(scale=scale,dual_bits=dual_bits,factor_bits=factor_bits,upper=float(upper.hi),numeric_objective=model.value/scale+src['constant_upper'],correction=float(correction.hi)/scale,seconds=time.monotonic()-t,solver_status=model.status,solver_iterations=model.solver_stats.num_iters,solver_solve_seconds=model.solver_stats.solve_time,variables=N,blocks=len(blocks),initial_diagnostic=initial_diagnostic,dual=lam.tolist(),psd_factors=factors,moment_blocks=blocks,monomials=list(map(list,keys)))


 if return_moments:
  mp,co,m=original_maps[0];result['moment_matrix']=(mp@y.value+co).reshape(m,m).tolist()
 result['linear_moments']=[float(y.value[lookup[(j,)]]) for j in range(src.get('variable_count',nr))]
 return result
