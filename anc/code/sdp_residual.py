"""Numerical moment proposal with a separately evaluated valid dual upper."""
import os,sys,time,warnings
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
B=Path(__file__).resolve().parent


import numpy as np,cvxpy as cp
from scipy.sparse import coo_matrix
from interval_arithmetic import I
from fast_interval import mm
from polynomial_majorant import splits

def constraints_from_keys(keys):
    lookup={k:i for i,k in enumerate(keys)};N=len(keys)
    rows=[];cols=[];values=[];rhs=[];constraints=[];done=set()
    def add(terms,b):
        combined={}
        for j,v in terms:combined[j]=combined.get(j,0.)+v
        terms=tuple(sorted((j,v) for j,v in combined.items() if v));key=(terms,b)
        if key in done:return
        done.add(key);i=len(rhs);rhs.append(b);constraints.append(terms)
        for j,v in terms:rows.append(i);cols.append(j);values.append(v)
    for key in keys:
        if len(key)==1:continue
        z=lookup[key]
        for first,second in splits(key):
            a=lookup[first];b=lookup[second]
            add([(z,1.),(a,-1.)],0.)
            if a==b:
                for t in (0.,.25,.5,.75,1.):add([(a,2*t),(z,-1.)],t*t)
            else:
                add([(z,1.),(b,-1.)],0.);add([(a,1.),(b,1.),(z,-1.)],1.)
    A=coo_matrix((values,(rows,cols)),shape=(len(rhs),N)).tocsr()
    return A,rhs,constraints

def solve(source,max_iters=15000,eps=1e-6,time_limit=None,return_moments=False):
    start=time.monotonic();keys=[tuple(k) for k in source['monomials']];lookup={k:i for i,k in enumerate(keys)};N=len(keys)
    A,rhs,constraints=constraints_from_keys(keys)
    extra=source.get('extra_linear_inequalities',[])
    if extra:
        from scipy.sparse import vstack
        rows=[];cols=[];values=[];new_rhs=[]
        for i,cut in enumerate(extra):
            row=[(lookup[tuple(key)],float(value)) for key,value in cut['terms']]
            constraints.append(tuple(row));new_rhs.append(float(cut['rhs']))
            for j,v in row:rows.append(i);cols.append(j);values.append(v)
        A=vstack([A,coo_matrix((values,(rows,cols)),shape=(len(extra),N)).tocsr()],format='csr');rhs=rhs+new_rhs
    terms=[tuple(k) for k in source['moment_terms']];m=len(terms)
    rr=[];cc=[];vv=[];constant=np.zeros(m*m)
    for i,first in enumerate(terms):
        for j,second in enumerate(terms):
            key=tuple(sorted(first+second));row=i*m+j
            if not key:constant[row]=1.
            else:rr.append(row);cc.append(lookup[key]);vv.append(1.)
    mapping=coo_matrix((vv,(rr,cc)),shape=(m*m,N)).tocsr()
    y=cp.Variable(N);Y=cp.reshape(mapping@y+constant,(m,m),order='C')
    linear=A@y<=rhs;psd=Y>>0
    scale=1e6;c=np.array(source['objective'])*scale
    problem=cp.Problem(cp.Maximize(c@y),[linear,psd,y>=0,y<=1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        options={} if time_limit is None else {'time_limit_secs':max(.01,time_limit)}
        if source.get('proposal_solver')=='CLARABEL':
            problem.solve(solver='CLARABEL',max_iter=120,time_limit=90. if time_limit is None else max(.01,time_limit),
                          tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,verbose=False)
        else:
            problem.solve(solver='SCS',eps=eps,max_iters=max_iters,verbose=False,**options)
    solver_seconds=time.monotonic()-start
    if linear.dual_value is None or psd.dual_value is None:raise RuntimeError(problem.status)
    dual=np.floor(np.maximum(0.,linear.dual_value)*2**20)/2**20
    S=(psd.dual_value+psd.dual_value.T)/2
    evals,vecs=np.linalg.eigh(S);factor=np.round(vecs*np.sqrt(np.maximum(evals,0.))[None,:]*2**24)/2**24
    positive=mm(I(factor),I(factor.T))
    true_c=I(source['objective'])*scale;reslo=true_c.lo.copy();reshi=true_c.hi.copy()
    for lam,terms0 in zip(dual,constraints):
        if not lam:continue
        for j,v in terms0:
            z=I(reslo[j],reshi[j])-I(lam)*v;reslo[j]=z.lo;reshi[j]=z.hi
    constant_psd=I(0.)
    for i,first in enumerate(terms):
        for j,second in enumerate(terms):
            key=tuple(sorted(first+second));value=positive[i,j]
            if not key:constant_psd=constant_psd+value
            else:
                k=lookup[key];z=I(reslo[k],reshi[k])+value;reslo[k]=z.lo;reshi[k]=z.hi
    correction=I(np.maximum(0.,reshi)).sum()
    upper=((I(dual)*I(rhs)).sum()+constant_psd+correction)/scale+source['constant_upper']
    dimensions=sum(len(key)==1 for key in keys)
    first=np.array([float(y.value[lookup[(j,)]]) for j in range(dimensions)])
    variance=np.array([max(0.,float(y.value[lookup[(j,j)]])-first[j]**2) for j in range(dimensions)])
    sensitivity=np.zeros(dimensions)
    for key,coef in zip(keys,source['objective']):
        for j in set(key):sensitivity[j]+=abs(coef)
    branch=int(np.argmax(variance*sensitivity))
    result=dict(status='PASS',solver_status=problem.status,upper=float(upper.hi),numeric_objective=problem.value/scale+source['constant_upper'],
        seconds=time.monotonic()-start,solver_seconds=solver_seconds,correction=float(correction.hi)/scale,
        scale=scale,dual=dual.tolist(),psd_factor=factor.tolist(),max_iters=max_iters,eps=eps,
        linear_moments=first.tolist(),moment_variance=variance.tolist(),branch_index=branch)
    if return_moments:result['moment_matrix']=Y.value.tolist()
    return result
