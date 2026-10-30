"""Bounded numerical reference search; the final certificate is separate."""
import os,sys,time
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'

import numpy as np,cvxpy as cp
from scipy.optimize import minimize

def propose(s,witness,iterations=8,starts=32,seed=92501,domain=None):
    a=s.a;p=a.p;ids=s.ids;lookup={int(i):ii for ii,i in enumerate(a.remaining)}
    dlo,dhi=(a.lo,a.hi) if domain is None else domain
    midpoint=(dlo+dhi)/2
    ir=np.array([lookup[int(i)] for i,j in ids]);jc=ids[:,1];low=dlo[a.remaining];high=dhi[a.remaining];ds=(high-low)[ir,jc]
    qr=np.array(a.qr,float);H=a.t@a.t.T/a.mu;mid=s.fit(midpoint);rng=np.random.default_rng(seed);records=[];pool=[]
    def add(x):
        xx=x[a.remaining];M=a.G+xx.T@xx/a.n;r=qr+xx.T@a.yr/a.n;theta=np.linalg.solve(M,r)
        g=M@a.beta-r;h=float(g@np.linalg.solve(M,g));K=a.t.T@M@a.t
        e,v=np.linalg.eigh((K+K.T)/2);factor=np.sqrt(np.maximum(0.,e*e/a.mu-e))[:,None]*v.T
        pool.append((h,factor,np.linalg.solve(a.t,theta-mid)))
    initial=np.clip(np.array(witness['x']),dlo,dhi);add(midpoint);add(initial);reference=mid.copy();start=time.monotonic()
    last=(initial[ids[:,0],jc]-dlo[ids[:,0],jc])/ds
    for iteration in range(iterations):
        z=cp.Variable(p);eta=cp.Variable();constraints=[]
        for h,factor,center in pool:constraints.append(h*1e6+cp.sum_squares(1000*factor@(z-center))<=eta)
        problem=cp.Problem(cp.Minimize(eta),constraints)
        proposal_backend='CLARABEL'
        proposal_error=None
        try:
            problem.solve(solver='CLARABEL',max_iter=150,tol_gap_abs=1e-8,tol_gap_rel=1e-9,tol_feas=1e-9)
        except cp.error.SolverError as exc:
            proposal_backend='SCS';proposal_error=str(exc)
            problem.solve(solver='SCS',eps=1e-7,max_iters=20000,verbose=False)
        if z.value is None or not np.isfinite(z.value).all():
            # This is a reference proposal, so any finite reference remains valid.
            # The subsequent original polynomial certificate supplies the guarantee.
            return reference,dict(seconds=time.monotonic()-start,iterations=iteration,
                starts=starts,seed=seed,trace=records,reference_fallback='last_finite_reference',
                proposal_status=problem.status,proposal_error=proposal_error)
        reference=mid+a.t@z.value;base=a.G@reference-qr
        constant=a.beta@a.G@a.beta-reference@a.G@reference-2*qr@(a.beta-reference)
        def objective(t):
            x=low.copy();x[ir,jc]=low[ir,jc]+ds*t
            eb=x@a.beta-a.yr;et=x@reference-a.yr;g=base+x.T@et/a.n;Hg=H@g
            value=constant+(eb@eb-et@et)/a.n+g@Hg
            grad=2/a.n*(eb[ir]*a.beta[jc]-et[ir]*reference[jc]+et[ir]*Hg[jc]+reference[jc]*(x@Hg)[ir])*ds
            return -value,-grad
        best=-np.inf;best_t=last
        for k in range(starts):
            init=last if k==0 else rng.uniform(0,1,len(ids))
            result=minimize(objective,init,jac=True,bounds=[(0,1)]*len(ids),method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10,'maxiter':200})
            if -result.fun>best:best=-result.fun;best_t=result.x.copy()
        last=best_t;table=midpoint.copy();table[ids[:,0],jc]=dlo[ids[:,0],jc]+ds*last
        s.lower(table,s.fit(table));add(table)
        records.append(dict(iteration=iteration,finite_pool_value=float(problem.value)/1e6,sampled_surrogate_maximum=best,original_lower=s.best,
                            proposal_backend=proposal_backend,proposal_error=proposal_error))
    return reference,dict(seconds=time.monotonic()-start,iterations=iterations,starts=starts,seed=seed,trace=records,guarantee='numerical proposal only; validate with uniform polynomial and dual certificate')
