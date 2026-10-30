"""Verified LP upper bounds for quadratic polynomials on [0,1]^d."""
import numpy as np
import warnings
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import coo_matrix
from interval_arithmetic import I

def bound(linear,quadratic,psd_rounds=3):
    n=len(linear);pairs=[(j,k) for j in range(n) for k in range(j,n)]
    objective=list(linear)
    objective.extend(float((I(quadratic[j,k])+I(quadratic[k,j])).hi) if j!=k else quadratic[j,j] for j,k in pairs)
    c=np.array(objective);ii=[];jj=[];vv=[];rhs=[];constraints=[]
    def add(terms,b):
        row=len(rhs);constraints.append(terms);rhs.append(b)
        for col,v in terms:ii.append(row);jj.append(col);vv.append(v)
    for z,(j,k) in enumerate(pairs,start=n):
        if j==k:
            add([(z,1.),(j,-1.)],0.)
            for t in (0.,.25,.5,.75,1.):add([(j,2*t),(z,-1.)],t*t)
        else:
            add([(z,1.),(j,-1.)],0.)
            add([(z,1.),(k,-1.)],0.)
            add([(j,1.),(k,1.),(z,-1.)],1.)
    cuts=[]
    for iteration in range(psd_rounds+1):
        A=coo_matrix((vv,(ii,jj)),shape=(len(rhs),len(c))).tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',OptimizeWarning)
            ans=linprog(-c,A_ub=A,b_ub=rhs,bounds=(0.,1.),method='highs',options={'threads':1})
        if not ans.success:raise RuntimeError(ans.message)
        if iteration==psd_rounds:break
        moment=np.ones((n+1,n+1));moment[0,1:]=ans.x[:n];moment[1:,0]=ans.x[:n]
        for z,(j,k) in enumerate(pairs,start=n):moment[j+1,k+1]=moment[k+1,j+1]=ans.x[z]
        eigen,vectors=np.linalg.eigh(moment)
        bad=np.flatnonzero(eigen < -1e-6)
        if not len(bad):break
        for which in bad[:8]:
            v=np.round(vectors[:,which]*2**12)/2**12
            terms=[(j,-2*v[0]*v[j+1]) for j in range(n)]
            terms.extend((z,-v[j+1]*v[k+1]*(1 if j==k else 2)) for z,(j,k) in enumerate(pairs,start=n))
            add(terms,v[0]*v[0]);cuts.append(v.tolist())
    dual=np.maximum(0.,-ans.ineqlin.marginals)
    # Any nonnegative dual yields an upper bound after this residual correction.
    residual=I(c)
    l=residual.lo.copy();h=residual.hi.copy()
    for lam,terms in zip(dual,constraints):
        if not lam:continue
        for col,v in terms:
            interval=I(l[col],h[col])-I(lam)*v
            l[col]=interval.lo;h[col]=interval.hi
    dual_value=(I(dual)*I(rhs)).sum()+I(np.maximum(0.,h)).sum()
    return dict(upper=float(dual_value.hi),solution=ans.x[:n],dual=dual.tolist(),objective=c.tolist(),psd_cuts=cuts,residual_hi=h.tolist(),
                linear=list(map(float,linear)),quadratic=np.asarray(quadratic).tolist(),solver_numeric_upper=-float(ans.fun),
                residual_correction=float(I(np.maximum(0.,h)).sum().hi),verified=True)
