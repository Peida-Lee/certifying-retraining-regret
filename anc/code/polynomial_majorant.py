"""Lifted exact polynomial residual majorant on a completion box."""
import itertools
import warnings
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import coo_matrix
from interval_arithmetic import I
from fast_interval import mm,matvec

def splits(key):
    out=set();n=len(key)
    for mask in range(1,2**n-1):
        a=tuple(key[i] for i in range(n) if mask>>i&1)
        b=tuple(key[i] for i in range(n) if not mask>>i&1)
        out.add(tuple(sorted((a,b))))
    return sorted(out)

def lift_bound(poly,n,moment_terms=None,psd_rounds=8):
    keys=set(k for k in poly if k)
    for j in range(n):keys.add((j,))
    for key in list(keys):
        for size in range(1,len(key)):
            keys.update(itertools.combinations(key,size))
    keys=sorted(keys,key=lambda x:(len(x),x));lookup={key:j for j,key in enumerate(keys)}
    objective=np.array([float(poly[k].hi) if k in poly else 0. for k in keys])
    rr=[];cc=[];vv=[];rhs=[];constraints=[];done=set()
    def add(terms,b):
        combined={}
        for j,v in terms:combined[j]=combined.get(j,0.)+v
        terms=tuple(sorted((j,v) for j,v in combined.items() if v))
        key=(terms,b)
        if key in done:return
        done.add(key);i=len(rhs);rhs.append(b);constraints.append(terms)
        for j,v in terms:rr.append(i);cc.append(j);vv.append(v)
    for key in keys:
        if len(key)==1:continue
        z=lookup[key]
        for first,second in splits(key):
            if first not in lookup or second not in lookup:continue
            a=lookup[first];b=lookup[second]
            add([(z,1.),(a,-1.)],0.)
            if a==b:
                for t in (0.,.25,.5,.75,1.):add([(a,2*t),(z,-1.)],t*t)
            else:
                add([(z,1.),(b,-1.)],0.)
                add([(a,1.),(b,1.),(z,-1.)],1.)
    cuts=[]
    for iteration in range(psd_rounds+1):
        A=coo_matrix((vv,(rr,cc)),shape=(len(rhs),len(keys))).tocsr()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',OptimizeWarning)
            ans=linprog(-objective,A_ub=A,b_ub=rhs,bounds=(0.,1.),method='highs',options={'threads':1})
        if not ans.success:raise RuntimeError(ans.message)
        if iteration==psd_rounds or not moment_terms:break
        Y=np.empty((len(moment_terms),len(moment_terms)))
        for i,first in enumerate(moment_terms):
            for j,second in enumerate(moment_terms):
                key=tuple(sorted(first+second));Y[i,j]=1. if not key else ans.x[lookup[key]]
        eigen,vectors=np.linalg.eigh(Y);bad=np.flatnonzero(eigen < -1e-5)
        if not len(bad):break
        for which in bad[:8]:
            v=np.round(vectors[:,which]*2**12)/2**12;terms=[];constant=0.
            for i,first in enumerate(moment_terms):
                for j in range(i,len(moment_terms)):
                    key=tuple(sorted(first+moment_terms[j]));co=-v[i]*v[j]*(1 if i==j else 2)
                    if not key:constant-=co
                    else:terms.append((lookup[key],co))
            add(terms,constant);cuts.append(v.tolist())
    dual=np.maximum(0.,-ans.ineqlin.marginals)
    l=objective.copy();h=objective.copy()
    for lam,terms in zip(dual,constraints):
        if lam==0:continue
        for col,v in terms:
            interval=I(l[col],h[col])-I(lam)*v;l[col]=interval.lo;h[col]=interval.hi
    correction=I(np.maximum(0.,h)).sum()
    value=(I(dual)*I(rhs)).sum()+correction+poly.get((),I(0.))
    return dict(upper=float(value.hi),solution=np.array([ans.x[lookup[(j,)]] for j in range(n)]),
                variables=len(keys),constraints=len(rhs),dual=dual.tolist(),monomials=[list(k) for k in keys],
                objective=objective.tolist(),constant_upper=float(poly.get((),I(0.)).hi),psd_cuts=cuts,moment_terms=[list(k) for k in (moment_terms or [])],
                residual_correction=float(correction.hi),solver_numeric_upper=-float(ans.fun)+float(poly.get((),I(0.)).hi))

def certificate(solver,lo,hi,x0,reference=None,build_only=False):
    a=solver.a;c=(lo+hi)/2
    x0=np.where(x0>=c,hi,lo);theta=solver.fit(x0) if reference is None else reference;solver.lower(x0,theta)
    active=[(int(i),int(j)) for i,j in solver.ids if lo[i,j]!=hi[i,j]]
    ir=np.array([i for i,j in active]);jc=np.array([j for i,j in active]);n=len(active)
    direction=I(np.where(x0[ir,jc]==lo[ir,jc],1.,-1.))
    delta=(I(hi[ir,jc])-I(lo[ir,jc]))*direction
    xx=x0[a.remaining];et=matvec(I(xx),I(theta))-I(a.yr)
    g0=matvec(I(a.t.T),matvec(solver.g,I(theta))-solver.r+matvec(I(xx.T),et)/a.n)
    coeff={():g0}
    xb=I(x0[ir]);err=matvec(xb,I(theta))-I(a.y[ir]);errb=matvec(xb,I(a.beta))-I(a.y[ir])
    linear=(mm(xb,I(a.t))*I(theta[jc,None])+err[:,None]*I(a.t[jc]))*delta[:,None]/a.n
    poly={}
    def add(key,value):poly[key]=poly.get(key,I(0.))+value
    q0=solver.beta_anchor-dot1(I(theta),matvec(solver.g,I(theta)))+2*dot1(solver.r,I(theta))
    q0=q0+((matvec(I(xx),I(a.beta))-I(a.yr)).square()-et.square()).sum()/a.n
    add((),q0)
    for j in range(n):
        coeff[(j,)]=linear[j]
        add((j,),2*(errb[j]*a.beta[jc[j]]-err[j]*theta[jc[j]])*delta[j]/a.n)
        for k in range(j,n):
            if ir[j]!=ir[k]:continue
            if j==k:
                coeff[(j,k)]=I(a.t[jc[j]])*theta[jc[j]]*delta[j].square()/a.n
                q=(I(a.beta[jc[j]]).square()-I(theta[jc[j]]).square())*delta[j].square()/a.n
            else:
                coeff[(j,k)]=(theta[jc[k]]*I(a.t[jc[j]])+theta[jc[j]]*I(a.t[jc[k]]))*delta[j]*delta[k]/a.n
                q=2*(I(a.beta[jc[j]])*a.beta[jc[k]]-I(theta[jc[j]])*theta[jc[k]])*delta[j]*delta[k]/a.n
            add((j,k),q)
    terms=list(coeff);C=I(np.array([coeff[k].lo for k in terms]),np.array([coeff[k].hi for k in terms]))
    gram=mm(C,I(C.lo.T,C.hi.T))/a.mu
    for j,first in enumerate(terms):
        for k in range(j,len(terms)):
            key=tuple(sorted(first+terms[k]));add(key,gram[j,k]*(1 if j==k else 2))
    if build_only:
        keys=set(k for k in poly if k)
        keys.update((j,) for j in range(n))
        for key in list(keys):
            for size in range(1,len(key)):keys.update(itertools.combinations(key,size))
        keys=sorted(keys,key=lambda k:(len(k),k))
        return dict(objective=[float(poly[k].hi) if k in poly else 0. for k in keys],
            coefficient_lower=[float(poly[k].lo) if k in poly else 0. for k in keys],
            monomials=[list(k) for k in keys],moment_terms=[list(k) for k in terms],
            constant_upper=float(poly[()].hi),constant_lower=float(poly[()].lo),
            theta=theta.tolist(),x0_uncertain=x0[ir,jc].tolist(),active_cells=[list(k) for k in active],
            T=a.t.tolist(),mu=a.mu,metric_rule=getattr(a,'residual_metric_rule',{'kind':'observed_rows'}))
    result=lift_bound(poly,n,moment_terms=terms)
    trial=x0.copy()
    for z,(i,j) in enumerate(active):trial[i,j]=np.clip(x0[i,j]+float(delta.lo[z]+delta.hi[z])/2*result['solution'][z],lo[i,j],hi[i,j])
    solver.lower(trial,solver.fit(trial))
    result.update(theta=theta.tolist(),x0_uncertain=x0[ir,jc].tolist())
    return result

def dot1(a,b):return (a*b).sum()
