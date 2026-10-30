"""Lifted row residuals with verified affine links for dense row certificates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time,itertools,argparse
from fractions import Fraction as F
B=Path(__file__).resolve().parent
import numpy as np,psutil
from observed_anchor import float_down,float_up
from planar_oracle import PlanarObservedAnchor as ObservedAnchor
from local_residual import LocalResidualSearch
from interval_arithmetic import I
from fast_interval import mm,matvec
from sdp_residual import solve

def build(original,source):
    lo=np.array(original['lo']);hi=np.array(original['hi']);y=np.array(original['y']);beta=np.array(original['beta']);theta=np.array(source['theta'])
    a=ObservedAnchor(lo,hi,y,beta,original['lambda_value']);a.t=np.array(source['T']);a.mu=source['mu'];s=LocalResidualSearch(a)
    ids=source['active_cells'];n=len(ids);rows=list(a.remaining);rowidx={i:z for z,i in enumerate(rows)};nv=n+len(rows)
    x0=(lo+hi)/2
    for (i,j),v in zip(ids,source['x0_uncertain']):x0[i,j]=v
    ds=np.array([(hi[i,j]-lo[i,j])*(1 if x0[i,j]==lo[i,j] else -1) for i,j in ids])
    rlo=[];span=[];relations=[]
    for i in rows:
        low=sum((min(F(lo[i,j])*F(theta[j]),F(hi[i,j])*F(theta[j])) for j in range(a.p)),F(0))-F(y[i])
        high=sum((max(F(lo[i,j])*F(theta[j]),F(hi[i,j])*F(theta[j])) for j in range(a.p)),F(0))-F(y[i])
        # Use a power-of-two span, enlarged once if the dyadic left endpoint needs it.
        step=2.**int(np.floor(np.log2(max(float(high-low),2**-30))))
        left=np.floor(float(low)/step)*step
        width=step
        while F(left)+F(width)<high:width*=2
        rlo.append(left);span.append(width)
        coef={n+rowidx[i]:1.}
        for z,(rr,j) in enumerate(ids):
            if rr==i:
                q=-F(ds[z])*F(theta[j])/F(width);v=float(q);assert F(v)==q;coef[z]=v
        rhs=(sum((F(x0[i,j])*F(theta[j]) for j in range(a.p)),F(0))-F(y[i])-F(left))/F(width)
        relations.append((coef,float_up(rhs)))
        relations.append(({k:-v for k,v in coef.items()},float_up(-rhs)))
    rlo=np.array(rlo);span=np.array(span);xx=x0[a.remaining]
    e=I(rlo);eb=matvec(I(xx),I(a.beta))-I(a.yr);et=matvec(I(xx),I(theta))-I(a.yr)
    cell_loss=source.get('loss_difference_rule')=='original_cells'
    g0=matvec(I(a.t.T),matvec(s.g,I(theta))-s.r+matvec(I(xx.T),e)/a.n)
    coeff={():g0};poly={}
    def add(key,v):poly[key]=poly.get(key,I(0.))+v
    add((),s.beta_anchor-(I(theta)*matvec(s.g,I(theta))).sum()+2*(s.r*I(theta)).sum()+(eb.square()-(et if cell_loss else e).square()).sum()/a.n)
    for z,(i,j) in enumerate(ids):
        row=rowidx[i];u=n+row;d=I(ds[z]);t=I(a.t[j])
        coeff[(z,)]=d*rlo[row]*t/a.n
        coeff[(z,u)]=d*span[row]*t/a.n
        add((z,),2*(eb[row]*beta[j]-(et[row]*theta[j] if cell_loss else 0))*d/a.n)
        for w in range(z,n):
            rr,k=ids[w]
            if rr==i:add((z,w),(I(beta[j])*beta[k]-(I(theta[j])*theta[k] if cell_loss else 0))*d*ds[w]*(1 if z==w else 2)/a.n)
    for row,i in enumerate(rows):
        u=n+row;coeff[(u,)]=span[row]*matvec(I(a.t.T),I(x0[i]))/a.n
        if not cell_loss:
            add((u,),-2*I(rlo[row])*span[row]/a.n);add((u,u),-I(span[row]).square()/a.n)
    terms=list(coeff);C=I(np.array([coeff[k].lo for k in terms]),np.array([coeff[k].hi for k in terms]));gram=mm(C,I(C.lo.T,C.hi.T))/a.mu
    for j,first in enumerate(terms):
        for k in range(j,len(terms)):add(tuple(sorted(first+terms[k])),gram[j,k]*(1 if j==k else 2))
    keys=set(k for k in poly if k);keys.update((j,) for j in range(nv))
    for key in list(keys):
        for size in range(1,len(key)):keys.update(itertools.combinations(key,size))
    keys=sorted(keys,key=lambda x:(len(x),x));cuts=[]
    def cut(terms,rhs):
        comb={}
        for k,v in terms:comb[tuple(sorted(k))]=comb.get(tuple(sorted(k)),F(0))+F(v)
        # Every lifted monomial is nonnegative. Rounding coefficients down
        # and the right side up preserves each RLT inequality.
        cuts.append(dict(terms=[[list(k),float_down(v)] for k,v in sorted(comb.items()) if v],rhs=float_up(F(rhs)),kind='residual_link'))
    for coef,rhs in relations:
        cut([((j,),v) for j,v in coef.items()],rhs)
        for j in range(nv):
            prod=[((k,j),v) for k,v in coef.items()]
            cut(prod+[((j,),-rhs)],0.)
            cut([((k,),v) for k,v in coef.items()]+[(k,-v) for k,v in prod]+[((j,),rhs)],rhs)
    # Complete all monomials required by the valid residual-link RLT rows.
    extra={tuple(k) for c in cuts for k,v in c['terms']};keys=sorted(set(keys)|extra,key=lambda x:(len(x),x))
    out=dict(objective=[float(poly.get(k,I(0.)).hi) for k in keys],coefficient_lower=[float(poly.get(k,I(0.)).lo) for k in keys],
             monomials=list(map(list,keys)),moment_terms=list(map(list,terms)),constant_upper=float(poly[()].hi),constant_lower=float(poly[()].lo),
             extra_linear_inequalities=cuts,residual_link_inequalities=cuts,domain_inequalities=[],active_cells=ids,residual_left=rlo.tolist(),residual_span=span.tolist(),theta=theta.tolist(),T=a.t.tolist(),mu=a.mu,x0_uncertain=source['x0_uncertain'],proposal_solver='CLARABEL')
    if cell_loss:out['loss_difference_rule']='original_cells'
    return out

