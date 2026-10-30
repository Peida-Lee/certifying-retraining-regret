"""Exact certificate-guided row-edge screening and residual-slab bounds."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from fractions import Fraction as F
from math import comb,lcm
import json,time,argparse
import numpy as np
from row_edge_exact import reconstruct
from edge_polynomial import zeta
B=Path(__file__).resolve().parent

def edge_powers(poly,k,free):
    den=lcm(*(v.denominator for v in poly.values()))
    keep=[j for j in range(k) if j!=free];loc={j:i for i,j in enumerate(keep)}
    power=np.zeros((5,1<<(k-1)),dtype=object)
    for key,value in poly.items():
        mask=sum(1<<loc[j] for j in set(key) if j!=free)
        power[key.count(free),mask]+=int(value*den)
    for row in power:zeta(row,k-1)
    return power,den

def bernstein(power):
    control=np.zeros_like(power)
    for q in range(5):
        for a in range(q+1):control[q]+=power[a]*(comb(q,a)*12//comb(4,a))
    return control

def halve(control):
    a,b,c,d,e=control;center=a+4*b+6*c+4*d+e
    return (np.array([16*a,8*(a+b),4*(a+2*b+c),2*(a+3*b+3*c+d),center],object),
            np.array([center,2*(b+3*c+3*d+e),4*(c+2*d+e),8*(d+e),16*e],object))

def screen(poly,k,cut,depth=4):
    """Keep an outer cover of every edge point with polynomial > cut."""
    result=[];scale=1<<depth
    for free in range(k):
        power,den=edge_powers(poly,k,free);control=bernstein(power)
        size=power.shape[1];left=np.full(size,scale,dtype=np.int64);right=np.zeros(size,dtype=np.int64)
        stack=[(control,np.arange(size),den*12,0,0)]
        while stack:
            c,ids,dn,level,pos=stack.pop();ub=np.max(c,axis=0)
            active=np.fromiter((int(v)*cut.denominator>cut.numerator*dn for v in ub),bool,len(ub))
            if not np.any(active):continue
            c=c[:,active];ids=ids[active]
            if level==depth:
                left[ids]=np.minimum(left[ids],pos);right[ids]=np.maximum(right[ids],pos+1);continue
            low,high=halve(c);stack.extend([(low,ids,dn*16,level+1,2*pos),(high,ids,dn*16,level+1,2*pos+1)])
        ids=np.flatnonzero(left<right)
        vertices=(ids&((1<<free)-1))|((ids>>free)<<(free+1))
        result.extend([free,int(v),int(left[i]),int(right[i])] for i,v in zip(ids,vertices))
    return dict(dimension=k,interval_scale=scale,edges=result,threshold=str(cut),arithmetic='exact integer Bernstein exclusion')

def restricted_upper(poly,k,allowed,residual,domain=None,tolerance=F(1,10000)):
    """Exact polynomial upper on screened edges intersected with a residual slab."""
    best=None;upper=None;count=0;scale=allowed['interval_scale'];edges=np.asarray(allowed['edges'],dtype=np.int64)
    if not len(edges):return None,dict(edges=0)
    for free in range(k):
        selected=edges[edges[:,0]==free]
        if not len(selected):continue
        power,den=edge_powers(poly,k,free);vertex=selected[:,1]
        ids=(vertex&((1<<free)-1))|((vertex>>(free+1))<<free);power=power[:,ids]
        ld=scale;left=selected[:,2].astype(object);right=selected[:,3].astype(object)
        if domain is not None:
            rd=lcm(*(F(v).denominator for v in list(residual)+list(domain)))
            ri=[int(F(v)*rd) for v in residual];dl,dh=[int(F(v)*rd) for v in domain]
            base=np.full(len(vertex),ri[0],object)
            for j in range(k):base+=((vertex>>j)&1).astype(object)*ri[j+1]
            slope=ri[free+1]
            if slope:
                ld=lcm(scale,abs(slope));left*=ld//scale;right*=ld//scale
                low=(dl-base if slope>0 else base-dh)*(ld//abs(slope))
                high=(dh-base if slope>0 else base-dl)*(ld//abs(slope))
                left=np.maximum(left,low);right=np.minimum(right,high)
                valid=np.fromiter((a<=b for a,b in zip(left,right)),bool,len(left))
            else:valid=np.fromiter((dl<=b<=dh for b in base),bool,len(base))
            if not np.any(valid):continue
            left=left[valid];right=right[valid];power=power[:,valid]
        shifted=np.zeros_like(power);span=right-left
        for degree in range(5):
            for exponent in range(degree,5):
                shifted[degree]+=power[exponent]*comb(exponent,degree)*left**(exponent-degree)*span**degree*ld**(4-exponent)
        stack=[(bernstein(shifted),den*ld**4*12,0)];count+=power.shape[1]
        while stack:
            c,dn,depth=stack.pop();ub=np.max(c,axis=0)
            lb=F(int(max(max(c[0]),max(c[-1]))),dn);best=lb if best is None else max(best,lb)
            cutoff=(best+tolerance)*dn
            active=np.fromiter((v>cutoff for v in ub),bool,len(ub))
            if np.any(~active):
                val=F(int(max(ub[~active])),dn);upper=val if upper is None else max(upper,val)
            if not np.any(active):continue
            c=c[:,active]
            if depth>=24:
                val=F(int(max(ub[active])),dn);upper=val if upper is None else max(upper,val);continue
            low,high=halve(c);stack.extend([(low,dn*16,depth+1),(high,dn*16,depth+1)])
    if best is None:return None,dict(edges=0)
    return max(best,upper) if upper is not None else best,dict(edges=int(count),sample_lower=float(best))

