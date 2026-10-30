"""Exact integer Bernstein bounds on univariate row-edge polynomials."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from fractions import Fraction as F
from math import comb,lcm
import sys,json,time,argparse
B=Path(__file__).resolve().parent
import numpy as np,psutil
from observed_anchor import float_up

def multiply(a,b):
    out={}
    for k,v in a.items():
        for q,w in b.items():
            key=tuple(sorted(k+q));out[key]=out.get(key,F(0))+v*w
    return {k:v for k,v in out.items() if v}

def zeta(a,k):
    for j in range(k):
        step=1<<j;view=a.reshape(-1,2*step)
        view[:,step:]+=view[:,:step]
    return a

def edge_upper(poly,k,tolerance=F(1,10**11),max_depth=24,residual=None,domain=None):
    den=lcm(*(v.denominator for v in poly.values()))
    coeff={key:int(value*den) for key,value in poly.items()}
    vertex=np.zeros(1<<k,dtype=object)
    for key,value in coeff.items():vertex[sum(1<<j for j in set(key))]+=value
    zeta(vertex,k)
    if residual is None:
        bestq=F(max(vertex),den)
    else:
        assert domain is not None and len(residual)==k+1 and domain[0]<=domain[1]
        rd=lcm(*(F(v).denominator for v in list(residual)+list(domain)))
        ri=[int(F(v)*rd) for v in residual];dl,dh=[int(F(v)*rd) for v in domain]
        rv=np.full(1<<k,ri[0],dtype=object)
        codes=np.arange(1<<k,dtype=np.uint32)
        for j in range(k):rv+=((codes>>j)&1).astype(object)*ri[j+1]
        valid=np.fromiter((dl<=v<=dh for v in rv),bool,len(rv))
        bestq=F(max(vertex[valid]),den) if np.any(valid) else -F(sum(abs(v) for v in coeff.values()),den)
    final=bestq;counts=0;maxdepth=0
    for free in range(k):
        keep=[j for j in range(k) if j!=free];loc={j:i for i,j in enumerate(keep)}
        power=np.zeros((5,1<<(k-1)),dtype=object)
        for key,value in coeff.items():
            degree=key.count(free);mask=sum(1<<loc[j] for j in set(key) if j!=free)
            power[degree,mask]+=value
        for row in power:zeta(row,k-1)
        local_den=den
        if residual is not None:
            codes=np.arange(1<<(k-1),dtype=np.uint32);base=np.full(len(codes),ri[0],dtype=object)
            for j,physical in enumerate(keep):base+=((codes>>j)&1).astype(object)*ri[physical+1]
            slope=ri[free+1]
            if slope:
                span_den=abs(slope)
                left=np.maximum(0,dl-base if slope>0 else base-dh)
                right=np.minimum(span_den,dh-base if slope>0 else base-dl)
                valid=np.fromiter((a<=b for a,b in zip(left,right)),bool,len(base))
                if not np.any(valid):continue
                left=left[valid];right=right[valid];power=power[:,valid];width=right-left
                shifted=np.zeros_like(power)
                for degree in range(5):
                    for exponent in range(degree,5):
                        shifted[degree]+=power[exponent]*comb(exponent,degree)*left**(exponent-degree)*width**degree*span_den**(4-exponent)
                power=shifted;local_den*=span_den**4
            else:
                valid=np.fromiter((dl<=v<=dh for v in base),bool,len(base))
                if not np.any(valid):continue
                power=power[:,valid]
        bern=np.zeros_like(power)
        for q in range(5):
            for a in range(q+1):bern[q]+=power[a]*(comb(q,a)*12//comb(4,a))
        stack=[(bern,local_den*12,0)]
        while stack:
            control,scale,depth=stack.pop();ub=np.max(control,axis=0)
            leaflower=max(max(control[0]),max(control[-1]));bestq=max(bestq,F(leaflower,scale))
            cutoff=(bestq+tolerance)*scale
            active=np.fromiter((v>cutoff for v in ub),dtype=bool,count=len(ub))
            if np.any(~active):final=max(final,F(max(ub[~active]),scale))
            if not np.any(active):continue
            control=control[:,active];counts+=control.shape[1];maxdepth=max(maxdepth,depth)
            if depth>=max_depth or counts>1000000:
                final=max(final,F(max(ub[active]),scale));continue
            a,b,c,d,e=control
            center=a+4*b+6*c+4*d+e
            left=np.array([16*a,8*(a+b),4*(a+2*b+c),2*(a+3*b+3*c+d),center],dtype=object)
            right=np.array([center,2*(b+3*c+3*d+e),4*(c+2*d+e),8*(d+e),16*e],dtype=object)
            stack.extend([(left,scale*16,depth+1),(right,scale*16,depth+1)])
    return max(final,bestq),dict(binary_vertices=1<<k,edges=k*(1<<(k-1)),subdivided_edges=counts,maximum_depth=maxdepth,lower=float(bestq))

