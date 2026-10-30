"""Exact leaf elimination for two-row star components of ridge regression.

Each leaf has rows (u, ell), (alpha, t), u and t in separate positive
intervals. Candidate is (1,0,...,0). All calculations are Fractions.
"""
from fractions import Fraction as F
import math,heapq

def quadmax(A,B,C,lo,hi):
    points=[lo,hi]
    if A<0:
        v=-B/(2*A)
        if lo<v<hi:points.append(v)
    x=max(points,key=lambda z:A*z*z+B*z+C)
    return A*x*x+B*x+C,x

def pair_query(s,n,lam,ell,alpha,y,ulo,uhi,tlo,thi):
    """Return exact maximum, u, t, and profiled leaf coefficient."""
    K=n*lam+ell*ell;r2=alpha*s-y;cand2=(alpha-y)**2
    candidates=[]
    for t in (tlo,thi):
        D=K+t*t;v=ell*s;w=-2*ell+t*r2
        A=1-s*s+v*v/D;B=-4+4*s+2*v*w/D;C=4-4-r2*r2+w*w/D+cand2
        value,u=quadmax(A,B,C,ulo,uhi);leaf=-(ell*(u*s-2)+t*r2)/D
        candidates.append((value,u,t,leaf))
    if ell:
        low,high=sorted((K*r2/thi,K*r2/tlo));a=ell*s;b=-2*ell
        if a:
            l,h=sorted(((low-b)/a,(high-b)/a));l=max(l,ulo);h=min(h,uhi)
        elif low<=b<=high:l,h=ulo,uhi
        else:l,h=F(1),F(0)
        if l<=h:
            c=n*lam/K
            value,u=quadmax(1-c*s*s,-4+4*c*s,4-4*c+cand2,l,h)
            r1=ell*(u*s-2)
            t=K*r2/r1 if r1 else tlo
            assert tlo<=t<=thi
            leaf=-(r1+t*r2)/(K+t*t)
            candidates.append((value,u,t,leaf))
    return max(candidates,key=lambda row:row[0])

def certify(groups,n,lam,target=F(1,100000)):
    lam=F(lam);groups=[tuple(map(F,g)) for g in groups]
    y2=sum((4+g[2]*g[2] for g in groups),F(0))/n
    q=y2/lam;radius=F(math.isqrt(q.numerator//q.denominator)+1)
    curvature=lam+sum((max(g[3]**2,g[4]**2)+g[1]**2 for g in groups),F(0))/n
    cache={};best=None;best_s=None
    def query(s):
        nonlocal best,best_s
        if s not in cache:
            witnesses=[pair_query(s,n,lam,*g) for g in groups]
            value=lam*(1-s*s)+sum((v[0] for v in witnesses),F(0))/n
            cache[s]=(value,witnesses)
            if best is None or value>best:best=value;best_s=s
        return cache[s][0]
    def upper(lo,hi):return max(query(lo),query(hi))+curvature*(hi-lo)**2/4
    counter=0;lo=-radius;hi=radius;ub=upper(lo,hi);heap=[(-ub,counter,lo,hi)];tree=[dict(id=0,lo=str(lo),hi=str(hi),upper=str(ub),children=[])]
    while heap and -heap[0][0]-best>target:
        neg,idx,lo,hi=heapq.heappop(heap)
        if -neg<=best:continue
        mid=(lo+hi)/2;query(mid)
        for cl,ch in ((lo,mid),(mid,hi)):
            counter+=1;u=min(-neg,upper(cl,ch));tree[idx]['children'].append(counter)
            tree.append(dict(id=counter,parent=idx,lo=str(cl),hi=str(ch),upper=str(u),children=[]))
            if u>best:heapq.heappush(heap,(-u,counter,cl,ch))
    ub=max(best,-heap[0][0] if heap else best)
    return dict(status='PASS',lower=str(best),upper=str(ub),width=str(ub-best),target=str(target),radius=str(radius),curvature=str(curvature),
        n=n,lambda_value=str(lam),groups=[[str(v) for v in row] for row in groups],tree=tree,
        points=[dict(s=str(s),value=str(v),pairs=[[str(x) for x in row] for row in w]) for s,(v,w) in cache.items()],
        witness_s=str(best_s),witness=[[str(x) for x in row] for row in cache[best_s][1]])
