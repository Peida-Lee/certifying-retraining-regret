"""Exact reconstruction of original row polynomials and coupling PSD proofs."""
from pathlib import Path
from fractions import Fraction as F
from math import lcm
import sys,json,time,argparse
B=Path(__file__).resolve().parent
import numpy as np,psutil
from audit_sdp_exact import denominator,int_array
from observed_anchor import float_up
from edge_polynomial import edge_upper

def certify_psd(proposal,mu):
    H=[[[F(v) for v in row] for row in h] for h in proposal['quadratic']]
    d=[list(map(F,row)) for row in proposal['linear']];m=len(H);p=len(d[0]);N=1+m*p
    a=F(10**6)*F(proposal['gscale'])**2/mu
    Q=[[F(0)]*N for _ in range(N)];Q[0][0]=F(proposal['alpha'])
    for i in range(m):
        for j in range(p):
            ii=1+i*p+j;Q[0][ii]=Q[ii][0]=d[i][j]
            for k in range(m):
                for ell in range(p):Q[ii][1+k*p+ell]=(H[i][j][ell] if i==k else F(0))-(a if j==ell else F(0))
    qf=np.asarray(Q,dtype=float);vals,V=np.linalg.eigh(qf);assert min(vals)>0
    S=np.round((V/np.sqrt(vals)[None,:])*2**40)/2**40
    dq=lcm(*(v.denominator for row in Q for v in row));ds=denominator(S.flat)
    Qi=np.array([[int(v*dq) for v in row] for row in Q],dtype=object);Si=int_array(S,ds)
    checked=Si.T@Qi@Si;den=dq*ds*ds
    margin=min(F(int(checked[i,i]-sum(abs(checked[i,j]) for j in range(N) if j!=i)),den) for i in range(N))
    assert margin>0
    return H,d,Q,dict(dimension=N,congruence_lower=str(margin),status='PASS',cross_blocks='exact original metric')

def reconstruct(data,proposal):
    o=data['original'];s=data['problem'];lo=np.asarray(o['lo']);hi=np.asarray(o['hi']);n,p=lo.shape
    y=list(map(F,o['y']));beta=list(map(F,o['beta']));theta=list(map(F,s['theta']));T=[[F(v) for v in row] for row in s['T']]
    x=(lo+hi)/2
    for (i,j),v in zip(s['active_cells'],s['x0_uncertain']):x[i,j]=v
    rows=np.flatnonzero(np.any(lo!=hi,axis=1));fixed=np.flatnonzero(np.all(lo==hi,axis=1));m=len(rows);lam=F(o['lambda_value']);gs=F(proposal['gscale'])
    gf=[lam*v for v in theta];constant=lam*sum((v*v-w*w for v,w in zip(beta,theta)),F(0))
    def dot(a,b):return sum((v*w for v,w in zip(a,b)),F(0))
    for i in fixed:
        xx=list(map(F,x[i]));eb=dot(xx,beta)-y[i];et=dot(xx,theta)-y[i]
        constant+=(eb*eb-et*et)/n
        for j in range(p):gf[j]+=xx[j]*et/n
    gf=[dot(gf,[T[j][k] for j in range(p)]) for k in range(p)]
    H,d,Q,psd=certify_psd(proposal,F(s['mu']));polys=[]
    for ii,i in enumerate(rows):
        jj=np.flatnonzero(lo[i]!=hi[i]);k=len(jj);xx=list(map(F,x[i]));eb=dot(xx,beta)-y[i];et=dot(xx,theta)-y[i]
        delta=[(F(hi[i,j])-F(lo[i,j]))*(1 if x[i,j]==lo[i,j] else -1) for j in jj]
        grad={():[(dot([v*et/n for v in xx],[T[j][q] for j in range(p)])+gf[q]/m)/gs for q in range(p)]}
        poly={():F(10**6)*(eb*eb-et*et)/n}
        for a,j in enumerate(jj):
            grad[(a,)]=[delta[a]*(et*T[j][q]+theta[j]*dot(xx,[T[t][q] for t in range(p)]))/n/gs for q in range(p)]
            poly[(a,)]=F(2*10**6)*delta[a]*(eb*beta[j]-et*theta[j])/n
            for b in range(a,k):
                ell=jj[b];key=(a,b)
                grad[key]=[delta[a]*delta[b]*(theta[j]*T[j][q] if a==b else theta[ell]*T[j][q]+theta[j]*T[ell][q])/n/gs for q in range(p)]
                poly[key]=F(10**6)*delta[a]*delta[b]*(beta[j]*beta[ell]-theta[j]*theta[ell])*(1 if a==b else 2)/n
        terms=list(grad);dc=lcm(*(v.denominator for key in terms for v in grad[key]));dh=lcm(*(v.denominator for row in H[ii] for v in row))
        C=np.array([[int(v*dc) for v in grad[key]] for key in terms],dtype=object);Hi=np.array([[int(v*dh) for v in row] for row in H[ii]],dtype=object)
        gram=C@Hi@C.T;den=dc*dc*dh
        for a,key in enumerate(terms):
            poly[key]=poly.get(key,F(0))+2*dot(grad[key],d[ii])
            for b in range(a,len(terms)):
                product=tuple(sorted(key+terms[b]));poly[product]=poly.get(product,F(0))+F(int(gram[a,b])*(1 if a==b else 2),den)
        residual=[et]+[delta[a]*theta[j] for a,j in enumerate(jj)]
        polys.append((k,{key:value for key,value in poly.items() if value},residual))
    return constant,polys,psd

