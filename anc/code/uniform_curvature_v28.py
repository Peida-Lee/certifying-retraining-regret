"""Uniform interval-row curvature from the matrix Young inequality."""
from fractions import Fraction as F
from math import lcm
from pathlib import Path
import sys,copy
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import numpy as np
from audit_sdp_exact import denominator,int_array,dot,qvalue
from observed_anchor import float_down


def curvature(original,tau):
    lo=np.array(original['lo']);hi=np.array(original['hi']);c=(lo+hi)/2;half=(hi-lo)/2;n,p=c.shape
    fixed=np.all(lo==hi,axis=1);remaining=np.flatnonzero(~fixed);dc=denominator(c.flat);C=int_array(c,dc)
    dg=n*dc*dc;Gi=C[fixed].T@C[fixed];lam=F(original['lambda_value']);assert (lam*dg).denominator==1
    for j in range(p):Gi[j,j]+=int(lam*dg)
    G=[[F(int(v),dg) for v in row] for row in Gi]
    U=C[remaining].T@C[remaining];dhalf=denominator(half.flat);D=int_array(half,dhalf)
    counts=np.sum(lo!=hi,axis=1);dd=np.sum(counts[:,None]*D*D,axis=0)
    t=F(tau);assert 0<t<1
    lower=[[G[i][j]+(1-t)*F(int(U[i,j]),dg) for j in range(p)] for i in range(p)]
    for j in range(p):lower[j][j]-=(1/t-1)*F(int(dd[j]),n*dhalf*dhalf)
    return lower,G


def choose(data,tau):
    new=copy.deepcopy(data);G,_=curvature(new['original'],tau);gf=np.array(G,float)
    T=np.round(np.linalg.inv(np.linalg.cholesky(gf)).T*2**20)/2**20
    dm=lcm(*(v.denominator for row in G for v in row));M=np.array([[int(v*dm) for v in row] for row in G],object)
    dt=denominator(T.flat);Ti=int_array(T,dt);K=Ti.T@M@Ti;den=dm*dt*dt
    mu=min(F(int(K[i,i]-sum(abs(K[i,j]) for j in range(len(G)) if j!=i)),den) for i in range(len(G)));assert mu>0
    new['problem']['T']=T.tolist();new['problem']['mu']=float_down(mu)
    new['problem']['uniform_curvature_rule']=dict(kind='matrix_young',tau=float(tau),
        delta_bound='diagonal sum of row uncertain-cell count times squared halfwidth, divided by n')
    return new


def audit(data):
    o=data['original'];s=data['problem'];rule=s['uniform_curvature_rule'];assert rule['kind']=='matrix_young'
    lo=np.array(o['lo']);hi=np.array(o['hi']);n,p=lo.shape;assert np.all(np.isfinite(lo)) and np.all(lo<=hi)
    assert set(map(tuple,s['active_cells']))==set(map(tuple,np.argwhere(lo!=hi)))
    G,G0=curvature(o,rule['tau']);T=np.array(s['T']);dt=denominator(T.flat);Ti=int_array(T,dt)
    dm=lcm(*(v.denominator for row in G for v in row));Gi=np.array([[int(v*dm) for v in row] for row in G],object)
    K=Ti.T@Gi@Ti;den=dm*dt*dt
    mu=min(F(int(K[i,i]-sum(abs(K[i,j]) for j in range(p) if j!=i)),den) for i in range(p))
    assert 0<F(s['mu'])<=mu
    beta=list(map(F,o['beta']));theta=list(map(F,data['witness']['theta']));x=np.array(data['witness']['x']);y=list(map(F,o['y']))
    assert x.shape==lo.shape and np.all(x>=lo) and np.all(x<=hi)
    value=F(o['lambda_value'])*sum((b*b-t*t for b,t in zip(beta,theta)),F(0))
    for row,yi in zip(x,y):
        xx=list(map(F,row));value+=((dot(xx,beta)-yi)**2-(dot(xx,theta)-yi)**2)/n
    assert F(data['result']['lower'])<=value
    return dict(status='PASS',exact_metric=True,uniform_curvature='matrix_young',tau=rule['tau'],original_witness=True)
