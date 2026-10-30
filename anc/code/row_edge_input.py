"""Original dense-row problems, feasible search and row features."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
from math import comb
import sys,json,time,argparse,warnings
B=Path(__file__).resolve().parent
import numpy as np
from scipy.optimize import minimize
from sklearn.datasets import load_diabetes

def prepare(data):
    src=data['problem'];o=data['original'];lo=np.array(o['lo']);hi=np.array(o['hi'])
    beta=np.array(o['beta']);theta=np.array(src['theta']);T=np.array(src['T']);mu=src['mu'];y=np.array(o['y']);n,p=lo.shape
    x0=(lo+hi)/2
    for (i,j),v in zip(src['active_cells'],src['x0_uncertain']):x0[i,j]=v
    remaining=np.flatnonzero(np.any(lo!=hi,axis=1));fixed=np.all(lo==hi,axis=1);m=len(remaining)
    gf=(o['lambda_value']*theta+x0[fixed].T@(x0[fixed]@theta-y[fixed])/n)@T
    const=o['lambda_value']*(beta@beta-theta@theta)+np.sum((x0[fixed]@beta-y[fixed])**2-(x0[fixed]@theta-y[fixed])**2)/n
    rows=[]
    for i in remaining:
        jj=np.flatnonzero(lo[i]!=hi[i]);k=len(jj);ds=(hi[i,jj]-lo[i,jj])*np.where(x0[i,jj]==lo[i,jj],1.,-1.)
        codes=np.arange(1<<k,dtype=np.uint32);z=((codes[:,None]>>np.arange(k))&1).astype(float)
        x=np.repeat(x0[i,None,:],len(z),axis=0);x[:,jj]+=z*ds
        eb=x@beta-y[i];et=x@theta-y[i];gamma=(x*et[:,None])@T/n+gf/m
        rows.append(dict(row=int(i),jj=jj,ds=ds,z=z,x=x,eb=eb,et=et,gamma=gamma,f=(eb**2-et**2)/n,n=n,beta=beta,theta=theta,T=T,x0=x0[i],y=y[i]))
    gs=2.**np.ceil(np.log2(max(np.max(np.abs(row['gamma'])) for row in rows)))
    for row in rows:row['gamma']/=gs;row['gscale']=gs;row['f']*=1e6
    return rows,const*1e6,1e6*gs**2/mu

def at_point(row,z):
    x=row['x0'].copy();x[row['jj']]+=row['ds']*z
    # Recover the fixed anchor allocation from the saved zero vertex.
    anchor=row['gamma'][0]-(row['x0']*(row['x0']@row['theta']-row['y']))@row['T']/row['n']/row['gscale']
    eb=x@row['beta']-row['y'];et=x@row['theta']-row['y']
    g=(x*et)@row['T']/row['n']/row['gscale']+anchor
    return g,(eb**2-et**2)/row['n']*1e6

def oracle(row,H,d,tolerance=1e-4):
    g=row['gamma'];values=row['f']+np.sum((g@H)*g,axis=1)+2*g@d
    eligible=np.ones(len(values),bool)
    if 'residual_domain' in row:
        low,high=row['residual_domain'];eligible=(row['et']>=low)&(row['et']<=high)
    pool=np.flatnonzero(eligible)
    best=float(np.max(values[pool])) if len(pool) else -np.inf;bound=best
    bestz=row['z'][pool[np.argmax(values[pool])]].copy() if len(pool) else None
    cuts=[row['z'][int(j)].copy() for j in pool[np.argsort(values[pool])[-4:]]]
    maxactive=0
    for free,j in enumerate(row['jj']):
        ids=np.flatnonzero(row['z'][:,free]==0);g0=g[ids];x=row['x'][ids];eb=row['eb'][ids];et=row['et'][ids]
        ds=row['ds'][free];n=row['n'];T=row['T'];theta=row['theta'];beta=row['beta'];gs=row['gscale']
        g1=(ds*et[:,None]*T[j]+ds*theta[j]*(x@T))/(n*gs)
        g2=ds**2*theta[j]*T[j]/(n*gs);Hg0=g0@H+d
        coeff=np.array([values[ids],2*ds*(eb*beta[j]-et*theta[j])/n*1e6+2*np.sum(g1*Hg0,axis=1),
          np.full(len(ids),ds**2*(beta[j]**2-theta[j]**2)/n*1e6)+np.sum((g1@H)*g1,axis=1)+2*Hg0@g2,
          2*g1@H@g2,np.full(len(ids),g2@H@g2)])
        left=np.zeros(len(ids));right=np.ones(len(ids))
        if 'residual_domain' in row:
            low,high=row['residual_domain'];slope=ds*theta[j]
            if slope:
                v1=(low-et)/slope;v2=(high-et)/slope
                left=np.maximum(0.,np.minimum(v1,v2));right=np.minimum(1.,np.maximum(v1,v2));valid=left<=right
            else:valid=(et>=low)&(et<=high)
            if not np.any(valid):continue
            left=left[valid];right=right[valid];ids=ids[valid];coeff=coeff[:,valid]
            shifted=np.zeros_like(coeff);span=right-left
            for degree in range(5):
                for exponent in range(degree,5):shifted[degree]+=coeff[exponent]*comb(exponent,degree)*left**(exponent-degree)*span**degree
            coeff=shifted
        control=np.zeros_like(coeff)
        for q in range(5):
            for a in range(q+1):control[q]+=coeff[a]*comb(q,a)/comb(4,a)
        for values0,positions in [(control[0],left),(control[-1],right)]:
            top=np.argsort(values0)[-4:]
            for index in top:
                point=row['z'][ids[index]].copy();point[free]=positions[index]
                if values0[index]>best:
                    best=float(values0[index]);bestz=point.copy()
                if 'residual_domain' in row:cuts.append(point)
        stack=[(control,ids,left,right,0)]
        while stack:
            c,idx,left,right,depth=stack.pop();ub=np.max(c,axis=0);active=ub>best+tolerance
            if np.any(~active):bound=max(bound,float(np.max(ub[~active])))
            if not np.any(active):continue
            c=c[:,active];idx=idx[active];left=left[active];right=right[active];maxactive=max(maxactive,len(idx))
            a,b,cc,dd,e=c;center=(a+4*b+6*cc+4*dd+e)/16;mid=(left+right)/2
            q=int(np.argmax(center))
            if center[q]>best:
                best=float(center[q]);bestz=row['z'][idx[q]].copy();bestz[free]=mid[q];cuts.append(bestz.copy())
            if depth>=14:
                bound=max(bound,float(np.max(ub[active])));continue
            lc=np.array([a,(a+b)/2,(a+2*b+cc)/4,(a+3*b+3*cc+dd)/8,center])
            rc=np.array([center,(b+3*cc+3*dd+e)/8,(cc+2*dd+e)/4,(dd+e)/2,e])
            stack.extend([(lc,idx,left,mid,depth+1),(rc,idx,mid,right,depth+1)])
    return max(bound,best),best,bestz,cuts,maxactive



def problem(name,**unused):
    if name=='dense_diabetes':
        o=load_diabetes();xx=np.round((o.data-o.data.mean(0))/np.ptp(o.data,axis=0)*1024)/1024
        x=np.column_stack([np.ones(len(xx)),xx]);y=np.round((o.target-o.target.min())/np.ptp(o.target)*1024)/1024;k=10
    else:
        k=int(name.rsplit('_',1)[1]);rng=np.random.default_rng(77116)
        xx=np.round(rng.normal(size=(128,16))/4*1024)/1024;x=np.column_stack([np.ones(128),xx])
        coeff=np.array([1/(j+2) for j in range(17)]);noise=np.round(rng.normal(size=128)*.1*1024)/1024
        y=np.round((x@coeff+noise)*1024)/1024
    lam=1/32;beta=np.linalg.solve(x.T@x/len(x)+lam*np.eye(x.shape[1]),x.T@y/len(x))
    lo=x.copy();hi=x.copy();lo[:4,1:k+1]-=1/16;hi[:4,1:k+1]+=1/16
    return (lo,hi,y,beta,lam),dict(dataset=name,k=k,uncertain_rows=4,original_target=1e-8)

def witness_search(s,starts=4096,seed=92901):
 a=s.a;ids=s.ids;lookup={int(i):j for j,i in enumerate(a.remaining)};ir=np.array([lookup[int(i)] for i,j in ids]);jc=ids[:,1]
 low=a.lo[a.remaining];ds=(a.hi-a.lo)[ids[:,0],jc];qr=np.array(a.qr,float);rng=np.random.default_rng(seed);best=-np.inf;bestz=None
 def obj(z):
  x=low.copy();x[ir,jc]=low[ir,jc]+ds*z;M=a.G+x.T@x/a.n;r=qr+x.T@a.yr/a.n;g=M@a.beta-r;u=np.linalg.solve(M,g);theta=a.beta-u
  eb=x@a.beta-a.yr;et=x@theta-a.yr;gradient=2/a.n*(eb[ir]*a.beta[jc]-et[ir]*theta[jc])*ds
  return -float(g@u)*1e6,-gradient*1e6
 for k in range(starts):
  z=rng.uniform(0,1,len(ids)) if k%2 else rng.integers(0,2,len(ids)).astype(float)
  fit=minimize(obj,z,jac=True,bounds=[(0,1)]*len(ids),method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-8,'maxiter':300})
  if -fit.fun>best:best=-fit.fun;bestz=fit.x.copy()
 x=a.c.copy();x[ids[:,0],jc]=a.lo[ids[:,0],jc]+ds*np.clip(bestz,0,1);s.lower(x,s.fit(x))
