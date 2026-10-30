"""Conic coupling proposals separated by sparse row-edge oracles."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from math import comb
import json,time,warnings
import numpy as np
import cvxpy as cp
from row_edge_input import prepare,at_point
B=Path(__file__).resolve().parent

class EdgeOracle:
    def __init__(self,row,cover):
        self.row=row;self.cover=cover;e=np.asarray(cover['edges'],dtype=np.int64)
        self.free=e[:,0];self.vertex=e[:,1];self.left=e[:,2]/cover['interval_scale'];self.right=e[:,3]/cover['interval_scale']
        x=row['x'][self.vertex];et=row['et'][self.vertex];eb=row['eb'][self.vertex]
        j=row['jj'][self.free];ds=row['ds'][self.free];n=row['n'];gs=row['gscale'];T=row['T'];theta=row['theta'];beta=row['beta']
        self.g0=row['gamma'][self.vertex]
        self.g1=(ds[:,None]*et[:,None]*T[j]+(ds*theta[j])[:,None]*(x@T))/(n*gs)
        self.g2=(ds**2*theta[j])[:,None]*T[j]/(n*gs)
        self.f0=row['f'][self.vertex];self.f1=2*ds*(eb*beta[j]-et*theta[j])/n*1e6
        self.f2=ds**2*(beta[j]**2-theta[j]**2)/n*1e6
        self.et=et;self.slope=ds*theta[j];self.k=len(row['jj'])

    def intervals(self,domain):
        left=self.left.copy();right=self.right.copy()
        if domain is not None:
            low,high=domain;slope=self.slope;nonzero=slope!=0
            a=np.zeros(len(slope));b=np.zeros(len(slope))
            a[nonzero]=(low-self.et[nonzero])/slope[nonzero];b[nonzero]=(high-self.et[nonzero])/slope[nonzero]
            left[nonzero]=np.maximum(left[nonzero],np.minimum(a[nonzero],b[nonzero]))
            right[nonzero]=np.minimum(right[nonzero],np.maximum(a[nonzero],b[nonzero]))
            right[(~nonzero)&((self.et<low)|(self.et>high))]=-1
        ids=np.flatnonzero(left<=right)
        return ids,left[ids],right[ids]

    def point(self,edge,t):
        z=((self.vertex[edge]>>np.arange(self.k))&1).astype(float);z[self.free[edge]]=t;return z

    def seeds(self,domain,limit=16):
        ids,left,right=self.intervals(domain)
        if not len(ids):return []
        rng=np.random.default_rng(92021);ii=rng.choice(len(ids),min(limit,len(ids)),replace=False)
        return [self.point(ids[q],t) for q in ii for t in (left[q],right[q])]

    def evaluate(self,H,d,domain=None,tolerance=1e-5):
        ids,left,right=self.intervals(domain)
        if not len(ids):return dict(upper=-np.inf,points=[],edges=0)
        g0=self.g0[ids];g1=self.g1[ids];g2=self.g2[ids];hg=g0@H+d
        power=np.array([self.f0[ids]+np.sum((g0@H)*g0,axis=1)+2*g0@d,
            self.f1[ids]+2*np.sum(g1*hg,axis=1),
            self.f2[ids]+np.sum((g1@H)*g1,axis=1)+2*np.sum(g2*hg,axis=1),
            2*np.sum((g1@H)*g2,axis=1),np.sum((g2@H)*g2,axis=1)])
        shifted=np.zeros_like(power);span=right-left
        for q in range(5):
            for r in range(q,5):shifted[q]+=power[r]*comb(r,q)*left**(r-q)*span**q
        control=np.zeros_like(shifted)
        for q in range(5):
            for r in range(q+1):control[q]+=shifted[r]*comb(q,r)/comb(4,r)
        values=np.concatenate([control[0],control[-1]]);top=np.argsort(values)[-min(len(values),2048):][::-1]
        pts=[];point_keys=set()
        for t in top:
            point=self.point(ids[t%len(ids)],left[t] if t<len(ids) else right[t-len(ids)])
            key=tuple(point)
            if key in point_keys:continue
            point_keys.add(key);pts.append(point)
            if len(pts)>=24:break
        best=float(np.max(values));bound=best;stack=[(control,ids,left,right,0)]
        while stack:
            c,idx,l,r,depth=stack.pop();ub=np.max(c,axis=0);active=ub>best+tolerance
            if np.any(~active):bound=max(bound,float(np.max(ub[~active])))
            if not np.any(active):continue
            c=c[:,active];idx=idx[active];l=l[active];r=r[active]
            if depth>=20:bound=max(bound,float(np.max(ub[active])));continue
            a,b,cc,dd,e=c;center=(a+4*b+6*cc+4*dd+e)/16;mid=(l+r)/2
            q=int(np.argmax(center))
            if center[q]>best:best=float(center[q]);pts.append(self.point(idx[q],mid[q]))
            lc=np.array([a,(a+b)/2,(a+2*b+cc)/4,(a+3*b+3*cc+dd)/8,center])
            rc=np.array([center,(b+3*cc+3*dd+e)/8,(cc+2*dd+e)/4,(dd+e)/2,e])
            stack.extend([(lc,idx,l,mid,depth+1),(rc,idx,mid,r,depth+1)])
        return dict(upper=max(bound,best),lower=best,points=pts,edges=len(ids))

def solve(rows,oracles,constant,coupling,domains,seconds=150,initial=None,initial_points=None,lower=-np.inf,stop_gap=8e-9):
    start=time.monotonic();m=len(rows);p=len(rows[0]['beta']);points=[];seen=[]
    for i,oracle in enumerate(oracles):
        pts=oracle.seeds(domains[i])
        if initial_points is not None:
            for z in initial_points[i]:
                z=np.asarray(z);et=rows[i]['x0']@rows[i]['theta']-rows[i]['y']+sum(rows[i]['ds']*rows[i]['theta'][rows[i]['jj']]*z)
                if domains[i] is None or domains[i][0]-1e-13<=et<=domains[i][1]+1e-13:pts.append(z)
        if initial is not None:pts.extend(oracle.evaluate(np.array(initial['quadratic'][i]),np.array(initial['linear'][i]),domains[i])['points'])
        unique={tuple(v):v for v in pts};seen.append(set(unique));points.append(list(unique.values()))
        if not pts:return dict(status='EMPTY',upper=-np.inf,seconds=time.monotonic()-start)
    alpha=cp.Variable();lin=[cp.Variable(p) for _ in rows];quad=[cp.Variable((p,p),symmetric=True) for _ in rows];tau=cp.Variable(m)
    matrix=[[cp.reshape(alpha,(1,1),order='C')]+[cp.reshape(q,(1,p),order='C') for q in lin]]
    for i in range(m):matrix.append([cp.reshape(lin[i],(p,1),order='C')]+[quad[i]-coupling*np.eye(p) if i==j else -coupling*np.eye(p) for j in range(m)])
    psd=cp.bmat(matrix);best=None;history=[]
    if initial is not None:
        upper=constant+initial['alpha'];stats=[]
        for i,oracle in enumerate(oracles):
            res=oracle.evaluate(np.array(initial['quadratic'][i]),np.array(initial['linear'][i]),domains[i]);upper+=res['upper'];stats.append({k:v for k,v in res.items() if k!='points'})
        prior=dict(initial);prior['residual_domains']=domains
        best=dict(status='NUMERICAL_PROPOSAL',upper=upper/1e6,proposal=prior,row_oracles=stats,branch_candidates=[],points=[[v.tolist() for v in row] for row in points])
    for iteration in range(35):
        constraints=[psd>>0];row_constraints=[]
        for i,pts in enumerate(points):
            vals=[at_point(rows[i],z) for z in pts];g=np.array([q[0] for q in vals]);f=np.array([q[1] for q in vals])
            gg=(g[:,:,None]*g[:,None,:]).reshape(len(g),p*p)
            c=tau[i]>=f+gg@cp.reshape(quad[i],(p*p,),order='C')+2*g@lin[i];constraints.append(c);row_constraints.append(c)
        problem=cp.Problem(cp.Minimize(constant+alpha+cp.sum(tau)),constraints)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:problem.solve(solver='CLARABEL',max_iter=150,time_limit=min(30,max(1,seconds-(time.monotonic()-start))),tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-8)
            except cp.error.SolverError:problem.solve(solver='SCS',eps=1e-6,max_iters=50000,time_limit_secs=30)
        if alpha.value is None:raise RuntimeError(problem.status)
        Y=(psd.value+psd.value.T)/2;vals=np.linalg.eigvalsh(Y);shift=max(0.,-float(vals[0]))+1e-9*max(1.,float(vals[-1]))
        upper=constant+float(alpha.value)+shift;rowstats=[];added=0;branches=[]
        newpoints=[]
        for i,oracle in enumerate(oracles):
            res=oracle.evaluate(quad[i].value+shift*np.eye(p),lin[i].value,domains[i]);upper+=res['upper']
            rowstats.append({k:v for k,v in res.items() if k!='points'});newpoints.append(res['points'])
            weights=np.maximum(0,np.asarray(row_constraints[i].dual_value));sw=np.sum(weights)
            theta=rows[i]['theta'];jj=rows[i]['jj'];et=rows[i]['x0']@theta-rows[i]['y']+np.asarray(points[i])@(rows[i]['ds']*theta[jj])
            mean=float(weights@et/sw) if sw else float(np.mean(et));var=float(weights@(et-mean)**2/sw) if sw else float(np.var(et))
            branches.append(dict(row=i,mean=mean,variance=var))
        rec=dict(iteration=iteration,numeric_relaxation=float(problem.value)/1e6,numeric_upper=upper/1e6,absolute_gap=upper/1e6-lower,seconds=time.monotonic()-start,cut_count=sum(map(len,points)),solver_status=problem.status)
        history.append(rec)
        if best is None or upper/1e6<best['upper']:
            proposal=dict(alpha=float(alpha.value)+shift,linear=[v.value.tolist() for v in lin],quadratic=[(v.value+shift*np.eye(p)).tolist() for v in quad],coupling=coupling,constant=constant,gscale=rows[0]['gscale'],residual_domains=domains,summary=rec)
            best=dict(status='NUMERICAL_PROPOSAL',upper=upper/1e6,proposal=proposal,row_oracles=rowstats,branch_candidates=branches,points=[[v.tolist() for v in row] for row in points])
        elif not best['branch_candidates']:
            best['branch_candidates']=branches
        for i,pts in enumerate(newpoints):
            for point in pts:
                key=tuple(point)
                if key not in seen[i]:points[i].append(point);seen[i].add(key);added+=1
        if best['upper']-lower<=stop_gap or not added or time.monotonic()-start>=seconds:break
    best.update(seconds=time.monotonic()-start,history=history);return best
