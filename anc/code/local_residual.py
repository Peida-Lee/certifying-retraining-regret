"""Verified local residual certificates on original completion subboxes.

For any fixed comparator t, h_beta(X) = J_X(beta)-J_X(t)
 + g_t(X)' M_X^{-1} g_t(X).  A fixed verified metric gives
 M_X >= mu (T T')^{-1}, so the second term is bounded by
 ||T' g_t(X)||^2/mu.  Both terms are enclosed on each subbox.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,time,heapq
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent

from observed_anchor import I,mm,matvec,dot,row_max,interval_q,abs_upper,down,up
from interval_arithmetic import quadratic_max_upper
from fast_interval import mm,matvec
import itertools

def row_linear_upper(a0,b0,ag,bg,lc,lg):
    """max of difference of squares plus affine term; original-row edges."""
    m,k=ag.lo.shape
    bound=np.full(m,-np.inf)
    if k==0:return (a0.square()-b0.square()+lc).hi
    for free in range(k):
        others=[j for j in range(k) if j!=free]
        for bits in itertools.product((-1.,1.),repeat=k-1):
            aa=a0;bb=b0;cc=lc
            for j,s in zip(others,bits):
                aa=aa+ag[:,j]*s;bb=bb+bg[:,j]*s;cc=cc+lg[:,j]*s
            qa=ag[:,free].square()-bg[:,free].square()
            qb=2*(aa*ag[:,free]-bb*bg[:,free])+lg[:,free]
            qc=aa.square()-bb.square()+cc
            bound=np.maximum(bound,quadratic_max_upper(qa,qb,qc))
    return bound

class LocalResidualSearch:
    def __init__(self,a):
        self.a=a
        self.g=interval_q(a.qG)
        self.r=I([float(v) for v in a.qr])
        # qr is rational, so enclose its conversion explicitly.
        from observed_anchor import float_down,float_up
        self.r=I([float_down(v) for v in a.qr],[float_up(v) for v in a.qr])
        self.beta_anchor=dot(I(a.beta),matvec(self.g,I(a.beta)))-2*dot(self.r,I(a.beta))
        self.ids=np.argwhere(a.rad>0)
        self.best=0.;self.witness=None;self.records=[]

    def fit(self,x):
        a=self.a
        xx=x[a.remaining]
        return np.linalg.solve(a.G+xx.T@xx/a.n,np.array(a.qr,float)+xx.T@a.yr/a.n)

    def lower(self,x,theta):
        a=self.a
        xx=x[a.remaining]
        q=self.beta_anchor-dot(I(theta),matvec(self.g,I(theta)))+2*dot(self.r,I(theta))
        q=q+((matvec(I(xx),I(a.beta))-I(a.yr)).square()-(matvec(I(xx),I(theta))-I(a.yr)).square()).sum()/a.n
        lower=max(0.,float(q.lo))
        if lower>self.best:
            self.best=lower
            self.witness=dict(x=x.tolist(),theta=theta.tolist(),lower=lower)
        return lower

    def residual_rows(self,lo,hi,theta):
        a=self.a;c=(lo+hi)/2;rad=(hi-lo)/2
        assert np.all(c-rad==lo) and np.all(c+rad==hi)
        cr=c[a.remaining];rr=rad[a.remaining]
        widths=np.take_along_axis(rr,a.cells,axis=1)
        ec=matvec(I(cr),I(theta))-I(a.yr)
        eg=I(widths)*I(theta[a.cells])
        bc=matvec(I(cr),I(a.beta))-I(a.yr)
        bg=I(widths)*I(a.beta[a.cells])
        rows,signs=row_max(bc,ec,bg,eg,1.)
        return c,rad,cr,widths,ec,eg,rows,signs

    def gradient(self,theta,cr,widths,ec,eg):
        a=self.a;m,p,k=a.m,a.p,a.k
        pc=mm(I(cr),I(a.t))
        pg=I(widths[:,None,:])*I(a.t[a.cells,:].transpose(0,2,1))
        def flat(v):return I(v.lo.reshape(m*p),v.hi.reshape(m*p))
        def flatg(v):return I(v.lo.reshape(m*p,k),v.hi.reshape(m*p,k))
        aa=(pc+ec[:,None])/2;bb=(pc-ec[:,None])/2
        ag=(pg+eg[:,None,:])/2;bg=(pg-eg[:,None,:])/2
        high,_=row_max(flat(aa),flat(bb),flatg(ag),flatg(bg),1.)
        neg,_=row_max(flat(bb),flat(aa),flatg(bg),flatg(ag),1.)
        ranges=I(-neg.hi.reshape(m,p),high.hi.reshape(m,p))
        base=matvec(I(a.t.T),matvec(self.g,I(theta))-self.r)
        return base+ranges.sum(axis=0)/a.n

    def slope_upper(self,lo,hi,x0,theta,cr,widths,ec,eg,refit_corner=True):
        """Joint objective/residual majorant retaining row-wise loss penalties."""
        a=self.a;c=(lo+hi)/2;rad=(hi-lo)/2
        x0=np.where(x0>=c,hi,lo)
        if refit_corner:theta=self.fit(x0)
        self.lower(x0,theta)
        ec=matvec(I(cr),I(theta))-I(a.yr)
        eg=I(widths)*I(theta[a.cells])
        cbar=(I(x0)+I(c))/2
        active=[(int(i),int(j)) for i,j in self.ids if rad[i,j]>0]
        if not active:return None
        ir=np.array([i for i,j in active]);jc=np.array([j for i,j in active])
        center=mm(cbar[ir],I(a.t))*I(theta[jc,None])+(matvec(cbar[ir],I(theta))-I(a.y[ir]))[:,None]*I(a.t[jc])
        center=center/a.n
        lookup={int(i):ii for ii,i in enumerate(a.remaining)}
        row_slots=np.array([a.cells[lookup[i]] for i in ir])
        generators=[]
        for slot in range(a.k):
            kk=row_slots[:,slot]
            gg=I(rad[ir,kk,None]/2)*(I(theta[jc,None])*I(a.t[kk])+I(theta[kk,None])*I(a.t[jc]))/a.n
            generators.append(gg)
        # The exact floating-point comparator may have a nonzero normal residual.
        xx=x0[a.remaining]
        er=matvec(I(xx),I(theta))-I(a.yr)
        g0=matvec(I(a.t.T),matvec(self.g,I(theta))-self.r+matvec(I(xx.T),er)/a.n)
        constant=g0.square().sum()/a.mu
        parts=[center]+generators
        kernel=I(np.zeros((len(active),len(active))))
        orientation=np.where(x0[ir,jc]==lo[ir,jc],1.,-1.)
        signed_parts=[v*I(orientation[:,None]) for v in parts]
        signed_kernel=mm(signed_parts[0],I(signed_parts[0].lo.T,signed_parts[0].hi.T))
        for v in parts:
            for w in parts:kernel=kernel+I(abs_upper(mm(v,I(w.lo.T,w.hi.T))))
        for vi,v in enumerate(signed_parts):
            for wi,w in enumerate(signed_parts):
                if vi==wi==0:continue
                radk=abs_upper(mm(v,I(w.lo.T,w.hi.T)))
                signed_kernel=signed_kernel+I(-radk,radk)
        linear=I(np.zeros(len(active)))
        for v in parts:linear=linear+I(abs_upper(matvec(v,g0)))
        dl=abs_upper(I(lo[ir,jc])-I(x0[ir,jc]));du=abs_upper(I(hi[ir,jc])-I(x0[ir,jc]))
        dv=I(np.maximum(dl,du))
        cost=(2*linear+matvec(kernel,dv))/a.mu
        weights=np.zeros((a.m,a.k))
        for z,(i,j) in enumerate(active):
            ii=lookup[i];jj=int(np.flatnonzero(a.cells[ii]==j)[0]);weights[ii,jj]=float((I(cost.hi[z])*a.n).hi)
        bc=matvec(I(cr),I(a.beta))-I(a.yr)
        bg=I(widths)*I(a.beta[a.cells])
        delta=I(cr)-I(x0[a.remaining])
        dcell=I(np.take_along_axis(delta.lo,a.cells,axis=1),np.take_along_axis(delta.hi,a.cells,axis=1))
        high=np.full(a.m,-np.inf)
        for signs in itertools.product((-1.,1.),repeat=a.k):
            sg=I(weights)*I(signs)
            lc=(sg*dcell).sum(axis=1)
            lg=sg*I(widths)
            high=np.maximum(high,row_linear_upper(bc,ec,bg,eg,lc,lg))
        anchor=self.beta_anchor-dot(I(theta),matvec(self.g,I(theta)))+2*dot(self.r,I(theta))
        upper=anchor+I(high).sum()/a.n+constant
        # Preserve signed cross-products in a lifted quadratic majorant.
        from quadratic_majorant import bound
        signed_linear=matvec(signed_parts[0],g0)
        for v in signed_parts[1:]:
            rr=abs_upper(matvec(v,g0));signed_linear=signed_linear+I(-rr,rr)
        qb=I(a.beta);qt=I(theta)
        eb=matvec(I(x0[ir]),qb)-I(a.y[ir])
        et=matvec(I(x0[ir]),qt)-I(a.y[ir])
        lin=(2*(eb*qb[jc]-et*qt[jc])*I(orientation)/a.n+2*signed_linear/a.mu)*dv
        same=(ir[:,None]==ir[None,:]).astype(float)
        qrow=(qb[jc,None]*qb[jc][None,:]-qt[jc,None]*qt[jc][None,:])*I(orientation[:,None]*orientation[None,:])*I(same)/a.n
        quad=(qrow+signed_kernel/a.mu)*dv[:,None]*dv[None,:]
        lp=bound(lin.hi,quad.hi)
        xx=x0[a.remaining]
        q0=anchor+((matvec(I(xx),qb)-I(a.yr)).square()-(matvec(I(xx),qt)-I(a.yr)).square()).sum()/a.n
        lp_upper=float((q0+constant+lp['upper']).hi)
        upper=I(min(float(upper.hi),lp_upper))
        scores=np.zeros(len(self.ids))
        ids_lookup={tuple(map(int,z)):ii for ii,z in enumerate(self.ids)}
        for z,key in enumerate(active):scores[ids_lookup[key]]=cost.hi[z]*dv.hi[z]
        # A fractional lifted solution guides branching; its table is feasible.
        trial=x0.copy()
        for z,(i,j) in enumerate(active):trial[i,j]=np.clip(x0[i,j]+orientation[z]*dv.hi[z]*lp['solution'][z],lo[i,j],hi[i,j])
        self.lower(trial,self.fit(trial))
        frac=lp['solution']*(1-lp['solution'])
        if np.max(frac)>1e-6:
            for z,key in enumerate(active):scores[ids_lookup[key]]*=max(1e-3,frac[z])
        reduced=np.array(lp['residual_hi'][:len(active)])
        # Keep the covariance score; reduced costs remain diagnostic.
        cut=None;gain=0.
        for z,key in enumerate(active):
            coeff=reduced[z]
            gap=max(0.,lp_upper-self.best)
            if abs(coeff)>gap and abs(coeff)>0:
                fraction=min(1.,float(up(up(gap)/abs(coeff))))
                if 1-fraction>gain:
                    gain=1-fraction
                    tcut=fraction if coeff<0 else 1-fraction
                    i,j=key
                    value=x0[i,j]+orientation[z]*dv.hi[z]*tcut
                    if lo[i,j]<value<hi[i,j]:cut=(ids_lookup[key],value)
        return dict(upper=float(upper.hi),constant=float(constant.hi),costs=cost.hi.tolist(),scores=scores,lp_upper=lp_upper,cut=cut,reduced_costs=reduced.tolist())

    def node(self,lo,hi,start_theta=None,iterations=2):
        a=self.a
        theta=self.fit((lo+hi)/2) if start_theta is None else start_theta.copy()
        for _ in range(iterations):
            c,rad,cr,widths,ec,eg,rows,signs=self.residual_rows(lo,hi,theta)
            x=c.copy()
            for ii,i in enumerate(a.remaining):
                for j,w,s in zip(a.cells[ii],widths[ii],signs[ii]):
                    if w:x[i,j]=np.clip(x[i,j]+w*s,lo[i,j],hi[i,j])
            fit=self.fit(x);self.lower(x,fit)
            theta=fit
        c,rad,cr,widths,ec,eg,rows,signs=self.residual_rows(lo,hi,theta)
        anchor=self.beta_anchor-dot(I(theta),matvec(self.g,I(theta)))+2*dot(self.r,I(theta))
        fixed=anchor+rows.sum()/a.n
        grad=self.gradient(theta,cr,widths,ec,eg)
        remainder=grad.square().sum()/a.mu
        upper=float((fixed+remainder).hi)
        slope=self.slope_upper(lo,hi,x,theta,cr,widths,ec,eg)
        if slope is not None:upper=min(upper,slope['upper'])
        score=[]
        residual=np.abs(cr@theta-a.yr)+np.sum(widths*np.abs(theta[a.cells]),axis=1)
        tn=np.linalg.norm(a.t,axis=1)
        cn=np.linalg.norm(cr@a.t,axis=1)
        lookup={int(row):ii for ii,row in enumerate(a.remaining)}
        for i,j in self.ids:
            ii=lookup[int(i)]
            score.append(rad[i,j]*(abs(theta[j])*cn[ii]+residual[ii]*tn[j]))
        if slope is not None:score=slope['scores']
        chosen=int(np.argmax(score));cut=None
        return dict(upper=upper,theta=theta,split=chosen,cut=cut,fixed_upper=float(fixed.hi),residual_upper=float(remainder.hi),slope_upper=None if slope is None else slope['upper'])

    def run(self,time_limit=10,target=1e-5,max_nodes=100000,initial_witness=None):
        start=time.monotonic();a=self.a
        theta=None
        if initial_witness is not None:
            x=np.array(initial_witness['x']);theta=self.fit(x);self.lower(x,theta)
        root=self.node(a.lo,a.hi,theta)
        heap=[(-root['upper'],0,a.lo.copy(),a.hi.copy(),root)]
        self.records=[dict(id=0,parent=None,upper=root['upper'],theta=root['theta'].tolist(),split=None,children=[])]
        expanded=0;trace=[];counter=0
        while heap:
            upper=max(self.best,-heap[0][0]);elapsed=time.monotonic()-start
            trace.append(dict(seconds=elapsed,lower=self.best,upper=upper,nodes=expanded))
            if upper-self.best<=target or elapsed>=time_limit or expanded>=max_nodes:break
            neg,idx,lo,hi,rec=heapq.heappop(heap)
            if -neg<=self.best:continue
            i,j=map(int,self.ids[rec['split']]);mid=(lo[i,j]+hi[i,j])/2
            if rec.get('cut') is not None:
                # Round the proposal onto an exact dyadic midpoint grid.
                step=(hi[i,j]-lo[i,j])/256
                pos=min(255,max(1,int(round((rec['cut']-lo[i,j])/step))))
                suggested=lo[i,j]+step*pos
                valid=True
                for l,u in ((lo[i,j],suggested),(suggested,hi[i,j])):
                    cc=(l+u)/2;rr=(u-l)/2
                    valid=valid and cc-rr==l and cc+rr==u
                if valid:mid=suggested
            self.records[idx]['split']=[i,j,mid]
            for side in (0,1):
                ll=lo.copy();uu=hi.copy()
                if side==0:uu[i,j]=mid
                else:ll[i,j]=mid
                child=self.node(ll,uu,rec['theta'],iterations=1)
                child['upper']=min(child['upper'],-neg)
                counter+=1
                self.records[idx]['children'].append(counter)
                self.records.append(dict(id=counter,parent=idx,side=side,upper=child['upper'],theta=child['theta'].tolist(),split=None,children=[],fixed_upper=child['fixed_upper'],residual_upper=child['residual_upper']))
                if child['upper']>self.best:heapq.heappush(heap,(-child['upper'],counter,ll,uu,child))
            expanded+=1
        upper=max(self.best,-heap[0][0] if heap else self.best)
        return dict(status='PASS',lower=self.best,upper=upper,absolute_gap=upper-self.best,nodes=expanded,root_remainder=root['residual_upper'],root_width=root['upper']-self.best,seconds=time.monotonic()-start,trace=trace,certificate=dict(nodes=self.records,T=a.t.tolist(),mu=a.mu,witness=self.witness))
