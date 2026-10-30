"""Reference proposal from the current relaxed moment adversary."""
import numpy as np

def propose(s,problem,moment,safeguard=False):
    a=s.a;ids=problem['active_cells'];x0=(a.lo+a.hi)/2
    for (i,j),v in zip(ids,problem['x0_uncertain']):x0[i,j]=v
    xx=x0[a.remaining];M=a.G+xx.T@xx/a.n;r=np.array(a.qr,float)+xx.T@a.yr/a.n
    terms=[tuple(k) for k in problem['moment_terms']];index={k:j for j,k in enumerate(terms)}
    MC=np.zeros((len(terms),a.p,a.p));RC=np.zeros((len(terms),a.p));MC[0]=M;RC[0]=r
    for z,(i,j) in enumerate(ids):
        d=(a.hi[i,j]-a.lo[i,j])*(1 if x0[i,j]==a.lo[i,j] else -1);t=index[(z,)]
        MC[t,j,:]+=d*x0[i]/a.n;MC[t,:,j]+=d*x0[i]/a.n;RC[t,j]=d*a.y[i]/a.n
        for w,(row,k) in enumerate(ids[z:],start=z):
            if row!=i:continue
            dd=(a.hi[row,k]-a.lo[row,k])*(1 if x0[row,k]==a.lo[row,k] else -1);t=index[(z,w)]
            if w==z:MC[t,j,j]+=d*d/a.n
            else:MC[t,j,k]+=d*dd/a.n;MC[t,k,j]+=d*dd/a.n
    Y=np.array(moment);means=Y[0];meanM=np.einsum('a,aij->ij',means,MC);meanR=means@RC
    L=np.einsum('ij,ajk->aik',a.t.T,MC);D=RC@a.t
    YL=(Y@L.reshape(len(terms),-1)).reshape(L.shape)
    curvature=np.einsum('aki,akj->ij',L,YL)/a.mu-meanM
    linear=np.einsum('aki,ak->i',L,Y@D)/a.mu-meanR
    vals,vecs=np.linalg.eigh((curvature+curvature.T)/2)
    candidate=vecs@((vecs.T@linear)/np.maximum(vals,1e-10))
    diagnostic=dict(minimum_numeric_curvature=float(min(vals)),norm_linear=float(np.linalg.norm(linear)))
    if safeguard:
        cutoff=max(1e-12,1e-8*max(0.,float(max(vals))))
        if min(vals)<=cutoff:
            try:candidate=np.linalg.solve((meanM+meanM.T)/2,meanR)
            except np.linalg.LinAlgError:candidate=s.fit(a.c)
            diagnostic['ill_conditioned_proposal']='expected_normal_equation_fit'
        # This is a proposal safeguard. Every proposed reference still receives
        # an independently verified uniform polynomial certificate.
        radius=np.sqrt(a.gradient_upper/a.mu)
        z=np.linalg.solve(a.t,candidate-a.beta);length=np.linalg.norm(z)
        if not np.isfinite(length):candidate=s.fit(a.c);diagnostic['nonfinite_fallback']=True
        elif length>radius:
            candidate=a.beta+a.t@(z*(radius/length));diagnostic['radius_clipped']=True
        diagnostic['proposal_radius']=float(radius)
    return candidate,diagnostic
