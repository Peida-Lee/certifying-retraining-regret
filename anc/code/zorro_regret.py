"""Scoped ZORRO fixed-point reimplementation and joint-symbol regret adapter.

Implements Zhu et al. (NeurIPS 2024), Appendix I, equations (22), (23), (39).
Labels are observed and every uncertain cell has its own shared symbol. Matrix
and polynomial coefficient enclosures are outward rounded. The exact original
ridge objective and all predictor columns are retained. This is a paper-based
reimplementation, not execution of the authors' notebook.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import time,itertools
import numpy as np
from observed_anchor import I,mm,matvec,dot,abs_upper,up,down


def outer(a,b):return a[:,None]*b[None,:]


def stack(columns):
    return I(np.column_stack([v.lo for v in columns]),np.column_stack([v.hi for v in columns]))


def inverse_enclosure(B):
    J=np.linalg.inv(B)
    error=I(np.eye(len(B)))-mm(I(J),I(B))
    q=float(I(abs_upper(error)).sum(axis=1).hi.max())
    assert q<1
    mag=float(I(abs(J)).sum(axis=1).hi.max())
    delta=float((I(q)*mag/(1-I(q))).hi)
    return I(down(J-delta),up(J+delta))


def enclosure(c,rad,y,beta,lam,deadline):
    start=time.monotonic();n,p=c.shape
    M=mm(I(c.T),I(c))/n+I(lam*np.eye(p))
    rhs=matvec(I(c.T),I(y))/n
    midM=c.T@c/n+lam*np.eye(p)
    wR=np.linalg.solve(midM,c.T@y/n)
    _,B=np.linalg.eigh(midM);B=np.round(B*2**24)/2**24
    J=inverse_enclosure(B)
    Q=mm(J,mm(M,I(B)))
    cells=[tuple(map(int,v)) for v in np.argwhere(rad)]
    linear={};quadratic={};labels={};pert=I(np.zeros((p,p)))
    for k,(i,j) in enumerate(cells):
        e=np.eye(p)[j];width=rad[i,j]
        linear[(k,)]=(outer(I(c[i]),I(e))+outer(I(e),I(c[i])))*(I(width)/n)
        labels[k]=I(e)*(I(width)*I(y[i])/n)
        transformed=(outer(matvec(J,I(c[i])),I(B[j]))+outer(J[:,j],matvec(I(B.T),I(c[i]))))*(I(width)/n)
        pert=pert+I(abs_upper(transformed))
        for ell in range(k,len(cells)):
            ii,jj=cells[ell]
            if ii!=i:continue
            if ell==k:
                part=outer(I(e),I(e))*(I(width)*I(width)/n)
                transformed=outer(J[:,j],I(B[j]))*(I(width)*I(width)/n)
            else:
                ee=np.eye(p)[jj]
                part=(outer(I(e),I(ee))+outer(I(ee),I(e)))*(I(width)*I(rad[ii,jj])/n)
                transformed=(outer(J[:,j],I(B[jj]))+outer(J[:,jj],I(B[j])))*(I(width)*I(rad[ii,jj])/n)
            quadratic[(k,ell)]=part
            pert=pert+I(abs_upper(transformed))
    H=-up(abs_upper(Q)+pert.hi)
    np.fill_diagonal(H,down(np.diag(Q.lo)-np.diag(pert.hi)))
    s=np.linalg.solve(H,np.ones(p));Hs=matvec(I(H),I(s))
    margin=float(Hs.lo.min())
    if not (min(s)>0 and margin>0):
        return dict(status='NEEDS_SPLITTING',seconds=time.monotonic()-start,
                    positive_inverse_test=False,minimum_diagonal=float(np.diag(H).min()),
                    minimum_test_vector=float(s.min()),minimum_test_margin=margin)
    D=np.column_stack([np.linalg.solve(midM,labels[k].lo-linear[(k,)].lo@wR) for k in range(len(cells))])
    force={():matvec(M,I(wR))-rhs}
    def add(key,value):
        key=tuple(sorted(key));force[key]=force.get(key,I(np.zeros(p)))+value
    MD=mm(M,I(D))
    for k in range(len(cells)):
        add((k,),MD[:,k]+matvec(linear[(k,)],I(wR))-labels[k])
    for key,A in linear.items():
        AD=mm(A,I(D))
        for k in range(len(cells)):add(key+(k,),AD[:,k])
    for key,A in quadratic.items():
        add(key,matvec(A,I(wR)))
        AD=mm(A,I(D))
        for k in range(len(cells)):add(key+(k,),AD[:,k])
    transformed=mm(J,stack(list(force.values())))
    r=I(abs_upper(transformed)).sum(axis=1).hi
    kval=np.linalg.solve(H,r)
    kval=up(np.maximum(kval,0.)*(1+1e-8)+1e-15*s)
    for _ in range(4):
        residual=matvec(I(H),I(kval)).lo-r
        if min(residual)>=0:break
        delta=float(up(max(0.,np.max(-residual/Hs.lo))))
        kval=up(kval+up(delta*s)+1e-16*s)
    assert min(kval)>=0 and min(matvec(I(H),I(kval)).lo-r)>=0
    # Preserve all data-symbol dependencies in g_beta^T(beta-theta).
    g={():matvec(M,I(beta))-rhs}
    for key,A in linear.items():g[key]=matvec(A,I(beta))-labels[key[0]]
    for key,A in quadratic.items():g[key]=matvec(A,I(beta))
    displacement={():I(beta)-I(wR)}
    for k in range(len(cells)):displacement[(k,)]=-I(D[:,k])
    for j in range(p):displacement[(len(cells)+j,)]=-I(B[:,j])*I(kval[j])
    poly={}
    for key,x in g.items():
        for key2,v in displacement.items():
            monomial=tuple(sorted(key+key2));value=dot(x,v)
            poly[monomial]=poly.get(monomial,I(0.))+value
    upper=I(0.);split_scores=np.zeros(len(cells))
    for monomial,coefficient in poly.items():
        for index in set(monomial):
            if index<len(cells):split_scores[index]+=float(abs_upper(coefficient))*monomial.count(index)
        if not monomial:upper=upper+coefficient
        elif all(monomial.count(k)%2==0 for k in set(monomial)):
            upper=upper+I(0.,max(0.,float(coefficient.hi)))
        else:upper=upper+I(0.,float(abs_upper(coefficient)))
    check_count=0;largest_violation=0.
    rng=np.random.default_rng(90271)
    for eps in rng.uniform(-1,1,size=(32,len(cells))):
        x=c.copy()
        for v,(i,j) in zip(eps,cells):x[i,j]+=rad[i,j]*v
        theta=np.linalg.solve(x.T@x/n+lam*np.eye(p),x.T@y/n)
        t=np.linalg.solve(B,theta-wR-D@eps)
        largest_violation=max(largest_violation,float(np.max(abs(t)-kval)))
        value=float((beta-theta)@(x.T@x/n+lam*np.eye(p))@(beta-theta))
        assert np.max(abs(t)-kval)<=1e-10 and value<=float(upper.hi)+1e-10
        check_count+=1
    return dict(status='PASS',upper=float(upper.hi),seconds=time.monotonic()-start,
                positive_inverse_test=True,minimum_test_margin=margin,
                fixed_point_box_verified=True,rounding_residuals_included=True,
                data_symbols=len(cells),nondata_symbols=p,forcing_monomials=len(force),
                regret_monomials=len(poly),sampled_normal_equations=check_count,
                largest_sample_enclosure_violation=largest_violation,
                basis=B.tolist(),inverse_lower=J.lo.tolist(),inverse_upper=J.hi.tolist(),
                real_component=wR.tolist(),data_component=D.tolist(),nondata_radii=kval.tolist(),
                fixed_point_matrix=H.tolist(),forcing_radius=r.tolist(),split_scores=split_scores.tolist())


def run(a,common,budget,witness):
    start=time.monotonic();deadline=start+max(.001,budget-common)
    queue=[(a.c.copy(),a.rad.copy(),0)];complete=[];attempts=[];peak=1
    while queue and time.monotonic()<deadline:
        c,rad,depth=queue.pop()
        result=enclosure(c,rad,a.y,a.beta,a.lam,deadline)
        attempts.append({k:v for k,v in result.items() if k not in ('basis','inverse_lower','inverse_upper','real_component','data_component','nondata_radii','fixed_point_matrix','forcing_radius')})
        if time.monotonic()>deadline:
            queue.append((c,rad,depth));break
        if result['status']=='PASS':complete.append(result);continue
        # Appendix J splitting keeps lambda fixed and covers the original box.
        cells=np.argwhere(rad);weights=[rad[i,j]*(np.linalg.norm(c[i])+rad[i,j]) for i,j in cells]
        i,j=cells[int(np.argmax(weights))];half=rad[i,j]/2
        rad=rad.copy();rad[i,j]=half
        for sign in (-1,1):
            cc=c.copy();cc[i,j]+=sign*half;queue.append((cc,rad.copy(),depth+1))
        peak=max(peak,len(queue)+len(complete))
    full=not queue
    zupper=max((r['upper'] for r in complete),default=float('inf')) if full else float('inf')
    upper=min(a.gradient_upper,zupper);lower=witness['regret_lower']
    return dict(status='PASS',fixed_point_cover_complete=full,lower=lower,upper=upper,
                absolute_gap=upper-lower,uncapped_zorro_upper=zupper if np.isfinite(zupper) else None,
                common_cap=a.gradient_upper,total_seconds=common+time.monotonic()-start,
                preprocessing_seconds=common,attempts=len(attempts),completed_boxes=len(complete),
                unresolved_boxes=len(queue),peak_boxes=peak,attempt_log=attempts,
                envelope_receipts=complete,objective_lambda=a.lam,
                implementation='paper equations 22,23,39; joint-symbol g_beta dot displacement adapter')
