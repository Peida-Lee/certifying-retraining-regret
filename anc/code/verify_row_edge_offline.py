"""Analytic checks for sparse edge covers and conditional exact bounds."""
from fractions import Fraction as F
from pathlib import Path
import itertools,json
import numpy as np
from row_edge_screen import screen,restricted_upper
from row_edge_oracle import EdgeOracle
from row_edge_input import prepare,oracle as full_oracle
from row_edge_exact import reconstruct,certify_psd

cases=[({(0,):F(1)},1,F(1),[F(0),F(1)],None),
       ({(0,):F(1),(0,0):F(-1)},1,F(1,4),[F(0),F(1)],None),
       ({(0,):F(1),(0,0):F(-1)},1,F(3,16),[F(0),F(1)],[F(0),F(1,4)]),
       ({():F(-1),(0,):F(2),(0,0):F(-2),(1,):F(2),(1,1):F(-2)},2,F(-1,2),[F(0),F(0),F(1)],None),
       ({(0,):F(1),(1,):F(1)},2,F(3,2),[F(0),F(0),F(1)],[F(1,4),F(1,2)]),
       ({(0,0):F(1),(0,0,0):F(-2),(0,0,0,0):F(1)},1,F(1,16),[F(0),F(1)],None)]
checks=[]
for i,(poly,k,opt,residual,domain) in enumerate(cases):
    cover=screen(poly,k,min(opt-F(1,10),F(-2)),depth=4)
    bound,_=restricted_upper(poly,k,cover,residual,domain,tolerance=F(1,10**11))
    assert opt<=bound<=opt+F(1,10**9),(i,opt,bound)
    checks.append(dict(case=i,status='PASS',optimum=str(opt),upper=str(bound)))
poly={(0,):F(1),(0,0):F(-1),(1,):F(2),(1,1):F(-2),(2,):F(-1,3)}
cut=F(1,8);cover=screen(poly,3,cut,4);edges=cover['edges'];scale=cover['interval_scale'];covered=0
for free in range(3):
    for fixed in itertools.product([0,1],repeat=2):
        z=list(fixed);z.insert(free,0);vertex=sum(v<<j for j,v in enumerate(z))
        for numerator in range(65):
            z[free]=F(numerator,64);value=sum(v*np.prod([z[j] for j in key]) for key,v in poly.items())
            if value>cut:
                assert any(f==free and v==vertex and F(l,scale)<=z[free]<=F(r,scale) for f,v,l,r in edges)
                covered+=1
rng=np.random.default_rng(92021);lo=rng.integers(-4,5,(6,4))/8;hi=lo.copy();hi[:2,:3]+=.25
y=rng.integers(-4,5,6)/8;beta=rng.integers(-4,5,4)/8;theta=rng.integers(-4,5,4)/8
ids=np.argwhere(lo!=hi);data=dict(original=dict(lo=lo.tolist(),hi=hi.tolist(),y=y.tolist(),beta=beta.tolist(),lambda_value=.25),problem=dict(theta=theta.tolist(),T=np.eye(4).tolist(),mu=.25,active_cells=ids.tolist(),x0_uncertain=lo[ids[:,0],ids[:,1]].tolist()))
rows,constant,coupling=prepare(data);m=len(rows);quad=(m+.25)*coupling*np.eye(4)
proposal=dict(alpha=1.,quadratic=[quad.tolist() for _ in rows],linear=[np.zeros(4).tolist() for _ in rows],gscale=rows[0]['gscale'])
constant,polys,_=reconstruct(data,proposal);oracle_checks=[]
for i,(row,(k,poly,residual)) in enumerate(zip(rows,polys)):
    all_edges=[[free,v,0,16] for free in range(k) for v in range(2**k) if not v&(1<<free)]
    cover=dict(interval_scale=16,dimension=k,edges=all_edges);engine=EdgeOracle(row,cover)
    low=float(min(row['et']));high=float(max(row['et']));domains=[None,[low+.2*(high-low),low+.8*(high-low)]]
    for domain in domains:
        numeric=engine.evaluate(quad,np.zeros(4),domain,tolerance=1e-7)
        exact,_=restricted_upper(poly,k,cover,residual,None if domain is None else list(map(F,domain)),tolerance=F(1,10**7))
        oldrow=dict(row)
        if domain is not None:oldrow['residual_domain']=domain
        legacy=full_oracle(oldrow,quad,np.zeros(4),tolerance=1e-7)
        assert abs(numeric['upper']-float(exact))<1e-6 and abs(numeric['upper']-legacy[0])<1e-6
        oracle_checks.append(dict(row=i,domain=domain,status='PASS',numeric=numeric['upper'],exact_upper=float(exact)))
# Independent exact original-objective identity at interior multivariate points.
H,dd,Q,psd=certify_psd(proposal,F(data['problem']['mu']));identity_checks=0
qlo=[[F(v) for v in row] for row in lo];qb=list(map(F,beta));qt=list(map(F,theta));qy=list(map(F,y));n=6;p=4;gs=F(proposal['gscale'])
dot=lambda a,b:sum((v*w for v,w in zip(a,b)),F(0))
for trial in range(20):
    zs=[[F(int(v),8) for v in rng.integers(0,9,3)] for _ in range(2)];x=[list(row) for row in qlo]
    for i,z in enumerate(zs):
        for j,t in enumerate(z):x[i][j]+=F(1,4)*t
    fixed_g=[F(1,4)*v for v in qt]
    for i in range(2,n):
        et=dot(x[i],qt)-qy[i]
        fixed_g=[g+xx*et/n for g,xx in zip(fixed_g,x[i])]
    blocks=[]
    for i in range(2):
        et=dot(x[i],qt)-qy[i];blocks.extend((xx*et/n+g/2)/gs for xx,g in zip(x[i],fixed_g))
    v=[F(1)]+blocks;quad_value=sum((v[i]*Q[i][j]*v[j] for i in range(len(v)) for j in range(len(v))),F(0))
    row_value=F(0)
    for z,(_,poly,_) in zip(zs,polys):
        for key,coef in poly.items():
            term=coef
            for j in key:term*=z[j]
            row_value+=term
    decomposed=constant+(F(proposal['alpha'])+row_value-quad_value)/10**6
    grad=[F(1,4)*w for w in qt]
    direct=F(1,4)*(dot(qb,qb)-dot(qt,qt))
    for i in range(n):
        eb=dot(x[i],qb)-qy[i];et=dot(x[i],qt)-qy[i]
        direct+=(eb*eb-et*et)/n;grad=[g+xx*et/n for g,xx in zip(grad,x[i])]
    direct+=dot(grad,grad)/F(data['problem']['mu'])
    assert decomposed==direct
    identity_checks+=1
report=dict(status='PASS',analytic_cases=checks,screened_points_checked=covered,sparse_oracles_against_exact=oracle_checks,exact_original_majorant_identities=identity_checks)
Path(__file__).with_name('row_edge_offline_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
