"""Independent rational reconstruction of original polynomials and SDP receipts.

Does not import the solver, interval arithmetic, or polynomial implementation.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
from math import lcm
import json,itertools,time,traceback,sys
import numpy as np,psutil
B=Path(__file__).resolve().parent;OLD=B
_allowed=psutil.Process().cpu_affinity()
_audit_cpu=int(os.environ.get('REGRET_AUDIT_CPU','8'))
psutil.Process().cpu_affinity([_audit_cpu if _audit_cpu in _allowed else _allowed[-1]])
def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def denominator(values):
    # Original floats are dyadic, but averaging by n introduces other factors.
    common=1
    for value in values:common=lcm(common,F(value).denominator)
    return common

def int_array(values,d):
    rows=[]
    for row in values:
        scaled=[F(v)*d for v in row]
        assert all(v.denominator==1 for v in scaled),'Nonintegral rational scaling'
        rows.append([v.numerator for v in scaled])
    return np.array(rows,dtype=object)
def qmat(A,b):return [dot(row,b) for row in A]
def qvalue(G,r,b):return dot(b,qmat(G,b))-2*dot(r,b)
def build_constraints(keys):
    lookup={key:i for i,key in enumerate(keys)};done=set();out=[]
    def add(terms,rhs):
        combined={}
        for j,v in terms:combined[j]=combined.get(j,F(0))+F(v)
        terms=tuple(sorted((j,v) for j,v in combined.items() if v));key=(terms,F(rhs))
        if key not in done:done.add(key);out.append(key)
    for key in keys:
        if len(key)==1:continue
        z=lookup[key];parts=set()
        for mask in range(1,2**len(key)-1):
            left=tuple(key[i] for i in range(len(key)) if mask>>i&1)
            right=tuple(key[i] for i in range(len(key)) if not mask>>i&1)
            parts.add(tuple(sorted((left,right))))
        for first,second in sorted(parts):
            a=lookup[first];b=lookup[second];add([(z,1),(a,-1)],0)
            if a==b:
                for t in (F(0),F(1,4),F(1,2),F(3,4),F(1)):add([(a,2*t),(z,-1)],t*t)
            else:add([(z,1),(b,-1)],0);add([(a,1),(b,1),(z,-1)],1)
    return out

def audit(file, original_override=None, details=None):
    payload=json.loads(file.read_text());rec=payload['result'];problem=payload['problem'];seed=rec['seed']
    dataset=rec.get('dataset','superconductivity');instance=rec.get('instance',f'{dataset}_{seed}')
    input_path=file.parent/f'{instance}_adaptive_energy.json'
    if not input_path.exists():input_path=OLD/f'runs/independent_masks/certificates/{dataset}_{seed}_adaptive_energy.json'
    original=original_override if original_override is not None else json.loads(input_path.read_text())['certificate']
    original_lo=np.array(original['lo']);original_hi=np.array(original['hi'])
    lo=np.array(problem.get('domain_lo',original_lo));hi=np.array(problem.get('domain_hi',original_hi))
    assert np.all(lo>=original_lo) and np.all(hi<=original_hi) and np.all(lo<=hi)
    y=list(map(F,original['y']));beta=list(map(F,original['beta']));lam=F(original['lambda_value']);n,p=lo.shape
    X=(lo+hi)/2;ids=[tuple(v) for v in problem['active_cells']]
    assert set(ids)==set(map(tuple,np.argwhere(lo!=hi)))
    for (i,j),v in zip(ids,problem['x0_uncertain']):
        assert v in (lo[i,j],hi[i,j]);X[i,j]=v
    theta=list(map(F,problem['theta']));T=[[F(v) for v in row] for row in problem['T']];mu=F(problem['mu'])
    dx=denominator(X.flat);dy=denominator(y);Xi=int_array(X,dx);Yi=np.array([int(v*dy) for v in y],dtype=object)
    den=n*dx*dx;gram=Xi.T@Xi
    assert (lam*den).denominator==1
    for j in range(p):gram[j,j]+=int(lam*den)
    M=[[F(int(v),den) for v in row] for row in gram];r=[F(int(v),n*dx*dy) for v in Xi.T@Yi]
    fixed=np.all(original_lo==original_hi,axis=1);Gint=Xi[fixed].T@Xi[fixed]
    for j in range(p):Gint[j,j]+=int(lam*den)
    G=[[F(int(v),den) for v in row] for row in Gint];rf=[F(int(v),n*dx*dy) for v in Xi[fixed].T@Yi[fixed]]
    floor=G
    if problem.get('metric_rule',{}).get('kind')=='young_row_floor':
        eta=F(problem['metric_rule']['eta']);assert 0<eta<=1
        floor=[row.copy() for row in G]
        for i in np.flatnonzero(~fixed):
            center=[(F(l)+F(u))/2 for l,u in zip(original_lo[i],original_hi[i])]
            radius=[(F(u)-F(l))/2 for l,u in zip(original_lo[i],original_hi[i])];k=sum(v!=0 for v in radius)
            for j in range(p):
                for l in range(p):floor[j][l]+=(1-eta)*center[j]*center[l]/n
                floor[j][j]-=(1/eta-1)*k*radius[j]*radius[j]/n
    dg=denominator(v for row in floor for v in row);GI=int_array(floor,dg)
    dt=denominator(v for row in T for v in row);Ti=int_array(T,dt);metric=Ti.T@GI@Ti;md=dg*dt*dt
    exact_mu=min(F(int(metric[j,j]-sum(abs(metric[j,k]) for k in range(p) if k!=j)),md) for j in range(p))
    assert 0<mu<=exact_mu
    g0=[v-w for v,w in zip(qmat(M,theta),r)];grad={():g0};poly={():qvalue(M,r,beta)-qvalue(M,r,theta)}
    def add(key,v):poly[key]=poly.get(key,F(0))+v
    for a,(i,j) in enumerate(ids):
        x=list(map(F,X[i]));d=F(hi[i,j]-lo[i,j])*(1 if X[i,j]==lo[i,j] else -1)
        et=dot(x,theta)-y[i];eb=dot(x,beta)-y[i]
        v=[d*theta[j]*xx/n for xx in x];v[j]+=d*et/n;grad[(a,)]=v
        add((a,),2*d*(eb*beta[j]-et*theta[j])/n)
        for b in range(a,len(ids)):
            krow,k=ids[b]
            if krow!=i:continue
            db=F(hi[i,k]-lo[i,k])*(1 if X[i,k]==lo[i,k] else -1)
            v=[F(0)]*p
            if a==b:
                v[j]=d*d*theta[j]/n;add((a,b),d*d*(beta[j]*beta[j]-theta[j]*theta[j])/n)
            else:
                v[j]+=d*db*theta[k]/n;v[k]+=d*db*theta[j]/n
                add((a,b),2*d*db*(beta[j]*beta[k]-theta[j]*theta[k])/n)
            grad[(a,b)]=v
    terms=list(grad);dc=denominator(v for key in terms for v in grad[key]);Ci=int_array([grad[key] for key in terms],dc)@Ti
    gram=Ci@Ci.T;gd=F((dc*dt)**2)*mu
    for j,key in enumerate(terms):
        for k in range(j,len(terms)):add(tuple(sorted(key+terms[k])),F(int(gram[j,k]))/gd*(1 if j==k else 2))
    keys=[tuple(k) for k in problem['monomials']];lookup={key:j for j,key in enumerate(keys)}
    for key,low,upper in zip(keys,problem['coefficient_lower'],problem['objective']):
        value=poly.get(key,F(0));assert F(low)<=value<=F(upper),('coefficient',key,float(value),low,upper)
    assert F(problem['constant_lower'])<=poly[()]<=F(problem['constant_upper'])
    assert all(key in lookup or not key for key in poly)
    constraints=build_constraints(keys)
    # Additional rows are conditional on their explicitly declared halfspaces.
    # Coverage of those halfspaces is verified by the aggregate-tree auditor.
    extra=problem.get('extra_linear_inequalities',[])
    if extra:
        nvars=len(ids);blocksize=1+2*nvars;assert len(extra)%blocksize==0
        def normalized(terms):
            out={}
            for key,val in terms:
                key=tuple(sorted(key));out[key]=out.get(key,F(0))+F(val)
            return {k:v for k,v in out.items() if v}
        for offset in range(0,len(extra),blocksize):
            first=extra[offset];aa=normalized(first['terms']);bb=F(first['rhs'])
            assert all(len(k)==1 for k in aa)
            expected=[(aa,bb)]
            for j in range(nvars):
                prod=[(tuple(sorted(k+(j,))),v) for k,v in aa.items()]
                expected.append((normalized(prod+[((j,),-bb)]),F(0)))
                expected.append((normalized(list(aa.items())+[(k,-v) for k,v in prod]+[((j,),bb)]),bb))
            for cut,(terms,rhs) in zip(extra[offset:offset+blocksize],expected):
                assert normalized(cut['terms'])==terms and F(cut['rhs'])==rhs
                constraints.append((tuple((lookup[k],v) for k,v in normalized(cut['terms']).items()),rhs))
    multipliers=list(map(F,rec['dual']));assert len(multipliers)==len(constraints) and min(multipliers)>=0
    scale=F(rec['scale']);residual=[F(v)*scale for v in problem['objective']];bound=F(0)
    for multiplier,(row,rhs) in zip(multipliers,constraints):
        bound+=multiplier*rhs
        if multiplier:
            for j,value in row:residual[j]-=multiplier*value
    if 'moment_blocks' in rec:
        blocks=rec['moment_blocks'];factors=rec['psd_factors']
        assert len(blocks)==len(factors) and blocks[0]['kind']=='gradient_moment'
        assert blocks[0]['basis']==problem['moment_terms'] and blocks[0]['polynomial']==[[[],1.]]
        for block in blocks[1:]:
            if block['kind']=='row_moment':
                group={j for j,ij in enumerate(ids) if ij[0]==block['row']}
                assert block['polynomial']==[[[],1.]]
                assert block['basis']==[k for k in problem['moment_terms'] if all(j in group for j in k)]
                continue
            assert block['kind']=='aggregate_halfspace_localizer'
            offset=block['base_cut'];assert offset%(1+2*len(ids))==0
            base=extra[offset]
            expected={():F(base['rhs'])}
            for k,v in base['terms']:expected[tuple(k)]=-F(v)
            actual={tuple(k):F(v) for k,v in block['polynomial']}
            assert actual==expected
            row=block['row'];assert block['basis']==[[]]+[[j] for j,ij in enumerate(ids) if ij[0]==row]
    else:
        blocks=[dict(basis=problem['moment_terms'],polynomial=[[[],1.]])];factors=[rec['psd_factor']]
    for block,rawfactor in zip(blocks,factors):
        factor=[[F(v) for v in row] for row in rawfactor];df=denominator(v for row in factor for v in row);Fi=int_array(factor,df);S=Fi@Fi.T
        moment_terms=[tuple(k) for k in block['basis']]
        assert len(factor)==len(moment_terms)
        for i,first in enumerate(moment_terms):
            for j,second in enumerate(moment_terms):
                for term,coefficient in block['polynomial']:
                    key=tuple(sorted(first+second+tuple(term)));v=F(int(S[i,j]),df*df)*F(coefficient)
                    if not key:bound+=v
                    else:residual[lookup[key]]+=v
    exact_upper=(bound+sum((max(F(0),v) for v in residual),F(0)))/scale+F(problem['constant_upper'])
    assert F(rec['upper'])>=exact_upper,('dual_upper',rec['upper'],float(exact_upper))
    w=payload['witness'];xw=np.array(w['x']);tw=list(map(F,w['theta']));assert np.all(xw>=lo) and np.all(xw<=hi)
    lower=qvalue(G,rf,beta)-qvalue(G,rf,tw)
    for i in np.flatnonzero(~fixed):
        row=list(map(F,xw[i]));lower+=((dot(row,beta)-y[i])**2-(dot(row,tw)-y[i])**2)/n
    assert F(rec['lower'])<=lower
    if details is not None:
        details.update(keys=keys,constraints=constraints,residual=residual,bound=bound,
            scale=scale,constant=F(problem['constant_upper']),exact_lower=lower,exact_upper=exact_upper)
    return dict(seed=seed,dataset=dataset,file=file.name,status='PASS',polynomial_coefficients=len(keys)+1,dual_constraints=len(constraints),PSD_by_exact_factor=True,
        exact_metric_lower=str(exact_mu),exact_dual_upper=str(exact_upper),verified_reported_width=rec['absolute_gap'],
        target=rec.get('target'),target_met=(rec['absolute_gap']<=rec['target']) if 'target' in rec else None)

if __name__=='__main__':
    folder=sys.argv[1] if len(sys.argv)>1 else 'sdp_existing'
    reports=[];start=time.monotonic()
    for file in sorted((B/folder).glob('*_receipt.json')):
        try:row=audit(file)
        except Exception as exc:row=dict(status='ERROR',file=file.name,error=str(exc),traceback=traceback.format_exc())
        reports.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('exact_metric_lower','exact_dual_upper')}),flush=True)
        (B/f'{folder}_exact_audit.json').write_text(json.dumps(dict(status='PASS' if all(r['status']=='PASS' for r in reports) else 'ERROR',reports=reports,seconds=time.monotonic()-start),indent=2))
