"""Rational reconstruction and small-Schur certificate for row-coordinate duals."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from fractions import Fraction as F
from pathlib import Path
import argparse,json,sys,time,traceback,copy
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import numpy as np,psutil
from edge_polynomial import edge_upper
from observed_anchor import float_up
from run_row_edge_refinement import audit_input


def dot(a,b):return sum((x*y for x,y in zip(a,b)),F(0))
def transpose(a):return list(map(list,zip(*a)))
def mm(a,b):return [[dot(row,col) for col in zip(*b)] for row in a]


def positive_ldl(a):
    n=len(a);work=[list(row) for row in a];pivots=[]
    assert all(a[i][j]==a[j][i] for i in range(n) for j in range(n))
    for k in range(n):
        pivot=work[k][k];assert pivot>0,(k,float(pivot));pivots.append(pivot)
        for i in range(k+1,n):
            for j in range(i,n):
                value=work[i][j]-work[i][k]*work[k][j]/pivot
                work[i][j]=work[j][i]=value
    return pivots


def inverse(a):
    n=len(a);q=[list(row)+[F(i==j) for j in range(n)] for i,row in enumerate(a)]
    for k in range(n):
        pivot=q[k][k];assert pivot
        q[k]=[v/pivot for v in q[k]]
        for i in range(n):
            if i!=k:
                co=q[i][k];q[i]=[v-co*w for v,w in zip(q[i],q[k])]
    ans=[row[n:] for row in q]
    assert mm(a,ans)==[[F(i==j) for j in range(n)] for i in range(n)]
    return ans


def dyadic_majorant(a,bits=64):
    """Return a symmetric dyadic PSD upper approximation using exact row errors."""
    den=1<<bits;n=len(a)
    out=[[F((v*den+F(1,2)).numerator//(v*den+F(1,2)).denominator,den) for v in row] for row in a]
    error=max(sum((abs(out[i][j]-a[i][j]) for j in range(n)),F(0)) for i in range(n))
    scaled=error*den;shift=F(-(-scaled.numerator//scaled.denominator)+1,den)
    for i in range(n):out[i][i]+=shift
    assert all(out[i][i]-a[i][i]>=sum(abs(out[i][j]-a[i][j]) for j in range(n) if j!=i) for i in range(n))
    return out,shift


def reconstruct(data,proposal):
    if proposal.get('reference_theta') is not None:
        data=copy.deepcopy(data);data['problem']['theta']=proposal['reference_theta']
    original=data['original'];src=data['problem'];lo=np.array(original['lo']);hi=np.array(original['hi']);n,p=lo.shape
    beta=list(map(F,original['beta']));theta=list(map(F,src['theta']));y=list(map(F,original['y']))
    T=[[F(v) for v in row] for row in src['T']];TT=transpose(T);dimension=len(TT);gs=F(proposal['gscale']);mu=F(src['mu']);lam=F(original['lambda_value'])
    assert gs>0 and mu>0
    x=(lo+hi)/2
    for (i,j),v in zip(src['active_cells'],src['x0_uncertain']):
        assert v in (lo[i,j],hi[i,j]);x[i,j]=v
    remaining=np.flatnonzero(np.any(lo!=hi,axis=1));fixed=np.flatnonzero(np.all(lo==hi,axis=1));m=len(remaining)
    gf=[lam*v for v in theta];constant=lam*sum((b*b-t*t for b,t in zip(beta,theta)),F(0))
    for i in fixed:
        xx=list(map(F,x[i]));eb=dot(xx,beta)-y[i];et=dot(xx,theta)-y[i]
        constant+=(eb*eb-et*et)/n
        for j in range(p):gf[j]+=xx[j]*et/n
    gf=[dot(gf,col) for col in TT]
    origins=[];rowdata=[]
    for ii,i in enumerate(remaining):
        jj=np.flatnonzero(lo[i]!=hi[i]);k=len(jj);xx=list(map(F,x[i]));eb=dot(xx,beta)-y[i];et=dot(xx,theta)-y[i]
        delta=[(F(hi[i,j])-F(lo[i,j]))*(1 if x[i,j]==lo[i,j] else -1) for j in jj]
        scale=list(map(F,proposal['coordinate_scales'][ii]));assert len(scale)==k+1 and all(s>0 for s in scale)
        raw=[dot(xx,col) for col in TT]
        V=[[raw[q]/scale[0]]+[T[j][q]/scale[a+1] for a,j in enumerate(jj)] for q in range(dimension)]
        indices=proposal.get('coordinate_indices',[list(range(len(v))) for v in proposal['coordinate_scales']])[ii]
        assert len(set(indices))==len(indices) and all(0<=j<=k for j in indices)
        assert all(V[q][j]==0 for q in range(dimension) for j in range(k+1) if j not in indices)
        V=[[r[j] for j in indices] for r in V]
        origin=[(raw[q]*et/n+gf[q]/m)/gs for q in range(dimension)];origins.append(origin)
        u={}
        def add_u(key,index,value):
            if key not in u:u[key]=[F(0)]*(k+1)
            u[key][index]+=value
        poly={():F(10**6)*(eb*eb-et*et)/n}
        for a,j in enumerate(jj):
            add_u((a,),0,delta[a]*theta[j]*scale[0]/n/gs)
            add_u((a,),a+1,delta[a]*et*scale[a+1]/n/gs)
            poly[(a,)]=F(2*10**6)*delta[a]*(eb*beta[j]-et*theta[j])/n
            for b,ell in enumerate(jj):
                add_u(tuple(sorted((a,b))),a+1,delta[a]*delta[b]*theta[ell]*scale[a+1]/n/gs)
                if b>=a:poly[(a,b)]=F(10**6)*delta[a]*delta[b]*(beta[j]*beta[ell]-theta[j]*theta[ell])*(1 if a==b else 2)/n
        partition_theta=list(map(F,data.get('partition_theta',proposal.get('partition_theta',src['theta']))))
        u={key:[v[j] for j in indices] for key,v in u.items()}
        rowdata.append(dict(k=k,coordinate_count=len(indices),V=V,u=u,poly=poly,residual=[dot(xx,partition_theta)-y[i]]+[delta[a]*partition_theta[j] for a,j in enumerate(jj)],row=int(i),jj=jj,delta=delta,x0=xx,origin=origin))
    g0=[sum((v[q] for v in origins),F(0)) for q in range(dimension)]
    c=F(10**6)*gs*gs/mu;constant=F(10**6)*constant+c*dot(g0,g0)
    cuts=[cut for cut in src.get('extra_linear_inequalities',[]) if all(len(key)==1 for key,co in cut['terms'])]
    multipliers=list(map(F,proposal.get('halfspace_multipliers',[])))
    assert len(multipliers)==len(cuts) and all(v>=0 for v in multipliers)
    cells=[tuple(v) for v in src['active_cells']];cell_lookup={v:j for j,v in enumerate(cells)}
    local_linear={}
    for cut,lam0 in zip(cuts,multipliers):
        constant+=lam0*F(cut['rhs'])
        for key,co in cut['terms']:
            ij=cells[key[0]];local_linear[ij]=local_linear.get(ij,F(0))-lam0*F(co)
    total=[[F(0)]*(dimension+1) for _ in range(dimension+1)];local_checks=[];polynomials=[]
    for ii,row in enumerate(rowdata):
        H=[list(map(F,r)) for r in proposal['quadratic'][ii]];d=list(map(F,proposal['linear'][ii]));k=row['k']
        size=row['coordinate_count'];assert len(H)==size and len(d)==size
        pivots=positive_ldl(H);inv=inverse(H);cross=[d]+row['V'];S=mm(mm(cross,inv),transpose(cross));up,shift=dyadic_majorant(S)
        for a in range(dimension+1):
            for b in range(dimension+1):total[a][b]+=up[a][b]
        local_checks.append(dict(row=row['row'],minimum_ldl_pivot=float(min(pivots)),rounding_shift=float_up(shift)))
        poly=row['poly'];u=row['u'];terms=list(u)
        for a,j in enumerate(row['jj']):poly[(a,)]=poly.get((a,),F(0))+local_linear.get((row['row'],int(j)),F(0))
        linear=[d[a]+c*dot([v[a] for v in row['V']],g0) for a in range(size)]
        for a,key in enumerate(terms):
            poly[key]=poly.get(key,F(0))+2*dot(u[key],linear)
            Hu=[dot(h,u[key]) for h in H]
            for b in range(a,len(terms)):
                product=tuple(sorted(key+terms[b]));poly[product]=poly.get(product,F(0))+(1 if a==b else 2)*dot(u[terms[b]],Hu)
        polynomials.append((k,{key:v for key,v in poly.items() if v},row['residual']))
    K=[[F(0)]*(dimension+1) for _ in range(dimension+1)];K[0][0]=F(proposal['alpha'])
    for q in range(1,dimension+1):K[q][q]=1/c
    central=[[K[i][j]-total[i][j] for j in range(dimension+1)] for i in range(dimension+1)]
    pivots=positive_ldl(central)
    certificate=dict(status='PASS',local_matrices=m,local_dimensions=[r['coordinate_count'] for r in rowdata],central_dimension=dimension+1,
        minimum_central_ldl_pivot=float(min(pivots)),row_checks=local_checks,
        proof='Exact local inverse plus dyadic PSD upper approximation; exact central LDL; original cross terms retained')
    return constant,polynomials,certificate


def verify(data,proposal,check_input=True,tolerance=F(1,1000000)):
    started=time.monotonic()
    if check_input and data['problem'].get('profile_columns') is not None:
        from profile_query_v31 import audit_input as audit_profile
        inp=audit_profile(data)
    elif check_input and data['problem'].get('uniform_curvature_rule'):
        from uniform_curvature_v28 import audit as audit_uniform
        inp=audit_uniform(data)
    else:inp=audit_input(data) if check_input else None
    constant,polys,psd=reconstruct(data,proposal);upper=constant+F(proposal['alpha']);rows=[]
    for i,(k,poly,residual) in enumerate(polys):
        domain=proposal.get('residual_domains',[None]*len(polys))[i]
        value,info=edge_upper(poly,k,tolerance=tolerance,residual=None if domain is None else residual,domain=None if domain is None else list(map(F,domain)))
        upper+=value;rows.append(dict(upper=str(value),**info))
    upper/=10**6;lower=F(data['result']['lower']);target=F(data['result'].get('target',3e-6))
    if data['problem'].get('profile_columns') is None and not any(proposal.get('residual_domains',[])) and not data['problem'].get('extra_linear_inequalities'):
        assert upper>=lower
    return dict(status='EXACT_AUDIT_PASS',input=inp,psd=psd,exact_upper=str(upper),exact_lower=str(lower),upper=float_up(upper),
                lower=float(lower),gap=float_up(upper-lower),target=float(target),target_met=upper-lower<=target,
                seconds=time.monotonic()-started,rows=rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--proposal',required=True);ap.add_argument('--cpu',type=int,default=12);args=ap.parse_args()
    psutil.Process().cpu_affinity([args.cpu]);out=Path(args.proposal);source=json.loads((out/'source.json').read_text());folder=Path(source['folder'])
    root=json.loads((folder/source['receipt']).read_text());result=json.loads((folder/'result.json').read_text())
    data=dict(original=json.loads((folder/'original_input.json').read_text()),problem=root['problem'],
              witness=json.loads((folder/'witness.json').read_text()),result=result)
    try:report=verify(data,json.loads((out/'proposal.json').read_text()))
    except Exception as exc:report=dict(status='ERROR',error=str(exc),traceback=traceback.format_exc())
    (out/'exact_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('psd','rows','exact_upper','exact_lower')}),flush=True)

if __name__=='__main__':main()
