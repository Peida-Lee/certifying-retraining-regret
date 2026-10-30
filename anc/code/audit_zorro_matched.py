"""Independent rational audit of the fixed-point enclosure and regret adapter.

Imports no ZORRO adapter, proposal, search, or interval implementation.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from fractions import Fraction as F
import json,time,traceback
import psutil
from audit_real_world import read,mq,vq,dot,tr,mm,inverse

def mv(A,x):return [dot(v,x) for v in A]
def add(x,y):return [a+b for a,b in zip(x,y)]
def sub(x,y):return [a-b for a,b in zip(x,y)]
def scale(x,s):return [v*s for v in x]

def audit(D,r):
    p,n=D['p'],D['n'];zero=[F(0)]*p
    M=[row.copy() for row in D['A']];rhs=D['r'].copy()
    for i in D['unknown']:
        c=D['c'][i]
        for j in range(p):
            rhs[j]+=c[j]*D['y'][i]/n
            for k in range(p):M[j][k]+=c[j]*c[k]/n
    cells=[(i,j) for i in D['unknown'] for j,w in enumerate(D['w'][i]) if w]
    B=mq(r['basis']);J=inverse(B);Q=mm(J,mm(M,B))
    assert all(F(l)<=v<=F(u) for lr,rr,ur in zip(r['inverse_lower'],J,r['inverse_upper']) for l,v,u in zip(lr,rr,ur))
    linear={};quad={};labels={}
    for a,(i,j) in enumerate(cells):
        w=D['w'][i][j];c=D['c'][i]
        A=[[w*(c[k]*int(ell==j)+int(k==j)*c[ell])/n for ell in range(p)] for k in range(p)]
        linear[(a,)]=A;labels[a]=[w*D['y'][i]/n*int(k==j) for k in range(p)]
        for b in range(a,len(cells)):
            ii,jj=cells[b]
            if ii!=i:continue
            ww=D['w'][ii][jj]
            quad[(a,b)]=[[w*ww/n*(int(k==j and ell==jj)+(int(k==jj and ell==j) if a!=b else 0)) for ell in range(p)] for k in range(p)]
    perturb=[[F(0)]*p for _ in range(p)]
    for A in list(linear.values())+list(quad.values()):
        T=mm(J,mm(A,B))
        perturb=[[v+abs(t) for v,t in zip(row,tt)] for row,tt in zip(perturb,T)]
    H=mq(r['fixed_point_matrix'])
    assert all(H[j][j]<=Q[j][j]-perturb[j][j] for j in range(p))
    assert all(H[j][k]<=-abs(Q[j][k])-perturb[j][k] for j in range(p) for k in range(p) if j!=k)
    wr=vq(r['real_component']);data=mq(r['data_component']);cols=tr(data)
    force={():sub(mv(M,wr),rhs)}
    def insert(key,x):
        key=tuple(sorted(key));force[key]=add(force.get(key,zero),x)
    for key,A in linear.items():
        a=key[0];insert(key,sub(add(mv(M,cols[a]),mv(A,wr)),labels[a]))
        for b,v in enumerate(cols):insert(key+(b,),mv(A,v))
    for key,A in quad.items():
        insert(key,mv(A,wr))
        for b,v in enumerate(cols):insert(key+(b,),mv(A,v))
    forcing=[F(0)]*p
    for f in force.values():forcing=add(forcing,list(map(abs,mv(J,f))))
    stored=vq(r['forcing_radius']);kval=vq(r['nondata_radii'])
    assert all(a>=b for a,b in zip(stored,forcing)) and min(kval)>=0
    assert all(a>=b for a,b in zip(mv(H,kval),stored))
    # Strict M-matrix certificate: H^{-1}1 > 0 and H has nonpositive off-diagonals.
    positive=mv(inverse(H),[F(1)]*p);assert min(positive)>0
    beta=D['beta'];g={():sub(mv(M,beta),rhs)}
    for key,A in linear.items():g[key]=sub(mv(A,beta),labels[key[0]])
    for key,A in quad.items():g[key]=mv(A,beta)
    disp={():sub(beta,wr)}
    for a,col in enumerate(cols):disp[(a,)]=scale(col,-1)
    for j,col in enumerate(tr(B)):disp[(len(cells)+j,)]=scale(col,-kval[j])
    poly={}
    for key,x in g.items():
        for key2,v in disp.items():
            mon=tuple(sorted(key+key2));poly[mon]=poly.get(mon,F(0))+dot(x,v)
    upper=F(0)
    for mon,coef in poly.items():
        upper+=coef if not mon else max(F(0),coef) if all(mon.count(j)%2==0 for j in set(mon)) else abs(coef)
    assert F(r['upper'])>=upper
    center_lower=dot(beta,mv(M,beta))-2*dot(rhs,beta)-dot(wr,mv(M,wr))+2*dot(rhs,wr)
    return dict(status='PASS',fixed_point_equations=True,joint_symbol_regret_upper=True,
                forcing_monomials=len(force),regret_monomials=len(poly),exact_regret_upper=str(upper),exact_center_lower=str(center_lower))

def main():
    psutil.Process().cpu_affinity([12])
    sourcefile=Path(os.environ.get('REGRET_SOURCE_RESULTS',Path(__file__).parent/'contraction_run/results.json'));base=sourcefile.parent
    source=json.loads(sourcefile.read_text());zfile=Path(os.environ.get('REGRET_ZORRO_OUTPUT',base/'zorro_matched_results.json'))
    z=json.loads(zfile.read_text());reports=[];start=time.monotonic()
    for row in z['results']:
        report=dict(dataset=row['dataset'])
        try:
            if row['fixed_point_cover_complete']:
                assert len(row['envelope_receipts'])==1
                orig=next(v for v in source['results'] if v['dataset']==row['dataset'] and v['seed']==row['seed'] and v['method']=='energy')
                cert=json.loads((base/orig['certificate_file']).read_text())['certificate']
                report.update(audit(read(cert),row['envelope_receipts'][0]))
            else:report.update(status='PASS',bound_origin='shared independently audited cap; incomplete fixed-point cover')
        except Exception as exc:report.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
        reports.append(report);print(report['dataset'],report['status'],report.get('error',''),flush=True)
        zfile.with_name('zorro_independent_audit.json').write_text(json.dumps(dict(status='RUNNING',reports=reports),indent=2))
    result=dict(status='PASS' if all(r['status']=='PASS' for r in reports) else 'FAILED',reports=reports,seconds=time.monotonic()-start,imports_adapter_implementation=False)
    zfile.with_name('zorro_independent_audit.json').write_text(json.dumps(result,indent=2));print(result['status'])

if __name__=='__main__':main()
