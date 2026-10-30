"""Independent Fraction reconstruction of the contraction certificate.

Imports only the independent arithmetic auditor, not the solver or verifier.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import json,time,itertools,traceback
import numpy as np
import psutil
from math import isqrt
from audit_real_world import (read,dot,quad,mm,tr,mq,vq,inverse,root,prod_range,
                             audit_constants,audit_witness,audit_tree,sqrtcheck)


def ceilq(x,bits=24):
    q=x*2**bits
    return F((q.numerator+q.denominator-1)//q.denominator,2**bits)


def fine_root(q,bits=128):
    """Independent upward rational root for comparing float-enclosed constants."""
    v=isqrt(q.numerator*2**(2*bits)//q.denominator)
    z=F(v,2**bits)
    return z if z*z>=q else z+F(1,2**bits)


def audit_centered_moment(D,cert,T,R,Hi,reference):
    p=D['p'];n=D['n']
    proj=mm(mm(R,Hi),tr(R))
    P=[[F(j==k)-v for k,v in enumerate(row)] for j,row in enumerate(proj)]
    TP=mm(T,P);TR=mm(T,R)
    g0=[dot(row,reference)-v for row,v in zip(D['A'],D['r'])]
    gl=[dot(col,g0) for col in tr(TP)];gu=gl.copy()
    ct=mm([D['c'][i] for i in D['unknown']],T)
    cp=mm(ct,P);cr=mm(ct,R)
    for i in D['unknown']:
        row=D['c'][i];js=[j for j,w in enumerate(D['w'][i]) if w]
        ec=dot(row,reference)-D['y'][i]
        eg=[D['w'][i][j]*reference[j] for j in js]
        for j,col in enumerate(tr(TP)):
            l,u=prod_range(dot(row,col),[D['w'][i][k]*col[k] for k in js],ec,eg)
            gl[j]+=l/n;gu[j]+=u/n
    rec=cert['centered_moment_receipt'];gn=F(rec['gradient_norm'])
    assert gn>=0 and gn*gn>=sum((max(abs(l),abs(u))**2 for l,u in zip(gl,gu)),F(0))
    for l,u,sl,su in zip(gl,gu,rec['gradient_lo'],rec['gradient_hi']):
        assert F(sl)<=l<=u<=F(su)
    # P T^T M_mid T R: observed anchor plus exact midpoint unknown rows.
    anchor=mm(tr(TP),mm(D['A'],TR))
    cross=[[v+sum((a[j]*b[k] for a,b in zip(cp,cr)),F(0))/n for k,v in enumerate(row)] for j,row in enumerate(anchor)]
    norm=lambda v:fine_root(dot(v,v))
    center=norm([v for row in cross for v in row])
    gp=[norm(row) for row in TP];gr=[norm(row) for row in TR]
    variation=F(0)
    for idx,i in enumerate(D['unknown']):
        dp=sum((w*g for w,g in zip(D['w'][i],gp)),F(0))
        dr=sum((w*g for w,g in zip(D['w'][i],gr)),F(0))
        variation+=(dp*norm(cr[idx])+norm(cp[idx])*dr+dp*dr)/n
    coupling=F(rec['coupling'])
    assert coupling>=center+variation
    return gn,coupling


def audit(D,cert):
    rec=cert['energy_receipt'];f={k:F(v) for k,v in rec['fractions'].items()}
    a,eps,G,d,im,mu,c=[f[k] for k in ('a','epsilon','G','d','inverse_sqrt_mu','mu','c')]
    oldR=(G+a*eps+(c/mu+a*im/2)*d+c*eps*im/2)/mu
    oldeta=(eps+a*oldR+(d+c*oldR)*im)**2
    # Reuse the independent original-row reconstruction for unchanged constants.
    oldf=f.copy();oldf.update(retained_radius=oldR,eta=oldeta,coordinate_radius=ceilq(root(oldR**2/f['gram_minimum'])))
    oldrec=rec.copy();oldrec.update(fractions={k:str(v) for k,v in oldf.items()},tail_target_met=oldeta<=F(1,400000))
    oldcert=cert.copy();oldcert.update(energy_receipt=oldrec,tail_error=str(oldeta),
        initial_upper=float(np.nextafter(float(F(cert['original_gradient_upper'])+oldeta),np.inf)))
    C=audit_constants(D,oldcert)
    T=mq(cert['T']);R=mq(cert['R']);U=mm(T,R);cols=tr(R);ucols=tr(U)
    Hi=inverse(mm(tr(R),R));n=D['n'];r=len(cols)
    ref=vq(rec['reference_beta'])
    trace=delta=F(0);centers=[];mixed_l=[F(0)]*r;mixed_u=[F(0)]*r
    nuisance=[i for i in D['unknown'] if i not in rec['signal_rows']]
    assert nuisance==rec['nuisance_rows']
    for i,receipt in zip(nuisance,rec['contraction_rows']):
        row=D['c'][i];js=[j for j,v in enumerate(D['w'][i]) if v];ws=[D['w'][i][j] for j in js]
        tc=[dot(row,col) for col in tr(T)];rc=[dot(tc,col) for col in cols]
        centers.append((tc,rc));vals=[];dvals=[]
        for signs in itertools.product((-1,1),repeat=len(js)):
            x=row.copy()
            for j,w,s in zip(js,ws,signs):x[j]+=w*s
            tx=[dot(x,col) for col in tr(T)];rx=[dot(tx,col) for col in cols]
            v=dot(tx,tx)-quad(rx,Hi);vals.append(v)
            dt=[v-c0 for v,c0 in zip(tx,tc)];dr=[v-c0 for v,c0 in zip(rx,rc)]
            dvals.append(dot(dt,dt)-quad(dr,Hi))
        assert vals==vq(receipt['discarded_vertex_norms_squared']) and max(vals)==F(receipt['discarded_maximum'])
        trace+=max(vals)/n;delta+=max(dvals)/n
        ec=dot(row,ref)-D['y'][i];eg=[w*ref[j] for j,w in zip(js,ws)]
        pc=[dot(row,col) for col in ucols];pg=[[w*col[j] for j,w in zip(js,ws)] for col in ucols]
        ranges=[prod_range(pc[j],pg[j],ec,eg) for j in range(r)]
        assert ranges==[tuple(vq(v)) for v in receipt['mixed_ranges']]
        for j,(l,u) in enumerate(ranges):mixed_l[j]+=l/n;mixed_u[j]+=u/n
    assert trace==f['discarded_trace'] and delta==f['discarded_delta_trace']
    cg=[[(dot(x,y)-dot(xr,[dot(h,yr) for h in Hi]))/n for y,yr in centers] for x,xr in centers]
    test=mq(rec['discarded_center_test']);tg=mm(tr(test),test);kg=mm(tr(test),mm(cg,test))
    gm=min(tg[j][j]-sum((abs(v) for k,v in enumerate(tg[j]) if k!=j),F(0)) for j in range(len(tg)))
    ku=max(kg[j][j]+sum((abs(v) for k,v in enumerate(kg[j]) if k!=j),F(0)) for j in range(len(kg)))
    assert gm>0 and f['discarded_center_operator']>=ku/gm>=0
    operator=(root(f['discarded_center_operator'])+root(delta))**2
    assert operator==f['discarded_operator'] and min(operator,trace)==f['discarded_energy']
    rho=f['discarded_energy'];gamma2=rho/(mu+rho)
    assert gamma2==f['contraction_squared'];sqrtcheck(f['contraction'],gamma2)
    b2=max(quad(list(z),Hi) for z in itertools.product(*zip(mixed_l,mixed_u)))
    assert b2==f['mixed_norm_squared'];sqrtcheck(f['mixed_norm'],b2)
    tt=min(im/2,root(rho)/mu)
    assert tt==f['inverse_cross_norm']
    radius=min(oldR,(G+min(a*eps,f['mixed_norm']+a*gamma2*eps)+(c/mu+a*tt)*d+c*tt*eps)/mu)
    assert radius==f['retained_radius']
    gr=[dot(row,ref)-v for row,v in zip(D['A'],D['r'])]
    gl=[dot(col,gr) for col in tr(T)];gu=gl.copy()
    for i in D['unknown']:
        row=D['c'][i];js=[j for j,v in enumerate(D['w'][i]) if v]
        ec=dot(row,ref)-D['y'][i];eg=[D['w'][i][j]*ref[j] for j in js]
        for j,col in enumerate(tr(T)):
            l,u=prod_range(dot(row,col),[D['w'][i][k]*col[k] for k in js],ec,eg)
            gl[j]+=l/n;gu[j]+=u/n
    exact_fullg2=sum((max(abs(l),abs(u))**2 for l,u in zip(gl,gu)),F(0))
    fullg2=F(cert['centered_moment_receipt']['full_gradient_squared'])
    assert fullg2>=exact_fullg2
    radius=min(radius,root(fullg2)/mu)
    cr=cert['contraction_receipt']
    assert radius==F(cr['retained_radius'])
    coord=ceilq(root(radius*radius/f['gram_minimum']))
    assert coord==F(cr['coordinate_radius'])==F(cert['coordinate_radius'])
    e2=min((eps+a*radius)**2,f['epsilon2']+2*f['mixed_norm']*radius+f['a2']*radius**2)
    eta=(f['contraction']*root(e2)+(d+c*radius)*im)**2
    assert eta==F(cr['tail'])
    if 'used_tail' in cr:
        gn,coupling=audit_centered_moment(D,cert,T,R,Hi,ref)
        moment=(gn+coupling*coord)**2/mu
        assert moment==F(cr['common_radius_moment_tail'])
        assert eta==F(cr['contraction_only_tail'])
        eta=min(eta,moment)
        assert eta==F(cr['used_tail'])
    assert eta==F(cert['tail_error'])
    C.update(eta=eta,radius=coord)
    return C


def main():
    psutil.Process().cpu_affinity([12])
    output=Path(os.environ.get('REGRET_OUTPUT',Path(__file__).resolve().parent/'contraction_run'))
    source=json.loads((output/'results.json').read_text());assert source['status']=='PASS'
    reports=[];start=time.monotonic();seeds=source.get('settings',{}).get('mask_seeds',source.get('plan',{}).get('seeds',[88211]))
    for row in source['results']:
        if row['method'] not in ('energy','adaptive_energy'):continue
        report={k:row.get(k) for k in ('dataset','seed','origin','row_rule','candidate_bits')}
        try:
            result=json.loads((output/row['certificate_file']).read_text());cert=result['certificate'];D=read(cert)
            C=audit(D,cert);audit_witness(D,cert,result)
            report.update(status='PASS',constants='PASS',feasible_original_witness='PASS',tail=str(C['eta']))
            if row['seed']==seeds[0] and row.get('origin','primary') in ('primary','independent_uniform'):report['complete_tree']=audit_tree(D,cert,result,C)
        except Exception as e:
            report.update(status='ERROR',error=str(e),traceback=traceback.format_exc())
        reports.append(report)
        (output/'independent_audit.json').write_text(json.dumps(dict(status='RUNNING',reports=reports,seconds=time.monotonic()-start),indent=2))
        print(row['dataset'],row['seed'],report['status'],report.get('error',''),flush=True)
    expected=sum(r['method'] in ('energy','adaptive_energy') for r in source['results'])
    report=dict(status='PASS' if len(reports)==expected and all(r['status']=='PASS' for r in reports) else 'FAILED',
                conditions=len(reports),full_trees=sum('complete_tree' in r for r in reports),reports=reports,
                seconds=time.monotonic()-start,imports_solver_implementation=False)
    (output/'independent_audit.json').write_text(json.dumps(report,indent=2));print(report['status'],flush=True)

if __name__=='__main__':main()
