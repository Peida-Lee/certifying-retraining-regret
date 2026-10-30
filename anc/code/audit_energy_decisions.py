"""Independent exact energy, midpoint and decision checks on disjoint pools."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from fractions import Fraction as F
import json,time,traceback
import numpy as np,psutil
from audit_contraction_real import audit,read,audit_witness,audit_tree
from audit_real_world import dot,quad,mq,vq,mm,tr

def midpoint(D,cert):
    n,p=D['n'],D['p'];xx=np.array([[int(v*1024) for v in row] for row in D['c']],dtype=np.int64)
    yy=np.array([int(v*1024) for v in D['y']],dtype=np.int64)
    assert n*int(abs(xx).max())**2<2**63 and n*int(abs(xx).max())*int(abs(yy).max())<2**63
    assert all(F(int(a),1024)==b for row,original in zip(xx,D['c']) for a,b in zip(row,original))
    gram=xx.T@xx;resp=xx.T@yy;den=n*1024**2
    M=[[F(int(gram[j,k]),den)+D['ridge']*int(j==k) for k in range(p)] for j in range(p)]
    rhs=[F(int(v),den) for v in resp]
    theta=vq(np.linalg.solve(np.array(M,float),np.array(rhs,float)));beta=D['beta']
    diff=quad(beta,M)-quad(theta,M)-2*dot(rhs,[b-t for b,t in zip(beta,theta)])
    residual=[dot(row,theta)-v for row,v in zip(M,rhs)]
    transformed=[dot(col,residual) for col in tr(mq(cert['T']))]
    correction=dot(transformed,transformed)/F(cert['mu'])
    return diff,diff+correction

def classify(row,meta,alpha):
    factor=1+F(str(alpha));lower=F(row.get('budget_lower',row['lower']));upper=F(row.get('budget_upper',row['upper']))
    if upper<=factor*F(meta['midpoint_lower']):return 'STOP'
    if lower>factor*F(meta['midpoint_upper']):return 'VIOLATION'
    return 'UNRESOLVED'

def main():
    psutil.Process().cpu_affinity([int(os.environ.get('REGRET_AUDIT_CPU','12'))]);output=Path(os.environ.get('REGRET_DECISION_OUTPUT',Path(__file__).parent/'energy_decisions'))
    saved=json.loads((output/'results.json').read_text());assert saved['status']=='PASS'
    alpha=saved['settings']['primary_increment_fraction'];first=saved['settings']['mask_seeds'][0]
    reports=[];started=time.monotonic();frozen_beta=None
    for meta in saved['conditions']:
        report=dict(pool=meta['pool'],seed=meta['seed'])
        try:
            arow=next(r for r in saved['results'] if r['pool']==meta['pool'] and r['seed']==meta['seed'] and r['method']=='adaptive_energy')
            crow=next(r for r in saved['results'] if r['pool']==meta['pool'] and r['seed']==meta['seed'] and r['method']=='contraction_energy')
            ar=json.loads((output/arow['certificate_file']).read_text());ac=ar['certificate'];D=read(ac)
            C=audit(D,ac);audit_witness(D,ac,ar)
            if frozen_beta is None:frozen_beta=D['beta']
            else:assert D['beta']==frozen_beta
            ml,mu=midpoint(D,ac)
            assert F(meta['midpoint_lower'])<=ml<=mu<=F(meta['midpoint_upper'])
            if meta['seed']==first:report['adaptive_complete_tree']=audit_tree(D,ac,ar,C)
            cr=json.loads((output/crow['certificate_file']).read_text());cc=cr['certificate']
            shared=('lo','hi','y','beta','lambda_value','T','mu','nu','R','hdiag','original_gradient_upper','coordinate_radius','center','energy_receipt','centered_moment_receipt')
            assert all(ac[k]==cc[k] for k in shared)
            assert F(cc['tail_error'])==F(ac['contraction_receipt']['tail'])
            audit_witness(D,cc,cr)
            if meta['seed']==first:
                if cc['tail_error']==ac['tail_error'] and cc['nodes']==ac['nodes'] and cc['points']==ac['points']:
                    report['pure_energy_tree']='identical mathematical tree and tail to audited adaptive tree'
                else:
                    Cpure=dict(C);Cpure['eta']=F(cc['tail_error']);report['pure_energy_complete_tree']=audit_tree(D,cc,cr,Cpure)
            for row in (r for r in saved['results'] if r['pool']==meta['pool'] and r['seed']==meta['seed']):
                assert classify(row,meta,alpha)==row['decision']
                assert row['saved_cells']==(33 if row['decision']=='STOP' else 0)
                if row['method'] in ('adaptive_energy','contraction_energy'):
                    assert F(row['budget_lower'])<=F(row['lower'])<=F(row['upper'])<=F(row['budget_upper'])
            report.update(status='PASS',adaptive_constants=True,pure_energy_constants_shared_and_verified=True,
                both_original_witnesses=True,midpoint_interval=True,frozen_candidate=True,exact_decision_rule=True)
        except Exception as exc:report.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
        reports.append(report);print(report['pool'],report['seed'],report['status'],report.get('error',''),flush=True)
        (output/'independent_audit.json').write_text(json.dumps(dict(status='RUNNING',reports=reports),indent=2))
    provenance=json.loads((output/'pool_provenance_local.json').read_text());ids=provenance['pool_ids']
    assert all(not set(ids[i]).intersection(ids[j]) for i in range(len(ids)) for j in range(i))
    result=dict(status='PASS' if len(reports)==15 and all(r['status']=='PASS' for r in reports) else 'FAILED',
        conditions=len(reports),core_energy_receipts=30,original_witnesses=30,adaptive_complete_trees=3,
        pure_energy_mathematical_tree_checks=3,disjoint_pools=True,exact_relative_tolerance='1/50',reports=reports,
        seconds=time.monotonic()-started,imports_solver_implementation=False)
    (output/'independent_audit.json').write_text(json.dumps(result,indent=2));print(result['status'])

if __name__=='__main__':main()
