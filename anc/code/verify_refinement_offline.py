"""Regenerate a small synthetic degree-four certificate and audit it exactly."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from fractions import Fraction
import json,tempfile
import numpy as np
from observed_anchor import ObservedAnchor
from local_residual import LocalResidualSearch
from polynomial_majorant import certificate
from sdp_residual import solve
from audit_sdp_exact import audit,denominator,int_array
from aggregate_split import constraints
from compressed_polynomial import build as compressed_build
from audit_compressed_exact import audit as compressed_audit
from localized_moment import solve as localized_solve
from halfspace_moment import solve as halfspace_solve
B=Path(__file__).resolve().parent

def verify():
    # Averaging binary input values can produce incomparable odd denominators.
    mixed=[[Fraction(1,6),Fraction(1,10),Fraction(-1,4)]]
    common=denominator(v for row in mixed for v in row)
    assert common==60 and int_array(mixed,common).tolist()==[[10,6,-15]]
    rejected_scale=False
    try:int_array(mixed,10)
    except AssertionError:rejected_scale=True
    assert rejected_scale
    lo=np.array([[1.,.125,-.25],[1.,.5,.125],[1.,-.5,.25],[1.,.25,.5]])
    hi=lo.copy();hi[0,1]+=.125;hi[0,2]+=.25;hi[1,1]+=.25
    y=np.array([.125,.5,-.25,.25]);beta=np.array([.125,.25,-.125]);lam=.25
    a=ObservedAnchor(lo,hi,y,beta,lam);s=LocalResidualSearch(a)
    s.lower(lo,s.fit(lo));theta=s.fit((lo+hi)/2)
    source=certificate(s,lo,hi,lo,theta,build_only=True)
    source.update(domain_lo=lo.tolist(),domain_hi=hi.tolist())
    original=dict(lo=lo.tolist(),hi=hi.tolist(),y=y.tolist(),beta=beta.tolist(),lambda_value=lam)
    reports=[]
    with tempfile.TemporaryDirectory(prefix='refinement_check_',dir=B) as tmp:
        for case in ('whole_box','conditional_halfspace'):
            current=dict(source)
            if case=='conditional_halfspace':
                current['extra_linear_inequalities']=constraints(np.array([1.,-.5,.25]),.5,0)
            result=solve(current,time_limit=5)
            result.update(dataset='synthetic_offline',seed=92001,lower=s.best,absolute_gap=result['upper']-s.best)
            file=Path(tmp)/f'{case}_receipt.json'
            file.write_text(json.dumps(dict(result=result,problem=current,witness=s.witness)))
            reports.append(audit(file,original))
        base=dict(theta=theta.tolist(),T=a.t.tolist(),mu=a.mu,active_cells=s.ids.tolist(),
                  x0_uncertain=[float(lo[i,j]) for i,j in s.ids])
        for case in ('lifted_residual_box','lifted_residual_halfspace'):
            current=compressed_build(original,base)
            if case.endswith('halfspace'):
                cuts=constraints(np.array([1.,-.5,.25]),.5,0)
                current['domain_inequalities']=cuts
                current['extra_linear_inequalities']=current['residual_link_inequalities']+cuts
            result=solve(current,time_limit=5)
            result.update(dataset='synthetic_offline',seed=92001,lower=s.best,absolute_gap=result['upper']-s.best)
            file=Path(tmp)/f'{case}_receipt.json'
            file.write_text(json.dumps(dict(result=result,problem=current,witness=s.witness)))
            reports.append(compressed_audit(file,original))
        for case in ('localized_box','localized_halfspace','localized_psd_halfspace_left','localized_psd_halfspace_right'):
            current=compressed_build(original,base)
            if 'halfspace' in case:
                cuts=constraints(np.array([1.,-.5,.25]),.5,int(case.endswith('right')))
                current['domain_inequalities']=cuts
                current['extra_linear_inequalities']=current['residual_link_inequalities']+cuts
                if '_psd_' in case:
                    nv=len(current['active_cells'])+len(current['residual_left'])
                    current['additional_moment_blocks']=[dict(kind='halfspace_localizer',domain_inequality_index=0,basis=[[]]+[[j] for j in range(nv)],polynomial=[[[],cuts[0]['rhs']]]+[[key,-v] for key,v in cuts[0]['terms']])]
            result=localized_solve(current,seconds=5)
            coefficients={tuple(k):(l,u) for k,l,u in zip(current['monomials'],current['coefficient_lower'],current['objective'])}
            current['monomials']=result['monomials']
            current['coefficient_lower']=[coefficients.get(tuple(k),(0.,0.))[0] for k in current['monomials']]
            current['objective']=[coefficients.get(tuple(k),(0.,0.))[1] for k in current['monomials']]
            result.update(dataset='synthetic_offline',seed=92001,lower=s.best,absolute_gap=result['upper']-s.best)
            file=Path(tmp)/f'{case}_receipt.json'
            file.write_text(json.dumps(dict(result=result,problem=current,witness=s.witness)))
            reports.append(compressed_audit(file,original))
        for case,side in [('original_row_halfspace_left',0),('original_row_halfspace_right',1)]:
            current=dict(source)
            current['extra_linear_inequalities']=constraints(np.array([1.,-.5,.25]),.5,side)
            current,result=halfspace_solve(current,seconds=5)
            result.update(dataset='synthetic_offline',seed=92001,lower=s.best,
                          absolute_gap=result['upper']-s.best,target=1e-5)
            file=Path(tmp)/f'{case}_receipt.json'
            file.write_text(json.dumps(dict(result=result,problem=current,witness=s.witness)))
            reports.append(audit(file,original))
            assert reports[-1]['target_met']==(result['absolute_gap']<=1e-5)
        # The checker must reject a PSD localizer with an incorrect slack.
        payload=json.loads(file.read_text())
        payload['result']['moment_blocks'][1]['polynomial'][0][1]+=1.
        wrong=Path(tmp)/'invalid_halfspace_receipt.json';wrong.write_text(json.dumps(payload))
        rejected=False
        try:audit(wrong,original)
        except AssertionError:rejected=True
        assert rejected
    return dict(status='PASS',synthetic_polynomials=len(reports),original_rows=True,
                conditional_RLT=True,lifted_residual_links=True,rowwise_box_localizers=True,
                halfspace_PSD_localizers=True,invalid_halfspace_rejected=True,
                study_target_labels_checked=True,exact_factor_and_dual=True,
                mixed_denominator_scaling=True,nonintegral_scaling_rejected=True,raw_records_opened=False)

if __name__=='__main__':print(json.dumps(verify()))
