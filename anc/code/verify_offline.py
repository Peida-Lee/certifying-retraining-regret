"""Exact checks of the stated mathematics, public summaries, and portable imports."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import csv,importlib,json
from audit_real_world import dot,quad,inverse,root

B=Path(__file__).resolve().parent


def main():
    import psutil
    allowed=psutil.Process().cpu_affinity()
    psutil.Process().cpu_affinity([10] if 10 in allowed else [allowed[-1]])
    imports=['interval_arithmetic','discounted_planar_oracle','dual_certificate',
        'rank_certificate','automatic_basis','observed_anchor','moment_tail',
        'residual_energy','epsilon_energy','leakage_energy','auto_energy',
        'anchor_energy','centered_signal_energy','planar_oracle','greedy_basis',
        'metric_basis','datasets','real_world_design','run_real_world','audit_real_world',
        'contraction_energy','audit_contraction_real','zorro_regret','audit_zorro_matched',
        'zorro_refined','fixed_point_contract','run_energy_decisions','audit_energy_decisions',
        'run_zorro_refined','run_independent_masks','audit_zorro_refined']
    for name in imports:importlib.import_module(name)
    K=[[F(i==j)+F(abs(i-j)==2,64) for j in range(4)] for i in range(4)]
    mu=F(63,64);nu=F(65,64);cc=(nu-mu)/2
    h=[F(1,64),-F(1,128),F(1,2**18),-F(1,2**18)]
    cases=0
    for scale in (F(1,16),F(1),F(32),F(1024)):
      for noise in (F(0),F(1,1024),F(1,32)):
       for z0 in ([F(0)]*4,[F(1,8),-F(1,16),F(0),F(0)]):
        rows=[[F(1),F(0),F(0),F(0)],[F(1,32),F(0),scale,F(0)],[F(0),F(1,32),F(0),scale]]
        res=[F(1,4),noise,-noise]
        M=[[K[i][j]+sum((v[i]*v[j] for v in rows),F(0)) for j in range(4)] for i in range(4)]
        g=[h[i]+sum((v[i]*e for v,e in zip(rows,res)),F(0)) for i in range(4)]
        hc=[h[i]-dot(K[i],z0) for i in range(4)];ec=[e-dot(v,z0) for v,e in zip(rows,res)]
        a=F(1,32);eps=root(ec[1]**2+ec[2]**2);G=root((hc[0]+ec[0])**2+hc[1]**2)
        d=root(hc[2]**2+hc[3]**2);im=root(1/mu)
        R=(G+a*eps+(cc/mu+a*im/2)*d+cc*eps*im/2)/mu
        eta=(eps+a*R+(d+cc*R)*im)**2
        disp=[dot(v,g) for v in inverse(M)];full=dot(g,disp)
        small=[[M[i][j] for j in range(2)] for i in range(2)];restricted=quad(g[:2],inverse(small))
        assert 0<=full-restricted<=eta
        assert sum(((disp[i]-z0[i])**2 for i in range(2)),F(0))<=R**2
        cases+=1
    # Positive-ridge finite-box interior counterexample, using original normal equations.
    beta=[F(1,3),F(0)];lam=F(1,3);values=[]
    for t in (-1,0,1):
        X=[[F(t),F(0)],[F(1),F(0)],[F(0),F(1)]];y=[F(0),F(1),F(0)]
        M=[[sum((x[i]*x[j]/3 for x in X),F(0))+lam*(i==j) for j in range(2)] for i in range(2)]
        rr=[sum((x[i]*z/3 for x,z in zip(X,y)),F(0)) for i in range(2)]
        g=[dot(M[i],beta)-rr[i] for i in range(2)];values.append(quad(g,inverse(M)))
    assert values==[F(0),F(1,54),F(0)]
    result_root=B.parent/'results'
    agg=json.loads((result_root/'real_world/public_energy_aggregate.json').read_text())
    assert len(agg)==4 and sum(r['methods']['energy']['successes'] for r in agg)==20
    assert sum(r['tail_successes'] for r in agg)==12
    audit=json.loads((result_root/'real_world/independent_audit_results.json').read_text())
    assert audit['status']=='PASS' and audit['conditions']==20 and audit['full_trees']==4
    with (result_root/'precision_and_scip/precision_row_summary.csv').open(newline='') as f:
        precision=list(csv.DictReader(f))
    assert len(precision)==48 and all(r['status']=='PASS' for r in precision)
    scip=json.loads((result_root/'precision_and_scip/SCIP_FOLLOWUP_REPORT.json').read_text())
    assert scip['status']=='PASS' and scip['runs']==12
    # Check the scoped dense-row adapter on a row with more than six unknowns.
    import numpy as np
    from planar_oracle import PlanarObservedAnchor,exact_row_max
    lo=np.vstack([np.r_[1.,np.zeros(8)],np.r_[1.,np.ones(8)]])
    hi=lo.copy();hi[0,1:]=1.
    anchored=PlanarObservedAnchor(lo,hi,np.array([.2,.3]),np.zeros(9),1/32)
    assert anchored.k==8
    # A full-column-rank but non-diagonally-dominant Gram must be certifiable.
    from observed_anchor import ObservedAnchor
    from moment_tail import tail_receipt
    small_lo=np.array([[1.,0.,.25],[1.,.5,-.25]])
    small_hi=small_lo.copy();small_hi[0,1]=.25
    small_anchor=ObservedAnchor(small_lo,small_hi,np.array([.1,.3]),np.array([.125,0.,0.]),1/32)
    nonorthogonal=np.array([[1.,2.],[0.,1.],[0.,0.]])
    gram_check=tail_receipt(small_anchor,nonorthogonal)
    assert gram_check['gram_lower_method']=='exact_inverse_norm'
    assert 0<F(gram_check['Hmu'])<=F(1,7)
    try:
        tail_receipt(small_anchor,np.array([[1.,2.],[0.,0.],[0.,0.]]))
    except (ValueError,ZeroDivisionError,StopIteration):
        pass
    else:
        raise AssertionError('A singular retained Gram was accepted')
    for k in (1,2,4,8):
        generators=[(F(j+1,16),F((-1)**j*(j+2),32)) for j in range(k)]
        value,w=exact_row_max(F(1,8),F(-1,4),generators,F(1,2),True)
        assert len(w)==k and all(-1<=s<=1 for s in w)
        assert value==(F(1,8)+sum((a*s for (a,b),s in zip(generators,w)),F(0)))**2-F(1,2)*(F(-1,4)+sum((b*s for (a,b),s in zip(generators,w)),F(0)))**2
    from audit_curvature_controls import audit as curvature_audit
    curvature=curvature_audit();assert curvature['status']=='PASS'
    automatic=json.loads((result_root/'energy_mechanism/auto_energy_audit_results.json').read_text())
    assert len(automatic['reports'])==6
    assert all(r['complete_binary_coverage_verified'] and r['every_node_upper_verified'] and r['original_full_witness_verified'] for r in automatic['reports'])
    paper_root=next(p for p in (B.parent/'paper/sections',B.parent.parent/'sections') if p.exists())
    width_source=(paper_root/'dimension_v78.tex').read_text()
    width_definition=r'\delta_r(\mathcal K)=\inf_{\dim S=r}\sup_{u\in\mathcal K}\operatorname{dist}_2(u,S)'
    assert width_definition in width_source and width_definition+'^2' not in width_source
    from verify_contraction_offline import verify as verify_contraction
    contraction=verify_contraction()
    current=json.loads((result_root/'contraction_real/aggregate.json').read_text())
    current_audit=json.loads((result_root/'contraction_real/independent_audit.json').read_text())
    assert len(current)==4 and all(a['median_paired_reduction']>0 for a in current)
    assert sum(a['methods']['energy']['successes'] for a in current)==20
    assert current_audit['status']=='PASS' and current_audit['conditions']==20 and current_audit['full_trees']==4
    matched=json.loads((result_root/'matched_controls/aggregate.json').read_text())
    matched_audit=json.loads((result_root/'matched_controls/independent_audit.json').read_text())
    assert matched['status']=='PASS' and matched['runs']==48 and matched['new_runs']==40
    assert {r['dataset'] for r in matched['conditions']}=={r['dataset'] for r in current}
    assert matched_audit['status']=='PASS' and matched_audit['energy_conditions']==24
    assert sum(a['energy_narrower_pairs'] for a in current)==19
    assert sum(a['equal_pairs'] for a in current)==1
    adaptive_audit=json.loads((result_root/'contraction_real/adaptive_independent_audit.json').read_text())
    assert adaptive_audit['status']=='PASS' and adaptive_audit['conditions']==40 and adaptive_audit['full_trees']==4
    with (result_root/'contraction_real/summary.csv').open(newline='') as f:current_rows=list(csv.DictReader(f))
    for row in current_rows:
        if row['method']=='energy':
            assert float(row['tail_error'])<=float(row['centered_moment_tail'])+1e-18
    external=result_root/'matched_external'
    scip_matched=json.loads((external/'scip_strict_results.json').read_text())
    assert scip_matched['status']=='PASS' and len(scip_matched['results'])==4
    assert all(r['same_saved_instance'] and r['search_tolerance']==1e-10 for r in scip_matched['results'])
    zorro_audit=json.loads((external/'zorro_independent_audit.json').read_text())
    assert zorro_audit['status']=='PASS' and sum(r.get('fixed_point_equations',False) for r in zorro_audit['reports'])==3
    midpoint=json.loads((external/'midpoint_increments.json').read_text())
    assert len(midpoint['conditions'])==20
    for r in midpoint['conditions']:
        assert 0<=r['midpoint_lower']<=r['midpoint_upper']
        assert 0<=r['increment_lower']<=r['increment_upper']
        assert r['midpoint_upper']-r['midpoint_lower']<1e-12
    uniform=json.loads((result_root/'independent_masks/aggregate.json').read_text())
    uniform_audit=json.loads((result_root/'independent_masks/independent_audit.json').read_text())
    assert uniform['status']=='PASS' and len(uniform['datasets'])==4
    assert all(v['adaptive_median_width']<v['moment_median_width'] for v in uniform['datasets'])
    assert sum(v['strict_pairs'] for v in uniform['datasets'])==14 and sum(v['equal_pairs'] for v in uniform['datasets'])==6
    assert uniform_audit['status']=='PASS' and uniform_audit['conditions']==20 and uniform_audit['full_trees']==4
    all_scip=json.loads((external/'scip_all_masks.json').read_text())
    assert all_scip['status']=='PASS' and len(all_scip['results'])==20
    refined_audit=json.loads((external/'zorro_refined_independent_audit.json').read_text())
    assert refined_audit['status']=='PASS' and all(v['complete_partition'] and v['all_parent_bounds_valid'] for v in refined_audit['reports'])
    assert sum(v['verified_model_bounds'] for v in refined_audit['reports'])==80
    decisions=result_root/'energy_decisions'
    operational=json.loads((decisions/'aggregate.json').read_text());op_audit=json.loads((decisions/'independent_audit.json').read_text())
    by_method={v['method']:v for v in operational['methods']}
    assert operational['conditions']==15 and operational['pools']==3
    assert by_method['contraction_energy']['stop']==by_method['adaptive_energy']['stop']==13
    assert by_method['moment_centered']['stop']==5 and by_method['coarse']['stop']==0
    assert by_method['contraction_energy']['saved_cells']-by_method['moment_centered']['saved_cells']==264
    assert op_audit['status']=='PASS' and op_audit['conditions']==15 and op_audit['core_energy_receipts']==30
    assert op_audit['adaptive_complete_trees']==op_audit['pure_energy_mathematical_tree_checks']==3
    assert op_audit['disjoint_pools'] and op_audit['exact_relative_tolerance']=='1/50'
    formats=json.loads((decisions/'primary_fixed_point_contract.json').read_text())
    assert formats['status']=='PASS' and formats['conditions']==20
    from fixed_point_contract import verify as verify_fixed_point
    toy=np.array([[1.,.25,-.5],[1.,-.125,.75]])
    contract=verify_fixed_point(dict(lo=toy.tolist(),hi=toy.tolist(),beta=[.125,-.0625,.5]))
    assert contract['exact_decoded_predictions']
    refinement=json.loads((result_root/'refinement_v92/audit_summary.json').read_text())
    for family,count in (('frozen_independent',20),('frozen_validation',20),('frozen_controls',40),('frozen_decisions',15)):
        report=refinement[family]
        assert report['conditions']==report['target_successes']==count
        assert all(v['status']=='PASS' and v['failures']==0 for v in report['audits'])
    assert refinement['multirank_exact']['target_successes']==18
    from run_adaptive_general_v33 import PROTOCOL as general_protocol
    general=json.loads((result_root/'general_adaptive_v33/all_conditions.json').read_text())
    general_summary=json.loads((result_root/'general_adaptive_v33/summary.json').read_text())
    assert all(general['protocol'][k]==v for k,v in general_protocol.items())
    assert sorted(r['condition'] for r in general['records'])==list(range(60))
    for r in general['records']:
        assert r['status']=='COMPLETED' and not r['errors']
        assert r['target_met_in_budget'] and r['independent_audit']['status']=='PASS'
        assert F(r['upper'])-F(r['lower'])<=F(r['target'])==F(3e-6)
        assert r['total_seconds']<=9000 and r['independent_audit']['geometry_only_dispatch']
    assert general_summary['certified']==general_summary['audited']==60
    print(json.dumps({'status':'PASS','portable_core_modules':len(imports),'exact_centered_energy_cases':cases,
        'positive_ridge_interior_example':'1/54','dense_row_adapter_unknowns':8,'feasible_planar_witnesses':4,
        'public_final_width_successes':'20/20','strict_tail_successes':'10/20',
        'real_world_datasets_with_narrower_median':4,
        'exact_contraction_normal_equation_cases':contraction['exact_normal_equation_cases'],
        'matched_primary_dataset_control_runs':48,'new_matched_searches':40,
        'matched_energy_condition_audits':24,
        'adaptive_exact_audits':40,'same_instance_strict_scip_runs':4,
        'zorro_exact_fixed_point_and_adapter_audits':3,'midpoint_increment_intervals':20,
        'response_independent_masks':20,'response_independent_exact_trees':4,
        'all_primary_scip_masks':20,'refined_zorro_exact_model_bounds':80,
        'core_energy_new_pool_conditions':15,'core_energy_STOPS':13,'moment_STOPS':5,
        'extra_policy_saved_cells':264,'exact_integer_inference':'PASS',
        'saved_independent_audits':'20conditions,4complete_trees','precision_runs':48,'scip_runs':12,
        'enhanced_curvature_normal_equation_cases':curvature['original_normal_equation_cases'],
        'automatic_energy_complete_trees':len(automatic['reports']),'classical_width_definition':'unsquared Euclidean distance',
        'refined_original_independent_successes':20,'refined_new_independent_successes':20,
        'refined_primary_and_control_successes':40,'refined_decision_successes':15,
        'exact_multirank_successes':18,
        'general_original_conditions':60,'general_target_successes':60,'general_independent_replays':60,
        'raw_records_opened':False,'timed_queues_rerun':False}))


if __name__=='__main__':main()

