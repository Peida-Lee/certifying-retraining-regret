"""Frozen core-energy acquisition decisions on three disjoint evaluation pools."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import time,json,traceback
from fractions import Fraction as F
from unittest.mock import patch
import numpy as np,pandas as pd,psutil
from project_paths import MODULE_DIR,DATA_CORE
from centered_signal_energy import ObservedAnchor,proposal,constants as reference_constants,CenteredSignalSearch
from contraction_energy import constants,ContractionEnergySearch,AdaptiveEnergySearch
from real_world_design import problem
from run_real_world import coarse
from observed_anchor import I,matvec
from zorro_refined import run as zorro_run

def midpoint_interval(a,witness):
    theta=np.array(witness['theta']);c=a.c;y=a.y
    def objective(v):return (matvec(I(c),I(v))-I(y)).square().sum()/a.n+a.lam*I(v).square().sum()
    feasible=objective(a.beta)-objective(theta)
    residual=matvec(I(c),I(theta))-I(y)
    normal=matvec(I(c.T),residual)/a.n+a.lam*I(theta)
    transformed=matvec(I(a.t.T),normal)
    return max(0.,float(feasible.lo)),float((feasible+transformed.square().sum()/a.mu).hi)

def label(lower,upper,midlo,midhi,alpha):
    factor=1+F(str(alpha))
    if F(upper)<=factor*F(midlo):return 'STOP'
    if F(lower)>factor*F(midhi):return 'VIOLATION'
    return 'UNRESOLVED'

def main():
    psutil.Process().cpu_affinity([int(os.environ.get('REGRET_TIMED_CPU','10'))])
    settings=json.loads((MODULE_DIR/'decision_settings.json').read_text());alpha=settings['primary_increment_fraction']
    spec=next(s for s in json.loads((MODULE_DIR/'contraction_settings.json').read_text())['designs'] if s['dataset']=='superconductivity')
    sourcefile=Path(os.environ.get('REGRET_SOURCE_RESULTS',MODULE_DIR/'contraction_run/results.json'));source=json.loads(sourcefile.read_text())
    development=next(r for r in source['results'] if r['dataset']=='superconductivity' and r['seed']==88211 and r['method'] in ('energy','adaptive_energy'))
    sourcecert=json.loads((sourcefile.parent/development['certificate_file']).read_text())['certificate'];beta=np.array(sourcecert['beta'])
    start=time.monotonic();df=pd.read_csv(DATA_CORE/'superconductivity/train.csv');xx=df.iloc[:,:-1].to_numpy(float);yy=df.iloc[:,-1].to_numpy(float)
    old=np.sort(np.random.default_rng(74001).choice(len(df),1024,replace=False))
    scale=np.sort(np.random.default_rng(84001).choice(np.setdiff1d(np.arange(len(df)),old),1024,replace=False))
    dev=np.sort(np.random.default_rng(84101).choice(np.setdiff1d(np.arange(len(df)),np.union1d(old,scale)),1024,replace=False))
    excluded=np.union1d(np.union1d(old,scale),dev)
    available=np.random.default_rng(settings['partition_seed']).permutation(np.setdiff1d(np.arange(len(df)),excluded))
    ids=[np.sort(available[k*settings['pool_size']:(k+1)*settings['pool_size']]) for k in range(settings['pools'])]
    assert all(not np.intersect1d(v,excluded).size for v in ids)
    assert all(not np.intersect1d(ids[j],ids[k]).size for j in range(len(ids)) for k in range(j))
    span=np.where(np.ptp(xx[scale],axis=0)>0,np.ptp(xx[scale],axis=0),1.);mean=xx[scale].mean(0)
    yscale=np.ptp(yy[scale]);ymin=yy[scale].min();pools=[]
    for v in ids:
        x=np.column_stack([np.ones(len(v)),np.round((xx[v]-mean)/span*1024)/1024]);y=np.round((yy[v]-ymin)/yscale*1024)/1024
        pools.append((x,y,['intercept']+list(df.columns[:-1])))
    preparation=time.monotonic()-start
    output=Path(os.environ.get('REGRET_DECISION_OUTPUT',MODULE_DIR/'energy_decisions'));output.mkdir(parents=True,exist_ok=True);(output/'certificates').mkdir(exist_ok=True)
    resultfile=output/'results.json';assert not resultfile.exists();rows=[];conditions=[];began=time.monotonic()
    provenance=dict(pool_ids=[v.tolist() for v in ids],excluded_rows=len(excluded),mutually_disjoint=True,
        disjoint_from_development_and_scaling=True,response_scale=yscale,pool_preparation_seconds=preparation,
        candidate=beta.tolist(),frozen_candidate=True,plan=settings)
    (output/'pool_provenance_local.json').write_text(json.dumps(provenance,indent=2))
    def save(status='RUNNING'):resultfile.write_text(json.dumps(dict(status=status,settings=settings,conditions=conditions,results=rows,seconds=time.monotonic()-began),indent=2))
    for pool,pooldata in enumerate(pools):
      for seed in settings['mask_seeds']:
        meta=dict(dataset='superconductivity',pool=pool,seed=seed,origin='fresh_energy_decision')
        try:
            start=time.monotonic()
            with patch('real_world_design.dataset',return_value=pooldata):
                args,design=problem('superconductivity',spec['lambda_value'],4,16,spec['width'],8,seed=seed,
                    pool_seed=settings['partition_seed']+pool,frozen_columns=spec['frozen_columns'],eligible_factor=2,
                    iterations=3,nuisance_width=spec['nuisance_width'],candidate_beta=beta)
            input_s=preparation+time.monotonic()-start
            assert np.array_equal(args[3],beta)
            start=time.monotonic();a=ObservedAnchor(*args,anchor='observed',preconditioner='anchor');anchor_s=time.monotonic()-start
            start=time.monotonic();caps,wit=coarse(a);coarse_s=time.monotonic()-start
            start=time.monotonic();midlo,midhi=midpoint_interval(a,wit);mid_s=time.monotonic()-start
            common=input_s+anchor_s+coarse_s+mid_s
            start=time.monotonic();R=proposal(a,design);basis_s=time.monotonic()-start
            start=time.monotonic();rec=constants(a,R,design);energy_s=time.monotonic()-start
            start=time.monotonic();ref=reference_constants(a,R,design,reference_only=True);reference_s=time.monotonic()-start
            meta.update(design,status='PASS',midpoint_lower=midlo,midpoint_upper=midhi,caps=caps,input_seconds=input_s,
                anchor_seconds=anchor_s,coarse_seconds=coarse_s,midpoint_seconds=mid_s,common_seconds=common,
                candidate_kind='fixed_development_beta',actual_retained_rank=3,signal_unknown_columns=1)
        except Exception as exc:
            meta.update(status='ERROR',error=str(exc),traceback=traceback.format_exc());conditions.append(meta);save();continue
        conditions.append(meta)
        coarse_label=label(caps['common_lower'],caps['common_cap'],midlo,midhi,alpha)
        rows.append(dict(dataset='superconductivity',pool=pool,seed=seed,method='coarse',status='PASS',
            lower=caps['common_lower'],upper=caps['common_cap'],total_seconds=common,decision=coarse_label,
            decision_seconds=common if coarse_label!='UNRESOLVED' else None,saved_cells=33 if coarse_label=='STOP' else 0))
        for method in settings['methods']:
            if method=='coarse':continue
            row=dict(dataset='superconductivity',pool=pool,seed=seed,method=method)
            try:
                if method in ('contraction_energy','adaptive_energy','moment_centered'):
                    start=time.monotonic()
                    solver=(ContractionEnergySearch(a,R,rec) if method=='contraction_energy' else AdaptiveEnergySearch(a,R,rec)
                            if method=='adaptive_energy' else CenteredSignalSearch(a,R,ref,'moment_centered'))
                    setup=time.monotonic()-start;pre=common+basis_s+(reference_s if method=='moment_centered' else energy_s)
                    solver.best_original=caps['common_lower'];solver.best_witness=wit
                    result=solver.run(time_limit=max(.001,10-pre-setup),absolute_tolerance=1e-10,relative_tolerance=0,record=True)
                    row.update(preprocessing_seconds=pre,setup_seconds=setup,total_seconds=pre+setup+result['seconds'])
                    events=[dict(lower=v['lower'],upper=v['upper'],seconds=pre+setup+v['seconds']) for v in result['trace']]
                elif method=='SCIP':
                    from run_scip import run as scip_run
                    result=scip_run(a,common,10.,1e6,wit,absolute_tolerance=1e-10)
                    result.update(upper=result['numeric_upper'],absolute_gap=result['numeric_width'])
                    events=[dict(lower=v['lower'],upper=v['numeric_upper'],seconds=v['seconds']) for v in result['events']]
                    row.update(total_seconds=result['total_seconds'],upper_kind=result['upper_kind'])
                elif method=='ZORRO_refined':
                    result=zorro_run(a,common,10.,wit);events=result['events'];row.update(total_seconds=result['total_seconds'])
                else:raise ValueError(method)
                fn=f'pool{pool}_{seed}_{method}.json';(output/'certificates'/fn).write_text(json.dumps(result,indent=2))
                budget_events=[dict(lower=caps['common_lower'],upper=caps['common_cap'],seconds=common)]
                budget_events.extend(v for v in events if v['seconds']<=10.)
                if row['total_seconds']<=10.:
                    budget_events.append(dict(lower=result['lower'],upper=result['upper'],seconds=row['total_seconds']))
                budget_lower=budget_events[-1]['lower'];budget_upper=budget_events[-1]['upper']
                final_label=label(budget_lower,budget_upper,midlo,midhi,alpha)
                decision_s=common if coarse_label!='UNRESOLVED' else next((v['seconds'] for v in events if v['seconds']<=10. and label(v['lower'],v['upper'],midlo,midhi,alpha)!='UNRESOLVED'),None)
                row.update(status='PASS',lower=result['lower'],upper=result['upper'],absolute_gap=result['absolute_gap'],
                    tail_error=result.get('tail_error'),decision=final_label,decision_seconds=decision_s,
                    saved_cells=33 if final_label=='STOP' else 0,coarse_already_decided=coarse_label!='UNRESOLVED',certificate_file='certificates/'+fn,
                    budget_lower=budget_lower,budget_upper=budget_upper,decision_interval_width=budget_upper-budget_lower)
            except Exception as exc:row.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
            rows.append(row);save();print('pool',pool,seed,method,row['status'],row.get('decision'),row.get('absolute_gap'),flush=True)
    expected=settings['pools']*len(settings['mask_seeds'])*len(settings['methods'])
    save('PASS' if len(rows)==expected and all(r['status']=='PASS' for r in rows) else 'COMPLETED_WITH_ERRORS')

if __name__=='__main__':main()
