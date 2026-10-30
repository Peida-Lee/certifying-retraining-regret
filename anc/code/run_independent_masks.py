"""Frozen response-independent signal/nuisance masks on the same four tasks."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,traceback,csv
import numpy as np,psutil
from project_paths import MODULE_DIR
B=MODULE_DIR
from datasets import dataset
from centered_signal_energy import ObservedAnchor,proposal,constants as reference_constants,CenteredSignalSearch
from contraction_energy import constants,AdaptiveEnergySearch
from run_real_world import coarse

def main():
 psutil.Process().cpu_affinity([int(os.environ.get('REGRET_TIMED_CPU','10'))])
 plan=json.loads((B/'independent_mask_settings.json').read_text());settings=json.loads((B/'contraction_settings.json').read_text())
 out=Path(os.environ.get('REGRET_INDEPENDENT_OUTPUT',B/'independent_masks'));out.mkdir(parents=True,exist_ok=True);(out/'certificates').mkdir(exist_ok=True)
 assert not (out/'results.json').exists();rows=[];conditions=[];started=time.monotonic()
 def save(status='RUNNING'):(out/'results.json').write_text(json.dumps(dict(status=status,plan=plan,conditions=conditions,results=rows,seconds=time.monotonic()-started),indent=2))
 for spec in settings['designs']:
  for seed in plan['seeds']:
   meta=dict(dataset=spec['dataset'],seed=seed,origin='independent_uniform',candidate_bits=4)
   try:
    start=time.monotonic();x,y,names=dataset(spec['dataset'],spec['pool_seed']);n,p=x.shape;lam=spec['lambda_value']
    chosen=np.random.default_rng(seed).choice(n,17,replace=False);signal=int(chosen[0]);nuisance=list(map(int,chosen[1:]));cols=spec['frozen_columns'];sj=cols['signal'];ns=cols['nuisance']
    lo=x.copy();hi=x.copy()
    ll=-1+np.floor((x[signal,sj]+1)/spec['width'])*spec['width'];lo[signal,sj]=ll;hi[signal,sj]=ll+spec['width']
    for order,i in enumerate(nuisance):
     for j in (ns[(2*order)%len(ns)],ns[(2*order+1)%len(ns)]):
      width=spec['nuisance_width'];ll=-1+np.floor((x[i,j]+1)/width)*width;lo[i,j]=ll;hi[i,j]=ll+width
    midpoint=(lo+hi)/2;beta=np.linalg.solve(midpoint.T@midpoint/n+lam*np.eye(p),midpoint.T@y/n);beta=np.round(beta*16)/16
    input_s=time.monotonic()-start
    meta.update(n=n,p=p,lambda_value=lam,pool_seed=spec['pool_seed'],signal_row=signal,signal_column=sj,
        nuisance_columns=ns,row_ids=nuisance,uncertain_cells=int(np.sum(lo!=hi)),signal_unknown_columns=1,
        structural_rank_lower_bound=1,actual_retained_rank=3,response_independent_mask=True)
    start=time.monotonic();a=ObservedAnchor(lo,hi,y,beta,lam,anchor='observed',preconditioner='anchor');anchor_s=time.monotonic()-start
    start=time.monotonic();caps,wit=coarse(a);coarse_s=time.monotonic()-start
    start=time.monotonic();R=proposal(a,meta);basis_s=time.monotonic()-start
    start=time.monotonic();rec=constants(a,R,meta);energy_s=time.monotonic()-start
    start=time.monotonic();ref=reference_constants(a,R,meta,reference_only=True);reference_s=time.monotonic()-start
    common=input_s+anchor_s+coarse_s+basis_s
    meta.update(status='PASS',caps=caps,input_seconds=input_s,anchor_seconds=anchor_s,coarse_seconds=coarse_s,proposal_seconds=basis_s)
   except Exception as exc:
    meta.update(status='ERROR',error=str(exc),traceback=traceback.format_exc());conditions.append(meta);save();continue
   conditions.append(meta)
   for method in plan['methods']:
    row={k:meta[k] for k in ('dataset','seed','origin','signal_unknown_columns','structural_rank_lower_bound','actual_retained_rank')};row['method']=method
    try:
     start=time.monotonic();solver=AdaptiveEnergySearch(a,R,rec) if method=='adaptive_energy' else CenteredSignalSearch(a,R,ref,'moment_centered');setup=time.monotonic()-start
     pre=common+(energy_s if method=='adaptive_energy' else reference_s);solver.best_original=caps['common_lower'];solver.best_witness=wit
     result=solver.run(time_limit=max(.001,10-pre-setup),absolute_tolerance=1e-10,relative_tolerance=0,record=True)
     filename=f"{meta['dataset']}_{seed}_{method}.json";(out/'certificates'/filename).write_text(json.dumps(result,indent=2));result.pop('certificate')
     row.update(result,status='PASS',method=method,preprocessing_seconds=pre,setup_seconds=setup,total_seconds=pre+setup+result['seconds'],
       target_met=result['absolute_gap']<=1e-5,tail_target_met=result['tail_error']<=2.5e-6,coarse_already_met=caps['common_gap']<=1e-5,certificate_file='certificates/'+filename)
    except Exception as exc:row.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
    rows.append(row);save();print(row['dataset'],seed,method,row['status'],row.get('absolute_gap'),flush=True)
 save('PASS' if len(rows)==40 and all(r['status']=='PASS' for r in rows) else 'COMPLETED_WITH_ERRORS')

if __name__=='__main__':main()
