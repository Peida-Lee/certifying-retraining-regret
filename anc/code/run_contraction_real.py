"""Full-cost comparison of contraction energy and centered moments."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import json,time,traceback,csv,copy
import psutil
from project_paths import MODULE_DIR
from real_world_design import problem
from centered_signal_energy import proposal,constants as reference_constants,ObservedAnchor,CenteredSignalSearch,MomentSearch
from contraction_energy import constants,ContractionEnergySearch,AdaptiveEnergySearch
from run_real_world import coarse


def main():
    psutil.Process().cpu_affinity([int(os.environ.get('REGRET_TIMED_CPU','10'))])
    settings=json.loads((MODULE_DIR/'contraction_settings.json').read_text())
    output=Path(os.environ.get('REGRET_OUTPUT',MODULE_DIR/'contraction_run'))
    output.mkdir(exist_ok=True,parents=True)
    resultfile=output/'results.json'
    assert not resultfile.exists(),'Use a fresh output directory for each timed run'
    (output/'certificates').mkdir(exist_ok=True)
    rows=[];conditions=[];start=time.monotonic()
    def save(status='RUNNING'):
        resultfile.write_text(json.dumps(dict(status=status,settings=settings,conditions=conditions,results=rows,
                                              seconds=time.monotonic()-start),indent=2))
    for spec in settings['designs']:
        for seed in settings['mask_seeds']:
            name=spec['dataset'];condition=dict(dataset=name,seed=seed)
            try:
                t=time.monotonic()
                args,meta=problem(name,spec['lambda_value'],4,16,spec['width'],8,
                    seed=seed,pool_seed=spec['pool_seed'],frozen_columns=spec['frozen_columns'],
                    eligible_factor=2,iterations=3,nuisance_width=spec['nuisance_width'])
                input_s=time.monotonic()-t
                t=time.monotonic();a=ObservedAnchor(*args,anchor='observed',preconditioner='anchor');anchor_s=time.monotonic()-t
                t=time.monotonic();caps,wit=coarse(a);coarse_s=time.monotonic()-t
                t=time.monotonic();R=proposal(a,meta);proposal_s=time.monotonic()-t
                t=time.monotonic();rec=constants(a,R,meta);energy_s=time.monotonic()-t
                t=time.monotonic();ref=reference_constants(a,R,meta,reference_only=True);reference_s=time.monotonic()-t
                condition.update(meta,status='PASS',caps=caps,input_seconds=input_s,anchor_seconds=anchor_s,
                    coarse_seconds=coarse_s,proposal_seconds=proposal_s,energy_seconds=energy_s,reference_seconds=reference_s,
                    energy_receipt=rec)
            except Exception as e:
                condition.update(status='ERROR',error=str(e),traceback=traceback.format_exc())
                conditions.append(condition);save();print(name,seed,'INPUT ERROR',traceback.format_exc(),flush=True);continue
            conditions.append(condition);save()
            for method in ('energy','moment_centered','full'):
                row=dict(dataset=name,seed=seed,method=method)
                try:
                    pre=input_s+anchor_s+coarse_s+(0 if method=='full' else proposal_s)
                    pre+=energy_s if method=='energy' else reference_s if method=='moment_centered' else 0
                    t=time.monotonic()
                    energy_class=ContractionEnergySearch if os.environ.get('REGRET_PURE_ENERGY')=='1' else AdaptiveEnergySearch
                    solver=(energy_class(a,R,rec) if method=='energy' else
                            CenteredSignalSearch(a,R,ref,mode='moment_centered') if method=='moment_centered' else MomentSearch(a,a.p))
                    setup_s=time.monotonic()-t
                    solver.best_original=caps['common_lower'];solver.best_witness=wit
                    remaining=max(.001,settings['complete_budget_seconds']-pre-setup_s)
                    result=solver.run(time_limit=remaining,absolute_tolerance=settings['search_tolerance'],relative_tolerance=0,record=True)
                    result.update(method=method,dataset=name,seed=seed)
                    fn=f'{name}_{seed}_{method}.json'
                    (output/'certificates'/fn).write_text(json.dumps(result,indent=2))
                    result.pop('certificate',None)
                    row.update(result,preprocessing_seconds=pre,setup_seconds=setup_s,total_seconds=pre+setup_s+result['seconds'],
                        target_met=result['absolute_gap']<=settings['final_width_target'],
                        certificate_file='certificates/'+fn,complete_budget_seconds=settings['complete_budget_seconds'])
                    print(name,seed,method,'gap',result['absolute_gap'],'tail',result['tail_error'],'seconds',row['total_seconds'],flush=True)
                except Exception as e:
                    row.update(status='ERROR',error=str(e),traceback=traceback.format_exc());print(name,seed,method,'ERROR',traceback.format_exc(),flush=True)
                rows.append(row);save()
    expected=len(settings['designs'])*len(settings['mask_seeds'])*3
    save('PASS' if len(rows)==expected and all(r['status']=='PASS' for r in rows) else 'COMPLETED_WITH_ERRORS')
    flat=[{k:v for k,v in r.items() if k not in ('trace','traceback','contraction_receipt')} for r in rows]
    keys=sorted(set().union(*(r.keys() for r in flat)))
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)

if __name__=='__main__':main()
