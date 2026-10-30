"""Matched precision/nuisance-row controls on the four primary datasets."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import time,json,csv,traceback,copy
import numpy as np
import psutil
from project_paths import MODULE_DIR
from datasets import dataset
from centered_signal_energy import ObservedAnchor,proposal,constants as reference_constants,CenteredSignalSearch
from contraction_energy import constants,ContractionEnergySearch,AdaptiveEnergySearch
from run_real_world import coarse


def main():
    psutil.Process().cpu_affinity([10])
    primary=Path(os.environ.get('REGRET_PRIMARY_RUN',MODULE_DIR/'contraction_run'))
    output=Path(os.environ.get('REGRET_PRECISION_OUTPUT',MODULE_DIR/'matched_controls'))
    output.mkdir(exist_ok=True,parents=True);(output/'certificates').mkdir(exist_ok=True)
    out=output/'results.json';assert not out.exists()
    src=json.loads((primary/'results.json').read_text());assert src['status']=='PASS'
    first=src['settings']['mask_seeds'][0];records=[];conditions=[];start=time.monotonic()
    def save(status='RUNNING'):
        out.write_text(json.dumps(dict(status=status,seed=first,conditions=conditions,results=records,
           seconds=time.monotonic()-start,primary_run='contraction_run',complete_budget_seconds=10,
           search_tolerance=1e-10,reused_primary_runs=8),indent=2))
    for spec in src['settings']['designs']:
        name=spec['dataset'];original=next(c for c in src['conditions'] if c['dataset']==name and c['seed']==first)
        base=next(r for r in src['results'] if r['dataset']==name and r['seed']==first and r['method']=='energy')
        cert=json.loads((primary/base['certificate_file']).read_text())['certificate']
        base_lo=np.array(cert['lo']);base_hi=np.array(cert['hi']);yy=np.array(cert['y']);lam=original['lambda_value']
        xx,_,_=dataset(name,original['pool_seed']);si,sj=original['signal_row'],original['signal_column']
        for rowrule in ('residual','uniform'):
            for bits in (0,8,4):
                condition=dict(dataset=name,row_rule=rowrule,candidate_bits=bits,seed=first)
                if rowrule=='residual' and bits==4:
                    condition.update(status='PASS',caps=original['caps'],coarse_already_met=original['caps']['common_gap']<=1e-5,
                        n=original['n'],p=original['p'],uncertain_cells=original['uncertain_cells'],reused_primary=True)
                    conditions.append(condition)
                    for method in ('energy','moment_centered'):
                        row=copy.deepcopy(next(r for r in src['results'] if r['dataset']==name and r['seed']==first and r['method']==method))
                        row.update(row_rule=rowrule,candidate_bits=bits,reused_primary=True,certificate_origin='primary',
                                   coarse_already_met=condition['coarse_already_met'])
                        records.append(row);print(name,rowrule,bits,method,'REUSED',row['absolute_gap'],flush=True)
                    save();continue
                try:
                    t=time.monotonic();lo=base_lo.copy();hi=base_hi.copy();meta=original.copy()
                    if rowrule=='uniform':
                        lo=xx.copy();hi=xx.copy();lo[si,sj]=base_lo[si,sj];hi[si,sj]=base_hi[si,sj]
                        rows=np.random.default_rng(first).choice(np.setdiff1d(np.arange(len(xx)),[si]),16,replace=False)
                        ns=original['nuisance_columns'];nw=original['nuisance_width']
                        for i in rows:
                            for j in [int(ns[(2*i+t)%len(ns)]) for t in range(2)]:
                                ll=-1+np.floor((xx[i,j]+1)/nw)*nw;lo[i,j]=ll;hi[i,j]=ll+nw
                        meta['row_ids']=rows.tolist()
                    midpoint=(lo+hi)/2
                    beta=np.linalg.solve(midpoint.T@midpoint/len(xx)+lam*np.eye(xx.shape[1]),midpoint.T@yy/len(xx))
                    if bits:beta=np.round(beta*2**bits)/2**bits
                    input_s=time.monotonic()-t
                    t=time.monotonic();a=ObservedAnchor(lo,hi,yy,beta,lam,anchor='observed',preconditioner='anchor');anchor_s=time.monotonic()-t
                    t=time.monotonic();caps,wit=coarse(a);cap_s=time.monotonic()-t
                    t=time.monotonic();R=proposal(a,meta);basis_s=time.monotonic()-t
                    t=time.monotonic();rec=constants(a,R,meta);energy_s=time.monotonic()-t
                    t=time.monotonic();ref=reference_constants(a,R,meta,reference_only=True);ref_s=time.monotonic()-t
                    common=input_s+anchor_s+cap_s+basis_s
                    condition.update(status='PASS',caps=caps,coarse_already_met=caps['common_gap']<=1e-5,n=a.n,p=a.p,
                       uncertain_cells=int(np.sum(lo!=hi)),input_seconds=input_s,anchor_seconds=anchor_s,
                       common_seconds=common,energy_seconds=energy_s,reference_seconds=ref_s,
                       signal_row=si,signal_column=sj,nuisance_rows=meta['row_ids'])
                except Exception as e:
                    condition.update(status='ERROR',error=str(e),traceback=traceback.format_exc());conditions.append(condition);save();continue
                conditions.append(condition)
                for method in ('energy','moment_centered'):
                    row=dict(dataset=name,row_rule=rowrule,candidate_bits=bits,seed=first,method=method)
                    try:
                        t=time.monotonic()
                        energy_class=ContractionEnergySearch if os.environ.get('REGRET_PURE_ENERGY')=='1' else AdaptiveEnergySearch
                        s=energy_class(a,R,rec) if method=='energy' else CenteredSignalSearch(a,R,ref,'moment_centered')
                        setup=time.monotonic()-t;pre=common+(energy_s if method=='energy' else ref_s)
                        s.best_original=caps['common_lower'];s.best_witness=wit
                        result=s.run(time_limit=max(.001,10-pre-setup),absolute_tolerance=1e-10,relative_tolerance=0,record=True)
                        fn=f'{name}_{rowrule}_{bits}_{method}.json'
                        (output/'certificates'/fn).write_text(json.dumps(result,indent=2));result.pop('certificate',None)
                        row.update(result,method=method,status='PASS',preprocessing_seconds=pre,setup_seconds=setup,
                          total_seconds=pre+setup+result['seconds'],target_met=result['absolute_gap']<=1e-5,
                          coarse_already_met=condition['coarse_already_met'],certificate_file='certificates/'+fn,
                          certificate_origin='matched',reused_primary=False)
                        print(name,rowrule,bits,method,'gap',row['absolute_gap'],'tail',row['tail_error'],'total',row['total_seconds'],flush=True)
                    except Exception as e:
                        row.update(status='ERROR',error=str(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
                    records.append(row);save()
    save('PASS' if len(records)==48 and all(r['status']=='PASS' for r in records) else 'COMPLETED_WITH_ERRORS')
    flat=[{k:v for k,v in r.items() if k not in ('trace','traceback','contraction_receipt')} for r in records]
    keys=sorted(set().union(*(r.keys() for r in flat)))
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)

if __name__=='__main__':main()
