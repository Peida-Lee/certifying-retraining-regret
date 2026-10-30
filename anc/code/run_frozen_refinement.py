"""Frozen response-independent masks; complete method costs and raw receipts."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,time,json,traceback,argparse
B=Path(__file__).resolve().parent;ROOT=B.parent;OLD=B


import numpy as np,psutil
from datasets import dataset
from centered_signal_energy import ObservedAnchor,proposal,constants as reference_constants,CenteredSignalSearch
from contraction_energy import constants,AdaptiveEnergySearch
from run_real_world import coarse

def make_instance(spec,seed):
    x,y,names=dataset(spec['dataset'],spec['pool_seed']);n,p=x.shape;lam=spec['lambda_value']
    chosen=np.random.default_rng(seed).choice(n,17,replace=False);signal=int(chosen[0]);nuisance=list(map(int,chosen[1:]));cols=spec['frozen_columns'];sj=cols['signal'];ns=cols['nuisance']
    lo=x.copy();hi=x.copy();ll=-1+np.floor((x[signal,sj]+1)/spec['width'])*spec['width'];lo[signal,sj]=ll;hi[signal,sj]=ll+spec['width']
    for order,i in enumerate(nuisance):
        for j in (ns[(2*order)%len(ns)],ns[(2*order+1)%len(ns)]):
            width=spec['nuisance_width'];ll=-1+np.floor((x[i,j]+1)/width)*width;lo[i,j]=ll;hi[i,j]=ll+width
    midpoint=(lo+hi)/2;beta=np.linalg.solve(midpoint.T@midpoint/n+lam*np.eye(p),midpoint.T@y/n);beta=np.round(beta*16)/16
    meta=dict(dataset=spec['dataset'],seed=seed,origin='independent_uniform',candidate_bits=4,n=n,p=p,lambda_value=lam,pool_seed=spec['pool_seed'],signal_row=signal,signal_column=sj,nuisance_columns=ns,row_ids=nuisance,uncertain_cells=int(np.sum(lo!=hi)),signal_unknown_columns=1,structural_rank_lower_bound=1,actual_retained_rank=3,response_independent_mask=True)
    return (lo,hi,y,beta,lam),meta

def run_one(spec,seed,out):
    from refinement_pipeline import refine
    start=time.monotonic();args,meta=make_instance(spec,seed);input_s=time.monotonic()-start
    start=time.monotonic();a=ObservedAnchor(*args,anchor='observed',preconditioner='anchor');anchor_s=time.monotonic()-start
    start=time.monotonic();caps,wit=coarse(a);coarse_s=time.monotonic()-start
    start=time.monotonic();R=proposal(a,meta);basis_s=time.monotonic()-start
    common=input_s+anchor_s+coarse_s+basis_s;meta.update(input_seconds=input_s,anchor_seconds=anchor_s,coarse_seconds=coarse_s,proposal_seconds=basis_s,caps=caps)
    instance=f"{spec['dataset']}_{seed}";(out/f'{instance}_meta.json').write_text(json.dumps(meta,indent=2))
    records={}
    for method in ('moment_centered','adaptive_energy'):
        start=time.monotonic();rec=reference_constants(a,R,meta,reference_only=True) if method=='moment_centered' else constants(a,R,meta)
        solver=CenteredSignalSearch(a,R,rec,'moment_centered') if method=='moment_centered' else AdaptiveEnergySearch(a,R,rec)
        setup=time.monotonic()-start;solver.best_original=caps['common_lower'];solver.best_witness=wit
        result=solver.run(time_limit=max(.001,10-common-setup),absolute_tolerance=1e-10,relative_tolerance=0,record=True)
        cost=common+setup+result['seconds'];result.update(dataset=spec['dataset'],seed=seed,complete_seconds=cost,setup_seconds=setup,common_seconds=common)
        (out/f'{instance}_{method}.json').write_text(json.dumps(result));records[method]=result
    energy=records['adaptive_energy'];final=refine(a,energy,energy['complete_seconds'],out,instance,meta)
    final.update(moment_width=records['moment_centered']['absolute_gap'],moment_seconds=records['moment_centered']['complete_seconds'],
                 energy_width=energy['absolute_gap'],energy_seconds=energy['complete_seconds'],energy_stop=energy['stop_reason'])
    (out/f'{instance}_result.json').write_text(json.dumps(final,indent=2))
    return final

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--dataset');parser.add_argument('--output',default='frozen_validation')
    parser.add_argument('--cpu',type=int,default=os.environ.get('REGRET_CPU'),help='Logical CPU; default reserves CPUs 0--3')
    args=parser.parse_args();allowed=sorted(psutil.Process().cpu_affinity());available=[cpu for cpu in allowed if cpu>=4]
    cpu=args.cpu if args.cpu is not None else (available[-1] if available else None)
    if cpu is None or cpu not in allowed:parser.error(f'Choose --cpu from {allowed}; default reserves logical CPUs 0--3')
    psutil.Process().cpu_affinity([cpu]);settings=json.loads((OLD/'contraction_settings.json').read_text());out=B/args.output;out.mkdir(exist_ok=True)
    for spec in settings['designs']:
        if args.dataset and spec['dataset']!=args.dataset:continue
        file=out/f"{spec['dataset']}_{args.seed}_result.json"
        if file.exists():continue
        try:result=run_one(spec,args.seed,out)
        except Exception as exc:
            result=dict(status='ERROR',dataset=spec['dataset'],seed=args.seed,error=str(exc),traceback=traceback.format_exc());file.write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('trace','tree','checkpoints')}),flush=True)

if __name__=='__main__':main()
