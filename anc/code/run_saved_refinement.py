"""Refine every regenerated, unchanged primary/control/independent/decision table."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,json,time,traceback
import numpy as np
import psutil
from observed_anchor import ObservedAnchor
from centered_signal_energy import proposal
from contraction_energy import constants,AdaptiveEnergySearch
from run_real_world import coarse
from refinement_pipeline import refine
B=Path(__file__).resolve().parent

def inputs(family,input_dir=None):
    defaults={'controls':['contraction_run','matched_controls'],
              'independent':['independent_masks'],'decisions':['energy_decisions']}
    folders=[Path(input_dir)] if input_dir else [B/s for s in defaults[family]]
    rows=[]
    for folder in folders:
        saved=json.loads((folder/'results.json').read_text())
        for row in saved['results']:
            wanted='adaptive_energy' if family in ('independent','decisions') else 'energy'
            if row.get('method')!=wanted or row.get('reused_primary'):continue
            path=folder/row['certificate_file']
            dataset=row.get('dataset','superconductivity');seed=row['seed']
            if family=='controls':
                primary=folder.name=='contraction_run'
                stem=f"{'primary' if primary else 'matched'}_{dataset}_{seed}_{row.get('row_rule','residual')}_{row.get('candidate_bits',4)}"
            else:stem=path.stem
            rows.append((path,stem,dataset,seed))
    return rows

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=['controls','independent','decisions'],required=True)
    parser.add_argument('--input-dir',help='A regenerated run directory containing results.json and certificates/')
    parser.add_argument('--dataset');parser.add_argument('--output')
    parser.add_argument('--cpu',type=int)
    args=parser.parse_args();allowed=psutil.Process().cpu_affinity()
    chosen=args.cpu if args.cpu is not None else (10 if 10 in allowed else allowed[-1])
    psutil.Process().cpu_affinity([chosen])
    out=B/(args.output or f'frozen_{args.family}');out.mkdir(parents=True,exist_ok=True)
    for file,instance,name,seed in inputs(args.family,args.input_dir):
        if args.dataset and args.dataset!=name:continue
        if (out/f'{instance}_result.json').exists():continue
        try:
            raw=json.loads(file.read_text());c=raw['certificate'];er=c['energy_receipt']
            lo=np.array(c['lo']);hi=np.array(c['hi']);si=er['signal_rows'][0]
            sj=int(np.flatnonzero(lo[si]!=hi[si])[0])
            meta=dict(dataset=name,seed=seed,signal_row=si,signal_column=sj,row_ids=er['nuisance_rows'])
            start=time.monotonic();a=ObservedAnchor(c['lo'],c['hi'],c['y'],c['beta'],c['lambda_value'])
            caps,wit=coarse(a);R=proposal(a,meta);rec=constants(a,R,meta)
            solver=AdaptiveEnergySearch(a,R,rec);solver.best_original=caps['common_lower'];solver.best_witness=wit
            pre=time.monotonic()-start
            energy=solver.run(time_limit=max(.001,10-pre),absolute_tolerance=1e-10,relative_tolerance=0,record=True)
            cost=time.monotonic()-start
            energy.update(dataset=name,seed=seed,complete_seconds=cost,source_instance=file.name,
                input_contract='fixed stored binary table; all new anchor/proposal/constants/search costs remeasured')
            (out/f'{instance}_adaptive_energy.json').write_text(json.dumps(energy))
            (out/f'{instance}_meta.json').write_text(json.dumps(meta))
            result=refine(a,energy,cost,out,instance,meta)
            result.update(original_width=raw['absolute_gap'],energy_width=energy['absolute_gap'],energy_seconds=cost,family=args.family)
        except Exception as exc:
            result=dict(status='ERROR',instance=instance,error=str(exc),traceback=traceback.format_exc())
        (out/f'{instance}_result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('trace','tree','checkpoints')}),flush=True)

if __name__=='__main__':main()
