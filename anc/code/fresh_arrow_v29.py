"""Fresh complete-cost diagnostic using common row-compressed rules."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,sys,time,argparse,traceback
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import numpy as np,psutil
from public_problems import problem
from observed_anchor import ObservedAnchor
from local_residual import LocalResidualSearch
from refinement_pipeline import multistart
from robust_reference import propose
from run_general_refinement import GRID
from uniform_curvature_v28 import choose
from compressed_edge_arrow_v27 import solve
from audit_compressed_arrow_v27 import verify
from pilot_arrow_tree_v28 import run as tree_run

CONFIG=dict(target=3e-6,tau=.5,witness_starts=64,reference_iterations=16,reference_starts=32,
    root_conic_seconds=90,root_iterations=12,max_tree_nodes=65,tree_cap_seconds=900,
    timing='Fresh original input, anchor, reference, numerical solve and exact verification included')


def run(cid,out,iterations=16,use_young=True,root_only=False):
    start=time.monotonic();name,width,spread,seed=GRID[cid]
    args,meta=problem(name=name,cells=128,width=width,spread=spread,seed=seed,subset_seed=74101)
    anchor=ObservedAnchor(*args);search=LocalResidualSearch(anchor);witness=multistart(search,starts=64)
    o=dict(lo=anchor.lo.tolist(),hi=anchor.hi.tolist(),y=anchor.y.tolist(),beta=anchor.beta.tolist(),lambda_value=anchor.lam)
    x0=np.where(np.array(search.witness['x'])>=anchor.c,anchor.hi,anchor.lo);ids=search.ids.tolist()
    src=dict(T=anchor.t.tolist(),mu=anchor.mu,theta=anchor.beta.tolist(),active_cells=ids,x0_uncertain=[float(x0[i,j]) for i,j in ids])
    data=dict(original=o,problem=src,witness=search.witness,result=dict(condition=cid,lower=search.best,target=CONFIG['target']))
    if use_young:
        data=choose(data,CONFIG['tau']);anchor.t=np.array(data['problem']['T']);anchor.mu=data['problem']['mu']
    theta,reference=propose(search,search.witness,iterations=iterations,starts=CONFIG['reference_starts'])
    data['problem']['theta']=theta.tolist();data['witness']=search.witness;data['result']['lower']=search.best
    (out/'input.json').write_text(json.dumps(data));(out/'reference.json').write_text(json.dumps(reference))
    prep=time.monotonic()-start
    def emit(best,item):
        (out/'proposal.json').write_text(json.dumps(best));print(json.dumps(dict(stage='root',**item)),flush=True)
    ans=solve(data,seconds=CONFIG['root_conic_seconds'],lower=search.best,max_iterations=CONFIG['root_iterations'],emit=emit)
    audit=verify(data,ans);(out/'exact_audit.json').write_text(json.dumps(audit,indent=2))
    report=dict(status=audit['status'],condition=cid,dataset=name,seed=seed,spread=spread,width=width,lower=search.best,upper=audit['upper'],gap=audit['gap'],target=CONFIG['target'],target_met=audit['target_met'],prep_seconds=prep,total_seconds=time.monotonic()-start,protocol=dict(CONFIG,reference_iterations=iterations,use_young=use_young))
    (out/'run_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    if not report['target_met'] and not root_only:
        tree=out/'tree';tree.mkdir(exist_ok=False);refined=tree_run(out,tree,CONFIG['max_tree_nodes'],CONFIG['tree_cap_seconds'])
        report.update(upper=refined['upper'],gap=refined['gap'],target_met=refined['target_met'],total_seconds=time.monotonic()-start,tree_nodes=len(refined['nodes']),tree=str(tree))
        (out/'run_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--cpu',type=int,required=True);p.add_argument('--iterations',type=int,default=16);p.add_argument('--observed-only',action='store_true');args=p.parse_args()
    psutil.Process().cpu_affinity([args.cpu]);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    try:run(args.condition,out,args.iterations,not args.observed_only)
    except Exception as exc:(out/'error.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc()),indent=2));raise
