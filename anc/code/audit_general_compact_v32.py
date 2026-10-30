"""Independent complete-stage replay for the compact original-grid algorithm."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import json,sys,time,argparse,traceback
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import psutil
from independent_moment_audit import audit as audit_moment
from audit_compressed_arrow_v27 import verify
from audit_profile_outer_v31 import audit as audit_profile
from audit_arrow_tree_v29 import audit as audit_rows
from run_general_compact_v32 import PROTOCOL


def audit(folder):
    start=time.monotonic();r=json.loads((folder/'result.json').read_text());assert r['protocol']==PROTOCOL
    first=json.loads((folder/'first_stage.json').read_text());original=json.loads((folder/'original_input.json').read_text());cert=first['certificate']
    assert all(cert[k]==original[k] for k in ('lo','hi','y','beta'));assert cert['lambda']==original['lambda_value']
    fa=audit_moment(folder/'first_stage.json',True);lower=F(first['lower']);upper=F(first['upper']);stages={'structural':(lower,upper)};checks={'structural':fa}
    if (folder/'root/exact_audit.json').exists():
        data=json.loads((folder/'root/input.json').read_text());proposal=json.loads((folder/'root/proposal.json').read_text());assert data['original']==original
        ra=verify(data,proposal);checks['compact_root']=dict(status=ra['status'],gap=ra['gap']);lower=max(lower,F(ra['lower']));upper=min(upper,F(ra['exact_upper']));stages['compact_root']=(lower,upper)
    if (folder/'profile/result.json').exists():
        pa=audit_profile(folder/'profile');checks['profile_cover']=pa;pr=json.loads((folder/'profile/result.json').read_text());upper=min(upper,F(pr['upper']));stages['profile_cover']=(lower,upper)
    if (folder/'rows/result.json').exists():
        pa=audit_rows(folder/'rows');checks['row_cover']=pa;pr=json.loads((folder/'rows/result.json').read_text());upper=min(upper,F(pr['upper']));stages['row_cover']=(lower,upper)
    for h in r['history']:
        if h['stage'] in stages:
            l,u=stages[h['stage']];assert F(h['lower'])<=l and F(h['upper'])>=u
    assert F(r['lower'])<=lower and F(r['upper'])>=upper and F(r['absolute_gap'])>=F(r['upper'])-F(r['lower'])
    assert r['target_met']==(F(r['upper'])-F(r['lower'])<=F(r['target']))
    assert r['target_met_in_budget']==any(h['seconds']<=r['cap_seconds'] and F(h['upper'])-F(h['lower'])<=F(r['target']) for h in r['history'])
    return dict(status='PASS',condition=r['condition'],target_met=r['target_met'],target_met_in_budget=r['target_met_in_budget'],gap=r['absolute_gap'],complete_algorithm_seconds=r['total_seconds'],seconds=time.monotonic()-start,checks=checks)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--cpu',type=int,default=12);a=p.parse_args();psutil.Process().cpu_affinity([a.cpu]);folder=Path(a.folder)
    try:r=audit(folder)
    except Exception as exc:r=dict(status='ERROR',error=str(exc),traceback=traceback.format_exc())
    (folder/'independent_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='checks'}),flush=True)
