"""Replay the chosen original-table certificate and account for complete timing."""
from pathlib import Path
from fractions import Fraction as F
import json,time,argparse,traceback
import psutil
from run_adaptive_general_v33 import PROTOCOL,dispatch
from audit_general_compact_v32 import audit as compact_audit
from audit_general_refinement import validate as moment_audit

def audit(folder):
    start=time.monotonic();r=json.loads((folder/'result.json').read_text());assert r['protocol']==PROTOCOL
    inner=folder/'engine';original=json.loads((inner/'original_input.json').read_text());route=dispatch(original['lo'],original['hi'])
    assert route==r['dispatch'];d=json.loads((folder/'dispatch.json').read_text());assert d['protocol']==PROTOCOL
    assert all(d[k]==v for k,v in route.items())
    checked=compact_audit(inner) if route['engine']=='compact' else moment_audit(inner)
    assert checked['status']=='PASS'
    child=json.loads((inner/'result.json').read_text());assert F(r['lower'])<=F(child['lower']) and F(r['upper'])>=F(child['upper'])
    assert F(r['absolute_gap'])>=F(r['upper'])-F(r['lower'])
    assert r['total_seconds']>=r['dispatch_seconds']+child['total_seconds'] and r['dispatch_seconds']>=d['seconds']
    assert len(r['history'])==len(child['history'])
    for h,c in zip(r['history'],child['history']):
        assert h['seconds']==r['dispatch_seconds']+c['seconds'] and h['lower']==c['lower'] and h['upper']==c['upper']
    assert r['target_met']==(F(r['upper'])-F(r['lower'])<=F(r['target']))
    assert r['target_met_in_budget']==any(h['seconds']<=r['cap_seconds'] and F(h['upper'])-F(h['lower'])<=F(r['target']) for h in r['history'])
    return dict(status='PASS',condition=r['condition'],engine=route['engine'],geometry_only_dispatch=True,original_witness=True,
        complete_cover=True,target_met=r['target_met'],target_met_in_budget=r['target_met_in_budget'],gap=r['absolute_gap'],seconds=time.monotonic()-start,
        complete_algorithm_seconds=r['total_seconds'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--cpu',type=int,required=True);a=p.parse_args();psutil.Process().cpu_affinity([a.cpu]);folder=Path(a.folder)
    try:r=audit(folder)
    except Exception as exc:r=dict(status='ERROR',error=str(exc),traceback=traceback.format_exc())
    (folder/'independent_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
