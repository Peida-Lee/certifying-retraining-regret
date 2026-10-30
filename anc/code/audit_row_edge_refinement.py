"""Replay every exact row-edge certificate and its complete covering tree."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from fractions import Fraction as F
import sys,json,time,argparse,traceback,math
import psutil
from run_row_edge_refinement import audit_input
from row_edge_exact import reconstruct
from edge_polynomial import edge_upper
from row_edge_screen import screen,restricted_upper
from observed_anchor import float_up
B=Path(__file__).resolve().parent

def check(out):
    start=time.monotonic();result=json.loads((out/'result.json').read_text());assert result['status'] in ('PASS','COMPLETED_WITH_ERRORS')
    data=json.loads((out/'input.json').read_text());audit_input(data);lower=F(data['result']['lower']);target=F(result['target'])
    assert target==F(1e-8) and F(result['exact_lower'])==lower
    nodes={v['id']:v for v in result['nodes']};root=json.loads((out/'node_000.json').read_text());const,root_polys,root_psd=reconstruct(data,root['proposal'])
    root_bounds=[edge_upper(poly,k,tolerance=F(1,10000))[0] for k,poly,residual in root_polys]
    root_upper=const+F(root['proposal']['alpha'])/10**6+sum(root_bounds,F(0))/10**6
    assert root_upper==F(root['exact']['exact_upper'])==F(nodes[0]['exact_upper']) and root_upper>=lower
    assert nodes[0]['parent'] is None and nodes[0]['domains']==[None]*len(root_polys)
    threshold=F(result['screen_threshold']);assert threshold==lower+F(99,100)*target
    covers=None
    if (out/'screen.json').exists():
        saved=json.loads((out/'screen.json').read_text());assert F(saved['root_upper'])==root_upper and F(saved['threshold'])==threshold
        covers=saved['rows']
        for (k,poly,residual),b,cover in zip(root_polys,root_bounds,covers):
            depth=int(math.log2(cover['interval_scale']));assert 2**depth==cover['interval_scale']
            expected=screen(poly,k,b-(root_upper-threshold)*10**6,depth);assert expected==cover
    reports=[dict(node=0,status='PASS',upper=float_up(root_upper))]
    checked={0:root_upper}
    for idx,node in sorted(nodes.items()):
        if idx==0:continue
        at=time.monotonic();assert covers is not None and node['parent'] in checked
        record=json.loads((out/f'node_{idx:03d}.json').read_text());prop=record['proposal'];assert prop['residual_domains']==node['domains']
        const,polys,psd=reconstruct(data,prop);value=const+F(prop['alpha'])/10**6;empty=False
        for (k,poly,residual),cover,domain in zip(polys,covers,node['domains']):
            bound,stats=restricted_upper(poly,k,cover,residual,None if domain is None else list(map(F,domain)))
            if bound is None:empty=True;value=threshold;break
            value+=bound/10**6
        if not empty:assert value==F(record['exact']['exact_upper'])
        value=min(value,checked[node['parent']]);assert value==F(node['exact_upper']);checked[idx]=value
        row=dict(node=idx,status='PASS',upper=float_up(value),seconds=time.monotonic()-at);reports.append(row);print(json.dumps(row),flush=True)
    frontier=[];visited=set()
    def visit(idx):
        assert idx not in visited;visited.add(idx);node=nodes[idx]
        if 'children' not in node:frontier.append(idx);return
        assert len(node['children'])==2;split=node['split'];row=split['row'];cut=F(split['cut']);bounds=node['domains'][row]
        if bounds is None:
            res=root_polys[row][2];lo=res[0]+sum((min(v,F(0)) for v in res[1:]),F(0));hi=res[0]+sum((max(v,F(0)) for v in res[1:]),F(0))
            bounds=list(map(F,split['original_interval']));assert bounds[0]<=lo and bounds[1]>=hi
        else:
            bounds=list(map(F,bounds));assert bounds==list(map(F,split['original_interval']))
        assert bounds[0]<cut<bounds[1]
        for side,child in enumerate(node['children']):
            n=nodes[child];assert n['parent']==idx
            assert list(map(F,n['domains'][row]))==([bounds[0],cut] if side==0 else [cut,bounds[1]])
            for j in range(len(root_polys)):
                if j!=row:assert n['domains'][j]==node['domains'][j]
            visit(child)
    visit(0);assert sorted(frontier)==sorted(result['active_leaves'])
    upper=min(root_upper,max(threshold,max(checked[i] for i in frontier)));assert upper==F(result['exact_upper'])
    assert F(result['upper'])>=upper and F(result['absolute_gap'])>=upper-lower
    met=upper-lower<=target;assert result['target_met']==met
    assert result['target_met_in_budget']==(met and result['total_seconds']<=result['cap_seconds'])
    return dict(status='PASS',condition=result['condition'],dataset=result['meta']['dataset'],original_input=True,exact_psd=True,exact_screening=True,complete_cover=True,
        target_met=met,target_met_in_budget=result['target_met_in_budget'],gap=float_up(upper-lower),lower=float(lower),upper=float_up(upper),
        algorithm_seconds=result['total_seconds'],seconds=time.monotonic()-start,checked_nodes=len(reports),unused_completed_nodes=len(nodes)-len(visited),nodes=reports)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--cpu',type=int,default=12);a=p.parse_args();psutil.Process().cpu_affinity([a.cpu]);out=B/a.folder
    try:r=check(out)
    except Exception as e:r=dict(status='ERROR',error=str(e),traceback=traceback.format_exc())
    (out/'independent_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='nodes'}),flush=True)
