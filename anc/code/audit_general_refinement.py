"""Exact original-table, halfspace dual and completed-cover audit."""
from pathlib import Path
from fractions import Fraction as F
import json,sys,time,traceback
from audit_sdp_exact import audit as polynomial_audit
from independent_moment_audit import audit as moment_audit

def validate(folder):
 result=json.loads((folder/'result.json').read_text());original=json.loads((folder/'original_input.json').read_text())
 if max(sum(l!=u for l,u in zip(ll,uu)) for ll,uu in zip(original['lo'],original['hi']))>6:
  import independent_moment_audit as checker
  from independent_planar_rows import independent_rowmax
  checker.rowmax=independent_rowmax
 first=json.loads((folder/'first_stage.json').read_text());fa=moment_audit(folder/'first_stage.json',True)
 nodes={n['id']:n for n in result['nodes']};payloads={};reports=[]
 # A cap can finish one child while the valid parent remains an active leaf.
 # That unused child's conditional certificate is checked, but it has no
 # coverage role until its complementary sibling is complete.
 cover_ids=set();roots=set()
 for i in result['active_leaves']:
  while nodes[i]['parent'] is not None:i=nodes[i]['parent']
  roots.add(i)
 todo=list(roots)
 while todo:
  i=todo.pop();cover_ids.add(i);todo.extend(nodes[i].get('children',[]))
 unused_children=0
 for i,node in sorted(nodes.items()):
  f=folder/node['receipt'];data=json.loads(f.read_text());payloads[i]=data
  reports.append(polynomial_audit(f,original))
  source=data['problem'];cuts=source.get('extra_linear_inequalities',[])
  if node['parent'] is None:
   assert not cuts
   inherited=min([F(first['upper'])]+[F(v['result']['upper']) for j,v in payloads.items() if j<i and nodes[j]['parent'] is None])
  else:
   parent=nodes[node['parent']];ps=payloads[node['parent']]['problem'];prior=ps.get('extra_linear_inequalities',[])
   assert cuts[:len(prior)]==prior
   assert source['active_cells']==ps['active_cells'] and source['x0_uncertain']==ps['x0_uncertain']
   if 'split' in parent:
    split=parent['split'];sign=1 if node['side']==0 else -1
    assert F(cuts[len(prior)]['rhs'])==sign*F(split['cut'])
    assert {tuple(k):F(v) for k,v in cuts[len(prior)]['terms']}=={(j,):sign*F(v) for j,v in enumerate(split['a']) if v}
   else:
    assert i not in cover_ids and node['side'] in (0,1)
    assert len(cuts)-len(prior)==1+2*len(source['active_cells'])
    unused_children+=1
   inherited=F(parent['upper'])
  assert F(node['upper'])>=min(inherited,F(data['result']['upper']))
 if result['active_leaves']:
  roots=set()
  for i in result['active_leaves']:
   while nodes[i]['parent'] is not None:i=nodes[i]['parent']
   roots.add(i)
  assert len(roots)==1;todo=list(roots);leaves=[]
  while todo:
   i=todo.pop();node=nodes[i]
   if 'children' in node:
    cc=node['children'];assert len(cc)==2 and {nodes[c]['side'] for c in cc}=={0,1}
    assert all(nodes[c]['parent']==i for c in cc);todo.extend(cc)
   else:leaves.append(i)
  assert set(leaves)==set(result['active_leaves'])
  assert F(result['upper'])>=min(F(first['upper']),max(F(nodes[i]['upper']) for i in leaves))
 else:assert F(result['upper'])>=F(first['upper'])
 # Every node's reported lower is checked against its original-table witness.
 proved_lower=max([F(first['lower'])]+[F(v['result']['lower']) for v in payloads.values()])
 assert F(result['lower'])<=proved_lower
 assert result['target_met']==(result['upper']-result['lower']<=result['target'])
 assert result['target_met_in_budget']==any(h['seconds']<=result['cap_seconds'] and h['upper']-h['lower']<=result['target'] for h in result['history'])
 return dict(status='PASS',condition=result['condition'],first_stage=fa,polynomial_receipts=len(reports),complete_halfspace_cover=True,
             original_witness=True,unused_conditional_children=unused_children,target_met=result['target_met'],target_met_in_budget=result['target_met_in_budget'],absolute_gap=result['absolute_gap'],reports=reports)

if __name__=='__main__':
 folder=Path(sys.argv[1]);start=time.monotonic()
 try:result=validate(folder)
 except Exception as exc:result=dict(status='ERROR',error=str(exc),traceback=traceback.format_exc())
 result['seconds']=time.monotonic()-start;(folder/'independent_audit.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ('reports','first_stage')}),flush=True)
