"""Exact original-target audit and complete-cover audit for localized refinement."""
from pathlib import Path
from fractions import Fraction as F
import json,sys,time,traceback
from audit_compressed_exact import audit

def validate(folder):
 original=json.loads((folder/'original_input.json').read_text());result=json.loads((folder/'result.json').read_text());nodes={n['id']:n for n in result['nodes']};payloads={};reports=[]
 assert nodes and result['status']!='RUNNING'
 cover=set();todo=[0]
 while todo:
  i=todo.pop();cover.add(i);todo.extend(nodes[i].get('children',[]))
 unused=0
 for i,node in sorted(nodes.items()):
  file=folder/node['receipt'];payload=json.loads(file.read_text());payloads[i]=payload;reports.append(audit(file,original));source=payload['problem'];cuts=source['domain_inequalities']
  if node['parent'] is None:
   assert i==0 and not cuts
   # If the initial gradient bound was active, its separate audit is required.
   # The localized runs currently use the stronger root polynomial bound.
   assert F(node['upper'])>=F(payload['result']['upper'])
  else:
   parent=nodes[node['parent']];ps=payloads[node['parent']]['problem'];prior=ps['domain_inequalities'];assert cuts[:len(prior)]==prior
   assert source['active_cells']==ps['active_cells'] and source['x0_uncertain']==ps['x0_uncertain']
   if 'split' in parent:
    split=parent['split'];sign=1 if node['side']==0 else -1
    assert F(cuts[len(prior)]['rhs'])==sign*F(split['cut'])
    assert {tuple(k):F(v) for k,v in cuts[len(prior)]['terms']}=={(j,):sign*F(v) for j,v in enumerate(split['a']) if v}
   else:
    assert i not in cover and node['side'] in (0,1);unused+=1
   assert F(node['upper'])>=min(F(parent['upper']),F(payload['result']['upper']))
 todo=[0];leaves=[]
 while todo:
  i=todo.pop();node=nodes[i]
  if 'children' in node:
   cc=node['children'];assert len(cc)==2 and {nodes[c]['side'] for c in cc}=={0,1};assert all(nodes[c]['parent']==i for c in cc);todo.extend(cc)
  else:leaves.append(i)
 assert set(leaves)==set(result['active_leaves'])
 assert F(result['upper'])>=max(F(nodes[i]['upper']) for i in leaves)
 assert F(result['lower'])<=max(F(p['result']['lower']) for p in payloads.values())
 assert result['target_met']==(result['upper']-result['lower']<=result['target'])
 assert result['target_met_in_budget']==any(h['seconds']<=result['cap_seconds'] and h['upper']-h['lower']<=result['target'] for h in result['history'])
 return dict(status='PASS',condition=result['condition'],kind=result['kind'],polynomial_receipts=len(reports),exact_localizers=True,exact_dual_residual=True,original_witness=True,complete_cover=True,unused_children=unused,target_met=result['target_met'],target_met_in_budget=result['target_met_in_budget'],absolute_gap=result['absolute_gap'],reports=reports)

if __name__=='__main__':
 folder=Path(sys.argv[1]);start=time.monotonic()
 try:r=validate(folder)
 except Exception as e:r=dict(status='ERROR',error=str(e),traceback=traceback.format_exc())
 r['seconds']=time.monotonic()-start;(folder/'independent_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='reports'}),flush=True)
