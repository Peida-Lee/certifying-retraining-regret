"""Rational verification of retained parent bounds and a complete box partition."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from fractions import Fraction as F
import sys,json,time,traceback,copy
import psutil,numpy as np
from audit_real_world import read,mq,vq,dot,mm,tr,inverse
from audit_zorro_matched import audit
def main():
    psutil.Process().cpu_affinity([int(os.environ.get('REGRET_AUDIT_CPU','12'))])
    sourcefile=Path(os.environ.get('REGRET_SOURCE_RESULTS',Path(__file__).parent/'contraction_run/results.json'));source=json.loads(sourcefile.read_text())
    out=Path(os.environ.get('REGRET_ZORRO_REFINED_OUTPUT',sourcefile.parent/'zorro_refined_results.json'));saved=json.loads(out.read_text());reports=[];started=time.monotonic()
    for result in saved['results']:
     report=dict(dataset=result['dataset'])
     try:
      base=next(r for r in source['results'] if r['dataset']==result['dataset'] and r['seed']==88211 and r['method']=='energy')
      cert=json.loads((sourcefile.parent/base['certificate_file']).read_text())['certificate'];D=read(cert)
      nodes=result['nodes'];m=len(result['source_cells']);assert nodes[0]['lo']==[-1.]*m and nodes[0]['hi']==[1.]*m
      assert result['common_cap']==cert['original_gradient_upper']
      leaves=[]
      for node in nodes:
       if node['children']:
        assert len(node['children'])==2;a,b=[nodes[i] for i in node['children']]
        lo,hi,al,ah,bl,bh=[vq(v) for v in (node['lo'],node['hi'],a['lo'],a['hi'],b['lo'],b['hi'])]
        changed=[j for j in range(m) if ah[j]!=hi[j]];assert len(changed)==1;j=changed[0]
        assert al==lo and bh==hi and lo[j]<ah[j]==bl[j]<hi[j]
        assert all(ah[k]==hi[k] and bl[k]==lo[k] for k in range(m) if k!=j)
       else:leaves.append(node['id'])
      assert sorted(leaves)==sorted(result['leaf_ids'])
      owners=set(n['bound_owner'] for n in nodes if n['bound_owner'] is not None)
      owner_reports={}
      for owner in sorted(owners):
       record=result['envelope_receipts'][owner];node=nodes[record['node_id']];sub=copy.deepcopy(cert)
       for k,(i,j) in enumerate(result['source_cells']):
        lower=D['c'][i][j]+D['w'][i][j]*F(node['lo'][k]);upper=D['c'][i][j]+D['w'][i][j]*F(node['hi'][k])
        sub['lo'][i][j]=float(lower);sub['hi'][i][j]=float(upper)
        assert F(sub['lo'][i][j])==lower and F(sub['hi'][i][j])==upper
       check=audit(read(sub),record['receipt']);owner_reports[owner]=check
      for node in nodes:
       parent_cap=F(result['common_cap']) if node['parent'] is None else F(nodes[node['parent']]['upper'])
       owner=node['bound_owner'];bound=F(result['common_cap']) if owner is None else F(result['envelope_receipts'][owner]['receipt']['upper'])
       assert F(node['upper'])>=min(parent_cap,bound)
       if owner is not None:
        owner_node=result['envelope_receipts'][owner]['node_id'];current=node['id']
        while current is not None and current!=owner_node:current=nodes[current]['parent']
        assert current==owner_node
      assert F(result['upper'])>=max(F(nodes[j]['upper']) for j in leaves)
      report.update(status='PASS',covered_leaves=len(leaves),verified_model_bounds=len(owners),
                    exact_fixed_point_and_adapter_checks=owner_reports,all_parent_bounds_valid=True,complete_partition=True)
     except Exception as exc:report.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
     reports.append(report);print(report['dataset'],report['status'],report.get('verified_model_bounds'),report.get('error',''),flush=True)
     out.with_name('zorro_refined_independent_audit.json').write_text(json.dumps(dict(status='RUNNING',reports=reports),indent=2))
    final=dict(status='PASS' if all(r['status']=='PASS' for r in reports) else 'FAILED',reports=reports,seconds=time.monotonic()-started,imports_refinement_or_adapter_implementation=False)
    out.with_name('zorro_refined_independent_audit.json').write_text(json.dumps(final,indent=2));print(final['status'])

if __name__=='__main__':main()
