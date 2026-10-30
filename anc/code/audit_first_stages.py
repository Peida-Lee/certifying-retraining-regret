"""Reconstruct every new structural first-stage certificate independently."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,time,traceback
B=Path(__file__).resolve().parent;
from audit_contraction_real import read,audit,audit_witness,audit_tree
import psutil
psutil.Process().cpu_affinity([int(os.environ.get('REGRET_AUDIT_CPU','10'))])
folder=B/(sys.argv[1] if len(sys.argv)>1 else 'frozen_validation');output=folder/'first_stage_audit.json'
reports=json.loads(output.read_text())['reports'] if output.exists() else [];done={r['file'] for r in reports if r['status']=='PASS'}
for file in sorted(folder.glob('*_adaptive_energy.json')):
 if file.name in done:continue
 try:
    result=json.loads(file.read_text());cert=result['certificate'];data=read(cert);constants=audit(data,cert);audit_witness(data,cert,result)
    row=dict(status='PASS',file=file.name,constants=True,original_witness=True)
    if 'wine_white' in file.name or '92411' in file.name:row['complete_tree']=audit_tree(data,cert,result,constants)
 except Exception as exc:row=dict(status='ERROR',file=file.name,error=str(exc),traceback=traceback.format_exc())
 reports.append(row);output.write_text(json.dumps(dict(status='PASS' if all(r['status']=='PASS' for r in reports) else 'ERROR',reports=reports),indent=2));print(json.dumps({k:v for k,v in row.items() if k!='complete_tree'}),flush=True)
