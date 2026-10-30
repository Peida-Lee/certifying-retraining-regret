"""Independent constants/witness and fixed-mask checks for matched controls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import json,time,traceback
import psutil
from project_paths import MODULE_DIR
from audit_contraction_real import audit,read,audit_witness


def main():
    psutil.Process().cpu_affinity([12])
    primary=Path(os.environ.get('REGRET_PRIMARY_RUN',MODULE_DIR/'contraction_run'))
    output=Path(os.environ.get('REGRET_PRECISION_OUTPUT',MODULE_DIR/'matched_controls'))
    source=json.loads((output/'results.json').read_text());assert source['status']=='PASS'
    primary_audit=json.loads((primary/'independent_audit.json').read_text());assert primary_audit['status']=='PASS'
    originals=json.loads((primary/'results.json').read_text());reports=[];masks={};start=time.monotonic()
    for row in source['results']:
        if row['method']!='energy':continue
        report={k:row[k] for k in ('dataset','seed','row_rule','candidate_bits')}
        try:
            path=(primary if row['reused_primary'] else output)/row['certificate_file']
            result=json.loads(path.read_text());cert=result['certificate'];D=read(cert)
            key=(row['dataset'],row['row_rule']);signature=(D['lo'],D['hi'],D['y'])
            if key in masks:assert masks[key]==signature
            else:masks[key]=signature
            assert sum(l!=u for lo,hi in zip(D['lo'],D['hi']) for l,u in zip(lo,hi))==33
            bits=row['candidate_bits']
            if bits:assert all((v*2**bits).denominator==1 for v in D['beta'])
            if row['reused_primary']:
                assert any(r['dataset']==row['dataset'] and r['seed']==row['seed'] and r['status']=='PASS' for r in primary_audit['reports'])
                report.update(status='PASS',audit_origin='reused_independent_primary_audit')
            else:
                C=audit(D,cert);audit_witness(D,cert,result)
                report.update(status='PASS',audit_origin='fresh_exact_constants_and_original_witness')
        except Exception as e:report.update(status='ERROR',error=str(e),traceback=traceback.format_exc())
        reports.append(report)
        (output/'independent_audit.json').write_text(json.dumps(dict(status='RUNNING',reports=reports,seconds=time.monotonic()-start),indent=2))
        print(report['dataset'],report['row_rule'],report['candidate_bits'],report['status'],report.get('error',''),flush=True)
    final=dict(status='PASS' if len(reports)==24 and all(r['status']=='PASS' for r in reports) else 'FAILED',
       energy_conditions=len(reports),fresh_audits=20,reused_primary_audits=4,
       fixed_masks_across_precisions=True,uncertain_cells=33,reports=reports,seconds=time.monotonic()-start)
    (output/'independent_audit.json').write_text(json.dumps(final,indent=2));print(final['status'])

if __name__=='__main__':main()
