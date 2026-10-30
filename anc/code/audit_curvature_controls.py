"""Independent normal-equation checks of enhanced baselines on original rows."""
from pathlib import Path
from fractions import Fraction as F
import json

def audit():
    data=json.loads((Path(__file__).resolve().parent.parent/'results/energy_mechanism/curvature_controls.json').read_text())
    reports=[];n=128;m=64;alpha=F(1,32);lam=F(1,2)
    for r in data['scales']:
      s=r['scale'];mu=F(r['full_metric_mu'])
      for pattern in (0,1,2):
        us=[F(3,4) if pattern==0 else F((3,4,5)[i%3],4) if pattern==1 else 1+F((-1)**i,8) for i in range(m)]
        ts=[F(s*16) if pattern==0 else F(s*8*(1+i%2)) if pattern==1 else s*8*F(5+2*(i%2),4) for i in range(m)]
        m00=lam+sum((u*u/n for u in us),F(0))+m*alpha*alpha/n
        cc=[alpha*t/n for t in ts];dd=[lam+t*t/n for t in ts]
        g=lam+sum((u*(u-2)/n for u in us),F(0))
        schur=m00-sum((c*c/d for c,d in zip(cc,dd)),F(0))
        omitted=g*g/schur-g*g/m00
        d_trans=[d/(s*s) for d in dd];c_trans=[c/s for c in cc]
        assert all(d>mu for d in d_trans)
        # This exact Schur complement checks the whole 65-dimensional transformed Gram.
        assert m00-mu-sum((c*c/(d-mu) for c,d in zip(c_trans,d_trans)),F(0))>0
        assert 0<=omitted<=F(r['energy']) and omitted<=F(r['enhanced_scalar']) and omitted<=F(r['metric_moment'])
        reports.append({'scale':s,'pattern':pattern,'normal_equations':'PASS','full_metric_lower':'PASS'})
    return {'status':'PASS','original_normal_equation_cases':len(reports),'reports':reports}

if __name__=='__main__':print(json.dumps(audit()))
