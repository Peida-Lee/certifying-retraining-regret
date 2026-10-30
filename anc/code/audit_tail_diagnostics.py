"""Independently check exact retained-Gram positivity and diagnostic arithmetic."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,math

def inverse(A):
    n=len(A);v=[row[:]+[F(i==j) for j in range(n)] for i,row in enumerate(A)]
    for k in range(n):
        pivot=next(i for i in range(k,n) if v[i][k]);v[k],v[pivot]=v[pivot],v[k]
        value=v[k][k];v[k]=[x/value for x in v[k]]
        for i in range(n):
            if i!=k:
                value=v[i][k];v[i]=[x-value*y for x,y in zip(v[i],v[k])]
    return [r[n:] for r in v]

def audit(folder):
    data=json.loads((folder/'results.json').read_text());assert data['status']=='PASS' and not data['errors']
    counts={'rank_curve':0,'sensitivity':0};fallback=[]
    for row in data['records']:
        R=[[F(v) for v in r] for r in row['basis']];H=[[F(v) for v in r] for r in row['gram']];n=row['rank']
        assert H==[[sum((r[i]*r[j] for r in R),F(0)) for j in range(n)] for i in range(n)]
        bound=F(row['Hmu']);assert bound>0
        if row['gram_lower_method']=='gershgorin':
            assert bound<=min(H[i][i]-sum(abs(H[i][j]) for j in range(n) if i!=j) for i in range(n))
        else:
            assert row['gram_lower_method']=='exact_inverse_norm';inv=inverse(H)
            assert all(sum(H[i][k]*inv[k][j] for k in range(n))==int(i==j) for i in range(n) for j in range(n))
            assert bound<=1/max(sum(abs(v) for v in r) for r in inv)
            fallback.append(dict(dataset=row['dataset'],spread=row['spread'],rank=n,Hmu=row['Hmu']))
        assert math.isclose(row['eta_r'],sum(row[k] for k in ('gradient_term','mixed_term','coupling_term')),rel_tol=1e-12)
        counts[row['phase']]+=1
    assert counts=={'rank_curve':60,'sensitivity':48}
    report=dict(status='PASS',counts=counts,exact_inverse_norm_cases=fallback,scope='Exact Gram construction and positivity; decomposition arithmetic')
    (folder/'independent_gram_audit.json').write_text(json.dumps(report,indent=2));return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder)))
