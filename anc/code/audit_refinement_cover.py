"""Exact original witness and completed subdivision-cover verification."""
from pathlib import Path
from fractions import Fraction as F
import json,sys,traceback
import numpy as np
from audit_sdp_exact import qvalue,dot,int_array,denominator
B=Path(__file__).resolve().parent

def audit(file):
    result=json.loads(file.read_text());instance=result['instance']
    original=json.loads((file.parent/f'{instance}_adaptive_energy.json').read_text())
    cert=original['certificate'];lo=np.array(cert['lo']);hi=np.array(cert['hi']);n,p=lo.shape
    entries={r['id']:r for r in result['tree']};domains={0:(lo,hi)};frontier=[0] if entries else []
    raw_uppers={};covered=[];polynomial_audits=[];witnesses=[cert['witness']]
    while frontier:
        node=frontier.pop();entry=entries[node];receipt=json.loads((file.parent/entry['receipt']).read_text())
        problem=receipt['problem'];ll,uu=domains[node]
        assert np.array_equal(problem['domain_lo'],ll) and np.array_equal(problem['domain_hi'],uu)
        inherited=original['upper'] if entry['parent'] is None else raw_uppers[entry['parent']]
        raw_uppers[node]=min(F(receipt['result']['upper']),F(inherited))
        assert F(entry['upper'])>=raw_uppers[node] or entry['parent'] is None
        witnesses.append(receipt['witness'])
        if 'children' in entry:
            i,j,cut=entry['split'];assert F(ll[i,j])<F(cut)<F(uu[i,j]);assert len(entry['children'])==2
            for side,child in enumerate(entry['children']):
                cl=ll.copy();cu=uu.copy()
                if side==0:cu[i,j]=cut
                else:cl[i,j]=cut
                domains[child]=(cl,cu);frontier.append(child)
        else:covered.append(node)
    assert set(covered)==set(result['active_leaves'])
    expected_upper=min(F(original['upper']),max((raw_uppers[v] for v in covered),default=F(original['upper'])))
    assert F(result['upper'])>=expected_upper
    witnessfile=file.parent/f'{instance}_witness.json'
    if witnessfile.exists():witnesses.append(json.loads(witnessfile.read_text()))
    beta=list(map(F,cert['beta']));yy=list(map(F,cert['y']));lam=F(cert['lambda_value']);exact_lowers=[]
    for witness in witnesses:
        x=np.array(witness['x']);theta=list(map(F,witness['theta']))
        assert x.shape==lo.shape and np.all(x>=lo) and np.all(x<=hi)
        dx=denominator(x.flat);dt=denominator(theta+beta);dy=denominator(yy)
        Xi=int_array(x,dx);bi=np.array([int(v*dt) for v in beta],dtype=object);ti=np.array([int(v*dt) for v in theta],dtype=object)
        # All denominators are powers of two, so this common scale is exact.
        d=max(dx*dt,dy);yi=np.array([int(v*d) for v in yy],dtype=object)
        eb=(Xi@bi)*(d//(dx*dt))-yi;et=(Xi@ti)*(d//(dx*dt))-yi
        value=F(int(eb@eb-et@et),n*d*d)+lam*(dot(beta,beta)-dot(theta,theta));exact_lowers.append(value)
    assert F(result['lower'])<=max(exact_lowers),('witness_lower',result['lower'],float(max(exact_lowers)))
    assert result['target_met']==(result['upper']-result['lower']<=result['target'])
    for checkpoint,event in result['checkpoints'].items():
        candidates=[e for e in result['trace'] if e['seconds']<=float(checkpoint)]
        assert event==(candidates[-1] if candidates else None)
    return dict(status='PASS',instance=instance,covered_leaves=len(covered),evaluated_nodes=len(entries),
                exact_original_witnesses=len(exact_lowers),target_met=result['target_met'],completed_deadlines=True)

if __name__=='__main__':
    folder=B/(sys.argv[1] if len(sys.argv)>1 else 'frozen_validation');reports=[]
    for file in sorted(folder.glob('*_result.json')):
        try:r=audit(file)
        except Exception as exc:r=dict(status='ERROR',file=file.name,error=str(exc),traceback=traceback.format_exc())
        reports.append(r);print(json.dumps(r),flush=True)
    (folder/'coverage_audit.json').write_text(json.dumps(dict(status='PASS' if reports and all(r['status']=='PASS' for r in reports) else 'ERROR',reports=reports),indent=2))
