"""Fixed rank curves and row/penalty controls with verified retained Gram bounds."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,csv,json,time,traceback
import numpy as np,psutil
from public_problems import problem
from observed_anchor import ObservedAnchor
from moment_tail import basis_proposal,tail_receipt

B=Path(__file__).resolve().parent

def snapshot(a):
    rng=np.random.default_rng(77031);ds=[]
    for index in range(9):
        x=a.c if index==0 else a.lo+(a.hi-a.lo)*rng.random(a.lo.shape)
        ds.append(np.linalg.solve(x.T@x/a.n+a.lam*np.eye(a.p),x.T@a.y/a.n)-a.beta)
    physical,_,_=np.linalg.svd(np.array(ds).T,full_matrices=False)
    mapped=np.linalg.solve(a.t,physical);mapped/=np.linalg.norm(mapped,axis=0)
    return np.round(mapped*2**24)/2**24

def run(out,cpu):
    psutil.Process().cpu_affinity([cpu]);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();records=[];errors=[]
    conditions=[(n,p,64,1.,'rank_curve') for n in ('diabetes','concrete','superconductivity') for p in ('concentrated','dispersed')]
    conditions += [(n,p,r,l,'sensitivity') for n in ('concrete','superconductivity') for p in ('concentrated','dispersed') for r in (64,128) for l in (.5,1.,2.)]
    for name,spread,nrows,mult,phase in conditions:
        try:
            tick=time.monotonic();args,meta=problem(name=name,spread=spread,cells=2*nrows)
            lo,hi,y,_,lam=args;lam*=mult;c=(lo+hi)/2
            beta=np.linalg.solve(c.T@c/len(c)+lam*np.eye(c.shape[1]),c.T@y/len(c))
            a=ObservedAnchor(lo,hi,y,beta,lam,anchor='observed',preconditioner='midpoint')
            common=time.monotonic()-tick
            for method in ('residual_gradient','snapshot_pod'):
                tick=time.monotonic();proposal=basis_proposal(a) if method=='residual_gradient' else snapshot(a)
                proposal_seconds=time.monotonic()-tick
                for rank in ((1,2,3,4,6) if phase=='rank_curve' else (3,)):
                    try:
                        tick=time.monotonic();r=proposal[:,:rank];t=tail_receipt(a,r)
                        g,coupling,radius=t['gradient_norm'],t['coupling_norm'],t['coordinate_radius']
                        terms=[g*g/a.mu,2*g*coupling*radius/a.mu,(coupling*radius)**2/a.mu]
                        row=dict(phase=phase,dataset=name,spread=spread,uncertain_rows=nrows,uncertain_cells=2*nrows,
                            lam=lam,method=method,rank=rank,G_r=g,C_r=coupling,R_s=radius,mu=a.mu,
                            eta_r=t['tail_bound'],gradient_term=terms[0],mixed_term=terms[1],coupling_term=terms[2],
                            dominant=('gradient','mixed','coupling')[int(np.argmax(terms))],
                            Hmu=t['Hmu'],gram_lower_method=t['gram_lower_method'],
                            common_seconds=common,proposal_seconds=proposal_seconds,verification_seconds=time.monotonic()-tick,
                            basis=r.tolist(),gram=[[str(v) for v in q] for q in t['qH']])
                        records.append(row)
                    except Exception as exc:errors.append(dict(dataset=name,spread=spread,phase=phase,rows=nrows,lam=lam,method=method,rank=rank,error=str(exc)))
        except Exception as exc:errors.append(dict(dataset=name,spread=spread,phase=phase,rows=nrows,lam_multiplier=mult,error=str(exc),traceback=traceback.format_exc()))
        report=dict(status='RUNNING',records=records,errors=errors,seconds=time.monotonic()-started,cpu=cpu)
        (out/'results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(dataset=name,spread=spread,rows=nrows,lambda_multiplier=mult,phase=phase,records=len(records),errors=len(errors))),flush=True)
    report.update(status='PASS' if not errors and len(records)==108 else 'COMPLETED_WITH_ERRORS',seconds=time.monotonic()-started)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    scalar=[{k:v for k,v in row.items() if k not in ('basis','gram')} for row in records]
    with (out/'diagnostics.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(scalar[0]));writer.writeheader();writer.writerows(scalar)
    print(json.dumps(dict(status=report['status'],records=len(records),errors=len(errors),seconds=report['seconds'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=B/'tail_diagnostics');p.add_argument('--cpu',type=int,default=10)
    a=p.parse_args();run(a.output,a.cpu)
