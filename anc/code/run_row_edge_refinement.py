"""Uniform complete-cost dense-row certification on the original ridge target.

All experiments use the same reference, row-edge dual, screening and tree rules.
The run constructs new inputs and witnesses and verifies each committed bound.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
from fractions import Fraction as F
from math import lcm
import sys,json,time,argparse,traceback
import numpy as np,psutil
B=Path(__file__).resolve().parent
from row_edge_input import problem
from planar_oracle import PlanarObservedAnchor
from local_residual import LocalResidualSearch
from row_edge_input import witness_search
from robust_reference import propose
from row_edge_input import prepare
from row_edge_oracle import EdgeOracle,solve
from row_edge_exact import reconstruct
from row_edge_screen import screen,restricted_upper
from edge_polynomial import edge_upper
from observed_anchor import float_up
from audit_sdp_exact import dot,qvalue,denominator,int_array

CONFIG=dict(version='v22',target=1e-8,complete_cap_seconds=1800,root_seconds=180,
    node_seconds=90,max_nodes=65,reference_iterations=8,reference_starts=32,
    witness_starts=4096,witness_seed=92901,screen_depth=4,screen_target_fraction='99/100',
    oracle_scaled_tolerance='1/10000',numeric_early_gap=8e-9,library_threads=1,
    split='largest row residual variance; dyadic mean clipped to inner 70 percent',
    cost='all input, anchor, feasible search, reference, numerical proposal, screening and exact bound evaluation; independent replay audit separately',
    errors='retain valid parent, record failure and continue other conditions')

def make_input(cid):
    names=['dense_diabetes','dense_synthetic_4','dense_synthetic_8','dense_synthetic_16']
    args,meta=problem(names[cid]);a=PlanarObservedAnchor(*args);s=LocalResidualSearch(a)
    witness_search(s,starts=CONFIG['witness_starts'],seed=CONFIG['witness_seed'])
    theta,reference=propose(s,s.witness,iterations=CONFIG['reference_iterations'],starts=CONFIG['reference_starts'])
    x=np.array(s.witness['x']);x0=np.where(x>=(a.lo+a.hi)/2,a.hi,a.lo);ids=s.ids.tolist()
    original=dict(lo=a.lo.tolist(),hi=a.hi.tolist(),y=a.y.tolist(),beta=a.beta.tolist(),lambda_value=a.lam)
    source=dict(theta=theta.tolist(),T=a.t.tolist(),mu=a.mu,active_cells=ids,x0_uncertain=[float(x0[i,j]) for i,j in ids])
    return dict(original=original,problem=source,witness=s.witness,result=dict(lower=s.best),meta=meta,reference=reference)

def audit_input(data):
    o=data['original'];s=data['problem'];lo=np.array(o['lo']);hi=np.array(o['hi']);n,p=lo.shape
    assert np.all(np.isfinite(lo)) and np.all(lo<=hi)
    ids=list(map(tuple,s['active_cells']));assert set(ids)==set(map(tuple,np.argwhere(lo!=hi)))
    x=(lo+hi)/2
    for (i,j),v in zip(ids,s['x0_uncertain']):assert v in (lo[i,j],hi[i,j]);x[i,j]=v
    fixed=np.all(lo==hi,axis=1);rows=np.flatnonzero(~fixed);lam=F(o['lambda_value']);assert lam>0
    beta=list(map(F,o['beta']));y=list(map(F,o['y']));dx=denominator(x.flat);dy=denominator(y)
    X=int_array(x,dx);Y=np.array([int(v*dy) for v in y],object);den=n*dx*dx;Gi=X[fixed].T@X[fixed]
    assert (lam*den).denominator==1
    for j in range(p):Gi[j,j]+=int(lam*den)
    G=[[F(int(v),den) for v in row] for row in Gi];r=[F(int(v),n*dx*dy) for v in X[fixed].T@Y[fixed]]
    T=np.array(s['T']);dt=denominator(T.flat);Ti=int_array(T,dt);metric=Ti.T@Gi@Ti;metric_den=den*dt*dt
    exact_mu=min(F(int(metric[j,j]-sum(abs(metric[j,k]) for k in range(p) if j!=k)),metric_den) for j in range(p));assert 0<F(s['mu'])<=exact_mu
    wx=np.array(data['witness']['x']);theta=list(map(F,data['witness']['theta']));assert wx.shape==lo.shape and np.all(wx>=lo) and np.all(wx<=hi)
    lower=qvalue(G,r,beta)-qvalue(G,r,theta)
    for i in rows:
        row=list(map(F,wx[i]));lower+=((dot(row,beta)-y[i])**2-(dot(row,theta)-y[i])**2)/n
    assert F(data['result']['lower'])<=lower
    return dict(status='PASS',exact_metric=True,original_witness=True,exact_witness_value=str(lower),uncertain_rows=len(rows),uncertain_cells=len(ids))

def full_cover(k):
    return dict(dimension=k,interval_scale=1,edges=[[j,v,0,1] for j in range(k) for v in range(1<<k) if not(v&(1<<j))])

def exact_proposal(data,proposal,covers=None,domains=None):
    const,polys,psd=reconstruct(data,proposal);value=const+F(proposal['alpha'])/10**6;bounds=[];stats=[]
    for i,(k,poly,residual) in enumerate(polys):
        if covers is None:bound,stat=edge_upper(poly,k,tolerance=F(1,10000))
        else:bound,stat=restricted_upper(poly,k,covers[i],residual,None if domains[i] is None else list(map(F,domains[i])))
        if bound is None:return None,polys,dict(status='EXACT_EMPTY',empty_row=i)
        bounds.append(bound);value+=bound/10**6;stats.append(stat)
    return value,polys,dict(status='PASS',psd=psd,exact_upper=str(value),scaled_row_uppers=list(map(str,bounds)),rows=stats)

def run(cid,out):
    start=time.monotonic();elapsed=lambda:time.monotonic()-start
    data=make_input(cid);(out/'input.json').write_text(json.dumps(data));input_audit=audit_input(data)
    (out/'input_audit.json').write_text(json.dumps(input_audit));rows,constant,coupling=prepare(data)
    lower=F(data['result']['lower']);threshold=lower+F(99,100)*F(CONFIG['target']);target=F(CONFIG['target'])
    nodes={};leaves=[];history=[];errors=[];root_upper=None;covers=None;screen_record=None
    def save(status):
        if not leaves:
            r=dict(status='ERROR',condition=cid,meta=data['meta'],target=float(target),target_met=False,target_met_in_budget=False,total_seconds=elapsed(),errors=errors,reason='No completed verified root certificate')
            (out/'result.json').write_text(json.dumps(r,indent=2));return r
        upper=max(threshold,max((F(nodes[i]['exact_upper']) for i in leaves),default=threshold))
        if root_upper is not None:upper=min(root_upper,upper)
        r=dict(status=status,condition=cid,meta=data['meta'],target=float(target),lower=float(lower),upper=float_up(upper),absolute_gap=float_up(upper-lower),target_met=upper-lower<=target,
           total_seconds=elapsed(),cap_seconds=CONFIG['complete_cap_seconds'],target_met_in_budget=upper-lower<=target and elapsed()<=CONFIG['complete_cap_seconds'],
           protocol=CONFIG,active_leaves=leaves,nodes=[{k:v for k,v in d.items() if k!='result'} for d in nodes.values()],history=history,errors=errors,
           exact_upper=str(upper),exact_lower=str(lower),screen_threshold=str(threshold))
        (out/'result.json').write_text(json.dumps(r,indent=2));return r
    try:
        raw_covers=[full_cover(len(row['jj'])) for row in rows];full_oracles=[EdgeOracle(row,cover) for row,cover in zip(rows,raw_covers)]
        m=len(rows);p=len(rows[0]['beta']);initial=dict(alpha=1.,linear=[np.zeros(p).tolist() for _ in rows],quadratic=[((m+.25)*coupling*np.eye(p)).tolist() for _ in rows],gscale=rows[0]['gscale'])
        root=solve(rows,full_oracles,constant,coupling,[None]*m,seconds=CONFIG['root_seconds'],initial=initial,lower=float(lower))
        root_upper,polys,checked=exact_proposal(data,root['proposal']);assert root_upper is not None and root_upper>=lower
        root.update(exact=checked);(out/'node_000.json').write_text(json.dumps(root))
        nodes[0]=dict(id=0,parent=None,domains=[None]*m,exact_upper=str(root_upper),result=root,completed_seconds=elapsed());leaves=[0]
        history.append(dict(seconds=elapsed(),lower=float(lower),upper=float_up(root_upper),stage='root'));save('RUNNING')
        print(json.dumps(dict(condition=cid,node=0,gap=float_up(root_upper-lower),seconds=elapsed())),flush=True)
        if root_upper-lower<=target:return save('PASS')
        del full_oracles,raw_covers
        bounds=list(map(F,checked['scaled_row_uppers']));covers=[];at=elapsed()
        for (k,poly,residual),b in zip(polys,bounds):covers.append(screen(poly,k,b-(root_upper-threshold)*10**6,CONFIG['screen_depth']))
        screen_record=dict(root_node=0,threshold=str(threshold),root_upper=str(root_upper),rows=covers,seconds=elapsed()-at)
        (out/'screen.json').write_text(json.dumps(screen_record));oracles=[EdgeOracle(row,cover) for row,cover in zip(rows,covers)]
        def evaluate(domains,parent):
            idx=len(nodes);prior=nodes[parent]['result'];ans=solve(rows,oracles,constant,coupling,domains,seconds=min(CONFIG['node_seconds'],max(1,CONFIG['complete_cap_seconds']-elapsed())),initial=prior.get('proposal'),initial_points=prior.get('points'),lower=float(lower))
            if 'proposal' not in ans:raise RuntimeError('Empty numerical domain needs an explicit exact certificate')
            value,_,cert=exact_proposal(data,ans['proposal'],covers,domains)
            if value is None:value=threshold
            ans['exact']=cert;(out/f'node_{idx:03d}.json').write_text(json.dumps(ans));upper=min(value,F(nodes[parent]['exact_upper']))
            nodes[idx]=dict(id=idx,parent=parent,domains=domains,exact_upper=str(upper),result=ans,completed_seconds=elapsed())
            print(json.dumps(dict(condition=cid,node=idx,parent=parent,gap=float_up(upper-lower),seconds=elapsed())),flush=True);return idx
        while max(F(nodes[i]['exact_upper']) for i in leaves)>threshold and elapsed()<CONFIG['complete_cap_seconds'] and len(nodes)+2<=CONFIG['max_nodes']:
            parent=max(leaves,key=lambda i:F(nodes[i]['exact_upper']));node=nodes[parent];options=[]
            for c in node['result']['branch_candidates']:
                i=c['row'];ids,left,right=oracles[i].intervals(node['domains'][i]);et=oracles[i].et[ids];slope=oracles[i].slope[ids]
                low=float(np.min(np.minimum(et+left*slope,et+right*slope)));high=float(np.max(np.maximum(et+left*slope,et+right*slope)))
                if high-low>1e-10:options.append((c['variance'],i,low,high,c['mean']))
            if not options:break
            _,row,low,high,mean=max(options);cut=float(np.round(np.clip(mean,low+.15*(high-low),high-.15*(high-low))*2**32)/2**32)
            bounds=node['domains'][row]
            if bounds is None:bounds=[float(np.min(rows[row]['et']))-1e-12,float(np.max(rows[row]['et']))+1e-12]
            children=[]
            for side in (0,1):
                if elapsed()>=CONFIG['complete_cap_seconds']:break
                domains=[None if d is None else list(d) for d in node['domains']];domains[row]=[bounds[0],cut] if side==0 else [cut,bounds[1]]
                children.append(evaluate(domains,parent))
            if len(children)!=2:break
            node.update(children=children,split=dict(row=row,cut=cut,original_interval=bounds));leaves.remove(parent);leaves.extend(children)
            upper=min(root_upper,max(threshold,max(F(nodes[i]['exact_upper']) for i in leaves)))
            history.append(dict(seconds=elapsed(),lower=float(lower),upper=float_up(upper),stage='exact_residual_cover'));save('RUNNING')
    except Exception as e:errors.append(dict(error=str(e),traceback=traceback.format_exc()))
    return save('PASS' if not errors else 'COMPLETED_WITH_ERRORS')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',type=int,required=True);p.add_argument('--cpu',type=int,required=True);p.add_argument('--output',default='dense_edge');a=p.parse_args();psutil.Process().cpu_affinity([a.cpu])
    out=B/a.output/f'condition_{a.condition:02d}';out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(CONFIG,indent=2))
    try:r=run(a.condition,out)
    except Exception as e:r=dict(status='ERROR',condition=a.condition,error=str(e),traceback=traceback.format_exc());(out/'result.json').write_text(json.dumps(r,indent=2))
    print(json.dumps({k:v for k,v in r.items() if k not in ('nodes','history','protocol','exact_upper','exact_lower')}),flush=True)
