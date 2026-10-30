"""Numerical small-cone reformulation of the row-edge coupling dual.

The row coordinates are exact algebraic coordinates before floating evaluation.
Numerical proposals require a separate rational certificate before publication.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse,json,sys,time,warnings,traceback,copy
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B/'source/code'))
import numpy as np, cvxpy as cp, psutil
from row_edge_input import prepare,at_point
from row_edge_oracle import EdgeOracle


def coordinates(row,z):
    z=np.asarray(z);dz=z*row['ds'];q=dz@row['theta'][row['jj']]
    e0=row['x0']@row['theta']-row['y']
    full=np.concatenate(([q],dz*(e0+q)))*row['coordinate_scale']/(row['n']*row['gscale'])
    return full[row.get('coordinate_indices',np.arange(len(full))) ]


def prepare_compressed(data):
    rows,constant,coupling=prepare(data)
    if data.get('fixed_gscale'):
        old=rows[0]['gscale'];new=data['fixed_gscale'];assert new>0
        for r in rows:r['gamma']*=old/new;r['gscale']=new
        coupling*=new*new/(old*old)
    origins=[r['gamma'][0].copy() for r in rows]
    g0=np.sum(origins,axis=0);constant+=coupling*(g0@g0)
    oracles=[];bases=[];points=[];max_error=0.
    for i,row in enumerate(rows):
        k=len(row['jj']);assert k<=3
        raw=np.column_stack([row['T'].T@row['x0']]+[row['T'][j] for j in row['jj']])
        norms=np.max(np.abs(raw),axis=0);indices=np.flatnonzero(norms>0)
        scales=2.**np.ceil(np.log2(np.where(norms>0,norms,1.)))
        basis=(raw/scales)[:,indices];row['coordinate_scale']=scales;row['coordinate_indices']=indices
        edges=[[j,v,0,1] for j in range(k) for v in range(1<<k) if not (v&(1<<j))]
        oracle=EdgeOracle(row,dict(edges=edges,interval_scale=1))
        partition_theta=np.asarray(data.get('partition_theta',data['problem']['theta']))
        row['partition_theta']=partition_theta
        oracle.et=row['x'][oracle.vertex]@partition_theta-row['y']
        oracle.slope=row['ds'][oracle.free]*partition_theta[row['jj'][oracle.free]]
        z0=np.asarray([oracle.point(t,0) for t in range(len(edges))])
        zm=np.asarray([oracle.point(t,.5) for t in range(len(edges))])
        z1=np.asarray([oracle.point(t,1) for t in range(len(edges))])
        c0=np.asarray([coordinates(row,z) for z in z0]);cm=np.asarray([coordinates(row,z) for z in zm]);c1=np.asarray([coordinates(row,z) for z in z1])
        for z,u in zip(zm,cm):max_error=max(max_error,float(np.max(np.abs(at_point(row,z)[0]-origins[i]-basis@u))))
        oracle.g0=c0;oracle.g2=2*(c1-2*cm+c0);oracle.g1=c1-c0-oracle.g2
        linear=2*coupling*(basis.T@g0)
        oracle.f0+=oracle.g0@linear;oracle.f1+=oracle.g1@linear;oracle.f2+=oracle.g2@linear
        oracles.append(oracle);bases.append(basis);points.append(list({tuple(z):z for z in oracle.seeds(None)}.values()))
    return rows,constant,coupling,g0,origins,bases,oracles,points,max_error


def recover(H,d,bases,coupling,constant,oracles,domains):
    """Small Schur complement proposal repair, with several uniform margins."""
    best=None;p=bases[0].shape[0]
    H=[(h+h.T)/2 for h in H]
    for margin in (1e-8,1e-6,1e-4,1e-2):
        hh=[h+max(margin,-float(np.linalg.eigvalsh(h)[0])+margin)*np.eye(len(h)) for h in H]
        inv=[np.linalg.inv(h) for h in hh]
        R=sum(v@w@v.T for v,w in zip(bases,inv))
        scale=max(1.,coupling*float(np.linalg.eigvalsh(R)[-1]))*(1+margin)
        hh=[h*scale for h in hh];inv=[w/scale for w in inv]
        J=np.eye(p)/coupling-sum(v@w@v.T for v,w in zip(bases,inv))
        t=sum(v@w@di for v,w,di in zip(bases,inv,d))
        a=sum(di@w@di for w,di in zip(inv,d))+t@np.linalg.solve(J,t)+margin
        vals=[o.evaluate(h,di,dom,tolerance=1e-6) for o,h,di,dom in zip(oracles,hh,d,domains)]
        upper=(constant+a+sum(v['upper'] for v in vals))/1e6
        if best is None or upper<best['upper']:
            best=dict(upper=upper,alpha=float(a),quadratic=[h.tolist() for h in hh],linear=[v.tolist() for v in d],repair_margin=margin,
                      minimum_local_eigenvalue=min(float(np.linalg.eigvalsh(h)[0]) for h in hh),
                      minimum_schur_eigenvalue=float(np.linalg.eigvalsh(J)[0]),row_oracles=[{k:v for k,v in r.items() if k!='points'} for r in vals])
    return best


def solve(data,seconds=180,solver='CLARABEL',domains=None,initial_points=None,lower=-np.inf,max_iterations=25,emit=None):
    start=time.monotonic()
    rows,constant,coupling,g0,origins,bases,oracles,points,max_error=prepare_compressed(data)
    m=len(rows);p=bases[0].shape[0];ranks=[q.shape[1] for q in bases]
    cells=[tuple(v) for v in data['problem']['active_cells']];cell_index={v:i for i,v in enumerate(cells)}
    halfspaces=[cut for cut in data['problem'].get('extra_linear_inequalities',[]) if all(len(key)==1 for key,co in cut['terms'])]
    whole=np.zeros((len(halfspaces),len(cells)));rhs=np.asarray([c['rhs'] for c in halfspaces])
    for j,cut in enumerate(halfspaces):
        for key,co in cut['terms']:whole[j,key[0]]+=co
    local_a=[whole[:,[cell_index[(r['row'],int(j))] for j in r['jj']]] for r in rows]
    multiplier=cp.Variable(len(halfspaces),nonneg=True) if halfspaces else None
    if domains is None:domains=[None]*m
    if any(d is not None for d in domains):points=[o.seeds(dom) for o,dom in zip(oracles,domains)]
    if initial_points is not None:
        for i,pts in enumerate(initial_points):
            for pt in pts:
                z=np.array(pt);r=rows[i];ptheta=r['partition_theta'];et=r['x0']@ptheta-r['y']+z@(r['ds']*ptheta[r['jj']])
                if domains[i] is None or domains[i][0]<=et<=domains[i][1]:points[i].append(z)
    points=[list({tuple(z):z for z in pts}.values()) for pts in points]
    alpha=cp.Variable();tau=cp.Variable(m);lin=[cp.Variable(r) for r in ranks]
    quad=[cp.Variable((r,r),symmetric=True) for r in ranks]
    slabs=[cp.Variable((p+1,p+1),symmetric=True) for _ in rows]
    K=cp.bmat([[cp.reshape(alpha,(1,1),order='C'),np.zeros((1,p))],[np.zeros((p,1)),np.eye(p)]])
    psd_constraints=[K-sum(slabs)>>0]
    for i,r in enumerate(ranks):
        cross=cp.vstack([cp.reshape(lin[i],(1,r),order='C'),np.sqrt(coupling)*bases[i]])
        psd_constraints.append(cp.bmat([[slabs[i],cross],[cross.T,quad[i]]])>>0)
    best=None;history=[]
    for iteration in range(max_iterations):
        constraints=list(psd_constraints);row_constraints=[]
        for i,pts in enumerate(points):
            if not pts:raise ValueError('Empty row domain')
            u=np.array([coordinates(rows[i],z) for z in pts])
            f=np.array([at_point(rows[i],z)[1] for z in pts])+2*coupling*(u@(bases[i].T@g0))
            uu=(u[:,:,None]*u[:,None,:]).reshape(len(u),ranks[i]**2)
            extra=-(np.array(pts)@local_a[i].T)@multiplier if halfspaces else 0.
            con=tau[i]>=f+uu@cp.reshape(quad[i],(ranks[i]**2,),order='C')+2*u@lin[i]+extra
            constraints.append(con);row_constraints.append(con)
        offset=rhs@multiplier if halfspaces else 0.
        problem=cp.Problem(cp.Minimize(constant+alpha+cp.sum(tau)+offset),constraints)
        remaining=seconds-(time.monotonic()-start)
        if remaining<=0:break
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if solver=='CLARABEL':
                problem.solve(solver=solver,max_iter=150,time_limit=min(45,remaining),tol_gap_abs=1e-8,tol_gap_rel=1e-8,tol_feas=1e-8)
            else:problem.solve(solver=solver,eps=1e-7,max_iters=25000,time_limit_secs=min(45,remaining))
        if alpha.value is None:raise RuntimeError(problem.status)
        hs=[q.value for q in quad];ds=[d.value for d in lin]
        lam=np.maximum(0.,multiplier.value) if halfspaces else np.array([])
        adjusted=[]
        for i,oracle in enumerate(oracles):
            current=copy.copy(oracle)
            if halfspaces:
                linear=lam@local_a[i];z0=np.array([oracle.point(j,0) for j in range(len(oracle.free))])
                current.f0=oracle.f0-z0@linear;current.f1=oracle.f1-linear[oracle.free]
            adjusted.append(current)
        proposal=recover(hs,ds,bases,coupling,constant+float(rhs@lam),adjusted,domains)
        proposal['halfspace_multipliers']=lam.tolist()
        branches=[];added=0;mean_table=(np.array(data['original']['lo'])+np.array(data['original']['hi']))/2
        for i,oracle in enumerate(adjusted):
            res=oracle.evaluate(hs[i],ds[i],domains[i],tolerance=1e-7)
            weights=np.maximum(0,np.asarray(row_constraints[i].dual_value));sw=weights.sum()
            row=rows[i];ptheta=row['partition_theta'];et=row['x0']@ptheta-row['y']+np.array(points[i])@(row['ds']*ptheta[row['jj']])
            mean=float(weights@et/sw) if sw else float(np.mean(et));var=float(weights@(et-mean)**2/sw) if sw else float(np.var(et))
            mean_z=weights@np.array(points[i])/sw if sw else np.mean(points[i],axis=0)
            mean_table[row['row']]=row['x0'];mean_table[row['row'],row['jj']]+=row['ds']*np.clip(mean_z,0,1)
            branches.append(dict(row=i,mean=mean,variance=var))
            seen={tuple(z) for z in points[i]}
            for z in res['points']:
                if tuple(z) not in seen:points[i].append(z);seen.add(tuple(z));added+=1
        item=dict(iteration=iteration,upper=proposal['upper'],numeric_relaxation=float(problem.value)/1e6,gap=proposal['upper']-lower,
                  seconds=time.monotonic()-start,added=added,solver_status=problem.status,solver_iterations=problem.solver_stats.num_iters)
        history.append(item)
        if best is None or proposal['upper']<best['upper']:
            n,p0=mean_table.shape;yt=np.array(data['original']['y']);ridge=data['original']['lambda_value']
            reference_fit=np.linalg.solve(mean_table.T@mean_table/n+ridge*np.eye(p0),mean_table.T@yt/n)
            best=dict(proposal,branch_candidates=branches,points=[[z.tolist() for z in pts] for pts in points],
                      reference_fit=reference_fit.tolist(),reference_theta=data['problem']['theta'],partition_theta=data.get('partition_theta',data['problem']['theta']))
        best.update(status='NUMERICAL_PROPOSAL',history=history,seconds=time.monotonic()-start,lower=lower,gap=best['upper']-lower,
                    maximum_coordinate_error=max_error,local_cone_sizes=[p+1+r for r in ranks],central_cone_size=p+1,
                    coordinate_scales=[r['coordinate_scale'].tolist() for r in rows],gscale=rows[0]['gscale'],residual_domains=domains)
        best['coordinate_indices']=[r['coordinate_indices'].tolist() for r in rows]
        if emit:emit(best,item)
        if best['gap']<2.7e-6 or not added:break
    return best


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--folder',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--seconds',type=float,default=180);ap.add_argument('--cpu',type=int,default=10)
    ap.add_argument('--solver',default='CLARABEL');ap.add_argument('--root',type=int)
    args=ap.parse_args();psutil.Process().cpu_affinity([args.cpu]);folder=Path(args.folder);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    result=json.loads((folder/'result.json').read_text());roots=[r for r in result['nodes'] if r['parent'] is None]
    root=max(roots,key=lambda r:r['id']) if args.root is None else next(r for r in result['nodes'] if r['id']==args.root)
    original=json.loads((folder/'original_input.json').read_text());source=json.loads((folder/root['receipt']).read_text())
    data=dict(original=original,problem=source['problem']);(out/'source.json').write_text(json.dumps(dict(folder=str(folder),receipt=root['receipt'])))
    def emit(best,item):
        (out/'proposal.json').write_text(json.dumps(best));print(json.dumps(item),flush=True)
    try:
        best=solve(data,args.seconds,args.solver,lower=result['lower'],emit=emit)
        (out/'summary.json').write_text(json.dumps({k:v for k,v in best.items() if k not in ('linear','quadratic','points','branch_candidates')},indent=2))
    except Exception as exc:
        (out/'error.json').write_text(json.dumps(dict(error=str(exc),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
