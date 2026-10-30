"""Algebra and rejection tests for the small-cone certificate, without raw data."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
from fractions import Fraction as F
import sys,json,copy
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'source/code'))
import numpy as np,psutil
from observed_anchor import ObservedAnchor
from local_residual import LocalResidualSearch
from compressed_edge_arrow_v27 import prepare_compressed,coordinates,recover
from audit_compressed_arrow_v27 import reconstruct,positive_ldl,dot,verify
from uniform_curvature_v28 import choose,curvature


def main():
    allowed=psutil.Process().cpu_affinity()
    psutil.Process().cpu_affinity([12] if 12 in allowed else [allowed[-1]])
    rng=np.random.default_rng(92927);cases=0;curvatures=0
    for k in (1,2,3):
        n,p=16,5;x=rng.integers(-4,5,(n,p))/8;x[:,0]=1;y=rng.integers(-8,9,n)/8
        beta=rng.integers(-3,4,p)/16;lo=x.copy();hi=x.copy();lo[:3,1:k+1]-=1/16;hi[:3,1:k+1]+=1/16
        a=ObservedAnchor(lo,hi,y,beta,1/8);search=LocalResidualSearch(a);search.lower(x,search.fit(x));ids=search.ids.tolist()
        data=dict(original=dict(lo=lo.tolist(),hi=hi.tolist(),y=y.tolist(),beta=beta.tolist(),lambda_value=1/8),
                  problem=dict(T=a.t.tolist(),mu=a.mu,theta=search.fit(x).tolist(),active_cells=ids,x0_uncertain=[lo[i,j] for i,j in ids]),
                  witness=search.witness,result=dict(lower=search.best,target=3e-6))
        for metric in (data,choose(data,.5)):
            rows,const,c,g0,origins,bases,oracles,pts,error=prepare_compressed(metric)
            H=[np.eye(k+1)*1e4 for _ in rows];d=[np.zeros(k+1) for _ in rows]
            proposal=recover(H,d,bases,c,const,oracles,[None]*len(rows));proposal.update(coordinate_scales=[r['coordinate_scale'].tolist() for r in rows],gscale=rows[0]['gscale'],residual_domains=[None]*len(rows),halfspace_multipliers=[])
            constant,polys,psd=reconstruct(metric,proposal);checked=verify(metric,proposal)
            assert checked['status']=='EXACT_AUDIT_PASS' and error<1e-11
            for rep in range(5):
                z=[rng.integers(0,17,k)/16 for _ in rows];direct=const+proposal['alpha'];reconstructed=constant+F(proposal['alpha'])
                for i,(row,zi) in enumerate(zip(rows,z)):
                    xx=row['x0'].copy();xx[row['jj']]+=zi*row['ds'];eb=xx@row['beta']-row['y'];et=xx@row['theta']-row['y'];u=coordinates(row,zi)
                    direct+=(eb*eb-et*et)/n*1e6+2*c*u@(bases[i].T@g0)+u@np.array(proposal['quadratic'][i])@u+2*u@proposal['linear'][i]
                    val=F(0)
                    for mon,co in polys[i][1].items():
                        term=co
                        for j in mon:term*=F(zi[j])
                        val+=term
                    reconstructed+=val
                assert abs(float(reconstructed)-direct)<=1e-8*max(1,abs(direct));cases+=1
            invalid=copy.deepcopy(proposal);invalid['quadratic'][0][0][0]=-1
            try:reconstruct(metric,invalid)
            except AssertionError:pass
            else:raise AssertionError('Indefinite local certificate accepted')
        for tau in (.25,.5,.75):
            lower,_=curvature(data['original'],tau)
            for rep in range(4):
                xx=lo+(hi-lo)*rng.integers(0,17,(n,p))/16;xf=[[F(v) for v in row] for row in xx]
                M=[[sum((row[i]*row[j] for row in xf),F(0))/n+F(i==j,8)-lower[i][j] for j in range(p)] for i in range(p)]
                for j in range(p):M[j][j]+=F(1,2**50)
                positive_ldl(M);curvatures+=1
    report=dict(status='PASS',polynomial_identity_cases=cases,uniform_curvature_cases=curvatures,indefinite_rejections=6,source='synthetic original rows',raw_records_opened=False)
    (B/'compressed_offline_v29.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
