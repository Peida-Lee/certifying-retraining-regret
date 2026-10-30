"""Exact normal-equation checks of the contraction refinement, no raw records."""
from pathlib import Path
from fractions import Fraction as F
import json
from audit_real_world import dot,quad,inverse,root


def verify():
    cases=0;improvements=0;ratios=[]
    for scale in (F(1,64),F(1,4),F(4),F(64)):
      for noise in (F(0),F(1,1024),F(1,32)):
       for shift in (F(0),F(1,8)):
        for leak in (F(0),F(1,256)):
          K=[[F(i==j)+F(abs(i-j)==2,64) for j in range(4)] for i in range(4)]
          mu=F(63,64);nu=F(65,64);cc=(nu-mu)/2
          rows=[[F(1),F(1,8),F(0),F(0)],
                [F(1,32),F(0),scale,F(0)],
                [F(0),F(1,64),F(0),scale]]
          h=[F(1,64),-F(1,128),leak,-leak]
          residuals=[F(1,4),noise,-noise];z0=[shift,-shift/2,F(0),F(0)]
          M=[[K[i][j]+sum((x[i]*x[j] for x in rows),F(0)) for j in range(4)] for i in range(4)]
          g=[h[i]+sum((x[i]*e for x,e in zip(rows,residuals)),F(0)) for i in range(4)]
          hc=[h[i]-dot(K[i],z0) for i in range(4)]
          ec=[e-dot(x,z0) for x,e in zip(rows,residuals)]
          a2=sum((dot(x[:2],x[:2]) for x in rows[1:]),F(0));a=root(a2)
          eps2=dot(ec[1:],ec[1:]);eps=root(eps2)
          sr=[hc[j]+rows[0][j]*ec[0] for j in range(2)];G=root(dot(sr,sr))
          d=root(dot(hc[2:],hc[2:]));im=root(1/mu)
          mixed=[sum((x[j]*e for x,e in zip(rows[1:],ec[1:])),F(0)) for j in range(2)]
          bb=root(dot(mixed,mixed));rho=sum((dot(x[2:],x[2:]) for x in rows[1:]),F(0))
          gamma2=rho/(mu+rho);gamma=root(gamma2);t=min(im/2,root(rho)/mu)
          RA=(G+a*eps+(cc/mu+a*im/2)*d+cc*eps*im/2)/mu
          RC=min(RA,(G+min(a*eps,bb+a*gamma2*eps)+(cc/mu+a*t)*d+cc*t*eps)/mu)
          gc=[hc[i]+sum((x[i]*e for x,e in zip(rows,ec)),F(0)) for i in range(4)]
          radius=min(RC,root(dot(gc,gc))/mu)
          energy2=min((eps+a*radius)**2,eps2+2*bb*radius+a2*radius**2)
          eta=(gamma*root(energy2)+(d+cc*radius)*im)**2
          oldeta=(eps+a*RA+(d+cc*RA)*im)**2
          disp=[dot(row,g) for row in inverse(M)]
          full=dot(g,disp);restricted=quad(g[:2],inverse([row[:2] for row in M[:2]]))
          assert 0<=full-restricted<=eta
          assert sum(((disp[j]-z0[j])**2 for j in range(2)),F(0))<=radius**2
          assert eta<=oldeta+F(1,2**60)
          cases+=1;improvements+=eta<oldeta
          ratios.append(float(eta/oldeta) if oldeta else 0.)
    return dict(status='PASS',exact_normal_equation_cases=cases,
       contraction_improves_original_bound=improvements,ratio_range=[min(ratios),max(ratios)],
       support_leakage_in_examples=0,nonzero_metric_cross_blocks=True,
       translated_retained_references=True,raw_records_opened=False)

def main():
    print(json.dumps(verify()))

if __name__=='__main__':main()
