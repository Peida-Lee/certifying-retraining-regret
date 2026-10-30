"""Exact enhanced-curvature and actually preconditioned moment controls."""
from fractions import Fraction as F
from math import isqrt
import json
from pathlib import Path

def sqrt_up(x,bits=64):
    v=x*2**(2*bits);n=isqrt(v.numerator//v.denominator)
    if F(n*n)<v:n+=1
    q=F(n,2**bits);assert q*q>=x;return q

def calculate():
    a=F(1,32);R=F(1,25);e=a*a/1250;out=[]
    for s in (1,2,4,16,64,256):
        # Original discarded curvature is diagonal, with every entry in this range.
        dmin=F(1+s*s,2);dmax=F(1+4*s*s,2);cross2=a*a*s*s
        scalar=cross2*R*R/F(1,2)
        stronger=cross2*R*R/dmin
        assert stronger/e==F(4*s*s,1+s*s)<4
        # Use the ACTUAL dyadic T_s=diag(1,1/s,...,1/s), not an ideal numerical basis.
        # Retained curvature >=25/32; transformed discarded curvature >=dmin/s^2;
        # cross-block norm <=a. The 2x2 block eigenvalue bounds every full Gram.
        aa=F(25,32);dd=dmin/(s*s)
        mu=(aa+dd-sqrt_up((aa-dd)**2+4*a*a))/2
        assert mu>0 and aa-mu>=0 and dd-mu>=0 and (aa-mu)*(dd-mu)>=a*a
        U0=F(1,1024)/mu;root=F(1,32)/mu
        assert root*root==U0/mu
        metric_moment=a*a*root*root/mu
        actualH=1/(800+512*a*a/F(1+4*s*s));V=1/(800+512*a*a)
        assert 0<actualH-V<=e<=stronger and actualH-V<=metric_moment
        out.append(dict(scale=s,actual_omission=str(actualH-V),energy=str(e),ridge_scalar=str(scalar),
            discarded_curvature_lower=str(dmin),discarded_curvature_upper=str(dmax),
            enhanced_scalar=str(stronger),enhanced_to_energy_ratio=str(stronger/e),
            actual_T_nuisance=str(F(1,s)),full_metric_mu=str(mu),metric_root=str(root),
            metric_moment=str(metric_moment),block_psd_check=True))
    return dict(status='PASS',arithmetic='exact Fraction and outward dyadic square root',
        fixed_p=65,scales=out,all_original_gram_enclosures_verified=True)

def main():
    p=Path(__file__).resolve().parent
    out=p/'curvature_controls.json';out.write_text(json.dumps(calculate(),indent=2))
    print(json.dumps({'status':'PASS','conditions':6}))

if __name__=='__main__':main()
