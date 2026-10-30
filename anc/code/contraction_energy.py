"""Observed-anchor energy with verified nuisance contraction and mixed moments.

The original energy certificate remains the scale-independent limiting bound.
The extra receipt uses the actual orthogonal discarded projection of each row.
"""
from fractions import Fraction as F
import itertools
import numpy as np
from centered_signal_energy import (
    constants as previous_constants, exact_geometry, dot, qform, product_box,
    sqrt_upper, float_up, CenteredSignalSearch,
)
from observed_anchor import I,mm,interval_q,eigen_gershgorin


def constants(a, R, meta):
    rec = previous_constants(a, R, meta)
    qr, cols, H, Hi, hmu, U, ucols, qt = exact_geometry(a, R)
    reference = list(map(F, rec['reference_beta']))
    r = len(cols)
    rho = F(0)
    mixed_lo = [F(0) for _ in range(r)]
    mixed_hi = [F(0) for _ in range(r)]
    row_receipts = []
    projected_centers=[]
    delta_trace=F(0)
    for i in rec['nuisance_rows']:
        c = list(map(F, a.c[i]))
        js = [j for j in range(a.p) if a.rad[i,j]]
        widths = [F(a.rad[i,j]) for j in js]
        ec = dot(c, reference) - F(a.y[i])
        eg = [w*reference[j] for j,w in zip(js,widths)]
        pc = [dot(c,col) for col in ucols]
        pg = [[w*col[j] for j,w in zip(js,widths)] for col in ucols]
        vals = []
        dvals=[]
        tc=[dot(c,col) for col in zip(*qt)]
        rc=[dot(tc,col) for col in cols]
        projected_centers.append((tc,rc))
        for signs in itertools.product((-1,1),repeat=len(js)):
            x = c.copy()
            for j,w,s in zip(js,widths,signs):
                x[j] += w*s
            tx = [dot(x,col) for col in zip(*qt)]
            retained = [dot(tx,col) for col in cols]
            discarded = dot(tx,tx)-qform(retained,Hi)
            assert discarded >= 0
            vals.append(discarded)
            delta=[v-vc for v,vc in zip(tx,tc)]
            dr=[v-vc for v,vc in zip(retained,rc)]
            dvals.append(dot(delta,delta)-qform(dr,Hi))
        ri = max(vals)
        rho += ri/a.n
        delta_trace+=max(dvals)/a.n
        ranges = [product_box(pc[j],pg[j],ec,eg) for j in range(r)]
        for j,(l,u) in enumerate(ranges):
            mixed_lo[j] += l/a.n
            mixed_hi[j] += u/a.n
        row_receipts.append(dict(row=i, discarded_vertex_norms_squared=list(map(str,vals)),
                                 discarded_maximum=str(ri), mixed_ranges=[[str(l),str(u)] for l,u in ranges]))
    center_gram=[[ (dot(v,w)-dot(vr,[dot(row,wr) for row in Hi]))/a.n
                   for w,wr in projected_centers] for v,vr in projected_centers]
    _,test=np.linalg.eigh(np.array(center_gram,float))
    test=np.round(test*2**24)/2**24
    tg=mm(I(test.T),I(test));gm=F(eigen_gershgorin(tg))
    assert gm>0
    kg=mm(I(test.T),mm(interval_q(center_gram),I(test)))
    center_op=F(eigen_gershgorin(kg,upper=True))/gm
    rho_operator=(sqrt_upper(max(F(0),center_op))+sqrt_upper(delta_trace))**2
    rho_trace=rho
    rho=min(rho,rho_operator)
    b2 = max(qform(list(z),Hi) for z in itertools.product(*zip(mixed_lo,mixed_hi)))
    ff = {k:F(v) for k,v in rec['fractions'].items()}
    mu,c,aa,eps,G,d = [ff[k] for k in ('mu','c','a','epsilon','G','d')]
    gamma2 = rho/(mu+rho)
    gamma = sqrt_upper(gamma2)
    t = min(ff['inverse_sqrt_mu']/2,sqrt_upper(rho)/mu)
    bb = sqrt_upper(b2)
    rhs_nuisance = min(aa*eps,bb+aa*gamma2*eps)
    radius = (G+rhs_nuisance+(c/mu+aa*t)*d+c*t*eps)/mu
    radius = min(radius,ff['retained_radius'])
    ff.update(discarded_energy=rho,discarded_trace=rho_trace,discarded_center_operator=center_op,
              discarded_delta_trace=delta_trace,discarded_operator=rho_operator, contraction_squared=gamma2,
              contraction=gamma, mixed_norm_squared=b2,mixed_norm=bb,
              inverse_cross_norm=t, retained_radius=radius)
    eta = tail_at_radius(ff,radius)
    coord = ceil_dyadic(sqrt_upper(radius*radius/hmu))
    ff.update(eta=eta,coordinate_radius=coord)
    rec['previous_energy_tail'] = rec['tail']
    rec.update(verifier='projected_contraction_energy_exact_rational',
               fractions={k:str(v) for k,v in ff.items()},tail=float_up(eta),
               tail_target_met=eta<=F(1,400000), contraction_rows=row_receipts,
               discarded_center_test=test.tolist(),
               mixed_gradient_intervals=[[str(l),str(u)] for l,u in zip(mixed_lo,mixed_hi)])
    return rec


def ceil_dyadic(x,bits=24):
    y=x*2**bits
    return F((y.numerator+y.denominator-1)//y.denominator,2**bits)


def tail_at_radius(ff,radius):
    # The mixed moment preserves cancellation before a norm is taken.
    energy2 = min((ff['epsilon']+ff['a']*radius)**2,
                  ff['epsilon2']+2*ff['mixed_norm']*radius+ff['a2']*radius**2)
    term = ff['contraction']*sqrt_upper(energy2)
    return (term+(ff['d']+ff['c']*radius)*ff['inverse_sqrt_mu'])**2


class ContractionEnergySearch(CenteredSignalSearch):
    def __init__(self,a,R,receipt):
        super().__init__(a,R,receipt,mode='energy')
        ff={k:F(v) for k,v in receipt['fractions'].items()}
        fullg2=F(self.centered_moment_receipt['full_gradient_squared'])
        radius=min(ff['retained_radius'],sqrt_upper(fullg2)/ff['mu'])
        self.qradius=ceil_dyadic(sqrt_upper(radius*radius/ff['gram_minimum']))
        self.qtail=tail_at_radius(ff,radius)
        self.radius=float_up(self.qradius)
        self.proof_mode='contraction_energy'
        self.initial_upper=float_up(F(a.gradient_upper)+self.qtail)
        self.contraction_receipt=dict(retained_radius=str(radius),coordinate_radius=str(self.qradius),
                                      tail=str(self.qtail),contraction_squared=str(ff['contraction_squared']),
                                      mixed_norm_squared=str(ff['mixed_norm_squared']))

    def run(self,**kwargs):
        ans=super().run(**kwargs)
        ans['contraction_receipt']=self.contraction_receipt
        if 'certificate' in ans:
            ans['certificate']['contraction_receipt']=self.contraction_receipt
        return ans


class AdaptiveEnergySearch(ContractionEnergySearch):
    """One search on the common domain; use the smallest verified tail."""
    def __init__(self,a,R,receipt):
        super().__init__(a,R,receipt)
        self.contraction_only_tail=self.qtail
        gn=F(self.centered_moment_receipt['gradient_norm'])
        coupling=F(self.centered_moment_receipt['coupling'])
        moment=(gn+coupling*self.qradius)**2/F(a.mu)
        self.qtail=min(self.qtail,moment)
        self.proof_mode='adaptive_energy'
        self.initial_upper=float_up(F(a.gradient_upper)+self.qtail)
        self.contraction_receipt.update(common_radius_moment_tail=str(moment),
            contraction_only_tail=str(self.contraction_only_tail),used_tail=str(self.qtail),
            selected_bound='contraction_energy' if self.qtail==self.contraction_only_tail else 'common_radius_moment')
