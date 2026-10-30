"""Conditional row moment matrices for aggregate branch halfspaces."""
import copy
from fractions import Fraction as F
import numpy as np
from localized_moment import solve as localized_solve


def strengthen(source):
    src=copy.deepcopy(source);n=len(src['active_cells']);cuts=src.get('extra_linear_inequalities',[])
    block=1+2*n
    assert len(cuts)%block==0
    groups={}
    for j,(row,col) in enumerate(src['active_cells']):groups.setdefault(row,[]).append(j)
    matrices=[dict(basis=src['moment_terms'],polynomial=[[[],1.]],kind='gradient_moment')]
    for offset in range(0,len(cuts),block):
        base=cuts[offset]
        polynomial=[[[],base['rhs']]]+[[k,float(-F(v))] for k,v in base['terms']]
        for row,group in groups.items():
            matrices.append(dict(basis=[[]]+[[j] for j in group],polynomial=polynomial,
                                 kind='aggregate_halfspace_localizer',base_cut=offset,row=int(row)))
    src['custom_moment_blocks']=matrices;src['localizing_solver']='SCS'
    return src


def compress_gradient_basis(src):
    """Project quadratic monomials onto the normal-residual coefficient span.

    The projected cone is a relaxation. Its lifted exact PSD factors, not
    a numerical rank claim, establish the certificate. Row cones are retained.
    """
    terms=src['moment_terms'];ids=src['active_cells'];theta=np.asarray(src['theta'])
    T=np.asarray(src['T']);lo=np.asarray(src['domain_lo']);hi=np.asarray(src['domain_hi'])
    delta=np.array([(hi[i,j]-lo[i,j])*(1 if x==lo[i,j] else -1)
                    for (i,j),x in zip(ids,src['x0_uncertain'])])
    linear=[i for i,k in enumerate(terms) if len(k)<2]
    quadratic=[i for i,k in enumerate(terms) if len(k)==2]
    C=[]
    for index in quadratic:
        a,b=terms[index];j=ids[a][1];k=ids[b][1]
        value=theta[j]*T[j] if a==b else theta[k]*T[j]+theta[j]*T[k]
        C.append(delta[a]*delta[b]*value)
    U,s,_=np.linalg.svd(np.asarray(C),full_matrices=False)
    rank=int(np.count_nonzero(s>max(s[0]*1e-12,1e-30))) if len(s) else 0
    transform=np.zeros((len(linear)+rank,len(terms)))
    transform[np.arange(len(linear)),linear]=1.
    transform[len(linear):,quadratic]=np.round(U[:,:rank].T*2**30)/2**30
    src['custom_moment_blocks'][0]['basis_transform']=transform.tolist()
    groups={}
    for j,(row,col) in enumerate(ids):groups.setdefault(row,set()).add(j)
    for row,group in groups.items():
        basis=[k for k in terms if all(j in group for j in k)]
        src['custom_moment_blocks'].append(dict(basis=basis,polynomial=[[[],1.]],kind='row_moment',row=int(row)))
    src['gradient_compression']=dict(full_order=len(terms),projected_order=len(transform),quadratic_rank=rank)
    return src


def solve(source,seconds=120,compress=False,return_moments=False):
    src=strengthen(source)
    if compress:src=compress_gradient_basis(src)
    result=localized_solve(src,seconds=seconds,return_moments=return_moments)
    oldkeys=list(map(tuple,src['monomials']));upper=dict(zip(oldkeys,src['objective']))
    lower=dict(zip(oldkeys,src['coefficient_lower']))
    src['monomials']=result['monomials']
    src['objective']=[upper.get(tuple(k),0.) for k in src['monomials']]
    src['coefficient_lower']=[lower.get(tuple(k),0.) for k in src['monomials']]
    return src,result
