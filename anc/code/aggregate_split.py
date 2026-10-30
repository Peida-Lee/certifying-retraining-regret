"""Continuous-cube RLT cuts for a covering aggregate-direction split."""
import numpy as np

def propose(source,matrix):
    keys=[tuple(k) for k in source['moment_terms']];look={k:i for i,k in enumerate(keys)};n=len(source['active_cells']);ii=[look[(j,)] for j in range(n)];Y=np.array(matrix);mean=Y[0,ii]
    covariance=Y[np.ix_(ii,ii)]-np.outer(mean,mean);values,vectors=np.linalg.eigh((covariance+covariance.T)/2)
    a=np.round(vectors[:,-1]*2**12)/2**12
    low=np.minimum(a,0).sum();high=np.maximum(a,0).sum();cut=float(np.round((a@mean)*2**16)/2**16)
    cut=min(high-2**-16,max(low+2**-16,cut))
    return a,cut,dict(top_variance=float(values[-1]),range=[float(low),float(high)])

def constraints(a,cut,side):
    a=np.asarray(a)*(-1 if side else 1);b=cut*(-1 if side else 1);n=len(a);cuts=[]
    def add(terms,rhs):
        merged={}
        for key,value in terms:merged[tuple(sorted(key))]=merged.get(tuple(sorted(key)),0.)+value
        cuts.append(dict(kind='aggregate_rlt',terms=[[list(k),float(v)] for k,v in sorted(merged.items()) if v],rhs=float(rhs)))
    add([((i,),a[i]) for i in range(n)],b)
    for j in range(n):
        add([((i,j),a[i]) for i in range(n)]+[((j,),-b)],0.)
        add([((i,),a[i]) for i in range(n)]+[((i,j),-a[i]) for i in range(n)]+[((j,),b)],b)
    return cuts
