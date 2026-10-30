"""BLAS-backed interval products with explicit dot-product rounding bounds."""
import numpy as np
from interval_arithmetic import I,up,down

def mm(a,b):
    k=a.lo.shape[-1]
    if k!=b.lo.shape[0]:raise ValueError('incompatible matrices')
    ac=a.lo+(a.hi-a.lo)/2;bc=b.lo+(b.hi-b.lo)/2
    ar=np.maximum(up(ac-a.lo),up(a.hi-ac));br=np.maximum(up(bc-b.lo),up(b.hi-bc))
    # gamma_k bounds any standard length-k dot product (also with FMA).
    eps=np.finfo(float).eps
    gamma=float(up(k*eps/(1-k*eps)))
    tiny=float(up(k*np.nextafter(0.,1.)))
    def positive(x,y):return up(up(x@y+tiny)/down(1-gamma))
    center=ac@bc
    error=up(up(gamma*positive(np.abs(ac),np.abs(bc)))+tiny)
    error=up(error+positive(np.abs(ac),br))
    error=up(error+positive(ar,np.abs(bc)))
    error=up(error+positive(ar,br))
    return I(down(center-error),up(center+error))

def matvec(a,b):
    c=mm(a,I(b.lo[:,None],b.hi[:,None]))
    return I(c.lo[:,0],c.hi[:,0])
