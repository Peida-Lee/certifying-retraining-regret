"""Full-budget ZORRO splitting with a valid inherited upper on every child."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import heapq,time
import numpy as np
from zorro_regret import enclosure
from observed_anchor import I,matvec

def center_lower(c,y,beta,lam,theta):
    def objective(v):return (matvec(I(c),I(v))-I(y)).square().sum()/len(y)+lam*I(v).square().sum()
    return max(0.,float((objective(beta)-objective(theta)).lo))

def run(a,common,budget,witness):
    start=time.monotonic();deadline=start+max(.001,budget-common)
    cells=[tuple(map(int,v)) for v in np.argwhere(a.rad)];m=len(cells)
    nodes=[dict(id=0,parent=None,lo=[-1.]*m,hi=[1.]*m,upper=a.gradient_upper,
                assessed=False,envelope_owner=None,bound_owner=None,children=[])]
    heap=[(-a.gradient_upper,0)];receipts=[];events=[];lower=witness['regret_lower'];attempts=0
    while heap and time.monotonic()<deadline:
        _,idx=heapq.heappop(heap);node=nodes[idx]
        if node['assessed']:
            scores=node['split_scores'];j=int(np.argmax(scores));middle=(node['lo'][j]+node['hi'][j])/2
            for side in (0,1):
                lo=node['lo'].copy();hi=node['hi'].copy()
                if side:lo[j]=middle
                else:hi[j]=middle
                child=dict(id=len(nodes),parent=idx,lo=lo,hi=hi,upper=node['upper'],assessed=False,
                    envelope_owner=node['envelope_owner'],bound_owner=node['bound_owner'],children=[])
                nodes.append(child);node['children'].append(child['id']);heapq.heappush(heap,(-child['upper'],child['id']))
            continue
        c=a.c.copy();rad=a.rad.copy()
        for k,(i,j) in enumerate(cells):
            c[i,j]+=a.rad[i,j]*(node['lo'][k]+node['hi'][k])/2
            rad[i,j]*=(node['hi'][k]-node['lo'][k])/2
        receipt=enclosure(c,rad,a.y,a.beta,a.lam,deadline);attempts+=1
        if time.monotonic()>deadline:
            heapq.heappush(heap,(-node['upper'],idx));break
        node['assessed']=True;node['envelope_status']=receipt['status']
        if receipt['status']=='PASS':
            owner=len(receipts);receipts.append(dict(node_id=idx,receipt=receipt));node['envelope_owner']=owner
            if receipt['upper']<node['upper']:node['upper']=receipt['upper'];node['bound_owner']=owner
            node['split_scores']=receipt['split_scores']
            lower=max(lower,center_lower(c,a.y,a.beta,a.lam,np.array(receipt['real_component'])))
        else:
            node['split_scores']=[float(rad[i,j]*(np.linalg.norm(c[i])+rad[i,j])) for i,j in cells]
        heapq.heappush(heap,(-node['upper'],idx))
        upper=-heap[0][0];assert lower<=upper
        events.append(dict(seconds=common+time.monotonic()-start,lower=lower,upper=upper,width=upper-lower,
                           attempts=attempts,leaves=len(heap),completed_envelopes=len(receipts)))
        if upper-lower<=1e-10:break
    upper=-heap[0][0];leaves=[idx for _,idx in heap]
    return dict(status='PASS',method='ZORRO_refined',lower=lower,upper=upper,absolute_gap=upper-lower,
        total_seconds=common+time.monotonic()-start,preprocessing_seconds=common,objective_lambda=a.lam,
        attempts=attempts,completed_envelopes=len(receipts),leaves=len(heap),events=events,nodes=nodes,
        envelope_receipts=receipts,leaf_ids=leaves,source_cells=cells,
        fixed_point_cover_complete=all(nodes[j]['envelope_owner'] is not None for j in leaves),
        inherited_parent_upper=True,common_cap=a.gradient_upper)
