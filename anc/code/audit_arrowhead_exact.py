"""Independent rational critical-point and complete-tree audit."""
from fractions import Fraction as F
from pathlib import Path
import json,itertools,traceback
B=Path(__file__).resolve().parent

def raw(u,t,s,z,n,lam,ell,alpha,y):
    return (u-2)**2+(alpha-y)**2-(u*s+ell*z-2)**2-(alpha*s+t*z-y)**2-n*lam*z*z

def profile(u,t,s,n,lam,ell,alpha,y):
    z=-(ell*(u*s-2)+t*(alpha*s-y))/(n*lam+ell*ell+t*t)
    return raw(u,t,s,z,n,lam,ell,alpha,y)

def qmax(f,l,h):
    c=f(F(0));b=(f(F(1))-f(F(-1)))/2;a=f(F(1))-b-c
    pts=[l,h]
    if a<0 and l<-b/(2*a)<h:pts.append(-b/(2*a))
    return max(f(x) for x in pts)

def pair_max(s,n,lam,g):
    ell,alpha,y,ul,uh,tl,th=g;values=[qmax(lambda u:profile(u,t,s,n,lam,ell,alpha,y),ul,uh) for t in (tl,th)]
    K=n*lam+ell*ell;r=alpha*s-y
    if ell:
        l,h=sorted((K*r/th,K*r/tl));a=ell*s;b=-2*ell
        if a:
            low,high=sorted(((l-b)/a,(h-b)/a));low=max(low,ul);high=min(high,uh)
        elif l<=b<=h:low,high=ul,uh
        else:low,high=F(1),F(0)
        if low<=high:values.append(qmax(lambda u:(u-2)**2+(alpha-y)**2-n*lam/K*(u*s-2)**2,low,high))
    return max(values)

def audit(file):
    data=json.loads(file.read_text());r=data['result'];rank=r['rank'];delta=F(r['delta']);ell=F(r['ell']);n=32*rank;lam=F(1/(2*rank));total_l=total_u=F(0);point_count=0;node_count=0
    for j,block in enumerate(data['blocks']):
        h=F(1,2**(j+2));groups=[tuple(map(F,g)) for g in block['groups']]
        expected=[(ell,F(1,32),F(1,32)+(1 if i<8 else -1)*delta,1-h,1+h,F(4),F(8)) for i in range(16)]
        assert groups==expected and F(block['lambda_value'])==lam and block['n']==n
        radius=F(block['radius']);curvature=F(block['curvature'])
        assert radius**2>=sum((4+g[2]**2 for g in groups),F(0))/(n*lam)
        assert curvature>=lam+sum((max(g[3]**2,g[4]**2)+g[1]**2 for g in groups),F(0))/n
        query={}
        for p in block['points']:
            s=F(p['s']);value=lam*(1-s*s)
            for g,w in zip(groups,p['pairs']):
                v,u,t,z=map(F,w);e,a,y,ul,uh,tl,th=g
                assert ul<=u<=uh and tl<=t<=th
                assert v==raw(u,t,s,z,n,lam,e,a,y)==pair_max(s,n,lam,g)
                value+=v/n
            assert value==F(p['value']);query[s]=value
        best=max(query.values());assert F(block['lower'])==best
        tree={v['id']:v for v in block['tree']};frontier=[0];leaves=[]
        assert F(tree[0]['lo'])==-radius and F(tree[0]['hi'])==radius
        while frontier:
            idx=frontier.pop();node=tree[idx];lo=F(node['lo']);hi=F(node['hi']);bound=max(query[lo],query[hi])+curvature*(hi-lo)**2/4
            if idx:bound=min(bound,F(tree[node['parent']]['upper']))
            assert F(node['upper'])>=bound
            if node['children']:
                l,r0=[tree[i] for i in node['children']];assert len(node['children'])==2
                assert F(l['lo'])==lo and F(l['hi'])==F(r0['lo']) and F(r0['hi'])==hi
                frontier.extend(node['children'])
            else:leaves.append(F(node['upper']))
        assert F(block['upper'])>=max(best,max(leaves))
        assert F(block['upper'])-best<=F(block['target'])
        total_l+=best;total_u+=F(block['upper']);point_count+=len(query);node_count+=len(tree)
    assert total_l==F(data['result']['lower']) and total_u==F(data['result']['upper'])
    assert total_u-total_l<=F(1,100000)
    return dict(status='PASS',condition=r['condition'],original_groups=16*rank,complete_tree_nodes=node_count,exact_queries=point_count,exact_gap=str(total_u-total_l),target_met=True)

if __name__=='__main__':
    reports=[]
    for file in sorted((B/'multirank_exact').glob('condition_*_receipt.json')):
        try:row=audit(file)
        except Exception as exc:row=dict(status='ERROR',file=file.name,error=str(exc),traceback=traceback.format_exc())
        reports.append(row)
    result=dict(status='PASS' if len(reports)==18 and all(r['status']=='PASS' for r in reports) else 'ERROR',conditions=len(reports),reports=reports)
    (B/'multirank_exact/independent_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
