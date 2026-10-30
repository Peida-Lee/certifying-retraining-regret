"""Independent rational audit of row-residual lifting and conic duals."""
from pathlib import Path
from fractions import Fraction as F
import json,time,sys,traceback
import numpy as np
from audit_sdp_exact import dot,qmat,qvalue,denominator,int_array,build_constraints

def audit(file,original,details=None):
    payload=json.loads(file.read_text());s=payload['problem'];r=payload['result']
    lo=np.array(original['lo']);hi=np.array(original['hi']);n,p=lo.shape;ids=list(map(tuple,s['active_cells']));d=len(ids)
    assert set(ids)==set(map(tuple,np.argwhere(lo!=hi)))
    fixed=np.all(lo==hi,axis=1);rows=list(np.flatnonzero(~fixed));rid={int(i):z for z,i in enumerate(rows)};nv=d+len(rows)
    y=list(map(F,original['y']));b=list(map(F,original['beta']));t=list(map(F,s['theta']));lam=F(original['lambda_value'])
    x=(lo+hi)/2
    for (i,j),v in zip(ids,s['x0_uncertain']):assert v in (lo[i,j],hi[i,j]);x[i,j]=v
    dx=denominator(x.flat);dy=denominator(y);X=int_array(x,dx);Yi=np.array([int(v*dy) for v in y],object)
    den=n*dx*dx;Gi=X[fixed].T@X[fixed];assert (lam*den).denominator==1
    for j in range(p):Gi[j,j]+=int(lam*den)
    G=[[F(int(v),den) for v in row] for row in Gi];ra=[F(int(v),n*dx*dy) for v in X[fixed].T@Yi[fixed]]
    T=[[F(v) for v in row] for row in s['T']];dt=denominator(v for row in T for v in row);Ti=int_array(T,dt)
    metric=Ti.T@Gi@Ti;metric_den=den*dt*dt
    exact_mu=min(F(int(metric[j,j]-sum(abs(metric[j,k]) for k in range(p) if k!=j)),metric_den) for j in range(p));mu=F(s['mu']);assert 0<mu<=exact_mu
    ds=[(F(hi[i,j])-F(lo[i,j]))*(1 if x[i,j]==lo[i,j] else -1) for i,j in ids]
    left=list(map(F,s['residual_left']));span=list(map(F,s['residual_span']));relations=[]
    for ii,i in enumerate(rows):
        low=sum((min(F(lo[i,j])*t[j],F(hi[i,j])*t[j]) for j in range(p)),F(0))-y[i]
        high=sum((max(F(lo[i,j])*t[j],F(hi[i,j])*t[j]) for j in range(p)),F(0))-y[i]
        assert span[ii]>0 and left[ii]<=low and left[ii]+span[ii]>=high
        c={(d+ii,):F(1)}
        for z,(rr,j) in enumerate(ids):
            if rr==i:c[(z,)]=-ds[z]*t[j]/span[ii]
        c={k:v for k,v in c.items() if v};rhs=(dot(list(map(F,x[i])),t)-y[i]-left[ii])/span[ii]
        relations.extend([(c,rhs),({k:-v for k,v in c.items()},-rhs)])
    g0=[v-w for v,w in zip(qmat(G,t),ra)]
    poly={():qvalue(G,ra,b)-qvalue(G,ra,t)}
    cell_loss=s.get('loss_difference_rule')=='original_cells'
    for ii,i in enumerate(rows):
        row=list(map(F,x[i]));eb=dot(row,b)-y[i]
        et=dot(row,t)-y[i]
        poly[()]+=(eb*eb-(et*et if cell_loss else left[ii]*left[ii]))/n
        g0=[g+v*left[ii]/n for g,v in zip(g0,row)]
    grad={():g0}
    def add(k,v):poly[k]=poly.get(k,F(0))+v
    for z,(i,j) in enumerate(ids):
        ii=rid[i];u=d+ii;v=[F(0)]*p;v[j]=ds[z]*left[ii]/n;grad[(z,)]=v
        v=[F(0)]*p;v[j]=ds[z]*span[ii]/n;grad[(z,u)]=v
        eb=dot(list(map(F,x[i])),b)-y[i];et=dot(list(map(F,x[i])),t)-y[i];add((z,),2*(eb*b[j]-(et*t[j] if cell_loss else 0))*ds[z]/n)
        for w in range(z,d):
            rr,k=ids[w]
            if rr==i:add((z,w),(b[j]*b[k]-(t[j]*t[k] if cell_loss else 0))*ds[z]*ds[w]*(1 if z==w else 2)/n)
    for ii,i in enumerate(rows):
        u=d+ii;grad[(u,)]=[F(v)*span[ii]/n for v in x[i]]
        if not cell_loss:
            add((u,),-2*left[ii]*span[ii]/n);add((u,u),-span[ii]*span[ii]/n)
    terms=list(grad);dc=denominator(v for k in terms for v in grad[k]);C=int_array([grad[k] for k in terms],dc)@Ti;gram=C@C.T;gd=F((dc*dt)**2)*mu
    for j,k in enumerate(terms):
        for l in range(j,len(terms)):add(tuple(sorted(k+terms[l])),F(int(gram[j,l]))/gd*(1 if j==l else 2))
    keys=list(map(tuple,s['monomials']));lookup={k:i for i,k in enumerate(keys)}
    assert all(k in lookup or not k for k in poly)
    for k,l,u in zip(keys,s['coefficient_lower'],s['objective']):assert F(l)<=poly.get(k,F(0))<=F(u),('polynomial',k)
    assert F(s['constant_lower'])<=poly[()]<=F(s['constant_upper'])
    cons=build_constraints(keys)
    def norm(items):
        z={}
        for k,v in items:
            k=tuple(sorted(k));z[k]=z.get(k,F(0))+F(v)
        return {k:v for k,v in z.items() if v}
    def validate_block(block,nvars,expected=None):
        assert len(block)==1+2*nvars
        a=norm(block[0]['terms']);rhs=F(block[0]['rhs'])
        if expected is not None:
            aa,bb=expected;assert all(a.get(k,F(0))<=aa.get(k,F(0)) for k in set(a)|set(aa)) and rhs>=bb
        assert all(len(k)==1 for k in a)
        want=[(a,rhs)]
        for j in range(nvars):
            prod=[(tuple(sorted(k+(j,))),v) for k,v in a.items()]
            want.append((norm(prod+[((j,),-rhs)]),F(0)))
            want.append((norm(list(a.items())+[(k,-v) for k,v in prod]+[((j,),rhs)]),rhs))
        for c,(aa,bb) in zip(block,want):
            actual=norm(c['terms']);bound=F(c['rhs'])
            assert all(actual.get(k,F(0))<=aa.get(k,F(0)) for k in set(actual)|set(aa)) and bound>=bb
            cons.append((tuple((lookup[k],v) for k,v in actual.items()),bound))
    links=s['residual_link_inequalities'];bs=1+2*nv;assert len(links)==len(relations)*bs
    for off,expected in zip(range(0,len(links),bs),relations):validate_block(links[off:off+bs],nv,expected)
    higher=s.get('higher_residual_links',[]);rules=s.get('higher_residual_link_rules',[]);assert len(higher)==len(rules)
    for cut,rule in zip(higher,rules):
        idx=rule['base_link'];assert 0<=idx<len(links) and idx%(1+2*nv)==0
        base=norm(links[idx]['terms']);base[()]=-F(links[idx]['rhs']);mult={():F(1)}
        for j,complement in rule['factors']:
            assert 0<=j<nv and type(complement) is bool
            new={}
            for k,v in mult.items():
                if complement:new[k]=new.get(k,F(0))+v
                z=tuple(sorted(k+(j,)));new[z]=new.get(z,F(0))+v*(-1 if complement else 1)
            mult=new
        expected={}
        for k,v in base.items():
            for z,w in mult.items():
                key=tuple(sorted(k+z));expected[key]=expected.get(key,F(0))+v*w
        assert expected.get((),F(0))==0
        actual=norm(cut['terms']);assert F(cut['rhs'])>=0
        assert all(actual.get(k,F(0))<=expected.get(k,F(0)) for k in set(actual)|set(expected))
        cons.append((tuple((lookup[k],v) for k,v in actual.items()),F(cut['rhs'])))
    cuts=s['domain_inequalities'];bs=1+2*d;assert len(cuts)%bs==0
    for off in range(0,len(cuts),bs):validate_block(cuts[off:off+bs],d)
    faces=s.get('extremizer_face_inequalities',[])
    if faces:
        assert s.get('extremizer_domain')=='at_most_one_interior_per_original_row'
    for cut in faces:
        j,k=cut['cell_pair'];assert 0<=j<k<d and ids[j][0]==ids[k][0]
        want={(j,k):F(1),(j,j,k):F(-1),(j,k,k):F(-1),(j,j,k,k):F(1)}
        actual=norm(cut['terms']);assert actual==want and F(cut['rhs'])==0
        cons.append((tuple((lookup[key],v) for key,v in actual.items()),F(0)))
    assert s['extra_linear_inequalities']==links+higher+cuts+faces
    dual=list(map(F,r['dual']));assert len(dual)==len(cons) and min(dual)>=0
    scale=F(r['scale']);res=[F(c)*scale for c in s['objective']];bound=F(0)
    for lam,(row,rhs) in zip(dual,cons):
        bound+=lam*rhs
        if lam:
            for j,v in row:res[j]-=lam*v
    mt=list(map(tuple,s['moment_terms']))
    expected_terms=[k for k in terms if not (len(k)==1 and k[0]>=d)] if s.get('moment_basis_rule') in ('omit_linear_residuals','facial_row_residuals') else terms
    assert mt==expected_terms
    if 'moment_blocks' in r:
        blocks=r['moment_blocks'];factors=r['psd_factors'];assert len(blocks)==len(factors)
        assert blocks[0]['basis']==s['moment_terms'] and blocks[0]['polynomial']==[[[],1.]]
        for block in blocks[1:]:
            pol=norm(block['polynomial'])
            if block.get('kind')=='row_edge_budget':
                assert s.get('extremizer_domain')=='at_most_one_interior_per_original_row'
                row=int(block['row']);group=[j for j,(rr,col) in enumerate(ids) if rr==row]
                assert group and block['basis']==[[]]+[[j] for j in group]
                want={():F(1,4)}
                for j in group:want[(j,)]=F(-1);want[(j,j)]=F(1)
                assert pol==want
            elif block.get('kind')=='halfspace_localizer':
                cut=cuts[block['domain_inequality_index']]
                want={k:-v for k,v in norm(cut['terms']).items()}
                if cut['rhs']:want[()]=F(cut['rhs'])
                assert pol==want
            else:
                single=[k for k,v in pol.items() if len(k)==1 and v==1]
                assert len(single)==1;j=single[0][0]
                assert 0<=j<nv and pol=={(j,):F(1),(j,j):F(-1)}
            assert all(all(0<=v<nv for v in k) for k in block['basis'])
    else:
        blocks=[dict(basis=s['moment_terms'],polynomial=[[[],1.]])];factors=[r['psd_factor']]
    for block,factor in zip(blocks,factors):
        assert len(factor)==len(block['basis']) and all(len(v)==len(factor) for v in factor)
        df=denominator(v for row in factor for v in row);Fi=int_array(factor,df);P=Fi@Fi.T
        for i,k in enumerate(block['basis']):
            for j,l in enumerate(block['basis']):
                for monomial,coefficient in block['polynomial']:
                    key=tuple(sorted(k+l+monomial));v=F(int(P[i,j]),df*df)*F(coefficient)
                    if key:res[lookup[key]]+=v
                    else:bound+=v
    upper=(bound+sum((max(F(0),v) for v in res),F(0)))/scale+F(s['constant_upper']);assert F(r['upper'])>=upper
    w=payload['witness'];wx=np.array(w['x']);wt=list(map(F,w['theta']));assert np.all(wx>=lo) and np.all(wx<=hi)
    lower=qvalue(G,ra,b)-qvalue(G,ra,wt)
    for i in rows:
        row=list(map(F,wx[i]));lower+=((dot(row,b)-y[i])**2-(dot(row,wt)-y[i])**2)/n
    assert F(r['lower'])<=lower
    if details is not None:
        details.update(keys=keys,residual=res,bound=bound,scale=scale,constant=F(s['constant_upper']),exact_relations=relations,cell_count=d,exact_lower=lower)
    return dict(status='PASS',file=file.name,original_polynomial=True,residual_links=True,
                row_face_extremizer_constraints=len(faces),exact_metric=True,exact_factor_dual=True,
                original_witness=True,polynomial_coefficients=len(keys)+1,exact_upper=str(upper))

def validate(folder):
    from independent_moment_audit import audit as first_audit
    import independent_moment_audit as first_module
    from independent_planar_rows import independent_rowmax
    first_module.rowmax=independent_rowmax
    result=json.loads((folder/'result.json').read_text());original=json.loads((folder/'original_input.json').read_text());first=json.loads((folder/'first_stage.json').read_text())
    fa=first_audit(folder/'first_stage.json',True);nodes={n['id']:n for n in result['nodes']};payloads={};reports=[]
    cover_ids=set();todo=[0] if nodes else []
    while todo:
        i=todo.pop();cover_ids.add(i);todo.extend(nodes[i].get('children',[]))
    unused=0
    for i,node in sorted(nodes.items()):
        file=folder/node['receipt'];reports.append(audit(file,original));payloads[i]=json.loads(file.read_text());source=payloads[i]['problem'];cuts=source['domain_inequalities']
        if node['parent'] is None:assert not cuts;inherited=F(first['upper'])
        else:
            parent=nodes[node['parent']];prior=payloads[node['parent']]['problem']['domain_inequalities'];assert cuts[:len(prior)]==prior
            assert source['active_cells']==payloads[node['parent']]['problem']['active_cells'] and source['x0_uncertain']==payloads[node['parent']]['problem']['x0_uncertain']
            if 'split' in parent:
                split=parent['split'];sgn=1 if node['side']==0 else -1;assert F(cuts[len(prior)]['rhs'])==sgn*F(split['cut'])
                assert {tuple(k):F(v) for k,v in cuts[len(prior)]['terms']}=={(j,):sgn*F(v) for j,v in enumerate(split['a']) if v}
            else:
                assert i not in cover_ids and node['side'] in (0,1);assert len(cuts)-len(prior)==1+2*len(source['active_cells']);unused+=1
            inherited=F(parent['upper'])
        assert F(node['upper'])>=min(inherited,F(payloads[i]['result']['upper']))
    if nodes:
        todo=[0];leaves=[]
        while todo:
            i=todo.pop();node=nodes[i]
            if 'children' in node:
                cc=node['children'];assert len(cc)==2 and {nodes[c]['side'] for c in cc}=={0,1};assert all(nodes[c]['parent']==i for c in cc);todo.extend(cc)
            else:leaves.append(i)
        assert set(leaves)==set(result['active_leaves'])
        assert F(result['upper'])>=min(F(first['upper']),max(F(nodes[i]['upper']) for i in leaves))
    assert F(result['lower'])<=max([F(first['lower'])]+[F(v['result']['lower']) for v in payloads.values()])
    assert result['target_met_in_budget']==any(h['seconds']<=result['cap_seconds'] and h['upper']-h['lower']<=result['target'] for h in result['history'])
    return dict(status='PASS',condition=result['condition'],polynomial_receipts=len(reports),residual_links=True,complete_cover=True,unused_conditional_children=unused,first_stage=fa,reports=reports,target_met_in_budget=result['target_met_in_budget'],absolute_gap=result['absolute_gap'])

if __name__=='__main__':
    folder=Path(sys.argv[1]);start=time.monotonic()
    try:r=validate(folder)
    except Exception as exc:r=dict(status='ERROR',error=str(exc),traceback=traceback.format_exc())
    r['seconds']=time.monotonic()-start;(folder/'independent_audit.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ('first_stage','reports')}))
