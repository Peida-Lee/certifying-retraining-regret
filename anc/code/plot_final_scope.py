"""Regenerate the manuscript scope figure from portable aggregate records."""
from pathlib import Path
import argparse,csv,json,os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
if os.name=='nt':
    import ctypes
    kernel=ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype=ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t))
    kernel.SetProcessAffinityMask.argtypes=(ctypes.c_void_p,ctypes.c_size_t)
    process_mask,system_mask=ctypes.c_size_t(),ctypes.c_size_t()
    assert kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),ctypes.byref(process_mask),ctypes.byref(system_mask))
    allowed=[i for i in range(8*ctypes.sizeof(ctypes.c_size_t)) if process_mask.value & (1<<i)]
    assert kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(),1<<(10 if 10 in allowed else max(allowed)))
elif hasattr(os,'sched_setaffinity'):
    allowed=os.sched_getaffinity(0)
    os.sched_setaffinity(0,{10 if 10 in allowed else max(allowed)})
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,LogNorm
import numpy as np

def main():
    root=Path(__file__).resolve().parent.parent
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('scope_final.pdf'))
    args=parser.parse_args()
    results=root/'results'
    general=json.loads((results/'general_adaptive_v33/summary.json').read_text())
    dense=json.loads((results/'refinement_v92/dense_edge_summary.json').read_text())
    structure=json.loads((results/'structure/structure_results.json').read_text())
    with (results/'tail_diagnostics/diagnostics.csv').open(newline='') as handle:
        diagnostics=list(csv.DictReader(handle))
    assert general['certified']==general['audited']==60
    assert dense['target_successes']==dense['conditions']==4
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8.5,
        'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
        'axes.linewidth':.6,'lines.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(6.75,5.3))
    axes=[fig.add_axes(p) for p in ([.13,.62,.36,.23],[.64,.62,.33,.23],
                                  [.13,.16,.36,.25],[.64,.16,.33,.25])]
    fig.text(.13,.96,'Certification and computation across experiment families',fontsize=10,fontweight='bold')
    fig.text(.13,.92,'Each panel uses its stated design and precision target.',fontsize=7.5)
    ax=axes[0]
    names=['diabetes','concrete','superconductivity']
    columns=[(1/32,'concentrated'),(1/32,'dispersed'),(1/8,'concentrated'),(1/8,'dispersed')]
    counts=np.zeros((3,4),int)
    for i,name in enumerate(names):
        for j,(width,spread) in enumerate(columns):
            selected=[r for r in general['groups'] if r['dataset']==name and r['width']==width and r['spread']==spread]
            assert len(selected)==1 and selected[0]['conditions']==5
            counts[i,j]=selected[0]['certified']
    ax.imshow(counts,vmin=0,vmax=5,aspect='auto',cmap=LinearSegmentedColormap.from_list('pass',['#f4f6f7','#8ac6ba']))
    for i in range(3):
        for j in range(4):ax.text(j,i,f'{counts[i,j]}/5',ha='center',va='center')
    ax.set_xticks(range(4),['1/32\nConc.','1/32\nDisp.','1/8\nConc.','1/8\nDisp.'])
    ax.set_yticks(range(3),['Diabetes','Concrete','Supercond.'])
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='both',length=0,pad=4)
    ax.spines[:].set_visible(False)
    ax.set_title('A  General coarsening: 60/60',loc='left',fontweight='bold',pad=10)
    ax.text(0,-.39,r'Width $\leq3\times10^{-6}$; all exact replays pass'+'\nComplete algorithm: at most 156.76 s',transform=ax.transAxes,va='top',fontsize=7)
    ax=axes[1]
    groups=[('concrete','concentrated'),('concrete','dispersed'),('superconductivity','concentrated'),('superconductivity','dispersed')]
    matrix=np.zeros((4,6))
    for i,(name,spread) in enumerate(groups):
        for j,(n,lam) in enumerate([(n,l) for n in (64,128) for l in (1/64,1/32,1/16)]):
            selected=[r for r in diagnostics if r['phase']!='rank_curve' and r['dataset']==name and r['spread']==spread and int(r['uncertain_rows'])==n and float(r['lam'])==lam and r['method']=='residual_gradient' and int(r['rank'])==3]
            assert len(selected)==1,(name,spread,n,lam,len(selected))
            matrix[i,j]=float(selected[0]['eta_r'])*1e6
    ax.imshow(matrix,aspect='auto',cmap='cividis',norm=LogNorm(vmin=matrix.min(),vmax=matrix.max()))
    for i in range(4):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=6.4,color='white' if matrix[i,j]<10 else '#111111')
    ax.set_yticks(range(4),['Concr. C','Concr. D','Super. C','Super. D']);ax.set_xticks(range(6),['0.5','1','2','0.5','1','2'])
    ax.tick_params(length=0,pad=3);ax.spines[:].set_visible(False);ax.axvline(2.5,color='white',lw=1.5)
    ax.set_title('B  Rank-three tail diagnostics',loc='left',fontweight='bold',pad=10)
    ax.text(.25,-.18,'64 rows',transform=ax.transAxes,ha='center',va='top',fontsize=7)
    ax.text(.75,-.18,'128 rows',transform=ax.transAxes,ha='center',va='top',fontsize=7)
    ax.text(0,-.39,r'$\lambda$ multiplier; tail in $10^{-6}$'+'\n24 bound diagnostics; two cells per row',transform=ax.transAxes,va='top',fontsize=7)
    ax=axes[2]
    rows=dense['rows'];x=np.arange(len(rows))
    assert all(r['target_met_in_budget'] and r['independent_audit']=='PASS' and r['gap']<=1e-8 for r in rows)
    ax.bar(x-.17,[r['algorithm_seconds'] for r in rows],width=.34,color='#258f9b',label='Algorithm')
    ax.bar(x+.17,[r['independent_audit_seconds'] for r in rows],width=.34,color='#97a2b0',label='Replay')
    ax.set_yscale('log');ax.set_ylim(.7,2000);ax.set_ylabel('Complete time (s)')
    ax.set_xticks(x,['Diabetes\n10 cols','Synthetic\n4 cols','Synthetic\n8 cols','Synthetic\n16 cols'])
    ax.set_title('C  Dense rows: 4/4 certified',loc='left',fontweight='bold',pad=10)
    ax.legend(frameon=False,loc='upper left',fontsize=6.5)
    ax.text(0,-.39,r'Original width target $10^{-8}$; all features retained'+'\n4 uncertain rows; independent replay shown separately',transform=ax.transAxes,va='top',fontsize=7)
    ax=axes[3]
    costs=sorted([r for r in structure['results'] if r['method']=='residual_energy' and r['alpha']=='1/32'],key=lambda r:r['p'])
    assert [r['p'] for r in costs]==[17,65,257]
    setup=np.array([r['construction_seconds']+r['anchor_setup_seconds']+r['method_setup_seconds'] for r in costs])
    search=np.array([r['search_wall_seconds'] for r in costs]);x=np.arange(3)
    ax.plot(x,setup,'s-',color='#0072b2',ms=3.7,label='Setup');ax.plot(x,search,'o--',color='#7a7d80',ms=3.7,mfc='white',label='Search')
    ax.set_yscale('log');ax.set_ylim(.008,130);ax.set_xlim(-.2,2.3)
    ax.set_xticks(x,[str(r['p']) for r in costs]);ax.set_xlabel('Original parameters (p)');ax.set_ylabel('Elapsed time (s)')
    ax.text(2.02,setup[-1]*1.25,f'{setup[-1]:.2f} s',ha='right',va='bottom',fontsize=7)
    ax.text(2.02,search[-1]*1.35,f'{search[-1]:.3f} s',ha='right',va='bottom',fontsize=7)
    ax.legend(frameon=False,loc='upper left',fontsize=7)
    ax.set_title('D  Structural family: setup cost',loc='left',fontweight='bold',pad=10)
    ax.text(0,-.39,r'$\alpha=1/32$, rank 1; single timing per size'+'\nSetup includes input, anchor and method receipt',transform=ax.transAxes,va='top',fontsize=7)
    for ax in axes[2:]:
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='#e6e9ec',lw=.5);ax.set_axisbelow(True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,metadata={'Title':'Certification and computation across experiment families','Creator':'Matplotlib'})
    fig.savefig(args.output.with_suffix('.png'),dpi=180)
    evidence={'general_successes':int(counts.sum()),'general_cells':counts.tolist(),
              'dense_successes':4,'dense_algorithm_seconds':[r['algorithm_seconds'] for r in rows],
              'tail_diagnostics_1e6':matrix.tolist(),'structural_setup_seconds':setup.tolist(),
              'structural_search_seconds':search.tolist(),'raw_records_opened':False}
    args.output.with_suffix('.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps({'status':'PASS','output':str(args.output),'general':60,'dense':4}))

if __name__=='__main__':main()
