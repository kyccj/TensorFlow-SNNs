import json, statistics as st, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad/eipnotion')
SP='/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad/'
inv=json.load(open(SP+'led1007b/inventory.json'))['runs']
SURF='#fcfcfb'; INK='#1f1f1d'; INK2='#5f5e58'; GRID='#e6e5e0'; OURS='#2a78d6'; BPSR='#eb6834'; BASE='#6b6a64'
def runs(ds,key,prefix):
    return [r for r in inv if r['ds']==ds and r['key']==key and r.get('completed') and not r.get('exclude') and r['seed']<900 and r['tag'].startswith(prefix)]
def grp(rs):
    g={}
    for r in rs: g.setdefault(r['knob'],[]).append(r)
    out=[]
    for k,v in g.items():
        a=[r['val_acc'] for r in v]; s=[r['spikes'] for r in v]
        out.append(dict(k=k,n=len(v),a=st.mean(a),sa=st.stdev(a) if len(a)>1 else 0,s=st.mean(s),ss=st.stdev(s) if len(s)>1 else 0,pts=[(r['spikes'],r['val_acc']) for r in v]))
    return sorted(out,key=lambda d:d['s'])
def fig(title, base, arms, fname, rep=None, xlabel='spikes per sample'):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    f,ax=plt.subplots(figsize=(7.2,4.6),dpi=200); f.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ba=[r[1] for r in base]; bs=[r[0] for r in base]; bm,bsd=st.mean(ba),(st.stdev(ba) if len(ba)>1 else 0)
    ax.axhspan(bm-bsd,bm+bsd,color=BASE,alpha=0.10,lw=0); ax.axhline(bm,color=BASE,lw=1,ls=(0,(4,3)))
    ax.scatter(bs,ba,s=14,color=BASE,alpha=0.35,lw=0,zorder=2)
    ax.errorbar([st.mean(bs)],[bm],xerr=[st.stdev(bs) if len(bs)>1 else 0],yerr=[bsd],fmt='D',ms=8,color=BASE,mec=SURF,mew=2,ecolor=BASE,elinewidth=1.5,capsize=0,zorder=4)
    ax.annotate(f'baseline (n={len(base)})',(st.mean(bs),bm),xytext=(8,8),textcoords='offset points',color=INK2,fontsize=9)
    for lab,g,col,mk in arms:
        xs=[d['s'] for d in g]; ys=[d['a'] for d in g]
        for d in g: ax.scatter([p[0] for p in d['pts']],[p[1] for p in d['pts']],s=14,color=col,alpha=0.30,lw=0,zorder=2)
        ax.plot(xs,ys,color=col,lw=2,zorder=3)
        ax.errorbar(xs,ys,yerr=[d['sa'] for d in g],fmt=mk,ms=8,color=col,mec=SURF,mew=2,ecolor=col,elinewidth=1.2,capsize=0,zorder=4)
        last=g[0]; ax.annotate(lab,(last['s'],last['a']),xytext=(10,-4),textcoords='offset points',ha='left',va='top',color=INK,fontsize=10,fontweight='bold')
        if rep:
            for d in g:
                if d['k']==rep.get(lab): ax.annotate(f"ρ {d['k']} (n={d['n']})",(d['s'],d['a']),xytext=(8,-18),textcoords='offset points',color=INK2,fontsize=9)
    ax.margins(x=0.08,y=0.10)
    ax.set_title(title,loc='left',color=INK,fontsize=12,fontweight='bold',pad=10)
    ax.set_xlabel(xlabel,color=INK2); ax.set_ylabel('val acc (%)',color=INK2)
    ax.grid(True,color=GRID,lw=0.8); ax.set_axisbelow(True)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v,p:f'{v/1000:.0f}K'))
    from matplotlib.lines import Line2D
    h=[Line2D([0],[0],color=BASE,marker='D',ls=(0,(4,3)),ms=7,label='baseline (mean ± SD)')]+[Line2D([0],[0],color=c,marker=m,lw=2,ms=7,label=l) for l,_,c,m in arms]
    ax.legend(handles=h,frameon=False,loc='lower right',fontsize=9,labelcolor=INK2)
    f.text(0.01,0.005,'큰 점 = 같은 ρ/λ 평균 ± SD, 작은 점 = 개별 런. 오른쪽 위가 좋음(같은 스파이크에서 높은 정확도).',fontsize=8,color=INK2) if False else None
    f.tight_layout(); f.savefig(SP+'pareto/'+fname,facecolor=SURF); plt.close(f); print('saved',fname)
b=[(r['spikes'],r['val_acc']) for r in runs('C10','base','r19c10')]
fig('ResNet-19 · CIFAR-10',b,[('ours_inter_ch',grp(runs('C10','ours_ch','r19c10')),OURS,'o'),('BPSR',grp(runs('C10','bpsr','r19c10')),BPSR,'s')],'r19_c10.png',rep={'ours_inter_ch':'3e-3'})
b=[(r['spikes'],r['val_acc']) for r in runs('C100','base','r19c100')]
fig('ResNet-19 · CIFAR-100',b,[('ours_inter_ch',grp(runs('C100','ours_ch','r19c100')),OURS,'o'),('BPSR',grp(runs('C100','bpsr','r19c100')),BPSR,'s')],'r19_c100.png',rep={'ours_inter_ch':'2e-3'})
b=[(r['spikes'],r['val_acc']) for r in runs('DVS','base','vggdvscm')]
fig('VGGSNN · CIFAR10-DVS',b,[('ours_inter_ch',grp(runs('DVS','ours_ch','vggdvscm')),OURS,'o')],'dvs.png',rep={'ours_inter_ch':'3e-3'})
# QK
R='/home/kyccj/runs/qkformer_wta_rev/'
def qj(f): d=json.load(open(R+f)); return (d['totals']['spikes_per_sample'],d['accuracy'])
qb=[qj('spikes_base.json')]+[qj(f'spikes_base_s{s}.json') for s in [43,45,46,47,48]]
def mk(k,files):
    pts=[qj(f) for f in files]; a=[p[1] for p in pts]; s=[p[0] for p in pts]
    return dict(k=k,n=len(pts),a=st.mean(a),sa=st.stdev(a) if len(a)>1 else 0,s=st.mean(s),ss=0,pts=pts)
qo=sorted([mk('7e-3',['spikes_ours_ch_rho7e-3.json']),mk('1.2e-2',['spikes_ours_ch_rho1.2e-2.json']+[f'spikes_ours_ch_rho1.2e-2_s{s}.json' for s in [43,44,45]])],key=lambda d:d['s'])
fig('QKFormer · CIFAR-10',qb,[('ours_inter_ch',qo,OURS,'o')],'qk_c10.png',rep={'ours_inter_ch':'1.2e-2'})
cb=[qj(f'spikes_c100_base_s{s}.json') for s in [43,44,45,46]]
co=sorted([mk('6e-3',['spikes_c100_ours_ch_rho6e-3_s43.json']),mk('1.2e-2',[f'spikes_c100_ours_ch_rho1.2e-2_s{s}.json' for s in [43,44,45,46]])],key=lambda d:d['s'])
fig('QKFormer · CIFAR-100',cb,[('ours_inter_ch',co,OURS,'o')],'qk_c100.png',rep={'ours_inter_ch':'6e-3'})
