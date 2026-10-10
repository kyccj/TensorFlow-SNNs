"""층별 ANCOVA: gini = a + b*ln(spike_count) + c*[arm].  CSV 프로토콜(최종 에폭 309).
   npz 는 팔당 n=1 이지만 CSV 는 같은 팔에서 여러 rho x 여러 시드를 쓸 수 있다 -> c 에 실제 CI 가 붙는다."""
import sys; sys.path.insert(0,'/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
from npz_neuron_dist_v2 import ROOTS, EXCL
import pandas as pd, numpy as np, os
from scipy import stats
def get(run):
    for r in ROOTS:
        a=os.path.join(r,run,'reg_neuron_detail.csv'); b=os.path.join(r,run,'reg_detail.csv')
        if os.path.exists(a) and os.path.exists(b) and os.path.getsize(a)>1000 and os.path.getsize(b)>1000:
            n=pd.read_csv(a); s=pd.read_csv(b)
            e=min(n.epoch.max(), s.epoch.max())
            if e<309: return None,None,None
            return e, n[n.epoch==e].set_index('layer'), s[s.epoch==e].set_index('layer')
    return None,None,None
FAM={
 'l2_lr':      ['r19c10-l2_lr-%s-s1'%r for r in ['2e-3','3e-3','4e-3','5e-3']],
 'plain L2':   ['r19c10-l2-5e-8-s1']+['r19c10-l2-5e-7-s%d'%i for i in range(1,6)]+['r19c10-l2-4p5e-7-s%d'%i for i in [1,2,3]]+['r19c10-l2-2p5e-7-s%d'%i for i in [1,2,3,4]]+['r19c10-l2-1e-7-s%d'%i for i in [1,2]]+['r19c10-l2-1p5e-7-s%d'%i for i in [1,2]]+['r19c10-l2-7p5e-7-s1','r19c10-l2-1p5e-6-s1'],
 'ours_ch':    ['r19c10-ours_ch-%s-s1'%r for r in ['2e-3','3e-3','4e-3']],
 'ours':       ['r19c10-prop-4e-3-s%d'%i for i in range(1,6)]+['r19c10-prop-3e-3-s%d'%i for i in [2,3,8]]+['r19c10-prop-5e-3-s1','r19c10-prop-6e-3-s1'],
 'ours_layer': ['r19c10-ours_layer-4e-3-s%d'%i for i in [1,2,3]]+['r19c10-ours_layer-3e-3-s%d'%i for i in [1,2,3]]+['r19c10-ours_layer-1e-3-s1','r19c10-ours_layer-2e-3-s1'],
 'ours_w1':    ['r19c10-ours_w1-3e-3-s%d'%i for i in [1,2,3]]+['r19c10-ours_w1-2p5e-3-s%d'%i for i in [1,2,3]]+['r19c10-ours_w1-2e-3-s%d'%i for i in [1,2,3]]+['r19c10-ours_w1-4e-3-s1','r19c10-ours_w1-5e-3-s1'],
}
D={}
for f,rs in FAM.items():
    D[f]=[]
    for r in rs:
        e,n,s=get(r)
        if n is None: continue
        D[f].append((r,n,s))
    print(f'{f:12s} n={len(D[f])}런: '+', '.join(r.replace("r19c10-","") for r,_,_ in D[f]))
LS=[l for l in D['ours'][0][1].index if l not in EXCL]

def ancova(A,B,layer):
    """B 대비 A 의 같은 스파이크에서의 gini 오프셋 c."""
    x=[];y=[];g=[]
    for f,gv in [(B,0),(A,1)]:
        for r,n,s in D[f]:
            sc=s.loc[layer,'spike_count']
            if sc<=0: return None
            x.append(np.log(sc)); y.append(n.loc[layer,'gini']); g.append(gv)
    x=np.array(x);y=np.array(y);g=np.array(g,float)
    X=np.column_stack([np.ones_like(x),x,g])
    if np.linalg.matrix_rank(X)<3: return None
    beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    res=y-X@beta; dof=len(y)-3
    if dof<2: return None
    s2=res@res/dof; cov=s2*np.linalg.pinv(X.T@X)
    se=np.sqrt(cov[2,2])
    return beta[2],se,beta[1],dof,len(y)

def tbl(A,B,title):
    print(f'\n=== {title} ===')
    print(f'  모형: gini = a + b*ln(spikes) + c*[{A}].  c>0 = 같은 스파이크에서 {A} 가 더 편중')
    print(f"{'layer':22s} {'n런':>4s} {'b(기울기)':>9s} {'c':>9s} {'SE':>8s} {'t':>7s} {'p':>8s}")
    cs=[];ts=[]
    for l in LS:
        r=ancova(A,B,l)
        if r is None: print(f'{l:22s}  판단 불가'); continue
        c,se,b,dof,nn=r; t=c/se; p=2*stats.t.sf(abs(t),dof)
        cs.append(c); ts.append(t)
        print(f"{l:22s} {nn:4d} {b:9.4f} {c:+9.4f} {se:8.4f} {t:+7.2f} {p:8.4f}")
    cs=np.array(cs);ts=np.array(ts)
    print(f"  18층 중 유효 {len(cs)}: c 중앙값 {np.median(cs):+.4f}  |t|>=2 인 층 {int((np.abs(ts)>=2).sum())}  |t|>=3 {int((np.abs(ts)>=3).sum())}  부호 +{int((cs>0).sum())}/-{int((cs<0).sum())}")
    # 층 전체 결합 (Stouffer)
    z=np.sum(ts)/np.sqrt(len(ts))
    print(f"  층 결합(Stouffer, 층 독립 가정) z={z:+.2f}  ※층끼리 상관이 있어 과대평가 가능")

tbl('ours_ch','l2_lr','Q1-a  ours_ch vs l2_lr (plain L2 + 같은 loss-ratio 제어기)')
tbl('ours_ch','plain L2','Q1-b  ours_ch vs plain L2 (고정 lambda)')
tbl('l2_lr','plain L2','Q1-대조  l2_lr vs plain L2 (제어기만 다른 두 plain L2) -> 0 이어야 정상')
tbl('ours','plain L2','Q1-대조  ours vs plain L2 (이미 갈린 것으로 아는 쌍)')
tbl('ours_layer','ours','Q2-a  ours_layer vs ours')
tbl('ours_w1','ours','Q2-b  ours_w1 vs ours')
