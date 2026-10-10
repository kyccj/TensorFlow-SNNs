"""l2_lr 2e-2 (제어기만 붙인 plain L2, n=1) vs ours_intra_ch 4e-3 (n=5) — 가장 깨끗한 짝.
   npz 가 없으므로 reg_neuron_detail.csv(최종 에폭 309) 프로토콜로만 대조한다.
   l2_lr 은 n=1 이라 σ 분모는 ours_intra_ch n=5 의 시드 SD (대리값)."""
import os, numpy as np, pandas as pd
from scipy import stats
SCR='/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
ROOTS=['/media/hdd1/kyccj/EIP/paper', os.path.join(SCR,'sejong')]
EXCL=('n_in','predictions_n')
def get(run):
    for r in ROOTS:
        a=os.path.join(r,run,'reg_neuron_detail.csv'); b=os.path.join(r,run,'reg_detail.csv')
        if os.path.exists(a) and os.path.exists(b):
            n=pd.read_csv(a); s=pd.read_csv(b)
            if n.empty or s.empty: continue
            e=min(n.epoch.max(),s.epoch.max())
            if e<309: continue
            return n[n.epoch==e].set_index('layer'), s[s.epoch==e].set_index('layer')
    return None,None
G={'ours_intra_ch 4e-3':['r19c10-prop-4e-3-s%d'%i for i in range(1,6)],
   'l2_lr 2e-2':['r19c10-l2_lr-2e-2-s1'],
   'l2_lr 1e-2':['r19c10-l2_lr-1e-2-s1'],
   'plain L2 4p5e-7':['r19c10-l2-4p5e-7-s%d'%i for i in [1,2,3]],
   'plain L2 5e-7':['r19c10-l2-5e-7-s%d'%i for i in range(1,6)],
   'ours_layer 4e-3':['r19c10-ours_layer-4e-3-s%d'%i for i in [1,2,3]],
   'ours_w1 3e-3':['r19c10-ours_w1-3e-3-s%d'%i for i in [1,2,3]]}
D={}
for k,rs in G.items():
    D[k]=[]
    for r in rs:
        n,s=get(r)
        if n is None: print('  MISS',r); continue
        D[k].append((r,n,s))
    print(f'{k:22s} n={len(D[k])}  '+', '.join(r.replace("r19c10-","") for r,_,_ in D[k]))
LS=[l for l in D['ours_intra_ch 4e-3'][0][1].index if l not in EXCL]
def col(k,l,c,src=1):
    return np.array([ (n if src==1 else s).loc[l,c] for _,n,s in D[k] ],float)

print('\n=== 총 스파이크 (reg_detail 최종에폭 층합, 18층) ===')
for k in G:
    v=np.array([s.loc[[l for l in LS],'spike_count'].sum() for _,_,s in D[k]])
    if len(v)>1: print(f"  {k:22s} n={len(v)}  {v.mean():12,.1f} ± {v.std(ddof=1):9,.1f}   개별 "+', '.join(f'{x:,.0f}' for x in v))
    else:        print(f"  {k:22s} n=1  {v[0]:12,.1f}      — SD 없음")

B='ours_intra_ch 4e-3'
for A in ['l2_lr 2e-2','plain L2 4p5e-7','ours_layer 4e-3','ours_w1 3e-3']:
    print(f'\n=== {A} vs {B} — 층별 (CSV 프로토콜, 최종에폭 309) ===')
    nA=len(D[A])
    print(f"    σ 분모: "+("자기 시드 SD (n=%d)"%nA if nA>1 else f"ours_intra_ch n=5 의 시드 SD (대리값) — {A} 는 n=1"))
    print(f"{'layer':22s} {'죽은뉴런비 A':>12s} {'B':>8s} {'Δ':>8s} {'σ':>7s} {'σ배':>6s} | "
          f"{'gini A':>8s} {'B':>8s} {'Δ':>8s} {'σ배':>6s} | {'top10 Δ':>9s} {'σ배':>6s} | {'스파이크비':>9s}")
    cnt=[0,0,0]
    for l in LS:
        out=[]
        for j,c in enumerate(['dead_neuron_ratio','gini','top10_share']):
            a=col(A,l,c); b=col(B,l,c); sd=(a.std(ddof=1) if nA>1 else b.std(ddof=1))
            sdb=b.std(ddof=1); s=np.sqrt(sd**2/nA+sdb**2/len(b))
            d=a.mean()-b.mean(); z=d/s if s>0 else np.nan
            if abs(z)>=2: cnt[j]+=1
            out.append((a.mean(),b.mean(),d,s,z))
        spa=np.mean([s.loc[l,'spike_count'] for _,_,s in D[A]]); spb=np.mean([s.loc[l,'spike_count'] for _,_,s in D[B]])
        print(f"{l:22s} {out[0][0]:12.4f} {out[0][1]:8.4f} {out[0][2]:+8.4f} {out[0][3]:7.4f} {out[0][4]:+6.1f} | "
              f"{out[1][0]:8.4f} {out[1][1]:8.4f} {out[1][2]:+8.4f} {out[1][4]:+6.1f} | {out[2][2]:+9.4f} {out[2][4]:+6.1f} | {spa/spb:9.3f}")
    print(f"  -> |σ배|>=2 인 층: 죽은뉴런비 {cnt[0]}/18, gini {cnt[1]}/18, top10 {cnt[2]}/18")
