"""P3 — (a) 팔 간 채널 번호가 정렬돼 있는가? (b) 시드 SD (dead_neuron_ratio 기반). 26-09-21"""
import sys,os,glob; sys.path.insert(0,'/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
from npz_neuron_dist_v2 import load, chan_sums
import numpy as np, pandas as pd
from scipy.stats import spearmanr
SCR='/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
ROOTS=['/media/hdd1/kyccj/EIP/paper', os.path.join(SCR,'sejong')]
EXCL=('n_in','predictions_n')
T={'baseline s11':'r19c10-base---s11','plain L2 5e-8 s1':'r19c10-l2-5e-8-s1',
   'plain L2 5e-7 s5':'r19c10-l2-5e-7-s5','ours_inter_ch 2e-3 s1':'r19c10-ours_ch-2e-3-s1',
   'ours_intra_ch 3e-3 w1 s1':'r19c10-ours_w1-3e-3-s1','ours_intra_ch 4e-3 s4':'r19c10-prop-4e-3-s4',
   'ours_layer 4e-3 s2':'r19c10-ours_layer-4e-3-s2'}
D={k:load(v) for k,v in T.items()}
LS=D['baseline s11'][1]
CS={k:{l:chan_sums(v[4][l]) for l in LS} for k,v in D.items()}
print('=== 표 H (P3a). 팔 쌍 간 **채널 번호가 같은 것을 가리키는가** — 층별 Spearman rho 의 18층 평균 ===')
print('    정렬돼 있으면 rho 가 0 에서 크게 떨어져야 한다. s1 끼리는 같은 시드다.')
ks=list(T)
print(f"{'':26s} " + ' '.join(f'{k[:11]:>12s}' for k in ks))
for a in ks:
    row=f'{a:26s} '
    for b in ks:
        if a==b: row+=f"{'-':>12s}"; continue
        r=np.mean([spearmanr(CS[a][l],CS[b][l]).correlation for l in LS if CS[a][l].size>3])
        row+=f'{r:12.4f}'
    print(row)

print()
print('=== 표 I (P3b). 시드 SD — 최종 에폭 dead_neuron_ratio 를 뉴런 수로 환산한 층합 (npz 아님, 전 런) ===')
NPL={l:int(np.asarray(D['baseline s11'][4][l]).size) for l in LS}
def dead_of(run):
    for r in ROOTS:
        f=os.path.join(r,run,'reg_neuron_detail.csv')
        if os.path.exists(f):
            df=pd.read_csv(f); e=df.epoch.max(); s=df[df.epoch==e].set_index('layer')
            return sum(float(s.loc[l,'dead_neuron_ratio'])*NPL[l] for l in LS if l in s.index)
    return None
GRP={'baseline':'r19c10-base---s*','plain L2 5e-7':'r19c10-l2-5e-7-s*','ours_intra_ch 4e-3 (w2)':'r19c10-prop-4e-3-s*',
     'ours_intra_ch 3e-3 (w2)':'r19c10-prop-3e-3-s*','ours_intra_ch 3e-3 (w1)':'r19c10-ours_w1-3e-3-s*',
     'ours_intra_ch 2p5e-3 (w1)':'r19c10-ours_w1-2p5e-3-s*','ours_layer 4e-3':'r19c10-ours_layer-4e-3-s*',
     'ours_layer 3e-3':'r19c10-ours_layer-3e-3-s*','ours_inter_ch 2e-3':'r19c10-ours_ch-2e-3-s*',
     'ours_inter_ch 3e-3':'r19c10-ours_ch-3e-3-s*','ours_inter_ch 4e-3':'r19c10-ours_ch-4e-3-s*'}
rd=pd.read_csv(os.path.join(SCR,'mech','regdetail_all.csv')).set_index('run')
for g,pat in GRP.items():
    runs=sorted({os.path.basename(p) for r in ROOTS for p in glob.glob(os.path.join(r,pat))})
    vals=[]
    for run in runs:
        d=dead_of(run)
        if d is None or run not in rd.index: continue
        if rd.loc[run,'ep']<300: print(f"   [제외] {run} ep{int(rd.loc[run,'ep'])} 미완주 (평균에서 뺌, 행은 남김)"); continue
        vals.append((run,d,rd.loc[run,'spikes']))
    if not vals: continue
    print(f'--- {g}  n={len(vals)}')
    for run,d,sp in vals: print(f"      {run:30s} dead_neuron {d:10,.0f}   spikes(train) {sp:11,.1f}")
    dv=np.array([v[1] for v in vals]); sv=np.array([v[2] for v in vals])
    sd=dv.std(ddof=1) if len(dv)>1 else np.nan
    sds=sv.std(ddof=1) if len(sv)>1 else np.nan
    print(f"      {'평균':30s} dead_neuron {dv.mean():10,.0f}  (SD {sd:,.0f})   spikes {sv.mean():11,.1f}  (SD {sds:,.1f})")
