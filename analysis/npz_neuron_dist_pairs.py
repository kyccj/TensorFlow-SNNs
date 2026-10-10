"""A쌍/B쌍 분포 차이의 유의성 (26-09-21).

npz 는 팔당 n=1 이라 시드 재현성을 못 잰다. 그래서 σ 분모를 npz 가 아니라
reg_neuron_detail.csv 의 시드 간 SD 에서 가져온다 (plain L2 5e-7 n=5, ours 4e-3 n=5,
baseline n=5, - 1- inversion 4e-3 n=2, 모두 최종 에폭 309).

프로토콜 주의: CSV gini 는 npz gini 와 층별로 r=0.99, 평균 오프셋 -0.026 으로 일치한다
(그래서 σ 분모로 쓸 수 있다). CSV dead_neuron_ratio 는 '배치 내 침묵' 이라 npz 의
'10,000장 전체 영구 침묵' 과 r=0.19~0.42 로 어긋난다 — 침묵률에는 이 σ 를 쓰지 않는다.
"""
import pandas as pd, numpy as np, glob, os
R='/media/hdd1/kyccj/EIP/paper'
groups={'plain L2 5e-7':[f'{R}/r19c10-l2-5e-7-s{i}' for i in range(1,6)],
        'ours 4e-3':[f'{R}/r19c10-prop-4e-3-s{i}' for i in range(1,6)],
        'baseline':[f'{R}/r19c10-base---s{i}' for i in [1,2,3,11,12]],
        '- 1- inv 4e-3':[f'{R}/r19c10-abl_noinv-4e-3-s1',f'{R}/r19c10-abl_noinv-4e-3-s3']}
LAY=None
out={}
for g,ds in groups.items():
    rows=[]
    for d in ds:
        f=os.path.join(d,'reg_neuron_detail.csv')
        if not os.path.exists(f): print('MISS',f); continue
        df=pd.read_csv(f)
        e=df.epoch.max()
        sub=df[df.epoch==e].set_index('layer')
        rows.append((os.path.basename(d),e,sub))
    out[g]=rows
    print(g, [(r[0],r[1],len(r[2])) for r in rows])

LAY=[l for l in out['plain L2 5e-7'][0][2].index if l not in ('n_in','predictions_n')]
print('\nlayers',len(LAY))
print('\n=== 시드 간 SD (마지막 에폭, reg_neuron_detail.csv 프로토콜) ===')
print(f"{'layer':24s} "+' '.join(f"{g[:13]:>28s}" for g in ['plain L2 5e-7','ours 4e-3']))
sds={}
for g in ['plain L2 5e-7','ours 4e-3','baseline','- 1- inv 4e-3']:
    sds[g]={}
    for l in LAY:
        v=np.array([r[2].loc[l,'gini'] for r in out[g]])
        dv=np.array([r[2].loc[l,'dead_neuron_ratio'] for r in out[g]])
        t10=np.array([r[2].loc[l,'top10_share'] for r in out[g]])
        sds[g][l]=dict(n=len(v),gini_m=v.mean(),gini_sd=v.std(ddof=1) if len(v)>1 else np.nan,
                       dead_m=dv.mean(),dead_sd=dv.std(ddof=1) if len(v)>1 else np.nan,
                       t10_m=t10.mean(),t10_sd=t10.std(ddof=1) if len(v)>1 else np.nan)
print(f"{'layer':24s} {'L2 gini':>10s} {'sd':>8s} {'ours gini':>10s} {'sd':>8s} {'diff':>8s} {'pooled sd':>10s} {'sigma':>7s}")
for l in LAY:
    a=sds['plain L2 5e-7'][l]; b=sds['ours 4e-3'][l]
    ps=np.sqrt((a['gini_sd']**2+b['gini_sd']**2)/2)
    d=b['gini_m']-a['gini_m']
    print(f"{l:24s} {a['gini_m']:10.4f} {a['gini_sd']:8.4f} {b['gini_m']:10.4f} {b['gini_sd']:8.4f} {d:8.4f} {ps:10.4f} {d/ps if ps>0 else np.nan:7.1f}")
import pickle; pickle.dump(sds,open('/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad/sds.pkl','wb'))

# --- 쌍별 유의성 ---
import sys; sys.path.insert(0,'/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
from npz_neuron_dist import *
import numpy as np, pickle
sds=pickle.load(open('/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad/sds.pkl','rb'))
data={n:load(t) for n,t in ARMS}
LS=data['baseline'][1]
g=lambda a,l: metrics(data[a][2][l])['gini']
sp=lambda a,l: metrics(data[a][2][l])['spikes_total']

def blk(title,hi,lo,sdhi,sdlo):
    print(f'\n=== {title} ===')
    print(f"{'layer':22s} {'spk비':>7s} {'gini_lo':>8s} {'gini_hi':>8s} {'diff':>8s} {'seedSD':>8s} {'sigma':>7s}")
    for l in LS:
        d=g(hi,l)-g(lo,l)
        s=np.sqrt((sds[sdhi][l]['gini_sd']**2+sds[sdlo][l]['gini_sd']**2)/2)
        print(f"{l:22s} {sp(hi,l)/sp(lo,l):7.3f} {g(lo,l):8.4f} {g(hi,l):8.4f} {d:+8.4f} {s:8.4f} {d/s:+7.1f}")

blk('A쌍  ours 4e-3 - plain L2 5e-7  (seedSD: 각 n=5)','ours 4e-3','plain L2 5e-7','ours 4e-3','plain L2 5e-7')
blk('B쌍  ours 5e-3 - (- 1- inversion 4e-2)  (seedSD 대리: ours 4e-3 n=5 / - 1- inv 4e-3 n=2)',
    'ours 5e-3','- 1- inversion 4e-2','ours 4e-3','- 1- inv 4e-3')
blk('참조  (- 1- inversion 4e-3) - baseline','- 1- inversion 4e-3','baseline','- 1- inv 4e-3','baseline')

print('\n=== 전역 요약 ===')
for n,_ in ARMS:
    m=metrics(data[n][3]); print(f"{n:22s} gini {m['gini']:.4f}  침묵 {m['silent']:,d} ({m['silent_pct']:.2f}%)  활성 {m['active']:,d}")
