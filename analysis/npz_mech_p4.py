"""P4 — 대역이 맞는 기준선으로 채널 사망 잔차를 다시 잰다 + sc_rate_mean(실측 가중 평균). 26-09-21"""
import os, glob, numpy as np, pandas as pd
SCR='/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
ROOTS=['/media/hdd1/kyccj/EIP/paper', os.path.join(SCR,'sejong')]
EXCL=('n_in','predictions_n')
def find(run):
    for r in ROOTS:
        f=os.path.join(r,run,'reg_detail.csv')
        if os.path.exists(f):
            h=os.path.join(r,run,'host.txt')
            srv='23' if (os.path.exists(h) and '23' in open(h).read() and '138' not in open(h).read()) else ('138' if r.startswith('/media') else '23')
            return f,srv
    return None,None
runs=set()
for r in ROOTS:
    for p in glob.glob(os.path.join(r,'r19c10-*')): runs.add(os.path.basename(p))
rows=[]
for run in sorted(runs):
    f,srv=find(run)
    if not f: continue
    try: df=pd.read_csv(f)
    except Exception: continue
    if 'spike_count' not in df.columns or df.empty: continue
    e=int(df.epoch.max())
    sub=df[(df.epoch==e)&(~df.layer.isin(EXCL))]
    if sub.empty: continue
    parts=run.split('-'); meth=parts[1]; lmb=parts[2]; seed=parts[3] if len(parts)>3 else ''
    rows.append(dict(run=run, method=meth, lmb=lmb, seed=seed, srv=srv, ep=e,
                     spikes=float(sub.spike_count.sum()),
                     sc_rate_mean=float(sub.sc_rate_mean.mean()) if 'sc_rate_mean' in sub else np.nan,
                     nlay=len(sub)))
R=pd.DataFrame(rows).sort_values(['method','spikes'])
pd.set_option('display.width',250)
print('=== 표 F. 전 런 최종에폭 층합 spike_count (reg_detail, raw). n 은 런 개수 ===')
for m,g in R.groupby('method'):
    if m not in ('base','l2','l2_lr','prop','ours_ch','ours_layer','ours_w1','ours_ch_b4','ours_ch_b7'): continue
    print(f'--- {m}  (n={len(g)} runs)')
    for _,r in g.iterrows():
        print(f"   {r.run:34s} {r.srv:>4s} ep{r.ep:4d} spikes {r.spikes:12,.1f}  sc_rate_mean {r.sc_rate_mean:8.4f}  layers {r.nlay}")
R.to_csv(os.path.join(SCR,'mech','regdetail_all.csv'),index=False)
