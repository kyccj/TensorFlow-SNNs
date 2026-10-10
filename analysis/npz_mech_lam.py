"""λ(adp_lambda)와 reg_R(= 층별 ||s*w|| 합)를 최종 에폭에서 뽑는다.
loss-ratio 제어기가 λ 를 정하므로, 같은 target 에서도 팔마다 λ·w^2/||x|| (뉴런이 받는 압력)이 다르다."""
import os,re,glob,numpy as np,pandas as pd
SCR='/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
ROOTS=['/media/hdd1/kyccj/EIP/paper', os.path.join(SCR,'sejong')]
EXCL=('n_in','predictions_n')
PAT=re.compile(r'adp_lambda: ([0-9.eE+-]+).*?reg_R: ([0-9.eE+-]+).*?task_loss_clean: ([0-9.eE+-]+).*?loss_ratio: ([0-9.eE+-]+)')
def logs(run):
    for r in ROOTS:
        for nm in ('train.log','train_repo.log','train_store.log'):
            f=os.path.join(r,run,nm)
            if os.path.exists(f): yield f
RUNS=['r19c10-base---s11','r19c10-l2-5e-8-s1','r19c10-l2-5e-7-s5',
      'r19c10-ours_ch-2e-3-s1','r19c10-ours_ch-3e-3-s1','r19c10-ours_ch-4e-3-s1',
      'r19c10-ours_ch_b4-3e-3-s1','r19c10-ours_ch_b7-3e-3-s1',
      'r19c10-ours_w1-3e-3-s1','r19c10-prop-4e-3-s4','r19c10-prop-5e-3-s1',
      'r19c10-ours_layer-4e-3-s1','r19c10-ours_layer-4e-3-s2','r19c10-ours_layer-4e-3-s3']
print(f"{'run':30s} {'lambda':>12s} {'reg_R':>12s} {'task_loss':>10s} {'loss_ratio':>11s} {'sum sc_loss':>12s} {'sc_rate_m':>10s} {'lam/R':>12s}")
for run in RUNS:
    last=None
    for f in logs(run):
        for ln in open(f,errors='ignore'):
            m=PAT.search(ln)
            if m: last=m.groups()
    scl=scr_=np.nan
    for r in ROOTS:
        p=os.path.join(r,run,'reg_detail.csv')
        if os.path.exists(p):
            df=pd.read_csv(p); e=df.epoch.max()
            s=df[(df.epoch==e)&(~df.layer.isin(EXCL))]
            scl=float(s.sc_loss.sum()); scr_=float(s.sc_rate_mean.mean()); break
    if last is None:
        print(f"{run:30s} {'(no log)':>12s} {'':>12s} {'':>10s} {'':>11s} {scl:12,.1f} {scr_:10.4f}")
        continue
    lam,R,tl,lr=[float(x) for x in last]
    print(f"{run:30s} {lam:12.4e} {R:12,.1f} {tl:10.4f} {lr:11.3e} {scl:12,.1f} {scr_:10.4f} {lam/max(R,1e-9):12.4e}")
