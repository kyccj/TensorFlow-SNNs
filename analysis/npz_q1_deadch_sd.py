"""Q1 — 채널 사망 잔차에 시드 SD 를 붙인다 (26-09-22).
   방법은 npz_mech_p4b.py 의 excess() 와 동일: plain L2 3점(baseline, l2 5e-8, l2 5e-7)을
   ln(spikes) 에서 선형보간해 예측 사망 채널 수를 만들고 실측에서 뺀다.
   달라진 것: 팔마다 시드가 여러 개라 잔차의 SD 를 실측한다."""
import json, os, numpy as np
from scipy import stats
SCR='/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
R=json.load(open(os.path.join(SCR,'seedsd','npz_seed_metrics.json')))

REF=['baseline','plain L2 5e-8','plain L2 5e-7']
rx=np.log([R[k][0]['spk_per_sample'] for k in REF]); ry=np.array([R[k][0]['dead_ch'] for k in REF],float)
def pred(spk):
    L=np.log(spk)
    if L>=rx[1]:
        f=(rx[0]-L)/(rx[0]-rx[1]); return ry[0]+f*(ry[1]-ry[0]), 'baseline~5e-8'
    f=(rx[1]-L)/(rx[1]-rx[2]); return ry[1]+f*(ry[2]-ry[1]), ('5e-8~5e-7'+('  [외삽]' if L<rx[2] else ''))
SLOPE=(ry[2]-ry[1])/(rx[2]-rx[1])

print('=== 표 Q1-a. 채널 사망: 개별 시드 전부 + 그룹 평균 행 (4,736 채널, 18 스파이킹층) ===')
print('    excess = 실측 사망채널 − plain L2 곡선 예측 (ln-spikes 선형보간). 기준 3점은 각 n=1')
print(f"{'arm':22s} {'seed':>5s} {'srv':>4s} {'val acc':>8s} {'spikes/sample':>14s} {'죽은채널':>9s} {'예측':>8s} {'excess':>8s}  구간")
ORDER=['baseline','plain L2 5e-8','plain L2 5e-7','ours_inter_ch 2e-3','ours_intra_ch 4e-3',
       'ours_layer 4e-3','ours_w1 3e-3','ours_intra_ch 5e-3','- 1- inversion 4e-3','- 1- inversion 4e-2']
E={}
for arm in ORDER:
    ex=[]; sp=[]; dc=[]
    for m in R[arm]:
        p,seg=pred(m['spk_per_sample']); e=m['dead_ch']-p
        ex.append(e); sp.append(m['spk_per_sample']); dc.append(m['dead_ch'])
        print(f"{arm:22s} {m['seed']:>5s} {m['srv']:>4s} {m['acc']:8.2f} {m['spk_per_sample']:14,.1f} "
              f"{m['dead_ch']:9,d} {p:8.1f} {e:+8.1f}  {seg}")
    ex=np.array(ex); sp=np.array(sp); dc=np.array(dc,float)
    E[arm]=dict(ex=ex,sp=sp,dc=dc,n=len(ex))
    if len(ex)>1:
        print(f"{'  └ 평균 (n=%d)'%len(ex):22s} {'':>5s} {'':>4s} "
              f"{np.mean([m['acc'] for m in R[arm]]):8.2f} {sp.mean():14,.1f} "
              f"{dc.mean():9,.1f} {'':>8s} {ex.mean():+8.1f}   SD(시드) 죽은채널 {dc.std(ddof=1):.1f} / excess {ex.std(ddof=1):.1f}")
    else:
        print(f"{'  └ (n=1, SD 없음)':22s}")
    print()

print('=== 표 Q1-b. 팔 내 시드 SD (자기 SD) ===')
print(f"{'arm':22s} {'n':>3s} {'죽은채널 평균':>13s} {'SD':>7s} {'SEM':>7s} {'excess 평균':>12s} {'SD':>7s} {'SEM':>7s}")
for arm in ORDER:
    a=E[arm]
    if a['n']<2:
        print(f"{arm:22s} {a['n']:3d} {a['dc'][0]:13,.0f} {'—':>7s} {'—':>7s} {a['ex'][0]:+12.1f} {'—':>7s} {'—':>7s}   판단 불가(n=1)")
        continue
    print(f"{arm:22s} {a['n']:3d} {a['dc'].mean():13,.1f} {a['dc'].std(ddof=1):7.1f} {a['dc'].std(ddof=1)/np.sqrt(a['n']):7.1f} "
          f"{a['ex'].mean():+12.1f} {a['ex'].std(ddof=1):7.1f} {a['ex'].std(ddof=1)/np.sqrt(a['n']):7.1f}")

A='ours_layer 4e-3'; B='ours_intra_ch 4e-3'
a,b=E[A],E[B]
d=a['ex'].mean()-b['ex'].mean()
print(f"\n=== Q1 핵심: {A} − {B} 의 excess 차 ===")
print(f"  어제 값(각 n=1): +152 vs +52 = 100채널.  오늘(n={a['n']} vs n={b['n']}): "
      f"{a['ex'].mean():+.1f} vs {b['ex'].mean():+.1f} = {d:+.1f}채널")
sa,sb=a['ex'].std(ddof=1),b['ex'].std(ddof=1)
# (1) 자기 SD (Welch)
se=np.sqrt(sa**2/a['n']+sb**2/b['n'])
dof=(sa**2/a['n']+sb**2/b['n'])**2/((sa**2/a['n'])**2/(a['n']-1)+(sb**2/b['n'])**2/(b['n']-1))
t=d/se; p=2*stats.t.sf(abs(t),dof)
print(f"  (1) 자기 SD 사용   : SD_layer={sa:.1f}(n={a['n']}, 자유도 {a['n']-1}) SD_intra={sb:.1f}(n={b['n']}) "
      f"-> SE={se:.1f}  t={t:+.2f}  Welch dof={dof:.1f}  p={p:.4f}   ({abs(t):.1f}σ)")
print(f"      95% CI {d-stats.t.ppf(0.975,dof)*se:+.1f} ~ {d+stats.t.ppf(0.975,dof)*se:+.1f} 채널")
# (2) 대리 SD: intra(n=5) 의 SD 를 layer 에도 쓴다
se2=sb*np.sqrt(1/a['n']+1/b['n']); dof2=b['n']-1
t2=d/se2; p2=2*stats.t.sf(abs(t2),dof2)
print(f"  (2) 대리 SD 사용   : 양쪽 모두 SD={sb:.1f} (ours_intra_ch n=5 의 SD 를 ours_layer 에 대입) "
      f"-> SE={se2:.1f}  t={t2:+.2f}  dof={dof2}  p={p2:.4f}   ({abs(t2):.1f}σ)")
print(f"      95% CI {d-stats.t.ppf(0.975,dof2)*se2:+.1f} ~ {d+stats.t.ppf(0.975,dof2)*se2:+.1f} 채널")
# (3) 합동 SD
sp_=np.sqrt(((a['n']-1)*sa**2+(b['n']-1)*sb**2)/(a['n']+b['n']-2)); se3=sp_*np.sqrt(1/a['n']+1/b['n']); dof3=a['n']+b['n']-2
t3=d/se3
print(f"  (3) 합동 SD        : SD_pooled={sp_:.1f} -> SE={se3:.1f}  t={t3:+.2f}  dof={dof3}  "
      f"p={2*stats.t.sf(abs(t3),dof3):.4f}   ({abs(t3):.1f}σ)")

print(f"\n=== Q1 보강: 기준곡선을 안 쓰는 직접 비교 (기준 3점의 n=1 불확실성을 제거) ===")
dd=a['dc'].mean()-b['dc'].mean()
lnd=np.log(a['sp'].mean())-np.log(b['sp'].mean())
adj=SLOPE*lnd
print(f"  원 사망채널 차       : {a['dc'].mean():.1f} − {b['dc'].mean():.1f} = {dd:+.1f}채널 (보정 없음)")
print(f"  스파이크 차          : {a['sp'].mean():,.0f} vs {b['sp'].mean():,.0f}  (ln 차 {lnd:+.4f}, {100*(np.exp(lnd)-1):+.2f}%)")
print(f"  plain L2 기울기      : {SLOPE:.1f} 채널/ln-spike (5e-8~5e-7 구간) -> 보정량 {adj:+.1f}채널")
print(f"  보정 후 차           : {dd-adj:+.1f}채널  (= excess 차 {d:+.1f} 과 동일해야 함)")
print(f"  기울기가 ±50% 틀려도 : 보정량이 {adj*0.5:+.1f}~{adj*1.5:+.1f} -> 차는 {dd-adj*0.5:+.1f}~{dd-adj*1.5:+.1f}채널")
print(f"  -> 두 팔이 같은 구간·거의 같은 스파이크라 기준곡선 오차가 차에서 상쇄된다.")

# ANCOVA (기준곡선 무관, 기울기도 데이터에서)
x=np.concatenate([np.log(a['sp']),np.log(b['sp'])]); y=np.concatenate([a['dc'],b['dc']])
g=np.concatenate([np.ones(a['n']),np.zeros(b['n'])])
X=np.column_stack([np.ones_like(x),x,g]); beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
res=y-X@beta; dof4=len(y)-3; s2=res@res/dof4; cov=s2*np.linalg.pinv(X.T@X)
print(f"\n  ANCOVA  dead_ch = a + b·ln(spikes) + c·[ours_layer]   (n={len(y)}런, 기준곡선 미사용)")
print(f"    b={beta[1]:+.1f}±{np.sqrt(cov[1,1]):.1f} 채널/ln-spike (기울기 자체가 유의하지 않을 수 있음)")
print(f"    c={beta[2]:+.1f}±{np.sqrt(cov[2,2]):.1f}  t={beta[2]/np.sqrt(cov[2,2]):+.2f}  dof={dof4}  "
      f"p={2*stats.t.sf(abs(beta[2]/np.sqrt(cov[2,2])),dof4):.4f}")

print(f"\n=== Q1 참고: ours_w1 3e-3 (n=3) 및 ours_inter_ch (n=1) ===")
for arm in ['ours_w1 3e-3','ours_inter_ch 2e-3','ours_intra_ch 5e-3']:
    a2=E[arm]
    if a2['n']>1:
        s=a2['ex'].std(ddof=1); se_=s/np.sqrt(a2['n'])
        dz=a2['ex'].mean()-b['ex'].mean()
        sew=np.sqrt(s**2/a2['n']+sb**2/b['n'])
        print(f"  {arm:22s} n={a2['n']} excess {a2['ex'].mean():+.1f}±{s:.1f}(SD,자기) "
              f"| vs ours_intra_ch 차 {dz:+.1f}  {abs(dz/sew):.1f}σ")
    else:
        print(f"  {arm:22s} n=1 excess {a2['ex'][0]:+.1f}  — σ 없음, 판단 불가")
