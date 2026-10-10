"""Q2 — 뉴런 분포 지표(gini / 영구침묵률 / 상위10% 점유)에 팔 내 시드 SD 를 붙인다. 26-09-22.
   전역 gini 의 σ 도 시드 SD 로 낸다 (뉴런 부트스트랩과 대조)."""
import json, os, numpy as np, sys
sys.path.insert(0,'/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
from scipy import stats
SCR='/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
R=json.load(open(os.path.join(SCR,'seedsd','npz_seed_metrics.json')))
LS=R['baseline'][0]['layers']
MULTI=['ours_intra_ch 4e-3','ours_layer 4e-3','ours_w1 3e-3']
SINGLE=['baseline','plain L2 5e-8','plain L2 5e-7','ours_inter_ch 2e-3','ours_intra_ch 5e-3']
ALL=SINGLE[:3]+MULTI+['ours_inter_ch 2e-3','ours_intra_ch 5e-3']

def col(arm,l,k): return np.array([m['lay'][l][k] for m in R[arm]],float)
def g(arm,k):     return np.array([m[k] for m in R[arm]],float)

for metric,key,fmt in [('영구침묵률(%)','silent_pct','%8.2f'),('gini','gini','%8.4f'),('상위10% 점유','top10','%8.4f')]:
    print(f'\n=== 표 Q2-{key}. 층별 {metric}: 평균 ± 시드 SD (자기 SD). 값 없는 σ 는 n=1 ===')
    hdr=f"{'layer':22s} {'뉴런수':>9s} "
    for a in ALL: hdr+=f"{a[:18]:>21s}"
    print(hdr)
    print(f"{'':22s} {'':>9s} " + ''.join(f"{'n=%d'%len(R[a]):>21s}" for a in ALL))
    for l in LS:
        line=f"{l:22s} {R['baseline'][0]['lay'][l]['n_neuron']:9,d} "
        for a in ALL:
            v=col(a,l,key)
            if len(v)>1: line+=f"{(fmt%v.mean()).strip():>13s}±{(fmt%v.std(ddof=1)).strip():>7s}"
            else:        line+=f"{(fmt%v[0]).strip():>13s}{'—':>8s}"
        print(line)
    line=f"{'전역(18층, 1,442,048)':22s} {1442048:9,d} "
    gk={'silent_pct':'g_silent_pct','gini':'g_gini','top10':'g_top10'}[key]
    for a in ALL:
        v=g(a,gk)
        if len(v)>1: line+=f"{(fmt%v.mean()).strip():>13s}±{(fmt%v.std(ddof=1)).strip():>7s}"
        else:        line+=f"{(fmt%v[0]).strip():>13s}{'—':>8s}"
    print(line)

print('\n=== 표 Q2-핵심. conv2_block1_conv1_n 영구침묵률: 개별 시드 전부 ===')
L='conv2_block1_conv1_n'
for a in ALL:
    for m in R[a]:
        print(f"  {a:22s} {m['seed']:>4s} {m['srv']:>4s}  침묵 {m['lay'][L]['silent_pct']:6.2f}%  "
              f"gini {m['lay'][L]['gini']:.4f}  상위10% {m['lay'][L]['top10']:.4f}  spk/샘플 {m['spk_per_sample']:,.0f}")
    v=col(a,L,'silent_pct')
    if len(v)>1: print(f"  {'  └ 평균 (n=%d)'%len(v):22s}       침묵 {v.mean():6.2f}% ± {v.std(ddof=1):.2f}(SD)")
    print()

A,B='ours_layer 4e-3','ours_intra_ch 4e-3'
va,vb=col(A,L,'silent_pct'),col(B,L,'silent_pct')
d=va.mean()-vb.mean(); sa,sb=va.std(ddof=1),vb.std(ddof=1)
se=np.sqrt(sa**2/len(va)+sb**2/len(vb))
dof=(sa**2/len(va)+sb**2/len(vb))**2/((sa**2/len(va))**2/(len(va)-1)+(sb**2/len(vb))**2/(len(vb)-1))
print(f"  어제: ours_intra_ch 75.8 vs ours_layer 77.4 = 1.6%p (각 n=1)")
print(f"  오늘: {vb.mean():.2f} (n={len(vb)}) vs {va.mean():.2f} (n={len(va)}) = {d:+.2f}%p")
print(f"  (1) 자기 SD : SD_layer={sa:.2f} SD_intra={sb:.2f} -> SE={se:.2f}  t={d/se:+.2f}  dof={dof:.1f}  "
      f"p={2*stats.t.sf(abs(d/se),dof):.4f}  ({abs(d/se):.1f}σ)")
se2=sb*np.sqrt(1/len(va)+1/len(vb))
print(f"  (2) 대리 SD (intra n=5 의 SD 를 layer 에도) : SE={se2:.2f}  t={d/se2:+.2f}  dof={len(vb)-1}  "
      f"p={2*stats.t.sf(abs(d/se2),len(vb)-1):.4f}  ({abs(d/se2):.1f}σ)")

print('\n=== 표 Q2-부트스트랩 대조. 전역 gini 의 σ: 시드 SD vs 뉴런 부트스트랩 ===')
print('    시드 SD 는 ours_intra_ch 4e-3 n=5 실측. 부트스트랩은 같은 팔 s1 한 런의 1,442,048 뉴런 재표집.')
import numpy as _np
gv=g('ours_intra_ch 4e-3','g_gini')
print(f"  시드 SD (n=5)      : {gv.mean():.4f} ± {gv.std(ddof=1):.4f}   [{', '.join(f'{x:.4f}' for x in gv)}]")
sys.path.insert(0,'/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
from npz_neuron_dist_v2 import load, gini
d1,ls1,per1,allv1,raw1=load('r19c10-prop-4e-3-s1')
rng=_np.random.default_rng(3); bs=[]
for _ in range(60):
    idx=rng.integers(0,allv1.size,allv1.size); bs.append(gini(allv1[idx]))
bs=_np.array(bs)
print(f"  뉴런 부트스트랩 SE : {bs.mean():.4f} ± {bs.std(ddof=1):.5f}  (60회)")
print(f"  비  율             : 시드 SD 가 부트스트랩 SE 의 {gv.std(ddof=1)/bs.std(ddof=1):.0f}배")
dg=g('ours_layer 4e-3','g_gini').mean()-gv.mean()
print(f"  -> ours_layer−ours_intra_ch 전역 gini 차 {dg:+.4f}: 시드 SD 기준 {abs(dg)/gv.std(ddof=1):.1f}σ, "
      f"부트스트랩 기준 {abs(dg)/bs.std(ddof=1):.0f}σ. 후자를 쓰면 안 된다.")
