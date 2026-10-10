"""P1 — 채널별 최대 발화의 분포: '죽는다' vs '줄어든다' 를 가른다. (26-09-21)

npz 는 테스트 10,000장 전체 누적이다. 팔당 n=1 (시드 SD 없음).
채널 = 각 층 마지막 축. fc1_n 은 (256,) 1D 라 채널=뉴런.
스파이크·채널 수는 raw 정수로 적는다.
"""
import sys; sys.path.insert(0,'/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
from npz_neuron_dist_v2 import ARMS, load, layers, chan_sums, gini
import numpy as np

# 2026-09-21 확정 명칭
NAME = {
 'baseline':'baseline', 'plain L2 5e-8':'plain L2 5e-8', 'plain L2 5e-7':'plain L2 5e-7',
 'ours 4e-3':'ours_intra_ch 4e-3 (w2)', 'ours 5e-3':'ours_intra_ch 5e-3 (w2)',
 'ours_w1 3e-3':'ours_intra_ch 3e-3 (w1)', 'ours_layer 4e-3':'ours_layer 4e-3',
 'ours_ch 2e-3':'ours_inter_ch 2e-3',
}
SEL=['baseline','plain L2 5e-8','plain L2 5e-7','ours_inter_ch 2e-3','ours_intra_ch 3e-3 (w1)',
     'ours_intra_ch 4e-3 (w2)','ours_intra_ch 5e-3 (w2)','ours_layer 4e-3']
RAWSEL={NAME[a]:(a,t,s) for a,t,s in [(a,t,s) for a,t,s in ARMS if a in NAME]}

D={}
for disp,(orig,tag,srv) in RAWSEL.items():
    d,ls,per,allv,raw = load(tag)
    D[disp]=dict(raw=raw, ls=ls, srv=srv, tag=tag, ns=int(d['n_samples']), T=int(d['T']), acc=float(d['acc']))
LS=D['baseline']['ls']
NS, T = D['baseline']['ns'], D['baseline']['T']
MAXC = NS*T   # 한 뉴런이 낼 수 있는 최대 스파이크 수 = 10,000 x 4

for k in D:
    r=D[k]['raw']
    D[k]['cmax']=np.concatenate([ (np.asarray(r[l]).reshape(-1,np.asarray(r[l]).shape[-1]).max(axis=0)
                                   if np.asarray(r[l]).ndim>1 else np.asarray(r[l]).copy()) for l in LS])
    D[k]['csum']=np.concatenate([chan_sums(r[l]) for l in LS])
    D[k]['nper']=np.concatenate([np.full(np.asarray(r[l]).shape[-1],
                                 int(np.asarray(r[l]).size/np.asarray(r[l]).shape[-1])) for l in LS])
    D[k]['spk_tot']=int(sum(np.asarray(r[l]).sum() for l in LS))

NCH=D['baseline']['cmax'].size
print(f'채널 총수 = {NCH:,d} (18 스파이킹층, n_in/predictions_n 제외).  샘플 {NS:,d}장, T={T}.  팔당 npz n=1')
print()
print('=== 표 A. 팔 요약 (스파이크는 테스트셋 전체 누적 raw 정수 / 샘플당) ===')
print(f"{'arm':28s} {'srv':>4s} {'acc':>6s} {'spikes/sample':>14s} {'누적 spikes':>16s} {'죽은채널':>9s} {'죽은뉴런':>11s}")
for k in SEL:
    a=D[k]; dead=int((a['cmax']==0).sum())
    dn=int(sum((np.asarray(a['raw'][l])==0).sum() for l in LS))
    print(f"{k:28s} {a['srv']:>4s} {a['acc']:6.2f} {a['spk_tot']/NS:14,.1f} {a['spk_tot']:16,d} {dead:9,d} {dn:11,d}")

print()
print('=== 표 B (P1). 채널별 최대 뉴런 발화수의 분위수 — raw 정수 (한 뉴런 최대 가능 = %d) ==='%MAXC)
qs=[0,1,5,10,25,50,75,90]
print(f"{'arm':28s} " + ' '.join(f'p{q:<3d}'.rjust(9) for q in qs))
for k in SEL:
    v=np.sort(D[k]['cmax'])
    print(f"{k:28s} " + ' '.join(f"{int(round(np.percentile(v,q))):9,d}" for q in qs))

print()
print('=== 표 C (P1, 결정적). 채널 최대 발화수의 구간별 채널 수 ===')
EDG=[0,1,41,401,4001,40001,10**9]
LBL=['= 0 (죽음)','1-40','41-400','401-4,000','4,001-40,000','> 40,000']
print(f"{'arm':28s} " + ' '.join(f'{l:>13s}' for l in LBL))
for k in SEL:
    v=D[k]['cmax']; row=[]
    for i in range(len(LBL)):
        row.append(int(((v>=EDG[i])&(v<EDG[i+1])).sum()))
    print(f"{k:28s} " + ' '.join(f'{c:13,d}' for c in row))

print()
print('=== 표 D (P1 보조). 채널 총발화수 구간별 채널 수 (채널 안 모든 뉴런 합) ===')
EDG2=[0,1,1001,10001,100001,1000001,10**12]
LBL2=['= 0','1-1,000','1,001-10,000','1e4-1e5','1e5-1e6','> 1e6']
print(f"{'arm':28s} " + ' '.join(f'{l:>13s}' for l in LBL2))
for k in SEL:
    v=D[k]['csum']; row=[int(((v>=EDG2[i])&(v<EDG2[i+1])).sum()) for i in range(len(LBL2))]
    print(f"{k:28s} " + ' '.join(f'{c:13,d}' for c in row))

print()
print('=== 표 E. 층별 죽은 채널 수 ===')
print(f"{'layer':22s} {'채널':>5s} " + ' '.join(f'{k[:13]:>14s}' for k in SEL))
off=0
for l in LS:
    nc=np.asarray(D['baseline']['raw'][l]).shape[-1]
    line=f"{l:22s} {nc:5,d} "
    for k in SEL:
        line += f"{int((D[k]['cmax'][off:off+nc]==0).sum()):14,d}"
    print(line); off+=nc
print(f"{'TOTAL':22s} {NCH:5,d} " + ' '.join(f"{int((D[k]['cmax']==0).sum()):14,d}" for k in SEL))
np.save('/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad/mech/ls.npy', np.array(LS,dtype=object), allow_pickle=True)
