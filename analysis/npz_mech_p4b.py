"""P4 — 대역이 맞는 기준선에서 채널 사망 잔차 재계산 + 사망의 '집중도'. 26-09-21. npz 팔당 n=1."""
import sys; sys.path.insert(0,'/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
from npz_neuron_dist_v2 import load, chan_sums
import numpy as np
T=[('baseline',             'r19c10-base---s11',        '138'),
   ('plain L2 5e-8',        'r19c10-l2-5e-8-s1',        '23'),
   ('ours_inter_ch 2e-3',   'r19c10-ours_ch-2e-3-s1',   '23'),
   ('plain L2 5e-7',        'r19c10-l2-5e-7-s5',        '138'),
   ('ours_intra_ch 4e-3 w2','r19c10-prop-4e-3-s4',      '138'),
   ('ours_layer 4e-3',      'r19c10-ours_layer-4e-3-s2','23'),
   ('ours_intra_ch 3e-3 w1','r19c10-ours_w1-3e-3-s1',   '23'),
   ('ours_intra_ch 5e-3 w2','r19c10-prop-5e-3-s1',      '138')]
D={}
for n,t,s in T:
    d,ls,per,allv,raw=load(t)
    cmax=np.concatenate([(np.asarray(raw[l]).reshape(-1,np.asarray(raw[l]).shape[-1]).max(axis=0)
                          if np.asarray(raw[l]).ndim>1 else np.asarray(raw[l]).copy()) for l in ls])
    npl=np.concatenate([np.full(np.asarray(raw[l]).shape[-1], int(np.asarray(raw[l]).size/np.asarray(raw[l]).shape[-1])) for l in ls])
    dn=int(sum((np.asarray(raw[l])==0).sum() for l in ls))
    D[n]=dict(srv=s,acc=float(d['acc']),spk=allv.sum()/10000.0,dead_ch=int((cmax==0).sum()),
              dead_n=dn, n_in_dead_ch=int(npl[cmax==0].sum()), ls=ls)

print('=== 표 J. 대역이 맞는 짝 비교 (회귀·외삽 없음). npz 팔당 n=1, 시드 SD 없음 ===')
print(f"{'arm':24s} {'srv':>4s} {'val acc':>8s} {'spikes/sample':>14s} {'죽은채널':>9s} {'죽은뉴런':>11s} {'죽은채널 속 뉴런':>15s} {'집중도':>8s}")
for n,_,_ in T:
    a=D[n]
    print(f"{n:24s} {a['srv']:>4s} {a['acc']:8.2f} {a['spk']:14,.1f} {a['dead_ch']:9,d} {a['dead_n']:11,d} "
          f"{a['n_in_dead_ch']:15,d} {100*a['n_in_dead_ch']/a['dead_n']:7.2f}%")

def excess(name, lo, hi):
    """plain-L2 곡선(baseline, l2 5e-8, l2 5e-7)을 ln(spikes) 에서 선형보간해 초과 사망 채널 수."""
    x=[D[k]['spk'] for k in ['baseline','plain L2 5e-8','plain L2 5e-7']]
    y=[D[k]['dead_ch'] for k in ['baseline','plain L2 5e-8','plain L2 5e-7']]
    lx=np.log(x); s=D[name]['spk']; L=np.log(s)
    if L>=lx[1]:   # 514k~405k 구간
        f=(lx[0]-L)/(lx[0]-lx[1]); pred=y[0]+f*(y[1]-y[0]); seg='baseline~5e-8'
    else:
        f=(lx[1]-L)/(lx[1]-lx[2]); pred=y[1]+f*(y[2]-y[1]); seg='5e-8~5e-7' + ('  [외삽]' if L<lx[2] else '')
    return pred, D[name]['dead_ch']-pred, seg

print()
print('=== 표 K. plain L2 곡선 대비 초과 사망 채널 (ln-spikes 선형보간, 기준 3점 각 n=1) ===')
print(f"{'arm':24s} {'spikes/sample':>14s} {'실측':>7s} {'plainL2 예측':>13s} {'초과':>8s}  구간")
for n in ['ours_inter_ch 2e-3','ours_intra_ch 4e-3 w2','ours_layer 4e-3','ours_intra_ch 3e-3 w1','ours_intra_ch 5e-3 w2']:
    p,e,seg=excess(n,None,None)
    print(f"{n:24s} {D[n]['spk']:14,.1f} {D[n]['dead_ch']:7,d} {p:13.1f} {e:+8.1f}  {seg}")
print()
print('   plain L2 기준곡선 3점:  ' + ' | '.join(f"{k} {D[k]['spk']:,.0f}->{D[k]['dead_ch']}" for k in ['baseline','plain L2 5e-8','plain L2 5e-7']))
lx=np.log([D[k]['spk'] for k in ['baseline','plain L2 5e-8','plain L2 5e-7']]); yy=[D[k]['dead_ch'] for k in ['baseline','plain L2 5e-8','plain L2 5e-7']]
print(f"   구간 기울기 (채널/ln-spike):  baseline~5e-8 {(yy[1]-yy[0])/(lx[1]-lx[0]):.1f}   5e-8~5e-7 {(yy[2]-yy[1])/(lx[2]-lx[1]):.1f}")
print('   -> 곡선이 선형이 아니다. 두 구간 기울기가 2.0배 차이나므로 대역 보정 자체가 불확실하다.')

print()
print('=== 표 L. 채널 안에서 선택적으로 죽은 뉴런 (= 죽은 뉴런 중 살아있는 채널에 속한 것) ===')
print('    이게 "채널 안 역가중" 의 직접 지문이다. 채널째 죽은 것과 구분된다. npz 팔당 n=1')
TOT=1442048
print(f"{'arm':24s} {'spikes/sample':>14s} {'죽은뉴런':>11s} {'채널째':>11s} {'채널안 선택사망':>14s} {'살아있는채널뉴런':>16s} {'비율':>8s}")
for n,_,_ in T:
    a=D[n]; iso=a['dead_n']-a['n_in_dead_ch']; alive=TOT-a['n_in_dead_ch']
    print(f"{n:24s} {a['spk']:14,.1f} {a['dead_n']:11,d} {a['n_in_dead_ch']:11,d} {iso:14,d} {alive:16,d} {100*iso/alive:7.3f}%")
