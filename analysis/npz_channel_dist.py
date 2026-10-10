import sys; sys.path.insert(0,'/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
from npz_neuron_dist_v2 import *
import numpy as np
SEL=['baseline','plain L2 5e-8','ours_ch 2e-3','plain L2 5e-7','ours 4e-3','ours_layer 4e-3','ours_w1 3e-3','ours 5e-3']
data={}
for name,tag,srv in ARMS:
    if name in SEL: data[name]=load(tag)
LS=data['baseline'][1]
CH={a:{l:chan_sums(data[a][4][l]) for l in LS} for a in SEL}
NC={l:CH['baseline'][l].size for l in LS}
print('=== 채널 단위 완전 침묵 채널 수 (테스트 10,000장 전체에서 해당 채널의 모든 뉴런·모든 타임스텝 발화 0) ===')
print('    fc1_n 은 (256,) 1D -> 채널=뉴런. 나머지는 마지막 축이 채널.  npz 각 팔 n=1')
print(f"{'layer':22s} {'채널수':>6s} " + ' '.join(f"{a[:12]:>13s}" for a in SEL))
tot={a:0 for a in SEL}; totc=0
for l in LS:
    totc+=NC[l]
    line=f"{l:22s} {NC[l]:6,d} "
    for a in SEL:
        d=int((CH[a][l]==0).sum()); tot[a]+=d
        line+=f"{d:13,d}"
    print(line)
print(f"{'TOTAL':22s} {totc:6,d} " + ' '.join(f"{tot[a]:13,d}" for a in SEL))
print(f"{'(%)':22s} {'':6s} " + ' '.join(f"{100*tot[a]/totc:12.2f}%" for a in SEL))

print('\n=== 채널 단위 gini (채널별 총 발화의 불평등) ===')
print(f"{'layer':22s} " + ' '.join(f"{a[:12]:>13s}" for a in SEL))
for l in LS:
    print(f"{l:22s} " + ' '.join(f"{gini(CH[a][l]):13.4f}" for a in SEL))
allc={a:np.concatenate([CH[a][l] for l in LS]) for a in SEL}
print(f"{'전역(18층 풀링)':22s} " + ' '.join(f"{gini(allc[a]):13.4f}" for a in SEL))

print('\n=== 거의 침묵 채널: 채널 총발화 < 10,000 (샘플당 평균 <1 스파이크) ===')
print(f"{'layer':22s} {'채널수':>6s} " + ' '.join(f"{a[:12]:>13s}" for a in SEL))
t2={a:0 for a in SEL}
for l in LS:
    line=f"{l:22s} {NC[l]:6,d} "
    for a in SEL:
        d=int((CH[a][l]<10000).sum()); t2[a]+=d
        line+=f"{d:13,d}"
    print(line)
print(f"{'TOTAL':22s} {totc:6,d} " + ' '.join(f"{t2[a]:13,d}" for a in SEL))
print(f"{'(%)':22s} {'':6s} " + ' '.join(f"{100*t2[a]/totc:12.2f}%" for a in SEL))
