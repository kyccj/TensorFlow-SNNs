"""npz 시드 SD (26-09-22). npz 18개 -> 팔 내 시드 SD 를 실측한다.

규약:
  * 전역 gini 의 σ 는 이제 **시드 SD** 로 낸다 (뉴런 부트스트랩 아님).
  * 집계 대상: 스파이킹 18층 (n_in, predictions_n 제외), 1,442,048 뉴런 / 4,736 채널.
  * 채널 = 마지막 축. fc1_n 은 (256,) 이라 채널=뉴런.
"""
import sys, os, json
sys.path.insert(0, '/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
import numpy as np
from npz_neuron_dist_v2 import load, chan_sums, gini, share_top, EXCL

SCR = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
OUT = os.path.join(SCR, 'seedsd')
os.makedirs(OUT, exist_ok=True)

ARMS = {  # arm -> [(tag, seed)]
 'baseline':            [('r19c10-base---s11','s11')],
 'plain L2 5e-8':       [('r19c10-l2-5e-8-s1','s1')],
 'plain L2 5e-7':       [('r19c10-l2-5e-7-s5','s5')],
 'ours_inter_ch 2e-3':  [('r19c10-ours_ch-2e-3-s1','s1')],
 'ours_intra_ch 4e-3':  [('r19c10-prop-4e-3-s%d'%i,'s%d'%i) for i in range(1,6)],
 'ours_intra_ch 5e-3':  [('r19c10-prop-5e-3-s1','s1')],
 'ours_layer 4e-3':     [('r19c10-ours_layer-4e-3-s%d'%i,'s%d'%i) for i in [1,2,3]],
 'ours_w1 3e-3':        [('r19c10-ours_w1-3e-3-s%d'%i,'s%d'%i) for i in [1,2,3]],
 '- 1- inversion 4e-3': [('r19c10-abl_noinv-4e-3-s1','s1')],
 '- 1- inversion 4e-2': [('r19c10-abl_noinv-4e-2-s1','s1')],
}

ROOTS = ['/media/hdd1/kyccj/EIP/paper', os.path.join(SCR, 'sejong')]
def server(tag):
    for r in ROOTS:
        h = os.path.join(r, tag, 'host.txt')
        if os.path.exists(h):
            t = open(h).read()
            if '161.122.23.23' in t or 'sejong' in t: return '23'
            if '161.122.23.138' in t or 'canus' in t: return '138'
    return '?'

def run_metrics(tag):
    d, ls, per, allv, raw = load(tag)
    rec = dict(tag=tag, acc=float(d['acc']), n_samples=int(d['n_samples']), T=int(d['T']),
               spikes_total=float(allv.sum()), spk_per_sample=float(allv.sum())/float(d['n_samples']),
               n_neuron=int(allv.size), layers=ls)
    # 전역
    rec['g_gini'] = gini(allv)
    rec['g_silent_pct'] = 100.0*float((allv==0).mean())
    rec['g_top10'] = share_top(allv, 0.10)
    ch_all = []
    lay = {}
    for l in ls:
        v = per[l]; c = chan_sums(raw[l]); ch_all.append(c)
        lay[l] = dict(n_neuron=int(v.size), n_ch=int(c.size),
                      gini=gini(v), silent_pct=100.0*float((v==0).mean()),
                      top10=share_top(v,0.10), spikes=float(v.sum()),
                      dead_ch=int((c==0).sum()), ch_gini=gini(c))
    ch_all = np.concatenate(ch_all)
    rec['n_ch'] = int(ch_all.size)
    rec['dead_ch'] = int((ch_all==0).sum())
    rec['ch_gini'] = gini(ch_all)
    rec['lay'] = lay
    del d, per, raw, allv
    return rec

R = {}
for arm, runs in ARMS.items():
    R[arm] = []
    for tag, seed in runs:
        m = run_metrics(tag); m['seed']=seed; m['srv']=server(tag)
        R[arm].append(m)
        print(f"loaded {arm:22s} {seed:4s} srv{m['srv']:>4s} acc {m['acc']*100 if m['acc']<1.5 else m['acc']:.2f} "
              f"spk/샘플 {m['spk_per_sample']:,.1f} dead_ch {m['dead_ch']:4d} g_gini {m['g_gini']:.4f}", flush=True)

with open(os.path.join(OUT,'npz_seed_metrics.json'),'w') as f:
    json.dump(R, f)
print('\nsaved', os.path.join(OUT,'npz_seed_metrics.json'))
