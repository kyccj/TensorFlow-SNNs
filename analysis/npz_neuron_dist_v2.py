"""npz 뉴런 분포 v2 — 신규 4팔(ours_ch / plain L2 5e-8 / ours_layer / ours_w1) 합류 (26-09-21).

규약 (이전 세션에서 확정, 그대로 유지):
  * 전역 gini 에 σ 를 붙이지 않는다. 뉴런 부트스트랩은 σ 분모가 아니다
    (N=1,442,048 이라 SE 가 1e-4 수준 → 170~191σ 가 나오고 무의미).
  * σ 는 층별 시드 SD 로만 만든다: reg_neuron_detail.csv 최종 에폭(309)에서
    같은 팔·같은 ρ/λ 의 여러 시드 간 SD.
  * 집계 대상: 스파이킹 18층 (n_in, predictions_n 제외), 총 1,442,048 뉴런.
"""
import numpy as np, os, pandas as pd

NPZ = '/home/kyccj/PycharmProjects/TensorFlow-SNNs/_npz'
SCR = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
EXCL = ('n_in', 'predictions_n')

ARMS = [
    ('baseline',            'r19c10-base---s11',        '138'),
    ('plain L2 5e-8',       'r19c10-l2-5e-8-s1',        '23'),
    ('ours_ch 2e-3',        'r19c10-ours_ch-2e-3-s1',   '23'),
    ('plain L2 5e-7',       'r19c10-l2-5e-7-s5',        '138'),
    ('ours 4e-3',           'r19c10-prop-4e-3-s4',      '138'),
    ('ours_layer 4e-3',     'r19c10-ours_layer-4e-3-s2','23'),
    ('ours_w1 3e-3',        'r19c10-ours_w1-3e-3-s1',   '23'),
    ('ours 5e-3',           'r19c10-prop-5e-3-s1',      '138'),
    ('- 1- inversion 4e-3', 'r19c10-abl_noinv-4e-3-s1', '138'),
    ('- 1- inversion 4e-2', 'r19c10-abl_noinv-4e-2-s1', '138'),
]

def layers(d):
    # 26-10-10: npz 에 __t/__first_t/logits 등이 추가됐다. 층 키 = __fired 짝이 있는 키만.
    ks = [k for k in d.files if not k.endswith('__fired') and k + '__fired' in d.files and k not in ('n_samples','T','acc')]
    return [k for k in ks if k not in EXCL]

def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); n = x.size; s = x.sum()
    if s <= 0: return 0.0
    idx = np.arange(1, n+1)
    return float((2.0*(idx*x).sum())/(n*s) - (n+1.0)/n)

def share_top(x, frac):
    x = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    k = max(1, int(round(x.size*frac)))
    return float(x[:k].sum()/x.sum())

def frac_neurons_for(x, target):
    x = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    c = np.cumsum(x)/x.sum()
    return float((int(np.searchsorted(c, target))+1)/x.size)

def metrics(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(n_neuron=int(v.size), silent=int((v==0).sum()),
                silent_pct=100.0*float((v==0).mean()), active=int((v>0).sum()),
                spikes_total=float(v.sum()), gini=gini(v),
                top1=share_top(v,0.01), top10=share_top(v,0.10),
                f50=frac_neurons_for(v,0.50), f80=frac_neurons_for(v,0.80),
                f90=frac_neurons_for(v,0.90), f99=frac_neurons_for(v,0.99))

def load(tag):
    """per: 1D 뉴런 벡터, shp: 원래 (H,W,C) 모양 (채널 단위 집계용)."""
    d = np.load(os.path.join(NPZ, tag+'.npz'))
    ls = layers(d)
    raw = {k: d[k].astype(np.float64) for k in ls}
    per = {k: raw[k].reshape(-1) for k in ls}
    allv = np.concatenate([per[k] for k in ls])
    return d, ls, per, allv, raw

def chan_sums(arr):
    """층 배열 -> 채널별 총 발화 (마지막 축이 채널). fc1 은 (256,) 이라 채널=뉴런."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1: return a.copy()
    return a.reshape(-1, a.shape[-1]).sum(axis=0)

# ---- reg_neuron_detail.csv 시드 SD ----
ROOTS = ['/media/hdd1/kyccj/EIP/paper', os.path.join(SCR, 'sejong')]

def rd(run):
    for r in ROOTS:
        f = os.path.join(r, run, 'reg_neuron_detail.csv')
        if os.path.exists(f):
            df = pd.read_csv(f)
            e = df.epoch.max()
            return e, df[df.epoch == e].set_index('layer')
    return None, None

def seed_sd(runs, label):
    """같은 설정 시드들의 층별 gini/dead/top10 평균·SD. 개별 런을 모두 보고한다."""
    got = []
    for r in runs:
        e, sub = rd(r)
        if sub is None: print(f'   MISS {r}'); continue
        got.append((r, e, sub))
    print(f'  {label}: n={len(got)}  ' + ', '.join(f'{r}(ep{e})' for r,e,_ in got))
    if not got: return None, 0
    LAY = [l for l in got[0][2].index if l not in EXCL]
    out = {}
    for l in LAY:
        v = np.array([s.loc[l,'gini'] for _,_,s in got])
        out[l] = dict(n=len(v), gini_m=float(v.mean()),
                      gini_sd=float(v.std(ddof=1)) if len(v)>1 else np.nan)
    return out, len(got)
