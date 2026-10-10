"""뉴런 단위 발화 분포 — 팔별 비교 (26-09-21).

입력: _npz/*.npz  (extract_neuron_spikes.py 산출물)
  '<layer>'        뉴런별 총 발화 횟수 (테스트셋 전체, T 누적)  <- 분포는 이걸 쓴다
  '<layer>__fired' 뉴런이 한 번이라도 발화한 '표본 수' (최대 n_samples)

집계 대상 층: 스파이킹 뉴런 18개 (n_in = 입력 인코딩, predictions_n = 출력 누적값 제외).
영구 침묵 = 테스트셋 10,000장 전체에서 총 발화 0.
"""
import numpy as np, os, json, sys

NPZ = '/home/kyccj/PycharmProjects/TensorFlow-SNNs/_npz'
ARMS = [
    ('baseline',            'r19c10-base---s11'),
    ('plain L2 5e-7',       'r19c10-l2-5e-7-s5'),
    ('ours 4e-3',           'r19c10-prop-4e-3-s4'),
    ('ours 5e-3',           'r19c10-prop-5e-3-s1'),
    ('- 1- inversion 4e-3', 'r19c10-abl_noinv-4e-3-s1'),
    ('- 1- inversion 4e-2', 'r19c10-abl_noinv-4e-2-s1'),
]
EXCL = ('n_in', 'predictions_n')


def layers(d):
    # 26-10-10: npz 에 __t/__first_t/logits 등이 추가됐다. 층 키 = __fired 짝이 있는 키만.
    ks = [k for k in d.files if not k.endswith('__fired') and k + '__fired' in d.files and k not in ('n_samples', 'T', 'acc')]
    return [k for k in ks if k not in EXCL]


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    s = x.sum()
    if s <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * s) - (n + 1.0) / n)


def share_top(x, frac):
    x = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    k = max(1, int(round(x.size * frac)))
    return float(x[:k].sum() / x.sum())


def frac_neurons_for(x, target):
    """스파이크의 target 비율을 담당하는 뉴런의 비율 (내림차순 누적)."""
    x = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    c = np.cumsum(x) / x.sum()
    k = int(np.searchsorted(c, target) + 1)
    return float(k / x.size)


def metrics(v):
    v = np.asarray(v, dtype=np.float64)
    return dict(
        n_neuron=int(v.size),
        silent=int((v == 0).sum()),
        silent_pct=100.0 * float((v == 0).mean()),
        active=int((v > 0).sum()),
        spikes_total=float(v.sum()),
        gini=gini(v),
        top1=share_top(v, 0.01),
        top10=share_top(v, 0.10),
        f50=frac_neurons_for(v, 0.50),
        f80=frac_neurons_for(v, 0.80),
        f90=frac_neurons_for(v, 0.90),
        f99=frac_neurons_for(v, 0.99),
    )


def load(tag):
    d = np.load(os.path.join(NPZ, tag + '.npz'))
    ls = layers(d)
    per = {k: d[k].reshape(-1).astype(np.float64) for k in ls}
    allv = np.concatenate([per[k] for k in ls])
    return d, ls, per, allv


def boot_gini(v, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = v.size
    out = np.empty(B)
    for i in range(B):
        out[i] = gini(v[rng.integers(0, n, n)])
    return out


if __name__ == '__main__':
    res = {}
    for name, tag in ARMS:
        d, ls, per, allv = load(tag)
        ns = int(d['n_samples']); T = int(d['T'])
        m = metrics(allv)
        m.update(name=name, tag=tag, n_samples=ns, T=T, acc=float(d['acc']),
                 n_layer=len(ls), spikes_per_sample=m['spikes_total'] / ns)
        m['per_layer'] = {k: metrics(per[k]) for k in ls}
        res[name] = m
    json.dump(res, open('/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad/dist.json', 'w'))

    print('=== 전역 (18층 풀링, n_neuron 동일) ===')
    hdr = f"{'arm':22s} {'acc':>6s} {'spikes/샘플':>12s} {'뉴런':>7s} {'침묵':>7s} {'침묵%':>7s} {'활성':>7s} {'gini':>7s} {'top1%':>7s} {'top10%':>7s} {'f50':>7s} {'f80':>7s} {'f90':>7s} {'f99':>7s}"
    print(hdr)
    for name, _ in ARMS:
        m = res[name]
        print(f"{name:22s} {m['acc']:6.2f} {m['spikes_per_sample']:12,.1f} {m['n_neuron']:7,d} {m['silent']:7,d} "
              f"{m['silent_pct']:7.2f} {m['active']:7,d} {m['gini']:7.4f} {100*m['top1']:7.2f} {100*m['top10']:7.2f} "
              f"{100*m['f50']:7.2f} {100*m['f80']:7.2f} {100*m['f90']:7.2f} {100*m['f99']:7.2f}")
