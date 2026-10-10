"""침묵 채널 실제 제거 측정 (26-09-26).

"같은 스파이크에서 제안법이 채널을 더 죽인다"를 배포 이득(정확도·params·MACs)으로 바꾼다.

순환 논리 차단:
  * 죽은 채널 판정 = **학습셋** 앞 N장 (기본 10,000장. N>10,000 이면 10,000장 시점 판정도 같이 저장)
  * 정확도 = **테스트셋** 10,000장
  학습셋은 datasets.load() 의 train_ds 를 쓰지 않는다. inference 모드에서도 RandAugment·
  RandomErase 가 걸리고 preprocessor 가 두 번 적용된다(image_cls.py). 대신 테스트와 똑같은
  변환(resize_with_crop → preprocess_input)을 train split 에 걸고, 그 변환이 로더의 test_ds 와
  비트 단위로 같은지 먼저 대조한다.

제거 = 그 채널 뉴런의 출력을 항상 0 으로 강제. Activation 층의 `mask_inh`(출력에 곱하는 기존
훅, activations.py) 에 채널 마스크를 넣는다. 모델은 다시 짜지 않는다.

시나리오:
  (a) residual 합류점(`*_out_n`) 제외 — conv1_conv_n(stem), `*_conv1_n`, fc1_n
  (b) 모든 스파이킹 층 (a + `*_out_n`)
  n_in(입력 부호화), predictions_n(출력) 은 제거 대상이 아니다.

params/MACs: 모델의 conv/dense 층마다 입력 쪽 스파이킹 층(src)과 출력 쪽 스파이킹 층(dst)을
Keras 그래프에서 역·순방향으로 따라가 찾는다. 가중치 [.., cin, cout] 는 src 채널 cin 이 죽었거나
dst 채널 cout 이 죽었으면 제거된다 (둘 다여도 한 번만 센다). MAC = 제거 가중치 × 출력 H×W
(dense 는 ×1). bias 는 dst 채널이 죽으면 제거. BN 은 추론 시 conv 에 접히므로 세지 않는다.
MACs 는 타임스텝 1회분(밀집 연산 기준). T 를 곱해도 비율은 같다.
`*_out_n` 채널 c 가 죽으면 그 블록의 conv2·conv0 필터 c 가 빠진다 (Add 채널 c 는 out_n c 로만
가므로). 다음 블록의 identity 경로는 가중치가 없어 0 을 더할 뿐이다.

쓰는 법:
    python analysis/prune_measure.py run <cfg_dir> <weights.h5> <tag> <out.json> <gpu> [n_train]
        gpu 는 '6' 또는 '' (CPU) 만 허용
    python analysis/prune_measure.py summary <json_dir>
"""
import sys, os, json, time

# ------------------------------------------------------------------------------------------
def summary(jdir):
    import numpy as np
    from scipy import stats
    ARMS = [('ours_layer 4e-3', 'r19c10-ours_layer-4e-3-s'),
            ('BPSR 6e-10', 'r19c10-bpsr-6e-10-s'),
            ('plain L2 5e-7', 'r19c10-l2-5e-7-s'),
            ('ours_intra_ch 4e-3', 'r19c10-prop-4e-3-s'),
            ('ours_layer_w1 2.5e-3', 'r19c10-ours_layer_w1-2p5e-3-s')]
    R = {}
    for arm, pre in ARMS:
        R[arm] = []
        for s in range(1, 6):
            p = os.path.join(jdir, f'{pre}{s}.json')
            if os.path.exists(p):
                d = json.load(open(p)); d['seed'] = f's{s}'; R[arm].append(d)
    for crit in sorted({c for a in R.values() for d in a for c in d['res']}):
        print(f'\n==================== 판정: {crit} ====================')
        for sc in ('a', 'b'):
            print(f'\n--- 시나리오 ({sc}) ---')
            hdr = (f"{'arm':22s} {'seed':>4s} {'srv':>4s} {'dead':>5s} {'test발화':>7s} {'잃는spk/샘플':>11s} "
                   f"{'acc전':>6s} {'acc후':>6s} {'Δacc':>6s} {'params%':>8s} {'MACs%':>7s}")
            print(hdr)
            for arm, runs in R.items():
                cols = {k: [] for k in ('dead', 'fired', 'lost', 'a0', 'a1', 'da', 'p', 'm')}
                for d in runs:
                    r = d['res'][crit][sc]
                    v = dict(dead=r['n_dead'], fired=r['n_dead_fired_test'], lost=r['test_spk_lost_per_sample'],
                             a0=d['acc_base'], a1=r['acc'], da=r['acc'] - d['acc_base'],
                             p=r['params_pct'], m=r['macs_pct'])
                    for k in cols: cols[k].append(v[k])
                    print(f"{arm:22s} {d['seed']:>4s} {d['srv']:>4s} {v['dead']:5d} {v['fired']:7d} {v['lost']:11.1f} "
                          f"{v['a0']:6.2f} {v['a1']:6.2f} {v['da']:+6.2f} {v['p']:8.2f} {v['m']:7.2f}")
                if runs:
                    n = len(runs)
                    f = lambda k, w, p: f"{np.mean(cols[k]):{w}.{p}f}±{np.std(cols[k], ddof=1) if n > 1 else 0:.{p}f}"
                    print(f"{'  평균 (n=%d)' % n:22s} {'':>4s} {'':>4s} {f('dead',5,1)} {f('fired',5,1)} {f('lost',7,1)} "
                          f"{f('a0',6,2)} {f('a1',6,2)} {f('da',6,2)} {f('p',6,2)} {f('m',6,2)}")
            # 학습셋 판정 집합 vs 테스트셋 판정 집합: 학습셋 판정 채널이 테스트에서 0번 발화하고
            # 개수까지 같으면 두 집합은 같다 (b 시나리오만 의미 있음 — 테스트 판정은 18층 전체).
            if sc == 'b':
                print('  학습셋 판정 == 테스트셋 판정 (개수, 테스트 발화 0):')
                for arm, runs in R.items():
                    print(f"    {arm:22s} " + ', '.join(
                        f"{d['seed']} {d['res'][crit]['b']['n_dead']}/{d['test_dead_ch']}" for d in runs))
            # 층별 죽은 채널 평균 / MACs 절감 기여 (%p)
            print(f'  층별 평균 죽은 채널 ({sc}):')
            for arm, runs in R.items():
                if not runs: continue
                ls = list(runs[0]['res'][crit][sc]['per_layer_dead'].keys())
                print(f"    {arm:22s} " + ' '.join(
                    f"{l.replace('_n', '').replace('conv', 'c').replace('_block', 'b')}:"
                    f"{np.mean([d['res'][crit][sc]['per_layer_dead'][l] for d in runs]):.0f}" for l in ls))
            print(f'  MACs 절감 기여 %p ({sc}), 가중치 층 → 상위 6:')
            for arm, runs in R.items():
                if not runs: continue
                tm = runs[0]['res'][crit][sc]['macs_total']
                wl = runs[0]['res'][crit][sc]['per_weight'].keys()
                c = {w: np.mean([d['res'][crit][sc]['per_weight'][w]['macs'] for d in runs]) / tm * 100 for w in wl}
                top = sorted(c.items(), key=lambda kv: -kv[1])[:6]
                print(f"    {arm:22s} " + ' '.join(f"{w}:{v:.1f}" for w, v in top))
            A = 'ours_layer 4e-3'
            for B in ('BPSR 6e-10', 'plain L2 5e-7', 'ours_intra_ch 4e-3'):
                for key, lab in (('macs_pct', 'MACs%'), ('params_pct', 'params%')):
                    x = [d['res'][crit][sc][key] for d in R[A]]
                    y = [d['res'][crit][sc][key] for d in R[B]]
                    if len(x) > 1 and len(y) > 1:
                        t, p = stats.ttest_ind(x, y, equal_var=False)
                        print(f"  Welch {lab:8s} {A} (n={len(x)}) {np.mean(x):.2f} vs {B} (n={len(y)}) {np.mean(y):.2f}"
                              f"  차 {np.mean(x) - np.mean(y):+.2f}%p  t={t:.2f} p={p:.4f}")


if __name__ == '__main__' and sys.argv[1] == 'summary':
    summary(sys.argv[2])
    sys.exit(0)

# ------------------------------------------------------------------------------------------
_, _cmd, cfg_dir, weights, tag, out_json, gpu = sys.argv[:7]
n_train = int(sys.argv[7]) if len(sys.argv) > 7 else 10000
assert _cmd == 'run'
# GPU 0~5 는 학습 중, 7 은 예약. 6 또는 CPU 만 쓴다.
assert gpu in ('6', ''), f'GPU {gpu!r} 금지 — 6 또는 "" 만'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
REPO = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.abspath(cfg_dir))

import numpy as np
import tensorflow as tf

# 장치 목록을 여기서 확정한다. config_sweep 이 import 시 CUDA_VISIBLE_DEVICES 를 다시 쓰지만
# TF 는 이미 초기화되어 있어 반영되지 않는다 (extract_neuron_spikes.py 와 같은 순서).
# CPU 로 돌릴 때 여러 건을 병렬로 띄우므로 건당 스레드를 제한한다 (PRUNE_THREADS, 기본 20).
_nth = int(os.environ.get('PRUNE_THREADS', '20'))
tf.config.threading.set_intra_op_parallelism_threads(_nth)
tf.config.threading.set_inter_op_parallelism_threads(2)
for _g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_g, True)
_gpus = tf.config.list_physical_devices('GPU')
print(f'[장치] CUDA_VISIBLE_DEVICES={gpu!r} GPU={_gpus} 스레드={_nth}', flush=True)
if gpu == '':
    assert not _gpus, 'CPU 모드인데 GPU 가 보인다'

from config_sweep import config
os.environ['CUDA_VISIBLE_DEVICES'] = gpu     # config_sweep 이 "6" 으로 덮어쓴 것을 되돌린다
conf = config.flags
conf.mode = 'inference'
# sejong 에서 가져온 체크포인트는 canus 런 디렉토리가 없다. config.set() 은 그 디렉토리를
# 뒤지므로, weights 가 놓인 트리(<root>/<tag>/<model>/<run>/ep-*.h5)의 <root>/<tag> 를 쓴다.
if not os.path.isdir(conf.root_model_save):
    conf.root_model_save = os.path.abspath(os.path.join(os.path.dirname(weights), '..', '..'))
conf.root_model_load = conf.root_model_save
config.set()
assert not config.train

import lib_snn, datasets                                             # noqa: E402
import tensorflow_datasets as tfds                                   # noqa: E402
from datasets.augmentation_cifar import resize_with_crop             # noqa: E402
from datasets.datasets import preprocessor_input_cifar10 as PP       # noqa: E402

lib_snn.utils.set_gpu()
train_ds_, valid_ds, test_ds, *_rest = datasets.datasets.load()
num_class, steps = _rest[-2], _rest[-1]
model = lib_snn.model_builder.model_builder(num_class, steps, valid_ds)
model.load_weights(weights)
print(f'[로드] {weights}', flush=True)
T = int(conf.time_step)
BS = int(config.batch_size_inference)


def clean_ds(split):
    """테스트와 같은 변환(증강 없음)."""
    ds = tfds.load('cifar10', split=split, shuffle_files=False, as_supervised=True)
    ds = ds.map(lambda i, l: resize_with_crop(i, l, 'cifar10', 32, 36 / 32, 10, 'torch', PP)) \
           .map(lambda i, l: (PP(i), l))
    return ds.batch(BS, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


# 변환 대조: 우리 test 파이프라인 == 로더 test_ds
_x0, _y0 = next(iter(test_ds))
_x1, _y1 = next(iter(clean_ds('test')))
assert _x0.shape == _x1.shape and np.array_equal(_x0.numpy(), _x1.numpy()) \
    and np.array_equal(_y0.numpy(), _y1.numpy()), '파이프라인 불일치'
print('[대조] 학습셋 변환 == 테스트 로더 변환 (첫 배치 일치)', flush=True)

# ---- 스파이킹 층 -----------------------------------------------------------------------
acts = {}
for l in model.layers:
    a = getattr(l, 'act', None)
    if isinstance(a, lib_snn.neurons.Neuron) and hasattr(a, 'spike_count'):
        acts[l.name] = l
PRUNABLE = [n for n in acts if n not in ('n_in', 'predictions_n')]
SC_A = [n for n in PRUNABLE if not n.endswith('_out_n')]
SC_B = list(PRUNABLE)
print(f'[층] 제거 후보 {len(PRUNABLE)}개, (a) {len(SC_A)}개: {SC_A}', flush=True)


# ---- 그래프: 가중치 층의 src / dst 스파이킹 층 -----------------------------------------
def _inb(l):
    ls = []
    for nd in l.inbound_nodes:
        il = nd.inbound_layers
        ls += il if isinstance(il, list) else [il]
    return ls


def _outb(l):
    return [nd.outbound_layer for nd in l.outbound_nodes]


def back_src(l):
    """가중치 층 입력을 거슬러 올라가 첫 스파이킹 층(채널 보존 경로만)."""
    ps = _inb(l)
    assert len(ps) == 1, (l.name, [p.name for p in ps])
    p = ps[0]
    while p.name not in acts:
        assert not p.weights or 'bn' in p.name or 'pool' in p.name, ('예상 밖 층', p.name)
        q = _inb(p)
        assert len(q) == 1, (p.name, [x.name for x in q])
        p = q[0]
    return p.name


def fwd_dst(l):
    """가중치 층 출력을 따라 내려가 첫 스파이킹 층. BN·Add 만 통과한다."""
    cs = _outb(l)
    assert len(cs) == 1, (l.name, [c.name for c in cs])
    c = cs[0]
    while c.name not in acts:
        assert 'bn' in c.name or c.name.endswith('_out'), ('예상 밖 층', c.name)
        nx = _outb(c)
        assert len(nx) == 1, (c.name, [x.name for x in nx])
        c = nx[0]
    return c.name


WL = []   # (name, kernel ndarray shape, has_bias, out_hw, src, dst)
for l in model.layers:
    k = [w for w in l.weights if w.name.split('/')[-1].startswith('kernel')]
    if not k:
        continue
    ks = tuple(k[0].shape)
    has_b = any(w.name.split('/')[-1].startswith('bias') for w in l.weights)
    osh = l.output_shape[0] if isinstance(l.output_shape, list) else l.output_shape
    hw = int(osh[1] * osh[2]) if len(ks) == 4 else 1
    WL.append(dict(name=l.name, ks=ks, bias=has_b, hw=hw, src=back_src(l), dst=fwd_dst(l)))
for w in WL:
    print(f"  {w['name']:24s} k{str(w['ks']):22s} HW={w['hw']:5d}  {w['src']:22s} -> {w['dst']}", flush=True)
# 채널 수 대조
for w in WL:
    cin, cout = w['ks'][-2], w['ks'][-1]
    assert acts[w['src']].output_shape[-1] == cin, (w['name'], 'cin')
    assert acts[w['dst']].output_shape[-1] == cout, (w['name'], 'cout')


def cost(dead):
    """dead: {layer: bool[C]} → (제거 params, 제거 MACs, 전체 params, 전체 MACs, 층별)"""
    rp = rm = tp = tm = 0
    per = {}
    for w in WL:
        kk = int(np.prod(w['ks'][:-2]))
        cin, cout = w['ks'][-2], w['ks'][-1]
        di = dead.get(w['src'], np.zeros(cin, bool))
        do = dead.get(w['dst'], np.zeros(cout, bool))
        m = di[:, None] | do[None, :]
        rw = kk * int(m.sum())
        rb = int(do.sum()) if w['bias'] else 0
        p_all = kk * cin * cout + (cout if w['bias'] else 0)
        rp += rw + rb; tp += p_all
        rm += rw * w['hw']; tm += kk * cin * cout * w['hw']
        per[w['name']] = dict(params=rw + rb, macs=rw * w['hw'])
    return rp, rm, tp, tm, per


# ---- 추론 루프 --------------------------------------------------------------------------
def set_masks(dead):
    for n, l in acts.items():
        if n in dead and dead[n].any():
            l.mask_inh = tf.constant((~dead[n]).astype(np.float32))
        else:
            l.mask_inh = None


def run_pass(ds, collect=False, snap_at=()):
    ch = {}
    snaps = {}
    n_seen = n_ok = 0
    for x, y in ds:
        for l in acts.values():
            l.act.reset_spike_count()
        out = model(x, training=False)
        p = tf.argmax(out, axis=-1).numpy()
        t = tf.argmax(y, axis=-1).numpy()
        n_ok += int((p == t).sum()); n_seen += int(x.shape[0])
        if collect:
            for n in PRUNABLE:
                sc = acts[n].act.spike_count.numpy()
                s = sc.reshape(-1, sc.shape[-1]).sum(axis=0).astype(np.float64)   # 배치·공간 합 → 채널
                ch[n] = ch.get(n, 0) + s
            if n_seen in snap_at:
                snaps[n_seen] = {k: v.copy() for k, v in ch.items()}
    return 100.0 * n_ok / n_seen, n_seen, ch, snaps


t0 = time.time()
set_masks({})
acc_base, n_test, ch_test, _ = run_pass(test_ds, collect=True)
print(f'[기준] 테스트 {n_test}장 정확도 {acc_base:.2f}%  ({time.time() - t0:.0f}s)', flush=True)

# 마스크 훅이 실제로 먹는지 확인 — stem 전 채널을 끄면 정확도가 무너져야 한다
set_masks({'conv1_conv_n': np.ones(acts['conv1_conv_n'].output_shape[-1], bool)})
acc_kill, _, _, _ = run_pass(test_ds.take(5))
set_masks({})
assert acc_kill < 30, f'마스크 훅이 안 먹음 (acc {acc_kill})'
print(f'[훅 확인] stem 전채널 제거 시 {acc_kill:.1f}% → 훅 작동', flush=True)

t0 = time.time()
snap_pts = tuple(s for s in (10000,) if s < n_train)
_, n_tr, ch_train, snaps = run_pass(clean_ds(f'train[:{n_train}]'), collect=True, snap_at=snap_pts)
print(f'[학습셋] {n_tr}장 수집 ({time.time() - t0:.0f}s)', flush=True)
crits = {f'train{n_tr // 1000}k': ch_train}
for k, v in snaps.items():
    crits[f'train{k // 1000}k'] = v

tot_test_spk = sum(float(v.sum()) for v in ch_test.values())
res = {}
for cname, chs in crits.items():
    res[cname] = {}
    dead_all = {n: chs[n] == 0 for n in PRUNABLE}
    for sc, lays in (('a', SC_A), ('b', SC_B)):
        dead = {n: dead_all[n] for n in lays}
        nd = int(sum(v.sum() for v in dead.values()))
        nf = int(sum(((ch_test[n] > 0) & dead[n]).sum() for n in lays))
        lost = float(sum(ch_test[n][dead[n]].sum() for n in lays))
        set_masks(dead)
        acc, _, _, _ = run_pass(test_ds)
        set_masks({})
        rp, rm, tp, tm, per = cost(dead)
        res[cname][sc] = dict(
            n_dead=nd, n_dead_fired_test=nf, test_spk_lost=lost,
            test_spk_lost_per_sample=lost / n_test, test_spk_lost_frac=lost / tot_test_spk,
            acc=acc, params_removed=rp, params_total=tp, params_pct=100.0 * rp / tp,
            macs_removed=rm, macs_total=tm, macs_pct=100.0 * rm / tm,
            per_layer_dead={n: int(dead[n].sum()) for n in lays},
            per_layer_dead_fired_test={n: int(((ch_test[n] > 0) & dead[n]).sum()) for n in lays},
            per_weight=per)
        print(f'[{cname} ({sc})] 죽은채널 {nd}, 그중 테스트 발화 {nf}, 잃는 스파이크 {lost / n_test:.1f}/샘플, '
              f'정확도 {acc_base:.2f}→{acc:.2f}, params -{100 * rp / tp:.2f}%, MACs -{100 * rm / tm:.2f}%', flush=True)

# 테스트셋 판정 죽은 채널 수 (npz 정의와 대조용)
test_dead = int(sum((ch_test[n] == 0).sum() for n in PRUNABLE))
srv = '?'
for h in (os.path.join(conf.root_model_save, 'host.txt'),
          os.path.join('/media/hdd1/kyccj/EIP/paper', tag, 'host.txt')):
    if os.path.exists(h):
        t_ = open(h).read()
        srv = '23' if ('161.122.23.23' in t_ or 'sejong' in t_) else ('138' if ('138' in t_ or 'canus' in t_) else '?')
        break
out = dict(tag=tag, weights=weights, srv=srv, T=T, n_test=n_test, n_train=n_tr, acc_base=acc_base,
           test_dead_ch=test_dead, test_spk_per_sample=tot_test_spk / n_test,
           layers_a=SC_A, layers_b=SC_B, weight_layers=[{k: (list(v) if k == 'ks' else v) for k, v in w.items()} for w in WL],
           res=res)
os.makedirs(os.path.dirname(out_json) or '.', exist_ok=True)
json.dump(out, open(out_json, 'w'), indent=1)
print(f'[저장] {out_json}  (테스트셋 판정 죽은 채널 {test_dead})', flush=True)
