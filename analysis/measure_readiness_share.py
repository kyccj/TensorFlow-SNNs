"""채널 합에서 vmem 준비도가 차지하는 비중을 잰다 (26-09-22).

왜 재나 -- `neurons.py` 의 두 분기가 readiness 를 다르게 다룬다:

    scp = tf.where(scp > 0, scp, g * readiness)     # 침묵 뉴런에 [0, gain) 부여

    'within_channel' :  sc_max = reduce_max(scp, axis=[1,2])   <- max 가 readiness 를 버린다
                        (침묵 뉴런의 readiness <= 1.0 이 발화 뉴런의 정수 카운트 >= 1 을
                         거의 못 이기므로)
    'channel'        :  ch     = reduce_sum(scp, axis=[1,2])   <- 1,024개가 전부 더해진다

가설: `channel` 분기에서 채널 합의 상당 부분이 문턱 아래 전하이고, 모든 채널이 같은
뉴런 수를 가지므로 그것이 거의 상수 바닥으로 깔린다. 그러면 ch/ch_max 가 1 쪽으로
눌리고 sc_rate = 1 - ch/ch_max 가 0 에 가까워져 규제가 약해진다. 이것이
`ours_inter_ch` 가 같은 rho 에서 스파이크를 32~61% 덜 줄이는 원인 후보다.

재는 것 (층마다):
  1. 채널 합 중 readiness 기여 비중
  2. sc_norm = ch/ch_max 의 분포 -- readiness 를 넣었을 때와 뺐을 때
  3. 그 결과 sc_rate = 1 - sc_norm 의 평균/분산이 얼마나 눌리는가

학습 경로를 건드리지 않는다. 체크포인트를 올려 순전파만 한다.

쓰는 법:
    python analysis/measure_readiness_share.py <run_dir> <weights.h5> [batches] [gpu]
    gpu 를 생략하거나 'cpu' 로 주면 CPU 로 돈다 (배치 몇 개면 비율은 수렴한다).
"""
import sys, os

run_dir = sys.argv[1]
weights = sys.argv[2]
max_batches = int(sys.argv[3]) if len(sys.argv) > 3 else 5
gpu = sys.argv[4] if len(sys.argv) > 4 else 'cpu'

os.environ['CUDA_VISIBLE_DEVICES'] = '' if gpu == 'cpu' else gpu
# 저장소 루트를 먼저 넣고(config, lib_snn, datasets 가 거기 있다) 그 앞에 run_dir 을 둔다.
# run_dir 의 config_sweep 이 우선이어야 학습 때와 같은 뉴런 배치가 나온다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(run_dir))

import numpy as np
import tensorflow as tf

for _g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass

from config_sweep import config
conf = config.flags
conf.mode = 'inference'
_rms = conf.root_model_save
if not os.path.isdir(_rms):
    for _root in ('/media/hdd1/kyccj/EIP/paper', '/media/hdd1/kyccj/EIP/archive'):
        _c = os.path.join(_root, os.path.basename(_rms))
        if os.path.isdir(_c):
            conf.root_model_save = _c
            break
conf.root_model_load = conf.root_model_save
config.set()
assert not config.train, 'train 모드로 남아 있음'

import lib_snn, datasets                                      # noqa: E402
lib_snn.utils.set_gpu()

_, _, test_ds, _, _, _, num_class, steps = datasets.datasets.load()
model = lib_snn.model_builder.model_builder(num_class, steps, test_ds)
model.load_weights(weights)
print(f'[로드] {weights}', flush=True)
print(f'[설정] vmem_gain={conf.reg_spike_vmem_gain} silent_only={conf.reg_spike_vmem_silent_only} '
      f'group={conf.reg_spike_maxnorm_group} T={conf.time_step}', flush=True)

acts = [(l.name, l.act) for l in model.layers
        if isinstance(getattr(l, 'act', None), lib_snn.neurons.Neuron)
        and hasattr(l.act, 'spike_count')]
T = conf.time_step
GAIN = float(conf.reg_spike_vmem_gain)

acc = {}   # name -> dict of running sums


def ta_last(x, t):
    """TensorArray 면 마지막 스텝을 읽고, 이미 텐서면 그대로 쓴다."""
    if hasattr(x, 'read'):
        return x.read(t - 1)
    return x


for bi, (xb, yb) in enumerate(test_ds):
    if bi >= max_batches:
        break
    for _, a in acts:
        a.reset_spike_count()
    model(xb, training=False)

    for name, a in acts:
        sc = a.spike_count.numpy()                    # [B, ...] T 동안 누적 (정수)
        if sc.ndim != 4:                              # fc 층은 채널 개념이 없다
            continue
        try:
            vm = ta_last(a.vmem, T).numpy()
            vth = ta_last(a.vth, T)
            vth = vth.numpy() if hasattr(vth, 'numpy') else np.asarray(vth)
        except Exception as e:
            print(f'  [건너뜀] {name}: vmem/vth 를 못 읽음 ({type(e).__name__})', flush=True)
            continue

        readiness = np.clip(np.divide(vm, vth, out=np.zeros_like(vm),
                                      where=np.asarray(vth) != 0), 0.0, 1.0)
        silent = sc <= 0
        # 학습 때와 같은 scp
        scp = np.where(silent, GAIN * readiness, sc).astype(np.float64)

        # 채널 합 (공간축 1,2) -- 'channel' 분기가 쓰는 양
        ch_full = scp.sum(axis=(1, 2))                        # [B, C]
        ch_fire = np.where(silent, 0.0, sc).sum(axis=(1, 2))  # readiness 없는 버전
        ch_rdy = ch_full - ch_fire

        def norm_spread(ch):
            m = ch.max(axis=1, keepdims=True)
            sn = np.divide(ch, m, out=np.zeros_like(ch), where=m != 0)
            return sn

        sn_full, sn_fire = norm_spread(ch_full), norm_spread(ch_fire)
        d = acc.setdefault(name, dict(n=0, rdy=0.0, tot=0.0, silent=0.0,
                                      sr_full=0.0, sr_fire=0.0,
                                      sd_full=0.0, sd_fire=0.0, C=scp.shape[-1]))
        d['n'] += 1
        d['rdy'] += ch_rdy.sum(); d['tot'] += ch_full.sum()
        d['silent'] += silent.mean()
        # sc_rate = 1 - sc_norm (beta=1)
        d['sr_full'] += (1 - sn_full).mean(); d['sr_fire'] += (1 - sn_fire).mean()
        d['sd_full'] += (1 - sn_full).std(axis=1).mean()
        d['sd_fire'] += (1 - sn_fire).std(axis=1).mean()
    print(f'  batch {bi} 완료', flush=True)

print('\n층별 — 채널 합에서 readiness 가 차지하는 비중과 그 결과 sc_rate 가 눌리는 정도\n')
hdr = (f"{'layer':<24}{'C':>5}{'침묵%':>8}{'readiness 비중':>15}"
       f"{'mean sc_rate':>14}{'(readiness 뺀 것)':>18}{'sd sc_rate':>12}{'(뺀 것)':>10}")
print(hdr); print('-' * len(hdr))
for name, d in acc.items():
    n = d['n']
    print(f"{name:<24}{d['C']:>5}{100*d['silent']/n:>7.1f}%{100*d['rdy']/d['tot']:>14.1f}%"
          f"{d['sr_full']/n:>14.4f}{d['sr_fire']/n:>18.4f}"
          f"{d['sd_full']/n:>12.4f}{d['sd_fire']/n:>10.4f}")
