"""문헌 지표 정의에 맞춘 스파이크 집계 -- run_paper.py 의 vggdvslit 조합 전용 (26-10-26).

lib_snn/__init__ 이 import 하지 않는다. vggdvslit 조합의 main_sweep.py 만 콜백 목록에 LitSpikeLogger 를 붙인다.
변수를 읽기만 한다 (학습·규제·logs 에 영향 없음). logs 에 키를 넣지 않는 것은 CSVLogger·로그 파서의 열을
바꾸지 않기 위해서다.

왜 따로 세나: 기존 s_count 에는 n_in 의 아날로그 입력 합이 들어가고 T 전체를 합한다. 문헌(UniSparse)은
발화율과 SOP 를 보고한다. 둘을 같은 열에 둘 수 없어서 정의를 맞춘 열을 따로 만든다.

정의 (시험 = fit 의 검증 데이터에서, 표본 수로 나눈다)
  hid_spk  = 은닉층(loc=='HID', n_in·n_fc 제외) 스파이크 수 / 표본, T 전체 합 (spike_count 합).
  fr       = hid_spk / (T x 은닉 뉴런 수).  UniSparse 의 전체 발화율 (층별 비영 비율 (x!=0).mean() 을
             층 원소 수로 가중 평균, main.py evaluate) 과 같은 양이다. 스파이크가 이진(0/1)일 때 spike_count 가
             비영 원소 수와 같다 -- CPU 검증 (26-10-26) 에서 은닉 출력값은 {0,1} 이고 spike_count 합 = 발화 횟수 합 =
             출력 비영 수였다.
  sop      = sum over conv of sum conv2d(x, ones_like(kernel)) / 표본  (UniSparse utils.py SOPMonitor.cal_sop_conv).
             x = 그 conv 의 실제 입력 (앞 뉴런층 스파이크, 사이에 avg pool 이 있으면 적용).
             입력이 n_in(아날로그 프레임)인 첫 conv 는 뺀다 (UniSparse VGGSNN skip=['conv1'] 과 같다).
             fc 는 넣지 않는다 (SOPMonitor 는 Conv2d·matmul 만 센다). pool·conv 가 선형이라 T 합 spike_count 에
             한 번 걸면 스텝별 합과 같다. 출력 단위는 M (1e6), UniSparse 의 sop / 1000**2 와 같다.
  in_sum   = n_in 출력 합 / 표본 (참고: s_count 에 들어가는 아날로그 입력 합).
CPU 검증 (26-10-26): conv 입력을 직접 받아 conv2d(x, ones_like(kernel)) 로 잰 SOP·은닉 출력 합과 이 콜백 값이
  상대오차 0.0 으로 같았다.
기존 s_count 와의 관계: fit 로그의 s_count 는 학습 배치(train 모드, 증강 입력)에서 센 값이다 (proc.py
  postproc_batch_train -> spike_count_batch_end, 에폭 끝 train_ds_num 으로 나눔. 검증 배치는 mode=='inference' 일
  때만 센다). 이 콜백은 검증(=시험) 배치를 eval 모드로 센다. 둘은 단계·모드가 달라 같은 열에 두지 않는다.
UniSparse 와 다른 점: 그쪽은 배치별 합의 평균을 args.batch_size 로 나눈다 (마지막 배치가 작으면 편향).
  여기서는 실제로 센 표본 수로 나눈다. vggdvslit 는 1000/40 이라 남는 배치가 없다.
뉴런 정의(LIF 0.9·reset_by_sub 대 τ=2·hard reset)가 다르므로 같은 정의로 세어도 값은 레시피 차이를 포함한다.
보고 규칙 (26-10-26 리뷰): 이 값도 s_count 와 같은 계측 체계에서 나온다. 그래서 s_count 와 뉴런별 npz 총합의 11%
  불일치가 풀렸다는 근거가 되지 않는다. 지금은 같은 조합 안의 상대 감소율 (baseline 대비 ours_inter_ch) 만 보고한다.
  문헌의 발화율·SOP 와 나란히 놓는 것은 두 조건이 모두 충족된 뒤에만 한다: (a) s_count/npz 불일치가 풀림,
  (b) 이 값을 독립 계수 (npz) 와 1% 안으로 맞춤.

산출: train.log 의 [LITSPK] 줄, 런 디렉토리(conf.root_model_save)의 lit_spikes.csv (에폭마다 한 줄,
epoch 열은 reg_detail.csv 와 같은 0 시작).
"""
import os

import numpy as np
import tensorflow as tf


def _inbound(layer):
    nodes = layer._inbound_nodes
    if len(nodes) != 1:
        raise ValueError(f'[LITSPK] {layer.name}: inbound node {len(nodes)}개 (1개여야 함)')
    il = nodes[0].inbound_layers
    if isinstance(il, (list, tuple)):
        if len(il) != 1:
            raise ValueError(f'[LITSPK] {layer.name}: 입력 층 {len(il)}개 (1개여야 함)')
        il = il[0]
    return il


def neuron_layers(model):
    """뉴런을 가진 Activation 층 (model.set_layers_w_neuron 과 같은 기준). model.layers_w_neuron 은 model.init
    (콜백 시작) 때 채워지므로, 그 전에 불려도 되게 층을 직접 훑는다."""
    import lib_snn
    return [l for l in model.layers if isinstance(l, lib_snn.activations.Activation)
            and isinstance(l.act, lib_snn.neurons.Neuron)]


def _read(v):
    """뉴런 카운터 변수 -> [B,...] 텐서 (분산 변수면 복제본을 배치 축으로 잇는다)."""
    vals = getattr(v, 'values', None)
    if isinstance(vals, (tuple, list)):
        return tf.concat([tf.convert_to_tensor(x) for x in vals], axis=0)
    return tf.convert_to_tensor(v)


def sop_plan(model):
    """conv 마다 (conv 층, 입력 뉴런층, [pool 층...]) 를 Keras 그래프에서 찾는다. 모르는 층을 만나면 멈춘다."""
    import lib_snn
    plan, skipped = [], []
    for conv in model.layers:
        if not isinstance(conv, lib_snn.layers.Conv2D):
            continue
        if conv.data_format != 'channels_last' or getattr(conv, 'groups', 1) != 1:
            raise ValueError(f'[LITSPK] {conv.name}: channels_last·groups=1 만 지원')
        src, pools = _inbound(conv), []
        while not (isinstance(src, lib_snn.activations.Activation)
                   and isinstance(src.act, lib_snn.neurons.Neuron)):
            if isinstance(src, tf.keras.layers.AveragePooling2D):
                pools.append(src)
            elif isinstance(src, tf.keras.layers.Dropout):
                pass                                   # 평가 때 항등
            else:
                raise ValueError(f'[LITSPK] {conv.name} 입력 경로에 모르는 층 {src.name} ({type(src).__name__})')
            src = _inbound(src)
        if src.act.loc == 'IN':
            skipped.append(conv.name)                  # 아날로그 입력 conv (UniSparse skip)
            continue
        plan.append((conv, src, pools[::-1]))          # pool 은 뉴런 -> conv 순서
    return plan, skipped


def conv_sop(x, conv, pools):
    """x [B,H,W,C] (T 합 스파이크) -> 그 conv 의 SOP 합 (배치 전체). conv(x, ones) 합 = filters x conv(sum_c x, ones 1채널) 합."""
    for p in pools:
        x = tf.nn.avg_pool2d(x, p.pool_size, p.strides, p.padding.upper())
    xs = tf.reduce_sum(x, axis=-1, keepdims=True)
    k = tf.ones(list(conv.kernel_size) + [1, 1], xs.dtype)
    y = tf.nn.conv2d(xs, k, strides=conv.strides, padding=conv.padding.upper(), dilations=conv.dilation_rate)
    return tf.reduce_sum(y) * float(conv.filters)


class LitSpikeLogger(tf.keras.callbacks.Callback):
    """검증(시험) 배치마다 뉴런 카운터를 읽어 위 정의대로 집계하고, 검증이 끝나면 [LITSPK] 줄과 CSV 한 줄을 남긴다.
    카운터는 배치 시작에 0 으로 돌아가므로 (proc.py preproc_batch) 배치 끝 값이 그 배치의 T 합이다."""

    def __init__(self, model, config):
        super().__init__()
        conf = config.flags
        self.T = int(conf.time_step)
        self.csv = os.path.join(conf.root_model_save, 'lit_spikes.csv')
        neu = neuron_layers(model)
        self.hid = [l for l in neu if l.act.loc == 'HID']
        self.inp = [l for l in neu if l.act.loc == 'IN']
        if not self.hid:
            raise ValueError('[LITSPK] 은닉 뉴런층이 없다')
        self.n_hid_neurons = sum(int(np.prod(list(l.act.dim)[1:])) for l in self.hid)
        self.plan, self.skipped = sop_plan(model)
        self.epoch = -1
        self._reset()
        print(f'[LITSPK] 은닉층 {len(self.hid)}개 ({self.n_hid_neurons} 뉴런), SOP conv '
              f'{[c.name for c, _, _ in self.plan]} (뺀 conv {self.skipped}), T={self.T}, csv={self.csv}', flush=True)

    def _reset(self):
        self.n = 0
        self.hid_spk = 0.0
        self.hid_int = 0.0
        self.in_sum = 0.0
        self.sop = 0.0
        self.layer_spk = {l.name: 0.0 for l in self.hid}

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_test_begin(self, logs=None):
        self._reset()

    def on_test_batch_end(self, batch, logs=None):
        b = None
        for l in self.hid:
            sc = _read(l.act.spike_count)
            b = int(sc.shape[0])
            s = float(tf.reduce_sum(sc))
            self.layer_spk[l.name] += s
            self.hid_spk += s
            self.hid_int += float(tf.reduce_sum(_read(l.act.spike_count_int)))
        for l in self.inp:
            self.in_sum += float(tf.reduce_sum(_read(l.act.spike_count)))
        for conv, src, pools in self.plan:
            self.sop += float(conv_sop(_read(src.act.spike_count), conv, pools))
        self.n += b or 0

    def result(self):
        n = max(self.n, 1)
        hid = self.hid_spk / n
        return {'n': self.n, 'hid_spk': hid, 'hid_spk_int': self.hid_int / n,
                'fr': hid / (self.T * self.n_hid_neurons), 'sop_m': self.sop / n / 1e6, 'in_sum': self.in_sum / n,
                **{f'spk:{k}': v / n for k, v in self.layer_spk.items()}}

    def on_test_end(self, logs=None):
        if self.n == 0:
            return
        r = self.result()
        print(f"[LITSPK] epoch {self.epoch + 1} (0 시작 {self.epoch}) n={r['n']} hid_spk/표본 {r['hid_spk']:.1f} "
              f"(발화 횟수 {r['hid_spk_int']:.1f}) fr {r['fr']:.5f} SOP {r['sop_m']:.3f}M/표본 "
              f"n_in 합 {r['in_sum']:.1f}", flush=True)
        try:
            new = not os.path.exists(self.csv)
            keys = list(r)
            with open(self.csv, 'a') as f:
                if new:
                    f.write('epoch,' + ','.join(keys) + '\n')
                f.write(f'{self.epoch},' + ','.join(f'{r[k]:.6g}' for k in keys) + '\n')
        except OSError as e:        # 기록 실패가 학습을 죽이면 안 된다
            print(f'[LITSPK] csv 기록 실패: {e}', flush=True)
