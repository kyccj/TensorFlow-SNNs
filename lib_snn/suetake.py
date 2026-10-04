"""Suetake et al. (arXiv 2302.01500) synaptic interaction penalty 대조군 보조 (26-10-04).

    Omega_syn = sum_l psi_l * sum_i S_{l,i}^p / B ,   loss = L_task + lambda_t * Omega_syn

이 파일은 규제 항 자체를 계산하지 않는다 (그건 neurons.py `Neuron._suetake_reg`).
여기서 하는 일은 둘:
  (1) psi_l -- 층 l 뉴런 하나가 내보내는 시냅스 수(fan-out) -- 를 Keras 그래프에서 한 번 계산해
      각 은닉 Neuron 의 `suetake_psi` 변수에 넣는다.
  (2) lambda 램프 계수 `ramp` (= lambda_t / lambda) 를 에폭 시작마다 갱신한다.
둘 다 proc.preproc_epoch_train_snn 에서 reg_spike_suetake 일 때만 불린다. 플래그가 꺼져 있으면
이 모듈은 import 조차 되지 않는다.

fan-out 정의 (층 단위 상수, 경계 패딩 무시):
    conv 소비자  : kh * kw * C_out / (stride_h * stride_w)
                   stride 2 이면 입력 뉴런 하나가 닿는 출력 위치가 평균 kh*kw/4 개라서 나눈다.
    dense 소비자 : 출력 유닛 수 (GAP 를 거쳐도 뉴런 하나가 출력 유닛 전부에 닿는다)
    여러 소비자(residual 의 conv1 + conv0 등)에 들어가면 합한다.
    가중치 없는 identity 지름길(Add 로 바로 가는 경로)은 conv/dense 가 아니라 세지 않는다.
소비자 찾기는 analysis/prune_measure.py 의 back_src 와 같다: 가중치 층 입력을 거슬러 올라가
처음 만나는 스파이킹 층(.act 가 Neuron)이 그 소비자의 원천이다.
"""
import tensorflow as tf

from config import config
conf = config.flags

# lambda_t / lambda. 램프를 끄면 1 그대로다. 그래프 안에서 읽히므로 Variable 이어야 한다
# (tf.function 이 처음 추적될 때의 값으로 굳지 않게).
ramp = tf.Variable(1.0, trainable=False, dtype=tf.float32, name='suetake_ramp')

_psi = None   # {스파이킹 층 이름: psi} -- 한 번만 계산


def _inbound(l):
    ls = []
    for nd in l.inbound_nodes:
        il = nd.inbound_layers
        ls += il if isinstance(il, list) else [il]
    return ls


def _is_spiking(l):
    import lib_snn
    return isinstance(getattr(l, 'act', None), lib_snn.neurons.Neuron)


def _back_src(l):
    """가중치 층 입력을 거슬러 올라가 첫 스파이킹 층. 단일 입력·가중치 없는 층만 통과한다."""
    ps = _inbound(l)
    assert len(ps) == 1, ('suetake: 입력이 하나가 아닌 가중치 층', l.name, [p.name for p in ps])
    p = ps[0]
    while not _is_spiking(p):
        assert not [w for w in p.weights if w.name.split('/')[-1].startswith('kernel')], \
            ('suetake: 스파이킹 층 전에 다른 가중치 층', l.name, p.name)
        q = _inbound(p)
        assert len(q) == 1, ('suetake: 다중 입력 층을 거슬러 올라감', l.name, p.name)
        p = q[0]
    return p.name


def fanout(model):
    """{스파이킹 층 이름: psi}, 표 행 목록."""
    psi, rows = {}, []
    for l in model.layers:
        k = [w for w in l.weights if w.name.split('/')[-1].startswith('kernel')]
        if not k:
            continue
        ks = tuple(int(x) for x in k[0].shape)
        src = _back_src(l)
        if len(ks) == 4:
            kh, kw, _, cout = ks
            sh, sw = (int(x) for x in l.strides)
            f = kh * kw * cout / (sh * sw)
            desc = f'conv {kh}x{kw}x{cout} s{sh}'
        elif len(ks) == 2:
            cout = ks[1]
            f = float(cout)
            desc = f'dense {cout}'
        else:
            raise ValueError(f'suetake: 모르는 kernel 모양 {l.name} {ks}')
        psi[src] = psi.get(src, 0.0) + f
        rows.append((src, l.name, desc, f))
    return psi, rows


def setup(model):
    """psi 를 계산해 은닉 Neuron 의 suetake_psi 에 넣고 표를 한 번 찍는다."""
    global _psi
    if _psi is not None:
        return
    psi, rows = fanout(model)
    print('[suetake] fan-out psi_l (Suetake et al. 2302.01500, 26-10-04)', flush=True)
    for src, dst, desc, f in rows:
        print(f'[suetake]   {src:24s} -> {dst:24s} {desc:22s} {f:9.1f}', flush=True)
    n_hid = 0
    for l in model.layers:
        if not _is_spiking(l):
            continue
        v = psi.get(l.name, 0.0)
        used = hasattr(l.act, 'suetake_psi')
        print(f'[suetake] psi {l.name:24s} loc={l.act.loc:3s} {v:9.1f}'
              f'{"" if used else "   (규제 범위 밖)"}', flush=True)
        if used:
            assert v > 0, f'suetake: 은닉층 {l.name} 의 conv/dense 소비자를 못 찾음'
            l.act.suetake_psi.assign(v)
            n_hid += 1
    assert n_hid > 0, 'suetake: suetake_psi 를 가진 은닉 Neuron 이 없다'
    _psi = psi


def epoch_begin(model, epoch):
    """에폭 시작: (처음 한 번) psi 설정, 램프 계수 갱신. epoch 은 Keras 0-기준."""
    setup(model)
    r = (epoch + 1) / float(conf.train_epoch) if conf.reg_spike_suetake_ramp else 1.0
    r = min(max(r, 0.0), 1.0)
    ramp.assign(r)
    print(f'[suetake] epoch {epoch + 1}: ramp={r:.6g} lambda_t={conf.reg_spike_out_const * r:.6g}',
          flush=True)
