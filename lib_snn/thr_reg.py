"""임계값 경로 대조군 두 팔 (26-10-20, docs/paper/compare-candidates-261020.md 3.2 B).

두 방법 모두 스파이크 벌점의 기울기가 가중치로 가지 않고 '임계값 역할을 하는 파라미터'로만 간다.
손실(add_loss)에는 아무것도 안 더한다. 그래서 가중치를 포함한 모든 파라미터의 과제 기울기는 플래그를
꺼 둔 것과 같고, 벌점 몫은 아래 파라미터에만 더해진다. 원본을 그대로 돌린 것이 아니라 우리 레시피
(R19, AdamW, asym 서로게이트, reset_by_sub) 위의 재구현이다 -- 논문에는 † 를 붙이고 아래 차이를 적는다.

2a UniSparse 스파이크 단독 -- Shi, Bu, Yu, ICML 2026 (PMLR 306:111711), 원 코드 xyshi2000/UniSparse
   (--lmbda-weight 0 --lmbda-neuron lambda, 즉 가중치 프루닝 없이 스파이크 항만. 저자 그림의 'Ours (Neuron Only)')
     E_l   = g_s^(l) * sum_c mean_{t,b,h,w} s^(l)                       식 7 의 스파이크 항 (utils.calc_penalty)
     gamma <- gamma - lambda*lr*dE_l/dgamma,  beta <- beta - lambda*lr*dE_l/dbeta     식 18-19 (31-32)
   * dE_l/d(gamma,beta) 는 층 l 바로 앞 BN(들)의 gamma,beta 에만 간다 (원 코드 prev_bns, 정리 4.1 의
     'BN affine = 다음 뉴런의 학습 가능한 임계값'). BN 입력(conv 출력)과 지름길 입력(Identity 스파이크)은
     상수로 둔다 -- 원 코드는 pre_norm_input 을 detach 해 두었다가 국소 순전파를 다시 돈다. 여기서는 t=T 에
     그 층 LIF 를 T 스텝 다시 돌려(그림자 순전파) 중첩 GradientTape 로 미분한다. 적분·누설·발화(서로게이트)·
     리셋은 Neuron 메서드를 그대로 불러 주 경로와 같은 연산이고 스파이크 값도 비트 단위로 같다.
   * 갱신 방식 (reg_spike_unisparse_update):
       'direct'(기본)  optimizer.step 뒤에 옵티마이저 밖에서 lambda*lr*dE/d(gamma,beta) 를 직접 뺀다. 원 코드
                       main.py 423-429행 그대로이고, 저자들도 AdamW 를 쓴 Spikformer 설정(configs/imagenet_spikformer.yaml
                       optimizer: adamw)에서 이 방식을 그대로 썼다. 그래서 'UniSparse 갱신식을 AdamW 위에 이식'한 판이다.
                       단 SGD 와 달리 과제 몫 스텝이 Adam 으로 ~lr 에 묶이므로 평형점이 식 14 (grad L + lambda*grad E = 0)
                       와 다르다 -- lambda*|dE/dgamma| 가 1 을 넘는 채널은 과제 손실과 무관하게 벌점이 이긴다.
       'optimizer'     lambda*dE/d(gamma,beta) 를 gamma·beta 의 과제 기울기에 더해 AdamW 에 넣는다 (식 14 목적 충실).
     두 판의 차이는 30에폭 착지 시험 1회로 비교한다. 에폭마다 층별 lambda*|dE| / |dL| 비를 찍는다.
   * g_s = sum_{conv 소비자} kh*kw*C_out. 원 코드 calc_penalty 의 next_weight 와 같이 stride 로 나누지 않고
     Linear 소비자는 세지 않는다. g_s=0 인 은닉층은 벌점이 없다 (R19: fc1_n, conv4_block2_out_n -- 원 코드
     SEW-R19 의 마지막 블록 뉴런과 epilogue LIF 도 벌점이 없다). g_s>0 인데 바로 앞 BN 이 없으면 멈춘다
     (BN 이 conv 층 안에 있는 모델 -- 조용히 벌점이 빠지는 것을 막는다).
   * 원 코드의 BasicBlock(지름길이 뉴런 앞에서 더해짐) 국소 순전파는 Identity 지름길 입력을 빼먹는다.
     원 R19 는 SEW 블록이라 해당이 없다. 우리 R19 는 Add 가 뉴런 앞이라 실제 막전위(지름길 포함)에서
     서로게이트를 잰다 -- 식 17 의 Theta'(v[t]-vth) 그대로다. 단 식 17 은 리셋을 떼어 낸(ẑ) 기울기이고
     우리는 리셋을 통하는 BPTT 기울기다.
   * 원 코드는 optimizer.step() 이 gamma,beta 를 갱신한 '뒤' 값으로 국소 순전파를 한다. 여기서는 같은 배치
     순전파(갱신 전 값)에서 기울기를 만들고 step 뒤에 적용한다 -- 한 스텝(학습률 크기)의 차이.
   * 원 코드 lr 은 에폭 단위(timm 스케줄의 그 에폭 값), 여기는 스텝 단위 현재 학습률 (워밍업 20에폭 포함).
   * 원 코드(ResNet 설정)는 BN affine 에 weight decay 가 없다 (create_optimizer_v2 에 weight_decay 없음, 자체
     weight_decay() 는 SparseConv2d 가중치만). 우리 레시피는 AdamW(wd)가 gamma·beta 도 감쇠한다 (baseline 과 공유).

2b Activity Pruning AT-LIF -- Bu, Shi, Yu, NeurIPS 2025, 원 코드 putshua/Activity-Pruning-SNN
     o = theta * s,  s = H(m - theta)       식 10 (층당 학습 스칼라 theta, 원 코드 ATLIF.thresh, 초기값 1.0)
     과제 기울기:  do/dm = 서로게이트,  do/dtheta = -do/dm                    식 13
     갱신:  theta <- theta - lr*dL/dtheta + lambda*lr * (1/T) sum_t mean_{뉴런} sum_b s * do/dm      식 12
            (둘째 항은 원 코드 Surrogate.forward 의 thre_updates, utils.threshold_update)
   * theta 는 AdamW 에 넣지 않는다. 식 12 는 같은 학습률 eta 의 경사 한 스텝이므로 옵티마이저 밖에서 위 식
     그대로(SGD 형) 갱신한다 (reg_spike_atlif_theta_task='sgd', 기본). AdamW 에 넣으면 과제 몫 스텝이 기울기
     크기와 무관하게 ~lr 로 정규화되어 규제 몫(lambda*lr*통계)과 같거나 커지고, theta 가 과제에 끌려 하한까지
     내려갈 수 있다 (10-20 리뷰). 'none' 은 과제 몫을 버리고 규제로만 움직인다 -- 원 코드의 grad_thre 는
     -(g*tmp*tmp).mean() (합 대신 원소 평균) 이라 과제 몫이 규제 몫보다 수 자릿수 작아 사실상 이쪽이다.
     어느 쪽이든 에폭마다 층별 과제 몫·규제 몫 delta-theta 를 따로 찍는다.
   * theta 하한 ATLIF_THETA_MIN 은 갱신 뒤 tf.maximum 으로 직접 건다 (Keras constraint 에 기대지 않는다 --
     옵티마이저 밖 변수라 적용되지 않는다). 2*하한 아래로 내려가면 에폭 시작에 런을 멈춘다.
   * 구현: 정규화 좌표 v' = m/theta 에서 기존 LIF 를 vth=1 로 그대로 돌린다. reset_by_sub 이므로 v' - s 는
     원 좌표의 m - theta*s 이고, 발화 조건 v' >= 1 은 m >= theta 와 같다. do/dm = theta*ds/dm = ds/dv' 는
     우리 서로게이트 그 자체다 (원 코드 삼각형 max(1-|m/theta-1|,0) 자리).
   * do/dtheta = -do/dm 은 발화 함수 입력에 -(theta-sg(theta))/sg(theta) 를 더해 만든다 (값은 정확히 0). 출력 배율
     theta 의 직접항(do/dtheta 의 +s)은 식 13 과 원 코드처럼 넣지 않는다. 리셋에는 이 항을 넣지 않고 s 만
     theta 기울기를 실어 나른다 (원 코드 mem*(1 - spike/thresh.detach()) 와 같은 구조).
   * 우리 레시피는 reset_by_sub (원 코드는 hard reset), 누설 위치(적분 뒤 0.9배)도 우리 것을 따른다.
   * 원 코드는 평가 순전파에서도 규제 통계를 쌓아 다음 에폭 첫 학습 스텝에 한꺼번에 더한다 (main.py evaluate 가
     update_value 를 0 으로 돌리지 않는다; C10 기준 에폭당 theta 상승이 약 1.2배). 여기서는 학습 스텝만 쓴다.
   * 근거로 쓸 수 있는 것은 AP 표1 의 lambda_s=0 행(CIFAR-10 R20, DVS)뿐이다. ImageNet-100 행과 공개 cifar10.yaml
     기본값(eta2=1e-4)은 가중치로 가는 L1 활동 규제를 함께 쓴다 -- 이 팔은 그 항이 없는 순수 AT-LIF 다.

공통: 서로게이트는 우리 것(fire_func, 레시피의 'asym')을 그대로 쓴다 -- 원 코드는 UniSparse ATan,
AP 삼각형. 같은 레시피 원칙 (compare-candidates 3.2 B).
reg_detail.csv 의 sc_loss 열은 건드리지 않는다 (baseline 과 같이 0). E_l 과 theta 규제 통계는 train.log 의
에폭 줄로만 찍는다 -- 다른 팔의 sc_loss(규제 손실)와 뜻이 달라 같은 열에 두면 잘못 읽힌다.

이 모듈은 플래그(reg_spike_unisparse / reg_spike_atlif)가 켜져 있을 때만 import 된다.
"""
import math

import tensorflow as tf

from absl import flags
conf = flags.FLAGS

# AT-LIF theta 의 수치 하한. 원 코드에는 없다. 0 이하가 되면 x/theta 가 뒤집혀 런이 죽는다.
# 정상 범위(~1 이상)에서는 닿지 않는다. 2*하한 아래면 에폭 시작에 멈추고, 0.1 아래면 경고한다.
ATLIF_THETA_MIN = 1e-2
ATLIF_THETA_ABORT = 2 * ATLIF_THETA_MIN
ATLIF_THETA_WARN = 0.1

_PLAN = {}        # id(Neuron) -> 계획 dict. setup() 이 한 번 채운다.
_STATE = {'model': None, 'steps': None, 'note_route': False}


############################################################
# 그래프 탐색 (lib_snn/suetake.py 와 같은 방식)
############################################################
def _inbound(l):
    ls = []
    for nd in l.inbound_nodes:
        il = nd.inbound_layers
        ls += il if isinstance(il, list) else [il]
    return ls


def _is_spiking(l):
    import lib_snn
    return isinstance(getattr(l, 'act', None), lib_snn.neurons.Neuron)


def _is_bn(l):
    import lib_snn
    return isinstance(l, lib_snn.layers.BatchNormalization)


def _kernel(l):
    return [w for w in l.weights if w.name.split('/')[-1].startswith('kernel')]


def _back_src(l):
    """가중치 층 입력을 거슬러 올라가 첫 스파이킹 층 이름. 단일 입력·가중치 없는 층만 통과한다."""
    ps = _inbound(l)
    assert len(ps) == 1, ('thr_reg: 입력이 하나가 아닌 가중치 층', l.name, [p.name for p in ps])
    p = ps[0]
    while not _is_spiking(p):
        assert not _kernel(p), ('thr_reg: 스파이킹 층 전에 다른 가중치 층', l.name, p.name)
        q = _inbound(p)
        assert len(q) == 1, ('thr_reg: 다중 입력 층을 거슬러 올라감', l.name, p.name)
        p = q[0]
    return p.name


def conv_fanout(model):
    """UniSparse g_s: {스파이킹 층 이름: sum_{conv 소비자} kh*kw*C_out}.
    원 코드 utils.calc_penalty 의 next_weight = sum(conv.out_channels*kernel_size[0]*kernel_size[1]).
    stride 로 나누지 않고, Linear(Dense) 소비자는 next_convs 에 없으므로 세지 않는다."""
    gs = {}
    for l in model.layers:
        k = _kernel(l)
        if not k or len(k[0].shape) != 4:
            continue
        kh, kw, _, cout = (int(x) for x in k[0].shape)
        src = _back_src(l)
        gs[src] = gs.get(src, 0.0) + float(kh * kw * cout)
    return gs


def _prev_bns(l):
    """뉴런(Activation 층) 바로 앞의 BN 층들. 원 코드 prev_bns 와 같다:
    BN -> 뉴런 이면 [BN], Add([지름길, BN]) -> 뉴런 이면 Add 입력 중 BN 인 것 (conv0_bn·conv2_bn).
    Identity 지름길(앞 블록 스파이크)은 BN 이 아니라 빠진다. 그 밖(Dense -> 뉴런 등)이면 []."""
    import lib_snn
    ps = _inbound(l)
    assert len(ps) == 1, ('thr_reg: 뉴런 입력이 하나가 아님', l.name)
    p = ps[0]
    if _is_bn(p):
        return [p]
    if isinstance(p, lib_snn.layers.Add):
        return [q for q in _inbound(p) if _is_bn(q)]
    return []


############################################################
# 설정
############################################################
def _lam():
    return float(conf.reg_spike_unisparse_lambda if conf.reg_spike_unisparse else conf.reg_spike_atlif_lambda)


def _check_recipe():
    """그림자 순전파(2a)와 정규화 좌표(2b)는 run_type_hid 의 RATE·LIF/IF·reset_by_sub 경로를 그대로
    다시 쓴다. 다른 설정이면 조용히 다른 동역학을 미분하게 되므로 멈춘다."""
    assert not (conf.reg_spike_unisparse and conf.reg_spike_atlif), \
        'reg_spike_unisparse 와 reg_spike_atlif 는 같이 켜지 않는다 (각각 독립 대조군)'
    assert not conf.reg_spike_out and not conf.reg_spike_suetake and not conf.reg_spike_loss_ratio, \
        '임계값 경로 대조군은 순수 팔이다 -- reg_spike_out / suetake / loss-ratio 와 같이 켜지 않는다'
    # lambda 기본값은 -1 (미지정). 플래그만 켜고 lambda 를 빠뜨리면 baseline 과 구별 안 되는 팔이 오류 없이
    # 돈다. 0 은 시험용(과제 몫만 재기)으로 명시할 때만 허용한다. 1e999(=inf) 같은 오타도 여기서 막는다.
    lam = _lam()
    assert math.isfinite(lam) and lam >= 0.0, \
        ('thr_reg: lambda 를 0 이상 유한값으로 지정해야 한다 (기본 -1 = 미지정)', lam)
    # atlif 의 출력 교체(neurons.py 맨 끝)가 _ch_taylor_tap 출력을 덮어쓴다
    assert conf.reg_spike_ch_importance == 'spike', conf.reg_spike_ch_importance
    # 그림자 순전파가 integration/fire_func 를 한 번 더 부른다 -- 부수효과(누적·기록·출력)가 두 번 생긴다
    assert not (conf.f_tot_psp or conf.f_positive_vmem or conf.f_neg_cap_vmem), \
        'thr_reg: f_tot_psp / f_positive_vmem / f_neg_cap_vmem 미지원'
    assert not (conf.debug_surro_grad or conf.verbose_snn_train), \
        'thr_reg: debug_surro_grad / verbose_snn_train 미지원'
    assert conf.nn_mode == 'SNN' and not conf.snn_training_spatial_first
    assert conf.neural_coding == 'RATE', conf.neural_coding
    assert conf.n_reset_type == 'reset_by_sub', conf.n_reset_type
    assert conf.binary_spike and not conf.integer_spike
    assert not (conf.leak_time_dep or conf.adaptive_vth or conf.adaptive_vth_SEL or conf.vth_rand_static)
    assert not (conf.noise_en or conf.SEL_noise_en_spike or conf.low_test or conf.bn_per_step)
    assert tf.keras.mixed_precision.global_policy().name == 'float32'


def _check_bn(bn, ndim):
    assert bn.built and bn.center and bn.scale, bn.name
    assert bn.trainable, ('thr_reg: 얼린 BN -- 그림자 순전파는 배치 통계로 재정규화한다', bn.name)
    assert list(bn.axis) == [ndim - 1], ('thr_reg: channels-last BN 만', bn.name, bn.axis)
    assert not bn.fused and not bn.renorm and bn.virtual_batch_size is None and bn.adjustment is None, \
        ('thr_reg: 일반 BN 만 지원', bn.name)
    assert not getattr(bn, 'temporal_batch', None), ('thr_reg: 단계별 BN 미지원', bn.name)
    assert not getattr(bn, 'f_skip_bn', False), bn.name


def _scalar(name):
    return tf.Variable(0.0, trainable=False, name=name)


def setup(model):
    """처음 한 번: 어느 뉴런이 어떤 파라미터를 갱신할지 정한다. 첫 train_step 추적 전에 불려야 한다
    (proc.preproc_epoch_train_snn, 에폭 0 시작 -- fit 의 분산 전략 scope 안, 옵티마이저 build 전)."""
    if _STATE['model'] is not None:
        # 계획은 id(Neuron) 로 묶여 있다. 한 프로세스에서 모델을 다시 지으면 이전 모델 계획이 남는다
        assert _STATE['model'] is model, 'thr_reg: 한 프로세스에서 두 번째 모델 -- 지원하지 않는다'
        return
    _check_recipe()
    # 옵티마이저 밖 갱신(assign_sub/assign)은 복제본 집계를 거치지 않는다. 런은 GPU 1장(OneDeviceStrategy)
    assert model.distribute_strategy.num_replicas_in_sync == 1, 'thr_reg: 단일 복제본 전용'
    lam = _lam()
    if conf.reg_spike_unisparse:
        mode = conf.reg_spike_unisparse_update
        gs = conv_fanout(model)
        print('[unisparse] UniSparse spike-only (Shi, Bu, Yu ICML 2026), lambda={:g}, update={}'
              .format(lam, mode), flush=True)
        for l in model.layers:
            if not _is_spiking(l) or l.act.loc != 'HID':
                continue
            bns = _prev_bns(l)
            g = gs.get(l.name, 0.0)
            print('[unisparse]   {:24s} g_s={:9.1f}  prev_bn={}{}'.format(
                l.name, g, [b.name for b in bns], '' if g > 0 else '   (벌점 없음: conv 소비자 없음, 원 코드와 같음)'),
                flush=True)
            if g <= 0:
                continue
            # g_s>0 인데 앞 BN 이 없으면 (BN 이 conv 층 안에 있는 모델 등) 이 층은 조용히 빠진다 -- 멈춘다
            assert bns, ('unisparse: conv 소비자가 있는데 바로 앞 BN 층이 없다 (use_bn 내장 BN 미지원)', l.name)
            assert len(l.act.dim) == 4, ('thr_reg: conv 뉴런만 (원 코드 mean(dim=(0,1,3,4)))', l.name)
            pairs = []
            for bn in bns:
                _check_bn(bn, len(l.act.dim))
                assert int(bn.gamma.shape[-1]) == int(l.act.dim[-1]), (bn.name, l.name)
                bn._thr_stash = True        # lib_snn/layers.py: BN 입력 참조를 남긴다
                for var in (bn.gamma, bn.beta):
                    pairs.append((var, tf.Variable(tf.zeros_like(var), trainable=False,
                                                   name='us_acc_' + var.name.split(':')[0].replace('/', '_'))))
            _PLAN[id(l.act)] = {'kind': 'unisparse', 'name': l.name, 'bns': bns, 'gs': g, 'pairs': pairs,
                                'E': _scalar('us_E'), 'lg_pen': _scalar('us_pen'), 'lg_ratio': _scalar('us_ratio'),
                                'traced': False, 'routed': False}
        assert _PLAN, 'unisparse: 벌점을 받을 은닉층이 없다'
    if conf.reg_spike_atlif:
        mode = conf.reg_spike_atlif_theta_task
        print('[atlif] AT-LIF (Bu, Shi, Yu NeurIPS 2025), lambda_theta={:g}, theta 과제 갱신={}'
              .format(lam, mode), flush=True)
        for l in model.layers:
            if not _is_spiking(l) or l.act.loc != 'HID':
                continue
            n = l.act
            assert hasattr(n, 'atlif_theta'), ('atlif: Neuron.build 에서 theta 가 안 만들어짐', l.name)
            assert n.vth_init_const == 1.0, ('atlif: 정규화 좌표는 vth=1 을 가정', l.name, n.vth_init_const)
            _PLAN[id(n)] = {'kind': 'atlif', 'name': l.name, 'theta': n.atlif_theta, 'acc': n.atlif_reg,
                            'd_task': _scalar('at_dtask'), 'd_reg': _scalar('at_dreg'), 'g_last': _scalar('at_g'),
                            'traced': False, 'routed': False}
        assert _PLAN, 'atlif: 은닉층이 없다'
        print('[atlif] theta 는 AdamW 밖에서 갱신 (식 12, {}). weight decay·constraint 없음, 하한 {:g} 직접 적용'
              .format('theta -= lr*dL/dtheta; theta += lambda*lr*stat' if mode == 'sgd'
                      else 'theta += lambda*lr*stat, 과제 몫 버림', ATLIF_THETA_MIN), flush=True)
    _STATE['steps'] = _scalar('thr_steps')
    _STATE['model'] = model


def epoch_begin(model, epoch):
    """지난 에폭의 기록을 한 줄씩 찍고 0 으로 돌린다 (첫 에폭은 setup 만).
    AT-LIF theta 가 하한 근처면 런을 멈춘다 -- 입력이 1/theta 배로 커져 발화가 포화한 무의미한 점이 된다."""
    setup(model)
    steps = max(float(_STATE['steps'].numpy()), 1.0)
    if conf.reg_spike_unisparse:
        s = ' '.join('{}:{:.3f}/{:.3f}/{:.3g}'.format(p['name'], float(tf.reduce_mean(p['bns'][-1].gamma)),
                                                      float(tf.reduce_mean(p['bns'][-1].beta)), float(p['E']))
                     for p in _PLAN.values())
        print('[unisparse] epoch {} mean gamma/beta (마지막 앞 BN) / 마지막 스텝 E_l: {}'.format(epoch + 1, s), flush=True)
        if epoch > 0:
            s = ' '.join('{}:{:.2e}/{:.3g}'.format(p['name'], float(p['lg_pen']), float(p['lg_ratio']) / steps)
                         for p in _PLAN.values())
            print('[unisparse] epoch {} 지난 에폭 벌점 몫 sum|lambda*lr*dE| (원소 평균, {}) / '
                  '평균 lambda*|dE|÷|dL| (gamma·beta): {}'.format(
                      epoch + 1, '실제 이동' if conf.reg_spike_unisparse_update == 'direct' else 'SGD 환산', s),
                  flush=True)
        neg = [p['name'] for p in _PLAN.values() for bn in p['bns'] if float(tf.reduce_mean(bn.gamma)) < 0.0]
        if neg:
            print('[unisparse][경고] 평균 gamma<0 인 앞 BN: {} (beta 단조성이 뒤집힌다, UniSparse 부록 A)'.format(neg),
                  flush=True)
    if conf.reg_spike_atlif:
        s = ' '.join('{}:{:.4f}({:+.2e}/{:+.2e})'.format(p['name'], float(p['theta']), float(p['d_task']),
                                                         float(p['d_reg']))
                     for p in _PLAN.values())
        print('[atlif] epoch {} theta (지난 에폭 과제 몫/규제 몫): {}'.format(epoch + 1, s), flush=True)
        low = [(p['name'], float(p['theta'])) for p in _PLAN.values() if float(p['theta']) < ATLIF_THETA_WARN]
        if low:
            print('[atlif][경고] theta < {:g}: {}'.format(ATLIF_THETA_WARN, low), flush=True)
        bad = [(n, v) for n, v in low if v < ATLIF_THETA_ABORT]
        if bad:
            raise RuntimeError('[atlif] theta 가 하한 근처 ({:g} 미만): {} -- 발화가 포화한 무의미한 런이라 멈춘다'
                               .format(ATLIF_THETA_ABORT, bad))
    for p in _PLAN.values():
        for k in ('lg_pen', 'lg_ratio', 'd_task', 'd_reg'):
            if k in p:
                p[k].assign(0.0)
    _STATE['steps'].assign(0.0)


def _plan(n):
    assert _STATE['model'] is not None, 'thr_reg: setup() 전에 학습 스텝이 추적됨 (proc.preproc_epoch_train_snn 확인)'
    return _PLAN.get(id(n))


def _training(training):
    if training is None:
        return False
    if isinstance(training, bool):
        return training
    v = tf.get_static_value(training)
    assert v is not None, 'thr_reg: training 이 기호 텐서'
    return bool(v)


############################################################
# 2a UniSparse -- Neuron.call 에서 매 스텝 불린다
############################################################
def unisparse_step(n, inputs, t, training):
    if not _training(training):
        return
    p = _plan(n)
    if p is None:
        return
    T = conf.time_step
    # 스텝별 입력은 계획 dict 에 둔다 (Keras 층 속성으로 두면 리스트가 추적 대상으로 감싸진다)
    if t == 1:
        p['xs'] = []
    p['xs'].append(tf.stop_gradient(inputs))
    if t != T:
        return
    assert len(p['xs']) == T

    with tf.GradientTape() as tape:
        # BN 국소 재정규화: 값은 0, 기울기는 gamma(*tdbn 배율)·beta 로만. BN 은 시간평균 입력으로 한 번
        # 불리므로 (model.py temporal reduction) 모든 스텝에 같은 항이다.
        loc = 0.0
        for bn in p['bns']:
            u = tf.stop_gradient(tf.cast(bn._thr_in, bn._param_dtype))
            axes = [i for i in range(len(u.shape)) if i not in bn.axis]
            mean, var = bn._moments(u, axes, keep_dims=False)
            xhat = tf.stop_gradient((u - mean) * tf.math.rsqrt(var + bn.epsilon))
            scale = bn.gamma * bn.tdbn_scale if bn.en_tdbn else bn.gamma
            y = xhat * scale + bn.beta
            loc = loc + (y - tf.stop_gradient(y))
        # 그림자 순전파: run_type_hid 와 같은 연산 (적분 -> 누설 -> fire_func -> reset_by_sub)
        vmem = tf.zeros(shape=n.dim, dtype=n._dtype)
        ssum = 0.0
        for i in range(T):
            v = n.integration(p['xs'][i] + loc, vmem, i + 1)
            if n.n_type == 'LIF':
                v = n.leak(v, i + 1)
            s = n.fire_func(v)
            vmem = tf.subtract(v, s)
            ssum = ssum + tf.reduce_sum(s)
        b, h, w = (int(x) for x in n.dim[:3])
        energy = p['gs'] * ssum / float(T * b * h * w)     # = g_s * sum_c mean_{t,b,h,w} s
    grads = tape.gradient(energy, [var for var, _ in p['pairs']])
    for (var, acc), g in zip(p['pairs'], grads):
        acc.assign(tf.zeros_like(var) if g is None else g)
    p['E'].assign(tf.stop_gradient(energy))     # train.log 에폭 줄 (lambda 전 E_l)
    p['xs'] = []
    p['traced'] = True


############################################################
# 2b AT-LIF -- Neuron.call 의 은닉층 순전파를 대신한다
############################################################
def atlif_step(n, inputs, vmem_prev, t, training):
    """반환: (이진 스파이크 s, 정규화 막전위, 다음 층 출력 theta*s)."""
    theta = n.atlif_theta
    # 하한은 갱신 뒤에 직접 걸지만 여기서도 나눗셈을 지킨다 (값이 하한 이상이면 sg(theta) 그대로)
    th = tf.maximum(tf.stop_gradient(theta), ATLIF_THETA_MIN)
    v = n.integration(inputs / th, vmem_prev, t)
    if n.n_type == 'LIF':
        v = n.leak(v, t)
    # 값은 v 그대로 (theta - sg(theta) = 0). d(입력)/dtheta = -1/theta 라 do/dtheta = th*Theta'*(-1/th) = -do/dm (식 13)
    v_in = v - (theta - tf.stop_gradient(theta)) / th
    train = _training(training)
    if train:
        with tf.GradientTape() as tape:
            tape.watch(v_in)
            s = n.fire_func(v_in)
        dsdv = tape.gradient(s, v_in)       # 우리 서로게이트 Theta'(v') = do/dm
    else:
        s = n.fire_func(v_in)
    vmem = tf.subtract(v, s)                # reset_by_sub (정규화 좌표). theta 기울기는 s 로만 지나간다
    n.f_fire = tf.math.greater(s, tf.zeros_like(s))    # count_spike_default 가 읽는다 (fire() 와 같은 역할)
    out = th * s                            # 식 10. do/ds = theta, do/dtheta 는 위 v_in 경로만

    if train:
        p = _plan(n)
        assert p is not None, ('atlif: 계획에 없는 은닉층 (setup 이 다른 모델로 불림?)', n.name)
        r = tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(s * dsdv), axis=0))   # sum_b 뒤 뉴런 평균
        p['r'] = r if t == 1 else p['r'] + r
        if t == conf.time_step:
            p['acc'].assign(p['r'] / float(conf.time_step))     # 식 12 둘째 항의 통계 (lambda 전)
            p['traced'] = True
    return s, vmem, out


############################################################
# 옵티마이저 앞뒤 -- model.py train_step 에서 apply_gradients 바로 앞(route_grads)과 바로 뒤(apply_after_step)
############################################################
def _current_lr(optimizer):
    """방금 적용된 스텝의 학습률. 스케줄 객체가 있으면 iterations-1 에서 직접 계산한다
    (Keras 2.11/2.12 의 learning_rate 프로퍼티가 _current_learning_rate 를 돌려주는지에 기대지 않는다)."""
    lr = getattr(optimizer, '_learning_rate', None)
    if lr is None:
        lr = optimizer.learning_rate
    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        lr = lr(optimizer.iterations - 1)
    return tf.cast(lr, tf.float32)


def _mean_abs(ts):
    """여러 텐서·변수를 이어 붙인 원소들의 |x| 평균 (기록용)."""
    return tf.add_n([tf.reduce_sum(tf.abs(x)) for x in ts]) / float(sum(x.shape.num_elements() for x in ts))


def route_grads(grads_and_vars):
    """apply_gradients 에 넘길 (기울기, 변수) 목록을 고친다 (추적 시점 Python, 한 번).
    unisparse: 층별 lambda*|dE|/|dL| 비를 기록하고, update='optimizer' 면 gamma·beta 기울기에 lambda*dE 를 더한다.
    atlif: theta 를 목록에서 뺀다 -- AdamW 가 theta 를 건드리지 않게. 과제 기울기는 apply_after_step 이 쓴다."""
    assert _STATE['model'] is not None, 'thr_reg: setup() 전에 학습 스텝이 추적됨'
    gv = list(grads_and_vars)
    idx = {v.ref(): i for i, (_, v) in enumerate(gv)}
    lam = _lam()
    drop = set()
    for p in _PLAN.values():
        assert p['traced'], ('thr_reg: 이번 학습 스텝 순전파에서 갱신량이 안 만들어짐', p['name'])
        if p['kind'] == 'unisparse':
            pen, task = [], []
            for var, acc in p['pairs']:
                i = idx.get(var.ref())
                assert i is not None, ('unisparse: BN 변수가 학습 목록에 없다', var.name)
                g = gv[i][0]
                assert isinstance(g, tf.Tensor), ('unisparse: BN 기울기가 dense 텐서가 아님', var.name, type(g))
                pen.append(lam * acc)
                task.append(g)
                if conf.reg_spike_unisparse_update == 'optimizer':
                    gv[i] = (g + lam * acc, var)
            p['lg_ratio'].assign_add(_mean_abs(pen) / (_mean_abs(task) + 1e-30))
        else:
            i = idx.get(p['theta'].ref())
            assert i is not None, ('atlif: theta 가 학습 목록에 없다', p['name'])
            g = gv[i][0]
            assert isinstance(g, tf.Tensor), ('atlif: theta 기울기 없음', p['name'], type(g))
            p['g_task'] = g
            drop.add(i)
        p['routed'] = True
    if drop:
        gv = [x for i, x in enumerate(gv) if i not in drop]
    if not _STATE['note_route']:
        _STATE['note_route'] = True
        if conf.reg_spike_atlif:
            print('[atlif] train_step 추적: theta {}개를 AdamW 목록에서 뺐다 (남은 변수 {}개)'
                  .format(len(drop), len(gv)), flush=True)
        else:
            print('[unisparse] train_step 추적: 층 {}개, update={}'
                  .format(len(_PLAN), conf.reg_spike_unisparse_update), flush=True)
    return gv


def apply_after_step(optimizer):
    assert _STATE['model'] is not None, 'thr_reg: setup() 전에 학습 스텝이 추적됨'
    lr = _current_lr(optimizer)
    lam = _lam()
    for p in _PLAN.values():
        # 이 추적 안에서 순전파가 갱신량을 실제로 만들었는지 (training 플래그가 안 넘어오면 조용히 0 이 된다),
        # route_grads 가 같은 추적에서 불렸는지 (안 불렸으면 theta 가 AdamW 로 갔다)
        assert p['traced'] and p['routed'], ('thr_reg: 순전파/route_grads 가 이번 추적에 없음', p['name'])
        p['traced'] = False
        p['routed'] = False
        if p['kind'] == 'unisparse':
            accs = [acc for _, acc in p['pairs']]
            if conf.reg_spike_unisparse_update == 'direct':
                for var, acc in p['pairs']:
                    var.assign_sub(lam * lr * acc)
            p['lg_pen'].assign_add(lam * lr * _mean_abs(accs))
        else:
            reg = lam * lr * p['acc']
            task = -lr * p['g_task'] if conf.reg_spike_atlif_theta_task == 'sgd' else tf.zeros_like(reg)
            p['theta'].assign(tf.maximum(p['theta'] + task + reg, ATLIF_THETA_MIN))
            p['d_task'].assign_add(task)
            p['d_reg'].assign_add(reg)
            p['g_last'].assign(p['g_task'])
            p['g_task'] = None
    _STATE['steps'].assign_add(1.0)
