"""TET 손실 (Deng et al., ICLR 2022) -- run_paper.py 의 vggdvslit 조합 전용 (26-10-26, 리뷰 반영 26-10-26).

lib_snn/__init__ 이 import 하지 않는다. vggdvslit 조합의 main_sweep.py 만 모델을 만든 직후
apply(model, config, out='step') 을 부른다. 다른 조합의 학습 경로와 무관하다. 모델(neurons.py 등)은 고치지 않는다.

기호
    u_t = 스텝 t 의 fc 층 출력 (Dense, 스텝마다 독립).  문헌(TET·UniSparse)의 스텝 출력은 이것에 해당한다.
    v_t = n_fc 막전위.  neurons.py run_type_out 은 적분 뒤 감쇠라 v_t = a (v_{t-1} + u_t), v_{-1} = 0
          (a = n_fc 의 leak_const, 기본 0.9). 그래서 v_t = sum_{k<=t} a^{t-k+1} u_k 로 누적된다.
    p_t = softmax(v_t) = 모델의 스텝 t 출력 (a_predictions).  l_t = log p_t = v_t - LSE(v_t).

out='step' (vggdvslit 가 쓰는 판, 문헌식)
    u~_t = l_t / a - l_{t-1}   (l_{-1} = 0)
         = u_t - LSE(v_t)/a + LSE(v_{t-1})       뒤 두 항은 클래스와 무관한 상수다.
    softmax 는 클래스 무관 상수에 불변이므로 softmax(u~_t) = softmax(u_t). 그래서
      손실 L = (1/T) sum_t CE_logit(y_ls, u~_t)  는 문헌 TET 손실 (1/T) sum_t CE(y, u_t) (MSE 항 0) 과 값이 같고,
      평가 argmax sum_t u~_t = argmax sum_t u_t  는 문헌 평가 (UniSparse main.py:482 output.mean(0)) 와 같다.
    기울기: 손실이 u~ 의 클래스 무관 이동에 불변이라 dL/dv 는 문헌식 손실을 u_t 로 쓴 것과 같은 꼴이 되고,
      n_fc 의 시간 경로 (v_{t-1} -> v_t) 를 autograd 가 따라가면 dL/du_t 가 문헌식과 같아진다.
      CPU 탐침 (실제 VGGSNN, T=10, B=4. fc 층 출력 u_t 를 직접 받아 만든 문헌식 손실과 비교) 결과 26-10-26:
      손실 차 0.0, 전 가중치 기울기 상대 노름차 5.2e-7, 변수별 최대 상대오차 3.0e-6 (BN 앞 conv bias 8개는
      기준 기울기 절대값 <= 8e-7 이라 제외), 평가 argmax 일치. 합성 4만 표본 (u ~ N(0, 1~10^2)) 에서도 argmax 불일치 0.
    한계: p 를 1e-30 에서 자르므로 (l >= -69) 한 스텝에서 클래스 간 막전위 차가 약 69 를 넘는 클래스는
      u~ 가 틀리고 기울기가 0 이 된다. 문헌식에는 없는 한계다. 걸린 비율은 PFloorLogger 가 에폭마다 [PFLOOR] 줄로
      찍는다 (26-10-26, 읽기 전용).
      바닥은 기울기에만 걸리지 않는다. 평가 점수 sum_t u~_t 도 잘린 log p 로 계산되므로, 2위 경쟁 클래스가 바닥에 걸린
      스텝에서는 점수가 틀어질 수 있다. n_fc 누적 막전위는 T=10 동안 약 6.5배까지 쌓이므로 학습 후반에 위험이 커진다.
      투입 조건: 연기시험과 첫 몇 에폭의 [PFLOOR] train·val 비율이 0. 0 보다 큰 값이 이어지면 fc 층 출력 u_t 를
      직접 받는 방식으로 바꾼다 (모델 출력이 바뀌므로 배포 검토 대상).
out='cum' (옛 판, 비교용. 어느 조합도 이 판을 켜지 않는다)
    누적 출력에 건 TET 꼴 손실 L = (1/T) sum_t CE(y_ls, p_t), 평가는 argmax mean_t l_t = argmax mean_t v_t.
    mean_t v_t = sum_k w_k u_k, w_k = a (1 - a^{T-k}) / ((1-a) T) 라 앞쪽 가중 평가다
    (T=10, a=0.9: w_0 = 5.86/T, w_9 = 0.90/T, 6.5:1). 문헌 평가 (u_t 균등 가중) 와 다르다.

L_task = 손실 전체. proc.py 의 loss-ratio 가 logs['loss'] - lambda*R 로 구하므로 수정이 필요 없고,
           옵티마이저가 줄이는 목적함수와 규제 비율이 맞는다. 대신 rho 눈금이 T=4 조합과 다르다.
val loss 도 같은 손실 (+규제항)이다. label smoothing 은 conf.label_smoothing (우리 0.1, 문헌 0).

모델 출력을 바꾸는 방법: Model.snn_output_processing 은 호출 시점에 읽는 속성이다
(lib_snn/model.py 의 'all' 분기 -> [B,T,C]). 출력을 쓰는 곳은 train/test step 의 compute_loss·
compute_metrics 뿐이다 (proc.py·callbacks.py 는 y_pred 를 안 읽는다).
"""
import tensorflow as tf

P_FLOOR = 1e-30
OUT_MODES = ('step', 'cum')


def log_p(y_pred):
    """[B,T,C] 확률 -> log, 1e-30 에서 자른다."""
    return tf.math.log(tf.clip_by_value(y_pred, P_FLOOR, 1.0))


def step_logits(y_pred, leak):
    """[B,T,C] p_t = softmax(v_t) -> [B,T,C] u~_t = l_t/a - l_{t-1} (= u_t + 클래스 무관 상수)."""
    lp = log_p(y_pred)
    prev = tf.concat([tf.zeros_like(lp[:, :1]), lp[:, :-1]], axis=1)
    return lp / leak - prev


def score(y_pred, out, leak):
    """[B,T,C] -> [B,C] 평가 점수. step: sum_t u~_t (argmax = 문헌), cum: mean_t l_t (argmax = mean_t v_t)."""
    if out == 'step':
        return tf.reduce_sum(step_logits(y_pred, leak), axis=1)
    return tf.reduce_mean(log_p(y_pred), axis=1)


def tet_score(y_pred):
    """옛 판(cum) 점수. 검증 스크립트 호환용."""
    return score(y_pred, 'cum', 1.0)


class TETLoss(tf.keras.losses.Loss):
    """y_true [B,C] (one-hot), y_pred [B,T,C] (스텝별 softmax) -> 표본별 (1/T) sum_t CE.
    step: CE 는 u~_t 에 logit 입력으로 건다 (클리핑 없음, 문헌의 logit CE 와 같은 함수).
    cum : CE 는 기본 경로(model_builder 의 CategoricalCrossentropy)와 같은 확률 입력 (1e-7 클리핑).
          T=1 이면 기본 경로 손실과 같다."""

    def __init__(self, label_smoothing=0.0, out='step', leak=None, name='tet_loss', **kw):
        super().__init__(name=name, **kw)
        if out not in OUT_MODES:
            raise ValueError(f'[TET] out={out!r} ({OUT_MODES} 중 하나)')
        if out == 'step' and not (leak is not None and 0.0 < float(leak) <= 1.0):
            raise ValueError(f'[TET] out=step 은 0 < leak <= 1 이 필요하다 (받은 값 {leak!r})')
        self.label_smoothing = float(label_smoothing)
        self.out = out
        self.leak = None if leak is None else float(leak)
        # 26-10-26 1e-30 바닥 감시 (읽기 전용, 손실값·기울기 무관). PFloorLogger 가 에폭마다 읽고 0 으로 돌린다.
        # Keras 메트릭과 같은 ON_READ 변수라 복제본마다 따로 쌓이고 읽을 때 합쳐진다 (merge_call 없음).
        # 메트릭으로 두지 않는 것은 logs 에 키를 넣지 않기 위해서다 (lit_spike_metrics.py 와 같은 이유).
        self.floor_stats = None
        if out == 'step':
            def _v(name, init, agg):
                return tf.Variable(init, trainable=False, dtype=tf.float32, name=name,
                                   synchronization=tf.VariableSynchronization.ON_READ, aggregation=agg)
            S = tf.VariableAggregation.SUM
            self.floor_stats = {
                'elem_hit': _v('pfloor_elem_hit', 0.0, S),     # p < 1e-30 인 (표본,스텝,클래스) 원소 수
                'elem_all': _v('pfloor_elem_all', 0.0, S),     # 전체 원소 수
                'samp_hit': _v('pfloor_samp_hit', 0.0, S),     # 그런 원소가 하나라도 있는 표본 수
                'samp_all': _v('pfloor_samp_all', 0.0, S),     # 전체 표본 수
                # 가장 작은 log p (1e-37 에서 자른 값, -85.2 가 하한). 바닥 -69.1 까지 남은 여유를 본다
                'min_logp': _v('pfloor_min_logp', 0.0, tf.VariableAggregation.ONLY_FIRST_REPLICA),
            }

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y = tf.broadcast_to(y_true[:, None, :], tf.shape(y_pred))
        if self.out == 'step':
            self._count_floor(y_pred)
            ce = tf.keras.losses.categorical_crossentropy(
                y, step_logits(y_pred, self.leak), from_logits=True, label_smoothing=self.label_smoothing)
        else:
            ce = tf.keras.losses.categorical_crossentropy(
                y, y_pred, from_logits=False, label_smoothing=self.label_smoothing)    # [B,T]
        return tf.reduce_mean(ce, axis=1)

    def _count_floor(self, y_pred):
        """바닥에 걸린 p 를 센다. 손실 계산과 별개의 텐서로만 세어 손실값·기울기는 바뀌지 않는다."""
        st = self.floor_stats
        if st is None:
            return
        yp = tf.stop_gradient(y_pred)
        hit = tf.cast(yp < P_FLOOR, tf.float32)                          # [B,T,C]
        st['elem_hit'].assign_add(tf.reduce_sum(hit))
        st['elem_all'].assign_add(tf.cast(tf.size(hit), tf.float32))
        st['samp_hit'].assign_add(tf.reduce_sum(tf.reduce_max(hit, axis=[1, 2])))
        st['samp_all'].assign_add(tf.cast(tf.shape(hit)[0], tf.float32))
        # 1e-38 은 비정규수라 CPU 에서 0 으로 밀려 log 가 -inf 가 된다 -> 1e-37 에서 자른다
        lo = tf.reduce_min(tf.math.log(tf.clip_by_value(yp, 1e-37, 1.0)))
        st['min_logp'].assign(tf.minimum(st['min_logp'], lo))

    def get_config(self):
        c = super().get_config()
        c.update(label_smoothing=self.label_smoothing, out=self.out, leak=self.leak)
        return c


def make_metrics(out, leak, name_acc, name_acc5):
    def acc(y_true, y_pred):
        return tf.keras.metrics.categorical_accuracy(y_true, score(y_pred, out, leak))

    def acc5(y_true, y_pred):
        return tf.keras.metrics.top_k_categorical_accuracy(y_true, score(y_pred, out, leak), k=5)
    return [tf.keras.metrics.MeanMetricWrapper(acc, name=name_acc),
            tf.keras.metrics.MeanMetricWrapper(acc5, name=name_acc5)]


def out_leak(model, conf):
    """n_fc 의 감쇠 a 를 돌려준다. v_t = a (v_{t-1} + u_t), v_{-1}=0 꼴이 아니면 멈춘다 (u_t 복원식이 틀어진다)."""
    if conf.snn_output_type != 'VMEM':
        raise ValueError(f'[TET] snn_output_type={conf.snn_output_type!r} (VMEM 이어야 u_t 를 복원한다)')
    if conf.neural_coding in ('TEMPORAL', 'NON_LINEAR'):
        raise ValueError(f'[TET] neural_coding={conf.neural_coding!r} (integration_default 경로만)')
    if conf.f_positive_vmem or conf.leak_const_train or conf.leak_time_dep:
        raise ValueError('[TET] f_positive_vmem / leak_const_train / leak_time_dep 가 켜져 있으면 복원식이 틀어진다')
    if tf.keras.mixed_precision.global_policy().name != 'float32':
        raise ValueError('[TET] mixed precision 이면 n_fc 막전위가 float16 으로 잘린다 (fp32 전용)')
    import lib_snn
    # model.layers_w_neuron 은 model.init (콜백 시작) 때 채워진다. 여기서는 model_builder 직후라 층을 직접 훑는다.
    outs = [l for l in model.layers if isinstance(l, lib_snn.activations.Activation)
            and isinstance(l.act, lib_snn.neurons.Neuron) and l.act.loc == 'OUT']
    if len(outs) != 1:
        raise ValueError(f'[TET] 출력 뉴런층이 {len(outs)}개 (1개여야 함)')
    n = outs[0].act
    if n.n_type == 'LIF':
        lc = n.leak_const.numpy().ravel()
        if lc.size == 0 or not (lc == lc[0]).all():
            raise ValueError('[TET] n_fc leak_const 가 뉴런마다 다르다 (클래스 무관 상수가 안 된다)')
        a = float(lc[0])
    elif n.n_type == 'IF':
        a = 1.0
    else:
        raise ValueError(f'[TET] n_fc n_type={n.n_type!r}')
    if not 0.0 < a <= 1.0:
        raise ValueError(f'[TET] n_fc leak_const={a}')
    return a


class MemPeakLogger(tf.keras.callbacks.Callback):
    """에폭 끝에 GPU 메모리 peak 를 찍는다 (연기시험 실측용). TF 가 GPU 를 통째로 잡아 nvidia-smi 로는
    실제 사용량을 못 본다. GPU 가 없으면 아무것도 안 한다. 학습에는 영향이 없다."""

    def on_epoch_end(self, epoch, logs=None):
        try:
            if not tf.config.list_logical_devices('GPU'):
                return
            info = tf.config.experimental.get_memory_info('GPU:0')
            print(f'[MEM] epoch {epoch + 1} GPU:0 peak {info["peak"] / 2**30:.2f} GiB '
                  f'current {info["current"] / 2**30:.2f} GiB', flush=True)
        except Exception as e:      # 측정 실패가 학습을 죽이면 안 된다
            print(f'[MEM] 측정 실패: {e}', flush=True)


class PFloorLogger(tf.keras.callbacks.Callback):
    """26-10-26 TETLoss(out='step') 의 1e-30 바닥 감시. 학습 구간과 검증 구간을 따로 [PFLOOR] 줄로 찍는다.
    바닥에 걸린 원소는 u~ 가 틀리고 기울기가 0 이다 (모듈 머리 '한계'). 학습 구간 비율이 0 보다 큰 값으로
    이어지면 손실을 softmax 앞 logit 에서 받는 방식으로 바꿔야 한다. 읽고 0 으로 돌리기만 한다 (학습 무영향).
    min log p 의 바닥은 log(1e-30) = -69.08 이다."""

    def _stats(self):
        loss = getattr(self.model, 'loss', None)
        return getattr(loss, 'floor_stats', None)

    def _reset(self):
        st = self._stats()
        if st:
            for v in st.values():
                v.assign(tf.zeros_like(v))

    def _report(self, part, epoch):
        st = self._stats()
        if not st:
            return
        try:
            r = {k: float(v.numpy()) for k, v in st.items()}
            ef = r['elem_hit'] / max(r['elem_all'], 1.0)
            sf = r['samp_hit'] / max(r['samp_all'], 1.0)
            # 학습 구간 줄은 진행 막대 (\r, 줄바꿈 없음) 바로 뒤에 찍혀 같은 줄에 붙는다 -> 앞에 줄바꿈
            print(f'{chr(10) if part == "train" else ""}[PFLOOR] epoch {epoch + 1} {part}: p<1e-30 원소 비율 {ef:.3e} ({r["elem_hit"]:.0f}/{r["elem_all"]:.0f}), '
                  f'표본 비율 {sf:.3e} ({r["samp_hit"]:.0f}/{r["samp_all"]:.0f}), min log p {r["min_logp"]:.2f} '
                  f'(바닥 -69.08)', flush=True)
        except Exception as e:     # 측정 실패가 학습을 죽이면 안 된다
            print(f'[PFLOOR] 측정 실패: {e}', flush=True)

    def on_train_begin(self, logs=None):
        self.epoch = 0
        if self._stats() is None:
            print('[PFLOOR] model.loss 에 floor_stats 가 없다 (TETLoss out=step 이 아님) -- 감시하지 않는다', flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self._reset()

    def on_test_begin(self, logs=None):
        self._report('train', self.epoch)
        self._reset()

    def on_test_end(self, logs=None):
        self._report('val', self.epoch)
        self._reset()


def apply(model, config, out):
    """모델 출력을 [B,T,C] 로 바꾸고 TET 손실로 다시 compile 한다. out 은 'step'(문헌식) 또는 'cum'(옛 판).
    기본값을 두지 않는다 -- 생성된 main_sweep.py 에 어느 판인지 글자로 남게 한다.
    메트릭 이름은 기존 그대로 ('acc', 'acc-5') 두어 val_acc 모니터·체크포인트·로그 파서가 그대로 동작한다.
    main_sweep.py 의 dist_strategy.scope() 안에서 불린다."""
    conf = config.flags
    if out not in OUT_MODES:
        raise ValueError(f'[TET] out={out!r} ({OUT_MODES} 중 하나)')
    if conf.nn_mode != 'SNN' or conf.snn_training_spatial_first:
        raise ValueError('[TET] nn_mode=SNN, snn_training_spatial_first=False 에서만 쓴다')
    if not hasattr(model, 'snn_output_processing'):
        raise ValueError('[TET] lib_snn.model.Model 이 아니다 (snn_output_processing 없음)')
    if model.optimizer is None:
        raise ValueError('[TET] model_builder 가 compile 하기 전이다')
    leak = out_leak(model, conf) if out == 'step' else None
    model.snn_output_processing = 'all'
    model.compile(optimizer=model.optimizer,
                  loss=TETLoss(label_smoothing=conf.label_smoothing, out=out, leak=leak),
                  metrics=make_metrics(out, leak, config.metric_name_acc, config.metric_name_acc_top5),
                  run_eagerly=config.eager_mode)
    if out == 'step':
        desc = (f'out=step: u~_t = log p_t/{leak:g} - log p_(t-1) (= fc 출력 u_t + 클래스 무관 상수), '
                f'loss = mean_t CE_logit(u~_t), metric {config.metric_name_acc} = argmax sum_t u~_t (문헌 평가)')
    else:
        desc = (f'out=cum: loss = mean_t CE(p_t) (누적 출력에 건 TET 꼴), '
                f'metric {config.metric_name_acc} = argmax mean_t log p_t (앞쪽 가중 평가)')
    print(f'[TET] snn_output_processing=all, {desc}, label_smoothing={conf.label_smoothing}, '
          f'T={conf.time_step}', flush=True)
    return model
