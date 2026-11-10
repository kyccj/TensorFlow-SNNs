"""채움 평가셋 (datasets/cifar10_dvs_litpad.py) 용 지표 장치 두 가지 -- vggdvslitsgd 전용 (26-10-30).

lib_snn/__init__ 이 import 하지 않는다. run_paper_vgglit.py 가 vggdvslitsgd 의 main_sweep.py 에서만 부른다
(PAD_SWAPS). 둘 다 학습 손실·기울기·규제·loss-ratio 에 영향이 없다.

1. use_weighted_metrics(model, config, out): lib_snn/tet_loss.apply 바로 뒤에 부른다. 같은 TETLoss 객체·옵티마이저로
   다시 compile 하되 정확도 메트릭 (acc, acc-5) 을 metrics= 대신 weighted_metrics= 로 넘긴다.
   왜: Keras 2.12 MetricsContainer 는 metrics= 로 받은 메트릭을 표본 가중 없이 (mask 만) 갱신하고 weighted_metrics= 로
   받은 것만 가중을 쓴다 (keras/engine/compile_utils.py 605·610행). 그대로 두면 val_acc 가 채움 행까지 넣은 1024 행
   평균이 된다 (CPU 검증 26-10-30: 직접 1000장 0.0960, metrics= 0.0967, weighted_metrics= 0.0960).
   손실은 원래부터 가중을 쓴다 (LossesContainer). 학습 배치에는 가중이 없어 학습 acc 는 그대로다. 메트릭 이름도 그대로다.
2. LitSpikeLoggerPad: lib_snn/lit_spike_metrics.py LitSpikeLogger 의 채움 판. 부모는 배치의 모든 행을 세므로, 배치 b 의
   앞 valid_rows(b) 행만 센다 (채움 행 = 마지막 배치의 뒤쪽). 정의 (hid_spk·fr·sop·in_sum), [LITSPK] 줄 형식,
   lit_spikes.csv 열은 부모와 같고 n 은 실제 표본 수 (1000) 다.
   [EVALN] 줄 (검증 끝마다): 평가 배치 수 x 배치 = 행 수, 실제 표본 수, Keras acc 메트릭의 가중합·맞춘 수 (가중합 / 실제
   표본 가중). 가중이 반영됐으면 맞춘 수가 정수이고 val_acc = 맞춘 수 / 1000 이다. 연기시험에서 이 줄로 val 이 시험
   1000장 전부인지 본다.
"""
import tensorflow as tf

from datasets.cifar10_dvs_litpad import valid_rows
from lib_snn.lit_spike_metrics import LitSpikeLogger, _read, conv_sop


def use_weighted_metrics(model, config, out):
    """tet_loss.apply 로 compile 된 모델을 정확도 메트릭만 weighted_metrics 로 옮겨 다시 compile 한다 (머리 주석 1)."""
    import lib_snn.tet_loss as _tet
    loss = model.loss
    if not isinstance(loss, _tet.TETLoss) or loss.out != out:
        raise ValueError(f'[EVALN] tet_loss.apply(out={out!r}) 뒤에 불러야 한다 (지금 loss={loss!r})')
    model.compile(optimizer=model.optimizer, loss=loss,
                  weighted_metrics=_tet.make_metrics(out, loss.leak, config.metric_name_acc,
                                                     config.metric_name_acc_top5),
                  run_eagerly=config.eager_mode)
    print(f'[EVALN] {config.metric_name_acc}·{config.metric_name_acc_top5} 를 weighted_metrics 로 다시 compile '
          f'(같은 손실 객체·옵티마이저, 평가 채움 행 가중 0 이 정확도에도 걸린다)', flush=True)
    return model


class LitSpikeLoggerPad(LitSpikeLogger):
    """layout = datasets.cifar10_dvs_litpad.LAYOUT (load() 뒤에 채워진 것)."""

    def __init__(self, model, config, layout):
        super().__init__(model, config)
        for k in ('n_real', 'batch', 'n_rows', 'w_real'):
            if k not in layout:
                raise ValueError(f'[LITSPK] 채움 배치 정보 {k} 가 없다 (datasets.cifar10_dvs_litpad.load() 전에 불렸다?)')
        self.n_real = int(layout['n_real'])
        self.B = int(layout['batch'])
        self.n_rows = int(layout['n_rows'])
        self.w_real = float(layout['w_real'])
        self.metric_name = config.metric_name_acc
        self._warned = False
        print(f'[LITSPK] 채움 평가셋: 실제 {self.n_real} + 채움 {self.n_rows - self.n_real} = {self.n_rows} 행 '
              f'(배치 {self.B}), 채움 행은 세지 않는다', flush=True)

    def on_test_begin(self, logs=None):
        super().on_test_begin(logs)
        self._b = 0
        self._rows = 0

    def _warn(self, msg):
        if not self._warned:
            print(f'[LITSPK] 경고: {msg}', flush=True)
            self._warned = True

    def on_test_batch_end(self, batch, logs=None):
        # 배치 번호는 직접 센다 (Keras 의 batch 인자와 다르면 한 번 경고)
        if batch != self._b:
            self._warn(f'Keras 배치 번호 {batch} != 센 번호 {self._b}')
        nv = valid_rows(self._b, self.n_real, self.B)
        rows = None
        for l in self.hid:
            sc = _read(l.act.spike_count)
            rows = int(sc.shape[0])
            s = float(tf.reduce_sum(sc[:nv]))
            self.layer_spk[l.name] += s
            self.hid_spk += s
            self.hid_int += float(tf.reduce_sum(_read(l.act.spike_count_int)[:nv]))
        for l in self.inp:
            self.in_sum += float(tf.reduce_sum(_read(l.act.spike_count)[:nv]))
        for conv, src, pools in self.plan:
            self.sop += float(conv_sop(_read(src.act.spike_count)[:nv], conv, pools))
        if rows is not None and rows != self.B:
            self._warn(f'뉴런 카운터 배치 {rows} != 채움 배치 {self.B}')
        self.n += nv
        self._rows += rows or 0
        self._b += 1

    def on_test_end(self, logs=None):
        super().on_test_end(logs)
        cnt = tot = None
        try:
            for m in self.model.metrics:
                if m.name == self.metric_name and hasattr(m, 'count') and hasattr(m, 'total'):
                    cnt, tot = float(m.count.numpy()), float(m.total.numpy())
                    break
        except Exception as e:      # 읽기 실패가 학습을 죽이면 안 된다
            print(f'[EVALN] acc 메트릭 읽기 실패: {e}', flush=True)
        acc = (logs or {}).get(self.metric_name)
        if cnt is None:
            seen = 'acc 메트릭 못 읽음'
        else:
            hit = tot / self.w_real
            ok = abs(hit - round(hit)) < 1e-2 and abs(cnt - self.n_real * self.w_real) < 1e-2
            seen = (f'acc 메트릭 가중합 {cnt:.4f} (= 실제 {self.n_real} x {self.w_real:.6g}), 맞춘 수 {hit:.3f}'
                    + ('' if ok else ' -- 경고: 맞춘 수가 정수가 아니다. 정확도에 가중이 안 걸린 것 같다'))
        print(f'[EVALN] epoch {self.epoch + 1}: 평가 배치 {self._b} x {self.B} = {self._rows} 행, 실제 표본 {self.n} '
              f'(채움 {self._rows - self.n} 행 가중 0), {seen}, acc {acc}', flush=True)
