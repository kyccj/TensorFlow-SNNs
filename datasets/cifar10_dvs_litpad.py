"""CIFAR10-DVS 문헌 데이터 프로토콜 로더 -- 배치가 시험 1000장을 나누지 못할 때의 판 (26-10-30).
run_paper_vgglit.py 의 vggdvslitsgd 조합 (레시피 탐색 r2, 배치 64) 전용.

이 모듈은 아무도 import 하지 않는다. vggdvslitsgd 의 main_sweep.py 만 데이터 로드 전에 `datasets.cifar10_dvs = 이 모듈`
로 바꿔 끼운다 (vggdvslit 의 datasets/cifar10_dvs_lit.py 와 같은 방식, 바꿔 끼우는 글자는 run_paper_vgglit.py PAD_SWAPS).
데이터 프로토콜 (캐시·sha256 대조·분할·프레임·눈금·증강) 은 datasets/cifar10_dvs_lit.py 의 함수를 그대로 부른다.
여기서 바꾸는 것은 평가셋의 배치 구성뿐이다. 학습셋은 cifar10_dvs_lit.make_datasets 가 만든 것 그대로다.

왜 필요한가
  모델 배치가 고정이다. models/vggsnn.py 가 Input(batch_size=B) 로 모델을 만들고, 뉴런 상태 텐서 (lib_snn/neurons.py
  tf.zeros(dim)) 와 단계별 BN (temporal_batch, 정적 배치 필요) 이 배치 축을 포함한다. 그래서 학습 배치와 다른 평가 배치를
  둘 수 없다. B=64 면 1000 % 64 = 40 이라 기존 로더 (drop_remainder) 는 시험 960장만 평가한다.
채움 방식
  시험 1000장 (순서 그대로) 뒤에 앞쪽 24장을 복사해 붙여 1024 행 = 16 배치 x 64 로 만든다. 채움 행은 마지막 배치의
  뒤 24 행이다 (valid_rows). 표본 가중은 실제 표본 1024/1000, 채움 행 0 이다.
  - val_acc·val_acc-5: Keras 가중 평균 sum(w*a)/sum(w) = 1000장 평균. 단 Keras 는 compile 의 metrics= 에는 가중을
    안 쓰고 weighted_metrics= 에만 쓴다. 그래서 main_sweep.py 가 tet_loss.apply 뒤에
    lib_snn/lit_spike_metrics_pad.use_weighted_metrics 로 정확도 메트릭을 weighted_metrics 로 옮긴다 (run_paper_vgglit.py).
  - val_loss: Keras 는 배치 손실을 sum(w*l)/B 로 줄이고 (SUM_OVER_BATCH_SIZE) 배치마다 B 로 가중 평균한다
    (keras/engine/compile_utils.py LossesContainer). 전체가 sum(w*l)/1024 이므로 실제 표본 가중을 1024/1000 으로 두면
    sum(l)/1000 = 1000장 평균이 된다. 규제 항 (model.losses) 은 가중과 무관하게 배치 평균으로 더해진다. ours 의 평가
    규제 항에는 채움 행 스파이크가 들어가지만 학습·loss-ratio 와는 무관하다 (둘 다 학습 배치에서 센다).
  - 평가 모드에서 BN 은 이동평균, conv·pool·뉴런은 표본별이라 채움 행이 실제 표본의 출력을 바꾸지 않는다.
  - [LITSPK]: lib_snn/lit_spike_metrics_pad.py 가 LAYOUT 으로 채움 행을 빼고 센다 (n=1000). 같은 콜백이 [EVALN] 줄로
    Keras acc 메트릭의 가중합과 맞춘 수를 찍는다.
  - [PFLOOR] val: TETLoss 가 행 단위로 세므로 1024 행 기준이다 (채움 행은 실제 표본 복사본, 진단용).
  - s_count: 학습 배치에서 세므로 (lib_snn/proc.py postproc_batch_train) 이 장치와 무관하다.
학습셋: 9000 % 64 = 40 이라 drop_remainder 로 에폭마다 무작위 40장 (재셔플) 을 빼고 140 스텝 (기존 로더 그대로).
B 가 시험 표본 수를 나누면 (예: 40) 채움 없이 cifar10_dvs_lit.load() 와 같은 데이터셋을 돌려준다 (가중 없음).

맨 위에서는 표준 라이브러리만 import 한다 (cifar10_dvs_lit.py 와 같은 이유). numpy·tensorflow·config 는 함수 안에서.
"""
import os

# load() 가 채운다. lib_snn/lit_spike_metrics_pad.py 가 읽는다.
#   n_real 실제 시험 표본 수, batch 배치, n_rows 평가 행 수 (배치 배수), n_pad 채움 행 수, w_real 실제 표본 가중
LAYOUT = {}


def valid_rows(b, n_real, batch):
    """평가 배치 b (0 시작) 의 앞쪽 실제 표본 행 수. 채움 행은 마지막 배치의 뒤쪽에만 있다."""
    return max(0, min(batch, n_real - b * batch))


def pad_layout(n_real, batch):
    n_pad = (-n_real) % batch
    n_rows = n_real + n_pad
    return {'n_real': n_real, 'batch': batch, 'n_rows': n_rows, 'n_pad': n_pad,
            'w_real': (n_rows / n_real) if n_pad else 1.0}


def pad_eval(test1, lay):
    """test1 = 배치 1 로 묶인 시험셋 (순서 고정) -> (x, y, w) 를 lay['batch'] 로 묶은 평가셋 (정적 배치, 배치 수 확인)."""
    import tensorflow as tf
    B, n_pad = lay['batch'], lay['n_pad']
    w_real = tf.constant(lay['w_real'], tf.float32)
    w_pad = tf.constant(0.0, tf.float32)
    one = test1.unbatch()
    real = one.map(lambda x, y: (x, y, w_real))
    pad = one.take(n_pad).map(lambda x, y: (x, y, w_pad))
    ds = real.concatenate(pad).batch(B, drop_remainder=True)
    # unbatch 뒤라 배치 수를 tf.data 가 모른다. 적어 두고 실제와 다르면 첫 평가에서 멈추게 한다.
    ds = ds.apply(tf.data.experimental.assert_cardinality(lay['n_rows'] // B))
    return ds.prefetch(tf.data.AUTOTUNE)


def load():
    """datasets/cifar10_dvs.load() 와 같은 6-튜플. valid 와 test 는 같은 시험 1000장 (채움 행 가중 0) 이다.
    점검 (dataset, bn_per_step, 캐시 sha256) 은 cifar10_dvs_lit.load() 와 같다."""
    from config import config
    import datasets.cifar10_dvs_lit as lit
    conf = config.flags

    T = conf.time_step
    size = conf.cifar10_dvs_img_size
    B = config.batch_size
    if conf.dataset != 'CIFAR10_DVS':
        raise ValueError(f'[DVSLIT] dataset={conf.dataset!r} (CIFAR10_DVS 전용)')
    if not conf.bn_per_step:
        raise ValueError('[DVSLIT] bn_per_step=True 가 필요하다 (run_paper_vgglit.py COMBO_FLAGS)')
    allow_partial = os.environ.get(lit.ENV_ALLOW_PARTIAL) == '1'
    d = lit.cache_dir(T, size)

    conf.time_dim_size = T
    conf.input_data_time_dim = True

    train_ds, test_ds, n_tr, n_te, man = lit.make_datasets(d, T, size, B, allow_partial, verify_sha=True)
    lay = pad_layout(n_te, B)
    if lay['n_pad']:
        # 시험셋만 배치 1 로 다시 만든다 (같은 캐시, sha256 은 위에서 대조했다). 학습셋은 위의 것을 쓴다.
        _, test1, _, n_te1, _ = lit.make_datasets(d, T, size, 1, allow_partial, augment_train=False,
                                                  verify_sha=False)
        if n_te1 != n_te:
            raise RuntimeError(f'[DVSLIT] 시험 표본 수가 다르다 ({n_te1} vs {n_te})')
        test_ds = pad_eval(test1, lay)
    LAYOUT.clear()
    LAYOUT.update(lay)
    train_ds_num = (n_tr // B) * B
    valid_ds_num = n_te
    print(f'[DVSLIT] cache={d} key={man["key"]} train={n_tr} test={n_te} batch={B} '
          f'-> steps {n_tr // B} / {lay["n_rows"] // B}, 사용 {train_ds_num} / {valid_ds_num}장 '
          f'(평가 {lay["n_rows"]}행 = 시험 {n_te} + 채움 {lay["n_pad"]}행 가중 0, 실제 표본 가중 {lay["w_real"]:.6g}), '
          f'input x{lit.INPUT_SCALE}, clip aug (flip {lit.FLIP_P} + roll {lit.ROLL_MAX} / rot {lit.ROT_DEG:g} / '
          f'cutout {lit.CUTOUT_LEN})'
          + (f'  ** 부분 캐시 limit={man["limit"]} (검증용, sha 대조 안 함) **' if man.get('limit') is not None
             else ', sha256 = canus 기준'),
          flush=True)
    print('[DVSLIT] 범위: 문헌과 같은 것은 데이터 프로토콜뿐이고 모델·학습 레시피는 우리 것이다 (이 조합은 옵티마이저만 문헌 쪽으로 '
          '옮겼다 -- configs_paper/vggdvs_litsgd 머리 주석). 문헌 수치와 비교하지 않는다. valid = test 1000장 (best epoch on test, '
          '문헌 관례) -- 정확도는 최고 에폭·마지막 에폭 둘 다 보고하고, 다른 조합과 같은 표에 섞지 않는다.', flush=True)
    return train_ds, test_ds, test_ds, train_ds_num, valid_ds_num, valid_ds_num
