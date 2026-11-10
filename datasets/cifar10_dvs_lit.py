"""CIFAR10-DVS 문헌 데이터 프로토콜 로더 -- run_paper.py 의 vggdvslit 조합 전용 (26-10-26, 리뷰 반영 26-10-26).

이 모듈은 아무도 import 하지 않는다. vggdvslit 조합의 main_sweep.py 만 데이터 로드 전에
`datasets.cifar10_dvs = 이 모듈` 로 바꿔 끼운다 (datasets/datasets.py 가 호출 시점에
getattr(datasets, 'cifar10_dvs') 로 로더를 찾기 때문). 다른 조합과 기존 로더
(datasets/cifar10_dvs.py) 는 이 파일과 무관하다.

범위: 문헌과 같게 맞춘 것은 **데이터 프로토콜뿐**이다. 모델·학습 레시피는 우리 것이다 (아래 '다른 8축').
  그래서 이 조합의 정확도가 문헌 수치와 다르거나 같아도 그 원인을 프로토콜과 레시피로 나눌 수 없다.
  '문헌 설정'이라고 부르지 않는다. 비교 기준 수치는 UniSparse dense 83.50 +-0.48 하나만, 레시피가 다르다는
  단서를 붙여 옆에 둔다 (TET 83.17 은 0/1 프레임·flip+roll·MSE 항·Adam, Shi 82.4 는 프루닝 파이프라인 결과라 뺀다).
기준 = UniSparse (ICML 2026) 의 CIFAR10-DVS 전처리 (spikingjelly CIFAR10DVS frame + UniSparse main.py /
  utils/augment.py / utils/utils.py). Shi 2024 (ICLR) 는 같은 전처리 코드이되 보간 기본값이 bilinear 다
  (UniSparse 는 bicubic, TET 는 프레임별 max 정규화 후 PIL 8비트 bilinear). 여기서는 UniSparse 의 bicubic 을 골랐다. Activity Pruning (NeurIPS 2025)
  저장소에는 CIFAR10-DVS 설정이 없고 --dvs-augment 기본값이 꺼짐이라 같은 전처리를 썼는지 코드로 확인되지 않는다.
맞춘 것 (데이터 프로토콜):
  - 프레임: 이벤트 수 기준 T 등분 (spikingjelly split_by='number'). 조각 i 는 [i*(N//T), (i+1)*(N//T)),
    마지막 조각은 N 까지 (나머지를 받는다). 극성 2채널, 값은 이벤트 개수 그대로 (눈금 1.0, 문헌과 같다).
  - 128 -> 48: bicubic + antialias, 증강 전에 1회. TF tf.image.resize(antialias=True) 와 torchvision
    Resize(antialias, BICUBIC) 의 평균 절대차 1e-7 (실측). 이 결과를 캐시로 둔다 (build_dvslit_cache.py).
  - 학습 증강은 클립 단위 (모든 프레임·채널에 같은 값): 좌우 뒤집기 0.5 후
    {roll +-5px, 회전 +-30도 nearest·바깥 0, cutout 16x16} 중 하나를 1/3 씩. 시험은 증강 없음.
  - 분할: 클래스마다 파일명 사전순 앞 900 학습 / 뒤 100 시험 (UniSparse DatasetSplitter).
    파일명 cifar10_<class>_<id>.npz 의 사전순은 str(id) 의 사전순과 같다 ('.' 이 숫자보다 앞).
    그래서 시험 id 는 클래스마다 {91..99, 909..999} 다. 문헌과 같은 1000장이다.
  - BN 은 T·B 묶음 (run_paper COMBO_FLAGS 의 bn_per_step). 이 로더는 그것이 켜졌는지만 확인한다.
다른 8축 (모델·학습 레시피, 우리 것 / UniSparse):
  (1) 뉴런: LIF 0.9 + reset_by_sub + asym 0.6 surrogate / tau=2 decay_input + hard reset 0 + ATan + detach_reset
  (2) 출력: FC10 + n_fc 누적 막전위 (lib_snn/tet_loss.py 가 스텝 출력 u_t 를 복원해 손실·평가에 쓴다)
      / 스텝별 Linear 4608->100 + AvgPool1d(10)
  (3) 옵티마이저: AdamW 6e-3, wd 2e-2, COS (워밍업 10에폭 lr_init 1e-5 -> 6e-3, 하한 1e-5), 200에폭, 배치 40
      / SGD nesterov 0.025, wd 0, 워밍업 0, cos 하한 0, 300에폭, 배치 64
  (4) label smoothing 0.1 / 0
  (5) 초기화 glorot_uniform / kaiming_normal fan_out
  (6) BN momentum·eps 0.99·1e-4 / 0.9·1e-5 (torch momentum 0.1 = keras 0.9)
  (7) fp32 / AMP
  (8) conv bias 있음 (lib_snn.layers.Conv2D 기본, BN 바로 앞) / 없음 (bias=False)
검증셋 = 시험셋: valid 와 test 가 같은 1000장이라 val_acc 최고 에폭을 고르는 것이 곧 시험셋으로 고르는 것이다
  (best epoch on test, 문헌 관례). 기존 T=4 조합의 val (tfds 앞 10% 992장) 과 숫자의 성격이 다르므로
  vggdvslit 는 별도 표에 두고 기존 조합과 같은 표·그래프에 섞지 않는다.
좌표·극성은 spikingjelly 와 같게 둔다 (기존 로더는 W 축이 거울이고 채널 순서가 반대다):
  H = y_sj = tfds coords[:,0],  W = x_sj = 127 - tfds coords[:,1],  채널 = p_sj = tfds polarity.
  근거: events_tfds 와 sj 의 aedat 리더(load_raw_events / parse_raw_address)는 마스크·시프트가 같다.
  tfds builder 는 (127-x, y), not p 를, sj 는 (x=127-y, y=127-x), 1-p 를 내고 sj 는 p==0 을 채널 0 에 둔다.
라벨 번호는 tfds 순서 그대로다 (sj 는 폴더명 알파벳순이라 번호만 다르고 표본은 같다).
캐시 동일성: 서버마다 TF 버전이 달라 bicubic 결과가 비트 단위로 달라질 수 있다. 그러면 서버 효과와 방법 효과가
  섞이므로 로더는 캐시 파일의 sha256 을 canus 기준 캐시 (REF_SHA256) 와 대조해 다르면 멈춘다. 원격 서버에는
  다시 빌드하지 말고 canus 캐시를 복사한다 (원격 작업은 3등급).

우리 것을 유지한 것: one-hot 라벨, drop_remainder 배치(모델 배치 고정). 입력 눈금은 26-10-26 리뷰 뒤 0.1 에서
1.0 (문헌과 같은 개수 그대로) 으로 바꿨다. BN eps=1e-4 (lib_snn/layers.py) 라 스케일 불변이 아니고, 0.1 이면
초기 conv1 출력 분산(채널 중앙값 3.4e-4)에 eps 가 약 0.3배로 섞였다. 1.0 이면 약 0.003배다. n_in 출력 합은
s_count 에 들어가므로 (1.0 이면 표본당 약 2.8만, T=10 합) 문헌과 맞춘 스파이크는 lib_snn/lit_spike_metrics.py 가
n_in 을 빼고 따로 센다.

맨 위에서는 표준 라이브러리만 import 한다. run_paper.py 의 preflight 가 이 파일을 경로로 불러와
캐시 manifest·배열 헤더·sha256 을 확인하는데, run_paper.py 는 numpy 없는 python 으로도 돌 수 있기 때문이다.
numpy·tensorflow·config 는 쓰는 함수 안에서 불러온다.
"""
import ast
import hashlib
import json
import math
import os

# --- 프로토콜 상수 (manifest 와 대조한다. 바꾸면 캐시 키가 달라져야 한다) -------------------------
T_DEFAULT = 10
SIZE_DEFAULT = 48
SRC_HW = 128
N_CLASS = 10
PER_CLASS = 1000            # 클래스당 표본 수 (tfds 전수 스캔: 클래스마다 id 0..999 빠짐없음)
N_TRAIN_PER_CLASS = 900
RESIZE_METHOD = 'bicubic'
ANTIALIAS = True
LAYOUT = 'sj'               # H=y_sj, W=x_sj, ch=p_sj
SPLIT_RULE = 'per-class sorted(str(example_id)): first 900 train, last 100 test'
INPUT_SCALE = 1.0           # 문헌과 같은 이벤트 개수 그대로 (26-10-26 리뷰 뒤 0.1 -> 1.0, 머리 주석)
FORMAT_VERSION = 1

# 증강 (UniSparse DVSAugment, M=1 N=2: rotate 15*N=30도, cutout ratio N/6 -> int(48/3)=16, roll 2N+1=5)
FLIP_P = 0.5
ROLL_MAX = 5
ROT_DEG = 30.0
CUTOUT_LEN = 16

# canus 기준 캐시 (26-10-26 16:41 빌드, TF 2.12.1, numpy 1.23.5) 의 파일 sha256. 다른 서버·다른 TF 로 다시 빌드한
# 캐시는 bicubic 결과가 비트 단위로 다를 수 있어 거부한다 (서버 효과와 방법 효과가 섞이지 않게). 원격에는 이 캐시를
# 복사한다. 이 값을 바꾸는 것은 프로토콜을 바꾸는 것이다 -- 바꾸면 이전 vggdvslit 런과 섞지 않는다.
REF_SHA256 = {
    'T10_s48_bicubic_aa': {
        'test_id.npy': 'b061947c834d304d64ecce57fb3e3880bee57c2bdd246aeebab56c5f3ede2c66',
        'test_nev.npy': 'a9e6f06ecf82256c775ba5a54fb7949a5ddaf776728fa923f08b2da17021713f',
        'test_x.npy': '64462e100cf424029942eddeef099a34a7153597232ee699e8f53e154b26f48a',
        'test_y.npy': 'dbedcc90f6a6a0684902a0ff704e18a2de6fa912f41cb083c8d534c637c1a2f6',
        'train_id.npy': '7cd1302e7c957cecf5ac1095a4e3bd7f85ec784a8d0e1a6465da8b914ad32587',
        'train_nev.npy': 'fd71d960df5b46066b97872ed5af329caa17e1fff2eb08e7d2f0884fe32ccfb0',
        'train_x.npy': 'f5597eb19146dfe67df816b8d465ecaab85d841b932fabdc18ed6f5d83584f02',
        'train_y.npy': 'e6dd4dc85d5b6d6f221fe6c3ad7171e42264053100c7747dcc953511ea7fc9f7',
    },
}

N_TRAIN_FULL = N_CLASS * N_TRAIN_PER_CLASS                  # 9000
N_TEST_FULL = N_CLASS * (PER_CLASS - N_TRAIN_PER_CLASS)     # 1000

# 부분 캐시(검증용, build_dvslit_cache.py --limit-*)는 이 env 가 '1' 일 때만 받는다.
# 실제 런이 부분 캐시로 조용히 도는 사고를 막는다.
ENV_ALLOW_PARTIAL = 'EIP_DVSLIT_ALLOW_PARTIAL'


# --- 캐시 위치·manifest ------------------------------------------------------------------------
def cache_key(T=T_DEFAULT, size=SIZE_DEFAULT):
    return f'T{T}_s{size}_{RESIZE_METHOD}_aa'


def cache_root():
    """캐시 루트. env EIP_DVSLIT_CACHE 가 있으면 그것, 없으면 dirname(EIP_STORE)/dvslit_cache.
    canus 기본 /media/hdd1/kyccj/EIP/dvslit_cache, sejong·55 는 EIP_STORE=/srv2/kyccj/EIP/paper 라
    /srv2/kyccj/EIP/dvslit_cache. 키 디렉토리는 항상 그 아래에 붙인다."""
    r = os.environ.get('EIP_DVSLIT_CACHE')
    if r:
        return r
    store = os.environ.get('EIP_STORE', '/media/hdd1/kyccj/EIP/paper')
    return os.path.join(os.path.dirname(os.path.normpath(store)), 'dvslit_cache')


def cache_dir(T=T_DEFAULT, size=SIZE_DEFAULT):
    return os.path.join(cache_root(), cache_key(T, size))


def expected_fields(T=T_DEFAULT, size=SIZE_DEFAULT):
    """manifest 에서 반드시 같아야 하는 항목."""
    return {
        'format_version': FORMAT_VERSION,
        'key': cache_key(T, size),
        'T': T,
        'size': size,
        'src_hw': SRC_HW,
        'resize_method': RESIZE_METHOD,
        'antialias': ANTIALIAS,
        'layout': LAYOUT,
        'split_rule': SPLIT_RULE,
        'frame_rule': 'split_by=number (N//T per segment, last segment takes remainder), 2ch counts',
        'value_scale': 1.0,
        'dtype': 'float32',
    }


def npy_header(path):
    """.npy 헤더만 읽어 (descr, fortran_order, shape). numpy 없이 동작한다 (NPY 형식 1.x/2.x/3.x)."""
    with open(path, 'rb') as f:
        if f.read(6) != b'\x93NUMPY':
            raise RuntimeError(f'[DVSLIT] npy 형식 아님: {path}')
        major = f.read(2)[0]
        hlen = int.from_bytes(f.read(2 if major == 1 else 4), 'little')
        h = ast.literal_eval(f.read(hlen).decode('latin1'))
    return h['descr'], h['fortran_order'], tuple(h['shape'])


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_ref_sha(d, T=T_DEFAULT, size=SIZE_DEFAULT):
    """캐시 파일 전부의 sha256 을 REF_SHA256 (canus 기준) 과 대조한다. 1.84GB 를 읽는다 (canus HDD 약 9초)."""
    ref = REF_SHA256.get(cache_key(T, size))
    if ref is None:
        raise RuntimeError(f'[DVSLIT] 키 {cache_key(T, size)} 의 기준 sha256 이 없다 (REF_SHA256)')
    bad = [n for n, h in sorted(ref.items()) if sha256_file(os.path.join(d, n)) != h]
    if bad:
        raise RuntimeError(f'[DVSLIT] 캐시 파일 {bad} 가 canus 기준 캐시와 비트가 다르다 (다른 TF·서버에서 다시 빌드한 '
                           f'캐시로 보인다). canus 캐시를 복사해 쓴다: {d}')


def check_cache(d, T=T_DEFAULT, size=SIZE_DEFAULT, allow_partial=False, verify_sha=False):
    """캐시 디렉토리 d 를 확인하고 manifest 를 돌려준다. 어긋나면 RuntimeError.
    verify_sha=False 면 배열은 헤더만 읽는다. True 면 전체 캐시 파일의 sha256 을 canus 기준과 대조한다
    (부분 캐시는 기준이 없어 건너뛴다 -- 부분 캐시는 env 로 허락한 검증에서만 쓰인다)."""
    mp = os.path.join(d, 'manifest.json')
    if not os.path.exists(mp):
        raise RuntimeError(f'[DVSLIT] 캐시 manifest 없음: {mp} (build_dvslit_cache.py 로 먼저 만든다)')
    with open(mp) as f:
        man = json.load(f)
    for k, v in expected_fields(T, size).items():
        if man.get(k) != v:
            raise RuntimeError(f'[DVSLIT] manifest {k}={man.get(k)!r} (기대 {v!r}) -- 다른 설정으로 만든 캐시: {d}')
    partial = man.get('limit') is not None
    if partial and not allow_partial:
        raise RuntimeError(f'[DVSLIT] 부분 캐시(limit={man.get("limit")})는 검증용이다. '
                           f'실제 런에 쓰지 않는다: {d}')
    for split, n_full in (('train', N_TRAIN_FULL), ('test', N_TEST_FULL)):
        n = man[f'n_{split}']
        if not partial and n != n_full:
            raise RuntimeError(f'[DVSLIT] {split} 표본 {n} (기대 {n_full})')
        xd, xf, xs = npy_header(os.path.join(d, f'{split}_x.npy'))
        _, _, ys = npy_header(os.path.join(d, f'{split}_y.npy'))
        if xs != (n, T, size, size, 2) or xd != '<f4' or xf:
            raise RuntimeError(f'[DVSLIT] {split}_x 모양 {xs} {xd} (기대 {(n, T, size, size, 2)} <f4)')
        if ys != (n,):
            raise RuntimeError(f'[DVSLIT] {split}_y 모양 {ys} (기대 {(n,)})')
    if verify_sha and not partial:
        verify_ref_sha(d, T, size)
    return man


# --- 분할 ---------------------------------------------------------------------------------------
def lex_rank():
    """id(0..999) -> 클래스 안 사전순 순위. 순위 < 900 이면 학습, 아니면 시험."""
    import numpy as np
    order = sorted(range(PER_CLASS), key=str)
    rank = np.empty(PER_CLASS, np.int64)
    rank[order] = np.arange(PER_CLASS)
    return rank


def test_ids():
    import numpy as np
    r = lex_rank()
    return sorted(int(i) for i in np.nonzero(r >= N_TRAIN_PER_CLASS)[0])


# --- 프레임화 (numpy, tfds 이벤트 -> 128x128 개수) -----------------------------------------------
def events_to_frames(coords, polarity, T=T_DEFAULT):
    """tfds cifar10_dvs 이벤트 -> [T,128,128,2] float32 개수.
    spikingjelly cal_fixed_frames_number_segment_index(split_by='number') +
    integrate_events_segment_to_frame 와 같은 결과 (좌표·채널은 모듈 머리 주석)."""
    import numpy as np
    coords = np.asarray(coords)
    N = coords.shape[0]
    di = N // T
    if di > 0:
        idx = np.minimum(np.arange(N, dtype=np.int64) // di, T - 1)
    else:                       # N < T: sj 는 j_l=j_r=0 (빈 조각), 마지막 조각이 [0, N)
        idx = np.full(N, T - 1, np.int64)
    h = coords[:, 0].astype(np.int64)
    w = (SRC_HW - 1) - coords[:, 1].astype(np.int64)
    c = np.asarray(polarity).astype(np.int64)
    flat = ((idx * SRC_HW + h) * SRC_HW + w) * 2 + c
    cnt = np.bincount(flat, minlength=T * SRC_HW * SRC_HW * 2)
    return cnt.reshape(T, SRC_HW, SRC_HW, 2).astype(np.float32)


def segment_counts(N, T=T_DEFAULT):
    """조각별 이벤트 수 (검증용)."""
    import numpy as np
    di = N // T
    c = np.full(T, di, np.int64)
    c[-1] = N - di * (T - 1)
    return c


def resize_frames(fr, size=SIZE_DEFAULT):
    """[..., T,128,128,2] -> [..., T,size,size,2]. bicubic + antialias (TF)."""
    import numpy as np
    import tensorflow as tf
    fr = np.asarray(fr, np.float32)
    lead = fr.shape[:-3]
    x = fr.reshape((-1,) + fr.shape[-3:])
    y = tf.image.resize(x, (size, size), method=RESIZE_METHOD, antialias=ANTIALIAS).numpy()
    return y.reshape(lead + (size, size, 2)).astype(np.float32)


# --- 증강 (TF, 클립 단위) ------------------------------------------------------------------------
# 파라미터 뽑기와 적용을 나눴다. 적용은 결정적이라 검증에서 같은 파라미터로 비교할 수 있다.
def sample_aug_params(size=SIZE_DEFAULT):
    import tensorflow as tf
    return {
        'flip': tf.random.uniform([]) < FLIP_P,                                  # random.random() > 0.5
        'op': tf.random.uniform([], 0, 3, dtype=tf.int32),                       # 0 roll, 1 rotate, 2 cutout
        'dy': tf.random.uniform([], -ROLL_MAX, ROLL_MAX + 1, dtype=tf.int32),    # random.randint(-5, 5)
        'dx': tf.random.uniform([], -ROLL_MAX, ROLL_MAX + 1, dtype=tf.int32),
        'theta': tf.random.uniform([], -ROT_DEG, ROT_DEG) * (math.pi / 180.0),   # RandomRotation(30)
        'cy': tf.random.uniform([], 0, size, dtype=tf.int32),                    # np.random.randint(h)
        'cx': tf.random.uniform([], 0, size, dtype=tf.int32),
    }


def rotate_clip(x, theta):
    """x [T,H,W,C] 를 영상 중심((W-1)/2,(H-1)/2) 기준으로 theta(rad) 회전, nearest, 바깥 0.
    출력 (col,row) 가 입력 (cos*col - sin*row + ox, sin*col + cos*row + oy) 를 읽는다.
    torchvision F.rotate(angle=theta 도) 의 역행렬 [cos,-sin; sin,cos] 와 같은 방향이다."""
    import tensorflow as tf
    T, H, W, C = x.shape
    img = tf.reshape(tf.transpose(x, [1, 2, 0, 3]), [1, H, W, T * C])
    c, s = tf.cos(theta), tf.sin(theta)
    ox = ((W - 1) - (c * (W - 1) - s * (H - 1))) / 2.0
    oy = ((H - 1) - (s * (W - 1) + c * (H - 1))) / 2.0
    tr = tf.reshape(tf.stack([c, -s, ox, s, c, oy, 0.0, 0.0]), [1, 8])
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img, transforms=tf.cast(tr, tf.float32), output_shape=tf.constant([H, W], tf.int32),
        fill_value=tf.constant(0.0, tf.float32), interpolation='NEAREST', fill_mode='CONSTANT')
    return tf.transpose(tf.reshape(out, [H, W, T, C]), [2, 0, 1, 3])


def cutout_clip(x, cy, cx):
    """UniSparse Cutout: 중심 (cy,cx), 반변 8, 경계에서 자른 [y1,y2)x[x1,x2) 를 0 으로. 모든 T·C 에 같은 마스크."""
    import tensorflow as tf
    T, H, W, C = x.shape
    half = CUTOUT_LEN // 2
    y1 = tf.clip_by_value(cy - half, 0, H)
    y2 = tf.clip_by_value(cy + half, 0, H)
    x1 = tf.clip_by_value(cx - half, 0, W)
    x2 = tf.clip_by_value(cx + half, 0, W)
    r = tf.range(H)
    q = tf.range(W)
    in_y = tf.logical_and(r >= y1, r < y2)
    in_x = tf.logical_and(q >= x1, q < x2)
    hole = tf.logical_and(in_y[:, None], in_x[None, :])
    mask = tf.cast(tf.logical_not(hole), x.dtype)
    return x * mask[None, :, :, None]


def apply_aug(x, p):
    """x [T,H,W,C] (정적 모양), p = sample_aug_params(). 뒤집기 후 op 하나."""
    import tensorflow as tf
    x = tf.cond(p['flip'], lambda: tf.reverse(x, [2]), lambda: x)
    return tf.switch_case(p['op'], [
        lambda: tf.roll(x, shift=tf.stack([p['dy'], p['dx']]), axis=[1, 2]),
        lambda: rotate_clip(x, p['theta']),
        lambda: cutout_clip(x, p['cy'], p['cx']),
    ])


def augment(x):
    return apply_aug(x, sample_aug_params(x.shape[1]))


# --- tf.data 파이프라인 --------------------------------------------------------------------------
def make_datasets(d, T, size, batch_size, allow_partial=False, augment_train=True, verify_sha=False):
    """캐시 d -> (train_ds, test_ds, n_train, n_test, manifest). load() 와 검증이 같이 쓴다."""
    import numpy as np
    import tensorflow as tf
    man = check_cache(d, T, size, allow_partial, verify_sha)
    arrs = {s: (np.load(os.path.join(d, f'{s}_x.npy'), mmap_mode='r'),
                np.load(os.path.join(d, f'{s}_y.npy')).astype(np.int32)) for s in ('train', 'test')}

    def getter(split):
        xs, ys = arrs[split]

        def _np_get(i):
            i = int(i)
            return np.array(xs[i], dtype=np.float32), np.int32(ys[i])

        def _get(i):
            x, y = tf.numpy_function(_np_get, [i], [tf.float32, tf.int32])
            x.set_shape([T, size, size, 2])
            y.set_shape([])
            return x, y
        return _get

    def _post(train):
        def f(x, y):
            x = x * INPUT_SCALE
            if train and augment_train:
                x = augment(x)
            return x, tf.one_hot(y, N_CLASS)
        return f

    AUTO = tf.data.AUTOTUNE
    n_tr, n_te = len(arrs['train'][1]), len(arrs['test'][1])
    train_ds = (tf.data.Dataset.range(n_tr)
                .shuffle(n_tr, reshuffle_each_iteration=True)        # 표본 단위 전체 셔플 (RandomSampler)
                .map(getter('train'), num_parallel_calls=AUTO)
                .map(_post(True), num_parallel_calls=AUTO)
                .batch(batch_size, drop_remainder=True)
                .prefetch(AUTO))
    test_ds = (tf.data.Dataset.range(n_te)
               .map(getter('test'), num_parallel_calls=AUTO)
               .map(_post(False), num_parallel_calls=AUTO)
               .batch(batch_size, drop_remainder=True)
               .prefetch(AUTO))
    return train_ds, test_ds, n_tr, n_te, man


def load():
    """datasets/cifar10_dvs.load() 와 같은 6-튜플. valid 와 test 는 같은 시험 1000장이다
    (best epoch on test, 문헌 관례 -- 머리 주석). 전체 캐시는 sha256 을 canus 기준과 대조한 뒤 쓴다."""
    from config import config
    conf = config.flags

    T = conf.time_step
    size = conf.cifar10_dvs_img_size
    B = config.batch_size
    if conf.dataset != 'CIFAR10_DVS':
        raise ValueError(f'[DVSLIT] dataset={conf.dataset!r} (CIFAR10_DVS 전용)')
    # 문헌 BN 은 T*B 묶음이다 (데이터 프로토콜과 함께 맞추는 항목). 평균 BN 이면 학습 시 BN 통계가 시간 평균 입력에서 나온다.
    if not conf.bn_per_step:
        raise ValueError('[DVSLIT] bn_per_step=True 가 필요하다 (run_paper.py COMBO_FLAGS[vggdvslit])')
    allow_partial = os.environ.get(ENV_ALLOW_PARTIAL) == '1'
    d = cache_dir(T, size)

    conf.time_dim_size = T
    conf.input_data_time_dim = True

    train_ds, test_ds, n_tr, n_te, man = make_datasets(d, T, size, B, allow_partial, verify_sha=True)
    train_ds_num = (n_tr // B) * B
    valid_ds_num = (n_te // B) * B
    print(f'[DVSLIT] cache={d} key={man["key"]} train={n_tr} test={n_te} batch={B} '
          f'-> steps {n_tr // B} / {n_te // B}, 사용 {train_ds_num} / {valid_ds_num}장, '
          f'input x{INPUT_SCALE}, clip aug (flip {FLIP_P} + roll {ROLL_MAX} / rot {ROT_DEG:g} / cutout {CUTOUT_LEN})'
          + (f'  ** 부분 캐시 limit={man["limit"]} (검증용, sha 대조 안 함) **' if man.get('limit') is not None
             else ', sha256 = canus 기준'),
          flush=True)
    print('[DVSLIT] 범위: 문헌과 같은 것은 데이터 프로토콜뿐이고 모델·학습 레시피는 우리 것이다 (8축 차이: 뉴런, 출력층, '
          '옵티마이저·워밍업, label smoothing, 초기화, BN momentum·eps, fp32, conv bias). 문헌 수치와 비교하지 않는다. valid = test 1000장 '
          '(best epoch on test, 문헌 관례) -- 정확도는 최고 에폭·마지막 에폭 둘 다 보고하고, 기존 T=4 조합과 같은 표에 섞지 않는다.', flush=True)
    return train_ds, test_ds, test_ds, train_ds_num, valid_ds_num, valid_ds_num
