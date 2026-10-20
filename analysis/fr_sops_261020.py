"""발화율 (FR)·SOPs 계산과 문헌 비교표 (26-10-20, eip-stats).

요청: docs/paper/compare-candidates-261020.md 1.1 의 '학습 없는 분석' 중 문헌 비교표.
사용자 승인 10-20 ("학습 없는 분석 3건"). GPU 를 쓰지 않는다 (기존 npz 만 읽음, TF 는 CPU 로 fan-out 검산만).

질문
  Q1. R19 C10 T=4 대표점 (baseline, ours_inter_ch 3e-3, BPSR 1e-10, plain L2 1e-7) 과
      342K ablation 세트의 평균 발화율 (FR) 과 SOPs (이미지당) 는 얼마인가.
  Q2. 같은 스파이크에서 SOPs 가 팔마다 다른가 (ln SOPs ~ ln 스파이크 + 팔 + 서버).
  Q3. 문헌 (AP, UPR, UniSparse, SCA, STDS, ESL-SNN, SEENN, DT-SNN, Sorbaro, Spik4lite, ...) 과 한 표에.
      SOPs 축에서 우리가 약해 보이는 곳.

정의
  스파이킹 층: n_in (아날로그 직접 입력) 과 predictions_n (출력 적분기, 막전위를 냄) 을 뺀 18층,
      뉴런 N = 7·32·32·128 + 6·16·16·256 + 4·8·8·512 + 256 = 1,442,048.
  S   = 이미지당 스파이크 (T=4 스텝 합, npz 테스트 1만 장).
  FR_step = S / (N·T)  (뉴런당 스텝당 스파이크, Activity Pruning 의 'Avg FR' 과 같은 꼴)
  FR_T    = S / N      (뉴런당 T 동안 스파이크)
  SOPs = Σ_층 Σ_위치 spikes(위치) × fan-out(위치)   (이미지당, T 합)
      fan-out = 그 뉴런의 스파이크 하나가 다음 가중치 층에서 일으키는 누산 수 (경계 패딩 반영).
      규약 (주): BN 은 앞 conv 에 접힘 (0), identity shortcut 의 Add 는 0, 단계 입구 1×1 shortcut 은 셈,
      GAP 는 fc1 에 접힘 (conv4_block2_out_n 스파이크 하나 → fc1 256 누산), fc1_n → predictions 10,
      predictions_n → 0. 이진 스파이크가 일으키는 연산이라 모두 AC (누산) 다.
  첫 층: 입력이 아날로그 직접 입력 (input_spike_mode=REAL, 매 스텝 같은 값; npz n_in__t 네 스텝 합이 같음)
      → conv1_conv 는 MAC. 밀집 32·32·128·(3·3·3) = 3,538,944 MAC/스텝, 매 스텝 다시 계산하면 ×T = 14,155,776.
      입력이 스텝마다 같으므로 한 번만 계산하면 3,538,944.
  fan-out 출처: models/resnet.py (ResNet19: 128채널, 블록 [3,3,2], block_basic, Add 뒤 발화) 를 읽어 만든 사양을
      (1) analysis/r19_fanout.py 의 FANOUT 과, (2) 실제 Keras 그래프에서 뽑은 가중치 층 목록
      (scratchpad/prune_out/*.json 'weight_layers', analysis/prune_measure.py) 과, (3) TF CPU conv 기울기
      (d Σconv(x, 1) / dx = 그 입력을 읽는 출력 수) 와 대조한다.

입력 (읽기만)
  원장 스냅숏 scratchpad/an1019/snapshot/inventory.json (10-19 16:38): 런 선택·n·정확도·s_count.
  npz: scratchpad/an1019/npz_abl, scratchpad/an1019/npz, _npz_t, _npz, scratchpad/deep1005/npz
      (같은 태그가 여럿이면 스텝별 배열(__t) 이 있는 판을 쓰고, 총합이 같은지 확인한다).

쓰는 법:  CUDA_VISIBLE_DEVICES= ~/anaconda3/envs/venv_1/bin/python analysis/fr_sops_261020.py
산출:     analysis/fr_sops_261020.json, analysis/fr_sops_261020.md

검토 반영 (10-20 eip-critic 1차, 14건)
  - UniSparse 행: 부록 Table 4 (scratchpad/us/us.txt) 의 밀집 행을 넣음. FR·SOPs 정의는 공개 코드
    (scratchpad/repos/xyshi2000_UniSparse) 로 확인.
  - 추론만 일찍 끊기 (학습 T=4, 추론 T=t): logits_t[:, t] 는 누적 출력 막전위 (logits_t[:, 3] == logits).
    스텝별 스파이크 배열 (__t) 은 샘플 합이므로 '모든 샘플을 같은 스텝에서 끊는' 경우만 정확하다.
    → [10-20 정정] 틀린 전제였다. R19 의 BN 은 T 스텝 입력의 시간 평균에 걸려 스텝 t 판독은 'T=t 에서 멈춘 망' 이 아니다
      (analysis/early_exit_261020.md 0·6절). 판독값은 JSON 'step_readout_noncausal' (인용 금지) 로만 남기고,
      3b 절은 early_exit_261020 의 CPU 원자료 (같은 체크포인트를 T' 로 다시 돌림, 테스트 앞 1000장) 를 다시 읽어
      계산한 실측 (JSON 'truncation_measured') 으로 바꿨다. 입력 추가: scratchpad/an1020/ee/{cpu,recal}/*.npz,
      analysis/early_exit_261020.json (대조용).
  - 같은 스파이크 SOPs 곡선을 npz 축과 s_count 축 둘 다로.
  - 동등성 통과 최심점 (ours_inter_ch 4e-3) 의 npz SOPs.
  - SOPs 규약 변형 D (층 FR × 패딩 포함 밀집 FLOPs), E (conv 만, UniSparse 코드 규약에 가까움).
  - 서버 민감도 (baseline s_count 23 대 138, npz/s_count 비로 n=6 외삽).
"""
import glob
import json
import math
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''          # GPU 를 쓰지 않는다
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
from scipy import stats

ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
S = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
A = S + '/an1019'
INV = A + '/snapshot/inventory.json'
NPZ_DIRS = [A + '/npz_abl', A + '/npz', ROOT + '/_npz_t', ROOT + '/_npz', S + '/deep1005/npz']
PRUNE_DIRS = [S + '/prune_out', S + '/deep1005/prune']
OUT_JSON = ROOT + '/analysis/fr_sops_261020.json'
OUT_MD = ROOT + '/analysis/fr_sops_261020.md'
SKIP = {'n_in', 'predictions_n'}
T = 4
E_AC, E_MAC = 0.9e-12, 4.6e-12      # J, 45 nm (Horowitz 2014) — 문헌에서 흔히 쓰는 값, 참고용

sys.path.insert(0, ROOT + '/analysis')
from r19_fanout import FANOUT as FANOUT_OLD, reads_per_index, conv_map   # noqa: E402


# ---------------------------------------------------------------- 1. 구조 사양 (models/resnet.py 에서)
def r19_spec():
    """ResNet19 (CIFAR, 32×32): 가중치 층 (이름, k, stride, padding, C_in, C_out, src 뉴런층, dst 뉴런층, H_in).
    models/resnet.py: conv1_conv 3×3 s1 same 128 → BN → conv1_conv_n;
    stack1(conv2, stride=1), stack1(conv3, stride=2), stack1(conv4, stride=2), 블록 [3,3,2];
    block_basic: block1 만 conv0 (1×1, stride, lib_snn Conv2D 기본 padding='valid') shortcut, 나머지 Identity;
    conv1 3×3 stride same → BN → conv1_n → conv2 3×3 same → BN → Add(shortcut) → out_n (sew=False);
    GAP → fc1 Dense(256) → fc1_n → predictions Dense(10) → predictions_n (loc OUT)."""
    W = [('conv1_conv', 3, 1, 'same', 3, 128, 'n_in', 'conv1_conv_n', 32)]
    prev, C_prev, H_prev = 'conv1_conv_n', 128, 32
    ident = []
    for stg, nblk, C, stride in [('conv2', 3, 128, 1), ('conv3', 3, 256, 2), ('conv4', 2, 512, 2)]:
        for b in range(1, nblk + 1):
            s = stride if b == 1 else 1
            H = H_prev // s
            n1, no = f'{stg}_block{b}_conv1_n', f'{stg}_block{b}_out_n'
            W.append((f'{stg}_block{b}_conv1', 3, s, 'same', C_prev, C, prev, n1, H_prev))
            if b == 1:
                W.append((f'{stg}_block{b}_conv0', 1, s, 'valid', C_prev, C, prev, no, H_prev))
            else:
                ident.append((prev, no))
            W.append((f'{stg}_block{b}_conv2', 3, 1, 'same', C, C, n1, no, H))
            prev, C_prev, H_prev = no, C, H
    W.append(('fc1', 'gap', None, None, 512, 256, prev, 'fc1_n', H_prev))
    W.append(('predictions', 'dense', None, None, 256, 10, 'fc1_n', 'predictions_n', 1))
    return W, ident


SPEC, IDENT = r19_spec()


def build_fanout(spec):
    """뉴런층 → 위치별 fan-out 지도 (공간 [H,W] 또는 [units]). 채널 축은 C_out 곱으로 들어간다."""
    M = {}
    for name, k, s, pad, cin, cout, src, dst, H in spec:
        if k == 'gap':
            m = np.full((H, H), float(cout))           # GAP 를 fc1 에 접음: 스파이크 하나 → fc1 행 하나 (cout 누산)
        elif k == 'dense':
            m = np.full((cin,), float(cout))
        else:
            m = conv_map(H, H, k, s, pad, cout)
        M[src] = M[src] + m if src in M else m
    M['predictions_n'] = np.zeros((10,))
    return M


FANOUT = build_fanout(SPEC)


def build_fanout_flops(spec):
    """변형 D (spikingjelly/syops 식): 층 FR × 패딩 포함 밀집 FLOPs.
    입력 위치마다 같은 fan-out = k²·C_out/s² (경계 보정 없음). GAP·dense 는 주 규약과 같다."""
    M = {}
    for name, k, s, pad, cin, cout, src, dst, H in spec:
        if k == 'gap':
            m = np.full((H, H), float(cout))
        elif k == 'dense':
            m = np.full((cin,), float(cout))
        else:
            m = np.full((H, H), k * k * cout / (s * s))
        M[src] = M[src] + m if src in M else m
    M['predictions_n'] = np.zeros((10,))
    return M


FANOUT_D = build_fanout_flops(SPEC)
# 스파이킹 18층 (dst 순서, 중복 제거 — 단계 입구 out_n 은 conv0·conv2 두 가중치 층의 dst 다)
NEURON_LAYERS = list(dict.fromkeys(w[7] for w in SPEC if w[7] not in SKIP))
assert len(NEURON_LAYERS) == 18, NEURON_LAYERS
LAYER_SHAPE = {}
for name, k, s, pad, cin, cout, src, dst, H in SPEC:
    if dst in SKIP:
        continue
    if k in ('gap', 'dense'):
        LAYER_SHAPE[dst] = (cout,)
    else:
        LAYER_SHAPE[dst] = (H // s, H // s, cout)
N_NEURON = int(sum(np.prod(v) for v in LAYER_SHAPE.values()))
CONV1_MAC_STEP = 32 * 32 * 128 * 3 * 3 * 3            # 패딩 포함 (thop 식 밀집 MAC, UniSparse·AP 코드의 MACs 와 같은 규약)
CONV1_MAC_STEP_NOPAD = 94 * 94 * 3 * 128               # 패딩 0 을 뺀 값 (SOPs 와 같은 규약), −4.1%
# UniSparse SEW-ResNet19 의 FR 감시 뉴런 수 (SparseLIF 만; models/resnet.py sew_resnet19, submodules/blocks.py):
# prologue 32·32·128 + 단계1 3블록×2 (32·32·128) + 단계2 (3×2 + downsample 1)·16·16·256 + 단계3 (2×2 + 1)·8·8·512.
# epilogue 의 LIF 는 SparseNeuron 이 아니라 FR 에 들어가지 않는다.
US_N = 32 * 32 * 128 + 6 * 32 * 32 * 128 + 7 * 16 * 16 * 256 + 5 * 8 * 8 * 512


def check_fanout():
    out = {'n_neuron': N_NEURON}
    # (1) r19_fanout.py 와 같은가
    same = sorted(FANOUT) == sorted(FANOUT_OLD) and all(np.array_equal(FANOUT[k], FANOUT_OLD[k]) for k in FANOUT)
    out['equals_r19_fanout_py'] = bool(same)
    # (2) 실제 Keras 그래프의 가중치 층 목록과 같은가 (이름·커널·src·dst·출력 위치 수)
    wl = None
    for d in PRUNE_DIRS:
        for p in sorted(glob.glob(d + '/r19c10-*.json')):
            j = json.load(open(p))
            if 'weight_layers' in j:
                wl, wl_src = j['weight_layers'], p
                break
        if wl:
            break
    if wl:
        mine = {}
        for name, k, s, pad, cin, cout, src, dst, H in SPEC:
            if k == 'gap':
                ks, hw = [512, 256], 1
            elif k == 'dense':
                ks, hw = [cin, cout], 1
            else:
                ks, hw = [k, k, cin, cout], (H // s) ** 2
            mine[name] = (ks, hw, src, dst)
        theirs = {w['name']: (list(w['ks']), w['hw'], w['src'], w['dst']) for w in wl}
        out['keras_graph_file'] = wl_src
        out['keras_graph_match'] = bool(mine == theirs)
        out['keras_graph_diff'] = sorted(set(mine.items()) ^ set(theirs.items())) if mine != theirs else []
    # (3) TF CPU 로 conv 기울기 검산: d Σ conv(x, 1) / dx[i,j] = 그 입력을 읽는 (출력 위치) 수
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        res = []
        for H, k, s, pad in [(32, 3, 1, 'SAME'), (32, 3, 2, 'SAME'), (32, 1, 2, 'VALID'), (16, 3, 2, 'SAME'),
                             (16, 3, 1, 'SAME'), (8, 3, 1, 'SAME')]:
            x = tf.Variable(tf.zeros((1, H, H, 1)))
            with tf.GradientTape() as g:
                y = tf.reduce_sum(tf.nn.conv2d(x, tf.ones((k, k, 1, 1)), strides=s, padding=pad))
            tfmap = g.gradient(y, x).numpy()[0, :, :, 0]
            mymap = np.outer(reads_per_index(H, k, s, pad.lower()), reads_per_index(H, k, s, pad.lower()))
            res.append({'H': H, 'k': k, 's': s, 'pad': pad, 'max_abs_diff': float(np.abs(tfmap - mymap).max()),
                        'mean_reads': float(mymap.mean())})
        out['tf_grad_check'] = res
    except Exception as e:  # TF 가 없어도 나머지는 돈다
        out['tf_grad_check'] = f'skipped: {e!r}'
    # (4) 손 계산: conv2_block1_conv1_n → conv2_block1_conv2 (3×3 s1 same, 128)
    m = FANOUT['conv2_block1_conv1_n']
    out['hand_conv2_block1_conv1_n'] = {
        'interior_expected': 9 * 128, 'interior': float(m[5, 5]),
        'edge_expected': 6 * 128, 'edge': float(m[0, 5]),
        'corner_expected': 4 * 128, 'corner': float(m[0, 0]),
        'mean_expected': (94 / 32) ** 2 * 128, 'mean': float(m.mean())}
    # (5) 손 계산: conv2_block3_out_n → conv3_block1_conv1 (3×3 s2 same, 256) + conv3_block1_conv0 (1×1 s2 valid, 256)
    m = FANOUT['conv2_block3_out_n']
    out['hand_conv2_block3_out_n'] = {
        'even_even_interior_expected': 2 * 2 * 256 + 256, 'even_even_interior': float(m[4, 4]),
        'odd_odd_expected': 1 * 1 * 256, 'odd_odd': float(m[5, 5]),
        'even_odd_expected': 2 * 1 * 256, 'even_odd': float(m[4, 5]),
        'mean_expected': (47 / 32) ** 2 * 256 + 0.25 * 256, 'mean': float(m.mean())}
    # (6) 합 검산: Σ_입력 fan-out = 유효 입력을 읽는 밀집 MAC 수 (conv2_block1_conv2, 패딩 0 제외)
    out['sum_check_conv2_block1_conv1_n'] = {'sum_fanout_x_Cin': float(FANOUT['conv2_block1_conv1_n'].sum() * 128),
                                             'valid_macs': 94 * 94 * 128 * 128,
                                             'dense_macs_with_padding': 32 * 32 * 128 * 9 * 128}
    # 층 평균 fan-out (참고)
    out['mean_fanout'] = {k: float(v.mean()) for k, v in FANOUT.items()}
    # 밀집 동등 (모든 뉴런이 매 스텝 쏠 때 SOPs/스텝)
    out['dense_equiv_sops_per_step'] = float(sum(FANOUT[l].sum() * (LAYER_SHAPE[l][-1] if len(LAYER_SHAPE[l]) == 3 else 1)
                                                 for l in NEURON_LAYERS))
    return out


# ---------------------------------------------------------------- 2. 원장·npz
LED = {r['tag']: r for r in json.load(open(INV))['runs']}
INV_GEN = json.load(open(INV))['generated']


def npz_files(tag):
    return [os.path.join(d, tag + '.npz') for d in NPZ_DIRS if os.path.exists(os.path.join(d, tag + '.npz'))]


def npz_metrics(p):
    z = np.load(p)
    ns = float(z['n_samples'])
    assert int(z['T']) == T
    per_layer, sops, sops_id, gap_sp, sops_d = {}, 0.0, 0.0, 0.0, 0.0
    for l in NEURON_LAYERS:
        c = z[l].astype(np.float64)
        assert c.shape == LAYER_SHAPE[l], (l, c.shape)
        per_layer[l] = c.sum() / ns
        sp = c.sum(-1) if c.ndim == 3 else c
        sops += float((sp * FANOUT[l]).sum())
        sops_d += float((sp * FANOUT_D[l]).sum())
        if l == 'conv4_block2_out_n':
            gap_sp = c.sum()
    fc1_sp = float(z['fc1_n'].astype(np.float64).sum())
    # 변형 B: identity shortcut 의 Add 를 스파이크당 1 AC 로 셈
    id_add = sum(z[src].astype(np.float64).sum() for src, _ in IDENT)
    spk = sum(per_layer.values())
    m = {'file': p, 'n_samples': ns, 'npz_acc': float(z['acc']), 'npz_spk': spk, 'fr_step': spk / (N_NEURON * T), 'fr_T': spk / N_NEURON,
         'fr_step_layer_mean': float(np.mean([per_layer[l] / (np.prod(LAYER_SHAPE[l]) * T) for l in NEURON_LAYERS])),
         'sops': sops / ns, 'sops_per_spk': sops / ns / spk,
         'sops_varB_identity': (sops + id_add) / ns,
         # 변형 C: GAP 를 접지 않음 — 스파이크당 1 AC (GAP 누산) + fc1 밀집 512·256 MAC/스텝 (아날로그 입력)
         'sops_varC_gap_acs': (sops - 256.0 * gap_sp + gap_sp) / ns, 'varC_fc1_macs': 512 * 256 * T,
         # 변형 D: 층 FR × 패딩 포함 밀집 FLOPs (spikingjelly/syops 식)
         'sops_varD_flops': sops_d / ns,
         # 변형 E: conv 만 셈 (GAP→fc1, fc1→predictions 를 뺌). UniSparse SOPMonitor 는 conv 만 감시하고 첫 conv 를 뺀다
         'sops_varE_conv_only': (sops - 256.0 * gap_sp - 10.0 * fc1_sp) / ns,
         'stage_share': {}, 'has_t': any(k.endswith('__t') for k in z.files)}
    if 'n_in__t' in z.files:          # 직접 입력이 네 스텝 모두 원소 단위로 같은가
        a = z['n_in__t']
        m['nin_elementwise_equal'] = bool(all(np.array_equal(a[0], a[t]) for t in range(1, T)))
    if 'logits_t' in z.files and 'labels' in z.files:
        lt, lab = z['logits_t'], z['labels']
        # logits_t[:, t] 는 t 스텝까지 누적된 출력 막전위다 (logits_t[:, T-1] == logits). 스텝끼리 더하면 이중으로 센다.
        m['logits_last_eq'] = float(np.abs(lt[:, T - 1] - z['logits']).max())
        m['acc_t'] = [100 * float((lt[:, t].argmax(1) == lab).mean()) for t in range(T)]
        m['acc_t_last_vs_npz'] = m['acc_t'][T - 1] - float(z['acc'])
    for stg in ('conv1', 'conv2', 'conv3', 'conv4', 'fc1'):
        ls = [l for l in NEURON_LAYERS if l.startswith(stg)]
        m['stage_share'][stg] = 100 * sum(per_layer[l] for l in ls) / spk
    m['sops_stage_share'] = {}
    for stg in ('conv1', 'conv2', 'conv3', 'conv4', 'fc1'):
        ls = [l for l in NEURON_LAYERS if l.startswith(stg)]
        v = 0.0
        for l in ls:
            c = z[l].astype(np.float64)
            sp = c.sum(-1) if c.ndim == 3 else c
            v += float((sp * FANOUT[l]).sum())
        m['sops_stage_share'][stg] = 100 * v / sops
    if m['has_t']:
        st_spk = np.zeros(T); st_sops = np.zeros(T)
        for l in NEURON_LAYERS:
            ct = z[l + '__t'].astype(np.float64)
            for t in range(T):
                c = ct[t]
                sp = c.sum(-1) if c.ndim == 3 else c
                st_spk[t] += c.sum(); st_sops[t] += float((sp * FANOUT[l]).sum())
        m['step_spk_share'] = list(100 * st_spk / st_spk.sum())
        m['step_fr'] = list(st_spk / ns / N_NEURON)
        m['step_sops_share'] = list(100 * st_sops / st_sops.sum())
        m['step_spk'] = list(st_spk / ns)           # 이미지당, 스텝별
        m['step_sops'] = list(st_sops / ns)
        m['step_max_count'] = max(float(z[l + '__t'].max()) for l in NEURON_LAYERS)   # ≤ n_samples 면 이진 스파이크
        m['t_sum_check'] = float(abs(st_spk.sum() / ns - spk) / spk)
    # 에너지 (참고): AC × SOPs + MAC × 첫 층 (스텝마다 다시 계산)
    m['energy_uJ'] = 1e6 * (E_AC * m['sops'] + E_MAC * CONV1_MAC_STEP * T)
    m['energy_uJ_conv1_once'] = 1e6 * (E_AC * m['sops'] + E_MAC * CONV1_MAC_STEP)
    m['conv1_mac_energy_share'] = 100 * E_MAC * CONV1_MAC_STEP * T / (E_AC * m['sops'] + E_MAC * CONV1_MAC_STEP * T)
    return m


def pick_npz(tag):
    fs = npz_files(tag)
    if not fs:
        return None, []
    ms = [(f, npz_metrics(f)) for f in fs]
    totals = [round(m['npz_spk'], 3) for _, m in ms]
    chosen = next(((f, m) for f, m in ms if m['has_t']), ms[0])
    dup = [{'file': f, 'npz_spk': m['npz_spk'], 'has_t': m['has_t']} for f, m in ms]
    assert len(set(totals)) == 1, (tag, totals)            # 같은 체크포인트면 총합이 같아야 한다
    return chosen[1], dup


def valid(r):
    return bool(r and r.get('completed') and not r.get('exclude') and r.get('seed', 999) < 900 and r.get('epochs') == 310)


# ---------------------------------------------------------------- 3. 묶음
GROUPS = [  # (표시 이름, 원장 key, knob, 묶음 종류)
    ('baseline', 'base', '-', 'T1'),
    ('ours_inter_ch 3e-3', 'ours_ch', '3e-3', 'T1'),
    ('BPSR 1e-10', 'bpsr', '1e-10', 'T1'),
    ('plain L2 1e-7', 'l2', '1e-7', 'T1'),
    # ±0.2%p 동등성 (TOST) 을 통과한 가장 많이 줄인 n≥3 점 (results_261019 요약 2). plain L2 1.5e-7 은 npz 가 없다.
    ('ours_inter_ch 4e-3', 'ours_ch', '4e-3', 'DEEP'),
    ('- vmem 2e-3', 'ch_novmem', '2e-3', 'ABL'),
    ('- final_step 1.4e-3', 'ch_nofinal', '1.4e-3', 'ABL'),
    ('- 1- inversion 4.5e-3', 'ch_noinv', '4.5e-3', 'ABL'),
    ('- loss-ratio 1.25e-7', 'ch_nolr', '1.25e-7', 'ABL'),
]
ROWS = []
for gname, key, knob, kind in GROUPS:
    rs = sorted([r for r in LED.values() if r.get('ds') == 'C10' and r.get('key') == key and r.get('knob') == knob
                 and r.get('status') not in ('hold',)], key=lambda r: r['seed'])
    for r in rs:
        if r.get('seed', 0) >= 900:
            continue          # 연기시험 (2에폭) — 표에 넣지 않음
        row = {'group': gname, 'kind': kind, 'tag': r['tag'], 'seed': r['seed'], 'server': r['server'],
               'train_loss': r.get('train_loss'), 'train_acc': r.get('train_acc'), 'val_loss': r.get('val_loss'),
               'val_acc': r.get('val_acc'), 'spikes': r.get('spikes'), 'valid': valid(r), 'reason': ''}
        if not row['valid']:
            row['reason'] = r.get('exclude_reason') or ('미완주' if not r.get('completed') else '배제')
        m, dup = pick_npz(r['tag'])
        if m is None:
            if row['valid']:
                row['reason'] = 'npz 없음' + (' (세종 런, 추출 안 함)' if r['server'] == '23' else ' (런 폴더 없음/추출 안 함)')
            row['npz'] = None
        else:
            row['npz'] = m
            row['npz_dups'] = dup
            row['acc_gap'] = m['npz_acc'] - r['val_acc']
        ROWS.append(row)


def msd(v):
    v = np.asarray(v, float)
    return [float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float('nan'), int(v.size)]


def welch(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    se = math.sqrt(va + vb); df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    d = float(a.mean() - b.mean()); t = stats.t.ppf(0.975, df)
    return {'d': d, 'ci95': [d - t * se, d + t * se], 'p': float(2 * stats.t.sf(abs(d / se), df)), 'df': float(df),
            'n': [int(a.size), int(b.size)]}


def ratio_pct(a, b):
    """100·(mean a / mean b − 1), 95% CI 는 ln 비의 델타법 + Welch 자유도 (두 묶음의 산포 모두 전파)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ma, mb = a.mean(), b.mean()
    va = a.var(ddof=1) / a.size / ma ** 2; vb = b.var(ddof=1) / b.size / mb ** 2
    se = math.sqrt(va + vb); df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    lr = math.log(ma / mb); t = stats.t.ppf(0.975, df)
    return {'pct': 100 * (ma / mb - 1), 'ci95': [100 * (math.exp(lr - t * se) - 1), 100 * (math.exp(lr + t * se) - 1)],
            'n': [int(a.size), int(b.size)], 'df': float(df)}


METRICS = ['npz_spk', 'fr_step', 'fr_T', 'fr_step_layer_mean', 'sops', 'sops_per_spk', 'sops_varB_identity',
           'sops_varC_gap_acs', 'sops_varD_flops', 'sops_varE_conv_only', 'energy_uJ', 'energy_uJ_conv1_once',
           'conv1_mac_energy_share']
G = {}
for gname, key, knob, kind in GROUPS:
    rr = [r for r in ROWS if r['group'] == gname and r['valid'] and r['npz']]
    allv = [r for r in ROWS if r['group'] == gname and r['valid']]
    g = {'kind': kind, 'n_valid_ledger': len(allv), 'n_npz': len(rr), 'tags_npz': [r['tag'] for r in rr],
         'ledger': {k: msd([r[k] for r in allv]) for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes')},
         'ledger_npz_subset': {k: msd([r[k] for r in rr]) for k in ('val_acc', 'spikes')},
         'npz': {k: msd([r['npz'][k] for r in rr]) for k in METRICS},
         'stage_share': {s: msd([r['npz']['stage_share'][s] for r in rr]) for s in ('conv1', 'conv2', 'conv3', 'conv4', 'fc1')},
         'sops_stage_share': {s: msd([r['npz']['sops_stage_share'][s] for r in rr]) for s in ('conv1', 'conv2', 'conv3', 'conv4', 'fc1')}}
    rt = [r for r in rr if r['npz']['has_t']]
    if rt:
        g['step'] = {'n': len(rt), 'spk_share': [msd([r['npz']['step_spk_share'][t] for r in rt]) for t in range(T)],
                     'fr': [msd([r['npz']['step_fr'][t] for r in rt]) for t in range(T)],
                     'sops_share': [msd([r['npz']['step_sops_share'][t] for r in rt]) for t in range(T)]}
    G[gname] = g

BASE = [r for r in ROWS if r['group'] == 'baseline' and r['valid'] and r['npz']]
BASE_ALL = [r for r in ROWS if r['group'] == 'baseline' and r['valid']]
VS_BASE = {}
for gname, *_ in GROUPS[1:]:
    rr = [r for r in ROWS if r['group'] == gname and r['valid'] and r['npz']]
    allv = [r for r in ROWS if r['group'] == gname and r['valid']]
    if len(rr) < 2:
        continue
    VS_BASE[gname] = {k: ratio_pct([r['npz'][k] for r in rr], [r['npz'][k] for r in BASE])
                      for k in ('npz_spk', 'fr_step', 'sops', 'sops_per_spk', 'sops_varD_flops', 'sops_varE_conv_only', 'energy_uJ',
                                'energy_uJ_conv1_once')}
    VS_BASE[gname]['scount_n6'] = ratio_pct([r['spikes'] for r in allv], [r['spikes'] for r in BASE_ALL])
    VS_BASE[gname]['scount_npzsubset'] = ratio_pct([r['spikes'] for r in rr], [r['spikes'] for r in BASE])
    VS_BASE[gname]['dacc_n6'] = welch([r['val_acc'] for r in allv], [r['val_acc'] for r in BASE_ALL])
    # SOPs 감소 / 스파이크 감소 (npz 축) — 1 보다 작으면 SOPs 가 덜 준다
    VS_BASE[gname]['sops_to_spk_reduction'] = VS_BASE[gname]['sops']['pct'] / VS_BASE[gname]['npz_spk']['pct']

# 팔끼리 (표1 점, n=4 대 4)
OURS = [r for r in ROWS if r['group'] == 'ours_inter_ch 3e-3' and r['valid'] and r['npz']]
PAIR = {}
for gname, *_ in GROUPS[2:]:
    rr = [r for r in ROWS if r['group'] == gname and r['valid'] and r['npz']]
    if len(rr) < 2:
        continue
    PAIR[gname] = {k: welch([r['npz'][k] for r in OURS], [r['npz'][k] for r in rr])
                   for k in ('npz_spk', 'fr_step', 'sops', 'sops_per_spk')}
    PAIR[gname]['sops_pct'] = ratio_pct([r['npz']['sops'] for r in OURS], [r['npz']['sops'] for r in rr])
    PAIR[gname]['sops_per_spk_pct'] = ratio_pct([r['npz']['sops_per_spk'] for r in OURS], [r['npz']['sops_per_spk'] for r in rr])


# ---------------------------------------------------------------- 4. 같은 스파이크 SOPs (곡선 ANCOVA)
def ols(y, X):
    y = np.asarray(y, float); X = np.asarray(X, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b; dof = len(y) - X.shape[1]
    s2 = float(res @ res / dof); cov = s2 * np.linalg.inv(X.T @ X)
    return b, np.sqrt(np.diag(cov)), dof


ARMKEY = {'ours_ch': 'ours_inter_ch', 'bpsr': 'BPSR', 'l2': 'plain L2'}
CURVE = []
for t, r in LED.items():
    if r.get('ds') != 'C10' or r.get('key') not in ARMKEY or not valid(r):
        continue
    fs = npz_files(t)
    if not fs:
        continue
    m = npz_metrics(fs[0])
    CURVE.append({'tag': t, 'arm': ARMKEY[r['key']], 'knob': r['knob'], 'server': r['server'], 'spikes': r['spikes'],
                  'npz_spk': m['npz_spk'], 'sops': m['sops'], 'sops_per_spk': m['sops_per_spk']})


def ancova_sops(lo, hi, other, cov='npz_spk', y_key='sops'):
    """ln SOPs = a + b·ln(cov) + c·[ours] + d·[서버 23]. cov = 'npz_spk' (npz 축) 또는 'spikes' (원장 s_count 축).
    런 선택은 두 축 모두 원장 s_count 구간으로 한다 (같은 런 집합)."""
    rr = [c for c in CURVE if lo <= c['spikes'] <= hi and c['arm'] in ('ours_inter_ch', other)]
    y = [math.log(c[y_key]) for c in rr]
    X = [[1.0, math.log(c[cov]), 1.0 if c['arm'] == 'ours_inter_ch' else 0.0, 1.0 if c['server'] == '23' else 0.0]
         for c in rr]
    b, se, dof = ols(y, X)
    t = stats.t.ppf(0.975, dof)
    c, sc = b[2], se[2]
    # 기울기 차 검정 (팔 × ln 스파이크)
    X2 = [x + [x[1] * x[2]] for x in X]
    b2, se2, dof2 = ols(y, X2)
    return {'n': len(rr), 'cov': cov, 'y': y_key, 'n_by_arm': {a: sum(1 for q in rr if q['arm'] == a) for a in ('ours_inter_ch', other)},
            'scount_range': {a: [min(q['spikes'] for q in rr if q['arm'] == a), max(q['spikes'] for q in rr if q['arm'] == a)]
                             for a in ('ours_inter_ch', other)},
            'slope': float(b[1]), 'slope_se': float(se[1]),
            'ours_minus_other_pct': 100 * (math.exp(c) - 1),
            'ci95_pct': [100 * (math.exp(c - t * sc) - 1), 100 * (math.exp(c + t * sc) - 1)],
            'p': float(2 * stats.t.sf(abs(c / sc), dof)), 'dof': int(dof),
            'slope_diff': float(b2[4]), 'slope_diff_p': float(2 * stats.t.sf(abs(b2[4] / se2[4]), dof2))}


SAME, SAME_SC = {}, {}
for lab, lo, hi in [('200_420K', 200e3, 420e3), ('300_420K', 300e3, 420e3)]:
    SAME[lab] = {o: ancova_sops(lo, hi, o) for o in ('BPSR', 'plain L2')}
    SAME_SC[lab] = {o: ancova_sops(lo, hi, o, cov='spikes') for o in ('BPSR', 'plain L2')}
# 같은 s_count 에서 npz 스파이크가 팔마다 다른가 (s_count 축 SOPs 차가 측정 축 차이에서 오는지 확인)
AXIS = {lab: {o: ancova_sops(lo, hi, o, cov='spikes', y_key='npz_spk') for o in ('BPSR', 'plain L2')}
        for lab, lo, hi in [('200_420K', 200e3, 420e3), ('300_420K', 300e3, 420e3)]}

# 같은 정확도 대신 '감소 깊이' 축: npz 스파이크 감소율과 SOPs 감소율의 관계 (모든 곡선 런, baseline npz n=3 기준)
BMEAN_SPK = float(np.mean([r['npz']['npz_spk'] for r in BASE])); BMEAN_SOPS = float(np.mean([r['npz']['sops'] for r in BASE]))
for c in CURVE:
    c['spk_red'] = 100 * (c['npz_spk'] / BMEAN_SPK - 1); c['sops_red'] = 100 * (c['sops'] / BMEAN_SOPS - 1)


# ---------------------------------------------------------------- 4b. 스텝 t 판독 (비인과, 인용 금지 — 10-20 정정)
# [10-20 정정] 검토 전 판은 여기를 '추론만 일찍 끊기' 로 읽었다 ("스텝 t 까지의 동역학은 뒤 스텝과 무관하므로 정확히 얻는다").
# 틀렸다. R19 의 BN 은 T 스텝 입력의 시간 평균에 걸린다 (lib_snn/model.py f_temporal_reduction, conv2 부터 영향).
# 그래서 T=4 런의 스텝 t 출력·스텝별 스파이크에는 스텝 t+1~4 의 정보가 들어 있고, 'T=t 에서 멈춘 망' 의 값이 아니다
# (analysis/early_exit_261020.md 0절·6절). 아래 값 (JSON 'step_readout_noncausal') 은 그 판독을 보존만 한다.
# 실제로 T 를 줄인 실측은 다음 블록 (4b') 이다.
def paired_ci(d):
    d = np.asarray(d, float)
    if d.size < 2:
        return {'d': float(d.mean()), 'ci95': [None, None], 'n': int(d.size), 'p': None}
    se = d.std(ddof=1) / math.sqrt(d.size); tq = stats.t.ppf(0.975, d.size - 1)
    p = float(2 * stats.t.sf(abs(d.mean() / se), d.size - 1)) if se > 0 else None
    return {'d': float(d.mean()), 'ci95': [float(d.mean() - tq * se), float(d.mean() + tq * se)], 'n': int(d.size), 'p': p,
            'each': [float(x) for x in d]}


def trunc_runs(gname):
    return [r for r in ROWS if r['group'] == gname and r['valid'] and r['npz'] and r['npz'].get('acc_t')
            and r['npz'].get('step_spk')]


TBASE = trunc_runs('baseline')
TRUNC = {}
for gname in [g[0] for g in GROUPS]:
    rr = trunc_runs(gname)
    if not rr:
        continue
    tg = {'n': len(rr), 'tags': [r['tag'] for r in rr], 'servers': sorted({r['server'] for r in rr}), 'cut': {}}
    for tc in range(1, T + 1):
        spk = [sum(r['npz']['step_spk'][:tc]) for r in rr]
        sops = [sum(r['npz']['step_sops'][:tc]) for r in rr]
        acc = [r['npz']['acc_t'][tc - 1] for r in rr]
        bspk = [sum(r['npz']['step_spk']) for r in TBASE]
        bsops = [sum(r['npz']['step_sops']) for r in TBASE]
        bacc = [r['npz']['acc_t'][T - 1] for r in TBASE]
        e = [1e6 * (E_AC * s_ + E_MAC * CONV1_MAC_STEP * tc) for s_ in sops]
        be = [1e6 * (E_AC * s_ + E_MAC * CONV1_MAC_STEP * T) for s_ in bsops]
        c = {'acc_each': acc, 'acc': msd(acc), 'spk': msd(spk), 'sops': msd(sops),
             'own_spk_cut_pct': 100 * (np.mean(spk) / np.mean([sum(r['npz']['step_spk']) for r in rr]) - 1),
             'own_sops_cut_pct': 100 * (np.mean(sops) / np.mean([sum(r['npz']['step_sops']) for r in rr]) - 1),
             'dacc_paired_vs_own_T4': paired_ci([a - r['npz']['acc_t'][T - 1] for a, r in zip(acc, rr)])}
        if len(rr) >= 2 and len(TBASE) >= 2:
            c['spk_vs_base_T4'] = ratio_pct(spk, bspk)
            c['sops_vs_base_T4'] = ratio_pct(sops, bsops)
            c['energy_vs_base_T4'] = ratio_pct(e, be)
            c['dacc_vs_base_T4'] = welch(acc, bacc)
        tg['cut'][tc] = c
    TRUNC[gname] = tg
TRUNC['_note'] = ('인용 금지. T=4 를 다 돈 런의 스텝 t 판독 (logits_t[:, t-1], 스텝별 배열 앞 t 스텝 합) 이다. '
                  '시간 평균 BN 때문에 T=t 에서 멈춘 망의 값이 아니다 (early_exit_261020.md 0·6절). 실측은 truncation_measured.')


# ---------------------------------------------------------------- 4b'. 추론 T 를 실제로 줄인 실측 (10-20 정정)
# early_exit_261020 이 같은 체크포인트를 CPU 로 T' 에서 다시 돌린 원자료 (테스트 앞 1000장, 가중치 그대로) 를 여기서
# 다시 읽어 계산한다 (early_exit_261020.json 의 값과 대조해 assert).
#   orig4 : 원래 BN 이동통계, T=4                       (an1020/ee/cpu/{nm}_T4_1k.npz)
#   fix3  : BN 이동통계 T=4 그대로, T'=3                (an1020/ee/cpu/{nm}_T3_1k.npz)
#   rc4/rc3/rc2 : BN 이동통계만 T' 에서 학습셋 5000장 (학습 후반과 같은 증강) 으로 다시 잡음 (an1020/ee/recal/{nm}_T{T'}_aug.npz)
# SOPs: CPU 원자료에는 위치별 지도가 없다 → '층별 스파이크 × 같은 체크포인트 1만 장 npz (T=4) 의 층별 SOPs/스파이크' 로 추정.
EE_DIR = S + '/an1020/ee'
EE_JSON = ROOT + '/analysis/early_exit_261020.json'
EE_RUNS = [('base11', 'r19c10-base---s11', 'baseline'), ('base12', 'r19c10-base---s12', 'baseline'),
           ('base13', 'r19c10-base---s13', 'baseline'),
           ('ours2', 'r19c10-ours_ch-3e-3-s2', 'ours_inter_ch 3e-3'), ('ours3', 'r19c10-ours_ch-3e-3-s3', 'ours_inter_ch 3e-3'),
           ('ours4', 'r19c10-ours_ch-3e-3-s4', 'ours_inter_ch 3e-3')]
EE_COND = {'orig4': ('cpu', '_T4_1k', 4), 'fix3': ('cpu', '_T3_1k', 3),
           'rc4': ('recal', '_T4_aug', 4), 'rc3': ('recal', '_T3_aug', 3), 'rc2': ('recal', '_T2_aug', 2)}
EE_BOOT_B, EE_BOOT_SEED = 20000, 20261020          # early_exit_261020.py 와 같은 값


def layer_k(path):
    """1만 장 npz (T=4) 의 층별 SOPs/스파이크 (위치별 fan-out 을 정확히 반영)."""
    z = np.load(path)
    out = {}
    for l in NEURON_LAYERS:
        c = z[l].astype(np.float64)
        sp = c.sum(-1) if c.ndim == 3 else c
        out[l] = float((sp * FANOUT[l]).sum() / c.sum())
    return out


def sops_est_check(path):
    """추정식 점검: 같은 npz 의 스텝별 배열에서 앞 t 스텝 SOPs 를 (a) 위치별로 정확히, (b) T=4 층별 SOPs/스파이크 × 층별 스파이크로
    추정해 상대 오차를 본다 (스텝 사이 층 안 공간 분포 변화가 추정에 주는 영향의 크기)."""
    z = np.load(path)
    k = layer_k(path)
    out = {}
    for tc in (2, 3):
        ex, es = 0.0, 0.0
        for l in NEURON_LAYERS:
            ct = z[l + '__t'].astype(np.float64)[:tc]
            sp = ct.sum(-1) if ct.ndim == 4 else ct
            ex += float((sp * FANOUT[l]).sum())
            es += float(ct.sum()) * k[l]
        out[tc] = 100 * (es / ex - 1)
    return out


def ee_clustered(pairs, seed=EE_BOOT_SEED, B=EE_BOOT_B):
    """early_exit_261020.py 의 clustered() 와 같은 계산 (같은 이미지·여러 체크포인트의 대응 차, %p).
    체크포인트별 차, 체크포인트 수준 t (df=K-1), 이미지 군집 부트스트랩, 체크포인트×이미지 이원 부트스트랩."""
    D = np.stack([c.astype(float) - a.astype(float) for a, c in pairs])
    K_, N_ = D.shape
    per = 100 * D.mean(1)
    out = {'K': K_, 'n_img': N_, 'per_ckpt_pp': [float(x) for x in per], 'mean_pp': float(per.mean())}
    se = per.std(ddof=1) / math.sqrt(K_); q = stats.t.ppf(0.975, K_ - 1)
    out['t_ckpt'] = [float(per.mean() - q * se), float(per.mean() + q * se)]
    rng = np.random.default_rng(seed)
    bi, b2 = np.empty(B), np.empty(B)
    for b in range(B):
        ii = rng.integers(N_, size=N_)
        bi[b] = D[:, ii].mean()
        kk = rng.integers(K_, size=K_)
        b2[b] = D[kk][:, ii].mean()
    out['boot_img'] = [float(100 * np.quantile(bi, 0.025)), float(100 * np.quantile(bi, 0.975))]
    out['boot_2way'] = [float(100 * np.quantile(b2, 0.025)), float(100 * np.quantile(b2, 0.975))]
    return out


def paired_pct(x, b):
    """같은 체크포인트끼리 100·(x/b − 1) 의 평균과 체크포인트 수준 t 95% (df=n−1)."""
    d = 100 * (np.asarray(x, float) / np.asarray(b, float) - 1)
    se = d.std(ddof=1) / math.sqrt(d.size); q = stats.t.ppf(0.975, d.size - 1)
    return {'pct': float(d.mean()), 'ci95': [float(d.mean() - q * se), float(d.mean() + q * se)], 'sd': float(d.std(ddof=1)),
            'n': int(d.size), 'each': [float(v) for v in d], 'paired': True}


_row_by_tag = {r['tag']: r for r in ROWS}
TM = {'_note': ('추론 T 를 실제로 줄여 다시 돌린 실측 (CPU, 테스트 앞 1000장, 체크포인트 n=3/팔, 학습 서버 138). '
                '원자료: early_exit_261020 (6.1 BN 고정, 6.2 BN 재추정 증강판). SOPs 는 추정 (층별 스파이크 × 1만 장 npz 층별 SOPs/스파이크).'),
      'conditions': {'orig4': '원래 BN 이동통계, T=4', 'fix3': 'BN 이동통계 T=4 그대로, T\'=3',
                     'rc4': 'BN 이동통계만 T\'=4 에서 다시 잡음 (절차 효과 점검)', 'rc3': 'BN 이동통계만 T\'=3 에서 다시 잡음',
                     'rc2': 'BN 이동통계만 T\'=2 에서 다시 잡음'},
      'runs': {}, 'by_group': {}}
_COR = {}
for nm, tag, gname in EE_RUNS:
    r = _row_by_tag[tag]
    K = layer_k(r['npz']['file'])
    o = {'nm': nm, 'group': gname, 'seed': r['seed'], 'server': r['server'], 'npz_10k': r['npz']['file'],
         'sops_est_check_pct': sops_est_check(r['npz']['file']), 'cond': {}}
    y0 = None
    _COR[tag] = {}
    for cond, (sub, suf, TT) in EE_COND.items():
        p = os.path.join(EE_DIR, sub, nm + suf + '.npz')
        c = np.load(p)
        assert int(c['T']) == TT, p
        if sub == 'recal':
            assert str(c['mode']) == 'aug' and float(c['pipe_maxdiff']) == 0.0, p
        y = c['labels']
        if y0 is None:
            y0 = y
            zg = np.load(r['npz']['file'])
            assert (zg['labels'][:len(y)] == y).all()
            o['gpu_npz_vs_cpu_argmax_same_pct'] = 100 * float((zg['logits_t'][:len(y), -1].argmax(1) == c['logits_t'][:, -1].argmax(1)).mean())
            o['gpu_npz_acc_first1000'] = 100 * float((zg['logits_t'][:len(y), -1].argmax(1) == y).mean())
        assert (y == y0).all()
        cor = c['logits_t'][:, -1].argmax(1) == y
        lay = {l: float(c['spk_' + l].sum(1).mean()) for l in NEURON_LAYERS}
        spk = sum(lay.values()); sops = sum(lay[l] * K[l] for l in NEURON_LAYERS)
        o['cond'][cond] = {'T': TT, 'n_img': int(len(y)), 'acc': 100 * float(cor.mean()), 'spk': spk, 'sops_est': sops,
                           'energy_uJ': 1e6 * (E_AC * sops + E_MAC * CONV1_MAC_STEP * TT), 'file': p}
        _COR[tag][cond] = cor
    for cond in EE_COND:
        e = o['cond'][cond]
        for k_ in ('spk', 'sops_est', 'energy_uJ'):
            e[k_ + '_vs_own_orig4_pct'] = 100 * (e[k_] / o['cond']['orig4'][k_] - 1)
    TM['runs'][tag] = o
_btags = [t for _, t, g in EE_RUNS if g == 'baseline']
for gname in ('baseline', 'ours_inter_ch 3e-3'):
    tags = [t for _, t, g in EE_RUNS if g == gname]
    A = {}
    for cond in EE_COND:
        val = {k_: [TM['runs'][t]['cond'][cond][k_] for t in tags] for k_ in ('acc', 'spk', 'sops_est', 'energy_uJ')}
        bval = {k_: [TM['runs'][t]['cond']['orig4'][k_] for t in _btags] for k_ in ('acc', 'spk', 'sops_est', 'energy_uJ')}
        e = {'n_ckpt': len(tags), 'acc': msd(val['acc']), 'spk': msd(val['spk']), 'sops_est': msd(val['sops_est']),
             'energy_uJ': msd(val['energy_uJ']),
             'spk_vs_own_orig4_pct': msd([TM['runs'][t]['cond'][cond]['spk_vs_own_orig4_pct'] for t in tags]),
             'sops_vs_own_orig4_pct': msd([TM['runs'][t]['cond'][cond]['sops_est_vs_own_orig4_pct'] for t in tags])}
        if cond != 'orig4':
            e['dacc_vs_own_orig4'] = ee_clustered([(_COR[t]['orig4'], _COR[t][cond]) for t in tags])
        if cond in ('rc3', 'rc2'):
            e['dacc_vs_own_rc4'] = ee_clustered([(_COR[t]['rc4'], _COR[t][cond]) for t in tags])
        # baseline 원래 T=4 대비: baseline 은 같은 체크포인트라 짝지은 비, ours 는 다른 체크포인트라 독립 (n=3 대 3)
        for k_ in ('spk', 'sops_est', 'energy_uJ'):
            if gname == 'baseline':
                e[k_ + '_vs_base_orig4'] = None if cond == 'orig4' else paired_pct(val[k_], bval[k_])
            else:
                e[k_ + '_vs_base_orig4'] = ratio_pct(val[k_], bval[k_])
        if gname != 'baseline':
            e['dacc_vs_base_orig4'] = welch(val['acc'], bval['acc'])
        A[cond] = e
    TM['by_group'][gname] = A
TM['sops_est_check_pct_range'] = [min(min(TM['runs'][t]['sops_est_check_pct'].values()) for t in TM['runs']),
                                  max(max(TM['runs'][t]['sops_est_check_pct'].values()) for t in TM['runs'])]
TM['gpu_vs_cpu_argmax_same_pct_range'] = [min(TM['runs'][t]['gpu_npz_vs_cpu_argmax_same_pct'] for t in TM['runs']),
                                          max(TM['runs'][t]['gpu_npz_vs_cpu_argmax_same_pct'] for t in TM['runs'])]

# early_exit_261020.json 과 대조 (같은 원자료·같은 부트스트랩 → 같은 값이어야 한다)
_EE = json.load(open(EE_JSON))
_arm = {'baseline': 'baseline', 'ours_inter_ch 3e-3': 'ours_inter_ch'}
_chk = []
for nm, tag, gname in EE_RUNS:
    c_ = TM['runs'][tag]['cond']
    ra = _EE['recal']['aug']['runs'][nm]
    _chk.append(abs(c_['orig4']['acc'] - ra['acc_orig_T4']))
    _chk.append(abs(c_['fix3']['acc'] - _EE['cpu']['runs'][nm]['T3_1k']['acc_T3']))
    _chk.append(abs(-c_['fix3']['spk_vs_own_orig4_pct'] - _EE['cpu']['runs'][nm]['T3_1k']['spk_red_pct']))
    for cond, TT in (('rc4', '4'), ('rc3', '3'), ('rc2', '2')):
        _chk.append(abs(c_[cond]['acc'] - ra['T'][TT]['acc']))
        _chk.append(abs(c_[cond]['spk_vs_own_orig4_pct'] - ra['T'][TT]['spk_vs_orig_T4_pct']))
for gname, arm in _arm.items():
    A = TM['by_group'][gname]
    pairs_ = [(A['fix3']['dacc_vs_own_orig4'], _EE['cpu']['T3_1k_by_arm'][arm])] + \
             [(A[cond]['dacc_vs_own_orig4'], _EE['recal']['aug']['by_arm'][arm][TT]['vs_orig_T4'])
              for cond, TT in (('rc4', '4'), ('rc3', '3'), ('rc2', '2'))]
    for mine, ref in pairs_:
        _chk += [abs(mine['mean_pp'] - ref['mean_pp']), abs(mine['boot_2way'][0] - ref['boot_2way'][0]),
                 abs(mine['boot_2way'][1] - ref['boot_2way'][1]), abs(mine['t_ckpt'][0] - ref['t_ckpt']['lo']),
                 abs(mine['t_ckpt'][1] - ref['t_ckpt']['hi'])]
TM['check_vs_early_exit_json_maxabs'] = float(max(_chk))
assert TM['check_vs_early_exit_json_maxabs'] < 1e-6, TM['check_vs_early_exit_json_maxabs']
TM['ours_vs_base_T4_same1000_early_exit_lnwelch'] = _EE['recal']['aug']['ours_vs_base_T4_same1000']

# ---------------------------------------------------------------- 4c. 서버 민감도 (npz 축 baseline 이 138 서버 3런뿐)
_b138 = [r for r in BASE_ALL if r['server'] == '138']; _b23 = [r for r in BASE_ALL if r['server'] == '23']
_ratio138 = [r['npz']['npz_spk'] / r['spikes'] for r in BASE]
_oursv = [r for r in ROWS if r['group'] == 'ours_inter_ch 3e-3' and r['valid'] and r['npz']]
_bnpz_n6 = float(np.mean(_ratio138)) * float(np.mean([r['spikes'] for r in BASE_ALL]))
SRV = {'scount_23_vs_138': ratio_pct([r['spikes'] for r in _b23], [r['spikes'] for r in _b138]),
       'scount_23_vs_138_welch_p': float(stats.ttest_ind([r['spikes'] for r in _b23], [r['spikes'] for r in _b138],
                                                          equal_var=False).pvalue),
       'acc_23_vs_138': welch([r['val_acc'] for r in _b23], [r['val_acc'] for r in _b138]),
       'npz_over_scount_base138': msd(_ratio138),
       'npz_over_scount_ours': msd([r['npz']['npz_spk'] / r['spikes'] for r in _oursv]),
       'base_npz_n6_extrapolated': _bnpz_n6,
       'ours_npz_vs_extrapolated_pct': 100 * (float(np.mean([r['npz']['npz_spk'] for r in _oursv])) / _bnpz_n6 - 1)}

# ---------------------------------------------------------------- 5. 문헌
# 수치는 docs/paper/compare-candidates-261020.md (CC) 와 refs-audit.md (RA-x) 에서 옮겼다. 예외 (10-20 검토 반영):
# UniSparse 행은 부록 Table 4 원문, STDS 행은 AP Table 2 원문 (scratchpad 의 PDF 추출본) 을 직접 읽었다.
LIT = [
    dict(paper='Activity Pruning (AT-LIF)', venue='NeurIPS 2025', model='ResNet-20', data='CIFAR-10', T=8,
         base=92.89, meth=92.05, metric='FR', before=0.236, after=0.053, sops_before=256.69, sops_after=61.70,
         n='미기재', src='RA 10-02 1a (proceedings PDF), CC §2·4',
         comp='아니오. 모델·T 다름, baseline 이 우리보다 3.84%p 낮다. FR 정의는 같은 꼴 (전 뉴런·전 스텝 평균) 이나 입력·출력층 포함 여부 미확인. 감소율끼리만 참고'),
    dict(paper='Activity Pruning (AT-LIF, λθ 2e-3)', venue='NeurIPS 2025', model='ResNet-20', data='CIFAR-10', T=8,
         base=92.89, meth=91.06, metric='FR', before=0.236, after=0.043, sops_before=256.69, sops_after=48.85,
         n='미기재', src='RA 10-02 1a', comp='위와 같음'),
    dict(paper='Activity Pruning 표의 L1 규제 (AR)', venue='NeurIPS 2025', model='ResNet-20', data='CIFAR-10', T=8,
         base=92.89, meth=91.31, metric='FR', before=0.236, after=0.060, sops_before=256.69, sops_after=57.26,
         n='미기재', src='RA 10-02 1c', comp='위와 같음. 우리 plain L2·BPSR 과 같은 부류 (균등 벌점) 의 외부 값'),
    dict(paper='Activity Pruning (AT-LIF)', venue='NeurIPS 2025', model='SEW-ResNet-18', data='ImageNet-100', T=4,
         base=83.44, meth=82.40, metric='FR', before=0.138, after=0.038, sops_before=1276.42, sops_after=463.65,
         n='미기재', src='RA 10-02 1d, 10-13 §2', comp='아니오. 데이터셋·모델 다름 (T 는 같음)'),
    dict(paper='Activity Pruning (AT-LIF)', venue='NeurIPS 2025', model='VGGSNN', data='CIFAR10-DVS', T=10,
         base=82.70, meth=82.20, metric='FR', before=0.080, after=0.043, sops_before=1066.40, sops_after=626.57,
         n='미기재', src='RA 10-02 1e', comp='아니오. DVS·T=10'),
    dict(paper='UPR', venue='ICLR 2024', model='6Conv+2FC', data='CIFAR-10', T=8,
         base=92.84, meth=92.63, metric='SOPs', before=None, after=None, sops_before=776.12, sops_after=38.32,
         n='미기재', src='RA 10-13 §2 (원문 Table 1)',
         comp='아니오. 가중치·뉴런 마스크 프루닝 + 미세조정이라 SOPs 감소 대부분이 연결 제거에서 온다 (스파이크 감소와 분리 안 됨)'),
    dict(paper='UPR', venue='ICLR 2024', model='SEW-ResNet-18', data='ImageNet', T=4,
         base=63.18, meth=61.89, metric='SOPs', before=None, after=None, sops_before=1337.87, sops_after=311.70,
         n='미기재', src='CC §4·6 (10-20 확인), RA 10-13 §2', comp='아니오. 데이터셋·모델 다름, 프루닝'),
    dict(paper='UPR', venue='ICLR 2024', model='VGGSNN', data='CIFAR10-DVS', T=10,
         base=82.4, meth=81.9, metric='SOPs', before=None, after=None, sops_before=647.30, sops_after=47.81,
         n='미기재', src='RA 10-13 §2', comp='아니오. DVS·T=10, 프루닝'),
    # UniSparse: 부록 Table 4 원문 (PMLR PDF 추출본 scratchpad/us/us.txt 864–895행, 10-20 eip-stats 직접 확인).
    # 각주: "CIFAR-10 은 시드 3회" → C10 행의 ± 는 n=3 의 SD. FR 은 원문 % 를 분수로 바꿈.
    dict(paper='UniSparse (스파이크+가중치, λ 5e-7)', venue='ICML 2026', model='SEW-ResNet-19', data='CIFAR-10', T=4,
         base=95.16, meth=94.71, metric='FR', before=0.0683, after=0.0552, sops_before=923.01, sops_after=39.47, dec=4,
         n='n=3 (각주 "CIFAR-10 시드 3회"). 밀집 95.16 ±0.03 @923.01 ±16.69M, FR 6.83 ±0.10%; 희소 94.71 ±0.11 @39.47 ±0.38M, FR 5.52 ±0.07%, 가중치 밀도 7.29%',
         src='UniSparse 부록 Table 4 (scratchpad/us/us.txt, 10-20 직접 확인)',
         comp='부분적. 데이터셋·T·채널 구성 (128/256/512, 블록 [3,3,2]) 은 같다. 다른 점: SEW 덧셈 (identity 경로로 스파이크가 정수로 '
              '더해져 뒤 층들이 다시 읽음), LIF τ=2·ATan, 밀집 정확도가 우리보다 1.57%p 낮음, SOPs 감소의 대부분이 가중치 프루닝 '
              '(밀도 7.29%) 에서 옴. FR 정의는 같은 꼴 (코드: 원소 수로 가중한 (x≠0) 평균 = 전체 FR). SOPs 는 conv 만 세고 첫 conv 를 뺀다 (코드 SOPMonitor)'),
    dict(paper='UniSparse (λ 1e-6)', venue='ICML 2026', model='SEW-ResNet-19', data='CIFAR-10', T=4,
         base=95.16, meth=94.07, metric='FR', before=0.0683, after=0.0391, sops_before=923.01, sops_after=18.58, dec=4,
         n='n=3 (±0.07, 18.58 ±0.48M, FR 3.91 ±0.15%, 밀도 3.41%)', src='UniSparse 부록 Table 4', comp='위와 같음'),
    dict(paper='UniSparse (가장 희소한 점, λ 2.5e-6)', venue='ICML 2026', model='SEW-ResNet-19', data='CIFAR-10', T=4,
         base=95.16, meth=91.56, metric='FR', before=0.0683, after=0.0190, sops_before=923.01, sops_after=6.11, dec=4,
         n='n=3 (±0.04, 6.11 ±0.20M, FR 1.90 ±0.09%, 밀도 1.15%)', src='UniSparse 부록 Table 4', comp='위와 같음'),
    dict(paper='UniSparse (스파이크+가중치, λ 7e-8)', venue='ICML 2026', model='SEW-ResNet-18', data='ImageNet', T=4,
         base=63.78, meth=61.89, metric='FR', before=0.1537, after=0.1338, sops_before=1545.91, sops_after=275.17, dec=4,
         n='n 미확인. 표에는 ± (밀집 ±0.04, 희소 ±0.12) 가 있으나 본문은 "ImageNet 은 수준마다 1회" 라고 적어 서로 맞지 않는다',
         src='UniSparse 부록 Table 4', comp='아니오. 데이터셋·모델 다름, 가중치 프루닝 결합 (밀도 41.67%)'),
    dict(paper='SCA', venue='ICML 2024', model='VGG-16', data='CIFAR-10', T=4,
         base=91.14, meth=91.46, metric='파라미터 55.86% 남김', before=None, after=None, sops_before=None, sops_after=None,
         n='미기재', src='RA B4 (Table 1: +0.32 @55.86%)',
         comp='아니오. 지표가 남은 파라미터 비율이고 SynOps 는 그림뿐 (채널 0.8 제거 시 약 1/2). 구조적 프루닝'),
    dict(paper='STDS (AP 재구현값)', venue='ICML 2022', model='6Conv+2FC', data='CIFAR-10', T=None,
         base=None, meth=90.71, metric='FR', before=None, after=0.072, sops_before=None, sops_after=45.48,
         n='미기재', src='AP Table 2 (scratchpad/ap.txt 534–559행, 10-20 확인)',
         comp='아니오. AP 가 재구현한 값이다 (AP 본문 "reproduced implementations"). STDS 원문 값이 아니고 T 도 표에 없다. 가중치 프루닝 (스파이크를 직접 줄이지 않음)'),
    dict(paper='ESL-SNN', venue='AAAI 2023', model='미확인', data='미확인', T=None,
         base=None, meth=None, metric='미확인', before=None, after=None, sops_before=None, sops_after=None,
         n='—', src='CC §5 ("표 값은 아직 읽지 않았다")', comp='판단 불가 (표 값 미확인, eip-refs 대기). 가중치 프루닝 계열'),
    dict(paper='SEENN-I (조기 종료)', venue='NeurIPS 2023', model='ResNet-19', data='CIFAR-10', T=4,
         base=96.46, meth=96.38, metric='평균 T', before=4.0, after=1.20, sops_before=None, sops_after=None,
         n='미기재', src='CC §1.2·3.2 C (arXiv v2 Table 1, 그림의 T=4 96.46)',
         comp='부분적. 모델·데이터셋·T 같음. 지표가 평균 T 이고 스파이크·SOPs 미보고. SEENN 은 스텝별 CE (TET) 학습에 '
              '멈춘 스텝까지만 계산하는 망을 전제한다. 우리 R19 는 시간 평균 BN 이라 우리 스텝 몫 (스텝 4 38%, 스텝 1 4%, n=3) 은 '
              'T=4 를 다 돈 런 안의 분포이고, SEENN 의 스파이크 감소를 어림하는 데 쓰지 않는다 (10-20 정정; 스텝 몫을 그대로 쓴 '
              '가정 계산은 early_exit_261020 4절 (c), 측정 아님). 학습 없는 가장 가까운 대조는 3b 절 (T\'=3 + BN 재추정)'),
    dict(paper='SEENN-I (조기 종료)', venue='NeurIPS 2023', model='ResNet-19', data='CIFAR-10', T=4,
         base=96.46, meth=96.07, metric='평균 T', before=4.0, after=1.09, sops_before=None, sops_after=None,
         n='미기재', src='CC §3.2 C', comp='위와 같음'),
    dict(paper='DT-SNN (조기 종료)', venue='DAC 2023', model='ResNet-19', data='CIFAR-10', T=4,
         base=94.03, meth=93.87, metric='평균 T', before=4.0, after=1.27, sops_before=None, sops_after=None,
         n='미기재', src='RA B2 (이번에 재확인 안 함, CC §6)', comp='부분적. SEENN 과 같은 이유 + baseline 94.03 으로 약함'),
    dict(paper='Sorbaro 외', venue='Front. Neurosci. 2020', model='All-ConvNet (변환 SNN)', data='CIFAR-10', T=10,
         base=91.75, meth=91.73, metric='SynOps', before=None, after=None, sops_before=2179, sops_after=1593,
         n='미기재', src='RA #3 (Table 1, 09월 원문)', comp='아니오. 변환 SNN, Ndt=10, ANN 활성에 건 벌점'),
    dict(paper='Spik4lite', venue='ICML 2026', model='Spikformer-4-384', data='CIFAR-10', T=4,
         base=95.24, meth=94.15, metric='SOPs', before=None, after=None, sops_before=822.64, sops_after=321.49,
         n='미기재', src='RA 10-07 (c) (PMLR Table 1)', comp='아니오. 트랜스포머, 채널 물리 제거 + 미세조정. Jetson 실측 5.34→3.08 J'),
    dict(paper='SRSC (심사 중 제출본)', venue='ICLR 2027 제출', model='ResNet-19 등', data='CIFAR-10', T=None,
         base=None, meth=None, metric='FR·SOPs', before=None, after=None, sops_before=None, sops_after=None,
         n='미확인', src='RA 10-13 N1 (초록만)', fr_red_given=-49, sops_red_given=-24,
         comp='판단 불가. 초록만 ("±0.5% 안, FR −49%, SOPs −24%, relative to state-of-the-art methods"). 기준 방법·표 미확인'),
    dict(paper='MI-TRQR', venue='NeurIPS 2025', model='PSN-VGG', data='CIFAR10-DVS', T=4,
         base=None, meth=None, metric='FR', before=None, after=None, sops_before=None, sops_after=None,
         n='미확인', src='CC §2·4 (범위만: FR −42~−67%, 정확도 상승)', comp='아니오. DVS, PSN 뉴런. 행별 값은 이 문서들에 없음'),
    dict(paper='DSN', venue='arXiv 2610.21748', model='MS-ResNet18, SDT', data='—', T=None,
         base=None, meth=None, metric='이론 에너지', before=None, after=None, sops_before=None, sops_after=None,
         n='미확인', src='CC §4 (이론 에너지 −40~−73%)', comp='아니오. 휴면 게이팅 (스파이크 수를 줄이는 방법이 아님, 원문 서술)'),
]
for L in LIT:
    L['dacc'] = (round(L['meth'] - L['base'], 2) if (L.get('meth') is not None and L.get('base') is not None)
                 else L.get('dacc_given'))
    L['red_metric_pct'] = (100 * (L['after'] / L['before'] - 1) if (L.get('before') and L.get('after') is not None) else
                           L.get('fr_red_given'))
    L['red_sops_pct'] = (100 * (L['sops_after'] / L['sops_before'] - 1) if (L.get('sops_before') and L.get('sops_after'))
                         else L.get('sops_red_given'))


# UniSparse 밀집 R19 의 스파이크당 SOPs 추정 (FR × 뉴런 수 × T 로 이미지당 스파이크를 되돌림).
# SEW 덧셈 뒤의 정수 입력도 SOPMonitor 는 값만큼 센다 → '스파이크당' 은 같은 정의가 아니다 (참고값).
_us_spk = 0.0683 * US_N * T
US_EST = {'N_us': US_N, 'spk_per_img_dense': _us_spk, 'sops_per_spk_dense': 923.01e6 / _us_spk,
          'sops_per_spk_dense_with_our_N': 923.01e6 / (0.0683 * N_NEURON * T),
          'ratio_sops_per_spk_vs_our_base': 923.01e6 / _us_spk / G['baseline']['npz']['sops_per_spk'][0],
          'ratio_sops_per_spk_vs_our_base_our_N': 923.01e6 / (0.0683 * N_NEURON * T) / G['baseline']['npz']['sops_per_spk'][0],
          'dense_sops_ratio_us_over_our_base': 923.01e6 / G['baseline']['npz']['sops'][0],
          'fr_red_us_5e7': 100 * (0.0552 / 0.0683 - 1), 'sops_red_us_5e7': 100 * (39.47 / 923.01 - 1)}

# ---------------------------------------------------------------- 6. 저장
FAN = check_fanout()


def tf_conv_sops_check(tags):
    """독립 경로 검산: 실제 Keras 가중치 층 목록을 따라 각 층 입력의 위치별 스파이크 지도 (채널 합) 를
    TF conv2d (커널 전부 1, 같은 stride·padding) 에 통과시킨 합 × C_out 으로 SOPs 를 다시 센다."""
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
    except Exception as e:
        return f'skipped: {e!r}'
    wl = None
    for d in PRUNE_DIRS:
        for p_ in sorted(glob.glob(d + '/r19c10-*.json')):
            j = json.load(open(p_))
            if 'weight_layers' in j:
                wl = j['weight_layers']; break
        if wl:
            break
    if not wl:
        return 'skipped: weight_layers 없음'
    out = []
    for tag in tags:
        r = next((q for q in ROWS if q['tag'] == tag and q['npz']), None)
        if r is None:
            continue
        z = np.load(r['npz']['file']); ns = float(z['n_samples']); tot = 0.0
        for w_ in wl:
            if w_['src'] == 'n_in':
                continue
            x = z[w_['src']].astype(np.float64); ks = w_['ks']
            if len(ks) == 4:
                k, cout = ks[0], ks[3]; H = x.shape[0]; Ho = int(round(w_['hw'] ** 0.5))
                y = tf.nn.conv2d(tf.constant(x.sum(-1)[None, :, :, None], tf.float64), tf.ones((k, k, 1, 1), tf.float64),
                                 strides=H // Ho, padding='SAME' if k == 3 else 'VALID')
                tot += float(tf.reduce_sum(y)) * cout
            else:
                tot += x.sum() * ks[1]
        out.append({'tag': tag, 'tf_conv_sops': tot / ns, 'script_sops': r['npz']['sops'],
                    'rel_diff': tot / ns / r['npz']['sops'] - 1})
    return out


FAN['tf_conv_sops_check'] = tf_conv_sops_check(['r19c10-base---s11', 'r19c10-ours_ch-3e-3-s4', 'r19c10-bpsr-1e-10-s1',
                                                'r19c10-l2-1e-7-s3'])
R = {'generated': '2026-10-20', 'ledger_snapshot': INV_GEN, 'definitions': {
        'N_neuron': N_NEURON, 'T': T, 'conv1_mac_per_step': CONV1_MAC_STEP, 'E_AC': E_AC, 'E_MAC': E_MAC,
        'identity_edges': IDENT, 'spiking_layers': NEURON_LAYERS},
     'fanout_check': FAN, 'rows': ROWS, 'groups': G, 'vs_baseline': VS_BASE, 'pairs_vs_ours': PAIR,
     'same_spike_sops': SAME, 'same_spike_sops_scount_axis': SAME_SC, 'npz_vs_scount_axis': AXIS,
     'step_readout_noncausal': TRUNC, 'truncation_measured': TM,
     'server_sensitivity': SRV, 'unisparse_estimate': US_EST, 'curve': CURVE, 'literature': LIT,
     'baseline_npz_means': {'npz_spk': BMEAN_SPK, 'sops': BMEAN_SOPS}}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


json.dump(_clean(R), open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)
print('wrote', OUT_JSON)


# ---------------------------------------------------------------- 7. 보고서 (md)
def K(x):
    return '–' if x is None else f'{x:,.0f}'


def F(x, d=2):
    return '–' if x is None else f'{x:.{d}f}'


def PC(r, d=1):
    return f'{r["pct"]:+.{d}f}% [{r["ci95"][0]:+.{d}f}, {r["ci95"][1]:+.{d}f}]'


def MS(v, d=1, scale=1.0):
    return f'{v[0] / scale:,.{d}f} ±{v[1] / scale:,.{d}f}' if v[2] > 1 else f'{v[0] / scale:,.{d}f}'


def DCI(x, d=2):
    """차 [95% CI] (welch / paired_ci 결과)."""
    if x['ci95'][0] is None:
        return f'{x["d"]:+.{d}f}'
    return f'{x["d"]:+.{d}f} [{x["ci95"][0]:+.{d}f}, {x["ci95"][1]:+.{d}f}]'


def SC(x, d=2):
    return f'{x["ours_minus_other_pct"]:+.{d}f}% [{x["ci95_pct"][0]:+.{d}f}, {x["ci95_pct"][1]:+.{d}f}]'


md = []
w = md.append
g0, gO, gB, gL, gD = (G[k] for k in ('baseline', 'ours_inter_ch 3e-3', 'BPSR 1e-10', 'plain L2 1e-7', 'ours_inter_ch 4e-3'))
vO, vB, vL, vD = (VS_BASE[k] for k in ('ours_inter_ch 3e-3', 'BPSR 1e-10', 'plain L2 1e-7', 'ours_inter_ch 4e-3'))
pB, pL = PAIR['BPSR 1e-10'], PAIR['plain L2 1e-7']
sB, sL = SAME['200_420K']['BPSR'], SAME['200_420K']['plain L2']
sBc, sLc = SAME_SC['200_420K']['BPSR'], SAME_SC['200_420K']['plain L2']
aB, aL = AXIS['200_420K']['BPSR'], AXIS['200_420K']['plain L2']
uni = next(L for L in LIT if L['paper'].startswith('UniSparse (스파이크+가중치, λ 5e-7)'))
ap1 = LIT[0]
red_all = {k: VS_BASE[k]['sops_to_spk_reduction'] for k in VS_BASE}
tb, to = TRUNC['baseline'], TRUNC['ours_inter_ch 3e-3']
tb3, tb2, to3, to2 = tb['cut'][3], tb['cut'][2], to['cut'][3], to['cut'][2]     # 비인과 판독 (인용 금지, 검토 기록용)
ue = US_EST
# 10-20 정정: 추론 T 를 실제로 줄인 실측 (early_exit_261020 CPU 원자료)
mB, mO = TM['by_group']['baseline'], TM['by_group']['ours_inter_ch 3e-3']
_EEclean_b3 = _EE['recal']['clean']['by_arm']['baseline']['3']['vs_orig_T4']
_EEmaxsave = _EE['arms']['baseline']['real_max_saving_pct']


def CK(c, d=2):
    """대응 차 (체크포인트×이미지 이원 부트스트랩 95%)."""
    return f'{c["mean_pp"]:+.{d}f}%p [{c["boot_2way"][0]:+.{d}f}, {c["boot_2way"][1]:+.{d}f}]'


def CKT(c, d=2):
    return f'[{c["t_ckpt"][0]:+.{d}f}, {c["t_ckpt"][1]:+.{d}f}]'


def MSm(v, d=1):
    return f'{v[0]:+.{d}f} ±{v[1]:.{d}f}'

w('# 발화율 (FR) · SOPs 와 문헌 비교표 (R19 CIFAR-10, T=4, 26-10-20)')
w('')
w('- 요청: `docs/paper/compare-candidates-261020.md` 1.1 의 "문헌 비교표" (사용자 승인 10-20, "학습 없는 분석 3건"). 같은 절의 조기 종료 모의와 Suetake npz 는 따로 한다 (Suetake 행은 그 추출이 끝나면 더한다).')
w(f'- 작성: eip-stats. 생성: `analysis/fr_sops_261020.py` → `.json`, 이 문서. 원장 스냅숏 {INV_GEN} (`scratchpad/an1019/snapshot/inventory.json`) 으로 런·n 을 골랐다. **10-20 eip-critic 1차 검토 14건을 반영했다** (끝의 "검토 반영" 절).')
w('- **10-20 정정.** 검토 전 판의 3b 절 (추론만 일찍 끊기) 은 npz 의 스텝 t 판독을 \'T=t 에서 멈춘 망\' 으로 읽었다. R19 의 BN 이 T 스텝 입력의 시간 평균에 걸려 그 판독은 실제 조건이 아니다 (`analysis/early_exit_261020.md` 0·6절). 3b 절·요약 3·3절 스텝 몫 문단·5절 우리 행과 SEENN 행·6절 1번·\'우리 쪽에 남는 것\'·[LIMITATION]·\'다음\' 을 같은 체크포인트를 T\' 로 다시 돌린 실측으로 바꿨다 (끝의 \'검토 반영\' 마지막 줄).')
w('- 파일만 썼다. 큐·실험·서버는 건드리지 않았고 커밋하지 않았다. **GPU 를 쓰지 않았다** (기존 npz 를 읽기만; TF 는 CPU 로 fan-out 검산에만 씀). 3b 절은 early_exit_261020 이 CPU 로 만든 원자료 (`scratchpad/an1020/ee/{cpu,recal}`) 를 읽기만 했다.')
w('- 우리 수치는 npz (테스트 1만 장, 평가 모드 1회 추론) 에서 새로 계산했다. 정확도·s_count·손실은 원장 값이다. 문헌 수치는 `compare-candidates-261020.md` (CC) 와 `refs-audit.md` (RA) 에서 옮겼고 각 행에 출처를 적었다. 예외: UniSparse 행은 부록 Table 4 와 공개 코드, STDS 행은 Activity Pruning (AP) Table 2 를 scratchpad 의 원문 추출본으로 직접 읽었다.')
w('- **사용 제한.** 이 문서의 FR·SOPs 절대값은 npz 축 값이라 09-15 보류 ("`s_count` 와 npz 총합 11% 불일치, 원인 미상 — 해소 전까지 절대 스파이크 수를 논문에 쓰지 않는다", `docs/decisions.md`) 에 그대로 걸린다. 10-19 기록도 "원인 미해결" 이다. **문서 안의 비교에만 쓰고 논문 표에는 넣지 않는다.** 푸는 길은 둘이다: (a) 같은 체크포인트를 평가 모드로 s_count 코드 경로에 다시 돌려 npz 와 맞는지 본다 (GPU, 승인 필요), (b) 사용자가 "npz = 추론 비용 측정 축" 으로 정해 보류를 명시적으로 푼다.')
w('')
w('## 요약')
w('')
w(f'1. **우리 대표점의 FR·SOPs (npz 축).** baseline (npz n=3, 138 서버만) FR {g0["npz"]["fr_step"][0]:.4f} 스파이크/뉴런/스텝, SOPs {g0["npz"]["sops"][0] / 1e6:.1f}M/이미지. '
  f'ours_inter_ch 3e-3 (n=4) FR {gO["npz"]["fr_step"][0]:.4f}, SOPs {gO["npz"]["sops"][0] / 1e6:.1f}M. '
  f'BPSR 1e-10 (n=4) {gB["npz"]["fr_step"][0]:.4f} / {gB["npz"]["sops"][0] / 1e6:.1f}M, plain L2 1e-7 (n=4) {gL["npz"]["fr_step"][0]:.4f} / {gL["npz"]["sops"][0] / 1e6:.1f}M. '
  f'첫 층은 아날로그 직접 입력이라 MAC 이다: 3.54M/스텝 (패딩 포함), 매 스텝 다시 계산하면 이미지당 14.16M (방법과 무관한 고정 비용).')
w(f'2. **SOPs 는 스파이크보다 덜 준다.** ours_inter_ch 3e-3 (n=4 대 baseline n=3): 스파이크 {PC(vO["npz_spk"])} (npz 축), SOPs {PC(vO["sops"])}. '
  f'원장 s_count 축 (baseline n=6) 의 {vO["scount_n6"]["pct"]:.1f}% 가 SOPs 로는 {vO["sops"]["pct"]:.1f}% 가 된다. SOPs/스파이크가 {PC(vO["sops_per_spk"])} 늘기 때문이다. '
  f'세 팔 모두 그렇다 (SOPs 감소/스파이크 감소: ours {red_all["ours_inter_ch 3e-3"]:.3f}, BPSR {red_all["BPSR 1e-10"]:.3f}, plain L2 {red_all["plain L2 1e-7"]:.3f}, 각 n=4). '
  f'이 비는 CI 없는 기술값이고 감소 깊이에 따라 변하는 값이라 (4절 탄력성 0.84) 팔 사이 0.005 차는 해석하지 않는다. 감소가 fan-out 이 작은 32×32 층에 몰려서다 (SOPs 중 conv2 몫 baseline {g0["sops_stage_share"]["conv2"][0]:.1f}% → ours {gO["sops_stage_share"]["conv2"][0]:.1f}%).')
w(f'3. **학습 없이 추론 T 를 한 스텝 줄인 baseline 은 BN 을 어떻게 두느냐에 따라 갈린다 (10-20 정정).** 검토 전 판의 "T=3 에서 끊으면 Δ {DCI(tb3["dacc_paired_vs_own_T4"])}%p, 스파이크 {tb3["spk_vs_base_T4"]["pct"]:.1f}%" 는 T=4 를 다 돈 런의 스텝 3 판독이라 실제 조건이 아니다 (`early_exit_261020.md` 0·6절). '
  f'같은 체크포인트를 T\'=3 으로 다시 돌린 실측 (CPU, 테스트 앞 1000장, 체크포인트 n=3, 138; [ ] 는 체크포인트×이미지 이원 부트스트랩 95%): BN 이동통계를 T=4 값으로 두면 자기 T=4 대비 {CK(mB["fix3"]["dacc_vs_own_orig4"])} (스파이크 {mB["fix3"]["spk_vs_own_orig4_pct"][0]:+.1f}%). '
  f'BN 이동통계만 T\'=3 에서 학습셋으로 다시 잡으면 {CK(mB["rc3"]["dacc_vs_own_orig4"])}, 스파이크 {PC(mB["rc3"]["spk_vs_base_orig4"])}, SOPs (추정) {PC(mB["rc3"]["sops_est_vs_base_orig4"])} 다. '
  f'같은 1000장에서 ours 3e-3 T=4 는 baseline T=4 대비 스파이크 {PC(mO["orig4"]["spk_vs_base_orig4"])}, SOPs (추정) {PC(mO["orig4"]["sops_est_vs_base_orig4"])} (n=3 대 3) 이다. '
  f'**스파이크·SOPs 로는 \'T\'=3 + BN 재추정\' baseline 이 ours 보다 깊고, 정확도 손실은 이 규모로 ±0.2%p 마진 판정을 못 한다** (구간이 마진을 걸침). '
  f'ours 도 같이 줄이면 (T\'=3 + BN 재추정) baseline T=4 대비 스파이크 {mO["rc3"]["spk_vs_base_orig4"]["pct"]:.1f}%, 자기 T=4 대비 {CK(mO["rc3"]["dacc_vs_own_orig4"])} (n=3) 이나, 겹쳐 쓸 수 있다는 근거로는 쓰지 않는다 (3b 절 단서).')
w(f'4. **같은 스파이크에서 SOPs 차는 1% 미만이고, 측정 축에 따라 0 을 포함하거나 +0.5% 다.** ln SOPs 곡선 (200–420K, 서버 항 포함). npz 축: ours − BPSR {SC(sB)} (n={sB["n"]}), ours − plain L2 {SC(sL)} (n={sL["n"]}). '
  f's_count 축: ours − BPSR {SC(sBc)}, ours − plain L2 {SC(sLc)}. s_count 축의 +0.5% 는 같은 s_count 에서 ours 의 npz 스파이크가 {aL["ours_minus_other_pct"]:+.2f}% [{aL["ci95_pct"][0]:+.2f}, {aL["ci95_pct"][1]:+.2f}] 많은 측정 축 차이와 크기가 맞다. 경쟁 가중이 비싼 층을 골라 줄인다는 근거는 어느 축에도 없다.')
w(f'5. **표1 점끼리의 SOPs 순위는 말할 수 없다.** 표1 점에서는 ours 의 SOPs 가 BPSR 보다 {PC(pB["sops_pct"])} 많다 (n=4 대 4). 그러나 착지 (λ 선택) 차이다. '
  f'±0.2%p 동등성을 통과한 가장 많이 줄인 n≥3 점끼리 보면 순서가 뒤집힌다: ours 4e-3 SOPs {PC(vD["sops"])} (n=3 대 3), BPSR 1e-10 {PC(vB["sops"])} (n=4 대 3), plain L2 1.5e-7 은 npz 가 없다.')
w(f'6. **문헌 (UniSparse 정정).** UniSparse 자체의 밀집 SEW-ResNet-19 C10 T=4 는 SOPs 923.01M, FR 0.0683 (n=3) 이다. 우리 baseline SOPs ({g0["npz"]["sops"][0] / 1e6:.1f}M, n=3) 는 그 {1 / ue["dense_sops_ratio_us_over_our_base"]:.2f}배로 더 작다. '
  f'UniSparse 39.47M 과 ours {gO["npz"]["sops"][0] / 1e6:.0f}M 의 {gO["npz"]["sops"][0] / 1e6 / uni["sops_after"]:.1f}배 차는 가중치 프루닝 (밀도 7.29%) 에서 온다 (UniSparse 자체는 SOPs {ue["sops_red_us_5e7"]:.1f}% 인데 FR 은 {ue["fr_red_us_5e7"]:.1f}% 만 준다). '
  f'스파이크만 보면 UniSparse 는 FR {ue["fr_red_us_5e7"]:.1f}% @ −0.45%p (n=3, 프루닝 결합), ours 는 FR {vO["fr_step"]["pct"]:.1f}% @ −0.01%p (n=4) 다. 모델 구현이 달라 판정하지 않는다. 엄밀하게 비교 가능한 문헌 행은 여전히 없다 (5절 "비교 가능?" 열).')
w(f'7. **에너지 추정 (연산 수 프록시, 0.9 pJ/AC · 4.6 pJ/MAC, 45nm).** ours (n=4 대 baseline npz n=3) 는 {PC(vO["energy_uJ"])} (첫 층 MAC 을 매 스텝 셈) / {PC(vO["energy_uJ_conv1_once"])} (한 번만 셈). 첫 층 MAC 의 몫이 baseline {g0["npz"]["conv1_mac_energy_share"][0]:.1f}% → ours {gO["npz"]["conv1_mac_energy_share"][0]:.1f}% 로 커진다. 막전위 갱신·메모리 이동은 넣지 않았고, 실측과 비교하는 문구는 쓰지 않는다.')
w('')

# ---- 0. 정의·검산
fc = FAN
h1, h2, sc = fc['hand_conv2_block1_conv1_n'], fc['hand_conv2_block3_out_n'], fc['sum_check_conv2_block1_conv1_n']
n_t = [r for r in ROWS if r['npz'] and r['npz'].get('has_t')]
n_eq = sum(1 for r in n_t if r['npz'].get('nin_elementwise_equal'))
w('## 0. 정의와 fan-out 검산')
w('')
w(f'- **스파이킹 층**: n_in (아날로그 직접 입력, `input_spike_mode=REAL`) 과 predictions_n (막전위를 내는 출력 적분기) 을 뺀 18층. 뉴런 N = 7·(32·32·128) + 6·(16·16·256) + 4·(8·8·512) + 256 = **{N_NEURON:,}**.')
w('- **FR** = S / (N·T): 뉴런당 스텝당 스파이크 (S = 이미지당 스파이크, T=4 합). **FR_T** = S / N. AP (원문 4.1절 식: 테스트 샘플·뉴런·스텝 전체의 평균) 와 UniSparse (코드 `us_main.py` 465–522행: 각 뉴런층의 (x≠0) 평균을 원소 수로 가중) 둘 다 이 전체 FR 정의와 같은 꼴임을 확인했다. '
  'UniSparse 의 FR 에는 출력 앞 LIF 가 빠진다 (SparseNeuron 만 감시). 층마다 FR 을 내고 층을 같은 무게로 평균하면 값이 작아진다 (baseline {:.4f} 대 전체 {:.4f}, n=3, −18%).'.format(g0['npz']['fr_step_layer_mean'][0], g0['npz']['fr_step'][0]))
w('- **SOPs** = Σ_층 Σ_위치 (그 위치의 스파이크 수, 채널 합) × fan-out(위치). fan-out = 그 뉴런 스파이크 하나가 다음 가중치 층에서 일으키는 누산 수 (패딩 0 은 세지 않음). 이미지당, T 합. UPR·UniSparse 의 SOPs 정의와 같은 꼴이다. UniSparse 코드 (`SOPMonitor`) 는 입력 스파이크를 1 로 채운 마스크 커널로 conv2d 해 합하므로 패딩을 세지 않는 점도 같다. 다만 conv 층만 세고 첫 conv 를 빼며, SEW 덧셈 뒤의 정수 입력 (예: 2) 을 값만큼 센다.')
w('  - 주 규약: BN 은 앞 conv 에 접힘 (0). identity shortcut 의 Add 는 0. 단계 입구 1×1 shortcut (conv0) 은 셈. GAP 는 fc1 에 접음 (conv4_block2_out_n 스파이크 하나 → fc1 256 누산). fc1_n → predictions 10. predictions_n → 0. 이진 스파이크가 일으키므로 모두 AC (누산) 다 (스텝별 배열의 원소 최대값이 n_samples 이하라 이진임을 확인).')
w('')
w('**SOPs 규약별 값 (baseline npz n=3, ours 3e-3 n=4)** — 감소율은 규약에 강건하다 (차 ≤0.4%p). 절대값은 규약에 따라 약 ±7% 움직이고, 구현이 다르면 그보다 더 움직인다 (UniSparse 밀집 R19 의 스파이크당 SOPs 는 우리의 약 1.7배, 5절).')
w('')
w('| 규약 | baseline SOPs (M) | 주 규약 대비 | ours 3e-3 SOPs (M) | 주 규약 대비 | ours 감소율 [95%] |')
w('|---|---|---|---|---|---|')
for lab, k, kv in (('주 규약 (위치별 fan-out, 패딩 제외)', 'sops', 'sops'),
                   ('B. identity Add 를 스파이크당 1 AC', 'sops_varB_identity', None),
                   ('C. GAP 를 접지 않음 (GAP 누산 1 AC, fc1 밀집 MAC 0.52M 따로)', 'sops_varC_gap_acs', None),
                   ('D. 층 FR × 패딩 포함 밀집 FLOPs (spikingjelly·syops 식)', 'sops_varD_flops', 'sops_varD_flops'),
                   ('E. conv 만 셈 (GAP→fc1, fc1→출력 뺌; UniSparse 코드에 가까움)', 'sops_varE_conv_only', 'sops_varE_conv_only')):
    b_, o_ = g0['npz'][k][0], gO['npz'][k][0]
    red = PC(vO[kv]) if kv else '–'
    w(f'| {lab} | {b_ / 1e6:.2f} | {100 * (b_ / g0["npz"]["sops"][0] - 1):+.2f}% | {o_ / 1e6:.2f} | {100 * (o_ / gO["npz"]["sops"][0] - 1):+.2f}% | {red} |')
w('')
w(f'- 뉴런마다 스텝마다 붙는 편향 덧셈 (N·T = {N_NEURON * T / 1e6:.2f}M) 은 어느 규약에도 넣지 않았다.')
w(f'- **첫 층 (직접 입력)**: conv1_conv 는 실수 입력을 받으므로 MAC 이다. 입력이 네 스텝 모두 **원소 단위로 같다** (`n_in__t`, 스텝별 배열이 있는 npz {n_eq}/{len(n_t)}개). '
  f'밀집 MAC 은 패딩 포함 32·32·128·(3·3·3) = {CONV1_MAC_STEP:,}/스텝 (thop 식; UniSparse·AP 코드가 MACs 를 세는 규약과 같다), 패딩을 빼면 94·94·3·128 = {CONV1_MAC_STEP_NOPAD:,} (−4.1%) 다. '
  f'즉 SOPs (패딩 제외) 와 첫 층 MAC (패딩 포함) 은 규약이 다르다. 이 문서는 패딩 포함 값을 쓴다. 매 스텝 다시 계산하면 {CONV1_MAC_STEP * T:,}, 한 번만 계산하면 {CONV1_MAC_STEP:,} 이다. 방법과 무관하고 SOPs (AC) 에는 넣지 않는다.')
w(f'- **밀집 동등**: 모든 뉴런이 매 스텝 쏘면 SOPs 는 {fc["dense_equiv_sops_per_step"] / 1e9:.3f}G/스텝이다. baseline 의 SOPs 는 그 {100 * g0["npz"]["sops"][0] / (fc["dense_equiv_sops_per_step"] * T):.1f}%, ours 는 {100 * gO["npz"]["sops"][0] / (fc["dense_equiv_sops_per_step"] * T):.1f}%.')
w('')
w('**fan-out 출처와 검산 (다섯 겹)**')
w('')
w(f'1. 구조는 `models/resnet.py` 를 읽어 만들었다: ResNet19 = 초기 128채널, 블록 [3,3,2], block_basic (conv1 3×3 stride same → BN → conv1_n → conv2 3×3 same → BN → Add → out_n, sew=False), 각 단계 block1 만 1×1 conv0 shortcut (lib_snn Conv2D 기본 padding=\'valid\'), conv2 단계 stride 1·나머지 stride 2, GAP → fc1 Dense(256) → fc1_n → predictions Dense(10).')
w(f'2. 실제 Keras 그래프 (`analysis/prune_measure.py` 가 체크포인트를 올려 뽑은 가중치 층 22개: 이름·커널 모양·출력 위치 수·src/dst 스파이킹 층) 와 **{"일치 (22개 모두)" if fc.get("keras_graph_match") else "불일치"}** (`{os.path.basename(fc.get("keras_graph_file", ""))}`).')
w(f'3. 기존 `analysis/r19_fanout.py` (10-09 SynOps 분석에 쓴 것) 의 지도와 **{"모든 층에서 같다" if fc["equals_r19_fanout_py"] else "다르다"}**. baseline SOPs/스파이크 {g0["npz"]["sops_per_spk"][0]:.1f} (n=3) 는 10-09 보고 1,286.1 과 같다.')
tfc = fc['tf_grad_check']
if isinstance(tfc, list):
    w(f'4. TF (CPU) 로 패딩 규칙을 독립 검산했다: d Σ conv2d(x, 1) / dx = 그 입력을 읽는 출력 위치 수. 3×3 s1/s2 SAME (32, 16, 8), 1×1 s2 VALID 여섯 경우 모두 최대 차 {max(t["max_abs_diff"] for t in tfc):.0f}.')
tcs = fc.get('tf_conv_sops_check')
if isinstance(tcs, list) and tcs:
    w(f'5. SOPs 자체도 다른 경로로 다시 셌다: 실제 Keras 가중치 층 목록을 따라 각 층 입력의 위치별 스파이크 지도를 TF conv2d (커널 전부 1) 에 통과시킨 합 × C_out. {len(tcs)}런 ({", ".join(t["tag"].replace("r19c10-", "") for t in tcs)}) 에서 상대 차 최대 {max(abs(t["rel_diff"]) for t in tcs):.1e}. (eip-critic 이 numpy 로 따로 짠 fan-out 으로도 15런 모두 같았다.)')
w('')
w('**손 계산 (한 층씩)**')
w('')
w(f'- conv2_block1_conv1_n (32×32×128) → conv2_block1_conv2 (3×3, stride 1, same, 128채널): 안쪽 뉴런 3·3·128 = **{h1["interior_expected"]:,}** (코드 {h1["interior"]:,.0f}), 가장자리 2·3·128 = {h1["edge_expected"]:,} ({h1["edge"]:,.0f}), 모서리 2·2·128 = {h1["corner_expected"]:,} ({h1["corner"]:,.0f}). 한 축의 읽힘 합이 3·32 − 2 = 94 이므로 평균 (94/32)²·128 = **{h1["mean_expected"]:,.1f}** ({h1["mean"]:,.1f}).')
w(f'  - 합 검산: Σ fan-out × C_in = {sc["sum_fanout_x_Cin"]:,.0f} = 94²·128·128 = {sc["valid_macs"]:,} (패딩 0 을 뺀 밀집 MAC). 패딩까지 넣은 밀집 MAC 은 {sc["dense_macs_with_padding"]:,} 로 4.1% 크다.')
w(f'- conv2_block3_out_n (32×32×128) → conv3_block1_conv1 (3×3, stride 2, same, 256) + conv3_block1_conv0 (1×1, stride 2, valid, 256): TF same 은 앞 패딩 0·뒤 패딩 1 이라 짝수 좌표는 두 출력, 홀수 좌표는 한 출력이 읽는다. 짝·짝 안쪽 2·2·256 + 256 = **{h2["even_even_interior_expected"]:,}** ({h2["even_even_interior"]:,.0f}), 홀·홀 1·1·256 = {h2["odd_odd_expected"]} ({h2["odd_odd"]:,.0f}), 짝·홀 2·1·256 = {h2["even_odd_expected"]} ({h2["even_odd"]:,.0f}). 평균 (47/32)²·256 + ¼·256 = **{h2["mean_expected"]:.2f}** ({h2["mean"]:.2f}).')
mf = fc['mean_fanout']
w(f'- 층 평균 fan-out (참고): conv1_conv_n {mf["conv1_conv_n"]:,.1f}, 32×32 블록 안 {mf["conv2_block1_conv1_n"]:,.1f}, conv2→conv3 입구 {mf["conv2_block3_out_n"]:,.2f}, 16×16 블록 안 {mf["conv3_block1_conv1_n"]:,.1f}, conv3→conv4 입구 {mf["conv3_block3_out_n"]:,.2f}, 8×8 블록 안 {mf["conv4_block1_conv1_n"]:,.1f}, conv4_block2_out_n (→fc1) {mf["conv4_block2_out_n"]:,.0f}, fc1_n {mf["fc1_n"]:,.0f}. 32×32 층이 16×16·8×8 층보다 스파이크당 1.7~3.5배 싸다.')
w('')


# ---- 1·2. 런별 표
def run_table(groups, title, note):
    w(title)
    w('')
    w(note)
    w('')
    w('| 묶음 | 시드 | 서버 | train loss | train acc | val loss | val acc | spikes | npz 스파이크 | FR (/뉴런/스텝) | FR_T (/뉴런) | SOPs (M) | SOPs/스파이크 | 비고 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for gname in groups:
        rows = [r for r in ROWS if r['group'] == gname]
        for r in rows:
            n = r['npz']
            if n:
                cells = f'{K(n["npz_spk"])} | {n["fr_step"]:.4f} | {n["fr_T"]:.4f} | {n["sops"] / 1e6:.1f} | {n["sops_per_spk"]:,.0f}'
                note_ = '' if abs(r['acc_gap']) <= 0.02 else f'npz 정확도 차 {r["acc_gap"]:+.2f}'
            else:
                cells = '– | – | – | – | –'
                note_ = ''
            rsn = r['reason'] if r['reason'] else note_
            if not r['valid']:
                rsn = f'**평균에서 뺌**: {r["reason"]}'
            w(f'| {gname} | s{r["seed"]} | {r["server"]} | {F(r["train_loss"], 4)} | {F(r["train_acc"])} | {F(r["val_loss"], 4)} | {F(r["val_acc"])} | {K(r["spikes"])} | {cells} | {rsn} |')
        g = G[gname]; L_, N_ = g['ledger'], g['npz']
        w(f'| **{gname} 평균** | n={g["n_valid_ledger"]} (npz n={g["n_npz"]}) | | {MS(L_["train_loss"], 4)} | {MS(L_["train_acc"], 2)} | {MS(L_["val_loss"], 4)} | **{MS(L_["val_acc"], 2)}** | **{MS(L_["spikes"], 0)}** | {MS(N_["npz_spk"], 0)} | **{MS(N_["fr_step"], 4)}** | {MS(N_["fr_T"], 4)} | **{MS(N_["sops"], 1, 1e6)}** | {MS(N_["sops_per_spk"], 0)} | |')
    w('')


run_table(['baseline', 'ours_inter_ch 3e-3', 'BPSR 1e-10', 'plain L2 1e-7'],
          '## 1. 표1 점 런별 값 (R19 C10 T=4)',
          '- 앞 다섯 열 (train loss → spikes) 은 원장의 best 에폭 값이다 (spikes = s_count, 학습 패스 통계). npz 열은 테스트 1만 장 평가 모드 값이다. 평균 행의 ± 는 시드 간 SD.\n'
          '- 평균 행의 n: 원장 열은 완주·비배제 런 전부, npz 열은 npz 가 있는 런만. **baseline 의 npz 는 138 서버 s11~s13 3런뿐이다** (s14~s16 은 세종 런이라 추출하지 않았다). 그래서 npz 열의 baseline 평균은 부분집합 (n=3) 이다.\n'
          '- 모든 npz 의 테스트 정확도는 원장 val acc 와 0.02%p 안이다 (val 셋 = 테스트 셋).')
run_table(['ours_inter_ch 4e-3'],
          '### 1b. ±0.2%p 동등성을 통과한 가장 많이 줄인 n≥3 점 (참고)',
          '- `results_261019.md` 요약 2: ±0.2%p 동등성 (TOST) 통과 최심점은 ours_inter_ch 4e-3 (s_count −38.2%, n=3), plain L2 1.5e-7 (−39.9%, n=3), BPSR 1e-10 (−34.1%, n=4, 1절) 이다.\n'
          '- ours 4e-3 의 npz 는 `scratchpad/deep1005/npz` (10-05 추출) 다. s4 는 원장 스냅숏 시점에 미완주라 뺐다. 서버가 23 둘·138 하나로 baseline npz (138 셋) 와 섞여 있다.\n'
          '- plain L2 1.5e-7 (s1 23, s2·s3 138) 은 npz 가 없어 이 표에 넣지 못했다 (추출 필요, GPU 승인).')
run_table(['- vmem 2e-3', '- final_step 1.4e-3', '- 1- inversion 4.5e-3', '- loss-ratio 1.25e-7'],
          '## 2. 342K ablation 세트 런별 값',
          '- 구성요소를 하나씩 끈 팔을 ours_inter_ch 3e-3 (342K, 1절) 근처에 착지시킨 세트다 (`analysis/ablation_261019.md`). 같은 ρ 가 아니라 같은 착지를 노린 설계점이다.\n'
          '- `- final_step` s1·s2 는 미완주 (215·214 에폭) 라 평균에서 뺐다. `- loss-ratio` s1 은 세종 런이라 npz 가 없어 npz 열은 n=3 이다.')

# ---- 3. baseline 대비
srv = SRV
w('## 3. baseline 대비 감소 (세 측정 축)')
w('')
w('- 칸 = 100·(묶음 평균 / baseline 평균 − 1) [95% CI]. CI 는 두 묶음의 시드 산포를 ln 비의 델타법으로 함께 전파했다 (Welch 자유도). npz 계열 열의 baseline 은 npz n=3 (138 서버만), s_count 열의 baseline 은 원장 n=6.')
w('- "SOPs/스파이크 감소비" = SOPs 감소율 / npz 스파이크 감소율. 1 보다 작으면 SOPs 가 스파이크보다 덜 준다. CI 없는 기술값이고 감소 깊이에 따라 변한다 (4절 탄력성 0.84 → 깊이 줄일수록 비가 변함). 팔 사이 0.01 안팎의 차는 해석하지 않는다.')
w('- Δacc = 묶음 − baseline (원장 best val acc, n=6), Welch 95% CI.')
w('')
w('| 묶음 | n (npz / 원장) | s_count 축 | npz 스파이크 = FR | SOPs | SOPs/스파이크 | SOPs/스파이크 감소비 | 에너지 추정 (첫 층 매 스텝) | Δacc %p |')
w('|---|---|---|---|---|---|---|---|---|')
for gname in [k for k in VS_BASE]:
    v = VS_BASE[gname]
    w(f'| {gname} | {G[gname]["n_npz"]} / {G[gname]["n_valid_ledger"]} | {PC(v["scount_n6"])} | {PC(v["npz_spk"])} | {PC(v["sops"])} | {PC(v["sops_per_spk"])} | {v["sops_to_spk_reduction"]:.3f} | {PC(v["energy_uJ"])} | {v["dacc_n6"]["d"]:+.3f} [{v["dacc_n6"]["ci95"][0]:+.3f}, {v["dacc_n6"]["ci95"][1]:+.3f}] |')
w('')
w(f'- npz 축의 FR 감소율은 npz 스파이크 감소율과 같은 수다 (N·T 가 고정). ours 의 s_count 축 {vO["scount_n6"]["pct"]:.1f}% 와 npz 축 {vO["npz_spk"]["pct"]:.1f}% 의 차 {vO["npz_spk"]["pct"] - vO["scount_n6"]["pct"]:.1f}%p 중 약 {vO["scount_npzsubset"]["pct"] - vO["scount_n6"]["pct"]:.1f}%p 는 baseline 표본 차 (같은 3런 baseline 이면 s_count 축 {vO["scount_npzsubset"]["pct"]:.1f}%), 나머지는 npz/s_count 비가 baseline {srv["npz_over_scount_base138"][0]:.3f} (n=3), ours {srv["npz_over_scount_ours"][0]:.3f} (n=4) 로 다르기 때문이다.')
w(f'- **서버 민감도.** npz 축 baseline 은 138 서버 3런뿐이다. 23 서버 baseline 의 s_count 는 138 보다 {PC(srv["scount_23_vs_138"])} 많다 (n=3 대 3, Welch p={srv["scount_23_vs_138_welch_p"]:.2f}, 검출 안 됨). '
  f'138 의 npz/s_count 비 {srv["npz_over_scount_base138"][0]:.3f} 로 n=6 baseline 의 npz 를 외삽하면 ({srv["base_npz_n6_extrapolated"]:,.0f}) ours 의 npz 감소는 {srv["ours_npz_vs_extrapolated_pct"]:.1f}% 가 된다 (점추정, 외삽이라 CI 를 내지 않음). CI 에 들어가지 않은 약 1%p 의 편향이고 방향은 보수적이다 (우리 감소를 작게 보이게 함). 논문에 쓴다면 "SOPs −24% (npz 축, baseline n=3, 138 서버만)" 처럼 범위를 붙인다. baseline s14~s16 npz 를 추출하면 풀린다 (GPU 승인).')
w('- `- vmem` 은 같은 착지를 노렸지만 실제로는 스파이크가 더 남았다 (npz −22.5%, n=4). 그래서 SOPs 도 덜 줄었다. 정확도 Δ 는 이 표가 아니라 `ablation_261019.md` 의 같은 스파이크 비교로 읽는다.')
w('')
w('**SOPs 의 단계별 몫 (%, 평균 ±SD)** — 32×32 (conv1·conv2) 몫이 줄고 16×16·8×8 몫이 는다. 세 규제 팔이 같은 방향이다.')
w('')
w('| 묶음 | n | conv1 | conv2 | conv3 | conv4 | 스파이크 중 conv1 | 스파이크 중 conv2 |')
w('|---|---|---|---|---|---|---|---|')
for gname in ['baseline', 'ours_inter_ch 3e-3', 'BPSR 1e-10', 'plain L2 1e-7', 'ours_inter_ch 4e-3']:
    g = G[gname]; s_ = g['sops_stage_share']; t_ = g['stage_share']
    w(f'| {gname} | {g["n_npz"]} | {MS(s_["conv1"])} | {MS(s_["conv2"])} | {MS(s_["conv3"])} | {MS(s_["conv4"])} | {MS(t_["conv1"])} | {MS(t_["conv2"])} |')
w('')
w('**스텝별 스파이크 몫과 FR** (스텝별 배열이 있는 npz 만; n 이 묶음마다 다르다)')
w('')
w('| 묶음 | n | 스텝 1 | 스텝 2 | 스텝 3 | 스텝 4 | FR 스텝 1 → 4 |')
w('|---|---|---|---|---|---|---|')
for gname in ['baseline', 'ours_inter_ch 3e-3', 'BPSR 1e-10', 'plain L2 1e-7', '- vmem 2e-3', '- final_step 1.4e-3', '- 1- inversion 4.5e-3', '- loss-ratio 1.25e-7']:
    st = G[gname].get('step')
    if not st:
        continue
    jud = ' (n≤2, 판단하지 않음)' if st['n'] <= 2 else ''
    w(f'| {gname} | {st["n"]}{jud} | ' + ' | '.join(f'{x[0]:.1f}%' for x in st['spk_share']) +
      f' | {st["fr"][0][0]:.4f} → {st["fr"][3][0]:.4f} |')
w('')
w(f'- baseline 스텝 1 몫 {g0["step"]["spk_share"][0][0]:.1f}% (n=3) 는 `ablation_261019.json` `/npz/base_t_share` 와 같다. ours 3e-3 의 스텝별 n=3 은 s2~s4 (모두 138) 다. '
  f'**스파이크가 뒤 스텝에 몰려 있다** (스텝 4 몫 baseline {g0["step"]["spk_share"][3][0]:.1f}%, ours {gO["step"]["spk_share"][3][0]:.1f}%). '
  f'이 몫은 T=4 를 다 돈 런 안의 분포다. 시간 평균 BN 때문에 \'T 를 줄이면 이만큼 준다\' 로 바로 옮기지 않는다 (10-20 정정). '
  f'실제로 T\'=3 으로 다시 돌리면 baseline 스파이크가 {mB["fix3"]["spk_vs_own_orig4_pct"][0]:.1f}% (BN 고정) / {mB["rc3"]["spk_vs_own_orig4_pct"][0]:.1f}% (BN 재추정) 준다 (3b 절, 1000장, n=3). 스텝 비율 (25%) 보다 더 준다는 방향은 맞다.')
w('')

# ---- 3b. 추론 T 를 실제로 줄인 실측 (10-20 정정)
w('## 3b. 추론 T 를 줄이면 (학습 T=4, 추론 T\'; 모든 샘플 같은 T\') — 10-20 정정판')
w('')
w(f'- **정정.** 검토 전 판의 이 절은 npz 의 `logits_t[:, t−1]` 판독과 스텝별 배열의 앞 t 스텝 합을 \'T=t 에서 멈춘 망\' 으로 읽고 "스텝 t 까지의 계산은 뒤 스텝과 무관하다" 고 적었다. 틀렸다. '
  f'R19 의 BN 은 T 스텝 입력의 시간 평균 한 장에 걸리고 그 결과를 모든 스텝에 복사한다 (`lib_snn/model.py` f_temporal_reduction 분기). 그래서 conv2 부터 스텝 t 의 입력에 스텝 t+1~4 의 발화가 섞인다. '
  f'검토 전 값 (baseline T=3 Δ {DCI(tb3["dacc_paired_vs_own_T4"])}%p, 스파이크 {tb3["spk_vs_base_T4"]["pct"]:.1f}%, SOPs {tb3["sops_vs_base_T4"]["pct"]:.1f}%; T=2 Δ {DCI(tb2["dacc_paired_vs_own_T4"])}%p, 스파이크 {tb2["spk_vs_base_T4"]["pct"]:.1f}%) 은 실제로 돌린 어느 조건의 값도 아니다 (`early_exit_261020.md` 0·5·6절). JSON `step_readout_noncausal` 에 인용 금지 표시로만 남긴다.')
w('- 아래는 같은 체크포인트를 T\' 로 **다시 돌린** 실측이다. 원자료는 `early_exit_261020` 6절의 CPU 출력 (테스트 앞 1000장, 가중치 그대로) 이고, 이 스크립트가 다시 읽어 계산했다 '
  f'(같은 부트스트랩 B=20000; early_exit_261020.json 과 정확도·스파이크·구간 최대 차 {TM["check_vs_early_exit_json_maxabs"]:.1g}). '
  '두 조건: **BN 고정** = BN 이동통계를 T=4 값 그대로 둠. **BN 재추정** = BN 이동통계만 T\' 에서 학습셋 5000장 (학습 후반과 같은 증강) 으로 다시 잡음 (학습 없음, 가중치 그대로). '
  f'CPU 원래 T=4 예측은 GPU npz 의 같은 1000장과 {TM["gpu_vs_cpu_argmax_same_pct_range"][0]:.1f}~{TM["gpu_vs_cpu_argmax_same_pct_range"][1]:.1f}% 같다 (체크포인트 6개).')
w(f'- SOPs 는 **추정**이다. CPU 원자료에는 위치별 지도가 없어, 층별 스파이크 × 같은 체크포인트 1만 장 npz (T=4) 의 층별 SOPs/스파이크로 셌다. '
  f'점검: npz 스텝별 배열의 앞 2·3 스텝에 같은 추정을 걸면 위치별 정확값과 {TM["sops_est_check_pct_range"][0]:+.2f}~{TM["sops_est_check_pct_range"][1]:+.2f}% 다르다 (체크포인트 6개). '
  '에너지 추정 = SOPs × 0.9 pJ + 첫 층 MAC × T\' × 4.6 pJ. 스파이크는 18층 이미지당 (npz 축과 같은 정의, 1000장) 이고 원장 s_count 축이 아니다.')
w('- 앞 다섯 열 (train loss → spikes) 은 원장의 best 에폭 값이다 (spikes = s_count). 평균 행의 원장 열은 **이 3런만의 평균**이다 (원장 n: baseline 6, ours_inter_ch 3e-3 4). 그 뒤 열은 CPU 1000장 값이다.')
w('')
w('| 묶음 | 시드 | 서버 | train loss | train acc | val loss | val acc | spikes | 정확도 1000장: 원래 T=4 | BN 고정 T\'=3 | BN 재추정 T\'=4 | BN 재추정 T\'=3 | BN 재추정 T\'=2 | 스파이크 1000장: 원래 T=4 | BN 고정 T\'=3 | BN 재추정 T\'=3 | SOPs 추정 (M): 원래 T=4 | BN 재추정 T\'=3 |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for gname, A in (('baseline', mB), ('ours_inter_ch 3e-3', mO)):
    tags = [t for _, t, g in EE_RUNS if g == gname]
    for t in tags:
        r = _row_by_tag[t]; c_ = TM['runs'][t]['cond']
        w(f'| {gname} | s{r["seed"]} | {r["server"]} | {F(r["train_loss"], 4)} | {F(r["train_acc"])} | {F(r["val_loss"], 4)} | {F(r["val_acc"])} | {K(r["spikes"])} | '
          f'{c_["orig4"]["acc"]:.1f} | {c_["fix3"]["acc"]:.1f} | {c_["rc4"]["acc"]:.1f} | {c_["rc3"]["acc"]:.1f} | {c_["rc2"]["acc"]:.1f} | '
          f'{c_["orig4"]["spk"]:,.0f} | {c_["fix3"]["spk"]:,.0f} | {c_["rc3"]["spk"]:,.0f} | {c_["orig4"]["sops_est"] / 1e6:.1f} | {c_["rc3"]["sops_est"] / 1e6:.1f} |')
    sub = [_row_by_tag[t] for t in tags]
    led = {k_: msd([r[k_] for r in sub]) for k_ in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes')}
    w(f'| **{gname} 평균** | n={len(tags)} (원장 부분집합) | 138 | {MS(led["train_loss"], 4)} | {MS(led["train_acc"], 2)} | {MS(led["val_loss"], 4)} | {MS(led["val_acc"], 2)} | {MS(led["spikes"], 0)} | '
      f'**{MS(A["orig4"]["acc"], 2)}** | {MS(A["fix3"]["acc"], 2)} | {MS(A["rc4"]["acc"], 2)} | **{MS(A["rc3"]["acc"], 2)}** | {MS(A["rc2"]["acc"], 1)} | '
      f'{MS(A["orig4"]["spk"], 0)} | {MS(A["fix3"]["spk"], 0)} | **{MS(A["rc3"]["spk"], 0)}** | {MS(A["orig4"]["sops_est"], 1, 1e6)} | **{MS(A["rc3"]["sops_est"], 1, 1e6)}** |')
w('')
w('- 아래 표: "자기 원래 T=4 대비" 는 같은 체크포인트·같은 1000장의 대응 차 ([ ] 체크포인트×이미지 이원 부트스트랩 95%, 옆 열은 체크포인트 수준 t 95%, df=2). '
  '"baseline 원래 T=4 대비" 는 baseline 행이면 같은 체크포인트끼리의 짝지은 비 (체크포인트 수준 t), ours 행이면 다른 체크포인트라 독립 비교 (n=3 대 3, ln 비 델타법·Welch) 다.')
w('')
w('| 묶음, 추론 조건 | n (체크포인트) | 정확도 (1000장) | Δacc 자기 원래 T=4 대비 [이원 부트스트랩] | 체크포인트 수준 t | Δacc baseline 원래 T=4 대비 (Welch) | 스파이크 baseline 원래 T=4 대비 | SOPs 추정 baseline 원래 T=4 대비 | 에너지 추정 baseline 원래 T=4 대비 |')
w('|---|---|---|---|---|---|---|---|---|')
_cn = {'orig4': '원래 T=4', 'fix3': 'BN 고정 T\'=3', 'rc4': 'BN 재추정 T\'=4 (절차 점검)', 'rc3': 'BN 재추정 T\'=3', 'rc2': 'BN 재추정 T\'=2'}
for gname, A in (('baseline', mB), ('ours_inter_ch 3e-3', mO)):
    for cond in ('orig4', 'fix3', 'rc4', 'rc3', 'rc2'):
        e = A[cond]
        da = '–' if cond == 'orig4' else CK(e['dacc_vs_own_orig4'])
        dt = '–' if cond == 'orig4' else CKT(e['dacc_vs_own_orig4'])
        if gname == 'baseline':
            dw = '– (같은 체크포인트: 왼쪽 열)'
            sp = '기준' if cond == 'orig4' else PC(e['spk_vs_base_orig4'])
            so = '기준' if cond == 'orig4' else PC(e['sops_est_vs_base_orig4'])
            en = '기준' if cond == 'orig4' else PC(e['energy_uJ_vs_base_orig4'])
        else:
            dw = DCI(e['dacc_vs_base_orig4'])
            sp, so, en = PC(e['spk_vs_base_orig4']), PC(e['sops_est_vs_base_orig4']), PC(e['energy_uJ_vs_base_orig4'])
        w(f'| {gname}, {_cn[cond]} | {e["n_ckpt"]} | {MS(e["acc"], 2)} | {da} | {dt} | {dw} | {sp} | {so} | {en} |')
w('')
w(f'- **baseline T\'=3 + BN 재추정**: 자기 원래 T=4 대비 {CK(mB["rc3"]["dacc_vs_own_orig4"])} (체크포인트별 {", ".join(f"{v:+.1f}" for v in mB["rc3"]["dacc_vs_own_orig4"]["per_ckpt_pp"])}; 체크포인트 수준 t {CKT(mB["rc3"]["dacc_vs_own_orig4"])}), '
  f'스파이크 {PC(mB["rc3"]["spk_vs_base_orig4"])}, SOPs 추정 {PC(mB["rc3"]["sops_est_vs_base_orig4"])}, 에너지 추정 {PC(mB["rc3"]["energy_uJ_vs_base_orig4"])} (n=3). '
  f'같은 1000장에서 ours 3e-3 T=4 는 baseline T=4 대비 스파이크 {PC(mO["orig4"]["spk_vs_base_orig4"])}, SOPs 추정 {PC(mO["orig4"]["sops_est_vs_base_orig4"])} (n=3 대 3; 1만 장 npz 에서는 n=4 대 3 으로 {vO["npz_spk"]["pct"]:.1f}% / {vO["sops"]["pct"]:.1f}%, 1절·3절). '
  '**스파이크·SOPs 축에서는 학습 없는 이 대조가 ours 보다 깊다.** 정확도 손실이 0 인지 0.2~1%p 인지는 1000장 × 체크포인트 3개로 가리지 못한다 (구간이 ±0.2%p 마진을 걸침).')
w(f'- **절차 자체의 효과** (BN 재추정 T\'=4 − 원래 T=4): baseline {CK(mB["rc4"]["dacc_vs_own_orig4"])}, ours {CK(mO["rc4"]["dacc_vs_own_orig4"])} (각 n=3). '
  f'T\'=3 을 같은 판 T\'=4 와 비교하면 baseline {CK(mB["rc3"]["dacc_vs_own_rc4"])}, ours {CK(mO["rc3"]["dacc_vs_own_rc4"])} 다.')
w(f'- **BN 고정 T\'=3**: baseline {CK(mB["fix3"]["dacc_vs_own_orig4"])}, ours {CK(mO["fix3"]["dacc_vs_own_orig4"])} (각 n=3). 체크포인트 6개 모두 손실 쪽이고, 체크포인트 수준 t 구간의 위 끝 (baseline {mB["fix3"]["dacc_vs_own_orig4"]["t_ckpt"][1]:+.2f}, ours {mO["fix3"]["dacc_vs_own_orig4"]["t_ckpt"][1]:+.2f}%p) 이 −0.2%p 보다 낮다. '
  '이 손실의 대부분은 BN 이동통계가 T=4 에 맞춰져 있어 생긴다 (BN 재추정으로 대부분 사라짐, early_exit_261020 6.2절).')
w(f'- **T\'=2**: BN 을 다시 잡아도 정확도가 baseline {MS(mB["rc2"]["acc"], 1)}%, ours {MS(mO["rc2"]["acc"], 1)}% 로 무너진다 (체크포인트 6개 모두 90% 미만). 검토 전 판의 \'baseline T=2 Δ {DCI(tb2["dacc_paired_vs_own_T4"])}%p\' 도 판독값이었다.')
w(f'- **ours 를 같이 줄이면** (T\'=3 + BN 재추정): baseline 원래 T=4 대비 스파이크 {PC(mO["rc3"]["spk_vs_base_orig4"])}, SOPs 추정 {PC(mO["rc3"]["sops_est_vs_base_orig4"])}, 자기 원래 T=4 대비 {CK(mO["rc3"]["dacc_vs_own_orig4"])} (n=3). '
  '손실 구간이 넓어 두 방법을 겹쳐 쓸 수 있다는 근거로 쓰지 않는다.')
w(f'- **단서.** (1) T\' 를 결과를 보고 골랐다 (T\'=3·2 만 봄). (2) 체크포인트를 T=4 테스트 정확도로 골랐다 (val 셋 = 테스트 셋). 그래서 짝지은 차 (T\' − T=4) 는 T=4 쪽에 유리하게 치우쳐 있을 수 있다. '
  '(3) 학습 서버 138 의 체크포인트 n=3/팔, 테스트 앞 1000장, CPU 추론이다. 1만 장 재측정은 early_exit_261020 8절 첫 제안 (사용자 확인 필요). '
  f'(4) BN 재추정 결과는 재추정에 쓴 데이터에 민감하다: 증강 없는 학습셋 3000장으로 잡으면 baseline T\'=3 이 {CK(_EEclean_b3)} 다 (early_exit_261020 6.2절 민감도 판). '
  f'(5) \'모든 샘플 같은 T\'\' 이다. 샘플마다 다른 스텝에서 끝내는 SEENN 식 조기 종료는 이 망 (시간 평균 BN) 에서 npz 로 모의할 수 없고, 4스텝을 다 돈 뒤 출력만 일찍 읽는 방식으로는 baseline 이 최대 {_EEmaxsave["mean"]:.2f} ±{_EEmaxsave["sd"]:.2f}% (n=3) 만 아낀다 (early_exit_261020 3절). '
  '(6) SOPs·에너지는 추정이다 (위).')
w('')

# ---- 4. 같은 스파이크 SOPs
w('## 4. 같은 스파이크에서 SOPs (곡선, 두 측정 축)')
w('')
w('- 모형: `ln SOPs = a + b·ln(스파이크) + c·[ours_inter_ch] + d·[서버 23]`, 팔 둘씩. c → 100·(e^c − 1) = 같은 스파이크에서 ours 의 SOPs 가 몇 % 많은가. 스파이크 공변량을 npz 축과 원장 s_count 축 두 가지로 넣는다 (compare-candidates 3.1 규칙 4). 런 선택은 두 축 모두 원장 s_count 구간 (주 200–420K, 민감도 300–420K), 완주·비배제·npz 있음 — 같은 런 집합이다.')
w('')
w('| 구간 | 상대 | n (ours / 상대) | 축 | 기울기 b | ours − 상대 (같은 스파이크) [95%] | p | 기울기 차 p |')
w('|---|---|---|---|---|---|---|---|')
for lab in ('200_420K', '300_420K'):
    for o in ('BPSR', 'plain L2'):
        for ax, S_ in (('npz', SAME), ('s_count', SAME_SC)):
            x = S_[lab][o]
            w(f'| {lab.replace("_", "–")} | {o} | {x["n"]} ({x["n_by_arm"]["ours_inter_ch"]} / {x["n_by_arm"][o]}) | {ax} | {x["slope"]:.3f} ±{x["slope_se"]:.3f} | {SC(x)} | {x["p"]:.3f} | {x["slope_diff_p"]:.2f} |')
w('')
rg = sL['scount_range']
w(f'- **해석.** 같은 스파이크 SOPs 차는 1% 미만이다. npz 축에서는 두 상대 모두 0 을 포함하고, s_count 축에서는 plain L2 대비 {SC(sLc)} (n={sLc["n"]}) 로 0 을 벗어난다. '
  f's_count 축의 이 차는 같은 s_count 에서 ours 의 npz 스파이크가 plain L2 보다 {aL["ours_minus_other_pct"]:+.2f}% [{aL["ci95_pct"][0]:+.2f}, {aL["ci95_pct"][1]:+.2f}] (BPSR 대비 {aB["ours_minus_other_pct"]:+.2f}% [{aB["ci95_pct"][0]:+.2f}, {aB["ci95_pct"][1]:+.2f}]) 많은 측정 축 차이와 크기가 맞다 (같은 모형, y = ln npz 스파이크). 스파이크당 비용 차라는 근거는 없다.')
w(f'- 1차 판에서 쓴 "±0.5% 안이면 같다" 는 마진은 결과를 본 뒤에 정한 값이라 판정 기준으로 쓰지 않는다. 경쟁 가중이 비싼 층을 골라 줄인다는 (같은 스파이크에서 SOPs 가 적다는) 근거는 어느 축에도 없다.')
w(f'- **외삽 주의.** plain L2 런은 s_count {rg["plain L2"][0] / 1e3:.0f}–{rg["plain L2"][1] / 1e3:.0f}K 에만 있고 ours 는 {rg["ours_inter_ch"][0] / 1e3:.0f}–{rg["ours_inter_ch"][1] / 1e3:.0f}K 에 걸쳐 있어 200–420K 비교는 공통 기울기에 기댄다. 300–420K (ours {SAME["300_420K"]["plain L2"]["scount_range"]["ours_inter_ch"][0] / 1e3:.0f}K 부터) 에서도 결론은 같다 (표).')
w('- 기울기 b 는 npz 축 0.84, s_count 축 0.79~0.80 이다: 스파이크를 10% 줄이면 SOPs 는 약 8% 준다 (세 팔 공통, 기울기 차 검출 안 됨). 3절의 감소비 0.87 과 같은 사실이다.')
w('')
w('**표1 점끼리 (n=4 대 4, Welch)** — 착지 (λ 선택) 차이를 그대로 담는다. 방법 순위로 읽지 않는다.')
w('')
w('| ours − 상대 | npz 스파이크 | FR | SOPs | SOPs/스파이크 |')
w('|---|---|---|---|---|')
for o, p_ in (('BPSR 1e-10', pB), ('plain L2 1e-7', pL)):
    w(f'| {o} | {p_["npz_spk"]["d"]:+,.0f} [{p_["npz_spk"]["ci95"][0]:+,.0f}, {p_["npz_spk"]["ci95"][1]:+,.0f}] p={p_["npz_spk"]["p"]:.3f} | {p_["fr_step"]["d"]:+.4f} | {PC(p_["sops_pct"])} p={p_["sops"]["p"]:.3f} | {PC(p_["sops_per_spk_pct"], 2)} p={p_["sops_per_spk"]["p"]:.3f} |')
w('')
w('**어느 점을 대표로 고르느냐에 따라 SOPs 깊이 순위가 바뀐다** (baseline npz n=3 대비)')
w('')
w('| 기준 | ours_inter_ch | BPSR | plain L2 |')
w('|---|---|---|---|')
w(f'| 표1 점 | 3e-3: SOPs {PC(vO["sops"])} (n=4) | 1e-10: {PC(vB["sops"])} (n=4) | 1e-7: {PC(vL["sops"])} (n=4, 합동 SD 기준 손실 쪽) |')
w(f'| ±0.2%p 동등성 통과 최심점 | 4e-3: {PC(vD["sops"])} (n=3; s_count {vD["scount_n6"]["pct"]:.1f}%) | 1e-10: {PC(vB["sops"])} (n=4; s_count {vB["scount_n6"]["pct"]:.1f}%) | 1.5e-7: npz 없음 (s_count −39.9%, n=3) |')
w('')
w('- 표1 점에서는 BPSR 이 깊고 동등성 최심점에서는 ours 가 깊다. 두 기준 모두 착지 선택에 따른 것이라 **SOPs 깊이로 방법 순위를 말할 수 없다.** 같은 스파이크 비교 (위 곡선) 가 방법 비교의 기준이다.')
w('- ablation 네 팔과 ours 3e-3 의 SOPs/스파이크 차 (n=4 대 4, `- loss-ratio` 는 3): ' + ', '.join(
    f'{k} {PAIR[k]["sops_per_spk_pct"]["pct"]:+.2f}% [{PAIR[k]["sops_per_spk_pct"]["ci95"][0]:+.2f}, {PAIR[k]["sops_per_spk_pct"]["ci95"][1]:+.2f}]'
    for k in ('- vmem 2e-3', '- final_step 1.4e-3', '- 1- inversion 4.5e-3', '- loss-ratio 1.25e-7')) +
  ' (ours − 끈 팔). 모두 CI 가 0 을 포함한다. 구성요소가 SOPs/스파이크를 바꾼다는 근거는 없다.')
w('')

# ---- 5. 문헌 비교표
w('## 5. 문헌 비교표')
w('')
w('- 우리 행은 이 분석에서 계산했다 (절대값은 머리말의 사용 제한에 걸림). 문헌 행은 CC / RA 에서 옮겼고, UniSparse·STDS 행은 원문을 직접 읽었다. 감소율은 표의 전·후 값으로 내가 계산했다.')
w('- "FR 또는 지표" 는 그 논문이 낸 지표다. SEENN·DT-SNN 의 값은 평균 시간 스텝이다 (스파이크가 아님). UniSparse 의 FR 은 원문 % 를 분수로 바꿨다.')
w('- 외부 논문이 n·± 를 적지 않았으면 "미기재" 로 둔다.')
w('')
w('| 논문 | 모델 | 데이터 | T | baseline 정확도 | 방법 정확도 | Δacc %p | FR 또는 지표 (전 → 후) | SOPs M (전 → 후) | 감소율 (지표 / SOPs) | n·± | 비교 가능? | 출처 |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for gname, v in (('ours_inter_ch 3e-3', vO), ('BPSR 1e-10', vB), ('plain L2 1e-7', vL)):
    g = G[gname]
    w(f'| **{gname} (우리)** | ResNet-19 (128ch) | CIFAR-10 | 4 | {g0["ledger"]["val_acc"][0]:.2f} (n=6) | {g["ledger"]["val_acc"][0]:.2f} (n={g["n_valid_ledger"]}) | {v["dacc_n6"]["d"]:+.2f} [{v["dacc_n6"]["ci95"][0]:+.2f}, {v["dacc_n6"]["ci95"][1]:+.2f}] | FR {g0["npz"]["fr_step"][0]:.4f} → {g["npz"]["fr_step"][0]:.4f} | {g0["npz"]["sops"][0] / 1e6:.1f} → {g["npz"]["sops"][0] / 1e6:.1f} | {PC(v["fr_step"])} / {PC(v["sops"])} | 시드 n={g["n_npz"]} 대 baseline npz n=3 | 기준 (자기 비교) | 이 분석 |')
w(f'| **baseline, 추론 T\'=3 + BN 재추정 (학습 T=4, 학습 없음) (우리)** | ResNet-19 (128ch) | CIFAR-10 | 4→3 | {mB["orig4"]["acc"][0]:.2f} (CPU 1000장, n=3) | {mB["rc3"]["acc"][0]:.2f} (n=3) | {CK(mB["rc3"]["dacc_vs_own_orig4"])} (짝지은 차, 이원 부트스트랩) | 이미지당 스파이크 (1000장) {mB["orig4"]["spk"][0]:,.0f} → {mB["rc3"]["spk"][0]:,.0f} | {mB["orig4"]["sops_est"][0] / 1e6:.1f} → {mB["rc3"]["sops_est"][0] / 1e6:.1f} (추정) | {PC(mB["rc3"]["spk_vs_base_orig4"])} / {PC(mB["rc3"]["sops_est_vs_base_orig4"])} (추정) | 같은 체크포인트 n=3 (138), 테스트 앞 1000장 | 기준 (자기 비교). T\' 사후 선택. BN 이동통계 재추정이 조건이다 (T=4 값 그대로면 {CK(mB["fix3"]["dacc_vs_own_orig4"])}). 정확도 손실은 ±0.2%p 마진 판정 못 함. 검토 전 판의 \'추론 T=3\' 행 (판독값) 을 10-20 에 이 실측으로 바꿈 | 이 분석 3b 절, `early_exit_261020.md` 6절 |')
for L in LIT:
    T_ = L['T'] if L['T'] else '미기재' if L['paper'].startswith('STDS') else (L['T'] if L['T'] else '미확인')
    _d = L.get('dec') or {'FR': 3, '평균 T': 2}.get(L['metric'], 2)
    _f = (lambda x: f'{x:.{_d}f}')
    _fs = (lambda x: f'{x:,.0f}' if L['metric'] == 'SynOps' else f'{x:,.2f}')
    if L['metric'] in ('FR', '평균 T'):
        met = (f'{L["metric"]} {_f(L["before"])} → {_f(L["after"])}' if L.get('before') is not None else
               (f'{L["metric"]} → {_f(L["after"])}' if L.get('after') is not None else L['metric']))
    else:
        met = L['metric']
    sops = (f'{_fs(L["sops_before"])} → {_fs(L["sops_after"])}' if L.get('sops_before') and L.get('sops_after') else
            (f'? → {_fs(L["sops_after"])}' if L.get('sops_after') else '–'))
    red = (f'{L["red_metric_pct"]:+.1f}%' if L.get('red_metric_pct') is not None else '–') + ' / ' + \
          (f'{L["red_sops_pct"]:+.1f}%' if L.get('red_sops_pct') is not None else '–')
    if L['metric'] == 'SynOps':
        red = f'– / {L["red_sops_pct"]:+.1f}% (SynOps)'
    dacc = f'{L["dacc"]:+.2f}' if L.get('dacc') is not None else '–'
    w(f'| {L["paper"]} ({L["venue"]}) | {L["model"]} | {L["data"]} | {T_} | {F(L["base"]) if L.get("base") is not None else "–"} | {F(L["meth"]) if L.get("meth") is not None else "–"} | {dacc} | {met} | {sops} | {red} | {L["n"]} | {L["comp"]} | {L["src"]} |')
w('')
w('- 행 순서: 우리 네 행, Activity Pruning, UPR, UniSparse, SCA, STDS, ESL-SNN, SEENN, DT-SNN, Sorbaro, Spik4lite, SRSC, MI-TRQR, DSN.')
w('- Suetake (TMLR 2024) 는 우리 TF 구현 두 점이 있지만 npz 가 없어 이 표에서 뺐다. 별도 추출 뒤 우리 행과 같은 방식으로 더한다.')
w(f'- **UniSparse 와 우리 R19 의 절대 SOPs.** 밀집끼리 보면 UniSparse 923.01M (n=3) 이 우리 baseline {g0["npz"]["sops"][0] / 1e6:.1f}M (npz n=3) 의 {ue["dense_sops_ratio_us_over_our_base"]:.2f}배다. '
  f'FR 은 오히려 UniSparse 가 낮다 (0.0683 대 {g0["npz"]["fr_step"][0]:.4f}). 공개 코드로 센 FR 감시 뉴런 수 {ue["N_us"]:,} 와 T=4 로 이미지당 스파이크를 되돌리면 약 {ue["spk_per_img_dense"]:,.0f} 이고, 스파이크당 SOPs 는 약 {ue["sops_per_spk_dense"]:,.0f} 로 우리 baseline {g0["npz"]["sops_per_spk"][0]:,.0f} 의 {ue["ratio_sops_per_spk_vs_our_base"]:.1f}배다 (우리 N 을 쓰면 {ue["ratio_sops_per_spk_vs_our_base_our_N"]:.1f}배). '
  f'원인 후보는 SEW 덧셈이다: identity 경로로 스파이크가 정수로 더해져 뒤 층들이 같은 스파이크를 다시 읽고, SOPMonitor 는 그 정수 값만큼 센다 (따로 재지 않음). 그러니 UniSparse 39.47M 과 ours {gO["npz"]["sops"][0] / 1e6:.0f}M 의 {gO["npz"]["sops"][0] / 1e6 / uni["sops_after"]:.1f}배 차는 "우리 SOPs 가 크다" 가 아니라 가중치 프루닝 (밀도 7.29%; SOPs 는 {ue["sops_red_us_5e7"]:.1f}% 인데 FR 은 {ue["fr_red_us_5e7"]:.1f}% 만 준다) 에서 온다.')
_ap = {L['data']: L for L in LIT if L['paper'] == 'Activity Pruning (AT-LIF)'}
_r = {k: (v['red_metric_pct'], v['red_sops_pct'], v['red_sops_pct'] / v['red_metric_pct']) for k, v in _ap.items()}
w(f'- 문헌에서도 SOPs 감소가 FR 감소보다 작은 경우가 많다: Activity Pruning ImageNet-100 FR {_r["ImageNet-100"][0]:.1f}% / SOPs {_r["ImageNet-100"][1]:.1f}% (비 {_r["ImageNet-100"][2]:.2f}), DVS {_r["CIFAR10-DVS"][0]:.1f}% / {_r["CIFAR10-DVS"][1]:.1f}% ({_r["CIFAR10-DVS"][2]:.2f}). 우리 비 {red_all["ours_inter_ch 3e-3"]:.2f} (n=4) 는 그 범위다. CIFAR-10 R20 행은 {_r["CIFAR-10"][0]:.1f}% / {_r["CIFAR-10"][1]:.1f}% ({_r["CIFAR-10"][2]:.2f}) 로 거의 같다. UniSparse 는 프루닝이 섞여 이 비를 낼 수 없다.')
w('')

# ---- 6. 약해 보이는 곳
w('## 6. SOPs 축에서 우리가 약해 보이는 곳')
w('')
w(f'1. **학습 없이 추론 T 를 한 스텝 줄이고 BN 이동통계만 다시 잡은 baseline 이 스파이크·SOPs 로는 ours 보다 깊다 (10-20 정정).** 실측 (CPU, 테스트 앞 1000장, 체크포인트 n=3, 138): '
  f'스파이크 {PC(mB["rc3"]["spk_vs_base_orig4"])}, SOPs 추정 {PC(mB["rc3"]["sops_est_vs_base_orig4"])}, 자기 T=4 대비 {CK(mB["rc3"]["dacc_vs_own_orig4"])}. 같은 1000장에서 ours 3e-3 T=4 는 스파이크 {mO["orig4"]["spk_vs_base_orig4"]["pct"]:.1f}%, SOPs 추정 {mO["orig4"]["sops_est_vs_base_orig4"]["pct"]:.1f}% (n=3 대 3) 다. '
  f'정확도 손실은 ±0.2%p 마진 판정을 못 한다. BN 을 T=4 값으로 두면 {CK(mB["fix3"]["dacc_vs_own_orig4"])} 라 이 대조는 \'BN 재추정\' 조건이 붙어야 성립한다. '
  f'compare-candidates 체크리스트의 "더 센 상대" (SEENN·DT-SNN 계열의 가장 단순한 형태) 후보가 우리 체크포인트 안에 있다. 단서: T\' 사후 선택, T=4 기준 best 에폭 (낙관적), 138 n=3, 1000장. '
  f'검토 전 판의 "T=3 Δ {DCI(tb3["dacc_paired_vs_own_T4"])}, 스파이크 {tb3["spk_vs_base_T4"]["pct"]:.1f}%" 와 "ours 를 T=3 에서 끊으면 Δ {DCI(to3["dacc_paired_vs_own_T4"])}" 는 비인과 판독이라 인용하지 않는다. 판정에는 1만 장 재측정이 필요하다 (early_exit_261020 8절).')
w(f'2. **헤드라인 −30% 가 SOPs 로는 −24% 다.** s_count {vO["scount_n6"]["pct"]:.1f}% (n=4 대 6) → npz 스파이크 {vO["npz_spk"]["pct"]:.1f}% → SOPs {vO["sops"]["pct"]:.1f}% (n=4 대 3) → 에너지 추정 {vO["energy_uJ"]["pct"]:.1f}%. 측정 축을 바꿀 때마다 작아진다. npz 축 baseline 은 138 서버 n=3 뿐이고, n=6 으로 외삽하면 npz 감소는 {srv["ours_npz_vs_extrapolated_pct"]:.1f}% 다 (3절).')
w(f'3. **SOPs 깊이로 방법 순위를 말할 수 없다.** 표1 점에서는 BPSR 1e-10 ({vB["sops"]["pct"]:.1f}%) 이 ours 3e-3 ({vO["sops"]["pct"]:.1f}%) 보다 깊고, 동등성 최심점에서는 ours 4e-3 ({vD["sops"]["pct"]:.1f}%, n=3) 이 BPSR 1e-10 보다 깊다. 둘 다 착지 선택이다. plain L2 1.5e-7 은 npz 가 없다.')
w(f'4. **경쟁 가중이 SOPs 를 따로 줄이지 않는다.** 같은 스파이크 SOPs 차가 1% 미만이고 (npz 축 n=29·25 에서 0 포함, s_count 축 plain L2 대비 {sLc["ours_minus_other_pct"]:+.2f}% 는 ours 가 더 쓰는 방향), 감소가 fan-out 이 작은 32×32 층에 몰리는 것도 BPSR·plain L2 와 같다. "에너지에 맞춘 규제" 라고 쓸 근거가 없다 (Suetake·Sorbaro·UniSparse 는 fan-out 으로 가중한다).')
w(f'5. **가중치 프루닝과 결합한 행에는 절대 SOPs 로 한 자릿수 이상 진다.** UniSparse SEW-R19 C10 T=4 는 94.71% 에서 39.47M, 91.56% 에서 6.11M (n=3) 이고 ours 는 {gO["npz"]["sops"][0] / 1e6:.0f}M 이다 ({gO["npz"]["sops"][0] / 1e6 / uni["sops_after"]:.1f}배). 그러나 밀집끼리는 우리 baseline 이 UniSparse 밀집보다 작고 ({1 / ue["dense_sops_ratio_us_over_our_base"]:.2f}배), 이 차는 모델이 아니라 프루닝에서 온다 (5절). '
  f'Activity Pruning R20 C10 은 92.05% 에서 61.70M ({gO["npz"]["sops"][0] / 1e6 / ap1["sops_after"]:.1f}배) 이지만 밀집 256.69M 의 작은 모델이다.')
w('6. **감소 깊이가 얕은 쪽이다.** 정확도 손실 0.25%p 이내 행끼리 SOPs (또는 같은 꼴 지표) 감소: UPR −95.1% (C10 6Conv T=8, 프루닝), Sorbaro −26.9% (변환 SNN), 우리 −23.8% (n=4). 손실을 0.5~1%p 안팎까지 허용하면 UniSparse −95.7% (−0.45%p, 프루닝 결합, FR 은 −19.2%), Activity Pruning −76.0% (−0.84%p), Spik4lite −60.9% (−1.09%p). 우리는 s_count 축 −47~−51% 부터 손실이 시작된다 (CC 0절, n=3).')
w(f'7. **첫 층 MAC 이 고정 비용으로 남는다.** 직접 입력이라 conv1 이 이미지당 14.16M MAC (매 스텝 계산 시, 패딩 포함) 이고, 에너지 추정에서 몫이 baseline {g0["npz"]["conv1_mac_energy_share"][0]:.1f}% → ours {gO["npz"]["conv1_mac_energy_share"][0]:.1f}% 로 커진다. 스파이크 규제로는 줄일 수 없다. 입력이 스텝마다 같으므로 한 번만 계산하는 구현을 전제하면 이 몫은 1/4 이 된다 (그때 에너지 추정 {PC(vO["energy_uJ_conv1_once"])}).')
w(f'8. **FR 절대값도 유리하지 않다.** ours 의 FR {gO["npz"]["fr_step"][0]:.4f} (n=4) 는 Activity Pruning R20 의 무손실에 가장 가까운 행 0.053 과 UniSparse λ 5e-7 점 0.0552 (n=3) 보다 높다 (UniSparse 밀집 0.0683 보다는 낮고, 우리 baseline {g0["npz"]["fr_step"][0]:.4f} 은 그보다 높다). 모델·구현이 달라 FR 절대값끼리는 판정하지 않는다.')
w('9. **절대 FR·SOPs 는 09-15 보류에 걸린다.** npz 절대값 자체이므로 보류가 풀리기 전에는 논문 표에 넣지 않는다 (머리말).')
w('')
w(f'**우리 쪽에 남는 것 (과장하지 않음)**: 정확도 손실이 검출되지 않는 구간 (Δ −0.01 [−0.18, +0.17], n=4 대 6) 에서의 SOPs −24% 이고, baseline 정확도가 문헌 행보다 높다 (96.73, n=6). 밀집 SOPs 도 같은 이름의 UniSparse 모델보다 작다. 그러나 같은 구간에서 BPSR·plain L2 도 같은 크기의 SOPs 감소를 내고, 학습 없이 추론 T 를 한 스텝 줄이고 BN 이동통계만 다시 잡은 baseline 이 스파이크·SOPs 로는 더 깊다 (정확도 손실은 판정 못 함, 1000장 n=3, 3b 절).')
w('')

# ---- LIMITATION
w('## [LIMITATION]')
w('')
w('- **절대 FR·SOPs 는 09-15 보류 (s_count 와 npz 11% 불일치, 원인 미상) 에 걸린다.** npz/s_count 비가 팔마다 다르다 (baseline 1.053, ours 1.077). 원인을 밝히거나 사용자가 측정 축을 정하기 전에는 논문에 쓰지 않는다.')
w('- **문헌 행은 어느 것도 엄밀하게 비교할 수 없다.** 모델·T·레시피·baseline 정확도·FR/SOPs 의 층 포함 범위 (입력층·출력층·shortcut·GAP) 가 다르고, 대부분 n 을 적지 않았다. UniSparse 는 코드로 정의를 확인했지만 SEW 덧셈의 정수 입력과 가중치 프루닝이 섞여 스파이크만의 SOPs 감소는 분리되지 않는다. 감소율 (%) 끼리만 참고로 놓았고, "이긴다·진다" 는 문헌 행과의 판정으로 쓰지 않는다.')
w('- STDS 는 AP 의 재구현값이고 원문 표는 읽지 않았다. ESL-SNN 표 값은 미확인이다. SRSC 는 초록뿐이고 심사 중이다. MI-TRQR·DSN 은 범위 값만 있다. DT-SNN·Sorbaro 는 RA 값을 이번에 다시 확인하지 않았다.')
w('- **refs-audit 정정 필요 (eip-refs 에 넘김, 이 문서에서는 고치지 않음).** `docs/paper/refs-audit.md` 821·830·842행 부근의 "UniSparse 밀집 행 없음" 은 틀렸다 — 부록 Table 4 에 R19 C10 밀집 95.16 ±0.03 @923.01M, FR 6.83% (n=3) 와 ImageNet R18 밀집 63.78 @1545.91M 이 있다.')
w(f'- **npz 축 baseline 은 n=3 (138 서버만)** 이다. 원장 baseline 은 n=6. 23 서버 baseline 의 s_count 가 +3.2% (검출 안 됨) 라 npz 계열 감소율에 약 1%p 의 보수적 편향이 있을 수 있다 (3절). 규제 팔은 138·23 이 섞였다.')
w('- **추론 T 감소 (3b) 는 CPU 1000장 × 팔당 체크포인트 3개 (138) 의 실측이다** (10-20 정정 전에는 npz 비인과 판독을 썼다). T\' 를 사후에 골랐고, best 에폭 선택이 T=4 테스트 정확도에 맞춰져 있다. BN 재추정 T\'=3 의 정확도 손실은 이원 부트스트랩 구간 폭이 약 2%p 라 정해지지 않았고, 재추정 데이터 (증강 유무) 에 1%p 안팎 민감하다. SOPs·에너지는 층별 추정이다. 이 망 (시간 평균 BN) 의 샘플별 조기 종료는 npz 로 모의할 수 없다 (early_exit_261020).')
w('- 같은 스파이크 SOPs 곡선은 서버 항만 넣었고 실행 시기 (9월/10월) 항은 넣지 않았다. plain L2 는 s_count 322–394K 에만 있어 200–420K 비교는 공통 기울기에 기댄다. 스텝별 값은 스텝별 배열이 있는 npz 만이라 BPSR n=1, plain L2 n=2 로 판단하지 않는다.')
w('- SOPs 는 연산 수 프록시다. 메모리 접근, 막전위 갱신·누설, 편향 덧셈, 희소 데이터 이동 비용은 넣지 않았다. 에너지 추정 (0.9/4.6 pJ, 45nm FP32 값) 도 같은 한계를 가진다. 실측 (Speck·Jetson) 과의 관계는 이 분석이 말할 수 없다.')
w('- 정확도 비교는 이 분석의 범위가 아니다. Δacc 는 원장 best val acc (테스트 셋에서 고른 최고값, 낙관적) 의 Welch 이고, 비열등 판정은 `method_261019.md`·`results_261019.md` 를 따른다.')
w('- C100·VGGSNN·QKFormer·SDTv3 는 계산하지 않았다 (R19 C100 npz 없음). 문헌 비교표는 R19 C10 한 설정의 값이다.')
w('')
w('**다음 (GPU 또는 긴 CPU 계산이라 모두 승인 필요, 제안만)**: (1) 같은 체크포인트로 s_count 코드 경로와 npz 를 대조해 09-15 보류의 원인 확인. (2) baseline s14~s16 npz 추출 (n=6, 서버 균형). (3) plain L2 1.5e-7 s1~s3 npz 추출 (동등성 최심점 비교 완성). (4) T\'=3 + BN 재추정 대조를 테스트 1만 장으로 다시 잰다 (early_exit_261020 8절 첫 제안; 가중치를 바꾸지 않는 추론). 검토 전 판의 (4) "샘플별 스텝별 스파이크 재추출 (SEENN 식 조기 종료 모의용)" 은 10-20 정정으로 뺐다 — 시간 평균 BN 망에서는 그 배열로도 조기 종료를 모의할 수 없다.')
w('')

# ---- 검토 반영
w('## 검토 반영 (10-20 eip-critic 1차, 14건)')
w('')
w('| # | 지적 (요약) | 판정 | 조치 |')
w('|---|---|---|---|')
w(f'| 11 | UniSparse 밀집 SOPs 가 표에 없다는 서술과 "우리 SOPs 가 12.6배 크다" 해석이 1차 출처와 어긋남 | 받아들임 (높음) | 부록 Table 4 를 직접 읽어 R19 C10 밀집 923.01M·FR 6.83%, 희소 39.47M·FR 5.52% (n=3, 각주 "시드 3회") 를 넣었다. λ 1e-6·2.5e-6 행과 ImageNet R18 밀집 63.78 @1545.91M 도 넣었다. FR·SOPs 정의를 공개 코드로 확인했다. 요약 6·5절·6절 5번을 "밀집 SOPs 는 UniSparse 밀집보다 작고 ({1 / ue["dense_sops_ratio_us_over_our_base"]:.2f}배), 차이는 가중치 프루닝에서 온다" 로 다시 썼다. 스파이크당 SOPs 는 UniSparse 가 약 {ue["ratio_sops_per_spk_vs_our_base"]:.1f}배 (코드로 센 N 사용; 우리 N 이면 {ue["ratio_sops_per_spk_vs_our_base_our_N"]:.1f}배). refs-audit 정정은 eip-refs 에 넘김 |')
w(f'| 14 | 스텝별 몫 해석이 방향을 숨김, 추론 T=3 baseline 이 ours 보다 깊음 | 받아들임 (높음) | 3b 절 신설 (logits_t 로 정확히 계산, 수치 재현: baseline T=3 Δ {DCI(tb3["dacc_paired_vs_own_T4"])}, 스파이크 {tb3["spk_vs_base_T4"]["pct"]:.1f}%), 요약 3, 5절 행, 6절 1번. 단서 넷 (사후 선택·낙관적·n=3·샘플별 종료 아님) 을 적었다. SEENN 행 문구를 "스파이크 감소가 평균 T 감소율보다 클 가능성이 높다" 로 고쳤다. **→ 10-20 정정으로 대체 (표 아래 줄): 이 판독은 실제 조건이 아니었다** |')
w('| 1–4·13 | 절대 FR·SOPs 가 09-15 보류에 걸림 | 받아들임 (보류 유지) | 머리말 "사용 제한", 요약 1, 6절 9번, LIMITATION 에 명시. 논문 표에는 넣지 않는다. 해소 방법 (a)/(b) 와 baseline s14~s16 추출을 "다음" 에 제안 (GPU 승인 필요, 실행 안 함) |')
w(f'| 요약 4·6.2 | "SOPs 무손실 대표점은 BPSR 이 깊다" 는 점 선택에 따라 뒤집힘 | 받아들임 | 그 문장을 지웠다. 동등성 최심점 ours 4e-3 의 npz 를 넣어 계산 (1b 절, SOPs {vD["sops"]["pct"]:.1f}%, n=3) 하고 4절에 두 기준 표를 두었다. plain L2 1.5e-7 은 npz 없음 |')
w(f'| 7 | 같은 스파이크 SOPs 를 npz 축으로만 봄, ±0.5% 마진 사후 설정 | 받아들임 | s_count 축을 함께 보고 (ours − plain L2 {SC(sLc)}), "1% 미만, 축에 따라 0 포함 또는 +0.5%" 로 바꿨다. 마진은 판정 기준에서 뺐다. s_count 축 차가 측정 축 차이 (같은 s_count 에서 npz 스파이크 {aL["ours_minus_other_pct"]:+.2f}%) 와 크기가 맞음을 추가로 보였다. plain L2 의 s_count 범위 (외삽) 를 적었다 |')
w(f'| 12 | 문헌에서 흔한 FLOPs 규약 누락, conv1 MAC 패딩 규약 혼합 | 받아들임 | 규약 D (층 FR × 패딩 포함 FLOPs: baseline {g0["npz"]["sops_varD_flops"][0] / 1e6:.1f}M, +{100 * (g0["npz"]["sops_varD_flops"][0] / g0["npz"]["sops"][0] - 1):.1f}%; 감소율 {vO["sops_varD_flops"]["pct"]:.1f}%) 와 E (conv 만) 를 0절 표에 넣었다. conv1 MAC 은 패딩 포함 (UniSparse·AP 코드의 MACs 규약) 을 유지하고 패딩 제외 값 3,393,024 (−4.1%) 를 함께 적었다 |')
w(f'| 5 | npz 축 baseline 이 138 만 | 받아들임 | 3절에 서버 민감도 (23 대 138 s_count {srv["scount_23_vs_138"]["pct"]:+.1f}%, p={srv["scount_23_vs_138_welch_p"]:.2f}; n=6 외삽 시 ours npz {srv["ours_npz_vs_extrapolated_pct"]:.1f}%) 와 논문 표기 문구를 넣었다 |')
w('| 6 | 감소비 0.870/0.870/0.875 는 CI 없는 기술값, 깊이에 따라 변함 | 받아들임 | 요약 2·3절에 "팔 사이 차 해석 안 함, 깊이에 따라 변함" 명시 |')
w(f'| 9 | conv1 입력이 스텝마다 같다는 근거가 약함 | 받아들임 | 원소 단위 비교로 바꿨다 (`n_in__t`, {n_eq}/{len(n_t)}개 npz 에서 네 스텝 동일) |')
w('| 10 | 에너지 프록시의 한계 | 받아들임 | "연산 수 프록시, 45nm FP32 값" 표기 유지, 실측 비교 문구 없음 |')
w('| 13 | FR 정의 확인 보강 | 받아들임 | AP 식 (원문 4.1절) 과 UniSparse 코드 (원소 수 가중 (x≠0) 평균) 가 둘 다 전체 FR 임을 0절에 적었다 |')
w('| 1 | 18층·N=1,442,048 | 통과 | 변경 없음 |')
w('| 정보 | 조기 종료 모의의 전제 (logits_t 누적, predictions_n 은 판독값 아님, 스텝별 배열은 샘플 합) | 받아들임 | 3b 절 머리와 단서 (4) 에 적었다. 이 문서의 3b 는 "모든 샘플 같은 스텝" 정확 기준선이다. **→ 10-20 정정으로 대체: 시간 평균 BN 때문에 정확 기준선이 아니었다** |')
w('| 경미 | STDS 행은 AP 재구현값 | 받아들임 | AP Table 2 원문 ("reproduced implementations") 확인 후 "AP 재구현값", T 미기재로 고쳤다 |')
w('')
w(f'- **2026-10-20 정정 (eip-stats; `early_exit_261020.md` 검토 결과 반영).** 위 14번·\'정보\' 줄의 3b 절은 npz 의 `logits_t[:, t−1]` 판독과 스텝별 배열 앞 t 스텝 합을 \'T=t 에서 멈춘 망\' 으로 읽었다. '
  f'R19 의 BN 이 T 스텝 입력의 시간 평균에 걸려 conv2 부터 스텝 t 출력에 뒤 스텝 정보가 들어 있으므로 실제 조건이 아니다. '
  f'"baseline T=3 Δ {DCI(tb3["dacc_paired_vs_own_T4"])}%p, 스파이크 {tb3["spk_vs_base_T4"]["pct"]:.1f}%, SOPs {tb3["sops_vs_base_T4"]["pct"]:.1f}%" 를 지우고 같은 체크포인트를 T\' 로 다시 돌린 실측으로 바꿨다: '
  f'BN 고정 T\'=3 {CK(mB["fix3"]["dacc_vs_own_orig4"])}; BN 재추정 T\'=3 {CK(mB["rc3"]["dacc_vs_own_orig4"])}, 스파이크 {mB["rc3"]["spk_vs_base_orig4"]["pct"]:.1f}%, SOPs 추정 {mB["rc3"]["sops_est_vs_base_orig4"]["pct"]:.1f}% (CPU 1000장, 체크포인트 n=3, 138). '
  f'원자료는 early_exit_261020 과 같고, 다시 계산해 그 JSON 과 대조했다 (정확도·스파이크·구간 최대 차 {TM["check_vs_early_exit_json_maxabs"]:.1g}). '
  f'고친 곳: 머리말, 요약 3, 3절 스텝 몫 문단, 3b 절 전체, 5절 우리 행과 SEENN 행 비교 문구, 6절 1번, \'우리 쪽에 남는 것\', [LIMITATION], \'다음\' (4). JSON 은 `truncation` → `step_readout_noncausal` (인용 금지 표시) 로 이름을 바꾸고 `truncation_measured` 를 더했다. '
  f'결론 방향 (\'학습 없이 T 를 줄인 baseline 이 ours 보다 깊다\') 은 스파이크·SOPs 축에서 \'BN 재추정\' 조건으로만 남고, 정확도 근거는 ±0.2%p 마진을 걸치는 넓은 구간으로 바뀌었다. '
  f'다른 절의 수치 (1·1b·2·3·4·5절의 npz·원장 값) 는 바뀌지 않았다. 정정 전 판: `scratchpad/an1020/frsops_fix/fr_sops_261020_v2.{{md,json,py}}`.')
w('')
w('## 재현')
w('')
w('`CUDA_VISIBLE_DEVICES= ~/anaconda3/envs/venv_1/bin/python analysis/fr_sops_261020.py` (약 1분, CPU). 런별 값·묶음·곡선·추론 T 감소 실측 (`truncation_measured`)·비인과 스텝 판독 (`step_readout_noncausal`, 인용 금지)·서버 민감도·문헌 행 전부 `analysis/fr_sops_261020.json`. 3b 절은 `scratchpad/an1020/ee/{cpu,recal}/*.npz` 와 `analysis/early_exit_261020.json` (대조용) 이 있어야 돈다. 1차 판 문서는 `scratchpad/an1020/fr_sops_261020_v1.md` 에 남겼다.')
open(OUT_MD, 'w').write('\n'.join(md) + '\n')
print('wrote', OUT_MD)
