"""SEENN-I 식 조기 종료 모의 — 학습 없음 (R19 CIFAR-10, T=4, 26-10-20, eip-stats).

요청 (10-20 사용자 승인 "학습 없는 분석 3건", docs/paper/compare-candidates-261020.md 1.1 둘째 항목):
  npz 의 스텝별 출력 (logits_t) 에 신뢰도 임계값 종료를 걸어, baseline 과 ours_inter_ch 3e-3 에서
  정확도 × 이미지당 스파이크 곡선을 만들고 (a) 조기 종료만으로 ours 의 스파이크 감소에 닿는지,
  (b) ours 위에 조기 종료가 더해지는지, (c) SEENN-I 보고값과 어떻게 다른지를 본다.

이 스크립트가 확인한 것 (먼저 읽을 것):
  R19 의 BN 은 T 스텝 입력의 시간 평균 한 장에 걸리고 그 결과가 모든 스텝에 복사된다
  (flags.py `bn_per_step` 주석 10-15, lib_snn/model.py `f_temporal_reduction` 분기, models/resnet.py
  Conv→BN→뉴런). 그래서 conv2 부터는 스텝 1 의 발화가 앞 층의 스텝 1~4 전부에 의존한다 — 시간에 인과적이지
  않다. npz 의 logits_t[:, t] 는 'T=4 로 끝까지 돌린 뒤 t 스텝째 출력' 이고, 't 스텝에서 멈췄을 때의 출력' 이
  아니다. CPU 로 T'=1,2,3 을 직접 돌려 확인했다 (3 체크포인트, GPU 미사용, 아래 CPU_SCRIPT).
  그래서 이 파일은 세 가지를 따로 낸다.
    (가) 실현 가능한 조기 종료: T=4 계산 안에서 출력을 스텝 e 에서 읽고 멈춘다. 정확도는 logits_t 그대로
        (읽기 자체는 실현 가능), 스파이크는 '마지막 BN 뒤의 층' (conv4_block2_out_n, fc1_n) 의 e 이후 스텝만
        아낀다. 나머지 층은 BN 시간 평균 때문에 4 스텝을 다 돌아야 한다.
    (나) 비인과 판독 (인용 금지): 같은 정확도에 스파이크를 스텝 1..e 의 합으로 센 값. 요청 문구 그대로 계산한
        값이며, 실제로 낼 수 없는 값이다. 다른 분석이 같은 실수를 하지 않도록 기록만 한다.
    (다) T 를 줄여 다시 돌린 값 (재학습 없음): CPU 로 직접 잰 T'=1,2,3 (200장, T'=3 은 1000장 추가).

입력 (읽기만):
  npz (테스트 1만 장, 평가 모드, 스텝별 배열이 있는 새 추출기 산출물만):
    baseline s11~s13  /home/kyccj/PycharmProjects/TensorFlow-SNNs/_npz_t/            (서버 138)
    ours_inter_ch 3e-3 s2·s3  scratchpad/an1019/npz_abl/,  s4  scratchpad/an1019/npz/  (서버 138)
  원장 스냅샷 scratchpad/an1019/snapshot/inventory.json (10-19 16:38) — 표의 원장 열과 n.
  CPU 점검 산출물 scratchpad/an1020/ee/cpu/*.npz (CPU_SCRIPT 로 만듦).
  BN 재추정 산출물 scratchpad/an1020/ee/recal/*.npz (RECAL_SCRIPT 로 만듦, 10-20 검토 반영 2·3번).
  npz 32런 전체 (스텝별 배열이 있는 것) — 꼬리 층 몫·스텝 몫과 스파이크 수준의 관계 (검토 반영 4·7번).
쓰는 것: analysis/early_exit_261020.{json,md}. 그림 초안은 scratchpad/an1020/ee_rev/ 에만 쓴다
  (이 워크플로의 쓰기 범위 밖이라 EIP_fig/ 는 건드리지 않는다. EIP_fig/an_261020/ 의 PNG 는 검토 전 판).

10-20 검토 반영 (eip-critic needs_fixes, 9건): 문구 (1·4·5·6·9), 범위 표기와 BN 재추정 (2·3), CI 를 체크포인트 수준으로 (3),
  스텝 몫 해석 (7), SEENN 인용 (8). 항목별 처리는 md 끝 '검토 반영' 절.

쓰는 법:
  ~/anaconda3/envs/venv_1/bin/python analysis/early_exit_261020.py              # 계산 + json/md (+ 그림 초안, scratchpad)
  ~/anaconda3/envs/venv_1/bin/python analysis/early_exit_261020.py --write-cpu-script /경로/cpu_trunc.py
  ~/anaconda3/envs/venv_1/bin/python analysis/early_exit_261020.py --write-recal-script /경로/cpu_recal.py
    (CPU 점검·BN 재추정 스크립트를 파일로 꺼낸다. 돌리는 법은 그 파일 머리말.)
"""
import argparse, glob, hashlib, json, math, os, sys
import numpy as np
from scipy import stats

ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
SCR = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
A19 = SCR + '/an1019'
CPU_DIR = SCR + '/an1020/ee/cpu'
RECAL_DIR = SCR + '/an1020/ee/recal'
SNAP = A19 + '/snapshot/inventory.json'
OUT_JSON = ROOT + '/analysis/early_exit_261020.json'
OUT_MD = ROOT + '/analysis/early_exit_261020.md'
FIG_DIR = SCR + '/an1020/ee_rev'
FIG = FIG_DIR + '/early_exit_ko_26-10-20_rev.png'
FIG_OLD = 'EIP_fig/an_261020/early_exit_ko_26-10-20.png'
NPZ_ALL = [ROOT + '/_npz_t', A19 + '/npz', A19 + '/npz_abl']      # 스텝별 배열이 있는 npz 전부 (32런)
BOOT_B, BOOT_SEED = 20000, 20261020

# 스파이크 정의: ablation_261019.py / method_261019.py 의 npz 스파이크와 같은 18층 (입력층 n_in, 출력층 제외)
LAYERS = ['conv1_conv_n'] + [f'conv2_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')] + \
         [f'conv3_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')] + \
         [f'conv4_block{b}_{s}_n' for b in (1, 2) for s in ('conv1', 'out')] + ['fc1_n']
# 마지막 BN (conv4_block2_conv2_bn) 뒤에 있어 스텝별로 인과적인 층. 조기 종료로 뒤 스텝을 건너뛸 수 있는 것은 이 둘뿐.
#   conv4_block2_out_n = 뉴런(Add[항등 지름길 = conv4_block1_out_n 스파이크, BN(conv2) 상수])
#   fc1_n = 뉴런(Dense(GAP(conv4_block2_out_n)))  — GAP·Dense 에는 시간 평균이 없다 (layers.py, resnet.py)
#   conv4_block1_out_n 은 conv4_block2_conv1 의 BN 입력이라 4 스텝을 다 내야 한다.
TAIL = ['conv4_block2_out_n', 'fc1_n']

ARMS = {
    'baseline': [('r19c10-base---s11', ROOT + '/_npz_t'), ('r19c10-base---s12', ROOT + '/_npz_t'),
                 ('r19c10-base---s13', ROOT + '/_npz_t')],
    'ours_inter_ch': [('r19c10-ours_ch-3e-3-s2', A19 + '/npz_abl'), ('r19c10-ours_ch-3e-3-s3', A19 + '/npz_abl'),
                      ('r19c10-ours_ch-3e-3-s4', A19 + '/npz')],
}
# 원장에는 있으나 이 분석에서 뺀 런 (행은 남긴다)
LEDGER_ALL = {'baseline': [f'r19c10-base---s{s}' for s in range(11, 17)],
              'ours_inter_ch': [f'r19c10-ours_ch-3e-3-s{s}' for s in range(1, 5)]}
EXCL_REASON = {
    'r19c10-base---s14': 'npz 없음 (세종 런, 추출 안 함)',
    'r19c10-base---s15': 'npz 없음 (세종 런, 추출 안 함)',
    'r19c10-base---s16': 'npz 없음 (세종 런, 추출 안 함)',
    'r19c10-ours_ch-3e-3-s1': 'npz 가 10-02 옛 추출기 (_npz) 라 스텝별 배열·logits_t 없음',
}
# CPU 점검 (재학습 없이 T 를 줄여 다시 돌림). 키: (이름, 원장 태그, 팔)
#   200장 T'=1~4 는 처음 3개 (base11·base12·ours4), 1000장 T'=3·4 는 6개 전부 (검토 반영 3번으로 base13·ours2·ours3 추가).
CPU_RUNS = [('base11', 'r19c10-base---s11', 'baseline'), ('base12', 'r19c10-base---s12', 'baseline'),
            ('base13', 'r19c10-base---s13', 'baseline'),
            ('ours2', 'r19c10-ours_ch-3e-3-s2', 'ours_inter_ch'), ('ours3', 'r19c10-ours_ch-3e-3-s3', 'ours_inter_ch'),
            ('ours4', 'r19c10-ours_ch-3e-3-s4', 'ours_inter_ch')]
RECAL_T = [4, 3, 2, 1]
DELTAS = [0.0, -0.05, -0.10, -0.20, -0.50]          # 자기 T=4 정확도 대비 허용 손실 (%p)
SEENN = {'source': 'Li, Geller, Kim, Panda, NeurIPS 2023, arXiv 2304.01230v2 (PDF md5 2fcd9157…) — Table 1 (ResNet-19 CIFAR-10), '
                   'Fig. 3 와 Fig. 4 왼쪽 (SNN T=4 96.46; Fig. 1 은 TET R19 94.16/94.44/94.50 이라 이 값이 아님), '
                   '식 11 (스텝마다 CE = TET), §4.1 (TET 코드 틀, SGD 300에폭), §3.1 식 1 (LIF τ=0.5, 0 으로 리셋), '
                   'Table 5 (R19 C10 AET 1.1309 = 식 4, 가정 3.1 아래; empirical AET 1.1188 = 식 13), 부록 D (덧셈 0.9pJ, 곱셈 4.6pJ)',
         'points': [[96.07, 1.09], [96.38, 1.20], [96.44, 1.34]], 'own_T4': 96.46,
         'AET_eq4': 1.1309, 'AET_empirical_eq13': 1.1188,
         'recipe': 'LIF τ=0.5 (누설), 발화 시 0 으로 리셋, SGD 모멘텀 0.9 300에폭 lr 0.1 cosine wd 5e-4, Cutout + AutoAugment, TET 손실',
         'checked': '10-20 eip-stats 가 arXiv v2 PDF 를 pdftotext 로 읽어 대조. 검토 (8번) 지적대로 Fig. 1 → Fig. 3·4, '
                    'AET 를 두 정의로 나눔'}


# --------------------------------------------------------------------------------------------------
def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def msd(v):
    v = [float(x) for x in v]
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'n': len(v),
            'values': v}


def welch(a, b, alpha=0.05):
    """b − a, Welch t 구간."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b.mean() - a.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else float('inf')
    q = stats.t.ppf(1 - alpha / 2, df)
    p = 2 * stats.t.sf(abs(d / se), df) if se > 0 else float('nan')
    return {'d': float(d), 'lo': float(d - q * se), 'hi': float(d + q * se), 'p': float(p), 'df': float(df),
            'n_a': len(a), 'n_b': len(b)}


def paired_disc(c4, c3):
    """같은 이미지에서 T=4 맞고 T' 틀림 (b), T=4 틀리고 T' 맞음 (c). 차 (c-b)/n 과 95% 구간 (대응 비율 차 Wald), 정확 이항 p."""
    b, c, n = int((c4 & ~c3).sum()), int((~c4 & c3).sum()), len(c4)
    d = (c - b) / n
    se = math.sqrt(max(b + c - (c - b) ** 2 / n, 0)) / n
    p = stats.binomtest(c, b + c).pvalue if b + c else 1.0
    return {'n_img': n, 'lost': b, 'gained': c, 'd_pp': 100 * d, 'lo_pp': 100 * (d - 1.96 * se),
            'hi_pp': 100 * (d + 1.96 * se), 'p_exact': float(p)}


def holm(ps):
    """Holm 보정 p (단계 하강)."""
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return [float(x) for x in adj]


def clustered(pairs, seed=BOOT_SEED, B=BOOT_B):
    """같은 이미지 집합을 여러 체크포인트에서 대응 비교한 차 (%p). pairs = [(맞음_기준, 맞음_비교), ...] (체크포인트별, 같은 이미지 순서).
    (1) 체크포인트별 차, (2) 체크포인트 수준 t 구간 (df=K-1), (3) 이미지 군집 부트스트랩, (4) 체크포인트×이미지 이원 부트스트랩."""
    D = np.stack([c.astype(float) - a.astype(float) for a, c in pairs])       # [K, N], 값 -1/0/+1
    K, N = D.shape
    per = 100 * D.mean(1)
    out = {'K': K, 'n_img': N, 'per_ckpt_pp': [float(x) for x in per], 'mean_pp': float(per.mean())}
    if K > 1:
        se = per.std(ddof=1) / math.sqrt(K)
        q = stats.t.ppf(0.975, K - 1)
        out['t_ckpt'] = {'lo': float(per.mean() - q * se), 'hi': float(per.mean() + q * se), 'sd': float(per.std(ddof=1)), 'df': K - 1}
    rng = np.random.default_rng(seed)
    bi, b2 = np.empty(B), np.empty(B)
    for b in range(B):
        ii = rng.integers(N, size=N)
        bi[b] = D[:, ii].mean()
        kk = rng.integers(K, size=K)
        b2[b] = D[kk][:, ii].mean()
    out['boot_img'] = [float(100 * np.quantile(bi, 0.025)), float(100 * np.quantile(bi, 0.975))]
    out['boot_2way'] = [float(100 * np.quantile(b2, 0.025)), float(100 * np.quantile(b2, 0.975))]
    out['boot'] = {'B': B, 'seed': seed}
    return out


def ledger():
    runs = json.load(open(SNAP))
    by = {r['tag']: r for r in runs['runs']}
    return runs['generated'], by


# --------------------------------------------------------------------------------------------------
def load_run(tag, d):
    p = os.path.join(d, tag + '.npz')
    z = np.load(p)
    V = z['logits_t'].astype(np.float64)                 # [N, T, C]  출력층 막전위 (누적, 누설 없음)
    y = z['labels'].astype(np.int64)
    N, T, _ = V.shape
    # 출력층은 적분기 (layers.py: 마지막 층 n_type='OUT', neurons.py run_type_out: n_type!='LIF' 라 누설 없음),
    # 모델 출력 = 마지막 스텝 (snn_output_processing='last_ts'). 점검: logits == logits_t[:, T-1]
    assert np.abs(z['logits'].astype(np.float64) - V[:, -1]).max() == 0.0
    step = {l: z[l + '__t'].astype(np.float64).reshape(T, -1).sum(axis=1) / N for l in LAYERS}   # 이미지당 [T]
    S = sum(step.values())                                # 18층 스텝별 이미지당 스파이크
    tot = float(sum(z[l].astype(np.float64).sum() for l in LAYERS) / N)
    assert abs(tot - S.sum()) < 1e-6 * tot
    tail = sum(step[l] for l in TAIL)
    acc_npz = float(z['acc'])
    acc_chk = 100 * float((V[:, -1].argmax(1) == y).mean())
    assert abs(acc_npz - acc_chk) < 1e-9, (tag, acc_npz, acc_chk)
    pred_t = V.argmax(2)                                  # [N, T]
    correct_t = pred_t == y[:, None]
    # SEENN-I 신뢰도: 누적 출력 = 스텝별 출력의 평균. 여기서 스텝별 출력 = v_t - v_{t-1} 이므로 평균 = v_t / t.
    cs = np.stack([softmax(V[:, t] / (t + 1)).max(1) for t in range(T)], axis=1)
    cs_raw = np.stack([softmax(V[:, t]).max(1) for t in range(T)], axis=1)   # 민감도: 막전위 그대로
    # 라벨 없는 오라클: 예측이 그 뒤로 바뀌지 않는 첫 스텝 (T=4 예측을 그대로 재현하는 최소 스텝)
    stable = np.full(N, T)
    for t in range(T - 1, -1, -1):
        stable[np.all(pred_t[:, t:] == pred_t[:, -1:], axis=1)] = t + 1
    # 라벨 오라클 (SEENN 의 AET 와 같은 정의: 처음 맞는 스텝, 끝까지 틀리면 T)
    first_ok = np.where(correct_t.any(1), correct_t.argmax(1) + 1, T)
    return {'tag': tag, 'path': p, 'N': N, 'T': T, 'V': V, 'y': y, 'S': S, 'tot': tot, 'tail': tail,
            'step_layer': step, 'acc': acc_npz, 'correct_t': correct_t, 'cs': cs, 'cs_raw': cs_raw,
            'stable': stable, 'first_ok': first_ok}


def ee_curve(r, key='cs', ngrid=2001):
    """τ 를 훑어 각 이미지가 처음 CS ≥ τ 인 스텝에서 끝내게 한다 (없으면 T). 두 가지 스파이크 회계를 함께 낸다."""
    cs, V, y, T = r[key], r['V'], r['y'], r['T']
    q = np.quantile(cs[:, :T - 1].ravel(), np.linspace(0, 1, ngrid))
    taus = np.unique(np.concatenate([[0.0], q, [2.0]]))
    C = np.cumsum(r['S'])                                 # 스텝 1..t 의 이미지당 스파이크 (비인과 회계)
    tail_after = np.array([r['tail'][t + 1:].sum() for t in range(T)])   # 스텝 t+1 뒤의 꼬리 층 스파이크
    out = []
    for tau in taus:
        hit = cs >= tau
        hit[:, T - 1] = True
        e = hit.argmax(1)                                 # 0-기준 종료 스텝
        pred = V[np.arange(len(y)), e].argmax(1)
        acc = 100 * float((pred == y).mean())
        out.append({'tau': float(tau), 'acc': acc, 'meanT': float(e.mean() + 1),
                    'spk_real': float(r['tot'] - tail_after[e].mean()),
                    'spk_prefix': float(C[e].mean()),
                    'exit_share': [float((e == t).mean()) for t in range(T)]})
    return out


def at_delta(curve, acc4, d, kind):
    ok = [c for c in curve if c['acc'] >= acc4 + d - 1e-9]
    best = min(ok, key=lambda c: (c[kind], -c['acc']))
    return best


def at_meanT(curve, m):
    xs = np.array([c['meanT'] for c in curve])
    o = np.argsort(xs)
    xs = xs[o]
    f = lambda k: float(np.interp(m, xs, np.array([curve[i][k] for i in o])))
    return {'meanT': m, 'acc': f('acc'), 'spk_real': f('spk_real'), 'spk_prefix': f('spk_prefix')}


# --------------------------------------------------------------------------------------------------
def spk_of(c, k=None):
    """CPU npz 의 18층 이미지당 스파이크 (앞 k 스텝만이면 k)."""
    return float(sum((c['spk_' + l][:, :k] if k else c['spk_' + l]).sum(1).mean() for l in LAYERS))


def ref_npz(tag):
    return np.load(os.path.join(dict(ARMS['baseline'] + ARMS['ours_inter_ch'])[tag], tag + '.npz'))


def cpu_block():
    """CPU 로 T 를 줄여 다시 돌린 값 (재학습 없음, BN 이동통계는 T=4 그대로). 같은 체크포인트·같은 앞쪽 이미지에서 T=4 와 비교."""
    out = {'runs': {}, 'pooled_T3_1k_wald_superseded': None}
    pairs = {}
    for nm, tag, arm in CPU_RUNS:
        o = {'tag': tag, 'arm': arm, 'trunc_200': {}, 'T3_1k': None}
        z = ref_npz(tag)
        p4 = os.path.join(CPU_DIR, f'{nm}_T4.npz')
        if os.path.exists(p4):
            c4 = np.load(p4)
            y = c4['labels']
            n = len(y)
            assert (z['labels'][:n] == y).all()
            dmax = np.abs(c4['logits_t'] - z['logits_t'][:n]).max(axis=(1, 2))
            o['harness'] = {'n_img': n, 'argmax_same_pct': 100 * float((c4['logits_t'][:, -1].argmax(1) ==
                                                                          z['logits_t'][:n, -1].argmax(1)).mean()),
                            'img_maxdiff_lt_1e-5_pct': 100 * float((dmax < 1e-5).mean()),
                            'note': 'CPU 대 GPU 부동소수 차로 일부 이미지에서 스파이크가 뒤집힘. 예측은 같음'}
            a4 = c4['logits_t'][:, -1].argmax(1) == y
            for TT in (1, 2, 3, 4):
                p = os.path.join(CPU_DIR, f'{nm}_T{TT}.npz')
                if not os.path.exists(p):
                    continue
                c = np.load(p)
                assert int(c['T']) == TT and (c['labels'] == y).all()
                at = c['logits_t'][:, -1].argmax(1) == y
                rd = c4['logits_t'][:, TT - 1].argmax(1) == y
                caus = {l: float(np.abs(c['spk_' + l] - c4['spk_' + l][:, :TT]).max()) for l in LAYERS}
                o['trunc_200'][TT] = {'acc_run': 100 * float(at.mean()), 'spk_run': spk_of(c),
                                      'fc1_spk_run': float(c['spk_fc1_n'].sum(1).mean()),
                                      'fc1_spk_prefix_T4': float(c4['spk_fc1_n'][:, :TT].sum(1).mean()),
                                      'acc_readout_T4': 100 * float(rd.mean()), 'spk_prefix_T4': spk_of(c4, TT),
                                      'logit_maxdiff_vs_T4_prefix': float(np.abs(c['logits_t'] - c4['logits_t'][:, :TT]).max()),
                                      'layer_spike_maxdiff_vs_T4_prefix': caus,
                                      'first_diff_layer': next((l for l in LAYERS if caus[l] > 0), None),
                                      'vs_T4_run': paired_disc(a4, at) if TT < 4 else None}
            o['trunc_200']['spk_T4'] = spk_of(c4)
        p3, p4k = os.path.join(CPU_DIR, f'{nm}_T3_1k.npz'), os.path.join(CPU_DIR, f'{nm}_T4_1k.npz')
        if os.path.exists(p3) and os.path.exists(p4k):
            k3, k4 = np.load(p3), np.load(p4k)
            yk = k4['labels']
            assert (k3['labels'] == yk).all() and (z['labels'][:len(yk)] == yk).all()
            b4 = k4['logits_t'][:, -1].argmax(1) == yk
            b3 = k3['logits_t'][:, -1].argmax(1) == yk
            g4 = z['logits_t'][:len(yk), -1].argmax(1) == yk
            s3, s4 = spk_of(k3), spk_of(k4)
            caus = {l: float(np.abs(k3['spk_' + l] - k4['spk_' + l][:, :3]).max()) for l in LAYERS}
            cell = {}
            for l in LAYERS:
                dd_ = k3['spk_' + l] - k4['spk_' + l][:, :3]
                cell[l] = {'cells_diff': int((dd_ != 0).sum()), 'cells': int(dd_.size), 'sum_abs': float(np.abs(dd_).sum()),
                           'mean_spk_per_cell': float(k4['spk_' + l][:, :3].mean())}
            o['T3_1k_cell'] = cell
            o['T3_1k'] = {'n_img': len(yk), 'acc_T4_cpu': 100 * float(b4.mean()), 'acc_T4_gpu_npz': 100 * float(g4.mean()),
                          'argmax_same_cpu_gpu_pct': 100 * float((k4['logits_t'][:, -1].argmax(1) == z['logits_t'][:len(yk), -1].argmax(1)).mean()),
                          'acc_T3': 100 * float(b3.mean()), 'spk_T4': s4, 'spk_T3': s3,
                          'spk_red_pct': 100 * (1 - s3 / s4), 'diff': paired_disc(b4, b3),
                          'acc_readout_step3_T4': 100 * float((k4['logits_t'][:, 2].argmax(1) == yk).mean()),
                          'spk_prefix_step3_T4': spk_of(k4, 3),
                          'layer_spike_maxdiff_vs_T4_prefix': caus,
                          'first_diff_layer': next((l for l in LAYERS if caus[l] > 0), None),
                          'first_diff_layer_over1pct_cells': next((l for l in LAYERS if cell[l]['cells_diff'] > 0.01 * cell[l]['cells']), None)}
            pairs[nm] = (b4, b3)
        if o['trunc_200'] or o['T3_1k']:
            out['runs'][nm] = o
    # 검토 반영 3번: 3000쌍 Wald 는 체크포인트 간 분산과 이미지 반복을 빠뜨림 → 체크포인트 수준 t·부트스트랩으로 바꾼다
    first3 = [nm for nm in ('base11', 'base12', 'ours4') if nm in pairs]
    if first3:
        w = paired_disc(np.concatenate([pairs[k][0] for k in first3]), np.concatenate([pairs[k][1] for k in first3]))
        w['note'] = '검토 전 판의 값 (세 체크포인트 이미지를 독립으로 본 Wald). 체크포인트 간 분산이 빠져 좁다 — 인용 금지'
        out['pooled_T3_1k_wald_superseded'] = w
        out['T3_1k_first3_clustered'] = clustered([pairs[k] for k in first3])
    out['T3_1k_by_arm'] = {}
    for arm in ('baseline', 'ours_inter_ch'):
        ks = [nm for nm, _, a in CPU_RUNS if a == arm and nm in pairs]
        if ks:
            c = clustered([pairs[k] for k in ks])
            c['ckpts'] = ks
            c['spk_red_pct'] = msd([out['runs'][k]['T3_1k']['spk_red_pct'] for k in ks])
            out['T3_1k_by_arm'][arm] = c
    if len(out['T3_1k_by_arm']) == 2:
        out['T3_1k_arm_diff'] = welch(out['T3_1k_by_arm']['baseline']['per_ckpt_pp'], out['T3_1k_by_arm']['ours_inter_ch']['per_ckpt_pp'])
    if pairs:
        out['T3_1k_all_ckpts_ref'] = clustered([pairs[k] for k in pairs])
        out['T3_1k_all_ckpts_ref']['note'] = '팔이 섞인 합침 — 참고만'
    return out


RECAL_MODES = {'aug': {'suffix': '_aug', 'T': [4, 3, 2], 'desc': '학습셋 5000장, 학습 후반과 같은 증강 (crop·flip·RandAugment·Random Erasing; CutMix 는 반복 100,000 뒤 꺼짐)'},
               'clean': {'suffix': '', 'T': [4, 3, 2, 1], 'desc': '학습셋 3000장, 증강 없음 (테스트 전처리)'}}


def recal_block(mode):
    """BN 이동통계를 T' 에서 학습셋으로 다시 잡은 뒤 테스트 앞 1000장 (검토 반영 2·3번). 가중치는 그대로, 학습 없음.
    mode='aug' 가 원래 이동통계를 잡은 조건 (학습 후반의 증강) 과 맞춘 주 판, 'clean' 은 민감도."""
    suf, RT = RECAL_MODES[mode]['suffix'], RECAL_MODES[mode]['T']
    out = {'mode': mode, 'desc': RECAL_MODES[mode]['desc'], 'runs': {}, 'by_arm': {}}
    P = {}
    for nm, tag, arm in CPU_RUNS:
        p4o = os.path.join(CPU_DIR, f'{nm}_T4_1k.npz')
        if not os.path.exists(p4o):
            continue
        k4o = np.load(p4o)
        y = k4o['labels']
        a4o = k4o['logits_t'][:, -1].argmax(1) == y
        s4o = spk_of(k4o)
        p3o = os.path.join(CPU_DIR, f'{nm}_T3_1k.npz')
        a3o = (np.load(p3o)['logits_t'][:, -1].argmax(1) == y) if os.path.exists(p3o) else None
        r = {'tag': tag, 'arm': arm, 'acc_orig_T4': 100 * float(a4o.mean()), 'spk_orig_T4': s4o, 'T': {}}
        cor = {}
        for T in RT:
            p = os.path.join(RECAL_DIR, f'{nm}_T{T}{suf}.npz')
            if not os.path.exists(p):
                continue
            c = np.load(p)
            assert int(c['T']) == T and (c['labels'] == y).all() and float(c['pipe_maxdiff']) == 0.0
            assert str(c['mode']) == mode if 'mode' in c.files else mode == 'clean'
            at = c['logits_t'][:, -1].argmax(1) == y
            cor[T] = at
            dg = np.asarray(c['bn_diag'])
            s = spk_of(c)
            r['T'][T] = {'n_recal_img': int(c['n_recal_img']), 'n_img': len(y), 'acc': 100 * float(at.mean()), 'spk': s,
                         'spk_vs_orig_T4_pct': 100 * (s / s4o - 1), 'vs_orig_T4': paired_disc(a4o, at),
                         'vs_orig_sameT': paired_disc(a3o, at) if (T == 3 and a3o is not None) else None,
                         'bn_shift_median_sd': float(np.median(dg[:, 0])), 'bn_shift_max_layer_sd': float(dg[:, 0].max()),
                         'bn_varratio_median': float(np.median(dg[:, 1])),
                         'bn_varratio_range': [float(dg[:, 1].min()), float(dg[:, 1].max())],
                         'conv_half_vs_full_mean_sd_max': float(dg[:, 2].max()), 'conv_half_vs_full_var_rel_max': float(dg[:, 3].max())}
        for T in r['T']:
            if T != 4 and 4 in cor:
                r['T'][T]['vs_recal_T4'] = paired_disc(cor[4], cor[T])
                r['T'][T]['spk_vs_recal_T4_pct'] = 100 * (r['T'][T]['spk'] / r['T'][4]['spk'] - 1)
        out['runs'][nm] = r
        P[nm] = (a4o, cor)
    for arm in ('baseline', 'ours_inter_ch'):
        ks = [nm for nm, _, a in CPU_RUNS if a == arm and nm in P]
        A = {}
        for T in RT:
            kk = [k for k in ks if T in P[k][1]]
            if not kk:
                continue
            e = {'ckpts': kk, 'acc': msd([out['runs'][k]['T'][T]['acc'] for k in kk]),
                 'spk_vs_orig_T4_pct': msd([out['runs'][k]['T'][T]['spk_vs_orig_T4_pct'] for k in kk]),
                 'vs_orig_T4': clustered([(P[k][0], P[k][1][T]) for k in kk])}
            if T != 4 and all(4 in P[k][1] for k in kk):
                e['vs_recal_T4'] = clustered([(P[k][1][4], P[k][1][T]) for k in kk])
                e['spk_vs_recal_T4_pct'] = msd([out['runs'][k]['T'][T]['spk_vs_recal_T4_pct'] for k in kk])
            A[T] = e
        out['by_arm'][arm] = A
    # 팔 간 (참고, n=3 대 3): T'=3 재추정의 손실 크기 (자기 원래 T=4 대비, 체크포인트별 값) Welch
    # 같은 1000장에서 ours T=4 대 baseline T=4 스파이크 (원래 BN, CPU) — 6.2 의 % 와 같은 이미지 집합
    sb = [math.log(out['runs'][k]['spk_orig_T4']) for k, _, a in [(n_, t_, a_) for n_, t_, a_ in CPU_RUNS] if a == 'baseline' and k in out['runs']]
    so = [math.log(out['runs'][k]['spk_orig_T4']) for k, _, a in [(n_, t_, a_) for n_, t_, a_ in CPU_RUNS] if a == 'ours_inter_ch' and k in out['runs']]
    if len(sb) > 1 and len(so) > 1:
        w_ = welch(sb, so)
        w_.update({'pct': 100 * (math.exp(w_['d']) - 1), 'pct_lo': 100 * (math.exp(w_['lo']) - 1), 'pct_hi': 100 * (math.exp(w_['hi']) - 1)})
        out['ours_vs_base_T4_same1000'] = w_
    bt = [out['runs'][k]['T'][3]['vs_orig_T4']['d_pp'] for k in out['by_arm'].get('baseline', {}).get(3, {}).get('ckpts', [])]
    ot = [out['runs'][k]['T'][3]['vs_orig_T4']['d_pp'] for k in out['by_arm'].get('ours_inter_ch', {}).get(3, {}).get('ckpts', [])]
    if len(bt) > 1 and len(ot) > 1:
        out['T3_loss_ours_minus_base'] = welch(bt, ot)
    return out


def npz32_block(LED):
    """스텝별 배열이 있는 npz 전부 (32런): 실현 최대 절감 범위, 스텝 몫과 총 스파이크의 상관, 같은 스파이크 대조군의 스텝 몫 (검토 반영 4·7번)."""
    rows = []
    for d in NPZ_ALL:
        for f in sorted(glob.glob(d + '/*.npz')):
            tag = os.path.basename(f)[:-4]
            z = np.load(f)
            if 'logits_t' not in z.files:
                continue
            N, T = len(z['labels']), z['logits_t'].shape[1]
            st = {l: z[l + '__t'].astype(np.float64).reshape(T, -1).sum(1) / N for l in LAYERS}
            S = sum(st.values())
            tot = float(S.sum())
            tail = sum(st[l] for l in TAIL)
            led = LED.get(tag, {})
            rows.append({'tag': tag, 'server_train': led.get('server'), 'npz_spk': tot, 'ledger_spk': led.get('spikes'),
                         'npz_over_scount': tot / led['spikes'] if led.get('spikes') else None,
                         'step_share_pct': [float(100 * v / tot) for v in S], 'tail_share_pct': float(100 * tail.sum() / tot),
                         'real_max_saving_pct': float(100 * tail[1:].sum() / tot), 'npz_acc': float(z['acc'])})
    x = np.array([r['npz_spk'] for r in rows])
    cor = []
    for t in range(4):
        yv = np.array([r['step_share_pct'][t] for r in rows])
        rr = stats.pearsonr(x, yv)
        cor.append({'step': t + 1, 'r': float(rr[0]), 'p': float(rr[1])})
    groups = {'baseline (s11~s13)': [f'r19c10-base---s{s}' for s in (11, 12, 13)],
              'ours_inter_ch 3e-3 (s2~s4)': [f'r19c10-ours_ch-3e-3-s{s}' for s in (2, 3, 4)],
              'plain L2 1e-7 (s3, s4)': ['r19c10-l2-1e-7-s3', 'r19c10-l2-1e-7-s4'],
              'BPSR 1e-10 (s4)': ['r19c10-bpsr-1e-10-s4'],
              '- 1- inversion 4.5e-3 (s1~s4)': [f'r19c10-ch_noinv-4p5e-3-s{s}' for s in (1, 2, 3, 4)]}
    by = {r['tag']: r for r in rows}
    G = {}
    for g, tags in groups.items():
        rr = [by[t] for t in tags if t in by]
        G[g] = {'n': len(rr), 'servers': [r['server_train'] for r in rr],
                'ledger_spk': [r['ledger_spk'] for r in rr], 'npz_spk': [r['npz_spk'] for r in rr],
                'step1': [r['step_share_pct'][0] for r in rr], 'step4': [r['step_share_pct'][3] for r in rr]}
    rm = [r['real_max_saving_pct'] for r in rows]
    return {'n_runs': len(rows), 'rows': rows, 'real_max_saving_range': [min(rm), max(rm)],
            'corr_share_vs_npz_spk': cor, 'same_spike_groups': G}


# --------------------------------------------------------------------------------------------------
def main():
    gen, LED = ledger()
    R = {arm: [load_run(t, d) for t, d in runs] for arm, runs in ARMS.items()}
    res = {'generated_by': 'analysis/early_exit_261020.py', 'ledger_snapshot': gen, 'layers': LAYERS, 'tail_layers': TAIL,
           'deltas_pp': DELTAS, 'seenn': SEENN, 'runs': {}, 'arms': {}}
    for arm, rs in R.items():
        for r in rs:
            T = r['T']
            cur = ee_curve(r, 'cs')
            cur_raw = ee_curve(r, 'cs_raw')
            C = np.cumsum(r['S'])
            led = LED[r['tag']]
            row = {'arm': arm, 'server': led['server'], 'seed': led['seed'],
                   'ledger': {k: led[k] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes')},
                   'npz_acc': r['acc'], 'npz_spk': r['tot'],
                   'step_spk': [float(v) for v in r['S']], 'step_share_pct': [float(100 * v / r['tot']) for v in r['S']],
                   'tail_spk': float(r['tail'].sum()), 'tail_share_pct': float(100 * r['tail'].sum() / r['tot']),
                   'tail_step_spk': [float(v) for v in r['tail']],
                   # 어떤 종료 규칙이든 실현 가능한 최대 절감 = 모든 이미지가 스텝 1 에서 끝날 때 꼬리 층의 스텝 2~4
                   'real_max_saving_pct': float(100 * r['tail'][1:].sum() / r['tot']),
                   'readout_acc_t': [float(100 * r['correct_t'][:, t].mean()) for t in range(T)],
                   'prefix_spk_t': [float(v) for v in C],
                   'stable_oracle': {'meanT': float(r['stable'].mean()), 'acc': r['acc'],
                                     'exit_share': [float((r['stable'] == t + 1).mean()) for t in range(T)],
                                     'real_red_pct': float(100 * np.mean([r['tail'][s:].sum() for s in r['stable']]) / r['tot']),
                                     'prefix_red_pct': float(100 * (1 - C[r['stable'] - 1].mean() / r['tot']))},
                   'label_oracle_AET': float(r['first_ok'].mean()),
                   'curve': cur, 'curve_raw_cs': [{k: c[k] for k in ('tau', 'acc', 'meanT', 'spk_real', 'spk_prefix')}
                                                  for c in cur_raw]}
            for kind in ('spk_real', 'spk_prefix'):
                for src, cv in (('cs', cur), ('cs_raw', cur_raw)):
                    tab = {}
                    for d in DELTAS:
                        b = at_delta(cv, r['acc'], d, kind)
                        tab[f'{d:+.2f}'] = {'tau': b['tau'], 'acc': b['acc'], 'meanT': b['meanT'], 'spk': b[kind],
                                            'red_pct': 100 * (1 - b[kind] / r['tot']), 'exit_share': b['exit_share']}
                    row[f'at_delta_{kind}_{src}'] = tab
            row['at_meanT_seenn'] = [at_meanT(cur, m) for _, m in SEENN['points']]
            res['runs'][r['tag']] = row
    # ---- 팔 요약 (n=3 대 3)
    for arm, rs in R.items():
        rows = [res['runs'][r['tag']] for r in rs]
        a = {'n': len(rows), 'tags': [r['tag'] for r in rs]}
        a['npz_acc'] = msd([x['npz_acc'] for x in rows])
        a['npz_spk'] = msd([x['npz_spk'] for x in rows])
        a['step_share_pct'] = [msd([x['step_share_pct'][t] for x in rows]) for t in range(4)]
        a['readout_acc_t'] = [msd([x['readout_acc_t'][t] for x in rows]) for t in range(4)]
        a['tail_share_pct'] = msd([x['tail_share_pct'] for x in rows])
        a['real_max_saving_pct'] = msd([x['real_max_saving_pct'] for x in rows])
        a['label_oracle_AET'] = msd([x['label_oracle_AET'] for x in rows])
        a['stable_oracle_meanT'] = msd([x['stable_oracle']['meanT'] for x in rows])
        a['stable_oracle_real_red_pct'] = msd([x['stable_oracle']['real_red_pct'] for x in rows])
        a['stable_oracle_prefix_red_pct'] = msd([x['stable_oracle']['prefix_red_pct'] for x in rows])
        for kind in ('spk_real', 'spk_prefix'):
            for src in ('cs', 'cs_raw'):
                a[f'at_delta_{kind}_{src}'] = {k: {'red_pct': msd([x[f'at_delta_{kind}_{src}'][k]['red_pct'] for x in rows]),
                                                   'spk': msd([x[f'at_delta_{kind}_{src}'][k]['spk'] for x in rows]),
                                                   'meanT': msd([x[f'at_delta_{kind}_{src}'][k]['meanT'] for x in rows]),
                                                   'acc': msd([x[f'at_delta_{kind}_{src}'][k]['acc'] for x in rows])}
                                               for k in rows[0][f'at_delta_{kind}_{src}']}
        a['at_meanT_seenn'] = [{'meanT': m, 'acc': msd([x['at_meanT_seenn'][i]['acc'] for x in rows]),
                                'real_red_pct': msd([100 * (1 - x['at_meanT_seenn'][i]['spk_real'] / x['npz_spk']) for x in rows]),
                                'prefix_red_pct': msd([100 * (1 - x['at_meanT_seenn'][i]['spk_prefix'] / x['npz_spk']) for x in rows])}
                               for i, (_, m) in enumerate(SEENN['points'])]
        res['arms'][arm] = a
    # ---- 비교 (ours − baseline, Welch, n=3 대 3)
    B = [res['runs'][r['tag']] for r in R['baseline']]
    O = [res['runs'][r['tag']] for r in R['ours_inter_ch']]
    cmp = {}
    cmp['T4_ln_spk'] = welch([math.log(x['npz_spk']) for x in B], [math.log(x['npz_spk']) for x in O])
    cmp['T4_ln_spk']['pct'] = 100 * (math.exp(cmp['T4_ln_spk']['d']) - 1)
    cmp['T4_ln_spk']['pct_lo'] = 100 * (math.exp(cmp['T4_ln_spk']['lo']) - 1)
    cmp['T4_ln_spk']['pct_hi'] = 100 * (math.exp(cmp['T4_ln_spk']['hi']) - 1)
    cmp['T4_acc'] = welch([x['npz_acc'] for x in B], [x['npz_acc'] for x in O])
    # (b) 실현 가능한 조기 종료를 양쪽에 걸고, 각자 T=4 정확도에서 Δ 만큼 허용했을 때 스파이크 비
    for kind in ('spk_real', 'spk_prefix'):
        for k in B[0][f'at_delta_{kind}_cs']:
            lb = [math.log(x[f'at_delta_{kind}_cs'][k]['spk']) for x in B]
            lo = [math.log(x[f'at_delta_{kind}_cs'][k]['spk']) for x in O]
            w = welch(lb, lo)
            w.update({'pct': 100 * (math.exp(w['d']) - 1), 'pct_lo': 100 * (math.exp(w['lo']) - 1),
                      'pct_hi': 100 * (math.exp(w['hi']) - 1),
                      'acc_b': msd([x[f'at_delta_{kind}_cs'][k]['acc'] for x in B]),
                      'acc_o': msd([x[f'at_delta_{kind}_cs'][k]['acc'] for x in O])})
            cmp[f'ee_{kind}_{k}'] = w
            # 조기 종료 절감 비율 자체가 팔마다 같은가 (같으면 곱으로 쌓인다)
            cmp[f'ee_saving_{kind}_{k}'] = welch([x[f'at_delta_{kind}_cs'][k]['red_pct'] for x in B],
                                                 [x[f'at_delta_{kind}_cs'][k]['red_pct'] for x in O])
    cmp['real_max_saving_pct'] = welch([x['real_max_saving_pct'] for x in B], [x['real_max_saving_pct'] for x in O])
    cmp['step_share_pct'] = [welch([x['step_share_pct'][t] for x in B], [x['step_share_pct'][t] for x in O]) for t in range(4)]
    # 검토 반영 7번: Holm 보정 (몫은 합이 100 이라 독립 검정 4개가 아님 — 참고로만)
    hp = holm([w['p'] for w in cmp['step_share_pct']])
    for w, h in zip(cmp['step_share_pct'], hp):
        w['p_holm'] = h
    # 검토 반영 5번: 같은 6런을 원장 s_count 축으로 (n=3 대 3)
    cmp['T4_ln_spk_scount'] = welch([math.log(x['ledger']['spikes']) for x in B], [math.log(x['ledger']['spikes']) for x in O])
    for k, kk in (('pct', 'd'), ('pct_lo', 'lo'), ('pct_hi', 'hi')):
        cmp['T4_ln_spk_scount'][k] = 100 * (math.exp(cmp['T4_ln_spk_scount'][kk]) - 1)
    res['npz_over_scount'] = {arm: [res['runs'][r['tag']]['npz_spk'] / res['runs'][r['tag']]['ledger']['spikes'] for r in R[arm]]
                              for arm in R}
    res['compare_ours_minus_baseline'] = cmp
    # ---- SEENN 평균 T 를 스파이크로 옮긴 '가정 계산' (측정 아님)
    Sb = np.array([np.mean([x['step_share_pct'][t] for x in B]) for t in range(4)]) / 100
    Cb = np.cumsum(Sb)
    conv = []
    for acc_s, m in SEENN['points']:
        lo, hi = lp_bounds(Cb, m)
        conv.append({'acc': acc_s, 'meanT': m, 'loss_vs_own_T4_pp': acc_s - SEENN['own_T4'],
                     'spk_frac_if_proportional_to_T': m / 4,
                     'spk_frac_bounds_if_baseline_step_profile': [lo, hi]})
    res['seenn_conversion_assumption'] = {'points': conv, 'baseline_step_share_used': list(Sb),
                                          'note': '측정 아님. SEENN 은 스파이크 수를 보고하지 않았다. (i) 스파이크가 스텝 수에 비례한다는 가정, '
                                                  '(ii) 우리 baseline 의 스텝별 몫 (n=3, 시간 평균 BN 망) 을 그대로 쓴다는 가정 아래, '
                                                  '평균 T 가 같은 모든 종료 분포에 대한 최소·최대 (선형계획).'}
    res['cpu'] = cpu_block()
    res['recal'] = {m: recal_block(m) for m in ('aug', 'clean')}
    res['npz32'] = npz32_block(LED)
    res['npz_extract_host'] = {'r19c10-base---s11': 'canus (138)', 'r19c10-base---s12': 'canus (138)', 'r19c10-base---s13': 'canus (138)',
                               'r19c10-ours_ch-3e-3-s2': '55 서버 GPU 8·9 (체크포인트 사본 /srv2/kyccj/an1019/store)',
                               'r19c10-ours_ch-3e-3-s3': '55 서버 GPU 8·9 (체크포인트 사본 /srv2/kyccj/an1019/store)',
                               'r19c10-ours_ch-3e-3-s4': 'canus (138)',
                               'source': 'npz 옆 .log 의 load weight 경로, scratchpad/an1019/npz55_drv.sh'}
    res['exclusions'] = [{'tag': t, 'reason': EXCL_REASON[t]} for t in EXCL_REASON]
    res['ledger_rows'] = {arm: [{'tag': t, **{k: LED[t][k] for k in ('server', 'seed', 'train_loss', 'train_acc', 'val_loss',
                                                                    'val_acc', 'spikes', 'completed')}}
                                for t in ts] for arm, ts in LEDGER_ALL.items()}
    res['ledger_means'] = {arm: {k: msd([LED[t][k] for t in ts]) for k in ('val_acc', 'spikes')} for arm, ts in LEDGER_ALL.items()}
    res['inputs_md5'] = {r['tag']: md5(r['path']) for rs in R.values() for r in rs}
    res['cpu_script_md5'] = hashlib.md5(CPU_SCRIPT.encode()).hexdigest()
    res['recal_script_md5'] = hashlib.md5(RECAL_SCRIPT.encode()).hexdigest()
    json.dump(res, open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)
    print('[json]', OUT_JSON)
    write_md(res)
    make_fig(res)


def lp_bounds(C, m):
    """종료 스텝 분포 p (합 1, 평균 m) 에 대해 Σ p_t C_t 의 최소·최대."""
    from scipy.optimize import linprog
    T = len(C)
    A = np.array([np.ones(T), np.arange(1, T + 1)])
    b = np.array([1.0, m])
    lo = linprog(C, A_eq=A, b_eq=b, bounds=[(0, 1)] * T).fun
    hi = -linprog(-C, A_eq=A, b_eq=b, bounds=[(0, 1)] * T).fun
    return float(lo), float(hi)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------------
def f0(x):
    return f'{x:,.0f}'


def pm(s, nd=2, unit=''):
    if s['sd'] is None:
        return f"{s['mean']:.{nd}f}{unit}"
    return f"{s['mean']:.{nd}f} ±{s['sd']:.{nd}f}{unit}"


def ci(w, nd=2, key=('d', 'lo', 'hi'), unit=''):
    d, lo, hi = (w[k] for k in key)
    return f"{d:+.{nd}f}{unit} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def fcl(c, nd=2):
    """clustered() 결과를 한 줄로."""
    s = f"{c['mean_pp']:+.{nd}f}%p"
    parts = []
    if 't_ckpt' in c:
        parts.append(f"체크포인트 수준 t [{c['t_ckpt']['lo']:+.{nd}f}, {c['t_ckpt']['hi']:+.{nd}f}]")
    parts.append(f"이미지 부트스트랩 [{c['boot_img'][0]:+.{nd}f}, {c['boot_img'][1]:+.{nd}f}]")
    if c['K'] > 1:
        parts.append(f"체크포인트×이미지 이원 부트스트랩 [{c['boot_2way'][0]:+.{nd}f}, {c['boot_2way'][1]:+.{nd}f}]")
    return s + ' (' + ', '.join(parts) + ')'


def _margin_word(c, m=0.2):
    """체크포인트×이미지 이원 부트스트랩 구간으로 ±m %p 마진 판정."""
    lo, hi = c['boot_2way']
    if hi < -m:
        return f'손실이 남는다 (구간 위 끝 {hi:+.2f} < −{m}%p)'
    if lo > -m and hi < m:
        return f'±{m}%p 마진 안이다'
    if lo > 0:
        return '오히려 높다'
    return f'±{m}%p 마진 판정을 못 한다 (구간 [{lo:+.2f}, {hi:+.2f}] 이 마진을 걸침)'


def recal_verdict(res):
    RA = res['recal']['aug']['by_arm']
    cpu = res['cpu']['T3_1k_by_arm']
    b3, o3 = RA.get('baseline', {}).get(3), RA.get('ours_inter_ch', {}).get(3)
    if not (b3 and o3):
        return ''
    return (f"BN 고정 손실 (baseline {cpu['baseline']['mean_pp']:+.2f}, ours {cpu['ours_inter_ch']['mean_pp']:+.2f}%p) 의 대부분은 BN 이동통계 불일치였고, 남은 손실은 이 규모로 ±0.2%p 마진 판정을 못 한다. "
            f"**그래서 '학습 없이 T 를 한 스텝 줄이고 BN 만 다시 잡은 baseline' 은 스파이크 {b3['spk_vs_orig_T4_pct']['mean']:+.0f}% 로 ours_inter_ch T=4 ({res['compare_ours_minus_baseline']['T4_ln_spk']['pct']:+.1f}%, npz 축) 보다 깊은 학습 없는 대조가 될 수 있다** "
            "(정확도 손실이 마진 안이라는 근거도 밖이라는 근거도 아직 없음, 8절). "
            + (f"T'=2 는 BN 을 다시 잡아도 정확도 {min(RA['baseline'][2]['acc']['values'] + RA['ours_inter_ch'][2]['acc']['values']):.1f}~{max(RA['baseline'][2]['acc']['values'] + RA['ours_inter_ch'][2]['acc']['values']):.1f}% 로 무너진다 (체크포인트 6개). "
               if 2 in RA.get('baseline', {}) and 2 in RA.get('ours_inter_ch', {}) else '')
            + "검토 전 판의 \"재학습 없는 T 감소는 쓸 수 없다\" 는 BN 고정 조건에서만 맞는 말이다.")


def recal_discuss(res):
    RA, RCl = res['recal']['aug'], res['recal']['clean']
    cpu = res['cpu']['T3_1k_by_arm']
    L = []
    pe = []
    for RR, nm in ((RA, '증강판'), (RCl, '무증강판')):
        pe.append(nm + ' ' + ', '.join(f"{arm} {fcs(RR['by_arm'][arm][4]['vs_orig_T4'])}" for arm in ('baseline', 'ours_inter_ch') if 4 in RR['by_arm'].get(arm, {})))
    L.append(f"- 절차 자체의 효과 (재추정 T'=4 − 원래 T=4, 팔당 체크포인트 3개): {'; '.join(pe)}. 무증강판은 T'=4 에서도 정확도를 떨어뜨린다. "
             "원래 이동통계가 증강된 학습 이미지 위에서 잡혔기 때문으로 보인다 (추정). 그래서 T 감소의 효과는 증강판으로 읽고, 무증강판은 \'BN 통계를 어떤 데이터로 잡느냐에 따라 1%p 안팎이 움직인다\' 는 민감도로만 쓴다.")
    b3, o3 = RA['by_arm'].get('baseline', {}).get(3), RA['by_arm'].get('ours_inter_ch', {}).get(3)
    if b3 and o3:
        L.append(f"- **증강판 T'=3**: 원래 T=4 대비 baseline {fcs(b3['vs_orig_T4'])}, ours_inter_ch {fcs(o3['vs_orig_T4'])} (체크포인트별 baseline "
                 f"{', '.join(f'{v:+.1f}' for v in b3['vs_orig_T4']['per_ckpt_pp'])}, ours {', '.join(f'{v:+.1f}' for v in o3['vs_orig_T4']['per_ckpt_pp'])}). "
                 f"BN 고정 T'=3 의 손실 (baseline {cpu['baseline']['mean_pp']:+.2f}, ours {cpu['ours_inter_ch']['mean_pp']:+.2f}%p) 의 대부분은 BN 이동통계가 T=4 에 맞춰져 있어서 생긴 것이다. "
                 f"남은 손실이 0 인지 0.2~1%p 인지는 이 규모 (1000장 × 체크포인트 3개) 로 가리지 못한다 — baseline {_margin_word(b3['vs_orig_T4'])}, ours {_margin_word(o3['vs_orig_T4'])}.")
        L.append(f"- 스파이크: 증강판 T'=3 은 원래 T=4 대비 baseline {pm(b3['spk_vs_orig_T4_pct'], 1)}%, ours {pm(o3['spk_vs_orig_T4_pct'], 1)}% (같은 1000장, 18층, 평가 모드) 다. "
                 f"baseline 의 −39% 는 ours_inter_ch T=4 의 baseline 대비 {res['compare_ours_minus_baseline']['T4_ln_spk']['pct']:+.1f}% (npz 1만 장, n=3 대 3) 보다 깊다"
                 + (f" (같은 1000장으로 재면 ours T=4 는 baseline T=4 대비 {RA['ours_vs_base_T4_same1000']['pct']:+.1f}% [{RA['ours_vs_base_T4_same1000']['pct_lo']:+.1f}, {RA['ours_vs_base_T4_same1000']['pct_hi']:+.1f}], n=3 대 3). " if 'ours_vs_base_T4_same1000' in RA else '. ')
                 + "**그래서 \'학습 없이 T 를 한 스텝 줄이고 BN 만 다시 잡은 baseline\' 은 ours 보다 스파이크를 더 줄이는 대조가 될 수 있다.** 정확도 손실이 ±0.2%p 안이라는 근거도, 밖이라는 근거도 아직 없다.")
    b2, o2 = RA['by_arm'].get('baseline', {}).get(2), RA['by_arm'].get('ours_inter_ch', {}).get(2)
    if b2 and o2:
        bad = max(b2['acc']['values'] + o2['acc']['values']) < 90
        L.append(f"- 증강판 T'=2: 원래 T=4 대비 baseline {fcs(b2['vs_orig_T4'])} (정확도 {pm(b2['acc'], 1)}%), ours {fcs(o2['vs_orig_T4'])} (정확도 {pm(o2['acc'], 1)}%), "
                 f"스파이크 baseline {pm(b2['spk_vs_orig_T4_pct'], 1)}%, ours {pm(o2['spk_vs_orig_T4_pct'], 1)}%."
                 + (" BN 을 다시 잡아도 T'=2 는 쓸 수 없는 수준이다 (체크포인트 6개 모두 정확도 90% 미만)." if bad else ''))
    c2 = RCl['by_arm'].get('baseline', {}).get(2)
    if c2:
        L.append(f"- 무증강판 T'=2·1 은 정확도가 baseline {pm(c2['acc'], 1)}% · {pm(RCl['by_arm']['baseline'][1]['acc'], 1)}% 로 무너진다 (체크포인트 3개).")
    sh = []
    for T in (4, 3, 2):
        v = [RA['runs'][k]['T'][T]['bn_shift_median_sd'] for k in RA['runs'] if T in RA['runs'][k]['T']]
        if v:
            sh.append(f"T'={T} {min(v):.3f}~{max(v):.3f}")
    if sh:
        L.append(f"- BN 평균이 원래 값에서 움직인 크기 (원래 SD 단위, 20층 중앙값, 증강판, 체크포인트 6개): {', '.join(sh)}. T'=4 에서는 원래 값을 거의 그대로 재현하므로 증강판 절차가 원래 통계를 잡은 방식과 맞는다는 점검도 된다.")
    L.append("- 정밀도: 1000장이라 체크포인트당 차의 표준오차가 약 0.3~0.7%p 이고, 팔당 체크포인트 3개라 체크포인트 수준 t 구간은 df=2 로 넓다 (표·JSON `recal.aug.by_arm.*.*.vs_orig_T4.t_ckpt`). "
             "스파이크 % 는 같은 1000장·같은 18층 정의끼리의 비교다 (ours − baseline 은 1000장에서도 1만 장 npz 와 같은 값이 나와 이미지 집합 차는 작다). "
             "다만 1만 장 npz (GPU) 축이고 원장 s_count 축이 아니다. "
             "무증강판 T'=4 파일 6개는 증강 모드를 넣기 전 판의 스크립트로 만들었다 (무증강 경로는 같음, 사본 scratchpad `an1020/ee/cpu_recal_v1_clean.py`).")
    return '\n'.join(L)


def sec7_t(res):
    RA = res['recal']['aug']['by_arm']
    e = RA.get('baseline', {}).get(3)
    s = '- 1.2 의 3번 (시간축 대조, T=2·T=1 학습) 은 그대로 유효하다. 그 T 로 학습하면 BN 평균도 그 T 위에서 잡힌다. '
    if e:
        s += (f"다만 검토 전 판의 \"재학습 없는 T 감소는 쓸 수 없다\" 는 뺐다. BN 이동통계만 다시 잡으면 baseline T'=3 이 원래 T=4 대비 {fcs(e['vs_orig_T4'])} (이원 부트스트랩), "
              f"스파이크 {pm(e['spk_vs_orig_T4_pct'], 1)}% 다 (6.2절, 증강판, 체크포인트 3개 × 1000장). 학습 없는 대조 (T 감소 + BN 재추정) 를 시간축 대조에 넣을지는 사용자가 정한다 (8절).")
    return s


def frsops_recal(res):
    e = res['recal']['aug']['by_arm'].get('baseline', {}).get(3)
    if not e:
        return ''
    return (f"BN 을 다시 잡으면 (증강판) {fcs(e['vs_orig_T4'])}, 스파이크 {pm(e['spk_vs_orig_T4_pct'], 1)}% 다 (6.2절). "
            "그 문서의 정확도 −0.02 [−0.10, +0.06]%p 는 실제로 돌린 어느 조건의 값도 아닌 판독값이다. 결론 방향 (\'학습 없이 T 를 한 스텝 줄인 baseline 이 ours 보다 깊다\') 은 "
            "BN 재추정을 붙이면 살아날 수 있지만, 정확도 근거는 6.2 의 넓은 구간으로 바뀌고 \'BN 재추정\' 이라는 조건이 붙는다.")


def sec8(res):
    return ("- **(먼저 권함, 사용자 확인 필요) T 감소 + BN 재추정 대조를 1만 장으로.** 6.2 에서 baseline T'=3 + BN 재추정이 스파이크 −39% 에 정확도 손실을 가리지 못하는 대조로 나왔다. "
            "논문 주장 (같은 정확도에서 스파이크 감소) 에 가장 직접적인 위협이라, 테스트 1만 장 × 팔당 체크포인트 3개 (baseline s11~s13, ours 3e-3 s2~s4) 로 T'=3 을 다시 잰다 (T'=2 는 BN 을 다시 잡아도 무너져 뺀다). "
            "가중치를 바꾸지 않는 추론이라 학습 경로·큐와 무관하다. CPU 로 체크포인트·T\' 하나에 약 20~40분 (추정: 5000장 추정 + 1만 장 평가, 이번 1000장 실측이 다른 작업과 겹쳐 10~12분) 이고, "
            "GPU 를 쓰면 짧아지지만 GPU 사용은 확인이 필요하다. 추정 장 수 (5000 → 1만) 와 난수를 바꾼 민감도도 함께 잰다.\n"
            "- 위 대조와 함께 T=2/T=3 으로 학습한 baseline (compare-candidates 1.2 의 3번) 을 두어야 \'T 만 줄이면 되지 않나\' 에 대한 답이 완성된다. BN 재추정만 있으면 학습 대조가 없다는 지적, 학습 대조만 있으면 학습 없는 대조를 빠뜨렸다는 지적을 받는다.")


def fcs(c, nd=2):
    """짧게: 평균 [이원 부트스트랩 95%] (체크포인트 K개)."""
    return f"{c['mean_pp']:+.{nd}f}%p [{c['boot_2way'][0]:+.{nd}f}, {c['boot_2way'][1]:+.{nd}f}]"


def write_md(res):
    A, Rn, C, cpu, N32 = res['arms'], res['runs'], res['compare_ours_minus_baseline'], res['cpu'], res['npz32']
    RA, RCl = res['recal']['aug'], res['recal']['clean']
    b, o = A['baseline'], A['ours_inter_ch']
    nm_arm = {nm: arm for nm, _, arm in CPU_RUNS}
    nm_tag = {nm: tag for nm, tag, _ in CPU_RUNS}
    c200 = [cpu['runs'][nm] for nm, _, _ in CPU_RUNS if nm in cpu['runs'] and cpu['runs'][nm]['trunc_200']]
    c1k = [cpu['runs'][nm] for nm, _, _ in CPU_RUNS if nm in cpu['runs'] and cpu['runs'][nm]['T3_1k']]
    rng = lambda v, nd=1: f'{min(v):.{nd}f}~{max(v):.{nd}f}'
    t2_run = rng([r['trunc_200'][2]['acc_run'] for r in c200])
    t2_read = rng([r['trunc_200'][2]['acc_readout_T4'] for r in c200])
    fc1_t2 = rng([r['trunc_200'][2]['fc1_spk_run'] for r in c200])
    fc1_t2p = rng([r['trunc_200'][2]['fc1_spk_prefix_T4'] for r in c200])
    tail_rng = rng([x['tail_share_pct'] for x in Rn.values()], 2)
    t3_spk = rng([-r['T3_1k']['spk_red_pct'] for r in c1k])
    t3_loss = rng([r['T3_1k']['diff']['d_pp'] for r in c1k])
    t3_vs_prefix = rng([100 * (r['T3_1k']['spk_T3'] / r['T3_1k']['spk_prefix_step3_T4'] - 1) for r in c1k], 1)
    gap2 = rng([x['readout_acc_t'][1] - x['npz_acc'] for x in Rn.values()], 2)
    fd200 = sorted({r['trunc_200'][t]['first_diff_layer'] for r in c200 for t in (1, 2, 3)} - {None})
    fd1k = sorted({r['T3_1k']['first_diff_layer_over1pct_cells'] for r in c1k} - {None})
    c1cells = [(nm, cpu['runs'][nm]['T3_1k_cell']['conv1_conv_n'], cpu['runs'][nm]['T3_1k_cell']['conv2_block1_conv1_n'])
               for nm, _, _ in CPU_RUNS if nm in cpu['runs'] and cpu['runs'][nm]['T3_1k']]
    conv1_200 = all(r['trunc_200'][t]['layer_spike_maxdiff_vs_T4_prefix']['conv1_conv_n'] == 0 for r in c200 for t in (1, 2, 3))
    conv1_bad = [(nm, c1) for nm, c1, _ in c1cells if c1['cells_diff'] > 0]
    c2frac = [100 * c2['cells_diff'] / c2['cells'] for _, _, c2 in c1cells]
    c2rel = [100 * c2['sum_abs'] / c2['cells'] / c2['mean_spk_per_cell'] for _, _, c2 in c1cells]
    conv1_txt = (f"200장 비교 (체크포인트 3개 × T'=1·2·3) 에서 conv1 발화는 {'모두 비트 단위로 같았다' if conv1_200 else '일부 달랐다'}. "
                 f"1000장 T'=3 비교에서는 체크포인트 {len(c1cells) - len(conv1_bad)}개가 비트 단위로 같았고"
                 + (', ' + ', '.join(f"{nm_tag[nm]} 는 {c1['cells']} (이미지, 스텝) 칸 중 {c1['cells_diff']}칸에서 합 {c1['sum_abs']:.0f} 스파이크 차 (CPU 부동소수 차 수준)" for nm, c1 in conv1_bad) if conv1_bad else '')
                 + f". 반면 conv2_block1_conv1_n 은 칸의 {min(c2frac):.0f}~{max(c2frac):.0f}% 가 다르고, 칸당 평균 절대 차가 그 층 스파이크의 {min(c2rel):.1f}~{max(c2rel):.1f}% 다 (체크포인트 {len(c1cells)}개). "
                 f"처음 달라지는 층 (칸의 1% 넘게 다름) 은 {', '.join(sorted(set(fd200) | set(fd1k)))} 이다.")
    ba, oa = cpu['T3_1k_by_arm']['baseline'], cpu['T3_1k_by_arm']['ours_inter_ch']
    rb, ro = RA['by_arm'].get('baseline', {}), RA['by_arm'].get('ours_inter_ch', {})
    sc = C['T4_ln_spk_scount']
    ratio_b, ratio_o = res['npz_over_scount']['baseline'], res['npz_over_scount']['ours_inter_ch']
    L = []
    w = L.append
    w('# SEENN-I 식 조기 종료 모의 — 학습 없음 (R19 CIFAR-10, T=4, 26-10-20, 검토 반영판)')
    w('')
    w('- 요청: 사용자 10-20 승인 "학습 없는 분석 3건" (`docs/paper/compare-candidates-261020.md` 1.1) 중 조기 종료 모의. '
      'npz 스텝별 출력에 신뢰도 임계값 종료를 걸어 baseline 과 ours_inter_ch 3e-3 의 정확도 × 스파이크 곡선을 만들고, '
      '(a) 조기 종료만으로 ours 의 감소에 닿는지, (b) ours 위에 더해지는지, (c) SEENN-I 보고값과 비교한다.')
    w(f"- 입력 (읽기만): 테스트 1만 장 npz — baseline s11~s13 (n=3), ours_inter_ch 3e-3 s2~s4 (n=3). 학습 서버는 모두 138 이고, "
      f"npz 추출 기계는 섞여 있다 (ours s2·s3 은 55 서버 GPU 8·9, 나머지 4런은 canus). 원장 스냅샷 {res['ledger_snapshot']} "
      f"(원장 n: baseline 6, ours_inter_ch 3e-3 4). 스텝별 배열이 없는 런은 뺐다 (1절 표에 사유). 스텝 몫 비교에는 스텝별 배열이 있는 npz {N32['n_runs']}런 전체도 썼다 (2절).")
    w('- 큐·실험·서버는 건드리지 않았다. GPU 는 쓰지 않았다. CPU 로 추론만 돌렸다: (i) BN 이동통계를 그대로 두고 T 만 줄임 — 체크포인트 3개 × 200장 (T\'=1~4), '
      '체크포인트 6개 × 1000장 (T\'=3·4); (ii) BN 이동통계만 T\' 에서 학습셋으로 다시 잡음 — 체크포인트 6개 × 1000장, 증강판 (5000장, T\'=2·3·4) 과 무증강판 (3000장, T\'=1~4). 가중치는 바꾸지 않았다 (6절).')
    w(f"- 쓴 파일: `analysis/early_exit_261020.{{py,json,md}}`. 그림 초안은 scratchpad `an1020/ee_rev/` 에만 있다. `{FIG_OLD}` 는 검토 전 판 그림이다 (이 워크플로의 쓰기 범위 밖이라 다시 그리지 않았다). "
      'CPU 원자료는 scratchpad `an1020/ee/cpu/`·`an1020/ee/recal/`, 스크립트는 py 안의 `CPU_SCRIPT`·`RECAL_SCRIPT` 다.')
    _b3 = RA['by_arm'].get('baseline', {}).get(3)
    _rv = (f"BN 이동통계만 학습셋으로 다시 잡으면 (학습 없음, 새 측정) baseline T'=3 이 스파이크 {_b3['spk_vs_orig_T4_pct']['mean']:+.0f}% 에 정확도 {fcs(_b3['vs_orig_T4'])} 로, "
           "ours 보다 깊은 학습 없는 대조가 될 수 있다 (6.2절)") if _b3 else 'BN 이동통계를 다시 잡으면 달라진다 (6.2절)'
    w('- **검토 반영 (10-20, eip-critic needs_fixes 9건).** 바뀐 판정: (1) 스텝 t 출력은 \'스텝 t 에서 멈춘 출력\' 이 아니라 \'4스텝을 다 돈 뒤 일찍 읽은 출력\' 이다 (문구). '
      f'(2) "재학습 없는 T 감소는 쓸 수 없다" 는 일반화를 뺐다. 그 결과는 \'T=4 BN 이동통계 고정\' 조건에서만 맞는다. {_rv}. '
      '(3) T\'=3 손실의 구간을 체크포인트 수준으로 다시 냈다. (4) 스텝 몫 차이는 ours 의 성질이 아니라 스파이크 수준 효과로 보인다 — 해석을 철회했다. '
      '(5) SEENN 인용 오류 (그림 번호, AET 정의) 를 고쳤다. (6) 같은 오해가 다른 문서 5곳에 있다 (7절). 항목별 처리는 끝의 \'검토 반영\' 절.')
    w('')
    # ---- 한 줄
    r3b, r3o = rb.get(3), ro.get(3)
    w('## 한 줄')
    w('')
    w(f"**npz 의 스텝별 출력으로는 조기 종료를 모의할 수 없다.** R19 의 BN 이 T 스텝 입력의 시간 평균에 걸려 망이 시간에 인과적이지 않다. 스텝 t 출력은 스텝 t 에서 계산을 멈춘 출력이 아니고, "
      f"4스텝을 다 돈 뒤 일찍 읽는 용도로만 쓸 수 있다. 그렇게 읽는 조기 종료가 아낄 수 있는 스파이크는 마지막 BN 뒤 두 층의 뒤 스텝뿐이라 "
      f"**최대 {pm(b['real_max_saving_pct'])}% (baseline n=3; 시간 평균 BN 망, 18층 npz 스파이크 기준)** 다. "
      f"T 를 실제로 줄여 다시 돌리는 것은 BN 을 어떻게 두느냐에 달렸다 (아래 [ ] 는 체크포인트×이미지 이원 부트스트랩 95%, 팔당 체크포인트 3개 × 1000장). "
      f"**T=4 BN 이동통계를 그대로 두면** T'=3 이 자기 T=4 대비 baseline {fcs(ba)}, ours_inter_ch {fcs(oa)} 이고 T'=2 는 우연 수준 ({t2_run}%, 200장 × 체크포인트 3개) 이다.")
    if r3b and r3o:
        w(f"**BN 이동통계만 T' 에서 다시 잡으면** (학습셋 5000장을 학습 후반과 같은 증강으로 순전파, 가중치 그대로) T'=3 은 자기 원래 T=4 대비 baseline {fcs(r3b['vs_orig_T4'])}, "
          f"ours_inter_ch {fcs(r3o['vs_orig_T4'])} 이고, 스파이크는 baseline {pm(r3b['spk_vs_orig_T4_pct'], 1)}%, ours {pm(r3o['spk_vs_orig_T4_pct'], 1)}% (원래 T=4 대비, 같은 1000장) 다. "
          f"{recal_verdict(res)}")
    w(f"(a) T=4 를 끝까지 도는 지금 망에서는 조기 종료만으로 ours_inter_ch 의 감소 (npz 축 {C['T4_ln_spk']['pct']:+.1f}%, s_count 축 {sc['pct']:+.1f}%, n=3 대 3) 에 닿지 않는다. "
      f"(b) 같은 조건에서 ours 에 걸어도 1% 미만만 더해진다. 인과적 망의 조기 종료 위에 ours 가 쌓이는지는 이 분석이 말하지 못한다. "
      f"(c) SEENN-I 는 스텝마다 CE 를 거는 학습과 멈춘 스텝까지만 계산하는 망을 전제해 우리 판독과 같은 축에 놓을 수 없다. 스파이크 축 비교에는 학습이 필요하다 (8절).")
    w('')
    # ---- 0
    w('## 0. 왜 스텝별 출력을 그대로 쓸 수 없나')
    w('')
    w('- 구조: R19 의 각 conv 층은 `Conv → BN → 뉴런` 이다 (`models/resnet.py` block_basic 88~100행, 첫 층 330~334행). '
      'BN 노드는 시간 루프에서 `f_temporal_reduction` 분기를 탄다 (`lib_snn/model.py` 3459~3505행, `_run_internal_graph_snn_t_first_k2`; `snn_training_spatial_first=False` 가 기본이라 학습·추론 모두 이 경로; '
      '결정 기록 10-15 항목 (3) 도 같은 경로를 실제 경로로 적음). 입력의 T 스텝 평균 한 장에 BN 을 걸고, 그 결과를 모든 스텝에 복사한다. '
      '`flags.py` 의 `bn_per_step` 주석 (10-15)이 같은 내용이다. `bn_per_step` 은 기본 False 이고 VGGSNN × CIFAR10-DVS 전용이다.')
    w('- 결과: conv1 (입력이 정적 이미지라 상관없음) 다음 층부터는 스텝 1 의 입력 전류가 앞 층 스텝 1~4 발화의 평균이다. '
      'T=4 런의 스텝 t 출력에는 이미 스텝 t+1~4 의 정보가 들어 있다.')
    w('- 출력: 출력층은 누설 없는 적분기다 (`layers.py` 198행 n_type=OUT, `neurons.py` 3073~3087행). 모델 출력은 마지막 스텝 막전위다 (`last_ts`). '
      '점검: npz 의 `logits == logits_t[:, 3]` 이 6런 모두 최대 차 0 이다. 그래서 SEENN-I 의 "누적 출력 평균" 은 `logits_t[:, t] / t` 로 잡았다 '
      '(argmax 는 같고, 신뢰도 크기만 다르다). 막전위를 그대로 쓴 민감도 판도 JSON `curve_raw_cs` 에 있다.')
    w('- **그래서 스텝 t 출력은 스텝 t 에서 계산을 멈춘 출력이 아니다. 몸통 (마지막 BN 까지) 을 4스텝 다 돈 뒤 출력을 일찍 읽는 용도로만 쓸 수 있다.** '
      '3절의 \'실현 가능한 조기 종료\' 는 이 읽기다.')
    w(f"- **직접 확인 (CPU, 6절)**: 같은 체크포인트를 T'=1·2·3 으로 다시 돌린 결과를 T=4 런의 앞 T' 스텝과 비교했다. 근거 규모는 체크포인트 3개 × 200장 (T'=1·2·3) 과 "
      f"체크포인트 6개 × 1000장 (T'=3) 이다. {conv1_txt} "
      "아래 표는 BN 이동통계를 T=4 값 그대로 둔 결과다 (BN 을 다시 잡은 결과는 6.2절).")
    w('')
    w("| 체크포인트 (학습 서버) | 장 수 | T'=1 다시 돌림 / T=4 의 스텝 1 판독 | T'=2 다시 돌림 / 스텝 2 판독 | T'=3 다시 돌림 / 스텝 3 판독 | T=4 |")
    w('|---|---|---|---|---|---|')
    for nm, tag, arm in CPU_RUNS:
        r = cpu['runs'].get(nm)
        if not r or not r['trunc_200']:
            continue
        t = r['trunc_200']
        w(f"| {tag} ({res['runs'][tag]['server']}) | 200 | {t[1]['acc_run']:.1f} / {t[1]['acc_readout_T4']:.1f} | "
          f"{t[2]['acc_run']:.1f} / {t[2]['acc_readout_T4']:.1f} | {t[3]['acc_run']:.1f} / {t[3]['acc_readout_T4']:.1f} | {t[4]['acc_run']:.1f} |")
    w('')
    w("- 조건: T=4 BN 이동통계 고정, T 만 줄임. 위 표의 정확도는 각 200장이라 칸마다 ±1.2~3.5%p (이항 표준오차) 다. T'=1·2 가 우연 수준 (10 클래스 = 10%) 으로 떨어지는 것은 세 체크포인트에서 모두 같다. "
      f"T'=2 의 붕괴는 마지막 층이 덜 쏴서가 아니다. fc1 스파이크는 T'=2 런이 이미지당 {fc1_t2}개, T=4 런의 앞 2스텝이 {fc1_t2p}개다. "
      "어느 층에서 무엇이 바뀌어 틀리는지는 따지지 않았다 (층별 값은 JSON `cpu`).")
    w('')
    # ---- 1
    w('## 1. 사용한 런과 T=4 값')
    w('')
    w('npz 열은 테스트 1만 장, 평가 모드 값이다. 원장 spikes 는 s_count 축 (학습 패스)이다. npz/s_count 비가 팔마다 달라 (아래) 절대 스파이크 수는 인용하지 않는다 (s_count·npz 불일치가 풀릴 때까지 보류). '
      '"꼬리 층" 은 conv4_block2_out_n + fc1_n 이다. 시간 평균 BN 망에서 조기 종료로 뒤 스텝을 건너뛸 수 있는 층은 이 둘뿐이다. "실현 최대 절감" 은 모든 이미지가 스텝 1 에서 끝날 때 아끼는 몫이다 (18층 npz 스파이크 기준).')
    w('')
    w('| 팔 | seed | 학습 서버 | train loss | train acc | val loss | val acc | spikes | npz 정확도 | npz 스파이크/이미지 | 스텝 1 / 2 / 3 / 4 몫 % | 꼬리 층 몫 % | 실현 최대 절감 % | 비고 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for arm in ('baseline', 'ours_inter_ch'):
        knob = '' if arm == 'baseline' else ' 3e-3'
        for lr in res['ledger_rows'][arm]:
            t = lr['tag']
            if t in Rn:
                x = Rn[t]
                sh = ' / '.join(f'{v:.1f}' for v in x['step_share_pct'])
                host = res['npz_extract_host'][t].split(' (')[0]
                w(f"| {arm}{knob} | s{lr['seed']} | {lr['server']} | {lr['train_loss']:.4f} | {lr['train_acc']:.2f} | {lr['val_loss']:.4f} | "
                  f"{lr['val_acc']:.2f} | {f0(lr['spikes'])} | {x['npz_acc']:.2f} | {f0(x['npz_spk'])} | {sh} | {x['tail_share_pct']:.2f} | "
                  f"{x['real_max_saving_pct']:.2f} | npz 추출 {host} |")
            else:
                w(f"| {arm}{knob} | s{lr['seed']} | {lr['server']} | {lr['train_loss']:.4f} | {lr['train_acc']:.2f} | {lr['val_loss']:.4f} | "
                  f"{lr['val_acc']:.2f} | {f0(lr['spikes'])} | – | – | – | – | – | 제외: {EXCL_REASON[t]} |")
        a = A[arm]
        lm = res['ledger_means'][arm]
        rows_in = [lr for lr in res['ledger_rows'][arm] if lr['tag'] in Rn]
        mm = lambda k: np.mean([lr[k] for lr in rows_in])
        sh = ' / '.join(f"{s['mean']:.1f}" for s in a['step_share_pct'])
        w(f"| **{arm}{knob} 평균 (npz 있는 런)** | n=3 | 138 | {mm('train_loss'):.4f} | {mm('train_acc'):.2f} | {mm('val_loss'):.4f} | "
          f"**{mm('val_acc'):.2f}** | {f0(mm('spikes'))} | {pm(a['npz_acc'])} | {f0(a['npz_spk']['mean'])} ±{f0(a['npz_spk']['sd'])} | {sh} | "
          f"{pm(a['tail_share_pct'])} | {pm(a['real_max_saving_pct'])} | 원장 전체 n={lm['val_acc']['n']}: val acc {pm(lm['val_acc'])}, spikes {f0(lm['spikes']['mean'])} |")
    w('')
    w(f"- 같은 6런 (n=3 대 3, 원장 n=6/4 중 부분집합), ours_inter_ch − baseline 스파이크: **npz 축 {ci(C['T4_ln_spk'], 1, ('pct', 'pct_lo', 'pct_hi'), '%')}, "
      f"s_count 축 {ci(sc, 1, ('pct', 'pct_lo', 'pct_hi'), '%')}**. 두 축이 약 {abs(C['T4_ln_spk']['pct'] - sc['pct']):.1f}%p 다른 것은 npz/s_count 비가 팔마다 달라서다 "
      f"(baseline {rng(ratio_b, 3)}, ours {rng(ratio_o, 3)}). 정확도 차 {ci(C['T4_acc'])}%p 는 시드 SD (0.10~0.13%p) 보다 작아 n=3 으로는 가리지 못한다. "
      f"(10-19 분석의 −27.4% 는 ours n=4 대 baseline n=3 이다.)")
    w('')
    # ---- 2
    w('## 2. 스텝별 스파이크 몫 (요청 항목, 서술만)')
    w('')
    w('| 스텝 | baseline (n=3) % | ours_inter_ch 3e-3 (n=3) % | 차 (ours − baseline) %p [95%] | p (Welch) / Holm | 스텝 t 판독 정확도 baseline / ours (비인과, 참고) |')
    w('|---|---|---|---|---|---|')
    for t in range(4):
        st = C['step_share_pct'][t]
        w(f"| {t + 1} | {pm(b['step_share_pct'][t])} | {pm(o['step_share_pct'][t])} | {ci(st)} | {st['p']:.3f} / {st['p_holm']:.3f} | "
          f"{pm(b['readout_acc_t'][t])} / {pm(o['readout_acc_t'][t])} |")
    w('')
    cor = {c['step']: c for c in N32['corr_share_vs_npz_spk']}
    G = N32['same_spike_groups']
    hp_ = [C['step_share_pct'][t]['p_holm'] for t in range(4)]
    edge = [str(t + 1) for t in range(4) if 0.04 <= hp_[t] < 0.05]
    w(f"- Holm 보정 p 가 0.05 아래인 스텝: {', '.join(str(t + 1) for t in range(4) if hp_[t] < 0.05) or '없음'} (n=3 대 3)."
      + (f" 스텝 {'·'.join(edge)} 은 0.04~0.05 로 경계다." if edge else '') + " 몫은 합이 100 이라 네 검정은 독립이 아니다.")
    w(f"- **이 차이를 'ours 가 발화를 뒤 스텝으로 민다' 로 읽지 않는다.** 스텝별 배열이 있는 npz {N32['n_runs']}런 전체에서 스텝 1 몫은 총 스파이크와 r={cor[1]['r']:+.2f}, "
      f"스텝 4 몫은 r={cor[4]['r']:+.2f} 다 (npz 축, 방법 섞임). 스파이크가 줄면 뒤 스텝 몫이 커지는 경향이라, baseline 대비 차이는 스파이크 수준으로 설명된다. "
      "같은 스파이크 수준 (원장 축 32~35만) 에서 보면 방향이 반대다 — ours 가 대조군보다 덜 뒤로 쏠려 있다:")
    w('')
    w('| 묶음 | n | 학습 서버 | 원장 spikes | 스텝 1 몫 % | 스텝 4 몫 % |')
    w('|---|---|---|---|---|---|')
    for g, d in G.items():
        if g.startswith('baseline'):
            continue
        w(f"| {g} | {d['n']} | {'/'.join(d['servers'])} | {', '.join(f0(v) for v in d['ledger_spk'])} | "
          f"{', '.join(f'{v:.2f}' for v in d['step1'])} | {', '.join(f'{v:.2f}' for v in d['step4'])} |")
    w('')
    w('- 위 표는 n 이 1~4 로 고르지 않고 (BPSR n=1, plain L2 n=2) 같은 스파이크 ANCOVA 도 하지 않았다. 방향만 적는다. 게다가 시간 평균 BN 망이라 이 몫에는 조기 종료상의 의미가 없다 (0절).')
    w(f"- 판독 정확도 열은 비인과 값이다. 스텝 2 판독 − T=4 정확도는 {gap2}%p (6런)다. 하지만 같은 체크포인트를 BN 고정 T'=2 로 돌리면 우연 수준이다 (0절 표).")
    w('')
    # ---- 3
    w('## 3. 실현 가능한 조기 종료 — 시간 평균 BN 망에서, 18층 npz 스파이크 기준')
    w('')
    w('- 규칙: SEENN-I 와 같다. 스텝 t 의 누적 출력 평균 (`logits_t[:, t] / t`) 의 softmax 최댓값이 τ 이상이면 그 스텝 예측을 쓰고 끝낸다. '
      '끝까지 넘지 못하면 스텝 4 에서 끝낸다. τ 는 스텝 1~3 신뢰도의 분위수 2,001개로 훑었다. 몸통은 4스텝을 다 돌고, 출력만 일찍 읽는다 (0절).')
    w('- 스파이크 회계: 꼬리 층 (conv4_block2_out_n, fc1_n) 의 종료 뒤 스텝만 뺀다. 이미지마다 그 층의 스텝별 발화가 테스트셋 평균과 같다고 근사했다 (npz 는 이미지별 스텝별 값이 없음). '
      f'꼬리 층 전체가 스파이크의 {tail_rng}% (6런) 라 이 근사의 영향은 그보다 작다.')
    w('- τ 는 같은 테스트셋에서 골랐다 (낙관적). val 셋 = 테스트 셋이라 따로 고를 데이터가 없다.')
    w('')
    w('| 허용 손실 (자기 T=4 대비, %p) | baseline 절감 % (n=3) | baseline 평균 T | ours 절감 % (n=3) | ours 평균 T | 같은 Δ 에서 ours/baseline 스파이크 % [95%] |')
    w('|---|---|---|---|---|---|')
    for k in b['at_delta_spk_real_cs']:
        bb, oo = b['at_delta_spk_real_cs'][k], o['at_delta_spk_real_cs'][k]
        cc = C[f'ee_spk_real_{k}']
        w(f"| {k} | {pm(bb['red_pct'])} | {pm(bb['meanT'])} | {pm(oo['red_pct'])} | {pm(oo['meanT'])} | "
          f"{ci(cc, 1, ('pct', 'pct_lo', 'pct_hi'), '%')} |")
    w('')
    rmr = N32['real_max_saving_range']
    w(f"- 어떤 τ 로도 아낄 수 있는 스파이크는 최대 baseline {pm(b['real_max_saving_pct'])}%, ours {pm(o['real_max_saving_pct'])}% (각 n=3) 다. "
      f"그 최대를 쓰려면 모든 이미지가 스텝 1 에서 끝나야 하고, 그러면 정확도가 스텝 1 판독값 ({pm(b['readout_acc_t'][0])}%) 으로 떨어진다. "
      "이 상한은 테스트셋 평균 근사가 들어가지 않는다 (모든 이미지가 같은 스텝에서 끝나므로).")
    w(f"- 경계: (a) 지금 체크포인트를 시간 평균 BN 그대로 실행할 때만 성립한다. (b) 18층 npz 스파이크 수 기준이고, 계산량이나 지연 시간이 아니다. "
      f"(c) 꼬리 층 몫은 총 스파이크가 적을수록 커진다. npz {N32['n_runs']}런 전체가 {rmr[0]:.2f}~{rmr[1]:.2f}% 라 ours 의 상한이 baseline 보다 큰 것은 ours 의 성질이 아니다. "
      "(d) 인과적 망 (SEENN 류) 의 조기 종료 이득과는 무관하다. **그래서 이 결과를 'ours 가 조기 종료와 직교한다' 는 근거로 인용하지 않는다.**")
    w(f"- 라벨 없이 T=4 예측을 그대로 재현하는 최소 스텝 (예측이 그 뒤로 안 바뀌는 첫 스텝) 오라클: 평균 T baseline {pm(b['stable_oracle_meanT'])}, "
      f"ours {pm(o['stable_oracle_meanT'])} (n=3), 실현 절감 baseline {pm(b['stable_oracle_real_red_pct'])}%. 판독 자체는 비인과라 이 평균 T 는 계산 시간이 아니다.")
    w('')
    # ---- 4
    cr0 = C['ee_spk_real_+0.00']
    w('## 4. 질문별 답')
    w('')
    w(f"**(a) 조기 종료만으로 baseline 정확도에서 ours_inter_ch 만큼 줄어드나? — T=4 를 끝까지 도는 지금 망에서는 아니다.** 실현 가능한 조기 종료의 절감은 손실 0 에서 "
      f"{pm(b['at_delta_spk_real_cs']['+0.00']['red_pct'])}%, 손실 0.10%p 허용에서 {pm(b['at_delta_spk_real_cs']['-0.10']['red_pct'])}%, "
      f"상한 {pm(b['real_max_saving_pct'])}% 다 (모두 baseline n=3, 시간 평균 BN 망, 18층 npz 스파이크). ours_inter_ch 3e-3 은 npz 축 {C['T4_ln_spk']['pct']:+.1f}% "
      f"[{C['T4_ln_spk']['pct_lo']:+.1f}, {C['T4_ln_spk']['pct_hi']:+.1f}], s_count 축 {sc['pct']:+.1f}% [{sc['pct_lo']:+.1f}, {sc['pct_hi']:+.1f}] (n=3 대 3) 다. "
      f"T 를 실제로 줄여 다시 돌리는 것은 조기 종료가 아니라 \'모든 이미지 같은 T\' 이고, 결과는 BN 처리에 달렸다 (6절).")
    w('')
    w(f"**(b) ours 위에 조기 종료가 더해지나? — T=4 를 끝까지 도는 지금 망에서는 1% 미만이다** (ours 상한 {pm(o['real_max_saving_pct'])}%, n=3). 각 팔의 T=4 정확도에서 손실 0 으로 맞추면 "
      f"ours+조기 종료 / baseline+조기 종료 스파이크는 {ci(cr0, 1, ('pct', 'pct_lo', 'pct_hi'), '%')} (n=3 대 3)다. "
      f"T=4 끼리의 {C['T4_ln_spk']['pct']:+.1f}% 와 사실상 같다. 조기 종료 절감 크기의 팔 간 차 (손실 0): {ci(C['ee_saving_spk_real_+0.00'], 3)}%p. "
      "경계: (a) 상한이 1.1% 이하라 이 결과는 정의에서 거의 따라 나온다 (3절). (b) τ 를 테스트셋에서 골라 낙관적이다. (c) '손실 0' 은 각 팔 자기 T=4 정확도 기준이고 공통 정확도가 아니다 "
      f"(두 팔의 T=4 정확도 차 {C['T4_acc']['d']:+.2f}%p 는 n=3 대 3 으로 가리지 못함). (d) 인과적 망의 조기 종료는 스파이크를 수십 % 줄일 수 있다 (SEENN 평균 T 1.09~1.34). "
      "**그 위에 ours 가 쌓이는지는 이 분석이 말하지 못한다. 'ours 는 조기 종료와 직교·보완' 같은 일반 주장으로 옮기지 않는다.**")
    w('')
    w('**(c) SEENN-I 와의 비교 — 같은 축에 놓을 수 없다.**')
    w('')
    w('| | SEENN-I (보고값, n 미기재) | 우리 baseline, T=4 계산 안 판독 (n=3) | 우리 ours_inter_ch, 같은 판독 (n=3) |')
    w('|---|---|---|---|')
    for i, (acc_s, m) in enumerate(SEENN['points']):
        bb, oo = b['at_meanT_seenn'][i], o['at_meanT_seenn'][i]
        w(f"| 평균 T {m:.2f} | {acc_s:.2f} (자기 T=4 {SEENN['own_T4']:.2f} 대비 {acc_s - SEENN['own_T4']:+.2f}) | "
          f"정확도 {pm(bb['acc'])}, 실현 절감 {pm(bb['real_red_pct'])}% | 정확도 {pm(oo['acc'])}, 실현 절감 {pm(oo['real_red_pct'])}% |")
    w(f"| 라벨 오라클 평균 T — empirical AET (식 13, 처음 맞는 스텝) | {SEENN['AET_empirical_eq13']:.4f} (Table 5) | {pm(b['label_oracle_AET'], 3)} (비인과 판독) | {pm(o['label_oracle_AET'], 3)} (비인과 판독) |")
    w(f"| 참고: AET (식 4, 가정 3.1 '한 번 맞으면 뒤로도 맞음' 아래) | {SEENN['AET_eq4']:.4f} (Table 5, Fig. 4 의 ★1.131) | 계산 안 함 | 계산 안 함 |")
    w('')
    w('- 출처: arXiv 2304.01230v2. Table 1 (세 점), Fig. 3 과 Fig. 4 왼쪽 (SNN T=4 96.46; 검토 전 판의 \'Fig. 1\' 은 틀림 — Fig. 1 은 TET R19 94.16/94.44/94.50), Table 5 (AET 두 정의). '
      '우리 라벨 오라클 (`first_ok`) 은 empirical AET 정의와 같다.')
    w('- 프로토콜 차이: (1) SEENN 은 스텝마다 CE 를 건다 (식 11, TET). 우리 학습은 마지막 스텝 출력에만 CE 를 건다 (`last_ts`). '
      '(2) SEENN 은 평균 T 를 비용으로 센다. 멈춘 스텝 뒤로는 계산하지 않는 망 (시간에 인과적)을 전제한다. TET 코드 틀 위에 구현했다 (§4.1). 우리 R19 는 시간 평균 BN 이라 그 전제가 서지 않는다. '
      '(3) 레시피: SEENN 은 LIF τ=0.5 (누설), 발화 시 0 으로 리셋, SGD 300에폭 (lr 0.1 cosine, wd 5e-4), Cutout + AutoAugment (§3.1 식 1, §4.1). '
      '우리 baseline 은 AdamW 310에폭, RandAugment·Random Erasing·CutMix (반복 10만까지) 이고 원장 96.73 ±0.10 (n=6) 이다. SEENN 자기 T=4 는 96.46 이다. '
      '(4) SEENN 은 스파이크 수를 보고하지 않는다. 연산 수 기반 샘플당 에너지 (부록 D: 덧셈 0.9pJ, 곱셈 4.6pJ) 가 Fig. 3 에 막대그림으로만 있고 수치 표는 없다. '
      '우리 에너지 추정과 같은 정의인지는 확인하지 않았다. '
      '(5) 우리 판독에서 평균 T 1.09~1.34 의 정확도가 SEENN 보다 낮다. 스텝 1 출력을 학습하지 않았기 때문일 것으로 **추정**한다 (검증 안 함). 그 판독조차 몸통 4스텝을 다 돌아야 얻는다.')
    scv = res['seenn_conversion_assumption']['points']
    w(f"- 가정 계산 (측정 아님): SEENN 의 평균 T 1.09 / 1.20 / 1.34 는, 스파이크가 스텝 수에 비례하면 T=4 의 "
      f"{100 * scv[0]['spk_frac_if_proportional_to_T']:.0f} / {100 * scv[1]['spk_frac_if_proportional_to_T']:.0f} / {100 * scv[2]['spk_frac_if_proportional_to_T']:.0f}% 다. "
      f"우리 baseline 의 스텝별 몫을 그대로 쓰면 {100 * scv[0]['spk_frac_bounds_if_baseline_step_profile'][0]:.1f}~{100 * scv[0]['spk_frac_bounds_if_baseline_step_profile'][1]:.1f} / "
      f"{100 * scv[1]['spk_frac_bounds_if_baseline_step_profile'][0]:.1f}~{100 * scv[1]['spk_frac_bounds_if_baseline_step_profile'][1]:.1f} / "
      f"{100 * scv[2]['spk_frac_bounds_if_baseline_step_profile'][0]:.1f}~{100 * scv[2]['spk_frac_bounds_if_baseline_step_profile'][1]:.1f}% 다 (종료 분포에 따른 최소~최대). "
      f"어느 가정이든 ours_inter_ch 의 −28~−30% 보다 훨씬 깊다. 다만 이는 SEENN 레시피 (스텝별 CE + 인과적 망)를 R19 에 학습해야 확인된다. "
      f"**'SEENN 류가 스파이크 축에서 우리를 앞설 수 있다'는 compare-candidates 4절 (우리가 질 수 있는 곳) 의 위협은 이 분석으로 해소되지 않는다.**")
    w('')
    # ---- 5
    w('## 5. 이렇게 읽으면 틀린다 — 비인과 판독 (인용 금지)')
    w('')
    w('요청 문구 그대로 계산하면 (정확도 = 스텝 t 판독, 스파이크 = 스텝 1..t 의 합) 아래 값이 나온다. 0절과 6절의 이유로 **실현할 수 없는 값** 이다. '
      '나중에 누가 npz 로 같은 계산을 하더라도 이 숫자를 쓰지 않게 하려고 남긴다 (실제로 다른 문서들이 이 읽기를 썼다, 7절).')
    w('')
    w('| 허용 손실 %p | baseline "절감" % (n=3) | ours "절감" % (n=3) | ours/baseline "스파이크" % [95%] |')
    w('|---|---|---|---|')
    for k in b['at_delta_spk_prefix_cs']:
        bb, oo = b['at_delta_spk_prefix_cs'][k], o['at_delta_spk_prefix_cs'][k]
        cc = C[f'ee_spk_prefix_{k}']
        w(f"| {k} | {pm(bb['red_pct'], 1)} | {pm(oo['red_pct'], 1)} | {ci(cc, 1, ('pct', 'pct_lo', 'pct_hi'), '%')} |")
    w('')
    w(f"- 예: 이 회계로는 baseline 이 손실 0.05%p 안에서 {b['at_delta_spk_prefix_cs']['-0.05']['red_pct']['mean']:.0f}% 를 \"줄인다\". "
      f"같은 체크포인트를 BN 고정 T'=3 으로 실제로 돌리면 스파이크는 {t3_spk}% 지만 정확도가 체크포인트마다 {t3_loss}%p 바뀐다 (1000장, 체크포인트 6개, 6.1절).")
    w('')
    # ---- 6
    w('## 6. CPU 점검 — 재학습 없이 T 를 줄여 다시 돌림')
    w('')
    w('- 방법: `extract_neuron_spikes.py` 와 같은 순서로 모델을 올린다. `config.set()` 을 원래 T=4 로 부른 뒤 (경로 이름에 ts-4 가 들어가므로), 모델을 만들기 직전에 `conf.time_step` 만 바꾼다. '
      'CPU 만 썼다 (`CUDA_VISIBLE_DEVICES=""`). 스크립트는 py 의 `CPU_SCRIPT` (6.1) 와 `RECAL_SCRIPT` (6.2) 다. 평가는 테스트셋 앞 1000장이다 (200장 표는 0절).')
    hs = [r['T3_1k']['argmax_same_cpu_gpu_pct'] for r in c1k]
    w(f"- 하네스 점검: CPU T=4 와 GPU npz 의 같은 1000장 예측 일치는 {rng(hs, 1)}% (체크포인트 {len(hs)}개) 다. 나머지는 CPU·GPU 부동소수 차로 일부 발화가 뒤집힌 것이다.")
    w('')
    w("### 6.1 BN 이동통계 고정 (T=4 값 그대로), T 만 줄임")
    w('')
    w("| 체크포인트 (팔, 학습 서버) | 장 수 | T=4 정확도 (CPU) | T'=3 정확도 | 대응 차 %p (잃음/얻음, 정확 p) | T'=3 스파이크 % (T=4 대비) | T=4 런 스텝 3 판독 정확도 (비인과) |")
    w('|---|---|---|---|---|---|---|')
    for nm, tag, arm in CPU_RUNS:
        r = cpu['runs'].get(nm)
        if not r or not r['T3_1k']:
            continue
        k = r['T3_1k']
        dd = k['diff']
        w(f"| {tag} ({arm}, {res['runs'][tag]['server']}) | {k['n_img']} | {k['acc_T4_cpu']:.1f} | {k['acc_T3']:.1f} | "
          f"{dd['d_pp']:+.1f} ({dd['lost']}/{dd['gained']}, p={dd['p_exact']:.2g}) | {-k['spk_red_pct']:+.1f} | {k['acc_readout_step3_T4']:.1f} |")
    for arm, c in cpu['T3_1k_by_arm'].items():
        w(f"| **{arm} (체크포인트 {c['K']}개)** | {c['n_img']} | | | **{fcl(c)}** | -{pm(c['spk_red_pct'], 1)} | |")
    w('')
    sup, f3 = cpu.get('pooled_T3_1k_wald_superseded'), cpu.get('T3_1k_first3_clustered')
    allc = cpu.get('T3_1k_all_ckpts_ref')
    w(f"- 팔마다 체크포인트 3개가 모두 손실 쪽이다. 손실 구간의 위 끝은 baseline {ba['t_ckpt']['hi']:+.2f}, ours {oa['t_ckpt']['hi']:+.2f}%p (체크포인트 수준 t, df=2) 로 −0.2%p 보다 낮다 — 비열등 마진 안에 들지 않는다.")
    if sup and f3:
        w(f"- 검토 전 판의 합친 구간 {sup['d_pp']:+.2f} [{sup['lo_pp']:+.2f}, {sup['hi_pp']:+.2f}] (체크포인트 3개 = baseline 2 + ours 1, 3000쌍 Wald) 은 같은 이미지 반복과 체크포인트 간 분산을 빠뜨려 좁았다. "
          f"같은 3개로 다시 내면 {fcl(f3)} 다. 팔이 섞인 6개 합침은 참고만: {fcl(allc)}.")
    w(f"- 팔 간 차 (ours − baseline, 손실 크기) {ci(cpu['T3_1k_arm_diff'])}%p (체크포인트 3개 대 3개, Welch). 체크포인트 간 SD 가 baseline {ba['t_ckpt']['sd']:.2f} / ours {oa['t_ckpt']['sd']:.2f}%p 라 판단하지 않는다.")
    w(f"- T'=3 런의 총 스파이크는 T=4 런의 앞 3스텝 합보다 {t3_vs_prefix}% 다르다 (1000장). 층별로 보면 conv2 이후가 다르다 (JSON `cpu.runs.*.T3_1k.layer_spike_maxdiff_vs_T4_prefix`). 총합이 비슷하다고 같은 계산은 아니다.")
    w('')
    w("### 6.2 BN 이동통계만 T' 에서 다시 잡음 (학습 없음, 가중치 그대로) — 검토 반영 2·3번")
    w('')
    w("- 방법 (precise BN): T' 로 만든 모델에 학습셋 앞 이미지를 흘린다. 이 동안 각 BN 은 그 배치의 통계로 정규화하고, "
      "배치 평균 m_b·분산 v_b 를 모아 모집단 평균 = mean(m_b), 분산 = mean(v_b) + var(m_b) 로 이동통계를 바꾼다 (BN 20층). 정규화 식은 원래 코드 경로 그대로다. "
      "테스트 전처리는 test_ds 첫 배치와 최대 차 0 으로 대조했다. 체크포인트 파일에는 쓰지 않았다. 평가는 테스트 앞 1000장 (6.1 과 같은 이미지) 이다.")
    w(f"- 두 판. **주 판 (aug)**: {RECAL_MODES['aug']['desc']}. 원래 이동통계를 잡은 조건 (학습 후반, 같은 증강, 순전파는 배치 통계) 과 맞췄다. "
      f"**민감도 판 (clean)**: {RECAL_MODES['clean']['desc']}. 두 판 모두 T'=4 에도 같은 절차를 걸어, \'재추정 T\'=4 − 원래 T=4\' 로 절차 자체의 효과를 잰다.")
    w('')
    w("| 체크포인트 (팔) | 원래 T=4 | aug T'=4 / 3 / 2 | clean T'=4 / 3 / 2 / 1 | 스파이크 % (원래 T=4 대비) aug T'=3 / 2 | BN 평균 이동 (원래 SD 단위, 층 중앙값) aug T'=4 / 3 / 2 · clean T'=4 |")
    w('|---|---|---|---|---|---|')
    for nm, tag, arm in CPU_RUNS:
        ra, rcl = RA['runs'].get(nm), RCl['runs'].get(nm)
        if not ra and not rcl:
            continue
        ga = lambda T, k, f='{:.1f}': (f.format(ra['T'][T][k]) if ra and T in ra['T'] else '–')
        gc = lambda T, k, f='{:.1f}': (f.format(rcl['T'][T][k]) if rcl and T in rcl['T'] else '–')
        acc0 = (ra or rcl)['acc_orig_T4']
        w(f"| {tag} ({arm}) | {acc0:.1f} | {ga(4, 'acc')} / {ga(3, 'acc')} / {ga(2, 'acc')} | {gc(4, 'acc')} / {gc(3, 'acc')} / {gc(2, 'acc')} / {gc(1, 'acc')} | "
          f"{ga(3, 'spk_vs_orig_T4_pct', '{:+.1f}')} / {ga(2, 'spk_vs_orig_T4_pct', '{:+.1f}')} | "
          f"{ga(4, 'bn_shift_median_sd', '{:.3f}')} / {ga(3, 'bn_shift_median_sd', '{:.3f}')} / {ga(2, 'bn_shift_median_sd', '{:.3f}')} · {gc(4, 'bn_shift_median_sd', '{:.3f}')} |")
    w('')
    w("| 판, 팔, T' | 체크포인트 | 정확도 (1000장) | 차: 재추정 T' − 원래 T=4 (%p) [이원 부트스트랩] | 체크포인트별 차 | 체크포인트 수준 t 95% | 차: 재추정 T' − 같은 판 T'=4 (%p) | 스파이크 % (원래 T=4 대비) |")
    w('|---|---|---|---|---|---|---|---|')
    for RR in (RA, RCl):
        for arm, AA in RR['by_arm'].items():
            for T in RECAL_MODES[RR['mode']]['T']:
                if T not in AA:
                    continue
                e = AA[T]
                vt = e['vs_orig_T4']
                w(f"| {RR['mode']}, {arm}, T'={T} | {len(e['ckpts'])} | {pm(e['acc'], 2)} | {fcs(vt)} | {', '.join(f'{v:+.1f}' for v in vt['per_ckpt_pp'])} | "
                  f"[{vt['t_ckpt']['lo']:+.2f}, {vt['t_ckpt']['hi']:+.2f}] | {fcs(e['vs_recal_T4']) if 'vs_recal_T4' in e else '–'} | {pm(e['spk_vs_orig_T4_pct'], 1)} |")
    w('')
    w(recal_discuss(res))
    w('')
    # ---- 7
    w('## 7. 다른 문서에 대한 함의 (이 문서는 고치지 않음, 각 담당이 고칠 것)')
    w('')
    w('- compare-candidates 1.1 의 "우리 baseline 은 스텝 1 의 스파이크 몫이 3.7% 라서 일찍 끝내면 스파이크가 크게 줄 수 있다" 는 지금 망에 맞지 않는다. 스텝 1 의 발화도 스텝 2~4 를 다 돌아야 계산된다.')
    w(sec7_t(res))
    w('- 3.5 판정표의 "조기 종료 모의가 0.1pp 이하 손실로 −30% 를 넘김" 행은 npz 판독으로는 판정할 수 없다. 모의가 성립하지 않아서이지, SEENN 류가 진다는 근거가 아니다. '
      "가장 가까운 실측은 6.2 의 \'T'=3 + BN 재추정\' 이고, 스파이크는 −30% 를 넘지만 정확도 손실이 0.1%p 이하인지는 1000장 × 체크포인트 3개로 판정할 수 없다 (8절 첫 제안).")
    w('- **같은 오해 (`logits_t[:, t]` = \'t 스텝에서 멈춘 출력\') 가 다른 문서 5곳에 있다.** 숫자 자체는 판독값으로 맞지만, \'추론을 t 스텝에서 끊으면\' 으로 읽은 결론은 6절 결과와 충돌한다.')
    w('  - `analysis/fr_sops_261020.md` 3b절·요약 3번·5절 \'baseline, 추론 T=3\' 행·6절 1번: "스텝 t 까지의 계산은 뒤 스텝과 무관하므로 … 정확히 얻는다 (모의가 아님)" 와 '
      f"\"추론만 한 스텝 일찍 끊은 baseline 이 ours 보다 깊다 (T=3 Δ -0.02, 스파이크 -38.3%)\". BN 고정으로 실제 T'=3 은 baseline {fcs(ba)} (이원 부트스트랩, 체크포인트 3개 × 1000장) 이다 (6.1절). {frsops_recal(res)}")
    w('  - `analysis/inter_vs_bpsr_addendum_261010.md` 머리 3번·19행·3절 ("T\'=k 에서 멈추면"), `analysis/ablation_mech_261010.md` 6절 ("T 를 줄여 추론하면"), '
      '`analysis/ablation_mech_addendum_261010.md` 2c절, `docs/decisions.md` 10-10 항목 (384행 "스텝 3/4 에서 멈춰도 정확도 −0.10 이하", 387행 "T\'=2 에서 멈추면 스파이크 −73%").')
    w('')
    # ---- 8
    w('## 8. 제안 (실행 안 함, 사용자 결정)')
    w('')
    w('- SEENN 식 비교 (인과적 망 + 스텝별 학습) 를 스파이크 축으로 하려면 학습이 필요하다. 길은 두 가지다.')
    w('  - (1) TF, 같은 레시피: `bn_per_step` 을 R19 로 넓히고 (지금은 VGGSNN × DVS 밖에서 켜면 model_builder 가 멈춤), **스텝별 CE (TET) 손실을 새로 구현**한 뒤 학습한다. '
      'TF 에는 TET 손실 플래그가 없다 (`snn_out_proc=\'all\'` 로 스텝별 출력을 꺼낼 수만 있음). 학습 경로 변경이라 연기시험과 배포 승인이 필요하다 (3등급).')
    w('  - (2) 별도 env·폴더: 공개 SEENN PyTorch 코드 (github.com/Intelligent-Computing-Lab-Yale/SEENN, TET 틀) 로 R19 C10 을 돌리고 스파이크를 센다. TF 학습 경로를 건드리지 않는다 '
      '(CLAUDE.md 가 Codex 에 맡길 일로 둔 \'PyTorch 재현\'). 프레임워크·레시피 차이 (뉴런 τ·리셋, 옵티마이저, 증강) 를 함께 적어야 한다.')
    w(sec8(res))
    w('')
    # ---- LIMITATION
    w('## [LIMITATION]')
    w('')
    w('- **이 분석은 SEENN 류 (스텝별 학습 + 인과적 망)가 우리 R19 C10 에서 스파이크 축으로 ours_inter_ch 를 앞서는지 말할 수 없다.** 학습이 필요하다 (8절의 두 길).')
    w('- 3절의 판독·조기 종료 정확도는 실현 가능한 읽기다 (몸통 4스텝을 다 돈 뒤 출력만 일찍 읽음). 그때의 "평균 T" 는 계산 시간이 아니다.')
    w(f'- 실현 스파이크 회계는 꼬리 층의 스텝별 발화를 이미지와 무관한 테스트셋 평균으로 근사했다. 꼬리 층이 전체의 {tail_rng}% 라 결론에는 영향이 없다. 스파이크는 18층 npz 수이고 계산량·지연이 아니다.')
    w('- npz 런: 스텝별 배열이 있는 baseline s11~s13 과 ours 3e-3 s2~s4 만 썼다 (각 n=3, 학습 서버 138, npz 추출 기계는 canus 와 55 서버). 원장 전체 (baseline n=6, ours n=4) 로는 재지 않았다. 서버 23 런은 포함되지 않았다.')
    w('- CPU 점검: 200장 표는 체크포인트 3개, T\'=3 과 BN 재추정은 체크포인트 6개 (팔당 3개) × 1000장이다. 정확도 차의 체크포인트 수준 구간은 df=2 라 넓다. 팔 간 차는 판단하지 않는다.')
    w('- **학습 없는 T 감소 (T\'=3 + BN 재추정) 의 정확도 손실은 이 분석으로 정해지지 않았다.** 1000장 × 팔당 체크포인트 3개라 이원 부트스트랩 구간이 약 2%p 폭이다. 0 인지 0.5~1%p 인지에 따라 ours 와의 비교 결론이 바뀐다 (8절 첫 제안).')
    w('- BN 재추정은 두 설정 (증강 5000장, 무증강 3000장) 만 했다. 장 수·난수를 바꾼 민감도는 재지 않았다. 절반 표본 대비 통계 차는 JSON `recal.*.runs.*.T.*.conv_half_vs_full_*` 에 있다. '
      '평가는 1000장이라 체크포인트당 정확도 차의 표준오차가 약 0.3~0.7%p 다.')
    w('- τ 는 테스트셋에서 골랐다. val = test 라 고를 데이터가 따로 없다.')
    w('- 2절의 같은 스파이크 대조군 비교는 n 이 1~4 이고 ANCOVA 를 하지 않은 서술이다.')
    w('- SEENN 수치는 arXiv v2 PDF 에서 직접 읽었다. NeurIPS proceedings 판과 대조하지 않았다. SEENN 코드의 BN 구현은 읽지 않았다. "인과적 망" 은 평균 T 를 비용으로 셌다는 점에서 추론한 것이다.')
    w('')
    # ---- 검토 반영
    w('## 검토 반영 (10-20 eip-critic, needs_fixes 9건)')
    w('')
    w('| # | 지적 (요지) | 처리 | 바뀐 곳 |')
    w('|---|---|---|---|')
    for row in REVIEW_ROWS(res):
        w('| ' + ' | '.join(row) + ' |')
    w('')
    open(OUT_MD, 'w').write('\n'.join(L) + '\n')
    print('[md]', OUT_MD)


def REVIEW_ROWS(res):
    C, cpu, N32 = res['compare_ours_minus_baseline'], res['cpu'], res['npz32']
    sc = C['T4_ln_spk_scount']
    f3 = cpu.get('T3_1k_first3_clustered')
    cor = {c['step']: c['r'] for c in N32['corr_share_vs_npz_spk']}
    hp = [C['step_share_pct'][t]['p_holm'] for t in range(4)]
    rmr = N32['real_max_saving_range']
    c1ok = sum(1 for o in cpu['runs'].values() if o.get('T3_1k_cell') and o['T3_1k_cell']['conv1_conv_n']['cells_diff'] == 0)
    return [
        ['1', '스텝 t 출력 비인과 — 통과, 문구만 (3절이 같은 출력을 \'실현 가능한 조기 종료\' 로 쓰는 것과 충돌)',
         '받아들임. "스텝 t 출력은 스텝 t 에서 멈춘 출력이 아니다. 몸통을 4스텝 다 돈 뒤 일찍 읽는 용도로만 쓸 수 있다" 로 고치고 근거 규모를 적었다. '
         f"체크포인트 6개 × 1000장 (T'=3) 을 더 돌렸다: conv1 은 {c1ok}개가 비트 단위로 같고 나머지도 부동소수 수준 차뿐, conv2_block1_conv1 부터 크게 다름 (6개 모두)",
         '한 줄, 0절, 3절 머리'],
        ['2', 'T\'=2 붕괴는 \'T=4 BN 이동통계 고정\' 조건뿐 — 7절의 "재학습 없는 T 감소는 쓸 수 없다" 는 근거를 넘음',
         '받아들임. 조건을 모든 T 감소 수치에 붙였다. 7절 일반화 문장을 뺐다. 살리는 측정 (BN 이동통계를 T\' 에서 학습셋으로 다시 잡기, CPU, 학습 없음) 을 체크포인트 6개에 두 판으로 했다: '
         '증강판 (학습 후반과 같은 증강, 5000장, T\'=2~4) 과 무증강판 (3000장, T\'=1~4). 무증강판은 T\'=4 에서도 정확도를 떨어뜨려 (절차 효과) 증강판을 주 판으로 둔다. '
         '결과: 증강판 T\'=3 은 BN 고정 손실의 대부분을 되찾아 \'학습 없는 T 감소\' 가 ours 를 위협하는 대조가 될 수 있음 — 판정이 바뀐 가장 큰 항목',
         '한 줄, 0절, 6.2절 (신설), 7절'],
        ['3', 'T\'=3 합친 CI 가 3000쌍 독립 Wald 라 과소 — 체크포인트 수준 구간으로',
         f"받아들임. 체크포인트별 값, 팔별 체크포인트 수준 t (df=2), 이미지 군집·이원 부트스트랩 (B={BOOT_B}) 을 낸다. 체크포인트를 팔당 3개로 늘렸다 (base13·ours2·ours3 추가). "
         f"검토 전 3개로 다시 낸 값 {fcl(f3) if f3 else '–'} — 검토자 값 (이미지 [-4.70, -2.67], 이원 [-5.00, -2.43], t [-5.08, -2.25]) 과 부트스트랩 난수 차 안에서 맞음. "
         "팔이 섞인 합침은 참고로만 둔다. 논문용 대조 (BN 재추정 T\'=3, T=2/3 학습 baseline) 는 8절",
         '6.1절, 6.2절, 8절'],
        ['4', '조기 종료 상한 0.82/1.07% — 통과, 경계 명시',
         f"받아들임. 표·본문에 '시간 평균 BN 망, 18층 npz 스파이크 기준' 을 붙이고 경계 (a)~(d) 를 적었다. npz {N32['n_runs']}런 상한 범위 {rmr[0]:.2f}~{rmr[1]:.2f}% 를 직접 계산. 'ours 가 조기 종료와 직교' 근거로 쓰지 말 것을 명시",
         '한 줄, 1절, 3절'],
        ['5', '-27.8% 의 서버 표기와 축', f"받아들임. '학습 서버 138, npz 추출 기계 섞임 (ours s2·s3 = 55 서버)' 로 고침 (npz 로그의 load weight 경로로 확인). s_count 축 {sc['pct']:+.1f}% [{sc['pct_lo']:+.1f}, {sc['pct_hi']:+.1f}] 를 함께 적고 npz/s_count 비를 팔별로 적음. 절대 스파이크 수는 인용 보류로 표시",
         '머리, 1절, 4절 (a)'],
        ['6', '조기 종료를 양쪽에 걸면 -27.9% — 통과, 경계 명시',
         "받아들임. 'T=4 를 끝까지 도는 지금 망에서' 를 붙이고 경계 (a)~(d) 를 적었다. 일반 주장 금지를 명시", '한 줄, 4절 (b)'],
        ['7', '스텝 몫 차이 — 해석 보류 (스파이크 수준 효과)',
         f"받아들임. Holm p {', '.join(f'{p:.3f}' for p in hp)} 를 표에 넣었다. npz {N32['n_runs']}런 상관 (스텝 1 r={cor[1]:+.2f}, 스텝 4 r={cor[4]:+.2f}) 과 같은 스파이크 대조군 (plain L2 n=2, BPSR n=1, - 1- inversion n=4) 의 몫을 표로 붙이고, 'ours 가 뒤 스텝으로 민다' 해석을 하지 않는다고 적었다",
         '2절'],
        ['8', 'SEENN 인용 오류 (Fig. 번호, AET 정의, \'비교도 못 함\', 원인 추정)',
         "받아들임. PDF (md5 2fcd9157…) 로 다시 대조: 96.46 은 Fig. 3·Fig. 4 왼쪽, AET 1.1309 (식 4) 와 empirical AET 1.1188 (식 13) 을 나눠 적고 우리 오라클은 후자와 같은 정의. "
         "에너지는 Fig. 3 막대그림 (부록 D 연산 단가) 만 있음, 레시피 차이 (τ=0.5·0 리셋·SGD 300에폭·Cutout+AutoAugment) 추가. (5) 는 '추정' 으로 낮춤. JSON `seenn` 도 고침",
         '4절 (c), JSON seenn'],
        ['9', '"학습 경로 변경이 필요하다" 가 단정적이고 길이 빠짐',
         "받아들임. '길은 두 가지' 로 바꿈: (1) TF bn_per_step R19 확장 + TET 손실 신규 구현 + 학습 (3등급), (2) 별도 env 의 SEENN PyTorch 재현 + 스파이크 계수 (프레임워크 차이 명시). TF 에 TET 손실 플래그가 없음을 flags.py 에서 확인",
         '8절, [LIMITATION]'],
    ]


# --------------------------------------------------------------------------------------------------
def make_fig(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.edgecolor'] = '#555555'
    plt.rcParams['axes.linewidth'] = 0.8
    INK, INK2, GRID = '#222222', '#555555', '#e6e6e6'
    COL = {'baseline': '#222222', 'ours_inter_ch': '#009E73'}      # an_261019_fig.py 와 같은 범주 색
    MK = {'baseline': 'o', 'ours_inter_ch': 'D'}
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6), gridspec_kw={'width_ratios': [1.05, 1]})
    # ---- 왼쪽: T 를 줄여 다시 돌린 값 vs T=4 런 안의 스텝별 판독 (같은 체크포인트·같은 200장)
    cpu = res['cpu']['runs']
    rc = res['recal']['aug']['runs']
    for nm, tag, arm in CPU_RUNS:
        if nm not in cpu or not cpu[nm]['trunc_200']:
            continue
        t = cpu[nm]['trunc_200']
        if nm in rc:                                           # BN 재추정 (1000장): 점선 + 삼각형
            TT = rc[nm]['T']
            ks = [k for k in (1, 2, 3, 4) if k in TT]
            ax1.plot([TT[k]['spk'] for k in ks], [TT[k]['acc'] for k in ks], ls=':', lw=1.6, color=COL[arm], zorder=4)
            ax1.scatter([TT[k]['spk'] for k in ks], [TT[k]['acc'] for k in ks], s=46, marker='^', color=COL[arm],
                        edgecolors='white', linewidths=0.8, zorder=6)
        xs_r = [t[k]['spk_run'] for k in (1, 2, 3, 4)]
        ys_r = [t[k]['acc_run'] for k in (1, 2, 3, 4)]
        xs_p = [t[k]['spk_prefix_T4'] for k in (1, 2, 3, 4)]
        ys_p = [t[k]['acc_readout_T4'] for k in (1, 2, 3, 4)]
        c = COL[arm]
        ax1.plot(xs_p, ys_p, ls=(0, (4, 3)), lw=1.2, color=c, alpha=0.75, zorder=2)
        ax1.scatter(xs_p, ys_p, s=34, marker=MK[arm], facecolors='white', edgecolors=c, linewidths=1.3, zorder=3)
        ax1.plot(xs_r, ys_r, ls='-', lw=1.8, color=c, zorder=4)
        ax1.scatter(xs_r, ys_r, s=36, marker=MK[arm], color=c, edgecolors='white', linewidths=0.8, zorder=5)
    # T' 라벨 (baseline s11 기준 위치)
    t = cpu['base11']['trunc_200']
    for k, xy in ((1, (22e3, 18)), (2, (166e3, 19)), (3, (322e3, 86))):
        ax1.text(*xy, f"T'={k}", fontsize=8, color=INK2)
    for k, xy in ((1, (24e3, 41)), (2, (125e3, 101)), (3, (300e3, 101))):
        ax1.text(*xy, f'스텝 {k}', fontsize=8, color=INK2)
    ax1.annotate('T=4', (t[4]['spk_run'], t[4]['acc_run']), xytext=(t[4]['spk_run'] - 35e3, t[4]['acc_run'] - 9),
                 fontsize=8, color=INK2)
    ax1.set_xlim(0, 560e3)
    ax1.set_ylim(0, 103)
    ax1.set_xlabel('이미지당 스파이크 수')
    ax1.set_ylabel('테스트 정확도 (%)')
    ax1.set_title("재학습 없이 T 를 줄여 다시 돌린 값: BN 고정 (실선) / BN 재추정 (점선 ▲)\nvs T=4 런 안의 스텝별 판독 (파선, 몸통 4스텝을 다 돈 뒤의 읽기)",
                  fontsize=9.5, loc='left', color=INK)
    ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1e3:.0f}K'))
    ax1.grid(True, color=GRID, lw=0.6)
    ax1.text(0.98, 0.36, '체크포인트 3개: baseline s11·s12, ours_inter_ch 3e-3 s4 (CPU)\n'
             "실선 ● = BN 이동통계 T=4 그대로, T' 로 다시 돌림 (앞 200장)\n"
             "점선 ▲ = BN 이동통계만 T' 에서 학습셋 5000장 (증강) 으로 다시 잡음 (앞 1000장)\n"
             "파선 ○ = T=4 런의 스텝 t 출력·앞 t 스텝 스파이크 (앞 200장)",
             transform=ax1.transAxes, ha='right', va='center', fontsize=7.5, color=INK2, linespacing=1.4)
    # ---- 오른쪽: 실현 가능한 조기 종료 곡선 (T=4 계산 안, 10,000장, n=3 대 3)
    for tag, row in res['runs'].items():
        arm = row['arm']
        c = COL[arm]
        cur = sorted(row['curve'], key=lambda q: q['spk_real'])    # τ 를 낮출수록 왼쪽 (스파이크 적음)·아래
        xs = [q['spk_real'] for q in cur]
        ys = [q['acc'] for q in cur]
        ax2.plot(xs, ys, lw=1.2, color=c, alpha=0.85, zorder=3)
        ax2.scatter([row['npz_spk']], [row['npz_acc']], s=40, marker=MK[arm], color=c, edgecolors='white', linewidths=0.8, zorder=5)
    for arm in ('baseline', 'ours_inter_ch'):
        a = res['arms'][arm]
        ax2.errorbar([a['npz_spk']['mean']], [a['npz_acc']['mean']], xerr=[a['npz_spk']['sd']], yerr=[a['npz_acc']['sd']],
                     fmt=MK[arm], ms=9, mfc='white', mec=COL[arm], mew=1.6, ecolor=COL[arm], elinewidth=1.0, capsize=2, zorder=6)
    ax2.set_xlim(330e3, 530e3)
    ax2.set_ylim(96.0, 97.0)
    ax2.set_xlabel('이미지당 스파이크 수 (테스트 1만 장, 평가 모드)')
    ax2.set_ylabel('테스트 정확도 (%)')
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1e3:.0f}K'))
    ax2.grid(True, color=GRID, lw=0.6)
    mb = res['arms']['baseline']['real_max_saving_pct']['mean']
    ax2.set_title(f'실현 가능한 조기 종료 (몸통 4스텝 다 돈 뒤 출력만 일찍 읽음; 시간 평균 BN 망)\n아낄 수 있는 것은 마지막 두 층의 뒤 스텝뿐: 최대 {mb:.2f}% (18층 npz 스파이크)',
                  fontsize=9.5, loc='left', color=INK)
    ax2.text(0.46, 0.12, '선 = 신뢰도 임계값을 훑은 조기 종료 (런마다 하나)\n임계값을 낮추면 스파이크는 1% 안에서만 줄고\n정확도는 아래로 떨어진다 (세로축 96.0~97.0 만 표시)\n'
             '작은 점 = 런의 T=4 값\n큰 속 빈 점 = 팔 평균 ± 시드 SD (각 n=3, 서버 138)',
             transform=ax2.transAxes, ha='center', va='bottom', fontsize=7.5, color=INK2, linespacing=1.4)
    handles = [Line2D([], [], color=COL['baseline'], marker='o', lw=1.8, label='baseline'),
               Line2D([], [], color=COL['ours_inter_ch'], marker='D', lw=1.8, label='ours_inter_ch 3e-3'),
               Line2D([], [], color=INK2, lw=1.8, label="실제 계산 (왼쪽: BN 고정 T' / 오른쪽: 조기 종료)"),
               Line2D([], [], color=INK2, lw=1.6, ls=':', marker='^', label="BN 재추정 T' (왼쪽)"),
               Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 3)), marker='o', mfc='white', label='T=4 런의 스텝별 판독 (비인과)')]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('R19 CIFAR-10 T=4: 시간 평균 BN 망이라 스텝 t 출력은 \'t 에서 멈춘 출력\' 이 아니다 (26-10-20, 검토 반영 초안)', fontsize=10.5,
                 x=0.01, ha='left', color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(FIG, dpi=200, facecolor='white')
    print('[fig]', FIG)


# --------------------------------------------------------------------------------------------------
CPU_SCRIPT = r'''"""10-20 eip-stats: 인과성 점검 (CPU 전용, GPU 안 씀). extract_neuron_spikes.py 와 같은 순서로
모델을 올리되 conf.time_step 만 바꿔 앞 몇 배치만 돌린다. T=4 결과가 기존 npz 와 같은지(하네스 검증),
T=T' 결과가 T=4 의 앞 T' 스텝과 같은지(인과성) 를 본다. 저장소에는 쓰지 않는다.
쓰는 법 (저장소 루트에서, LD_LIBRARY_PATH=~/anaconda3/envs/venv_1/lib):
  python cpu_trunc.py <run_dir> <weights.h5> <out.npz> <T> <batches>
"""
import sys, os
run_dir, weights, out_npz, T_new, nb = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/kyccj/PycharmProjects/TensorFlow-SNNs')
sys.path.insert(0, os.path.abspath(run_dir))
import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)
assert not tf.config.list_physical_devices('GPU'), 'GPU 가 보인다'
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
T_orig = conf.time_step
config.set()          # 경로 이름에 ts-4 가 들어가므로 set() 은 원래 T 로 부르고
conf.time_step = T_new  # 모델을 만들기 직전에 T 만 바꾼다 (time_step 은 경로 이름과 모델 빌드에서만 쓰임)
assert not config.train
import lib_snn, datasets, callbacks  # noqa
train_ds, valid_ds, test_ds, train_n, valid_n, test_n, num_class, steps = datasets.datasets.load()
model = lib_snn.model_builder.model_builder(num_class, steps, valid_ds)
model.load_weights(weights)
print(f'[로드] {weights}  T_orig={T_orig} T={conf.time_step}', flush=True)
acts = [(l.name, l.act) for l in model.layers if isinstance(getattr(l, 'act', None), lib_snn.neurons.Neuron) and hasattr(l.act, 'spike_count')]
out_acts = [(n, a) for n, a in acts if getattr(a, 'loc', None) == 'OUT']
out_name, out_act = out_acts[-1]
T = conf.time_step
LT, LAB, SPK = [], [], {}
for bi, (x, y) in enumerate(test_ds):
    if bi >= nb:
        break
    for _, a in acts:
        a.reset_spike_count()
    out = model(x, training=False)
    t = tf.argmax(y, axis=-1).numpy() if len(y.shape) > 1 else y.numpy().reshape(-1)
    for name, a in acts:
        if a is out_act:
            continue
        st = a.out.stack().numpy().astype(np.float32)          # [T, B, ...]
        ps = st.reshape(T, st.shape[1], -1).sum(axis=2).T        # [B, T] 표본별 스텝별 발화 수
        SPK.setdefault(name, []).append(ps)
    LT.append(np.moveaxis(out_act.out.stack().numpy(), 0, 1).astype(np.float32))
    LAB.append(t)
    print(f'batch {bi} done', flush=True)
save = {'logits_t': np.concatenate(LT), 'labels': np.concatenate(LAB), 'T': np.array(T)}
for k, v in SPK.items():
    save['spk_' + k] = np.concatenate(v)
np.savez_compressed(out_npz, **save)
print('[저장]', out_npz, save['logits_t'].shape, flush=True)
'''

RECAL_SCRIPT = r'''"""10-20 eip-stats (검토 반영 2·3번): BN 이동통계를 T' 에서 다시 잡은 뒤 정확도를 잰다. CPU 전용, 학습 없음.
가중치는 그대로 두고 BN 의 moving_mean / moving_variance 만 학습셋 순전파로 다시 추정한다 (precise BN).
- 추정 순전파: 학습셋 앞 n_recal*100 장. mode='clean' 은 테스트와 같은 전처리 (증강 없음), mode='aug' 는 학습 마지막 구간과 같은 증강
  (random crop+flip → RandAugment → Random Erasing → 정규화; CutMix 는 mix_off_iter=100,000 반복 뒤 꺼지므로 넣지 않음). 각 BN 은 그 배치의 통계로 정규화한다.
  배치 평균 m_b, 배치 분산 v_b 를 모아 모집단 평균 = mean(m_b), 분산 = mean(v_b) + var(m_b) 로 둔다 (배치 크기 같음).
- 정규화 식은 원래 코드 경로 그대로다: 추정 중에는 moving 변수에 배치 통계를 넣고 원래 call (training=False) 을 부른다.
- 저장소·체크포인트에는 아무것도 쓰지 않는다 (가중치 파일을 다시 저장하지 않음).
쓰는 법 (저장소 루트에서, LD_LIBRARY_PATH=~/anaconda3/envs/venv_1/lib, PYTHONDONTWRITEBYTECODE=1):
  python cpu_recal.py <run_dir> <weights.h5> <out.npz> <T> <n_recal_batches> <n_eval_batches> [threads] [clean|aug]
"""
import sys, os
run_dir, weights, out_npz = sys.argv[1], sys.argv[2], sys.argv[3]
T_new, n_rec, n_ev = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
nth = int(sys.argv[7]) if len(sys.argv) > 7 else 6
mode = sys.argv[8] if len(sys.argv) > 8 else 'clean'
assert mode in ('clean', 'aug')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/kyccj/PycharmProjects/TensorFlow-SNNs')
sys.path.insert(0, os.path.abspath(run_dir))
import numpy as np
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(nth)
tf.config.threading.set_inter_op_parallelism_threads(2)
assert not tf.config.list_physical_devices('GPU'), 'GPU 가 보인다'
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
T_orig = conf.time_step
config.set()
conf.time_step = T_new
assert not config.train
import lib_snn, datasets, callbacks  # noqa
import tensorflow_datasets as tfds
import layers_new.batch_normalization as LBN
from datasets.augmentation_cifar import resize_with_crop, resize_with_crop_aug
import datasets.augmentation_cifar as AC
from datasets.datasets import preprocessor_input_cifar10 as PP
train_ds, valid_ds, test_ds, train_n, valid_n, test_n, num_class, steps = datasets.datasets.load()
model = lib_snn.model_builder.model_builder(num_class, steps, valid_ds)
model.load_weights(weights)
print(f'[로드] {weights}  T_orig={T_orig} T={conf.time_step}', flush=True)
T = conf.time_step

# ---- BN 층과 원래 이동통계
bns = [l for l in model.layers if isinstance(l, LBN.BatchNormalization)]
for l in bns:
    assert not l.renorm and l.virtual_batch_size is None and l.adjustment is None and not l.f_skip_bn, l.name
    assert list(l.axis) in ([3], [-1]), (l.name, l.axis)
orig = {l.name: (l.moving_mean.numpy().astype(np.float64), l.moving_variance.numpy().astype(np.float64)) for l in bns}
print(f'[BN] {len(bns)} 층', flush=True)

MODE = {'batch': False}
ACC = {}
_orig_call = LBN.BatchNormalization.call


def _patched(self, inputs, training=None):
    if MODE['batch']:
        x = tf.cast(inputs, tf.float32)
        ax = list(range(len(x.shape) - 1))
        m = tf.reduce_mean(x, ax)
        v = tf.reduce_mean(tf.square(x - m), ax)
        a = ACC.setdefault(self.name, {'m': [], 'v': []})
        a['m'].append(m.numpy().astype(np.float64))
        a['v'].append(v.numpy().astype(np.float64))
        self.moving_mean.assign(m)
        self.moving_variance.assign(v)
    return _orig_call(self, inputs, training)


LBN.BatchNormalization.call = _patched

# ---- 추정용 학습셋 (테스트와 같은 전처리). 같은 함수로 테스트 앞 100장을 만들어 test_ds 첫 배치와 대조한다.
def pipe(split):
    d = tfds.load('cifar10', split=split, as_supervised=True, shuffle_files=False)
    d = d.map(lambda i, l: resize_with_crop(i, l, 'cifar10', 32, 36 / 32, num_class, 'torch', PP)) \
         .map(lambda i, l: (PP(i), l))
    return d.batch(100, drop_remainder=True)


def pipe_aug(split):
    # image_cls.py 학습 분기 (data_aug_mix='cutmix') 와 같은 순서. CutMix 는 학습 후반 (반복 > mix_off_iter) 에 꺼져 있었다.
    assert conf.data_aug_mix == 'cutmix' and conf.mix_off_iter == 500 * 200, (conf.data_aug_mix, conf.mix_off_iter)
    import keras_cv_local as keras_cv
    d = tfds.load('cifar10', split=split, as_supervised=True, shuffle_files=False)
    d = d.map(lambda i, l: resize_with_crop_aug(i, l, 'cifar10', 32, 36 / 32, num_class, 'torch', PP))
    if conf.randaug_en:
        ra = keras_cv.preprocessing.RandAugment(value_range=(0, 255), magnitude=conf.randaug_mag, magnitude_stddev=conf.randaug_mag_std,
                                                augmentations_per_image=conf.randaug_n, rate=conf.randaug_rate)
        d = d.map(lambda i, l: (ra(tf.cast(i, tf.uint8)), l))
    if conf.rand_erase_en:
        re_ = AC.RandomErasing()
        d = d.map(lambda i, l: (re_._erase(i), l))
    d = d.map(lambda i, l: (PP(i), l))
    return d.batch(100, drop_remainder=True)


tf.random.set_seed(20261020)
x_ref, y_ref = next(iter(test_ds))
x_my, y_my = next(iter(pipe('test[:100]')))
pipe_diff = float(np.abs(x_ref.numpy() - x_my.numpy()).max())
lab_same = bool((np.argmax(y_ref.numpy(), -1) == np.argmax(y_my.numpy(), -1)).all())
print(f'[전처리 대조] test 앞 100장 최대 차 {pipe_diff}  라벨 같음 {lab_same}', flush=True)
assert pipe_diff == 0.0 and lab_same

# ---- 추정
MODE['batch'] = True
half = {}
for bi, (x, y) in enumerate((pipe if mode == 'clean' else pipe_aug)(f'train[:{n_rec * 100}]')):
    model(x, training=False)
    if bi + 1 == n_rec // 2:
        half = {k: (np.mean(a['m'], 0), np.mean(a['v'], 0) + np.var(a['m'], 0)) for k, a in ACC.items()}
    print(f'recal batch {bi} done', flush=True)
MODE['batch'] = False
assert all(len(a['m']) == n_rec for a in ACC.values()) and len(ACC) == len(bns), (len(ACC), len(bns))
diag = {}
for l in bns:
    a = ACC[l.name]
    mu, var = np.mean(a['m'], 0), np.mean(a['v'], 0) + np.var(a['m'], 0)
    l.moving_mean.assign(mu.astype(np.float32))
    l.moving_variance.assign(var.astype(np.float32))
    om, ov = orig[l.name]
    hm, hv = half[l.name]
    diag[l.name] = [float(np.median(np.abs(mu - om) / np.sqrt(ov + l.epsilon))),      # 평균 이동 / 원래 표준편차 (채널 중앙값)
                    float(np.median(var / (ov + 1e-12))),                              # 분산 비 (채널 중앙값)
                    float(np.median(np.abs(mu - hm) / np.sqrt(var + l.epsilon))),      # 절반 표본 대비 평균 차 (수렴 점검)
                    float(np.median(np.abs(var / (hv + 1e-12) - 1)))]                  # 절반 표본 대비 분산 상대차
print('[BN 갱신 완료]', flush=True)

# ---- 평가 (테스트 앞 n_ev 배치, cpu_trunc 와 같은 기록)
acts = [(l.name, l.act) for l in model.layers if isinstance(getattr(l, 'act', None), lib_snn.neurons.Neuron) and hasattr(l.act, 'spike_count')]
out_name, out_act = [(n, a) for n, a in acts if getattr(a, 'loc', None) == 'OUT'][-1]
LT, LAB, SPK = [], [], {}
for bi, (x, y) in enumerate(test_ds):
    if bi >= n_ev:
        break
    for _, a in acts:
        a.reset_spike_count()
    model(x, training=False)
    t = tf.argmax(y, axis=-1).numpy() if len(y.shape) > 1 else y.numpy().reshape(-1)
    for name, a in acts:
        if a is out_act:
            continue
        st = a.out.stack().numpy().astype(np.float32)
        SPK.setdefault(name, []).append(st.reshape(T, st.shape[1], -1).sum(axis=2).T)
    LT.append(np.moveaxis(out_act.out.stack().numpy(), 0, 1).astype(np.float32))
    LAB.append(t)
    print(f'eval batch {bi} done', flush=True)
save = {'logits_t': np.concatenate(LT), 'labels': np.concatenate(LAB), 'T': np.array(T), 'n_recal_img': np.array(n_rec * 100), 'mode': np.array(mode),
        'pipe_maxdiff': np.array(pipe_diff), 'bn_names': np.array(sorted(diag)),
        'bn_diag': np.array([diag[k] for k in sorted(diag)])}
for k, v in SPK.items():
    save['spk_' + k] = np.concatenate(v)
np.savez_compressed(out_npz, **save)
acc = 100 * float((save['logits_t'][:, -1].argmax(1) == save['labels']).mean())
print('[저장]', out_npz, save['logits_t'].shape, f'acc={acc:.2f}', flush=True)
'''

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--write-cpu-script', default=None)
    ap.add_argument('--write-recal-script', default=None)
    a = ap.parse_args()
    if a.write_cpu_script:
        open(a.write_cpu_script, 'w').write(CPU_SCRIPT)
        print('[cpu script]', a.write_cpu_script)
        sys.exit(0)
    if a.write_recal_script:
        open(a.write_recal_script, 'w').write(RECAL_SCRIPT)
        print('[recal script]', a.write_recal_script)
        sys.exit(0)
    main()
