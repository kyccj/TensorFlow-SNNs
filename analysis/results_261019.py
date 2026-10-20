"""EIP 현재 결과 요약 (2026-10-19) -- 노션 갱신의 단일 출처.

입력 (모두 읽기만 한다):
  - 원장 고정 사본 (10-19 16:38): <SNAP>/inventory.json (R19 C10/C100, VGGSNN DVS), qk.json, sdt.json
  - QKFormer 재측정: ~/runs/qkformer_wta_rev/spikes_*.json (정확도·스파이크), 각 런 summary.csv (train/val loss)
  - SDTv3 에폭별 log.txt 사본: <SCR>/sdt_logs/*.log.txt (sejong /srv2/kyccj/EIP/sdtv3_full/<run>/log.txt 를
    10-19 읽기 전용으로 복사. 원장 사본의 에폭 수까지만 쓴다)
  - Speck2f 칩 측정값: docs/decisions.md 10-15 00:0x / 10-15 05:42 / 10-16 12:1x 항 (아래 SPECK 에 옮겨 적음)
출력: analysis/results_261019.json (모든 수치), analysis/results_261019.md (보고서)

정의
  - 정확도 = 런 전체 best val acc, spikes = 그 에폭의 s_count (원장 정의). QK 는 model_best 체크포인트 fp32 재측정값.
    SDTv3 는 마지막 에폭 값 (best 는 파인튜닝 1에폭 근처라 쓰지 않음).
  - 평균 행: 완주·비배제 런만. SD 는 표본 SD (ddof=1).
  - Δacc = 조건 평균 - baseline 평균 (%p). Δspikes = 조건 평균 스파이크 / baseline 평균 - 1 (%).
  - SE(Δ) = sp * sqrt(1/n + 1/n_base), sp = 그 설정의 합동 시드 SD (n>=2 조건들의 조건 내 분산을 자유도로 가중 평균).
    t = Δ/SE, 양측 p (자유도 = 합동 자유도). 최소 검출 차 MDD = (t_.975 + t_.80) * SE (80% 검정력).
  - 무손실 (과제 정의) = 조건 평균 정확도 >= baseline 시드 최저값. n>=3 조건만 판정 (n<=2 는 판단 불가).
    '최대' = 통과한 조건 중 스파이크 감소가 가장 큰 것, '연속' = 그보다 덜 깎은 n>=3 조건이 모두 통과한 최대.
  - 같은 스파이크 비교 (ANCOVA): acc = a + b ln(spikes) + c_arm + d [server=23], c_arm 이 기준 팔 대비 오프셋.
"""
import csv, glob, json, math, os, re, time
from collections import OrderedDict, defaultdict

import numpy as np
from scipy import stats

ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
SCR = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad/an1019'
SNAP = os.path.join(SCR, 'snapshot')
QKR = os.path.expanduser('~/runs/qkformer_wta_rev')
QKBASE = os.path.expanduser('~/runs/qkformer_baseline/c10_baseline')
OUT_JSON = os.path.join(ROOT, 'analysis/results_261019.json')
OUT_MD = os.path.join(ROOT, 'analysis/results_261019.md')

INV = json.load(open(os.path.join(SNAP, 'inventory.json')))
RUNS = INV['runs']
QKJ = json.load(open(os.path.join(SNAP, 'qk.json')))
SDTJ = json.load(open(os.path.join(SNAP, 'sdt.json')))

ABL_NAME = {'ch_novmem': '- vmem', 'ch_nofinal': '- final_step', 'ch_noinv': '- 1- inversion', 'ch_nolr': '- loss-ratio'}
ARM_NAME = {'base': 'baseline', 'ours_ch': 'ours_inter_ch', 'l2': 'plain L2', 'bpsr': 'BPSR', **ABL_NAME}

# 원장 사유를 결정 기록으로 보강 (원장은 '미완주(Nep)' 만 적는다)
EXTRA_REASON = {
    'r19c10-l2-5e-8-s4': '10-19 사용자 중단 (결과 아님)',
    'r19c10-l2-1p5e-7-s4': '10-19 사용자 중단 (결과 아님)',
    'r19c10-ch_nofinal-1p4e-3-s1': '10-14 final_step 확정 후 사용자 승인 중단',
    'r19c10-ch_nofinal-1p4e-3-s2': '10-14 final_step 확정 후 사용자 승인 중단',
    'r19c10-ch_nofinal-1e-3-s1': '10-14 final_step 확정 후 사용자 승인 중단',
    'r19c10-ch_nofinal-1p6e-3-s1': '10-14 final_step 확정 후 사용자 승인 중단',
    'r19c10-ch_nofinal-1p8e-3-s1': '10-14 final_step 확정 후 사용자 승인 중단',
    'r19c10-ch_nofinal-7e-4-s1': '10-14 final_step 확정 후 사용자 승인 중단',
    'r19c10-ch_nofinal-2p8e-3-s1': '10-14 final_step 확정 후 사용자 승인 중단',
}


def kval(k):
    try:
        return float(k)
    except (TypeError, ValueError):
        return -1.0


def r_reason(r):
    """평균에 넣지 않는 사유. None 이면 포함."""
    if r['seed'] >= 900:
        return '연기시험 (2에폭)'
    st = r.get('status')
    if st == 'done':
        return None
    if st == 'excluded':
        return '원장 배제: ' + (r.get('exclude_reason') or '')
    if st == 'running':
        return f"진행 중 ({r.get('epochs')}에폭)"
    if st in ('hold', 'todo'):
        return f'큐 {st} (미시작)'
    base = r.get('exclude_reason') or st
    ex = EXTRA_REASON.get(r['tag'])
    return base + (f' -- {ex}' if ex else (' -- 큐 fail' if r.get('queue') == 'fail' else ''))


# ---------------------------------------------------------------- 통계 도구
def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=1)) if len(xs) > 1 else None
    return m, s


def pooled_sd(groups):
    """groups: [list of acc] -> (sp, df)."""
    ss, df = 0.0, 0
    for g in groups:
        if len(g) >= 2:
            ss += float(np.sum((np.array(g) - np.mean(g)) ** 2))
            df += len(g) - 1
    return (math.sqrt(ss / df) if df else None), df


def delta_stats(n, m, nb, mb, sp, dfp):
    d = m - mb
    se = sp * math.sqrt(1.0 / n + 1.0 / nb)
    t = d / se
    p = 2 * stats.t.sf(abs(t), dfp)
    mdd = (stats.t.ppf(0.975, dfp) + stats.t.ppf(0.80, dfp)) * se
    return dict(d=d, se=se, t=t, p=p, mdd=mdd)


def ancova(rows, ref, band, use_server=True):
    """rows: dict(arm, acc, spikes, server). 반환: 팔별 오프셋 (ref 대비)."""
    rr = [r for r in rows if band[0] <= r['spikes'] <= band[1]]
    arms = [ref] + sorted({r['arm'] for r in rr} - {ref})
    arms = [a for a in arms if any(r['arm'] == a for r in rr)]
    if ref not in arms:
        return None
    X, y = [], []
    has23 = len({r['server'] for r in rr}) > 1 and use_server
    for r in rr:
        row = [1.0, math.log(r['spikes'])] + [1.0 if r['arm'] == a else 0.0 for a in arms[1:]]
        if has23:
            row.append(1.0 if r['server'] == '23' else 0.0)
        X.append(row)
        y.append(r['acc'])
    X, y = np.array(X), np.array(y)
    N, P = X.shape
    if N - P < 3:
        return None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float(res @ res) / (N - P)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tq = stats.t.ppf(0.975, N - P)
    out = dict(band=list(band), N=N, df=N - P, resid_sd=math.sqrt(s2), server_term=has23,
               n_by_arm={a: sum(1 for r in rr if r['arm'] == a) for a in arms},
               slope=dict(b=float(beta[1]), se=float(se[1]), p=float(2 * stats.t.sf(abs(beta[1] / se[1]), N - P))),
               offsets={})
    for i, a in enumerate(arms[1:]):
        j = 2 + i
        c, s = float(beta[j]), float(se[j])
        out['offsets'][a] = dict(c=c, se=s, lo=c - tq * s, hi=c + tq * s,
                                 p=float(2 * stats.t.sf(abs(c / s), N - P)))
        # 같은 정확도에서의 스파이크 비 (비교 팔 / 기준 팔) = exp(-c/b), 델타법 95% 구간 (검토 반영 10-19)
        g = np.zeros(P)
        g[1], g[j] = c / beta[1] ** 2, -1.0 / beta[1]
        lr, lse = -c / float(beta[1]), math.sqrt(float(g @ cov @ g))
        out['offsets'][a].update(iso_ratio=math.exp(lr), iso_lo=math.exp(lr - tq * lse), iso_hi=math.exp(lr + tq * lse))
    if has23:
        j = X.shape[1] - 1
        out['server23'] = dict(d=float(beta[j]), se=float(se[j]))
    return out


def in_range(mean, bmin):
    """평균 >= baseline 시드 최저값. 정확도는 소수 둘째 자리 값이라 부동소수 오차를 반올림으로 없앤다
    (예: 96.60, 96.59, 96.58 의 평균이 96.589999... 로 계산되는 것)."""
    return round(mean, 6) >= round(bmin, 6)


def holm(ps):
    """Holm 보정 p (입력 순서 유지)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def tost_p(d, se, df, margin=0.2):
    """동등성 검정 (TOST, 마진 ±margin %p): 두 단측 p 중 큰 값. < 0.05 이면 '차이가 ±margin 안' 이 입증된다."""
    return float(max(stats.t.sf((d + margin) / se, df), stats.t.sf((margin - d) / se, df)))


def _pt(c):
    return None if c is None else dict(knob=c['knob'], n=c['n'], dspk=c['dspk'], acc=c['val_acc'],
                                       dacc=c['dacc_d'], se=c['dacc_se'], p=c['dacc_p'], mdd=c['dacc_mdd'],
                                       servers=c.get('servers'), tost=c.get('tost_p'))


def lossless_of(cs):
    """한 팔의 조건들 -> 무손실 판정 두 가지.
    (1) 시드 범위 기준 (과제 정의): 평균 >= baseline 최저 시드, n>=3 만.
    (2) 검정 기준 (참고): Δ<0 이고 p<0.05 인 n>=3 조건이 처음 나오기 전까지. '차이가 검출되지 않음' 이지 '같다' 가 아니다."""
    cs = sorted(cs, key=lambda c: -c['dspk'])           # 덜 깎은 것부터
    j = [c for c in cs if c['judgeable']]
    ok = [c for c in j if c['in_seed_range']]
    best = min(ok, key=lambda c: c['dspk']) if ok else None
    cont = None
    for c in j:
        if not c['in_seed_range']:
            break
        cont = c
    first_fail = next((c for c in j if not c['in_seed_range']), None)
    sig = next((c for c in j if c['dacc_d'] < 0 and c['dacc_p'] < 0.05), None)
    tcont = None
    for c in j:
        if c is sig:
            break
        tcont = c
    # 보정 없는 순차 검정이라, 첫 검출 뒤 더 줄인 조건이 검출되지 않으면 첫 검출은 위양성일 수 있다 (검토 반영)
    after = [c for c in j if sig is not None and c['dspk'] < sig['dspk'] and c['dacc_p'] >= 0.05]
    teq = [c for c in j if c.get('tost_p') is not None and c['tost_p'] < 0.05]
    tbest = min(teq, key=lambda c: c['dspk']) if teq else None
    return dict(max=_pt(best), contiguous=_pt(cont), first_fail=_pt(first_fail),
                test_first_sig=_pt(sig), test_contiguous=_pt(tcont),
                test_after_sig_nonsig=[_pt(c) for c in after], tost_max=_pt(tbest),
                n12=[dict(knob=c['knob'], n=c['n'], dspk=c['dspk'], acc=c['val_acc'], in_range=c['in_seed_range'])
                     for c in cs if not c['judgeable']])


# ---------------------------------------------------------------- 공통 표 생성 (원장 기반 설정)
def build_setting(name, runs, base_key, arm_keys, knob_of=lambda r: r['knob']):
    """runs: 원장 행들 (한 설정). 반환: dict(conds=[...], base=..., excluded=[...], running=[...], hold=[...])."""
    inc, exc, running, hold = [], [], [], []
    for r in runs:
        why = r_reason(r)
        rec = dict(tag=r['tag'], arm=r['_arm'], knob=knob_of(r), seed=r['seed'], server=r['server'],
                   epochs=r.get('epochs'), train_loss=r.get('train_loss'), train_acc=r.get('train_acc'),
                   val_loss=r.get('val_loss'), val_acc=r.get('val_acc'), spikes=r.get('spikes'),
                   s30=r.get('s30'), finished=r.get('finished', ''), reason=why)
        if why is None:
            inc.append(rec)
        elif r.get('status') == 'running':
            running.append(rec)
        elif r.get('status') in ('hold', 'todo'):
            hold.append(rec)
        else:
            exc.append(rec)
    groups = OrderedDict()
    for rec in sorted(inc, key=lambda x: (arm_keys.index(x['arm']) if x['arm'] in arm_keys else 99, kval(x['knob']), x['seed'])):
        groups.setdefault((rec['arm'], rec['knob']), []).append(rec)
    sp, dfp = pooled_sd([[x['val_acc'] for x in g] for g in groups.values()])
    base = groups.get((base_key, '-'))
    bm = mean_sd([x['val_acc'] for x in base])
    bs = mean_sd([x['spikes'] for x in base])
    bmin, bmax = min(x['val_acc'] for x in base), max(x['val_acc'] for x in base)
    conds = []
    for (arm, knob), g in groups.items():
        c = dict(arm=arm, knob=knob, n=len(g), runs=g)
        for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes', 's30'):
            c[k], c[k + '_sd'] = mean_sd([x[k] for x in g])
        c['servers'] = sorted({x['server'] for x in g})
        if arm != base_key:
            c['dspk'] = c['spikes'] / bs[0] - 1
            c.update({'dacc_' + k: v for k, v in delta_stats(c['n'], c['val_acc'], len(base), bm[0], sp, dfp).items()})
            c['in_seed_range'] = in_range(c['val_acc'], bmin)
            c['judgeable'] = c['n'] >= 3
            c['tost_p'] = tost_p(c['dacc_d'], c['dacc_se'], dfp)
            c['servers_by_seed'] = {x['seed']: x['server'] for x in g}
        conds.append(c)
    lossless = {arm: lossless_of([c for c in conds if c['arm'] == arm]) for arm in arm_keys if arm != base_key}
    return dict(name=name, pooled_sd=sp, pooled_df=dfp,
                base=dict(n=len(base), acc=bm[0], acc_sd=bm[1], spikes=bs[0], acc_min=bmin, acc_max=bmax),
                conds=conds, lossless=lossless, excluded=exc, running=running, hold=hold)


def r19(ds):
    out = []
    for r in RUNS:
        if r['ds'] != ds or not r['tag'].startswith('r19'):
            continue
        if r['key'] not in ARM_NAME:
            continue
        r = dict(r)
        r['_arm'] = ARM_NAME[r['key']]
        out.append(r)
    return out


ARMS_C10 = ['baseline', 'ours_inter_ch', 'plain L2', 'BPSR', '- vmem', '- final_step', '- 1- inversion', '- loss-ratio']
ARMS_C100 = ['baseline', 'ours_inter_ch', 'plain L2', 'BPSR']
C10 = build_setting('R19 CIFAR-10', r19('C10'), 'baseline', ARMS_C10)
C100 = build_setting('R19 CIFAR-100', r19('C100'), 'baseline', ARMS_C100)


def dvs_f0():
    out = []
    for r in RUNS:
        if r['ds'] != 'DVS':
            continue
        if not (r['tag'].startswith('vggdvsf0-') or r['tag'] == 'vggdvs-base-f0-s1'):
            continue
        r = dict(r)
        r['_arm'] = {'base': 'baseline', 'ours_ch': 'ours_inter_ch'}[r['key']]
        out.append(r)
    return out


DVS = build_setting('VGGSNN CIFAR10-DVS 첫 프레임 (vggdvsf0)', dvs_f0(), 'baseline', ['baseline', 'ours_inter_ch'])

# 옛 DVS 레시피 (짧은 참고)
DVS_OLD = []
for r in RUNS:
    if r['ds'] != 'DVS' or r['tag'].startswith('vggdvsf0-') or r['tag'] == 'vggdvs-base-f0-s1':
        continue
    DVS_OLD.append(dict(tag=r['tag'], arm=r['arm'], knob=r['knob'], seed=r['seed'], server=r['server'],
                        status=r.get('status'), epochs=r.get('epochs'), val_acc=r.get('val_acc'), spikes=r.get('spikes'),
                        train_loss=r.get('train_loss'), train_acc=r.get('train_acc'), val_loss=r.get('val_loss'),
                        reason=r_reason(r)))

# ---------------------------------------------------------------- 같은 스파이크 비교 (ANCOVA)
def anc_rows(S, arms):
    rows = []
    for c in S['conds']:
        if c['arm'] in arms:
            for x in c['runs']:
                rows.append(dict(arm=c['arm'], acc=x['val_acc'], spikes=x['spikes'], server=x['server'],
                                 knob=c['knob'], seed=x['seed'], tag=x.get('tag')))
    return rows


ANC = {
    'C10_methods': {b: ancova(anc_rows(C10, ['ours_inter_ch', 'plain L2', 'BPSR']), 'ours_inter_ch', b)
                    for b in [(200e3, 420e3), (280e3, 420e3), (150e3, 420e3)]},
    'C10_ablation': {b: ancova(anc_rows(C10, ['ours_inter_ch'] + list(ABL_NAME.values())), 'ours_inter_ch', b)
                     for b in [(200e3, 420e3), (280e3, 420e3)]},
    'C100_methods': {b: ancova(anc_rows(C100, ['ours_inter_ch', 'plain L2', 'BPSR']), 'ours_inter_ch', b)
                     for b in [(220e3, 420e3), (280e3, 420e3)]},
}
# 서버 항 없이도 (민감도)
ANC['C10_methods_noserver'] = ancova(anc_rows(C10, ['ours_inter_ch', 'plain L2', 'BPSR']), 'ours_inter_ch', (200e3, 420e3), use_server=False)
ANC['C10_ablation_noserver'] = ancova(anc_rows(C10, ['ours_inter_ch'] + list(ABL_NAME.values())), 'ours_inter_ch', (200e3, 420e3), use_server=False)


# 342K 구간 직접 비교 (각 n=4): inter 3e-3 vs 각 ablation 팔 / 표1 (plain L2 1e-7, BPSR 1e-10)
def cond(S, arm, knob):
    return next((c for c in S['conds'] if c['arm'] == arm and c['knob'] == knob), None)


def pair(S, a, b):
    sp, dfp = S['pooled_sd'], S['pooled_df']
    d = delta_stats(b['n'], b['val_acc'], a['n'], a['val_acc'], sp, dfp)
    return dict(ref=f"{a['arm']} {a['knob']}", arm=f"{b['arm']} {b['knob']}", n_ref=a['n'], n=b['n'],
                acc_ref=a['val_acc'], acc=b['val_acc'], spk_ref=a['spikes'], spk=b['spikes'],
                dacc=d['d'], se=d['se'], p=d['p'], mdd=d['mdd'], dspk=b['spikes'] / a['spikes'] - 1)


REF = cond(C10, 'ours_inter_ch', '3e-3')
ABL342 = [pair(C10, REF, cond(C10, a, k)) for a, k in
          [('- vmem', '2e-3'), ('- final_step', '1.4e-3'), ('- 1- inversion', '4.5e-3'), ('- loss-ratio', '1.25e-7')]]
TABLE1 = [pair(C10, REF, cond(C10, a, k)) for a, k in [('plain L2', '1e-7'), ('BPSR', '1e-10')]]
for x, ph in zip(ABL342, holm([x['p'] for x in ABL342])):
    x['p_holm'] = ph
for A in [a for k in ('C10_ablation', 'C10_methods', 'C100_methods') for a in ANC[k].values()]:
    ks = list(A['offsets'])
    for k, ph in zip(ks, holm([A['offsets'][k]['p'] for k in ks])):
        A['offsets'][k]['p_holm'] = ph


def curve_offsets(S, ref_arm, arms, band):
    """기준 팔만으로 acc = a + b ln(s) 를 맞추고 (서버 항 없음), 다른 팔의 조건별로 같은 스파이크 예측값과의 평균 차.
    SE^2 = resid_sd^2/n + x̄ᵀ Cov(β) x̄ (x̄ = 그 조건 런들의 [1, ln s] 평균)."""
    ref = [x for c in S['conds'] if c['arm'] == ref_arm for x in c['runs'] if band[0] <= x['spikes'] <= band[1]]
    X = np.array([[1.0, math.log(x['spikes'])] for x in ref])
    y = np.array([x['val_acc'] for x in ref])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float(res @ res) / (len(y) - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    out = dict(band=list(band), n_ref=len(y), b=float(beta[1]), resid_sd=math.sqrt(s2), conds=[])
    for c in S['conds']:
        if c['arm'] not in arms:
            continue
        xs = np.array([[1.0, math.log(x['spikes'])] for x in c['runs']])
        d = float(np.mean([x['val_acc'] for x in c['runs']]) - np.mean(xs @ beta))
        xb = xs.mean(axis=0)
        se = math.sqrt(s2 / len(xs) + float(xb @ cov @ xb))
        out['conds'].append(dict(arm=c['arm'], knob=c['knob'], n=c['n'], spikes=c['spikes'], acc=c['val_acc'],
                                 pred=float(np.mean(xs @ beta)), d=d, se=se,
                                 p=float(2 * stats.t.sf(abs(d / se), len(y) - 2))))
    return out


CURVE_ABL = curve_offsets(C10, 'ours_inter_ch', list(ABL_NAME.values()), (200e3, 420e3))
CURVE_MET = curve_offsets(C10, 'ours_inter_ch', ['plain L2', 'BPSR'], (200e3, 420e3))


# ---------------------------------------------------------------- QKFormer (재측정)
def qk_summary(d):
    p = os.path.join(d, 'summary.csv')
    rows = list(csv.reader(open(p)))
    h = rows[0]
    by_ep = OrderedDict()
    for r in rows[1:]:
        if not r or r[0] == h[0]:
            continue
        by_ep[int(r[0])] = dict(zip(h, r))      # --resume 로 겹친 에폭은 뒤의 것
    eps = list(by_ep.values())
    best = None
    for e in eps:                              # timm CheckpointSaver: 엄격히 더 클 때만 갱신 -> 최고값의 첫 에폭
        if best is None or float(e['eval_top1']) > float(best['eval_top1']):
            best = e
    tl = best.get('train_task_loss') or best.get('train_loss')
    return dict(epochs=len(eps), best_epoch=int(best['epoch']) + 1, log_best=float(best['eval_top1']),
                train_loss=float(tl), val_loss=float(best['eval_loss']))


def qk_seed(d):
    m = re.search(r'^seed:\s*(\d+)', open(os.path.join(d, 'args.yaml')).read(), re.M)
    return int(m.group(1)) if m else None


QK_RUNS = [  # (설정, 팔, ρ, 런 폴더, 재측정 json 이름, 원장 qk.json 이름)
    ('C10', 'baseline', '-', QKBASE, 'base', 'baseline'),
    ('C10', 'baseline', '-', f'{QKR}/base_s43', 'base_s43', 'baseline s43'),
    ('C10', 'baseline', '-', None, None, 'baseline s44'),
    ('C10', 'baseline', '-', f'{QKR}/base_s45', 'base_s45', 'baseline s45'),
    ('C10', 'baseline', '-', f'{QKR}/base_s46', 'base_s46', 'baseline s46'),
    ('C10', 'baseline', '-', f'{QKR}/base_s47', 'base_s47', 'baseline s47'),
    ('C10', 'baseline', '-', f'{QKR}/base_s48', 'base_s48', 'baseline s48'),
    ('C10', 'ours_inter_ch', '1e-3', f'{QKR}/ours_ch_rho1e-3', 'ours_ch_rho1e-3', 'ours_ch_rho1e-3'),
    ('C10', 'ours_inter_ch', '2e-3', f'{QKR}/ours_ch_rho2e-3', 'ours_ch_rho2e-3', 'ours_ch_rho2e-3'),
    ('C10', 'ours_inter_ch', '3e-3', f'{QKR}/ours_ch_rho3e-3', 'ours_ch_rho3e-3', 'ours_ch_rho3e-3'),
    ('C10', 'ours_inter_ch', '5e-3', f'{QKR}/ours_ch_rho5e-3', 'ours_ch_rho5e-3', 'ours_ch_rho5e-3'),
    ('C10', 'ours_inter_ch', '7e-3', f'{QKR}/ours_ch_rho7e-3', 'ours_ch_rho7e-3', 'ours_ch_rho7e-3'),
    ('C10', 'ours_inter_ch', '1.2e-2', f'{QKR}/ours_ch_rho1.2e-2', 'ours_ch_rho1.2e-2', 'ours_ch_rho1.2e-2'),
    ('C10', 'ours_inter_ch', '1.2e-2', f'{QKR}/ours_ch_rho1.2e-2_s43', 'ours_ch_rho1.2e-2_s43', 'ours_ch_rho1.2e-2 s43'),
    ('C10', 'ours_inter_ch', '1.2e-2', f'{QKR}/ours_ch_rho1.2e-2_s44', 'ours_ch_rho1.2e-2_s44', 'ours_ch_rho1.2e-2 s44'),
    ('C10', 'ours_inter_ch', '1.2e-2', f'{QKR}/ours_ch_rho1.2e-2_s45', 'ours_ch_rho1.2e-2_s45', 'ours_ch_rho1.2e-2 s45'),
    ('C10', 'ours_inter_ch', '2e-2', f'{QKR}/ours_ch_rho2e-2', 'ours_ch_rho2e-2', 'ours_ch_rho2e-2'),
    ('C100', 'baseline', '-', f'{QKR}/c100_base_s43', 'c100_base_s43', 'c100_baseline s43'),
    ('C100', 'baseline', '-', f'{QKR}/c100_base_s44', 'c100_base_s44', 'c100_baseline s44'),
    ('C100', 'baseline', '-', f'{QKR}/c100_base_s45', 'c100_base_s45', 'c100_baseline s45'),
    ('C100', 'baseline', '-', f'{QKR}/c100_base_s46', 'c100_base_s46', 'c100_baseline s46'),
] + [('C100', 'ours_inter_ch', rho, f'{QKR}/c100_ours_ch_rho{rho}_s{s}', f'c100_ours_ch_rho{rho}_s{s}',
      f'c100_ours_ch_rho{rho}_s{s}') for rho in ('6e-3', '1.2e-2') for s in (43, 44, 45, 46)]
QK_REF = [  # 참고 팔 (재측정 있음, 이 요약의 주 비교 대상 아님)
    ('C10', 'plain L2', lam, f'{QKR}/l2_{lam}', f'l2_{lam}') for lam in ('5e-7', '1e-6', '3e-6', '5e-6', '1e-5', '2e-5')
] + [('C10', 'ours_layer_w1', '5e-3', f'{QKR}/ours_layer_w1_rho5e-3', 'ours_layer_w1_rho5e-3'),
     ('C10', 'ours_layer_w1', '1e-2', f'{QKR}/ours_layer_w1_rho1e-2', 'rho1e-2'),
     ('C10', 'ours_layer_w1', '1e-2', f'{QKR}/ours_layer_w1_rho1e-2_s43', 'ours_layer_w1_rho1e-2_s43')]
QKJOB = {j['name']: j for j in QKJ['jobs']}


def qk_rec(ds, arm, rho, d, sj, ledger=None):
    rec = dict(ds=ds, arm=arm, knob=rho, dir=d, server='138', ledger_name=ledger)
    led = QKJOB.get(ledger) if ledger else None
    if led:
        rec['server'] = led.get('server') or '138'
        rec['ledger_status'] = led['status']
        rec['log_best_ledger'] = led.get('acc')
    if d is None or sj is None or not os.path.exists(os.path.join(QKR, f'spikes_{sj}.json')):
        rec.update(seed=int(re.search(r's(\d+)$', ledger).group(1)) if ledger and re.search(r's(\d+)$', ledger) else None,
                   reason='재측정 없음 (sejong 런, 체크포인트 fp32 재측정 안 함) -- 학습 로그 최고 %s' % (led.get('acc') if led else '?'))
        return rec
    j = json.load(open(os.path.join(QKR, f'spikes_{sj}.json')))
    s = qk_summary(d)
    rec.update(seed=qk_seed(d), epochs=s['epochs'], best_epoch=s['best_epoch'], log_best=s['log_best'],
               train_loss=s['train_loss'], train_acc=None, val_loss=s['val_loss'],
               val_acc=float(j['accuracy']), spikes=float(j['totals']['spikes_per_sample']),
               remeasure_json=f'spikes_{sj}.json', reason=None)
    if s['epochs'] < 410:
        rec['reason'] = f"미완주 ({s['epochs']}에폭)"
    return rec


QK = [qk_rec(*x) for x in QK_RUNS]
QK_REFR = [qk_rec(*x) for x in QK_REF]
QK_PENDING = [dict(name=j['name'], status=j['status'], epochs=j['epochs']) for j in QKJ['jobs']
              if j['status'] in ('todo', 'running', 'stale') and ('ours_ch' in j['name'])]


def qk_setting(ds):
    rs = [r for r in QK if r['ds'] == ds]
    S = dict(runs=rs)
    inc = [r for r in rs if r.get('reason') is None]
    groups = OrderedDict()
    for r in sorted(inc, key=lambda r: (r['arm'] != 'baseline', kval(r['knob']), r['seed'])):
        groups.setdefault((r['arm'], r['knob']), []).append(r)
    sp, dfp = pooled_sd([[x['val_acc'] for x in g] for g in groups.values()])
    base = groups[('baseline', '-')]
    bm, bsd = mean_sd([x['val_acc'] for x in base])
    bs = mean_sd([x['spikes'] for x in base])[0]
    bmin = min(x['val_acc'] for x in base)
    conds = []
    for (arm, knob), g in groups.items():
        c = dict(arm=arm, knob=knob, n=len(g), runs=g)
        for k in ('train_loss', 'val_loss', 'val_acc', 'spikes', 'log_best'):
            c[k], c[k + '_sd'] = mean_sd([x[k] for x in g])
        if arm != 'baseline':
            c['dspk'] = c['spikes'] / bs - 1
            c.update({'dacc_' + k: v for k, v in delta_stats(c['n'], c['val_acc'], len(base), bm, sp, dfp).items()})
            c['in_seed_range'] = in_range(c['val_acc'], bmin)
            c['judgeable'] = c['n'] >= 3
            c['tost_p'] = tost_p(c['dacc_d'], c['dacc_se'], dfp)
        conds.append(c)
    S.update(pooled_sd=sp, pooled_df=dfp, base=dict(n=len(base), acc=bm, acc_sd=bsd, spikes=bs, acc_min=bmin,
                                                    acc_max=max(x['val_acc'] for x in base)),
             conds=conds,
             lossless={'ours_inter_ch': lossless_of([c for c in conds if c['arm'] == 'ours_inter_ch'])},
             excluded=[r for r in rs if r.get('reason')])
    return S


QK10, QK100 = qk_setting('C10'), qk_setting('C100')


# ---------------------------------------------------------------- SDTv3
def sdt_log(name, max_ep=None):
    p = os.path.join(SCR, 'sdt_logs', f'{name}.log.txt')
    rows = []
    for l in open(p):
        try:
            j = json.loads(l)
        except Exception:
            continue
        if 'test_acc1' in j:
            rows.append(j)
    return rows[:max_ep] if max_ep else rows


SDTJOB = {j['dir']: j for j in SDTJ['jobs']}
SDT_MAIN = []
CTRL = SDTJOB['main_ctrl']
for d, rho, path in [('main_ctrl', '-', 'orig a45daf6'), ('main_rho0.05', '0.05', 'fast 92581b0'), ('main_rho0.1', '0.1', 'fast 92581b0'),
                     ('main_rho0.2', '0.2', 'fast 92581b0'), ('main_rho0.3', '0.3', 'orig a45daf6'),
                     ('main_rho1.0', '1.0', 'orig a45daf6'), ('main_rho3.0', '3.0', 'orig a45daf6')]:
    j = SDTJOB[d]
    lg = sdt_log(d, j['epochs'])
    last5 = lg[-5:]
    rec = dict(dir=d, arm='baseline (대조, 규제 없음)' if rho == '-' else 'ours_inter_ch', knob=rho, server='23', status=j['status'],
               epochs=j['epochs'], total=j['total'], code_path=path,
               train_loss=j['train_loss'], train_acc=j['train_acc'], val_loss=j['val_loss'],
               val_acc=j['val_last'], spikes=j['spikes'], best_over_run=j['val_acc'],
               last5_acc=float(np.mean([r['test_acc1'] for r in last5])),
               last5_spikes=float(np.mean([r['test_spikes'] for r in last5])),
               log_last_matches=abs(lg[-1]['test_acc1'] - j['val_last']) < 1e-6)
    if rho != '-':
        rec['dacc'] = rec['val_acc'] - CTRL['val_last']
        rec['dspk'] = rec['spikes'] / CTRL['spikes'] - 1
        c5 = sdt_log('main_ctrl', CTRL['epochs'])[-5:]
        rec['dacc_last5'] = rec['last5_acc'] - float(np.mean([r['test_acc1'] for r in c5]))
        rec['dspk_last5'] = rec['last5_spikes'] / float(np.mean([r['test_spikes'] for r in c5])) - 1
    SDT_MAIN.append(rec)
_cl = sdt_log('main_ctrl', CTRL['epochs'])
SDT_CTRL_NOTE = dict(official_headonly=85.72, last=CTRL['val_last'], last20=float(np.mean([r['test_acc1'] for r in _cl[-20:]])),
                     first_epoch=_cl[0]['test_acc1'])
SDT_CTRL_NOTE['d_last'] = SDT_CTRL_NOTE['last'] - 85.72
SDT_CTRL_NOTE['d_last20'] = SDT_CTRL_NOTE['last20'] - 85.72

SDT_KD = []
kd_ctrl_log = sdt_log('main_kd_ctrl', SDTJOB['main_kd_ctrl']['epochs'])
for d in ('main_kd_ctrl', 'main_kd_rho0.3', 'main_kd_rho0.5', 'main_kd_rho1.0'):
    j = SDTJOB[d]
    lg = sdt_log(d, j['epochs'])
    rec = dict(dir=d, knob=j['knob'], status=j['status'], epochs=j['epochs'], total=j['total'], eta=j['eta'],
               train_loss=j['train_loss'], train_acc=j['train_acc'], val_loss=j['val_loss'],
               val_acc=j['val_last'], spikes=j['spikes'])
    if d != 'main_kd_ctrl':
        e = min(len(lg), len(kd_ctrl_log))       # 같은 에폭끼리 (둘 다 진행한 마지막 에폭)
        a, b = lg[e - 1], kd_ctrl_log[e - 1]
        rec['same_epoch'] = dict(epoch=e, acc=a['test_acc1'], spikes=a['test_spikes'], ctrl_acc=b['test_acc1'],
                                 ctrl_spikes=b['test_spikes'], dacc=a['test_acc1'] - b['test_acc1'],
                                 dspk=a['test_spikes'] / b['test_spikes'] - 1)
    SDT_KD.append(rec)

# ---------------------------------------------------------------- Speck2f 칩 (docs/decisions.md 10-15/10-16 에서 옮김, 153 서버, n_eval 1000)
SPECK = dict(
    source='docs/decisions.md: 2026-10-15 00:0x (표), 10-15 05:42 (baseline ep60 r1), 10-15 정정 (10-10 측정 시간 3.94 s), 10-16 12:1x 마지막 항 (체인 완료분)',
    setting='CIFAR10-DVS 9층 SNN (exp470 레시피: exp466 출발 교사 3개 앙상블 증류 60에폭 파인튜닝), Speck2f, bias_channel_pipeline n_eval 1000',
    measurements=[
        dict(model='baseline exp470 best', run='r1 (10-10)', sw_acc=64.7, hw_acc=63.0, sw_spikes=105685, mW=1.582, mJ=6.231, s_per_sample=3.94),
        dict(model='baseline exp470 best', run='r2 (10-14 재측정)', sw_acc=64.7, hw_acc=63.0, sw_spikes=105685, mW=1.714, mJ=6.173, s_per_sample=3.60),
        dict(model='baseline exp470 epoch_060', run='r1', sw_acc=64.1, hw_acc=62.5, sw_spikes=108180, mW=1.682, mJ=6.346, s_per_sample=3.77),
        dict(model='baseline exp470 epoch_060', run='r2', sw_acc=None, hw_acc=None, sw_spikes=None, mW=1.586, mJ=6.335, s_per_sample=None),
        dict(model='ours_inter_ch rho 1.0 best (에폭 42)', run='r1', sw_acc=64.1, hw_acc=62.9, sw_spikes=65075, mW=1.067, mJ=4.297, s_per_sample=4.03),
        dict(model='ours_inter_ch rho 1.0 best (에폭 42)', run='r2', sw_acc=None, hw_acc=None, sw_spikes=None, mW=1.081, mJ=4.261, s_per_sample=None),
        dict(model='ours_inter_ch rho 1.0 epoch_060', run='r1', sw_acc=None, hw_acc=62.8, chip_events=64759, sw_spikes=None, mW=1.173, mJ=4.212, s_per_sample=None),
        dict(model='ours_inter_ch rho 0.3 epoch_060', run='r1', sw_acc=None, hw_acc=63.2, chip_events=99464, sw_spikes=None, mW=1.444, mJ=5.765, s_per_sample=None),
    ],
    notes=['각 모델은 학습 1회 (n=1). r1/r2 는 같은 체크포인트의 측정 반복이지 시드 반복이 아니다.',
           'mW 는 실행마다 약 8% 흔들린다 (baseline best 1.582 vs 1.714). n=1000 칩 정확도의 SE 약 1.5%p, 이 레시피 시드 SD 약 1.1%p.',
           'mJ/표본 = 동적 전력 x 표본당 측정 시간 (벽시계). inter 쪽 측정 시간이 길어 에너지 감소가 전력 감소보다 작게 잡힌다.',
           'epoch_060 끼리 칩 이벤트 감소율 (-41.7%, -10.4%) 은 decisions 10-16 의 값. baseline epoch_060 의 칩 이벤트 수 자체는 기록에 없다.'])
_b = [m for m in SPECK['measurements'] if m['model'].startswith('baseline')]
_i = [m for m in SPECK['measurements'] if m['model'].startswith('ours_inter_ch rho 1.0 best')]
_b60 = [m for m in SPECK['measurements'] if m['model'] == 'baseline exp470 epoch_060']
SPECK['summary'] = dict(
    baseline_mW=float(np.mean([m['mW'] for m in _b])), baseline_mJ=float(np.mean([m['mJ'] for m in _b])), baseline_k=len(_b),
    inter_best_mW=float(np.mean([m['mW'] for m in _i])), inter_best_mJ=float(np.mean([m['mJ'] for m in _i])), inter_k=len(_i),
    sw_spike_change_best=65075 / 105685 - 1)
SPECK['summary']['dmW'] = SPECK['summary']['inter_best_mW'] / SPECK['summary']['baseline_mW'] - 1
SPECK['summary']['dmJ'] = SPECK['summary']['inter_best_mJ'] / SPECK['summary']['baseline_mJ'] - 1
_b60mW = float(np.mean([m['mW'] for m in _b60]))
SPECK['summary']['ep60'] = dict(baseline_mW=_b60mW, rho1_mW=1.173, rho03_mW=1.444,
                                d_rho1=1.173 / _b60mW - 1, d_rho03=1.444 / _b60mW - 1,
                                events_rho1=-0.417, events_rho03=-0.104)

# ================================================================ 검토 반영 (10-19 eip-critic) 추가 계산
# 각 항목은 md 의 "검토 반영" 절과 짝이다. 기존 수치는 바꾸지 않고 옆에 붙인다.
def ols(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    N, P = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    rss = float(res @ res)
    cov = rss / (N - P) * np.linalg.inv(X.T @ X)
    return beta, cov, rss, N - P


def coef(beta, cov, j, df):
    c, s = float(beta[j]), math.sqrt(float(cov[j, j]))
    tq = stats.t.ppf(0.975, df)
    return dict(c=c, se=s, lo=c - tq * s, hi=c + tq * s, p=float(2 * stats.t.sf(abs(c / s), df)), df=df)


def contrast(beta, cov, g, df):
    g = np.asarray(g, float)
    c, s = float(g @ beta), math.sqrt(float(g @ cov @ g))
    return dict(c=c, se=s, p=float(2 * stats.t.sf(abs(c / s), df)), df=df)


N_VAL = {'R19 CIFAR-10': 10000, 'R19 CIFAR-100': 10000, 'DVS': 992, 'QK': 10000}
# DVS 검증셋 크기: 모든 vggdvsf0 런의 정확도가 k/992 (소수 둘째 자리 반올림) 와 맞는다 -> 992장 (이미지 1장 = 0.1008%p)
_dvs_acc = [x['val_acc'] for c in DVS['conds'] for x in c['runs']]
DVS_N_CHECK = dict(n_val=992, n_runs=len(_dvs_acc),
                   all_match=all(abs(round(round(a * 992 / 100) * 100 / 992, 2) - a) < 1e-6 for a in _dvs_acc))


def judge_robust(S, c, n_val):
    """시드 범위 판정의 취약성: 기준과의 차 (이미지 수), 동점 여부, 런 하나 빼기로 판정이 뒤집히는 시드."""
    bmin = S['base']['acc_min']
    accs = [(x['seed'], x['val_acc']) for x in c['runs']]
    m = float(np.mean([a for _, a in accs]))
    margin = m - bmin
    flips = []
    for sd, _ in accs:
        rest = [a for s2, a in accs if s2 != sd]
        if rest and in_range(float(np.mean(rest)), bmin) != in_range(m, bmin):
            flips.append(sd)
    return dict(knob=c['knob'], n=c['n'], margin=margin, margin_images=margin / 100 * n_val * c['n'],
                tie=abs(round(m, 6) - round(bmin, 6)) < 5e-4, loo_flip_seeds=flips)


def sub_cond(S, c, runs):
    """조건을 런 일부로 다시 계산 (합동 SD 는 원래 값 유지)."""
    acc = [x['val_acc'] for x in runs]
    m = float(np.mean(acc))
    d = delta_stats(len(runs), m, S['base']['n'], S['base']['acc'], S['pooled_sd'], S['pooled_df'])
    spk = float(np.mean([x['spikes'] for x in runs]))
    return dict(c, n=len(runs), runs=runs, val_acc=m, spikes=spk, dspk=spk / S['base']['spikes'] - 1,
                servers=sorted({x['server'] for x in runs}),
                dacc_d=d['d'], dacc_se=d['se'], dacc_p=d['p'], dacc_mdd=d['mdd'],
                in_seed_range=in_range(m, S['base']['acc_min']), judgeable=len(runs) >= 3,
                tost_p=tost_p(d['d'], d['se'], S['pooled_df']))


def capped_lossless(S, arm, cap=4):
    """10-17 '반복 n=4 로 통일' 규칙: n>4 조건은 시드 번호가 작은 4개만 쓰고 판정을 다시 한다."""
    cs = []
    for c in S['conds']:
        if c['arm'] != arm:
            continue
        cs.append(sub_cond(S, c, sorted(c['runs'], key=lambda x: x['seed'])[:cap]) if c['n'] > cap else c)
    changed = [c['knob'] for c in S['conds'] if c['arm'] == arm and c['n'] > cap]
    out = lossless_of(cs)
    out['capped_conds'] = [dict(knob=k, **{kk: vv for kk, vv in sub_cond(S, cond(S, arm, k), sorted(cond(S, arm, k)['runs'], key=lambda x: x['seed'])[:cap]).items()
                                         if kk in ('n', 'val_acc', 'dspk', 'dacc_d', 'dacc_se', 'dacc_p', 'in_seed_range')}) for k in changed]
    return out


REV = {}
# (1) 시드 범위 판정 취약성 -- 각 설정·팔의 '최대' 점과 '연속' 점
REV['robust'] = {}
for key, S, nv, arms in (('C10', C10, 10000, ARMS_C10[1:]), ('C100', C100, 10000, ARMS_C100[1:]), ('DVS', DVS, 992, ['ours_inter_ch']),
                         ('QK10', QK10, 10000, ['ours_inter_ch']), ('QK100', QK100, 10000, ['ours_inter_ch'])):
    REV['robust'][key] = {}
    for arm in arms:
        ll = S['lossless'][arm]
        pts = {}
        for which in ('max', 'contiguous', 'first_fail'):
            if ll.get(which):
                pts[which] = judge_robust(S, cond(S, arm, ll[which]['knob']), nv)
        REV['robust'][key][arm] = pts
# C100 ours_inter_ch 3e-3, DVS 2e-3·1e-2 처럼 '처음 떨어지는' 점의 모자란 양도 같이 (위 first_fail 에 포함)

# (2) n=4 통일 규칙 (10-17) 으로 다시 판정 -- plain L2 (C10 은 2.5e-7·5e-7 이 n=5)
REV['capped4'] = {'C10': {arm: capped_lossless(C10, arm) for arm in ['ours_inter_ch', 'plain L2', 'BPSR']},
                  'C100': {arm: capped_lossless(C100, arm) for arm in ['ours_inter_ch', 'plain L2', 'BPSR']}}

# (3) 경계 쌍의 서버 구성과 ours_inter_ch 안 서버 항 (200–420K)
_ri = [r for r in anc_rows(C10, ['ours_inter_ch']) if 200e3 <= r['spikes'] <= 420e3]
_b, _cv, _, _df = ols([[1.0, math.log(r['spikes']), float(r['server'] == '23')] for r in _ri], [r['acc'] for r in _ri])
REV['inter_server_term'] = dict(n=len(_ri), n23=sum(r['server'] == '23' for r in _ri), **coef(_b, _cv, 2, _df))

# (4) 기울기 동질성 (C10 방법 비교 200–420K): 팔마다 기울기를 따로 둔 모형 대 공통 기울기 모형
def slope_homog(rows, ref, band):
    rr = [r for r in rows if band[0] <= r['spikes'] <= band[1]]
    arms = [ref] + sorted({r['arm'] for r in rr} - {ref})
    base = lambda r: [1.0, math.log(r['spikes'])] + [float(r['arm'] == a) for a in arms[1:]] + [float(r['server'] == '23')]
    y = [r['acc'] for r in rr]
    _, _, rss0, _ = ols([base(r) for r in rr], y)
    _, _, rss1, df1 = ols([base(r) + [float(r['arm'] == a) * math.log(r['spikes']) for a in arms[1:]] for r in rr], y)
    q = len(arms) - 1
    F = ((rss0 - rss1) / q) / (rss1 / df1)
    return dict(F=F, df1=q, df2=df1, p=float(stats.f.sf(F, q, df1)))


REV['slope_homog'] = {'C10_methods_200-420K': slope_homog(anc_rows(C10, ['ours_inter_ch', 'plain L2', 'BPSR']), 'ours_inter_ch', (200e3, 420e3)),
                      'C10_ablation_200-420K': slope_homog(anc_rows(C10, ['ours_inter_ch'] + list(ABL_NAME.values())), 'ours_inter_ch', (200e3, 420e3)),
                      'C100_methods_220-420K': slope_homog(anc_rows(C100, ['ours_inter_ch', 'plain L2', 'BPSR']), 'ours_inter_ch', (220e3, 420e3))}

# (5) ablation 세부 (200–420K, 서버 항 포함)
_AB = list(ABL_NAME.values())
_rab = [r for r in anc_rows(C10, ['ours_inter_ch'] + _AB) if 200e3 <= r['spikes'] <= 420e3]
_yab = [r['acc'] for r in _rab]
# (5a) - vmem × 서버: - vmem 오프셋을 서버별로
_X = [[1.0, math.log(r['spikes'])] + [float(r['arm'] == a) for a in _AB] + [float(r['server'] == '23'),
      float(r['arm'] == '- vmem' and r['server'] == '23')] for r in _rab]
_b, _cv, _, _df = ols(_X, _yab)
_iv = 2 + _AB.index('- vmem')
_g = np.zeros(len(_b)); _g[_iv] = 1.0; _g[-1] = 1.0
REV['vmem_by_server'] = dict(at138=coef(_b, _cv, _iv, _df), at23=contrast(_b, _cv, _g, _df), diff_23_minus_138=coef(_b, _cv, len(_b) - 1, _df),
                             n138=sum(r['arm'] == '- vmem' and r['server'] == '138' for r in _rab),
                             n23=sum(r['arm'] == '- vmem' and r['server'] == '23' for r in _rab))
# (5b) - vmem 런 하나 빼기 (팔 공통 ANCOVA)
_loo = []
for r0 in [r for r in _rab if r['arm'] == '- vmem']:
    A = ancova([r for r in _rab if r is not r0], 'ours_inter_ch', (200e3, 420e3))
    o = A['offsets']['- vmem']
    _loo.append(dict(knob=r0['knob'], seed=r0['seed'], server=r0['server'], acc=r0['acc'], c=o['c'], p=o['p']))
REV['vmem_loo'] = dict(runs=_loo, n_sig=sum(x['p'] < 0.05 for x in _loo), max_p=max(x['p'] for x in _loo))
# (5c) - final_step 을 ρ 별 오프셋으로
_fs = ['1.4e-3', '2.2e-3']
_oth = [a for a in _AB if a != '- final_step']
_X = [[1.0, math.log(r['spikes'])] + [float(r['arm'] == a) for a in _oth] + [float(r['arm'] == '- final_step' and r['knob'] == k) for k in _fs]
      + [float(r['server'] == '23')] for r in _rab]
_b, _cv, _, _df = ols(_X, _yab)
_i14, _i22 = 2 + len(_oth), 3 + len(_oth)
_g = np.zeros(len(_b)); _g[_i14], _g[_i22] = 1.0, -1.0
REV['final_step_by_rho'] = {'1.4e-3': coef(_b, _cv, _i14, _df), '2.2e-3': coef(_b, _cv, _i22, _df), 'diff_1.4_minus_2.2': contrast(_b, _cv, _g, _df)}
# (5d) 342K 기준점 (ours_inter_ch 3e-3) 의 시드 하나 빼기
_ref_loo = []
for sd in [x['seed'] for x in REF['runs']]:
    rr = [x for x in REF['runs'] if x['seed'] != sd]
    rc_ = dict(REF, n=len(rr), val_acc=float(np.mean([x['val_acc'] for x in rr])), spikes=float(np.mean([x['spikes'] for x in rr])))
    _ref_loo.append(dict(drop_seed=sd, ref_acc=rc_['val_acc'],
                         arms={x['arm']: dict(d=p_['dacc'], p=p_['p']) for x, p_ in
                               zip(ABL342, [pair(C10, rc_, cond(C10, *x['arm'].rsplit(' ', 1))) for x in ABL342])}))
REV['ref342_loo'] = _ref_loo
# (5e) 보정 전 p<0.05 전부 (ablation)
_unc = []
for x in ABL342:
    if x['p'] < 0.05:
        _unc.append(dict(where='342K 직접 비교', what=x['arm'], d=x['dacc'], p=x['p']))
for band, A in ANC['C10_ablation'].items():
    for k, o in A['offsets'].items():
        if o['p'] < 0.05:
            _unc.append(dict(where=f'같은 스파이크 ANCOVA {band[0] / 1e3:.0f}–{band[1] / 1e3:.0f}K', what=k, d=o['c'], p=o['p']))
for c in CURVE_ABL['conds']:
    if c['p'] < 0.05:
        _unc.append(dict(where='곡선 대비 (조건별)', what=f"{c['arm']} {c['knob']} (n={c['n']})", d=c['d'], p=c['p']))
for k, o in REV['final_step_by_rho'].items():
    if k != 'diff_1.4_minus_2.2' and o['p'] < 0.05:
        _unc.append(dict(where='ρ 별 오프셋 ANCOVA 200–420K', what=f'- final_step {k}', d=o['c'], p=o['p']))
REV['ablation_uncorrected_sig'] = _unc

# (6) 스파이크 측정 축: 원장 s_count 대 npz (테스트 1만 장 1회 추론, 평가 모드) 총합
#     npz 는 이 시각 이전에 쓰인 파일만 쓴다 (같은 날 다른 추출이 진행 중이라 고정).
NPZ_DIRS = [os.path.join(ROOT, '_npz'), os.path.join(os.path.dirname(SCR), 'deep1005', 'npz'), os.path.join(SCR, 'npz'), os.path.join(SCR, 'npz_abl')]
NPZ_CUTOFF = '2026-10-19 17:10'
_cut = time.mktime(time.strptime(NPZ_CUTOFF, '%Y-%m-%d %H:%M'))
NPZ_SKIP = {'n_in', 'predictions_n'}       # 입력·출력층은 뺀다 (inter_deep_261005 의 npz 정의와 같음)


def npz_spikes(tag):
    for d in NPZ_DIRS:
        p = os.path.join(d, tag + '.npz')
        if os.path.exists(p) and os.path.getmtime(p) <= _cut:
            try:
                z = np.load(p)
                ns = float(z['n_samples'])
                tot = sum(float(z[k].astype(np.float64).sum()) for k in z.files
                          if not k.endswith('__fired') and k + '__fired' in z.files and k not in NPZ_SKIP)
                return dict(path=p, spikes=tot / ns, acc=float(z['acc']))
            except Exception:
                return None
    return None


_nr = []
for c in C10['conds']:
    for x in c['runs']:
        m = npz_spikes(x['tag'])
        if m:
            _nr.append(dict(arm=c['arm'], knob=c['knob'], seed=x['seed'], server=x['server'], s=x['spikes'], npz=m['spikes'],
                            ratio=m['spikes'] / x['spikes'], acc_npz=m['acc'], acc=x['val_acc'], file=m['path']))
_arms_npz = sorted({r['arm'] for r in _nr} - {'ours_inter_ch'})
_y = [r['ratio'] for r in _nr]
_b0, _cv0, _rss0, _df0 = ols([[1.0, math.log(r['s'])] for r in _nr], _y)
_b1, _cv1, _rss1, _df1 = ols([[1.0, math.log(r['s'])] + [float(r['arm'] == a) for a in _arms_npz] for r in _nr], _y)
_q = len(_arms_npz)
_F = ((_rss0 - _rss1) / _q) / (_rss1 / _df1)
_bs, _cvs, _, _dfs = ols([[1.0, math.log(r['s']), float(r['server'] == '23')] for r in _nr], _y)
_diff = [r['npz'] - r['s'] for r in _nr]
_bd, _cvd, _, _dfd = ols([[1.0, r['s']] for r in _nr], _diff)


def npz_pred_ratio(s):
    return float(_b0[0] + _b0[1] * math.log(s))


_byarm = defaultdict(list)
for r in _nr:
    _byarm[r['arm']].append(r['ratio'])
NPZ = dict(cutoff=NPZ_CUTOFF, dirs=NPZ_DIRS, n_runs=len(_nr), runs=_nr,
           ratio_by_arm={a: dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, n=len(v)) for a, v in _byarm.items()},
           fit_ratio_lnS=dict(a=float(_b0[0]), b=float(_b0[1]), b_se=math.sqrt(float(_cv0[1, 1])), resid_sd=math.sqrt(_rss0 / _df0), n=len(_nr)),
           arm_terms=dict(ref='ours_inter_ch', terms={a: dict(c=float(_b1[2 + i]), se=math.sqrt(float(_cv1[2 + i, 2 + i]))) for i, a in enumerate(_arms_npz)},
                          F=_F, p=float(stats.f.sf(_F, _q, _df1))),
           server_term=coef(_bs, _cvs, 2, _dfs),
           diff=dict(mean=float(np.mean(_diff)), sd=float(np.std(_diff, ddof=1)), min=float(min(_diff)), max=float(max(_diff)),
                     fit_a=float(_bd[0]), fit_b=float(_bd[1])),
           acc_match=dict(max_abs=float(max(abs(r['acc_npz'] - r['acc']) for r in _nr))))
_npz_by_tag = {(r['arm'], r['knob'], r['seed']): r['npz'] for r in _nr}


def npz_axis(S, c):
    """조건의 npz 축 스파이크: 실측 npz 가 있으면 그 값, 없으면 s_count × 곡선 비 (예측). 반환: (평균, 실측 런 수)."""
    vals, k = [], 0
    for x in c['runs']:
        v = _npz_by_tag.get((c['arm'], c['knob'], x['seed']))
        if v is None:
            v = x['spikes'] * npz_pred_ratio(x['spikes'])
        else:
            k += 1
        vals.append(v)
    return float(np.mean(vals)), k


_bc = cond(C10, 'baseline', '-')
_bnpz, _bk = npz_axis(C10, _bc)
NPZ['baseline'] = dict(npz_axis=_bnpz, measured=_bk, n=_bc['n'])
NPZ['conds'] = []
for c in C10['conds']:
    if c['arm'] == 'baseline':
        continue
    v, k = npz_axis(C10, c)
    NPZ['conds'].append(dict(arm=c['arm'], knob=c['knob'], n=c['n'], measured=k, dspk_scount=c['dspk'], npz_axis=v, dspk_npz=v / _bnpz - 1))
_ref_npz = npz_axis(C10, REF)[0]
NPZ['abl342'] = [dict(arm=x['arm'], dspk_scount=x['dspk'], dspk_npz=npz_axis(C10, cond(C10, *x['arm'].rsplit(' ', 1)))[0] / _ref_npz - 1,
                      measured=npz_axis(C10, cond(C10, *x['arm'].rsplit(' ', 1)))[1]) for x in ABL342]
NPZ['table1'] = [dict(arm=x['arm'], dspk_scount=x['dspk'], dspk_npz=npz_axis(C10, cond(C10, *x['arm'].rsplit(' ', 1)))[0] / _ref_npz - 1,
                      measured=npz_axis(C10, cond(C10, *x['arm'].rsplit(' ', 1)))[1]) for x in TABLE1]
# 같은 정확도 비교에 주는 영향: 팔 항 (같은 s_count 에서의 비 차) × 정확도 기울기
NPZ['acc_offset_shift_bound'] = max(abs(t['c']) for t in NPZ['arm_terms']['terms'].values()) * ANC['C10_methods'][(200e3, 420e3)]['slope']['b']
REV['npz'] = NPZ

# (7) QK C100: 검정 방식별 p, 로그 최고값 축
_qb = cond(QK100, 'baseline', '-')['runs']
_q6 = cond(QK100, 'ours_inter_ch', '6e-3')['runs']
_w = stats.ttest_ind([x['val_acc'] for x in _q6], [x['val_acc'] for x in _qb], equal_var=False)
_bs_ = {x['seed']: x for x in _qb}
_pr = [(x['val_acc'] - _bs_[x['seed']]['val_acc'], x['seed']) for x in _q6 if x['seed'] in _bs_]
_pt_ = stats.ttest_1samp([d for d, _ in _pr], 0.0)
_lg_groups = [[x['log_best'] for x in c['runs']] for c in QK100['conds']]
_lsp, _ldf = pooled_sd(_lg_groups)
_lb = [x['log_best'] for x in _qb]
_l6 = [x['log_best'] for x in _q6]
_ld = delta_stats(len(_l6), float(np.mean(_l6)), len(_lb), float(np.mean(_lb)), _lsp, _ldf)
REV['qk_c100'] = dict(pooled=dict(d=cond(QK100, 'ours_inter_ch', '6e-3')['dacc_d'], p=cond(QK100, 'ours_inter_ch', '6e-3')['dacc_p']),
                      welch=dict(t=float(_w.statistic), p=float(_w.pvalue)),
                      paired=dict(diffs=[dict(seed=s, d=d) for d, s in _pr], t=float(_pt_.statistic), p=float(_pt_.pvalue), n=len(_pr)),
                      log_axis=dict(base_mean=float(np.mean(_lb)), base_min=float(min(_lb)), arm_mean=float(np.mean(_l6)), sp=_lsp, df=_ldf,
                                    d=_ld['d'], se=_ld['se'], p=_ld['p'], in_seed_range=in_range(float(np.mean(_l6)), float(min(_lb)))))

# (8) QK 재측정 − 로그 최고값: 팔별 평균, 양 끝 런
_qrows = [dict(ds=ds, arm=c['arm'], knob=c['knob'], seed=x['seed'], d=x['val_acc'] - x['log_best'])
          for ds, S in (('C10', QK10), ('C100', QK100)) for c in S['conds'] for x in c['runs']]
_qg = OrderedDict()
for r in _qrows:
    _qg.setdefault((r['ds'], r['arm'], r['knob']), []).append(r['d'])
_qmin = min(_qrows, key=lambda r: r['d'])
_qmax = max(_qrows, key=lambda r: r['d'])
REV['qk_remeasure_minus_log'] = dict(n=len(_qrows), min=_qmin, max=_qmax,
                                     by_cond=[dict(ds=k[0], arm=k[1], knob=k[2], n=len(v), mean=float(np.mean(v)),
                                                   sd=float(np.std(v, ddof=1)) if len(v) > 1 else None) for k, v in _qg.items()])

# (9) Speck: 같은 체크포인트 종류끼리
_bb = [m for m in SPECK['measurements'] if m['model'] == 'baseline exp470 best']
_bbmW, _bbmJ = float(np.mean([m['mW'] for m in _bb])), float(np.mean([m['mJ'] for m in _bb]))
_b60mJ = float(np.mean([m['mJ'] for m in _b60]))
REV['speck'] = dict(best_vs_best=dict(baseline_mW=_bbmW, inter_mW=SPECK['summary']['inter_best_mW'], dmW=SPECK['summary']['inter_best_mW'] / _bbmW - 1,
                                      baseline_mJ=_bbmJ, inter_mJ=SPECK['summary']['inter_best_mJ'], dmJ=SPECK['summary']['inter_best_mJ'] / _bbmJ - 1),
                    ep60_vs_ep60=dict(baseline_mW=_b60mW, inter_mW=1.173, dmW=1.173 / _b60mW - 1,
                                      baseline_mJ=_b60mJ, inter_mJ=4.212, dmJ=4.212 / _b60mJ - 1))
REV['dvs_n_val'] = DVS_N_CHECK


# ================================================================ 저장 (JSON)
def clean(o):
    if isinstance(o, dict):
        return {str(k) if not isinstance(k, tuple) else f'{k[0]:.0f}-{k[1]:.0f}': clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


RESULT = dict(generated='2026-10-19', snapshot=INV['generated'], sources=dict(
    inventory=os.path.join(SNAP, 'inventory.json'), qk=os.path.join(SNAP, 'qk.json'), sdt=os.path.join(SNAP, 'sdt.json'),
    qk_remeasure=QKR + '/spikes_*.json', sdt_logs=os.path.join(SCR, 'sdt_logs')),
    r19_c10=C10, r19_c100=C100, dvs_f0=DVS, dvs_old=DVS_OLD, ancova=ANC, ablation_342k=ABL342, table1_340k=TABLE1,
    curve_offsets_ablation=CURVE_ABL, curve_offsets_methods=CURVE_MET, sdt_ctrl_note=SDT_CTRL_NOTE,
    qk_c10=QK10, qk_c100=QK100, qk_reference=QK_REFR, qk_pending=QK_PENDING,
    sdt_main=SDT_MAIN, sdt_kd=SDT_KD, speck=SPECK, review_261019=REV)
json.dump(clean(RESULT), open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)


# ================================================================ 보고서 (MD)
def f(x, d=2):
    return '' if x is None else f'{x:.{d}f}'


def fi(x):
    return '' if x is None else f'{x:,.0f}'


def pct(x, d=1):
    return '' if x is None else f'{x * 100:+.{d}f}%'


def pp(x, d=2):
    return '' if x is None else f'{x:+.{d}f}'


def sd_(m, s, d=2):
    return f(m, d) if s is None else f'{m:.{d}f} ±{s:.{d}f}'


def p_(p):
    return '' if p is None else ('<0.001' if p < 0.001 else f'{p:.3f}')


def peq(p):
    s = p_(p)
    return s if s.startswith('<') else '=' + s


def kn(arm, knob):
    return arm if knob in ('-', None) else f'{arm} {knob}'


L = []
w = L.append


def setting_table(S, arms):
    w('| 방법 | ρ/λ | seed | 서버 | train loss | train acc | val loss | val acc | spikes | spikes vs base | Δacc vs base (±SE) | p | S30/S1 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for c in S['conds']:
        if c['arm'] not in arms:
            continue
        for x in c['runs']:
            w(f"| {c['arm']} | {c['knob']} | s{x['seed']} | {x['server']} | {f(x['train_loss'], 4)} | {f(x['train_acc'])} | "
              f"{f(x['val_loss'], 4)} | {f(x['val_acc'])} | {fi(x['spikes'])} |  |  |  | {f(x['s30'], 3)} |")
        flag = '' if c['n'] >= 4 else ' ⚠n<4'
        if c['arm'] == 'baseline':
            w(f"| **baseline 평균** | | **n={c['n']}** | {'·'.join(c['servers'])} | {sd_(c['train_loss'], c['train_loss_sd'], 4)} | "
              f"{sd_(c['train_acc'], c['train_acc_sd'])} | {sd_(c['val_loss'], c['val_loss_sd'], 4)} | "
              f"**{sd_(c['val_acc'], c['val_acc_sd'])}** | **{fi(c['spikes'])}** |  |  |  | {sd_(c['s30'], c['s30_sd'], 3)} |")
        else:
            w(f"| **{c['arm']} {c['knob']} 평균** | | **n={c['n']}**{flag} | {'·'.join(c['servers'])} | {sd_(c['train_loss'], c['train_loss_sd'], 4)} | "
              f"{sd_(c['train_acc'], c['train_acc_sd'])} | {sd_(c['val_loss'], c['val_loss_sd'], 4)} | "
              f"**{sd_(c['val_acc'], c['val_acc_sd'])}** | **{fi(c['spikes'])}** | {pct(c['dspk'])} | "
              f"{pp(c['dacc_d'])} ±{c['dacc_se']:.2f} | {p_(c['dacc_p'])} | {sd_(c['s30'], c['s30_sd'], 3)} |")


def excl_table(S):
    rows = S['excluded']
    if not rows:
        w('(없음)')
        return
    w('| 방법 | ρ/λ | seed | 서버 | 에폭 | train loss | train acc | val loss | val acc | spikes | 사유 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for x in sorted(rows, key=lambda x: (x['arm'], kval(x['knob']), x['seed'])):
        w(f"| {x['arm']} | {x['knob']} | s{x['seed']} | {x['server']} | {x['epochs']} | {f(x['train_loss'], 4)} | {f(x['train_acc'])} | "
          f"{f(x['val_loss'], 4)} | {f(x['val_acc'])} | {fi(x['spikes'])} | {x['reason']} |")


def ptxt(m, with_test=True):
    s = f"{m['knob']} (스파이크 {pct(m['dspk'])}, 평균 {m['acc']:.2f}, n={m['n']}{' ⚠n<4' if m['n'] < 4 else ''}"
    if m.get('servers'):
        s += f", 서버 {'·'.join(m['servers'])}"
    if with_test:
        s += f", Δ {pp(m['dacc'])} ±{m['se']:.2f}, p{peq(m['p'])}, 최소 검출 차 {m['mdd']:.2f}"
    return s + ')'


def img_txt(x):
    x = abs(x)
    return f'{x:.1f}장' if x < 10 else f'{x:.0f}장'


def rob_txt(r):
    """시드 범위 판정의 취약성 서술 (검토 반영 10-19)."""
    if r is None:
        return ''
    if r['tie']:
        s = "기준과 동점 (차 0.000) — '≥' 규칙으로만 통과, 판정 근거로 인용하지 않음"
    else:
        s = (f"최저 시드와의 차 {r['margin']:+.3f}%p ({r['n']}런 합 이미지 {img_txt(r['margin_images'])} "
             f"{'초과' if r['margin'] > 0 else '부족'})")
    if r['loo_flip_seeds']:
        s += f", 런 하나 빼기로 판정이 뒤집힘 ({'·'.join('s' + str(x) for x in r['loo_flip_seeds'])})"
    return s


def lossless_line(S, arm, rkey=None):
    ll = S['lossless'][arm]
    rb = REV['robust'].get(rkey, {}).get(arm, {}) if rkey else {}
    s = f"- **{arm}** — 시드 범위 기준: "
    if ll['max'] is None:
        s += 'n≥3 통과 조건 없음'
    else:
        s += '평균이 baseline 최저 시드 이상인 마지막 n≥3 점 ' + ptxt(ll['max'])
        if rb.get('max'):
            s += ' — ' + rob_txt(rb['max'])
    if ll['contiguous'] and ll['max'] and ll['contiguous']['knob'] != ll['max']['knob']:
        s += f"; 연속 구간은 {ll['contiguous']['knob']} ({pct(ll['contiguous']['dspk'])}) 까지"
    if ll['first_fail']:
        s += f"; 처음 떨어지는 n≥3 조건 {ptxt(ll['first_fail'], False)}"
        if rb.get('first_fail'):
            s += ' — ' + rob_txt(rb['first_fail'])
    s += '. 검정 기준 (보정 없는 순차 검정): '
    if ll['test_first_sig']:
        tc = ll['test_contiguous']
        s += (f"baseline 아래가 처음 검출되는 n≥3 조건 {ptxt(ll['test_first_sig'])}"
              + (f", 그 앞까지 {tc['knob']} ({pct(tc['dspk'])})" if tc else ''))
        if ll.get('test_after_sig_nonsig'):
            s += (' — 그보다 더 줄인 ' + ', '.join(f"{c['knob']} (Δ {pp(c['dacc'])}, p{peq(c['p'])})" for c in ll['test_after_sig_nonsig'])
                  + ' 은 검출되지 않아 첫 검출은 위양성일 수 있음')
    else:
        s += 'n≥3 조건 중 baseline 아래로 검출된 조건 없음'
    t = ll.get('tost_max')
    s += '. ±0.2%p 동등성 (TOST): ' + ('통과 조건 없음' if not t else
                                     f"통과하는 가장 많이 줄인 n≥3 조건 {t['knob']} ({pct(t['dspk'])}, n={t['n']}, p{peq(t['tost'])})")
    n12 = ll['n12']
    if n12:
        s += '. n≤2 (판단 불가): ' + ', '.join(f"{b['knob']} ({pct(b['dspk'])}, {b['acc']:.2f}, n={b['n']})" for b in n12)
    w(s + '.')


def anc_table(A):
    w('| 구간 (스파이크) | N | 팔별 n | 기울기 b (%p/ln) | 비교 팔 | 같은 스파이크 오프셋 c (%p) | 95% CI | p | Holm p | 같은 정확도 스파이크 비 [95%] | 잔차 SD |')
    w('|---|---|---|---|---|---|---|---|---|---|---|')
    for a in A.values():
        if a is None:
            continue
        ns = ', '.join(f'{k} {v}' for k, v in a['n_by_arm'].items())
        weak = a['slope']['p'] >= 0.05
        for arm, o in a['offsets'].items():
            iso = f"{o['iso_ratio']:.2f} [{o['iso_lo']:.2f}, {o['iso_hi']:.2f}]" + (' (기울기 불확실)' if weak else '')
            w(f"| {a['band'][0] / 1e3:.0f}–{a['band'][1] / 1e3:.0f}K | {a['N']} | {ns} | {a['slope']['b']:+.3f} ±{a['slope']['se']:.3f} "
              f"(p{peq(a['slope']['p'])}) | {arm} | {o['c']:+.3f} ±{o['se']:.3f} | [{o['lo']:+.3f}, {o['hi']:+.3f}] | {p_(o['p'])} | "
              f"{p_(o.get('p_holm'))} | {iso} | {a['resid_sd']:.3f} |")


def curve_table(CO):
    w(f"| 팔 | ρ/λ | n | spikes | val acc | ours_inter_ch 곡선 예측 | 곡선 대비 차 (±SE) | p | 비고 |")
    w('|---|---|---|---|---|---|---|---|---|')
    lo, hi = CO['band']
    for c in CO['conds']:
        ext = '' if lo <= c['spikes'] <= hi else '곡선 구간 밖 (외삽)'
        w(f"| {c['arm']} | {c['knob']} | {c['n']} | {fi(c['spikes'])} | {c['acc']:.2f} | {c['pred']:.2f} | {pp(c['d'], 3)} ±{c['se']:.3f} | {p_(c['p'])} | {ext} |")


def offs_txt(A, keys=None):
    return ', '.join(f"{k} {o['c']:+.3f} [{o['lo']:+.3f}, {o['hi']:+.3f}] (n={A['n_by_arm'][k]}, p{peq(o['p'])})"
                     for k, o in A['offsets'].items() if keys is None or k in keys)


# ---------------- 머리
def mdd_n4(S):
    """n=4 조건 대 baseline 의 최소 검출 차 (80% 검정력, 양측 0.05)."""
    return delta_stats(4, 0.0, S['base']['n'], 0.0, S['pooled_sd'], S['pooled_df'])['mdd']


NP = REV['npz']
w('# EIP 현재 결과 요약 (2026-10-19)')
w('')
w(f"원장 고정 사본 {INV['generated']} 기준 (모든 보고서가 같은 n 을 쓰도록 고정). 생성: `analysis/results_261019.py` → `.json` (모든 수치) / `.md` (이 문서). "
  '10-19 eip-critic 검토를 반영해 고쳤다 (맨 끝 "검토 반영" 절).')
w('QKFormer 는 `~/runs/qkformer_wta_rev/spikes_*.json` 재측정값 (체크포인트 fp32 재측정), SDTv3 는 원장 `sdt.json` + sejong `log.txt` 읽기 전용 사본, Speck2f 는 `docs/decisions.md` 10-15·10-16 항에서 옮김.')
w('')
w('**정의.** 정확도 = 런 전체 best val acc, spikes = 그 에폭의 s_count (샘플당, 원값). **이 문서의 절대 스파이크 수와 감소율은 모두 원장 s_count 축이다** '
  f"(npz 총합 축으로 바꾸면 baseline 대비 감소율이 1~2%p 작아진다 — 1.7 절). 평균 행은 완주·비배제 런만. "
  'Δacc 의 ± 는 차이의 표준오차 SE = sp·√(1/n + 1/n_base), sp 는 설정별 합동 시드 SD (n≥2 조건들의 조건 내 분산을 합친 값 — 조건마다 SD 를 따로 쓰면 n=3 에서 너무 흔들려서). '
  'p 는 그 SE 로 계산한 양측 t 검정 (자유도 = 합동 자유도). 최소 검출 차 = 80% 검정력으로 볼 수 있는 가장 작은 차 ((t₀.₉₇₅+t₀.₈₀)·SE).')
w('')
w('**정확도 유지 판정 세 가지 — 어느 것도 "손실이 없다" 를 증명하지 않는다.** '
  '(1) **시드 범위 기준 (이번 요청의 정의)**: 조건 평균 정확도 ≥ baseline 시드 최저값. n≥3 조건만 판정, n≤2 는 "판단 불가". '
  '"마지막 점" = 통과한 조건 중 스파이크를 가장 많이 줄인 것, "연속" = 그보다 덜 줄인 n≥3 조건이 모두 통과한 범위. '
  '이 기준은 느슨하다 — 평균이 baseline 최저 시드와 같으면 통과하므로 baseline 평균보다 시드 SD 1~1.5배 낮아도 통과한다. 그래서 이 문서는 통과한 마지막 점을 "무손실 한계" 라고 부르지 않는다. '
  '(2) **검정 기준 (보정 없는 순차 검정, 참고)**: 스파이크를 덜 줄인 조건부터 보며 Δ<0 이고 p<0.05 인 n≥3 조건이 처음 나오기 전까지. 조건 약 10개를 보정 없이 차례로 검정하므로 첫 검출이 위양성일 수 있고, 검출되지 않음은 "같다" 가 아니다. '
  '(3) **±0.2%p 동등성 검정 (TOST, 참고)**: Δ 의 90% 구간이 (−0.2, +0.2) 안에 들어가면 통과. 마진 0.2%p 는 합의된 값이 아니다 (10-14 inter 타당성 분석에서 쓴 값).')
w('')
w('**같은 스파이크 비교 (ANCOVA).** acc = a + b·ln(spikes) + c·[팔] + d·[서버 23]; c 가 기준 팔 (ours_inter_ch) 대비 같은 스파이크에서의 정확도 차 (c>0 이면 비교 팔이 더 높음). '
  '"같은 정확도 스파이크 비" = exp(−c/b) = 같은 정확도를 내는 데 비교 팔이 쓰는 스파이크 / 기준 팔 (1 미만이면 비교 팔이 덜 씀), 95% 구간은 델타법. 여러 팔을 한 번에 비교한 p 는 Holm 보정 값을 같이 적는다.')
w('')
w('표 규약: 열 순서 train loss → train acc → val loss → val acc → spikes. 반복 런은 개별값 뒤 **평균 ±SD (n)** 행. ⚠n<4 = 반복 4회 미만. '
  'S30/S1 = 30에폭 스파이크 / 1에폭 스파이크 (0.19 미만은 원장 배제 규칙).')
w('')

# ---------------- 요약
cl10, c1 = C10['lossless'], C100['lossless']
i3 = cond(C10, 'ours_inter_ch', '3e-3')
RB = REV['robust']
cap_l2 = REV['capped4']['C10']['plain L2']
cap_l2_25 = next(x for x in cap_l2['capped_conds'] if x['knob'] == '2.5e-7')
isrv = REV['inter_server_term']
npc = {(c['arm'], c['knob']): c for c in NP['conds']}
w('## 요약')
w('')
w('**먼저: 이 문서의 어떤 수치도 우리 방법 (ours_inter_ch) 고유의 정확도·스파이크 우위를 보여주지 않는다.** R19 CIFAR-10 에서 스파이크를 약 −47~−52% 줄이면서 평균이 baseline 최저 시드에 닿는 것은 plain L2·BPSR 도 같고, '
  '같은 스파이크 비교는 세 방법을 구별하지 못한다 (정확도 ±0.07%p, 스파이크로는 ±20% 안팎). Speck2f 칩 전력과 VGGSNN DVS 수치는 학습 n=1 이거나 판정 불가 폭이 커서 **실측 사례**로만 쓴다. '
  '절대 스파이크 수는 원장 s_count 축이다.')
w('')
w(f"1. **R19 CIFAR-10 대표점** ours_inter_ch ρ 3e-3: {i3['val_acc']:.2f} ±{i3['val_acc_sd']:.2f} @ {fi(i3['spikes'])} (n={i3['n']}, 138·23), "
  f"baseline {C10['base']['acc']:.2f} ±{C10['base']['acc_sd']:.2f} @ {fi(C10['base']['spikes'])} (n={C10['base']['n']}) 대비 "
  f"Δacc {pp(i3['dacc_d'])} ±{i3['dacc_se']:.2f} (p{peq(i3['dacc_p'])}), 스파이크 {pct(i3['dspk'])} (npz 축 {pct(npc[('ours_inter_ch', '3e-3')]['dspk_npz'])}). "
  f"baseline 아래가 검출되지 않을 뿐이고 최소 검출 차가 {i3['dacc_mdd']:.2f}%p 라 그보다 작은 손실은 못 본다. 합동 시드 SD {C10['pooled_sd']:.3f}%p (자유도 {C10['pooled_df']}).")
mi, ml, mb = cl10['ours_inter_ch'], cl10['plain L2'], cl10['BPSR']
ri, rl, rbp = RB['C10']['ours_inter_ch']['max'], RB['C10']['plain L2']['max'], RB['C10']['BPSR']['max']
w(f"2. **R19 C10 시드 범위 기준 — 평균이 baseline 최저 시드 ({C10['base']['acc_min']:.2f}) 이상인 마지막 n≥3 점**: "
  f"ours_inter_ch {ptxt(mi['max'], False)}, plain L2 {ptxt(ml['max'], False)}, BPSR {ptxt(mb['max'], False)}. "
  f"세 점 모두 Δ {pp(min(mi['max']['dacc'], ml['max']['dacc'], mb['max']['dacc']))}~{pp(max(mi['max']['dacc'], ml['max']['dacc'], mb['max']['dacc']))}%p 이고 시드 하나에 판정이 뒤집힌다 "
  f"(ours_inter_ch {'·'.join('s' + str(x) for x in ri['loo_flip_seeds'])} 빼면 불통과, plain L2 {'·'.join('s' + str(x) for x in rl['loo_flip_seeds'])}, BPSR {'·'.join('s' + str(x) for x in rbp['loo_flip_seeds'])}; BPSR 은 기준과 동점 차 0.000). "
  f"plain L2 2.5e-7 을 10-17 'n=4 통일' 규칙대로 s1~s4 만 쓰면 {cap_l2_25['val_acc']:.2f} 로 불통과 (Δ {pp(cap_l2_25['dacc_d'])}, p{peq(cap_l2_25['dacc_p'])}) 이고 plain L2 의 마지막 점은 {ptxt(cap_l2['max'], False)} 가 된다. "
  f"±0.2%p 동등성 검정은 세 점 모두 불통과이고, 동등성 기준으로 통과하는 가장 많이 줄인 n≥3 점도 ours_inter_ch {mi['tost_max']['knob']} ({pct(mi['tost_max']['dspk'])}, n={mi['tost_max']['n']}), "
  f"plain L2 {ml['tost_max']['knob']} ({pct(ml['tost_max']['dspk'])}, n={ml['tost_max']['n']}), BPSR {mb['tost_max']['knob']} ({pct(mb['tost_max']['dspk'])}, n={mb['tost_max']['n']}) 로 겹친다. 경계 쌍은 ρ 와 서버가 겹친다 (ours_inter_ch 5.5e-3 전부 138, 6e-3 전부 23) — 다만 ours_inter_ch 안의 서버 항은 {isrv['c']:+.3f} ±{isrv['se']:.3f} (n={isrv['n']}) 라 서버 탓이라는 근거는 없다. "
  '→ **이 기준으로 방법 간 우열을 말할 수 없다.**')
a = ANC['C10_methods'][(200e3, 420e3)]
oB, oL = a['offsets']['BPSR'], a['offsets']['plain L2']
sh = REV['slope_homog']['C10_methods_200-420K']
w(f"3. **같은 스파이크 방법 비교 (R19 C10, 200–420K, 서버 항 포함)**: ours_inter_ch 대비 {offs_txt(a)}. 기준 n={a['n_by_arm']['ours_inter_ch']}. "
  f"**구별되지 않음** (95% CI 반폭 약 {max((o['hi'] - o['lo']) / 2 for o in a['offsets'].values()):.2f}%p). 스파이크 축으로 옮기면 같은 정확도에서 BPSR 이 ours_inter_ch 의 "
  f"{oB['iso_ratio']:.2f}배 [{oB['iso_lo']:.2f}, {oB['iso_hi']:.2f}], plain L2 가 {oL['iso_ratio']:.2f}배 [{oL['iso_lo']:.2f}, {oL['iso_hi']:.2f}] 를 쓴다 — BPSR 이 스파이크를 최대 약 {(1 - oB['iso_lo']) * 100:.0f}% 덜 쓸 가능성을 배제하지 못한다. "
  f"'같은 성능' 이나 '동등' 이라고 쓰지 않는다. 팔마다 기울기가 같다는 가정은 기각되지 않음 (F p={sh['p']:.2f}). "
  f"표1 지점 (각 n=4): ours_inter_ch 3e-3 {TABLE1[0]['acc_ref']:.2f} @ {fi(TABLE1[0]['spk_ref'])} / "
  f"plain L2 1e-7 {TABLE1[0]['acc']:.2f} @ {fi(TABLE1[0]['spk'])} / BPSR 1e-10 {TABLE1[1]['acc']:.2f} @ {fi(TABLE1[1]['spk'])} — BPSR 가 스파이크 {pct(TABLE1[1]['dspk'])} 에 Δ {pp(TABLE1[1]['dacc'])} ±{TABLE1[1]['se']:.2f}.")
ab = ANC['C10_ablation'][(200e3, 420e3)]
vs = REV['vmem_by_server']
vl = REV['vmem_loo']
rl2 = next(x for x in REV['ref342_loo'] if x['drop_seed'] == 2)
fsr = REV['final_step_by_rho']
_nunc = REV['ablation_uncorrected_sig']
w('4. **R19 C10 ablation (기준 ours_inter_ch 3e-3, n=4)**: 342K 직접 비교 (각 n=4) ' + ', '.join(
    f"{x['arm']} {pp(x['dacc'])} ±{x['se']:.2f} (p={x['p']:.4f}, Holm {p_(x['p_holm'])})" for x in ABL342) +
  f". 팔의 모든 ρ 를 쓴 같은 스파이크 오프셋 (200–420K): {offs_txt(ab)}. "
  f"**4팔 모두 Holm 보정 후 p<0.05 없음.** 보정 전 p<0.05 는 {len(_nunc)}건 (- vmem {sum(u['what'].startswith('- vmem') for u in _nunc)}건, "
  f"- final_step {sum(u['what'].startswith('- final_step') for u in _nunc)}건; 목록은 1.5 절). "
  f"- vmem 신호는 138 서버에만 있고 (138 {vs['at138']['c']:+.3f} n={vs['n138']} / 23 {vs['at23']['c']:+.3f} n={vs['n23']}, 차 p={vs['diff_23_minus_138']['p']:.3f}), "
  f"런 하나를 빼면 9번 중 {9 - vl['n_sig']}번 p≥0.05 (2e-3 s3 96.42 를 빼면 p={vl['max_p']:.3f}). "
  f"342K 직접 비교는 기준점의 s2=96.89 에 기댄다 (s2 를 빼면 - vmem {pp(rl2['arms']['- vmem 2e-3']['d'])}, - final_step {pp(rl2['arms']['- final_step 1.4e-3']['d'])}, 둘 다 p>0.1). "
  f"- final_step 은 1.4e-3 이 곡선 아래 (p={fsr['1.4e-3']['p']:.3f}, 보정 전), 2.2e-3 은 곡선과 구별 안 됨 (p={fsr['2.2e-3']['p']:.2f}), 두 ρ 의 차 p={fsr['diff_1.4_minus_2.2']['p']:.3f} → 팔 공통 오프셋은 해석하지 않는다. "
  f"최소 검출 차 {ABL342[0]['mdd']:.2f}%p → **'기여 검출 안 됨' 이지 '기여 없음' 이 아니다.**")
c3 = cond(C100, 'ours_inter_ch', '3e-3')
a100 = ANC['C100_methods'][(220e3, 420e3)]
r3 = RB['C100']['ours_inter_ch']['first_fail']
w(f"5. **R19 CIFAR-100**: baseline {C100['base']['acc']:.2f} ±{C100['base']['acc_sd']:.2f} @ {fi(C100['base']['spikes'])} (n={C100['base']['n']}). "
  f"ours_inter_ch 시드 범위 기준 마지막 점 {ptxt(c1['ours_inter_ch']['max'])}; 3e-3 ({pct(c3['dspk'])}, n=3) 은 평균 {c3['val_acc']:.3f} 로 최저 {C100['base']['acc_min']:.2f} 에 {C100['base']['acc_min'] - c3['val_acc']:.3f}%p 모자람 "
  f"(3런 합 이미지 약 {img_txt(r3['margin_images'])}; Δ {pp(c3['dacc_d'])}, p{peq(c3['dacc_p'])}). plain L2 는 {ptxt(c1['plain L2']['max'], False)} 까지 통과. "
  f"**시드 범위 판정은 어느 ρ 에 n≥3 을 줬는지와 이미지 몇 장 차에 좌우되고**, 같은 스파이크 비교 (220–420K, 같은 날 정한 구간이라 사전 등록 아님) 는 {offs_txt(a100)} — "
  f"±0.15%p (스파이크로는 plain L2 {a100['offsets']['plain L2']['iso_lo']:.2f}~{a100['offsets']['plain L2']['iso_hi']:.2f}배, BPSR {a100['offsets']['BPSR']['iso_lo']:.2f}~{a100['offsets']['BPSR']['iso_hi']:.2f}배) 안에서 구별 불가.")
dl = DVS['lossless']['ours_inter_ch']
_d4 = sorted([c for c in DVS['conds'] if c['arm'] == 'ours_inter_ch' and c['n'] >= 4], key=lambda c: -c['dspk'])
_dseq = ' → '.join(f"{c['knob']} {'통과' if c['in_seed_range'] else '불통과'}" for c in _d4)
_dr = {c['knob']: judge_robust(DVS, c, 992) for c in _d4}
w(f"6. **VGGSNN CIFAR10-DVS (첫 프레임, 실측 사례)**: baseline {DVS['base']['acc']:.2f} ±{DVS['base']['acc_sd']:.2f} @ {fi(DVS['base']['spikes'])} (n={DVS['base']['n']}, 전부 138); 합동 SD {DVS['pooled_sd']:.2f}%p (R19 C10 의 {DVS['pooled_sd'] / C10['pooled_sd']:.1f}배). "
  f"스파이크 {pct(max(c['dspk'] for c in _d4), 0)}~{pct(min(c['dspk'] for c in _d4), 0)} 의 n=4 조건 {len(_d4)}개 모두 Δ {pp(min(c['dacc_d'] for c in _d4))}~{pp(max(c['dacc_d'] for c in _d4))}, p≥{min(c['dacc_p'] for c in _d4):.2f} "
  f"— n=4 대 4 의 최소 검출 차가 {_d4[0]['dacc_mdd']:.2f}%p 라 **baseline 과 1%p 미만의 차는 판정 불가**. 시드 범위 판정은 {_dseq} 로 단조롭지 않고, "
  f"떨어지는 2e-3·1e-2 는 4런 합 이미지 {img_txt(_dr['2e-3']['margin_images'])}·{img_txt(_dr['1e-2']['margin_images'])} 부족 (검증셋 992장, 1장 = 0.101%p) — 이미지 1장 미만 차로 갈린다. "
  "'−60.6% 무손실' 을 단독으로 인용하지 않는다.")
q12 = cond(QK10, 'ours_inter_ch', '1.2e-2')
q6 = cond(QK100, 'ours_inter_ch', '6e-3')
qc = REV['qk_c100']
w(f"7. **QKFormer (재측정, 전부 138)**: C10 baseline {QK10['base']['acc']:.2f} ±{QK10['base']['acc_sd']:.2f} @ {fi(QK10['base']['spikes'])} (n={QK10['base']['n']}); "
  f"ρ 1.2e-2 {q12['val_acc']:.2f} ±{q12['val_acc_sd']:.2f} @ {fi(q12['spikes'])} (n={q12['n']}, {pct(q12['dspk'])}, Δ {pp(q12['dacc_d'])} ±{q12['dacc_se']:.2f}, p{peq(q12['dacc_p'])}) "
  f"→ baseline 아래가 검출되지 않음 (최소 검출 차 {q12['dacc_mdd']:.2f}%p — 이보다 작은 손실은 못 봄). ρ 2e-2 96.00 @ −44% 는 n=1 (판단 불가). "
  f"C100 baseline {QK100['base']['acc']:.2f} ±{QK100['base']['acc_sd']:.2f} (n={QK100['base']['n']}); ρ 6e-3 {q6['val_acc']:.2f} ±{q6['val_acc_sd']:.2f} (n={q6['n']}, {pct(q6['dspk'])}): "
  f"재측정 기준 Δ {pp(q6['dacc_d'])} (p {qc['pooled']['p']:.3f}~{qc['paired']['p']:.3f}, 검정 방식에 따라: 합동 SD {qc['pooled']['p']:.3f} / Welch {qc['welch']['p']:.3f} / 같은 시드 짝 {qc['paired']['p']:.3f}). "
  f"학습 로그 최고값 기준으로는 Δ {pp(qc['log_axis']['d'])} (p={qc['log_axis']['p']:.3f}) 이지만 시드 범위는 통과 ({qc['log_axis']['arm_mean']:.3f} ≥ 로그 최저 {qc['log_axis']['base_min']:.2f}). "
  "ρ<6e-3 (−10.6% 보다 약한 규제) 은 시험하지 않았다.")
s01 = next(x for x in SDT_MAIN if x['knob'] == '0.1')
s02 = next(x for x in SDT_MAIN if x['knob'] == '0.2')
w(f"8. **SDTv3 171M ImageNet 파인튜닝 (비KD, 마지막 에폭, 각 n=1, 23 서버)**: 대조 {CTRL['val_last']:.3f} @ {fi(CTRL['spikes'])}. "
  f"ρ 0.1 {pp(s01['dacc'], 3)}%p / {pct(s01['dspk'])}, ρ 0.2 {pp(s02['dacc'], 3)}%p / {pct(s02['dspk'])}. ImageNet 시드 편차 미측정 → 정확도 유지 판정 불가 (서술만). KD 4런 진행 중.")
sp_ = SPECK['summary']
spr = REV['speck']
w(f"9. **Speck2f 칩 (CIFAR10-DVS 9층, 실측 사례, 각 모델 학습 n=1)**: ours_inter_ch ρ 1.0 best — SW 스파이크 {pct(sp_['sw_spike_change_best'])}, "
  f"칩 동적 전력 {sp_['inter_best_mW']:.3f} mW (측정 {sp_['inter_k']}회 평균) vs baseline {sp_['baseline_mW']:.3f} mW (best 2회 + epoch_060 2회) = {pct(sp_['dmW'])}. "
  f"같은 체크포인트 종류끼리는 best 끼리 {pct(spr['best_vs_best']['dmW'])}, **epoch_060 끼리 {pct(spr['ep60_vs_ep60']['dmW'])}** (보수적인 같은 조건 값). "
  f"에너지/표본 {pct(sp_['dmJ'])} (best 끼리 {pct(spr['best_vs_best']['dmJ'])}, epoch_060 끼리 {pct(spr['ep60_vs_ep60']['dmJ'])}). SW 정확도 64.7 → 64.1 (−0.6%p), 칩 정확도 63.0 → 62.9 (n=1000 표본 SE 약 1.5%p).")
w('10. **진행 중·주의**: R19 C10 ours_inter_ch 6런 (1.5e-3 s4, 2e-3 s4, 4e-3 s4, 5e-3 s2·s3·s4), SDTv3 KD 4런. '
  'R19 C10 plain L2 5e-8 s4·1.5e-7 s4 는 10-19 사용자 중단 — 결과 아님 (해당 조건은 n=3 으로 남음). QK C10 ρ 2e-2 s43~45 는 대기 스크립트 기동 승인 대기 (미시작).')
w('')

# ---------------- 1. R19 C10
w('## 1. R19 CIFAR-10')
w('')
w(f"baseline 시드 범위 {C10['base']['acc_min']:.2f}~{C10['base']['acc_max']:.2f} (n={C10['base']['n']}). 합동 시드 SD {C10['pooled_sd']:.3f}%p (자유도 {C10['pooled_df']}, baseline·ours_inter_ch·plain L2·BPSR·ablation 팔의 n≥2 조건 전부). "
  f"n=4 조건 대 baseline 의 최소 검출 차 {mdd_n4(C10):.2f}%p.")
w('')
w('### 1.1 baseline + ours_inter_ch (ρ 전부)')
w('')
setting_table(C10, ['baseline', 'ours_inter_ch'])
w('')
w('### 1.2 plain L2')
w('')
setting_table(C10, ['plain L2'])
w('')
w('### 1.3 BPSR')
w('')
setting_table(C10, ['BPSR'])
w('')
w('### 1.4 정확도 유지 판정과 같은 스파이크 방법 비교')
w('')
for arm in ['ours_inter_ch', 'plain L2', 'BPSR']:
    lossless_line(C10, arm, 'C10')
w('')
w(f"- **n=4 통일 규칙 (10-17) 으로 다시 판정**: n=5 인 plain L2 2.5e-7·5e-7 을 시드 번호가 작은 4개 (s1~s4) 로 줄이면 " +
  ', '.join(f"{x['knob']} {x['val_acc']:.2f} (Δ {pp(x['dacc_d'])}, p{peq(x['dacc_p'])}, {'통과' if x['in_seed_range'] else '불통과'})" for x in cap_l2['capped_conds']) +
  f" → plain L2 의 시드 범위 마지막 점은 {ptxt(cap_l2['max'], False)}. ours_inter_ch·BPSR 은 n≤4 라 바뀌지 않는다.")
w(f"- **경계 쌍의 ρ·서버 겹침**: ours_inter_ch 5.5e-3 (통과) 는 전부 138, 6e-3 (불통과) 는 전부 23. ours_inter_ch 런만으로 맞춘 서버 항 (200–420K, n={isrv['n']}, 23 서버 {isrv['n23']}런): "
  f"{isrv['c']:+.3f} ±{isrv['se']:.3f} [{isrv['lo']:+.3f}, {isrv['hi']:+.3f}] → 서버 차이라는 근거는 없지만 ±0.08%p 를 배제하지도 못한다.")
w('')
w('**표1 지점 (약 320~340K, 각 n=4).**')
w('')
w('| 방법 | n | 서버 (런 수) | val acc | spikes | ours_inter_ch 3e-3 대비 Δacc (±SE) | p | 최소 검출 차 | spikes 차 (s_count) | spikes 차 (npz 축) |')
w('|---|---|---|---|---|---|---|---|---|---|')


def srv_txt(c):
    from collections import Counter
    k = Counter(x['server'] for x in c['runs'])
    return ' · '.join(f'{s} ({k[s]})' for s in sorted(k))


w(f"| ours_inter_ch 3e-3 | {REF['n']} | {srv_txt(REF)} | {REF['val_acc']:.2f} ±{REF['val_acc_sd']:.2f} | {fi(REF['spikes'])} |  |  |  |  |  |")
for x, xn in zip(TABLE1, NP['table1']):
    c = cond(C10, *x['arm'].rsplit(' ', 1))
    w(f"| {x['arm']} | {x['n']} | {srv_txt(c)} | {c['val_acc']:.2f} ±{c['val_acc_sd']:.2f} | {fi(x['spk'])} | {pp(x['dacc'])} ±{x['se']:.2f} | {p_(x['p'])} | {x['mdd']:.2f} | {pct(x['dspk'])} | {pct(xn['dspk_npz'])} (npz 실측 {xn['measured']}/{x['n']}) |")
w('')
w('스파이크가 5~6% 다르므로 같은 스파이크 비교는 아래 ANCOVA 로 본다. 주 구간 200–420K 는 이전 분석 (10-09) 과 같은 구간이고 나머지는 민감도 (사전 등록된 구간은 없다).')
w('')
anc_table(ANC['C10_methods'])
nsv = ANC['C10_methods_noserver']
w('')
w('서버 항을 빼면 (200–420K): ' + ', '.join(f"{k} {o['c']:+.3f} [{o['lo']:+.3f}, {o['hi']:+.3f}]" for k, o in nsv['offsets'].items()) + '. '
  f"서버 항 자체는 {a['server23']['d']:+.3f} ±{a['server23']['se']:.3f}. 팔마다 기울기를 따로 둔 모형과의 비교 F p={sh['p']:.2f} (공통 기울기 가정 유지).")
w('')
w('- **경계**: 정확도 축의 ±0.07%p 는 스파이크 축으로 ±20% 안팎이다. "BPSR 과 같은 스파이크에서 같은 정확도" 라고도 말할 수 없다 — 같은 정확도에서 BPSR 이 스파이크를 최대 약 '
  f"{(1 - oB['iso_lo']) * 100:.0f}% 덜 쓸 가능성도, {(oB['iso_hi'] - 1) * 100:.0f}% 더 쓸 가능성도 남아 있다. 이 구간은 델타법 근사다 (기울기 b 의 불확실성이 크면 구간이 비대칭으로 넓어진다).")
w('')
w(f"조건별로 ours_inter_ch 곡선 (200–420K, n={CURVE_MET['n_ref']}, b={CURVE_MET['b']:+.3f}%p/ln, 잔차 SD {CURVE_MET['resid_sd']:.3f}) 과 비교:")
w('')
curve_table(CURVE_MET)
w('')
w('- 조건별 p 는 다중 비교 보정 전 값이다 (조건 10개).')
w('')
w('### 1.5 ablation (ours_inter_ch 에서 한 요소만 끈 팔)')
w('')
w('`- vmem` = ch_novmem, `- final_step` = ch_nofinal, `- 1- inversion` = ch_noinv, `- loss-ratio` = ch_nolr (loss-ratio 대신 고정 λ). 각 팔의 342K 구간 조건은 n=4.')
w('')
setting_table(C10, list(ABL_NAME.values()))
w('')
w('**342K 구간 직접 비교 (기준 ours_inter_ch 3e-3, n=4).**')
w('')
w('| 팔 | n | 서버 (런 수) | val acc | spikes | 기준 대비 Δacc (±SE) | p | Holm p (4팔) | 최소 검출 차 | 기준 대비 spikes (s_count) | 기준 대비 spikes (npz 축, 예측) |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
w(f"| ours_inter_ch 3e-3 (기준) | {REF['n']} | {srv_txt(REF)} | {REF['val_acc']:.2f} ±{REF['val_acc_sd']:.2f} | {fi(REF['spikes'])} |  |  |  |  |  |  |")
for x, xn in zip(ABL342, NP['abl342']):
    c = cond(C10, *x['arm'].rsplit(' ', 1))
    w(f"| {x['arm']} | {x['n']} | {srv_txt(c)} | {c['val_acc']:.2f} ±{c['val_acc_sd']:.2f} | {fi(x['spk'])} | {pp(x['dacc'])} ±{x['se']:.2f} | {x['p']:.4f} | {p_(x['p_holm'])} | {x['mdd']:.2f} | {pct(x['dspk'])} | {pct(xn['dspk_npz'])} |")
w('')
_spk_t = []
for x in ABL342:
    c = cond(C10, *x['arm'].rsplit(' ', 1))
    tt = stats.ttest_ind([r['spikes'] for r in c['runs']], [r['spikes'] for r in REF['runs']], equal_var=False)
    _spk_t.append(f"{x['arm']} {pct(x['dspk'])} (Welch p={tt.pvalue:.3f})")
w('- npz 축 값은 ablation 팔에 npz 가 없어 (실측 0/4) s_count × 곡선 비로 예측한 값이다 (1.7 절). 측정한 팔 사이의 비 차는 같은 s_count 에서 약 0.3% (95% 구간으로도 0.6% 안쪽) 다. '
  's_count 축 스파이크 차 자체의 검정 (n=4 대 4): ' + ', '.join(_spk_t) + ' — - vmem 만 스파이크가 기준보다 많은 쪽으로 구별되고, 나머지 ±2% 안팎의 차는 방향을 주장할 크기가 아니다.')
w('')
w('**같은 스파이크 오프셋 (ANCOVA, 팔의 완주·비배제 런 전부).**')
w('')
anc_table(ANC['C10_ablation'])
nsv = ANC['C10_ablation_noserver']
w('')
w('서버 항을 빼면 (200–420K): ' + ', '.join(f"{k} {o['c']:+.3f} [{o['lo']:+.3f}, {o['hi']:+.3f}]" for k, o in nsv['offsets'].items()) + '. '
  f"기울기 동질성 F p={REV['slope_homog']['C10_ablation_200-420K']['p']:.2f}.")
w('')
w(f"**조건별 곡선 대비 차** (ours_inter_ch 곡선 200–420K, n={CURVE_ABL['n_ref']}):")
w('')
curve_table(CURVE_ABL)
w('')
w(f"- 조건별 p 는 다중 비교 보정 전 값이다 (조건 {len(CURVE_ABL['conds'])}개). n=1 조건의 SE 는 곡선 잔차 SD 그대로라 크다.")
w('- **보정 전 p<0.05 전부** (어느 것도 Holm 보정 후에는 0.05 아래가 아니다):')
for u in REV['ablation_uncorrected_sig']:
    w(f"  - {u['where']}: {u['what']} {pp(u['d'], 3)}, p={u['p']:.4f}")
w(f"- **- vmem 의 서버 쏠림** (200–420K, 팔×서버 모형): 138 {vs['at138']['c']:+.3f} ±{vs['at138']['se']:.3f} (n={vs['n138']}, p={vs['at138']['p']:.3f}), "
  f"23 {vs['at23']['c']:+.3f} ±{vs['at23']['se']:.3f} (n={vs['n23']}, p={vs['at23']['p']:.2f}), 차 {vs['diff_23_minus_138']['c']:+.3f} ±{vs['diff_23_minus_138']['se']:.3f} (p={vs['diff_23_minus_138']['p']:.3f}).")
w('- **- vmem 런 하나 빼기** (팔 공통 ANCOVA 200–420K): ' + ', '.join(f"{x['knob']} s{x['seed']} ({x['server']}, {x['acc']:.2f}) → {pp(x['c'], 3)} p={x['p']:.3f}" for x in vl['runs'])
  + f". {len(vl['runs'])}번 중 {vl['n_sig']}번만 p<0.05.")
w('- **342K 기준점 시드 하나 빼기** (기준 ours_inter_ch 3e-3): ' + '; '.join(
    f"s{x['drop_seed']} 빼면 기준 {x['ref_acc']:.3f} → - vmem {pp(x['arms']['- vmem 2e-3']['d'])} (p={x['arms']['- vmem 2e-3']['p']:.3f}), - final_step {pp(x['arms']['- final_step 1.4e-3']['d'])} (p={x['arms']['- final_step 1.4e-3']['p']:.3f})"
    for x in REV['ref342_loo']) + '. 기준점은 s2=96.89 한 런에 기대고 있다 (기준 3e-3 s5 는 큐 hold).')
w(f"- **- final_step 은 ρ 마다 따로 본다**: ρ 별 오프셋 ANCOVA (200–420K, 서버 항 포함) 에서 1.4e-3 (n=4, 346K) {pp(fsr['1.4e-3']['c'], 3)} ±{fsr['1.4e-3']['se']:.3f} (p={fsr['1.4e-3']['p']:.3f}, 보정 전) 은 곡선 아래, "
  f"2.2e-3 (n=3, 286K) {pp(fsr['2.2e-3']['c'], 3)} ±{fsr['2.2e-3']['se']:.3f} (p={fsr['2.2e-3']['p']:.2f}) 은 곡선과 구별 안 됨. 두 ρ 의 차 {pp(fsr['diff_1.4_minus_2.2']['c'], 3)} ±{fsr['diff_1.4_minus_2.2']['se']:.3f} "
  f"(p={fsr['diff_1.4_minus_2.2']['p']:.3f}) — 팔 전체에 공통 오프셋을 두는 가정이 이 팔에서는 깨지므로 팔 공통 오프셋 {ab['offsets']['- final_step']['c']:+.3f} 는 해석하지 않는다.")
w(f"- n=4 대 4 의 최소 검출 차는 {ABL342[0]['mdd']:.2f}%p 다. 이 절의 결론은 **'네 요소 모두 정확도 기여가 검출되지 않음'** 이지 '기여 없음' 이 아니다.")
w('- 각 팔의 정확도 유지 판정 (참고):')
for arm in ABL_NAME.values():
    lossless_line(C10, arm, 'C10')
w('')
w('### 1.6 진행 중 · 배제 · 대기')
w('')
w('**진행 중 (평균에 넣지 않음, 현재까지의 best 값):**')
w('')
w('| 방법 | ρ | seed | 서버 | 에폭 | train loss | train acc | val loss | val acc | spikes |')
w('|---|---|---|---|---|---|---|---|---|---|')
for x in sorted(C10['running'], key=lambda x: (kval(x['knob']), x['seed'])):
    w(f"| {x['arm']} | {x['knob']} | s{x['seed']} | {x['server']} | {x['epochs']}/310 | {f(x['train_loss'], 4)} | {f(x['train_acc'])} | {f(x['val_loss'], 4)} | {f(x['val_acc'])} | {fi(x['spikes'])} |")
w('')
w('**배제·미완주 (평균에서 뺌, 행은 남김):**')
w('')
excl_table(C10)
w('')
w('**큐 대기 (hold, 미시작):** ' + ', '.join(f"{kn(x['arm'], x['knob'])} s{x['seed']}" for x in C10['hold']) + '.')
w('')
w('이 요약에서 뺀 C10 팔 (원장에는 있음): ours_intra_ch (prop), ours_layer(_w1), ours_w1, st30/st60/end40, taylor, × intra, b4/b7, Suetake, 1-softmax, plain L2 + rho, 옛 intra ablation (abl_*).')
w('')
w('### 1.7 스파이크 측정 축: 원장 s_count 대 npz 총합')
w('')
fr = NP['fit_ratio_lnS']
at = NP['arm_terms']
w(f"npz = 테스트 1만 장을 평가 모드로 한 번 추론해 층별 발화 수를 저장한 파일 (입력·출력층 제외 합, 이미지당). {NP['cutoff']} 이전에 쓰인 npz 만 썼다 (같은 날 다른 추출이 진행 중). "
  f"이 문서의 C10 런 중 npz 가 있는 런 n={NP['n_runs']} (baseline {NP['ratio_by_arm']['baseline']['n']}, ours_inter_ch {NP['ratio_by_arm']['ours_inter_ch']['n']}, plain L2 {NP['ratio_by_arm']['plain L2']['n']}, BPSR {NP['ratio_by_arm']['BPSR']['n']}; ablation 팔 0). "
  f"npz 의 정확도는 원장 best val acc 와 최대 {NP['acc_match']['max_abs']:.2f}%p 차라 같은 체크포인트다.")
w('')
w('| 팔 | n | npz / s_count (평균 ±SD) |')
w('|---|---|---|')
for arm_ in ['baseline', 'ours_inter_ch', 'plain L2', 'BPSR']:
    r_ = NP['ratio_by_arm'][arm_]
    w(f"| {arm_} | {r_['n']} | {r_['mean']:.4f} ±{r_['sd']:.4f} |")
w('')
w(f"- **비는 팔보다 스파이크 수준에 따라 달라진다.** 비 = a + b·ln(s_count) 로 맞추면 b = {fr['b']:+.4f} ±{fr['b_se']:.4f} (n={fr['n']}, 잔차 SD {fr['resid_sd']:.4f}) — 스파이크가 줄수록 비가 커진다. "
  f"npz − s_count 는 런마다 {NP['diff']['min'] / 1e3:,.1f}K~{NP['diff']['max'] / 1e3:,.1f}K (평균 {NP['diff']['mean'] / 1e3:,.1f}K) 로, 대략 고정분 {NP['diff']['fit_a'] / 1e3:,.1f}K + s_count 의 {NP['diff']['fit_b'] * 100:.1f}% 꼴이다. "
  "팔마다 비가 1.05~1.09 로 달라 보인 것 (이전 분석) 은 대부분 팔마다 잰 스파이크 수준이 달라서다.")
w(f"- 같은 s_count 에서 팔 항 (기준 ours_inter_ch): " + ', '.join(f"{k} {v['c']:+.4f} ±{v['se']:.4f}" for k, v in at['terms'].items())
  + f" (F p={at['p']:.3f}; 95% 구간으로도 0.6% 안쪽). 서버 항 {NP['server_term']['c']:+.4f} ±{NP['server_term']['se']:.4f}. 팔 항의 최대 0.34% 를 정확도로 옮기면 같은 스파이크 비교 오프셋이 약 {NP['acc_offset_shift_bound']:.3f}%p 바뀐다 → **같은 스파이크 방법 비교의 결론에는 영향이 없다.**")
w('- **baseline 대비 감소율은 npz 축에서 1~2%p 작아진다** (낮은 스파이크일수록 비가 커서). npz 축 값 = 실측 npz 가 있으면 그 값, 없으면 s_count × 곡선 비 (예측). baseline npz 축 '
  f"{fi(NP['baseline']['npz_axis'])} (실측 {NP['baseline']['measured']}/{NP['baseline']['n']}).")
w('')
w('| 팔 | ρ/λ | n | npz 실측 런 | 스파이크 vs base (s_count) | 스파이크 vs base (npz 축) |')
w('|---|---|---|---|---|---|')
for c in NP['conds']:
    if c['n'] < 3 and c['arm'] not in ('ours_inter_ch',):
        continue
    w(f"| {c['arm']} | {c['knob']} | {c['n']} | {c['measured']}/{c['n']} | {pct(c['dspk_scount'])} | {pct(c['dspk_npz'])} |")
w('')
w('- 원인은 아직 모른다 (09-15 부터 보류). ablation 4팔은 npz 가 없어 위 곡선이 그 팔에도 맞는지 확인하지 못했다 — 끈 요소가 층별 발화 분포를 바꾸면 비가 다를 수 있다.')
w('')

# ---------------- 2. R19 C100
w('## 2. R19 CIFAR-100')
w('')
w(f"baseline 시드 범위 {C100['base']['acc_min']:.2f}~{C100['base']['acc_max']:.2f} (n={C100['base']['n']}). 합동 시드 SD {C100['pooled_sd']:.3f}%p (자유도 {C100['pooled_df']}). n=4 조건 대 baseline 의 최소 검출 차 {mdd_n4(C100):.2f}%p.")
w('')
setting_table(C100, ARMS_C100)
w('')
for arm in ['ours_inter_ch', 'plain L2', 'BPSR']:
    lossless_line(C100, arm, 'C100')
w('')
_l2c = [c for c in C100['conds'] if c['arm'] == 'plain L2' and c['n'] == 1 and 0.25 < -c['dspk'] < 0.42]
w(f"- **시드 범위 판정이 갈린 이유는 같은 스파이크 정확도 차이가 아니다** (아래 ANCOVA 오프셋 ≈ 0). ours_inter_ch 는 −31% 점 (3e-3, n=3) 이 baseline 최저에 {C100['base']['acc_min'] - c3['val_acc']:.3f}%p (3런 합 이미지 약 {img_txt(r3['margin_images'])}) 모자라 끊겼고, "
  f"plain L2 는 −43% 점 (5e-7, n=3, 전부 138) 이 통과했다. 그 사이 plain L2 점은 n=1 뿐이다 (" + ', '.join(f"{c['knob']} {c['val_acc']:.2f} ({pct(c['dspk'])})" for c in _l2c) + '). '
  '즉 이 판정은 어느 ρ 에 n≥3 을 줬는지와 이미지 몇 장 차에 좌우된다. 원인을 시드 운으로 단정하지는 않는다 (검증하지 않았다).')
w(f"- 같은 스파이크 비교는 ±0.15%p 안에서 구별 불가다. 스파이크 축으로는 같은 정확도에서 plain L2 {a100['offsets']['plain L2']['iso_ratio']:.2f}배 [{a100['offsets']['plain L2']['iso_lo']:.2f}, {a100['offsets']['plain L2']['iso_hi']:.2f}], "
  f"BPSR {a100['offsets']['BPSR']['iso_ratio']:.2f}배 [{a100['offsets']['BPSR']['iso_lo']:.2f}, {a100['offsets']['BPSR']['iso_hi']:.2f}]. 주 구간 220–420K 는 10-19 방법 분석 (method_261019.py) 에서 같은 날 정한 구간이다 (사전 등록 아님). "
  f"기울기 동질성 F p={REV['slope_homog']['C100_methods_220-420K']['p']:.2f}.")
w('')
w('**같은 스파이크 오프셋 (ANCOVA, 기준 ours_inter_ch).**')
w('')
anc_table(ANC['C100_methods'])
w('')
w('**배제·미완주:**')
w('')
excl_table(C100)
w('')
w('**큐 대기 (hold):** ' + ', '.join(f"{kn(x['arm'], x['knob'])} s{x['seed']}" for x in C100['hold']) + '. C100 BPSR 반복은 사용자 지시로 보류 (10-16).')
w('')

# ---------------- 3. DVS
w('## 3. VGGSNN CIFAR10-DVS — 첫 프레임 레시피 (vggdvsf0, 논문 채택 프로토콜)')
w('')
w('레시피: nda (프레임마다 독립 증강, CutMix 없음), 200에폭, T=4, 입력은 프레임 0 을 T 번 반복 (시간 정보를 쓰지 않는 정적 입력). '
  'baseline s1 은 같은 설정의 수동 패치 런 `_ours/vggdvs-base-f0-s1` (run_paper 구현과 바이트 동일, 10-15 확인).')
w('')
w(f"baseline 시드 범위 {DVS['base']['acc_min']:.2f}~{DVS['base']['acc_max']:.2f} (n={DVS['base']['n']}). 합동 시드 SD {DVS['pooled_sd']:.3f}%p (자유도 {DVS['pooled_df']}). 전부 138 서버. "
  f"검증셋은 992장이다 (런 {REV['dvs_n_val']['n_runs']}개의 정확도가 모두 k/992 와 맞음) — 이미지 1장 = 0.101%p.")
w('')
setting_table(DVS, ['baseline', 'ours_inter_ch'])
w('')
lossless_line(DVS, 'ours_inter_ch', 'DVS')
w('')
w(f"- **판정 가능한 범위**: 스파이크 {pct(max(c['dspk'] for c in _d4), 0)}~{pct(min(c['dspk'] for c in _d4), 0)} 의 n=4 조건 {len(_d4)}개 모두 baseline 과의 차가 검출되지 않는다 (Δ {pp(min(c['dacc_d'] for c in _d4))}~{pp(max(c['dacc_d'] for c in _d4))}). "
  f"n=4 대 4 의 최소 검출 차가 {_d4[0]['dacc_mdd']:.2f}%p 라 **1%p 미만의 차는 이 n 으로 판정 불가**다.")
w(f"- 시드 범위 판정은 {_dseq} 로 단조롭지 않다. 판정을 가르는 차: " + ', '.join(
    f"{k} {_dr[k]['margin']:+.4f}%p (4런 합 이미지 {img_txt(_dr[k]['margin_images'])} {'초과' if _dr[k]['margin'] > 0 else '부족'})" for k in [c['knob'] for c in _d4])
  + ' — 2e-3·1e-2 는 이미지 1장 미만 차로 떨어진다. 잡음이 판정을 좌우한다.')
w('- best 에폭을 같은 992장에서 고르므로 정확도가 위로 부풀려진다 (모든 팔 공통). 입력은 시간 정보가 없는 정적 입력이다. 전부 138 서버라 서버 혼입은 없다.')
w('')
w('**배제:**')
w('')
excl_table(DVS)
w('')
_x3 = next(x for x in DVS['excluded'] if x['knob'] == '3e-2')
w(f"- 3e-2 s1 은 원장 배제 규칙 (S30/S1<0.19) 에 걸렸다. 이 규칙은 R19 에서 정한 것이라 DVS 에 그대로 맞는지는 확인하지 않았다 (값 자체도 baseline 최저보다 {DVS['base']['acc_min'] - _x3['val_acc']:.2f}%p 낮음, n=1).")
w('')
w('**옛 DVS 레시피 (논문 표에서 내림, 참고만).**')
w('')
w('| 레시피 | 방법 | ρ | seed | 서버 | train loss | train acc | val loss | val acc | spikes | 상태 |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
for x in sorted(DVS_OLD, key=lambda x: (x['tag'].split('-')[0], x['arm'], kval(x['knob']), x['seed'])):
    if x['status'] in ('hold', 'todo') or (x['reason'] or '').startswith('연기시험'):
        continue
    w(f"| {x['tag'].split('-')[0]} | {x['arm']} | {x['knob']} | s{x['seed']} | {x['server']} | {f(x.get('train_loss'), 4)} | {f(x['train_acc'])} | {f(x.get('val_loss'), 4)} | {f(x['val_acc'])} | {fi(x['spikes'])} | {x['reason'] or '완주'} |")
w('')
w('- vggdvs = 3월 레시피 (nda) + 평균 프레임 입력 (74% 대, 과적합), vggdvscm = 11월 레시피 (클립 단위 증강 + CutMix) + 평균 프레임 (baseline 79.74, n=3), '
  'vggdvscmtd = vggdvscm + 단계별 BN (n=1), `(frame0)` = 수동 첫 프레임 패치. 이 레시피들의 ours_inter_ch 는 각 n=1 이라 판단 불가.')
w('')

# ---------------- 4. QK
w('## 4. QKFormer (CIFAR-10 / CIFAR-100, 410에폭, 재측정)')
w('')
w('val acc·spikes 는 model_best 체크포인트 fp32 재측정값 (spikes_*.json). train loss (과제 손실)·val loss 는 학습 로그 summary.csv 의 model_best 에폭 값 (AMP 학습 중 평가) — '
  '**같은 행에서 val loss 는 로그 값, val acc 는 재측정 값이다.** train acc 는 QKFormer 학습 로그에 기록되지 않음 (mixup/cutmix). "로그 best" 는 학습 중 평가 최고값 — 보고값 아님. '
  '재측정 정확도를 공개 QKFormer 수치나 학습 로그 값과 섞어 쓰지 않는다.')
w('')
for S, nm, rk in ((QK10, 'CIFAR-10', 'QK10'), (QK100, 'CIFAR-100', 'QK100')):
    w(f'### {nm}')
    w('')
    w(f"baseline 시드 범위 {S['base']['acc_min']:.2f}~{S['base']['acc_max']:.2f} (n={S['base']['n']}). 합동 시드 SD {S['pooled_sd']:.3f}%p (자유도 {S['pooled_df']}). n=4 조건 대 baseline 의 최소 검출 차 {mdd_n4(S):.2f}%p.")
    w('')
    w('| 방법 | ρ | seed | 서버 | best 에폭 | train loss | train acc | val loss | val acc (재측정) | spikes (재측정) | 로그 best | spikes vs base | Δacc vs base (±SE) | p |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for c in S['conds']:
        for x in c['runs']:
            w(f"| {c['arm']} | {c['knob']} | s{x['seed']} | {x['server']} | {x['best_epoch']} | {f(x['train_loss'], 4)} | — | {f(x['val_loss'], 4)} | {f(x['val_acc'])} | {fi(x['spikes'])} | {f(x['log_best'])} |  |  |  |")
        flag = '' if c['n'] >= 4 else ' ⚠n<4'
        if c['arm'] == 'baseline':
            w(f"| **baseline 평균** | | **n={c['n']}** | 138 | | {sd_(c['train_loss'], c['train_loss_sd'], 4)} | — | {sd_(c['val_loss'], c['val_loss_sd'], 4)} | **{sd_(c['val_acc'], c['val_acc_sd'])}** | **{fi(c['spikes'])}** | {f(c['log_best'])} |  |  |  |")
        else:
            w(f"| **{c['arm']} {c['knob']} 평균** | | **n={c['n']}**{flag} | 138 | | {sd_(c['train_loss'], c['train_loss_sd'], 4)} | — | {sd_(c['val_loss'], c['val_loss_sd'], 4)} | **{sd_(c['val_acc'], c['val_acc_sd'])}** | **{fi(c['spikes'])}** | {f(c['log_best'])} | {pct(c['dspk'])} | {pp(c['dacc_d'])} ±{c['dacc_se']:.2f} | {p_(c['dacc_p'])} |")
    w('')
    lossless_line(S, 'ours_inter_ch', rk)
    for x in S['excluded']:
        w(f"- 평균에서 뺀 런: {x['arm']} s{x['seed']} ({x['server']}) — {x['reason']}.")
    if S is QK100:
        w(f"- **C100 ρ 6e-3 의 p 는 검정 방식에 따라 {qc['pooled']['p']:.3f}~{qc['paired']['p']:.3f}**: 합동 SD (세 조건의 분산을 합침, 1.2e-2 의 좁은 분산이 섞임) p={qc['pooled']['p']:.3f}, "
          f"두 조건만 쓴 Welch p={qc['welch']['p']:.3f}, 같은 시드 짝 (" + ', '.join(f"s{x['seed']} {pp(x['d'])}" for x in qc['paired']['diffs']) + f") 짝 t 검정 p={qc['paired']['p']:.3f}. "
          f"학습 로그 최고값 축으로는 baseline {qc['log_axis']['base_mean']:.3f} (최저 {qc['log_axis']['base_min']:.2f}) 대 {qc['log_axis']['arm_mean']:.3f}, Δ {pp(qc['log_axis']['d'])} ±{qc['log_axis']['se']:.2f} (p={qc['log_axis']['p']:.3f}) 이지만 "
          f"시드 범위 기준으로는 {'통과' if qc['log_axis']['in_seed_range'] else '불통과'} → **'정확도 유지 n≥3 점이 없다' 는 측정 방식에 따라 달라진다.** ρ<6e-3 (−10.6% 보다 약한 규제) 은 시험하지 않았다.")
    w('')
qn1 = [c for c in QK10['conds'] if c['arm'] == 'ours_inter_ch' and c['n'] == 1]
w('- C10 n=1 사다리: ' + ', '.join(f"{c['knob']} {c['val_acc']:.2f} ({pct(c['dspk'])}, {'범위 안' if c['in_seed_range'] else '최저 아래'})" for c in qn1)
  + f" — baseline 시드 범위 {QK10['base']['acc_min']:.2f}~{QK10['base']['acc_max']:.2f}. n=1 이라 판정에 쓰지 않는다. ρ 2e-2 반복 3런 (s43~45) 은 기동 승인 대기.")
w('- 대기·중단 (qk.json): ' + ', '.join(f"{x['name']} ({x['status']}, {x['epochs']}에폭)" for x in QK_PENDING) + '.')
qrm = REV['qk_remeasure_minus_log']
w(f"- **재측정 − 로그 최고값** (n={qrm['n']} 런): 전체 범위 {qrm['min']['d']:+.2f}~{qrm['max']['d']:+.2f}%p, 양 끝 모두 baseline 런 ({qrm['min']['ds']} s{qrm['min']['seed']}, {qrm['max']['ds']} s{qrm['max']['seed']}). 조건별 평균: "
  + ', '.join(f"{x['ds']} {kn(x['arm'], x['knob'])} {x['mean']:+.3f}" + (f" ±{x['sd']:.2f}" if x['sd'] is not None else '') + f" (n={x['n']})" for x in qrm['by_cond'] if x['n'] >= 4)
  + ' — n=4 이상 조건끼리 평균 차가 비슷해 한 팔에만 불리한 편향은 보이지 않지만, 원인 모를 −0.2~−0.4%p 의 계통 차이고 런마다 흩어짐 (SD 약 0.07~0.17) 이 시드 SD 와 비슷하다.')
w('')
w('**참고 팔 (재측정 있음, 각 n=1~2, 이 요약의 판정 대상 아님).**')
w('')
w('| 방법 | ρ/λ | seed | val acc (재측정) | spikes | spikes vs C10 base |')
w('|---|---|---|---|---|---|')
for x in QK_REFR:
    if x.get('val_acc') is None:
        continue
    w(f"| {x['arm']} | {x['knob']} | s{x['seed']} | {f(x['val_acc'])} | {fi(x['spikes'])} | {pct(x['spikes'] / QK10['base']['spikes'] - 1)} |")
w('')

# ---------------- 5. SDT
nt = SDT_CTRL_NOTE
w('## 5. SDTv3 171M (1×8, SDT_D=8, T=1) ImageNet-1k 파인튜닝 — 23 서버')
w('')
w('공식 체크포인트 (171M-1x8) 에서 30에폭 파인튜닝. 비KD 런의 정확도는 head 하나 기준. **각 조건 n=1, ImageNet 시드 편차 미측정** → 정확도 유지 판정 불가, 아래는 서술만. '
  '값은 마지막 에폭 (원장 sdt.json, 학습 루프 안 평가). train loss 는 과제 손실 (규제항 제외). "마지막 5에폭" 은 log.txt 사본의 마지막 5에폭 평균 (한 에폭 값의 흔들림 확인용).')
w('')
w('| 조건 | 코드 경로 | 에폭 | train loss | train acc | val loss | val acc (마지막) | spikes (마지막) | Δacc vs 대조 | spikes vs 대조 | 마지막 5에폭 val acc | 마지막 5에폭 Δacc / spikes |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|')
for x in SDT_MAIN:
    nm = '대조 (규제 없음)' if x['knob'] == '-' else f"ours_inter_ch ρ {x['knob']}"
    d5 = '' if x['knob'] == '-' else f"{pp(x['dacc_last5'], 3)} / {pct(x['dspk_last5'])}"
    w(f"| {nm} | {x['code_path']} | {x['epochs']}/{x['total']} | {f(x['train_loss'], 4)} | {f(x['train_acc'])} | {f(x['val_loss'], 4)} | **{f(x['val_acc'], 3)}** | {fi(x['spikes'])} | "
      f"{pp(x.get('dacc'), 3)} | {pct(x.get('dspk'))} | {f(x['last5_acc'], 3)} | {d5} |")
w('')
w('- 서술 (n=1): ρ 0.05·0.1 은 대조와 정확도 차 0.03%p 안에서 스파이크 −3%·−6%. ρ 0.2 −0.19%p (−11.5%), ρ 0.3 −0.27%p (−15.9%), ρ 1.0 −1.87%p (−33%), ρ 3.0 은 스파이크도 덜 줄고 (−30%) 정확도 −3.8%p. '
  '마지막 5에폭 평균으로 바꿔도 같은 모양. 시드 편차를 모르므로 0.2%p 안팎 차이가 손실인지 노이즈인지 말할 수 없다.')
w(f"- 대조 런 자체가 공식 출발점 (head 기준 85.72) 대비 마지막 에폭 {pp(nt['d_last'])}%p, 마지막 20에폭 평균 ({nt['last20']:.2f}) {pp(nt['d_last20'])}%p (n=1). "
  f"**단 두 값은 평가 경로가 다르다**: 85.72 는 별도 평가 스크립트 (eval_headonly, 50,000장) 값이고, 파인튜닝 값은 학습 루프 안 평가 (test_acc1) 다. 학습 루프 안 에폭 1 값이 {nt['first_epoch']:.2f} 로 85.72 보다 높아서, "
  '이 −0.5~−0.6%p 에는 평가 경로 차이나 초기 상승이 섞여 있을 수 있다. 규제 런과 대조 런의 비교는 둘 다 학습 루프 안 평가라 이 문제가 없다.')
w('- 코드 경로: orig (a45daf6) 와 fast (92581b0) 는 같은 기울기 (smoke 확인, 10-13). ρ 0.05·0.1·0.2 만 fast.')
w('')
w('**KD 런 (진행 중, 진행 상황만).** 정확도는 (head + head_kd)/2 평균 출력 기준이라 비KD 런과 눈금이 다르다 (공식 ckpt: 평균 85.21 vs head 85.72). KD 런끼리만 비교한다.')
w('')
w('| 조건 | 에폭 | 예상 완주 | train loss | train acc | val loss | val acc (현재) | spikes (현재) | 같은 에폭 대조+KD 대비 (에폭: 값 vs 대조) | Δacc | spikes 차 |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
for x in SDT_KD:
    nm = '대조+KD' if x['dir'] == 'main_kd_ctrl' else f"ours_inter_ch ρ {x['knob'].replace('+KD', '')}+KD"
    se_ = x.get('same_epoch')
    tail = '|  |  |  |' if not se_ else f"| {se_['epoch']}: {se_['acc']:.3f} vs {se_['ctrl_acc']:.3f} | {pp(se_['dacc'], 3)} | {pct(se_['dspk'])} |"
    w(f"| {nm} | {x['epochs']}/{x['total']} | {x['eta']} | {f(x['train_loss'], 4)} | {f(x['train_acc'])} | {f(x['val_loss'], 4)} | {f(x['val_acc'], 3)} | {fi(x['spikes'])} {tail}")
w('')
w('- 학습 중간값이고 n=1 이다. 같은 에폭 비교는 그 에폭 한 점이라 흔들림이 크다. KD 정확도 유지 범위는 완주 (ρ 1.0+KD·대조+KD 10-20, ρ 0.3·0.5+KD 10-22 무렵) 후 판단.')
w('')

# ---------------- 6. Speck
w('## 6. Speck2f 칩 실측 (153 서버, CIFAR10-DVS 9층) — 실측 사례 (학습 n=1)')
w('')
w(f"출처: {SPECK['source']}. 설정: {SPECK['setting']}.")
w('')
w('| 모델 | 측정 | SW acc | 칩 acc | SW 스파이크/표본 | 칩 이벤트 | 동적 전력 (mW) | 에너지/표본 (mJ) | 표본당 측정 시간 (s) |')
w('|---|---|---|---|---|---|---|---|---|')
for m in SPECK['measurements']:
    w(f"| {m['model']} | {m['run']} | {f(m.get('sw_acc'), 1)} | {f(m.get('hw_acc'), 1)} | {fi(m.get('sw_spikes'))} | {fi(m.get('chip_events'))} | {m['mW']:.3f} | {m['mJ']:.3f} | {f(m.get('s_per_sample'))} |")
w('')
w(f"- baseline 측정 {sp_['baseline_k']}회 평균 {sp_['baseline_mW']:.3f} mW / {sp_['baseline_mJ']:.3f} mJ (best 2회 + epoch_060 2회, 서로 다른 체크포인트 둘을 합침), ours_inter_ch ρ 1.0 best {sp_['inter_k']}회 평균 {sp_['inter_best_mW']:.3f} mW / {sp_['inter_best_mJ']:.3f} mJ "
  f"→ 전력 {pct(sp_['dmW'])}, 에너지 {pct(sp_['dmJ'])}, SW 스파이크 {pct(sp_['sw_spike_change_best'])}.")
w(f"- **같은 체크포인트 종류끼리**: best 끼리 (baseline {spr['best_vs_best']['baseline_mW']:.3f} mW, 2회) 전력 {pct(spr['best_vs_best']['dmW'])} / 에너지 {pct(spr['best_vs_best']['dmJ'])}; "
  f"**epoch_060 끼리 (baseline {spr['ep60_vs_ep60']['baseline_mW']:.3f} mW 2회 대 ρ 1.0 1회) 전력 {pct(spr['ep60_vs_ep60']['dmW'])}** / 에너지 {pct(spr['ep60_vs_ep60']['dmJ'])}. 보수적인 같은 조건 전력 값은 {pct(spr['ep60_vs_ep60']['dmW'])} 이다.")
e60 = sp_['ep60']
w(f"- 같은 에폭 60 끼리 칩 이벤트: ρ 1.0 {pct(e60['events_rho1'])}. ρ 0.3 (epoch_060, 1회) 은 전력 {pct(e60['d_rho03'])}, 칩 이벤트 {pct(e60['events_rho03'])}.")
w('- inter best 의 SW 정확도는 baseline best 보다 −0.6%p (64.7 → 64.1, n=1). 원본 측정 파일은 이 분석에서 다시 읽지 않았다 (decisions.md 에서 옮겨 적음).')
for note in SPECK['notes']:
    w(f'- {note}')
w('')

# ---------------- 7. 종합
w('## 7. 설정별 정확도 유지 판정 한눈에')
w('')
w('"시드 범위 마지막 점" 은 무손실 한계가 아니다 (정의 참고). 최소 검출 차 = n=4 조건 대 그 설정 baseline.')
w('')
w('| 설정 | baseline (n) | 시드 범위 | 합동 SD | 최소 검출 차 (n=4) | ours_inter_ch 시드 범위 마지막 점 (n) | 연속 구간 | ours_inter_ch 검정 기준 (보정 없음, 첫 검출 전까지) | ours_inter_ch ±0.2%p 동등성 통과 최대 | 다른 방법 시드 범위 마지막 점 (n) | 비고 |')
w('|---|---|---|---|---|---|---|---|---|---|---|')


def llstr(ll, key='max'):
    m = ll.get(key) if ll else None
    return '없음' if not m else f"{m['knob']} {pct(m['dspk'])} (n={m['n']})"


def tstr(ll):
    if not ll['test_first_sig']:
        return '검출 없음 (시험한 n≥3 전부)'
    tc = ll['test_contiguous']
    return (f"{tc['knob']} {pct(tc['dspk'])}" if tc else '없음') + f" (첫 검출 {ll['test_first_sig']['knob']} {pct(ll['test_first_sig']['dspk'])})"


def eqstr(ll):
    t = ll.get('tost_max')
    return '없음' if not t else f"{t['knob']} {pct(t['dspk'])} (n={t['n']})"


li = QK10['lossless']['ours_inter_ch']
lj = QK100['lossless']['ours_inter_ch']
w(f"| R19 C10 | {C10['base']['acc']:.2f} @ {fi(C10['base']['spikes'])} (6) | {C10['base']['acc_min']:.2f}~{C10['base']['acc_max']:.2f} | {C10['pooled_sd']:.3f} | {mdd_n4(C10):.2f} | {llstr(cl10['ours_inter_ch'])} — 런 하나 빼기로 뒤집힘 ({'·'.join('s' + str(x) for x in RB['C10']['ours_inter_ch']['max']['loo_flip_seeds'])}) | {llstr(cl10['ours_inter_ch'], 'contiguous')} | {tstr(cl10['ours_inter_ch'])} | {eqstr(cl10['ours_inter_ch'])} | "
  f"plain L2 {llstr(cl10['plain L2'])} (n=4 규칙이면 {llstr(cap_l2)}); BPSR {llstr(cl10['BPSR'])}{' (기준과 동점)' if RB['C10']['BPSR']['max']['tie'] else ''} | 진행 중 6런 |")
w(f"| R19 C100 | {C100['base']['acc']:.2f} @ {fi(C100['base']['spikes'])} (6) | {C100['base']['acc_min']:.2f}~{C100['base']['acc_max']:.2f} | {C100['pooled_sd']:.3f} | {mdd_n4(C100):.2f} | {llstr(c1['ours_inter_ch'])} | {llstr(c1['ours_inter_ch'], 'contiguous')} | {tstr(c1['ours_inter_ch'])} | {eqstr(c1['ours_inter_ch'])} | "
  f"plain L2 {llstr(c1['plain L2'])}; BPSR 없음 | 3e-3 은 3런 합 이미지 {img_txt(RB['C100']['ours_inter_ch']['first_fail']['margin_images'])} 부족으로 불통과 |")
w(f"| VGGSNN DVS 첫 프레임 | {DVS['base']['acc']:.2f} @ {fi(DVS['base']['spikes'])} (4) | {DVS['base']['acc_min']:.2f}~{DVS['base']['acc_max']:.2f} | {DVS['pooled_sd']:.3f} | {mdd_n4(DVS):.2f} | {llstr(DVS['lossless']['ours_inter_ch'])} | {llstr(DVS['lossless']['ours_inter_ch'], 'contiguous')} | {tstr(DVS['lossless']['ours_inter_ch'])} | {eqstr(DVS['lossless']['ours_inter_ch'])} | (비교 방법 없음) | 1%p 미만 차는 판정 불가, 판정이 이미지 1장 미만 차로 갈림 |")
w(f"| QKFormer C10 | {QK10['base']['acc']:.2f} @ {fi(QK10['base']['spikes'])} (6) | {QK10['base']['acc_min']:.2f}~{QK10['base']['acc_max']:.2f} | {QK10['pooled_sd']:.3f} | {mdd_n4(QK10):.2f} | {llstr(li)} | {llstr(li, 'contiguous')} | {tstr(li)} | {eqstr(li)} | (n≥3 비교 방법 없음) | ρ 2e-2 n=1 −44% |")
w(f"| QKFormer C100 | {QK100['base']['acc']:.2f} @ {fi(QK100['base']['spikes'])} (4) | {QK100['base']['acc_min']:.2f}~{QK100['base']['acc_max']:.2f} | {QK100['pooled_sd']:.3f} | {mdd_n4(QK100):.2f} | {llstr(lj)} | {llstr(lj, 'contiguous')} | {tstr(lj)} (p 0.01~0.07, 검정 방식에 따라) | {eqstr(lj)} | | 로그 최고값 축이면 6e-3 시드 범위 통과; ρ<6e-3 미시험 |")
w(f"| SDTv3 ImageNet | 대조 {CTRL['val_last']:.2f} @ {fi(CTRL['spikes'])} (1) | 미측정 | 미측정 | | 판단 불가 (n=1) | | | | | KD 4런 진행 중 |")
w(f"| Speck2f 칩 | 칩 acc 63.0, {sp_['baseline_mW']:.3f} mW (학습 1) | 미측정 | | | 판단 불가 (n=1); ρ 1.0 SW 스파이크 {pct(sp_['sw_spike_change_best'])}, 칩 전력 {pct(sp_['dmW'])} (epoch_060 끼리 {pct(spr['ep60_vs_ep60']['dmW'])}) | | | | | 실측 사례 |")
w('')

# ---------------- LIMITATION
x55 = cond(C10, 'ours_inter_ch', '5.5e-3')
w('## [LIMITATION]')
w('')
w('- **이 문서의 수치는 방법의 우위나 새로움을 뒷받침하지 않는다.** R19 C10 에서 −47~−52% 를 시드 범위 안에서 줄이는 것은 plain L2·BPSR 도 같다. '
  '문헌 대비 위치 (eip-critic 이 든 예: 2026 "The Sparsity Ceiling", ICLR 2024 Shi 등) 는 이 문서에서 확인하지 않았다 — refs-audit 소관.')
w(f"- **시드 범위 기준이 느슨하고 경계점이 취약하다.** 평균이 baseline 최저 시드와 같으면 통과한다 (예: R19 C10 ours_inter_ch 5.5e-3 은 통과하지만 Δ {pp(x55['dacc_d'])} ±{x55['dacc_se']:.2f}, p{peq(x55['dacc_p'])}, n=3, s2 를 빼면 불통과). "
  '검정 기준은 반대로 "검출되지 않음" 을 "같음" 으로 읽게 만들고, 보정 없는 순차 검정이라 위양성도 있다 (plain L2 1e-7). ±0.2%p 동등성 검정 (TOST) 을 참고로 붙였지만 마진은 합의된 값이 아니다. 어느 기준을 논문에 쓸지는 정하지 않았다.')
w('- **시드 범위 마지막 점으로 방법 간 우열을 말할 수 없다.** R19 C10·C100 에서 plain L2·BPSR 의 시드 범위 마지막 점이 ours_inter_ch 와 같거나 더 크고, 같은 스파이크 ANCOVA 는 모두 0 을 포함한다. '
  '그렇다고 "우리 방법이 못하다" 의 근거도 아니다 — 현재 n 으로는 약 0.1~0.2%p (스파이크로는 ±20% 안팎) 보다 작은 차이를 볼 수 없다.')
w(f"- **스파이크 축 (s_count 대 npz).** 절대 스파이크 수와 감소율은 원장 s_count 축이다. npz 총합은 s_count 보다 런마다 {NP['diff']['min'] / 1e3:,.0f}K~{NP['diff']['max'] / 1e3:,.0f}K 많고 (비 1.05~1.13, 스파이크가 적을수록 큼), "
  f"그래서 baseline 대비 감소율이 npz 축에서 1~2%p 작다 (예: ours_inter_ch 3e-3 {pct(npc[('ours_inter_ch', '3e-3')]['dspk_scount'])} → {pct(npc[('ours_inter_ch', '3e-3')]['dspk_npz'])}, 5.5e-3 {pct(npc[('ours_inter_ch', '5.5e-3')]['dspk_scount'])} → {pct(npc[('ours_inter_ch', '5.5e-3')]['dspk_npz'])}, "
  f"plain L2 2.5e-7 {pct(npc[('plain L2', '2.5e-7')]['dspk_scount'])} → {pct(npc[('plain L2', '2.5e-7')]['dspk_npz'])}, BPSR 3e-10 {pct(npc[('BPSR', '3e-10')]['dspk_scount'])} → {pct(npc[('BPSR', '3e-10')]['dspk_npz'])}). "
  '불일치 원인은 미해결이다. 같은 s_count 에서 측정한 팔 사이의 비 차는 약 0.3% (95% 구간으로도 0.6% 안쪽) 라 같은 스파이크 비교에는 영향이 없지만, ablation 4팔은 npz 가 없어 확인하지 못했다.')
w('- ANCOVA 는 ln(spikes) 직선 + 서버 항 + 팔 공통 오프셋을 가정한다. - final_step 은 두 ρ 의 오프셋이 달라 (p=0.012) 이 가정이 깨진다. 구간 경계에 따라 포함 런이 바뀌며, C10 200–420K 는 이전 분석과 맞춘 것, C100 220–420K 는 같은 날 정한 것이고 둘 다 사전 등록된 구간이 아니다. ablation 4팔은 Holm 보정을 같이 적었다.')
w('- R19 C10 ours_inter_ch 1.5e-3·2e-3·4e-3 (n=3)·5e-3 (n=1) 은 진행 중인 런이 끝나면 평균이 바뀐다. 이 문서의 n 은 10-19 16:38 고정 사본 기준.')
w('- VGGSNN DVS 는 시드 SD 가 커서 n=4 대 4 로도 약 1%p 보다 작은 차이를 볼 수 없고, 시드 범위 판정은 이미지 1장 미만 차로 갈린다. 첫 프레임 입력은 시간 정보를 쓰지 않는 정적 입력이다.')
w(f"- QKFormer 정확도는 fp32 재측정값이고 학습 로그 최고값과 전체 범위 {qrm['min']['d']:+.2f}~{qrm['max']['d']:+.2f}%p 다르다 (n={qrm['n']} 런, 양 끝 모두 baseline 런; 조건별 평균은 4절). "
  '재측정 차이의 원인은 확인되지 않았다. baseline s44 (23) 는 재측정이 없어 뺐다. QK val loss 는 재측정이 아니라 학습 중 평가 값이다. QK C100 6e-3 의 판정은 측정 축 (재측정/로그) 과 검정 방식에 따라 달라진다.')
w('- SDTv3·Speck 은 조건당 학습 1회다. 칩 측정 반복 (r1/r2) 은 측정 잡음만 줄이고 학습 시드 편차는 반영하지 않는다. Speck baseline 쪽은 서로 다른 체크포인트 둘을 합쳤다. '
  'Speck 수치는 decisions.md 에서 옮긴 것이고 153 서버의 원본 측정 파일은 이 분석에서 다시 읽지 않았다. SDTv3 대조 런의 "공식 출발점 대비" 차는 평가 경로가 다른 두 값의 비교다.')
w('- 정확도는 val=test 에서 고른 best 에폭이라 모든 팔이 위로 부풀려진다 (이전 분석에서 R19 C10 약 0.15~0.18%p, 팔 간 같음).')
w('- R19 train loss 는 Keras 로그의 loss (규제항 포함 여부는 이 문서에서 확인하지 않음), QK·SDT train loss 는 과제 손실이다. 설정 간 train loss 를 비교하지 말 것.')
w('- 진행 중 런의 값은 그 시점까지의 best 이며 결과가 아니다. SDTv3 KD 의 같은 에폭 비교도 중간값이다.')
w('')

# ---------------- 검토 반영
w('## 검토 반영 (10-19 eip-critic, 판정 needs_fixes)')
w('')
w('검토자의 수치 재계산은 전부 이 문서와 일치했다. 지적 15건을 아래처럼 처리했다. 기존 수치 (원장 s_count 축 평균·Δ·p·ANCOVA) 는 바뀌지 않았고, 옆에 붙인 계산은 json 의 `review_261019` 에 있다.')
w('')
RV = [
    ('[major] npz/s_count 축', '반영 + 일부 정정',
     f"1.7 절 신설, [LIMITATION]·정의에 '원장 s_count 축' 명시. npz 를 직접 다시 계산했다 (n={NP['n_runs']} 런). 검토의 '팔마다 비가 다르다 (1.053~1.090)' 는 대부분 스파이크 수준 차이였다: "
     f"비 = a + b·ln(s) 의 b={fr['b']:+.4f}, 같은 s_count 에서 팔 항 약 0.3% (95% 구간 0.6% 안쪽, F p={at['p']:.3f}). 그래서 '342K 스파이크 차가 비의 불확실성 (최대 약 6%) 보다 작다' 는 측정한 팔에 대해서는 과장이다. "
     "npz 축 감소율은 곡선으로 다시 계산 (검토 값과 0.4%p 안). ablation 4팔은 npz 가 없어 예측값으로만 적고 미확인이라고 표시 (같은 날 다른 추출이 진행 중, 이 문서는 17:10 이전 파일만 씀)."),
    ('[major] 세 \'무손실 한계\'', '반영',
     "'무손실 한계' 를 '평균이 baseline 최저 시드 이상인 마지막 n≥3 점' 으로 바꾸고, 기준과의 차 (이미지 수)·동점·시드 하나 빼기로 뒤집히는 시드를 모든 판정 옆에 붙였다. "
     "plain L2 를 10-17 n=4 규칙 (s1~s4) 으로 다시 판정 (2.5e-7 불통과 → 마지막 점 1.5e-7 −39.9%). ±0.2%p TOST 를 참고 기준으로 추가 (세 경계점 모두 불통과). 경계 쌍의 ρ·서버 겹침과 ours_inter_ch 안 서버 항을 적었다."),
    ('[minor] ANCOVA 스파이크 축 구간', '반영', "모든 ANCOVA 표에 '같은 정확도 스파이크 비 [95%]' 열을 넣고 (델타법), 요약 3·5 에 BPSR 0.94 [0.76, 1.17] 등을 적었다. 기울기 동질성 F 검정 추가. '같은 성능·동등' 표현 없음."),
    ('[minor] BPSR 3e-10 부동소수', '반영', "'기준과 동점 (차 0.000) — ≥ 규칙으로만 통과, 판정 근거로 인용하지 않음' 으로 표시 (동점 판정을 일반 규칙으로 만듦)."),
    ('[minor] ablation 보정 전 p', '반영', "보정 전 p<0.05 를 전부 나열 (8건, p 를 넷째 자리까지: - vmem 342K p=0.0498). - vmem 의 서버 쏠림 (팔×서버 모형), - vmem 런 하나 빼기 9번, 기준점 3e-3 시드 하나 빼기를 계산해 적었다. '기여 검출 안 됨' 유지."),
    ('[minor] - final_step 두 ρ', '반영', "ρ 별 오프셋 ANCOVA 로 1.4e-3 −0.152 (p=0.008), 2.2e-3 +0.052 (p=0.39), 차 p=0.012 를 적고 '반대 방향' 표현을 '2.2e-3 은 곡선과 구별 안 됨' 으로 고침. 팔 공통 오프셋은 해석하지 않는다고 명시."),
    ('[minor] C100 \'시드 운\'', '반영', "검증하지 않은 인과 서술을 지우고 'ρ 에 n≥3 을 줬는지와 이미지 몇 장 차에 좌우, 같은 스파이크 ±0.15%p (스파이크 0.79~1.26배) 안에서 구별 불가' 로 바꿈. 220–420K 가 같은 날 정한 구간임을 적음."),
    ('[minor] DVS', '반영', "검증셋 992장 (정확도 값 간격으로 확인), 판정 순서 (통과/불통과 반복), 이미지 1장 미만 차, best 에폭 선택 편향을 적고 '−60.6% 무손실' 단독 인용을 금지."),
    ('[minor] QK C100', '반영', "합동 SD·Welch·짝 t 검정 p (0.010~0.068) 와 로그 최고값 축 결과 (시드 범위 통과) 를 적고, ρ<6e-3 미시험을 명시."),
    ('[minor] QK 재측정 − 로그', '반영', "조건별 평균 차를 적고 [LIMITATION] 문구를 '전체 범위, 양 끝 모두 baseline 런' 으로 고침. 같은 행의 val loss 는 로그 값임을 표 설명에 적음."),
    ('[minor] SDTv3 대조', '반영', "85.72 (eval_headonly 50,000장) 와 학습 루프 안 평가의 경로 차이, 에폭 1 값 85.92 를 적음."),
    ('[minor] Speck', '반영', "best 끼리 −34.8%, epoch_060 끼리 −28.2% (보수적 값) 를 헤드라인 옆에 나란히 적고 SW 정확도 −0.6%p 를 적음. 원본 측정 파일 재확인은 하지 않음 (153 서버 접근 범위 밖)."),
    ('[minor] 최소 검출 차', '반영', "모든 정확도 유지 판정 문장과 7절 표에 최소 검출 차를 붙임 (R19 C10 n=4 0.19, QK C10 0.31 등)."),
    ('[minor] 표 규약', '반영', "표1·342K 표에 서버 (런 수) 열 추가, 옛 DVS 레시피 표를 고정 열 순서로 바꿈 (train loss·val loss 추가), 검정 기준을 '보정 없는 순차 검정' 으로 표기하고 첫 검출 뒤 비검출 조건이 있으면 위양성 가능 표시 (plain L2 1e-7)."),
    ('[minor] 경쟁 문헌', '일부 반영', "요약 머리에 '고유 우위 검출 안 됨' 을 먼저 적고 칩·DVS 를 '실측 사례' 로 표시. 문헌 인용 (Sparsity Ceiling, Shi 등) 은 이 문서에서 검증하지 않아 근거로 넣지 않고 refs-audit 확인 대상으로만 적음."),
]
w('| 지적 | 처리 | 바꾼 것 / 이유 |')
w('|---|---|---|')
for a_, b_, c_ in RV:
    w(f'| {a_} | {b_} | {c_} |')
w('')

open(OUT_MD, 'w').write('\n'.join(L) + '\n')
print('wrote', OUT_JSON, OUT_MD)
