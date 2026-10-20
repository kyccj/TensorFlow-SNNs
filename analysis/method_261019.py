"""ours_inter_ch 방법 분석 갱신 (26-10-19, eip-stats).

10-14 판 (docs/paper/inter_justification_261014.md, analysis/inter_vs_bpsr_261009.md,
struct_compare_261007.md, inter_results_261009.md) 을 10-14 이후 도착한 자료로 다시 계산한다.

질문
  Q1. 설정별 정확도-스파이크 곡선: ours_inter_ch 가 baseline 정확도를 어디까지 지키며 스파이크를
      얼마나 줄이나 (R19 C10/C100, VGGSNN 첫 프레임, QK C10/C100, SDTv3).
  Q2. 같은 스파이크 비교: plain L2·BPSR 대비 (R19 C10·C100, QK C10 은 plain L2 만).
      ANCOVA acc = a + b·ln(spikes) + c·[팔] + d·[서버 23] 의 c, 같은 정확도 스파이크 비 exp(−c/b)
      (부트스트랩), 최소 검출 차 (α=0.05 양측, 검정력 0.8).
  Q3. 구조: 죽은 채널, MACs 제거, 이미지당 활성 뉴런, 단계별 배분, 집중도 (지니).
      npz (테스트 모드, R19 C10) + reg_neuron_detail.csv / reg_detail.csv (모든 런, C10·C100).
  Q4. 10-14 결론 대비 변화 → 주장 계층 (md 에서).
  (덧붙임) 구성요소 (ablation) 4팔: 같은 스파이크 정확도와 CSV 구조 지표 — 정밀 기전은 별도 분석.

입력 (읽기만)
  원장 스냅숏 an1019/snapshot/inventory.json (10-19 16:38) + sdt.json. 런 선택·n 은 여기 따른다.
  QK 재측정 ~/runs/qkformer_wta_rev/spikes_*.json (fp32 model_best), 손실은 각 런 summary.csv.
  npz _npz/, scratchpad/deep1005/npz/, an1019/npz/ (10-19 추출 4런: 표1 ours_ch 3e-3 s4,
      bpsr 1e-10 s4, l2 1e-7 s3·s4).
  CSV 138 = /media/hdd1/kyccj/EIP/paper/<tag>/ (없으면 저장소 루트 <tag>/), 23 = an1019/csv23/<tag>/
      (세종 /srv2/kyccj/EIP/paper 에서 읽기 전용 rsync).
  MACs 층 사상 scratchpad/prune_out/*.json.

정의
  정확도 = best val acc, 스파이크 = 그 에폭 s_count (원장). SDTv3 는 파인튜닝이라 best 가 에폭 1 근처라서
  마지막 에폭 값 (대조 런 마지막 에폭과 비교, decisions 10-17 과 같은 기준). QK 는 fp32 재측정.
  유효 런 = 완주 (R19 310, VGG 200 에폭), 배제 아님 (S30/S1 ≥ 0.19), 시드 < 900 (연기시험 제외).

2판 (10-19, eip-critic 검토 반영): REV 블록 ('review' 키) 추가 — s_count 측정 축 (학습 배치 통계) 확인과
  테스트 npz 축 %, 비열등 판정 기준별 (합동 SD·마지막 10에폭·마지막 에폭), Holm 보정, - vmem 서버 상호작용·4팔 모형,
  - 1- inversion 4팔 구조 모형, 활성 뉴런 팔별 기울기, 뉴런 희소성 위치, VGGSNN 추세. 난수를 쓰지 않아 1판 수치는 그대로.

쓰는 법:  ~/anaconda3/envs/venv_1/bin/python analysis/method_261019.py
산출:     analysis/method_261019.json, analysis/method_261019.md
"""
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
S = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
A = S + '/an1019'
INV = A + '/snapshot/inventory.json'
SDT = A + '/snapshot/sdt.json'
QKD = os.path.expanduser('~/runs/qkformer_wta_rev')
NPZ_DIRS = [ROOT + '/_npz', S + '/deep1005/npz', A + '/npz']
PRUNE_DIRS = [S + '/prune_out', S + '/deep1005/prune']
HDD = '/media/hdd1/kyccj/EIP/paper'
CSV23 = A + '/csv23'
OUT_JSON = ROOT + '/analysis/method_261019.json'
SKIP_L = {'n_in', 'predictions_n'}
STAGES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1']
NB = 10000
RNG = np.random.default_rng(20261019)

KEYNAME = {'base': 'baseline', 'ours_ch': 'ours_inter_ch', 'l2': 'plain L2', 'bpsr': 'BPSR',
           'ch_novmem': '- vmem', 'ch_nofinal': '- final_step', 'ch_noinv': '- 1- inversion',
           'ch_nolr': '- loss-ratio'}
R19_KEYS = list(KEYNAME)


# ---------------------------------------------------------------- 통계 도구
def msd(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if v.size > 1 else float('nan')), int(v.size)


def welch(a, b, sd_pool=None, conf=(0.90, 0.95)):
    """a − b. n=1 쪽은 sd_pool 로 SE 를 메운다 (자유도는 합동 자유도를 sd_pool[1] 로)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ma, mb = a.mean(), b.mean()
    va = a.var(ddof=1) if a.size > 1 else (sd_pool[0] ** 2 if sd_pool else np.nan)
    vb = b.var(ddof=1) if b.size > 1 else (sd_pool[0] ** 2 if sd_pool else np.nan)
    se = math.sqrt(va / a.size + vb / b.size)
    if a.size > 1 and b.size > 1:
        df = (va / a.size + vb / b.size) ** 2 / ((va / a.size) ** 2 / (a.size - 1) + (vb / b.size) ** 2 / (b.size - 1))
    else:
        df = sd_pool[1] if sd_pool else np.nan
    d = ma - mb
    out = {'d': float(d), 'se': float(se), 'df': float(df), 'p': float(2 * stats.t.sf(abs(d / se), df))}
    for c in conf:
        t = stats.t.ppf(0.5 + c / 2, df)
        out[f'ci{int(c * 100)}'] = [float(d - t * se), float(d + t * se)]
    out['mdd80'] = float((stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * se)
    return out


def pooled_t(a, b, sd, df, conf=(0.90, 0.95)):
    """a − b, SE 를 합동 시드 SD 로 (검토 반영 10-19). 자유도 = 합동 자유도."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = float(a.mean() - b.mean()); se = float(sd * math.sqrt(1 / a.size + 1 / b.size))
    out = {'d': d, 'se': se, 'df': float(df), 'p': float(2 * stats.t.sf(abs(d / se), df))}
    for c in conf:
        t = stats.t.ppf(0.5 + c / 2, df)
        out[f'ci{int(c * 100)}'] = [d - t * se, d + t * se]
    return out


def holm(pmap):
    """Holm 단계 보정 (단조성 강제). pmap: {이름: p}."""
    order = sorted(pmap, key=pmap.get); m = len(order); adj = {}; prev = 0.0
    for i, k in enumerate(order):
        prev = max(prev, min(1.0, (m - i) * pmap[k])); adj[k] = prev
    return adj


def ols(y, X):
    y = np.asarray(y, float); X = np.asarray(X, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    dof = len(y) - X.shape[1]
    s2 = float(res @ res / dof) if dof > 0 else float('nan')
    cov = s2 * np.linalg.pinv(X.T @ X)
    return b, cov, dof, s2


def design(rows, arms, ref, xkey, server=True, quad=False, x0=None, slopes=False):
    """rows: dict(arm, x, server). 열: 1, x−x0, [x²], 팔 더미(ref 제외), [서버 23], [팔×x]."""
    x = np.log([r[xkey] for r in rows])
    if x0 is None:
        x0 = float(x.mean())
    xc = x - x0
    cols = [np.ones_like(xc), xc]
    names = ['a', 'b']
    if quad:
        cols.append(xc ** 2); names.append('b2')
    for a in arms:
        if a == ref:
            continue
        cols.append(np.array([r['arm'] == a for r in rows], float)); names.append('arm:' + a)
    srv = np.array([r['server'] == '23' for r in rows], float)
    if server and 0 < srv.sum() < len(rows):
        cols.append(srv); names.append('srv23')
    if slopes:
        for a in arms:
            if a == ref:
                continue
            cols.append(xc * np.array([r['arm'] == a for r in rows], float)); names.append('bx:' + a)
    return np.column_stack(cols), names, x0


def ancova(rows, ycol, arms, ref, xkey='spikes', server=True, quad=False, logy=False, boot=True):
    """ref = 대상 팔. c[o] = ref − o (같은 x). 같은 정확도 스파이크 비 ref/o = exp(−c/b)."""
    rr = [r for r in rows if r.get(ycol) is not None and r.get(xkey) and r['arm'] in arms]
    if len(rr) < len(arms) + 3 or len({r['arm'] for r in rr}) < 2:
        return None
    y = np.array([r[ycol] for r in rr], float)
    if logy:
        y = np.log(y)
    X, names, x0 = design(rr, arms, ref, xkey, server, quad)
    b, cov, dof, s2 = ols(y, X)
    if dof < 1:
        return None
    tc = stats.t.ppf(0.975, dof)
    out = {'n': len(rr), 'by_arm': {a: sum(r['arm'] == a for r in rr) for a in arms}, 'dof': dof,
           'resid_sd': math.sqrt(s2), 'b': float(b[1]), 'b_se': float(math.sqrt(cov[1, 1])),
           'b_p': float(2 * stats.t.sf(abs(b[1] / math.sqrt(cov[1, 1])), dof)),
           'server_in_model': 'srv23' in names, 'x_range': [float(min(r[xkey] for r in rr)), float(max(r[xkey] for r in rr))],
           'contrast': {}}
    if 'srv23' in names:
        j = names.index('srv23'); out['srv23'] = float(b[j]); out['srv23_se'] = float(math.sqrt(cov[j, j]))
    # 기울기 차이 F (팔×x)
    Xs, ns_, _ = design(rr, arms, ref, xkey, server, quad, x0=x0, slopes=True)
    bs, covs, dofs, s2s = ols(y, Xs)
    if dofs > 0:
        rss0 = s2 * dof; rss1 = s2s * dofs; q = Xs.shape[1] - X.shape[1]
        F = ((rss0 - rss1) / q) / (rss1 / dofs) if q > 0 and rss1 > 0 else float('nan')
        out['slope_diff_F_p'] = float(stats.f.sf(F, q, dofs)) if q > 0 else float('nan')
    for o in arms:
        if o == ref:
            continue
        j = names.index('arm:' + o)
        c = -float(b[j]); se = float(math.sqrt(cov[j, j]))
        e = {'c': c, 'se': se, 'ci95': [c - tc * se, c + tc * se], 'p': float(2 * stats.t.sf(abs(c / se), dof)),
             'mdd80': float((stats.t.ppf(0.975, dof) + stats.t.ppf(0.80, dof)) * se),
             'n_ref': out['by_arm'][ref], 'n_o': out['by_arm'][o]}
        if logy:
            e['pct'] = 100 * (math.exp(c) - 1)
            e['pct_ci95'] = [100 * (math.exp(e['ci95'][0]) - 1), 100 * (math.exp(e['ci95'][1]) - 1)]
        if not logy and b[1] > 0:
            g = -c / b[1]
            e['spk_ratio'] = math.exp(g)
        out['contrast'][o] = e
    if boot and not logy:
        bs_c = defaultdict(list); bs_r = defaultdict(list); bad = defaultdict(int)
        idx_by = {a: [i for i, r in enumerate(rr) if r['arm'] == a] for a in arms}
        for _ in range(NB):
            sel = []
            for a in arms:
                ii = idx_by[a]
                if ii:
                    sel += list(RNG.choice(ii, size=len(ii), replace=True))
            rs = [rr[i] for i in sel]
            if len({r['server'] for r in rs}) < 2 and out['server_in_model']:
                Xb, nb_, _ = design(rs, arms, ref, xkey, False, quad, x0=x0)
            else:
                Xb, nb_, _ = design(rs, arms, ref, xkey, server, quad, x0=x0)
            yb = np.array([r[ycol] for r in rs], float)
            try:
                bb, _, dfb, _ = ols(yb, Xb)
            except np.linalg.LinAlgError:
                continue
            for o in arms:
                if o == ref or 'arm:' + o not in nb_:
                    continue
                c = -bb[nb_.index('arm:' + o)]
                bs_c[o].append(c)
                if bb[1] > 0:
                    bs_r[o].append(math.exp(-c / bb[1]))
                else:
                    bad[o] += 1
        for o in out['contrast']:
            if bs_c[o]:
                out['contrast'][o]['boot_c_ci95'] = [float(np.percentile(bs_c[o], 2.5)), float(np.percentile(bs_c[o], 97.5))]
            if bs_r[o]:
                out['contrast'][o]['boot_ratio_ci95'] = [float(np.percentile(bs_r[o], 2.5)), float(np.percentile(bs_r[o], 97.5))]
                out['contrast'][o]['boot_ratio_med'] = float(np.median(bs_r[o]))
            out['contrast'][o]['boot_slope_nonpos'] = bad[o] / NB
    return out


def at_x0(rows, ycol, arms, ref, x0, xkey='spikes'):
    """팔마다 기울기를 따로 둔 모형 (서버 공통) 에서 x0 의 예측 차 ref − o + 부트스트랩 CI."""
    rr = [r for r in rows if r.get(ycol) is not None and r['arm'] in arms]
    lx0 = math.log(x0)

    def fit(rs):
        X, names, _ = design(rs, arms, ref, xkey, True, False, x0=lx0, slopes=True)
        y = np.array([r[ycol] for r in rs], float)
        b, cov, dof, s2 = ols(y, X)
        return b, cov, dof, names
    b, cov, dof, names = fit(rr)
    out = {'x0': x0, 'n': len(rr), 'dof': dof, 'contrast': {}}
    tc = stats.t.ppf(0.975, dof) if dof > 0 else float('nan')
    for o in arms:
        if o == ref:
            continue
        j = names.index('arm:' + o)
        c = -float(b[j]); se = float(math.sqrt(cov[j, j]))
        out['contrast'][o] = {'c': c, 'se': se, 'ci95': [c - tc * se, c + tc * se],
                              'p': float(2 * stats.t.sf(abs(c / se), dof)) if dof > 0 else float('nan')}
    idx_by = {a: [i for i, r in enumerate(rr) if r['arm'] == a] for a in arms}
    bc = defaultdict(list)
    for _ in range(NB):
        sel = []
        for a in arms:
            sel += list(RNG.choice(idx_by[a], size=len(idx_by[a]), replace=True))
        rs = [rr[i] for i in sel]
        try:
            bb, _, _, nb_ = fit(rs)
        except Exception:
            continue
        for o in out['contrast']:
            bc[o].append(-bb[nb_.index('arm:' + o)])
    for o in out['contrast']:
        out['contrast'][o]['boot_ci95'] = [float(np.percentile(bc[o], 2.5)), float(np.percentile(bc[o], 97.5))]
    return out


# ---------------------------------------------------------------- 런 수집 (원장 스냅숏)
inv = json.load(open(INV))
LF = ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes')


def r19_ok(r):
    if r.get('status') in ('hold',) or r.get('seed') is None:
        return None
    full = 310
    if r['seed'] >= 900:
        return '연기시험 (2에폭)'
    if r.get('exclude') and r.get('exclude_reason'):
        return r['exclude_reason']
    if not r.get('completed') or r.get('epochs') != full:
        return f"미완주 ({r.get('epochs')}에폭, {r.get('status')})"
    return ''


R19, R19_EXCL = [], []
for r in inv['runs']:
    if r.get('ds') not in ('C10', 'C100') or r['key'] not in KEYNAME:
        continue
    if r['key'] in ('ch_novmem', 'ch_nofinal', 'ch_noinv', 'ch_nolr') and r['ds'] != 'C10':
        continue
    why = r19_ok(r)
    if why is None:
        continue
    row = {'tag': r['tag'], 'ds': r['ds'], 'key': r['key'], 'arm': KEYNAME[r['key']], 'knob': r['knob'],
           'seed': r['seed'], 'server': r['server'], 'epochs': r['epochs'], **{k: r.get(k) for k in LF}}
    if why:
        row['reason'] = why
        R19_EXCL.append(row)
    else:
        R19.append(row)

VGG, VGG_EXCL = [], []
for r in inv['runs']:
    t = r['tag']
    if not (t.startswith('vggdvsf0-') or t == 'vggdvs-base-f0-s1'):
        continue
    key = 'base' if '-base-' in t else 'ours_ch'
    row = {'tag': t, 'ds': 'DVS-f0', 'key': key, 'arm': KEYNAME[key], 'knob': r['knob'] if key != 'base' else '-',
           'seed': int(t.rsplit('-s', 1)[1]), 'server': r['server'], 'epochs': r['epochs'], **{k: r.get(k) for k in LF}}
    if r.get('exclude') or not r.get('completed') or r['epochs'] != 200:
        row['reason'] = r.get('exclude_reason') or f"미완주 ({r['epochs']}에폭)"
        VGG_EXCL.append(row)
    else:
        VGG.append(row)


# QK: fp32 재측정 JSON + summary.csv (best eval_top1 행의 손실)
def qk_row(name, ds, key, knob, seed):
    j = json.load(open(f'{QKD}/spikes_{name}.json'))
    ck = j['checkpoint']; rd = os.path.dirname(ck)
    tl = vl = la = None
    sp = os.path.join(rd, 'summary.csv')
    if os.path.exists(sp):
        rows = []
        for q in csv.DictReader(open(sp)):
            try:
                float(q['eval_top1']); rows.append(q)
            except (TypeError, ValueError):
                continue                          # 재개 때 다시 쓴 머리글 행
        if rows:
            best = max(rows, key=lambda q: float(q['eval_top1']))
            tl = float(best['train_loss']); vl = float(best['eval_loss']); la = float(best['eval_top1'])
    return {'log_acc': la, 'tag': 'qk-' + name, 'ds': ds, 'key': key, 'arm': KEYNAME[key], 'knob': knob, 'seed': seed, 'server': '138',
            'train_loss': tl, 'train_acc': None, 'val_loss': vl, 'val_acc': float(j['accuracy']),
            'spikes': float(j['totals']['spikes_per_sample']), 'ckpt': ck}


QK = []
for nm, sd in [('base', 42), ('base_s43', 43), ('base_s45', 45), ('base_s46', 46), ('base_s47', 47), ('base_s48', 48)]:
    QK.append(qk_row(nm, 'QK-C10', 'base', '-', sd))
for nm, kn, sd in [('ours_ch_rho1e-3', '1e-3', 42), ('ours_ch_rho2e-3', '2e-3', 42), ('ours_ch_rho3e-3', '3e-3', 42),
                   ('ours_ch_rho5e-3', '5e-3', 42), ('ours_ch_rho7e-3', '7e-3', 42),
                   ('ours_ch_rho1.2e-2', '1.2e-2', 42), ('ours_ch_rho1.2e-2_s43', '1.2e-2', 43),
                   ('ours_ch_rho1.2e-2_s44', '1.2e-2', 44), ('ours_ch_rho1.2e-2_s45', '1.2e-2', 45),
                   ('ours_ch_rho2e-2', '2e-2', 42)]:
    QK.append(qk_row(nm, 'QK-C10', 'ours_ch', kn, sd))
for nm, kn in [('l2_5e-7', '5e-7'), ('l2_1e-6', '1e-6'), ('l2_3e-6', '3e-6'), ('l2_5e-6', '5e-6'),
               ('l2_1e-5', '1e-5'), ('l2_2e-5', '2e-5')]:
    QK.append(qk_row(nm, 'QK-C10', 'l2', kn, 42))
for sd in (43, 44, 45, 46):
    QK.append(qk_row(f'c100_base_s{sd}', 'QK-C100', 'base', '-', sd))
    QK.append(qk_row(f'c100_ours_ch_rho6e-3_s{sd}', 'QK-C100', 'ours_ch', '6e-3', sd))
    QK.append(qk_row(f'c100_ours_ch_rho1.2e-2_s{sd}', 'QK-C100', 'ours_ch', '1.2e-2', sd))

# SDTv3 (ImageNet-1k, 공식 ckpt 30에폭 파인튜닝, 비KD 본 런만; KD 4런은 진행 중이라 별도 표기)
SDTR, SDT_RUNNING = [], []
for j in json.load(open(SDT))['jobs']:
    if j['ds'] != 'SDT1K' or '보정' in str(j['knob']):
        continue
    key = 'base' if j['arm'] == 'baseline' else 'ours_ch'
    row = {'tag': 'sdt-' + j['dir'], 'ds': 'SDTv3', 'key': key, 'arm': KEYNAME[key], 'knob': j['knob'], 'seed': 0,
           'server': j['server'], 'epochs': j['epochs'], 'train_loss': j['train_loss'], 'train_acc': j['train_acc'],
           'val_loss': j['val_loss'], 'val_acc': j['val_last'], 'val_best': j['val_acc'], 'spikes': j['spikes']}
    if j['status'] != 'done' or 'KD' in str(j['knob']):
        row['reason'] = f"{j['status']} ({j['epochs']}/{j['total']}에폭)" + (' KD' if 'KD' in str(j['knob']) else '')
        SDT_RUNNING.append(row)
    else:
        SDTR.append(row)


# ---------------------------------------------------------------- Q1 곡선
def kf(k):
    try:
        return float(k)
    except Exception:
        return -1.0


def curve(rows, ds, margins, base_key='base'):
    rs = [r for r in rows if r['ds'] == ds]
    base = [r for r in rs if r['key'] == base_key]
    grp = defaultdict(list)
    for r in rs:
        if r['key'] == 'ours_ch':
            grp[r['knob']].append(r)
    bm, bsd, bn = msd([r['val_acc'] for r in base])
    bs = float(np.mean([r['spikes'] for r in base]))
    # 합동 시드 SD (n≥2 묶음 + baseline)
    ss = df_ = 0.0
    for g in list(grp.values()) + [base]:
        if len(g) > 1:
            v = np.array([r['val_acc'] for r in g]); ss += ((v - v.mean()) ** 2).sum(); df_ += len(g) - 1
    sdp = math.sqrt(ss / df_) if df_ > 0 else float('nan')
    out = {'ds': ds, 'baseline': {'rows': base, 'mean': bm, 'sd': bsd, 'n': bn, 'spikes': bs,
                                  'mean_cols': {k: float(np.mean([r[k] for r in base if r.get(k) is not None]))
                                                if any(r.get(k) is not None for r in base) else None for k in LF}},
           'sd_pool': sdp, 'sd_pool_df': df_, 'groups': []}
    for kn in sorted(grp, key=kf):
        g = grp[kn]
        m, sd, n = msd([r['val_acc'] for r in g])
        sp = float(np.mean([r['spikes'] for r in g]))
        w = welch([r['val_acc'] for r in g], [r['val_acc'] for r in base], sd_pool=(sdp, df_))
        e = {'knob': kn, 'rows': g, 'n': n, 'mean': m, 'sd': sd, 'spikes': sp, 'spk_pct': 100 * (sp / bs - 1),
             'mean_cols': {k: float(np.mean([r[k] for r in g if r.get(k) is not None]))
                           if any(r.get(k) is not None for r in g) else None for k in LF},
             'delta': w}
        e['ni'] = {str(mg): bool(w['ci90'][0] > -mg) for mg in margins}
        # 검토 반영 (10-19): n≤4 묶음은 그룹 안 SD 가 작게 나오면 Welch 가 과신한다 → 합동 시드 SD 판정을 함께 둔다
        if n >= 2 and df_ > 0:
            e['delta_pool'] = pooled_t([r['val_acc'] for r in g], [r['val_acc'] for r in base], sdp, df_)
            e['ni_pool'] = {str(mg): bool(e['delta_pool']['ci90'][0] > -mg) for mg in margins}
        e['servers'] = sorted({r['server'] for r in g})
        out['groups'].append(e)
    return out


CURVES = {
    'R19-C10': curve(R19, 'C10', (0.15, 0.2)),
    'R19-C100': curve(R19, 'C100', (0.15, 0.2, 0.3)),
    'VGG-DVS-f0': curve(VGG, 'DVS-f0', (0.2, 0.5)),
    'QK-C10': curve(QK, 'QK-C10', (0.15, 0.2)),
    'QK-C100': curve(QK, 'QK-C100', (0.15, 0.2, 0.3)),
}
# SDT: 대조 n=1 이라 Welch 불가 → 원값 차이만
sb = [r for r in SDTR if r['key'] == 'base'][0]
CURVES['SDTv3'] = {'ds': 'SDTv3', 'baseline': {'rows': [sb], 'mean': sb['val_acc'], 'n': 1, 'spikes': sb['spikes']},
                   'groups': [{'knob': r['knob'], 'rows': [r], 'n': 1, 'mean': r['val_acc'], 'spikes': r['spikes'],
                               'spk_pct': 100 * (r['spikes'] / sb['spikes'] - 1), 'd_raw': r['val_acc'] - sb['val_acc']}
                              for r in sorted([r for r in SDTR if r['key'] == 'ours_ch'], key=lambda q: kf(q['knob']))],
                   'running': SDT_RUNNING}


# ---------------------------------------------------------------- Q2 같은 스파이크 비교
def band(rows, ds, arms, lo, hi):
    return [r for r in rows if r['ds'] == ds and r['arm'] in arms and lo <= r['spikes'] <= hi]


ARM3 = ['ours_inter_ch', 'plain L2', 'BPSR']
Q2 = {}
for lab, lo, hi in [('B1_300_420K', 300e3, 420e3), ('B2_280_420K', 280e3, 420e3), ('B3_180_420K', 180e3, 420e3)]:
    rr = band(R19, 'C10', ARM3, lo, hi)
    Q2[f'C10_{lab}'] = {'lin': ancova(rr, 'val_acc', ARM3, 'ours_inter_ch'),
                        'lin_noserver': ancova(rr, 'val_acc', ARM3, 'ours_inter_ch', server=False, boot=False)}
    if lab.startswith('B3'):
        Q2[f'C10_{lab}']['quad'] = ancova(rr, 'val_acc', ARM3, 'ours_inter_ch', quad=True, boot=False)
    for o in ('plain L2', 'BPSR'):
        Q2[f'C10_{lab}']['pair_' + o] = ancova(rr, 'val_acc', ['ours_inter_ch', o], 'ours_inter_ch')
Q2['C10_at342K'] = at_x0(band(R19, 'C10', ARM3, 280e3, 420e3), 'val_acc', ARM3, 'ours_inter_ch', 342075.0)


def grp_vals(rows, ds, key, knob):
    return [r for r in rows if r['ds'] == ds and r['key'] == key and r['knob'] == knob]


T1 = {'ours_inter_ch': grp_vals(R19, 'C10', 'ours_ch', '3e-3'), 'BPSR': grp_vals(R19, 'C10', 'bpsr', '1e-10'),
      'plain L2': grp_vals(R19, 'C10', 'l2', '1e-7')}
Q2['C10_table1'] = {a: {'n': len(v), 'acc': msd([r['val_acc'] for r in v]), 'spikes': msd([r['spikes'] for r in v])}
                    for a, v in T1.items()}
for o in ('BPSR', 'plain L2'):
    Q2['C10_table1']['vs_' + o] = {
        'acc': welch([r['val_acc'] for r in T1['ours_inter_ch']], [r['val_acc'] for r in T1[o]]),
        'spk_pct': 100 * (np.mean([r['spikes'] for r in T1['ours_inter_ch']]) / np.mean([r['spikes'] for r in T1[o]]) - 1),
        'lnspk': welch(np.log([r['spikes'] for r in T1['ours_inter_ch']]), np.log([r['spikes'] for r in T1[o]]))}
BASE10 = [r['val_acc'] for r in R19 if r['ds'] == 'C10' and r['key'] == 'base']
Q2['C10_table1']['vs_baseline'] = {a_: {**welch([r['val_acc'] for r in v], BASE10),
                                        'spk_pct': 100 * (np.mean([r['spikes'] for r in v]) / np.mean([r['spikes'] for r in R19 if r['ds'] == 'C10' and r['key'] == 'base']) - 1)}
                                   for a_, v in T1.items()}
# 같은 스파이크로 맞춘 표1 점: B2 기울기로 x 차이를 보정
b2 = Q2['C10_B2_280_420K']['lin']
for o in ('BPSR', 'plain L2'):
    v = Q2['C10_table1']['vs_' + o]
    v['adj_acc'] = v['acc']['d'] - b2['b'] * v['lnspk']['d']
    v['adj_se'] = math.sqrt(v['acc']['se'] ** 2 + (b2['b_se'] * v['lnspk']['d']) ** 2 + (b2['b'] * v['lnspk']['se']) ** 2)

for lab, lo, hi in [('C100_B_290_420K', 290e3, 420e3), ('C100_B_220_420K', 220e3, 420e3)]:
    rr = band(R19, 'C100', ARM3, lo, hi)
    Q2[lab] = {'lin': ancova(rr, 'val_acc', ARM3, 'ours_inter_ch')}
    for o in ('plain L2', 'BPSR'):
        Q2[lab]['pair_' + o] = ancova(rr, 'val_acc', ['ours_inter_ch', o], 'ours_inter_ch')
Q2['C100_at354K'] = at_x0(band(R19, 'C100', ARM3, 290e3, 420e3), 'val_acc', ['ours_inter_ch', 'plain L2'], 'ours_inter_ch', 353882.0)
for lab, lo, hi in [('QK_C10_500_800K', 500e3, 800e3), ('QK_C10_450_950K', 450e3, 950e3)]:
    rr = [r for r in QK if r['ds'] == 'QK-C10' and r['arm'] in ('ours_inter_ch', 'plain L2') and lo <= r['spikes'] <= hi]
    Q2[lab] = {'lin': ancova(rr, 'val_acc', ['ours_inter_ch', 'plain L2'], 'ours_inter_ch', server=False)}

# ---------------------------------------------------------------- Q3 구조 (npz)


def weight_layers():
    for d in PRUNE_DIRS:
        for p in sorted(glob.glob(d + '/r19c10-*.json')):
            j = json.load(open(p))
            if 'weight_layers' in j:
                return j['weight_layers']
    raise SystemExit('weight_layers 없음')


WL = weight_layers()


def prune_cost(dead):
    mr = mt = 0
    for w in WL:
        ks = w['ks']; cin, cout = ks[-2], ks[-1]; kk = int(np.prod(ks[:-2])) if len(ks) > 2 else 1
        di = dead.get(w['src'], 0); do = dead.get(w['dst'], 0)
        mr += (di * cout + do * cin - di * do) * kk * w['hw']; mt += cin * cout * kk * w['hw']
    return 100 * mr / mt


def gini_sorted(v):
    v = np.sort(np.asarray(v, float).ravel()); n = v.size; s = v.sum()
    if s == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float((2 * (i * v).sum()) / (n * s) - (n + 1) / n)


def npz_path(tag):
    for d in NPZ_DIRS:
        p = os.path.join(d, tag + '.npz')
        if os.path.exists(p):
            return p
    return None


def npz_metrics(p):
    z = np.load(p)
    ns = float(z['n_samples'])
    allc = []; spk = fired = 0.0; dead_by = {}; st = {s: 0.0 for s in STAGES}
    chg, chw = [], []
    for k in z.files:
        if k.endswith('__fired') or k + '__fired' not in z.files or k in SKIP_L:
            continue
        c = z[k].astype(np.float64); f = z[k + '__fired'].astype(np.float64)
        allc.append(c.ravel()); spk += c.sum(); fired += f.sum(); st[k.split('_')[0]] += c.sum()
        if c.ndim == 3:
            tot = c.reshape(-1, c.shape[-1]).sum(0)
            dead_by[k] = int((tot == 0).sum()); chg.append(gini_sorted(tot)); chw.append(tot.sum())
        else:
            dead_by[k] = int((c == 0).sum())
    chw = np.array(chw)
    m = {'npz_spk': spk / ns, 'npz_acc': float(z['acc']), 'dead_ch': int(sum(dead_by.values())),
         'dead_ch_conv': int(sum(v for k, v in dead_by.items() if not k.startswith('fc1'))),
         'macs_b': prune_cost(dead_by), 'macs_a': prune_cost({k: v for k, v in dead_by.items() if '_out_n' not in k}),
         'active_img': fired / ns, 'gini_neuron': gini_sorted(np.concatenate(allc)),
         'gini_ch_w': float((np.array(chg) * chw).sum() / chw.sum()), 'steps_per_active': spk / fired}
    for s in STAGES:
        m['share_' + s] = 100 * st[s] / spk
    return m


NPZ_ROWS = []
for r in R19:
    if r['ds'] != 'C10':
        continue
    p = npz_path(r['tag'])
    if p:
        NPZ_ROWS.append({**r, **npz_metrics(p), 'npz_file': p})
NPZ_METRICS = ['dead_ch', 'macs_a', 'macs_b', 'active_img', 'gini_ch_w', 'gini_neuron', 'steps_per_active',
               'share_conv1', 'share_conv2', 'share_conv3', 'share_conv4']
Q3 = {'npz_n': {a: sum(r['arm'] == a for r in NPZ_ROWS) for a in KEYNAME.values()}, 'npz': {}}
for lab, lo, hi in [('B1_300_420K', 300e3, 420e3), ('B3_180_420K', 180e3, 420e3)]:
    rr = [r for r in NPZ_ROWS if r['arm'] in ARM3 and lo <= r['spikes'] <= hi]
    Q3['npz'][lab] = {m: ancova(rr, m, ARM3, 'ours_inter_ch', xkey='npz_spk', logy=(m == 'active_img'), boot=False)
                      for m in NPZ_METRICS}
# 표1 점 (n=4 끼리) 평균·Welch
T1N = {a: [r for r in NPZ_ROWS if r['tag'] in {q['tag'] for q in v}] for a, v in T1.items()}
BASEN = [r for r in NPZ_ROWS if r['key'] == 'base']
Q3['table1'] = {'n': {a: len(v) for a, v in T1N.items()}, 'baseline_n': len(BASEN), 'mean': {}, 'vs': {}}
for a, v in list(T1N.items()) + [('baseline', BASEN)]:
    Q3['table1']['mean'][a] = {m: msd([r[m] for r in v]) for m in NPZ_METRICS + ['npz_spk']}
for o in ('BPSR', 'plain L2'):
    if T1N[o] and T1N['ours_inter_ch']:
        Q3['table1']['vs'][o] = {m: welch([r[m] for r in T1N['ours_inter_ch']], [r[m] for r in T1N[o]])
                                 for m in NPZ_METRICS + ['npz_spk']}


# ---------------------------------------------------------------- Q3 구조 (CSV, 모든 런)
NEUR = {}
for k in ['conv1_conv_n'] + [f'conv2_block{i}_{s}_n' for i in (1, 2, 3) for s in ('conv1', 'out')]:
    NEUR[k] = 32 * 32 * 128
for k in [f'conv3_block{i}_{s}_n' for i in (1, 2, 3) for s in ('conv1', 'out')]:
    NEUR[k] = 16 * 16 * 256
for k in [f'conv4_block{i}_{s}_n' for i in (1, 2) for s in ('conv1', 'out')]:
    NEUR[k] = 8 * 8 * 512
NEUR['fc1_n'] = 256
NTOT = sum(NEUR.values())


def csv_dir(r):
    if r['server'] == '23':
        d = os.path.join(CSV23, r['tag'])
    else:
        d = os.path.join(HDD, r['tag'])
        if not os.path.exists(os.path.join(d, 'reg_neuron_detail.csv')):
            d = os.path.join(ROOT, r['tag'])
    return d if os.path.exists(os.path.join(d, 'reg_neuron_detail.csv')) else None


def best_epoch_of(tag, server):
    f = A + '/snapshot/' + ('inv23.csv' if server == '23' else 'inv138.csv')
    for q in csv.DictReader(open(f)):
        if q['run_dir'] == '_paper/' + tag:
            return int(q['best_epoch'])
    return None


def csv_metrics(r):
    d = csv_dir(r)
    be = best_epoch_of(r['tag'], r['server'])
    if d is None or be is None:
        return None
    e = str(be - 1)                                      # CSV epoch 는 0부터 (ep-0307 가중치 = CSV 306)
    act = 0.0; gnum = gden = 0.0; ok = 0
    for q in csv.DictReader(open(os.path.join(d, 'reg_neuron_detail.csv'))):
        if q['epoch'] != e or q['layer'] not in NEUR:
            continue
        act += NEUR[q['layer']] * (1 - float(q['dead_neuron_ratio'])); ok += 1
    st = {s: 0.0 for s in STAGES}; lay = {}
    for q in csv.DictReader(open(os.path.join(d, 'reg_detail.csv'))):
        if q['epoch'] != e or q['layer'] not in NEUR:
            continue
        v = float(q['spike_count']); st[q['layer'].split('_')[0]] += v; lay[q['layer']] = v
    if ok != len(NEUR) or not lay:
        return None
    tot = sum(st.values())
    for q in csv.DictReader(open(os.path.join(d, 'reg_neuron_detail.csv'))):
        if q['epoch'] != e or q['layer'] not in NEUR:
            continue
        gnum += float(q['gini']) * lay[q['layer']]; gden += lay[q['layer']]
    m = {'csv_active_img': act, 'csv_active_pct': 100 * act / NTOT, 'csv_spk': tot, 'csv_gini_layer_w': gnum / gden,
         'csv_epoch': int(e)}
    for s in STAGES:
        m['csv_share_' + s] = 100 * st[s] / tot
    return m


CSV_ROWS = []
CSV_MISSING = []
for r in R19:
    m = csv_metrics(r)
    if m:
        CSV_ROWS.append({**r, **m})
    else:
        CSV_MISSING.append(r['tag'])
CSV_METRICS = ['csv_active_img', 'csv_gini_layer_w', 'csv_share_conv1', 'csv_share_conv2', 'csv_share_conv3', 'csv_share_conv4']
Q3['csv'] = {}
for lab, ds, lo, hi in [('C10_B1_300_420K', 'C10', 300e3, 420e3), ('C10_B3_180_420K', 'C10', 180e3, 420e3),
                        ('C100_B_290_420K', 'C100', 290e3, 420e3), ('C100_B_220_420K', 'C100', 220e3, 420e3)]:
    rr = [r for r in CSV_ROWS if r['ds'] == ds and r['arm'] in ARM3 and lo <= r['spikes'] <= hi]
    Q3['csv'][lab] = {m: ancova(rr, m, ARM3, 'ours_inter_ch', logy=(m == 'csv_active_img'), boot=False) for m in CSV_METRICS}
Q3['csv_base'] = {ds: {m: msd([r[m] for r in CSV_ROWS if r['ds'] == ds and r['key'] == 'base']) for m in CSV_METRICS + ['csv_spk']}
                  for ds in ('C10', 'C100')}
Q3['csv_table1'] = {a: {m: msd([r[m] for r in CSV_ROWS if r['tag'] in {q['tag'] for q in v}]) for m in CSV_METRICS + ['csv_spk']}
                    for a, v in T1.items()}
# npz ↔ CSV 활성 뉴런 일치 (같은 런)
both = [r for r in CSV_ROWS if npz_path(r['tag']) and r['ds'] == 'C10']
if both:
    xs = np.array([[r['csv_active_img'], npz_metrics(npz_path(r['tag']))['active_img']] for r in both])
    Q3['csv_vs_npz_active'] = {'n': len(both), 'r': float(np.corrcoef(xs[:, 0], xs[:, 1])[0, 1]),
                               'ratio_mean': float((xs[:, 0] / xs[:, 1]).mean())}

# ---------------------------------------------------------------- 덧붙임: 구성요소 (ablation) — 정확도·CSV 구조
ABL = ['- vmem', '- final_step', '- 1- inversion', '- loss-ratio']
ABLQ = {}
for a in ABL:
    rr = [r for r in R19 if r['ds'] == 'C10' and r['arm'] in ('ours_inter_ch', a) and 180e3 <= r['spikes'] <= 420e3]
    ABLQ[a] = {'curve_180_420K': ancova(rr, 'val_acc', ['ours_inter_ch', a], 'ours_inter_ch'),
               'csv_active_180_420K': ancova([r for r in CSV_ROWS if r['tag'] in {q['tag'] for q in rr}],
                                             'csv_active_img', ['ours_inter_ch', a], 'ours_inter_ch', logy=True, boot=False)}
ABL_PT = {'- vmem': '2e-3', '- final_step': '1.4e-3', '- 1- inversion': '4.5e-3', '- loss-ratio': '1.25e-7'}
ABL_KEY = {'- vmem': 'ch_novmem', '- final_step': 'ch_nofinal', '- 1- inversion': 'ch_noinv', '- loss-ratio': 'ch_nolr'}
for a in ABL:
    g = grp_vals(R19, 'C10', ABL_KEY[a], ABL_PT[a])
    ABLQ[a]['point'] = {'knob': ABL_PT[a], 'rows': g, 'acc': msd([r['val_acc'] for r in g]),
                        'spikes': msd([r['spikes'] for r in g]),
                        'vs_inter3e-3': welch([r['val_acc'] for r in g], [r['val_acc'] for r in T1['ours_inter_ch']]),
                        'spk_pct_vs_inter3e-3': 100 * (np.mean([r['spikes'] for r in g]) / np.mean([r['spikes'] for r in T1['ours_inter_ch']]) - 1),
                        'csv': {m: msd([r[m] for r in CSV_ROWS if r['tag'] in {q['tag'] for q in g}]) for m in CSV_METRICS}}


# ---------------------------------------------------------------- 민감도: 체크포인트 기준 (best → 마지막 10에폭 평균)
# 에폭별 val acc·s_count: 138 은 train.log 를 직접, 23 은 세종에서 grep 한 줄만 받아 an1019/ep/ep23.txt 에 둠.
EP = defaultdict(list)
for f in ('ep23.txt', 'ep138.txt'):
    for line in open(f'{A}/ep/{f}'):
        q = line.split()
        EP[q[0]].append((100 * float(q[q.index('val_acc:') + 1]), float(q[q.index('s_count:') + 1])))
EP_CHECK = {'runs': 0, 'mismatch': 0}
for r in R19:
    v = EP.get(r['tag'])
    if not v or len(v) != 310:
        r['last10_acc'] = r['last10_spk'] = None
        continue
    EP_CHECK['runs'] += 1
    if abs(max(a for a, _ in v) - r['val_acc']) > 0.006:
        EP_CHECK['mismatch'] += 1
    r['last10_acc'] = float(np.mean([a for a, _ in v[-10:]]))
    r['last10_spk'] = float(np.mean([b for _, b in v[-10:]]))
    r['last_acc'] = v[-1][0]


def relabel(rows, ak, sk):
    return [{**r, 'val_acc': r[ak], 'spikes': r[sk]} for r in rows if r.get(ak) is not None]


R19L = relabel(R19, 'last10_acc', 'last10_spk')
SENS = {'ep_check': EP_CHECK}
for lab, ds, lo, hi in [('C10_B1_300_420K', 'C10', 300e3, 420e3), ('C10_B2_280_420K', 'C10', 280e3, 420e3),
                        ('C10_B3_180_420K', 'C10', 180e3, 420e3), ('C100_B_290_420K', 'C100', 290e3, 420e3)]:
    SENS[lab] = ancova(band(R19L, ds, ARM3, lo, hi), 'val_acc', ARM3, 'ours_inter_ch')
T1L = {a: [r for r in R19L if r['tag'] in {q['tag'] for q in v}] for a, v in T1.items()}
SENS['C10_table1'] = {a: {'acc': msd([r['val_acc'] for r in v]), 'spikes': msd([r['spikes'] for r in v])} for a, v in T1L.items()}
for o in ('BPSR', 'plain L2'):
    SENS['C10_table1']['vs_' + o] = welch([r['val_acc'] for r in T1L['ours_inter_ch']], [r['val_acc'] for r in T1L[o]])
SENS['curves_last10'] = {'R19-C10': curve(R19L, 'C10', (0.15, 0.2)), 'R19-C100': curve(R19L, 'C100', (0.15, 0.2, 0.3))}
for a in ABL:
    rr = [r for r in R19L if r['ds'] == 'C10' and r['arm'] in ('ours_inter_ch', a) and 180e3 <= r['spikes'] <= 420e3]
    SENS['abl_' + a] = ancova(rr, 'val_acc', ['ours_inter_ch', a], 'ours_inter_ch', boot=False)


# ---------------------------------------------------------------- 영향 런 점검 (하나 빼기, 서버별)
def loo(rows, arms, ref, other):
    out = []
    for i, r in enumerate(rows):
        rs = rows[:i] + rows[i + 1:]
        a = ancova(rs, 'val_acc', arms, ref, boot=False)
        if a:
            e = a['contrast'][other]
            out.append({'drop': r['tag'], 'c': e['c'], 'p': e['p']})
    return {'c_min': min(o['c'] for o in out), 'c_max': max(o['c'] for o in out),
            'p_max': max(o['p'] for o in out), 'n_sig05': sum(o['p'] < 0.05 for o in out), 'n': len(out),
            'most_influential': sorted(out, key=lambda o: o['c'])[:2] + sorted(out, key=lambda o: -o['c'])[:2]}


INFL = {}
rrb1 = band(R19, 'C10', ARM3, 300e3, 420e3)
INFL['C10_B1_inter_vs_BPSR'] = loo(rrb1, ARM3, 'ours_inter_ch', 'BPSR')
INFL['C10_B1_inter_vs_L2'] = loo(rrb1, ARM3, 'ours_inter_ch', 'plain L2')
rv = [r for r in R19 if r['ds'] == 'C10' and r['arm'] in ('ours_inter_ch', '- vmem') and 180e3 <= r['spikes'] <= 420e3]
INFL['vmem_curve'] = loo(rv, ['ours_inter_ch', '- vmem'], 'ours_inter_ch', '- vmem')
for sv in ('138', '23'):
    INFL['vmem_server_' + sv] = ancova([r for r in rv if r['server'] == sv], 'val_acc', ['ours_inter_ch', '- vmem'],
                                       'ours_inter_ch', server=False, boot=False)

# ---------------------------------------------------------------- QK: 짝 비교 (n=1 대 n=1, 판단 불가 표기용) · 보간
QKC10 = [r for r in QK if r['ds'] == 'QK-C10']
l2q = sorted([r for r in QKC10 if r['key'] == 'l2'], key=lambda r: r['spikes'])
sdq = CURVES['QK-C10']['sd_pool']


def interp_l2(x):
    lx = math.log(x)
    for a, b in zip(l2q, l2q[1:]):
        if a['spikes'] <= x <= b['spikes']:
            w = (lx - math.log(a['spikes'])) / (math.log(b['spikes']) - math.log(a['spikes']))
            return a['val_acc'] + w * (b['val_acc'] - a['val_acc']), math.sqrt((1 - w) ** 2 + w ** 2)
    return None, None


QKP = []
for kn in ('7e-3', '1.2e-2', '2e-2'):
    g = [r for r in QKC10 if r['key'] == 'ours_ch' and r['knob'] == kn]
    x = float(np.mean([r['spikes'] for r in g])); m = float(np.mean([r['val_acc'] for r in g]))
    li, f = interp_l2(x)
    if li is None:
        continue
    se = sdq * math.sqrt(1 / len(g) + f ** 2)
    QKP.append({'knob': kn, 'n_inter': len(g), 'spikes': x, 'inter': m, 'l2_interp': li, 'd': m - li, 'se': se,
                'ci95': [m - li - 1.96 * se, m - li + 1.96 * se]})
# 같은 비교를 학습 로그 최고값 기준으로 (재측정이 런마다 −0.0~−0.46 다르게 낮아서)
l2q_log = sorted([r for r in QKC10 if r['key'] == 'l2' and r.get('log_acc') is not None], key=lambda r: r['spikes'])
for e in QKP:
    g = [r for r in QKC10 if r['key'] == 'ours_ch' and r['knob'] == e['knob']]
    lx = math.log(e['spikes'])
    for a, b in zip(l2q_log, l2q_log[1:]):
        if a['spikes'] <= e['spikes'] <= b['spikes']:
            w = (lx - math.log(a['spikes'])) / (math.log(b['spikes']) - math.log(a['spikes']))
            e['log_inter'] = float(np.mean([r['log_acc'] for r in g]))
            e['log_l2_interp'] = a['log_acc'] + w * (b['log_acc'] - a['log_acc'])
            e['log_d'] = e['log_inter'] - e['log_l2_interp']
Q2['QK_C10_interp'] = {'pairs': QKP, 'sd_pool': sdq,
                       'l2_3e-6_other_seeds_log': {j['name']: [j['acc'], j.get('server')] for j in json.load(open(A + '/snapshot/qk.json'))['jobs']
                                                   if j['name'].startswith('plain L2 λ 3e-6')},
                       'note': 'plain L2 는 ρ 마다 시드 42 한 런. 보간 SE 는 합동 시드 SD 로 근사 (곡선 모양 불확실성 제외).'}

# ---------------------------------------------------------------- 검토 반영 (eip-critic 10-19) — 추가 계산
# 난수를 쓰지 않는다 (위 부트스트랩 결과가 바뀌지 않도록).
REV = {}

# (1) 측정 축: s_count 는 학습 배치 통계. CSV 층 합 (n_in 포함) = s_count 인지 전수 확인
def csv_total(r):
    d = csv_dir(r); be = best_epoch_of(r['tag'], r['server'])
    if d is None or be is None:
        return None
    e = str(be - 1)
    return sum(float(q['spike_count']) for q in csv.DictReader(open(os.path.join(d, 'reg_detail.csv'))) if q['epoch'] == e)


_chk = [(r['tag'], csv_total(r), r['spikes']) for r in R19]
_chk = [c for c in _chk if c[1] is not None]
REV['csv_eq_scount'] = {'runs': len(_chk), 'mismatch_rel_1e-4': sum(abs(a - b) / b > 1e-4 for _, a, b in _chk),
                        'max_rel': max(abs(a - b) / b for _, a, b in _chk)}
_rat = defaultdict(list)
for r in NPZ_ROWS:
    _rat[r['arm']].append(r['npz_spk'] / r['spikes'])
REV['npz_over_scount_all'] = {a: msd(v) for a, v in _rat.items()}
_bn = [r for r in NPZ_ROWS if r['key'] == 'base']
_bn_npz = float(np.mean([r['npz_spk'] for r in _bn])); _bn_sc = float(np.mean([r['spikes'] for r in _bn]))
REV['npz_axis_base'] = {'n': len(_bn), 'tags': [r['tag'] for r in _bn], 'npz': _bn_npz, 'scount': _bn_sc,
                        'ratio': msd([r['npz_spk'] / r['spikes'] for r in _bn])}
# 묶음별 두 축 (그 묶음 런이 모두 npz 를 가진 경우만)
REV['npz_axis_groups'] = {}
_npzt = {r['tag']: r for r in NPZ_ROWS}
_grp = defaultdict(list)
for r in R19:
    if r['ds'] == 'C10' and r['key'] != 'base':
        _grp[(r['arm'], r['knob'])].append(r)
for (a_, kn), g in _grp.items():
    if all(r['tag'] in _npzt for r in g):
        npz_m = float(np.mean([_npzt[r['tag']]['npz_spk'] for r in g])); sc_m = float(np.mean([r['spikes'] for r in g]))
        REV['npz_axis_groups'][f'{a_}|{kn}'] = {'n': len(g), 'npz': npz_m, 'scount': sc_m,
                                                'pct_npz': 100 * (npz_m / _bn_npz - 1),
                                                'pct_scount_same3': 100 * (sc_m / _bn_sc - 1),
                                                'ratio': msd([_npzt[r['tag']]['npz_spk'] / r['spikes'] for r in g])}

# (2) 비열등 판정의 여러 기준 (표1 점, baseline n=6 대비)
_b10 = [r for r in R19 if r['ds'] == 'C10' and r['key'] == 'base']
_sdp, _dfp = CURVES['R19-C10']['sd_pool'], CURVES['R19-C10']['sd_pool_df']
REV['ni_t1'] = {}
for a_, v in T1.items():
    REV['ni_t1'][a_] = {'best_welch': welch([r['val_acc'] for r in v], [r['val_acc'] for r in _b10]),
                        'best_pooled': pooled_t([r['val_acc'] for r in v], [r['val_acc'] for r in _b10], _sdp, _dfp),
                        'last10_welch': welch([r['last10_acc'] for r in v], [r['last10_acc'] for r in _b10]),
                        'last_welch': welch([r['last_acc'] for r in v], [r['last_acc'] for r in _b10]),
                        'last10_mean': float(np.mean([r['last10_acc'] for r in v])),
                        'last_mean': float(np.mean([r['last_acc'] for r in v]))}
REV['ni_t1']['baseline'] = {'last10_mean': float(np.mean([r['last10_acc'] for r in _b10])),
                            'last_mean': float(np.mean([r['last_acc'] for r in _b10])), 'n': len(_b10)}

# (3) 표1 npz Welch 24개의 Holm 보정
_pm = {}
for o in ('BPSR', 'plain L2'):
    for m_ in NPZ_METRICS + ['npz_spk']:
        if Q3['table1']['vs'].get(o):
            _pm[f'{o}|{m_}'] = Q3['table1']['vs'][o][m_]['p']
REV['t1_npz_holm'] = {'m': len(_pm), 'adj': holm(_pm)}

# (4) 구성요소 4팔 검정의 Holm 보정, - vmem 서버 상호작용, - vmem 4팔 모형
_pa = {a_: ABLQ[a_]['curve_180_420K']['contrast'][a_]['p'] for a_ in ABL}
REV['abl_holm'] = {'raw': _pa, 'adj': holm(_pa)}
_rv = [r for r in R19 if r['ds'] == 'C10' and r['arm'] in ('ours_inter_ch', '- vmem') and 180e3 <= r['spikes'] <= 420e3]
_x = np.log([r['spikes'] for r in _rv]); _x = _x - _x.mean()
_am = np.array([r['arm'] == '- vmem' for r in _rv], float); _sv = np.array([r['server'] == '23' for r in _rv], float)
_bb, _cv, _dof, _ = ols([r['val_acc'] for r in _rv], np.column_stack([np.ones_like(_x), _x, _am, _sv, _am * _sv]))
REV['vmem_x_server'] = {'n': len(_rv), 'dof': _dof, 'coef': float(_bb[4]), 'se': float(math.sqrt(_cv[4, 4])),
                        'p': float(2 * stats.t.sf(abs(_bb[4] / math.sqrt(_cv[4, 4])), _dof)),
                        'inter_minus_vmem_138': float(-_bb[2]), 'inter_minus_vmem_23': float(-(_bb[2] + _bb[4])),
                        'n_vmem_by_server': {s: int(sum(1 for r in _rv if r['arm'] == '- vmem' and r['server'] == s)) for s in ('138', '23')}}
ARM4V = ['- vmem', 'ours_inter_ch', 'BPSR', 'plain L2']
REV['vmem_4arm_180_420K'] = ancova([r for r in R19 if r['ds'] == 'C10' and r['arm'] in ARM4V and 180e3 <= r['spikes'] <= 420e3],
                                   'val_acc', ARM4V, '- vmem', boot=False)

# (5) - 1- inversion 의 구조: 4팔 모형 (기준 = - 1- inversion), 전체·138 만
ARM4I = ['- 1- inversion', 'ours_inter_ch', 'BPSR', 'plain L2']
REV['noinv_struct'] = {}
for lab, lo, hi in (('300_420K', 300e3, 420e3), ('180_420K', 180e3, 420e3)):
    for sel in ('all', '138'):
        rr = [r for r in CSV_ROWS if r['ds'] == 'C10' and r['arm'] in ARM4I and lo <= r['spikes'] <= hi and (sel == 'all' or r['server'] == sel)]
        act = ancova(rr, 'csv_active_img', ARM4I, '- 1- inversion', logy=True, boot=False, server=(sel == 'all'))
        gin = ancova(rr, 'csv_gini_layer_w', ARM4I, '- 1- inversion', boot=False, server=(sel == 'all'))
        e = {'active': act, 'gini': gin}
        # inter − BPSR 의 log 차 중 1− 반전이 설명하는 몫 = (noinv − inter) / (BPSR − inter)
        ci_ = act['contrast']
        l_ni = ci_['ours_inter_ch']['c']; l_nb = ci_['BPSR']['c']          # noinv − inter, noinv − BPSR (log)
        e['inter_minus_bpsr_pct'] = 100 * (math.exp(-l_ni + l_nb) - 1)
        e['share_explained_by_inversion'] = l_ni / (l_ni - l_nb) if (l_ni - l_nb) != 0 else None
        REV['noinv_struct'][f'{lab}_{sel}'] = e
REV['noinv_runs'] = sorted([(r['knob'], r['seed'], r['server'], r['spikes']) for r in R19 if r['arm'] == '- 1- inversion'])
REV['gini_points'] = {f'{a_}|{kn}': msd([r['csv_gini_layer_w'] for r in CSV_ROWS if r['ds'] == 'C10' and r['arm'] == a_ and r['knob'] == kn])
                      for a_, kn in (('ours_inter_ch', '3e-3'), ('BPSR', '1e-10'), ('plain L2', '1e-7'), ('- 1- inversion', '4.5e-3'))}


# (6) 활성 뉴런: 팔마다 기울기를 따로 둔 모형으로 스파이크 수준별 차 (해석적 CI, 서버 항 포함)
def sep_slope_at(rows, ycol, arms, ref, x0s, logy=False):
    rr = [r for r in rows if r['arm'] in arms]
    y = np.log([r[ycol] for r in rr]) if logy else np.array([r[ycol] for r in rr], float)
    out = {'n': len(rr), 'by_arm': {a: sum(r['arm'] == a for r in rr) for a in arms},
           'x_range': {a: [min(r['spikes'] for r in rr if r['arm'] == a), max(r['spikes'] for r in rr if r['arm'] == a)] for a in arms},
           'at': {}}
    for x0 in x0s:
        X, names, _ = design(rr, arms, ref, 'spikes', True, False, x0=math.log(x0), slopes=True)
        b, cov, dof, _ = ols(y, X); tc = stats.t.ppf(0.975, dof); o = {}
        for a in arms:
            if a == ref:
                continue
            j = names.index('arm:' + a); c = -float(b[j]); se_ = float(math.sqrt(cov[j, j]))
            o[a] = ({'pct': 100 * (math.exp(c) - 1), 'pct_ci95': [100 * (math.exp(c - tc * se_) - 1), 100 * (math.exp(c + tc * se_) - 1)]}
                    if logy else {'c': c, 'ci95': [c - tc * se_, c + tc * se_]})
        out['at'][str(int(x0))] = o
    return out


REV['active_sep'] = {
    'C10_300_420K': sep_slope_at([r for r in CSV_ROWS if r['ds'] == 'C10' and 300e3 <= r['spikes'] <= 420e3], 'csv_active_img', ARM3, 'ours_inter_ch', (400e3, 342075, 310e3), logy=True),
    'C10_180_420K': sep_slope_at([r for r in CSV_ROWS if r['ds'] == 'C10' and 180e3 <= r['spikes'] <= 420e3], 'csv_active_img', ARM3, 'ours_inter_ch', (400e3, 342075, 310e3, 250e3), logy=True),
    'C100_290_420K': sep_slope_at([r for r in CSV_ROWS if r['ds'] == 'C100' and 290e3 <= r['spikes'] <= 420e3], 'csv_active_img', ARM3, 'ours_inter_ch', (400e3, 353882, 310e3), logy=True)}
_cr = defaultdict(list)
for r in CSV_ROWS:
    if r['tag'] in _npzt:
        _cr[r['arm']].append(r['csv_active_img'] / _npzt[r['tag']]['active_img'])
REV['csv_over_npz_active'] = {a: msd(v) for a, v in _cr.items()}


# (7) 뉴런 희소성의 위치 (Shi 2024 비교용): 테스트 1만 장에서 한 번도 안 쏜 뉴런 비율, 산 채널 비율, 이미지당 활성 비율
def never_fire_frac(p):
    z = np.load(p); tot = dead = 0
    for k in z.files:
        if k.endswith('__fired') or k + '__fired' not in z.files or k in SKIP_L:
            continue
        c = z[k]; tot += c.size; dead += int((c == 0).sum())
    return dead / tot


REV['sparsity_position'] = {}
for a_, v in list(T1N.items()) + [('baseline', BASEN)]:
    REV['sparsity_position'][a_] = {'n': len(v), 'never_fire_neuron_pct': msd([100 * never_fire_frac(r['npz_file']) for r in v]),
                                    'alive_channel_pct': msd([100 * (1 - r['dead_ch'] / 4736) for r in v]),
                                    'active_per_img_pct': msd([100 * r['active_img'] / NTOT for r in v])}

# (8) VGGSNN 첫 프레임 전체 추세 (n=1 인 ρ 2e-2 제외 / 포함)
REV['vgg_trend'] = {}
_vb = [r for r in VGG if r['key'] == 'base']; _vbs = float(np.mean([r['spikes'] for r in _vb]))
_v1e2 = float(np.mean([r['spikes'] for r in VGG if r['knob'] == '1e-2']))
for lab, rr in (('excl_2e-2', [r for r in VGG if r['knob'] != '2e-2']), ('incl_2e-2', VGG)):
    x = np.log([r['spikes'] for r in rr]); bb, cv, dof, s2 = ols([r['val_acc'] for r in rr], np.column_stack([np.ones_like(x), x]))
    se_ = float(math.sqrt(cv[1, 1])); L = math.log(_v1e2 / _vbs)
    REV['vgg_trend'][lab] = {'n': len(rr), 'slope': float(bb[1]), 'se': se_, 'p': float(2 * stats.t.sf(abs(bb[1] / se_), dof)),
                             'pred_at_1e-2': float(bb[1] * L), 'pred_se': se_ * abs(L), 'spk_pct_1e-2': 100 * (_v1e2 / _vbs - 1)}
_vr = [r['val_acc'] for r in VGG if r['key'] != 'base' and r['knob'] != '2e-2']
REV['vgg_reg_mean_minus_base'] = {'n_reg': len(_vr), 'n_base': len(_vb), 'd': float(np.mean(_vr) - np.mean([r['val_acc'] for r in _vb]))}

# ---------------------------------------------------------------- 저장
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (isinstance(o, float) and math.isnan(o)) or (isinstance(o, np.floating) and np.isnan(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


RESULT = {'generated': '2026-10-19', 'snapshot': inv['generated'], 'curves': CURVES, 'q2': Q2, 'q3': Q3,
          'ablation': ABLQ, 'sens_last10': SENS, 'influence': INFL, 'excluded_r19': R19_EXCL, 'excluded_vgg': VGG_EXCL, 'csv_missing': CSV_MISSING,
          't1_rows': T1, 'npz_rows': [{k: v for k, v in r.items()} for r in NPZ_ROWS], 'review': REV}

# ---------------------------------------------------------------- md
def write_md(R, path):
    L = []
    w = L.append

    def F(x, d=2, sign=False):
        if x is None:
            return '–'
        return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'

    def K(x):
        return '–' if x is None else f'{x:,.0f}'

    def P(p):
        if p is None:
            return ''
        return 'p<0.001' if p < 0.001 else (f'p={p:.3f}' if p < 0.1 else f'p={p:.2f}')

    def CI(c, d=2):
        return f'[{c[0]:+.{d}f}, {c[1]:+.{d}f}]'

    def G(cv, kn):
        return next(g for g in R['curves'][cv]['groups'] if g['knob'] == kn)

    def C(q, o):
        return q['contrast'][o]

    cur = R['curves']; q2 = R['q2']; q3 = R['q3']; ab = R['ablation']; se = R['sens_last10']; inf = R['influence']
    c10, c100, dvs, qk10, qk100, sdt = (cur[k] for k in ('R19-C10', 'R19-C100', 'VGG-DVS-f0', 'QK-C10', 'QK-C100', 'SDTv3'))
    t1 = q2['C10_table1']; b1 = q2['C10_B1_300_420K']['lin']; b2 = q2['C10_B2_280_420K']['lin']; b3 = q2['C10_B3_180_420K']['lin']
    cb = q2['C100_B_290_420K']['lin']
    qkp = {e['knob']: e for e in q2['QK_C10_interp']['pairs']}
    s1 = se['C10_B1_300_420K']; s3 = se['C10_B3_180_420K']

    w('# ours_inter_ch 방법 분석 갱신 (2026-10-19)')
    w('')
    w('- 요청 (사용자 10-19): "지금 결과로 … 방법 분석 해봐야할거같아. 지난번에 했던것처럼 우리 방법 분석이 좀 필요할 것 같고 ablation 분석도".')
    w('  이 문서는 방법(ours_inter_ch) 분석이다. 구성요소(ablation) 는 4절에 정확도·구조 요약만 두고, 요소별 기전은 `analysis/ablation_261019.*` 가 맡는다.')
    w(f'- 작성: eip-stats. 생성: `analysis/method_261019.py` → `.json`, 이 문서. 원장 스냅숏 {R["snapshot"]} (모든 보고서가 같은 n).')
    w('- 파일만 썼다. 큐·실험·서버·노션은 건드리지 않았고 커밋하지 않았다. npz 4런은 1판 때 canus GPU 0·4 에서 추론만 했다 (띄우기 직전 메모리 < 1 GiB 확인, GPU 당 1프로세스). 2판의 새 계산은 GPU 를 쓰지 않았다 (기존 npz·CSV·로그만 읽음).')
    w('- 이전 판: `docs/paper/inter_justification_261014.md` (주장 계층), `analysis/inter_vs_bpsr_261009.md`, `struct_compare_261007.md`, `inter_results_261009.md`.')
    w('- **10-19 2판: eip-critic 검토 반영.** 구조 주장 둘과 일반성 주장 하나를 A → B 로 내렸고, 스파이크 측정 축 (s_count = 학습 배치 통계) 을 정정했고, 비열등 판정이 ±0.15 경계 위에 있음을 적었다. 바뀐 곳은 끝의 "검토 반영" 절에 모았다.')
    w('')
    w('**정의**')
    w('- 정확도 = best val acc, 스파이크 = 그 에폭의 s_count (원장). 예외: SDTv3 는 공식 체크포인트 파인튜닝이라 best 가 에폭 1 근처에 생기므로 **마지막 에폭** 값 (대조 런 마지막 에폭과 비교). QK 는 fp32 model_best 를 테스트 셋에서 재측정한 값.')
    ce = R['review']['csv_eq_scount']; rb = R['review']['npz_axis_base']; ra = R['review']['npz_over_scount_all']
    w(f'- **s_count 는 학습 배치 통계다 (2판 정정).** 그 에폭의 학습 패스 (증강된 학습 50,000장, 학습 모드 순전파) 에서 센 이미지당 스파이크다 — `lib_snn/proc.py` 의 `postproc_batch_train_snn → spike_count_batch_end`, 에폭 끝에 `train_ds_num` 으로 나눔; 테스트 배치는 `mode==\'inference\'` 일 때만 센다. reg_detail.csv 의 같은 에폭 층 합 (n_in 포함) 은 s_count 와 같다 ({ce["runs"]}런, 불일치 {ce["mismatch_rel_1e-4"]}). 그래서 CSV 지표도 학습 패스 통계다.')
    w(f'- **테스트 셋 npz 와의 비가 팔마다 다르다:** npz/s_count = baseline {rb["ratio"][0]:.4f} (n={rb["n"]}, s11~s13, 138), ours_inter_ch {ra["ours_inter_ch"][0]:.3f} (n={ra["ours_inter_ch"][2]}), BPSR {ra["BPSR"][0]:.3f} (n={ra["BPSR"][2]}), plain L2 {ra["plain L2"][0]:.3f} (n={ra["plain L2"][2]}). 그래서 **baseline 대비 % 는 측정 축에 따라 1.5~2%p 달라진다** (R19 C10 표에 두 축을 함께 적었다). 규제 팔끼리의 비교 (inter 대 BPSR·plain L2) 는 비가 같아 영향이 없다. 09-15 결정 "불일치 해소 전 절대 스파이크 수를 논문에 쓰지 않는다" 는 그대로 유효하다 (학습 패스 대 테스트 셋이라는 차이가 불일치를 전부 설명하는지는 확인하지 않았다).')
    w('- **R19 의 val 셋은 테스트 셋이다.** best 는 310 에폭 중 테스트 셋에서 고른 최고값이라 절대 정확도가 낙관적이다 (예: inter 3e-3 best 96.72, 마지막 10에폭 평균 96.53, n=4). 팔끼리 비교는 마지막 10에폭 민감도로 확인했다.')
    w('- 유효 런 = 완주 (R19 310, VGG 200 에폭), S30/S1 ≥ 0.19 (원장 배제 규칙), 시드 < 900. 미완주·배제 런은 표 끝 부록에 사유와 함께 남겼다.')
    w('- 같은 스파이크 비교 = `acc = a + b·ln(spikes) + c·[팔] + d·[서버 23]` (ANCOVA). c = inter − 상대 팔. 같은 정확도 스파이크 비 = exp(−c/b) (inter/상대, 1 보다 작으면 inter 가 적게 씀), 95% 구간은 팔 안 재표본 부트스트랩 10,000회.')
    w('- 최소 검출 차 = (t₀.₉₇₅ + t₀.₈₀) × SE. 양측 5%, 검정력 80% 로 잡을 수 있는 가장 작은 차이.')
    w('- 비열등 (baseline 대비) = Δacc 의 90% 구간 하한이 −마진 보다 큼 (한쪽 5%). 마진은 아직 사용자 결정 전이라 ±0.15·±0.2 를 둘 다 적는다. **두 마진은 자료를 본 뒤 함께 적은 것이고 미리 정한 값이 아니다.** n=1 묶음은 합동 시드 SD 로 구간을 그렸지만 판정에 쓰지 않는다.')
    w('- **합동 SD 판정 (2판 추가):** n≤4 묶음은 그룹 안 SD 가 우연히 작으면 Welch 구간이 좁아진다. 그래서 같은 차를 합동 시드 SD (아래) 로 다시 판정해 함께 적는다. 둘이 다르면 "경계" 로 읽는다.')
    w('- 다중비교: 표 안의 p 는 따로 적지 않으면 보정 전이다. 여러 검정을 한 묶음으로 본 곳은 Holm 보정값을 함께 적었다.')
    w('- 시드 SD (정확도, 합동): R19 C10 {:.3f} (자유도 {:.0f}), C100 {:.3f} ({:.0f}), VGGSNN 첫 프레임 {:.3f} ({:.0f}), QK C10 {:.3f} ({:.0f}), QK C100 {:.3f} ({:.0f}).'.format(
        c10['sd_pool'], c10['sd_pool_df'], c100['sd_pool'], c100['sd_pool_df'], dvs['sd_pool'], dvs['sd_pool_df'],
        qk10['sd_pool'], qk10['sd_pool_df'], qk100['sd_pool'], qk100['sd_pool_df']))
    w('')
    w('---')
    w('')

    # ---------------------------------------------------------- 0. 결론
    g3 = G('R19-C10', '3e-3'); g55 = G('R19-C10', '5.5e-3'); g45 = G('R19-C10', '4.5e-3')
    gc2 = G('R19-C100', '2e-3'); gc3 = G('R19-C100', '3e-3'); gc1 = G('R19-C100', '1e-3')
    gq = G('QK-C10', '1.2e-2'); gq2 = G('QK-C10', '2e-2'); gqc = G('QK-C100', '6e-3')
    gd = {k: G('VGG-DVS-f0', k) for k in ('1e-3', '2e-3', '3e-3', '5e-3', '1e-2')}
    l3 = next(g for g in se['curves_last10']['R19-C10']['groups'] if g['knob'] == '3e-3')
    w('## 0. 결론')
    w('')
    rv = R['review']; ni = rv['ni_t1']; nag = rv['npz_axis_groups']; g6 = G('R19-C10', '6e-3')
    ARM3_ = ['ours_inter_ch', 'plain L2', 'BPSR']
    _bestm = {'ours_inter_ch': R['q2']['C10_table1']['ours_inter_ch']['acc'][0], 'BPSR': R['q2']['C10_table1']['BPSR']['acc'][0],
              'plain L2': R['q2']['C10_table1']['plain L2']['acc'][0], 'baseline': R['curves']['R19-C10']['baseline']['mean']}
    gap_l10 = [_bestm[k] - ni[k]['last10_mean'] for k in _bestm]
    sp = rv['sparsity_position']; asep = rv['active_sep']; vt = rv['vgg_trend']
    w('**한 줄:** ours_inter_ch 는 R19 C10 에서 스파이크 −30% (s_count; 테스트 셋 npz 로는 −27%) 까지 baseline 과 비열등이다 — **±0.2 마진이면 best·마지막 10에폭·마지막 에폭 세 기준 모두 성립하고, ±0.15 마진은 경계 위다.** 같은 스파이크에서 BPSR·plain L2 보다 정확도가 높다는 근거도, 같은 정확도에서 스파이크가 적다는 근거도 없다 (R19 C10·C100, 각 n=4 대표점과 곡선 전체). 구조 차이 (같은 스파이크에서 이미지당 활성 뉴런이 적고 채널을 더 끈다) 는 재현되지만 **보조 근거 (B 등급)** 다 — maxnorm 계열 공통 성질이고, 크기가 09-15 에 기각한 크기와 같으며, 뉴런 희소성으로는 Shi 2024 (UPR) 와 거리가 멀다.')
    w('')
    w('1. **baseline 을 지키는 범위 (Q1)**')
    a3 = nag['ours_inter_ch|3e-3']
    w(f'   - R19 C10: ρ 3e-3 (n=4) **−{-g3["spk_pct"]:.1f}% (s_count, 학습 패스, baseline n=6) / {a3["pct_npz"]:+.1f}% (테스트 npz, baseline n=3)** 에서 Δacc {F(g3["delta"]["d"], 3, True)} %p. 90% 구간: Welch {CI(g3["delta"]["ci90"], 3)} (하한이 −0.15 와 {g3["delta"]["ci90"][0] + 0.15:.3f} 차), 합동 SD {CI(g3["delta_pool"]["ci90"], 3)}, 마지막 10에폭 {F(ni["ours_inter_ch"]["last10_welch"]["d"], 3, True)} {CI(ni["ours_inter_ch"]["last10_welch"]["ci90"], 3)}, 마지막 에폭 {F(ni["ours_inter_ch"]["last_welch"]["d"], 3, True)} {CI(ni["ours_inter_ch"]["last_welch"]["ci90"], 3)}. → **±0.2 는 네 판정 모두 성립, ±0.15 는 마지막 에폭에서 못 넘는다 (경계).**')
    w(f'   - −38~−43% (ρ 4e-3·4.5e-3 각 n=3, 5e-3 n=1) 은 판정 불가. **손실이 시작되는 점은 −47~−51% 이고 서버가 섞여 있다:** ρ 5.5e-3 (n=3, −{-g55["spk_pct"]:.1f}%, 3런 모두 138) 은 Welch {F(g55["delta"]["d"], 3, True)} {CI(g55["delta"]["ci90"], 3)} {P(g55["delta"]["p"])} 로 손실이지만 그룹 SD 가 {g55["sd"]:.3f} 로 합동 SD 의 1/3 이라, 합동 SD 로는 {CI(g55["delta_pool"]["ci90"], 3)} {P(g55["delta_pool"]["p"])} 로 손실이 확인되지 않는다. ρ 6e-3 (n=3, −{-g6["spk_pct"]:.1f}%, 3런 모두 23) 은 두 판정 모두 손실 (합동 SD {P(g6["delta_pool"]["p"])}).')
    w(f'   - R19 C100: ρ 2e-3 (n=4) **−{-gc2["spk_pct"]:.1f}%** (s_count, npz 없음) 에서 {F(gc2["delta"]["d"], 3, True)}, Welch {CI(gc2["delta"]["ci90"], 3)} (그룹 SD {gc2["sd"]:.3f}), 합동 SD {CI(gc2["delta_pool"]["ci90"], 3)} → ±0.2 비열등 (두 판정 모두). ρ 3e-3 (n=3, −{-gc3["spk_pct"]:.1f}%) 은 {F(gc3["delta"]["d"], 3, True)}, Welch {CI(gc3["delta"]["ci90"], 3)} {P(gc3["delta"]["p"])}, 합동 SD {CI(gc3["delta_pool"]["ci90"], 3)} {P(gc3["delta_pool"]["p"])} 로 손실.')
    w(f'   - VGGSNN 첫 프레임: ρ 1e-3~1e-2 (각 n=4, −{-gd["1e-3"]["spk_pct"]:.0f}~−{-gd["1e-2"]["spk_pct"]:.0f}%) 모두 baseline {F(dvs["baseline"]["mean"])} ±{F(dvs["baseline"]["sd"])} (n=4) 과 **차가 검출되지 않는다** (Δ {F(min(g["delta"]["d"] for g in gd.values()), 2, True)}~{F(max(g["delta"]["d"] for g in gd.values()), 2, True)}, 모두 p≥{min(g["delta"]["p"] for g in gd.values()):.2f}, 최소 검출 차 {min(g["delta"]["mdd80"] for g in gd.values()):.1f}~{max(g["delta"]["mdd80"] for g in gd.values()):.1f}%p). **다만 점추정과 추세는 손실 쪽이다:** 규제 {rv["vgg_reg_mean_minus_base"]["n_reg"]}런 평균이 baseline 보다 {rv["vgg_reg_mean_minus_base"]["d"]:+.2f}, 24런 (ρ 2e-2 n=1 제외) 의 정확도–ln 스파이크 기울기 {vt["excl_2e-2"]["slope"]:.2f} ±{vt["excl_2e-2"]["se"]:.2f} ({P(vt["excl_2e-2"]["p"])}) 는 −75% 에서 약 {vt["excl_2e-2"]["pred_at_1e-2"]:+.1f}%p 를 가리킨다 (2e-2 포함 n=25 면 {P(vt["incl_2e-2"]["p"])}). 무손실로 읽으면 안 된다.')
    w(f'   - QK C10: ρ 1.2e-2 (n=4) −{-gq["spk_pct"]:.1f}% 에서 {F(gq["delta"]["d"], 3, True)} {CI(gq["delta"]["ci90"])} (90% 하한 {F(gq["delta"]["ci90"][0], 3, True)}: ±0.2 마진은 못 넘고 ±0.25 면 겨우 비열등). ρ 2e-2 (n=1) −{-gq2["spk_pct"]:.1f}% 에서 {F(gq2["delta"]["d"], 2, True)} 이지만 n=1 → 판단 불가.')
    w(f'   - QK C100: ρ 6e-3 (n=4) **−{-gqc["spk_pct"]:.1f}% 에서 이미 {F(gqc["delta"]["d"], 2, True)} {CI(gqc["delta"]["ci90"])}, {P(gqc["delta"]["p"])}** → 무손실 구간을 찾지 못했다.')
    sg = {g['knob']: g for g in sdt['groups']}
    w(f'   - SDTv3 (ImageNet, 각 n=1, 마지막 에폭): ρ 0.1 −{-sg["0.1"]["spk_pct"]:.1f}% 에서 {F(sg["0.1"]["d_raw"], 2, True)}, ρ 0.2 −{-sg["0.2"]["spk_pct"]:.1f}% 에서 {F(sg["0.2"]["d_raw"], 2, True)}, ρ 0.3 −{-sg["0.3"]["spk_pct"]:.1f}% 에서 {F(sg["0.3"]["d_raw"], 2, True)}. ImageNet 시드 SD 를 재지 않아 판단 불가. KD 4런은 진행 중 (10-20~22 완주 예정).')
    w('')
    w('2. **plain L2·BPSR 대비 같은 스파이크 정확도 (Q2) — 차이 검출 안 됨**')
    vb = t1['vs_BPSR']; vl = t1['vs_plain L2']
    bpsr_less = 100 * (1 - t1['BPSR']['spikes'][0] / t1['ours_inter_ch']['spikes'][0])
    w(f'   - 표1 점 (R19 C10, 각 n=4): ours_inter_ch {F(t1["ours_inter_ch"]["acc"][0])} @ {K(t1["ours_inter_ch"]["spikes"][0])}, BPSR {F(t1["BPSR"]["acc"][0])} @ {K(t1["BPSR"]["spikes"][0])}, plain L2 {F(t1["plain L2"]["acc"][0])} @ {K(t1["plain L2"]["spikes"][0])}. **BPSR 이 inter 보다 스파이크 {bpsr_less:.1f}% 적은 점에서 정확도 차가 검출되지 않는다** (inter − BPSR {F(vb["acc"]["d"], 2, True)} {CI(vb["acc"]["ci95"])}, 최소 검출 차 {F(vb["acc"]["mdd80"])}; 기울기로 스파이크 차를 보정하면 {F(vb["adj_acc"], 3, True)} ±{F(vb["adj_se"], 3)}). 동등성 검정은 하지 않았으므로 "같다" 가 아니라 "구분되지 않는다" 다. plain L2 대비 보정 차는 {F(vl["adj_acc"], 2, True)} ±{F(vl["adj_se"])}.')
    nb_ = ni['BPSR']; nl_ = ni['plain L2']; ab_ = nag['BPSR|1e-10']; al_ = nag['plain L2|1e-7']
    w(f'   - baseline (n=6) 대비로는 BPSR 1e-10 (n=4) 도 {t1["vs_baseline"]["BPSR"]["spk_pct"]:+.1f}% (s_count) / {ab_["pct_npz"]:+.1f}% (npz) 에서 {nb_["best_welch"]["d"]:+.3f}, Welch 하한 {nb_["best_welch"]["ci90"][0]:+.4f}, 합동 SD 하한 {nb_["best_pooled"]["ci90"][0]:+.4f}, 마지막 에폭 하한 {nb_["last_welch"]["ci90"][0]:+.3f} → **inter 와 마찬가지로 ±0.15 경계 위, ±0.2 는 성립.** "−30% 에서 무손실" 은 inter 만의 결과가 아니다. plain L2 1e-7 (n=4, {t1["vs_baseline"]["plain L2"]["spk_pct"]:+.1f}% / {al_["pct_npz"]:+.1f}%) 은 {nl_["best_welch"]["d"]:+.3f}, Welch {CI(nl_["best_welch"]["ci90"], 3)} 이지만 합동 SD 로는 {CI(nl_["best_pooled"]["ci90"], 3)} 로 손실 쪽이다.')
    w(f'   - ANCOVA, 300~420K (n={b1["n"]}: inter {b1["by_arm"]["ours_inter_ch"]}, plain L2 {b1["by_arm"]["plain L2"]}, BPSR {b1["by_arm"]["BPSR"]}): inter − BPSR {F(C(b1, "BPSR")["c"], 3, True)} {CI(C(b1, "BPSR")["ci95"], 3)} {P(C(b1, "BPSR")["p"])}, inter − plain L2 {F(C(b1, "plain L2")["c"], 3, True)} {CI(C(b1, "plain L2")["ci95"], 3)} {P(C(b1, "plain L2")["p"])}. 180~420K (n={b3["n"]}): {F(C(b3, "BPSR")["c"], 3, True)} {CI(C(b3, "BPSR")["ci95"], 3)} / {F(C(b3, "plain L2")["c"], 3, True)} {CI(C(b3, "plain L2")["ci95"], 3)}.')
    w(f'   - 같은 정확도 스파이크 비 (180~420K, 기울기 {F(b3["b"], 3)} ±{F(b3["b_se"], 3)} 로 잘 잡히는 구간): inter/BPSR **{F(C(b3, "BPSR")["spk_ratio"], 2)} {CI(C(b3, "BPSR")["boot_ratio_ci95"]).replace("+", "")}**, inter/plain L2 {F(C(b3, "plain L2")["spk_ratio"], 2)} {CI(C(b3, "plain L2")["boot_ratio_ci95"]).replace("+", "")}. 300~420K 는 기울기가 0 과 구분되지 않아 ({F(b1["b"], 2)} ±{F(b1["b_se"], 2)}) 비가 정의되지 않는다 (구간 [{C(b1, "BPSR")["boot_ratio_ci95"][0]:.2f}, {C(b1, "BPSR")["boot_ratio_ci95"][1]:.2f}]).')
    w(f'   - 최소 검출 차: 300~420K 에서 {F(C(b1, "BPSR")["mdd80"])}%p, 180~420K 에서 {F(C(b3, "BPSR")["mdd80"])}%p. BPSR 대비 우위가 있다 해도 95% 상한이 {F(C(b1, "BPSR")["ci95"][1], 2, True)} (300~420K) / {F(C(b3, "BPSR")["ci95"][1], 2, True)}%p (180~420K) 라, 그보다 큰 우위는 이 자료로 **배제**된다.')
    w(f'   - C100 (290~420K, n={cb["n"]}): inter − plain L2 {F(C(cb, "plain L2")["c"], 2, True)} {CI(C(cb, "plain L2")["ci95"])}, inter − BPSR {F(C(cb, "BPSR")["c"], 2, True)} {CI(C(cb, "BPSR")["ci95"])}.')
    w(f'   - QK C10 (plain L2 만, ρ 마다 n=1): 662K 에서 {F(qkp["1.2e-2"]["d"], 2, True)} {CI(qkp["1.2e-2"]["ci95"])}. 524K 에서 inter ρ 2e-2 (n=1) 가 plain L2 보간보다 {F(qkp["2e-2"]["d"], 2, True)} 높지만 n=1 대 n=1 이고, 학습 로그 최고값 기준이면 {F(qkp["2e-2"]["log_d"], 2, True)} 로 줄어든다 → 판단 불가.')
    w('   - **사용자 요구 ("BPSR 대비 같은 정확도면 스파이크 적게 / 같은 스파이크면 정확도 높게") 는 지금 자료로 보일 수 없다.** 자료가 가리키는 것은 차이가 검출되지 않는다는 것이다 (180~420K 의 95% 구간이 ±0.1%p 안, 같은 정확도 스파이크 비 0.9~1.25 안 — 선형·공통 기울기 가정, 동등성 검정은 하지 않음).')
    w('')
    tn = q3['table1']['n']
    tv = q3['table1']['vs']; tm = q3['table1']['mean']
    w('3. **구조 (Q3) — 차이는 재현되지만 보조 근거 (B 등급, 2판에서 내림)**')
    n1 = q3['npz']['B1_300_420K']
    cc10 = q3['csv']['C10_B1_300_420K']; cc100 = q3['csv']['C100_B_290_420K']
    hb = rv['t1_npz_holm']['adj']
    if tv.get('BPSR'):
        w(f'   - 표1 점 npz (테스트 1만 장, inter {tn["ours_inter_ch"]} / BPSR {tn["BPSR"]} / plain L2 {tn["plain L2"]} 런, baseline {q3["table1"]["baseline_n"]}): 죽은 채널 (/4,736) inter {F(tm["ours_inter_ch"]["dead_ch"][0], 1)} vs BPSR {F(tm["BPSR"]["dead_ch"][0], 1)} vs plain L2 {F(tm["plain L2"]["dead_ch"][0], 1)} (baseline {F(tm["baseline"]["dead_ch"][0], 1)}); 지울 수 있는 밀집 MACs (시나리오 a) {F(tm["ours_inter_ch"]["macs_a"][0], 1)} / {F(tm["BPSR"]["macs_a"][0], 1)} / {F(tm["plain L2"]["macs_a"][0], 1)}% (baseline {F(tm["baseline"]["macs_a"][0], 1)}). 단 inter 쪽 npz 스파이크가 6% 많은 점끼리이고, n=4 Welch 24개 검정이라 죽은 채널 (p_Holm {hb["BPSR|dead_ch"]:.2f} / {hb["plain L2|dead_ch"]:.2f})·MACs a ({hb["BPSR|macs_a"]:.2f} / {hb["plain L2|macs_a"]:.2f})·활성 뉴런 ({hb["BPSR|active_img"]:.2f} / {hb["plain L2|active_img"]:.2f}) 은 보정 뒤 유의하지 않다. 주 근거는 아래 같은 스파이크 ANCOVA 다.')
    w(f'   - 같은 스파이크 (npz, 300~420K, n={n1["dead_ch"]["n"]}: inter {n1["dead_ch"]["by_arm"]["ours_inter_ch"]}, plain L2 {n1["dead_ch"]["by_arm"]["plain L2"]}, BPSR {n1["dead_ch"]["by_arm"]["BPSR"]}): 죽은 채널 BPSR 대비 {F(C(n1["dead_ch"], "BPSR")["c"], 1, True)} {CI(C(n1["dead_ch"], "BPSR")["ci95"], 1)} (채널의 {100 * C(n1["dead_ch"], "BPSR")["c"] / 4736:.2f}%p), MACs (a) {F(C(n1["macs_a"], "BPSR")["c"], 2, True)} {CI(C(n1["macs_a"], "BPSR")["ci95"])}%p, 이미지당 활성 뉴런 {F(C(n1["active_img"], "BPSR")["pct"], 1, True)}% {CI(C(n1["active_img"], "BPSR")["pct_ci95"], 1)}.')
    a10 = asep['C10_300_420K']['at']; a100 = asep['C100_290_420K']['at']
    w(f'   - 활성 뉴런 감소는 스파이크 수준에 따라 다르다 (팔마다 기울기를 따로 둔 모형, CSV 지표, BPSR 대비): C10 400K {a10["400000"]["BPSR"]["pct"]:+.1f}%, **342K {a10["342075"]["BPSR"]["pct"]:+.1f}% {CI(a10["342075"]["BPSR"]["pct_ci95"], 1)}**, 310K {a10["310000"]["BPSR"]["pct"]:+.1f}% (n=27); C100 **354K {a100["353882"]["BPSR"]["pct"]:+.1f}% {CI(a100["353882"]["BPSR"]["pct_ci95"], 1)}** (n=25, C100 BPSR 은 n=4 이고 293~342K 에만 있어 342K 위는 외삽). 스파이크를 줄일수록 차가 커진다. 공통 기울기 모형의 숫자 하나 (C10 {F(C(cc10["csv_active_img"], "BPSR")["pct"], 1, True)}%, C100 {F(C(cc100["csv_active_img"], "BPSR")["pct"], 1, True)}%) 는 기울기 차가 유의해 (기울기 차 C10 {P(cc10["csv_active_img"]["slope_diff_F_p"])}, C100 {P(cc100["csv_active_img"]["slope_diff_F_p"])}) 구간 평균일 뿐이다. CSV 와 npz 의 활성 뉴런 비는 팔마다 다르지만 (baseline {rv["csv_over_npz_active"]["baseline"][0]:.3f} n={rv["csv_over_npz_active"]["baseline"][2]}, 규제 팔 {min(rv["csv_over_npz_active"][a_][0] for a_ in ARM3_):.3f}~{max(rv["csv_over_npz_active"][a_][0] for a_ in ARM3_):.3f}), 같은 구간의 팔 대비는 CSV {F(C(cc10["csv_active_img"], "BPSR")["pct"], 1, True)}% (n={cc10["csv_active_img"]["n"]}) 와 npz {F(C(n1["active_img"], "BPSR")["pct"], 1, True)}% (n={n1["active_img"]["n"]}) 로 맞는다.')
    w(f'   - **위치 (2판 추가):** 표1 점에서 inter 는 채널의 {sp["ours_inter_ch"]["alive_channel_pct"][0]:.1f}% 가 살아 있고 (BPSR {sp["BPSR"]["alive_channel_pct"][0]:.1f}%), 테스트 1만 장에서 한 번도 안 쏜 뉴런은 {sp["ours_inter_ch"]["never_fire_neuron_pct"][0]:.1f}% (BPSR {sp["BPSR"]["never_fire_neuron_pct"][0]:.1f}%, baseline {sp["baseline"]["never_fire_neuron_pct"][0]:.1f}%), 이미지당 활성 뉴런은 전체의 {sp["ours_inter_ch"]["active_per_img_pct"][0]:.1f}% (BPSR {sp["BPSR"]["active_per_img_pct"][0]:.1f}%) 다 (각 n=4). Shi 2024 (ICLR, UPR) 는 CIFAR-10 −0.21% 손실에 뉴런 21% 만 남긴다 (09-15 인용) — **뉴런 희소성으로 비교할 위치가 아니다** (09-15 결정 그대로).')
    w(f'   - **등급을 내린 이유:** (1) 활성 뉴런 감소는 maxnorm 계열 공통 성질이다 (10-14 E8: ours_layer·ours_intra_ch 대비 −1.0~−1.2%). (2) 죽은 채널 +{C(n1["dead_ch"], "BPSR")["c"]:.1f}/4,736 ({100 * C(n1["dead_ch"], "BPSR")["c"] / 4736:.2f}%p) 는 09-15 에 "구조적으로 쓸 수 없다" 고 기각한 크기 (+50개, 1.06%p) 와 같다. (3) 10-10 사용자 결정에서 구조 지표는 보조 근거다. (4) 이벤트 구동 하드웨어의 효율 이득이 아니다 (같은 스파이크면 SynOps 차 0.4% 이하, 10-09) — 밀집 연산·가중치 메모리에서만 의미가 있고 실측이 없다.')
    w(f'   - 단계 배분: BPSR 대비 inter 는 conv1 몫이 {F(C(cc10["csv_share_conv1"], "BPSR")["c"], 1, True)}%p (C10) / {F(C(cc100["csv_share_conv1"], "BPSR")["c"], 1, True)}%p (C100) 크다 — conv1 을 덜 줄인다. plain L2 와는 차이 {F(C(cc10["csv_share_conv1"], "plain L2")["c"], 1, True)} / {F(C(cc100["csv_share_conv1"], "plain L2")["c"], 1, True)}%p.')
    w('')
    va = ab['- vmem']; vi = ab['- 1- inversion']
    w('4. **구성요소 (요약, R19 C10, 342K 근처 각 n=4)**')
    ah = rv['abl_holm']['adj']; vx = rv['vmem_x_server']; v4 = rv['vmem_4arm_180_420K']['contrast']
    w(f'   - `- vmem` 은 같은 스파이크에서 inter 곡선보다 낮은 쪽이다: inter − (- vmem) {F(C(va["curve_180_420K"], "- vmem")["c"], 3, True)} {CI(C(va["curve_180_420K"], "- vmem")["ci95"], 3)} {P(C(va["curve_180_420K"], "- vmem")["p"])} (n={va["curve_180_420K"]["by_arm"]["ours_inter_ch"]}/{va["curve_180_420K"]["by_arm"]["- vmem"]}), 마지막 10에폭 기준 {F(C(se["abl_- vmem"], "- vmem")["c"], 3, True)}. **그러나 구성요소 4팔 검정을 Holm 으로 보정하면 p={ah["- vmem"]:.2f} 로 유의하지 않다.** 서버별로는 138 {F(C(inf["vmem_server_138"], "- vmem")["c"], 2, True)}, 23 {F(C(inf["vmem_server_23"], "- vmem")["c"], 2, True)} 이지만 팔×서버 상호작용 p={vx["p"]:.3f} 이고 23 의 - vmem 은 n={vx["n_vmem_by_server"]["23"]} 이라, "138 에서만 보인다" 도 "서버와 무관하다" 도 판정할 수 없다. BPSR·plain L2 를 함께 넣은 4팔 모형 (180~420K) 에서 - vmem − BPSR {F(v4["BPSR"]["c"], 3, True)} {CI(v4["BPSR"]["ci95"], 3)} ({P(v4["BPSR"]["p"])}), - vmem − plain L2 {F(v4["plain L2"]["c"], 3, True)} {CI(v4["plain L2"]["ci95"], 3)} ({P(v4["plain L2"]["p"])}) 로 균등 규제보다 낮은 쪽이다 → vmem 의 몫은 "inter 를 균등 규제보다 낫게" 가 아니라 "inter 가 균등 규제보다 떨어지지 않게" 로 읽힌다 (이것도 보정 전 p).')
    ns3 = rv['noinv_struct']['300_420K_all']; ns18 = rv['noinv_struct']['180_420K_all']; ns3h = rv['noinv_struct']['300_420K_138']
    gp = rv['gini_points']
    w(f'   - `- 1- inversion` (n={vi["curve_180_420K"]["by_arm"]["- 1- inversion"]}; 대표 4런은 모두 ρ 4.5e-3·138 서버) 은 정확도 차가 검출되지 않는다 ({F(C(vi["curve_180_420K"], "- 1- inversion")["c"], 3, True)} {CI(C(vi["curve_180_420K"], "- 1- inversion")["ci95"], 3)}; "그대로" 가 아니라 "차가 검출되지 않음"). 구조는 달라진다: **층 안 뉴런 지니 상승은 사라지고 균등 규제보다도 낮아진다** (대표점 지니 inter 3e-3 {gp["ours_inter_ch|3e-3"][0]:.3f}, BPSR 1e-10 {gp["BPSR|1e-10"][0]:.3f}, plain L2 1e-7 {gp["plain L2|1e-7"][0]:.3f}, - 1- inversion 4.5e-3 {gp["- 1- inversion|4.5e-3"][0]:.3f}, 각 n=4; 같은 스파이크 4팔 모형 - 1- inversion − BPSR {ns3["gini"]["contrast"]["BPSR"]["c"]:+.3f} {CI(ns3["gini"]["contrast"]["BPSR"]["ci95"], 3)}, 300~420K). **활성 뉴런 감소는 사라지지 않는다.** 1− 반전이 설명하는 몫은 inter − BPSR 차의 약 {100 * ns18["share_explained_by_inversion"]:.0f}~{100 * ns3["share_explained_by_inversion"]:.0f}% 이고, - 1- inversion − BPSR 이 300~420K {ns3["active"]["contrast"]["BPSR"]["pct"]:+.2f}% {CI(ns3["active"]["contrast"]["BPSR"]["pct_ci95"], 2)} (n={ns3["active"]["by_arm"]["- 1- inversion"]}), 180~420K {ns18["active"]["contrast"]["BPSR"]["pct"]:+.2f}% {CI(ns18["active"]["contrast"]["BPSR"]["pct_ci95"], 2)} (n={ns18["active"]["by_arm"]["- 1- inversion"]}) 로 남는다 (같은 모형의 inter − BPSR 은 {ns3["inter_minus_bpsr_pct"]:+.1f}% / {ns18["inter_minus_bpsr_pct"]:+.1f}%). 138 서버만 보면 300~420K {ns3h["active"]["contrast"]["BPSR"]["pct"]:+.2f}% {CI(ns3h["active"]["contrast"]["BPSR"]["pct_ci95"], 2)} 로 0 을 포함한다.')
    w(f'   - `- final_step`·`- loss-ratio` 는 정확도 차 검출 안 됨 ({F(C(ab["- final_step"]["curve_180_420K"], "- final_step")["c"], 3, True)}, {F(C(ab["- loss-ratio"]["curve_180_420K"], "- loss-ratio")["c"], 3, True)}; 최소 검출 차 0.14~0.16%p; Holm p {ah["- final_step"]:.2f}, {ah["- loss-ratio"]:.2f}).')
    w('')
    w(f'5. **10-14 대비:** BPSR 대비 정확도 오프셋은 +0.050 (p=0.27) → +0.02 (p=0.64) 로 약해졌고, 표1 점에서는 BPSR 이 스파이크 {bpsr_less:.1f}% 적은 점에서 정확도 차가 검출되지 않는다. 구조 차이 (활성 뉴런·죽은 채널) 는 n=4 표1 점·C100 으로 재현됐지만 2판에서 보조 근거로 내렸다. 자세한 대조는 5절.')
    w('')
    w('---')
    w('')

    # ---------------------------------------------------------- 1. 곡선 표
    w('## 1. 정확도–스파이크 곡선 (Q1)')
    w('')
    w('### 1.0 요약')
    w('')
    w('| 설정 | baseline (n) | 비열등 ±0.15 이 서는 가장 깊은 감소 (n≥3, Welch 와 합동 SD 둘 다) | 비열등 ±0.2 (둘 다) | 손실이 확인된 첫 점 — Welch (90% 상한 < 0) | 손실이 확인된 첫 점 — 합동 SD | 대표점 최소 검출 차 |')
    w('|---|---|---|---|---|---|---|')

    def deepest(cv, mg):
        gs = [g for g in cur[cv]['groups'] if g['n'] >= 3 and g.get('ni', {}).get(mg) and g.get('ni_pool', {}).get(mg)]
        if not gs:
            return '없음'
        g = min(gs, key=lambda g: g['spk_pct'])
        return f'ρ {g["knob"]} (n={g["n"]}) {g["spk_pct"]:+.1f}%, Δ {g["delta"]["d"]:+.3f}, 하한 {g["delta"]["ci90"][0]:+.3f} / {g["delta_pool"]["ci90"][0]:+.3f}'

    def firstloss(cv, key='delta'):
        gs = [g for g in cur[cv]['groups'] if g['n'] >= 3 and key in g and g[key]['ci90'][1] < 0]
        if not gs:
            return '없음 (n≥3 묶음 중)'
        g = max(gs, key=lambda g: g['spk_pct'])
        return f'ρ {g["knob"]} (n={g["n"]}, {"·".join(g["servers"])}) {g["spk_pct"]:+.1f}%, Δ {g[key]["d"]:+.3f} {CI(g[key]["ci90"], 3)}'
    rep = {'R19-C10': '3e-3', 'R19-C100': '2e-3', 'VGG-DVS-f0': '3e-3', 'QK-C10': '1.2e-2', 'QK-C100': '6e-3'}
    nm = {'R19-C10': 'R19 C10', 'R19-C100': 'R19 C100', 'VGG-DVS-f0': 'VGGSNN DVS 첫 프레임', 'QK-C10': 'QK C10', 'QK-C100': 'QK C100'}
    for cv in rep:
        c = cur[cv]; g = G(cv, rep[cv])
        w(f'| {nm[cv]} | {F(c["baseline"]["mean"])} ±{F(c["baseline"]["sd"])} @ {K(c["baseline"]["spikes"])} (n={c["baseline"]["n"]}) | {deepest(cv, "0.15") if "0.15" in g["ni"] else "–"} | {deepest(cv, "0.2")} | {firstloss(cv)} | {firstloss(cv, "delta_pool")} | ρ {rep[cv]} (n={g["n"]}): {F(g["delta"]["mdd80"])}%p |')
    w(f'| SDTv3 (n=1) | {F(sdt["baseline"]["mean"], 3)} @ {K(sdt["baseline"]["spikes"])} (대조, n=1) | 판단 불가 (n=1) | 판단 불가 | 판단 불가 | 판단 불가 | 시드 SD 미측정 |')
    w('')
    w('- 하한 두 값 = Welch / 합동 SD. R19 C10 ρ 3e-3 의 Welch 하한은 −0.15 와 0.002 차이다 — ±0.15 는 경계로 읽는다. 마지막 에폭 기준이면 하한이 {:+.3f} 로 ±0.15 를 못 넘는다 (2.1).'.format(ni['ours_inter_ch']['last_welch']['ci90'][0]))
    w('- R19 C10 의 손실 시작점은 판정 방법에 따라 ρ 5.5e-3 (−47%, 3런 모두 138) 또는 ρ 6e-3 (−51%, 3런 모두 23) 이다. 곡선 끝에 서버가 섞여 있어 "−47~−51%" 로만 쓴다.')
    w('- VGGSNN 은 시드 SD 가 커서 ±0.2 는 물론 ±0.5 마진도 n=4 로 세울 수 없다. "손실이 검출되지 않음" 만 말할 수 있고, 검출 한계가 약 1.1~1.3%p 다. 점추정과 전체 추세는 손실 쪽이다 (결론 1).')
    w('- QK C10 ρ 1.2e-2 의 90% 하한은 −0.249 로 ±0.25 마진이면 비열등이다. QK C100 은 ρ 6e-3 (−10.6%) 에서 이미 손실이므로, 무손실 점을 찾으려면 더 약한 ρ (예: 3e-3) 가 필요하다.')
    w('- 스파이크 축: R19·VGGSNN 은 s_count (학습 패스), QK 는 테스트 셋 fp32 재측정, SDTv3 는 원장 기록값이다. 설정끼리 % 를 직접 비교하지 않는다.')
    w('')

    def crow(r, extra='', lab='baseline', npzcol=False):
        return (f'| {lab} | s{r["seed"]} | {r["server"]} | {F(r.get("train_loss"), 4)} | {F(r.get("train_acc"))} | {F(r.get("val_loss"), 4)} | '
                f'{F(r["val_acc"])} | {K(r["spikes"])} | |' + (' |' if npzcol else '') + f' | | {extra}|')

    def ctable(cv, title, note='', npzcol=False):
        c = cur[cv]
        w(f'### {title}')
        w('')
        if note:
            w(note)
            w('')
        h_npz = ' 테스트 npz 축 (baseline n=3) |' if npzcol else ''
        w(f'| 묶음 | 시드 | 서버 | train loss | train acc | val loss | val acc | spikes | baseline 대비 스파이크 |{h_npz} Δacc, Welch [90%] | Δacc, 합동 SD [90%] | 비열등 (마진: Welch / 합동 SD) |')
        w('|---|---|---|---|---|---|---|---|---|' + ('---|' if npzcol else '') + '---|---|---|')
        b = c['baseline']
        for r in b['rows']:
            w(crow(r, npzcol=npzcol))
        mc = b.get('mean_cols', {})
        w(f'| **baseline 평균** | **n={b["n"]}** | | {F(mc.get("train_loss"), 4)} | {F(mc.get("train_acc"))} | {F(mc.get("val_loss"), 4)} | '
          f'**{F(b["mean"])} ±{F(b.get("sd"))}** | **{K(b["spikes"])}** | – |' + (' – |' if npzcol else '') + ' – | – | |')
        for g in c['groups']:
            for r in g['rows']:
                w(crow(r, lab=f'ρ {g["knob"]}', npzcol=npzcol))
            mc = g['mean_cols']; d = g['delta']
            if g['n'] >= 3:
                ni = ', '.join(f'±{k} {"성립" if v else "아님"} / {"성립" if g["ni_pool"][k] else "아님"}' for k, v in g['ni'].items())
            else:
                ni = 'n=1 판정 안 함' if g['n'] == 1 else 'n=2 판정 안 함'
            sd_ = f' ±{F(g["sd"])}' if g['n'] > 1 else ''
            dp = g.get('delta_pool')
            pcell = f'{dp["d"]:+.3f} {CI(dp["ci90"], 3)} {P(dp["p"])}' if dp else '–'
            ncell = ''
            if npzcol:
                e = R['review']['npz_axis_groups'].get(f'ours_inter_ch|{g["knob"]}')
                ncell = (f' {e["pct_npz"]:+.1f}% (n={e["n"]}) |' if e and e['n'] == g['n'] else ' – |')
            w(f'| **ρ {g["knob"]} 평균** | **n={g["n"]}** | | {F(mc.get("train_loss"), 4)} | {F(mc.get("train_acc"))} | {F(mc.get("val_loss"), 4)} | '
              f'**{F(g["mean"])}{sd_}** | **{K(g["spikes"])}** | {g["spk_pct"]:+.1f}% |{ncell} {d["d"]:+.3f} {CI(d["ci90"], 3)} {P(d["p"])} | {pcell} | {ni} |')
        w('')

    ctable('R19-C10', '1.1 R19 CIFAR-10 (T=4, 310 에폭)',
           '- 진행 중인 반복 (ρ 1.5e-3 s4, 2e-3 s4, 4e-3 s4, 5e-3 s2·s3·s4) 은 부록 A 에 행만 남기고 평균에서 뺐다. 그 런들이 끝나면 −38~−43% 구간이 n=4 로 판정 가능해진다.\n'
           '- "baseline 대비 스파이크" 는 s_count (학습 패스) 축, baseline n=6. "테스트 npz 축" 은 묶음의 모든 런에 npz 가 있을 때만 적었고, baseline 은 npz 가 있는 s11~s13 (138) 3런이다. 같은 3런 기준 s_count 축이면 ρ 3e-3 은 {:+.1f}% 다 (n=6 기준 −30.1% 와 다른 것은 baseline 표본 차이).\n'
           '- 합동 SD = {:.3f} (자유도 {:.0f}). n=1 묶음의 Welch 칸도 합동 SD 로 그린 것이라 두 칸이 같다.'.format(R['review']['npz_axis_groups']['ours_inter_ch|3e-3']['pct_scount_same3'], c10['sd_pool'], c10['sd_pool_df']), npzcol=True)
    ctable('R19-C100', '1.2 R19 CIFAR-100 (T=4, 310 에폭)')
    ctable('VGG-DVS-f0', '1.3 VGGSNN CIFAR10-DVS 첫 프레임 레시피 (T=4, 200 에폭, 138)',
           '- baseline s1 은 수동 패치 런 `_ours/vggdvs-base-f0-s1` (10-15 결정으로 같은 레시피 baseline 에 포함). val 은 992장이라 정확도 눈금이 약 0.1%p. ρ 3e-2 s1 (79.94 @ 31,108) 은 S30/S1=0.169 로 원장 배제 규칙에 걸려 뺐다 (부록 A).\n'
           '- 묶음별로는 차가 검출되지 않지만 (최소 검출 차 1.1~1.3%p), 규제 20런 평균은 baseline 보다 {:+.2f} 이고, 24런 (2e-2 제외) 전체 추세 기울기 {:.2f} ±{:.2f}%p/ln 스파이크 ({}) 는 −75% 에서 {:+.2f} ±{:.2f}%p 를 가리킨다. 2e-2 (n=1) 를 넣으면 기울기 {:.2f} ({}). val 992장에서 200 에폭 중 최고값을 고르므로 best 자체의 잡음도 크다. → "차가 검출되지 않음, 점추정·추세는 손실 쪽".'.format(
               R['review']['vgg_reg_mean_minus_base']['d'], R['review']['vgg_trend']['excl_2e-2']['slope'], R['review']['vgg_trend']['excl_2e-2']['se'], P(R['review']['vgg_trend']['excl_2e-2']['p']),
               R['review']['vgg_trend']['excl_2e-2']['pred_at_1e-2'], R['review']['vgg_trend']['excl_2e-2']['pred_se'], R['review']['vgg_trend']['incl_2e-2']['slope'], P(R['review']['vgg_trend']['incl_2e-2']['p'])))
    ctable('QK-C10', '1.4 QKFormer CIFAR-10 (T=4, 410 에폭, fp32 재측정, 138)',
           '- train acc 는 QK 기록에 없다 (–). train loss·val loss 는 각 런 summary.csv 의 best eval_top1 행 (규제항 포함). ρ 1e-3~7e-3·2e-2 는 시드 42 한 런씩.')
    ctable('QK-C100', '1.5 QKFormer CIFAR-100 (T=4, 410 에폭, fp32 재측정, 138)')
    w('### 1.6 SDTv3 ImageNet-1k (공식 체크포인트 30 에폭 파인튜닝, T=1, 23, 각 n=1)')
    w('')
    w('val acc 는 마지막 에폭 (head 출력, 50,000장). best 열은 참고용 (에폭 1 근처).')
    w('')
    w('| 묶음 | 서버 | train loss | train acc | val loss | val acc (마지막) | spikes | 대조 대비 스파이크 | Δacc (원값) | best val acc |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    r = sdt['baseline']['rows'][0]
    w(f'| 대조 (규제 없음) | 23 | {F(r["train_loss"], 4)} | {F(r["train_acc"])} | {F(r["val_loss"], 4)} | {F(r["val_acc"], 3)} | {K(r["spikes"])} | – | – | {F(r["val_best"], 3)} |')
    for g in sdt['groups']:
        r = g['rows'][0]
        w(f'| ρ {g["knob"]} | 23 | {F(r["train_loss"], 4)} | {F(r["train_acc"])} | {F(r["val_loss"], 4)} | {F(r["val_acc"], 3)} | {K(r["spikes"])} | {g["spk_pct"]:+.1f}% | {g["d_raw"]:+.3f} | {F(r["val_best"], 3)} |')
    for r in sdt['running']:
        lab_ = '대조+KD' if r['key'] == 'base' else f'ρ {r["knob"]}'
        w(f'| {lab_} (진행 중, {r["reason"]}) | 23 | {F(r["train_loss"], 4)} | {F(r["train_acc"])} | {F(r["val_loss"], 4)} | ({F(r["val_acc"], 3)}) | ({K(r["spikes"])}) | – | – | – |')
    w('')
    w('- KD 런의 정확도는 (head + head_kd)/2 평균 출력 기준이라 비KD 와 눈금이 다르다 (10-15 정정). KD 끼리만 비교한다.')
    w('- ρ 3.0 이 ρ 1.0 보다 스파이크가 많은 것은 n=1 이라 원인을 말할 수 없다.')
    w('')
    w('---')
    w('')

    # ---------------------------------------------------------- 2. 같은 스파이크
    w('## 2. 같은 스파이크 비교 — plain L2·BPSR (Q2)')
    w('')
    w('### 2.1 표1 점 (R19 C10, 각 n=4)')
    w('')
    w('| 방법 | ρ/λ | 시드 | 서버 | train loss | train acc | val loss | val acc | spikes |')
    w('|---|---|---|---|---|---|---|---|---|')
    T1R = {'ours_inter_ch': ('3e-3', 'ours_ch'), 'BPSR': ('1e-10', 'bpsr'), 'plain L2': ('1e-7', 'l2')}
    for a, (kn, key) in T1R.items():
        runs = R['t1_rows'][a]
        for r in runs:
            w(f'| {a} | {kn} | s{r["seed"]} | {r["server"]} | {F(r["train_loss"], 4)} | {F(r["train_acc"])} | {F(r["val_loss"], 4)} | {F(r["val_acc"])} | {K(r["spikes"])} |')
        m = lambda k: float(np.mean([r[k] for r in runs]))
        w(f'| **{a} 평균** | {kn} | **n={len(runs)}** | | {F(m("train_loss"), 4)} | {F(m("train_acc"))} | {F(m("val_loss"), 4)} | **{F(m("val_acc"))} ±{F(float(np.std([r["val_acc"] for r in runs], ddof=1)))}** | **{K(m("spikes"))}** |')
    w('')
    w('| 비교 (각 n=4) | 정확도 차 [95%] | 최소 검출 차 | 스파이크 차 (inter/상대 − 1) | 스파이크 차를 기울기 (280~420K b={:.3f}) 로 보정한 정확도 차 | 마지막 10에폭 평균 기준 정확도 차 [95%] |'.format(b2['b']))
    w('|---|---|---|---|---|---|')
    for o in ('BPSR', 'plain L2'):
        v = t1['vs_' + o]; sv = se['C10_table1']['vs_' + o]
        w(f'| inter − {o} | {F(v["acc"]["d"], 3, True)} {CI(v["acc"]["ci95"])} {P(v["acc"]["p"])} | {F(v["acc"]["mdd80"])} | {v["spk_pct"]:+.1f}% | {F(v["adj_acc"], 3, True)} ±{F(v["adj_se"], 3)} | {F(sv["d"], 3, True)} {CI(sv["ci95"])} {P(sv["p"])} |')
    w('')
    w(f'- 표1 의 세 점은 스파이크가 다르다: inter 가 BPSR 보다 {t1["vs_BPSR"]["spk_pct"]:+.1f}%, plain L2 보다 {t1["vs_plain L2"]["spk_pct"]:+.1f}% 많다 (거꾸로 말하면 BPSR 이 inter 보다 {bpsr_less:.1f}% 적다). 기울기로 스파이크 차를 보정한 inter − BPSR 은 {t1["vs_BPSR"]["adj_acc"]:+.3f} ±{t1["vs_BPSR"]["adj_se"]:.3f} 로 **차가 검출되지 않는다** (n=4 대 n=4 의 최소 검출 차 {t1["vs_BPSR"]["acc"]["mdd80"]:.2f}%p, 95% 구간이 ±0.15 보다 넓어 동등하다고는 말할 수 없다).')
    w('')
    w('**baseline (n=6) 대비 비열등 — 기준별 (90% 구간, 소수 셋째 자리)**')
    w('')
    w('| 팔 (n=4) | s_count 축 | 테스트 npz 축 (baseline n=3) | best, Welch | best, 합동 SD | 마지막 10에폭 평균, Welch | 마지막 에폭, Welch | ±0.15 (네 기준) | ±0.2 (네 기준) |')
    w('|---|---|---|---|---|---|---|---|---|')
    for a_, gk in (('ours_inter_ch', 'ours_inter_ch|3e-3'), ('BPSR', 'BPSR|1e-10'), ('plain L2', 'plain L2|1e-7')):
        e = ni[a_]; ks = ('best_welch', 'best_pooled', 'last10_welch', 'last_welch')
        def jud(mg):
            ok = [e[k]['ci90'][0] > -mg for k in ks]
            return '모두 성립' if all(ok) else ('모두 아님' if not any(ok) else f'{sum(ok)}/4 성립')
        cells = ' | '.join(f'{e[k]["d"]:+.3f} {CI(e[k]["ci90"], 3)}' for k in ks)
        w(f'| {a_} | {t1["vs_baseline"][a_]["spk_pct"]:+.1f}% | {nag[gk]["pct_npz"]:+.1f}% | {cells} | {jud(0.15)} | {jud(0.2)} |')
    w('')
    w(f'- 하한이 마진에 얼마나 붙어 있나: inter Welch {ni["ours_inter_ch"]["best_welch"]["ci90"][0]:+.4f} (−0.15 와 {ni["ours_inter_ch"]["best_welch"]["ci90"][0] + 0.15:.4f} 차), BPSR Welch {ni["BPSR"]["best_welch"]["ci90"][0]:+.4f} ({ni["BPSR"]["best_welch"]["ci90"][0] + 0.15:.4f} 차), BPSR 합동 SD {ni["BPSR"]["best_pooled"]["ci90"][0]:+.4f} (합동 SD 를 0.110 으로 반올림하면 −0.150 아래로 내려간다). **"inter·BPSR 둘 다 ±0.15 비열등" 은 "둘 다 ±0.15 경계 위" 로 읽어야 한다.** 쓸 수 있는 문장: "±0.2 마진이면 네 기준 모두 비열등, ±0.15 는 경계".')
    w(f'- 마지막 10에폭 평균의 절대값: baseline {ni["baseline"]["last10_mean"]:.2f} (n=6), inter 3e-3 {ni["ours_inter_ch"]["last10_mean"]:.2f}, BPSR 1e-10 {ni["BPSR"]["last10_mean"]:.2f}, plain L2 1e-7 {ni["plain L2"]["last10_mean"]:.2f} (각 n=4). best 보다 {min(gap_l10):.2f}~{max(gap_l10):.2f}%p 낮다 (best 는 테스트 셋에서 고른 최고값).')
    w('- 즉 −30% 대 비열등은 BPSR 도 (더 깊은 −34% / npz −32% 에서) 같은 경계로 성립한다. plain L2 1e-7 은 합동 SD·마지막 에폭 기준으로 손실 쪽이다.')
    w('')
    w('### 2.2 ANCOVA — 구간별 (R19 C10)')
    w('')
    w('| 구간 | n (inter / plain L2 / BPSR) | 기울기 b (%p / ln 스파이크) | 기울기 차 p | inter − plain L2 [95%] | 최소 검출 차 | inter − BPSR [95%] | 최소 검출 차 | 같은 정확도 스파이크 비 inter/plain L2 [부트 95%] | inter/BPSR [부트 95%] |')
    w('|---|---|---|---|---|---|---|---|---|---|')

    def arow(lab, a):
        cl = C(a, 'plain L2'); cbp = C(a, 'BPSR')

        def rr_(e):
            if 'boot_ratio_ci95' not in e:
                return '–'
            return f'{e.get("spk_ratio", float("nan")):.2f} [{e["boot_ratio_ci95"][0]:.2f}, {e["boot_ratio_ci95"][1]:.2f}]'
        return (f'| {lab} | {a["by_arm"]["ours_inter_ch"]} / {a["by_arm"]["plain L2"]} / {a["by_arm"]["BPSR"]} | {F(a["b"], 3)} ±{F(a["b_se"], 3)} {P(a["b_p"])} | '
                f'{F(a.get("slope_diff_F_p"), 2)} | {F(cl["c"], 3, True)} {CI(cl["ci95"], 3)} {P(cl["p"])} | {F(cl["mdd80"])} | '
                f'{F(cbp["c"], 3, True)} {CI(cbp["ci95"], 3)} {P(cbp["p"])} | {F(cbp["mdd80"])} | {rr_(cl)} | {rr_(cbp)} |')
    w(arow('300~420K (10-14 주 구간)', b1))
    w(arow('280~420K', b2))
    w(arow('180~420K', b3))
    w(arow('300~420K, 마지막 10에폭 평균', s1))
    w(arow('180~420K, 마지막 10에폭 평균', s3))
    w('')
    nb = q2['C10_B3_180_420K']
    w(f'- 서버 항을 빼도 같다 (300~420K: plain L2 {F(C(q2["C10_B1_300_420K"]["lin_noserver"], "plain L2")["c"], 3, True)}, BPSR {F(C(q2["C10_B1_300_420K"]["lin_noserver"], "BPSR")["c"], 3, True)}). 180~420K 에 2차항을 넣어도 {F(C(nb["quad"], "plain L2")["c"], 3, True)} / {F(C(nb["quad"], "BPSR")["c"], 3, True)}. 서버 효과 (23 − 138) 는 {F(b1.get("srv23"), 3, True)}%p.')
    xb = q2['C10_at342K']
    w(f'- 팔마다 기울기를 따로 둔 모형으로 342,075 (inter 3e-3 평균) 에서 비교 (280~420K, n={xb["n"]}): inter − plain L2 {F(C(xb, "plain L2")["c"], 3, True)} {CI(C(xb, "plain L2")["boot_ci95"], 3)}, inter − BPSR {F(C(xb, "BPSR")["c"], 3, True)} {CI(C(xb, "BPSR")["boot_ci95"], 3)} (부트스트랩).')
    ib = inf['C10_B1_inter_vs_BPSR']; il = inf['C10_B1_inter_vs_L2']
    w(f'- 하나 빼기 (300~420K, {ib["n"]}런): inter − BPSR 은 {F(ib["c_min"], 3, True)}~{F(ib["c_max"], 3, True)} 로 0.05%p 를 넘지 않는다. inter − plain L2 는 {F(il["c_min"], 3, True)}~{F(il["c_max"], 3, True)} 이고 {il["n_sig05"]}/{il["n"]} 경우만 p<0.05 — plain L2 5e-8 s1·s3 (96.77·96.78) 을 빼면 커지고 plain L2 1e-7 s2 (96.41) 를 빼면 작아진다.')
    w('- 300~420K 는 10-05 에 결과를 보고 고른 구간이다 (10-14 지적). 미리 정한 구간이 아니므로 이 구간의 p 는 탐색적이다.')
    w(f'- 검정력: 300~420K 에서 inter − plain L2 +0.08 을 80% 로 잡으려면 잔차 SD {F(b1["resid_sd"], 3)} 기준 팔당 약 {int(math.ceil(2 * ((1.96 + 0.84) * b1["resid_sd"] / max(abs(C(b1, "plain L2")["c"]), 1e-9)) ** 2))}런, BPSR 대비 +0.02 는 약 {int(math.ceil(2 * ((1.96 + 0.84) * b1["resid_sd"] / max(abs(C(b1, "BPSR")["c"]), 1e-9)) ** 2))}런이 필요하다 (두 집단 근사). n≤4 규칙으로는 정확도 우위를 확정하는 실험을 설계할 수 없다.')
    w('')
    w('### 2.3 R19 C100')
    w('')
    w('| 구간 | n (inter / plain L2 / BPSR) | 기울기 b | inter − plain L2 [95%] | 최소 검출 차 | inter − BPSR [95%] | 최소 검출 차 | 같은 정확도 스파이크 비 inter/plain L2 | inter/BPSR |')
    w('|---|---|---|---|---|---|---|---|---|')
    for lab, a in [('290~420K', cb), ('220~420K', q2['C100_B_220_420K']['lin']), ('290~420K, 마지막 10에폭', se['C100_B_290_420K'])]:
        cl = C(a, 'plain L2'); cbp = C(a, 'BPSR')
        w(f'| {lab} | {a["by_arm"]["ours_inter_ch"]} / {a["by_arm"]["plain L2"]} / {a["by_arm"]["BPSR"]} | {F(a["b"], 3)} ±{F(a["b_se"], 3)} | {F(cl["c"], 3, True)} {CI(cl["ci95"], 3)} {P(cl["p"])} | {F(cl["mdd80"])} | '
          f'{F(cbp["c"], 3, True)} {CI(cbp["ci95"], 3)} {P(cbp["p"])} | {F(cbp["mdd80"])} | '
          f'{F(cl.get("spk_ratio"))} [{F(cl.get("boot_ratio_ci95", [None, None])[0])}, {F(cl.get("boot_ratio_ci95", [None, None])[1])}] | {F(cbp.get("spk_ratio"))} [{F(cbp.get("boot_ratio_ci95", [None, None])[0])}, {F(cbp.get("boot_ratio_ci95", [None, None])[1])}] |')
    w('')
    x1 = q2['C100_at354K']
    w(f'- 354K (inter 2e-3 n=4 평균) 에서 기울기를 따로 둔 비교: inter − plain L2 {F(C(x1, "plain L2")["c"], 3, True)} {CI(C(x1, "plain L2")["boot_ci95"], 3)} (n={x1["n"]}).')
    w('- C100 BPSR 은 1.5e-10 (n=1, 342K) 과 3e-10 (n=3, 293~303K) 뿐이라 inter 대표점 (354K) 근처의 BPSR 은 한 런이다. BPSR 대비는 구간 아래쪽 끝 비교에 가깝다. BPSR C100 반복은 사용자 지시로 보류 중이다.')
    w('')
    w('### 2.4 QK C10 (plain L2 만)')
    w('')
    a = q2['QK_C10_500_800K']['lin']; a2 = q2['QK_C10_450_950K']['lin']
    w(f'- ANCOVA 500~800K (n={a["n"]}: inter {a["by_arm"]["ours_inter_ch"]}, plain L2 {a["by_arm"]["plain L2"]}): inter − plain L2 {F(C(a, "plain L2")["c"], 3, True)} {CI(C(a, "plain L2")["ci95"], 3)} {P(C(a, "plain L2")["p"])}, 최소 검출 차 {F(C(a, "plain L2")["mdd80"])}. **기울기 차 p={F(a["slope_diff_F_p"], 3)}** — plain L2 는 스파이크를 줄일수록 정확도가 빨리 떨어지고 inter 는 평탄해서 공통 기울기 모형이 맞지 않는다. 450~950K (n={a2["n"]}) 는 {F(C(a2, "plain L2")["c"], 3, True)} {CI(C(a2, "plain L2")["ci95"], 3)}, 기울기 차 p={F(a2["slope_diff_F_p"], 3)}.')
    w('- 그래서 점별로 본다 (plain L2 는 ρ 마다 시드 42 한 런, 이웃 두 점을 ln 스파이크로 직선 보간):')
    w('')
    w('| inter 점 | n | spikes | inter (재측정) | plain L2 보간 (재측정) | 차 [95%] | inter (학습 로그 최고) | plain L2 보간 (로그) | 차 (로그) |')
    w('|---|---|---|---|---|---|---|---|---|')
    for kn in ('1.2e-2', '2e-2'):
        e = qkp[kn]
        w(f'| ρ {kn} | {e["n_inter"]} | {K(e["spikes"])} | {F(e["inter"])} | {F(e["l2_interp"])} | {F(e["d"], 2, True)} {CI(e["ci95"])} | {F(e.get("log_inter"))} | {F(e.get("log_l2_interp"))} | {F(e.get("log_d"), 2, True)} |')
    w('')
    oth = q2['QK_C10_interp']['l2_3e-6_other_seeds_log']
    w(f'- 524K 의 +0.60 은 plain L2 3e-6 s42 의 재측정값 (95.38, 자기 학습 로그 95.74 보다 −0.36) 에 크게 기댄다. 같은 λ 의 다른 시드 학습 로그 최고는 {", ".join(f"{k.split()[-1]} {v[0]} ({v[1]})" for k, v in oth.items())} 이고 재측정이 없다. 재측정이 학습 로그보다 낮은 폭이 런마다 0.00~0.46 으로 흩어져 (QK 30런) 단일 런 비교는 이 잡음보다 작은 차이를 말할 수 없다. → **판단 불가.** 이 점은 inter 가 plain L2 보다 나을 수 있는 유일한 후보라 반복 가치가 있다 (6절).')
    w('')
    w('---')
    w('')

    # ---------------------------------------------------------- 3. 구조
    w('## 3. 구조 — 스파이크 감소가 어디에 어떻게 나뉘는가 (Q3)')
    w('')
    w('**지표 정의** (`struct_compare_261007` 과 같음)')
    w('- 죽은 채널: 테스트 1만 장 전체에서 공간 모든 위치가 한 번도 안 쏜 채널 (/4,736 = conv 4,480 + fc1 256). "완전 침묵 채널" 과 같은 말.')
    w('- MACs 제거 %: 죽은 채널을 물리 제거할 때 줄어드는 밀집 MACs 비율. 시나리오 a = 잔차 합류점 (`_out_n`) 채널은 지우지 않음 (논문 보고용), b = 모두 지움.')
    w('- 이미지당 활성 뉴런: 이미지 한 장에서 T 스텝 동안 한 번이라도 쏜 뉴런 수 (전체 1,442,048). ln 으로 회귀하고 % 로 환산.')
    w('- 채널 지니 (층 스파이크 가중): 층 안 채널별 총 스파이크의 지니. 0 = 고르게, 1 = 한 채널에 몰림. 뉴런 지니: 전체 뉴런의 총 스파이크 지니.')
    w('- 스텝/활성쌍: 뉴런이 어떤 이미지에서 쐈을 때 T=4 중 평균 몇 스텝 쐈나.')
    w('- 단계 몫: 전체 스파이크 중 conv1 (1층)·conv2 (6층, 32×32)·conv3 (6층, 16×16)·conv4 (4층, 8×8) 의 비율 (%).')
    w('- CSV 지표 (모든 런, reg_neuron_detail.csv · reg_detail.csv 의 best 에폭 행): 활성 뉴런 = Σ층 뉴런 수 × (1 − 표본별 0 스파이크 비율), 층 안 뉴런 지니 (층 스파이크 가중), 단계 몫. **2판 정정: 이 CSV 는 "학습 중 평가 기록" 이 아니라 그 에폭 학습 배치에서 누적한 통계다** (s_count 와 같은 경로, 정의 절).')
    w('- npz 지표는 테스트 1만 장을 추론 모드로 한 번 돌린 값이다. CSV 와 npz 는 데이터 (증강된 학습 셋 대 테스트 셋) 와 모드가 다르다.')
    w('')
    w('### 3.1 표1 점 npz (R19 C10)')
    w('')
    if tv.get('BPSR'):
        w(f'baseline n={q3["table1"]["baseline_n"]} (s11~s13), inter 3e-3 n={tn["ours_inter_ch"]}, BPSR 1e-10 n={tn["BPSR"]}, plain L2 1e-7 n={tn["plain L2"]}. 값은 평균 ±SD. 차는 Welch (같은 스파이크 보정 없음, 스파이크 차는 첫 행).')
        w('')
        w('| 지표 | baseline | ours_inter_ch | BPSR | plain L2 | inter − BPSR [95%] | inter − plain L2 [95%] |')
        w('|---|---|---|---|---|---|---|')
        lab = {'npz_spk': 'npz 스파이크 (이미지당)', 'dead_ch': '죽은 채널 (/4,736)', 'macs_a': 'MACs 제거 a (%)', 'macs_b': 'MACs 제거 b (%)',
               'active_img': '이미지당 활성 뉴런', 'gini_ch_w': '채널 지니 (가중)', 'gini_neuron': '뉴런 지니', 'steps_per_active': '스텝/활성쌍',
               'share_conv1': 'conv1 몫 (%)', 'share_conv2': 'conv2 몫 (%)', 'share_conv3': 'conv3 몫 (%)', 'share_conv4': 'conv4 몫 (%)'}
        dd = {'npz_spk': 0, 'dead_ch': 1, 'macs_a': 2, 'macs_b': 2, 'active_img': 0, 'gini_ch_w': 3, 'gini_neuron': 3, 'steps_per_active': 3}
        for m, l_ in lab.items():
            d = dd.get(m, 2)
            cell = lambda a_: f'{tm[a_][m][0]:,.{d}f} ±{tm[a_][m][1]:,.{d}f}' if tm[a_][m][1] is not None else f'{tm[a_][m][0]:,.{d}f}'
            vb_ = tv['BPSR'][m]; vl_ = tv['plain L2'][m]
            w(f'| {l_} | {cell("baseline")} | {cell("ours_inter_ch")} | {cell("BPSR")} | {cell("plain L2")} | {vb_["d"]:+,.{d}f} [{vb_["ci95"][0]:+,.{d}f}, {vb_["ci95"][1]:+,.{d}f}] {P(vb_["p"])} | {vl_["d"]:+,.{d}f} [{vl_["ci95"][0]:+,.{d}f}, {vl_["ci95"][1]:+,.{d}f}] {P(vl_["p"])} |')
        w('')
        w(f'- **p 는 보정 전이다.** 이 표는 지표 12개 × 비교 2개 = {rv["t1_npz_holm"]["m"]}개 Welch 검정 (n=4 대 n=4) 이다. Holm 보정 뒤에도 p<0.05 인 것: ' + ', '.join(f'{k.split("|")[1]} ({k.split("|")[0]}, {v:.3f})' for k, v in sorted(rv['t1_npz_holm']['adj'].items(), key=lambda kv: kv[1]) if v < 0.05) + '. 죽은 채널·MACs·활성 뉴런은 보정 뒤 0.09~0.26 이라, 이 표는 방향만 보이고 근거는 3.2 의 같은 스파이크 ANCOVA (n=25) 다.')
        w(f'- inter 는 BPSR·plain L2 보다 npz 스파이크가 {100 * (tm["ours_inter_ch"]["npz_spk"][0] / tm["BPSR"]["npz_spk"][0] - 1):+.1f}% / {100 * (tm["ours_inter_ch"]["npz_spk"][0] / tm["plain L2"]["npz_spk"][0] - 1):+.1f}% 많은데도 죽은 채널이 많고 활성 뉴런이 적다. 스파이크가 많을수록 죽은 채널은 줄고 활성 뉴런은 느는 방향이므로 (3.2 기울기), 같은 스파이크로 맞추면 차이는 더 커지는 쪽이다 — 이 표의 차는 보수적이다.')
        w('- 10-19 에 추출한 4런: inter 3e-3 s4, BPSR 1e-10 s4, plain L2 1e-7 s3 (23, 읽기 전용 사본)·s4. npz 테스트 정확도는 원장 best 와 0.01%p 안 (96.65 / 96.81 / 96.57 / 96.68) — val 셋이 곧 테스트 셋이라는 확인이기도 하다.')
        w('')
        w('**뉴런 희소성의 위치 (2판 추가, 표1 점 npz)**')
        w('')
        w('| 지표 | baseline (n=3) | ours_inter_ch (n=4) | BPSR (n=4) | plain L2 (n=4) |')
        w('|---|---|---|---|---|')
        for k_, l_ in (('alive_channel_pct', '산 채널 비율 (%)'), ('never_fire_neuron_pct', '테스트 1만 장에서 한 번도 안 쏜 뉴런 (%)'), ('active_per_img_pct', '이미지당 활성 뉴런 (% of 1,442,048)')):
            w(f'| {l_} | ' + ' | '.join(f'{sp[a_][k_][0]:.1f} ±{sp[a_][k_][1]:.1f}' for a_ in ('baseline', 'ours_inter_ch', 'BPSR', 'plain L2')) + ' |')
        w('')
        w('- 비교 기준: Shi 2024 (ICLR, UPR) 는 CIFAR-10 에서 −0.21% 손실에 뉴런 21% 만 남기고, −2.19% 에 8.23% 를 남긴다 (09-15 decisions 인용, 이번에 원문 재확인 안 함). inter 는 구조적으로 지울 수 있는 뉴런 (한 번도 안 쏜 뉴런) 이 약 20% 라, 뉴런 희소성 축에서는 경쟁 위치가 아니다. 09-15 결정 "뉴런 희소성으로는 이길 수 없다" 를 바꿀 자료가 아니다.')
        w('')
    else:
        w('(npz 추출 미완 — 표1 점 n 이 모자라 생략)')
        w('')
    w('### 3.2 같은 스파이크 npz ANCOVA (R19 C10)')
    w('')
    w('x = ln(npz 스파이크), 서버 항 포함. 구간은 원장 스파이크로 고름. 칸 = inter − 상대 [95%] p.')
    w('')
    w('| 지표 | 구간 | n (inter / plain L2 / BPSR) | inter − plain L2 | inter − BPSR | 기울기 차 p |')
    w('|---|---|---|---|---|---|')
    for m in ('dead_ch', 'macs_a', 'macs_b', 'active_img', 'gini_ch_w', 'gini_neuron', 'steps_per_active', 'share_conv1', 'share_conv2', 'share_conv3', 'share_conv4'):
        for bl, lab_ in (('B1_300_420K', '300~420K'), ('B3_180_420K', '180~420K')):
            a = q3['npz'][bl][m]
            if not a:
                continue

            def cc(o):
                e = C(a, o)
                if 'pct' in e:
                    return f'{e["pct"]:+.1f}% [{e["pct_ci95"][0]:+.1f}, {e["pct_ci95"][1]:+.1f}] {P(e["p"])}'
                d = 1 if m == 'dead_ch' else (3 if m.startswith('gini') or m == 'steps_per_active' else 2)
                return f'{e["c"]:+.{d}f} [{e["ci95"][0]:+.{d}f}, {e["ci95"][1]:+.{d}f}] {P(e["p"])}'
            w(f'| {m} | {lab_} | {a["by_arm"]["ours_inter_ch"]} / {a["by_arm"]["plain L2"]} / {a["by_arm"]["BPSR"]} | {cc("plain L2")} | {cc("BPSR")} | {F(a.get("slope_diff_F_p"), 3)} |')
    w('')
    w('- 기울기 차 p 가 작은 지표는 스파이크를 더 줄일수록 inter 와 상대의 차이가 벌어진다는 뜻이다 (180~420K 의 차가 300~420K 보다 큰 이유).')
    w('')
    w('### 3.3 CSV 구조 지표 — 모든 런 (R19 C10·C100)')
    w('')
    cna = rv['csv_over_npz_active']
    w(f'npz 가 없는 런까지 넣는다. CSV 활성 뉴런과 npz 활성 뉴런은 r={q3["csv_vs_npz_active"]["r"]:.4f} (n={q3["csv_vs_npz_active"]["n"]}, 같은 런) 이지만, 이 r 은 baseline 부터 강규제까지 범위가 넓어서 커진 값이다. 비 (CSV/npz) 는 팔마다 다르다: baseline {cna["baseline"][0]:.3f} (n={cna["baseline"][2]}), inter {cna["ours_inter_ch"][0]:.3f} ±{cna["ours_inter_ch"][1]:.3f} (n={cna["ours_inter_ch"][2]}), BPSR {cna["BPSR"][0]:.3f} ±{cna["BPSR"][1]:.3f} (n={cna["BPSR"][2]}), plain L2 {cna["plain L2"][0]:.3f} ±{cna["plain L2"][1]:.3f} (n={cna["plain L2"][2]}). 그래서 "같은 양" 이 아니다. 근거가 되는 것은 같은 구간의 팔 대비가 두 측정에서 맞는다는 점이다 (C10 300~420K BPSR 대비 CSV {C(cc10["csv_active_img"], "BPSR")["pct"]:+.1f}% n={cc10["csv_active_img"]["n"]}, npz {C(n1["active_img"], "BPSR")["pct"]:+.1f}% n={n1["active_img"]["n"]}).')
    w('')
    w('- **공통 기울기 모형의 숫자는 구간 평균이다.** 활성 뉴런은 팔마다 기울기가 달라 (기울기 차 p 아래 표), 스파이크 수준을 정해 읽어야 한다:')
    w('')
    w('| 설정·구간 | n (inter / plain L2 / BPSR) | BPSR 이 있는 스파이크 범위 | 400K: inter − BPSR | 342K / 354K: inter − BPSR | 310K: inter − BPSR | 342K / 354K: inter − plain L2 |')
    w('|---|---|---|---|---|---|---|')
    for k_, lab_, mid in (('C10_300_420K', 'C10 300~420K', '342075'), ('C10_180_420K', 'C10 180~420K', '342075'), ('C100_290_420K', 'C100 290~420K', '353882')):
        e = asep[k_]; at = e['at']
        def pc(x0, o):
            v = at[x0][o]; return f'{v["pct"]:+.1f}% {CI(v["pct_ci95"], 1)}'
        w(f'| {lab_} | {e["by_arm"]["ours_inter_ch"]} / {e["by_arm"]["plain L2"]} / {e["by_arm"]["BPSR"]} | {K(e["x_range"]["BPSR"][0])}~{K(e["x_range"]["BPSR"][1])} | {pc("400000", "BPSR")} | {pc(mid, "BPSR")} | {pc("310000", "BPSR")} | {pc(mid, "plain L2")} |')
    w('')
    w('- 팔마다 기울기를 따로 둔 모형 (서버 항 포함, 해석적 95% 구간). 스파이크를 줄일수록 차가 커진다. C100 의 BPSR 은 n=4 이고 293~342K 에만 있어 400K 칸은 외삽이다.')
    w('')
    w('| 지표 | 설정·구간 | n (inter / plain L2 / BPSR) | inter − plain L2 [95%] | inter − BPSR [95%] |')
    w('|---|---|---|---|---|')
    for m, l_ in (('csv_active_img', '이미지당 활성 뉴런'), ('csv_gini_layer_w', '층 안 뉴런 지니'), ('csv_share_conv1', 'conv1 몫 (%p)'),
                  ('csv_share_conv2', 'conv2 몫 (%p)'), ('csv_share_conv3', 'conv3 몫 (%p)'), ('csv_share_conv4', 'conv4 몫 (%p)')):
        for bl, lab_ in (('C10_B1_300_420K', 'C10 300~420K'), ('C10_B3_180_420K', 'C10 180~420K'), ('C100_B_290_420K', 'C100 290~420K'), ('C100_B_220_420K', 'C100 220~420K')):
            a = q3['csv'][bl][m]
            if not a:
                continue

            def cc(o):
                e = C(a, o)
                if 'pct' in e:
                    return f'{e["pct"]:+.1f}% [{e["pct_ci95"][0]:+.1f}, {e["pct_ci95"][1]:+.1f}] {P(e["p"])}'
                d = 3 if 'gini' in m else 2
                return f'{e["c"]:+.{d}f} [{e["ci95"][0]:+.{d}f}, {e["ci95"][1]:+.{d}f}] {P(e["p"])}'
            w(f'| {l_} | {lab_} | {a["by_arm"]["ours_inter_ch"]} / {a["by_arm"]["plain L2"]} / {a["by_arm"]["BPSR"]} | {cc("plain L2")} | {cc("BPSR")} |')
    w('')
    cbs = q3['csv_base']
    w(f'- 기준: baseline C10 (n={cbs["C10"]["csv_active_img"][2]}) 활성 뉴런 {K(cbs["C10"]["csv_active_img"][0])}, 층 안 지니 {F(cbs["C10"]["csv_gini_layer_w"][0], 3)}; C100 (n={cbs["C100"]["csv_active_img"][2]}) {K(cbs["C100"]["csv_active_img"][0])}, {F(cbs["C100"]["csv_gini_layer_w"][0], 3)}.')
    w('')
    w('### 3.4 해석')
    w('')
    w(f'- 같은 스파이크에서 inter 는 **더 적은 뉴런이 더 몰아서** 쏜다 (활성 뉴런 342K 에서 BPSR 대비 {asep["C10_300_420K"]["at"]["342075"]["BPSR"]["pct"]:+.1f}%, 층 안 지니 +0.05). BPSR·plain L2 는 발화하는 뉴런의 스텝을 고르게 깎는다 (10-09 의 스텝/활성쌍 결과와 같은 방향). 이 모양 차이는 C10 과 C100 에서 비슷한 크기다 (C100 354K {asep["C100_290_420K"]["at"]["353882"]["BPSR"]["pct"]:+.1f}%).')
    w('- **이 모양 차이의 대부분은 inter 고유가 아니다.** 활성 뉴런 감소는 maxnorm 계열 (ours_intra_ch·ours_layer) 공통이고 inter 와 그 둘의 차는 −1.0~−1.2% 뿐이다 (10-14 E8, 인용). 구성요소로는 1− 반전이 inter − BPSR 차의 약 60~70% 를 설명한다 (4절).')
    w('- 채널 단위로 보면 그 차이의 일부가 "채널 전체가 꺼짐" 으로 나타난다 (죽은 채널·MACs). 다만 채널 사망은 inter 스파이크 감소의 약 9% 뿐이고 (10-14 인용, n=19), 대부분은 산 채널 안의 연속적인 감소다. 크기도 채널의 1.2%p 로 09-15 에 기각한 크기 (1.06%p) 와 같다.')
    w(f'- 단계 배분은 BPSR 과 1~2%p 안에서만 다르다 (inter 가 conv1 을 덜, conv2·conv3 를 조금 더 줄임; npz 표1 점 conv1 몫 inter {F(tm["ours_inter_ch"]["share_conv1"][0], 1)}% vs BPSR {F(tm["BPSR"]["share_conv1"][0], 1)}%, baseline {F(tm["baseline"]["share_conv1"][0], 1)}%). plain L2 와는 300~420K 에서 차이가 없다. 층 배분은 주장의 근거가 될 만큼 크지 않다.')
    w('- **정확도와는 연결되지 않는다.** 같은 스파이크에서 정확도 차가 검출되지 않는데 구조만 다르므로, 구조 차이는 "같은 정확도·같은 스파이크에서 다른 해를 찾는다" 로만 쓸 수 있다. 10-14 에 적은 팔 안 상관 (죽은 채널이 많을수록 정확도가 낮음, r=−0.38, 인용) 도 그대로다.')
    w('- **이벤트 구동 하드웨어의 효율 이득이 아니다** (같은 스파이크면 SynOps 차 0.4% 이하, 10-09 인용). 밀집 연산·가중치 메모리에서만 의미가 있고, 채널을 실제로 지운 뒤의 정확도·지연은 재지 않았다.')
    w('')
    w('---')
    w('')

    # ---------------------------------------------------------- 4. 구성요소
    w('## 4. 구성요소 (ablation) — 정확도와 CSV 구조 요약 (R19 C10)')
    w('')
    w('대표점은 ours_inter_ch ρ 3e-3 (96.72 ±0.12 @ 342,075, n=4) 와 같은 구간에 착지하도록 ρ 를 고른 각 n=4 묶음. 곡선 비교는 inter 27런 (180~420K) 과 해당 팔 전부 (다른 ρ 포함), 서버 항 포함.')
    w('')
    w('| 팔 | 대표 ρ/λ | n | val acc | spikes | 팔 − inter 3e-3 정확도 차 [95%] | 스파이크 차 (팔/inter 3e-3) | 곡선 대비 (inter − 팔) [95%] 보정 전 p (n) | Holm p (4팔) | 마지막 10에폭 기준 | 같은 정확도 스파이크 비 inter/팔 | 활성 뉴런 inter − 팔 (%) | 층 안 지니 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for a_ in ('- vmem', '- final_step', '- 1- inversion', '- loss-ratio'):
        v = ab[a_]; pt = v['point']; cv_ = C(v['curve_180_420K'], a_); sl = C(se['abl_' + a_], a_); ca = C(v['csv_active_180_420K'], a_)
        w(f'| {a_} | {pt["knob"]} | {pt["acc"][2]} | {F(pt["acc"][0])} ±{F(pt["acc"][1])} | {K(pt["spikes"][0])} | {F(pt["vs_inter3e-3"]["d"], 3, True)} {CI(pt["vs_inter3e-3"]["ci95"])} | {pt["spk_pct_vs_inter3e-3"]:+.1f}% | '
          f'{F(cv_["c"], 3, True)} {CI(cv_["ci95"], 3)} {P(cv_["p"])} ({v["curve_180_420K"]["by_arm"]["ours_inter_ch"]}/{v["curve_180_420K"]["by_arm"][a_]}) | {R["review"]["abl_holm"]["adj"][a_]:.2f} | {F(sl["c"], 3, True)} {CI(sl["ci95"], 3)} | '
          f'{F(cv_.get("spk_ratio"))} [{F(cv_.get("boot_ratio_ci95", [None, None])[0])}, {F(cv_.get("boot_ratio_ci95", [None, None])[1])}] | {F(ca["pct"], 1, True)} {CI(ca["pct_ci95"], 1)} | {F(pt["csv"]["csv_gini_layer_w"][0], 3)} |')
    w(f'| (ours_inter_ch) | 3e-3 | 4 | {F(t1["ours_inter_ch"]["acc"][0])} ±{F(t1["ours_inter_ch"]["acc"][1])} | {K(t1["ours_inter_ch"]["spikes"][0])} | – | – | – | – | – | – | – | {F(q3["csv_table1"]["ours_inter_ch"]["csv_gini_layer_w"][0], 3)} |')
    w('')
    vx = rv['vmem_x_server']; v4 = rv['vmem_4arm_180_420K']
    w(f'- `- vmem` 서버별: 138 {F(C(inf["vmem_server_138"], "- vmem")["c"], 3, True)} {CI(C(inf["vmem_server_138"], "- vmem")["ci95"], 3)} (n={inf["vmem_server_138"]["by_arm"]["ours_inter_ch"]}/{inf["vmem_server_138"]["by_arm"]["- vmem"]}), 23 {F(C(inf["vmem_server_23"], "- vmem")["c"], 3, True)} {CI(C(inf["vmem_server_23"], "- vmem")["ci95"], 3)} (n={inf["vmem_server_23"]["by_arm"]["ours_inter_ch"]}/{inf["vmem_server_23"]["by_arm"]["- vmem"]}). 팔×서버 상호작용 {vx["coef"]:+.3f} ±{vx["se"]:.3f}, p={vx["p"]:.3f} (n={vx["n"]}) — 23 의 - vmem 이 n={vx["n_vmem_by_server"]["23"]} 이라 서버 차이를 판정할 수 없다. 하나 빼기 범위 {F(inf["vmem_curve"]["c_min"], 3, True)}~{F(inf["vmem_curve"]["c_max"], 3, True)}, 최대 p {F(inf["vmem_curve"]["p_max"], 3)}.')
    w(f'- `- vmem` 4팔 모형 (180~420K, n={v4["n"]}: - vmem {v4["by_arm"]["- vmem"]}, inter {v4["by_arm"]["ours_inter_ch"]}, BPSR {v4["by_arm"]["BPSR"]}, plain L2 {v4["by_arm"]["plain L2"]}): - vmem − inter {C(v4, "ours_inter_ch")["c"]:+.3f} {CI(C(v4, "ours_inter_ch")["ci95"], 3)} {P(C(v4, "ours_inter_ch")["p"])}, − BPSR {C(v4, "BPSR")["c"]:+.3f} {CI(C(v4, "BPSR")["ci95"], 3)} {P(C(v4, "BPSR")["p"])}, − plain L2 {C(v4, "plain L2")["c"]:+.3f} {CI(C(v4, "plain L2")["ci95"], 3)} {P(C(v4, "plain L2")["p"])} (보정 전). vmem 을 빼면 균등 규제보다 낮아지는 쪽이다.')
    w('')
    w('**- 1- inversion 의 구조 — 4팔 같은 스파이크 모형 (기준 = - 1- inversion, CSV 지표)**')
    w('')
    w('| 구간·서버 | n (- 1- inv / inter / BPSR / plain L2) | 활성 뉴런: - 1- inv − inter | − BPSR | − plain L2 | 같은 모형 inter − BPSR | 1− 반전이 설명하는 몫 | 지니: - 1- inv − BPSR | − plain L2 |')
    w('|---|---|---|---|---|---|---|---|---|')
    for k_, lab_ in (('300_420K_all', '300~420K, 두 서버'), ('180_420K_all', '180~420K, 두 서버'), ('300_420K_138', '300~420K, 138 만'), ('180_420K_138', '180~420K, 138 만')):
        e = rv['noinv_struct'][k_]; ac = e['active']['contrast']; gc = e['gini']['contrast']; ba = e['active']['by_arm']
        w(f'| {lab_} | {ba["- 1- inversion"]} / {ba["ours_inter_ch"]} / {ba["BPSR"]} / {ba["plain L2"]} | {ac["ours_inter_ch"]["pct"]:+.1f}% {CI(ac["ours_inter_ch"]["pct_ci95"], 1)} | {ac["BPSR"]["pct"]:+.2f}% {CI(ac["BPSR"]["pct_ci95"], 2)} | {ac["plain L2"]["pct"]:+.1f}% {CI(ac["plain L2"]["pct_ci95"], 1)} | {e["inter_minus_bpsr_pct"]:+.1f}% | {100 * e["share_explained_by_inversion"]:.0f}% | {gc["BPSR"]["c"]:+.3f} {CI(gc["BPSR"]["ci95"], 3)} | {gc["plain L2"]["c"]:+.3f} {CI(gc["plain L2"]["ci95"], 3)} |')
    w('')
    w(f'- 1− 반전을 빼면 **지니 상승은 사라지고 균등 규제보다도 낮아진다** (균등 규제보다 고르게 쏜다). 반전이 없으면 가장 많이 쏘는 채널이 가장 큰 가중 (sc/max = 1) 을 받으므로 (`run_paper.py` ch_noinv 주석), 고르게 쏘는 쪽으로 가는 것과 방향이 맞다 (기전은 확인하지 않음). **활성 뉴런 감소는 줄지만 BPSR 대비 {rv["noinv_struct"]["300_420K_all"]["active"]["contrast"]["BPSR"]["pct"]:+.1f}~{rv["noinv_struct"]["180_420K_all"]["active"]["contrast"]["BPSR"]["pct"]:+.1f}% 가 남는다.** 1판의 "활성 뉴런 감소와 지니 상승이 사라진다", "지니가 baseline 수준" 은 틀렸다 — baseline 지니 ({F(q3["csv_base"]["C10"]["csv_gini_layer_w"][0], 3)}) 는 489K 의 점이라 같은 스파이크 비교가 아니었고, 활성 뉴런 대비는 다른 구간의 숫자를 나란히 놓은 것이었다.')
    w(f'- 한계: - 1- inversion 은 5런 중 4런이 ρ 4.5e-3 한 점·138 서버라 이 팔의 기울기를 따로 추정할 수 없다 (구조 지표는 팔마다 기울기가 다르다, 4개 모형의 기울기 차 p 최대 {max(e["active"]["slope_diff_F_p"] for e in rv["noinv_struct"].values()):.3f}). 138 만 보면 300~420K 의 BPSR 대비는 0 을 포함한다. 런: ' + ', '.join(f'ρ {k} s{sd} ({sv}, {sk:,.0f})' for k, sd, sv, sk in rv['noinv_runs']) + '.')
    w('- 요소별 기전 (침묵 뉴런, 채널 사망, 학습 시간) 은 `analysis/ablation_261019.*` 를 본다.')
    w('')
    w('---')
    w('')

    # ---------------------------------------------------------- 5. 변화 + 주장 계층
    w('## 5. 10-14 결론 대비 변화와 주장 계층')
    w('')
    w('### 5.1 무엇이 바뀌었나')
    w('')
    w('| 항목 | 10-14 (inter_justification) | 10-19 (이 문서) | 방향 |')
    w('|---|---|---|---|')
    w(f'| BPSR 대비 같은 스파이크 정확도 (300~420K) | +0.050 [−0.040, +0.139] p=0.27 (inter 12 / BPSR 6) | {F(C(b1, "BPSR")["c"], 3, True)} {CI(C(b1, "BPSR")["ci95"], 3)} {P(C(b1, "BPSR")["p"])} (12 / 7) | 약해짐 — BPSR 1e-10 s4 96.82 추가 |')
    w(f'| BPSR 표1 점 | 96.66 ±0.07 @ 319,481 (n=3) | {F(t1["BPSR"]["acc"][0])} ±{F(t1["BPSR"]["acc"][1])} @ {K(t1["BPSR"]["spikes"][0])} (n=4) | BPSR 이 inter 보다 스파이크 {bpsr_less:.1f}% 적은 점에서 정확도 차 검출 안 됨 |')
    w(f'| plain L2 표1 점 | 96.55 ±0.13 @ 325,908 (n=3) | {F(t1["plain L2"]["acc"][0])} ±{F(t1["plain L2"]["acc"][1])} @ {K(t1["plain L2"]["spikes"][0])} (n=4) | 거의 같음 |')
    w(f'| plain L2 대비 (300~420K) | +0.084 p=0.054 (다른 팔 합동 모형) | {F(C(b1, "plain L2")["c"], 3, True)} {CI(C(b1, "plain L2")["ci95"], 3)} {P(C(b1, "plain L2")["p"])}; 마지막 10에폭 {F(C(s1, "plain L2")["c"], 3, True)} {P(C(s1, "plain L2")["p"])} | 그대로 경계, 넓은 구간에서는 0 |')
    w(f'| 같은 정확도 스파이크 비 inter/BPSR | 1.01 [0.87, 1.15] (200~420K) | {F(C(b3, "BPSR")["spk_ratio"])} [{C(b3, "BPSR")["boot_ratio_ci95"][0]:.2f}, {C(b3, "BPSR")["boot_ratio_ci95"][1]:.2f}] (180~420K) | 같음 (우위 없음) |')
    w(f'| R19 C100 대표점 | ρ 2e-3 80.65 @ 353,882 (n=4); ρ 1e-3 n=3 | ρ 2e-3 그대로; ρ 1e-3 n=4 {F(gc1["mean"])} @ {K(gc1["spikes"])} ({gc1["spk_pct"]:+.1f}%, Δ {gc1["delta"]["d"]:+.2f}) | 같음 |')
    w(f'| R19 C100 plain L2·BPSR 대비 | +0.007 [−0.154, +0.168] (다른 팔 합동, n=25) | plain L2 {F(C(cb, "plain L2")["c"], 2, True)}, BPSR {F(C(cb, "BPSR")["c"], 2, True)} (plain L2 2e-7 n=4 추가) | 같음 (차이 없음) |')
    w(f'| 구조: 활성 뉴런 (BPSR 대비) | −5.7% (C10 npz, n=36, 5팔 모형) | C10 {F(C(cc10["csv_active_img"], "BPSR")["pct"], 1, True)}% (CSV, n=27), C100 {F(C(cc100["csv_active_img"], "BPSR")["pct"], 1, True)}% (n=25); 342K/354K 에서 {asep["C10_300_420K"]["at"]["342075"]["BPSR"]["pct"]:+.1f}% / {asep["C100_290_420K"]["at"]["353882"]["BPSR"]["pct"]:+.1f}% | C100 로 재현. 단 2판에서 B 로 내림 (maxnorm 공통, Shi 2024) |')
    if tv.get('BPSR'):
        w(f'| 구조: 죽은 채널 (BPSR 대비, 300~420K) | +49.0 [+30.2, +67.8] (n=36) | {F(C(n1["dead_ch"], "BPSR")["c"], 1, True)} {CI(C(n1["dead_ch"], "BPSR")["ci95"], 1)} (n={n1["dead_ch"]["n"]}, 세 팔 모형) | 재현. 표1 n=4 Welch 는 Holm 뒤 비유의, 크기는 09-15 기각과 같음 → B |')
    w(f'| VGGSNN DVS | 평균 프레임 레시피, 각 n=1, 판단 불가 | 첫 프레임 레시피 baseline n=4, ρ 5개 n=4: −75% 까지 차이 검출 안 됨 (검출 한계 ~1.2%p) | 새 자료, 단 해상도 낮음 |')
    w(f'| QK C10 | ρ 1.2e-2 −29.2% Δ −0.06 (n=4); 사다리 n=1 | 같음 + ρ 2e-2 −44% Δ −0.02 (n=1) | 새 점 (n=1) |')
    w(f'| QK C100 | ρ 6e-3 n=1 −10.9% Δ −0.10 | ρ 6e-3 n=4 Δ {F(gqc["delta"]["d"], 2, True)} {P(gqc["delta"]["p"])} | **약해짐: −10% 에서 이미 손실** |')
    w(f'| SDTv3 | ρ 0.3·1.0·3.0 (n=1) | ρ 0.05·0.1·0.2 추가 (n=1): −6% 까지 차 없음, −11% 부터 −0.19 | 새 점 (n=1, 판단 불가) |')
    w(f'| - vmem 효과 | −0.096 [−0.188, −0.005], n=8, 138 에서만 | {F(-C(va["curve_180_420K"], "- vmem")["c"], 3, True)} (부호를 10-14 와 맞춤), n=9; Holm p={rv["abl_holm"]["adj"]["- vmem"]:.2f}, 서버 상호작용 p={rv["vmem_x_server"]["p"]:.3f} | 약해짐 (보정 뒤 비유의, 서버 차 판정 불가) |')
    w(f'| - 1- inversion 구조 | 10-14: ρ 1e-2 n=1, 정확도만 (+0.054) | 지니 상승 사라짐, 활성 뉴런 감소는 BPSR 대비 {rv["noinv_struct"]["180_420K_all"]["active"]["contrast"]["BPSR"]["pct"]:+.1f}% 남음 (n=5) | 새 자료 (1판 문구 정정) |')
    w(f'| 측정 축 | 10-14: npz 축 −28.3% 를 함께 적음 | s_count (학습 패스) −30.1% / 테스트 npz −27.4% (baseline n=3) | 2판에서 다시 함께 적음 |')
    w('')
    w('### 5.2 주장 계층 (10-19)')
    w('')
    w('**A. 강함 — 경계 문구와 함께 쓸 수 있다**')
    w('')
    w(f'1. **R19 C10 에서 −30% 까지 baseline 과 비열등 (±0.2 마진).** ρ 3e-3 (n=4), 스파이크 {g3["spk_pct"]:+.1f}% (s_count, 학습 패스) / {nag["ours_inter_ch|3e-3"]["pct_npz"]:+.1f}% (테스트 npz, baseline n=3). Δ {g3["delta"]["d"]:+.3f}; 90% 하한 Welch {g3["delta"]["ci90"][0]:+.3f}, 합동 SD {g3["delta_pool"]["ci90"][0]:+.3f}, 마지막 10에폭 {ni["ours_inter_ch"]["last10_welch"]["ci90"][0]:+.3f}, 마지막 에폭 {ni["ours_inter_ch"]["last_welch"]["ci90"][0]:+.3f}. 함께 적을 것: **±0.15 는 경계** (마지막 에폭에서 못 넘음), 마진은 사후에 적은 값, BPSR 1e-10 (n=4) 도 {t1["vs_baseline"]["BPSR"]["spk_pct"]:+.1f}% / {nag["BPSR|1e-10"]["pct_npz"]:+.1f}% 에서 같은 경계로 비열등, plain L2 1e-7 (n=4) 은 합동 SD·마지막 에폭 기준 손실 쪽. "유일" 은 쓰지 않는다. 절대 정확도는 best (테스트 셋 선택) 라 낙관적이다.')
    w(f'2. **R19 C100 에서 −21.5% 까지 비열등** (ρ 2e-3, n=4, ±0.2 마진, Welch·합동 SD 둘 다; s_count 축, npz 없음). −31% (ρ 3e-3, n=3) 에서는 {gc3["delta"]["d"]:+.3f} 손실 (합동 SD {P(gc3["delta_pool"]["p"])}).')
    w('')
    w('**B. 단서가 붙어야 쓸 수 있다 (부록·보조 문장)**')
    w('')
    w(f'3. *(1판 A3 에서 내림)* **같은 스파이크에서 이미지당 활성 뉴런이 균등 규제보다 적다** — 342K 에서 BPSR 대비 {asep["C10_300_420K"]["at"]["342075"]["BPSR"]["pct"]:+.1f}% (C10, n=27), 354K 에서 {asep["C100_290_420K"]["at"]["353882"]["BPSR"]["pct"]:+.1f}% (C100, n=25); 스파이크를 줄일수록 커진다. 함께 적을 것: maxnorm 계열 공통 성질 (ours_layer·ours_intra_ch 대비 −1.0~−1.2%, 10-14 E8); 1− 반전이 그 차의 약 60~70% 를 설명; Shi 2024 (UPR) 는 −0.21% 손실에 뉴런 21% 만 남기는데 inter 는 한 번도 안 쏜 뉴런이 약 {sp["ours_inter_ch"]["never_fire_neuron_pct"][0]:.0f}% — 뉴런 희소성으로 경쟁할 위치가 아니다; 이벤트 구동 하드웨어 이득 아님; 10-10 결정대로 보조 근거.')
    if tv.get('BPSR'):
        w(f'4. *(1판 A4 에서 내림)* **R19 C10 에서 같은 스파이크에 채널을 더 끈다** (죽은 채널 BPSR 대비 {F(C(n1["dead_ch"], "BPSR")["c"], 1, True)} = 채널의 {100 * C(n1["dead_ch"], "BPSR")["c"] / 4736:.2f}%p, MACs a {F(C(n1["macs_a"], "BPSR")["c"], 2, True)}%p, npz n=25). 함께 적을 것: 09-15 에 같은 크기 (+50개, 1.06%p) 를 "구조적으로 쓸 수 없다" 고 기각한 이력; inter 도 채널의 {sp["ours_inter_ch"]["alive_channel_pct"][0]:.0f}% 가 살아 있음; 표1 n=4 Welch 는 Holm 뒤 비유의; baseline 도 MACs a {F(tm["baseline"]["macs_a"][0], 1)}% 를 이미 지울 수 있음; SynOps 이득 없음; 실제로 지운 뒤의 정확도·지연 측정 없음; C100·QK·DVS 는 재지 않음.')
    w('5. *(1판 A5 에서 내림)* **스파이크 감소 방향은 R19·QK·VGGSNN·SDTv3·Speck 칩에서 같다.** 함께 적을 것: 벌점을 걸면 스파이크가 주는 것은 어느 방법에서나 기대되는 결과라 기여 주장이 아니다; 칩 전력 −35% (10-15 인용) 는 **학습 n=1 런 하나를 반복 측정**한 값이다; 무손실 폭은 과제가 어려울수록 준다 (C10 −30% → C100 −21% → QK C100 −10% 에서 이미 손실).')
    w(f'6. **BPSR·plain L2 와 같은 스파이크에서 정확도 차가 검출되지 않는다.** 180~420K 에서 BPSR 대비 {F(C(b3, "BPSR")["c"], 3, True)} {CI(C(b3, "BPSR")["ci95"], 3)}, plain L2 대비 {F(C(b3, "plain L2")["c"], 3, True)} {CI(C(b3, "plain L2")["ci95"], 3)} (n=63; 선형·공통 기울기·best 기준). 동등성 검정은 하지 않았다. "정확도를 잃지 않고 구조가 다르다" 의 앞부분으로만 쓴다.')
    w(f'7. VGGSNN 첫 프레임: −75% (ρ 1e-2, n=4) 까지 baseline 과 **차가 검출되지 않는다.** 함께 적을 것: 검출 한계 ~1.2%p (시드 SD {dvs["sd_pool"]:.2f}); 점추정·추세는 손실 쪽 (−75% 에서 약 {vt["excl_2e-2"]["pred_at_1e-2"]:+.1f}%p, 규제 20런 평균 {rv["vgg_reg_mean_minus_base"]["d"]:+.2f}); 첫 프레임 입력은 정적 입력 (메모리 dvs-input-protocol). 무손실로 쓰지 않는다.')
    w(f'8. QK C10: −29% (ρ 1.2e-2, n=4) 에서 Δ {F(gq["delta"]["d"], 2, True)} {CI(gq["delta"]["ci90"])} — ±0.25 마진이면 비열등. QK 스파이크는 테스트 셋 재측정 축.')
    w(f'9. `- vmem` 은 같은 스파이크에서 정확도가 낮은 쪽이다 ({F(C(va["curve_180_420K"], "- vmem")["c"], 3, True)}, n=27/9, 보정 전 {P(C(va["curve_180_420K"], "- vmem")["p"])}). 함께 적을 것: **구성요소 4팔 Holm 보정 뒤 p={rv["abl_holm"]["adj"]["- vmem"]:.2f} 로 유의하지 않음**; 서버 차이는 판정 불가 (23 의 - vmem n={rv["vmem_x_server"]["n_vmem_by_server"]["23"]}, 상호작용 p={rv["vmem_x_server"]["p"]:.3f}); 효과 방향은 BPSR·plain L2 보다 낮아지는 쪽 → "vmem 이 inter 를 균등 규제보다 떨어지지 않게 한다" 로만 쓴다. "필수 요소" 로 쓰지 않는다.')
    w(f'10. `- 1- inversion`: 정확도 차 검출 안 됨 (n=5, 4런은 한 ρ·138). 1− 반전을 빼면 지니 상승은 사라지고 균등 규제보다도 낮아진다; 활성 뉴런 감소는 줄지만 BPSR 대비 {rv["noinv_struct"]["300_420K_all"]["active"]["contrast"]["BPSR"]["pct"]:+.1f}~{rv["noinv_struct"]["180_420K_all"]["active"]["contrast"]["BPSR"]["pct"]:+.1f}% 가 남는다.')
    w('11. 학습 비용이 다른 경쟁 범위 팔과 같다 (10-14 E20, 인용).')
    w('')
    w('**C. 쓰면 안 되는 문장**')
    w('')
    w('- BPSR 보다 같은 스파이크에서 정확도가 높다 / 같은 정확도에서 스파이크가 적다. "BPSR 대비 −17%" 류의 수치. (표1 점에서는 오히려 BPSR 이 스파이크 {:.1f}% 적은 점에서 정확도 차가 검출되지 않는다.)'.format(bpsr_less))
    w('- inter 와 BPSR 이 같은 스파이크에서 "정확도가 같다" (동등성 검정을 하지 않았다 — "차가 검출되지 않는다" 로 쓴다).')
    w('- 구조 지표 (활성 뉴런 −6%, 죽은 채널 +56) 를 헤드라인·기여로 쓰는 것. 뉴런 희소성으로 Shi 2024 와 겨루는 문장.')
    w('- "1− 반전을 빼면 활성 뉴런 감소가 사라진다", "지니가 baseline 수준으로 돌아간다" (1판 문구, 틀림).')
    w('- "vmem 이 필수", "vmem 이 inter 를 균등 규제보다 낫게 만든다", "vmem 효과는 138 서버에서만 보인다" (판정 불가).')
    w('- 절대 스파이크 수와 "baseline 대비 −30%" 를 축 표기 없이 쓰는 것 (s_count 는 학습 패스 통계, 09-15 보류 유효).')
    w('- plain L2 보다 정확도가 높다 (300~420K +0.08 p=0.077 은 결과를 보고 고른 구간·넓히면 0).')
    w('- QK C10 저스파이크 (−44%) 에서 plain L2 보다 낫다 (n=1 대 n=1, 측정 기준에 따라 +0.60 → +0.24).')
    w('- QK C100 무손실. QK C100 은 −10.6% 에서 −0.30%p (n=4).')
    w('- VGGSNN 에서 inter 가 정확도를 올린다 (ρ 1e-3 +0.18, 시드 SD 0.5 안).')
    w('- SDTv3 무손실 범위 (n=1, 시드 SD 미측정).')
    w('- 구조 차이가 정확도 이점의 원인이다. 이벤트 구동 하드웨어에서 같은 스파이크로 더 적은 에너지.')
    w('- 10-14 C 항목 (−30% 유일 무손실, intra·layer 보다 나은 경쟁 범위, 생물학 근거, "채널 경쟁 가중은 새롭다") 은 그대로 유효.')
    w('')
    w('---')
    w('')
    w('## 6. 다음 측정 제안 (실행은 승인 후)')
    w('')
    w('| 순위 | 항목 | 무엇을 가르나 | 비용 | 등급 |')
    w('|---|---|---|---|---|')
    w('| 1 | 진행 중 R19 C10 inter 반복 (4e-3 s4, 5e-3 s2~4, 1.5e-3·2e-3 s4) 완주 뒤 이 스크립트 재실행 | −38~−43% 구간 비열등 판정 | 재실행 수 분 | 1 |')
    w('| 2 | QK C10 ρ 2e-2 시드 43~45 (대기 스크립트 작성됨) + plain L2 3e-6 s43 재측정 (s44 는 23) | QK 저스파이크에서 plain L2 대비 우위가 실재하는가 — 이번 분석에서 우위 후보가 남은 유일한 점 | 학습 3런 × 약 33시간 + 재측정 수 분 | 3 (학습), 재측정은 GPU 승인 |')
    w('| 3 | R19 C100 npz 추출 (inter 2e-3 n=4, BPSR 3e-10 n=3, plain L2 2e-7 n=4) + 표1 점에서 죽은 채널을 실제로 지운 뒤 재학습 없이 정확도·실측 지연 | 구조 주장 (B3·B4) 을 다시 올릴 근거가 있는가 — C100 에서도 서는가, 지우면 실제로 빨라지는가 | 11런 × 약 17분 + 지연 측정 | GPU 승인 |')
    w('| 4 | QK C100 더 약한 ρ (예: 3e-3) | QK C100 의 무손실 점이 있는가 | 1런 33시간 (n=1 탐색) | 3 |')
    w('| 5 | 동등성 마진 결정 (±0.15 / ±0.2) | A1·A2·B6 의 문장 형태. ±0.15 를 쓰려면 여유가 0.002 뿐이다 | 문서 | 사용자 결정 |')
    w('| 6 | - 1- inversion 을 다른 ρ (예: 3e-3·6e-3) 로, 23 서버에서도 1~2런 | 이 팔의 기울기와 서버 효과 — 지금은 4런이 한 ρ·138 | 런당 약 30시간 | 3 |')
    w('| 7 | - vmem 을 23 서버에서 더 (다른 ρ) | "138 에서만" 인지 판정 — 지금 23 의 - vmem 은 n=3 (ρ 셋 각 1런) | 런당 약 30시간 | 3 |')
    w('| 8 | (보류 권고) hold 중인 inter 3e-3 s5·baseline s17~19 완주로 n=5 / n=9 | ±0.15 경계 여유를 늘림 | 4런 × 30시간 | 3, **반복 상한 n=4 (10-17 사용자 지시) 와 충돌 — 사용자 결정** |')
    w('')
    w('BPSR 대비 정확도 우위를 n≤4 로 확정하는 실험은 제안하지 않는다 (2.2 의 검정력 계산 — 팔당 수십~수백 런).')
    w('')
    w('---')
    w('')
    w('## 부록 A. 평균에서 뺀 런 (사유)')
    w('')
    w('| 런 | 서버 | 에폭 | val acc | spikes | 사유 |')
    w('|---|---|---|---|---|---|')
    for r in R['excluded_r19'] + R['excluded_vgg']:
        w(f'| {r["tag"]} | {r["server"]} | {r.get("epochs")} | {F(r.get("val_acc"))} | {K(r.get("spikes"))} | {r["reason"]} |')
    w('')
    w('## 부록 B. 재현')
    w('')
    w('- `~/anaconda3/envs/venv_1/bin/python analysis/method_261019.py` (약 2분, 부트스트랩 10,000회 고정 난수 20261019).')
    w('- 입력 스냅숏·세종 CSV 사본·에폭별 val acc 추출본은 scratchpad `an1019/` 아래 (snapshot/, csv23/, ep/, npz/). 10-19 npz 4런 추출 기록 `an1019/npz_drv.log`.')
    w(f'- 에폭별 기록 점검: {R["sens_last10"]["ep_check"]["runs"]}런 모두 학습 로그 최고값 = 원장 val acc (불일치 {R["sens_last10"]["ep_check"]["mismatch"]}).')
    w('')

    # ---------------------------------------------------------- 검토 반영
    w('## 검토 반영 (eip-critic 10-19, 2판)')
    w('')
    w('critic 판정 "needs_fixes" 의 12개 지적을 모두 다시 계산해 확인했다. critic 이 적은 수치는 모두 재현됐다. **12건 모두 반영** (11번은 문서에만 반영하고 논문 주값 결정은 사용자에게 넘김). 그 안의 세부 수치 2건 (1− 반전 몫, Holm 단조성) 은 부분 기각 — 표 끝 두 행.')
    w('')
    w('| # | 지적 | 처리 | 바뀐 곳 |')
    w('|---|---|---|---|')
    w('| 1 | A3·A4 (활성 뉴런·죽은 채널) 를 "강함" 에 둔 것은 10-10·09-15·10-14 결정과 충돌 | 반영 — B3·B4 로 내림. Shi 2024·maxnorm 공통·09-15 기각 이력을 문장마다 붙임. 뉴런 희소성 위치 표 추가 (한 번도 안 쏜 뉴런 inter {:.1f}% / BPSR {:.1f}%, 산 채널 {:.1f}%) | 0절 3, 3.1, 3.4, 5.2 |'.format(sp['ours_inter_ch']['never_fire_neuron_pct'][0], sp['BPSR']['never_fire_neuron_pct'][0], sp['ours_inter_ch']['alive_channel_pct'][0]))
    w('| 2 | s_count 는 학습 배치 통계, CSV 도 "평가 기록" 아님, baseline 대비 % 가 축마다 다름 | 반영 — 코드 경로 확인 (`proc.py`), CSV 층 합 = s_count {}런 불일치 {}. 정의 절 정정, R19 C10 표와 표1 점에 테스트 npz 축 % 를 함께 적음 (inter {:+.1f}%, BPSR {:+.1f}%) | 정의, 0절 1·2, 1.1, 2.1, 3 정의 |'.format(ce['runs'], ce['mismatch_rel_1e-4'], nag['ours_inter_ch|3e-3']['pct_npz'], nag['BPSR|1e-10']['pct_npz']))
    w('| 3 | - 1- inversion: "활성 뉴런 감소가 사라진다" 는 틀림, 지니 비교 대상 오류 | 반영 (critic 판정 그대로 수용) — 4팔 같은 스파이크 모형을 새로 계산: 지니 상승은 사라지고 균등 규제보다 낮아짐, 활성 뉴런은 BPSR 대비 {:+.1f}~{:+.1f}% 남음, 1− 반전이 설명하는 몫 약 60~70% | 0절 4, 4절 표, 5.2 B10, C |'.format(rv['noinv_struct']['300_420K_all']['active']['contrast']['BPSR']['pct'], rv['noinv_struct']['180_420K_all']['active']['contrast']['BPSR']['pct']))
    w('| 4 | - vmem: Holm 뒤 비유의, "138 에서만" 도 판정 불가, 4팔 모형에서 균등 규제보다 낮음 | 반영 — Holm p={:.2f}, 상호작용 p={:.3f}, 4팔 모형 − BPSR {:+.3f} / − plain L2 {:+.3f}. B9 문장 교체 | 0절 4, 4절, 5.1, 5.2 B9 |'.format(rv['abl_holm']['adj']['- vmem'], rv['vmem_x_server']['p'], rv['vmem_4arm_180_420K']['contrast']['BPSR']['c'], rv['vmem_4arm_180_420K']['contrast']['plain L2']['c']))
    w('| 5 | −30% 비열등이 ±0.15 경계 위, 마진 사후 지정, 마지막 에폭 행 없음 | 반영 — 하한 소수 셋째 자리, 합동 SD·마지막 10에폭·마지막 에폭 판정 표 추가. 문장을 "±0.2 는 네 기준 모두, ±0.15 는 경계" 로. n=5 로 늘리는 제안은 반복 상한 n=4 (10-17) 와 충돌해 6절 8번에 사용자 결정으로만 둠 | 0절 1, 1.0, 2.1, 5.2 A1 |')
    w('| 6 | "BPSR 이 6.1% 적다" 는 방향 오류 (5.7%), "같다" 는 과대 | 반영 — {:.1f}% 로 정정, "차가 검출되지 않는다 (+0.02 [−0.17, +0.21], 최소 검출 차 0.25)" | 0절 2, 2.1, 5.1, C |'.format(bpsr_less))
    w('| 7 | −47% 손실은 작은 그룹 SD 탓, 합동 SD 로는 확인 안 됨 | 반영 — 모든 n≥2 묶음에 합동 SD 판정 열 추가, 손실 시작점을 "−47~−51% (서버 교란)" 으로 | 정의, 0절 1, 1.0, 1.1·1.2 |')
    w('| 8 | 활성 뉴런 공통 기울기 숫자는 잘못 지정된 모형의 평균, r=0.9998 과장 | 반영 — 팔별 기울기 모형으로 400/342/310K 값 표, CSV/npz 비가 팔마다 다름을 명시 | 0절 3, 3.3 |')
    w('| 9 | 표1 npz p 는 24개 검정의 보정 전 값 | 반영 — "보정 전" 명시, Holm 보정값 기재 (죽은 채널·MACs·활성 뉴런 0.09~0.26) | 0절 3, 3.1 |')
    w('| 10 | VGGSNN "−75% 까지 차 없음" 이 무손실로 읽힘 | 반영 — 규제 20런 평균 {:+.2f}, 추세 기울기 {:.2f} ±{:.2f} ({}), −75% 에서 약 {:+.1f}%p 를 함께 적음 | 0절 1, 1.3, 5.2 B7 |'.format(rv['vgg_reg_mean_minus_base']['d'], vt['excl_2e-2']['slope'], vt['excl_2e-2']['se'], P(vt['excl_2e-2']['p']), vt['excl_2e-2']['pred_at_1e-2']))
    w('| 11 | best 는 테스트 셋에서 고른 값 → 절대값 낙관적 | 반영 (문서) — 정의·2.1 에 마지막 10에폭 절대값 기재. 논문 표의 주값을 바꿀지는 사용자 결정으로 넘김 | 정의, 2.1, LIMITATION |')
    w('| 12 | A5 (방향 일반성) 는 기여 주장이 아님, 칩은 학습 n=1 | 반영 — B5 로 내리고 "학습 n=1 반복 측정" 을 붙임 | 5.2 |')
    w('| – | C100 ρ 3e-3 의 −0.49 / −0.50 반올림 불일치 | 반영 — 소수 셋째 자리로 통일 ([−0.495, −0.052]) | 0절 1, 1.2 |')
    w('| – | "1− 반전 몫 약 60%" 의 수치 | 부분 기각 — 구간에 따라 {:.0f}% (180~420K) ~ {:.0f}% (300~420K) 라 "60~70%" 로 적음 | 4절 |'.format(100 * rv['noinv_struct']['180_420K_all']['share_explained_by_inversion'], 100 * rv['noinv_struct']['300_420K_all']['share_explained_by_inversion']))
    w(f'| – | Holm 보정값: critic 은 - 1- inversion 0.81 | 부분 기각 — 단조성을 강제한 Holm 은 - loss-ratio·- 1- inversion 둘 다 1.00 이다 (결론은 같음) | 4절 표 |')
    w('')
    w('- 1판에서 바뀌지 않은 수치: 모든 Welch·ANCOVA·부트스트랩 값 (같은 난수, 새 계산은 난수를 쓰지 않음). 런 선택·n 은 스냅숏 그대로 (137런).')
    w('')

    # ---------------------------------------------------------- LIMITATION
    w('## [LIMITATION]')
    w('')
    w('- **정확도 우위에 대한 검정력이 없다.** 같은 스파이크 정확도 차의 최소 검출 차가 0.09~0.13%p (R19 C10), 0.21~0.31%p (C100), 0.4~0.5%p (QK) 다. 이보다 작은 우위·열위는 "없다" 가 아니라 "보이지 않는다" 다. 반대로 BPSR 대비 +0.11%p 보다 큰 우위는 이 자료로 배제된다. 동등성 검정은 하지 않았다.')
    w(f'- **비열등 판정은 경계 위에 있다.** R19 C10 ±0.15 하한과의 여유가 inter {ni["ours_inter_ch"]["best_welch"]["ci90"][0] + 0.15:.4f}, BPSR {ni["BPSR"]["best_welch"]["ci90"][0] + 0.15:.4f} (Welch) 뿐이고, 마진은 자료를 본 뒤 적은 값이다. 반복 상한 n=4 안에서는 여유를 늘릴 수 없다.')
    w('- **스파이크 측정 축.** R19·VGGSNN 의 s_count 와 CSV 지표는 학습 배치 통계다. 테스트 셋 npz 와의 비가 baseline 1.053, 규제 팔 1.07~1.09 로 달라 baseline 대비 % 가 축마다 1.5~2%p 다르다. npz 축 baseline 은 138 서버 3런뿐이다. 학습 패스 대 테스트 셋 차이가 09-15 의 11% 불일치를 전부 설명하는지는 확인하지 않았다.')
    w(f'- **절대 정확도는 테스트 셋에서 고른 best 다.** R19 의 val 셋 = 테스트 셋. 표1 점과 baseline 의 마지막 10에폭 평균은 best 보다 {min(gap_l10):.2f}~{max(gap_l10):.2f}%p 낮다.')
    w('- **300~420K 는 사후에 고른 구간이다.** 주 구간의 p 는 탐색적이다. 넓은 구간 (180~420K) 은 곡선 모양 (직선 가정) 에 기댄다 — 2차항을 넣어도 결론은 같았다.')
    w('- **같은 정확도 스파이크 비는 기울기에 기댄다.** 300~420K 처럼 기울기가 0 과 구분되지 않는 구간에서는 비가 정의되지 않는다 (구간이 0~2.7 로 퍼짐).')
    w('- **구조 지표는 팔마다 기울기가 다르다.** 공통 기울기 숫자는 구간 평균이고, 스파이크 수준을 정해 읽어야 한다. C100 의 BPSR 은 293~342K 에만 있어 그 위는 외삽이다.')
    w('- **구조 지표의 범위.** 죽은 채널·MACs 는 R19 C10 npz (테스트 1만 장 1회 추론) 뿐이다. C100 은 CSV 지표 (활성 뉴런·지니·단계 몫) 만 있고, QK·DVS·SDT 는 구조를 재지 않았다. MACs 는 밀집 계산값이고 실측이 없다. Shi 2024 수치는 09-15 기록 인용이며 원문을 다시 확인하지 않았다.')
    w('- **다중비교.** 구성요소 4팔과 표1 npz 24개는 Holm 보정을 적었지만, 구간·지표를 여럿 본 나머지 표의 p 는 보정 전이다.')
    w('- **서버 교란.** 서버 항을 넣었지만 팔마다 서버 비율이 다르다. R19 C10 곡선 끝 (ρ 5.5e-3 은 138 만, 6e-3 은 23 만), - 1- inversion (4/5 런이 138), - vmem (23 은 n=3) 에서 서버와 처리가 겹친다.')
    w('- **QK 비교의 측정 잡음.** fp32 재측정값이 학습 로그 최고값보다 런마다 0.00~0.46 낮아, n=1 끼리의 비교는 판단할 수 없다. plain L2 는 ρ 마다 시드 하나다.')
    w('- **VGGSNN·SDTv3 의 해상도.** VGGSNN 은 시드 SD 0.5%p 로 n=4 가 1%p 를 겨우 가르고, 점추정·추세는 손실 쪽이다. SDTv3 는 n=1 이고 ImageNet 시드 SD 를 재지 않았다. KD 런은 진행 중이다.')
    w('- **진행 중인 런.** R19 C10 inter 6런, plain L2 2런 (사용자 중단 뒤 재투입) 은 평균에서 뺐다. 끝나면 −38~−43% 구간 판정이 바뀔 수 있다.')
    w('- **인용 값.** SynOps (10-09), 채널 사망 몫 9% (10-05), 팔 안 상관 r=−0.38, maxnorm 계열 비교 (10-14 E8), 학습 비용 (10-14), Speck 칩 측정 (10-15), Shi 2024 (09-15) 는 다시 계산하지 않았다.')
    w('- **이 분석은 경쟁 범위 (intra·layer) 비교를 갱신하지 않았다.** 새 자료가 BPSR·plain L2 쪽에만 들어왔다.')
    open(path, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    json.dump(clean(RESULT), open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)
    print('json →', OUT_JSON)
    write_md(json.load(open(OUT_JSON)), OUT_JSON.replace('.json', '.md'))
    print('md →', OUT_JSON.replace('.json', '.md'))
