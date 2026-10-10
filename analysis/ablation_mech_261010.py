"""ours_inter_ch 어블레이션 기전 분석 + BPSR 대비 우위 위치 (R19-C10, 26-10-10).

질문 (사용자 10-10 13:20): 각 요소가 '정확도-스파이크 곡선 위치' 말고 무엇을 바꾸나.
  1. vmem       — 같은 rho / 같은 lambda 에서 스파이크, 층 배분, 뉴런 분포, 어느 뉴런이 더 쏘나
  2. final_step — 스텝별 스파이크 비중·최초 발화 스텝 (npz __t/__first_t)
  3. loss-ratio — lambda 궤적, 착지 스파이크의 시드 변동
  4. 1-inversion— 가중 w 분포, 침묵 비율, 지니
  5. BPSR 대비  — 대역별 같은 스파이크 Δacc, 검정력, 구조 지표
  6. 스텝별 정확도 (logits_t)

입력 (모두 읽기 전용):
  원장  scratchpad/led1007b/inventory.json (10-10 13:20 집계)
  학습 로그  scratchpad/abl1010/logs*.json  ← collect_logs.py 가 두 서버 런 폴더의 train.log,
            reg_detail.csv, reg_neuron_detail.csv 를 읽어 만든 것 (10-10 13:27~13:30)
  npz  _npz/, scratchpad/deep1005/npz/ (뉴런별 총합), _npz_t/ (스텝별, 10-10 추출)

쓰는 법:  python analysis/ablation_mech_261010.py      → .json, .md
"""
import glob, json, os, sys
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
INV = S + '/led1007b/inventory.json'
LOGS = [S + '/abl1010/' + f for f in ('logs_138.json', 'logs_23.json', 'logs2_138.json', 'logs2_23.json')]
NPZ_DIRS = [os.path.join(ROOT, '_npz'), S + '/deep1005/npz']
NPZ_T = os.path.join(ROOT, '_npz_t')
OUT_JSON = os.path.join(ROOT, 'analysis', 'ablation_mech_261010.json')
OUT_MD = os.path.join(ROOT, 'analysis', 'ablation_mech_261010.md')
PRUNE_DIRS = [S + '/prune_out', S + '/deep1005/prune']

NAME = {'ours_ch': 'ours_inter_ch', 'ch_novmem': '- vmem', 'ch_nofinal': '- final_step',
        'ch_nolr': '- loss-ratio', 'ch_noinv': '- 1- inversion', 'bpsr': 'BPSR', 'base': 'baseline',
        'prop': 'ours_intra_ch', 'abl_novmem': 'ours_intra_ch - vmem',
        'abl_nofinal': 'ours_intra_ch - final_step', 'abl_nolr': 'ours_intra_ch - loss-ratio',
        'abl_noinv': 'ours_intra_ch - 1- inversion (반전·within_channel·vmem 동시 제거)'}
STAGES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1']
SKIP = {'n_in', 'predictions_n'}
NEU = {}  # 층 → 뉴런 수
for l in ['conv1_conv_n'] + [f'conv2_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')]:
    NEU[l] = 32 * 32 * 128
for l in [f'conv3_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')]:
    NEU[l] = 16 * 16 * 256
for l in [f'conv4_block{b}_{s}_n' for b in (1, 2) for s in ('conv1', 'out')]:
    NEU[l] = 8 * 8 * 512
NEU['fc1_n'] = 256


def knob_val(k):
    return float(k.replace('p', '.')) if k not in ('-', '') else float('nan')


# ---------------------------------------------------------------- 통계 도구
def ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b; df = len(y) - X.shape[1]
    s2 = (r @ r) / df if df > 0 else float('nan')
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return b, se, cov, df, float(np.sqrt(s2))


def pval(t, df):
    return float(2 * stats.t.sf(abs(t), df)) if df > 0 else float('nan')


def ancova(x0, y0, x1, y1):
    """y = a + b·x + c·g. g=1 이 두 번째 집단. 기울기 차 p 도 함께."""
    x = np.r_[x0, x1]; y = np.r_[y0, y1]; g = np.r_[np.zeros(len(x0)), np.ones(len(x1))]
    X = np.c_[np.ones_like(x), x, g]
    b, se, cov, df, sd = ols(X, y)
    out = {'c': float(b[2]), 'se': float(se[2]), 'p': pval(b[2] / se[2], df), 'df': int(df),
           'b': float(b[1]), 'b_se': float(se[1]), 'b_p': pval(b[1] / se[1], df), 'resid_sd': sd,
           'n0': int(len(x0)), 'n1': int(len(x1))}
    if len(x0) >= 3 and len(x1) >= 3:
        X2 = np.c_[X, x * g]
        b2, se2, _, df2, _ = ols(X2, y)
        out['slope_diff_p'] = pval(b2[3] / se2[3], df2)
    return out


def line_pred(x, y, x0):
    """y = a + b·x 적합 후 x0 에서 예측값, 예측 SE (평균), 새 런 하나의 예측 SE."""
    X = np.c_[np.ones(len(x)), x]
    b, se, cov, df, sd = ols(X, y)
    v = np.array([1.0, x0])
    m = float(v @ b); sm = float(np.sqrt(v @ cov @ v))
    return m, sm, float(np.sqrt(sm ** 2 + sd ** 2)), df, sd, float(b[1])


def mean_sd(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float('nan')), len(v)


# ---------------------------------------------------------------- 자료
inv_all = json.load(open(INV))
runs = {}
excluded = []
for r in inv_all['runs']:
    if r['ds'] != 'C10' or not r['tag'].startswith('r19c10-') or r['key'] not in NAME:
        continue
    if r['seed'] >= 900:
        continue
    if not r.get('completed') or r.get('exclude') or r.get('val_acc') is None:
        excluded.append({'tag': r['tag'], 'arm': NAME[r['key']], 'server': r['server'],
                         'epochs': r.get('epochs'), 'val_acc': r.get('val_acc'), 'spikes': r.get('spikes'),
                         'reason': r.get('exclude_reason') or r.get('status')})
        continue
    runs[r['tag']] = dict(r, rho=knob_val(r['knob']), arm=NAME[r['key']])

LOG = {}
for f in LOGS:
    LOG.update(json.load(open(f)))


def log_metrics(tag):
    """train.log 에폭별 lambda/R, reg_detail·reg_neuron_detail 마지막 에폭 층 집계."""
    out = {}
    L = LOG.get(tag)
    if not L:
        return out
    ep = L.get('epochs') or []
    r = runs[tag]
    if ep and 'adp_lambda' in ep[-1]:
        lam = np.array([e['adp_lambda'] for e in ep]); R = np.array([e['reg_R'] for e in ep])
        tl = np.array([e['task_loss_clean'] for e in ep])
        out.update(lam_end=float(lam[-1]), lam_mean=float(lam.mean()), lam_pre=float(lam[:200].mean()),
                   lam_post=float(lam[200:].mean()), lam_traj=lam.tolist(), R_traj=R.tolist(),
                   L_traj=tl.tolist(), R_end=float(R[-1]))
    if ep:
        out['s_traj'] = [e.get('s_count') for e in ep]
        out['loss_traj'] = [e.get('loss') for e in ep]
    rd = L.get('rd'); rn = L.get('rnd')
    if rd:
        last = str(max(int(k) for k in rd['by_epoch']))
        lay = rd['layers']; rows = rd['by_epoch'][last]
        ci = {c: i for i, c in enumerate(rd['cols'])}
        spk = {l: rows[i][ci['spike_count']] for i, l in enumerate(lay) if l in NEU and rows[i]}
        w = {l: rows[i][ci['sc_rate_mean']] for i, l in enumerate(lay) if l in NEU and rows[i]}
        scl = {l: rows[i][ci['sc_loss']] for i, l in enumerate(lay) if l in NEU and rows[i]}
        tot = sum(spk.values())
        out['tr_spk'] = tot
        out['tr_ep'] = int(last)
        for s in STAGES:
            out[f'share_{s}'] = 100 * sum(v for l, v in spk.items() if l.startswith(s)) / tot
        nN = sum(NEU[l] for l in w)
        out['w_mean'] = sum(w[l] * NEU[l] for l in w) / nN          # 뉴런 가중 평균 w
        for s in STAGES:
            ls = [l for l in w if l.startswith(s)]
            out[f'w_{s}'] = sum(w[l] * NEU[l] for l in ls) / sum(NEU[l] for l in ls)
        out['R_rd'] = sum(scl.values())
        # 에폭별 학습 중 총 스파이크 (lambda 없는 팔의 rho_eff 계산용)
        out['tr_spk_traj'] = [sum(x[ci['spike_count']] for i, x in enumerate(rd['by_epoch'][str(e)])
                                  if lay[i] in NEU and x) for e in sorted(int(k) for k in rd['by_epoch'])]
        out['R_rd_traj'] = [sum(x[ci['sc_loss']] for i, x in enumerate(rd['by_epoch'][str(e)])
                                if lay[i] in NEU and x) for e in sorted(int(k) for k in rd['by_epoch'])]
        if rn:
            rows2 = rn['by_epoch'][last]; c2 = {c: i for i, c in enumerate(rn['cols'])}
            d = {l: rows2[i] for i, l in enumerate(rn['layers']) if l in NEU and rows2[i]}
            nN = sum(NEU[l] for l in d)
            out['dead_pct'] = 100 * sum(d[l][c2['dead_neuron_ratio']] * NEU[l] for l in d) / nN
            out['gini_sw'] = sum(d[l][c2['gini']] * spk[l] for l in d) / tot      # 층 스파이크 가중
            out['top10_sw'] = 100 * sum(d[l][c2['top10_share']] * spk[l] for l in d) / tot
            for s in STAGES:
                ls = [l for l in d if l.startswith(s)]
                out[f'dead_{s}'] = 100 * sum(d[l][c2['dead_neuron_ratio']] * NEU[l] for l in ls) / sum(NEU[l] for l in ls)
                ss = sum(spk[l] for l in ls)
                out[f'gini_{s}'] = sum(d[l][c2['gini']] * spk[l] for l in ls) / ss if ss > 0 else float('nan')
    return out


for t in runs:
    runs[t]['log'] = log_metrics(t)


def arm_runs(key, lo=None, hi=None, need=None):
    o = [r for r in runs.values() if r['key'] == key]
    if lo is not None:
        o = [r for r in o if lo <= r['spikes'] <= hi]
    if need:
        o = [r for r in o if need in r['log']]
    return sorted(o, key=lambda r: (r['rho'], r['seed']))


res = {'generated_from': {'inventory': INV, 'inventory_generated': inv_all.get('generated'), 'logs': LOGS},
       'excluded': excluded}

# ---------------------------------------------------------------- inter 기준 곡선 (같은 스파이크 Δacc)
INTER = arm_runs('ours_ch')
BAND_FIT = (200e3, 420e3)
fit_runs = [r for r in INTER if BAND_FIT[0] <= r['spikes'] <= BAND_FIT[1]]
fx = np.log([r['spikes'] for r in fit_runs]); fy = np.array([r['val_acc'] for r in fit_runs])
res['inter_fit'] = {'band': BAND_FIT, 'n': len(fit_runs)}


def same_spike_delta(r):
    m, sm, sp, df, sd, b = line_pred(fx, fy, np.log(r['spikes']))
    return {'pred': m, 'delta': r['val_acc'] - m, 'se': sp, 'z': (r['val_acc'] - m) / sp}


# inter 곡선에서 '같은 스파이크' 지표 예측 (로그 지표용)
def inter_metric_pred(key_metric, r, band=(200e3, 420e3)):
    pts = [(np.log(q['spikes']), q['log'][key_metric]) for q in INTER
           if band[0] <= q['spikes'] <= band[1] and key_metric in q['log']]
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    m, sm, sp, df, sd, b = line_pred(x, y, np.log(r['spikes']))
    v = r['log'][key_metric]
    return {'inter_pred': m, 'value': v, 'delta': v - m, 'se': sp, 'z': (v - m) / sp, 'n_fit': len(x)}


# ================================================================ 1. vmem
NOV = arm_runs('ch_novmem')
sec1 = {}
# 1a 같은 rho 에서 착지 스파이크: ln(spk) = a + b ln(rho) + c[novmem]
RHO_LO, RHO_HI = 1.0e-3, 5.0e-3
i_r = [r for r in INTER if RHO_LO <= r['rho'] <= RHO_HI]
sec1['same_rho_spikes'] = ancova(np.log([r['rho'] for r in i_r]), np.log([r['spikes'] for r in i_r]),
                                 np.log([r['rho'] for r in NOV]), np.log([r['spikes'] for r in NOV]))
sec1['same_rho_spikes']['rho_range'] = [RHO_LO, RHO_HI]
# 같은 rho 짝 (inter 평균 vs -vmem 단일)
pairs = []
for q in NOV:
    grp = [r for r in INTER if abs(r['rho'] - q['rho']) < 1e-12]
    if grp:
        ms, ss, n = mean_sd([r['spikes'] for r in grp]); ma, sa, _ = mean_sd([r['val_acc'] for r in grp])
        pairs.append({'rho': q['rho'], 'inter_n': n, 'inter_spk': ms, 'inter_spk_sd': ss, 'inter_acc': ma,
                      'nov_spk': q['spikes'], 'nov_acc': q['val_acc'], 'ratio': q['spikes'] / ms})
sec1['same_rho_pairs'] = pairs
# 1b 같은 lambda: ln(spk) = a + b ln(lambda) + c[novmem]  (lambda 는 학습 전체 평균, 마지막 에폭 값 둘 다)
for lk in ('lam_mean', 'lam_end', 'lam_pre'):
    a0 = [r for r in i_r if lk in r['log']]; a1 = [r for r in NOV if lk in r['log']]
    sec1[f'same_{lk}_spikes'] = ancova(np.log([r['log'][lk] for r in a0]), np.log([r['spikes'] for r in a0]),
                                       np.log([r['log'][lk] for r in a1]), np.log([r['spikes'] for r in a1]))
# 같은 rho 에서 lambda 차
a0 = [r for r in i_r if 'lam_mean' in r['log']]
sec1['same_rho_lam_mean'] = ancova(np.log([r['rho'] for r in a0]), np.log([r['log']['lam_mean'] for r in a0]),
                                   np.log([r['rho'] for r in NOV]), np.log([r['log']['lam_mean'] for r in NOV]))
# 1c 같은 (학습 모드) 스파이크에서 R (= 스파이크에 걸린 가중 벌점의 크기) 과 평균 w
BAND_V = (250e3, 430e3)
i_b = [r for r in INTER if BAND_V[0] <= r['spikes'] <= BAND_V[1] and 'tr_spk' in r['log']]
n_b = [r for r in NOV if 'tr_spk' in r['log']]
mech = {}
for k in ['R_rd', 'w_mean'] + [f'w_{s}' for s in STAGES[:4]]:
    ylog = k == 'R_rd'
    f = (lambda v: np.log(v)) if ylog else (lambda v: v)
    mech[k] = ancova(np.log([r['log']['tr_spk'] for r in i_b]), f(np.array([r['log'][k] for r in i_b])),
                     np.log([r['log']['tr_spk'] for r in n_b]), f(np.array([r['log'][k] for r in n_b])))
    mech[k]['log_y'] = ylog
sec1['mech_same_spk'] = mech
# 1d 같은 스파이크에서 층 배분·분포 (학습 모드 마지막 에폭, reg_detail/reg_neuron_detail)
dist = {}
for k in [f'share_{s}' for s in STAGES[:4]] + ['dead_pct', 'gini_sw', 'top10_sw'] + \
         [f'dead_{s}' for s in STAGES[:4]] + [f'gini_{s}' for s in STAGES[:4]]:
    dist[k] = ancova(np.log([r['log']['tr_spk'] for r in i_b]), [r['log'][k] for r in i_b],
                     np.log([r['log']['tr_spk'] for r in n_b]), [r['log'][k] for r in n_b])
sec1['dist_same_spk'] = dist
sec1['band'] = BAND_V
# 1e 같은 스파이크 Δacc (원장)
sec1['acc_rows'] = [{'tag': r['tag'], **same_spike_delta(r)} for r in NOV]
d = np.array([x['delta'] for x in sec1['acc_rows']]); se = np.array([x['se'] for x in sec1['acc_rows']])
wts = 1 / se ** 2
sec1['acc_pooled'] = {'delta': float((d * wts).sum() / wts.sum()), 'se': float(np.sqrt(1 / wts.sum())), 'n': len(d)}
sec1['acc_pooled']['p'] = float(2 * stats.norm.sf(abs(sec1['acc_pooled']['delta'] / sec1['acc_pooled']['se'])))
# 1f ours_intra_ch 에서 같은 비교 (rho 4e-3 에 양쪽 n=5)
P4 = [r for r in arm_runs('prop') if abs(r['rho'] - 4e-3) < 1e-12]
N4 = [r for r in arm_runs('abl_novmem') if abs(r['rho'] - 4e-3) < 1e-12]
intra = {'prop4': [r['tag'] for r in P4], 'nov4': [r['tag'] for r in N4]}
for k, f in (('spikes', np.log), ('val_acc', lambda v: v)):
    a = f(np.array([r[k] for r in P4])); b_ = f(np.array([r[k] for r in N4]))
    t_, p_ = stats.ttest_ind(b_, a, equal_var=False)
    intra[k] = {'prop_mean': float(np.mean([r[k] for r in P4])), 'prop_sd': float(np.std([r[k] for r in P4], ddof=1)),
                'nov_mean': float(np.mean([r[k] for r in N4])), 'nov_sd': float(np.std([r[k] for r in N4], ddof=1)),
                'diff': float(b_.mean() - a.mean()),
                'se': float(np.sqrt(a.var(ddof=1) / len(a) + b_.var(ddof=1) / len(b_))), 'p': float(p_)}
# intra 같은 스파이크 정확도: prop 전체 (유효) 와 abl_novmem 전체로 ANCOVA
PA = arm_runs('prop'); NA = arm_runs('abl_novmem')
intra['acc_same_spk'] = ancova(np.log([r['spikes'] for r in PA]), [r['val_acc'] for r in PA],
                               np.log([r['spikes'] for r in NA]), [r['val_acc'] for r in NA])
intra['same_rho_spikes'] = ancova(np.log([r['rho'] for r in PA]), np.log([r['spikes'] for r in PA]),
                                  np.log([r['rho'] for r in NA]), np.log([r['spikes'] for r in NA]))
# intra 같은 스파이크 분포 (reg_neuron_detail)
pa = [r for r in PA if 'dead_pct' in r['log']]; na = [r for r in NA if 'dead_pct' in r['log']]
intra['dist'] = {k: ancova(np.log([r['log']['tr_spk'] for r in pa]), [r['log'][k] for r in pa],
                           np.log([r['log']['tr_spk'] for r in na]), [r['log'][k] for r in na])
                 for k in ['dead_pct', 'gini_sw', 'top10_sw', 'w_mean'] + [f'share_{s}' for s in STAGES[:4]]}
intra['R_same_spk'] = ancova(np.log([r['log']['tr_spk'] for r in pa]), np.log([r['log']['R_rd'] for r in pa]),
                             np.log([r['log']['tr_spk'] for r in na]), np.log([r['log']['R_rd'] for r in na]))
intra['n_dist'] = [len(pa), len(na)]
sec1['intra'] = intra
# 두 경쟁 범위 합산 (사후, 역분산 가중) — 같은 스파이크 Δacc
a_ = sec1['acc_pooled']; b_ = intra['acc_same_spk']
w1, w2 = 1 / a_['se'] ** 2, 1 / b_['se'] ** 2
cm = (a_['delta'] * w1 + b_['c'] * w2) / (w1 + w2); cs = float(np.sqrt(1 / (w1 + w2)))
sec1['acc_combined_posthoc'] = {'delta': float(cm), 'se': cs, 'p': float(2 * stats.norm.sf(abs(cm / cs)))}
res['vmem'] = sec1

# ================================================================ 2. final_step
NOF = arm_runs('ch_nofinal')
sec2 = {}
x = np.log([r['rho'] for r in i_r]); y = np.log([r['spikes'] for r in i_r])
rows = []
for q in NOF:
    m, sm, sp, df, sd, b = line_pred(x, y, np.log(q['rho']))
    rows.append({'tag': q['tag'], 'rho': q['rho'], 'spikes': q['spikes'], 'inter_pred_spk': float(np.exp(m)),
                 'ratio': float(q['spikes'] / np.exp(m)), 'ln_ratio_se': sp, 'z': float((np.log(q['spikes']) - m) / sp),
                 **{'acc_' + k: v for k, v in same_spike_delta(q).items()}})
    for k in ['dead_pct', 'gini_sw', 'top10_sw', 'w_mean'] + [f'share_{s}' for s in STAGES[:4]]:
        if k in q['log']:
            rows[-1]['m_' + k] = inter_metric_pred(k, q, BAND_V)
    rows[-1]['lam_mean'] = q['log'].get('lam_mean')
    # 같은 lambda (학습 평균) 에서 inter 가 착지할 스파이크 — R 정의가 final_step 에 따라 달라
    # 같은 rho 가 같은 세기가 아니므로 lambda 기준도 함께 본다.
    a0 = [r for r in i_r if 'lam_mean' in r['log']]
    m2, sm2, sp2, *_ = line_pred(np.log([r['log']['lam_mean'] for r in a0]), np.log([r['spikes'] for r in a0]),
                                np.log(q['log']['lam_mean']))
    rows[-1]['same_lam'] = {'inter_pred_spk': float(np.exp(m2)), 'ratio': float(q['spikes'] / np.exp(m2)),
                            'ln_se': sp2, 'z': float((np.log(q['spikes']) - m2) / sp2), 'n_fit': len(a0)}
    m3, sm3, sp3, *_ = line_pred(np.log([r['rho'] for r in a0]), np.log([r['log']['lam_mean'] for r in a0]), np.log(q['rho']))
    rows[-1]['lam_vs_inter'] = {'inter_pred_lam': float(np.exp(m3)), 'ratio': float(q['log']['lam_mean'] / np.exp(m3))}
sec2['same_rho'] = rows
# intra 판 (abl_nofinal vs prop, 같은 rho)
intra2 = []
for q in arm_runs('abl_nofinal'):
    grp = [r for r in PA if abs(r['rho'] - q['rho']) < 1e-12]
    intra2.append({'tag': q['tag'], 'rho': q['rho'], 'spikes': q['spikes'], 'val_acc': q['val_acc'],
                   'prop_n': len(grp), 'prop_spk': float(np.mean([r['spikes'] for r in grp])) if grp else None,
                   'prop_acc': float(np.mean([r['val_acc'] for r in grp])) if grp else None})
sec2['intra_same_rho'] = intra2
xa = np.log([r['rho'] for r in PA]); ya = np.log([r['spikes'] for r in PA])
NFA = arm_runs('abl_nofinal')
sec2['intra_same_rho_ancova'] = ancova(xa, ya, np.log([r['rho'] for r in NFA]), np.log([r['spikes'] for r in NFA]))
sec2['intra_acc_same_spk'] = ancova(np.log([r['spikes'] for r in PA]), [r['val_acc'] for r in PA],
                                    np.log([r['spikes'] for r in NFA]), [r['val_acc'] for r in NFA])


# 2b 시간축 (npz_t)
def npz_t_metrics(p):
    z = np.load(p)
    ns = float(z['n_samples']); T = int(z['T'])
    o = {'n_samples': ns, 'T': T, 'acc': float(z['acc'])}
    tot_t = np.zeros(T); first = np.zeros(T + 1)
    st_t = {s: np.zeros(T) for s in STAGES}; st_first = {s: np.zeros(T + 1) for s in STAGES}
    for k in z.files:
        if not k.endswith('__t'):
            continue
        l = k[:-3]
        if l in SKIP or l not in NEU:
            continue
        a = z[k].reshape(T, -1).sum(1).astype(np.float64)
        f = z[l + '__first_t'].astype(np.float64)
        tot_t += a; first += f
        st = l.split('_')[0]
        st_t[st] += a; st_first[st] += f
    o['step_share'] = (100 * tot_t / tot_t.sum()).tolist()
    o['spk_per_img'] = float(tot_t.sum() / ns)
    o['step_spk_per_img'] = (tot_t / ns).tolist()
    fired = first[:T].sum()
    o['first_share_of_fired'] = (100 * first[:T] / fired).tolist()
    o['silent_pair_pct'] = float(100 * first[T] / first.sum())
    o['mean_first_t'] = float((np.arange(1, T + 1) * first[:T]).sum() / fired)
    for s in STAGES:
        o[f'step_share_{s}'] = (100 * st_t[s] / st_t[s].sum()).tolist() if st_t[s].sum() > 0 else None
        ff = st_first[s][:T].sum()
        o[f'first_share_{s}'] = (100 * st_first[s][:T] / ff).tolist() if ff > 0 else None
        o[f'mean_first_t_{s}'] = float((np.arange(1, T + 1) * st_first[s][:T]).sum() / ff) if ff > 0 else None
    # 스텝별 정확도 (logits_t: 각 스텝까지 돌렸을 때의 출력)
    if 'logits_t' in z.files:
        lt = z['logits_t']; lab = z['labels']
        o['acc_t'] = [float(100 * (lt[:, t].argmax(1) == lab).mean()) for t in range(T)]
        o['acc_final'] = float(100 * (z['pred'] == lab).mean())
        # 스텝 t 에서 정답·확신 (softmax 최대값 기준 early-exit 은 하지 않음 — 고정 T' 절단만)
        o['agree_t_final'] = [float(100 * (lt[:, t].argmax(1) == z['pred']).mean()) for t in range(T)]
    return o


npzt = {}
for p in sorted(glob.glob(NPZ_T + '/*.npz')):
    tag = os.path.basename(p)[:-4]
    try:
        npzt[tag] = npz_t_metrics(p)
    except Exception as e:  # 추출 중 파일
        npzt[tag] = {'error': str(e)}
sec2['npz_t'] = npzt
res['final_step'] = sec2

# ================================================================ 3. loss-ratio
NOL = arm_runs('ch_nolr')
sec3 = {}
# lambda 궤적 요약 (inter, rho 묶음별)
EPS = [0, 4, 9, 29, 99, 199, 200, 309]
traj = {}
for rho in sorted(set(r['rho'] for r in INTER)):
    grp = [r for r in INTER if r['rho'] == rho and 'lam_traj' in r['log']]
    if not grp:
        continue
    lt = np.array([r['log']['lam_traj'] for r in grp])
    traj[f'{rho:g}'] = {'n': len(grp), 'lam_at': {e: float(lt[:, e].mean()) for e in EPS},
                        'lam_cv_end': float(100 * lt[:, -1].std(ddof=1) / lt[:, -1].mean()) if len(grp) > 1 else None,
                        'lam_mean': float(lt.mean()),
                        'max_over_min_pre': float((lt[:, 4:200].max(1) / lt[:, 4:200].min(1)).mean()),
                        'tags': [r['tag'] for r in grp]}
sec3['inter_lam'] = traj
# 고정 lambda 팔의 실효 rho (= lambda·R / L) 궤적 — R 은 reg_detail sc_loss 합, L 은 학습 loss
nolr = []
for q in NOL:
    lg = q['log']; lam = q['rho']
    R = np.array(lg['R_rd_traj']); Ltr = np.array(lg['loss_traj'], float)
    # loss = task + lambda·R 이므로 task = loss − lambda·R
    task = Ltr - lam * R
    rho_eff = lam * R / task
    nolr.append({'tag': q['tag'], 'lambda': lam, 'rho_eff_at': {e: float(rho_eff[e]) for e in EPS},
                 'rho_eff_mean': float(rho_eff.mean()), 'spikes': q['spikes'], 'val_acc': q['val_acc'],
                 **{'acc_' + k: v for k, v in same_spike_delta(q).items()}})
sec3['nolr'] = nolr
# inter 의 같은 시점 R·L 로 본 rho (검산: 상수여야 함)
# 착지 스파이크 시드 변동 (같은 rho / 같은 lambda, n>=3 묶음)
def pooled_cv(groups):
    num = den = 0
    for g in groups:
        v = np.log(g)
        num += ((v - v.mean()) ** 2).sum(); den += len(v) - 1
    return float(100 * np.sqrt(num / den)), int(den)


gi = [[r['spikes'] for r in INTER if r['rho'] == rho] for rho in sorted(set(r['rho'] for r in INTER))]
gi = [g for g in gi if len(g) >= 3]
BP = arm_runs('bpsr')
gb = [[r['spikes'] for r in BP if r['rho'] == rho] for rho in sorted(set(r['rho'] for r in BP))]
gb = [g for g in gb if len(g) >= 3]
PR = [[r['spikes'] for r in PA if r['rho'] == rho] for rho in sorted(set(r['rho'] for r in PA))]
PR = [g for g in PR if len(g) >= 3]
cvi, dfi = pooled_cv(gi); cvb, dfb = pooled_cv(gb); cvp, dfp = pooled_cv(PR)
F = (cvi / cvb) ** 2
sec3['landing_cv'] = {'inter': [cvi, dfi], 'bpsr_fixed_lambda': [cvb, dfb], 'intra': [cvp, dfp],
                      'F_inter_vs_bpsr': F, 'p': float(2 * min(stats.f.sf(F, dfi, dfb), stats.f.cdf(F, dfi, dfb)))}
# 착지 정확도 변동 (같은 묶음)
def pooled_sd(groups):
    num = den = 0
    for g in groups:
        v = np.asarray(g, float); num += ((v - v.mean()) ** 2).sum(); den += len(v) - 1
    return float(np.sqrt(num / den)), int(den)


ai = [[r['val_acc'] for r in INTER if r['rho'] == rho] for rho in sorted(set(r['rho'] for r in INTER))]
ab = [[r['val_acc'] for r in BP if r['rho'] == rho] for rho in sorted(set(r['rho'] for r in BP))]
sec3['acc_sd'] = {'inter': pooled_sd([g for g in ai if len(g) >= 3]), 'bpsr': pooled_sd([g for g in ab if len(g) >= 3])}
# 시드 사이 착지 차이가 학습 어디서 생기나: inter 3e-3 (n=4) 와 BPSR 3e-11 (n=3) 의 에폭별 s_count CV
def ep_cv(tags, eps):
    m = np.array([runs[t]['log']['s_traj'] for t in tags], float)
    return {e: float(100 * m[:, e].std(ddof=1) / m[:, e].mean()) for e in eps}


EPS2 = [9, 29, 99, 199, 249, 309]
sec3['ep_cv'] = {
    'inter_3e-3': {'n': 4, 'cv': ep_cv([r['tag'] for r in INTER if r['rho'] == 3e-3], EPS2)},
    'inter_6e-3': {'n': 3, 'cv': ep_cv([r['tag'] for r in INTER if r['rho'] == 6e-3], EPS2)},
    'bpsr_3e-11': {'n': 3, 'cv': ep_cv([r['tag'] for r in BP if r['rho'] == 3e-11], EPS2)},
    'bpsr_1e-10': {'n': 3, 'cv': ep_cv([r['tag'] for r in BP if r['rho'] == 1e-10], EPS2)}}
res['loss_ratio'] = sec3

# ================================================================ 4. 1- inversion
NOI = arm_runs('ch_noinv')
sec4 = {'rows': []}
for q in NOI:
    row = {'tag': q['tag'], 'rho': q['rho'], 'spikes': q['spikes'], **{'acc_' + k: v for k, v in same_spike_delta(q).items()}}
    for k in ['w_mean'] + [f'w_{s}' for s in STAGES[:4]] + ['dead_pct', 'gini_sw', 'top10_sw'] + \
             [f'share_{s}' for s in STAGES[:4]] + [f'dead_{s}' for s in STAGES[:4]] + [f'gini_{s}' for s in STAGES[:4]]:
        if k in q['log']:
            row['m_' + k] = inter_metric_pred(k, q, (180e3, 330e3))
    sec4['rows'].append(row)


# 4b 채널 단위 (npz): intra 의 abl_noinv (반전·within_channel·vmem 동시 제거) vs ours_intra_ch — 참고
def npz_path(tag):
    for d in NPZ_DIRS:
        p = os.path.join(d, tag + '.npz')
        if os.path.exists(p):
            return p
    return None


def weight_layers():
    for d in PRUNE_DIRS:
        for q in sorted(glob.glob(d + '/r19c10-*.json')):
            j = json.load(open(q))
            if 'weight_layers' in j:
                return j['weight_layers']
    return None


WL = weight_layers()


def macs_removed_pct(dead):
    mr = mt = 0
    for w in WL:
        ks = w['ks']; cin, cout = ks[-2], ks[-1]; kk = int(np.prod(ks[:-2])) if len(ks) > 2 else 1
        di = dead.get(w['src'], 0); do = dead.get(w['dst'], 0)
        mr += (di * cout + do * cin - di * do) * kk * w['hw']; mt += cin * cout * kk * w['hw']
    return 100 * mr / mt


def npz_struct(p):
    z = np.load(p); ns = float(z['n_samples'])
    dead = {}; act = 0.0; spk = 0.0; chsize = {}; deadsize = {'big': 0, 'small': 0}
    for l in NEU:
        c = z[l].astype(np.float64); f = z[l + '__fired'].astype(np.float64)
        act += f.sum(); spk += c.sum()
        tot = c.reshape(-1, c.shape[-1]).sum(0) if c.ndim == 3 else c
        dead[l] = int((tot == 0).sum())
    o = {'dead_ch': int(sum(dead.values())), 'active_img': act / ns, 'npz_spk': spk / ns,
         'dead_by_stage': {s: int(sum(v for l, v in dead.items() if l.startswith(s))) for s in STAGES}}
    if WL:
        o['macs_rm_pct'] = macs_removed_pct(dead)
    return o


# 같은 rho 의 baseline 채널 크기로 어떤 채널이 죽었나: 기준 = baseline s11~13 의 채널 공간합 순위.
# 독립 학습 망은 채널 번호가 정렬돼 있지 않으므로 (decisions 09-20) 채널 단위 대응은 하지 않는다.
st_rows = []
for key in ('ours_ch', 'bpsr', 'base', 'prop', 'abl_noinv'):
    for r in arm_runs(key):
        p = npz_path(r['tag'])
        if p:
            st_rows.append({'tag': r['tag'], 'key': key, 'spikes': r['spikes'], 'val_acc': r['val_acc'], **npz_struct(p)})
sec4['npz_struct'] = [x for x in st_rows if x['key'] in ('prop', 'abl_noinv')]
res['inversion'] = sec4

# ================================================================ 5. BPSR 대비
sec5 = {}
# 5a BPSR 묶음마다 그 스파이크에서 inter 국소 직선 예측과 비교
loc = []
for lam in sorted(set(r['rho'] for r in BP)):
    g = [r for r in BP if r['rho'] == lam]
    ms = float(np.exp(np.mean(np.log([r['spikes'] for r in g]))))
    ma, sa, n = mean_sd([r['val_acc'] for r in g])
    win = [r for r in INTER if ms / 1.25 <= r['spikes'] <= ms * 1.25]
    if len(win) < 4:
        loc.append({'bpsr_lambda': lam, 'n_bpsr': n, 'bpsr_spk': ms, 'bpsr_acc': ma, 'n_inter': len(win), 'note': 'inter 런 부족'})
        continue
    m, sm, sp, df, sd, b = line_pred(np.log([r['spikes'] for r in win]), [r['val_acc'] for r in win], np.log(ms))
    sdb = sa if n > 1 else sd
    se_d = float(np.sqrt(sm ** 2 + sdb ** 2 / n))
    dlt = m - ma
    loc.append({'bpsr_lambda': lam, 'n_bpsr': n, 'bpsr_spk': ms, 'bpsr_acc': ma, 'bpsr_sd': sa,
                'n_inter': len(win), 'inter_lo': min(r['spikes'] for r in win), 'inter_hi': max(r['spikes'] for r in win),
                'inter_pred': m, 'inter_pred_se': sm, 'delta': dlt, 'se': se_d,
                'p': float(2 * stats.t.sf(abs(dlt / se_d), df + n - 1)), 'df': int(df + n - 1)})
sec5['local'] = loc
# 5b 대역별 ANCOVA
bands = [(370e3, 430e3), (300e3, 420e3), (280e3, 360e3), (220e3, 330e3), (180e3, 260e3), (180e3, 430e3)]
bres = []
for lo, hi in bands:
    a = [r for r in INTER if lo <= r['spikes'] <= hi]; b_ = [r for r in BP if lo <= r['spikes'] <= hi]
    if len(b_) < 2 or len(a) < 2:
        bres.append({'band': [lo, hi], 'n_inter': len(a), 'n_bpsr': len(b_), 'note': '표본 부족'}); continue
    # BPSR 를 g=0, inter 를 g=1 로 → c = inter − BPSR
    o = ancova(np.log([r['spikes'] for r in b_]), [r['val_acc'] for r in b_],
               np.log([r['spikes'] for r in a]), [r['val_acc'] for r in a])
    o.update(band=[lo, hi], n_inter=len(a), n_bpsr=len(b_))
    bres.append(o)
sec5['bands'] = bres
# 5c 검정력
z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.80)
main = [b for b in bres if b['band'] == [300e3, 420e3]][0]
sig = main['resid_sd']; dlt = main['c']
n_arm = 2 * (sig * (z_a + z_b) / dlt) ** 2 / 2 * 2  # 집단당 n (두 집단 평균 차, 공통 SD)
n_arm = 2 * (z_a + z_b) ** 2 * sig ** 2 / dlt ** 2
se_now = main['se']; se_need = abs(dlt) / (z_a + z_b)
sec5['power'] = {'delta': dlt, 'sigma': sig, 'n_per_arm_80': float(n_arm), 'n_now': [main['n_inter'], main['n_bpsr']],
                 'se_now': se_now, 'se_needed': se_need, 'scale_runs': float((se_now / se_need) ** 2),
                 'mde_n4': float((z_a + z_b) * sig * np.sqrt(2 / 4)),
                 'mde_now': float((z_a + z_b) * se_now),
                 'power_now': float(stats.norm.cdf(abs(dlt) / se_now - z_a))}
for nn in (4, 8, 12, 24):
    sec5['power'][f'mde_n{nn}'] = float((z_a + z_b) * sig * np.sqrt(2 / nn))
# 5d 구조 지표 재계산 (npz): 죽은 채널, MACs, 이미지당 활성 뉴런 — 300~420K ANCOVA
srow = [x for x in st_rows if x['key'] in ('ours_ch', 'bpsr')]
srow_b = [x for x in st_rows if x['key'] == 'base']
sec5['struct_rows'] = srow + srow_b
stc = {}
for lo, hi in ((300e3, 420e3), (180e3, 430e3)):
    a = [x for x in srow if x['key'] == 'ours_ch' and lo <= x['spikes'] <= hi]
    b_ = [x for x in srow if x['key'] == 'bpsr' and lo <= x['spikes'] <= hi]
    d = {}
    for k, f in (('dead_ch', lambda v: v), ('macs_rm_pct', lambda v: v), ('active_img', np.log), ('val_acc', lambda v: v)):
        if k not in a[0]:
            continue
        d[k] = ancova(np.log([x['spikes'] for x in b_]), f(np.array([x[k] for x in b_])),
                      np.log([x['spikes'] for x in a]), f(np.array([x[k] for x in a])))
    stc[f'{int(lo/1e3)}-{int(hi/1e3)}K'] = d
sec5['struct_ancova'] = stc
res['bpsr'] = sec5

# ================================================================ 6. 스텝별 정확도 (npz_t 의 logits_t)
def tgroup(prefix):
    return [(t, o) for t, o in sorted(npzt.items()) if t.startswith(prefix) and 'error' not in o]


TG = [('baseline', 'r19c10-base---'), ('ours_inter_ch 1.5e-3', 'r19c10-ours_ch-1p5e-3-'), ('BPSR 3e-11', 'r19c10-bpsr-3e-11-')]
sec6 = {}
for nm, pf in TG:
    g = tgroup(pf)
    if not g:
        continue
    A = np.array([o['acc_t'] for _, o in g]); D = A - A[:, [-1]]
    cs = np.array([np.cumsum(o['step_share']) for _, o in g])
    st = np.array([o['step_share'] for _, o in g]); ft = np.array([o['first_share_of_fired'] for _, o in g])
    sec6[nm] = {'n': len(g), 'tags': [t for t, _ in g], 'acc_t': A.mean(0).tolist(),
                'acc_t_sd': (A.std(0, ddof=1) if len(g) > 1 else np.full(A.shape[1], np.nan)).tolist(),
                'drop_vs_T4': D.mean(0).tolist(),
                'drop_vs_T4_sd': (D.std(0, ddof=1) if len(g) > 1 else np.full(A.shape[1], np.nan)).tolist(),
                'cum_spk_share': cs.mean(0).tolist(), 'step_share': st.mean(0).tolist(),
                'step_share_sd': (st.std(0, ddof=1) if len(g) > 1 else np.full(4, np.nan)).tolist(),
                'first_share': ft.mean(0).tolist(),
                'first_share_sd': (ft.std(0, ddof=1) if len(g) > 1 else np.full(4, np.nan)).tolist(),
                'mean_first_t': float(np.mean([o['mean_first_t'] for _, o in g])),
                'mean_first_t_sd': float(np.std([o['mean_first_t'] for _, o in g], ddof=1)) if len(g) > 1 else float('nan'),
                'silent_pair_pct': float(np.mean([o['silent_pair_pct'] for _, o in g]))}
    for stg in STAGES:
        sec6[nm][f'step_share_{stg}'] = np.mean([o[f'step_share_{stg}'] for _, o in g], 0).tolist()
        sec6[nm][f'first_share_{stg}'] = np.mean([o[f'first_share_{stg}'] for _, o in g], 0).tolist()
        sec6[nm][f'mean_first_t_{stg}'] = float(np.mean([o[f'mean_first_t_{stg}'] for _, o in g]))


def wdiff(a, b, key, idx):
    """Welch 차 (a − b) 와 SE, p. n<2 쪽은 nan."""
    va = np.array([o[key][idx] if isinstance(o[key], list) else o[key] for _, o in tgroup(a)])
    vb = np.array([o[key][idx] if isinstance(o[key], list) else o[key] for _, o in tgroup(b)])
    if len(va) < 2 or len(vb) < 2:
        return {'diff': float(va.mean() - vb.mean()), 'se': float('nan'), 'p': float('nan'), 'n': [len(va), len(vb)]}
    t_, p_ = stats.ttest_ind(va, vb, equal_var=False)
    return {'diff': float(va.mean() - vb.mean()),
            'se': float(np.sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb))), 'p': float(p_), 'n': [len(va), len(vb)]}


def drop_list(pf, t):
    return [o['acc_t'][t] - o['acc_t'][-1] for _, o in tgroup(pf)]


cmp6 = {}
for nm, pf in TG[1:]:
    cmp6[nm + ' − baseline'] = {
        'last_step_share': wdiff(pf, TG[0][1], 'step_share', 3),
        'first_step_share': wdiff(pf, TG[0][1], 'step_share', 0),
        'mean_first_t': wdiff(pf, TG[0][1], 'mean_first_t', 0),
        'first_at_t4': wdiff(pf, TG[0][1], 'first_share_of_fired', 3)}
    a = np.array(drop_list(pf, 1)); b_ = np.array(drop_list(TG[0][1], 1))
    if len(a) >= 2 and len(b_) >= 2:
        t_, p_ = stats.ttest_ind(a, b_, equal_var=False)
        cmp6[nm + ' − baseline']['drop_T2'] = {'diff': float(a.mean() - b_.mean()),
                                               'se': float(np.sqrt(a.var(ddof=1) / len(a) + b_.var(ddof=1) / len(b_))), 'p': float(p_)}
cmp6['ours_inter_ch 1.5e-3 − BPSR 3e-11'] = {k: wdiff(TG[1][1], TG[2][1], key, i) for k, key, i in
                                             (('last_step_share', 'step_share', 3), ('first_step_share', 'step_share', 0),
                                              ('mean_first_t', 'mean_first_t', 0), ('first_at_t4', 'first_share_of_fired', 3))}
a = np.array(drop_list(TG[1][1], 1)); b_ = np.array(drop_list(TG[2][1], 1))
if len(a) >= 2 and len(b_) >= 2:
    t_, p_ = stats.ttest_ind(a, b_, equal_var=False)
    cmp6['ours_inter_ch 1.5e-3 − BPSR 3e-11']['drop_T2'] = {'diff': float(a.mean() - b_.mean()),
        'se': float(np.sqrt(a.var(ddof=1) / len(a) + b_.var(ddof=1) / len(b_))), 'p': float(p_)}
sec6['compare'] = cmp6
res['steps'] = sec6

res['runs'] = {t: {k: v for k, v in r.items() if k != 'log'} | {'log': {k: v for k, v in r['log'].items()
                                                                       if not k.endswith('_traj')}}
               for t, r in runs.items()}
json.dump(res, open(OUT_JSON, 'w'), indent=1, default=float)
print('json', OUT_JSON)


# ================================================================ md
def F(v, d=2, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '—'
    return (f'{v:+.{d}f}' if sign else f'{v:.{d}f}')


def P(p):
    if p is None or not np.isfinite(p):
        return '—'
    return f'{p:.2g}' if p >= 1e-4 else f'{p:.1e}'


def K(v):
    return f'{v:,.0f}'


def ratio_txt(c, se):
    """ln 오프셋 → % 와 ±SE(%)."""
    return f'{100 * (np.exp(c) - 1):+.1f}% ±{100 * se:.1f}'


def lam_txt(v):
    return f'{v:.3g}' if v is not None else '—'


def run_row(r, extra=()):
    lg = r['log']
    cells = [r['arm'], f"{r['rho']:g}", f"s{r['seed']}", r['server'], F(r['train_loss'], 4), F(r['train_acc']),
             F(r['val_loss'], 4), F(r['val_acc']), K(r['spikes'])]
    for k, d in extra:
        v = lg.get(k)
        cells.append(lam_txt(v) if k.startswith('lam') else (F(v, d) if v is not None else '—'))
    return '| ' + ' | '.join(cells) + ' |'


def mean_row(grp, label, extra=()):
    def ms(k, d, src=None):
        v = [(q['log'].get(k) if src == 'log' else q[k]) for q in grp]
        v = [x for x in v if x is not None]
        if not v:
            return '—'
        if len(v) == 1:
            return F(v[0], d) if d is not None else K(v[0])
        return (F(np.mean(v), d) if d is not None else K(np.mean(v))) + ' ±' + (F(np.std(v, ddof=1), d) if d is not None else K(np.std(v, ddof=1)))
    cells = [f'**{label}**', '', f'**n={len(grp)}**', '', ms('train_loss', 4), ms('train_acc', 2), ms('val_loss', 4),
             f"**{ms('val_acc', 2)}**", f"**{ms('spikes', None)}**"]
    for k, d in extra:
        v = [q['log'].get(k) for q in grp if q['log'].get(k) is not None]
        if not v:
            cells.append('—')
        elif k.startswith('lam'):
            cells.append(f'{np.mean(v):.3g}')
        else:
            cells.append(F(np.mean(v), d) + (' ±' + F(np.std(v, ddof=1), d) if len(v) > 1 else ''))
    return '| ' + ' | '.join(cells) + ' |'


J = res
v1 = J['vmem']; v2 = J['final_step']; v3 = J['loss_ratio']; v4 = J['inversion']; v5 = J['bpsr']; v6 = J['steps']
o = []
w = o.append
sr = v1['same_rho_spikes']; sl = v1['same_lam_mean_spikes']; ia = v1['intra']
nof = v2['same_rho'][0] if v2['same_rho'] else None
nol = v3['nolr'][0] if v3['nolr'] else None
noi = v4['rows'][0] if v4['rows'] else None
mainb = [b for b in v5['bands'] if b['band'] == [300e3, 420e3]][0]
pw = v5['power']
st6 = v6.get('compare', {})
s_int = v6.get('ours_inter_ch 1.5e-3'); s_base = v6.get('baseline'); s_bp = v6.get('BPSR 3e-11')
n_bp_t = s_bp['n'] if s_bp else 0

w('# ours_inter_ch 어블레이션 — 요소별 기전과 BPSR 대비 우위 위치 (R19-C10, 26-10-10)\n')
w(f"생성: `analysis/ablation_mech_261010.py` → `.json`, 이 문서. 원장 {J['generated_from']['inventory_generated']} 집계 "
  "(310 에폭 완주·S30/S1≥0.19·seed<900). 학습 로그 = 두 서버 런 폴더의 `train.log`·`reg_detail.csv`·`reg_neuron_detail.csv` "
  "(10-10 13:27~13:30 읽음). npz = `_npz`/`deep1005/npz` (뉴런별 총합), `_npz_t` (스텝별, 10-10 추출분). "
  "ours_inter_ch 의 런 키는 `ours_ch`, 어블레이션 키는 `ch_novmem`·`ch_nofinal`·`ch_nolr`·`ch_noinv` (한 요소만 끈 팔).\n")

w('## 요약\n')
w(f"1. **vmem — 같은 ρ·같은 λ 에서 더 많이 쏘게 한다 (스파이크 감소를 누그러뜨린다). 정확도 이득은 경향만.** "
  f"inter 에서 `- vmem` 은 같은 ρ 에서 스파이크 {ratio_txt(sr['c'], sr['se'])} (p={P(sr['p'])}, inter n={sr['n0']} / `- vmem` n={sr['n1']}), "
  f"같은 λ (학습 평균) 에서 {ratio_txt(sl['c'], sl['se'])} (p={P(sl['p'])}). 이유: vmem 이 채널 가중 w 를 낮춘다 — 같은 스파이크에서 평균 w "
  f"`- vmem` {F(v1['mech_same_spk']['w_mean']['c'], 3, True)} ±{F(v1['mech_same_spk']['w_mean']['se'], 3)} (p={P(v1['mech_same_spk']['w_mean']['p'])}), "
  f"벌점 크기 R {ratio_txt(v1['mech_same_spk']['R_rd']['c'], v1['mech_same_spk']['R_rd']['se'])}. "
  f"intra 에서는 효과가 훨씬 크다: ρ 4e-3 에서 {K(ia['spikes']['prop_mean'])} vs {K(ia['spikes']['nov_mean'])} (각 n=5, `- vmem` {100*(np.exp(ia['spikes']['diff'])-1):+.1f}%, p={P(ia['spikes']['p'])}). "
  f"같은 스파이크 정확도는 `- vmem` 이 inter {F(v1['acc_pooled']['delta'], 3, True)} ±{F(v1['acc_pooled']['se'], 3)} (p={P(v1['acc_pooled']['p'])}, n=6), "
  f"intra {F(ia['acc_same_spk']['c'], 3, True)} ±{F(ia['acc_same_spk']['se'], 3)} (p={P(ia['acc_same_spk']['p'])}, n={ia['acc_same_spk']['n0']}/{ia['acc_same_spk']['n1']}) 로 둘 다 낮은 쪽이나 각각은 유의하지 않다.")
if nof:
    w(f"2. **final_step — 스파이크를 마지막 스텝으로 몰지 않는다. 바꾸는 것은 '같은 ρ 에서 얼마나 줄이나' 다.** "
      f"`- final_step` (n=1) 은 같은 ρ 에서 스파이크 {100*(nof['ratio']-1):+.0f}% (z={F(nof['z'],1)}), 같은 λ 에서 {100*(nof['same_lam']['ratio']-1):+.0f}% (z={F(nof['same_lam']['z'],1)}) 로 훨씬 세게 줄이고, "
      f"같은 스파이크 정확도는 {F(nof['acc_delta'],2,True)} ±{F(nof['acc_se'],2)} 로 차이 없다. 시간축 (npz_t): inter (n={s_int['n'] if s_int else 0}) 의 마지막 스텝 몫은 baseline (n={s_base['n'] if s_base else 0}) 보다 "
      f"{F(st6['ours_inter_ch 1.5e-3 − baseline']['last_step_share']['diff'],2,True)}%p (p={P(st6['ours_inter_ch 1.5e-3 − baseline']['last_step_share']['p'])}) 이고, final_step 이 없는 BPSR (n={n_bp_t}) 은 "
      f"{F(st6['BPSR 3e-11 − baseline']['last_step_share']['diff'],2,True)}%p — 스파이크를 줄이면 앞 스텝이 조금 비는 것은 두 방법 공통이고 inter 쪽이 더 작다. `- final_step` npz 가 없어 요소 단독의 시간 효과는 재지 못했다.")
if nol:
    li = v3['inter_lam'].get('0.003')
    w(f"3. **loss-ratio — λ 를 학습 내내 움직여 ρ 를 고정한다; 고정 λ 는 실효 ρ 가 {F(min(nol['rho_eff_at'].values())*1e3,1)}e-3 → {F(max(nol['rho_eff_at'].values())*1e3,1)}e-3 로 1.6배 흔들린다.** "
      f"inter ρ 3e-3 (n={li['n']}) 의 λ 는 5에폭 {li['lam_at'][4]:.3g} → 200에폭 직전 {li['lam_at'][199]:.3g} (+{100*(li['lam_at'][199]/li['lam_at'][4]-1):.0f}%) → 혼합 증강 해제(200에폭) 뒤 {li['lam_at'][309]:.3g}. "
      f"착지 스파이크의 시드 변동은 inter 가 오히려 크다: 같은 ρ 묶음 합동 CV {F(v3['landing_cv']['inter'][0],2)}% (자유도 {v3['landing_cv']['inter'][1]}) vs 고정 λ 인 BPSR {F(v3['landing_cv']['bpsr_fixed_lambda'][0],2)}% (자유도 {v3['landing_cv']['bpsr_fixed_lambda'][1]}), F 검정 p={P(v3['landing_cv']['p'])}. "
      f"`- loss-ratio` 는 n=1 이라 그 팔 자체의 시드 변동은 **판단 불가**. 같은 스파이크 정확도 {F(nol['acc_delta'],2,True)} ±{F(nol['acc_se'],2)}.")
if noi:
    w(f"4. **1−inversion — 분포 모양을 정반대로 만든다 (n=1 이지만 효과가 시드 산포의 20배 이상).** `- 1- inversion` (w = 채널합/최대) 은 같은 스파이크의 inter 대비 평균 w "
      f"{F(noi['m_w_mean']['value'],2)} vs {F(noi['m_w_mean']['inter_pred'],2)}, 층 스파이크 가중 지니 {F(noi['m_gini_sw']['delta'],3,True)} (z={F(noi['m_gini_sw']['z'],0)}), 상위 10% 몫 "
      f"{F(noi['m_top10_sw']['delta'],1,True)}%p (z={F(noi['m_top10_sw']['z'],0)}) — 큰 채널이 눌려 발화가 고르게 퍼진다. 정확도는 {F(noi['acc_delta'],2,True)} ±{F(noi['acc_se'],2)} 로 차이 없다. "
      "채널 수(죽은 채널)는 이 팔 npz 가 없어 못 셌다.")
w(f"5. **BPSR 대비 — 정확도로는 지금도, 현실적인 n 으로도 우위를 못 보인다. 주장은 구조 지표로 둔다.** 같은 스파이크 Δacc (inter − BPSR): 300~420K {F(mainb['c'],3,True)} ±{F(mainb['se'],3)} (p={P(mainb['p'])}, n={mainb['n_inter']}/{mainb['n_bpsr']}); "
  f"BPSR 묶음별로는 412K {F(v5['local'][0]['delta'],2,True)}, 319K {F(v5['local'][1]['delta'],2,True)}, 237K {F(v5['local'][2]['delta'],2,True)} (p={P(v5['local'][2]['p'])}) — 높은 구간에서만 inter 가 앞서는 경향. "
  f"+{F(pw['delta'],3)} 를 80% 검정력으로 보이려면 팔당 약 {pw['n_per_arm_80']:.0f}런 (잔차 SD {F(pw['sigma'],3)}) 이 필요하고, 조건당 n=4 상한에서 보이는 최소 차는 ±{F(pw['mde_n4'],2)}%p 다. "
  f"반면 같은 스파이크 (300~420K) 에서 죽은 채널 {F(v5['struct_ancova']['300-420K']['dead_ch']['c'],1,True)} ±{F(v5['struct_ancova']['300-420K']['dead_ch']['se'],1)}/4,736 (p={P(v5['struct_ancova']['300-420K']['dead_ch']['p'])}), "
  f"제거 시 MACs {F(v5['struct_ancova']['300-420K']['macs_rm_pct']['c'],2,True)} ±{F(v5['struct_ancova']['300-420K']['macs_rm_pct']['se'],2)}%p, 이미지당 활성 뉴런 {ratio_txt(v5['struct_ancova']['300-420K']['active_img']['c'], v5['struct_ancova']['300-420K']['active_img']['se'])} 는 이미 유의하다 (재계산).")
if s_int and s_base:
    w(f"6. **스텝별 정확도 (보너스).** T=4 로 학습한 망을 2스텝에서 자르면 (스파이크 약 {F(s_int['cum_spk_share'][1],0)}%만 씀) 정확도 변화가 inter {F(s_int['drop_vs_T4'][1],2,True)} ±{F(s_int['drop_vs_T4_sd'][1],2)} (n={s_int['n']}), "
      f"baseline {F(s_base['drop_vs_T4'][1],2,True)} ±{F(s_base['drop_vs_T4_sd'][1],2)} (n={s_base['n']})" + (f", BPSR {F(s_bp['drop_vs_T4'][1],2,True)} (n={s_bp['n']})" if s_bp else '') +
      f"; 3스텝 (약 {F(s_int['cum_spk_share'][2],0)}%) 은 모두 −0.1 이내. inter − baseline 차 {F(st6['ours_inter_ch 1.5e-3 − baseline']['drop_T2']['diff'],2,True)} ±{F(st6['ours_inter_ch 1.5e-3 − baseline']['drop_T2']['se'],2)} (p={P(st6['ours_inter_ch 1.5e-3 − baseline']['drop_T2']['p'])}) — n=3 이라 **아직 판단 불가**.")
w('')

w('## 정의\n')
w('- **ρ**: loss-ratio 목표. 매 에폭 끝에 λ = ρ·L/R 로 다음 에폭 λ 를 정한다 (L = 과제 손실, R = 가중 스파이크 벌점의 원값). 그래서 ρ 가 같으면 "벌점/과제 손실" 비가 같다.')
w('- **λ 평균**: `train.log` 의 `adp_lambda` 를 310 에폭 평균한 값. **λ 마지막**: 310 에폭 값.')
w('- **w (채널 가중)**: inter 의 벌점 가중 `sc_rate` = 1 − (채널 공간합)/(층 안 최대 채널 공간합). 1등 채널 0, 침묵 채널 1. vmem 이 켜지면 한 번도 안 쏜 위치는 0 대신 막전위/문턱 (0~1) 을 채널 합에 더한다. `w 평균` = `reg_detail.csv` 의 `sc_rate_mean` 을 뉴런 수로 가중한 18층 평균 (마지막 에폭, 학습 모드).')
w('- **R**: 18층 `sc_loss` 합 = Σ_층 ‖S∘w‖₂ (S = 이미지별 T 합 스파이크, 배치 평균). 같은 스파이크에서 R 이 크면 스파이크 하나에 걸린 가중이 크다.')
w('- **학습 모드 지표** (`reg_detail`/`reg_neuron_detail`, 마지막 에폭 배치 평균): 층 몫 = 단계 스파이크/전체, 침묵 % = 이미지 하나에서 한 번도 안 쏜 뉴런 비율 (뉴런 수 가중), 지니·상위 10% 몫 = 이미지마다 층 안 뉴런 스파이크의 집중도 (층 스파이크 가중 평균). 원장 스파이크 (평가 모드, 베스트 에폭) 와 다른 자료다.')
w('- **같은 스파이크 Δacc**: inter 완주 런 200~420K 직선 `val acc = a + b·ln(spikes)` 예측값과의 차. ± 는 새 런 하나의 예측 SE (곡선 SE + 잔차 SD). ANCOVA = `y = a + b·ln(x) + c·[팔]`, c 가 오프셋.')
w('- **npz_t**: 테스트 1만 장 1회 추론. `__t` = 스텝별 스파이크 합, `__first_t` = (뉴런, 이미지) 쌍의 최초 발화 스텝 히스토그램, `logits_t` = 각 스텝까지 돌린 출력. 스파이킹 18층.')
w('- 단계: conv1 (1층, 32×32), conv2 (6층, 32×32), conv3 (6층, 16×16), conv4 (4층, 8×8), fc1.\n')

# ---------------- 1. vmem
w('## 1. vmem (막전위 준비도 가중)\n')
w('### 1a. 착지값 — 같은 ρ 의 inter 와 `- vmem`\n')
EXTRA = [('lam_mean', None), ('lam_end', None), ('w_mean', 3), ('R_rd', 0), ('tr_spk', 0)]
hdr = '| 팔 | ρ | seed | 서버 | train loss | train acc | val loss | val acc | spikes | λ 평균 | λ 마지막 | w 평균 | R | 학습 모드 스파이크 |'
w(hdr); w('|' + '---|' * hdr.count('|')[:-1] if False else '|' + '---|' * (hdr.count('|') - 1))
for rho in sorted(set(r['rho'] for r in NOV)):
    g = [r for r in INTER if r['rho'] == rho]
    for r in g:
        w(run_row(r, EXTRA))
    if g:
        w(mean_row(g, f'ours_inter_ch {rho:g} 평균', EXTRA))
    for r in [q for q in NOV if q['rho'] == rho]:
        w(run_row(r, EXTRA))
w('')
w('- `- vmem` 은 ρ 마다 n=1. 2e-3 시드 2~4 는 미완주 (7절).')
w('- 짝별 스파이크 비 (`- vmem` / inter 평균): ' + ', '.join(f"ρ {p['rho']:g} {p['ratio']:.3f} (inter n={p['inter_n']})" for p in v1['same_rho_pairs']) + '.\n')

w('### 1b. "같은 값에서 더 많이 쏘나" — 세 가지 정의\n')
w('| 정의 | 모형 | n (inter / `- vmem`) | `- vmem` 의 스파이크 (inter 대비) | p | 기울기 차 p |')
w('|---|---|---|---|---|---|')
w(f"| 같은 ρ | ln(spikes) ~ ln ρ + [팔], ρ 1e-3~5e-3 | {sr['n0']} / {sr['n1']} | {ratio_txt(sr['c'], sr['se'])} | {P(sr['p'])} | {P(sr.get('slope_diff_p'))} |")
for lk, nm in (('lam_mean', '같은 λ (310 에폭 평균)'), ('lam_pre', '같은 λ (0~199 에폭 평균)'), ('lam_end', '같은 λ (마지막 에폭)')):
    q = v1[f'same_{lk}_spikes']
    w(f"| {nm} | ln(spikes) ~ ln λ + [팔] | {q['n0']} / {q['n1']} | {ratio_txt(q['c'], q['se'])} | {P(q['p'])} | {P(q.get('slope_diff_p'))} |")
q = v1['same_rho_lam_mean']
w(f"| (참고) 같은 ρ 에서 λ 평균 | ln λ ~ ln ρ + [팔] | {q['n0']} / {q['n1']} | λ {ratio_txt(q['c'], q['se'])} | {P(q['p'])} | {P(q.get('slope_diff_p'))} |")
iq = ia['same_rho_spikes']
w(f"| intra: 같은 ρ (ρ 1.8e-3~5e-3) | ln(spikes) ~ ln ρ + [팔] | {iq['n0']} / {iq['n1']} | {ratio_txt(iq['c'], iq['se'])} | {P(iq['p'])} | {P(iq.get('slope_diff_p'))} |")
w(f"| intra: ρ 4e-3 직접 | 두 평균 | 5 / 5 | {K(ia['spikes']['prop_mean'])} ±{K(ia['spikes']['prop_sd'])} → {K(ia['spikes']['nov_mean'])} ±{K(ia['spikes']['nov_sd'])} ({100*(np.exp(ia['spikes']['diff'])-1):+.1f}%) | {P(ia['spikes']['p'])} | — |")
w('')
w('- 답: **그렇다.** vmem 이 있으면 같은 ρ 에서 inter 는 약 5~6%, intra 는 약 27~30% 더 쏜 채로 착지한다. 같은 λ 로 맞추면 inter 차이는 더 커진다 (−7%) — vmem 이 R 을 낮춰 loss-ratio 가 λ 를 오히려 6% 올려 주는데도 스파이크가 더 남기 때문이다.')
w('- 뉴런 하나하나의 "더 쏘는 뉴런" 은 `- vmem` 팔의 npz 가 없어 재지 못했다 (inter·intra 모두). 아래 1c·1d 는 층·단계 수준의 대리 지표다.\n')

w('### 1c. 기전 — vmem 은 스파이크에 걸리는 가중을 낮춘다\n')
w(f"같은 학습 모드 스파이크 (원장 {int(BAND_V[0]/1e3)}~{int(BAND_V[1]/1e3)}K) 에서 `- vmem` − inter. 오프셋 c (± SE):\n")
w('| 지표 | inter n | `- vmem` n | c | p | 기울기 b (inter 쪽) |')
w('|---|---|---|---|---|---|')
for k, nm in (('R_rd', 'R (ln)'), ('w_mean', 'w 평균 (18층)'), ('w_conv1', 'w conv1'), ('w_conv2', 'w conv2'), ('w_conv3', 'w conv3'), ('w_conv4', 'w conv4')):
    q = v1['mech_same_spk'][k]
    c = ratio_txt(q['c'], q['se']) if q['log_y'] else f"{F(q['c'],3,True)} ±{F(q['se'],3)}"
    w(f"| {nm} | {q['n0']} | {q['n1']} | {c} | {P(q['p'])} | {F(q['b'],3)} |")
q = ia['dist']['w_mean']; qr = ia['R_same_spk']
w(f"| intra: w 평균 | {q['n0']} | {q['n1']} | {F(q['c'],3,True)} ±{F(q['se'],3)} | {P(q['p'])} | {F(q['b'],3)} |")
w(f"| intra: R (ln) | {qr['n0']} | {qr['n1']} | {ratio_txt(qr['c'], qr['se'])} | {P(qr['p'])} | {F(qr['b'],3)} |")
w('')
w('- inter: vmem 을 끄면 같은 스파이크에서 평균 w 가 0.07 오르고 (모든 단계, 뒤 단계일수록 크게: conv1 +0.03 → conv3·conv4 +0.08) R 이 약 14% 커진다. 침묵 위치가 막전위만큼 채널 합에 더해져 **거의 쏠 뻔한 채널** 의 순위가 올라가고 w 가 내려가기 때문이다. 쏜 뉴런이 받는 벌점 기울기는 λ·w²/‖S∘w‖ (층마다) 라 분모도 함께 줄어 정확히 비례하지는 않지만, 결과로 같은 스파이크에서 R 이 14% 작다 — 같은 λ 에서 벌점이 덜 걸린다 → 1b 의 "같은 λ 에서 더 쏨". (해석은 `lib_snn/neurons.py` 976–1004 의 코드 경로에서 추론한 것이다.)')
w('- intra: w 평균은 마찬가지로 오르지만 (+0.056) R 은 같은 스파이크에서 차이가 없다 (p=' + P(qr['p']) + '). intra 에서 vmem 이 바꾸는 것은 **쏘는 뉴런의 가중이 아니라 침묵 뉴런의 가중** — wta_rev 역전파는 안 쏜 뉴런에도 w² 만큼 기울기를 주는데, vmem 이 없으면 침묵 뉴런 전부가 w=1 로 최대한 눌리고, 있으면 문턱 근처 뉴런의 w 가 내려가 덜 눌린다. intra 의 착지 차이 (+27~30%) 가 inter (+5~6%) 보다 훨씬 큰 것과 맞다 (inter 는 채널 단위라 이 효과가 채널 평균으로 희석된다). 이 해석은 코드 경로 (`lib_snn/neurons.py` 976–1004, `lib_snn/layers.py` l2_norm_wta_rev) 와 위 수치로 추론한 것이고 뉴런 단위로 확인하지 않았다.\n')

w('### 1d. 층 배분과 분포 — 같은 스파이크에서 `- vmem` − inter\n')
w('| 지표 | inter n | `- vmem` n | c ± SE | p |')
w('|---|---|---|---|---|')
lab = {'share_conv1': '층 몫 conv1 (%)', 'share_conv2': '층 몫 conv2 (%)', 'share_conv3': '층 몫 conv3 (%)', 'share_conv4': '층 몫 conv4 (%)',
       'dead_pct': '이미지당 침묵 뉴런 % (18층)', 'gini_sw': '지니 (층 스파이크 가중)', 'top10_sw': '상위 10% 뉴런 몫 (%)',
       'dead_conv1': '침묵 % conv1', 'dead_conv2': '침묵 % conv2', 'dead_conv3': '침묵 % conv3', 'dead_conv4': '침묵 % conv4',
       'gini_conv1': '지니 conv1', 'gini_conv2': '지니 conv2', 'gini_conv3': '지니 conv3', 'gini_conv4': '지니 conv4'}
for k, q in v1['dist_same_spk'].items():
    d = 3 if 'gini' in k else 2
    w(f"| {lab[k]} | {q['n0']} | {q['n1']} | {F(q['c'],d,True)} ±{F(q['se'],d)} | {P(q['p'])} |")
w('')
w('- 차이는 모두 1%p 이하로 작다. p<0.05 인 것 (15개 지표 중): `- vmem` 이 conv3 몫 +0.60%p, conv1 침묵 +0.78%p, conv3·conv4 침묵 −0.30·−0.06%p, 지니 −0.006. Bonferroni 기준 (0.05/15=0.0033) 을 통과하는 지표는 없다 (가장 작은 p = conv3 몫 ' + P(v1['dist_same_spk']['share_conv3']['p']) + '). 하나하나를 결론으로 쓰지 않는다.')
w('- 방향만 읽으면: vmem 이 있으면 **conv1 에서 이미지당 침묵 뉴런이 적고** (문턱 근처 뉴런이 가끔 쏜다) 그만큼 conv3 몫이 줄어든다. "vmem 이 더 쏘게 하는 뉴런 = 입력층 쪽의 문턱 근처 뉴런" 가설과 맞지만, 뉴런 단위 확인 (baseline 발화율 중간대인지) 은 `- vmem` npz 가 필요하다.')
qd = ia['dist']
w(f"- intra (n={ia['n_dist'][0]}/{ia['n_dist'][1]}): `- vmem` 이 상위 10% 몫 {F(qd['top10_sw']['c'],2,True)} ±{F(qd['top10_sw']['se'],2)}%p (p={P(qd['top10_sw']['p'])}), conv1 몫 {F(qd['share_conv1']['c'],2,True)} ±{F(qd['share_conv1']['se'],2)}%p (p={P(qd['share_conv1']['p'])}), 침묵 % {F(qd['dead_pct']['c'],2,True)} (p={P(qd['dead_pct']['p'])}). intra 에서는 vmem 이 스파이크를 덜 몰리게 한다.\n")

w('### 1e. 같은 스파이크 정확도 (원장)\n')
w('| 런 | spikes | inter 곡선 예측 | Δacc ± SE | z |')
w('|---|---|---|---|---|')
for q in v1['acc_rows']:
    w(f"| {q['tag']} | {K(runs[q['tag']]['spikes'])} | {F(q['pred'])} | {F(q['delta'],2,True)} ±{F(q['se'],2)} | {F(q['z'],1,True)} |")
ap = v1['acc_pooled']; ac = v1['acc_combined_posthoc']
w(f"| *inter `- vmem` 합동 (역분산)* | | | *{F(ap['delta'],3,True)} ±{F(ap['se'],3)}* (p={P(ap['p'])}, n={ap['n']}) | |")
w(f"| *intra `- vmem` ANCOVA (전 구간)* | | | *{F(ia['acc_same_spk']['c'],3,True)} ±{F(ia['acc_same_spk']['se'],3)}* (p={P(ia['acc_same_spk']['p'])}, n={ia['acc_same_spk']['n0']}/{ia['acc_same_spk']['n1']}) | |")
w(f"| *두 범위 사후 합산* | | | *{F(ac['delta'],3,True)} ±{F(ac['se'],3)}* (p={P(ac['p'])}) | |")
w('')
w(f"- 10-09 문서의 `- vmem` 합동 −0.067 ±0.047 (p=0.16) 과 점추정이 같다. SE 차이는 합동 방식 (10-09 는 평균의 SE, 여기는 역분산 가중) 때문이다.")
w(f"- 두 경쟁 범위를 합치면 p={P(ac['p'])} 이지만 **결과를 보고 합친 사후 합산**이고, intra 와 inter 는 다른 방법이다. 논문에는 \"vmem 을 빼면 같은 스파이크에서 정확도가 0.07%p 낮은 경향 (각 범위에서 p=0.11·0.20)\" 까지만 쓸 수 있다.")
w(f"- intra ρ 4e-3 직접 비교 (각 n=5): 정확도 {F(ia['val_acc']['prop_mean'])} ±{F(ia['val_acc']['prop_sd'])} vs {F(ia['val_acc']['nov_mean'])} ±{F(ia['val_acc']['nov_sd'])} (Δ {F(ia['val_acc']['diff'],2,True)} ±{F(ia['val_acc']['se'],2)}, p={P(ia['val_acc']['p'])}) — 그러나 스파이크가 −23% 달라 같은 비용 비교가 아니다.\n")

# ---------------- 2. final_step
w('## 2. final_step (T 합산에만 벌점)\n')
if nof:
    r = runs[nof['tag']]
    w('### 2a. 착지값\n')
    hdr = '| 팔 | ρ | seed | 서버 | train loss | train acc | val loss | val acc | spikes | λ 평균 | λ 마지막 | w 평균 | R | 학습 모드 스파이크 |'
    w(hdr); w('|' + '---|' * (hdr.count('|') - 1))
    g = [q for q in INTER if q['rho'] == 2e-3]
    for q in g:
        w(run_row(q, EXTRA))
    w(mean_row(g, 'ours_inter_ch 2e-3 평균', EXTRA))
    w(run_row(r, EXTRA))
    w('')
    w('| 비교 | inter 예측 (같은 ρ/λ) | `- final_step` | 비 | z (inter 시드 산포 기준) |')
    w('|---|---|---|---|---|')
    w(f"| 같은 ρ (2.2e-3) 착지 스파이크 | {K(nof['inter_pred_spk'])} | {K(nof['spikes'])} | {nof['ratio']:.3f} | {F(nof['z'],1)} |")
    w(f"| 같은 λ 평균 ({nof['lam_mean']:.3g}) 착지 스파이크 | {K(nof['same_lam']['inter_pred_spk'])} | {K(nof['spikes'])} | {nof['same_lam']['ratio']:.3f} | {F(nof['same_lam']['z'],1)} |")
    w(f"| 같은 ρ 에서 λ 평균 | {nof['lam_vs_inter']['inter_pred_lam']:.3g} | {nof['lam_mean']:.3g} | {nof['lam_vs_inter']['ratio']:.3f} | — |")
    w(f"| 같은 스파이크 정확도 | {F(nof['acc_pred'])} | {F(r['val_acc'])} | Δ {F(nof['acc_delta'],2,True)} ±{F(nof['acc_se'],2)} | {F(nof['acc_z'],1,True)} |")
    for k in ['share_conv1', 'share_conv2', 'share_conv3', 'share_conv4', 'dead_pct', 'gini_sw', 'top10_sw', 'w_mean']:
        q = nof.get('m_' + k)
        if q:
            d = 3 if k in ('gini_sw', 'w_mean') else 2
            w(f"| 같은 스파이크 {lab.get(k, 'w 평균')} | {F(q['inter_pred'],d)} | {F(q['value'],d)} | Δ {F(q['delta'],d,True)} ±{F(q['se'],d)} | {F(q['z'],1,True)} |")
    w('')
    ia2 = v2['intra_same_rho_ancova']; ib2 = v2['intra_acc_same_spk']
    w(f"- `- final_step` 은 n=1 이다. 그러나 착지 차 (z=−8~−10) 는 inter 시드 산포 (ln 잔차 SD {F(sr['resid_sd'],3)}) 의 8배 이상이라 방향은 확실하다: **final_step 을 끄면 같은 ρ·같은 λ 에서 스파이크를 22~25% 더 줄인다.** 층 배분·분포·w 는 같은 스파이크에서 inter 와 구분되지 않는다 (|z|≤1.3).")
    w(f"- intra 도 같다: `abl_nofinal` (n=4, ρ 2e-3~4e-3) 은 같은 ρ 에서 {ratio_txt(ia2['c'], ia2['se'])} (p={P(ia2['p'])}, ours_intra_ch n={ia2['n0']}), 같은 스파이크 정확도 {F(ib2['c'],3,True)} ±{F(ib2['se'],3)} (p={P(ib2['p'])}). 개별: " +
      ', '.join(f"{q['tag'].split('-',1)[1]} {K(q['spikes'])} / {F(q['val_acc'])}" + (f" (ours_intra_ch 같은 ρ n={q['prop_n']}: {K(q['prop_spk'])} / {F(q['prop_acc'])})" if q['prop_n'] else ' (같은 ρ ours_intra_ch 없음)') for q in v2['intra_same_rho']) + '.')
    w('- 주의: final_step 을 끄면 R 의 정의가 바뀐다 (매 스텝 누적 스파이크로 가중을 다시 계산해 스텝마다 벌점). 그래서 "같은 ρ" 는 같은 세기가 아니다. 같은 λ 비교도 같은 결론이다 (−25%).')
    w('- 해석: final_step 은 정확도-스파이크 곡선을 옮기지 않고 **ρ 눈금을 늘린다** (같은 ρ 에서 덜 줄임). 곡선 위 위치로는 이득이 안 보인다 (10-09 결론과 같음). 요소로서의 근거는 "가중을 T 전체 발화로 정한다" 는 설계 논리이고, 수치로는 보이지 않는다.\n')

w('### 2b. 시간축 — 스텝별 스파이크 비중과 최초 발화 스텝 (npz_t)\n')
avail = ', '.join(f"{nm} n={v6[nm]['n']}" for nm in v6 if nm != 'compare')
miss_ = [t for t in ('r19c10-bpsr-3e-11-s2', 'r19c10-bpsr-3e-11-s3') if t not in npzt or 'error' in npzt.get(t, {})]
w(f"사용 가능 런: {avail}." + (f" 추출 중이라 뺀 런: {', '.join(miss_)} (이 문서 생성 시점)." if miss_ else '') + "\n")
w('| 묶음 | n | spikes/이미지 (npz) | t1 % | t2 % | t3 % | t4 % | 최초 발화 t1 % | t2 % | t3 % | t4 % | 평균 최초 발화 스텝 | 침묵 (뉴런·이미지) % |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for nm in [x for x in v6 if x != 'compare']:
    q = v6[nm]
    spk = np.mean([npzt[t]['spk_per_img'] for t in q['tags']])
    ss = ' | '.join(F(x, 1) + (' ±' + F(y, 1) if np.isfinite(y) else '') for x, y in zip(q['step_share'], q['step_share_sd']))
    fs = ' | '.join(F(x, 1) + (' ±' + F(y, 1) if np.isfinite(y) else '') for x, y in zip(q['first_share'], q['first_share_sd']))
    w(f"| {nm} | {q['n']} | {K(spk)} | {ss} | {fs} | {F(q['mean_first_t'],3)}" + (f" ±{F(q['mean_first_t_sd'],3)}" if np.isfinite(q['mean_first_t_sd']) else '') + f" | {F(q['silent_pair_pct'],2)} |")
w('')
w('개별 런:\n')
w('| 런 | 서버 | spikes/이미지 (npz) | t1 % | t2 % | t3 % | t4 % | 평균 최초 발화 스텝 | 침묵 % |')
w('|---|---|---|---|---|---|---|---|---|')
for t, q in sorted(npzt.items()):
    if 'error' in q:
        continue
    w(f"| {t} | {runs[t]['server'] if t in runs else '—'} | {K(q['spk_per_img'])} | " + ' | '.join(F(x, 2) for x in q['step_share']) + f" | {F(q['mean_first_t'],3)} | {F(q['silent_pair_pct'],2)} |")
w('')
w('단계별 마지막 스텝 (t4) 몫 % / 평균 최초 발화 스텝:\n')
w('| 묶음 | ' + ' | '.join(STAGES) + ' |')
w('|---|' + '---|' * len(STAGES))
for nm in [x for x in v6 if x != 'compare']:
    q = v6[nm]
    w(f"| {nm} (n={q['n']}) | " + ' | '.join(f"{F(q[f'step_share_{s}'][3],1)} / {F(q[f'mean_first_t_{s}'],2)}" for s in STAGES) + ' |')
w('')
w('Welch 비교 (첫 묶음 = inter − baseline, 각 n=3):\n')
w('| 지표 | 차 ± SE | p |')
w('|---|---|---|')
c6 = st6.get('ours_inter_ch 1.5e-3 − baseline', {})
for k, nm in (('last_step_share', 't4 스파이크 몫 (%p)'), ('first_step_share', 't1 스파이크 몫 (%p)'), ('first_at_t4', '최초 발화가 t4 인 쌍의 몫 (%p)'), ('mean_first_t', '평균 최초 발화 스텝')):
    if k in c6:
        q = c6[k]; d = 3 if k == 'mean_first_t' else 2
        w(f"| {nm} | {F(q['diff'],d,True)} ±{F(q['se'],d)} | {P(q['p'])} |")
b6 = st6.get('BPSR 3e-11 − baseline', {})
ib6 = st6.get('ours_inter_ch 1.5e-3 − BPSR 3e-11', {})
for k, nm in (('last_step_share', 't4 스파이크 몫 (%p)'), ('first_step_share', 't1 스파이크 몫 (%p)'), ('first_at_t4', '최초 발화가 t4 인 쌍의 몫 (%p)'), ('mean_first_t', '평균 최초 발화 스텝')):
    if k in ib6:
        q = ib6[k]; d = 3 if k == 'mean_first_t' else 2
        w(f"| (inter − BPSR, n={q['n'][0]}/{q['n'][1]}) {nm} | {F(q['diff'],d,True)}" + (f" ±{F(q['se'],d)}" if np.isfinite(q['se']) else '') + f" | {P(q['p'])} |")
if b6:
    w(f"| (BPSR 3e-11 − baseline, BPSR n={b6['last_step_share']['n'][0]}) t4 몫 | {F(b6['last_step_share']['diff'],2,True)} | — |")
    w(f"| (BPSR 3e-11 − baseline) 평균 최초 발화 스텝 | {F(b6['mean_first_t']['diff'],3,True)} | — |")
w('')
w(f"- 답: **final_step 은 스파이크를 마지막 스텝에 몰지 않고 앞 스텝을 비우지도 않는다 (적어도 inter 전체로 보면).** inter 의 스텝 배분은 baseline 과 거의 같다 (t4 몫 {F(c6['last_step_share']['diff'],2,True)}%p, 평균 최초 발화 {F(c6['mean_first_t']['diff'],3,True)} 스텝, p≈0.04 이나 n=3/3). 스파이크를 비슷하게 (−14~15%) 줄인 BPSR — final_step 이 없는 방법 — 이 오히려 더 뒤로 민다 (t4 {F(b6['last_step_share']['diff'],2,True)}%p, 최초 발화 {F(b6['mean_first_t']['diff'],3,True)} 스텝, BPSR n={b6['last_step_share']['n'][0]}).")
w('- 즉 "스파이크를 줄이면 t1·t2 발화가 조금 늦어진다" 는 규제 일반의 현상이고 final_step 고유의 효과로 볼 근거가 없다. 차이는 conv1 에서 가장 크다 (단계 표).')
w('- `- final_step` 런의 npz 가 없어 final_step **단독** 효과 (켜고 끈 같은 방법 비교) 는 재지 못했다. 재려면 `r19c10-ch_nofinal-2p2e-3-s1` (서버 23) 1런을 같은 추출기로 돌리면 된다 (약 17분, GPU 1장).\n')

# ---------------- 3. loss-ratio
w('## 3. loss-ratio λ 제어\n')
w('### 3a. inter 의 λ 궤적 (`adp_lambda`, 묶음 평균)\n')
w('| ρ | n | 1에폭 | 5에폭 | 10에폭 | 30에폭 | 100에폭 | 200에폭 (직전) | 201에폭 | 310에폭 | 5~200에폭 최대/최소 | 310에폭 λ 시드 CV % |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|')
for k, q in v3['inter_lam'].items():
    la = q['lam_at']
    w(f"| {k} | {q['n']} | " + ' | '.join(f"{la[str(e)] if str(e) in la else la[e]:.3g}" for e in EPS) +
      f" | {F(q['max_over_min_pre'],2)} | {F(q['lam_cv_end'],2) if q['lam_cv_end'] is not None else '—'} |")
w('')
w('- 200에폭에서 λ 가 약 40% 떨어지는 것은 혼합 증강 해제 (`mix_off_iter = 500×200`) 로 과제 손실 L 이 1.20 → 0.74 로 내려가기 때문이다 (λ = ρ·L/R). 그 전 (5~200에폭) 에는 R 이 스파이크와 함께 줄어들어 λ 가 1.27~1.94배 오른다 — 스파이크가 줄수록 압력을 키워 **벌점/과제 손실 비를 일정하게** 유지한다.\n')
w('### 3b. 고정 λ (`- loss-ratio`) 의 실효 ρ\n')
if nol:
    r = runs[nol['tag']]
    hdr = '| 팔 | ρ/λ | seed | 서버 | train loss | train acc | val loss | val acc | spikes | λ 평균 | λ 마지막 | w 평균 | R | 학습 모드 스파이크 |'
    w(hdr); w('|' + '---|' * (hdr.count('|') - 1))
    g = [q for q in INTER if q['rho'] == 3e-3]
    for q in g:
        w(run_row(q, EXTRA))
    w(mean_row(g, 'ours_inter_ch 3e-3 평균', EXTRA))
    w(run_row(r, EXTRA).replace(f"| {r['rho']:g} |", f"| λ={r['rho']:g} |", 1))
    w('')
    w('| 에폭 | 1 | 5 | 10 | 30 | 100 | 200 | 201 | 310 | 평균 |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    w('| `- loss-ratio` 실효 ρ (= λ·R/L) | ' + ' | '.join(f"{nol['rho_eff_at'][str(e)] if str(e) in nol['rho_eff_at'] else nol['rho_eff_at'][e]:.4f}" for e in EPS) + f" | {nol['rho_eff_mean']:.4f} |")
    w('| inter 3e-3 (검산: 정의상 상수) | ' + ' | '.join('0.0030' for _ in EPS) + ' | 0.0030 |')
    w('')
    w(f"- 고정 λ 1.25e-7 은 학습 전반에 실효 ρ 0.0021~0.0026, 혼합 증강 해제 뒤 0.0035 로 움직인다. 착지 스파이크 {K(r['spikes'])} 는 inter 3e-3 평균 342,075 (n=4) 와 같고, 같은 스파이크 정확도 {F(nol['acc_delta'],2,True)} ±{F(nol['acc_se'],2)} 로 차이 없다.")
    w('- 실효 ρ 는 `reg_detail.csv` 의 `sc_loss` 합을 R, 학습 loss − λR 을 L 로 계산했다 (`- loss-ratio` 는 로그에 `reg_R` 이 없다).\n')
w('### 3c. 착지 스파이크·정확도의 시드 변동\n')
lc = v3['landing_cv']; asd = v3['acc_sd']
w('| 묶음 | λ 방식 | 같은 ρ/λ 묶음 (n≥3) 합동 자유도 | 착지 스파이크 CV % | val acc SD (%p) |')
w('|---|---|---|---|---|')
w(f"| ours_inter_ch | loss-ratio | {lc['inter'][1]} | {F(lc['inter'][0],2)} | {F(asd['inter'][0],3)} |")
w(f"| ours_intra_ch | loss-ratio | {lc['intra'][1]} | {F(lc['intra'][0],2)} | — |")
w(f"| BPSR | 고정 λ | {lc['bpsr_fixed_lambda'][1]} | {F(lc['bpsr_fixed_lambda'][0],2)} | {F(asd['bpsr'][0],3)} |")
w(f"| `- loss-ratio` | 고정 λ | 0 (n=1) | 판단 불가 | 판단 불가 |")
w('')
w(f"F 검정 (inter/BPSR 스파이크 CV 분산비) {F(lc['F_inter_vs_bpsr'],2)}, p={P(lc['p'])}.\n")
w('에폭별 학습 중 평가 스파이크 (`s_count`) 의 시드 간 CV %:\n')
w('| 묶음 | n | 10 | 30 | 100 | 200 | 250 | 310 |')
w('|---|---|---|---|---|---|---|---|')
for k, q in v3['ep_cv'].items():
    w(f"| {k} | {q['n']} | " + ' | '.join(F(q['cv'][str(e)] if str(e) in q['cv'] else q['cv'][e], 2) for e in EPS2) + ' |')
w('')
w('- 답: λ 는 loss-ratio 아래에서 학습 내내 움직이고 (전반 +27~94%, 200에폭 −40%), 그 결과 **실효 ρ 가 일정**하다. 고정 λ 는 같은 착지점을 줄 수 있지만 실효 ρ 가 1.6배 흔들린다.')
w('- "loss-ratio 가 착지를 더 일정하게 한다" 는 **지지되지 않는다**: inter 의 착지 CV 는 고정 λ 인 BPSR 의 2배 (p=0.057). 다만 BPSR 은 벌점 모양도 다르므로 이 비교는 λ 방식만의 효과가 아니다. 요소 단독 (같은 inter 에서 고정 λ) 의 시드 변동은 `- loss-ratio` n=1 이라 판단 불가 — 재려면 같은 λ 로 시드 2런이 더 필요하다.')
w('- 이 요소의 실익으로 쓸 수 있는 것은 "ρ 하나로 C10·C100·다른 모델에서 같은 눈금 (벌점/과제 손실 비)" 이라는 사용성 쪽이다. 수치 근거: R19 C10 에서 ρ 와 착지 스파이크의 관계가 ln-ln 직선 (기울기 ' + F(sr['b'], 3) + ', 잔차 SD ' + F(sr['resid_sd'], 3) + ', n=' + str(sr['n0']) + ').\n')

# ---------------- 4. inversion
w('## 4. 1−inversion (w = 1 − 채널합/최대)\n')
if noi:
    r = runs[noi['tag']]
    hdr = '| 팔 | ρ | seed | 서버 | train loss | train acc | val loss | val acc | spikes | λ 평균 | λ 마지막 | w 평균 | R | 학습 모드 스파이크 |'
    w(hdr); w('|' + '---|' * (hdr.count('|') - 1))
    for rho in (6e-3, 7e-3):
        g = [q for q in INTER if q['rho'] == rho]
        for q in g:
            w(run_row(q, EXTRA))
        w(mean_row(g, f'ours_inter_ch {rho:g} 평균', EXTRA))
    w(run_row(r, EXTRA))
    w('')
    w(f"같은 스파이크 ({K(noi['spikes'])}) 에서 inter 곡선 (180~330K, n={noi['m_w_mean']['n_fit']}) 예측 대비. z 는 inter 시드 산포 기준 (새 런 하나의 예측 SE):\n")
    w('| 지표 | inter 예측 | `- 1- inversion` | Δ ± SE | z |')
    w('|---|---|---|---|---|')
    for k in ['w_mean', 'w_conv1', 'w_conv2', 'w_conv3', 'w_conv4', 'gini_sw', 'top10_sw', 'gini_conv1', 'gini_conv2', 'gini_conv3', 'gini_conv4',
              'dead_pct', 'dead_conv1', 'dead_conv2', 'dead_conv3', 'dead_conv4', 'share_conv1', 'share_conv2', 'share_conv3', 'share_conv4']:
        q = noi.get('m_' + k)
        if q:
            d = 3 if ('gini' in k or k.startswith('w')) else 2
            w(f"| {lab.get(k, 'w 평균 (18층)' if k == 'w_mean' else k.replace('w_', 'w '))} | {F(q['inter_pred'],d)} | {F(q['value'],d)} | {F(q['delta'],d,True)} ±{F(q['se'],d)} | {F(q['z'],1,True)} |")
    w(f"| 정확도 (원장) | {F(noi['acc_pred'])} | {F(r['val_acc'])} | {F(noi['acc_delta'],2,True)} ±{F(noi['acc_se'],2)} | {F(noi['acc_z'],1,True)} |")
    w('')
    w('- 반전을 끄면 가중이 "많이 쏘는 채널일수록 세게" 로 뒤집힌다 (평균 w 0.81 → 0.34, conv4 0.20). 그 결과 같은 스파이크에서 **이미지 안 발화가 훨씬 고르게 퍼진다**: 지니 −0.15, 상위 10% 몫 −14.5%p (z −19~−27), conv2 지니 0.71 → 0.49. 이미지당 침묵 뉴런은 −0.6%p (더 많은 뉴런이 조금씩 쏜다).')
    w('- **어떤 채널이 눌리나**: 반전이 있으면 (inter) 작은 채널이 세게 (w≈1) 눌려 꺼지고 1등 채널은 면제, 반전이 없으면 큰 채널이 눌린다. inter 의 큰 지니·상위 10% 몫과 `- 1- inversion` 의 작은 값은 이 차이와 맞다.')
    w('- 층 배분도 달라진다: conv2 몫 −2.9%p, conv3 +2.9%p (z ±5.5~6.1).')
    w('- n=1 이지만 지니·몫의 z 가 −19~−53 이라 방향은 시드 산포로 설명되지 않는다. 정확도는 같은 스파이크에서 차이 없다.\n')
w('### 4b. 채널 수 (npz) — intra 의 `- 1- inversion` (참고)\n')
w('inter `- 1- inversion` 의 npz 는 없다. 대신 intra 의 `abl_noinv` 는 npz 가 있다. **이 팔은 반전·within_channel·vmem 세 요소를 함께 뺀 것** (decisions 09-20) 이라 반전 단독 효과가 아니다.\n')
w('| 팔 | ρ | seed | 서버 | val acc | spikes | 죽은 채널 (/4,736) | 제거 시 MACs 감소 % | 이미지당 활성 뉴런 | conv1 | conv2 | conv3 | conv4 | fc1 |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
ns_ = v4['npz_struct']
for key, nm in (('prop', 'ours_intra_ch'), ('abl_noinv', 'ours_intra_ch - 1- inversion (3요소)')):
    for rho in sorted(set(runs[x['tag']]['rho'] for x in ns_ if x['key'] == key)):
        g = [x for x in ns_ if x['key'] == key and runs[x['tag']]['rho'] == rho]
        for x in g:
            rr = runs[x['tag']]
            w(f"| {nm} | {rr['rho']:g} | s{rr['seed']} | {rr['server']} | {F(x['val_acc'])} | {K(x['spikes'])} | {x['dead_ch']} | {F(x.get('macs_rm_pct'),2)} | {K(x['active_img'])} | " +
              ' | '.join(str(x['dead_by_stage'][s]) for s in STAGES) + ' |')
        if len(g) > 1:
            w(f"| **{nm} {rho:g} 평균** | | **n={len(g)}** | | {F(np.mean([x['val_acc'] for x in g]))} | {K(np.mean([x['spikes'] for x in g]))} | **{np.mean([x['dead_ch'] for x in g]):.0f} ±{np.std([x['dead_ch'] for x in g], ddof=1):.0f}** | {F(np.mean([x['macs_rm_pct'] for x in g]),2)} | {K(np.mean([x['active_img'] for x in g]))} | " +
              ' | '.join(f"{np.mean([x['dead_by_stage'][s] for x in g]):.0f}" for s in STAGES) + ' |')
w('')
pr3 = [x for x in ns_ if x['key'] == 'prop' and abs(runs[x['tag']]['rho'] - 3e-3) < 1e-12]
ni3 = [x for x in ns_ if x['key'] == 'abl_noinv' and abs(runs[x['tag']]['rho'] - 2.4e-2) < 1e-12]
if pr3 and ni3:
    d_ = np.mean([x['dead_ch'] for x in ni3]) - np.mean([x['dead_ch'] for x in pr3])
    se_ = np.sqrt(np.var([x['dead_ch'] for x in ni3], ddof=1) / len(ni3) + np.var([x['dead_ch'] for x in pr3], ddof=1) / len(pr3))
    a_ = np.mean([x['active_img'] for x in ni3]) / np.mean([x['active_img'] for x in pr3]) - 1
    res['inversion']['intra_pair'] = {'dead_diff': float(d_), 'se': float(se_), 'active_ratio': float(a_), 'n': [len(pr3), len(ni3)]}
    w(f"- 스파이크가 거의 같은 짝 (ours_intra_ch 3e-3 n={len(pr3)} ≈ 238K vs 3요소 제거 2.4e-2 n={len(ni3)} ≈ 233K): 죽은 채널 {d_:+.0f} ±{se_:.0f}, 이미지당 활성 뉴런 {100*a_:+.1f}%. "
      "반전 (+within_channel+vmem) 을 빼면 채널을 덜 죽이고 더 많은 뉴런이 쏜다 — inter 의 분포 결과 (4a) 와 같은 방향이다.\n")

# ---------------- 5. BPSR
w('## 5. BPSR 대비 — 어디서 앞서고, 무엇이 필요한가\n')
w('### 5a. BPSR 각 묶음의 스파이크에서 inter 곡선과 비교\n')
w('inter 런 중 BPSR 묶음 스파이크의 ±25% 안 (ln 직선 국소 적합) 예측값과 BPSR 평균의 차. ± = √(예측 SE² + BPSR SD²/n).\n')
w('| BPSR λ | n | BPSR spikes | BPSR val acc ±SD | inter 런 수 (범위) | inter 예측 | Δacc (inter − BPSR) | p |')
w('|---|---|---|---|---|---|---|---|')
for q in v5['local']:
    if 'note' in q:
        w(f"| {q['bpsr_lambda']:g} | {q['n_bpsr']} | {K(q['bpsr_spk'])} | {F(q['bpsr_acc'])} | {q['n_inter']} | — | {q['note']} | — |"); continue
    w(f"| {q['bpsr_lambda']:g} | {q['n_bpsr']} | {K(q['bpsr_spk'])} | {F(q['bpsr_acc'])} ±{F(q['bpsr_sd'])} | {q['n_inter']} ({K(q['inter_lo'])}~{K(q['inter_hi'])}) | {F(q['inter_pred'])} ±{F(q['inter_pred_se'])} | {F(q['delta'],3,True)} ±{F(q['se'],3)} | {P(q['p'])} |")
w('')
w('### 5b. 대역별 ANCOVA `acc = a + b·ln(spikes) + c·[inter]`\n')
w('| 대역 | n (inter / BPSR) | c (inter − BPSR) ± SE | p | 기울기 b ± SE (p) | 기울기 차 p |')
w('|---|---|---|---|---|---|')
for q in v5['bands']:
    if 'note' in q:
        w(f"| {int(q['band'][0]/1e3)}~{int(q['band'][1]/1e3)}K | {q['n_inter']} / {q['n_bpsr']} | {q['note']} | | | |"); continue
    w(f"| {int(q['band'][0]/1e3)}~{int(q['band'][1]/1e3)}K | {q['n_inter']} / {q['n_bpsr']} | {F(q['c'],3,True)} ±{F(q['se'],3)} | {P(q['p'])} | {F(q['b'],2)} ±{F(q['b_se'],2)} ({P(q['b_p'])}) | {P(q.get('slope_diff_p'))} |")
w('')
w('- 경향: **300K 이상에서는 inter 가 +0.01~+0.05 앞서고, 260K 아래에서는 −0.06 뒤진다.** 어느 대역도 유의하지 않다 (가장 작은 p 는 BPSR 237K 묶음의 국소 비교 −0.06, p=' + P(v5['local'][2]['p']) + ', inter 쪽 불리).')
w('- 대역 경계는 미리 정하지 않았다. 여러 대역을 본 것 자체가 다중 비교다.\n')
w('### 5c. 검정력 — 300~420K 의 +0.042 를 유의하게 하려면\n')
w('| 항목 | 값 |')
w('|---|---|')
w(f"| 현재 효과 (점추정) | {F(pw['delta'],3,True)}%p |")
w(f"| 잔차 SD (ANCOVA, 300~420K) | {F(pw['sigma'],3)} (시드 SD 0.10~0.11 보다 작다 — 기울기가 일부 흡수) |")
w(f"| 현재 SE / 검정력 | ±{F(pw['se_now'],3)} (n={pw['n_now'][0]}/{pw['n_now'][1]}) / {100*pw['power_now']:.0f}% |")
w(f"| 80% 검정력 (양측 α=0.05) 에 필요한 SE | ±{F(pw['se_needed'],3)} → 지금 런 수의 약 {pw['scale_runs']:.1f}배 |")
w(f"| 팔당 필요 n (σ={F(pw['sigma'],3)}) | {pw['n_per_arm_80']:.0f} |")
w(f"| 팔당 필요 n (σ=0.10, 시드 SD) | {2*(stats.norm.ppf(0.975)+stats.norm.ppf(0.8))**2*0.01/pw['delta']**2:.0f} |")
w(f"| 한 비교점에서 팔당 n 일 때 보이는 최소 차 (80%, σ={F(pw['sigma'],3)}) | n=4: ±{F(pw['mde_n4'],2)}, n=8: ±{F(pw['mde_n8'],2)}, n=12: ±{F(pw['mde_n12'],2)}, n=24: ±{F(pw['mde_n24'],2)} |")
w('')
w('- 결론: +0.04%p 는 조건당 n=4 상한 (10-07 결정) 안에서는 **보일 수 없는 크기**다. 팔당 50~90런이 필요하다. "같은 스파이크에서 정확도가 더 높다" 를 주장으로 세우는 길은 사실상 닫혀 있다.\n')
w('### 5d. 정확도 밖 지표 — 같은 스파이크에서 이미 유의 (npz 재계산)\n')
w('구조 지표는 10-09 와 같은 정의 (죽은 채널 = 테스트 1만 장 동안 공간 전체에서 한 번도 안 쏜 채널; MACs = 그 채널을 지웠을 때 밀집 MACs 감소; 활성 뉴런 = 이미지당 한 번이라도 쏜 뉴런 수). 공변량은 원장 ln(spikes).\n')
w('| 대역 | 지표 | n (inter / BPSR) | c (inter − BPSR) ± SE | p | 기울기 차 p |')
w('|---|---|---|---|---|---|')
for band, d in v5['struct_ancova'].items():
    for k, nm in (('dead_ch', '죽은 채널 (/4,736)'), ('macs_rm_pct', '제거 시 MACs 감소 %p'), ('active_img', '이미지당 활성 뉴런'), ('val_acc', 'val acc (같은 npz 런 집합)')):
        q = d.get(k)
        if not q:
            continue
        cc = ratio_txt(q['c'], q['se']) if k == 'active_img' else f"{F(q['c'],2,True)} ±{F(q['se'],2)}"
        w(f"| {band} | {nm} | {q['n1']} / {q['n0']} | {cc} | {P(q['p'])} | {P(q.get('slope_diff_p'))} |")
w('')
w('개별값 (npz):\n')
w('| 팔 | ρ/λ | seed | 서버 | val acc | spikes | 죽은 채널 | MACs 감소 % | 이미지당 활성 뉴런 |')
w('|---|---|---|---|---|---|---|---|---|')
for key, nm in (('base', 'baseline'), ('ours_ch', 'ours_inter_ch'), ('bpsr', 'BPSR')):
    rows_ = [x for x in v5['struct_rows'] if x['key'] == key]
    knobs = sorted(set(runs[x['tag']]['knob'] for x in rows_), key=lambda v: knob_val(v) if v != '-' else 0)
    for kn in knobs:
        g = [x for x in rows_ if runs[x['tag']]['knob'] == kn]
        for x in g:
            rr = runs[x['tag']]
            w(f"| {nm} | {'-' if np.isnan(rr['rho']) else format(rr['rho'], 'g')} | s{rr['seed']} | {rr['server']} | {F(x['val_acc'])} | {K(x['spikes'])} | {x['dead_ch']} | {F(x['macs_rm_pct'],2)} | {K(x['active_img'])} |")
        if len(g) > 1:
            w(f"| **{nm} 평균** | | **n={len(g)}** | | {F(np.mean([x['val_acc'] for x in g]))} ±{F(np.std([x['val_acc'] for x in g], ddof=1))} | {K(np.mean([x['spikes'] for x in g]))} | **{np.mean([x['dead_ch'] for x in g]):.0f} ±{np.std([x['dead_ch'] for x in g], ddof=1):.0f}** | {F(np.mean([x['macs_rm_pct'] for x in g]),2)} ±{F(np.std([x['macs_rm_pct'] for x in g], ddof=1),2)} | {K(np.mean([x['active_img'] for x in g]))} |")
w('')
sa = v5['struct_ancova']['300-420K']
w(f"- 10-09 문서 (+49.8 / +3.44 / −5.7%) 와 개별 런 값은 일치하고, ANCOVA 오프셋이 +48.8 / +3.38 / −5.4% 로 조금 다르다 (공변량·런 집합 정의 차이로 보이나 확인하지 않았다). 결론은 같다.")
w('- **논문 주장안**: "같은 스파이크·같은 정확도에서 inter 는 채널 단위로 더 희소하다." 근거 = 300~420K 에서 Δacc ' + f"{F(mainb['c'],3,True)} ±{F(mainb['se'],3)} (p={P(mainb['p'])}, 구분 안 됨) + 죽은 채널 {F(sa['dead_ch']['c'],1,True)} (p={P(sa['dead_ch']['p'])}), MACs {F(sa['macs_rm_pct']['c'],2,True)}%p (p={P(sa['macs_rm_pct']['p'])}), 활성 뉴런 {ratio_txt(sa['active_img']['c'], sa['active_img']['se'])} (p={P(sa['active_img']['p'])}). "
  '지표 수가 셋이고 p 가 모두 3e-4 이하라 Bonferroni 를 적용해도 남는다. 차이는 스파이크를 더 줄일수록 커진다 (기울기 차 p≤0.003, 전 구간에서 죽은 채널 ' + f"{F(v5['struct_ancova']['180-430K']['dead_ch']['c'],0,True)}, MACs {F(v5['struct_ancova']['180-430K']['macs_rm_pct']['c'],1,True)}%p).")
w('- 이 주장의 한계: "하드웨어 이득" 은 밀집 MACs 계산이다. 실제 가속기에서 채널 제거로 시간·에너지가 주는지는 재지 않았다. SynOps (스파이크 기반 연산량) 는 10-09 에서 BPSR 과 실질 차이 없음 (0.4%p 이하) 이었다 — **이벤트 구동 하드웨어에서는 이득이 없을 수 있다.**\n')

# ---------------- 6. steps
w('## 6. 스텝별 정확도 — T 를 줄여 추론하면 (logits_t)\n')
w('T=4 로 학습한 망을 t 스텝까지만 돌린 출력의 정확도 (테스트 1만 장). 스파이크 몫 = 그 스텝까지 누적 스파이크 / 4스텝 전체.\n')
w('| 묶음 | n | T\'=1 | T\'=2 | T\'=3 | T\'=4 | T\'=2 − T\'=4 | T\'=3 − T\'=4 | 누적 스파이크 % (T\'=1/2/3) |')
w('|---|---|---|---|---|---|---|---|---|')
for nm in [x for x in v6 if x != 'compare']:
    q = v6[nm]
    def ms_(i, key='acc_t', sdk='acc_t_sd'):
        return F(q[key][i]) + (' ±' + F(q[sdk][i]) if np.isfinite(q[sdk][i]) else '')
    w(f"| {nm} | {q['n']} | {ms_(0)} | {ms_(1)} | {ms_(2)} | {ms_(3)} | {F(q['drop_vs_T4'][1],2,True)}" + (f" ±{F(q['drop_vs_T4_sd'][1],2)}" if np.isfinite(q['drop_vs_T4_sd'][1]) else '') +
      f" | {F(q['drop_vs_T4'][2],2,True)}" + (f" ±{F(q['drop_vs_T4_sd'][2],2)}" if np.isfinite(q['drop_vs_T4_sd'][2]) else '') + f" | {F(q['cum_spk_share'][0],1)} / {F(q['cum_spk_share'][1],1)} / {F(q['cum_spk_share'][2],1)} |")
w('')
w('개별 런:\n')
w('| 런 | 서버 | T\'=1 | T\'=2 | T\'=3 | T\'=4 | 저장된 예측 정확도 |')
w('|---|---|---|---|---|---|---|')
for t, q in sorted(npzt.items()):
    if 'error' in q or 'acc_t' not in q:
        continue
    w(f"| {t} | {runs[t]['server'] if t in runs else '—'} | " + ' | '.join(F(x) for x in q['acc_t']) + f" | {F(q['acc_final'])} |")
w('')
if 'drop_T2' in c6:
    q = c6['drop_T2']
    w(f"- T'=3 (스파이크 약 61%) 은 세 묶음 모두 손실 0.10%p 이하. T'=2 (약 26~28%) 에서 inter 는 {F(s_int['drop_vs_T4'][1],2,True)}, baseline {F(s_base['drop_vs_T4'][1],2,True)} — 차 {F(q['diff'],2,True)} ±{F(q['se'],2)} (p={P(q['p'])}, n=3/3). baseline 은 시드 하나 (s11) 가 −0.29 로 떨어지고 나머지는 −0.06·−0.05 다. **n=3 으로는 판단 불가.**")
w('- T\'=1 정확도 (43~67%) 는 시드마다 크게 다르고 방법과 무관해 보인다.')
w('- 이 결과는 "추론 T 를 줄여 스파이크를 더 아끼는" 별도 축의 가능성만 보여 준다. 학습을 T=4 로 하고 추론만 줄이는 것이라 규제 방법 비교와는 다른 질문이다.\n')

# ---------------- 7. 배제
w('## 7. 배제·미완주 (평균·회귀에서 뺌, 행은 남김)\n')
w('| 런 | 팔 | 서버 | 에폭 | val acc | spikes | 사유 |')
w('|---|---|---|---|---|---|---|')
for x in sorted(J['excluded'], key=lambda x: x['tag']):
    if x['arm'] in ('ours_inter_ch', '- vmem', '- final_step', '- loss-ratio', '- 1- inversion', 'BPSR', 'baseline') or 'intra' in x['arm']:
        w(f"| {x['tag']} | {x['arm']} | {x['server']} | {x.get('epochs') or '—'} | {F(x['val_acc']) if x['val_acc'] is not None else '—'} | {K(x['spikes']) if x['spikes'] else '—'} | {x['reason']} |")
npz_missing = [t for t in ('r19c10-bpsr-3e-11-s2', 'r19c10-bpsr-3e-11-s3') if t not in npzt or 'error' in npzt.get(t, {})]
if npz_missing:
    w('\nnpz_t 미포함 (추출 중): ' + ', '.join(f'`{t}`' for t in npz_missing) + '.')
w('\n로그 없음: baseline s11~13 (`train.log` 없음, `reg_detail` 은 s11·s12 만), ours_intra_ch 138 런 대부분 (`train.log` 없음 → λ 비교에서 빠짐, `reg_detail` 은 있음).\n')

# ---------------- LIMITATION
w('## [LIMITATION]\n')
w('- **어블레이션 팔은 거의 n=1** 이다 (`- vmem` 은 ρ 6개 각 n=1, 나머지 셋은 각 n=1). 착지 스파이크·분포 지표처럼 효과가 시드 산포의 수 배 이상인 것은 방향을 말할 수 있지만, 정확도 효과 (0.1%p 대) 는 말할 수 없다.')
w('- **뉴런 단위 "vmem 이 더 쏘게 하는 뉴런" 은 재지 못했다.** `ch_novmem`·`ch_nofinal`·`ch_noinv`·`ch_nolr` 와 intra `abl_novmem` 의 npz 가 없다. 1d 는 학습 모드 층 집계 대리 지표다. 재려면 `ch_novmem` 3e-3 s1 (138, 가중치 있음) 과 inter 3e-3 s2 를 같은 추출기로 돌려 baseline 발화율 구간별로 비교해야 한다 (각 약 17분, GPU 1장 — 실행은 승인 사항).')
w(f"- **final_step 단독의 시간 효과는 측정 안 됨.** 2b 는 inter (final_step 있음) vs baseline·BPSR (없음) 비교라 다른 요소와 섞여 있다. BPSR npz_t 는 n={n_bp_t}.")
w('- **"같은 ρ" 는 요소마다 같은 세기가 아니다.** final_step·inversion 을 끄면 R 의 정의 (가중·합산 위치) 가 바뀐다. vmem 은 R 정의가 같아 같은 ρ 비교가 가장 공정하다. λ 는 loss-ratio 가 정하는 내생 변수라 "같은 λ" 비교는 인과가 아니라 대응 관계다.')
w('- 학습 모드 지표 (`reg_detail`, `reg_neuron_detail`) 는 마지막 에폭의 학습 배치 (증강 포함) 평균이고, 원장 스파이크 (평가, 베스트 에폭) 와 다르다. 지니·상위 10% 는 이미지 안 층별 값의 평균이라 npz 의 데이터셋 전체 지니와 정의가 다르다.')
w('- 1d·4a 는 지표를 15~20개 검정했다. 다중 비교 보정은 하지 않았고, 결론은 z 가 매우 큰 것 (4a) 또는 방향만 (1d) 으로 제한했다.')
w('- vmem 의 정확도 사후 합산 (p=0.04) 은 두 경쟁 범위를 결과를 본 뒤 합친 값이다. 논문 근거로 쓰지 않는다.')
w('- BPSR 비교의 대역·국소 창 (±25%) 은 미리 정한 것이 아니다. 검정력 계산은 정규 근사이고 잔차 SD 를 고정값으로 썼다.')
w(f"- 스텝별 정확도는 inter·baseline 각 n=3, BPSR n={n_bp_t} 이고, 추론 T 를 줄이는 것은 학습 규제와 다른 질문이다.")
w('- R19·CIFAR-10·T=4 한 설정이다. C100·다른 모델의 어블레이션은 없다.')

open(OUT_MD, 'w').write('\n'.join(o) + '\n')
json.dump(res, open(OUT_JSON, 'w'), indent=1, default=float)
print('md', OUT_MD)
