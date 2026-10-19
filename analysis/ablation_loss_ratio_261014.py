#!/usr/bin/env python
"""R19 CIFAR-10 (ResNet-19, T=4): ablation '- loss-ratio' — 26-10-14.

사용자 요청 (10-14): "어블레이션 각 방법 분석도 해줄 수 있어?" → 이 파일은 loss-ratio 몫.
팔: ch_nolr (ours_inter_ch 에서 loss-ratio 만 끄고 고정 λ=1.25e-7). 교차 확인 (섞지 않음):
  abl_nolr  = ours_intra_ch (run 키 prop) 에서 loss-ratio 만 끈 팔 (고정 λ 2e-7/3e-7/5e-7)
  l2_lr     = plain L2 에 loss-ratio 를 붙인 팔 vs l2 (plain L2, 고정 λ). 두 팔 설정 차이는 λ 제어뿐.
  lw1_nolr  = ours_layer_w1 에서 끈 팔 — 2에폭 런 하나뿐이라 쓸 수 없음 (행만 남김).
  BPSR      = 고정 λ 방법 (착지 시드 변동 비교용 참고).

수치 정의 (원장·nofinal_vs_inter_261014 와 같음)
  val acc  = 310 에폭 중 최고 val_acc (같은 값이면 첫 에폭), spikes = 그 에폭 s_count (원값, 이미지당)
  train loss (규제 포함 'loss') / train acc / val loss = 그 에폭 값
  λ_used[e] = 에폭 e 동안 쓰인 λ. loss-ratio 런은 에폭 e-1 끝에 찍힌 adp_lambda (e=0 은 0),
              고정 λ 런은 상수.
  R[e] = reg_detail.csv 의 sc_loss 18층 합 (배치 평균, λ 곱하기 전 원값) = 로그의 reg_R
  L[e] = 과제 손실 = loss − λ_used·R  (loss-ratio 런은 로그의 task_loss_clean 과 같음)
  실효 ρ[e] = λ_used[e]·R[e] / L[e]   (loss-ratio 런은 정의상 ≈ρ, 고정 λ 런은 움직인다)
  혼합 증강 해제 = mix_off_iter 500×200 → 0-기준 에폭 200 부터 (config_snn_training.py:153)

값은 학습 로그에서 직접 다시 읽는다. 원장 inventory.json 은 교차검증·S30/S1 배제 판정에만 쓴다.
  138 = 로컬 _paper/<tag>/train.log, /media/hdd1/kyccj/EIP/paper/<tag>/reg_*.csv
  23  = sejong ~/PycharmProjects/TensorFlow-SNNs/_paper/<tag>/config_sweep.py, /srv2/kyccj/EIP/paper/<tag>/
        (ssh 로 수집기를 stdin 으로 돌려 읽기만 한다 — 원격에 파일을 남기지 않음)
통계 도구 (welch, ols, fieller, mde, n_needed, parse_conf) 는 nofinal_vs_inter_261014.py 에서 가져와
같은 스파이크 ANCOVA 의 숫자가 다른 팔과 비교 가능하게 한다.

쓰는 법:
  ~/anaconda3/envs/venv_1/bin/python analysis/ablation_loss_ratio_261014.py [--refetch]
산출: analysis/ablation_loss_ratio_261014.md (그리고 scratchpad 에 수치 원본 json)
"""
import argparse, gzip, importlib.util, json, math, os, re, subprocess, sys, time
import numpy as np
from scipy import stats

ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
SCR = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
CACHE = SCR + '/lr1014'
COLLECT = CACHE + '/collect.py'
INV = SCR + '/led1007b/inventory.json'
OUT_MD = ROOT + '/analysis/ablation_loss_ratio_261014.md'
OUT_JSON = CACHE + '/ablation_loss_ratio_261014.json'
EPOCHS = 310
MIX_OFF = 200
PATS = ['r19c10-base---s1[1-6]', 'r19c10-ours_ch-[0-9]*', 'r19c10-ch_nolr-*', 'r19c10-prop-[0-9]*',
        'r19c10-abl_nolr-*', 'r19c10-l2-[0-9]*', 'r19c10-l2_lr-*', 'r19c10-bpsr-[0-9]*', 'r19c10-lw1_nolr-*',
        # C100: ρ 눈금이 데이터셋 사이에 옮겨지는지 (2e) 만 본다
        'r19c100-base---s*', 'r19c100-ours_ch-[0-9]*', 'r19c100-prop-[0-9]*', 'r19c100-l2-[0-9]*', 'r19c100-bpsr-[0-9]*']
NAME = re.compile(r'^r19c(10|100)-(base|ours_ch|ch_nolr|prop|abl_nolr|l2_lr|l2|bpsr|lw1_nolr)-(.*)-s(\d+)$')
LABEL = {'base': 'baseline', 'ours_ch': 'ours_inter_ch', 'ch_nolr': '- loss-ratio',
         'prop': 'ours_intra_ch', 'abl_nolr': 'ours_intra_ch - loss-ratio', 'l2': 'plain L2',
         'l2_lr': 'plain L2 + loss-ratio', 'bpsr': 'BPSR', 'lw1_nolr': 'ours_layer_w1 - loss-ratio'}
FIXED_KEYS = ('ch_nolr', 'abl_nolr', 'l2', 'bpsr', 'lw1_nolr')
NEU = {}
for _l in ['conv1_conv_n'] + [f'conv2_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')]:
    NEU[_l] = 32 * 32 * 128
for _l in [f'conv3_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')]:
    NEU[_l] = 16 * 16 * 256
for _l in [f'conv4_block{b}_{s}_n' for b in (1, 2) for s in ('conv1', 'out')]:
    NEU[_l] = 8 * 8 * 512
NEU['fc1_n'] = 256
STAGES = ['conv1', 'conv2', 'conv3', 'conv4']
BAND = (200e3, 420e3)        # ours_inter_ch 곡선 범위 (nofinal 과 같음)
BAND_INTRA = (150e3, 300e3)  # ours_intra_ch 곡선 범위 (abl_nolr 3점 포함)
BAND_L2 = (200e3, 440e3)     # plain L2 곡선 범위 (l2_lr 2e-3 431K 포함)
RHO_LOCAL = (1.5e-3, 5e-3)   # ours ln S–ln ρ 국소 직선 범위
NBOOT = 20000
PRE = (1, MIX_OFF)           # 혼합 증강 구간에서 λ 를 평균할 에폭 [1, 200)
POST = (MIX_OFF, EPOCHS)     # 해제 뒤 [200, 310)

_spec = importlib.util.spec_from_file_location('nofinal', ROOT + '/analysis/nofinal_vs_inter_261014.py')
NF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(NF)
welch, ols, fieller, mde, n_needed, tcrit, parse_conf = (NF.welch, NF.ols, NF.fieller, NF.mde, NF.n_needed,
                                                         NF.tcrit, NF.parse_conf)


# ============================================================================ 읽기
def fetch(refetch):
    os.makedirs(CACHE, exist_ok=True)
    c23, c138 = CACHE + '/c23.json.gz', CACHE + '/c138.json.gz'
    if refetch or not os.path.exists(c23):
        pat = ' '.join("'" + p + "'" for p in PATS)
        cmd = ['ssh', '-p', '23456', '-o', 'BatchMode=yes', 'kyccj@161.122.23.23',
               f'python3 - /home/kyccj/PycharmProjects/TensorFlow-SNNs/_paper /srv2/kyccj/EIP/paper {pat}']
        out = subprocess.run(cmd, stdin=open(COLLECT), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             check=True).stdout
        with gzip.open(c23, 'wb') as f:
            f.write(out)
    if refetch or not os.path.exists(c138):
        out = subprocess.run([sys.executable, '-I', COLLECT, ROOT + '/_paper', '/media/hdd1/kyccj/EIP/paper'] + PATS,
                             stdout=subprocess.PIPE, check=True).stdout
        with gzip.open(c138, 'wb') as f:
            f.write(out)
    return json.load(gzip.open(c138)), json.load(gzip.open(c23))


def knob_val(k):
    return float(k.replace('p', '.')) if k not in ('-', '') else float('nan')


def build(refetch):
    c138, c23 = fetch(refetch)
    inv = {r['tag']: r for r in json.load(open(INV))['runs']}
    raw, dup = {}, []
    for tag, r in c138.items():
        if r['epochs']:
            raw[tag] = (r, '138')
    for tag, r in c23.items():
        if r['epochs']:
            if tag in raw:
                dup.append(tag)
            else:
                raw[tag] = (r, '23')
    # 원장에만 있는 hold/stale 행도 표에 남긴다
    for tag, iv in inv.items():
        if NAME.match(tag) and tag not in raw and iv.get('ds') == 'C10' and NAME.match(tag).group(2) in (
                'ch_nolr', 'abl_nolr', 'l2_lr', 'lw1_nolr'):
            raw[tag] = ({'epochs': [], 'conf': '', 'rd': None, 'rnd': None}, iv.get('server') or '-')
    runs = []
    for tag, (r, server) in sorted(raw.items()):
        m = NAME.match(tag)
        if not m:
            continue
        ds, key, knob, seed = 'C' + m.group(1), m.group(2), m.group(3), int(m.group(4))
        conf_txt = r.get('conf') or (c23.get(tag, {}).get('conf') if server == '138' else c138.get(tag, {}).get('conf')) or ''
        conf = parse_conf(conf_txt) if conf_txt else {}
        rows = r['epochs']
        rec = {'tag': tag, 'ds': ds, 'key': key, 'arm': LABEL[key], 'knob': '-' if key == 'base' else knob,
               'kv': knob_val(knob) if key != 'base' else float('nan'), 'seed': seed, 'server': server,
               'epochs': len(rows), 'conf': conf, 'rd': r.get('rd'), 'rnd': r.get('rnd'), 'rows': rows}
        if rows:
            b = max(rows, key=lambda x: x['val_acc'])
            rec.update(best_epoch=rows.index(b) + 1, train_loss=b['loss'], train_acc=b['acc'] * 100,
                       val_loss=b['val_loss'], val_acc=b['val_acc'] * 100, spikes=b['s_count'])
        iv = inv.get(tag, {})
        reasons = []
        if iv.get('status') in ('hold', 'todo') and not rows:
            reasons.append(f'{iv.get("status")} (미시작)')
        elif len(rows) < EPOCHS:
            reasons.append(f'연기시험({len(rows)}ep)' if seed >= 900 else
                           (f'미완주({len(rows)}ep, 원장 없음)' if not iv else f'미완주({len(rows)}ep)'))
        elif len(rows) > EPOCHS:
            reasons.append(f'에폭 줄 과다({len(rows)})')
        if 'S30/S1' in (iv.get('exclude_reason') or ''):
            reasons.append([x for x in iv['exclude_reason'].split('; ') if 'S30' in x][0])
        # 팔 확인: λ 제어 플래그와 손잡이
        if conf:
            lr_on = conf.get('reg_spike_loss_ratio') is True
            want_lr = key in ('ours_ch', 'prop', 'l2_lr')
            if key != 'base' and lr_on != want_lr:
                reasons.append(f'λ 제어 불일치(conf loss_ratio={lr_on})')
            kn = conf.get('reg_spike_loss_ratio_target') if lr_on else conf.get('reg_spike_out_const')
            if key != 'base' and (kn is None or abs(float(kn) - rec['kv']) > 1e-15):
                reasons.append(f'손잡이 불일치(dir={knob}, conf={kn})')
            rms = str(conf.get('root_model_save', ''))
            srv_rms = '138' if '/media/hdd1' in rms else ('23' if '/srv2' in rms else None)
            if srv_rms and rows and srv_rms != server:
                reasons.append(f'서버 불일치(읽은 곳 {server}, 저장 경로 {srv_rms})')
        rec['exclude_reason'] = '; '.join(reasons)
        rec['use'] = bool(rows) and not reasons
        if iv and rows:
            rec['inv_check'] = {'val_acc_diff': None if iv.get('val_acc') is None else round(rec['val_acc'] - iv['val_acc'], 4),
                                'spikes_diff': None if iv.get('spikes') is None else round(rec['spikes'] - iv['spikes'], 1),
                                'server_match': iv.get('server') == server}
        runs.append(rec)
    for rec in runs:
        rec.update(traj_metrics(rec))
        rec.update(struct_metrics(rec))
    return runs, dup, inv


# ============================================================================ 런별 궤적·구조 지표
def traj_metrics(r):
    rows = r['rows']
    if len(rows) != EPOCHS:
        return {}
    o = {}
    s = np.array([x['s_count'] for x in rows])
    loss = np.array([x['loss'] for x in rows])
    o['s_traj'] = s
    o['sec'] = np.array([x.get('_sec') or np.nan for x in rows], float)
    o['sec_med'] = float(np.nanmedian(o['sec'][1:]))
    # 혼합 증강 해제 반응: 해제 직전 10에폭 vs 마지막 10에폭 평균 (평가 스파이크)
    o['s_pre10'] = float(s[MIX_OFF - 10:MIX_OFF].mean())
    o['s_last10'] = float(s[-10:].mean())
    o['d_mix'] = math.log(o['s_last10'] / o['s_pre10'])
    if r['key'] == 'base':
        return o
    if 'adp_lambda' in rows[-1]:
        logged = np.array([x['adp_lambda'] for x in rows])
        lam = np.r_[0.0, logged[:-1]]
        o['lam_used'] = lam
        o['R'] = np.array([x['reg_R'] for x in rows])
        o['L'] = np.array([x['task_loss_clean'] for x in rows])
        o['rho_eff'] = np.array([x['loss_ratio'] for x in rows])
    else:
        lam = np.full(EPOCHS, r['kv'])
        o['lam_used'] = lam
        rd = r.get('rd')
        if rd and 'sc_loss' in rd['sums'] and len(rd['sums']['sc_loss']) == EPOCHS and max(rd['sums']['sc_loss']) > 0:
            R = np.array(rd['sums']['sc_loss'])
            L = loss - lam * R
            o['R'], o['L'], o['rho_eff'] = R, L, lam * R / L
    o['lam_mean'] = float(lam[1:].mean())
    o['lam_pre'] = float(lam[PRE[0]:PRE[1]].mean())
    o['lam_post'] = float(lam[POST[0]:POST[1]].mean())
    o['lam_last'] = float(lam[-10:].mean())
    if 'rho_eff' in o:
        re_ = o['rho_eff']
        o['rho_mean'] = float(re_[1:].mean())
        o['rho_pre'] = float(re_[PRE[0]:PRE[1]].mean())
        o['rho_post'] = float(re_[POST[0]:POST[1]].mean())
        o['rho_last'] = float(re_[-10:].mean())
        o['R_end'] = float(o['R'][-1])
        o['L_pre10'] = float(o['L'][MIX_OFF - 10:MIX_OFF].mean())
        o['L_post10'] = float(o['L'][MIX_OFF + 1:MIX_OFF + 11].mean())
        o['R_pre10'] = float(o['R'][MIX_OFF - 10:MIX_OFF].mean())
        o['R_post10'] = float(o['R'][MIX_OFF + 1:MIX_OFF + 11].mean())
    return o


def struct_metrics(r):
    """reg_detail / reg_neuron_detail 마지막 에폭 (학습 모드, 배치 평균) 층 집계."""
    rd, rn = r.get('rd'), r.get('rnd')
    if not rd or r['epochs'] != EPOCHS:
        return {}
    last = str(rd['last'])
    if rd['last'] != EPOCHS - 1:
        return {}
    L = rd['layers'][last]
    spk = {l: L[l]['spike_count'] for l in NEU if l in L}
    tot = sum(spk.values())
    o = {'tr_spk': tot}
    for s in STAGES + ['fc1']:
        o[f'share_{s}'] = 100 * sum(v for l, v in spk.items() if l.startswith(s)) / tot
    if 'sc_rate_mean' in rd['cols'] and any(L[l]['sc_rate_mean'] > 0 for l in spk):
        w = {l: L[l]['sc_rate_mean'] for l in spk}
        nN = sum(NEU[l] for l in w)
        o['w_mean'] = sum(w[l] * NEU[l] for l in w) / nN
        for s in STAGES:
            ls = [l for l in w if l.startswith(s)]
            o[f'w_{s}'] = sum(w[l] * NEU[l] for l in ls) / sum(NEU[l] for l in ls)
        o['lnR_rd'] = math.log(sum(L[l]['sc_loss'] for l in spk)) if sum(L[l]['sc_loss'] for l in spk) > 0 else None
    if rn and str(rn['last']) == last:
        N = rn['layers'][last]
        d = {l: N[l] for l in NEU if l in N}
        nN = sum(NEU[l] for l in d)
        o['dead_pct'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in d) / nN
        o['gini_sw'] = sum(d[l]['gini'] * spk[l] for l in d) / tot
        o['top10_sw'] = 100 * sum(d[l]['top10_share'] * spk[l] for l in d) / tot
        for s in STAGES:
            ls = [l for l in d if l.startswith(s)]
            o[f'dead_{s}'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in ls) / sum(NEU[l] for l in ls)
    return o


# ============================================================================ 통계
def A(rs, k):
    return np.array([r[k] for r in rs], float)


def ancova(ref, arm, center, srv=True, quad=False, y='val_acc', ylog=False):
    data = ref + arm
    x = np.log(A(data, 'spikes')) - math.log(center)
    yy = A(data, y)
    if ylog:
        yy = np.log(yy)
    g = np.array([0.0] * len(ref) + [1.0] * len(arm))
    cols, names = [np.ones_like(x), x], ['a', 'b']
    if quad:
        cols.append(x ** 2); names.append('c')
    cols.append(g); names.append('d')
    sv = np.array([1.0 if r['server'] == '23' else 0.0 for r in data])
    if srv and 0 < sv.sum() < len(sv):
        cols.append(sv); names.append('e')
    f = ols(np.column_stack(cols), yy, names)
    f['n_ref'], f['n_arm'] = len(ref), len(arm)
    return f


def same_acc_ratio(f):
    nm = f['names']
    i_b, i_d = nm.index('b'), nm.index('d')
    b, d = f['beta'][i_b], f['beta'][i_d]
    vb, vd, cdb = f['cov'][i_b, i_b], f['cov'][i_d, i_d], f['cov'][i_b, i_d]
    th = d / b
    lo, hi, kind = fieller(d, b, vd, vb, cdb, f['df'])
    return {'theta': th, 'ratio_pct': (math.exp(-th) - 1) * 100, 'fieller': kind,
            'lo_pct': None if lo is None else (math.exp(-hi) - 1) * 100,
            'hi_pct': None if lo is None else (math.exp(-lo) - 1) * 100}


def one_vs_group(x, grp, sd=None, df=None):
    """단일 런 x 와 묶음 평균 비교. sd 를 안 주면 묶음 자체 SD (자유도 n-1)."""
    g = np.asarray(grp, float)
    if sd is None:
        sd, df = g.std(ddof=1), len(g) - 1
    se = sd * math.sqrt(1 + 1 / len(g))
    dlt = x - g.mean()
    t = dlt / se
    return {'diff': dlt, 'se': se, 'df': df, 'lo': dlt - tcrit(df) * se, 'hi': dlt + tcrit(df) * se,
            'p': 2 * stats.t.sf(abs(t), df), 'n_grp': len(g), 'grp_mean': g.mean()}


def line_fit(x, y):
    X = np.c_[np.ones(len(x)), x]
    f = ols(X, y, ['a', 'b'])
    return f


def line_pred(f, x0):
    v = np.array([1.0, x0])
    m = float(v @ f['beta'])
    sm = float(np.sqrt(v @ f['cov'] @ v))
    return m, sm, math.sqrt(sm ** 2 + f['resid_sd'] ** 2)


def pooled_resid(rs, val, group='knob', server=True):
    """같은 손잡이 묶음 고정효과 (+서버) 잔차 SD. n≥2 묶음만."""
    by = {}
    for r in rs:
        by.setdefault(r[group], []).append(r)
    by = {k: v for k, v in by.items() if len(v) >= 2}
    sel = [r for v in by.values() for r in v]
    if not sel:
        return None
    keys = sorted(by)
    cols = [[1.0 if r[group] == k else 0.0 for r in sel] for k in keys]
    sv = [1.0 if r['server'] == '23' else 0.0 for r in sel]
    names = ['g_' + k for k in keys]
    mixed = [k for k in keys if len({r['server'] for r in by[k]}) > 1]
    if server and mixed:
        cols.append(sv); names.append('srv23')
    X = np.array(cols).T
    y = np.array([val(r) for r in sel])
    f = ols(X, y, names)
    # 서버 항 없는 단순 합동 SD
    num = sum(((np.array([val(r) for r in v]) - np.mean([val(r) for r in v])) ** 2).sum() for v in by.values())
    den = sum(len(v) - 1 for v in by.values())
    return {'sd': f['resid_sd'], 'df': f['df'], 'sd_plain': math.sqrt(num / den), 'df_plain': den,
            'groups': {k: len(v) for k, v in by.items()}, 'srv': f['coef'].get('srv23'), 'n': len(sel),
            'servers': {'138': sum(r['server'] == '138' for r in sel), '23': sum(r['server'] == '23' for r in sel)}}


def ftest(a, b):
    F = (a['sd'] / b['sd']) ** 2
    p = 2 * min(stats.f.sf(F, a['df'], b['df']), stats.f.cdf(F, a['df'], b['df']))
    return {'F': F, 'p': p, 'df1': a['df'], 'df2': b['df']}


# ============================================================================ 분석
def analyze(runs):
    res = {}
    U = [r for r in runs if r['use'] and r['ds'] == 'C10']
    by = lambda k: sorted([r for r in U if r['key'] == k], key=lambda r: (r['kv'] if r['key'] != 'base' else 0, r['seed']))
    base, ours, nolr = by('base'), by('ours_ch'), by('ch_nolr')
    prop, anolr, l2, l2lr, bpsr = by('prop'), by('abl_nolr'), by('l2'), by('l2_lr'), by('bpsr')
    res['n'] = {k: len(v) for k, v in (('base', base), ('ours_ch', ours), ('ch_nolr', nolr), ('prop', prop),
                                       ('abl_nolr', anolr), ('l2', l2), ('l2_lr', l2lr), ('bpsr', bpsr))}
    # 짝 팔의 override 차이 (λ 제어 키·저장 경로·시드 제외). 옛 런에 없는 키는 flags.py 기본값 (둘 다 False) 로 본다.
    DROP = {'root_model_save', 'run_seed', 'reg_spike_loss_ratio', 'reg_spike_loss_ratio_target',
            'reg_spike_loss_ratio_start_ep', 'reg_spike_out_const'}
    DEF = {'reg_spike_maxnorm_no_inv': False, 'reg_spike_wta_rev_w1': False}
    def sig(r):
        d = {k: v for k, v in r['conf'].items() if k not in DROP}
        for k, v in DEF.items():
            d.setdefault(k, v)
        return d
    cdiff = {}
    for a_, b_ in (('ours_ch', 'ch_nolr'), ('prop', 'abl_nolr'), ('l2', 'l2_lr')):
        ra, rb = by(a_), by(b_)
        keys = set().union(*[set(sig(r)) for r in ra + rb])
        cdiff[f'{a_}|{b_}'] = {k: [sorted({repr(sig(r).get(k)) for r in ra}), sorted({repr(sig(r).get(k)) for r in rb})]
                               for k in sorted(keys)
                               if {repr(sig(r).get(k)) for r in ra} != {repr(sig(r).get(k)) for r in rb}}
    res['conf_diff'] = cdiff
    q = nolr[0]
    s0, a0 = q['spikes'], q['val_acc']
    ob = [r for r in ours if BAND[0] <= r['spikes'] <= BAND[1]]
    res['ours_band_n'] = len(ob)
    res['ours_band_rhos'] = sorted({r['knob'] for r in ob}, key=knob_val)

    # ---------------------------------------------------------------- 1. 같은 스파이크 정확도
    sec1 = {}
    for nm, kw in (('lin', dict(srv=False)), ('lin_srv', dict(srv=True)), ('quad', dict(srv=False, quad=True)),
                   ('quad_srv', dict(srv=True, quad=True))):
        f = ancova(ob, nolr, s0, **kw)
        sec1[nm] = f
    sec1['same_acc'] = same_acc_ratio(sec1['lin_srv'])
    # 가설 확인: nofinal 과 같은 ours 곡선 (n=27) 인가
    # 국소: ours 3e-3 (같은 착지) 와 ±12% 창
    g3 = [r for r in ours if r['knob'] == '3e-3']
    pool = pooled_resid(ours, lambda r: r['val_acc'], server=False)
    sec1['pooled_acc_sd'] = pool
    sec1['vs_3e3'] = one_vs_group(a0, A(g3, 'val_acc'), sd=pool['sd_plain'], df=pool['df_plain'])
    sec1['vs_3e3_spk'] = math.log(s0 / A(g3, 'spikes').mean())
    win = [r for r in ours if s0 * 0.88 <= r['spikes'] <= s0 * 1.12]
    sec1['win12'] = one_vs_group(a0, A(win, 'val_acc'), sd=pool['sd_plain'], df=pool['df_plain'])
    sec1['win12_tags'] = [r['tag'] for r in win]
    sec1['win12_spk_mean'] = A(win, 'spikes').mean()
    # baseline 대비 (단일 런 vs n=6): baseline 자체 SD (자유도 5) 와 합동 SD 두 가지
    bacc, bspk = A(base, 'val_acc'), A(base, 'spikes')
    sec1['vs_base'] = {'base_own_sd': one_vs_group(a0, bacc),
                       'pooled_sd': one_vs_group(a0, bacc, sd=pool['sd_plain'], df=pool['df_plain']),
                       'spk_pct': (s0 / bspk.mean() - 1) * 100,
                       'lnspk': one_vs_group(math.log(s0), np.log(bspk)),
                       'base_mean': bacc.mean(), 'base_sd': bacc.std(ddof=1), 'base_spk': bspk.mean(), 'n_base': len(base)}
    sec1['vs_base']['spk_lo'] = (math.exp(sec1['vs_base']['lnspk']['lo'] + math.log(bspk.mean()) - np.log(bspk).mean()) - 1) * 100
    sec1['vs_base']['spk_hi'] = (math.exp(sec1['vs_base']['lnspk']['hi'] + math.log(bspk.mean()) - np.log(bspk).mean()) - 1) * 100
    # 검정력: ours 곡선을 지금대로 두고 nolr 반복만 늘릴 때 (nofinal 4절과 같은 식)
    f = sec1['lin_srv']
    se_d = f['coef']['d']['se']
    var_curve = max(se_d ** 2 - f['resid_sd'] ** 2 / len(nolr), 0.0)
    k = stats.t.ppf(0.975, f['df']) + stats.t.ppf(0.8, f['df'])
    sec1['mde'] = {'curve_se': math.sqrt(var_curve),
                   'by_n': {m: k * math.sqrt(pool['sd_plain'] ** 2 / m + var_curve) for m in (1, 3, 4, 6)},
                   'floor': k * math.sqrt(var_curve)}
    res['sec1'] = sec1

    # 교차 확인 1: intra (prop vs abl_nolr), 교차 확인 2: plain L2 (l2 vs l2_lr)
    cross = {}
    pb = [r for r in prop if BAND_INTRA[0] <= r['spikes'] <= BAND_INTRA[1]]
    ab = [r for r in anolr if BAND_INTRA[0] <= r['spikes'] <= BAND_INTRA[1]]
    cen = math.exp(np.log(A(ab, 'spikes')).mean())
    cross['intra'] = {'lin': ancova(pb, ab, cen, srv=False), 'lin_srv': ancova(pb, ab, cen, srv=True),
                      'n_ref': len(pb), 'n_arm': len(ab), 'ref_range': (A(pb, 'spikes').min(), A(pb, 'spikes').max()),
                      'arm_spikes': A(ab, 'spikes').tolist()}
    fi = line_fit(np.log(A(pb, 'spikes')), A(pb, 'val_acc'))
    cross['intra']['each'] = []
    for r in ab:
        m, sm, sp = line_pred(fi, math.log(r['spikes']))
        cross['intra']['each'].append({'tag': r['tag'], 'pred': m, 'delta': r['val_acc'] - m, 'se': sp,
                                       'z': (r['val_acc'] - m) / sp})
    lb = [r for r in l2 if BAND_L2[0] <= r['spikes'] <= BAND_L2[1]]
    lr_ = [r for r in l2lr if BAND_L2[0] <= r['spikes'] <= BAND_L2[1]]
    cen = math.exp(np.log(A(lr_, 'spikes')).mean())
    cross['l2'] = {'lin': ancova(lb, lr_, cen, srv=False), 'lin_srv': ancova(lb, lr_, cen, srv=True),
                   'n_ref': len(lb), 'n_arm': len(lr_), 'ref_range': (A(lb, 'spikes').min(), A(lb, 'spikes').max()),
                   'arm_range': (A(lr_, 'spikes').min(), A(lr_, 'spikes').max())}
    # 겹치는 구간만 (l2_lr 이 l2 범위 밖이면 외삽이므로)
    hi_l2 = A(lb, 'spikes').max()
    lr_in = [r for r in lr_ if r['spikes'] <= hi_l2 * 1.0]
    if len(lr_in) >= 2:
        cen2 = math.exp(np.log(A(lr_in, 'spikes')).mean())
        cross['l2']['lin_overlap'] = ancova(lb, lr_in, cen2, srv=False)
        cross['l2']['n_arm_overlap'] = len(lr_in)
    res['cross'] = cross

    # ---------------------------------------------------------------- 2. 같은 ρ / 같은 λ: 눈금
    sec2 = {}
    # ours 의 ln S–ln ρ 는 ρ 1.5e-3~1.2e-2 전체로는 휘어 있다 (2차 잔차 SD 0.021 vs 직선 0.064).
    # 고정 λ 런 (342K) 근처만 쓰는 국소 직선: ρ 1.5e-3~5e-3 (10-10 의 n=17 과 같은 범위).
    oloc = [r for r in ours if RHO_LOCAL[0] <= r['kv'] <= RHO_LOCAL[1]]
    xr = np.log(A(oloc, 'kv'))
    yr = np.log(A(oloc, 'spikes'))
    fr = line_fit(xr, yr)
    b_, a_ = fr['beta'][1], fr['beta'][0]
    rho_star = math.exp((math.log(s0) - a_) / b_)
    va, vb, cab = fr['cov'][0, 0], fr['cov'][1, 1], fr['cov'][0, 1]
    lo, hi, kind = fieller(math.log(s0) - a_, b_, va + fr['resid_sd'] ** 2, vb, -cab, fr['df'])
    fq = ols(np.c_[np.ones(len(ours)), np.log(A(ours, 'kv')), np.log(A(ours, 'kv')) ** 2], np.log(A(ours, 'spikes')), ['a', 'b', 'c'])
    fl = line_fit(np.log(A(ours, 'kv')), np.log(A(ours, 'spikes')))
    sec2['rho_fit'] = {'n': len(oloc), 'slope': b_, 'slope_se': fr['coef']['b']['se'], 'resid_sd': fr['resid_sd'],
                       'rho_range': RHO_LOCAL, 'rho_star': rho_star,
                       'rho_star_lo': math.exp(min(lo, hi)) if lo is not None else None,
                       'rho_star_hi': math.exp(max(lo, hi)) if lo is not None else None,
                       'full_n': len(ours), 'full_lin_sd': fl['resid_sd'], 'full_quad_sd': fq['resid_sd'],
                       'full_quad_c_p': fq['coef']['c']['p']}
    preds = {}
    for k_ in ('rho_mean', 'rho_pre', 'rho_post', 'rho_last'):
        m, sm, sp = line_pred(fr, math.log(q[k_]))
        preds[k_] = {'value': q[k_], 'pred_spk': math.exp(m), 'resid': math.log(s0) - m, 'z': (math.log(s0) - m) / sp}
    sec2['nolr_rho_preds'] = preds
    lam_fits = {}
    for k_ in ('lam_mean', 'lam_pre', 'lam_post', 'lam_last'):
        f_ = line_fit(np.log(A(oloc, k_)), yr)
        m, sm, sp = line_pred(f_, math.log(q[k_]))
        lam_fits[k_] = {'slope': f_['beta'][1], 'slope_se': f_['coef']['b']['se'], 'resid_sd': f_['resid_sd'],
                        'pred_spk': math.exp(m), 'resid': math.log(s0) - m, 'z': (math.log(s0) - m) / sp,
                        'ours3e3_mean': A(g3, k_).mean(), 'nolr_vs_3e3': q[k_] / A(g3, k_).mean() - 1, 'n': len(oloc)}
    sec2['lam_fits'] = lam_fits

    # 2b. 착지를 정하는 구간의 무게 α: ln λ_혼합(α) = (1−α)·ln λ_해제전 + α·ln λ_해제뒤.
    #     α = 해제 뒤 구간 (에폭 200~309, 시간의 110/309 = 0.356) 이 착지에 미치는 몫.
    #     설계 A (plain L2): 고정 λ 의 용량-반응선에 loss-ratio 런의 λ_혼합(α) 를 넣어 착지 오차 평균 0 인 α.
    #     설계 B (inter·intra): loss-ratio 런의 ln S ~ ln λ_혼합(α) 선에 고정 λ 런을 넣어 오차 0 인 α.
    def lmix(r, a):
        return (1 - a) * math.log(r['lam_pre']) + a * math.log(r['lam_post'])
    grid = np.linspace(-0.5, 1.5, 4001)

    def design_b(ref, arm):
        def resid(a):
            f_ = line_fit(np.array([lmix(r, a) for r in ref]), np.log(A(ref, 'spikes')))
            return np.array([math.log(r['spikes']) - (f_['beta'][0] + f_['beta'][1] * math.log(r['kv'])) for r in arm])
        out = {'n_ref': len(ref), 'n_arm': len(arm)}
        mres = np.array([resid(a).mean() for a in grid])
        k = np.where(np.diff(np.sign(mres)) != 0)[0]
        out['alpha'] = float(grid[k[0]]) if len(k) else None
        each = []
        for j in range(len(arm)):
            mm = np.array([resid(a)[j] for a in grid])
            kk = np.where(np.diff(np.sign(mm)) != 0)[0]
            each.append({'tag': arm[j]['tag'], 'alpha': float(grid[kk[0]]) if len(kk) else None})
        out['each'] = each
        out['resid_at'] = {str(a): float(resid(a).mean()) for a in (0.0, 0.356, 1.0)}
        return out
    alpha = {'time_share': (EPOCHS - MIX_OFF) / (EPOCHS - 1)}
    alpha['inter'] = design_b(oloc, nolr)
    alpha['intra'] = design_b(prop, anolr)
    l2fit = line_fit(np.log(A(l2, 'kv')), np.log(A(l2, 'spikes')))

    def design_a(fixed, lr_runs, fit=None):
        f_ = fit or line_fit(np.log(A(fixed, 'kv')), np.log(A(fixed, 'spikes')))
        rp = np.array([math.log(r['spikes']) - (f_['beta'][0] + f_['beta'][1] * lmix(r, 0)) for r in lr_runs])
        ro = np.array([math.log(r['spikes']) - (f_['beta'][0] + f_['beta'][1] * lmix(r, 1)) for r in lr_runs])
        return rp, ro
    rp, ro = design_a(l2, l2lr, l2fit)
    each = rp / (rp - ro)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(NBOOT):
        i = rng.choice(len(l2), len(l2)); j = rng.choice(len(l2lr), len(l2lr))
        rpb, rob = design_a([l2[t] for t in i], [l2lr[t] for t in j])
        if abs(rpb.mean() - rob.mean()) > 1e-9:
            bs.append(rpb.mean() / (rpb.mean() - rob.mean()))
    in_rng = [r for r in l2lr if r['lam_post'] >= A(l2, 'kv').min()]
    rp4, ro4 = design_a(l2, in_rng, l2fit)
    alpha['l2'] = {'n_ref': len(l2), 'n_arm': len(l2lr), 'alpha': float(rp.mean() / (rp.mean() - ro.mean())),
                   'each': [{'tag': r['tag'], 'alpha': float(a)} for r, a in zip(l2lr, each)],
                   'each_mean': float(each.mean()), 'each_sd': float(each.std(ddof=1)),
                   'boot_lo': float(np.percentile(bs, 2.5)), 'boot_hi': float(np.percentile(bs, 97.5)),
                   'alpha_in_range': float(rp4.mean() / (rp4.mean() - ro4.mean())), 'n_in_range': len(in_rng),
                   'out_range': [r['knob'] for r in l2lr if r not in in_rng],
                   'lam_min_fixed': float(A(l2, 'kv').min())}
    # 런별 α 가 시간 비율 (0.356) 과 다른가 (단일 표본 t; 같은 기준선을 공유하므로 독립은 아님)
    for k_ in ('intra', 'l2'):
        v = np.array([e['alpha'] for e in alpha[k_]['each'] if e['alpha'] is not None])
        if len(v) >= 2:
            tt = (v.mean() - alpha['time_share']) / (v.std(ddof=1) / math.sqrt(len(v)))
            alpha[k_]['t_vs_share'] = {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'n': len(v), 't': float(tt),
                                       'p': float(2 * stats.t.sf(abs(tt), len(v) - 1))}
    sec2['alpha'] = alpha
    # 단순 요약값으로 본 착지 오차 (참고): plain L2 설계 A
    l2x = {'n_ref': len(l2), 'slope': l2fit['beta'][1], 'slope_se': l2fit['coef']['b']['se'],
           'resid_sd': l2fit['resid_sd'], 'by': {}}
    for k_ in ('lam_mean', 'lam_pre', 'lam_post', 'lam_last'):
        zs = []
        for r in l2lr:
            m, sm, sp = line_pred(l2fit, math.log(r[k_]))
            zs.append({'tag': r['tag'], 'lam': r[k_], 'pred_spk': math.exp(m), 'resid': math.log(r['spikes']) - m,
                       'z': (math.log(r['spikes']) - m) / sp})
        rz = np.array([z['resid'] for z in zs])
        tt = rz.mean() / (rz.std(ddof=1) / math.sqrt(len(rz)))
        l2x['by'][k_] = {'arm': zs, 'mean_resid': float(rz.mean()), 'sd_resid': float(rz.std(ddof=1)), 't': float(tt),
                         'p': float(2 * stats.t.sf(abs(tt), len(rz) - 1)), 'rms': float(np.sqrt((rz ** 2).mean()))}
    sec2['l2'] = l2x
    intra = {}
    for k_ in ('lam_mean', 'lam_pre', 'lam_post', 'lam_last'):
        f_ = line_fit(np.log(A(prop, k_)), np.log(A(prop, 'spikes')))
        zs = []
        for r in anolr:
            m, sm, sp = line_pred(f_, math.log(r[k_]))
            zs.append({'tag': r['tag'], 'pred_spk': math.exp(m), 'resid': math.log(r['spikes']) - m,
                       'z': (math.log(r['spikes']) - m) / sp})
        intra[k_] = {'n_ref': len(prop), 'slope': f_['beta'][1], 'resid_sd': f_['resid_sd'], 'arm': zs,
                     'mean_resid': float(np.mean([z['resid'] for z in zs]))}
    sec2['intra'] = intra

    # 2c. 데이터셋 사이 눈금: 같은 손잡이에서 C10 과 C100 의 스파이크 감소 (각 데이터셋 baseline 대비)
    #     ln(S/S_base) = a + b·ln(손잡이) + c·[C100]. c = C100 에서 같은 손잡이가 덜 줄이는 정도 (ln).
    #     손잡이 범위는 두 데이터셋이 겹치는 구간만 (국소 직선).
    U100 = [r for r in runs if r['use'] and r['ds'] == 'C100']
    b100 = [r for r in U100 if r['key'] == 'base']
    sb10, sb100 = math.exp(np.log(A(base, 'spikes')).mean()), math.exp(np.log(A(b100, 'spikes')).mean())
    xfer = {'base_c10': {'n': len(base), 'spikes': sb10, 'acc': A(base, 'val_acc').mean()},
            'base_c100': {'n': len(b100), 'spikes': sb100, 'acc': A(b100, 'val_acc').mean(),
                          'servers': sorted(r['server'] for r in b100)}, 'methods': {}}
    for key in ('ours_ch', 'prop', 'l2', 'bpsr'):
        c10 = [r for r in U if r['key'] == key]
        c100 = [r for r in U100 if r['key'] == key]
        if len(c100) < 2:
            continue
        lo_k = max(A(c10, 'kv').min(), A(c100, 'kv').min())
        hi_k = min(A(c10, 'kv').max(), A(c100, 'kv').max())
        a10 = [r for r in c10 if lo_k <= r['kv'] <= hi_k]
        a100 = [r for r in c100 if lo_k <= r['kv'] <= hi_k]
        data = a10 + a100
        y = np.array([math.log(r['spikes'] / (sb10 if r['ds'] == 'C10' else sb100)) for r in data])
        x = np.log(A(data, 'kv'))
        g = np.array([0.0] * len(a10) + [1.0] * len(a100))
        f_ = ols(np.c_[np.ones_like(x), x, g], y, ['a', 'b', 'c'])
        # 같은 손잡이 묶음별 감소 %
        grp = []
        for kn in sorted({r['knob'] for r in data}, key=knob_val):
            g10 = [r for r in a10 if r['knob'] == kn]; g100 = [r for r in a100 if r['knob'] == kn]
            grp.append({'knob': kn, 'n10': len(g10), 'n100': len(g100),
                        'red10': (math.exp(np.log(A(g10, 'spikes')).mean()) / sb10 - 1) * 100 if g10 else None,
                        'red100': (math.exp(np.log(A(g100, 'spikes')).mean()) / sb100 - 1) * 100 if g100 else None,
                        'acc10': A(g10, 'val_acc').mean() - xfer['base_c10']['acc'] if g10 else None,
                        'acc100': A(g100, 'val_acc').mean() - xfer['base_c100']['acc'] if g100 else None})
        xfer['methods'][key] = {'range': (lo_k, hi_k), 'n10': len(a10), 'n100': len(a100), 'c': f_['coef']['c'],
                                'b': f_['coef']['b'], 'df': f_['df'], 'groups': grp}
    # 기전: 같은 ρ 에서 C100 λ / C10 λ 와 L 비 (loss-ratio 팔만)
    lamx = {}
    for key in ('ours_ch', 'prop'):
        for kn in sorted({r['knob'] for r in U100 if r['key'] == key}, key=knob_val):
            g10 = [r for r in U if r['key'] == key and r['knob'] == kn and 'lam_post' in r]
            g100 = [r for r in U100 if r['key'] == key and r['knob'] == kn and 'lam_post' in r]
            if g10 and g100:
                lamx[f'{key} {kn}'] = {'n10': len(g10), 'n100': len(g100),
                                       'lam_pre_ratio': A(g100, 'lam_pre').mean() / A(g10, 'lam_pre').mean(),
                                       'lam_post_ratio': A(g100, 'lam_post').mean() / A(g10, 'lam_post').mean(),
                                       'L_pre_ratio': A(g100, 'L_pre10').mean() / A(g10, 'L_pre10').mean(),
                                       'L_post_ratio': A(g100, 'L_post10').mean() / A(g10, 'L_post10').mean(),
                                       'R_ratio': A(g100, 'R_pre10').mean() / A(g10, 'R_pre10').mean(),
                                       'L10': (A(g10, 'L_pre10').mean(), A(g10, 'L_post10').mean()),
                                       'L100': (A(g100, 'L_pre10').mean(), A(g100, 'L_post10').mean())}
    xfer['lam_ratio'] = lamx
    xfer['c100_runs'] = [{k: r.get(k) for k in ('tag', 'key', 'knob', 'seed', 'server', 'train_loss', 'train_acc',
                                                'val_loss', 'val_acc', 'spikes', 'exclude_reason')} for r in runs
                         if r['ds'] == 'C100']
    sec2['xfer'] = xfer
    res['sec2'] = sec2

    # ---------------------------------------------------------------- 3. 기전
    sec3 = {}
    EPS = [1, 5, 10, 30, 100, 199, 200, 201, 250, 309]
    # 3a. λ 궤적 (λ_used) — ours 는 ρ 묶음 평균, prop·l2_lr 도
    def lam_table(rs):
        out = {}
        for kn in sorted({r['knob'] for r in rs}, key=knob_val):
            g = [r for r in rs if r['knob'] == kn and 'lam_used' in r]
            if not g:
                continue
            M = np.array([r['lam_used'] for r in g])
            out[kn] = {'n': len(g), 'at': {e: float(M[:, e].mean()) for e in EPS},
                       'rise_5_199': float((M[:, 199] / M[:, 5]).mean()),
                       'drop_199_201': float((M[:, 201] / M[:, 199]).mean()),
                       'last_cv': float(100 * M[:, -1].std(ddof=1) / M[:, -1].mean()) if len(g) > 1 else None}
        return out
    sec3['lam'] = {'ours_ch': lam_table(ours), 'prop': lam_table(prop), 'l2_lr': lam_table(l2lr)}

    def pooled_ratio(rs, a, b):
        v = np.array([math.log(r['lam_used'][b] / r['lam_used'][a]) for r in rs if 'lam_used' in r])
        return {'mean_pct': (math.exp(v.mean()) - 1) * 100, 'min_pct': (math.exp(v.min()) - 1) * 100,
                'max_pct': (math.exp(v.max()) - 1) * 100, 'n': len(v)}
    sec3['lam_summary'] = {k: {'rise': pooled_ratio(v, 5, 199), 'drop': pooled_ratio(v, 199, 201)}
                           for k, v in (('ours_ch', ours), ('prop', prop), ('l2_lr', l2lr))}
    # L, R 분해 (해제 전후 10에폭 평균): λ ∝ L/R 이므로 ln 변화 = ln L 변화 − ln R 변화
    def lr_decomp(rs):
        v = [r for r in rs if 'L_pre10' in r]
        dL = np.array([math.log(r['L_post10'] / r['L_pre10']) for r in v])
        dR = np.array([math.log(r['R_post10'] / r['R_pre10']) for r in v])
        return {'n': len(v), 'dL_pct': (math.exp(dL.mean()) - 1) * 100, 'dR_pct': (math.exp(dR.mean()) - 1) * 100,
                'dL_sd': float(dL.std(ddof=1)) if len(v) > 1 else None,
                'L_pre': float(np.mean([r['L_pre10'] for r in v])), 'L_post': float(np.mean([r['L_post10'] for r in v]))}
    sec3['LR_decomp'] = {k: lr_decomp(v) for k, v in (('ours_ch', ours), ('ch_nolr', nolr), ('prop', prop),
                                                       ('abl_nolr', anolr), ('l2_lr', l2lr))}
    # 3b. 고정 λ 팔의 실효 ρ 궤적
    def rho_rows(rs):
        return [{'tag': r['tag'], 'knob': r['knob'], 'at': {e: float(r['rho_eff'][e]) for e in EPS},
                 'mean': r['rho_mean'], 'pre': r['rho_pre'], 'post': r['rho_post'], 'last': r['rho_last'],
                 'min_pre': float(r['rho_eff'][PRE[0]:PRE[1]].min()), 'max_post': float(r['rho_eff'][POST[0]:].max()),
                 'post_over_pre': r['rho_post'] / r['rho_pre']} for r in rs if 'rho_eff' in r]
    sec3['rho_fixed'] = {'ch_nolr': rho_rows(nolr), 'abl_nolr': rho_rows(anolr)}
    # loss-ratio 런의 실현 비 (검산): ρ 대비 평균 편차 (에폭 0·200 제외)
    def realized(rs):
        v = []
        for r in rs:
            if 'rho_eff' not in r:
                continue
            m = np.r_[r['rho_eff'][1:MIX_OFF], r['rho_eff'][MIX_OFF + 1:]]
            v.append(m.mean() / r['kv'])
        return {'n': len(v), 'mean_ratio': float(np.mean(v)), 'min': float(np.min(v)), 'max': float(np.max(v))}
    sec3['rho_realized'] = {k: realized(v) for k, v in (('ours_ch', ours), ('prop', prop), ('l2_lr', l2lr))}

    # 3c. 스파이크 궤적: ch_nolr vs ours 3e-3 (같은 착지, n=4) — 에폭별 비와 z (ours 3e-3 시드 SD 기준)
    S3 = np.array([r['s_traj'] for r in g3])
    sq = q['s_traj']
    TR = [0, 1, 2, 5, 10, 30, 60, 100, 150, 199, 200, 201, 210, 250, 300, 309]
    traj = []
    for e in TR:
        lv = np.log(S3[:, e])
        sd = lv.std(ddof=1)
        z = (math.log(sq[e]) - lv.mean()) / (sd * math.sqrt(1 + 1 / len(lv))) if sd > 0 else None
        traj.append({'ep': e, 'ours_mean': float(S3[:, e].mean()), 'ours_cv': float(100 * sd), 'nolr': float(sq[e]),
                     'ratio_pct': (sq[e] / math.exp(lv.mean()) - 1) * 100, 'z': z})
    sec3['traj_vs_3e3'] = {'n': len(g3), 'rows': traj, 'tags': [r['tag'] for r in g3]}
    # 3d. 혼합 증강 해제 반응 d_mix = ln(마지막10 / 해제직전10). 착지 스파이크 공변량 ANCOVA
    def dmix_cmp(ref, arm, label):
        out = {'label': label, 'n_ref': len(ref), 'n_arm': len(arm),
               'ref_mean_pct': (math.exp(A(ref, 'd_mix').mean()) - 1) * 100,
               'ref_sd': float(A(ref, 'd_mix').std(ddof=1)),
               'arm_vals_pct': [(math.exp(v) - 1) * 100 for v in A(arm, 'd_mix')]}
        data = ref + arm
        x = np.log(A(data, 'spikes')) - np.log(A(data, 'spikes')).mean()
        g = np.array([0.0] * len(ref) + [1.0] * len(arm))
        f = ols(np.c_[np.ones_like(x), x, g], A(data, 'd_mix'), ['a', 'b', 'd'])
        out['ancova'] = f['coef']
        out['ancova_df'] = f['df']
        out['resid_sd'] = f['resid_sd']
        if len(arm) >= 2:
            out['welch'] = welch(A(arm, 'd_mix'), A(ref, 'd_mix'))
        return out
    sec3['dmix'] = {'base': {'n': len(base), 'mean_pct': (math.exp(A(base, 'd_mix').mean()) - 1) * 100,
                             'sd': float(A(base, 'd_mix').std(ddof=1)),
                             'vals_pct': [(math.exp(v) - 1) * 100 for v in A(base, 'd_mix')]},
                    'inter': dmix_cmp(ours, nolr, 'ours_inter_ch vs - loss-ratio'),
                    'intra': dmix_cmp(prop, anolr, 'ours_intra_ch vs ours_intra_ch - loss-ratio'),
                    'l2': dmix_cmp(l2lr, l2, 'plain L2 + loss-ratio (기준) vs plain L2 (고정 λ)'),
                    'bpsr': {'n': len(bpsr), 'mean_pct': (math.exp(A(bpsr, 'd_mix').mean()) - 1) * 100,
                             'sd': float(A(bpsr, 'd_mix').std(ddof=1))}}
    # 묶음별 d_mix 평균 (참고)
    sec3['dmix_groups'] = {}
    for k_, rs in (('ours_ch', ours), ('prop', prop), ('l2', l2), ('l2_lr', l2lr), ('bpsr', bpsr),
                   ('ch_nolr', nolr), ('abl_nolr', anolr)):
        sec3['dmix_groups'][k_] = [{'knob': kn, 'n': len(g), 'spikes': A(g, 'spikes').mean(),
                                    'd_mix_pct': (math.exp(A(g, 'd_mix').mean()) - 1) * 100}
                                   for kn in sorted({r['knob'] for r in rs}, key=knob_val)
                                   for g in [[r for r in rs if r['knob'] == kn]]]

    # 3e. 착지 시드 변동 (같은 손잡이 묶음 n≥2, 묶음 고정효과 + 서버)
    lnS = lambda r: math.log(r['spikes'])
    acc = lambda r: r['val_acc']
    cv = {k: pooled_resid(v, lnS) for k, v in (('ours_ch', ours), ('prop', prop), ('bpsr', bpsr), ('l2', l2))}
    cv['base'] = pooled_resid(base, lnS)
    cv3 = {k: pooled_resid([r for r in v if sum(x['knob'] == r['knob'] for x in v) >= 3], lnS)
           for k, v in (('ours_ch', ours), ('prop', prop), ('bpsr', bpsr), ('l2', l2))}
    accsd = {k: pooled_resid(v, acc) for k, v in (('ours_ch', ours), ('prop', prop), ('bpsr', bpsr), ('l2', l2))}
    accsd['base'] = pooled_resid(base, acc)
    ft = {'ours_vs_bpsr': ftest(cv['ours_ch'], cv['bpsr']), 'ours_vs_l2': ftest(cv['ours_ch'], cv['l2']),
          'prop_vs_l2': ftest(cv['prop'], cv['l2']), 'prop_vs_bpsr': ftest(cv['prop'], cv['bpsr']),
          'ours_vs_base': ftest(cv['ours_ch'], cv['base']), 'l2_vs_base': ftest(cv['l2'], cv['base']),
          'bpsr_vs_base': ftest(cv['bpsr'], cv['base'])}
    # 고정 λ 두 방법 합동 vs loss-ratio 두 방법 합동
    def pool2(a, b):
        df = a['df'] + b['df']
        return {'sd': math.sqrt((a['sd'] ** 2 * a['df'] + b['sd'] ** 2 * b['df']) / df), 'df': df}
    lrp, fxp = pool2(cv['ours_ch'], cv['prop']), pool2(cv['bpsr'], cv['l2'])
    ft['lr_vs_fixed_pooled'] = ftest(lrp, fxp)
    ft['lr_vs_fixed_pooled'].update(sd_lr=lrp['sd'], sd_fixed=fxp['sd'])
    # 선 잔차 (l2_lr 은 손잡이당 n=1 이라 같은 손잡이 SD 를 못 잰다 → ln S ~ ln knob 직선 잔차로 비교)
    line_res = {}
    for k_, rs in (('ours_ch', ours), ('prop', prop), ('bpsr', bpsr), ('l2', l2), ('l2_lr', l2lr)):
        f_ = line_fit(np.log(A(rs, 'kv')), np.log(A(rs, 'spikes')))
        line_res[k_] = {'n': len(rs), 'resid_sd': f_['resid_sd'], 'df': f_['df'], 'slope': f_['beta'][1]}
    sec3['cv'] = {'pooled': cv, 'pooled_n3': cv3, 'acc_sd': accsd, 'F': ft, 'line_resid': line_res}

    # 3f. loss-ratio 되먹임의 부호: 같은 ρ 묶음 안에서 시드별 ln λ_post 와 ln S 의 기울기 (묶음 고정효과)
    def within_slope(rs, ykey, xkey='spikes', ylog=True):
        grp = {}
        for r in rs:
            if ykey in r:
                grp.setdefault(r['knob'], []).append(r)
        grp = {k: v for k, v in grp.items() if len(v) >= 2}
        xs, ys = [], []
        for v in grp.values():
            x = np.log(A(v, xkey)); y = np.log(A(v, ykey)) if ylog else A(v, ykey)
            xs.extend(x - x.mean()); ys.extend(y - y.mean())
        xs, ys = np.array(xs), np.array(ys)
        b = (xs @ ys) / (xs @ xs)
        res_ = ys - b * xs
        df = len(xs) - len(grp) - 1
        se = math.sqrt((res_ @ res_) / df / (xs @ xs))
        return {'slope': b, 'se': se, 'df': df, 'p': 2 * stats.t.sf(abs(b / se), df), 'n': len(xs), 'groups': len(grp)}
    fb = {}
    for k_, rs in (('ours_ch', ours), ('prop', prop)):
        fb[k_] = {'lam_post': within_slope(rs, 'lam_post'), 'lam_last': within_slope(rs, 'lam_last'),
                  'R_end': within_slope(rs, 'R_end')}
    # 용량-반응 탄력 k = −d lnS / d lnλ: ours 는 λ_post 직선 기울기 (ρ 사이), plain L2 는 고정 λ 직선
    k_ours = -sec2['lam_fits']['lam_post']['slope']
    k_l2 = -l2fit['beta'][1]
    for k_, kk in (('ours_ch', k_ours), ('prop', -sec2['intra']['lam_post']['slope'])):
        s_l = fb[k_]['lam_post']['slope']
        fb[k_]['k'] = kk
        fb[k_]['amp'] = 1 / (1 + kk * s_l) if (1 + kk * s_l) > 0 else None
        # s_l 의 95% 구간 끝에서
        lo_ = s_l - tcrit(fb[k_]['lam_post']['df']) * fb[k_]['lam_post']['se']
        hi_ = s_l + tcrit(fb[k_]['lam_post']['df']) * fb[k_]['lam_post']['se']
        fb[k_]['amp_range'] = [1 / (1 + kk * hi_) if 1 + kk * hi_ > 0 else None,
                               1 / (1 + kk * lo_) if 1 + kk * lo_ > 0 else None]
    fb['k_l2'] = k_l2
    sec3['feedback'] = fb

    # 3g. 같은 스파이크 구조 지표: ch_nolr vs ours 곡선 (200~420K), abl_nolr vs prop, l2_lr vs l2
    MET = ['w_mean', 'lnR_rd', 'dead_pct', 'gini_sw', 'top10_sw'] + [f'share_{s}' for s in STAGES] + \
          [f'dead_{s}' for s in STAGES] + [f'w_{s}' for s in STAGES]
    def struct_cmp(ref, arm, mets):
        out = {}
        for mt in mets:
            rr = [r for r in ref if r.get(mt) is not None]
            aa = [r for r in arm if r.get(mt) is not None]
            if len(rr) < 4 or not aa:
                continue
            f_ = line_fit(np.log(A(rr, 'spikes')), A(rr, mt))
            dl = []
            for r in aa:
                m, sm, sp = line_pred(f_, math.log(r['spikes']))
                dl.append({'tag': r['tag'], 'value': r[mt], 'pred': m, 'delta': r[mt] - m, 'se': sp, 'z': (r[mt] - m) / sp})
            o = {'n_ref': len(rr), 'n_arm': len(aa), 'each': dl, 'resid_sd': f_['resid_sd']}
            if len(aa) >= 2:
                cen = math.exp(np.log(A(aa, 'spikes')).mean())
                data = rr + aa
                x = np.log(A(data, 'spikes')) - math.log(cen)
                g = np.array([0.0] * len(rr) + [1.0] * len(aa))
                ff = ols(np.c_[np.ones_like(x), x, g], A(data, mt), ['a', 'b', 'd'])
                o['ancova'] = ff['coef']['d']
                o['ref_at_center'] = ff['coef']['a']['est']
            out[mt] = o
        return out
    sec3['struct'] = {'inter': struct_cmp(ob, nolr, MET),
                      'intra': struct_cmp([r for r in prop if BAND_INTRA[0] <= r['spikes'] <= BAND_INTRA[1]], anolr, MET),
                      'l2': struct_cmp(lb, lr_, ['dead_pct', 'gini_sw', 'top10_sw'] + [f'share_{s}' for s in STAGES] +
                                       [f'dead_{s}' for s in STAGES])}
    res['sec3'] = sec3

    # ---------------------------------------------------------------- 4. 학습 시간
    sec4 = {}
    for k_, rs in (('base', base), ('ours_ch', ours), ('ch_nolr', nolr), ('prop', prop), ('abl_nolr', anolr),
                   ('l2', l2), ('l2_lr', l2lr), ('bpsr', bpsr)):
        for sv in ('138', '23'):
            v = [r['sec_med'] for r in rs if r['server'] == sv and not math.isnan(r.get('sec_med', float('nan')))]
            if v:
                sec4[f'{k_}@{sv}'] = {'n': len(v), 'median': float(np.median(v)), 'min': float(min(v)),
                                      'max': float(max(v)), 'vals': v}
    # ch_nolr (23) vs ours_ch (23): 단일 런 vs 묶음
    o23 = [r['sec_med'] for r in ours if r['server'] == '23']
    sec4['nolr_vs_ours23'] = one_vs_group(q['sec_med'], o23)
    l2v = [r['sec_med'] for r in l2 if r['server'] == '138']
    l2lrv = [r['sec_med'] for r in l2lr if r['server'] == '138']
    sec4['l2lr_vs_l2_138'] = welch(l2lrv, l2v)
    pv = [r['sec_med'] for r in prop if r['server'] == '138']
    av = [r['sec_med'] for r in anolr if r['server'] == '138']
    sec4['anolr_vs_prop_138'] = welch(av, pv) if len(av) >= 2 else None
    res['sec4'] = sec4
    return res, dict(base=base, ours=ours, nolr=nolr, prop=prop, anolr=anolr, l2=l2, l2lr=l2lr, bpsr=bpsr, g3=g3,
                     ob=ob)


# ============================================================================ 보고
def fp(p):
    if p is None:
        return '–'
    return f'{p:.3f}' if p >= 0.001 else f'{p:.1e}'


def K(x):
    return f'{x:,.0f}'


def E(x, d=2):
    return f'{x:.{d}e}'.replace('e-0', 'e-').replace('e+0', 'e+')


def KN(k):
    return k.replace('p', '.')


def row(r, extra):
    if 'val_acc' not in r:
        return (f'| {r["arm"]} | {KN(r["knob"])} | s{r["seed"]} | {r["server"]} | – | – | – | – | – | – | '
                + ' | '.join('–' for _ in extra) + f' | {r["exclude_reason"]} |')
    ex = ' | '.join(f(r) for f in extra)
    return (f'| {r["arm"]} | {KN(r["knob"])} | s{r["seed"]} | {r["server"]} | {r["best_epoch"]} | {r["train_loss"]:.4f} | '
            f'{r["train_acc"]:.2f} | {r["val_loss"]:.4f} | {r["val_acc"]:.2f} | {K(r["spikes"])} | {ex} | '
            f'{r["exclude_reason"] or ""} |')


def mean_row(rs, label, knob, extra_mean):
    a = lambda k: A(rs, k)
    sd = f' ±{a("val_acc").std(ddof=1):.2f}' if len(rs) > 1 else ''
    ssd = f' ±{a("spikes").std(ddof=1):,.0f}' if len(rs) > 1 else ''
    ex = ' | '.join(f(rs) for f in extra_mean)
    return (f'| **{label} 평균** | {KN(knob)} | n={len(rs)} | {"/".join(sorted({r["server"] for r in rs}))} | | '
            f'{a("train_loss").mean():.4f} | {a("train_acc").mean():.2f} | {a("val_loss").mean():.4f} | '
            f'**{a("val_acc").mean():.2f}{sd}** | {a("spikes").mean():,.0f}{ssd} | {ex} | |')


def write_md(runs, res, G, dup):
    L = []
    w = L.append
    s1, s2, s3, s4, cr = res['sec1'], res['sec2'], res['sec3'], res['sec4'], res['cross']
    q = G['nolr'][0]
    n = res['n']
    bm = s1['vs_base']
    ls = s1['lin_srv']['coef']['d']
    pool = s1['pooled_acc_sd']
    cvp = s3['cv']['pooled']
    F = s3['cv']['F']
    lf = s2['lam_fits']
    rf = s2['rho_fit']
    rp = s2['nolr_rho_preds']
    al = s2['alpha']
    xf = s2['xfer']
    fx = s3['rho_fixed']['ch_nolr'][0]
    fxa = s3['rho_fixed']['abl_nolr']
    fb = s3['feedback']
    dm = s3['dmix']
    lam_o = s3['lam_summary']['ours_ch']
    dLo = s3['LR_decomp']['ours_ch']
    l2x = s2['l2']
    tl = s3['traj_vs_3e3']['rows']
    t199 = [x for x in tl if x['ep'] == 199][0]
    t30 = [x for x in tl if x['ep'] == 30][0]
    t100 = [x for x in tl if x['ep'] == 100][0]
    t309 = [x for x in tl if x['ep'] == 309][0]
    g3spk = A(G['g3'], 'spikes').mean()
    sti, sii, sll = s3['struct']['inter'], s3['struct']['intra'], s3['struct']['l2']
    zs = [abs(sti[m]['each'][0]['z']) for m in sti]
    xo, xp, xl, xb = (xf['methods'][k] for k in ('ours_ch', 'prop', 'l2', 'bpsr'))
    pct = lambda v: (math.exp(v) - 1) * 100
    ci_, cl = cr['intra'], cr['l2']
    fpool = F['lr_vs_fixed_pooled']
    go3 = [g for g in xo['groups'] if g['knob'] == '3e-3'][0]
    gl25 = [g for g in xl['groups'] if g['knob'] == '2p5e-7'][0]
    lrL = [v[k] for v in xf['lam_ratio'].values() for k in ('L_pre_ratio', 'L_post_ratio')]
    lrl = [v[k] for v in xf['lam_ratio'].values() for k in ('lam_pre_ratio', 'lam_post_ratio')]
    dg_lr = [x['d_mix_pct'] for k in ('ours_ch', 'prop', 'l2_lr') for x in s3['dmix_groups'][k]]
    dg_fx = [x['d_mix_pct'] for k in ('ch_nolr', 'abl_nolr', 'l2', 'bpsr') for x in s3['dmix_groups'][k]]
    pof = [fx['post_over_pre']] + [x['post_over_pre'] for x in fxa]
    alphas = [al['inter']['alpha'], al['intra']['alpha'], al['l2']['alpha']]

    w('# `- loss-ratio` 어블레이션 — λ 제어가 바꾸는 것 (R19 CIFAR-10, 26-10-14)')
    w('')
    w('- 요청 (사용자 10-14): "어블레이션 각 방법 분석도 해줄 수 있어?" — 이 문서는 loss-ratio 몫. 기전 분석 중심.')
    w('- 팔: `- loss-ratio` (run 키 `ch_nolr`) = `ours_inter_ch` (run 키 `ours_ch`) 에서 loss-ratio 만 끄고 λ 를 1.25e-7 로 고정. '
      'override 블록 대조로 두 팔의 설정 차이는 λ 제어 (`reg_spike_loss_ratio`, 손잡이 ρ→`reg_spike_out_const`) 뿐임을 확인 '
      '(교차 확인 쌍 `prop`/`abl_nolr`, `l2`/`l2_lr` 도 같음; 옛 런에 없는 `reg_spike_maxnorm_no_inv`·`reg_spike_wta_rev_w1` 은 기본값 False).')
    w('- 용어: **loss-ratio** = 매 에폭 끝에 다음 에폭 λ 를 λ = ρ·L/R 로 다시 푸는 제어 (L = 과제 손실, R = 가중 스파이크 벌점의 원값, λ 곱하기 전; '
      '`proc.py` 1240~1296). **실효 ρ** = 그 에폭에 실제로 걸린 벌점/과제 손실 비 = λ·R/L (loss-ratio 런은 정의상 ≈ρ, 고정 λ 런은 움직인다). '
      '**혼합 증강 해제** = 0-기준 에폭 200 부터 mixup/cutmix 를 끔 (`mix_off_iter = 500×200`). '
      '**해제 반응** = 마지막 10에폭 평균 평가 스파이크 / 해제 직전 10에폭 (190~199) 평균 − 1. '
      '**α** = 착지에 대한 해제 뒤 구간 λ 의 무게 (2c). **착지** = best 에폭의 스파이크 수. **용량-반응선** = 고정 λ 런들의 ln λ – ln 착지 스파이크 직선. '
      '**손잡이** = 팔의 세기 설정값 (loss-ratio 팔은 ρ, 고정 λ 팔은 λ).')
    w('- 정확도 = 310 에폭 중 best val acc, spikes = 그 에폭 s_count (원값, 이미지당). train.log·reg_detail.csv·reg_neuron_detail.csv 를 '
      f'두 서버에서 직접 다시 읽음 (sejong 은 ssh 로 수집기를 stdin 실행, 읽기만). 캐시 `{CACHE}/c138.json.gz`, `c23.json.gz` '
      f'({time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(CACHE + "/c23.json.gz")))} 수집).')
    w(f'- 대상 (완주·비배제): `- loss-ratio` **n={n["ch_nolr"]}** (s1, 23; s2·s3 는 hold), `ours_inter_ch` n={n["ours_ch"]} '
      f'(정확도 곡선 200~420K 안 n={res["ours_band_n"]}, ρ 눈금 국소 직선 ρ 1.5e-3~5e-3 n={rf["n"]}), baseline n={n["base"]} (138 3 · 23 3). '
      f'교차 확인 (섞지 않음): `ours_intra_ch` n={n["prop"]} vs `ours_intra_ch - loss-ratio` (`abl_nolr`) n={n["abl_nolr"]} (λ 3개 각 n=1); '
      f'`plain L2` (고정 λ) n={n["l2"]} vs `plain L2 + loss-ratio` (`l2_lr`) n={n["l2_lr"]} (ρ 5개 각 n=1); 참고 BPSR (고정 λ) n={n["bpsr"]}; '
      f'데이터셋 눈금 (2d) 에 C100 baseline·`ours_inter_ch`·`ours_intra_ch`·`plain L2`·BPSR. '
      '`ours_layer_w1 - loss-ratio` (`lw1_nolr`) 는 2에폭 런 (s94) 하나뿐이라 쓰지 않았다.')
    w('- 재현: `~/anaconda3/envs/venv_1/bin/python analysis/ablation_loss_ratio_261014.py [--refetch]`. 같은 스파이크 ANCOVA 와 통계 함수는 '
      '`nofinal_vs_inter_261014.py` 의 것을 그대로 불러 쓴다 (다른 팔 분석과 숫자 비교 가능). 수치 원본 json 은 scratchpad (`'
      + OUT_JSON + '`).')
    w('')

    # ------------------------------------------------------------------ 요약
    w('## 요약')
    w('')
    w(f'1. **같은 스파이크 정확도 — 판단 불가 (n=1).** `- loss-ratio` {q["val_acc"]:.2f}% @ {K(q["spikes"])} vs `ours_inter_ch` 곡선 '
      f'(n={res["ours_band_n"]}): 오프셋 d = **{ls["est"]:+.3f}%p** [95% CI {ls["lo"]:+.3f}, {ls["hi"]:+.3f}], p={fp(ls["p"])} (선형+서버, '
      f'nofinal 분석과 같은 모형). 같은 착지의 `ours_inter_ch` 3e-3 (n=4, {K(g3spk)}) 대비 {s1["vs_3e3"]["diff"]:+.3f}%p '
      f'[{s1["vs_3e3"]["lo"]:+.2f}, {s1["vs_3e3"]["hi"]:+.2f}]. 교차 확인 두 쌍도 0 근처 — intra (끈 팔 n=3) {ci_["lin"]["coef"]["d"]["est"]:+.3f}%p '
      f'[{ci_["lin"]["coef"]["d"]["lo"]:+.2f}, {ci_["lin"]["coef"]["d"]["hi"]:+.2f}], plain L2 (켠 팔 n=5 − 끈 팔) {cl["lin"]["coef"]["d"]["est"]:+.3f}%p '
      f'[{cl["lin"]["coef"]["d"]["lo"]:+.2f}, {cl["lin"]["coef"]["d"]["hi"]:+.2f}]. 어느 쌍에서도 약 ±0.2%p 보다 큰 효과는 없고, 그보다 작은 효과는 볼 수 없다. '
      f'baseline (n=6) 대비 {bm["base_own_sd"]["diff"]:+.3f}%p [{bm["pooled_sd"]["lo"]:+.2f}, {bm["pooled_sd"]["hi"]:+.2f}] · 스파이크 {bm["spk_pct"]:+.1f}%.')
    w(f'2. **λ 를 과제 손실에 묶는다.** loss-ratio 아래 λ 는 혼합 증강 구간에 오르고 (에폭 5→199 {lam_o["rise"]["mean_pct"]:+.0f}%, '
      f'범위 {lam_o["rise"]["min_pct"]:+.0f}~{lam_o["rise"]["max_pct"]:+.0f}%, n={lam_o["rise"]["n"]}) 해제 때 {lam_o["drop"]["mean_pct"]:+.0f}% 떨어진다 '
      f'(n={lam_o["drop"]["n"]}, 범위 {lam_o["drop"]["min_pct"]:+.0f}~{lam_o["drop"]["max_pct"]:+.0f}%). 떨어지는 이유는 L 이 {dLo["dL_pct"]:+.0f}% '
      f'(R 은 {dLo["dR_pct"]:+.0f}%). 고정 λ 는 같은 L 하락을 그대로 받아 **실효 ρ 가 {E(fx["pre"])} → {E(fx["post"])} (×{fx["post_over_pre"]:.2f})** 로 오른다 '
      f'(intra 고정 λ 3런 ×' + '/'.join(f'{x["post_over_pre"]:.2f}' for x in fxa) + ').')
    w(f'3. **같은 착지, 다른 경로.** `- loss-ratio` 는 `ours_inter_ch` ρ ≈ {rf["rho_star"]*1e3:.2f}e-3 (95% [{rf["rho_star_lo"]*1e3:.2f}, '
      f'{rf["rho_star_hi"]*1e3:.2f}]e-3) 와 같은 곳에 착지한다. 그러나 혼합 증강 구간에는 실효 ρ 가 낮아 같은 착지의 `ours_inter_ch` 3e-3 (n=4) 보다 '
      f'스파이크가 많고 (에폭 100 {t100["ratio_pct"]:+.1f}%, 199 {t199["ratio_pct"]:+.1f}%, z={t199["z"]:+.1f}), 해제 뒤 따라잡아 309 에서 '
      f'{t309["ratio_pct"]:+.1f}% (z={t309["z"]:+.1f}). 해제 반응 {dm["inter"]["arm_vals_pct"][0]:+.1f}% vs `ours_inter_ch` {dm["inter"]["ref_mean_pct"]:+.1f}% '
      f'(n={dm["inter"]["n_ref"]}; 착지 보정 차 {dm["inter"]["ancova"]["d"]["est"]:+.3f} ln, p={fp(dm["inter"]["ancova"]["d"]["p"])} — n=1 이라 기준 팔 잔차 SD 를 가정). '
      f'같은 방향이 교차 쌍에서 n 을 갖고 보인다: intra 고정 λ {np.mean(dm["intra"]["arm_vals_pct"]):+.1f}% (n=3) vs {dm["intra"]["ref_mean_pct"]:+.1f}% '
      f'(n={dm["intra"]["n_ref"]}), 차 {dm["intra"]["ancova"]["d"]["est"]:+.3f} ln (p={fp(dm["intra"]["ancova"]["d"]["p"])}); plain L2 고정 λ '
      f'(n={dm["l2"]["n_arm"]}) − loss-ratio (n={dm["l2"]["n_ref"]}) {dm["l2"]["ancova"]["d"]["est"]:+.3f} ln (p={fp(dm["l2"]["ancova"]["d"]["p"])}). '
      f'규제 없는 baseline 도 해제 뒤 {dm["base"]["mean_pct"]:+.1f}% (n=6) 줄어든다.')
    w(f'4. **착지는 해제 뒤 λ 에 시간 비율보다 큰 무게를 둔다.** ln λ_혼합 = (1−α)·ln λ_해제전 + α·ln λ_해제뒤 로 놓고 고정 λ 와 loss-ratio 의 착지가 맞는 α: '
      f'inter **{al["inter"]["alpha"]:.2f}** (n=1), intra **{al["intra"]["alpha"]:.2f}** (n=3, 런별 '
      + '/'.join(f'{e["alpha"]:.2f}' for e in al['intra']['each']) + f'), plain L2 **{al["l2"]["alpha"]:.2f}** (n=5, 부트스트랩 95% '
      f'[{al["l2"]["boot_lo"]:.2f}, {al["l2"]["boot_hi"]:.2f}]). 해제 뒤 구간의 시간 비율은 {al["time_share"]:.2f} — 착지 무게는 그 '
      f'{min(alphas) / al["time_share"]:.1f}~{max(alphas) / al["time_share"]:.1f}배 '
      f'(런별 α 대 {al["time_share"]:.2f}: intra p={fp(al["intra"]["t_vs_share"]["p"])}, plain L2 p={fp(al["l2"]["t_vs_share"]["p"])}). '
      'loss-ratio 는 무거운 구간에서 λ 를 38% 내리므로, 같은 평균 λ 라면 고정 λ 보다 덜 줄인다.')
    w(f'5. **ρ 눈금이 데이터셋 사이에 옮겨진다 (C10 → C100, 같은 R19).** 같은 손잡이에서 각 데이터셋 baseline 대비 스파이크 비의 C100 차 (ln): '
      f'loss-ratio 방법 `ours_inter_ch` **{xo["c"]["est"]:+.3f}** [{xo["c"]["lo"]:+.3f}, {xo["c"]["hi"]:+.3f}] (n={xo["n10"]}/{xo["n100"]}), '
      f'`ours_intra_ch` {xp["c"]["est"]:+.3f} [{xp["c"]["lo"]:+.3f}, {xp["c"]["hi"]:+.3f}] (n={xp["n10"]}/{xp["n100"]}); 고정 λ 방법 `plain L2` '
      f'**{xl["c"]["est"]:+.3f}** [{xl["c"]["lo"]:+.3f}, {xl["c"]["hi"]:+.3f}] (n={xl["n10"]}/{xl["n100"]}), BPSR {xb["c"]["est"]:+.3f} '
      f'[{xb["c"]["lo"]:+.3f}, {xb["c"]["hi"]:+.3f}] (n={xb["n10"]}/{xb["n100"]}). 예: ρ 3e-3 → C10 {go3["red10"]:+.1f}% (n={go3["n10"]}) / C100 '
      f'{go3["red100"]:+.1f}% (n={go3["n100"]}); plain L2 λ 2.5e-7 → C10 {gl25["red10"]:+.1f}% (n={gl25["n10"]}) / C100 {gl25["red100"]:+.1f}% (n={gl25["n100"]}). '
      f'기전: C100 의 과제 손실 L 이 {min(lrL):.1f}~{max(lrL):.1f}배라 loss-ratio 가 λ 를 {min(lrl):.1f}~{max(lrl):.1f}배로 올린다 (2d 표). '
      f'단 이것은 벌점 모양이 다른 방법 사이의 비교이고, 같은 ρ 의 정확도 손실은 데이터셋마다 다르다 (ρ 3e-3: C10 {go3["acc10"]:+.2f}, C100 {go3["acc100"]:+.2f}%p).')
    w(f'6. **착지 시드 변동 — loss-ratio 가 줄이지 않는다.** 같은 손잡이 묶음 ln 스파이크 잔차 SD: `ours_inter_ch` {cvp["ours_ch"]["sd"]*100:.2f}% '
      f'(자유도 {cvp["ours_ch"]["df"]}), `ours_intra_ch` {cvp["prop"]["sd"]*100:.2f}% ({cvp["prop"]["df"]}) vs 고정 λ `plain L2` '
      f'{cvp["l2"]["sd"]*100:.2f}% ({cvp["l2"]["df"]}), BPSR {cvp["bpsr"]["sd"]*100:.2f}% ({cvp["bpsr"]["df"]}); baseline {cvp["base"]["sd"]*100:.2f}% '
      f'({cvp["base"]["df"]}). loss-ratio 합동 / 고정 λ 합동 SD 비 {fpool["sd_lr"] / fpool["sd_fixed"]:.2f} (F={fpool["F"]:.2f}, p={fp(fpool["p"])}); '
      f'BPSR 만 따로 보면 p={fp(F["ours_vs_bpsr"]["p"])}, plain L2 와는 p={fp(F["ours_vs_l2"]["p"])}. 되먹임 부호가 이 방향과 맞다: 같은 ρ 에서 더 쏘는 시드는 '
      f'R 이 커서 λ 가 작다 (묶음 안 d ln λ_해제뒤 / d ln S = {fb["ours_ch"]["lam_post"]["slope"]:+.2f} ±{fb["ours_ch"]["lam_post"]["se"]:.2f}, '
      f'p={fp(fb["ours_ch"]["lam_post"]["p"])}, n={fb["ours_ch"]["lam_post"]["n"]}) → 선형 근사로 시드 차이를 ×{fb["ours_ch"]["amp"]:.2f} '
      f'[{fb["ours_ch"]["amp_range"][0]:.2f}, {fb["ours_ch"]["amp_range"][1]:.2f}] 키운다. `- loss-ratio` 자체의 시드 변동은 n=1 이라 **판단 불가**.')
    gi_ = sii['gini_sw']['ancova']; ti_ = sii['top10_sw']['ancova']
    w(f'7. **같은 스파이크에서 분포.** inter: 지표 {len(sti)}개 모두 `ours_inter_ch` 곡선 시드 산포 안 (|z| 최대 {max(zs):.1f}, n=1; 지표 수를 생각하면 기대 범위). '
      f'intra: 고정 λ 3런이 같은 스파이크에서 지니 {gi_["est"]:+.3f} ±{gi_["se"]:.3f}, 상위 10% 몫 {ti_["est"]:+.2f} ±{ti_["se"]:.2f}%p '
      f'(p={fp(gi_["p"])}, {fp(ti_["p"])}) — 발화가 덜 몰린다. inter 의 같은 지표도 같은 방향 (z={sti["gini_sw"]["each"][0]["z"]:+.1f}, '
      f'{sti["top10_sw"]["each"][0]["z"]:+.1f}), plain L2 는 차이 없음 ({sll["gini_sw"]["ancova"]["est"]:+.3f}, p={fp(sll["gini_sw"]["ancova"]["p"])}). '
      '원인은 확인하지 않았다 (가설: 감소가 해제 뒤로 몰리는 경로 차이). 채널 단위 (죽은 채널·활성 뉴런) 는 npz 가 없어 못 쟀다.')
    w(f'8. **학습 시간 차이 없음**: 에폭당 중앙값 `- loss-ratio` {q["sec_med"]:.0f}s vs `ours_inter_ch` (23) '
      f'{s4["ours_ch@23"]["median"]:.0f}s (n={s4["ours_ch@23"]["n"]}), 차 {s4["nolr_vs_ours23"]["diff"]:+.1f}s '
      f'[{s4["nolr_vs_ours23"]["lo"]:+.1f}, {s4["nolr_vs_ours23"]["hi"]:+.1f}]. 제어기는 에폭 끝 스칼라 한 번이다.')
    w('')
    w('**판정**: loss-ratio 가 바꾸는 것은 **λ 의 시간·데이터셋 배분** 이다. (i) λ 를 과제 손실에 묶어, 혼합 증강 해제처럼 L 이 바뀔 때 '
      f'벌점/과제 손실 비를 일정하게 둔다 — 고정 λ 는 해제 뒤 실효 ρ 가 ×{min(pof):.2f}~{max(pof):.2f} 로 오르고 스파이크가 그때 몰아서 준다. '
      f'(ii) 같은 이유로 ρ 가 C10·C100 사이에서 같은 스파이크 감소를 주는 눈금이 된다 (고정 λ 방법은 같은 λ 에서 C100 이 ln '
      f'{min(xl["c"]["est"], xb["c"]["est"]):.2f}~{max(xl["c"]["est"], xb["c"]["est"]):.2f} 덜 줄인다). '
      '정확도 (n=1, 교차 쌍 포함 ±0.2%p 안 판단 불가), 착지 재현성 (줄이지 않음, 되먹임은 오히려 키우는 부호) 에는 이득이 보이지 않는다. '
      '논문에 쓸 수 있는 것: "loss-ratio 는 정확도 기여가 아니라 세기 눈금의 이식성 (같은 모델, 다른 데이터셋) 과 학습 중 L 변화에 대한 '
      '자동 보정을 준다" — 단 데이터셋 이식성은 다른 벌점 방법과의 비교 (같은 벌점의 고정 λ C100 런 없음) 라는 단서를 단다. '
      '"정확도를 올린다", "착지를 안정시킨다" 는 쓸 수 없다.')
    w('')

    # ------------------------------------------------------------------ 0
    w('## 0. 런별 값 (best 에폭)')
    w('')
    w('열: train loss (규제 포함 `loss`) → train acc → val loss → val acc → spikes, 뒤에 파생: spikes vs baseline, '
      'λ 평균 (에폭 1~309 에 쓰인 λ), 해제 뒤 λ 평균 (에폭 200~309), 실효 ρ 평균 (1~309; 고정 λ `plain L2` 는 R 을 기록하지 않아 –), '
      '해제 반응, 에폭당 시간 중앙값 (초). 평균 행은 배제 행을 뺀 값, ± 는 시드 SD.')
    w('')
    bspk = A(G['base'], 'spikes').mean()
    extra = [lambda r: f'{(r["spikes"] / bspk - 1) * 100:+.1f}%',
             lambda r: E(r['lam_mean']) if 'lam_mean' in r else '–',
             lambda r: E(r['lam_post']) if 'lam_post' in r else '–',
             lambda r: E(r['rho_mean']) if 'rho_mean' in r else '–',
             lambda r: f'{(math.exp(r["d_mix"]) - 1) * 100:+.1f}%' if 'd_mix' in r else '–',
             lambda r: f'{r["sec_med"]:.0f}' if 'sec_med' in r else '–']
    extra_m = [lambda rs: f'{(A(rs, "spikes").mean() / bspk - 1) * 100:+.1f}%',
               lambda rs: E(A(rs, 'lam_mean').mean()) if all('lam_mean' in r for r in rs) else '–',
               lambda rs: E(A(rs, 'lam_post').mean()) if all('lam_post' in r for r in rs) else '–',
               lambda rs: E(A(rs, 'rho_mean').mean()) if all('rho_mean' in r for r in rs) else '–',
               lambda rs: f'{(math.exp(A(rs, "d_mix").mean()) - 1) * 100:+.1f}%',
               lambda rs: f'{np.median(A(rs, "sec_med")):.0f}']
    hdr = ('| 팔 | ρ / λ | seed | 서버 | best ep | train loss | train acc | val loss | val acc | spikes | spikes vs base | '
           'λ 평균 | 해제 뒤 λ | 실효 ρ 평균 | 해제 반응 | s/에폭 | 비고 |')
    sep = '|' + '---|' * 17

    def block(key, label, ds='C10'):
        allr = [r for r in runs if r['key'] == key and r['ds'] == ds]
        for kn in sorted({r['knob'] for r in allr}, key=lambda k: knob_val(k) if k != '-' else 0):
            g = sorted([r for r in allr if r['knob'] == kn], key=lambda r: r['seed'])
            for r in g:
                w(row(r, extra))
            gu = [r for r in g if r['use']]
            if gu:
                w(mean_row(gu, label, kn, extra_m))

    w('### 0a. `ours_inter_ch` 계열 (주 비교)')
    w('')
    w(hdr); w(sep)
    block('base', 'baseline')
    block('ch_nolr', '- loss-ratio')
    block('ours_ch', 'ours_inter_ch')
    w('')
    w('- `- loss-ratio` s2·s3: 원장 status=hold (10-13, final_step 판정 대기로 큐에서 보류). 시작 안 됨.')
    w('- `ours_inter_ch` ρ 1.6e-2 이상 4런은 원장 S30/S1<0.19 배제 (전부 115K 이하라 곡선·창에 안 듦). 3e-3 s901 은 2에폭 연기시험.')
    chk = [r for r in runs if r.get('inv_check')]
    bad = [r['tag'] for r in chk if (r['inv_check']['val_acc_diff'] is not None and abs(r['inv_check']['val_acc_diff']) > 0.005)
           or (r['inv_check']['spikes_diff'] is not None and abs(r['inv_check']['spikes_diff']) > 1)
           or not r['inv_check']['server_match']]
    w(f'- 원장 (`inventory.json`, 10-14 12:50) 교차검증: 로그에서 다시 읽은 런 {len(chk)}개 (C10·C100) 중 val acc·spikes·서버 불일치 {len(bad)}건'
      + (f' ({", ".join(bad)})' if bad else '') + '.' + (f' 두 서버 모두 로그가 있는 런 {len(dup)}개: {", ".join(dup)} (138 값을 씀).' if dup else ''))
    w('')
    w('### 0b. 교차 확인 팔 (섞지 않음)')
    w('')
    w(hdr); w(sep)
    block('abl_nolr', 'ours_intra_ch - loss-ratio')
    block('prop', 'ours_intra_ch')
    block('l2_lr', 'plain L2 + loss-ratio')
    block('l2', 'plain L2')
    block('lw1_nolr', 'ours_layer_w1 - loss-ratio')
    w('')
    w(f'- BPSR (C10 n={n["bpsr"]}) 은 착지 시드 변동·데이터셋 눈금 비교에만 쓰여 표를 생략한다 (값은 inter_vs_bpsr_261009.md 와 같음; 원장 교차검증 포함).')
    w('- `ours_intra_ch` 은 5e-3 s1 (23) 만 빼고 138. 2e-3 s3 은 원장 stale (0에폭) 이라 행이 없다. `abl_nolr` 3런·`l2_lr` 5런 (사용분) 은 모두 138.')
    w('')

    # ------------------------------------------------------------------ 1
    w('## 1. 같은 스파이크에서 정확도')
    w('')
    w(f'`acc = a + b·x [+ c·x²] + d·[- loss-ratio] [+ e·[서버 23]]`, x = ln(spikes) − ln({K(q["spikes"])}). '
      f'`ours_inter_ch` 200~420K n={res["ours_band_n"]} (ρ {", ".join(KN(k) for k in res["ours_band_rhos"])}) + `- loss-ratio` n=1. '
      'n=1 이라 d 는 그 런 하나가 곡선에서 떨어진 거리이고 SE 에 시드 잡음 전체가 들어간다.')
    w('')
    w('| 모형 | n | d (%p) | 95% CI | p | b (%p / ln) | e (서버 23) | 잔차 SD |')
    w('|---|---|---|---|---|---|---|---|')
    for nm, lab in (('lin', '선형'), ('lin_srv', '선형 + 서버'), ('quad', '2차'), ('quad_srv', '2차 + 서버')):
        f = s1[nm]
        d = f['coef']['d']
        e = f['coef'].get('e')
        w(f'| {lab} | {f["n"]} | **{d["est"]:+.3f}** | [{d["lo"]:+.3f}, {d["hi"]:+.3f}] | {fp(d["p"])} | '
          f'{f["coef"]["b"]["est"]:+.3f} ±{f["coef"]["b"]["se"]:.3f} | '
          + (f'{e["est"]:+.3f} (p={fp(e["p"])})' if e else '–') + f' | {f["resid_sd"]:.3f} |')
    w('')
    sa = s1['same_acc']
    w(f'- 같은 정확도에서 스파이크 비 (선형+서버, θ=d/b): {sa["ratio_pct"]:+.1f}%, Fieller 95% '
      + (f'[{sa["lo_pct"]:+.1f}, {sa["hi_pct"]:+.1f}]%' if sa['fieller'] == 'bounded' else '구간 무한') + '. 정확도 축 d 를 주 결과로 쓴다.')
    v3 = s1['vs_3e3']
    w(f'- 국소: 같은 착지의 `ours_inter_ch` 3e-3 (n=4, {K(g3spk)}, 스파이크 차 {s1["vs_3e3_spk"]*100:+.2f}%) 대비 '
      f'{v3["diff"]:+.3f}%p [{v3["lo"]:+.3f}, {v3["hi"]:+.3f}], p={fp(v3["p"])} (SE = 합동 SD·√(1+1/4)). '
      f'±12% 창 (n={s1["win12"]["n_grp"]}): {s1["win12"]["diff"]:+.3f}%p [{s1["win12"]["lo"]:+.3f}, {s1["win12"]["hi"]:+.3f}].')
    w(f'- 합동 SD = {pool["sd_plain"]:.3f}%p (`ours_inter_ch` 같은 ρ n≥2 묶음 {len(pool["groups"])}개, 자유도 {pool["df_plain"]}; nofinal 분석의 0.112 와 같은 정의).')
    md = s1['mde']
    w(f'- 최소 검출 차 (α=0.05, 검정력 0.8, ours 곡선 지금대로): `- loss-ratio` n=1 → {md["by_n"][1]:.2f}, n=3 → {md["by_n"][3]:.2f}, '
      f'n=4 → {md["by_n"][4]:.2f}, n=6 → {md["by_n"][6]:.2f}%p, 곡선 오차 하한 {md["floor"]:.2f}%p.')
    w('')
    w('### 1b. baseline 대비 (단일 런 vs n=6)')
    w('')
    w('| 기준 SD | Δacc (%p) | 95% CI | p | 자유도 |')
    w('|---|---|---|---|---|')
    for lab, x in (('baseline 자체 SD (n=6)', bm['base_own_sd']), (f'합동 SD {pool["sd_plain"]:.3f}', bm['pooled_sd'])):
        w(f'| {lab} | {x["diff"]:+.3f} | [{x["lo"]:+.3f}, {x["hi"]:+.3f}] | {fp(x["p"])} | {x["df"]} |')
    w('')
    w(f'- 스파이크 {bm["spk_pct"]:+.1f}% (baseline ln 스파이크 SD 로 본 95% [{bm["spk_lo"]:+.1f}, {bm["spk_hi"]:+.1f}]%).')
    w('')
    w('### 1c. 교차 확인 — 다른 방법에서 loss-ratio 를 껐을 때')
    w('')
    w('| 쌍 | 기준 곡선 n | 비교 팔 n | 스파이크 범위 | d (%p, 선형) | 95% CI | p | d (선형+서버) |')
    w('|---|---|---|---|---|---|---|---|')
    d1, d2 = ci_['lin']['coef']['d'], ci_['lin_srv']['coef']['d']
    w(f'| `ours_intra_ch - loss-ratio` (λ 2e-7·3e-7·5e-7 각 n=1) − `ours_intra_ch` | {ci_["n_ref"]} | {ci_["n_arm"]} | {K(BAND_INTRA[0])}~{K(BAND_INTRA[1])} | '
      f'**{d1["est"]:+.3f}** | [{d1["lo"]:+.3f}, {d1["hi"]:+.3f}] | {fp(d1["p"])} | {d2["est"]:+.3f} (p={fp(d2["p"])}) |')
    d1, d2 = cl['lin']['coef']['d'], cl['lin_srv']['coef']['d']
    w(f'| `plain L2 + loss-ratio` (ρ 2e-3~1e-2 각 n=1) − `plain L2` | {cl["n_ref"]} | {cl["n_arm"]} | {K(BAND_L2[0])}~{K(BAND_L2[1])} | '
      f'**{d1["est"]:+.3f}** | [{d1["lo"]:+.3f}, {d1["hi"]:+.3f}] | {fp(d1["p"])} | {d2["est"]:+.3f} (p={fp(d2["p"])}) |')
    if 'lin_overlap' in cl:
        d3 = cl['lin_overlap']['coef']['d']
        w(f'| 위 행, `plain L2` 범위 안 (≤{K(cl["ref_range"][1])}) 만 | {cl["n_ref"]} | {cl["n_arm_overlap"]} | 겹치는 구간 | '
          f'{d3["est"]:+.3f} | [{d3["lo"]:+.3f}, {d3["hi"]:+.3f}] | {fp(d3["p"])} | – |')
    w('')
    w('- 부호 주의: intra 행은 "끈 팔 − 켠 팔", plain L2 행은 "켠 팔 − 끈 팔".')
    w('- intra 개별점 (곡선 대비): ' + ', '.join(f'{KN(x["tag"].replace("r19c10-abl_nolr-", "λ "))} {x["delta"]:+.3f}%p (z={x["z"]:+.1f})' for x in ci_['each'])
      + f'. 5e-7 ({K(ci_["arm_spikes"][-1])}) 은 `ours_intra_ch` 최저점 {K(ci_["ref_range"][0])} 아래라 외삽.')
    w(f'- plain L2: `plain L2 + loss-ratio` 2e-3 (431K)·3e-3 (396K) 은 `plain L2` 범위 ({K(cl["ref_range"][0])}~{K(cl["ref_range"][1])}) 밖이거나 경계라 '
      '세 번째 행에서 뺐다. 세 쌍 모두 95% 구간이 0 을 포함하고 폭이 ±0.15~0.22%p 다.')
    w('')

    # ------------------------------------------------------------------ 2
    w('## 2. 세기 눈금 — 같은 ρ·같은 λ 에서')
    w('')
    w('loss-ratio 를 끄면 손잡이가 ρ 에서 λ 로 바뀌어 "같은 ρ" 가 정의되지 않는다. 대신 (a) 고정 λ 런이 `ours_inter_ch` 의 어느 ρ 와 같은 곳에 착지하는지, '
      '(b) 그 런의 실효 ρ·λ 의 어느 요약이 착지를 맞히는지, (c) 착지를 정하는 구간의 무게 α, (d) 같은 손잡이가 데이터셋 사이에서 같은 감소를 주는지 본다.')
    w('')
    w(f'### 2a. ρ 눈금 — `ours_inter_ch` ln S ~ ln ρ (국소, ρ 1.5e-3~5e-3, n={rf["n"]})')
    w('')
    w(f'- 기울기 {rf["slope"]:+.3f} ±{rf["slope_se"]:.3f} (ρ 2배에 스파이크 {(2 ** rf["slope"] - 1) * 100:+.1f}%), 잔차 SD {rf["resid_sd"]*100:.1f}% (ln). '
      f'전체 범위 (n={rf["full_n"]}, ρ 1.5e-3~1.2e-2) 는 휘어 있어 (직선 잔차 {rf["full_lin_sd"]*100:.1f}% vs 2차 {rf["full_quad_sd"]*100:.1f}%, 2차 항 p={fp(rf["full_quad_c_p"])}) 국소 직선을 쓴다. '
      '10-10 (n=17) 은 −0.298, 3.2%.')
    w(f'- `- loss-ratio` 착지 {K(q["spikes"])} 에 해당하는 ρ = **{rf["rho_star"]*1e3:.2f}e-3** (95% [{rf["rho_star_lo"]*1e3:.2f}, {rf["rho_star_hi"]*1e3:.2f}]e-3, 새 런 하나의 잔차 포함).')
    w('')
    w('| `- loss-ratio` 실효 ρ 요약 | 값 | 그 ρ 의 `ours_inter_ch` 예측 착지 | 실제/예측 | z |')
    w('|---|---|---|---|---|')
    for k_, lab in (('rho_pre', '혼합 증강 구간 (에폭 1~199) 평균'), ('rho_post', '해제 뒤 (200~309) 평균'),
                    ('rho_last', '마지막 10에폭 평균'), ('rho_mean', '전 구간 (1~309) 평균')):
        x = rp[k_]
        w(f'| {lab} | {E(x["value"])} | {K(x["pred_spk"])} | {(math.exp(x["resid"]) - 1) * 100:+.1f}% | {x["z"]:+.1f} |')
    w('')
    w(f'### 2b. λ 눈금 — `ours_inter_ch` ln S ~ ln λ_요약 (국소 n={lf["lam_mean"]["n"]}), `- loss-ratio` 는 λ=1.25e-7 상수')
    w('')
    w('| λ 요약 (loss-ratio 런) | 기울기 | 잔차 SD | `ours_inter_ch` 3e-3 (n=4) 값 | 1.25e-7 / 그 값 − 1 | 1.25e-7 에서 예측 착지 | 실제/예측 | z |')
    w('|---|---|---|---|---|---|---|---|')
    for k_, lab in (('lam_pre', '혼합 증강 구간 평균'), ('lam_post', '해제 뒤 평균'), ('lam_last', '마지막 10에폭'), ('lam_mean', '전 구간 평균')):
        x = lf[k_]
        w(f'| {lab} | {x["slope"]:+.3f} | {x["resid_sd"]*100:.1f}% | {E(x["ours3e3_mean"])} | {x["nolr_vs_3e3"]*100:+.1f}% | '
          f'{K(x["pred_spk"])} | {(math.exp(x["resid"]) - 1) * 100:+.1f}% | {x["z"]:+.1f} |')
    w('')
    w('- 어느 한 구간의 요약도 착지를 정확히 맞히지 않는다: 해제 전 값으로 보면 실제가 예측보다 적고 (−8%), 해제 뒤 값으로 보면 많다 (+5~8%). '
      '착지는 두 구간 λ 의 가중 평균이 정한다 — 2c 에서 그 무게를 잰다.')
    w('')
    w('### 2c. 착지를 정하는 구간의 무게 α')
    w('')
    w('ln λ_혼합(α) = (1−α)·ln λ_해제전 + α·ln λ_해제뒤. α=0.356 이면 시간 비율대로 (해제 뒤 110 / 309 에폭), α>0.356 이면 후반이 더 무겁다. '
      '고정 λ 런은 λ_혼합 = λ 이므로 α 와 무관하다. 고정 λ 와 loss-ratio 가 같은 "λ_혼합 → 착지" 관계에 오르는 α 를 찾는다.')
    w('')
    w('| 쌍 | 설계 | 기준 n | 비교 n | α | 런별 α | α=0 / 0.356 / 1 에서 착지 오차 평균 (ln) |')
    w('|---|---|---|---|---|---|---|')
    ai, aa, a2 = al['inter'], al['intra'], al['l2']
    w(f'| inter (`ours_inter_ch` vs `- loss-ratio`) | loss-ratio 선 (국소) 에 고정 λ 1점 | {ai["n_ref"]} | {ai["n_arm"]} | **{ai["alpha"]:.2f}** | – | '
      + ' / '.join(f'{v:+.3f}' for v in ai['resid_at'].values()) + ' |')
    w(f'| intra (`ours_intra_ch` vs `- loss-ratio`) | loss-ratio 선에 고정 λ 3점 | {aa["n_ref"]} | {aa["n_arm"]} | **{aa["alpha"]:.2f}** | '
      + ', '.join(f'{e["alpha"]:.2f}' for e in aa['each']) + f' (t 대 0.356: p={fp(aa["t_vs_share"]["p"])}) | '
      + ' / '.join(f'{v:+.3f}' for v in aa['resid_at'].values()) + ' |')
    w(f'| plain L2 (`plain L2` vs `+ loss-ratio`) | 고정 λ 용량-반응선 (n={a2["n_ref"]}) 에 loss-ratio 5점 | {a2["n_ref"]} | {a2["n_arm"]} | **{a2["alpha"]:.2f}** '
      f'[부트스트랩 {a2["boot_lo"]:.2f}, {a2["boot_hi"]:.2f}] | ' + ', '.join(f'{e["alpha"]:.2f}' for e in a2['each'])
      + f' (ρ 2e-3→1e-2; t 대 0.356: p={fp(a2["t_vs_share"]["p"])}) | – |')
    w('')
    w(f'- 세 쌍 모두 α = {min(alphas):.2f}~{max(alphas):.2f} 로 시간 비율 {al["time_share"]:.2f} 의 {min(alphas) / al["time_share"]:.1f}~{max(alphas) / al["time_share"]:.1f}배. '
      '해제 뒤 구간 (학습률도 cosine 으로 내려가는 중) 의 λ 가 착지를 더 많이 정한다.')
    w(f'- plain L2 의 런별 α 는 ρ 가 클수록 커진다 ({a2["each"][0]["alpha"]:.2f}→{a2["each"][-1]["alpha"]:.2f}). '
      + ', '.join(f'ρ {KN(t)}' for t in a2['out_range']) + f' 는 해제 뒤 λ 가 고정 λ 최소 {E(a2["lam_min_fixed"])} 아래라 외삽이고, '
      f'범위 안 {a2["n_in_range"]}점만 쓰면 α={a2["alpha_in_range"]:.2f}.')
    w('- 참고 — 단순 요약값 하나로 본 plain L2 착지 오차 (고정 λ 선 기준, ln, 평균 ±SD, n=5): '
      + '; '.join(f'{lab} {l2x["by"][k_]["mean_resid"]*100:+.1f}% ±{l2x["by"][k_]["sd_resid"]*100:.1f} (p={fp(l2x["by"][k_]["p"])})'
                  for k_, lab in (('lam_pre', '해제 전'), ('lam_mean', '전 구간'), ('lam_post', '해제 뒤'), ('lam_last', '마지막 10'))) + '.')
    w('')
    w('### 2d. 데이터셋 사이 눈금 — 같은 손잡이에서 C10 과 C100 의 스파이크 감소')
    w('')
    w(f'`ln(S / S_baseline) = a + b·ln(손잡이) + c·[C100]`, 손잡이 범위는 두 데이터셋이 겹치는 구간. c = 같은 손잡이에서 C100 의 S/S_baseline 이 C10 보다 큰 정도 '
      '(ln; c=+0.1 이면 baseline 대비 남는 스파이크가 약 10.5% 더 많다 = 덜 줄인다). '
      f'baseline (기하평균): C10 {K(xf["base_c10"]["spikes"])} (n={xf["base_c10"]["n"]}), C100 {K(xf["base_c100"]["spikes"])} (n={xf["base_c100"]["n"]}, 서버 '
      + '/'.join(xf['base_c100']['servers']) + ').')
    w('')
    w('| 방법 | λ 방식 | 손잡이 범위 | n (C10 / C100) | c (ln) | 95% CI | p |')
    w('|---|---|---|---|---|---|---|')
    for key, lab, how in (('ours_ch', 'ours_inter_ch', 'loss-ratio (ρ)'), ('prop', 'ours_intra_ch', 'loss-ratio (ρ)'),
                          ('l2', 'plain L2', '고정 λ'), ('bpsr', 'BPSR', '고정 λ')):
        x = xf['methods'][key]
        w(f'| {lab} | {how} | {E(x["range"][0], 1)}~{E(x["range"][1], 1)} | {x["n10"]} / {x["n100"]} | **{x["c"]["est"]:+.3f}** | '
          f'[{x["c"]["lo"]:+.3f}, {x["c"]["hi"]:+.3f}] | {fp(x["c"]["p"])} |')
    w('')
    w('같은 손잡이 묶음별 (스파이크 감소 % / val acc − 그 데이터셋 baseline, %p; n):')
    w('')
    w('| 방법 | 손잡이 | C10 | C100 |')
    w('|---|---|---|---|')
    for key, lab in (('ours_ch', 'ours_inter_ch'), ('prop', 'ours_intra_ch'), ('l2', 'plain L2'), ('bpsr', 'BPSR')):
        for g in xf['methods'][key]['groups']:
            c10 = f'{g["red10"]:+.1f}% / {g["acc10"]:+.2f} (n={g["n10"]})' if g['n10'] else '–'
            c100 = f'{g["red100"]:+.1f}% / {g["acc100"]:+.2f} (n={g["n100"]})' if g['n100'] else '–'
            w(f'| {lab} | {KN(g["knob"])} | {c10} | {c100} |')
    w('')
    w('기전 — 같은 ρ 에서 C100 / C10 비 (loss-ratio 팔, 해제 전·뒤 10에폭 평균):')
    w('')
    w('| 팔 ρ | n (C10 / C100) | L 해제 전 비 | L 해제 뒤 비 | R 비 | λ 해제 전 비 | λ 해제 뒤 비 |')
    w('|---|---|---|---|---|---|---|')
    for k_, x in xf['lam_ratio'].items():
        kk_, kn_ = k_.split(' ')
        w(f'| {LABEL[kk_]} {KN(kn_)} | {x["n10"]} / {x["n100"]} | {x["L_pre_ratio"]:.2f} | '
          f'{x["L_post_ratio"]:.2f} | {x["R_ratio"]:.2f} | {x["lam_pre_ratio"]:.2f} | {x["lam_post_ratio"]:.2f} |')
    w('')
    w('- C100 은 과제 손실이 약 2배 (100 클래스) 이고 R 은 약 0.9배라, loss-ratio 가 λ 를 2.1~2.4배로 올린다. 고정 λ 는 이 보정이 없어 같은 λ 가 C100 에서 상대적으로 약하다 '
      '(plain L2 c=+0.36 → 같은 λ 에서 C100 의 S/S_base 가 약 1.43배).')
    w('- 이 비교는 loss-ratio 방법 (inter·intra) 과 고정 λ 방법 (plain L2·BPSR) 사이다. 같은 벌점에서 λ 제어만 바꾼 C100 런 (`- loss-ratio` 나 `plain L2 + loss-ratio` 의 C100) 은 없다. '
      'λ ∝ L 이라는 기전은 로그로 확인했다.')
    w('- 이식되는 것은 스파이크 감소이지 정확도 손실이 아니다: ρ 3e-3 에서 C10 −0.01%p (n=4) / C100 −0.27%p (n=3). 무손실 ρ 는 데이터셋마다 다시 골라야 한다.')
    w('- 모델 사이에서는 같은 ρ 가 같은 감소를 주지 않는다: QKFormer C10 ρ 1.2e-2 는 약 −30% (결정 기록 10-11 인용, 재계산 안 함), 이 문서의 R19 C10 1.2e-2 (n=1) 는 −69.8%. '
      'QKFormer 는 PyTorch 이식이라 R 의 정의·T 등 구현 차가 섞여 있어 원인을 가를 수 없다.')
    w('')

    # ------------------------------------------------------------------ 3
    w('## 3. 기전')
    w('')
    w('### 3a. λ 궤적 (에폭 e 에 쓰인 λ, 묶음 평균)')
    w('')
    w('| 팔 | ρ | n | 1 | 5 | 30 | 100 | 199 | 200 | 201 | 309 | 199/5 | 201/199 | 309 λ 시드 CV % |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k_, lab in (('ours_ch', 'ours_inter_ch'), ('prop', 'ours_intra_ch'), ('l2_lr', 'plain L2 + loss-ratio')):
        for kn, x in s3['lam'][k_].items():
            a = x['at']
            w(f'| {lab} | {KN(kn)} | {x["n"]} | {E(a[1])} | {E(a[5])} | {E(a[30])} | {E(a[100])} | {E(a[199])} | {E(a[200])} | '
              f'{E(a[201])} | {E(a[309])} | {x["rise_5_199"]:.2f} | {x["drop_199_201"]:.2f} | '
              + (f'{x["last_cv"]:.1f}' if x['last_cv'] is not None else '–') + ' |')
    w('| - loss-ratio | λ 1.25e-7 | 1 | 1.25e-7 | 1.25e-7 | 1.25e-7 | 1.25e-7 | 1.25e-7 | 1.25e-7 | 1.25e-7 | 1.25e-7 | 1.00 | 1.00 | – |')
    w('')
    for k_, lab in (('ours_ch', 'ours_inter_ch'), ('prop', 'ours_intra_ch'), ('l2_lr', 'plain L2 + loss-ratio')):
        x = s3['lam_summary'][k_]
        dd = s3['LR_decomp'][k_]
        w(f'- {lab} (n={x["rise"]["n"]}): 에폭 5→199 λ {x["rise"]["mean_pct"]:+.0f}% (범위 {x["rise"]["min_pct"]:+.0f}~{x["rise"]["max_pct"]:+.0f}), '
          f'199→201 {x["drop"]["mean_pct"]:+.0f}% ({x["drop"]["min_pct"]:+.0f}~{x["drop"]["max_pct"]:+.0f}). 해제 전후 10에폭 평균 L '
          f'{dd["L_pre"]:.3f}→{dd["L_post"]:.3f} ({dd["dL_pct"]:+.0f}%), R {dd["dR_pct"]:+.0f}%.')
    w('- λ = ρ·L/R 이므로 ln λ 변화 = ln L 변화 − ln R 변화. 해제 때 L 이 40% 떨어지고 R 은 거의 그대로라 λ 가 L 만큼 떨어진다 (세 방법 모두 −37~−38%). '
      '혼합 증강 구간의 상승은 R 이 스파이크와 함께 줄어서다. 상승 폭은 ρ 가 클수록 크고 (스파이크를 더 줄이므로), `ours_intra_ch` 은 작다 (+13%).')
    w('- 에폭 0 은 loss-ratio 런에서 λ=0 (규제 없는 첫 에폭), 고정 λ 런은 첫 배치부터 λ 가 걸린다. 에폭 0~2 스파이크는 두 팔이 같다 (3c).')
    w('')
    w('### 3b. 고정 λ 의 실효 ρ (λ·R/L)')
    w('')
    w('| 팔 | λ | 1 | 5 | 30 | 100 | 199 | 200 | 201 | 250 | 309 | 해제 전 평균 | 해제 뒤 평균 | 뒤/전 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k_, lab in (('ch_nolr', '- loss-ratio'), ('abl_nolr', 'ours_intra_ch - loss-ratio')):
        for x in s3['rho_fixed'][k_]:
            a = x['at']
            w(f'| {lab} | {KN(x["knob"])} | ' + ' | '.join(E(a[e]) for e in (1, 5, 30, 100, 199, 200, 201, 250, 309))
              + f' | {E(x["pre"])} | {E(x["post"])} | **{x["post_over_pre"]:.2f}** |')
    rr = s3['rho_realized']
    w('')
    w(f'- 검산: loss-ratio 런의 실현 비 (에폭 0·200 제외 평균 / ρ) = `ours_inter_ch` {rr["ours_ch"]["mean_ratio"]:.4f} '
      f'({rr["ours_ch"]["min"]:.4f}~{rr["ours_ch"]["max"]:.4f}, n={rr["ours_ch"]["n"]}), `ours_intra_ch` {rr["prop"]["mean_ratio"]:.4f} (n={rr["prop"]["n"]}), '
      f'`plain L2 + loss-ratio` {rr["l2_lr"]["mean_ratio"]:.4f} (n={rr["l2_lr"]["n"]}) — 제어기가 설계대로 비를 고정한다.')
    w('- 고정 λ 의 실효 ρ 는 혼합 증강 구간 안에서도 움직인다: 에폭 5 무렵 가장 높고 (첫 에폭들에 L 이 먼저 떨어지고 R 은 오히려 오른다) '
      '이후 199 까지 R 이 스파이크와 함께 줄어 낮아진다. 10-10 보고 (0.0021~0.0026 → 0.0035, 1.6배) 는 "그 에폭에 쓰인 λ" 정의로 다시 계산해도 같다.')
    w('')
    w(f'### 3c. 스파이크 궤적 — `- loss-ratio` vs 같은 착지의 `ours_inter_ch` 3e-3 (n={s3["traj_vs_3e3"]["n"]})')
    w('')
    w('| 에폭 | ours 3e-3 평균 | ours 시드 CV % | - loss-ratio | 차 | z |')
    w('|---|---|---|---|---|---|')
    for x in tl:
        w(f'| {x["ep"]} | {K(x["ours_mean"])} | {x["ours_cv"]:.2f} | {K(x["nolr"])} | {x["ratio_pct"]:+.1f}% | '
          + (f'{x["z"]:+.1f}' if x['z'] is not None else '–') + ' |')
    w('')
    w('- 평가 모드 s_count (이미지당). z = (ln nolr − ln ours 평균) / (ours 시드 SD·√(1+1/4)). n=1 대 n=4 라 개별 에폭 z 는 거칠지만, 에폭 60~210 '
      '에서 (표의 에폭 모두) z>2.6 이고 309 에서 0.4 로 돌아온다.')
    w('- 해석: 혼합 증강 구간 `- loss-ratio` 의 실효 ρ (2.1~2.6e-3) 가 `ours_inter_ch` 3e-3 보다 낮아 덜 줄이고, 해제 뒤 실효 ρ 3.5e-3 으로 더 세게 줄여 따라잡는다.')
    w('')
    w('### 3d. 혼합 증강 해제 반응')
    w('')
    w('착지 스파이크 차를 보정: `d_mix = a + b·ln(착지 스파이크) + d·[비교 팔]`, d_mix = ln(마지막 10에폭 / 해제 직전 10에폭). d 는 ln 단위.')
    w('')
    w('| 비교 (비교 팔 − 기준 팔) | 기준 팔 n | 기준 평균 | 기준 SD (ln) | 비교 팔 n | 비교 팔 값 | d (ln) | 95% CI | p |')
    w('|---|---|---|---|---|---|---|---|---|')
    labs = {'inter': '`- loss-ratio` − `ours_inter_ch`', 'intra': '`ours_intra_ch - loss-ratio` − `ours_intra_ch`',
            'l2': '`plain L2` (고정) − `plain L2 + loss-ratio`'}
    for k_ in ('inter', 'intra', 'l2'):
        x = dm[k_]
        d = x['ancova']['d']
        vals = x['arm_vals_pct']
        vtxt = ', '.join(f'{v:+.1f}%' for v in vals) if len(vals) <= 3 else f'평균 {np.mean(vals):+.1f}% (범위 {min(vals):+.1f}~{max(vals):+.1f})'
        w(f'| {labs[k_]} | {x["n_ref"]} | {x["ref_mean_pct"]:+.1f}% | {x["ref_sd"]:.3f} | {x["n_arm"]} | {vtxt} | '
          f'**{d["est"]:+.3f}** | [{d["lo"]:+.3f}, {d["hi"]:+.3f}] | {fp(d["p"])} |')
    w(f'| (참고) baseline | {dm["base"]["n"]} | {dm["base"]["mean_pct"]:+.1f}% | {dm["base"]["sd"]:.3f} | | | | | |')
    w(f'| (참고) BPSR (고정 λ) | {dm["bpsr"]["n"]} | {dm["bpsr"]["mean_pct"]:+.1f}% | {dm["bpsr"]["sd"]:.3f} | | | | | |')
    w('')
    w('묶음별 해제 반응 (손잡이: n, 착지 스파이크, 해제 반응):')
    w('')
    w('| 팔 | 묶음 |')
    w('|---|---|')
    for k_, lab in (('ours_ch', 'ours_inter_ch'), ('ch_nolr', '- loss-ratio'), ('prop', 'ours_intra_ch'),
                    ('abl_nolr', 'ours_intra_ch - loss-ratio'), ('l2_lr', 'plain L2 + loss-ratio'), ('l2', 'plain L2'), ('bpsr', 'BPSR')):
        w(f'| {lab} | ' + '; '.join(f'{KN(x["knob"])} (n={x["n"]}, {x["spikes"]/1e3:.0f}K, {x["d_mix_pct"]:+.1f}%)'
                                     for x in s3['dmix_groups'][k_]) + ' |')
    w('')
    w(f'- 규제가 없어도 (baseline) 해제 뒤 스파이크는 {dm["base"]["mean_pct"]:+.1f}% 준다. loss-ratio 런 묶음은 λ 가 38% 떨어져 {max(dg_lr):+.1f}~{min(dg_lr):+.1f}%, '
      f'고정 λ 런 묶음은 실효 ρ 가 올라 {max(dg_fx):+.1f}~{min(dg_fx):+.1f}% (착지 스파이크를 보정해도 차가 남는다).')
    w(f'- inter 행은 n=1 이라 p 는 `ours_inter_ch` 잔차 SD ({dm["inter"]["resid_sd"]:.3f}) 를 그 런에도 적용한 값이다. intra (n=3) 와 plain L2 (n={dm["l2"]["n_arm"]}/{dm["l2"]["n_ref"]}) 는 방향이 같고 각자 유의하다.')
    w('')
    w('### 3e. 착지 시드 변동')
    w('')
    w('같은 손잡이 (ρ 또는 λ) n≥2 묶음. ln 스파이크·val acc 를 묶음 고정효과 (+두 서버가 섞인 묶음이 있으면 서버 항) 로 적합한 잔차 SD.')
    w('')
    w('| 팔 | λ 방식 | 묶음 (손잡이: n) | 서버 138/23 | ln 스파이크 잔차 SD | 자유도 | 서버 항 없이 | n≥3 묶음만 | val acc 잔차 SD (%p, 자유도) |')
    w('|---|---|---|---|---|---|---|---|---|')
    for k_, lab, how in (('ours_ch', 'ours_inter_ch', 'loss-ratio'), ('prop', 'ours_intra_ch', 'loss-ratio'),
                         ('l2', 'plain L2', '고정 λ'), ('bpsr', 'BPSR', '고정 λ'), ('base', 'baseline', '규제 없음')):
        x = cvp[k_]
        x3 = s3['cv']['pooled_n3'].get(k_)
        xa = s3['cv']['acc_sd'][k_]
        w(f'| {lab} | {how} | ' + ', '.join(f'{KN(g)}: {m}' for g, m in sorted(x['groups'].items(), key=lambda t: knob_val(t[0]) if t[0] != '-' else 0))
          + f' | {x["servers"]["138"]}/{x["servers"]["23"]} | **{x["sd"]*100:.2f}%** | {x["df"]} | {x["sd_plain"]*100:.2f}% ({x["df_plain"]}) | '
          + (f'{x3["sd"]*100:.2f}% ({x3["df"]})' if x3 else '–') + f' | {xa["sd"]:.3f} ({xa["df"]}) |')
    w('')
    w('| F 검정 (ln 스파이크 분산비) | F | 자유도 | p |')
    w('|---|---|---|---|')
    for k_, lab in (('ours_vs_bpsr', 'ours_inter_ch / BPSR'), ('ours_vs_l2', 'ours_inter_ch / plain L2'),
                    ('prop_vs_bpsr', 'ours_intra_ch / BPSR'), ('prop_vs_l2', 'ours_intra_ch / plain L2'),
                    ('lr_vs_fixed_pooled', 'loss-ratio 두 방법 합동 / 고정 λ 두 방법 합동'),
                    ('ours_vs_base', 'ours_inter_ch / baseline'), ('l2_vs_base', 'plain L2 / baseline'), ('bpsr_vs_base', 'BPSR / baseline')):
        x = F[k_]
        w(f'| {lab} | {x["F"]:.2f} | {x["df1"]}, {x["df2"]} | {fp(x["p"])} |')
    w('')
    lr_ = s3['cv']['line_resid']
    w('같은 손잡이 반복이 없는 `plain L2 + loss-ratio` 를 넣으려고 ln S ~ ln 손잡이 직선 잔차 SD 도 본다 (직선이 안 맞는 몫이 섞임):')
    w('')
    w('| 팔 | n | 기울기 | 직선 잔차 SD | 자유도 |')
    w('|---|---|---|---|---|')
    for k_, lab in (('ours_ch', 'ours_inter_ch (전체 ρ, 휘어 있음)'), ('prop', 'ours_intra_ch'), ('l2_lr', 'plain L2 + loss-ratio'),
                    ('bpsr', 'BPSR'), ('l2', 'plain L2')):
        x = lr_[k_]
        w(f'| {lab} | {x["n"]} | {x["slope"]:+.3f} | {x["resid_sd"]*100:.2f}% | {x["df"]} |')
    w('')
    w(f'- 10-10 보고 (inter 2.17% 자유도 17 vs BPSR 1.10% 자유도 8, p=0.057) 의 갱신: 같은 자료로 서버 항을 넣으면 inter {cvp["ours_ch"]["sd"]*100:.2f}% vs BPSR '
      f'{cvp["bpsr"]["sd"]*100:.2f}%, p={fp(F["ours_vs_bpsr"]["p"])}. 그러나 고정 λ 인 `plain L2` ({cvp["l2"]["sd"]*100:.2f}%) 와는 구별되지 않고 (p={fp(F["ours_vs_l2"]["p"])}), '
      f'규제 없는 baseline 도 {cvp["base"]["sd"]*100:.2f}% 다. BPSR 이 유독 좁은 것이지 loss-ratio 가 넓힌다고 할 근거는 약하다 (합동 p={fp(fpool["p"])}).')
    w(f'- `plain L2 + loss-ratio` 의 직선 잔차 ({lr_["l2_lr"]["resid_sd"]*100:.2f}%, 자유도 {lr_["l2_lr"]["df"]}) 는 고정 λ `plain L2` ({lr_["l2"]["resid_sd"]*100:.2f}%, 자유도 {lr_["l2"]["df"]}) 보다 작다 — '
      '같은 벌점 안에서 loss-ratio 가 산포를 키운다는 신호는 없다 (자유도 3 이라 판단 불가).')
    w('')
    w('### 3f. loss-ratio 되먹임의 부호')
    w('')
    w('같은 ρ 묶음 안에서 시드별 ln λ 를 ln 착지 스파이크에 회귀 (묶음 고정효과).')
    w('')
    w('| 팔 | 묶음 수 | n | d ln λ_해제뒤 / d ln S | 95% CI | p | d ln λ_마지막10 / d ln S | d ln R_마지막 / d ln S |')
    w('|---|---|---|---|---|---|---|---|')
    for k_, lab in (('ours_ch', 'ours_inter_ch'), ('prop', 'ours_intra_ch')):
        x = fb[k_]
        a, b, c = x['lam_post'], x['lam_last'], x['R_end']
        w(f'| {lab} | {a["groups"]} | {a["n"]} | **{a["slope"]:+.2f}** ±{a["se"]:.2f} | [{a["slope"] - tcrit(a["df"]) * a["se"]:+.2f}, '
          f'{a["slope"] + tcrit(a["df"]) * a["se"]:+.2f}] | {fp(a["p"])} | {b["slope"]:+.2f} ±{b["se"]:.2f} (p={fp(b["p"])}) | {c["slope"]:+.2f} ±{c["se"]:.2f} |')
    w('')
    ao, ap = fb['ours_ch'], fb['prop']
    w('- 음의 기울기 = 같은 ρ 에서 더 쏘는 시드일수록 R 이 커서 λ = ρL/R 가 작아진다 → 그 시드는 덜 눌려 더 쏜 채로 남는다 (시드 차이를 키우는 고리). 고정 λ 에는 이 고리가 없다.')
    w(f'- 크기 (선형 근사): 고정 λ 에서 시드 섭동 ε 가 ln S 를 ε 만큼 옮긴다면 loss-ratio 아래서는 ε / (1 + k·s) (k = −d ln S / d ln λ, s = 위 기울기). '
      f'`ours_inter_ch`: k = {ao["k"]:.3f} (국소 ln S ~ ln λ_해제뒤), s = {ao["lam_post"]["slope"]:+.2f} → **×{ao["amp"]:.2f}** '
      f'(s 의 95% 끝에서 ×{fmt_amp(ao["amp_range"][0])}~×{fmt_amp(ao["amp_range"][1])}). `ours_intra_ch`: s 가 유의하지 않음 (p={fp(ap["lam_post"]["p"])}), ×{fmt_amp(ap["amp"])}.')
    w(f'- 예측 배율 ×{ao["amp"]:.2f} 는 관측 SD 비 (loss-ratio 합동 / 고정 λ 합동 {fpool["sd_lr"] / fpool["sd_fixed"]:.2f}, p={fp(fpool["p"])}) 와 방향이 같고 크기는 작다. '
      '요소 단독의 효과는 `- loss-ratio` 반복 런으로만 확인된다.')
    w('')
    w('### 3g. 같은 스파이크에서 가중·분포 (학습 모드, 마지막 에폭, reg_detail / reg_neuron_detail)')
    w('')
    w('`ours_inter_ch` 곡선 (200~420K) 의 직선 `지표 = a + b·ln(spikes)` 예측과 `- loss-ratio` 값의 차. z = 차 / 새 런 하나의 예측 SE.')
    w('')
    w('| 지표 | 곡선 n | 곡선 예측 | - loss-ratio | 차 | z |')
    w('|---|---|---|---|---|---|')
    MLAB = {'w_mean': 'w 평균 (뉴런 가중)', 'lnR_rd': 'ln R', 'dead_pct': '침묵 뉴런 % (배치 단위, 학습 모드)',
            'gini_sw': '지니 (층 스파이크 가중)', 'top10_sw': '상위 10% 몫 %',
            **{f'share_{s}': f'{s} 스파이크 몫 %' for s in STAGES}, **{f'dead_{s}': f'{s} 침묵 %' for s in STAGES},
            **{f'w_{s}': f'{s} w' for s in STAGES}}
    for mt, x in sti.items():
        e = x['each'][0]
        dg = 3 if mt in ('w_mean', 'lnR_rd', 'gini_sw') or mt.startswith('w_') else 2
        w(f'| {MLAB.get(mt, mt)} | {x["n_ref"]} | {e["pred"]:.{dg}f} | {e["value"]:.{dg}f} | {e["delta"]:+.{dg}f} | {e["z"]:+.1f} |')
    w('')
    w('교차 확인 (ANCOVA 오프셋 ± SE; intra 는 끈 팔 − 켠 팔, plain L2 는 켠 팔 − 끈 팔):')
    w('')
    w('| 지표 | intra n (켠/끈) | intra 오프셋 | p | plain L2 n (끈/켠) | plain L2 오프셋 | p |')
    w('|---|---|---|---|---|---|---|')
    for mt in ['w_mean', 'lnR_rd', 'dead_pct', 'gini_sw', 'top10_sw'] + [f'share_{s}' for s in STAGES] + [f'w_{s}' for s in STAGES]:
        a, b = sii.get(mt), sll.get(mt)
        dg = 4 if mt in ('w_mean', 'lnR_rd', 'gini_sw') or mt.startswith('w_') else 2
        ca = (f'{a["n_ref"]}/{a["n_arm"]} | {a["ancova"]["est"]:+.{dg}f} ±{a["ancova"]["se"]:.{dg}f} | {fp(a["ancova"]["p"])}'
              if a and 'ancova' in a else '– | – | –')
        cb = (f'{b["n_ref"]}/{b["n_arm"]} | {b["ancova"]["est"]:+.{dg}f} ±{b["ancova"]["se"]:.{dg}f} | {fp(b["ancova"]["p"])}'
              if b and 'ancova' in b else '– | – | –')
        w(f'| {MLAB.get(mt, mt)} | {ca} | {cb} |')
    w('')
    w('- intra 개별 z (지니): ' + ', '.join(f'{KN(e["tag"].replace("r19c10-abl_nolr-", "λ "))} {e["z"]:+.1f}' for e in sii['gini_sw']['each'])
      + '; (상위 10%): ' + ', '.join(f'{e["z"]:+.1f}' for e in sii['top10_sw']['each']) + '. 5e-7 은 곡선 범위 밖 외삽이지만 범위 안 두 점도 같은 방향.')
    w('- 읽는 법: 고정 λ 는 같은 스파이크에서 발화를 덜 몰리게 한다 (intra 유의, inter 같은 방향 n=1, plain L2 는 없음 — plain L2 에는 순위 가중이 없다). '
      f'가중 w 와 R 의 차는 작다 (intra w {sii["w_mean"]["ancova"]["est"]:+.4f}, ln R {sii["lnR_rd"]["ancova"]["est"]:+.4f}). 경로 차 (고정 λ 는 감소가 해제 뒤로 몰림, 3c·3d) 가 분포 차로 남는다는 가설과 맞지만 검증하지 않았다.')
    w('- `plain L2` 계열은 w 가 없고 고정 λ 런은 R 을 기록하지 않아 칸이 비었다. 채널 단위 지표 (죽은 채널 수, 이미지당 활성 뉴런, MACs) 는 npz 가 필요한데 '
      '`- loss-ratio`·`abl_nolr`·`l2_lr` 의 npz 는 없다 (5절).')
    w('')

    # ------------------------------------------------------------------ 4
    w('## 4. 학습 시간 (에폭당 초; 런별 중앙값 (에폭 1~309) → 묶음 중앙값)')
    w('')
    w('| 팔 | 서버 | n | 중앙값 | 최소~최대 |')
    w('|---|---|---|---|---|')
    for k_, lab in (('base', 'baseline'), ('ours_ch', 'ours_inter_ch'), ('ch_nolr', '- loss-ratio'), ('prop', 'ours_intra_ch'),
                    ('abl_nolr', 'ours_intra_ch - loss-ratio'), ('l2_lr', 'plain L2 + loss-ratio'), ('l2', 'plain L2'), ('bpsr', 'BPSR')):
        for sv in ('138', '23'):
            x = s4.get(f'{k_}@{sv}')
            if x:
                w(f'| {lab} | {sv} | {x["n"]} | {x["median"]:.0f} | {x["min"]:.0f}~{x["max"]:.0f} |')
    w('')
    x = s4['nolr_vs_ours23']
    w(f'- `- loss-ratio` (23) − `ours_inter_ch` (23, n={x["n_grp"]}): {x["diff"]:+.1f}s [{x["lo"]:+.1f}, {x["hi"]:+.1f}] (단일 런 vs 묶음 SD).')
    x = s4['l2lr_vs_l2_138']
    w(f'- `plain L2 + loss-ratio` − `plain L2` (둘 다 138): {x["diff"]:+.1f}s [{x["lo"]:+.1f}, {x["hi"]:+.1f}], p={fp(x["p"])} (n={x["na"]}/{x["nb"]}). '
      '`plain L2` 138 런 중 359~366s 인 7런은 모두 09-14 이전·당일 투입분이라 시기 (동시 실행 수·코드 판) 차로 본다 — 제어기가 학습을 빠르게 할 경로는 없다.')
    x = s4['anolr_vs_prop_138']
    if x:
        w(f'- `ours_intra_ch - loss-ratio` − `ours_intra_ch` (138): {x["diff"]:+.1f}s [{x["lo"]:+.1f}, {x["hi"]:+.1f}], p={fp(x["p"])} (n={x["na"]}/{x["nb"]}).')
    w(f'- 제어기 계산은 에폭 끝 스칼라 한 번 (`proc.py` 1240~1296). R 을 모으는 `list_sc_loss` 는 `reg_spike_log_detail=True` 인 모든 팔이 이미 모은다. '
      f'baseline 대비 학습 비용 (23: {s4["base@23"]["median"]:.0f} → {s4["ours_ch@23"]["median"]:.0f}s, '
      f'{(s4["ours_ch@23"]["median"] / s4["base@23"]["median"] - 1) * 100:+.0f}%) 은 규제 항 자체의 몫이고 loss-ratio 와 무관하다.')
    w('')

    # ------------------------------------------------------------------ 5
    w('## 5. 판정 가능 / 불가 / 필요한 측정')
    w('')
    w('**판정 가능**')
    w('')
    w(f'- loss-ratio 는 λ 를 L 에 묶는다: 혼합 증강 구간 +{lam_o["rise"]["mean_pct"]:.0f}% (n={lam_o["rise"]["n"]}, 전 런 같은 방향), 해제 때 {lam_o["drop"]["mean_pct"]:+.0f}% '
      f'(원인 L {dLo["dL_pct"]:+.0f}%). 실현 비는 ρ 의 ±0.1% 안.')
    w(f'- 고정 λ 의 실효 ρ 는 해제 뒤 ×{fx["post_over_pre"]:.2f} (inter n=1), ×' + '/'.join(f'{x["post_over_pre"]:.2f}' for x in fxa)
      + ' (intra n=3). 정의상 L 하락에서 나오므로 n 이 작아도 결정적이다.')
    w(f'- 고정 λ 는 해제 뒤에 더 줄인다: plain L2 쌍 (n={dm["l2"]["n_arm"]}/{dm["l2"]["n_ref"]}) p={fp(dm["l2"]["ancova"]["d"]["p"])}, intra (n=3/{dm["intra"]["n_ref"]}) '
      f'p={fp(dm["intra"]["ancova"]["d"]["p"])}, inter (n=1) 같은 방향.')
    w(f'- 착지는 해제 뒤 λ 에 시간 비율 ({al["time_share"]:.2f}) 보다 큰 무게 (α {min(alphas):.2f}~{max(alphas):.2f}) 를 둔다: intra p={fp(al["intra"]["t_vs_share"]["p"])} (n=3), plain L2 p={fp(al["l2"]["t_vs_share"]["p"])} (n=5).')
    w(f'- 같은 ρ 는 C10·C100 에서 같은 스파이크 감소를 준다 (inter c={xo["c"]["est"]:+.3f}, 95% ±{(xo["c"]["hi"] - xo["c"]["lo"]) / 2:.3f}); 같은 λ 는 아니다 '
      f'(plain L2 c={xl["c"]["est"]:+.2f}, BPSR {xb["c"]["est"]:+.2f}). 단 방법 간 비교.')
    w(f'- 학습 비용 차이 없음 (inter 차 {s4["nolr_vs_ours23"]["diff"]:+.1f}s).')
    w('')
    w('**판정 불가**')
    w('')
    w(f'- 같은 스파이크 정확도: `- loss-ratio` n=1, 95% 구간 [{ls["lo"]:+.2f}, {ls["hi"]:+.2f}]%p. "기여한다/안 한다" 둘 다 말할 수 없다. '
      f'n=3 이 돼도 최소 검출 차 {s1["mde"]["by_n"][3]:.2f}%p, n=4 {s1["mde"]["by_n"][4]:.2f}%p — 0.1%p 대 차이는 조건당 n=4 상한으로 못 본다.')
    w('- `- loss-ratio` 자체의 착지 시드 변동 (요소 단독 재현성): n=1. 3e·3f 는 벌점 모양이 다른 방법 사이 비교와 되먹임 부호로 본 간접 근거다.')
    w('- 같은 스파이크 분포 차 (지니·상위 10%) 가 inter 에서도 있는지: n=1 (z −2.2·−1.6). intra 에서만 유의.')
    w('- 채널 단위 구조 (죽은 채널·활성 뉴런): npz 없음.')
    w('- 같은 벌점에서 λ 제어만 바꿨을 때의 데이터셋 이식성: C100 고정 λ inter 런 없음.')
    w('')
    w('**필요한 측정**')
    w('')
    w('| 측정 | 답하는 질문 | GPU | 비용 | 상태 |')
    w('|---|---|---|---|---|')
    w(f'| `- loss-ratio` λ 1.25e-7 s2·s3 (R19 C10, 310ep) | 요소 단독 착지 시드 변동, 같은 스파이크 정확도 (n=3, MDE ≈{s1["mde"]["by_n"][3]:.2f}%p), '
      'inter 에서 지니·상위 10% 차 | 예, 런당 1장 | 런당 약 30시간 (348s×310) | 큐 hold (final_step 판정 대기) — 재투입은 3등급 (사용자 승인) |')
    w('| `- loss-ratio` s1 npz 추출 (sejong 체크포인트 ep-0310 있음) — `ours_inter_ch` 3e-3 npz 는 있음 | 같은 스파이크 죽은 채널·활성 뉴런 차 | 예, 1장 | 약 17분 | 미실행 |')
    w('| `- loss-ratio` R19 C100 1~2런 (λ = C10 값, 그리고 ×2.2) | 같은 벌점에서 데이터셋 이식성 (2d 의 방법 간 비교를 요소 단독으로) | 예 | 런당 약 30시간 | 제안만 |')
    w('| (GPU 불필요) 위 런 완주 뒤 이 스크립트 `--refetch` | 전 표 갱신 | 아니오 | 약 2분 | – |')
    w('')

    # ------------------------------------------------------------------ 6
    w('## 6. 10-10 결과 중 갱신·정정')
    w('')
    w('| 항목 | 10-10 (ablation_mech_261010.md) | 10-14 (이 문서) | 상태 |')
    w('|---|---|---|---|')
    w(f'| 같은 스파이크 정확도 | +0.01 ±0.10 (inter 곡선 n=17, 새 런 예측 SE) | {ls["est"]:+.3f} [{ls["lo"]:+.2f}, {ls["hi"]:+.2f}] (ANCOVA, ours n={res["ours_band_n"]}, 선형+서버) | 같은 결론, 방법 통일 |')
    w(f'| 고정 λ 실효 ρ | 0.0021~0.0026 → 0.0035 (1.6배) | 해제 전 {E(fx["pre"])} → 뒤 {E(fx["post"])} (×{fx["post_over_pre"]:.2f}); intra 3런 ×' + '/'.join(f'{x["post_over_pre"]:.2f}' for x in fxa) + ' 추가 | 확인·보강 |')
    w(f'| inter λ 궤적 | 3e-3 (n=4): 5ep 1.43e-7 → 200 직전 1.9e-7 (+34%) → 1.08e-7 | 전 ρ n={lam_o["rise"]["n"]}: +{lam_o["rise"]["mean_pct"]:.0f}% (ρ 클수록 큼), {lam_o["drop"]["mean_pct"]:+.0f}%. 3e-3 은 같은 값 | 확인 (n 확대) |')
    w(f'| "고정 λ 도 같은 곳에 착지" | 착지 342,306 = inter 3e-3 342,075 | 같은 착지, **다른 경로** (199 에폭 +{t199["ratio_pct"]:.0f}%, z={t199["z"]:.1f} → 309 +{t309["ratio_pct"]:.1f}%). 착지는 해제 뒤 λ 에 무게 α={min(alphas):.2f}~{max(alphas):.2f} | **해석 추가** |')
    w(f'| 착지 CV inter vs BPSR | 2.17% (df 17) vs 1.10% (df 8), p=0.057 → "loss-ratio 재현성 이득 없음" | 서버 항 포함 {cvp["ours_ch"]["sd"]*100:.2f}% vs {cvp["bpsr"]["sd"]*100:.2f}%, p={fp(F["ours_vs_bpsr"]["p"])}; '
      f'그러나 고정 λ plain L2 {cvp["l2"]["sd"]*100:.2f}% 와 구별 안 됨 (p={fp(F["ours_vs_l2"]["p"])}), baseline {cvp["base"]["sd"]*100:.2f}% | **정정**: "loss-ratio 가 산포를 2배 키운다" 로 읽으면 안 된다. "줄이지 않는다" 까지 |')
    w(f'| ln S–ln ρ 직선 (사용성 근거) | 기울기 −0.298, 잔차 3.2%, n=17 | ρ 1.5e-3~5e-3 n={rf["n"]}: {rf["slope"]:+.3f}, {rf["resid_sd"]*100:.1f}%. 전체 범위는 휘어 있음 (2차 p={fp(rf["full_quad_c_p"])}) | 범위 명시 |')
    w(f'| loss-ratio 의 실익 = "ρ 하나로 C10·C100·다른 모델에서 같은 눈금" | 제안 (근거 없음) | C10↔C100 스파이크 눈금은 성립 (inter c={xo["c"]["est"]:+.3f}), 고정 λ 는 불성립 (+0.33~0.36). 정확도 경계·모델 간은 불성립 | **근거 추가·범위 축소** |')
    w('')

    # ------------------------------------------------------------------ 한계
    w('## [LIMITATION]')
    w('')
    w('1. 주 비교 팔 `- loss-ratio` 는 **n=1** 이다. 정확도·시드 변동·구조 지표 모두 이 런 하나가 `ours_inter_ch` 시드 산포 안에 드는지만 말한다. '
      '"loss-ratio 를 빼도 같다" 는 결론이 아니다 — 95% 구간 밖의 큰 차이만 배제된다. n=1 의 p (3d inter 행 등) 는 그 런의 시드 산포를 기준 팔과 같다고 가정한다.')
    w('2. 교차 확인 팔은 손잡이당 n=1 이다 (`abl_nolr` 3개, `l2_lr` 5개). 여러 손잡이를 묶어 n 을 만든 것이라 "손잡이에 따라 효과가 다르다" 는 가능성 (plain L2 α 가 ρ 와 함께 커짐) 을 배제하지 못한다. '
      '모두 138 서버다.')
    w('3. 데이터셋 이식성 (2d) 과 착지 시드 변동 (3e·3f) 은 벌점 모양이 다른 방법 사이의 비교다. loss-ratio 단독 효과는 λ ∝ L 기전 (로그로 확인) 과 되먹임 부호로 뒷받침할 뿐, '
      '같은 벌점에서 λ 제어만 바꾼 C100 런·반복 런이 없어 직접 검증하지 않았다. 되먹임 배율은 선형 근사다. C100 값은 이 스크립트가 로그에서 다시 읽은 best 에폭 값 (원장과 일치) 이다.')
    w('4. "같은 λ" 비교의 λ 는 loss-ratio 가 정하는 내생 변수다. ln S ~ ln λ 직선은 ρ 를 바꿔 얻은 대응 관계이지 λ 를 직접 조작한 인과가 아니다 (2b·2c 의 inter·intra). '
      'plain L2 쌍만 고정 λ 를 직접 조작한 용량-반응선이다. α 의 2구간 혼합 모형은 단순화이고 구간 안 λ 변화는 평균으로 뭉갰다.')
    w('5. 혼합 증강 해제와 학습률 cosine 감쇠 후반이 겹친다 (해제 = 에폭 200/310). "해제" 와 "후반 학습률" 효과는 이 설계로 가를 수 없다 — 두 팔에 같게 작용한다는 가정만 쓴다. '
      'α 가 큰 것도 "최근 압력이 남는다" 와 "학습률이 낮아 늦은 변화가 지워지지 않는다" 를 구별하지 못한다.')
    w('6. 구조 지표는 학습 모드 배치 평균 (reg_neuron_detail) 이고 평가 모드 npz 와 정의가 다르다. 침묵 뉴런 % 는 배치 단위라 이미지·데이터셋 단위 침묵과 다르다. '
      'inter 의 17개 지표 z 는 다중 비교 보정 없이 보고했다.')
    w('7. best 에폭 기준은 310 에폭 최대값 선택이라 시드 잡음을 위로 끌어올린다. spikes 는 s_count 원값이고 npz 총합과의 11% 불일치 (09-15 보류) 는 해소되지 않았다.')
    w('8. 학습 시간은 서버의 동시 실행 런 수에 좌우된다. 혼잡도를 통제하지 않았다.')
    open(OUT_MD, 'w').write('\n'.join(L) + '\n')


def fmt_amp(v):
    return '∞' if v is None else f'{v:.2f}'


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if k not in ('cov', 'beta', 'rd', 'rnd', 'rows', 'conf')}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refetch', action='store_true')
    a = ap.parse_args()
    runs, dup, inv = build(a.refetch)
    res, G = analyze(runs)
    json.dump(jsonable({'generated': '2026-10-14', 'runs': runs, 'results': res}), open(OUT_JSON, 'w'),
              ensure_ascii=False, indent=1)
    write_md(runs, res, G, dup)
    print(f'wrote {OUT_MD}\nwrote {OUT_JSON}')


if __name__ == '__main__':
    main()
