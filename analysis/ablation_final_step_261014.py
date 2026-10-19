#!/usr/bin/env python
"""ours_inter_ch 어블레이션 — final_step 요소 (R19 CIFAR-10, T=4) — 26-10-14.

사용자 요청 (10-14): "어블레이션 각 방법 분석도 해줄 수 있어?" 중 final_step 몫.
팔: ch_nofinal (= '- final_step', ours_inter_ch 에서 reg_spike_final_step 만 끔).
교차 확인: abl_nofinal (ours_intra_ch = prop 에서 끔), lw1_nofinal (ours_layer_w1 에서 끔 — 런 없음).

묻는 것
  1. 같은 스파이크 정확도 오프셋 (nofinal_vs_inter_261014.py 와 같은 ANCOVA: ours_ch 200~420K + 선형 + 서버)
  2. 같은 ρ / 같은 λ 에서 스파이크 — final_step 이 '규제 세기 눈금' 을 얼마나 바꾸나
  3. 기전 — λ·R 비 분해, 스텝별 노름으로 본 기울기 배율, 학습 모드 지표 (w, 침묵, 지니, 상위 10%),
     npz 채널 구조, 시간축, 부분 누적 순위
  4. 학습 시간 (에폭당 초, 같은 시각 짝 비교)

수치 정의 (원장·논문과 같음)
  val acc = 310 에폭 중 최고 val_acc (같은 값이면 첫 에폭), spikes = 그 에폭 s_count (eval, 이미지당 원값)
  train loss (규제 포함 'loss') / train acc / val loss = 그 에폭 값
  λ_end, R_end, L_end = 310 에폭 줄의 adp_lambda, reg_R, task_loss_clean (λ_end 는 다음 에폭용 = ρ·L_end/R_end)
  학습 모드 지표 = reg_detail.csv / reg_neuron_detail.csv 마지막 에폭 (배치 평균)

입력 (전부 읽기 전용)
  138: _paper/<tag>/train.log, config_sweep.py; /media/hdd1/kyccj/EIP/paper/<tag>/{reg_detail,reg_neuron_detail}.csv, host.txt
  23 : ssh -p 23456 kyccj@161.122.23.23 에서 아래 REMOTE_PY 를 돌려 같은 항목을 JSON 으로 받는다 (캐시 --cache23)
  원장: scratchpad/led1007b/inventory.json (교차검증 + S30/S1 배제 판정)
  npz: _npz/, scratchpad/deep1005/npz/ (뉴런별 총합), _npz_t/ (스텝별)

쓰는 법
  ~/anaconda3/envs/venv_1/bin/python analysis/ablation_final_step_261014.py [--refetch]
산출: analysis/ablation_final_step_261014.md (수치 원본 JSON 은 scratchpad/afs1014/ 에)
"""
import argparse, glob, gzip, json, math, os, re, subprocess, sys
from datetime import datetime
import numpy as np
from scipy import stats

ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
SCR = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
HDD = '/media/hdd1/kyccj/EIP/paper'
DEF_INV = SCR + '/led1007b/inventory.json'
DEF_CACHE = SCR + '/afs1014/sejong.json.gz'
OUT_JSON = SCR + '/afs1014/ablation_final_step_261014.json'
OUT_MD = ROOT + '/analysis/ablation_final_step_261014.md'
NPZ_DIRS = [ROOT + '/_npz', SCR + '/deep1005/npz', ROOT + '/_npz_t']
NPZ_T = ROOT + '/_npz_t'
EPOCHS = 310
PATTERNS = ['r19c10-base---s1[1-6]', 'r19c10-ours_ch-[0-9]*', 'r19c10-ch_nofinal-*', 'r19c10-ch_novmem-*',
            'r19c10-ch_nolr-*', 'r19c10-ch_noinv-*', 'r19c10-prop-[0-9]*', 'r19c10-abl_nofinal-*', 'r19c10-lw1_*']
NAME_RE = re.compile(r'^r19c10-(base|ours_ch|ch_nofinal|ch_novmem|ch_nolr|ch_noinv|prop|abl_nofinal|lw1_[a-z]+)-(.*)-s(\d+)$')
ARM = {'base': 'baseline', 'ours_ch': 'ours_inter_ch', 'ch_nofinal': '- final_step', 'ch_novmem': '- vmem',
       'ch_nolr': '- loss-ratio', 'ch_noinv': '- 1- inversion', 'prop': 'ours_intra_ch',
       'abl_nofinal': 'ours_intra_ch - final_step'}
BAND = (200e3, 420e3)                    # ours_ch 곡선 범위 (nofinal_vs_inter_261014 와 같음)
NOF_RHO = '2.2e-3'
STAGES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1']
LAYERS = ['conv1_conv_n'] + [f'conv2_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')] + \
         [f'conv3_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')] + \
         [f'conv4_block{b}_{s}_n' for b in (1, 2) for s in ('conv1', 'out')] + ['fc1_n']
NEU = {}
for _l in LAYERS:
    if _l.startswith(('conv1', 'conv2')):
        NEU[_l] = 32 * 32 * 128
    elif _l.startswith('conv3'):
        NEU[_l] = 16 * 16 * 256
    elif _l.startswith('conv4'):
        NEU[_l] = 8 * 8 * 512
    else:
        NEU[_l] = 256
TOTN = sum(NEU.values())

NUM = re.compile(r'([A-Za-z_][\w\-]*): ([-+0-9.eE]+|nan|inf)')
SEC = re.compile(r'- (\d+)s (\d+)(?:ms|us|s)/step')
ASSIGN = re.compile(r"^\s*conf\.([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.M)

# sejong 에서 도는 읽기 전용 수집기 (stdlib 만). 로컬 load_local 과 같은 파싱.
REMOTE_PY = r'''
import os, re, json, csv, glob, sys
ROOT = '/srv2/kyccj/EIP/paper'; CODE = os.path.expanduser('~/PycharmProjects/TensorFlow-SNNs/_paper')
PATS = __PATS__
NUM = re.compile(r'([A-Za-z_][\w\-]*): ([-+0-9.eE]+|nan|inf)')
SEC = re.compile(r'- (\d+)s (\d+)(?:ms|us|s)/step')
def last_rows(p):
    if not os.path.exists(p): return None
    rows = list(csv.DictReader(open(p)))
    if not rows: return None
    last = max(int(r['epoch']) for r in rows)
    cols = [c for c in rows[0] if c not in ('epoch', 'layer')]
    return {'epoch': last, 'cols': cols,
            'rows': {r['layer']: [float(r[c]) for c in cols] for r in rows if int(r['epoch']) == last}}
out = {}
for pat in PATS:
    for d in sorted(glob.glob(os.path.join(ROOT, pat))):
        tag = os.path.basename(d); lg = os.path.join(d, 'train.log')
        if not os.path.exists(lg): continue
        ep = []
        for line in re.split(r'[\r\n]', open(lg, errors='ignore').read()):
            if 'val_acc:' in line and '500/500' in line:
                kv = {k: float(v) for k, v in NUM.findall(line.split('step', 1)[-1])}
                m = SEC.search(line); kv['_sec'] = float(m.group(1)) if m else None
                ep.append(kv)
        try: host = open(os.path.join(d, 'host.txt')).read().strip()
        except Exception: host = None
        conf = ''
        cf = os.path.join(CODE, tag, 'config_sweep.py')
        if os.path.exists(cf):
            t = open(cf, errors='ignore').read(); i = t.find('# >>> run_paper overrides'); j = t.find('# <<< run_paper overrides')
            if i >= 0 and j > i: conf = t[i:j + 30]
        out[tag] = {'epochs': ep, 'host': host, 'conf': conf,
                    'rd': last_rows(os.path.join(d, 'reg_detail.csv')),
                    'rnd': last_rows(os.path.join(d, 'reg_neuron_detail.csv'))}
json.dump(out, sys.stdout)
'''


# ============================================================================ 읽기
def _lit(s):
    s = s.split('#')[0].strip()
    if s[:1] in ("'", '"'):
        return s.strip('\'"')
    if s in ('True', 'False'):
        return s == 'True'
    try:
        return float(s)
    except ValueError:
        return s


def parse_conf(txt):
    i, j = txt.find('# >>> run_paper overrides'), txt.find('# <<< run_paper overrides')
    if i < 0 or j <= i:
        return {}
    return {k: _lit(v) for k, v in ASSIGN.findall(txt[i:j + 30])}


def parse_log(txt):
    ep = []
    for line in re.split(r'[\r\n]', txt):
        if 'val_acc:' in line and '500/500' in line:
            kv = {k: float(v) for k, v in NUM.findall(line.split('step', 1)[-1])}
            m = SEC.search(line)
            kv['_sec'] = float(m.group(1)) if m else None
            ep.append(kv)
    return ep


def last_rows(p):
    import csv
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    last = max(int(r['epoch']) for r in rows)
    cols = [c for c in rows[0] if c not in ('epoch', 'layer')]
    return {'epoch': last, 'cols': cols,
            'rows': {r['layer']: [float(r[c]) for c in cols] for r in rows if int(r['epoch']) == last}}


def load_local():
    out = {}
    for pat in PATTERNS:
        for d in sorted(glob.glob(os.path.join(ROOT, '_paper', pat))):
            tag = os.path.basename(d)
            lg = os.path.join(d, 'train.log')
            if not os.path.exists(lg):
                continue
            h = os.path.join(HDD, tag)
            src = h if os.path.exists(os.path.join(h, 'reg_detail.csv')) else d
            try:
                host = open(os.path.join(h, 'host.txt')).read().strip()
            except Exception:
                host = None
            cf = os.path.join(d, 'config_sweep.py')
            out[tag] = {'epochs': parse_log(open(lg, errors='ignore').read()), 'host': host,
                        'conf': open(cf, errors='ignore').read() if os.path.exists(cf) else '',
                        'rd': last_rows(os.path.join(src, 'reg_detail.csv')),
                        'rnd': last_rows(os.path.join(src, 'reg_neuron_detail.csv')), 'src': 'local'}
    return out


def load_sejong(cache, refetch):
    if refetch or not os.path.exists(cache):
        code = REMOTE_PY.replace('__PATS__', repr(PATTERNS))
        r = subprocess.run(['ssh', '-p', '23456', '-o', 'BatchMode=yes', 'kyccj@161.122.23.23',
                            '~/anaconda3/envs/venv_eip/bin/python -'], input=code.encode(),
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, timeout=900)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with gzip.open(cache, 'wb') as f:
            f.write(r.stdout)
    d = json.loads(gzip.open(cache, 'rt').read())
    for v in d.values():
        v['src'] = 'sejong'
    return d


def host_start(h):
    if not h:
        return None
    m = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2})', h)
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M') if m else None


def host_gpu(h):
    m = re.search(r'\bgpu(\d+)', h or '')
    return int(m.group(1)) if m else None


def rd_metrics(rd, rnd):
    """reg_detail 마지막 에폭 → 층 집계 (ablation_mech_261010.py log_metrics 와 같은 정의)."""
    if not rd:
        return {}
    ci = {c: i for i, c in enumerate(rd['cols'])}
    rows = {l: v for l, v in rd['rows'].items() if l in NEU}
    spk = {l: v[ci['spike_count']] for l, v in rows.items()}
    w = {l: v[ci['sc_rate_mean']] for l, v in rows.items()}
    R = {l: v[ci['sc_loss']] for l, v in rows.items()}
    tot = sum(spk.values())
    o = {'rd_epoch': rd['epoch'], 'tr_spk': tot, 'R_rd': sum(R.values()), 'R_layer': R, 'spk_layer': spk}
    if sum(R.values()) > 0:          # baseline 은 규제가 없어 w·R 이 0 으로 찍힌다 — 비운다
        o['w_mean'] = sum(w[l] * NEU[l] for l in w) / sum(NEU[l] for l in w)
    for s in STAGES:
        o[f'share_{s}'] = 100 * sum(v for l, v in spk.items() if l.startswith(s)) / tot
        ls = [l for l in w if l.startswith(s)]
        if 'w_mean' in o:
            o[f'w_{s}'] = sum(w[l] * NEU[l] for l in ls) / sum(NEU[l] for l in ls)
            o[f'R_{s}'] = sum(R[l] for l in ls)
    if rnd:
        c2 = {c: i for i, c in enumerate(rnd['cols'])}
        d = {l: v for l, v in rnd['rows'].items() if l in NEU}
        o['dead_pct'] = 100 * sum(d[l][c2['dead_neuron_ratio']] * NEU[l] for l in d) / sum(NEU[l] for l in d)
        o['gini_sw'] = sum(d[l][c2['gini']] * spk[l] for l in d) / tot
        o['top10_sw'] = 100 * sum(d[l][c2['top10_share']] * spk[l] for l in d) / tot
        for s in STAGES[:4]:
            ls = [l for l in d if l.startswith(s)]
            o[f'dead_{s}'] = 100 * sum(d[l][c2['dead_neuron_ratio']] * NEU[l] for l in ls) / sum(NEU[l] for l in ls)
    return o


def expected_arm(conf):
    """override 블록 → (팔 이름, 근거). 이 분석에서 쓰는 팔만 엄격하게 판정."""
    if not conf.get('reg_spike_out', False):
        return 'base'
    g = conf.get('reg_spike_maxnorm_group')
    fs = conf.get('reg_spike_final_step')
    vm = float(conf.get('reg_spike_vmem_gain', 1.0))
    ninv = conf.get('reg_spike_maxnorm_no_inv', False)
    lr = conf.get('reg_spike_loss_ratio', False)
    w1 = conf.get('reg_spike_wta_rev_w1', False)
    if not (conf.get('reg_spike_out_sc_maxnorm') is True and conf.get('reg_spike_out_wta_rev') is True
            and not conf.get('reg_spike_out_sc_maxnorm_plain', False) and not w1
            and float(conf.get('reg_spike_shape_beta', 1.0)) == 1.0
            and float(conf.get('reg_spike_loss_ratio_start_ep', 0) or 0) == 0):
        return 'other'
    if g == 'channel':
        if lr and vm == 1.0 and not ninv:
            return 'ours_ch' if fs else 'ch_nofinal'
        if lr and vm == 0.0 and not ninv and fs:
            return 'ch_novmem'
        if lr and vm == 1.0 and ninv and fs:
            return 'ch_noinv'
        if not lr and vm == 1.0 and not ninv and fs:
            return 'ch_nolr'
    if g == 'within_channel' and lr and vm == 1.0 and not ninv:
        return 'prop' if fs else 'abl_nofinal'
    return 'other'


def build(inv_path, cache, refetch):
    raw = load_local()
    sj = load_sejong(cache, refetch)
    both = sorted(set(raw) & set(sj))
    for k, v in sj.items():
        if k not in raw:
            raw[k] = v
    inv = {r['tag']: r for r in json.load(open(inv_path))['runs']}
    runs = []
    for tag, r in sorted(raw.items()):
        m = NAME_RE.match(tag)
        if not m:
            continue
        key, knob, seed = m.group(1), m.group(2), int(m.group(3))
        conf = parse_conf(r['conf']) if r.get('conf') else {}
        server = '138' if r['src'] == 'local' else '23'
        rows = r['epochs']
        arm = expected_arm(conf) if conf else 'no_conf'
        rec = {'tag': tag, 'key': key, 'knob': knob.replace('p', '.') if key != 'base' else '-', 'seed': seed,
               'server': server, 'epochs': len(rows), 'arm_conf': arm, 'host': r.get('host'),
               'start': host_start(r.get('host')), 'gpu': host_gpu(r.get('host')),
               'rho': conf.get('reg_spike_loss_ratio_target') if conf.get('reg_spike_loss_ratio') else None,
               'lam_fixed': conf.get('reg_spike_out_const') if (conf and not conf.get('reg_spike_loss_ratio')) else None,
               'conf': conf}
        secs = [e['_sec'] for e in rows if e.get('_sec') is not None]
        rec['secs'] = secs
        rec['sec_med'] = float(np.median(secs[1:])) if len(secs) > 2 else None
        if rows:
            b = max(rows, key=lambda x: x.get('val_acc', -1))
            rec.update(best_epoch=rows.index(b) + 1, train_loss=b['loss'], train_acc=b['acc'] * 100,
                       val_loss=b['val_loss'], val_acc=b['val_acc'] * 100, spikes=b['s_count'])
            last = rows[-1]
            if 'adp_lambda' in last:
                lam = np.array([e['adp_lambda'] for e in rows if 'adp_lambda' in e])
                rec.update(lam_end=last['adp_lambda'], lam_mean=float(lam.mean()),
                           R_end=last.get('reg_R'), L_end=last.get('task_loss_clean'),
                           lr_end=last.get('loss_ratio'))
        rec.update(rd_metrics(r.get('rd'), r.get('rnd')))
        iv = inv.get(tag, {})
        reasons = []
        if seed >= 900:
            reasons.append(f'연기시험({len(rows)}ep)')
        elif len(rows) < EPOCHS:
            reasons.append(f'미완주({len(rows)}ep)')
        elif len(rows) > EPOCHS:
            reasons.append(f'에폭 줄 과다({len(rows)})')
        if 'S30/S1' in (iv.get('exclude_reason') or ''):
            reasons.append([x for x in iv['exclude_reason'].split('; ') if 'S30' in x][0])
        if iv.get('server') in ('138', '23') and iv['server'] != server:
            reasons.append(f"서버 불일치(읽은 곳 {server}, 원장 {iv['server']})")
        if key in ('base', 'ours_ch', 'ch_nofinal', 'ch_novmem', 'ch_nolr', 'ch_noinv', 'prop', 'abl_nofinal') \
                and arm != key:
            reasons.append(f'팔 불일치(dir={key}, conf={arm})')
        if key not in ('base', 'ch_nolr') and rec['rho'] is not None and abs(float(rec['knob']) - rec['rho']) > 1e-12:
            reasons.append(f"ρ 불일치(dir={rec['knob']}, conf={rec['rho']})")
        rec['exclude_reason'] = '; '.join(reasons)
        rec['use'] = not reasons
        if iv and rows:
            rec['inv_check'] = {'val_acc_diff': None if iv.get('val_acc') is None else round(rec['val_acc'] - iv['val_acc'], 4),
                                'spikes_diff': None if iv.get('spikes') is None else round(rec['spikes'] - iv['spikes'], 1),
                                'server_match': iv.get('server') == server}
        runs.append(rec)
    return runs, both, inv


# ============================================================================ 통계
def tcrit(df):
    return stats.t.ppf(0.975, df)


def ols(X, y, names):
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    res = y - X @ beta
    df = n - k
    s2 = float(res @ res / df)
    cov = s2 * XtXi
    se = np.sqrt(np.diag(cov))
    co = {nm: {'est': float(beta[i]), 'se': float(se[i]), 'lo': float(beta[i] - tcrit(df) * se[i]),
               'hi': float(beta[i] + tcrit(df) * se[i]), 't': float(beta[i] / se[i]),
               'p': float(2 * stats.t.sf(abs(beta[i] / se[i]), df))} for i, nm in enumerate(names)}
    return {'coef': co, 'df': int(df), 'n': int(n), 'resid_sd': math.sqrt(s2), 'beta': beta, 'cov': cov,
            'names': list(names), 'X': X, 'y': y}


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    return {'diff': float(d), 'se': se, 'df': df, 'p': float(2 * stats.t.sf(abs(d / se), df)),
            'lo': float(d - tcrit(df) * se), 'hi': float(d + tcrit(df) * se), 'na': len(a), 'nb': len(b)}


def fieller(num, den, vn, vd, cnd, df):
    """θ = num/den 의 Fieller 95% 구간 (den 이 유의해야 유계)."""
    t2 = tcrit(df) ** 2
    A = den * den - t2 * vd
    B = -2 * (num * den - t2 * cnd)
    C = num * num - t2 * vn
    if A <= 0:
        return None, None
    r = math.sqrt(max(B * B - 4 * A * C, 0))
    return (-B - r) / (2 * A), (-B + r) / (2 * A)


def ancova(base, arm, yf, xf=lambda r: math.log(r['spikes']), server=True, x0=None):
    """y = a + b·x + c·[arm] (+ e·[23]). x 는 x0 로 중심화. 반환: c (같은 x 에서의 차) 와 적합 정보."""
    data = list(base) + list(arm)
    x = np.array([xf(r) for r in data])
    if x0 is not None:
        x = x - x0
    y = np.array([yf(r) for r in data], float)
    g = np.array([0.0] * len(base) + [1.0] * len(arm))
    cols, names = [np.ones_like(x), x, g], ['a', 'b', 'c']
    sv = np.array([1.0 if r['server'] == '23' else 0.0 for r in data])
    if server and 0 < sv.sum() < len(sv):
        cols.append(sv)
        names.append('e')
    f = ols(np.c_[tuple(cols)], y, names)
    f['n_base'], f['n_arm'] = len(base), len(arm)
    return f


def slim(f):
    return {'coef': f['coef'], 'df': f['df'], 'n': f['n'], 'resid_sd': f['resid_sd'],
            'n_base': f.get('n_base'), 'n_arm': f.get('n_arm')}


def zpred_line(xs, ys, x0, y0):
    """n=1 값의 직선 예측 대비 잔차와 z (새 관측 하나의 예측 SE)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    f = ols(np.c_[np.ones_like(xs), xs], ys, ['a', 'b'])
    v = np.array([1.0, x0])
    pred = float(v @ f['beta'])
    se = math.sqrt(float(v @ f['cov'] @ v) + f['resid_sd'] ** 2)
    z = (y0 - pred) / se
    return {'pred': pred, 'obs': float(y0), 'resid': float(y0 - pred), 'se': se, 'z': float(z),
            'p': float(2 * stats.t.sf(abs(z), f['df'])), 'n_fit': len(xs), 'df': f['df']}


def n_needed(delta, sd, target=0.8):
    """두 표본 t (조건당 같은 n, α=0.05 양측) 로 검정력 target 에 필요한 n — nofinal_vs_inter_261014.py 와 같다."""
    for n in range(2, 5000):
        df = 2 * n - 2
        ncp = abs(delta) / (sd * math.sqrt(2 / n))
        tc = tcrit(df)
        if stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp) >= target:
            return n
    return None


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); s = x.sum()
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * s)) if s > 0 else float('nan')


# ============================================================================ npz
def npz_path(tag, dirs=NPZ_DIRS):
    for d in dirs:
        p = os.path.join(d, tag + '.npz')
        if os.path.exists(p):
            return p
    return None


def chan(p):
    """채널 단위 구조 (ablation_mech_addendum_261010.py chan 과 같은 정의)."""
    z = np.load(p); ns = float(z['n_samples'])
    o = {'dead_ch': 0, 'dead_by_stage': {s: 0 for s in STAGES}, 'n_ch': 0}
    gw = tw = sw = 0.0; act = 0.0; sil = 0; spk = 0.0
    for l in LAYERS:
        c = z[l].astype(np.float64)
        ch = c.reshape(-1, c.shape[-1]).sum(0) if c.ndim == 3 else c
        d = int((ch == 0).sum()); o['dead_ch'] += d; o['dead_by_stage'][l.split('_')[0]] += d; o['n_ch'] += ch.size
        s = ch.sum(); k = max(1, int(round(0.1 * ch.size)))
        gw += gini(ch) * s; tw += np.sort(ch)[::-1][:k].sum() / s * s; sw += s
        act += z[l + '__fired'].sum(); sil += int((c == 0).sum()); spk += c.sum()
    o.update(gini_ch=gw / sw, top10_ch_pct=100 * tw / sw, active_img_pct=100 * act / ns / TOTN,
             silent_neu_pct=100 * sil / TOTN, npz_spk=spk / ns, npz_acc=float(z['acc']) if 'acc' in z.files else None)
    return o


def tmetrics(p):
    """스텝별 몫·최초 발화 (ablation_mech_addendum_261010.py tmetrics 와 같은 정의) + 층별 스텝 합."""
    z = np.load(p); ns = float(z['n_samples']); T = int(z['T'])
    tot_t = np.zeros(T); first = np.zeros(T + 1)
    st_t = {s: np.zeros(T) for s in STAGES}
    lay_t = {}
    for l in LAYERS:
        a = z[l + '__t'].reshape(T, -1).sum(1).astype(np.float64)
        lay_t[l] = a
        tot_t += a; first += z[l + '__first_t'].astype(np.float64); st_t[l.split('_')[0]] += a
    fired = first[:T].sum()
    o = {'spk_per_img': float(tot_t.sum() / ns), 'step_share': (100 * tot_t / tot_t.sum()).tolist(),
         'first_share': (100 * first[:T] / fired).tolist(),
         'mean_first_t': float((np.arange(1, T + 1) * first[:T]).sum() / fired),
         'lay_t': {l: v.tolist() for l, v in lay_t.items()}}
    for s in STAGES:
        o[f'share_t_{s}'] = (100 * st_t[s] / st_t[s].sum()).tolist()
    return o


def partial_rank(p):
    """이미지 합산 채널 누적 합 C_t (t 까지) 와 C_T 의 순위 상관, 그리고 w_t = 1 − C_t/max C_t 와 w_T 의 차.
    규제가 실제로 쓰는 것은 이미지별 값이라 이 합산 지표는 이미지별 순위 차이를 과소평가할 수 있다."""
    z = np.load(p); T = int(z['T'])
    out = {t: {'rho': [], 'dw': [], 'zero': [], 'n': []} for t in range(1, T)}
    for l in LAYERS:
        if l == 'fc1_n':
            continue
        a = z[l + '__t'].astype(np.float64)            # (T, H, W, C)
        C = np.cumsum(a.sum((1, 2)), axis=0)          # (T, C)
        wT = 1 - C[-1] / C[-1].max()
        for t in range(1, T):
            Ct = C[t - 1]
            r = stats.spearmanr(Ct, C[-1]).correlation if Ct.max() > Ct.min() else float('nan')
            wt = 1 - Ct / Ct.max() if Ct.max() > 0 else np.ones_like(Ct)
            out[t]['rho'].append(r); out[t]['dw'].append(float(np.mean(np.abs(wt - wT))))
            out[t]['zero'].append(float(np.mean(Ct == 0))); out[t]['n'].append(C.shape[1])
    return {t: {'spearman_med': float(np.nanmedian(v['rho'])), 'spearman_min': float(np.nanmin(v['rho'])),
                'mean_abs_dw': float(np.average(v['dw'], weights=v['n'])),
                'zero_ch_frac': float(np.average(v['zero'], weights=v['n']))} for t, v in out.items()}


# ============================================================================ 분석
def analyze(runs, inv):
    res = {}
    use = [r for r in runs if r['use']]
    by = lambda k: [r for r in use if r['key'] == k]
    base, ours_all, prop, anf = by('base'), by('ours_ch'), by('prop'), by('abl_nofinal')
    nof_all = by('ch_nofinal')
    nof = [r for r in nof_all if r['knob'] == NOF_RHO]
    ours = [r for r in ours_all if BAND[0] <= r['spikes'] <= BAND[1]]
    res['n'] = {'base': len(base), 'ours_all': len(ours_all), 'ours_band': len(ours), 'nofinal': len(nof),
                'nofinal_all_complete': len(nof_all), 'prop': len(prop), 'abl_nofinal': len(anf),
                'lw1_any': len([r for r in runs if r['key'].startswith('lw1_')])}
    res['ours_band_rhos'] = sorted({r['knob'] for r in ours}, key=float)
    s0 = float(np.exp(np.mean(np.log([r['spikes'] for r in nof]))))
    a0 = float(np.mean([r['val_acc'] for r in nof]))
    res['nof_center'] = {'spikes_geo': s0, 'spikes_mean': float(np.mean([r['spikes'] for r in nof])), 'acc': a0}
    x0 = math.log(float(np.mean([r['spikes'] for r in nof])))   # nofinal_vs_inter 와 같은 중심 (산술 평균)

    # 0. 설정 차이 확인: ch_nofinal 과 ours_ch override 블록 diff (경로·시드·ρ·final_step 제외)
    ign = {'reg_spike_loss_ratio_target', 'reg_spike_final_step', 'root_model_save', 'root_tensorboard',
           'name_model_load', 'random_seed', 'seed', 'exp_set_name', 'path_stat'}
    def cdict(r):
        return {k: v for k, v in r['conf'].items() if k not in ign and 'path' not in k and 'root' not in k}
    diffs = {}
    for q in nof_all:
        ref = [r for r in ours_all if r['server'] == q['server']][0]
        a, b = cdict(q), cdict(ref)
        diffs[q['tag']] = {'ref': ref['tag'], 'diff': {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}}
    res['conf_diff'] = {k: {'ref': v['ref'], 'diff': {kk: [str(x) for x in vv] for kk, vv in v['diff'].items()}}
                        for k, v in diffs.items()}

    # 1. 같은 스파이크 정확도 (nofinal_vs_inter_261014 와 같은 모형)
    acc = lambda r: r['val_acc']
    A = {}
    A['lin'] = ancova(ours, nof, acc, server=False, x0=x0)
    A['lin_srv'] = ancova(ours, nof, acc, server=True, x0=x0)
    xq = lambda r: math.log(r['spikes']) - x0
    data = ours + nof
    Xq = np.c_[np.ones(len(data)), [xq(r) for r in data], [xq(r) ** 2 for r in data],
               [1.0 if r['key'] == 'ch_nofinal' else 0.0 for r in data], [1.0 if r['server'] == '23' else 0.0 for r in data]]
    A['quad_srv'] = ols(Xq, [acc(r) for r in data], ['a', 'b', 'q', 'c', 'e'])
    # 사전 등록 민감도: 가설을 만든 s1 을 뺀 집합 (n=2)
    A['lin_srv_no_s1'] = ancova(ours, [r for r in nof if r['seed'] != 1], acc, server=True, x0=x0)
    res['acc_ancova'] = {k: slim(v) for k, v in A.items()}
    # 합동 시드 SD (ours 같은 ρ 묶음, 전 범위)
    grp = {}
    for r in ours_all:
        grp.setdefault(r['knob'], []).append(r['val_acc'])
    pn = sum((len(v) - 1) * np.var(v, ddof=1) for v in grp.values() if len(v) > 1)
    pdf = sum(len(v) - 1 for v in grp.values() if len(v) > 1)
    res['pooled_sd'] = {'sd': math.sqrt(pn / pdf), 'df': pdf, 'rhos': sorted([k for k, v in grp.items() if len(v) > 1], key=float)}
    # baseline 대비
    res['vs_base'] = {'acc': welch([r['val_acc'] for r in nof], [r['val_acc'] for r in base]),
                      'lnspk': welch(np.log([r['spikes'] for r in nof]), np.log([r['spikes'] for r in base])),
                      'base_acc': float(np.mean([r['val_acc'] for r in base])),
                      'base_spk': float(np.mean([r['spikes'] for r in base]))}

    # 2. 같은 ρ / 같은 λ
    lnrho = lambda r: math.log(r['rho'])
    lns = lambda r: math.log(r['spikes'])
    S = {}
    f = ancova(ours, nof, lns, xf=lnrho, server=True)
    i_b, i_c = f['names'].index('b'), f['names'].index('c')
    b, c = f['beta'][i_b], f['beta'][i_c]
    th = c / b
    lo, hi = fieller(c, b, f['cov'][i_c, i_c], f['cov'][i_b, i_b], f['cov'][i_b, i_c], f['df'])
    S['same_rho'] = {'fit': slim(f), 'spk_ratio': math.exp(c), 'spk_ratio_lo': math.exp(f['coef']['c']['lo']),
                     'spk_ratio_hi': math.exp(f['coef']['c']['hi']), 'rho_mult': math.exp(th),
                     'rho_mult_lo': math.exp(min(lo, hi)) if lo is not None else None,
                     'rho_mult_hi': math.exp(max(lo, hi)) if lo is not None else None}
    # 2차 항 검사 (ln ρ 에서 곡률)
    dq = ours + nof
    xr = np.array([lnrho(r) for r in dq]); xr_c = xr - xr.mean()
    Xr = np.c_[np.ones(len(dq)), xr_c, xr_c ** 2, [1.0 if r['key'] == 'ch_nofinal' else 0.0 for r in dq],
               [1.0 if r['server'] == '23' else 0.0 for r in dq]]
    fq = ols(Xr, [lns(r) for r in dq], ['a', 'b', 'q', 'c', 'e'])
    S['same_rho_quad'] = slim(fq)
    # 2차 곡선에서 ρ 배율: ours 곡선 f(x)=a+bx+qx² 가 f(x0)+c 가 되는 x' (x0 = ln 2.2e-3 − 평균), 배율 exp(x'−x0)
    x0r = math.log(float(NOF_RHO)) - xr.mean()
    def qmult(bv):
        a_, b_, q_, c_ = bv[0], bv[1], bv[2], bv[3]
        tgt = b_ * x0r + q_ * x0r ** 2 + c_
        disc = b_ * b_ + 4 * q_ * tgt
        if disc < 0 or abs(q_) < 1e-12:
            return None
        roots = [(-b_ + sgn * math.sqrt(disc)) / (2 * q_) for sgn in (1, -1)]
        xr_ = min(roots, key=lambda v: abs(v - x0r))
        return math.exp(xr_ - x0r)
    rng = np.random.default_rng(0)
    draws = rng.multivariate_normal(fq['beta'], fq['cov'], 20000)
    ms = np.array([m for m in (qmult(d_) for d_ in draws) if m is not None])
    S['same_rho_quad_mult'] = {'mult': qmult(fq['beta']), 'lo': float(np.percentile(ms, 2.5)),
                               'hi': float(np.percentile(ms, 97.5)), 'n_draw': int(len(ms))}
    # 국소: ρ 4e-3~5.5e-3 의 ours 만으로 (ln s ~ ln ρ) 역산
    loc = [r for r in ours_all if 4e-3 <= r['rho'] <= 5.5e-3]
    fl = ols(np.c_[np.ones(len(loc)), [lnrho(r) for r in loc]], [lns(r) for r in loc], ['a', 'b'])
    rho_eq = math.exp((float(np.mean([lns(r) for r in nof])) - fl['beta'][0]) / fl['beta'][1])
    S['local_rho_eq'] = {'n': len(loc), 'rhos': sorted({r['knob'] for r in loc}, key=float), 'rho_eq': rho_eq,
                         'mult': rho_eq / float(NOF_RHO), 'b': float(fl['beta'][1]), 'b_se': fl['coef']['b']['se']}
    # 같은 ρ 짝 (ours ρ 2e-3 n=3 과 2.2e-3 nofinal 은 ρ 가 10% 다름 → 위 회귀가 주결과)
    for lk in ('lam_end', 'lam_mean'):
        ok = [r for r in ours if r.get(lk)]
        fl2 = ancova(ok, nof, lns, xf=lambda r, lk=lk: math.log(r[lk]), server=True)
        S[f'same_{lk}'] = {'fit': slim(fl2), 'spk_ratio': math.exp(fl2['coef']['c']['est']),
                           'lo': math.exp(fl2['coef']['c']['lo']), 'hi': math.exp(fl2['coef']['c']['hi'])}
    # 같은 스파이크에서 λ, R, L (ln) → ρ 분해: ln ρ = ln λ + ln R − ln L (λ_end = ρ·L_end/R_end)
    dec = {}
    for k in ('lam_end', 'lam_mean', 'R_end', 'L_end', 'R_rd', 'tr_spk'):
        ok = [r for r in ours if r.get(k)]
        fk = ancova(ok, nof, lambda r, k=k: math.log(r[k]), server=True, x0=x0)
        dec[k] = {'fit': slim(fk), 'ratio': math.exp(fk['coef']['c']['est']), 'lo': math.exp(fk['coef']['c']['lo']),
                  'hi': math.exp(fk['coef']['c']['hi']), 'p': fk['coef']['c']['p'],
                  'ours_at_nof': math.exp(fk['coef']['a']['est']) if 'e' not in fk['names'] else None}
    dec['check_identity'] = {'lam_x_R_over_L': dec['lam_end']['ratio'] * dec['R_end']['ratio'] / dec['L_end']['ratio'],
                             'rho_ratio_from_mult': 1 / S['same_rho']['rho_mult']}
    S['same_spk_decomp'] = dec
    # 런별 λ·R 표 (nofinal 3 + ours 근처 ρ)
    res['same_rho'] = S

    # 3a. 기울기 배율 추정 — 스텝별 노름 (npz_t 의 층·스텝 스파이크 몫)
    #   final: g = λ_f·w²/R_l,f (모든 t 같음).  nofinal: g_t = λ_n·w_t²/R_l,t,  R_l,n = Σ_t R_l,t.
    #   w_t ≈ w 가정 → R_l,t ≈ R_l,n·√p_lt / Σ_s √p_ls.  같은 압력이면 λ_f/λ_n = K = E_t[R_l,f / R_l,t].
    pt = npz_path('r19c10-ch_nofinal-2p2e-3-s1', [NPZ_T])
    K = {}
    if pt:
        tm = tmetrics(pt)
        Rn = {l: float(np.mean([r['R_layer'][l] for r in nof if 'R_layer' in r])) for l in LAYERS}
        Nn = {l: float(np.mean([r['spk_layer'][l] for r in nof if 'spk_layer' in r])) for l in LAYERS}
        Rf = {}
        for l in LAYERS:
            ok = [r for r in ours if 'R_layer' in r and r['R_layer'].get(l, 0) > 0]
            fx = np.array([math.log(r['spikes']) - x0 for r in ok]); fy = np.array([math.log(r['R_layer'][l]) for r in ok])
            Rf[l] = math.exp(ols(np.c_[np.ones_like(fx), fx], fy, ['a', 'b'])['beta'][0])
        per = {}
        for l in LAYERS:
            p = np.array(tm['lay_t'][l], float); p = p / p.sum()
            sq = np.sqrt(p)
            ratio = Rf[l] / Rn[l]
            k_spk = ratio * sq.sum() ** 2
            k_uni = ratio * sq.sum() * np.mean(1 / sq)
            per[l] = {'p': p.tolist(), 'Rf_over_Rn': ratio, 'K_spk': float(k_spk), 'K_uni': float(k_uni),
                      'rel_push_t': (1 / sq / np.mean(1 / sq)).tolist(), 'N': Nn[l]}
        wN = np.array([per[l]['N'] for l in LAYERS])
        K['per_layer'] = per
        K['K_spk'] = float(np.average([per[l]['K_spk'] for l in LAYERS], weights=wN))
        K['K_uni'] = float(np.average([per[l]['K_uni'] for l in LAYERS], weights=wN))
        K['Rf_over_Rn_w'] = float(np.average([per[l]['Rf_over_Rn'] for l in LAYERS], weights=wN))
        for s in STAGES:
            ls = [l for l in LAYERS if l.startswith(s)]
            ww = [per[l]['N'] for l in ls]
            K[f'stage_{s}'] = {'K_spk': float(np.average([per[l]['K_spk'] for l in ls], weights=ww)),
                               'K_uni': float(np.average([per[l]['K_uni'] for l in ls], weights=ww)),
                               'p': np.average([per[l]['p'] for l in ls], axis=0, weights=ww).tolist(),
                               't1_vs_t4_push': float(np.average([per[l]['rel_push_t'][0] / per[l]['rel_push_t'][3] for l in ls], weights=ww))}
        K['K_meas_lam_end'] = 1 / dec['lam_end']['ratio']
        K['K_meas_lo'] = 1 / dec['lam_end']['hi']
        K['K_meas_hi'] = 1 / dec['lam_end']['lo']
        K['Rn_over_Rf_meas'] = dec['R_end']['ratio']
        K['pred_rho_mult_spk'] = K['K_spk'] / dec['R_end']['ratio'] * dec['L_end']['ratio']
        K['pred_rho_mult_uni'] = K['K_uni'] / dec['R_end']['ratio'] * dec['L_end']['ratio']
    res['grad_mult'] = K

    # 3b. 학습 모드 지표 (같은 스파이크, ours 200~420K vs nofinal n=3)
    mech = {}
    for k in ['w_mean', 'dead_pct', 'gini_sw', 'top10_sw'] + [f'share_{s}' for s in STAGES[:4]] + \
             [f'w_{s}' for s in STAGES[:4]] + [f'dead_{s}' for s in STAGES[:4]]:
        ok = [r for r in ours if k in r]; nk = [r for r in nof if k in r]
        if len(ok) < 5 or not nk:
            continue
        fk = ancova(ok, nk, lambda r, k=k: r[k], server=True, x0=x0)
        e_ = fk['coef']['e']['est'] if 'e' in fk['coef'] else 0.0
        mech[k] = {'fit': slim(fk), 'c': fk['coef']['c'], 'nof_vals': [r[k] for r in nk],
                   'ours_at_nof': fk['coef']['a']['est'] + e_ * float(np.mean([r['server'] == '23' for r in nk]))}
        # 10-10 (n=1, s1 만) 과 비교용: s1 단독 z
        s1 = [r for r in nk if r['seed'] == 1]
        if s1:
            mech[k]['s1_z'] = zpred_line([math.log(r['spikes']) for r in ok], [r[k] for r in ok],
                                         math.log(s1[0]['spikes']), s1[0][k])
    res['train_mode'] = mech

    # 3c. npz 채널 구조 (nofinal s1 만 npz 있음, n=1)
    st = {}
    pn = npz_path('r19c10-ch_nofinal-2p2e-3-s1')
    if pn:
        nofc = chan(pn)
        ours_np = [(r, npz_path(r['tag'])) for r in ours]
        ours_np = [(r, p) for r, p in ours_np if p]
        oc = [(r, chan(p)) for r, p in ours_np]
        st['ours_tags'] = [r['tag'] for r, _ in oc]
        st['nof'] = nofc
        sp_nof = [r for r in nof if r['seed'] == 1][0]['spikes']
        st['nof_ledger_spk'] = sp_nof
        for k in ('dead_ch', 'gini_ch', 'top10_ch_pct', 'active_img_pct', 'silent_neu_pct'):
            st[k] = zpred_line([math.log(r['spikes']) for r, _ in oc], [o[k] for _, o in oc], math.log(sp_nof), nofc[k])
        for s in STAGES[:4]:
            st[f'dead_{s}'] = zpred_line([math.log(r['spikes']) for r, _ in oc], [o['dead_by_stage'][s] for _, o in oc],
                                        math.log(sp_nof), nofc['dead_by_stage'][s])
        st['npz_over_ledger'] = {'nof': nofc['npz_spk'] / sp_nof,
                                 'ours': [o['npz_spk'] / r['spikes'] for r, o in oc]}
        bc = [chan(npz_path(f'r19c10-base---s{i}')) for i in (11, 12, 13) if npz_path(f'r19c10-base---s{i}')]
        st['base_mean'] = {k: float(np.mean([o[k] for o in bc])) for k in ('dead_ch', 'gini_ch', 'top10_ch_pct',
                                                                          'active_img_pct', 'silent_neu_pct')}
        st['base_n'] = len(bc)
        # npz 일관성: 같은 런이 _npz 와 _npz_t 에 둘 다 있을 때 뉴런 총합이 같은가
        a, b_ = npz_path('r19c10-ours_ch-1p5e-3-s1', [ROOT + '/_npz']), npz_path('r19c10-ours_ch-1p5e-3-s1', [NPZ_T])
        if a and b_:
            za, zb = np.load(a), np.load(b_)
            st['npz_consistency_max_rel'] = float(max(abs(za[l].sum() - zb[l].sum()) / max(za[l].sum(), 1) for l in LAYERS))
    res['npz_struct'] = st

    # 3d. 시간축 (npz_t) — addendum 재계산 + 스텝별 기울기 배율 + 부분 누적 순위
    T_ = {}
    grp_t = {'baseline': [f'r19c10-base---s{i}' for i in (11, 12, 13)],
             'ours_inter_ch 1.5e-3': [f'r19c10-ours_ch-1p5e-3-s{i}' for i in (1, 2, 3)],
             '- vmem 3e-3 (final_step 켜짐)': ['r19c10-ch_novmem-3e-3-s1'],
             '- final_step 2.2e-3': ['r19c10-ch_nofinal-2p2e-3-s1']}
    tmv = {}
    for g, tags in grp_t.items():
        ms = [tmetrics(npz_path(t, [NPZ_T])) for t in tags if npz_path(t, [NPZ_T])]
        tmv[g] = ms
        T_[g] = {'n': len(ms), 'spk_per_img': float(np.mean([m['spk_per_img'] for m in ms])),
                 'step_share': np.mean([m['step_share'] for m in ms], 0).tolist(),
                 'step_share_sd': (np.std([m['step_share'] for m in ms], 0, ddof=1).tolist() if len(ms) > 1 else None),
                 'first_share': np.mean([m['first_share'] for m in ms], 0).tolist(),
                 'mean_first_t': float(np.mean([m['mean_first_t'] for m in ms])),
                 **{f'share_t_{s}': np.mean([m[f'share_t_{s}'] for m in ms], 0).tolist() for s in STAGES}}
    # 스파이크 수준 추세 (baseline → inter 1.5e-3 직선, addendum 과 같음) 에서 nofinal 잔차
    xb, xi = math.log(T_['baseline']['spk_per_img']), math.log(T_['ours_inter_ch 1.5e-3']['spk_per_img'])
    trend = {}
    def scal(m, k):
        if k.startswith('step'):
            return m['step_share'][int(k[-1]) - 1]
        if k.startswith('last_'):
            return m[f'share_t_{k[5:]}'][3]
        if k.startswith('first1_'):
            return m[f'share_t_{k[7:]}'][0]
        return m[k]
    for k in ['step1', 'step4', 'mean_first_t'] + [f'last_{s}' for s in STAGES[:4]] + [f'first1_{s}' for s in STAGES[:4]]:
        vb = [scal(m, k) for m in tmv['baseline']]; vi = [scal(m, k) for m in tmv['ours_inter_ch 1.5e-3']]
        yb, yi = np.mean(vb), np.mean(vi); sdb, sdi = np.std(vb, ddof=1), np.std(vi, ddof=1)
        o = {}
        for g in ('- final_step 2.2e-3', '- vmem 3e-3 (final_step 켜짐)'):
            m = tmv[g][0]
            d = (math.log(m['spk_per_img']) - xi) / (xi - xb)
            pred = yi + d * (yi - yb)
            se = math.sqrt((1 + d) ** 2 * sdi ** 2 / 3 + d ** 2 * sdb ** 2 / 3 + sdi ** 2)
            o[g] = {'obs': scal(m, k), 'pred': pred, 'resid': scal(m, k) - pred, 'se': se, 'z': (scal(m, k) - pred) / se}
        trend[k] = o
    T_['trend'] = trend
    if pt:
        T_['partial_rank_nofinal'] = partial_rank(pt)
        p15 = npz_path('r19c10-ours_ch-1p5e-3-s1', [NPZ_T])
        if p15:
            T_['partial_rank_ours15_s1'] = partial_rank(p15)
    res['time_axis'] = T_

    # 3e. intra 교차 확인 (abl_nofinal vs prop)
    I = {}
    fi = ancova(prop, anf, lns, xf=lnrho, server=True)
    i_b, i_c = fi['names'].index('b'), fi['names'].index('c')
    b, c = fi['beta'][i_b], fi['beta'][i_c]
    lo, hi = fieller(c, b, fi['cov'][i_c, i_c], fi['cov'][i_b, i_b], fi['cov'][i_b, i_c], fi['df'])
    I['same_rho'] = {'fit': slim(fi), 'spk_ratio': math.exp(c), 'lo': math.exp(fi['coef']['c']['lo']),
                     'hi': math.exp(fi['coef']['c']['hi']), 'rho_mult': math.exp(c / b),
                     'rho_mult_lo': math.exp(min(lo, hi)) if lo is not None else None,
                     'rho_mult_hi': math.exp(max(lo, hi)) if lo is not None else None}
    xi0 = math.log(float(np.mean([r['spikes'] for r in anf])))
    I['acc_lin'] = slim(ancova(prop, anf, acc, server=False, x0=xi0))
    I['acc_lin_srv'] = slim(ancova(prop, anf, acc, server=True, x0=xi0))
    for k in ('lam_end', 'R_end', 'L_end'):
        ok = [r for r in prop if r.get(k)]; nk = [r for r in anf if r.get(k)]
        fk = ancova(ok, nk, lambda r, k=k: math.log(r[k]), server=True, x0=xi0)
        I[f'same_spk_{k}'] = {'ratio': math.exp(fk['coef']['c']['est']), 'lo': math.exp(fk['coef']['c']['lo']),
                              'hi': math.exp(fk['coef']['c']['hi']), 'n': [len(ok), len(nk)]}
    I['prop_range'] = [min(r['spikes'] for r in prop), max(r['spikes'] for r in prop)]
    I['prop_rhos'] = sorted({r['knob'] for r in prop}, key=float)
    res['intra'] = I

    # 4. 학습 시간
    tm_runs = [r for r in runs if r['sec_med'] and r['seed'] < 900 and r['epochs'] >= 10
               and r['key'] in ('base', 'ours_ch', 'ch_nofinal', 'ch_novmem', 'ch_nolr', 'ch_noinv', 'prop', 'abl_nofinal')
               and 'S30' not in r['exclude_reason'] and '불일치' not in r['exclude_reason']]
    summ = {}
    for r in tm_runs:
        summ.setdefault((r['key'], r['server']), []).append(r['sec_med'])
    res['time_summary'] = {f'{k}|{s}': {'n': len(v), 'median': float(np.median(v)), 'min': float(min(v)), 'max': float(max(v))}
                           for (k, s), v in sorted(summ.items())}
    # 같은 시각 짝: 같은 서버·시작 시각 차 ≤ 3분 인 묶음 안에서 nofinal 과 final_step 켜진 inter 계열
    pairs = []
    nf_t = [r for r in tm_runs if r['key'] == 'ch_nofinal' and r['start']]
    seen = set()
    for q in nf_t:
        if q['tag'] in seen:
            continue
        mates = [r for r in tm_runs if r['server'] == q['server'] and r['start']
                 and abs((r['start'] - q['start']).total_seconds()) <= 180]
        nfs = [r for r in mates if r['key'] == 'ch_nofinal']
        fon = [r for r in mates if r['key'] in ('ours_ch', 'ch_novmem', 'ch_nolr', 'ch_noinv')]
        seen |= {r['tag'] for r in nfs}
        if not fon:
            continue
        W = min(sum(r['secs']) for r in nfs + fon)      # 모두가 돌고 있던 구간 (초)
        def win_med(r):
            cs = np.cumsum(r['secs'])
            v = [s for s, c_ in zip(r['secs'], cs) if c_ <= W][1:]
            return float(np.median(v)), len(v) + 1
        o = {'server': q['server'], 'start': q['start'].strftime('%m-%d %H:%M'), 'window_h': W / 3600,
             'runs': [{'tag': r['tag'], 'key': r['key'], 'gpu': r['gpu'], 'med': win_med(r)[0], 'ep_in_win': win_med(r)[1],
                       'med_all': r['sec_med'], 'epochs': r['epochs']} for r in nfs + fon]}
        mn = np.mean([x['med'] for x in o['runs'] if x['key'] == 'ch_nofinal'])
        mf = np.mean([x['med'] for x in o['runs'] if x['key'] != 'ch_nofinal'])
        o['ratio'] = float(mn / mf)
        tot_n = np.mean([sum(r['secs']) for r in nfs if r['epochs'] >= EPOCHS]) if any(r['epochs'] >= EPOCHS for r in nfs) else None
        tot_f = np.mean([sum(r['secs']) for r in fon if r['epochs'] >= EPOCHS]) if any(r['epochs'] >= EPOCHS for r in fon) else None
        o['total_h'] = {'nofinal': tot_n / 3600 if tot_n else None, 'final_on': tot_f / 3600 if tot_f else None}
        pairs.append(o)
    res['time_pairs'] = pairs
    res['time_runs'] = [{'tag': r['tag'], 'key': r['key'], 'server': r['server'], 'gpu': r['gpu'],
                         'start': r['start'].strftime('%m-%d %H:%M') if r['start'] else None, 'epochs': r['epochs'],
                         'sec_med': r['sec_med']} for r in tm_runs]

    # 5. 검정력 (같은 스파이크 정확도)
    sd = res['pooled_sd']['sd']
    f = A['lin_srv']
    se_d = f['coef']['c']['se']
    var_curve = max(se_d ** 2 - f['resid_sd'] ** 2 / len(nof), 0.0)
    k_ = stats.t.ppf(0.975, f['df']) + stats.t.ppf(0.8, f['df'])
    res['power'] = {'curve_se': math.sqrt(var_curve),
                    'mde_by_n': {str(m): k_ * math.sqrt(sd ** 2 / m + var_curve) for m in (3, 4, 6, 10, 20)},
                    'floor': k_ * math.sqrt(var_curve),
                    'n_for_obs': n_needed(f['coef']['c']['est'], sd)}
    return res


# ============================================================================ md
def F(v, d=3, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '–'
    return (f'{v:+.{d}f}' if sign else f'{v:.{d}f}')


def K_(v):
    return f'{v:,.0f}'


def Pv(p):
    return '–' if p is None else (f'{p:.2g}' if p < 0.001 else f'{p:.3f}')


def E(v):
    return '–' if v is None else f'{v:.3g}'


def pct(r):
    return f'{(r - 1) * 100:+.1f}%'


def run_row(r, base_spk, extra=''):
    if not r['use'] and ('미완주' in r['exclude_reason'] or '연기시험' in r['exclude_reason']):
        return (f"| {ARM.get(r['key'], r['key'])} | {r['knob']} | s{r['seed']} | {r['server']} | – | – | – | – | – | – | – | – | – | – | "
                f"{F(r.get('sec_med'), 0)} | {extra or r['exclude_reason']} (값 비움) |")
    return (f"| {ARM.get(r['key'], r['key'])} | {r['knob']} | s{r['seed']} | {r['server']} | {r.get('best_epoch', '–')} | "
            f"{F(r.get('train_loss'), 4)} | {F(r.get('train_acc'), 2)} | {F(r.get('val_loss'), 4)} | {F(r.get('val_acc'), 2)} | "
            f"{K_(r['spikes']) if r.get('spikes') else '–'} | {pct(r['spikes'] / base_spk) if r.get('spikes') else '–'} | "
            f"{E(r.get('lam_end'))} | {K_(r['R_end']) if r.get('R_end') else '–'} | {F(r.get('w_mean'), 3)} | "
            f"{F(r.get('sec_med'), 0)} | {extra or r['exclude_reason']} |")


def mean_row(rs, label, base_spk):
    g = lambda k: [r[k] for r in rs if r.get(k) is not None]
    m = lambda k: float(np.mean(g(k))) if g(k) else None
    sd = lambda k: float(np.std(g(k), ddof=1)) if len(g(k)) > 1 else None
    servers = '/'.join(sorted({r['server'] for r in rs}))
    return (f"| **{label} 평균** | {rs[0]['knob']} | n={len(rs)} | {servers} | | {F(m('train_loss'), 4)} | {F(m('train_acc'), 2)} | "
            f"{F(m('val_loss'), 4)} | **{F(m('val_acc'), 2)} ±{F(sd('val_acc'), 2)}** | **{K_(m('spikes'))}** ±{K_(sd('spikes')) if sd('spikes') else '–'} | "
            f"{pct(m('spikes') / base_spk)} | {E(m('lam_end'))} | {K_(m('R_end')) if m('R_end') else '–'} | {F(m('w_mean'), 3)} | "
            f"{F(m('sec_med'), 0)} | |")


HDR = ('| 팔 | ρ | seed | 서버 | best ep | train loss | train acc | val loss | val acc | spikes | spikes vs base | '
       'λ_end | R_end | w 평균 | 초/에폭 | 비고 |\n|' + '---|' * 16)


def coef_txt(c, d=3, unit=''):
    return f"{c['est']:+.{d}f}{unit} [{c['lo']:+.{d}f}, {c['hi']:+.{d}f}], p={Pv(c['p'])}"


def write_md(runs, res, both, gen):
    use = [r for r in runs if r['use']]
    base = sorted([r for r in use if r['key'] == 'base'], key=lambda r: r['seed'])
    bs = res['vs_base']['base_spk']
    nof = sorted([r for r in use if r['key'] == 'ch_nofinal' and r['knob'] == NOF_RHO], key=lambda r: r['seed'])
    nof_x = sorted([r for r in runs if r['key'] == 'ch_nofinal' and not r['use']], key=lambda r: (float(r['knob']), r['seed']))
    ours_all = [r for r in use if r['key'] == 'ours_ch']
    ours_x = [r for r in runs if r['key'] == 'ours_ch' and not r['use']]
    A = res['acc_ancova']; S = res['same_rho']; dec = S['same_spk_decomp']; Kg = res['grad_mult']
    I = res['intra']; tm = res['train_mode']; st = res['npz_struct']; T_ = res['time_axis']
    c_ls = A['lin_srv']['coef']['c']
    qm = S['same_rho_quad_mult']
    mults = [S['same_rho']['rho_mult'], qm['mult'], S['local_rho_eq']['mult']]
    mrange = f"×{min(mults):.2f}~×{max(mults):.2f}"
    tp = [p['ratio'] for p in res['time_pairs']]
    trange = f"+{(min(tp) - 1) * 100:.0f}~{(max(tp) - 1) * 100:.0f}%" if tp else '–'
    _ts = res['time_summary']
    _rt = lambda k1, k2, sv: (_ts[f'{k1}|{sv}']['median'] / _ts[f'{k2}|{sv}']['median']) if (f'{k1}|{sv}' in _ts and f'{k2}|{sv}' in _ts) else None
    r138, r23 = _rt('ch_nofinal', 'ours_ch', '138'), _rt('ch_nofinal', 'ours_ch', '23')
    sig_tm = [k for k, q in tm.items() if q['c']['p'] < 0.05]
    sig_tm_bonf = [k for k, q in tm.items() if q['c']['p'] < 0.05 / max(len(tm), 1)]
    L = []
    w = L.append
    w('# ours_inter_ch 어블레이션 — final_step (R19 CIFAR-10, T=4, 26-10-14)\n')
    w(f'- 요청 (사용자 10-14): "어블레이션 각 방법 분석도 해줄 수 있어?" — 이 문서는 **final_step** 몫. 팔 `- final_step` (run 키 `ch_nofinal`) = `ours_inter_ch` (run 키 `ours_ch`) 에서 `reg_spike_final_step` 하나만 끈 것. 교차 확인: `ours_intra_ch - final_step` (`abl_nofinal`, 제안법 intra 에서 끔), `lw1_nofinal` (런 없음 — 두 서버 모두 디렉토리 없음).')
    w(f'- 정확도 = 310 에폭 중 best val acc, spikes = 그 에폭 s_count (eval, 이미지당 원값). λ_end·R_end·L_end = 310 에폭 줄의 `adp_lambda`·`reg_R`·`task_loss_clean`. 학습 모드 지표 = `reg_detail.csv`/`reg_neuron_detail.csv` 마지막 에폭. train.log·csv 를 두 서버에서 직접 다시 읽음 ({gen}). 원장 (`inventory.json`) 은 교차검증과 S30/S1 배제 판정에만 씀.')
    w(f'- 재현: `~/anaconda3/envs/venv_1/bin/python analysis/ablation_final_step_261014.py` (세종 캐시 `scratchpad/afs1014/sejong.json.gz`, `--refetch` 로 ssh 재수집). 수치 원본 JSON: `scratchpad/afs1014/ablation_final_step_261014.json`.')
    w(f'- 앞선 분석 인용: `analysis/nofinal_vs_inter_261014.md` (같은 스파이크 비교·검정력), `ablation_mech_261010.md` 2절, `ablation_mech_addendum_261010.md` 2절.\n')

    # ---------------- 요약
    sr = S['same_rho']; lr = S['local_rho_eq']
    w('## 요약\n')
    w(f"1. **같은 스파이크 정확도: 차이 검출 안 됨 (판정: 0.08%p 보다 큰 손실·0.17%p 보다 큰 이득은 배제, 그 안은 판단 불가).** "
      f"ANCOVA (ours_inter_ch 200~420K n={A['lin_srv']['n_base']} + `- final_step` n={A['lin_srv']['n_arm']}, 선형 + 서버) 오프셋 "
      f"**{coef_txt(c_ls)}**%p. nofinal_vs_inter_261014 와 같은 값 (재계산 일치). baseline (n={res['n']['base']}) 대비 "
      f"{res['vs_base']['acc']['diff']:+.3f}%p [{res['vs_base']['acc']['lo']:+.3f}, {res['vs_base']['acc']['hi']:+.3f}] · 스파이크 "
      f"{(math.exp(res['vs_base']['lnspk']['diff']) - 1) * 100:+.1f}%.")
    w(f"2. **final_step 이 바꾸는 것은 '규제 세기 눈금' 이다.** 같은 ρ 에서 `- final_step` 의 스파이크는 ours 의 "
      f"**{pct(sr['spk_ratio'])}** [{pct(sr['spk_ratio_lo'])}, {pct(sr['spk_ratio_hi'])}] (n={sr['fit']['n_base']}/{sr['fit']['n_arm']}, p={Pv(sr['fit']['coef']['c']['p'])}). "
      f"ρ 로 환산하면 `- final_step` ρ 2.2e-3 = ours ρ **{2.2 * qm['mult']:.1f}e-3** 상당, 배율 **×{qm['mult']:.2f}** [{qm['lo']:.2f}, {qm['hi']:.2f}] "
      f"(ln ρ–ln s 곡선이 휘어 (2차 항 p={Pv(S['same_rho_quad']['coef']['q']['p'])}) 2차 곡선을 주결과로 씀; 직선 ×{sr['rho_mult']:.2f}, 국소 ρ 4e-3~5.5e-3 직선 ×{lr['mult']:.2f} → 모형에 따라 {mrange}).")
    w(f"3. **ρ 배율의 분해 (같은 스파이크, n={dec['lam_end']['fit']['n_base']}/{dec['lam_end']['fit']['n_arm']}).** "
      f"λ_end 는 `- final_step` 이 ours 의 {dec['lam_end']['ratio']:.3f} 배 [{dec['lam_end']['lo']:.3f}, {dec['lam_end']['hi']:.3f}] — 같은 스파이크에 도달하는 데 λ 가 "
      f"**1/{1 / dec['lam_end']['ratio']:.2f}** 만 든다. R_end 는 {dec['R_end']['ratio']:.3f} 배 [{dec['R_end']['lo']:.3f}, {dec['R_end']['hi']:.3f}], "
      f"L_end 는 {dec['L_end']['ratio']:.3f} 배. 곱 λ·R/L = {dec['check_identity']['lam_x_R_over_L']:.3f} = 1/{1 / dec['check_identity']['lam_x_R_over_L']:.2f} — 같은 스파이크에서의 ρ 비 (런마다 λ_end = ρ·L/R 이라 항등식). "
      f"즉 **λ 한 단위가 주는 압력이 약 {1 / dec['lam_end']['ratio']:.1f} 배** 이고, 제어기는 R 이 {dec['R_end']['ratio']:.2f} 배 커진 만큼만 λ 를 깎아 순 효과가 ρ 눈금 {mrange} 로 남는다 (회귀 방향·곡률에 따라).")
    if Kg:
        w(f"4. **압력 {1 / dec['lam_end']['ratio']:.1f} 배의 원인 = 스텝별 노름의 분모.** final_step 은 층마다 ‖(ΣS)⊙w‖ 하나로 나누고, 끄면 스텝마다 ‖S_t⊙w_t‖ 로 나눈다. "
          f"한 스텝의 노름은 전체의 일부라 같은 λ 에서 뉴런당 기울기가 커진다. npz_t 스텝 몫 (`- final_step` s1, n=1) 과 층별 R 로 추정한 배율 K = "
          f"**{Kg['K_spk']:.2f}** (스파이크 가중) / {Kg['K_uni']:.2f} (스텝 균등) vs 실측 λ 비 **{Kg['K_meas_lam_end']:.2f}** [{Kg['K_meas_lo']:.2f}, {Kg['K_meas_hi']:.2f}] — 실측은 스파이크 가중 추정과 같고 스텝 균등 추정은 CI 밖. "
          f"스파이크 가중 K 로 예측한 ρ 배율 ×{Kg['pred_rho_mult_spk']:.2f} vs 실측 ×{qm['mult']:.2f} (직선 ×{sr['rho_mult']:.2f}). "
          f"가장 적게 쏘는 t1 (몫 {Kg['stage_conv2']['p'][0] * 100:.1f}%, conv2) 에서 뉴런당 기울기가 t4 의 약 {Kg['stage_conv2']['t1_vs_t4_push']:.1f} 배 (conv2) — t1 을 가장 세게 누른다.")
    w(f"5. **10-10 문서 정정 1건.** addendum 2절의 '이른 스텝 스파이크가 남은 스텝 수만큼 여러 번 벌점에 들어간다' 는 코드와 다르다. "
      f"`spike_reg = spike` (그 스텝 S_t) 라 스파이크 하나는 자기 스텝에서 **한 번만** 벌점을 받는다 (neurons.py:1132-1135). 이른 스텝을 더 누르는 것은 반복 벌점이 아니라 위 4 의 작은 분모 때문이다. "
      f"방향 예측 (t1 몫↓, 뒤 스텝↑) 은 addendum 의 관측 방향과 같다.")
    tw = res['time_pairs']
    if tw:
        rr = ', '.join(f"{p['server']} {p['start']} ×{p['ratio']:.3f}" for p in tw)
        w(f"6. **학습 시간 {trange}** (같은 시각에 시작한 짝 {len(tw)}묶음, 에폭당 중앙값 비: {rr}; 310 에폭 총 시간 +{min(p['total_h']['nofinal'] - p['total_h']['final_on'] for p in tw):.1f}~{max(p['total_h']['nofinal'] - p['total_h']['final_on'] for p in tw):.1f}시간). 규제 블록이 T 번 (매 스텝) 돌기 때문 — final_step 은 t=T 에서 1 번.")
    w(f"7. **intra 에서도 부호가 같다.** `abl_nofinal` (n={I['same_rho']['fit']['n_arm']}) 은 같은 ρ 에서 스파이크 {pct(I['same_rho']['spk_ratio'])} "
      f"[{pct(I['same_rho']['lo'])}, {pct(I['same_rho']['hi'])}], ρ 배율 ×{I['same_rho']['rho_mult']:.2f}; 같은 스파이크 정확도 "
      f"{coef_txt(I['acc_lin_srv']['coef']['c'])}%p (ours_intra_ch n={I['acc_lin_srv']['n_base']}).")
    w(f"8. **같은 스파이크에서 학습 모드 분포 지표는 구분되지 않는다** (n=3, 3c 절: 16개 중 p<0.05 {len(sig_tm)}개, 16개 보정 시 {len(sig_tm_bonf)}개) — 10-10 결론 (n=1) 이 n=3 으로 유지된다. "
      f"npz 채널 구조 (n=1, 3d 절) 는 '덜 집중' 방향 하나가 보인다 (conv3 죽은 채널 {st['dead_conv3']['resid']:+.0f}, z={st['dead_conv3']['z']:+.1f}; 채널 지니 z={st['gini_ch']['z']:+.1f}, 상위 10% 채널 몫 z={st['top10_ch_pct']['z']:+.1f}) — n=1 이라 가설.\n")

    w(f"**판정**: final_step 은 정확도-스파이크 곡선을 움직인다는 증거가 없고 (n=3, 오프셋 95% CI [{c_ls['lo']:+.2f}, {c_ls['hi']:+.2f}]%p), ρ 눈금 (×{qm['mult']:.1f}, 모형에 따라 {mrange}) 과 학습 시간 ({trange}) 을 바꾼다. "
      f"기전은 \"벌점의 분모를 층마다 T 합의 노름 하나로 둔다\" 는 것 — 끄면 스텝별 노름이 분모가 되어 같은 λ 에서 약 {1 / dec['lam_end']['ratio']:.1f} 배 세지고, 스파이크가 가장 적은 t1 의 뉴런당 기울기가 t4 의 2~5배가 된다. "
      f"논문 ablation 에는 \"같은 스파이크에서 정확도 차 검출 안 됨 (n=3, CI), ρ 눈금 ×{qm['mult']:.1f}, 학습 시간 {trange}\" 로 쓰고, \"final_step 이 정확도에 기여한다\" 는 쓸 수 없다.\n")

    # ---------------- 정의
    w('## 정의\n')
    w('- **final_step (켜짐, 제안법)**: 규제 블록이 t=T 에서 한 번만 돈다. 벌점 = λ·Σ_l ‖(Σ_t S_t)⊙w_l‖₂, w = 1 − c/max c (c = 채널 공간합, t=T 의 누적 발화 + 침묵 뉴런의 막전위 준비도). backward (wta_rev) 는 모든 뉴런·모든 t 에 λ·w²/R_l 을 준다 (R_l = 그 층의 노름).')
    w('- **`- final_step` (꺼짐)**: 규제 블록이 매 스텝 돈다. 벌점 = λ·Σ_l Σ_t ‖S_t⊙w_{l,t}‖₂, w_{l,t} 는 t 까지의 누적 발화 (+ t 의 준비도) 로 매 스텝 다시 계산. 스텝 t 의 뉴런은 λ·w_t²/R_{l,t} 를 받는다 (R_{l,t} = 그 스텝만의 노름).')
    w('- **R**: loss-ratio 가 쓰는 원값 벌점. 켜짐 R = Σ_l ‖(ΣS)⊙w‖, 꺼짐 R = Σ_l Σ_t ‖S_t⊙w_t‖ (sc_loss_snap 이 스텝마다 누적, neurons.py:1213-1218). **λ_{e+1} = ρ·L_e/R_e** (L = 규제 뺀 과제 손실).')
    w('- **ρ 배율**: `- final_step` 이 ρ 에서 착지한 스파이크를 ours 가 내려면 필요한 ρ 의 비 (ln s = a + b·ln ρ + c·[nofinal] + e·[23] 에서 exp(c/b), Fieller 구간).')
    w('- **기울기 배율 K**: 같은 스파이크·같은 w 에서 λ 한 단위가 뉴런에 주는 기울기의 비 (꺼짐/켜짐) = E[R_{l,f}/R_{l,t}]. 같은 압력이면 λ_f/λ_n = K 여야 한다.')
    w('- **같은 스파이크 차 (ANCOVA)**: y = a + b·ln(spikes) + c·[nofinal] + e·[서버 23], ours_inter_ch 200~420K 완주·비배제. c 가 오프셋. 정확도 축은 nofinal_vs_inter_261014 와 같은 모형.')
    w('- **이미지당 침묵 %** (학습 모드): 한 이미지에서 T 동안 한 번도 안 쏜 뉴런 비율 (배치 평균, 뉴런 수 가중). **지니·상위 10%**: 층 안 뉴런 스파이크 집중도 (층 스파이크 가중). **죽은 채널** (npz): 테스트 1만 장 합이 0 인 채널.\n')

    # ---------------- 0. 런 표
    w('## 0. 런별 값 (best 에폭)\n')
    w('열: train loss (규제 포함) → train acc → val loss → val acc → spikes, 뒤에 파생 (baseline 대비 스파이크, λ_end, R_end, 학습 모드 w 평균, 에폭당 초 중앙값). 평균 행은 배제 행을 뺀 값, ± 는 시드 SD.\n')
    w(HDR)
    for r in base:
        w(run_row(r, bs))
    w(mean_row(base, 'baseline', bs))
    for r in nof:
        w(run_row(r, bs))
    w(mean_row(nof, '- final_step', bs))
    for r in nof_x:
        w(run_row(r, bs, r['exclude_reason'] + (' — 진행 중, 평균·검정 제외' if r['seed'] < 900 else '')))
    for k in sorted({r['knob'] for r in ours_all}, key=float):
        g = sorted([r for r in ours_all if r['knob'] == k], key=lambda r: r['seed'])
        for r in g:
            w(run_row(r, bs, '' if BAND[0] <= r['spikes'] <= BAND[1] else '200~420K 밖 (곡선 제외)'))
        w(mean_row(g, 'ours_inter_ch', bs))
    for r in sorted(ours_x, key=lambda r: (float(r['knob']), r['seed'])):
        w(run_row(r, bs))
    w('')
    nchk = [r for r in runs if r.get('inv_check') and r['use']]
    bad = [r['tag'] for r in nchk if (r['inv_check']['val_acc_diff'] or 0) != 0 or abs(r['inv_check']['spikes_diff'] or 0) > 1
           or not r['inv_check']['server_match']]
    w(f"- 원장 교차검증: 쓰인 런 {len(nchk)}개 중 val acc·spikes·서버 불일치 {len(bad)}건{(' (' + ', '.join(bad) + ')') if bad else ''}. 두 서버에 같은 이름 디렉토리가 있는 런 {len(both)}개는 원장 서버(138)를 따름 ({', '.join(both) if both else '없음'}).")
    w(f"- 설정 차이 (override 블록, 경로·시드 제외): " + '; '.join(f"{k} vs {v['ref']}: {', '.join(f'{kk} {vv[0]}→{vv[1]}' for kk, vv in v['diff'].items()) or '없음'}" for k, v in res['conf_diff'].items() if k in [r['tag'] for r in nof]) + '. 두 팔의 차이는 `reg_spike_final_step` 과 ρ 뿐.\n')

    # ---------------- 1. 같은 스파이크 정확도
    w('## 1. 같은 스파이크 정확도 (nofinal_vs_inter_261014 와 같은 방법, 재계산)\n')
    w('| 모형 | n (ours / nofinal) | 오프셋 c (%p) | 95% CI | p | 기울기 b (%p / ln) | 서버 e | 잔차 SD |\n|---|---|---|---|---|---|---|---|')
    for nm, lab in (('lin', '선형'), ('lin_srv', '선형 + 서버 (주결과)'), ('quad_srv', '2차 + 서버'),
                    ('lin_srv_no_s1', '선형 + 서버, s1 제외 (사전 등록 민감도)')):
        f = A[nm]; c = f['coef']['c']; b = f['coef']['b']; e = f['coef'].get('e')
        nn = f"{f.get('n_base') or f['n'] - 3} / {f.get('n_arm') or 3}"
        w(f"| {lab} | {nn} | **{c['est']:+.3f}** | [{c['lo']:+.3f}, {c['hi']:+.3f}] | {Pv(c['p'])} | {b['est']:+.3f} ±{b['se']:.3f} | "
          f"{(F(e['est'], 3, True) + ' (p=' + Pv(e['p']) + ')') if e else '–'} | {f['resid_sd']:.3f} |")
    w('')
    pw = res['power']
    w(f"- 같은 ρ 시드 간 합동 SD = **{res['pooled_sd']['sd']:.3f}%p** (ours_inter_ch, 자유도 {res['pooled_sd']['df']}). 관측 오프셋 {c_ls['est']:+.3f}%p 를 검정력 0.8 로 보려면 조건당 n≈{pw['n_for_obs']}. "
      f"ours 곡선을 지금대로 두고 `- final_step` 만 늘릴 때 최소 검출 차: n=3 {pw['mde_by_n']['3']:.2f}, n=4 {pw['mde_by_n']['4']:.2f}, n=6 {pw['mde_by_n']['6']:.2f}, n=10 {pw['mde_by_n']['10']:.2f}, 하한 {pw['floor']:.2f}%p.")
    w(f"- s1 제외 (n=2) 는 사전 등록 규칙 (10-14) 의 민감도 확인용이고 n≤2 라 **판단 불가**. 부호는 같다.")
    vb = res['vs_base']
    w(f"- baseline 대비 (Welch, `- final_step` n={vb['acc']['na']} vs baseline n={vb['acc']['nb']}): 정확도 {vb['acc']['diff']:+.3f}%p [{vb['acc']['lo']:+.3f}, {vb['acc']['hi']:+.3f}] p={Pv(vb['acc']['p'])}, "
      f"스파이크 {(math.exp(vb['lnspk']['diff']) - 1) * 100:+.1f}% [{(math.exp(vb['lnspk']['lo']) - 1) * 100:+.1f}, {(math.exp(vb['lnspk']['hi']) - 1) * 100:+.1f}]. baseline {vb['base_acc']:.2f}% @ {K_(vb['base_spk'])}.")
    w('- 지배 관계·창 비교·같은 정확도 스파이크 비는 nofinal_vs_inter_261014.md 1·2·3절 (같은 데이터) 을 인용한다: 창 ±5/8/12% Δ +0.085/+0.053/+0.042 (전부 p≥0.19), 같은 정확도 스파이크 차 −10.4% [−35.9, +21.7], 시드 단위 지배는 양방향.\n')

    # ---------------- 2. 같은 ρ / λ
    w('## 2. 같은 ρ · 같은 λ 에서 스파이크 — 규제 세기 눈금\n')
    fr = sr['fit']
    w(f"ours 는 200~420K 완주·비배제 n={fr['n_base']} (ρ {', '.join(res['ours_band_rhos'])}), `- final_step` ρ 2.2e-3 n={fr['n_arm']}.\n")
    w('| 비교 | 모형 | `- final_step` / ours 스파이크 | 95% CI | p | 환산 배율 |\n|---|---|---|---|---|---|')
    w(f"| 같은 ρ | ln s ~ ln ρ + 팔 + 서버 | **{sr['spk_ratio']:.3f}** ({pct(sr['spk_ratio'])}) | [{sr['spk_ratio_lo']:.3f}, {sr['spk_ratio_hi']:.3f}] | {Pv(fr['coef']['c']['p'])} | ρ ×{sr['rho_mult']:.2f} [{sr['rho_mult_lo']:.2f}, {sr['rho_mult_hi']:.2f}] (기울기 b={fr['coef']['b']['est']:.3f} ±{fr['coef']['b']['se']:.3f}) |")
    fq_ = S['same_rho_quad']
    w(f"| 같은 ρ (2차 곡선, 주결과) | ln s ~ ln ρ + (ln ρ)² + 팔 + 서버 | {math.exp(fq_['coef']['c']['est']):.3f} ({pct(math.exp(fq_['coef']['c']['est']))}) | [{math.exp(fq_['coef']['c']['lo']):.3f}, {math.exp(fq_['coef']['c']['hi']):.3f}] | {Pv(fq_['coef']['c']['p'])} | **ρ ×{qm['mult']:.2f}** [{qm['lo']:.2f}, {qm['hi']:.2f}] (계수 공분산에서 {qm['n_draw']}회 추출) |")
    for lk, lab in (('lam_end', 'λ_end (310 에폭)'), ('lam_mean', 'λ 평균 (310 에폭)')):
        q = S[f'same_{lk}']
        w(f"| 같은 {lab} | ln s ~ ln λ + 팔 + 서버 | {q['spk_ratio']:.3f} ({pct(q['spk_ratio'])}) | [{q['lo']:.3f}, {q['hi']:.3f}] | {Pv(q['fit']['coef']['c']['p'])} | – |")
    w('')
    fq = S['same_rho_quad']
    w(f"- ln ρ 2차 항: q={fq['coef']['q']['est']:+.3f} (p={Pv(fq['coef']['q']['p'])}); 2차 모형의 팔 차 c={fq['coef']['c']['est']:+.3f} (선형 {fr['coef']['c']['est']:+.3f}). "
      f"국소 직선 (ours ρ {', '.join(lr['rhos'])}, n={lr['n']}) 으로 역산하면 ρ_eq = {lr['rho_eq'] * 1e3:.2f}e-3, 배율 ×{lr['mult']:.2f}.")
    w(f"- ρ 배율은 모형에 따라 {mrange} (직선 전역 ×{sr['rho_mult']:.2f}, 2차 ×{qm['mult']:.2f}, 국소 ×{lr['mult']:.2f}). 직선 전역은 곡률을 무시해 작게 나온다. 착지 스파이크의 시드 산포 (ln SD 약 2%) 를 기울기 (−0.4) 로 나누면 ρ 환산에 ±5% 가 붙는다.")
    w(f"- 10-10 (n=1) 의 '같은 ρ −22% (z=−8.0), 같은 λ −25%' 는 n=3 으로 {pct(sr['spk_ratio'])}·{pct(S['same_lam_mean']['spk_ratio'])} 로 **유지**. 10-13/10-14 결정 기록의 '약 2배 (ours 4.7e-3 상당, 2.1배)' 는 2차·국소 값과 맞는다.\n")
    w('### 2b. 같은 스파이크에서 λ · R · L — ρ 배율의 분해\n')
    w(f"ln y ~ ln(spikes) + 팔 + 서버 (ours 200~420K n={dec['lam_end']['fit']['n_base']}, `- final_step` n={dec['lam_end']['fit']['n_arm']}). 비 = `- final_step` / 같은 스파이크의 ours.\n")
    w('| 양 | 비 | 95% CI | p | 뜻 |\n|---|---|---|---|---|')
    lab = {'lam_end': ('λ_end', '같은 스파이크에 필요한 λ'), 'lam_mean': ('λ 평균', '학습 전체 평균'),
           'R_end': ('R_end (reg_R)', '원값 벌점 — R 정의가 다름'), 'L_end': ('L_end (과제 손실)', ''),
           'R_rd': ('R (reg_detail 합)', 'R_end 와 같은 양, 배치 평균 csv'), 'tr_spk': ('학습 모드 스파이크', 'eval 스파이크와의 관계')}
    for k in ('lam_end', 'lam_mean', 'R_end', 'R_rd', 'L_end', 'tr_spk'):
        q = dec[k]
        w(f"| {lab[k][0]} | **{q['ratio']:.3f}** | [{q['lo']:.3f}, {q['hi']:.3f}] | {Pv(q['p'])} | {lab[k][1]} |")
    w('')
    ci = dec['check_identity']
    w(f"- 항등식 검산: 런마다 λ_end = ρ·L_end/R_end 이므로 (λ 비)·(R 비)/(L 비) = {ci['lam_x_R_over_L']:.3f} = 1/{1 / ci['lam_x_R_over_L']:.2f} 는 'ln ρ 를 ln s 에 회귀한' ρ 비와 같다 (OLS 는 y 에 선형). 같은 ρ 회귀 (ln s ~ ln ρ) 의 직선 1/×{sr['rho_mult']:.2f} = {ci['rho_ratio_from_mult']:.3f} 와 맞고, 곡률을 넣은 ×{qm['mult']:.2f} 보다는 조금 작다 — 회귀 방향·곡률 차이.")
    w(f"- 읽는 법: 같은 스파이크에서 `- final_step` 은 λ 를 **{1 / dec['lam_end']['ratio']:.2f} 배 적게** 쓰고도 같은 곳에 착지한다 → λ 한 단위의 압력이 그만큼 크다 (3a 절). "
      f"한편 R 은 {dec['R_end']['ratio']:.2f} 배라 loss-ratio 가 λ 를 1/{dec['R_end']['ratio']:.2f} 로 줄인다. 압력 ×{1 / dec['lam_end']['ratio']:.2f} 을 λ ÷{dec['R_end']['ratio']:.2f} 가 일부 상쇄 → ρ 눈금 ×{1 / ci['lam_x_R_over_L']:.2f} (같은 스파이크 회귀; 같은 ρ 회귀로는 {mrange}).")
    w(f"- 10-14 결정 기록의 '같은 스파이크에서 λ_f/λ_nf ≈ 2.7' 은 이 표의 1/{dec['lam_end']['ratio']:.3f} = {1 / dec['lam_end']['ratio']:.2f} (n=3) 로 확인된다.\n")

    # 런별 λ·R 표
    w('런별 (310 에폭 줄; 같은 스파이크 근처 ours 와 나란히):\n')
    w('| 팔 | ρ | seed | 서버 | spikes | λ_end | λ 평균 | R_end | L_end | λ·R/L (= ρ) |\n|---|---|---|---|---|---|---|---|---|---|')
    near = sorted([r for r in ours_all if 250e3 <= r['spikes'] <= 320e3], key=lambda r: -r['spikes'])
    for r in nof + near:
        w(f"| {ARM[r['key']]} | {r['knob']} | s{r['seed']} | {r['server']} | {K_(r['spikes'])} | {E(r['lam_end'])} | {E(r['lam_mean'])} | {K_(r['R_end'])} | {r['L_end']:.4f} | {r['lam_end'] * r['R_end'] / r['L_end']:.2e} |")
    w('')

    # ---------------- 3. 기전
    w('## 3. 기전\n')
    w('### 3a. final_step 한 플래그가 바꾸는 네 가지\n')
    w('| | final_step 켜짐 (ours_inter_ch) | 꺼짐 (`- final_step`) | 확인 |\n|---|---|---|---|')
    w('| (1) 언제 | 규제 블록이 t=T 에서 1 번 (neurons.py:804) | 매 스텝 T=4 번 | 학습 시간 4절 |')
    w('| (2) 무엇에 벌점 | ΣS (0~4 정수, T 합) — `self.out` TensorArray 합 (1132-1133) | S_t (그 스텝의 0/1) (1135). 스파이크 하나는 자기 스텝에서 한 번만 벌점 | 코드 |')
    w('| (3) 가중 w 의 근거 | t=T 의 전체 누적 발화 + t=T 막전위 준비도 | t 까지의 **부분 누적** 발화 + t 의 준비도, 매 스텝 다시 계산 | 3d 절 (부분 순위) |')
    w(f"| (4) 분모·ρ 눈금 | R_l = ‖(ΣS)⊙w‖ 하나; λ = ρL/R | R_{{l,t}} = ‖S_t⊙w_t‖ 스텝별; R = Σ_t R_{{l,t}}; λ = ρL/R | 2절 (ρ ×{qm['mult']:.2f}), 3b |")
    w('')
    if Kg:
        w('### 3b. 기울기 배율 K — 스텝별 노름이 압력을 얼마나 키우나\n')
        w('wta_rev backward 는 뉴런 (발화 여부 무관) 에 λ·w²/(그 노름) 을 준다. 켜짐은 모든 t 가 같은 R_{l,f} 로 나뉘고, 꺼짐은 스텝 t 가 R_{l,t} 로 나뉜다. '
          'w_t ≈ w (발화 뉴런의 평균 w² 가 스텝마다 같다) 라고 두면 R_{l,t} ∝ √(그 스텝 스파이크 수) 이므로 R_{l,t} = R_{l,n}·√p_t / Σ_s √p_s (p_t = 층의 스텝 몫, npz_t). '
          '그러면 K_l = (R_{l,f}/R_{l,n})·(Σ√p)² (스파이크 가중 평균) 또는 (R_{l,f}/R_{l,n})·(Σ√p)·mean_t(1/√p_t) (스텝 균등 — 비발화 뉴런까지 기울기를 받는 wta_rev 에 더 가깝다). '
          f'R_{{l,n}} = `- final_step` 3런 평균 (reg_detail), R_{{l,f}} = ours 200~420K 층별 ln R ~ ln s 직선의 같은 스파이크 값, p_t = `- final_step` s1 npz_t (n=1, 테스트셋 eval).\n')
        w('| 단계 | 스텝 몫 t1/t2/t3/t4 (%) | K (스파이크 가중) | K (스텝 균등) | 뉴런당 기울기 t1 ÷ t4 |\n|---|---|---|---|---|')
        for s in STAGES:
            q = Kg[f'stage_{s}']
            w(f"| {s} | {' / '.join(f'{x * 100:.1f}' for x in q['p'])} | {q['K_spk']:.2f} | {q['K_uni']:.2f} | {q['t1_vs_t4_push']:.2f} |")
        w(f"| **전체 (층 스파이크 가중)** | | **{Kg['K_spk']:.2f}** | **{Kg['K_uni']:.2f}** | |")
        w('')
        w(f"- 실측 (2b 절): λ_f/λ_n = **{Kg['K_meas_lam_end']:.2f}** [{Kg['K_meas_lo']:.2f}, {Kg['K_meas_hi']:.2f}] (n=3). **스파이크 가중 추정 {Kg['K_spk']:.2f} 와 같고**, 스텝 균등 추정 {Kg['K_uni']:.2f} 은 CI 밖이다. "
          f"이 추정으로 예측한 ρ 배율 K·(R_f/R_n)·(L_n/L_f) = ×{Kg['pred_rho_mult_spk']:.2f} (스파이크 가중) / ×{Kg['pred_rho_mult_uni']:.2f} (스텝 균등) vs 실측 ×{qm['mult']:.2f} (2차) · ×{sr['rho_mult']:.2f} (직선).")
        w('- 해석 (검증 안 됨): 착지점은 모든 뉴런에 고르게 주는 기울기가 아니라 **그 스텝에 발화하는 (문턱 근처) 뉴런이 받는 기울기** 로 정해지는 것으로 보인다. 대리 기울기가 문턱 근처에서만 크다는 것과 맞는 해석이다. '
          '이 추정의 입력 (스텝 몫 p_t 는 npz_t, R 비는 reg_detail) 은 착지 스파이크와 독립이라, 일치는 순환 논리가 아니다.')
        w(f"- R_{{l,f}}/R_{{l,n}} (층 스파이크 가중) = {Kg['Rf_over_Rn_w']:.3f} — 같은 스파이크에서 스텝별 노름의 합이 T 합의 노름보다 {1 / Kg['Rf_over_Rn_w']:.2f} 배 크다 (삼각부등식 Σ_t‖x_t‖ ≥ ‖Σ_t x_t‖ 의 실제 크기).")
        w('- 그래서 "같은 ρ 에서 2배 세다" 는 효율 차이가 아니라 분모 정의의 차이다. 두 팔의 ρ 는 같은 단위가 아니다.')
        w('- 한계: w_t ≈ w 가정은 검증하지 못했다 (스텝별 R_{l,t} 와 w_t 는 로그에 없다). t1 은 부분 누적이 작아 침묵 뉴런의 준비도가 w 를 주로 정한다. p_t 는 eval (테스트셋) 몫이고 n=1.\n')

    w('### 3c. 학습 모드 지표 — 같은 스파이크에서 무엇이 다른가 (n=3, 10-10 은 n=1)\n')
    w(f"y ~ ln(spikes) + 팔 + 서버, ours 200~420K vs `- final_step` 2.2e-3. c = `- final_step` − 같은 스파이크 ours. 마지막 열은 10-10 과 같은 s1 단독 z (새 관측 예측 SE 기준).\n")
    w('| 지표 | ours 예측 @ nofinal 스파이크 | `- final_step` 3런 | c | 95% CI | p | s1 단독 z (10-10 방식) |\n|---|---|---|---|---|---|---|')
    labs = {'w_mean': 'w 평균 (18층, 뉴런 가중)', 'dead_pct': '이미지당 침묵 뉴런 %', 'gini_sw': '지니 (층 스파이크 가중)',
            'top10_sw': '상위 10% 뉴런 몫 %', 'share_conv1': '층 몫 conv1 %', 'share_conv2': '층 몫 conv2 %',
            'share_conv3': '층 몫 conv3 %', 'share_conv4': '층 몫 conv4 %', 'w_conv1': 'w conv1', 'w_conv2': 'w conv2',
            'w_conv3': 'w conv3', 'w_conv4': 'w conv4', 'dead_conv1': '침묵 % conv1', 'dead_conv2': '침묵 % conv2',
            'dead_conv3': '침묵 % conv3', 'dead_conv4': '침묵 % conv4'}
    for k, q in tm.items():
        c = q['c']
        d = 3 if k.startswith(('w_', 'gini')) else 2
        z1 = q.get('s1_z')
        w(f"| {labs.get(k, k)} | {q['ours_at_nof']:.{d}f} | {' / '.join(f'{v:.{d}f}' for v in q['nof_vals'])} | **{c['est']:+.{d}f}** | [{c['lo']:+.{d}f}, {c['hi']:+.{d}f}] | {Pv(c['p'])} | {F(z1['z'], 1, True) if z1 else '–'} |")
    w('')
    sig = [k for k, q in tm.items() if q['c']['p'] < 0.05]
    w(f"- p<0.05 인 지표: {', '.join(labs.get(k, k) + ' (' + format(tm[k]['c']['est'], '+.3f') + ')' for k in sig) if sig else '없음'}. "
      f'16개를 검정했으니 우연으로도 약 0.8개가 p<0.05 로 나온다; 16개 보정 (p<0.0031) 을 넘는 지표는 {len(sig_tm_bonf)}개. s1 단독 z (10-10 방식) 는 모두 |z|≤1.4 — 10-10 결론과 같다.')
    w('- `reg_detail` 의 firing_rate 열은 두 팔에서 정의가 다르다 (켜짐: ΣS/T 평균, 꺼짐: 마지막 스텝 S_T 평균) — 쓰지 않았다. sc_loss 열도 정의가 달라 (위 R) 비교는 2b 절의 비로만 한다.\n')

    if st:
        w('### 3d. npz 채널 구조 (테스트 1만 장, `- final_step` s1 n=1)\n')
        w(f"ours 는 200~420K 중 npz 있는 {len(st['ours_tags'])}런으로 ln(spikes) 직선; `- final_step` s1 (원장 {K_(st['nof_ledger_spk'])}) 의 잔차 z 는 새 관측 하나의 예측 SE 기준 (자유도 {st['dead_ch']['df']}).\n")
        w('| 지표 | ours 예측 | `- final_step` s1 | 잔차 | z | p | baseline (n=' + str(st['base_n']) + ') |\n|---|---|---|---|---|---|---|')
        lab3 = {'dead_ch': '죽은 채널 (/4,736)', 'gini_ch': '채널 지니', 'top10_ch_pct': '상위 10% 채널 몫 %',
                'active_img_pct': '이미지당 활성 뉴런 %', 'silent_neu_pct': '테스트셋 침묵 뉴런 %',
                'dead_conv1': '죽은 채널 conv1', 'dead_conv2': '죽은 채널 conv2', 'dead_conv3': '죽은 채널 conv3', 'dead_conv4': '죽은 채널 conv4'}
        for k in ('dead_ch', 'gini_ch', 'top10_ch_pct', 'active_img_pct', 'silent_neu_pct', 'dead_conv1', 'dead_conv2', 'dead_conv3', 'dead_conv4'):
            q = st[k]
            bm = st['base_mean'].get(k)
            d = 3 if 'gini' in k else (2 if 'pct' in k else 1)
            w(f"| {lab3[k]} | {q['pred']:.{d}f} | {q['obs']:.{d}f} | {q['resid']:+.{d}f} | {q['z']:+.1f} | {Pv(q['p'])} | {F(bm, d) if bm is not None else '–'} |")
        w('')
        on = st['npz_over_ledger']
        w(f"- npz/원장 스파이크 비: `- final_step` s1 {on['nof']:.3f}, ours {np.mean(on['ours']):.3f} ±{np.std(on['ours'], ddof=1):.3f} (n={len(on['ours'])}) — 같은 정의로 비교 가능. "
          f"_npz 와 _npz_t 에 둘 다 있는 런 (ours 1.5e-3 s1) 의 층별 총합 최대 상대차 {st.get('npz_consistency_max_rel', float('nan')):.1e}.")
        w('- n=1 이라 |z|<2 는 시드 산포 안이다. 방향으로만 읽는다.\n')

    w('### 3e. 시간축 (npz_t) — addendum 재계산 + 정정\n')
    w('| 묶음 | n | npz 스파이크/이미지 | t1 % | t2 % | t3 % | t4 % | 평균 최초 발화 스텝 | conv1 t4 몫 % |\n|---|---|---|---|---|---|---|---|---|')
    for g, q in T_.items():
        if g in ('trend',) or g.startswith('partial'):
            continue
        w(f"| {g} | {q['n']} | {K_(q['spk_per_img'])} | " + ' | '.join(f'{x:.2f}' for x in q['step_share']) + f" | {q['mean_first_t']:.3f} | {q['share_t_conv1'][3]:.2f} |")
    w('')
    w('스파이크 수준 추세 (baseline n=3 → ours 1.5e-3 n=3 직선의 외삽) 대비 잔차 (addendum 과 같은 방법):\n')
    w('| 지표 | `- final_step` 잔차 ± SE (z) | 검사점 `- vmem` (final_step 켜짐) 잔차 ± SE (z) |\n|---|---|---|')
    tl = {'step1': 't1 몫 %', 'step4': 't4 몫 %', 'mean_first_t': '평균 최초 발화 스텝', 'last_conv1': 't4 몫 conv1 %',
          'last_conv2': 't4 몫 conv2 %', 'last_conv3': 't4 몫 conv3 %', 'last_conv4': 't4 몫 conv4 %',
          'first1_conv1': 't1 몫 conv1 %', 'first1_conv2': 't1 몫 conv2 %', 'first1_conv3': 't1 몫 conv3 %', 'first1_conv4': 't1 몫 conv4 %'}
    for k, o in T_['trend'].items():
        a_, b_ = o['- final_step 2.2e-3'], o['- vmem 3e-3 (final_step 켜짐)']
        d = 3 if k == 'mean_first_t' else 2
        w(f"| {tl[k]} | {a_['resid']:+.{d}f} ±{a_['se']:.{d}f} ({a_['z']:+.1f}) | {b_['resid']:+.{d}f} ±{b_['se']:.{d}f} ({b_['z']:+.1f}) |")
    w('')
    w('- addendum 결과 재현: t4 잔차 +0.41 ±0.38 (z≈+1), conv1 t4 +1.03 (z≈+6) — 위 표와 같다. 방향 (꺼지면 뒤 스텝으로) 은 3b 의 예측 (작은 분모의 t1 을 가장 세게 누름) 과 맞는다. 크기는 n=1·외삽이라 **판단 불가**.')
    w('- **정정**: addendum 2절 셋째 줄 "이른 스텝 스파이크가 남은 스텝 수만큼 여러 번 벌점에 들어간다" → 틀림. 꺼짐에서 벌점 대상은 그 스텝의 S_t 라 스파이크마다 한 번이다. '
      '이른 스텝이 더 눌리는 이유는 t1 의 노름 R_{l,1} 이 작아 (스파이크 몫 2~7%) 뉴런당 기울기가 t4 의 2~5배라서다 (3b 표 마지막 열).')
    if 'partial_rank_nofinal' in T_:
        pr = T_['partial_rank_nofinal']; p15 = T_.get('partial_rank_ours15_s1')
        w('\n부분 누적 순위 — 꺼짐에서 스텝 t 의 가중은 t 까지의 누적으로 정해진다. 이미지 합산 채널 누적 합 C_t 와 C_T 의 비교 (17 conv 층):\n')
        w('| t | Spearman(C_t, C_T) 중앙값 (최소) | 평균 \\|w_t − w_T\\| | C_t=0 채널 비율 | ours 1.5e-3 s1 (켜짐 망) 중앙값 |\n|---|---|---|---|---|')
        for t in sorted(pr, key=int):
            q = pr[t]; q2 = p15[t] if p15 else None
            w(f"| {t} | {q['spearman_med']:.3f} ({q['spearman_min']:.3f}) | {q['mean_abs_dw']:.3f} | {q['zero_ch_frac'] * 100:.1f}% | {q2['spearman_med']:.3f} |" if q2 else
              f"| {t} | {q['spearman_med']:.3f} ({q['spearman_min']:.3f}) | {q['mean_abs_dw']:.3f} | {q['zero_ch_frac'] * 100:.1f}% | – |")
        w('')
        w(f"- 이미지 합산으로 t=1 의 채널 순위는 T 의 순위와 꽤 다르고 (Spearman 중앙값 {pr[1]['spearman_med']:.2f}, 최소 {pr[1]['spearman_min']:.2f}, 평균 |w_1 − w_T| {pr[1]['mean_abs_dw']:.2f}), t=2 부터 거의 같다 ({pr[2]['spearman_med']:.2f}). "
          f"final_step 이 켜진 망 (ours 1.5e-3 s1) 도 t=1 순위 상관이 {p15[1]['spearman_med']:.2f} 로 비슷해, 이것은 팔의 결과가 아니라 T=4 SNN 의 발화 시간 구조다. C_t=0 비율에는 죽은 채널 (약 11%) 이 들어 있다. "
          '실제 가중은 **이미지별** 이고 이미지 하나의 t=1 발화는 훨씬 성기므로 이 표는 순위 차이를 과소평가한다 — 이미지별 값은 npz 에 없다. 즉 꺼짐에서 t1·t2 벌점의 가중은 최종 순위와 다른 채널을 겨냥한다. '
          '채널 공간합 경쟁 (inter) 은 t=1 에서도 값이 연속적이라, final_step 의 원래 근거 (neurons.py:801 주석, "t=1 에서 max=1 이라 1-maxnorm 이 0/1 이분법으로 무너진다") 는 뉴런 단위 max (intra·층 전체) 에 해당하고 inter 에는 약하게만 적용된다.\n')

    w('### 3f. intra 교차 확인 — `ours_intra_ch - final_step` (`abl_nofinal`) vs `ours_intra_ch` (`prop`)\n')
    anf = sorted([r for r in use if r['key'] == 'abl_nofinal'], key=lambda r: (float(r['knob']), r['seed']))
    prop = sorted([r for r in use if r['key'] == 'prop'], key=lambda r: (float(r['knob']), r['seed']))
    w(HDR)
    for k in sorted({r['knob'] for r in prop}, key=float):
        g = [r for r in prop if r['knob'] == k]
        for r in g:
            w(run_row(r, bs))
        w(mean_row(g, 'ours_intra_ch', bs))
    for k in sorted({r['knob'] for r in anf}, key=float):
        g = [r for r in anf if r['knob'] == k]
        for r in g:
            w(run_row(r, bs))
        w(mean_row(g, 'ours_intra_ch - final_step', bs))
    for r in [r for r in runs if r['key'] in ('prop', 'abl_nofinal') and not r['use']]:
        w(run_row(r, bs))
    w('')
    si = I['same_rho']; fi = si['fit']
    w('| 비교 | inter (`- final_step` vs ours_inter_ch) | intra (`abl_nofinal` vs ours_intra_ch) | 부호 일치 |\n|---|---|---|---|')
    w(f"| n (기준 / 끈 팔) | {sr['fit']['n_base']} / {sr['fit']['n_arm']} | {fi['n_base']} / {fi['n_arm']} | |")
    w(f"| 같은 ρ 스파이크 | {pct(sr['spk_ratio'])} [{pct(sr['spk_ratio_lo'])}, {pct(sr['spk_ratio_hi'])}] | {pct(si['spk_ratio'])} [{pct(si['lo'])}, {pct(si['hi'])}] | {'예' if (sr['spk_ratio'] - 1) * (si['spk_ratio'] - 1) > 0 else '아니오'} |")
    w(f"| ρ 배율 | ×{sr['rho_mult']:.2f} [{sr['rho_mult_lo']:.2f}, {sr['rho_mult_hi']:.2f}] | ×{si['rho_mult']:.2f} [{si['rho_mult_lo']:.2f}, {si['rho_mult_hi']:.2f}] | |" if si['rho_mult_lo'] else
      f"| ρ 배율 | ×{sr['rho_mult']:.2f} | ×{si['rho_mult']:.2f} (구간 유계 아님) | |")
    ca, ci_ = A['lin_srv']['coef']['c'], I['acc_lin_srv']['coef']['c']
    w(f"| 같은 스파이크 정확도 (선형+서버) %p | {coef_txt(ca)} | {coef_txt(ci_)} | {'예' if ca['est'] * ci_['est'] > 0 else '부호 다름'} — 둘 다 0 포함, 판단 불가 |")
    for k, labk in (('lam_end', 'λ_end'), ('R_end', 'R_end')):
        a_ = dec[k]; b_ = I[f'same_spk_{k}']
        w(f"| 같은 스파이크 {labk} 비 | {a_['ratio']:.3f} [{a_['lo']:.3f}, {a_['hi']:.3f}] | {b_['ratio']:.3f} [{b_['lo']:.3f}, {b_['hi']:.3f}] (n={b_['n'][0]}/{b_['n'][1]}) | {'예' if (a_['ratio'] - 1) * (b_['ratio'] - 1) > 0 else '아니오'} |")
    w('')
    w(f"- intra 는 ours_intra_ch 범위 {K_(I['prop_range'][0])}~{K_(I['prop_range'][1])} (ρ {', '.join(I['prop_rhos'])}) 전부를 곡선으로 씀. `abl_nofinal` 4e-3 s2 (155,422) 는 이 범위 아래라 약간 외삽.")
    w(f"- 같은 ρ 효과는 intra 가 더 작다 (ρ 배율 ×{si['rho_mult']:.2f} vs ×{sr['rho_mult']:.2f}). intra 의 w 는 채널 안 위치 경쟁이라 t 마다 다시 계산한 w_t 가 inter 보다 0/1 에 가깝고 (t=1 에서 위치 max=1) 분모 외의 효과가 섞인다 — 크기 차이의 원인은 이 자료로 가르지 못한다.")
    w(f"- `lw1_nofinal` (ours_layer_w1 에서 끔): run_paper.py 에 방법은 있으나 두 서버 모두 런 디렉토리가 없다 (n=0). 교차 확인 불가.\n")

    # ---------------- 4. 학습 시간
    w('## 4. 학습 시간 (에폭당 초, train.log 의 `- NNNs`, 첫 에폭 제외 중앙값)\n')
    w('### 4a. 같은 시각에 시작한 짝 (같은 서버, 시작 3분 이내) — 서버 부하가 같은 비교\n')
    w('| 서버 | 시작 | 겹친 구간 (h) | 런 | GPU | 구간 안 에폭 | 구간 중앙값 (초) | 전체 중앙값 | 에폭 |\n|---|---|---|---|---|---|---|---|---|')
    for p in res['time_pairs']:
        for x in p['runs']:
            w(f"| {p['server']} | {p['start']} | {p['window_h']:.1f} | {x['tag']} | {x['gpu']} | {x['ep_in_win']} | {x['med']:.0f} | {x['med_all']:.0f} | {x['epochs']} |")
        th = p['total_h']
        w(f"| | | | **`- final_step` / 켜짐 비** | | | **×{p['ratio']:.3f}** | | 310 에폭 총 시간: 꺼짐 {F(th['nofinal'], 1)}h, 켜짐 {F(th['final_on'], 1)}h |")
    w('')
    w('- 짝의 켜짐 쪽은 `- vmem`·`- loss-ratio`·`- 1- inversion` (모두 final_step 켜짐) 이다. ours_inter_ch 와 같은 시각에 시작한 `- final_step` 런은 없다. `- vmem` 은 준비도 계산을 건너뛰어 조금 가볍지만 세종 짝에서 `- loss-ratio`·`- 1- inversion` (준비도 계산 있음) 과 같은 수준이라 그 몫은 작다.')
    w('\n### 4b. 팔·서버별 (완주 + 진행 중 런, 런별 중앙값의 중앙값)\n')
    w('| 팔 | 서버 | n | 중앙값 (초) | 범위 |\n|---|---|---|---|---|')
    for k, v in res['time_summary'].items():
        key, srv = k.split('|')
        w(f"| {ARM.get(key, key)} | {srv} | {v['n']} | {v['median']:.0f} | {v['min']:.0f}~{v['max']:.0f} |")
    w('')
    ts = res['time_summary']
    def ratio(k1, k2, s):
        a, b = ts.get(f'{k1}|{s}'), ts.get(f'{k2}|{s}')
        return a['median'] / b['median'] if a and b else None
    r138, r23 = ratio('ch_nofinal', 'ours_ch', '138'), ratio('ch_nofinal', 'ours_ch', '23')
    w(f"- 팔 중앙값 비 `- final_step` / ours_inter_ch: 138 ×{F(r138, 3)}, 23 ×{F(r23, 3)} (런마다 시작 시각·서버 부하가 달라 짝 비교 4a 보다 거칠다). "
      f"intra (`abl_nofinal` / ours_intra_ch, 138): ×{F(ratio('abl_nofinal', 'prop', '138'), 3)}.")
    w('- 진행 중 `- final_step` 런 (1e-3·1.4e-3·1.6e-3·1.8e-3·7e-4·2.8e-3) 은 시간 측정에만 넣었다 (완주 전이라도 에폭당 시간은 유효). canus 의 이 런들은 GPU 6·7 포함 8장이 모두 찬 상태에서 돌아 서버 부하가 4a 짝과 다르다.\n')

    # ---------------- 5. 판정
    w('## 5. 판정 — 무엇을 말할 수 있나\n')
    w('| 질문 | 판정 | 근거 (n) |\n|---|---|---|')
    w(f"| final_step 을 끄면 같은 스파이크에서 정확도가 달라지나 | **판단 불가** (차이가 ±0.17%p 안이면 볼 수 없음). 0.08%p 보다 큰 손실·0.17%p 보다 큰 이득은 배제 | c={c_ls['est']:+.3f} [{c_ls['lo']:+.3f}, {c_ls['hi']:+.3f}], n=27/3 |")
    w(f"| 같은 ρ 에서 더 줄이나 | **판정 가능 — 예** ({pct(sr['spk_ratio'])}, ρ ×{qm['mult']:.2f} [{qm['lo']:.2f}, {qm['hi']:.2f}]; 모형에 따라 {mrange}) | n=27/3, p={Pv(sr['fit']['coef']['c']['p'])} |")
    w(f"| 왜 (λ·R) | **판정 가능** — 같은 스파이크에 λ 1/{1 / dec['lam_end']['ratio']:.2f}, R ×{dec['R_end']['ratio']:.2f} | n=27/3 |")
    w(f"| 왜 (기울기 분모) | **크기 일치, 가정 미검증** — 스파이크 가중 K {Kg.get('K_spk', float('nan')):.2f} vs 실측 λ 비 {Kg.get('K_meas_lam_end', float('nan')):.2f} [{Kg.get('K_meas_lo', float('nan')):.2f}, {Kg.get('K_meas_hi', float('nan')):.2f}] | 스텝 몫 n=1, w_t≈w 가정 |")
    sig_txt = ', '.join(labs.get(k, k) for k in sig) if sig else '없음'
    w(f"| 같은 스파이크에서 분포 (w·침묵·지니·상위10%·층 몫) | **차이 검출 안 됨** (16개 중 p<0.05: {sig_txt}; 16개 보정 시 {len(sig_tm_bonf)}개) | n=27/3 |")
    w(f"| 채널 구조 (죽은 채널·활성 뉴런) | **판단 불가** (n=1). 방향: 덜 집중 (conv3 죽은 채널 z={st['dead_conv3']['z']:+.1f}, 채널 지니 z={st['gini_ch']['z']:+.1f}), 이미지당 활성 뉴런은 같음 (z={st['active_img_pct']['z']:+.1f}) | n=1 |")
    w('| 시간축 (뒤 스텝으로 밀림) | **방향만** (t4 z≈+1, conv1 z≈+6 이나 외삽·n=1) | n=1 |')
    w(f"| 학습 시간 | **판정 가능 — {trange}** (같은 시각 짝 {len(res['time_pairs'])}묶음) | 런 {sum(len(p['runs']) for p in res['time_pairs'])}개 |" if res['time_pairs'] else '| 학습 시간 | 짝 없음 | |')
    w(f"| intra 와 부호 일치 | **예** (같은 ρ 감소·같은 스파이크 λ 감소·R 증가; 정확도는 둘 다 0 근처) — 크기는 intra 가 작음 (ρ ×{si['rho_mult']:.2f}) | intra n={fi['n_base']}/{fi['n_arm']} |")
    w('| ours_layer_w1 에서 | **판단 불가** — 런 없음 | n=0 |')
    w('')

    w('## 6. 10-10 문서에서 바뀌는 것\n')
    w('| 위치 | 10-10 | 지금 (10-14) |\n|---|---|---|')
    w(f"| ablation_mech 2a '같은 ρ −22% (z=−8.0), 같은 λ −25% (z=−10.0)' (n=1) | n=1 z | n=3: 같은 ρ {pct(sr['spk_ratio'])}, 같은 λ 평균 {pct(S['same_lam_mean']['spk_ratio'])}, 같은 λ_end {pct(S['same_lam_end']['spk_ratio'])} — **유지** |")
    w(f"| 2a '같은 스파이크 정확도 +0.06 ±0.10' (n=1) | n=1 | n=3: {c_ls['est']:+.3f} [{c_ls['lo']:+.3f}, {c_ls['hi']:+.3f}] — **유지** (0 포함) |")
    w(f"| 2a '층 배분·분포·w 는 같은 스파이크에서 구분되지 않는다 (\\|z\\|≤1.3)' | n=1 | n=3 ANCOVA (3c) — **유지** (16개 중 p<0.05 {len(sig_tm)}개: {sig_txt}; 16개 보정 시 {len(sig_tm_bonf)}개) |")
    w("| 2a '같은 ρ 에서 λ 평균 0.863' | 같은 ρ 비 | 같은 ρ 비는 착지 스파이크가 달라 해석이 어렵다 → 같은 스파이크 비 (2b: λ_end " + f"{dec['lam_end']['ratio']:.3f}) 로 바꿈 |")
    w("| 2a 주의 'final_step 을 끄면 R 정의가 바뀐다 (매 스텝 누적 스파이크로 가중을 다시 계산해 스텝마다 벌점)' | 정성 | **보완**: 벌점 대상은 누적이 아니라 그 스텝 S_t, 가중만 누적으로 다시 계산. 세기 차이의 주원인은 스텝별 노름 분모 (3b) |")
    w("| addendum 2절 '이른 스텝 스파이크가 남은 스텝 수만큼 여러 번 벌점' | 기전 설명 | **정정**: 스파이크당 벌점 1회. 이른 스텝은 노름이 작아 뉴런당 기울기가 큼 (3b) |")
    w(f"| ablation_mech 2a 해석 '요소로서의 근거는 설계 논리이고 수치로는 보이지 않는다' | | 정확도로는 유지. 수치로 보이는 것: ρ 눈금 ×{qm['mult']:.1f}, 학습 시간 {trange} |")
    w(f"| 10-14 결정 기록 '학습 시간 약 +16% (전체 평균), 같은 시각 짝 +17~19%' | | 재측정: 짝 {trange}, 팔 중앙값 비 138 ×{F(r138, 3)} · 23 ×{F(r23, 3)} — **유지** |")
    w('')

    w('## [LIMITATION]\n')
    w('1. `- final_step` 은 **한 ρ (2.2e-3, 약 −42% 스파이크), n=3**. 다른 스파이크 수준 (특히 −30% 구간) 은 진행 중인 6런 (각 n=1~2) 이 완주해야 곡선으로 볼 수 있다. 10-14 사전 등록 규칙 (d 의 95% 하한 > 0 일 때만 final_step 제거) 은 그 런들로 판정한다 — 이 문서는 판정하지 않는다.')
    w('2. 같은 스파이크 오프셋 CI [−0.08, +0.17]%p 안의 차이는 이 설계로 볼 수 없다. 조건당 n≤4 규칙 아래서는 하한이 0.17%p 근처다.')
    w('3. 기울기 배율 K 는 w_t ≈ w 와 R_{l,t} ∝ √(스텝 스파이크 수) 를 가정한 추정이다. 스텝별 R_{l,t}·w_t 는 로그에 없고, 스텝 몫은 eval (테스트셋) 의 n=1 이다. 실측 λ 비가 스파이크 가중 추정과 맞는다는 것은 일치이지 증명이 아니다.')
    w('4. 학습 모드 지표는 마지막 에폭 배치 평균, 정확도·스파이크는 best 에폭 eval 이다. 지표 16개를 보정 없이 검정했다.')
    w('5. npz 채널 구조·시간축은 `- final_step` s1 하나 (n=1) 이고, 이미지 합산이라 이미지별 부분 순위를 재지 못한다.')
    w('6. 학습 시간 짝 비교의 켜짐 쪽은 ours_inter_ch 가 아니라 같은 시각의 다른 어블레이션 팔 (final_step 켜짐) 이다. 서버·GPU 부하가 시점마다 달라 팔 전체 중앙값 비는 거칠다.')
    w('7. intra 교차 확인은 다른 방법 (채널 안 경쟁) 이고, `abl_nofinal` 은 ρ 3개 n=4 (같은 ρ 반복 2런은 4e-3 뿐). ours_layer_w1 판은 런이 없다.')
    w('8. spikes 는 s_count 원값이다. s_count 와 npz 총합의 불일치 (09-15 보류) 는 두 팔에 같은 정의를 쓰므로 비율 비교에는 영향이 작다고 보지만 해소되지 않았다.')
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inv', default=DEF_INV)
    ap.add_argument('--cache23', default=DEF_CACHE)
    ap.add_argument('--refetch', action='store_true')
    a = ap.parse_args()
    runs, both, inv = build(a.inv, a.cache23, a.refetch)
    res = analyze(runs, inv)
    gen = datetime.now().strftime('%Y-%m-%d %H:%M')
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    slim_runs = [{k: (v.strftime('%Y-%m-%d %H:%M') if isinstance(v, datetime) else v) for k, v in r.items()
                  if k not in ('secs', 'conf', 'R_layer', 'spk_layer')} for r in runs]
    json.dump({'generated': gen, 'runs': slim_runs, 'res': res}, open(OUT_JSON, 'w'), indent=1, ensure_ascii=False,
              default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, (np.floating, np.integer)) else str(o)))
    open(OUT_MD, 'w').write(write_md(runs, res, both, gen))
    print('wrote', OUT_MD, OUT_JSON)


if __name__ == '__main__':
    main()
