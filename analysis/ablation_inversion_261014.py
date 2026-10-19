#!/usr/bin/env python
"""R19 CIFAR-10 (ResNet-19, T=4): ours_inter_ch 의 '1- inversion' 요소 분석 — 26-10-14.

사용자 요청 (10-14): "어블레이션 각 방법 분석도 해줄 수 있어?" 중 inversion 몫.
팔: `- 1- inversion (inter)` = run 키 ch_noinv = ours_ch 에서 reg_spike_maxnorm_no_inv 만 True
    (w = c/max c 대신 1 − c/max c 를 쓰지 않음. c = 채널 공간합, vmem readiness 포함).
교차 확인: intra 쪽 abl_noinv (maxnorm_plain — 반전·within_channel·vmem 세 요소 동시 제거, decisions 09-20),
          abl_noinv_wc / lw1_noinv (단일 요소판 — 큐에 투입된 적 없음, 0런).

수치 정의 (원장·논문과 같음)
  val acc  = 310 에폭 중 최고 val_acc (같은 값이면 첫 에폭)
  spikes   = 그 에폭의 s_count (원값, 이미지당, 평가 모드)
  train loss / train acc / val loss = 그 에폭 값. train loss 는 규제 항 포함 'loss'.
값은 train.log 에서 직접 다시 읽는다 (원장 inventory.json 은 S30/S1 배제 판정과 교차검증에만 쓴다).
  138 = 로컬 _paper/<tag>/train.log (HDD 링크), reg_detail/reg_neuron_detail = /media/hdd1/kyccj/EIP/paper/<tag>/
  23  = sejong /srv2/kyccj/EIP/paper/<tag>/ 에서 ssh 로 읽기만 (에폭 줄·override 블록·host.txt·CSV)
팔 정체는 config_sweep.py 의 run_paper override 블록으로 확인한다.

쓰는 법:
  ~/anaconda3/envs/venv_1/bin/python analysis/ablation_inversion_261014.py [--refetch]
산출: analysis/ablation_inversion_261014.md (수치 원본 JSON 은 scratchpad/inv1014/ablation_inversion_261014.json)
"""
import argparse, glob, gzip, io, json, math, os, re, subprocess, tarfile
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
SCR = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
INV = SCR + '/led1007b/inventory.json'
CACHE = SCR + '/inv1014'
OUT_JSON = SCR + '/inv1014/ablation_inversion_261014.json'   # 과제 지시: analysis/ 에는 .py·.md 만
OUT_MD = ROOT + '/analysis/ablation_inversion_261014.md'
STORE138 = '/media/hdd1/kyccj/EIP/paper'
NPZ_DIRS = [ROOT + '/_npz', SCR + '/deep1005/npz']
PRUNE_DIRS = [SCR + '/prune_out', SCR + '/deep1005/prune']
EPOCHS = 310
PATTERNS = ['r19c10-base---s1[1-6]', 'r19c10-ours_ch-[0-9]*', 'r19c10-ch_noinv-*', 'r19c10-abl_noinv-*',
            'r19c10-prop-*']
NAME = re.compile(r'^r19c10-(base|ours_ch|ch_noinv|abl_noinv|prop)-(.*)-s(\d+)$')
EPOCH_KV = re.compile(r'([A-Za-z_][\w\-]*): ([\d.]+(?:[eE][+-]?\d+)?)')
TIME_RE = re.compile(r' - (\d+)s (\d+)(?:ms|s|us)/step - loss:')
ASSIGN = re.compile(r"^\s*conf\.([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.M)
ACC_RANGE = (200e3, 420e3)       # 정확도 ANCOVA 의 ours_ch 범위 (nofinal_vs_inter_261014 와 같음)
MECH_BAND = (180e3, 330e3)       # 학습 모드 기전 지표의 ours_ch 비교 대역 (ablation_mech_261010 4절과 같음)
NPZ_BAND = (150e3, 420e3)        # npz 구조 지표 곡선 대역
STAGES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1']
NEU = {}
for _l in ['conv1_conv_n'] + [f'conv2_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')]:
    NEU[_l] = 32 * 32 * 128
for _l in [f'conv3_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')]:
    NEU[_l] = 16 * 16 * 256
for _l in [f'conv4_block{b}_{s}_n' for b in (1, 2) for s in ('conv1', 'out')]:
    NEU[_l] = 8 * 8 * 512
NEU['fc1_n'] = 256
LABEL = {'base': 'baseline', 'ours_ch': 'ours_inter_ch', 'ch_noinv': '- 1- inversion (inter)',
         'prop': 'ours_intra_ch', 'abl_noinv': 'ours_intra_ch − (반전·within_channel·vmem)'}

REMOTE_TXT = ('cd /srv2/kyccj/EIP/paper && for d in ' + ' '.join(PATTERNS) + '; do '
              '[ -d "$d" ] || continue; echo "@@RUN $d"; '
              '[ -f "$d/train.log" ] && tr "\\r" "\\n" < "$d/train.log" | grep "val_loss"; '
              'echo "@@CONF $d"; '
              'sed -n "/# >>> run_paper overrides/,/# <<< run_paper overrides/p" '
              '~/PycharmProjects/TensorFlow-SNNs/_paper/$d/config_sweep.py 2>/dev/null; '
              'echo "@@HOST $d"; cat "$d/host.txt" 2>/dev/null; echo; '
              'echo "@@MTIME $d"; stat -c %Y "$d/train.log" 2>/dev/null; done')
REMOTE_CSV = ('cd /srv2/kyccj/EIP/paper && tar czf - $(for d in ' + ' '.join(PATTERNS) + '; do '
              'for f in reg_detail.csv reg_neuron_detail.csv; do [ -f "$d/$f" ] && echo "$d/$f"; done; done)')


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


def parse_epoch_lines(lines):
    rows = []
    for ln in lines:
        k = ln.find('loss: ')
        if 'val_loss' not in ln or k < 0:
            continue
        d = {}
        for key, v in EPOCH_KV.findall(ln[k:]):
            try:
                d[key] = float(v)
            except ValueError:
                pass
        m = TIME_RE.search(ln)
        d['sec'] = float(m.group(1)) if m else float('nan')
        if all(x in d for x in ('loss', 'acc', 'val_loss', 'val_acc', 's_count')):
            rows.append(d)
    return rows


def parse_csv(txt):
    """epoch,layer,... → {epoch: {layer: {col: val}}} (같은 에폭 중복이면 마지막 값)."""
    lines = [l for l in txt.replace('\r', '\n').split('\n') if l.strip()]
    if not lines:
        return {}
    cols = lines[0].split(',')
    out = defaultdict(dict)
    for ln in lines[1:]:
        p = ln.split(',')
        if len(p) != len(cols) or p[0] == 'epoch':
            continue
        try:
            e = int(p[0])
            out[e][p[1]] = {c: float(v) for c, v in zip(cols[2:], p[2:])}
        except ValueError:
            continue
    return dict(out)


def host_server(h):
    h = (h or '').strip().lower()
    return '138' if h.startswith('canus') else ('23' if h.startswith('sejong') else None)


def load_local():
    out = {}
    for pat in PATTERNS:
        for d in sorted(glob.glob(os.path.join(ROOT, '_paper', pat))):
            tag = os.path.basename(d)
            lg, cf = os.path.join(d, 'train.log'), os.path.join(d, 'config_sweep.py')
            if not os.path.exists(lg):
                continue
            txt = open(lg, errors='ignore').read().replace('\r', '\n')
            st = os.path.join(STORE138, tag)
            rec = {'rows': parse_epoch_lines(l for l in txt.split('\n') if 'val_loss' in l),
                   'conf': parse_conf(open(cf, errors='ignore').read()) if os.path.exists(cf) else {},
                   'host': open(st + '/host.txt').read().strip() if os.path.exists(st + '/host.txt') else '',
                   'mtime': os.path.getmtime(lg), 'src': 'local'}
            for nm in ('reg_detail', 'reg_neuron_detail'):
                p = os.path.join(st, nm + '.csv')
                rec[nm] = parse_csv(open(p).read()) if os.path.exists(p) else {}
            out[tag] = rec
    return out


def load_sejong(refetch):
    os.makedirs(CACHE, exist_ok=True)
    ft, fc = CACHE + '/ep23.txt.gz', CACHE + '/csv23.tar.gz'
    ssh = ['ssh', '-n', '-p', '23456', '-o', 'BatchMode=yes', 'kyccj@161.122.23.23']
    if refetch or not os.path.exists(ft):
        r = subprocess.run(ssh + [REMOTE_TXT], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        with gzip.open(ft, 'wb') as f:
            f.write(r.stdout)
    if refetch or not os.path.exists(fc):
        r = subprocess.run(ssh + [REMOTE_CSV], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        open(fc, 'wb').write(r.stdout)
    txt = gzip.open(ft, 'rt', errors='ignore').read()
    buf, tag, mode = {}, None, None
    for ln in txt.split('\n'):
        if ln.startswith('@@'):
            kind, tag = ln[2:].split(' ', 1)
            tag = tag.strip()
            mode = kind
            buf.setdefault(tag, {'RUN': [], 'CONF': '', 'HOST': '', 'MTIME': ''})
            continue
        if tag is None:
            continue
        if mode == 'RUN':
            buf[tag]['RUN'].append(ln)
        else:
            buf[tag][mode] += ln + '\n'
    csvs = defaultdict(dict)
    with tarfile.open(fc, 'r:gz') as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            t, f = m.name.split('/', 1)
            csvs[t][f.replace('.csv', '')] = parse_csv(tf.extractfile(m).read().decode(errors='ignore'))
    out = {}
    for tag, b in buf.items():
        mt = b['MTIME'].strip()
        out[tag] = {'rows': parse_epoch_lines(b['RUN']), 'conf': parse_conf(b['CONF']), 'host': b['HOST'].strip(),
                    'mtime': float(mt) if mt else None, 'src': 'sejong',
                    'reg_detail': csvs.get(tag, {}).get('reg_detail', {}),
                    'reg_neuron_detail': csvs.get(tag, {}).get('reg_neuron_detail', {})}
    return out


def arm_of(conf):
    """override 블록 → 팔. 단일 요인 차이만 허용 (나머지 설정이 다르면 'other')."""
    if not conf:
        return 'no_conf'
    if not conf.get('reg_spike_out', False):
        return 'base'
    common = (conf.get('reg_spike_final_step') is True and conf.get('reg_spike_loss_ratio') is True
              and float(conf.get('reg_spike_loss_ratio_start_ep', 0) or 0) == 0
              and float(conf.get('reg_spike_loss_ratio_end_ep', 0) or 0) == 0
              and float(conf.get('reg_spike_vmem_gain', 1.0)) == 1.0
              and conf.get('reg_spike_vmem_silent_only') is True
              and conf.get('reg_spike_wta_rev_w1', False) is False
              and float(conf.get('reg_spike_shape_beta', 1.0)) == 1.0
              and not conf.get('reg_spike_out_bpsr', False)
              and conf.get('reg_spike_out_wta_rev') is True and conf.get('reg_spike_out_sc') is True
              and conf.get('reg_spike_accum_loss', False) is False)
    if not common:
        return 'other'
    if conf.get('reg_spike_out_sc_maxnorm_plain') is True and conf.get('reg_spike_out_sc_maxnorm') is False:
        return 'abl_noinv'
    if conf.get('reg_spike_out_sc_maxnorm') is True and not conf.get('reg_spike_out_sc_maxnorm_plain', False):
        g, ni = conf.get('reg_spike_maxnorm_group'), conf.get('reg_spike_maxnorm_no_inv', False) is True
        if g == 'channel':
            return 'ch_noinv' if ni else 'ours_ch'
        if g == 'within_channel':
            return 'abl_noinv_wc' if ni else 'prop'
    return 'other'


def layer_metrics(rd, rn, ep):
    """reg_detail / reg_neuron_detail 의 한 에폭 (0-기준) → 층 집계."""
    if ep not in rd:
        return None
    lay = [l for l in NEU if l in rd[ep]]
    if len(lay) != len(NEU):
        return None
    spk = {l: rd[ep][l]['spike_count'] for l in lay}
    w = {l: rd[ep][l]['sc_rate_mean'] for l in lay}
    tot = sum(spk.values())
    nN = sum(NEU.values())
    o = {'tr_spk': tot, 'R_rd': sum(rd[ep][l]['sc_loss'] for l in lay),
         'w_mean': sum(w[l] * NEU[l] for l in lay) / nN}
    for s in STAGES:
        ls = [l for l in lay if l.startswith(s)]
        o[f'share_{s}'] = 100 * sum(spk[l] for l in ls) / tot
        o[f'w_{s}'] = sum(w[l] * NEU[l] for l in ls) / sum(NEU[l] for l in ls)
    if rn and ep in rn and all(l in rn[ep] for l in lay):
        d = rn[ep]
        o['dead_pct'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in lay) / nN
        o['gini_sw'] = sum(d[l]['gini'] * spk[l] for l in lay) / tot
        o['top10_sw'] = 100 * sum(d[l]['top10_share'] * spk[l] for l in lay) / tot
        o['wsd_mean'] = sum(d[l]['sc_rate_std'] * NEU[l] for l in lay) / nN
        for s in STAGES:
            ls = [l for l in lay if l.startswith(s)]
            ss = sum(spk[l] for l in ls)
            o[f'dead_{s}'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in ls) / sum(NEU[l] for l in ls)
            o[f'gini_{s}'] = sum(d[l]['gini'] * spk[l] for l in ls) / ss if ss > 0 else float('nan')
            o[f'wsd_{s}'] = sum(d[l]['sc_rate_std'] * NEU[l] for l in ls) / sum(NEU[l] for l in ls)
        o['_per_layer'] = {l: {'w': w[l], 'wsd': d[l]['sc_rate_std'], 'spk': spk[l]} for l in lay}
    return o


def add_cn(o, noinv):
    """채널 정규화 합 m = mean(c/max c) 와 그 산포 — 두 팔에서 같은 양.
    ours_ch: w = 1 − c/max → m = 1 − w. ch_noinv: w = c/max → m = w. sc_rate_std 는 std(1−x)=std(x) 라 양쪽 같은 양."""
    if o is None:
        return o
    f = (lambda v: v) if noinv else (lambda v: 1.0 - v)
    o['cn_mean'] = f(o['w_mean'])
    for s in STAGES:
        o[f'cn_{s}'] = f(o[f'w_{s}'])
    if '_per_layer' in o:
        pl = o['_per_layer']
        nN = sum(NEU.values())
        cv = {l: (pl[l]['wsd'] / f(pl[l]['w']) if f(pl[l]['w']) > 0 else float('nan')) for l in pl}
        o['cn_cv'] = sum(cv[l] * NEU[l] for l in pl) / nN
        for s in STAGES:
            ls = [l for l in pl if l.startswith(s)]
            o[f'cn_cv_{s}'] = sum(cv[l] * NEU[l] for l in ls) / sum(NEU[l] for l in ls)
    return o


def build_runs(refetch):
    loc = load_local()
    sj = load_sejong(refetch)
    inv = {r['tag']: r for r in json.load(open(INV))['runs']}
    tags = sorted(set(loc) | set(sj))
    runs, dups = [], []
    for tag in tags:
        m = NAME.match(tag)
        if not m:
            continue
        cands = [c for c in (loc.get(tag), sj.get(tag)) if c is not None]
        withrows = [c for c in cands if c['rows']]
        if len(withrows) > 1:
            dups.append({'tag': tag, 'epochs': {c['src']: len(c['rows']) for c in withrows}})
        if not withrows:
            continue
        c = max(withrows, key=lambda c: len(c['rows']))
        conf = c['conf'] or next((x['conf'] for x in cands if x['conf']), {})
        key, knob, seed = m.group(1), m.group(2).replace('p', '.'), int(m.group(3))
        server = host_server(c['host']) or ('138' if c['src'] == 'local' else '23')
        arm = arm_of(conf)
        rho = conf.get('reg_spike_loss_ratio_target') if arm not in ('base',) else None
        rows = c['rows']
        rec = {'tag': tag, 'key': key, 'knob': '-' if key == 'base' else knob, 'seed': seed, 'server': server,
               'src': c['src'], 'host': c['host'], 'mtime': c['mtime'], 'arm_conf': arm,
               'rho': None if rho is None else float(rho), 'epochs': len(rows)}
        b = max(rows, key=lambda x: x['val_acc'])
        rec.update(best_epoch=rows.index(b) + 1, train_loss=b['loss'], train_acc=b['acc'] * 100,
                   val_loss=b['val_loss'], val_acc=b['val_acc'] * 100, spikes=b['s_count'])
        rec['traj'] = {k: [r.get(k, float('nan')) for r in rows]
                       for k in ('s_count', 'val_acc', 'adp_lambda', 'reg_R', 'task_loss_clean', 'sec', 'loss')}
        if 'adp_lambda' in rows[-1]:
            lam = np.array(rec['traj']['adp_lambda'])
            rec['lam_mean'], rec['lam_end'] = float(lam.mean()), float(lam[-1])
        sec = np.array(rec['traj']['sec'][1:], float)
        rec['sec_med'] = float(np.nanmedian(sec)) if np.isfinite(sec).any() else None
        rec['s30'] = rows[29]['s_count'] / rows[0]['s_count'] if len(rows) >= 30 else None
        rd, rn = c['reg_detail'], c['reg_neuron_detail']
        last = max(rd) if rd else None
        rec['csv_last_ep'] = last
        rec['mech'] = add_cn(layer_metrics(rd, rn, last), key == 'ch_noinv') if last is not None else None
        rec['mech_traj'] = {}
        for e in (0, 9, 29, 59, 99, 149, 194, 199, 249, 309):
            mm = add_cn(layer_metrics(rd, rn, e), key == 'ch_noinv')
            if mm:
                mm.pop('_per_layer', None)
                rec['mech_traj'][e] = mm
        if rec['mech']:
            rec['mech'].pop('_per_layer', None)
        iv = inv.get(tag, {})
        reasons = []
        if len(rows) < EPOCHS:
            if seed >= 900:
                reasons.append(f'연기시험({len(rows)}ep)')
            elif tag == 'r19c10-ch_noinv-2p4e-2-s1':
                reasons.append(f'중단({len(rows)}ep, 10-06 사용자 지시)')
            else:
                reasons.append(f'미완주({len(rows)}ep)')
        elif len(rows) > EPOCHS:
            reasons.append(f'에폭 줄 과다({len(rows)})')
        if key != 'base' and (rho is None or abs(float(knob) - float(rho)) > 1e-12):
            reasons.append(f'ρ 불일치(dir={knob}, conf={rho})')
        if 'S30/S1' in (iv.get('exclude_reason') or ''):
            reasons.append([x for x in iv['exclude_reason'].split('; ') if 'S30' in x][0])
        if arm != key:
            reasons.append(f'팔 불일치(dir={key}, conf={arm})')
        rec['exclude_reason'] = '; '.join(reasons)
        rec['use'] = not reasons
        if iv and iv.get('val_acc') is not None:
            rec['inv_check'] = {'val_acc_diff': round(rec['val_acc'] - iv['val_acc'], 4),
                                'spikes_diff': round(rec['spikes'] - iv['spikes'], 1),
                                'server_match': iv.get('server') == server,
                                's30_diff': None if (iv.get('s30') is None or rec['s30'] is None)
                                else round(rec['s30'] - iv['s30'], 4)}
        runs.append(rec)
    return runs, dups, inv


# ============================================================================ 통계 도구
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
    co = {}
    for i, nm in enumerate(names):
        t = beta[i] / se[i]
        co[nm] = {'est': float(beta[i]), 'se': float(se[i]), 'lo': float(beta[i] - tcrit(df) * se[i]),
                  'hi': float(beta[i] + tcrit(df) * se[i]), 't': float(t), 'p': float(2 * stats.t.sf(abs(t), df))}
    return {'coef': co, 'df': df, 'n': n, 'resid_sd': math.sqrt(s2), 'beta': beta, 'cov': cov, 'names': names}


def line_pred(x, y, x0):
    """y = a + b·x 적합 → x0 에서 예측값, 평균 SE, 새 런 하나 예측 SE, 자유도, 잔차 SD, 기울기."""
    f = ols(np.c_[np.ones(len(x)), x], y, ['a', 'b'])
    v = np.array([1.0, x0])
    m = float(v @ f['beta'])
    sm = float(np.sqrt(v @ f['cov'] @ v))
    return {'pred': m, 'se_mean': sm, 'se_new': float(math.sqrt(sm ** 2 + f['resid_sd'] ** 2)), 'df': f['df'],
            'resid_sd': f['resid_sd'], 'slope': float(f['beta'][1]), 'slope_se': f['coef']['b']['se'], 'n': f['n']}


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    t = d / se
    return {'diff': float(d), 'se': se, 'df': float(df), 't': float(t), 'p': float(2 * stats.t.sf(abs(t), df)),
            'lo': float(d - tcrit(df) * se), 'hi': float(d + tcrit(df) * se), 'na': len(a), 'nb': len(b)}


def pval_t(z, df):
    return float(2 * stats.t.sf(abs(z), df))


def cond_means(rs, fval):
    """ρ 별 평균 (ln 값). 반환 [(ρ, 평균, n)] — ρ 오름차순."""
    by = defaultdict(list)
    for r in rs:
        by[r['rho']].append(fval(r))
    return sorted([(k, float(np.mean(v)), len(v)) for k, v in by.items()], key=lambda t: t[0])


def interp_rho_at(cm, y0, lnsd):
    """ρ 별 평균 ln 스파이크 곡선 (ln-ln 구간 선형) 에서 ln 스파이크 y0 를 내는 ρ.
    분산: y0 자체 (새 런 하나, lnsd²) + 두 이웃 평균 (lnsd²/n) 을 국소 기울기로 ρ 축에 옮김."""
    for (r1, y1, n1), (r2, y2, n2) in zip(cm, cm[1:]):
        if (y1 - y0) * (y2 - y0) <= 0 and y1 != y2:
            w = (y1 - y0) / (y1 - y2)
            lr = math.log(r1) + w * (math.log(r2) - math.log(r1))
            beta = (y2 - y1) / (math.log(r2) - math.log(r1))
            var = lnsd ** 2 * (1 + (1 - w) ** 2 / n1 + w ** 2 / n2) / beta ** 2
            return {'rho_eq': math.exp(lr), 'w': w, 'bracket': [r1, r2], 'n': [n1, n2], 'beta_local': beta,
                    'se_ln': math.sqrt(var)}
    return None


def interp_at_rho(cm, rho, lnsd):
    """같은 곡선에서 ρ 의 ln 스파이크 (구간 밖이면 끝 두 점으로 외삽, 표시)."""
    lr = math.log(rho)
    pts = [(math.log(r), y, n, r) for r, y, n in cm]
    for (x1, y1, n1, r1), (x2, y2, n2, r2) in zip(pts, pts[1:]):
        if x1 <= lr <= x2:
            w = (lr - x1) / (x2 - x1)
            return {'y': y1 + w * (y2 - y1), 'se': lnsd * math.sqrt((1 - w) ** 2 / n1 + w ** 2 / n2),
                    'bracket': [r1, r2], 'n': [n1, n2], 'extrap': False}
    (x1, y1, n1, r1), (x2, y2, n2, r2) = (pts[-2], pts[-1]) if lr > pts[-1][0] else (pts[0], pts[1])
    w = (lr - x1) / (x2 - x1)
    return {'y': y1 + w * (y2 - y1), 'se': lnsd * math.sqrt((1 - w) ** 2 / n1 + w ** 2 / n2),
            'bracket': [r1, r2], 'n': [n1, n2], 'extrap': True}


def pooled_sd(rs, fval):
    by = defaultdict(list)
    for r in rs:
        by[r['rho']].append(fval(r))
    num = sum((len(v) - 1) * np.var(v, ddof=1) for v in by.values() if len(v) > 1)
    df = sum(len(v) - 1 for v in by.values() if len(v) > 1)
    return (math.sqrt(num / df) if df else float('nan')), df


def gini(v):
    v = np.sort(np.asarray(v, float))
    n, tot = len(v), v.sum()
    if tot <= 0:
        return float('nan')
    idx = np.arange(1, n + 1)
    return float(2 * (idx * v).sum() / (n * tot) - (n + 1) / n)


# ============================================================================ npz
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
        ks = w['ks']
        cin, cout = ks[-2], ks[-1]
        kk = int(np.prod(ks[:-2])) if len(ks) > 2 else 1
        di, do = dead.get(w['src'], 0), dead.get(w['dst'], 0)
        mr += (di * cout + do * cin - di * do) * kk * w['hw']
        mt += cin * cout * kk * w['hw']
    return 100 * mr / mt


def npz_struct(p):
    """테스트 1만 장 1회 추론 (평가 모드). 채널 = conv 의 공간 합, fc1 은 뉴런 하나."""
    z = np.load(p)
    ns = float(z['n_samples'])
    dead, act, spk, nf = {}, 0.0, 0.0, 0.0
    gsum = tsum = 0.0
    lspk = {}
    g_st = defaultdict(float)
    s_st = defaultdict(float)
    for l in NEU:
        c = z[l].astype(np.float64)
        f = z[l + '__fired'].astype(np.float64)
        act += f.sum()
        spk += c.sum()
        nf += (c == 0).sum()
        ch = c.reshape(-1, c.shape[-1]).sum(0) if c.ndim == 3 else c
        dead[l] = int((ch == 0).sum())
        g = gini(ch)
        k = max(len(ch) // 10, 1)
        top = np.sort(ch)[::-1][:k].sum() / ch.sum() if ch.sum() > 0 else float('nan')
        lspk[l] = c.sum()
        gsum += g * c.sum()
        tsum += top * c.sum()
        st = l.split('_')[0]
        g_st[st] += g * c.sum()
        s_st[st] += c.sum()
    o = {'dead_ch': int(sum(dead.values())), 'active_img': act / ns, 'npz_spk': spk / ns,
         'never_fired_pct': 100 * nf / sum(NEU.values()),
         'ch_gini_sw': gsum / spk, 'ch_top10_sw': 100 * tsum / spk,
         'dead_by_stage': {s: int(sum(v for l, v in dead.items() if l.startswith(s))) for s in STAGES},
         'ch_gini_by_stage': {s: g_st[s] / s_st[s] for s in STAGES if s_st[s] > 0}}
    if WL:
        o['macs_rm_pct'] = macs_removed_pct(dead)
    return o


# ============================================================================ 분석
def analyze(runs):
    A = lambda rs, k: np.array([r[k] for r in rs], float)
    use = [r for r in runs if r['use']]
    base = [r for r in use if r['key'] == 'base']
    ours = [r for r in use if r['key'] == 'ours_ch']
    noi = [r for r in use if r['key'] == 'ch_noinv']
    prop = [r for r in use if r['key'] == 'prop']
    abl = [r for r in use if r['key'] == 'abl_noinv']
    res = {'n': {'base': len(base), 'ours_ch': len(ours), 'ch_noinv': len(noi), 'prop': len(prop),
                 'abl_noinv': len(abl)}}
    bm = {'val_acc': A(base, 'val_acc').mean(), 'val_acc_sd': A(base, 'val_acc').std(ddof=1),
          'spikes': A(base, 'spikes').mean(), 'spikes_sd': A(base, 'spikes').std(ddof=1), 'n': len(base)}
    res['base_mean'] = bm
    # 합동 시드 SD (ours_ch 같은 ρ 묶음)
    by = defaultdict(list)
    for r in ours:
        by[r['knob']].append(r)
    num = sum((len(v) - 1) * A(v, 'val_acc').var(ddof=1) for v in by.values() if len(v) > 1)
    dfp = sum(len(v) - 1 for v in by.values() if len(v) > 1)
    sd_pool = math.sqrt(num / dfp)
    lnsd = math.sqrt(sum((len(v) - 1) * np.log(A(v, 'spikes')).var(ddof=1) for v in by.values() if len(v) > 1) / dfp)
    res['pooled'] = {'acc_sd': sd_pool, 'df': dfp, 'ln_spk_sd': lnsd, 'n_rho': sum(len(v) > 1 for v in by.values())}

    # ---------------------------------------------------------------- 1. 같은 스파이크 정확도
    rng = [r for r in ours if ACC_RANGE[0] <= r['spikes'] <= ACC_RANGE[1]]
    data = rng + noi
    s0 = A(noi, 'spikes').mean()
    x = np.log(A(data, 'spikes')) - math.log(s0)
    y = A(data, 'val_acc')
    g = np.array([1.0 if r['key'] == 'ch_noinv' else 0.0 for r in data])
    sv = np.array([1.0 if r['server'] == '23' else 0.0 for r in data])
    one = np.ones_like(x)
    models = {'lin': (np.c_[one, x, g], ['a', 'b', 'd']),
              'lin_srv': (np.c_[one, x, g, sv], ['a', 'b', 'd', 'e']),
              'quad': (np.c_[one, x, x ** 2, g], ['a', 'b', 'c', 'd']),
              'quad_srv': (np.c_[one, x, x ** 2, g, sv], ['a', 'b', 'c', 'd', 'e'])}
    anc = {nm: ols(X, y, names) for nm, (X, names) in models.items()}
    res['ancova'] = {nm: {'coef': f['coef'], 'df': f['df'], 'n': f['n'], 'resid_sd': f['resid_sd']}
                     for nm, f in anc.items()}
    res['ancova_n'] = {'ours_ch': len(rng), 'ch_noinv': len(noi), 'ours_rhos': sorted({r['knob'] for r in rng}, key=float),
                       'ours_servers': {'138': sum(r['server'] == '138' for r in rng),
                                        '23': sum(r['server'] == '23' for r in rng)}}
    # 같은 정확도 스파이크 비 (선형+서버, θ = d/b)
    f = anc['lin_srv']
    b, d = f['coef']['b']['est'], f['coef']['d']['est']
    ib, idd = f['names'].index('b'), f['names'].index('d')
    vb, vd, cdb = f['cov'][ib, ib], f['cov'][idd, idd], f['cov'][ib, idd]
    th = d / b
    se_th = abs(th) * math.sqrt(vd / d ** 2 + vb / b ** 2 - 2 * cdb / (d * b))
    t2 = tcrit(f['df']) ** 2
    Aq, Bq, Cq = b * b - t2 * vb, -2 * (d * b - t2 * cdb), d * d - t2 * vd
    disc = Bq * Bq - 4 * Aq * Cq
    fl = None
    if Aq > 0 and disc >= 0:
        r1, r2 = (-Bq - math.sqrt(disc)) / (2 * Aq), (-Bq + math.sqrt(disc)) / (2 * Aq)
        fl = ((math.exp(-r2) - 1) * 100, (math.exp(-r1) - 1) * 100)
    res['same_acc'] = {'ratio_pct': (math.exp(-th) - 1) * 100,
                       'delta_lo': (math.exp(-(th + tcrit(f['df']) * se_th)) - 1) * 100,
                       'delta_hi': (math.exp(-(th - tcrit(f['df']) * se_th)) - 1) * 100,
                       'fieller': fl}
    # 검정력: ours 곡선 고정, noinv 반복 수 m 에 따른 최소 검출 차
    se_d = f['coef']['d']['se']
    var_curve = max(se_d ** 2 - f['resid_sd'] ** 2 / len(noi), 0.0)
    kk = tcrit(f['df']) + stats.t.ppf(0.8, f['df'])
    res['power'] = {'curve_se': math.sqrt(var_curve),
                    'mde_by_n': {str(m): kk * math.sqrt(sd_pool ** 2 / m + var_curve) for m in (1, 3, 4, 6, 10)},
                    'floor': kk * math.sqrt(var_curve),
                    'n_needed_obs': None}
    dd = abs(d)
    for m in range(1, 2001):
        if kk * math.sqrt(sd_pool ** 2 / m + var_curve) <= dd:
            res['power']['n_needed_obs'] = m
            break
    # 범위 민감도: ours 150~420K, 180~330K
    sens = {}
    for lo, hi in ((150e3, 420e3), (180e3, 330e3), (200e3, 300e3)):
        rr = [r for r in ours if lo <= r['spikes'] <= hi] + noi
        xx = np.log(A(rr, 'spikes')) - math.log(s0)
        X = np.c_[np.ones(len(rr)), xx, [1.0 if r['key'] == 'ch_noinv' else 0.0 for r in rr],
                  [1.0 if r['server'] == '23' else 0.0 for r in rr]]
        ff = ols(X, A(rr, 'val_acc'), ['a', 'b', 'd', 'e'])
        sens[f'{int(lo/1e3)}-{int(hi/1e3)}K'] = {'n_ours': len(rr) - len(noi), 'd': ff['coef']['d'], 'b': ff['coef']['b']}
    res['ancova_sens'] = sens
    # baseline 대비 (noinv n=1: 그 런의 시드 산포를 ours 합동 SD 로 가정)
    vbase = []
    for r in noi:
        da = r['val_acc'] - bm['val_acc']
        se = math.sqrt(sd_pool ** 2 + bm['val_acc_sd'] ** 2 / bm['n'])
        dfw = se ** 4 / (sd_pool ** 4 / dfp + (bm['val_acc_sd'] ** 2 / bm['n']) ** 2 / (bm['n'] - 1))
        lb = np.log(A(base, 'spikes'))
        dl = math.log(r['spikes']) - lb.mean()
        sel = math.sqrt(lnsd ** 2 + lb.var(ddof=1) / len(lb))
        vbase.append({'tag': r['tag'], 'acc_diff': da, 'acc_se': se, 'df': dfw, 'acc_lo': da - tcrit(dfw) * se,
                      'acc_hi': da + tcrit(dfw) * se, 'p': pval_t(da / se, dfw),
                      'spk_pct': (math.exp(dl) - 1) * 100, 'spk_lo': (math.exp(dl - 1.96 * sel) - 1) * 100,
                      'spk_hi': (math.exp(dl + 1.96 * sel) - 1) * 100})
    res['vs_base'] = vbase
    # 근처 ours (±8% 스파이크 창) 의 baseline 대비 — 같은 정의로
    w8 = [r for r in ours if s0 * 0.92 <= r['spikes'] <= s0 * 1.08]
    res['ours_win8'] = {'tags': [r['tag'] for r in w8], 'n': len(w8), 'acc': welch(A(w8, 'val_acc'), A(base, 'val_acc')),
                        'acc_mean': A(w8, 'val_acc').mean(), 'spk_mean': A(w8, 'spikes').mean(),
                        'spk_pct': (math.exp(np.log(A(w8, 'spikes')).mean() - np.log(A(base, 'spikes')).mean()) - 1) * 100}

    # ---------------------------------------------------------------- 2. 같은 ρ / 같은 λ 에서 스파이크
    sc = {}
    # ours: 완주 310 에폭 (연기시험 제외). 정확도 배제(S30/S1) 런은 눈금 추정에만 별도 포함.
    ours_all = [r for r in runs if r['key'] == 'ours_ch' and r['epochs'] == EPOCHS and r['seed'] < 900
                and r['arm_conf'] == 'ours_ch']
    for nm, pool in (('use', [r for r in ours_all if r['use']]), ('all310', ours_all)):
        xr = np.log(A(pool, 'rho'))
        ys = np.log(A(pool, 'spikes'))
        fit = ols(np.c_[np.ones(len(xr)), xr], ys, ['a', 'b'])
        a_, b_ = fit['coef']['a']['est'], fit['coef']['b']['est']
        rows = []
        for r in noi:
            lp = line_pred(xr, ys, math.log(r['rho']))
            # 같은 스파이크가 되는 ours ρ (역산) 와 배율
            lr_eq = (math.log(r['spikes']) - a_) / b_
            # 델타법: Var(lr_eq) ≈ [Var(ln s_new)+Var(a)+lr²Var(b)+2lr Cov(a,b)]/b²  (ln s_new 에 시드 산포 = 잔차 SD)
            cov = fit['cov']
            var = (fit['resid_sd'] ** 2 + cov[0, 0] + lr_eq ** 2 * cov[1, 1] + 2 * lr_eq * cov[0, 1]) / b_ ** 2
            fac = r['rho'] / math.exp(lr_eq)
            rows.append({'tag': r['tag'], 'rho': r['rho'], 'spikes': r['spikes'],
                         'ours_pred_spk': math.exp(lp['pred']), 'ratio_pct': (r['spikes'] / math.exp(lp['pred']) - 1) * 100,
                         'z': (math.log(r['spikes']) - lp['pred']) / lp['se_new'], 'df': lp['df'],
                         'rho_eq': math.exp(lr_eq), 'factor': fac,
                         'factor_lo': fac * math.exp(-tcrit(fit['df']) * math.sqrt(var)),
                         'factor_hi': fac * math.exp(tcrit(fit['df']) * math.sqrt(var))})
        sc[nm] = {'n': len(pool), 'slope': fit['coef']['b'], 'resid_sd': fit['resid_sd'],
                  'rho_range': [float(A(pool, 'rho').min()), float(A(pool, 'rho').max())], 'rows': rows}
    # 국소 (주 결과): ρ 별 평균 ln 스파이크를 ln-ln 구간 선형으로 잇는다. ln s–ln ρ 가 높은 ρ 에서 가팔라져
    # (8e-3→1.2e-2 기울기 −0.76, 1.6e-2→2e-2 −1.13) 전체 직선은 1e-2 에서 ours 스파이크를 과대 예측한다.
    # 곡선 = 310 에폭 완주·붕괴 아님 (best val acc ≥ 95%) — S30 배제 1.6e-2·2e-2 는 정확도 비교에서만 빠지고 눈금에는 넣는다.
    ours_curve = [r for r in ours_all if r['val_acc'] >= 95.0]
    cm = cond_means(ours_curve, lambda r: math.log(r['spikes']))
    loc = []
    for r in noi:
        ie = interp_rho_at(cm, math.log(r['spikes']), lnsd)
        ia = interp_at_rho(cm, r['rho'], lnsd)
        z = (math.log(r['spikes']) - ia['y']) / math.sqrt(lnsd ** 2 + ia['se'] ** 2)
        fac = r['rho'] / ie['rho_eq']
        tc = tcrit(dfp)
        loc.append({'tag': r['tag'], 'rho': r['rho'], 'spikes': r['spikes'], 'same_rho': ia,
                    'ours_spk_at_rho': math.exp(ia['y']), 'ratio_pct': (r['spikes'] / math.exp(ia['y']) - 1) * 100,
                    'z': z, 'same_spk': ie, 'factor': fac, 'factor_lo': fac * math.exp(-tc * ie['se_ln']),
                    'factor_hi': fac * math.exp(tc * ie['se_ln'])})
    sc['local'] = {'rows': loc, 'curve': [{'rho': c[0], 'spikes': math.exp(c[1]), 'n': c[2]} for c in cm],
                   'n_runs': len(ours_curve), 'lnsd': lnsd, 'df': dfp}
    # λ 도 같은 방식 (같은 스파이크에서 ours λ 평균을 ρ 묶음 평균 사이 보간)
    cmL = cond_means([r for r in ours_curve if 'lam_mean' in r], lambda r: math.log(r['lam_mean']))
    for r, lr_ in zip(noi, loc):
        ie = lr_['same_spk']
        r1, r2 = ie['bracket']
        y1 = [c[1] for c in cmL if c[0] == r1][0]
        y2 = [c[1] for c in cmL if c[0] == r2][0]
        lr_['ours_lam_same_spk'] = math.exp(y1 + ie['w'] * (y2 - y1))
        lr_['lam_ratio'] = r['lam_mean'] / lr_['ours_lam_same_spk']
    # λ (학습 310 에폭 평균) 기준
    lam_pool = [r for r in ours_all if r['use'] and 'lam_mean' in r]
    xl = np.log(A(lam_pool, 'lam_mean'))
    ys = np.log(A(lam_pool, 'spikes'))
    sc['lam'] = []
    for r in noi:
        lp = line_pred(xl, ys, math.log(r['lam_mean']))          # 같은 λ 에서 ours 가 착지할 스파이크
        lq = line_pred(ys, xl, math.log(r['spikes']))           # 같은 스파이크에서 ours 의 λ
        lq_end = line_pred(ys, np.log(A(lam_pool, 'lam_end')), math.log(r['spikes']))
        sc['lam'].append({'tag': r['tag'], 'lam_mean': r['lam_mean'], 'lam_end': r['lam_end'],
                          'same_lam_ours_spk': math.exp(lp['pred']),
                          'same_lam_ratio_pct': (r['spikes'] / math.exp(lp['pred']) - 1) * 100,
                          'same_lam_z': (math.log(r['spikes']) - lp['pred']) / lp['se_new'],
                          'same_spk_ours_lam': math.exp(lq['pred']), 'lam_ratio': r['lam_mean'] / math.exp(lq['pred']),
                          'lam_ratio_z': (math.log(r['lam_mean']) - lq['pred']) / lq['se_new'],
                          'lam_end_ratio': r['lam_end'] / math.exp(lq_end['pred']), 'n_fit': lp['n']})
    sc['lam_n'] = len(lam_pool)
    # λ = ρ·L/R 분해 (마지막 10 에폭 평균, train.log 의 task_loss_clean·reg_R)
    def tail(r, k, n=10):
        v = np.array(r['traj'][k][-n:], float)
        return float(np.nanmean(v))
    dec_pool = [r for r in ours_all if r['use'] and MECH_BAND[0] <= r['spikes'] <= MECH_BAND[1]]
    dec = []
    for r in noi:
        o = {'tag': r['tag']}
        for k in ('task_loss_clean', 'reg_R', 'adp_lambda'):
            lp = line_pred(np.log(A(dec_pool, 'spikes')), np.log([tail(q, k) for q in dec_pool]), math.log(r['spikes']))
            v = tail(r, k)
            o[k] = {'value': v, 'ours_pred': math.exp(lp['pred']), 'ratio': v / math.exp(lp['pred']),
                    'z': (math.log(v) - lp['pred']) / lp['se_new'], 'n_fit': lp['n']}
        dec.append(o)
    sc['decomp'] = dec
    # 2.4e-2 s1 (195 에폭 중단) — 같은 에폭 (195) 의 s_count 를 ours 와 비교, 착지 예측
    stale = [r for r in runs if r['tag'] == 'r19c10-ch_noinv-2p4e-2-s1']
    if stale:
        q = stale[0]
        E = q['epochs']                          # 195
        pool = [r for r in ours_curve if len(r['traj']['s_count']) >= E]
        lnsdE, dfE = pooled_sd(pool, lambda r: math.log(r['traj']['s_count'][E - 1]))
        cmE = cond_means(pool, lambda r: math.log(r['traj']['s_count'][E - 1]))
        sE = q['traj']['s_count'][E - 1]
        ieE = interp_rho_at(cmE, math.log(sE), lnsdE)
        iaE = interp_at_rho(cmE, q['rho'], lnsdE)
        # 1e-2 s1 (완주) 도 같은 에폭에서 같은 방식 — 두 ρ 의 배율이 맞는지
        e195 = []
        for r in noi:
            ie_ = interp_rho_at(cmE, math.log(r['traj']['s_count'][E - 1]), lnsdE)
            e195.append({'tag': r['tag'], 's_at_E': r['traj']['s_count'][E - 1], 'rho_eq': ie_['rho_eq'],
                         'factor': r['rho'] / ie_['rho_eq'], 'bracket': ie_['bracket'], 'se_ln': ie_['se_ln']})
        # 착지 예측: 붕괴 아닌 ours 런의 (best 에폭 스파이크 / 195 에폭 스파이크)
        rat = np.array([r['spikes'] / r['traj']['s_count'][E - 1] for r in pool])
        rat_hi = np.array([r['spikes'] / r['traj']['s_count'][E - 1] for r in pool if r['rho'] >= 8e-3])
        # noinv 1e-2 s1 (완주) 의 같은 비 — noinv 팔 자체의 보정값
        rat_noi = [r['spikes'] / r['traj']['s_count'][E - 1] for r in noi]
        chk = [{'tag': r['tag'], 's195': r['traj']['s_count'][E - 1],
                'pred_land': r['traj']['s_count'][E - 1] * float(np.median(rat)), 'actual': r['spikes']} for r in noi]
        sc['stale'] = {'tag': q['tag'], 'epochs': E, 'rho': q['rho'], 's_at_E': sE,
                       'val_acc_at_E': q['traj']['val_acc'][E - 1] * 100,
                       'best_val_acc_so_far': q['val_acc'], 'best_ep_so_far': q['best_epoch'],
                       'lnsd_E': lnsdE, 'df_E': dfE, 'n_pool': len(pool),
                       'same_rho': iaE, 'ours_at_rho_E': math.exp(iaE['y']),
                       'ratio_pct': (sE / math.exp(iaE['y']) - 1) * 100,
                       'z': (math.log(sE) - iaE['y']) / math.sqrt(lnsdE ** 2 + iaE['se'] ** 2),
                       'same_spk': ieE, 'rho_eq_at_E': ieE['rho_eq'], 'factor_at_E': q['rho'] / ieE['rho_eq'],
                       'factor_at_E_lo': q['rho'] / ieE['rho_eq'] * math.exp(-tcrit(dfE) * ieE['se_ln']),
                       'factor_at_E_hi': q['rho'] / ieE['rho_eq'] * math.exp(tcrit(dfE) * ieE['se_ln']),
                       'e195_noinv': e195,
                       'land_ratio_median': float(np.median(rat)), 'land_ratio_range': [float(rat.min()), float(rat.max())],
                       'land_ratio_hi_median': float(np.median(rat_hi)), 'n_hi': int(len(rat_hi)),
                       'land_ratio_noinv': rat_noi,
                       'pred_land': sE * float(np.median(rat)), 'pred_land_lo': sE * float(rat.min()),
                       'pred_land_hi': sE * float(rat.max()), 'pred_land_rhohi': sE * float(np.median(rat_hi)),
                       'pred_land_noinvcal': sE * float(np.median(rat_noi)) if rat_noi else None,
                       'check_1e-2': chk,
                       's30': q['s30'], 'traj_s': [q['traj']['s_count'][i] for i in (0, 29, 59, 99, 149, E - 1)],
                       'traj_acc': [q['traj']['val_acc'][i] * 100 for i in (0, 29, 59, 99, 149, E - 1)],
                       'last_sec': q['traj']['sec'][-1]}
        # 같은 ρ 짝: ours 2e-2 s1, 1.6e-2 s1 의 같은 에폭 값 (참고)
        sc['stale']['ours_same_epoch'] = [{'tag': r['tag'], 'rho': r['rho'], 's_at_E': r['traj']['s_count'][E - 1],
                                           'acc_at_E': r['traj']['val_acc'][E - 1] * 100, 'final_spk': r['spikes'],
                                           'final_acc': r['val_acc'], 'use': r['use'], 'excl': r['exclude_reason']}
                                          for r in ours_all if r['rho'] >= 8e-3 and len(r['traj']['s_count']) >= E]
    # intra: abl_noinv vs prop — 같은 ρ 와 같은 스파이크 (ρ 배율)
    prop_all = [r for r in runs if r['key'] == 'prop' and r['epochs'] == EPOCHS and r['arm_conf'] == 'prop']
    abl_all = [r for r in runs if r['key'] == 'abl_noinv' and r['epochs'] == EPOCHS and r['arm_conf'] == 'abl_noinv']
    intra = {}
    for nm, pool in (('prop', prop_all), ('abl_noinv', abl_all)):
        xr = np.log(A(pool, 'rho'))
        ys = np.log(A(pool, 'spikes'))
        fit = ols(np.c_[np.ones(len(xr)), xr], ys, ['a', 'b'])
        intra[nm] = {'n': len(pool), 'a': fit['coef']['a']['est'], 'b': fit['coef']['b'], 'resid_sd': fit['resid_sd']}
    p3 = [r for r in prop if abs(r['rho'] - 3e-3) < 1e-12]
    a24 = [r for r in abl if abs(r['rho'] - 2.4e-2) < 1e-12]
    intra['pair'] = {'prop3_n': len(p3), 'prop3_spk': A(p3, 'spikes').mean(), 'abl24_n': len(a24),
                     'abl24_spk': A(a24, 'spikes').mean(), 'factor': 2.4e-2 / 3e-3,
                     'spk_diff': welch(np.log(A(a24, 'spikes')), np.log(A(p3, 'spikes')))}
    # 같은 ρ (4e-3): abl_noinv 4e-3 (n=2) vs prop 4e-3 (n=5)
    p4 = [r for r in prop if abs(r['rho'] - 4e-3) < 1e-12]
    a4 = [r for r in abl if abs(r['rho'] - 4e-3) < 1e-12]
    intra['same_rho4'] = {'prop_n': len(p4), 'prop_spk': A(p4, 'spikes').mean(), 'abl_n': len(a4),
                          'abl_spk': A(a4, 'spikes').mean(),
                          'ln_diff': welch(np.log(A(a4, 'spikes')), np.log(A(p4, 'spikes'))) if len(a4) > 1 else None}
    sc['intra'] = intra
    res['scale'] = sc

    # ---------------------------------------------------------------- 3. 기전 (학습 모드, 마지막 에폭)
    mkeys = (['tr_spk', 'R_rd', 'w_mean', 'cn_mean', 'wsd_mean', 'cn_cv', 'gini_sw', 'top10_sw', 'dead_pct']
             + [f'cn_{s}' for s in STAGES[:4]] + [f'cn_cv_{s}' for s in STAGES[:4]] + [f'w_{s}' for s in STAGES[:4]]
             + [f'gini_{s}' for s in STAGES[:4]] + [f'dead_{s}' for s in STAGES[:4]] + [f'share_{s}' for s in STAGES[:4]])
    def mech_cmp(pool, xf, r):
        o = {'tag': r['tag'], 'x': xf(r), 'n_fit': len(pool), 'm': {}}
        for k in mkeys:
            if k == 'tr_spk' or k not in r['mech']:
                continue
            logy = k == 'R_rd'
            yv = [q['mech'][k] for q in pool]
            yy = np.log(yv) if logy else np.array(yv)
            lp = line_pred(np.log([xf(q) for q in pool]), yy, math.log(xf(r)))
            v = r['mech'][k]
            vv = math.log(v) if logy else v
            o['m'][k] = {'ours_pred': math.exp(lp['pred']) if logy else lp['pred'], 'value': v,
                         'delta': (math.exp(vv - lp['pred']) - 1) * 100 if logy else v - lp['pred'],
                         'se': lp['se_new'], 'z': (vv - lp['pred']) / lp['se_new'], 'df': lp['df'],
                         'p': pval_t((vv - lp['pred']) / lp['se_new'], lp['df']), 'log': logy}
        return o
    # 주: 10-10 과 같은 방법 — x = 평가 모드 spikes (원장 정의), 대역도 평가 모드 spikes 180~330K
    mpool = [r for r in ours if r['mech'] and MECH_BAND[0] <= r['spikes'] <= MECH_BAND[1]]
    res['mech'] = [mech_cmp(mpool, lambda q: q['spikes'], r) for r in noi]
    res['mech_pool'] = [r['tag'] for r in mpool]
    # 민감도: x = 학습 모드 스파이크 (같은 에폭의 같은 자료)
    mpool_tr = [r for r in ours if r['mech'] and MECH_BAND[0] <= r['mech']['tr_spk'] <= MECH_BAND[1]]
    res['mech_tr'] = [mech_cmp(mpool_tr, lambda q: q['mech']['tr_spk'], r) for r in noi]
    res['mech_pool_tr'] = [r['tag'] for r in mpool_tr]
    # 민감도: ours 비배제 전부 (대역 제한 없음)
    mpool_all = [r for r in ours if r['mech']]
    res['mech_all'] = [mech_cmp(mpool_all, lambda q: q['spikes'], r) for r in noi]
    # ours 의 m·지니가 스파이크에 따라 어떻게 변하나 (마지막 에폭, ours 전체 비배제) — 기울기 (지표 / ln 스파이크)
    allm = [r for r in ours if r['mech']]
    slopes = {}
    for k in ('cn_mean', 'gini_sw', 'top10_sw'):
        f_ = ols(np.c_[np.ones(len(allm)), np.log([r['spikes'] for r in allm])], [r['mech'][k] for r in allm], ['a', 'b'])
        slopes[k] = {'ours_b': f_['coef']['b'], 'n': len(allm)}
    res['mech_slopes'] = slopes
    # 같은 에폭 (195, CSV 0-기준 194) 비교: 2.4e-2 s1 (중단 런) 과 1e-2 s1
    E0 = 194
    ep_cmp = []
    qset = [r for r in runs if r['key'] == 'ch_noinv' and E0 in r['mech_traj'] and r['seed'] < 900]
    # 붕괴 아닌 ours (best val acc ≥ 95%, S30 배제 런 포함 — 기전 비교라 정확도 배제 기준을 쓰지 않는다)
    opool = [r for r in runs if r['key'] == 'ours_ch' and r['arm_conf'] == 'ours_ch' and r['seed'] < 900
             and E0 in r['mech_traj'] and r['val_acc'] >= 95.0]
    ekeys = ('cn_mean', 'cn_cv', 'gini_sw', 'top10_sw', 'dead_pct', 'share_conv2', 'share_conv3')
    # 195 에폭 지표의 ours 같은 ρ 시드 SD (합동)
    sd195 = {k: pooled_sd([r for r in opool if r['use']], lambda r, k=k: r['mech_traj'][E0][k]) for k in ekeys}
    for q in qset:
        t0 = q['mech_traj'][E0]['tr_spk']
        pool = [r for r in opool if t0 / 2 <= r['mech_traj'][E0]['tr_spk'] <= t0 * 2]
        o = {'tag': q['tag'], 'tr_spk': t0, 'n_fit': len(pool), 'pool_tags': [r['tag'] for r in pool], 'm': {}}
        if len(pool) >= 4:
            for k in ekeys:
                lp = line_pred(np.log([r['mech_traj'][E0]['tr_spk'] for r in pool]),
                               np.array([r['mech_traj'][E0][k] for r in pool]), math.log(t0))
                v = q['mech_traj'][E0][k]
                o['m'][k] = {'ours_pred': lp['pred'], 'value': v, 'delta': v - lp['pred'], 'se': lp['se_new'],
                             'z': (v - lp['pred']) / lp['se_new'], 'df': lp['df'],
                             'z_seed': (v - lp['pred']) / sd195[k][0]}
        ep_cmp.append(o)
    res['mech_ep195'] = ep_cmp
    res['mech_ep195_sd'] = {k: {'sd': v[0], 'df': v[1]} for k, v in sd195.items()}
    # 195 에폭: 스파이크를 줄일 때 m 이 어떻게 변하나 — ours 기울기 (붕괴 아닌 전부) vs noinv 두 점
    xo = np.log([r['mech_traj'][E0]['tr_spk'] for r in opool])
    tr195 = {}
    for k in ('cn_mean', 'gini_sw', 'top10_sw'):
        f_ = ols(np.c_[np.ones(len(xo)), xo], [r['mech_traj'][E0][k] for r in opool], ['a', 'b'])
        two = sorted(qset, key=lambda r: r['mech_traj'][E0]['tr_spk'])
        nb = None
        if len(two) == 2:
            nb = (two[1]['mech_traj'][E0][k] - two[0]['mech_traj'][E0][k]) / \
                 (math.log(two[1]['mech_traj'][E0]['tr_spk']) - math.log(two[0]['mech_traj'][E0]['tr_spk']))
        tr195[k] = {'ours_b': f_['coef']['b'], 'ours_n': len(opool), 'noinv_two_point_b': nb,
                    'ours_range': [float(np.exp(xo.min())), float(np.exp(xo.max()))]}
    res['mech_ep195_slope'] = tr195
    # 궤적: noinv 1e-2 s1 vs ours 7e-3 (n=3, 착지 스파이크 짝) · 2.4e-2 s1
    tr = {}
    o7 = [r for r in ours if abs(r['rho'] - 7e-3) < 1e-12]
    for e in (0, 9, 29, 59, 99, 149, 194, 199, 249, 309):
        row = {}
        for nm, grp in (('ours7', o7), ('noinv1e-2', [r for r in noi if abs(r['rho'] - 1e-2) < 1e-12]),
                        ('noinv2.4e-2', [r for r in runs if r['tag'] == 'r19c10-ch_noinv-2p4e-2-s1'])):
            g_ = [r for r in grp if e in r['mech_traj']]
            if not g_:
                continue
            row[nm] = {k: (float(np.mean([r['mech_traj'][e][k] for r in g_])),
                           float(np.std([r['mech_traj'][e][k] for r in g_], ddof=1)) if len(g_) > 1 else None)
                       for k in ('tr_spk', 'cn_mean', 'cn_cv', 'gini_sw', 'top10_sw', 'dead_pct')
                       if all(k in r['mech_traj'][e] for r in g_)}
            row[nm]['n'] = len(g_)
        tr[e + 1] = row
    res['mech_traj'] = tr
    res['mech_traj_ours7'] = [r['tag'] for r in o7]

    # ---------------------------------------------------------------- 4. npz (평가 모드)
    nz = []
    for r in runs:
        if r['key'] in ('ours_ch', 'ch_noinv', 'prop', 'abl_noinv', 'base') and r['use']:
            p = npz_path(r['tag'])
            if p:
                nz.append({'tag': r['tag'], 'key': r['key'], 'rho': r['rho'], 'server': r['server'],
                           'spikes': r['spikes'], 'val_acc': r['val_acc'], 'path': p, **npz_struct(p)})
    res['npz_rows'] = nz
    res['npz_noinv'] = [x['tag'] for x in nz if x['key'] == 'ch_noinv']
    # ours_ch npz 곡선 → noinv 스파이크에서 예측 (측정 목표)
    on = [x for x in nz if x['key'] == 'ours_ch' and NPZ_BAND[0] <= x['spikes'] <= NPZ_BAND[1]]
    tgt = {}
    for r in noi:
        o = {'n_fit': len(on), 'tags': [x['tag'] for x in on]}
        for k, lg in (('dead_ch', False), ('macs_rm_pct', False), ('active_img', True), ('never_fired_pct', False),
                      ('ch_gini_sw', False), ('ch_top10_sw', False)):
            if k not in on[0]:
                continue
            yy = np.log([x[k] for x in on]) if lg else np.array([x[k] for x in on])
            lp = line_pred(np.log([x['spikes'] for x in on]), yy, math.log(r['spikes']))
            o[k] = {'pred': math.exp(lp['pred']) if lg else lp['pred'], 'se_new': lp['se_new'],
                    'resid_sd': lp['resid_sd'], 'log': lg, 'df': lp['df'],
                    # 새 런 하나가 이 예측과 유의하게 (p<0.05) 다르려면 필요한 차
                    'z_thresh': tcrit(lp['df']) * lp['se_new']}
        tgt[r['tag']] = o
    res['npz_target'] = tgt
    # intra 짝: prop 3e-3 (npz 있는 것) vs abl_noinv 2.4e-2
    p3n = [x for x in nz if x['key'] == 'prop' and abs(x['rho'] - 3e-3) < 1e-12]
    a24n = [x for x in nz if x['key'] == 'abl_noinv' and abs(x['rho'] - 2.4e-2) < 1e-12]
    ip = {'prop_tags': [x['tag'] for x in p3n], 'abl_tags': [x['tag'] for x in a24n]}
    if len(p3n) >= 2 and len(a24n) >= 2:
        for k, lg in (('dead_ch', False), ('macs_rm_pct', False), ('active_img', True), ('never_fired_pct', False),
                      ('ch_gini_sw', False), ('ch_top10_sw', False), ('npz_spk', True), ('val_acc', False)):
            a_ = [x[k] for x in a24n]
            b_ = [x[k] for x in p3n]
            if lg:
                w_ = welch(np.log(a_), np.log(b_))
                ip[k] = {'abl': float(np.mean(a_)), 'prop': float(np.mean(b_)), 'pct': (math.exp(w_['diff']) - 1) * 100,
                         'lo': (math.exp(w_['lo']) - 1) * 100, 'hi': (math.exp(w_['hi']) - 1) * 100, 'p': w_['p']}
            else:
                w_ = welch(a_, b_)
                ip[k] = {'abl': float(np.mean(a_)), 'prop': float(np.mean(b_)), 'abl_sd': float(np.std(a_, ddof=1)),
                         'prop_sd': float(np.std(b_, ddof=1)), **w_}
        for s in STAGES:
            w_ = welch([x['dead_by_stage'][s] for x in a24n], [x['dead_by_stage'][s] for x in p3n])
            ip[f'dead_{s}'] = {'abl': float(np.mean([x['dead_by_stage'][s] for x in a24n])),
                               'prop': float(np.mean([x['dead_by_stage'][s] for x in p3n])), **w_}
        # npz 스파이크 차 (abl 쪽이 적음) 로 예상되는 죽은 채널 변화: prop npz 전부의 dead ~ ln(npz_spk) 기울기
        pn = [x for x in nz if x['key'] == 'prop']
        fp_ = ols(np.c_[np.ones(len(pn)), np.log([x['npz_spk'] for x in pn])], [x['dead_ch'] for x in pn], ['a', 'b'])
        dln = float(np.mean(np.log([x['npz_spk'] for x in a24n])) - np.mean(np.log([x['npz_spk'] for x in p3n])))
        ip['spk_adj'] = {'prop_npz_n': len(pn), 'dead_slope': fp_['coef']['b'], 'dln': dln,
                         'expected_dead_diff': fp_['coef']['b']['est'] * dln}
    res['npz_intra_pair'] = ip

    # intra 같은 스파이크 정확도: prop 곡선 vs abl_noinv (겹치는 범위 160~300K) + 스파이크 맞춘 짝
    rng_p = [r for r in prop if 160e3 <= r['spikes'] <= 300e3]
    rng_a = [r for r in abl if 160e3 <= r['spikes'] <= 300e3]
    dd_ = rng_p + rng_a
    s0i = A(rng_a, 'spikes').mean()
    X = np.c_[np.ones(len(dd_)), np.log(A(dd_, 'spikes')) - math.log(s0i),
              [1.0 if r['key'] == 'abl_noinv' else 0.0 for r in dd_]]
    fi = ols(X, A(dd_, 'val_acc'), ['a', 'b', 'd'])
    res['intra_acc'] = {'n_prop': len(rng_p), 'n_abl': len(rng_a), 'd': fi['coef']['d'], 'b': fi['coef']['b'],
                        'resid_sd': fi['resid_sd'],
                        'servers_prop': {s: sum(r['server'] == s for r in rng_p) for s in ('138', '23')},
                        'servers_abl': {s: sum(r['server'] == s for r in rng_a) for s in ('138', '23')},
                        'pair': {'acc': welch(A(a24, 'val_acc'), A(p3, 'val_acc')),
                                 'abl_mean': A(a24, 'val_acc').mean(), 'prop_mean': A(p3, 'val_acc').mean()}}
    # intra 학습 모드 기전 (같은 스파이크) — 정의가 팔마다 달라 w 는 비교 안 함, 분포 지표만
    pm = [r for r in prop if r['mech']]
    am = [r for r in abl if r['mech'] and 160e3 <= r['spikes'] <= 300e3]
    im = {}
    if len(am) >= 2:
        for k in ('gini_sw', 'top10_sw', 'dead_pct', 'share_conv2', 'share_conv3'):
            xx = np.r_[np.log([r['mech']['tr_spk'] for r in pm]), np.log([r['mech']['tr_spk'] for r in am])]
            yy = np.r_[[r['mech'][k] for r in pm], [r['mech'][k] for r in am]]
            gg = np.r_[np.zeros(len(pm)), np.ones(len(am))]
            ff = ols(np.c_[np.ones(len(xx)), xx, gg], yy, ['a', 'b', 'd'])
            im[k] = {'d': ff['coef']['d'], 'n': [len(pm), len(am)]}
    res['intra_mech'] = im

    # ---------------------------------------------------------------- 5. 학습 시간
    tt = defaultdict(list)
    for r in runs:
        if r['seed'] >= 900 or r['sec_med'] is None:
            continue
        if r['key'] in ('base', 'ours_ch', 'ch_noinv', 'prop', 'abl_noinv'):
            tt[(r['key'], r['server'])].append(r)
    tsum = {}
    for (k, s), rs in tt.items():
        v = A(rs, 'sec_med')
        tsum[f'{k}|{s}'] = {'n': len(rs), 'median': float(np.median(v)), 'min': float(v.min()), 'max': float(v.max()),
                            'tags': [r['tag'] for r in rs]}
    res['time'] = tsum
    # 같은 시각 비교: ch_noinv 런의 [시작, 끝] 과 50% 이상 겹친 sejong ours_ch 런
    def window(r):
        h = r['host'].split('\t')
        try:
            import datetime as _dt
            st = _dt.datetime.strptime(h[-1].strip(), '%Y-%m-%d %H:%M').timestamp()
        except Exception:
            return None
        return (st, r['mtime']) if r['mtime'] else None
    conc = []
    for q in [r for r in runs if r['key'] == 'ch_noinv' and r['seed'] < 900]:
        wq = window(q)
        if not wq:
            continue
        ov = []
        for r in runs:
            if r['server'] != q['server'] or r['seed'] >= 900 or r['tag'] == q['tag'] or r['sec_med'] is None:
                continue
            if r['key'] not in ('ours_ch', 'base', 'ch_noinv'):
                continue
            wr = window(r)
            if not wr:
                continue
            o_ = max(0.0, min(wq[1], wr[1]) - max(wq[0], wr[0]))
            if o_ >= 0.5 * (wq[1] - wq[0]):
                ov.append({'tag': r['tag'], 'sec_med': r['sec_med'], 'gpu': r['host'].split('\t')[2] if '\t' in r['host'] else ''})
        conc.append({'tag': q['tag'], 'sec_med': q['sec_med'], 'gpu': q['host'].split('\t')[2] if '\t' in q['host'] else '',
                     'start': wq[0], 'end': wq[1], 'overlap': ov})
    res['time_concurrent'] = conc
    return res


# ============================================================================ 보고
def fp(p):
    return f'{p:.3f}' if p >= 0.001 else f'{p:.1e}'


def sk(x):
    return f'{x:,.0f}'


def write_md(runs, res, dups):
    L = []
    w = L.append
    bm = res['base_mean']
    noi = [r for r in runs if r['key'] == 'ch_noinv' and r['use']]
    nr = noi[0]
    anc = res['ancova']
    ls, qs = anc['lin_srv']['coef']['d'], anc['quad_srv']['coef']['d']
    pw = res['power']
    sc = res['scale']
    lo = sc['local']['rows'][0]
    st = sc.get('stale')
    mo = res['mech'][0]
    mt = res['mech_tr'][0]
    mm = mo['m']
    ip = res['npz_intra_pair']
    ia = res['intra_acc']
    vb = res['vs_base'][0]
    sd_pool = res['pooled']['acc_sd']
    tm = res['time']
    e195 = {o['tag']: o for o in res['mech_ep195']}
    s195 = res['mech_ep195_slope']
    tg = res['npz_target'][nr['tag']]
    it = sc['intra']

    w('# `- 1- inversion (inter)` — 반전이 무엇을 바꾸나 (R19 CIFAR-10, 26-10-14)')
    w('')
    w('- 요청 (사용자 10-14): "어블레이션 각 방법 분석도 해줄 수 있어?" — 이 문서는 **1− inversion** 몫. '
      '`ours_inter_ch` (run 키 `ours_ch`) 의 채널 가중 w = 1 − c/max c 에서 반전만 끈 팔 `- 1- inversion (inter)` '
      '(run 키 `ch_noinv`, w = c/max c; c = 채널 공간합, 침묵 위치는 vmem readiness 포함) 을 본다.')
    w('- 정확도 = 310 에폭 중 best val acc, spikes = 그 에폭 s_count (원값, 이미지당, 평가 모드). train.log 에서 직접 다시 읽음. '
      '팔은 config override 블록으로 확인 (`ch_noinv` 와 `ours_ch` 의 차이는 `reg_spike_maxnorm_no_inv` 와 ρ 뿐).')
    w(f'- 대상: `ch_noinv` 완주·비배제 **n={len(noi)}** (ρ 1e-2 s1, 23) + 중단 1런 (ρ 2.4e-2 s1, 23, 195 에폭 — 정확도 비교에서 제외, '
      f'기전·눈금의 방향 확인에만 씀). `ours_ch` 완주·비배제 n={res["n"]["ours_ch"]}, baseline n={res["n"]["base"]}. '
      f'교차 확인: intra `abl_noinv` 완주·비배제 n={res["n"]["abl_noinv"]}, `prop` (= ours_intra_ch) n={res["n"]["prop"]}. '
      '`abl_noinv_wc`·`lw1_noinv` (반전만 끈 intra·layer 판) 는 **0런** — 큐에 투입된 적 없음. `- final_step`·`- vmem`·`- loss-ratio` 팔은 섞지 않음.')
    w('- 재현: `~/anaconda3/envs/venv_1/bin/python analysis/ablation_inversion_261014.py` '
      f'(sejong 캐시 `{CACHE}/ep23.txt.gz`·`csv23.tar.gz`, `--refetch` 로 다시 읽음). 수치 원본 `{OUT_JSON}`.')
    w('')

    # ------------------------------------------------------------ 요약
    w('## 요약')
    w('')
    w(f'1. **같은 스파이크 정확도 — 판정 불가 (n=1).** `- 1- inversion` {nr["val_acc"]:.2f}% @ {sk(nr["spikes"])} (23). '
      f'ours 곡선 (200~420K, n={res["ancova_n"]["ours_ch"]}) 대비 오프셋 d = **{ls["est"]:+.3f}%p** '
      f'[{ls["lo"]:+.3f}, {ls["hi"]:+.3f}] p={fp(ls["p"])} (선형+서버), 2차+서버 {qs["est"]:+.3f}%p '
      f'[{qs["lo"]:+.3f}, {qs["hi"]:+.3f}]. 지금 설계로 볼 수 있는 최소 차는 {pw["mde_by_n"]["1"]:.2f}%p (n=1), '
      f's2·s3 (hold) 가 오면 {pw["mde_by_n"]["3"]:.2f}%p. baseline (n={bm["n"]}) 대비 {vb["acc_diff"]:+.3f}%p '
      f'[{vb["acc_lo"]:+.3f}, {vb["acc_hi"]:+.3f}] · 스파이크 {vb["spk_pct"]:+.1f}%.')
    w(f'2. **같은 ρ 에서 덜 줄인다 — ρ 눈금이 약 1.5배 어긋난다 (완주 n=1 이라 규칙상 판단 보류, 신호는 큼).** ρ 1e-2 에서 ours (8e-3·1.2e-2 사이 보간) '
      f'{sk(lo["ours_spk_at_rho"])} 대비 `- 1- inversion` {sk(nr["spikes"])} (**{lo["ratio_pct"]:+.0f}%**, z={lo["z"]:+.1f}). '
      f'같은 스파이크를 내는 ours ρ 는 {lo["same_spk"]["rho_eq"]:.2e} (6e-3·7e-3 사이) → 배율 **{lo["factor"]:.2f}** '
      f'[{lo["factor_lo"]:.2f}, {lo["factor_hi"]:.2f}]. 중단 런 2.4e-2 s1 도 195 에폭에서 같은 에폭 ours 대비 배율 '
      f'{st["factor_at_E"]:.2f} [{st["factor_at_E_lo"]:.2f}, {st["factor_at_E_hi"]:.2f}] 로 같다 (1e-2 s1 의 195 에폭 배율 '
      f'{st["e195_noinv"][0]["factor"]:.2f}). 같은 스파이크에서 λ 는 {lo["lam_ratio"]:.2f}배, 과제 손실 L·벌점 R 은 '
      f'ours 와 같다 (비 {sc["decomp"][0]["task_loss_clean"]["ratio"]:.3f}·{sc["decomp"][0]["reg_R"]["ratio"]:.3f}) — '
      'loss-ratio 식 λ = ρL/R 에서 늘어난 것은 ρ (벌점 몫) 뿐이다.')
    w(f'3. **2.4e-2 s1 은 실패가 아니라 중단이다.** train.log 가 196 에폭 112/500 배치에서 오류 없이 끊기고, 큐 기록 '
      f'"10-06 사용자 지시로 중단(착지 116K 예상), 1e-2 로 재투입" 대로 같은 GPU (sejong gpu2) 에서 같은 시각 (10-06 02:31) 에 '
      f'1e-2 s1 이 시작됐다. 195 에폭까지 학습은 정상 (S30/S1={st["s30"]:.3f}, 195 에폭 val acc {st["val_acc_at_E"]:.2f}%). '
      f'ours 의 (착지 / 195 에폭) 스파이크 비 (중앙값 {st["land_ratio_median"]:.3f}, n={st["n_pool"]}) 로 착지 예측 '
      f'{sk(st["pred_land"])} (범위 {sk(st["pred_land_lo"])}~{sk(st["pred_land_hi"])}) — ours 비교 구간 (200~420K) 밖의 저스파이크 점이었다. '
      f'처음 ρ 를 "약 8배" 로 잡은 것은 intra `abl_noinv` (세 요소 제거) 의 배율 ({it["pair"]["factor"]:.0f}배) 을 옮긴 것으로, '
      f'단일 요소 inter 판의 실제 배율 ({lo["factor"]:.1f}배) 보다 5배 컸다.')
    cm, cv_, gs, tp, dp = mm['cn_mean'], mm['cn_cv'], mm['gini_sw'], mm['top10_sw'], mm['dead_pct']
    e2 = e195.get('r19c10-ch_noinv-2p4e-2-s1', {}).get('m', {})
    w(f'4. **기전 — 반전을 끄면 채널이 고르게 남는다 (완주 n=1 + 중단 1런, 같은 방향 — 규칙상 판단 보류, 신호는 ours 시드 SD 의 15~50배).** 두 팔에서 같은 양인 '
      f'm = mean(c/max c) (1등 채널 대비 평균 채널의 비율) 이 같은 스파이크에서 ours {cm["ours_pred"]:.3f} → **{cm["value"]:.3f}** '
      f'(z={cm["z"]:+.0f}), 채널 변동계수 {cv_["ours_pred"]:.2f} → {cv_["value"]:.2f} (z={cv_["z"]:+.0f}), 뉴런 지니 '
      f'{gs["delta"]:+.3f} (z={gs["z"]:+.0f}), 상위 10% 뉴런 몫 {tp["delta"]:+.1f}%p (z={tp["z"]:+.0f}). '
      + (f'중단 런 (195 에폭, 136K) 은 m {e2["cn_mean"]["ours_pred"]:.3f} → {e2["cn_mean"]["value"]:.3f}, 지니 {e2["gini_sw"]["delta"]:+.3f} 로 더 크다. '
         if e2 else '')
      + f'ours 는 스파이크를 줄일수록 채널이 모이지만 (195 에폭 m 기울기 {s195["cn_mean"]["ours_b"]["est"]:+.3f}/ln 스파이크, n={s195["cn_mean"]["ours_n"]}), '
      f'`- 1- inversion` 두 런은 {s195["cn_mean"]["noinv_two_point_b"]:+.3f} 로 1/3 수준이다. '
      '10-10 의 "평균 w 0.81 → 0.34" 는 대부분 정의 뒤집기 (w ↔ 1−w) 였고, 실제 분포 변화는 m 0.19 → 0.34 다.')
    w(f'5. **죽은 채널 — inter 팔은 npz 가 없어 측정 전.** ours npz 곡선 (n={tg["n_fit"]}) 의 같은 스파이크 예측은 '
      f'{tg["dead_ch"]["pred"]:.0f} ±{tg["dead_ch"]["se_new"]:.0f} / 4,736 — 측정값이 {tg["dead_ch"]["z_thresh"]:.0f} 개 이상 벗어나면 n=1 로도 '
      f'방향이 보인다. intra 교차 확인 (세 요소 제거, 스파이크 맞춘 짝 n=3 vs 3): 죽은 채널 {ip["dead_ch"]["diff"]:+.0f} '
      f'±{ip["dead_ch"]["se"]:.0f} (p={fp(ip["dead_ch"]["p"])}), 이미지당 활성 뉴런 {ip["active_img"]["pct"]:+.1f}% (p={fp(ip["active_img"]["p"])}), '
      f'채널 지니 {ip["ch_gini_sw"]["diff"]:+.3f} (p={fp(ip["ch_gini_sw"]["p"])}) — inter 의 학습 모드 결과와 같은 방향.')
    w(f'6. **intra 교차 확인은 같은 그림이지만 단일 요소가 아니다.** `abl_noinv` 은 반전·within_channel·vmem 을 함께 뺀 팔이다 '
      f'(neurons.py maxnorm_plain 가지, decisions 09-20). 같은 스파이크 정확도 {ia["d"]["est"]:+.3f}%p '
      f'[{ia["d"]["lo"]:+.3f}, {ia["d"]["hi"]:+.3f}] p={fp(ia["d"]["p"])} (prop n={ia["n_prop"]} vs abl n={ia["n_abl"]}, 160~300K; '
      f'서버가 팔과 겹친다 — prop 은 138 {ia["servers_prop"]["138"]}·23 {ia["servers_prop"]["23"]}, abl 은 전부 23). '
      f'스파이크 맞춘 짝 (abl 2.4e-2 n=3 vs prop 3e-3 n={it["pair"]["prop3_n"]}) {ia["pair"]["acc"]["diff"]:+.3f}%p '
      f'[{ia["pair"]["acc"]["lo"]:+.3f}, {ia["pair"]["acc"]["hi"]:+.3f}] p={fp(ia["pair"]["acc"]["p"])}. 둘 다 0 을 포함.')
    w(f'7. **학습 시간 — 차이 없음.** 코드 차이는 `sc_rate = 1 − sc_norm` → `sc_rate = sc_norm` 한 줄. 에폭당 중앙값 sejong: '
      f'`- 1- inversion` {tm["ch_noinv|23"]["median"]:.0f}s (n={tm["ch_noinv|23"]["n"]}), `ours_inter_ch` {tm["ours_ch|23"]["median"]:.0f}s '
      f'(n={tm["ours_ch|23"]["n"]}), baseline {tm["base|23"]["median"]:.0f}s (n={tm["base|23"]["n"]}).')
    w('')
    w('**판정**: 코드상 반전이 정하는 것은 **"어느 채널을 누르나"** 다 (반전 있음: 약한 채널, 없음: 1등 채널). 관측 (완주 n=1 + 중단 1런, '
      '아직 판단 보류): 반전이 있으면 발화가 소수 채널로 모이고 스파이크를 줄일수록 더 모이며, 없으면 채널이 평준화되고 '
      f'같은 스파이크까지 내리는 데 벌점 몫 ρ 가 약 {lo["factor"]:.1f}배 더 든다. 같은 스파이크 정확도는 n=1 이라 **판정 불가** '
      f'(관측 {ls["est"]:+.2f}%p, 검출 하한 {pw["mde_by_n"]["1"]:.2f}%p). "반전이 채널을 죽여 구조 희소성을 만든다" 는 inter 팔에서 '
      '**아직 측정 전** (npz 1런, GPU 약 17분) 이고, 지금 근거는 intra 세 요소 제거 팔뿐이다.')
    w('')

    # ------------------------------------------------------------ 정의
    w('## 정의')
    w('')
    w('- **c**: 채널 공간합 = Σ_위치 scp. scp = T 합 스파이크 (한 번이라도 쏜 위치) 또는 막전위/문턱 clip(0,1) (한 번도 안 쏜 위치, vmem). '
      '이미지마다 따로. **max c**: 같은 층·같은 이미지의 채널 중 최대 (neurons.py:1019-1033). fc1 (2D) 은 채널 묶음 없이 층 전체 max.')
    w('- **w** (`reg_detail.csv` 의 `sc_rate_mean`, 위치·채널·배치 평균): ours = 1 − c/max c (1등 채널 0, 침묵 채널 ≈1), '
      '`- 1- inversion` = c/max c (1등 채널 1, 침묵 채널 ≈0; neurons.py:1047-1062). 18층 평균은 뉴런 수 가중. '
      '**두 팔의 w 를 그대로 비교하면 정의가 뒤집힌 것까지 섞인다.**')
    w('- **m = mean(c/max c)** (이 문서에서 새로 씀): 두 팔에서 같은 양 — ours 는 1 − w, `- 1- inversion` 은 w. 1등 채널 대비 평균 채널의 비율; '
      '클수록 채널이 고르다. **채널 CV** = std(c/max c) / m. std 는 `reg_neuron_detail.csv` 의 `sc_rate_std` '
      '(std(1−x) = std(x) 라 두 팔 같은 양; 이미지마다 층 안 뉴런 위 표준편차 → 배치 평균, neurons.py:1229-1232). 층별 CV 를 뉴런 수 가중.')
    w('- **뉴런 지니·상위 10% 몫** (`reg_neuron_detail.csv`, neurons.py:1449-1462): 배치 (100장) 평균 뉴런별 T 합 스파이크의 층 안 집중도 → '
      '층 스파이크 가중 평균. 뉴런 단위 (채널 단위 아님). **이미지당 침묵 %** = 이미지 하나에서 한 번도 안 쏜 뉴런 비율의 배치 평균 (뉴런 수 가중).')
    w('- **학습 모드 지표**: 마지막 에폭 (310) 학습 배치 평균, 증강 포함. 원장 spikes (평가 모드, best 에폭) 와 다른 자료다.')
    w('- **같은 스파이크 Δ (기전)**: ours 런 (원장 spikes 180~330K, 10-10 과 같은 대역) 으로 `지표 = a + b·ln(spikes)` 적합 → '
      '`- 1- inversion` 의 spikes 에서 예측. **z** = (값 − 예측) / 새 런 하나의 예측 SE (곡선 SE + 잔차 SD). 잔차 SD 는 ours 시드 산포를 담는다; '
      '`- 1- inversion` 자신의 시드 산포는 n=1 이라 모른다. x 를 학습 모드 스파이크로 바꾼 값도 함께 적었다.')
    w('- **벌점 기울기** (layers.py:2024-2053, decisions 09-20): 스파이크 하나가 받는 기울기 = λ·w_i²/R_ℓ (× surrogate). 쏜 뉴런이든 안 쏜 뉴런이든 '
      'w² 만큼 받는다 (wta_rev). R_ℓ = ‖S_ℓ ⊙ w_ℓ‖₂, R = Σ_ℓ R_ℓ, loss-ratio: λ_{e+1} = ρ·L_e/R_e.')
    w('- **ρ 배율**: `- 1- inversion` 과 같은 스파이크를 내는 ours ρ 에 대한 `- 1- inversion` ρ 의 비. ours ρ 별 평균 ln 스파이크를 ln-ln 구간 선형으로 '
      '이은 곡선 위에서 역산 (주 결과). 구간 = 이웃 두 ρ 평균 SE + noinv 한 런의 시드 산포 (ours 합동 ln SD) 를 국소 기울기로 옮김, t(df=합동 자유도).')
    w('- **npz 지표** (테스트 1만 장 1회 추론, 평가 모드, best 가중치): **죽은 채널** = 테스트셋 합 0 인 채널 (conv 는 공간합, fc1 은 뉴런 하나; '
      '18층 4,736). **이미지당 활성 뉴런** = Σ 뉴런이 쏜 이미지 수 / 1만. **채널 지니·상위 10%** = 층 안 채널 합의 집중도를 층 스파이크로 가중. '
      '**제거 시 MACs 감소** = 죽은 채널을 지웠을 때 conv MAC 감소율.')
    w('')

    # ------------------------------------------------------------ 0. 런 표
    w('## 0. 런별 값 (best 에폭)')
    w('')
    w('열: train loss (규제 포함 `loss`) → train acc → val loss → val acc → spikes, 뒤에 파생. 평균 행은 배제 행을 뺀 값, ± 는 시드 SD.')
    w('')
    hdr = ('| 팔 | ρ | seed | 서버 | best ep / 에폭 | train loss | train acc | val loss | val acc | spikes | spikes vs base | λ 평균 | 에폭 s (중앙) | 비고 |\n'
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    w(hdr)
    bspk = bm['spikes']

    def row(r):
        lam = f'{r["lam_mean"]:.2e}' if 'lam_mean' in r else '–'
        sec = f'{r["sec_med"]:.0f}' if r['sec_med'] else '–'
        return (f'| {LABEL[r["key"]]} | {r["knob"]} | s{r["seed"]} | {r["server"]} | {r["best_epoch"]}/{r["epochs"]} | '
                f'{r["train_loss"]:.4f} | {r["train_acc"]:.2f} | {r["val_loss"]:.4f} | {r["val_acc"]:.2f} | {sk(r["spikes"])} | '
                f'{(r["spikes"] / bspk - 1) * 100:+.1f}% | {lam} | {sec} | {r["exclude_reason"] or ""} |')

    def mrow(rs, label, knob):
        A = lambda k: np.array([r[k] for r in rs], float)
        sd = f' ±{A("val_acc").std(ddof=1):.2f}' if len(rs) > 1 else ''
        ssd = f' ±{A("spikes").std(ddof=1):,.0f}' if len(rs) > 1 else ''
        return (f'| **{label} 평균** | {knob} | n={len(rs)} | {"/".join(sorted({r["server"] for r in rs}))} | | '
                f'{A("train_loss").mean():.4f} | {A("train_acc").mean():.2f} | {A("val_loss").mean():.4f} | '
                f'**{A("val_acc").mean():.2f}{sd}** | {A("spikes").mean():,.0f}{ssd} | '
                f'{(A("spikes").mean() / bspk - 1) * 100:+.1f}% | | | |')

    def block(key):
        rs = [r for r in runs if r['key'] == key]
        for k in sorted({r['knob'] for r in rs}, key=lambda s: -1 if s == '-' else float(s)):
            g = sorted([r for r in rs if r['knob'] == k], key=lambda r: r['seed'])
            for r in g:
                w(row(r))
            gu = [r for r in g if r['use']]
            if gu:
                w(mrow(gu, LABEL[key], k))

    for key in ('base', 'ch_noinv', 'ours_ch'):
        block(key)
    w('')
    w('intra 교차 확인 (같은 열 순서):')
    w('')
    w(hdr)
    for key in ('abl_noinv', 'prop'):
        block(key)
    w('')
    w('- 원장에 있으나 표에 없는 행: `ch_noinv` 1e-2 s2·s3 **hold** (10-13, final_step 결정 대기, 미시작), `abl_noinv` 4e-3 s4 '
      '(138, 0 에폭). `abl_noinv_wc`·`lw1_noinv` 는 원장·큐 모두 0행.')
    w('- `- 1- inversion` 2.4e-2 s1 의 best (91.94%, 181 에폭) 는 코사인 LR 이 다 내려가기 전 값이다 — 완주 런과 비교하지 않는다.')
    if dups:
        w('- 두 서버에 같은 이름의 런 (에폭 줄이 더 많은 쪽, 같으면 로컬): ' + '; '.join(f'{d["tag"]} {d["epochs"]}' for d in dups) + ' — 연기시험이라 결과에 안 쓰임.')
    mism = [r for r in runs if r.get('inv_check') and (abs(r['inv_check']['val_acc_diff']) > 0.005
                                                       or abs(r['inv_check']['spikes_diff']) > 1
                                                       or not r['inv_check']['server_match'])]
    nchk = sum(1 for r in runs if r.get('inv_check'))
    w(f'- 원장 (`inventory.json`, {res.get("inv_generated", "")}) 교차검증: {nchk}런 중 val acc·spikes·서버 불일치 {len(mism)}건'
      + (': ' + ', '.join(f'{r["tag"]} ({r["inv_check"]})' for r in mism) if mism else '') + '.')
    w('')

    # ------------------------------------------------------------ 1. 정확도
    w('## 1. 같은 스파이크 정확도 — `- 1- inversion` (n=1) vs ours 곡선')
    w('')
    w(f'`acc = a + b·x [+ c·x²] + d·[noinv] + e·[server=23]`, x = ln(spikes) − ln({sk(nr["spikes"])}). '
      f'ours 200~420K n={res["ancova_n"]["ours_ch"]} (138 {res["ancova_n"]["ours_servers"]["138"]} · 23 '
      f'{res["ancova_n"]["ours_servers"]["23"]}; ρ {", ".join(res["ancova_n"]["ours_rhos"])}) + noinv n={res["ancova_n"]["ch_noinv"]}. '
      '`nofinal_vs_inter_261014.py` 와 같은 방법·범위라 그 문서의 `- final_step` d (+0.048, n=3) 와 바로 비교된다.')
    w('')
    w('| 모형 | n | d (%p) | 95% CI | p | b (%p / ln) | e (서버 23) | 잔차 SD |')
    w('|---|---|---|---|---|---|---|---|')
    for nm, lab in (('lin', '선형'), ('lin_srv', '선형 + 서버'), ('quad', '2차'), ('quad_srv', '2차 + 서버')):
        f = anc[nm]
        c = f['coef']
        e = f'{c["e"]["est"]:+.3f} (p={fp(c["e"]["p"])})' if 'e' in c else '–'
        w(f'| {lab} | {f["n"]} | **{c["d"]["est"]:+.3f}** | [{c["d"]["lo"]:+.3f}, {c["d"]["hi"]:+.3f}] | {fp(c["d"]["p"])} | '
          f'{c["b"]["est"]:+.3f} ±{c["b"]["se"]:.3f} | {e} | {f["resid_sd"]:.3f} |')
    w('')
    w('범위 민감도 (선형 + 서버):')
    w('')
    w('| ours 범위 | ours n | d (%p) | 95% CI | p | b |')
    w('|---|---|---|---|---|---|')
    for k, v in res['ancova_sens'].items():
        w(f'| {k} | {v["n_ours"]} | {v["d"]["est"]:+.3f} | [{v["d"]["lo"]:+.3f}, {v["d"]["hi"]:+.3f}] | {fp(v["d"]["p"])} | '
          f'{v["b"]["est"]:+.3f} (p={fp(v["b"]["p"])}) |')
    w('')
    sa = res['same_acc']
    w(f'- 같은 정확도에서 스파이크 차 (θ = d/b, 선형+서버): {sa["ratio_pct"]:+.1f}% (델타법 [{sa["delta_lo"]:+.1f}, {sa["delta_hi"]:+.1f}]'
      + (f'; Fieller [{sa["fieller"][0]:+.1f}, {sa["fieller"][1]:+.1f}]' if sa['fieller'] else '; Fieller 구간 무한') + ') — 0 을 넓게 포함.')
    w(f'- 검출 하한 (ours 곡선 고정, noinv 반복 수 m, α=0.05·검정력 0.8, 시드 SD = ours 합동 {sd_pool:.3f}%p (자유도 {res["pooled"]["df"]}), '
      f'곡선 SE {pw["curve_se"]:.3f}): ' + ', '.join(f'm={k} → {v:.2f}%p' for k, v in pw['mde_by_n'].items())
      + f', m→∞ → {pw["floor"]:.2f}%p. 관측 |d|={abs(ls["est"]):.3f} 는 '
      + (f'm={pw["n_needed_obs"]} 에서 보인다.' if pw['n_needed_obs'] else 'noinv 를 아무리 늘려도 ours 곡선 오차 하한보다 작아 이 설계로는 안 보인다.'))
    w(f'- baseline 대비 (baseline {bm["val_acc"]:.2f} ±{bm["val_acc_sd"]:.2f}% @ {sk(bm["spikes"])}, n={bm["n"]}): '
      f'`- 1- inversion` {vb["acc_diff"]:+.3f}%p [{vb["acc_lo"]:+.3f}, {vb["acc_hi"]:+.3f}] p={fp(vb["p"])} '
      f'(n=1 의 시드 산포를 ours 합동 SD 로 가정), 스파이크 {vb["spk_pct"]:+.1f}% [{vb["spk_lo"]:+.1f}, {vb["spk_hi"]:+.1f}]. '
      f'같은 스파이크대 ours (±8% 창, n={res["ours_win8"]["n"]}: ρ 6e-3·7e-3) 는 {res["ours_win8"]["acc"]["diff"]:+.3f}%p '
      f'[{res["ours_win8"]["acc"]["lo"]:+.3f}, {res["ours_win8"]["acc"]["hi"]:+.3f}] p={fp(res["ours_win8"]["acc"]["p"])}, '
      f'스파이크 {res["ours_win8"]["spk_pct"]:+.1f}% — 이 스파이크대 (−53%) 는 ours 도 baseline 보다 유의하게 낮다.')
    w('- 판정: **판정 불가.** 점추정은 양수지만 CI 폭이 ±0.22%p 이고, n=1 이라 noinv 자신의 시드 산포도 모른다. '
      '10-10 문서의 "+0.05 ±0.10 로 차이 없다" 는 "차이를 볼 수 없다" 로 고쳐 읽는다.')
    w('')

    # ------------------------------------------------------------ 2. 눈금
    w('## 2. 같은 ρ·같은 λ 에서 스파이크 — 반전이 ρ 눈금을 얼마나 바꾸나')
    w('')
    w(f'ours 곡선 = 310 에폭 완주·붕괴 아님 (best val acc ≥ 95%) n={sc["local"]["n_runs"]} 의 ρ 별 평균 (S30 배제 1.6e-2·2e-2 는 정확도 비교에서만 '
      '빠지고 눈금에는 넣음). ln-ln 구간 선형 보간.')
    w('')
    w('| ours ρ | n | 평균 spikes |')
    w('|---|---|---|')
    for c in sc['local']['curve']:
        w(f'| {c["rho"]:.1e} | {c["n"]} | {sk(c["spikes"])} |')
    w('')
    w('| 비교 | 대상 | ours (보간 구간) | `- 1- inversion` | 차 | z | ρ 배율 [95%] |')
    w('|---|---|---|---|---|---|---|')
    sr, ss = lo['same_rho'], lo['same_spk']
    w(f'| 같은 ρ (1e-2) | 1e-2 s1, 310 에폭 | {sk(lo["ours_spk_at_rho"])} ({sr["bracket"][0]:.0e}~{sr["bracket"][1]:.1e}, n={sr["n"][0]}·{sr["n"][1]}) | '
      f'{sk(nr["spikes"])} | **{lo["ratio_pct"]:+.0f}%** | {lo["z"]:+.1f} | – |')
    w(f'| 같은 스파이크 | 1e-2 s1, 310 에폭 | ρ {ss["rho_eq"]:.2e} ({ss["bracket"][0]:.0e}~{ss["bracket"][1]:.0e}, n={ss["n"][0]}·{ss["n"][1]}, 국소 기울기 {ss["beta_local"]:.2f}) | ρ 1e-2 | – | – | **{lo["factor"]:.2f}** [{lo["factor_lo"]:.2f}, {lo["factor_hi"]:.2f}] |')
    sE, iE = st['same_rho'], st['same_spk']
    w(f'| 같은 ρ (2.4e-2) | 2.4e-2 s1, 195 에폭 | {sk(st["ours_at_rho_E"])} (1.6e-2~2e-2 에서 **외삽**) | {sk(st["s_at_E"])} | {st["ratio_pct"]:+.0f}% | {st["z"]:+.1f} | – |')
    w(f'| 같은 스파이크 | 2.4e-2 s1, 195 에폭 | ρ {iE["rho_eq"]:.2e} ({iE["bracket"][0]:.1e}~{iE["bracket"][1]:.0e}, n={iE["n"][0]}·{iE["n"][1]}) | ρ 2.4e-2 | – | – | **{st["factor_at_E"]:.2f}** [{st["factor_at_E_lo"]:.2f}, {st["factor_at_E_hi"]:.2f}] |')
    eo = st['e195_noinv'][0]
    w(f'| 같은 스파이크 | 1e-2 s1, 195 에폭 | ρ {eo["rho_eq"]:.2e} ({eo["bracket"][0]:.0e}~{eo["bracket"][1]:.0e}) | ρ 1e-2 | – | – | {eo["factor"]:.2f} |')
    w('')
    w(f'- 195 에폭 비교의 시드 산포는 ours 같은 ρ 묶음의 195 에폭 ln 스파이크 합동 SD {st["lnsd_E"] * 100:.1f}% (자유도 {st["df_E"]}). '
      '2.4e-2 의 같은 ρ 값은 ours 의 마지막 두 점 밖이라 외삽이다 — 배율 (같은 스파이크) 은 보간이라 이 문제가 없다.')
    w('- 두 ρ, 두 시점에서 배율이 1.48~1.55 로 같다 → 반전을 끄면 ρ 눈금이 **곱으로 약 1.5배** 밀리는 것으로 보인다 (ln-ln 곡선이 수평 이동). '
      '완주 1런 + 중단 1런 (둘 다 s1) 이라 규칙상 판단 보류 — s2·s3 가 오면 확정.')
    g_ = sc['use']
    a_ = sc['all310']
    w(f'- 참고 (전체 직선 ln s = α + β ln ρ): ours 비배제 n={g_["n"]} (β {g_["slope"]["est"]:.3f}) 로는 배율 {g_["rows"][0]["factor"]:.2f}, 같은 ρ 차 '
      f'{g_["rows"][0]["ratio_pct"]:+.0f}%; S30 배제 포함 n={a_["n"]} (β {a_["slope"]["est"]:.3f}) 로는 {a_["rows"][0]["factor"]:.2f}, {a_["rows"][0]["ratio_pct"]:+.0f}%. '
      'ours 의 ln s–ln ρ 는 높은 ρ 에서 가팔라져 (6e-3→7e-3 기울기 −0.53, 8e-3→1.2e-2 −0.76, 1.6e-2→2e-2 −1.13) 직선 하나로는 1e-2 의 ours 스파이크를 '
      '과대 예측한다 — 그래서 보간을 주 결과로 쓴다.')
    w('')
    lm = sc['lam'][0]
    dc = sc['decomp'][0]
    w(f'같은 λ·λ 분해 (λ 평균 = 310 에폭 평균; 분해는 마지막 10 에폭 평균을 ours 180~330K n={dc["adp_lambda"]["n_fit"]} 의 같은 스파이크 예측과 비교):')
    w('')
    w('| 양 | `- 1- inversion` | ours (같은 스파이크) | 비 | z |')
    w('|---|---|---|---|---|')
    w(f'| λ 평균 (310 에폭) | {nr["lam_mean"]:.3e} | {lo["ours_lam_same_spk"]:.3e} (6e-3·7e-3 보간) | {lo["lam_ratio"]:.2f} | – |')
    for k, lab in (('adp_lambda', 'λ (마지막 10 에폭)'), ('task_loss_clean', 'L 과제 손실 (마지막 10 에폭)'), ('reg_R', 'R 가중 벌점 (마지막 10 에폭)')):
        q = dc[k]
        w(f'| {lab} | {q["value"]:.4g} | {q["ours_pred"]:.4g} | {q["ratio"]:.3f} | {q["z"]:+.1f} |')
    w(f'| 같은 λ 평균에서 ours 착지 (전체 직선, n={sc["lam_n"]}) | {sk(nr["spikes"])} | {sk(lm["same_lam_ours_spk"])} | {lm["same_lam_ratio_pct"] / 100 + 1:.3f} | {lm["same_lam_z"]:+.1f} |')
    w('')
    w('- 같은 스파이크에서 L·R 이 ours 와 같으므로 λ 비 = ρ 비 ≈ 1.5. 즉 반전을 끄면 **같은 스파이크까지 내리는 데 벌점의 몫 (ρ = λR/L) 이 1.5배 '
      '더 든다.** 벌점 기울기 λ·w_i²/R_ℓ 는 반전이 있으면 약한 채널 (w≈1) 에, 없으면 1등 채널 (w≈1) 에 몰린다. 해석: 1등 채널은 과제 신호를 '
      '나르는 채널이라 과제 손실이 버텨 같은 압력에 덜 줄어든다. **이 해석은 직접 재지 않았다** (채널별 과제 기울기 대 벌점 기울기 크기가 필요).')
    w('')
    w('### 2b. 2.4e-2 s1 (195 에폭) — 실패인가 중단인가')
    w('')
    w(f'- 로그: sejong `/srv2/kyccj/EIP/paper/{st["tag"]}/train.log` 의 마지막 줄은 196 에폭 112/500 배치 진행 표시, 오류·Traceback 없음. '
      '체크포인트는 `ep-0181.weights.h5` (그때까지 best) 하나. host.txt 시작 10-05 07:28 (gpu2), 로그 마지막 수정 10-06 02:31.')
    w('- 같은 GPU (sejong gpu2) 에서 `ch_noinv 1e-2 s1` 이 10-06 02:31 에 시작 (host.txt). 큐 `_queue/QUEUE.tsv`: '
      '`fail:1 … ch_noinv 2.4e-2 1 … 10-06 사용자 지시로 중단(착지 116K 예상), 1e-2 로 재투입` / '
      '`done:sejong:gpu2 … ch_noinv 1e-2 1 … 2.4e-2 가 116K 로 착지 예상(-66%)이라 중단, 196에폭 스파이크로 역산한 rho`.')
    w(f'- 학습 상태: S30/S1={st["s30"]:.3f} (배제 기준 0.19 이상). 에폭 1/30/60/100/150/{st["epochs"]} 의 s_count '
      + ' → '.join(sk(v) for v in st['traj_s']) + ', val acc ' + ' → '.join(f'{v:.2f}' for v in st['traj_acc']) + '.')
    w(f'- 착지 예측: ours 붕괴 아닌 런 (n={st["n_pool"]}) 의 (best 에폭 spikes / 195 에폭 s_count) 중앙값 {st["land_ratio_median"]:.3f} '
      f'(범위 {st["land_ratio_range"][0]:.3f}~{st["land_ratio_range"][1]:.3f}; ρ≥8e-3 만 {st["land_ratio_hi_median"]:.3f}, n={st["n_hi"]}) → '
      f'**{sk(st["pred_land"])}** (범위 {sk(st["pred_land_lo"])}~{sk(st["pred_land_hi"])}). `- 1- inversion` 1e-2 s1 자신의 비 '
      f'({st["land_ratio_noinv"][0]:.3f}) 로 보정하면 {sk(st["pred_land_noinvcal"])}. 같은 방법을 1e-2 s1 에 쓰면 '
      + ', '.join(f'예측 {sk(c["pred_land"])} vs 실제 {sk(c["actual"])}' for c in st['check_1e-2']) + ' (+6%).')
    w('- **판정: 사용자 지시에 의한 중단이며 학습 실패가 아니다.** 착지 예측 약 10~12만 은 ours 비교 구간 (200~420K) 밖이었고, ours 의 '
      '같은 스파이크대 (1.6e-2·2e-2) 는 S30/S1 배제 구간이라 정확도 비교에 쓸 수 없는 점이었다.')
    w('')
    w('| 같은 에폭 비교 (ρ≥8e-3) | ρ | 195 에폭 s_count | 195 에폭 val acc | 착지 spikes | 착지 val acc | 비고 |')
    w('|---|---|---|---|---|---|---|')
    for o in sorted(st['ours_same_epoch'], key=lambda o: o['rho']):
        w(f'| {o["tag"]} | {o["rho"]:.1e} | {sk(o["s_at_E"])} | {o["acc_at_E"]:.2f} | {sk(o["final_spk"])} | {o["final_acc"]:.2f} | {o["excl"]} |')
    w(f'| **{st["tag"]}** | {st["rho"]:.1e} | {sk(st["s_at_E"])} | {st["val_acc_at_E"]:.2f} | (예측 {sk(st["pred_land"])}) | – | 중단 |')
    w('')
    w('### 2c. intra 교차 확인 — `abl_noinv` (세 요소 제거) 의 ρ 눈금')
    w('')
    w(f'- 같은 스파이크 짝: `abl_noinv` 2.4e-2 (n={it["pair"]["abl24_n"]}) {sk(it["pair"]["abl24_spk"])} vs `prop` 3e-3 '
      f'(n={it["pair"]["prop3_n"]}) {sk(it["pair"]["prop3_spk"])} (ln 차 {(math.exp(it["pair"]["spk_diff"]["diff"]) - 1) * 100:+.1f}%, '
      f'p={fp(it["pair"]["spk_diff"]["p"])}) → ρ 배율 **{it["pair"]["factor"]:.0f}**.')
    if it['same_rho4']['ln_diff']:
        w(f'- 같은 ρ 4e-3: `abl_noinv` (n={it["same_rho4"]["abl_n"]}) {sk(it["same_rho4"]["abl_spk"])} vs `prop` (n={it["same_rho4"]["prop_n"]}) '
          f'{sk(it["same_rho4"]["prop_spk"])} → {(math.exp(it["same_rho4"]["ln_diff"]["diff"]) - 1) * 100:+.0f}% '
          f'(p={fp(it["same_rho4"]["ln_diff"]["p"])}).')
    w(f'- 기울기 β (ln s / ln ρ, 310 에폭 완주 전부): prop {it["prop"]["b"]["est"]:.3f} ±{it["prop"]["b"]["se"]:.3f} (n={it["prop"]["n"]}), '
      f'abl_noinv {it["abl_noinv"]["b"]["est"]:.3f} ±{it["abl_noinv"]["b"]["se"]:.3f} (n={it["abl_noinv"]["n"]}) — 기울기가 달라 배율이 스파이크 수준마다 다르다.')
    w(f'- inter 단일 요소 (~{lo["factor"]:.1f}배) 와 intra 세 요소 ({it["pair"]["factor"]:.0f}배) 는 같은 "반전" 이름이지만 다른 변경이다. maxnorm_plain 은 '
      '침묵 뉴런에 w=0 (vmem 없음) 을 주고 층 전체 max 로 정규화한다 — 이 차이가 배율 차이의 후보지만 분리해 재지 않았다.')
    w('')

    # ------------------------------------------------------------ 3. 기전
    w('## 3. 기전 — 반전이 바꾸는 것 (학습 모드, 마지막 에폭)')
    w('')
    df_ = next(iter(mm.values()))['df']
    w(f'`- 1- inversion` 1e-2 s1 (원장 spikes {sk(nr["spikes"])}, 학습 모드 {sk(nr["mech"]["tr_spk"])}) vs 같은 스파이크의 ours 예측 '
      f'(ours 원장 spikes 180~330K, n={mo["n_fit"]}; 10-10 4절과 같은 대역·같은 x). z 의 자유도 {df_} — |z|≥{tcrit(df_):.2f} 이면 p<0.05 '
      '(ours 시드 산포 기준). 오른쪽 두 열: x 를 학습 모드 스파이크로 바꾼 값 (ours n=' + str(mt['n_fit']) + '), 10-10 문서 값.')
    w('')
    w('| 지표 | ours 예측 | `- 1- inversion` | Δ (± SE) | z | x=학습 모드: Δ (z) | 10-10 (Δ, z) |')
    w('|---|---|---|---|---|---|---|')
    old = {'w_mean': '-0.468, -52.9', 'gini_sw': '-0.153, -27.4', 'top10_sw': '-14.53, -19.5', 'dead_pct': '-0.62, -7.9',
           'w_conv1': '-0.565, -52.3', 'w_conv2': '-0.442, -44.2', 'w_conv3': '-0.439, -24.8', 'w_conv4': '-0.614, -100.6',
           'gini_conv1': '-0.084, -8.1', 'gini_conv2': '-0.225, -27.3', 'gini_conv3': '-0.101, -11.5', 'gini_conv4': '+0.000, +0.1',
           'dead_conv1': '-1.13, -1.7', 'dead_conv2': '-0.32, -1.7', 'dead_conv3': '-1.23, -5.6', 'dead_conv4': '-0.08, -1.3',
           'share_conv1': '-0.20, -0.4', 'share_conv2': '-2.86, -5.5', 'share_conv3': '+2.88, +6.1', 'share_conv4': '+0.18, +2.5'}
    lab = {'R_rd': 'R (18층 sc_loss 합)', 'w_mean': 'w 평균 (정의가 팔마다 반대 — 참고)', 'cn_mean': '**m = mean(c/max c)** (같은 양)',
           'wsd_mean': 'std(c/max c)', 'cn_cv': '**채널 CV** = std/m', 'gini_sw': '뉴런 지니 (층 스파이크 가중)',
           'top10_sw': '상위 10% 뉴런 몫 (%)', 'dead_pct': '이미지당 침묵 뉴런 %'}
    for s in STAGES[:4]:
        lab[f'cn_{s}'] = f'm {s}'
        lab[f'cn_cv_{s}'] = f'채널 CV {s}'
        lab[f'w_{s}'] = f'w {s} (참고)'
        lab[f'gini_{s}'] = f'지니 {s}'
        lab[f'dead_{s}'] = f'침묵 % {s}'
        lab[f'share_{s}'] = f'층 몫 {s} (%)'
    order = (['cn_mean', 'cn_cv', 'wsd_mean', 'w_mean', 'R_rd', 'gini_sw', 'top10_sw', 'dead_pct']
             + [f'cn_{s}' for s in STAGES[:4]] + [f'cn_cv_{s}' for s in STAGES[:4]] + [f'gini_{s}' for s in STAGES[:4]]
             + [f'dead_{s}' for s in STAGES[:4]] + [f'share_{s}' for s in STAGES[:4]] + [f'w_{s}' for s in STAGES[:4]])
    for k in order:
        if k not in mm:
            continue
        q, qt = mm[k], mt['m'].get(k)
        if q['log']:
            trc = f'{qt["delta"]:+.1f}% ({qt["z"]:+.1f})' if qt else '–'
            w(f'| {lab[k]} | {q["ours_pred"]:,.0f} | {q["value"]:,.0f} | {q["delta"]:+.1f}% (±{q["se"] * 100:.1f}%) | {q["z"]:+.1f} | {trc} | {old.get(k, "–")} |')
        else:
            trc = f'{qt["delta"]:+.3f} ({qt["z"]:+.1f})' if qt else '–'
            w(f'| {lab[k]} | {q["ours_pred"]:.3f} | {q["value"]:.3f} | {q["delta"]:+.3f} (±{q["se"]:.3f}) | {q["z"]:+.1f} | {trc} | {old.get(k, "–")} |')
    w('')
    ma = res['mech_all'][0]['m']
    w(f'- **대역을 넓히면**: ours 비배제 전부 (n={res["mech_all"][0]["n_fit"]}, 147~412K) 로 적합하면 m {ma["cn_mean"]["delta"]:+.3f} (z={ma["cn_mean"]["z"]:+.0f}), '
      f'채널 CV {ma["cn_cv"]["delta"]:+.3f} (z={ma["cn_cv"]["z"]:+.0f}), 지니 {ma["gini_sw"]["delta"]:+.3f} (z={ma["gini_sw"]["z"]:+.0f}), '
      f'상위 10% {ma["top10_sw"]["delta"]:+.1f}%p (z={ma["top10_sw"]["z"]:+.0f}) — 분포 지표는 같다. **이미지당 침묵 % 는 {ma["dead_pct"]["delta"]:+.2f}%p (z={ma["dead_pct"]["z"]:+.1f}) 로 약해져** '
      '대역에 따라 달라진다 (넓은 대역에서 직선이 안 맞아 잔차가 커짐) — 침묵 % 차는 확정된 결과로 쓰지 않는다.')
    w(f'- **10-10 값 재현**: 같은 대역·같은 x 로 다시 읽은 값이 10-10 표와 소수 셋째 자리까지 같다 (n 도 17 로 같음 — 10-10 이후 이 대역에 새 ours 런 없음). '
      f'x 를 학습 모드 스파이크로 바꾸면 침묵 % 만 −0.62 → {mt["m"]["dead_pct"]["delta"]:+.2f}%p 로 커지고 나머지는 같다.')
    w(f'- **"평균 w 0.81 → 0.34" (10-10) 는 대부분 정의 뒤집기다.** 채널 분포가 그대로였다면 `- 1- inversion` 의 w 평균은 1 − 0.81 = '
      f'{cm["ours_pred"]:.2f} 였을 것이다. 같은 양 m 으로 비교하면 {cm["ours_pred"]:.3f} → {cm["value"]:.3f}, 즉 **1등 채널 대비 평균 채널의 비율이 '
      f'{cm["ours_pred"] * 100:.0f}% → {cm["value"] * 100:.0f}%** 로 고르게 된 것이 실제 변화다. 채널 CV {cv_["ours_pred"]:.2f} → {cv_["value"]:.2f}. '
      '단계별로는 conv2 (m 0.17 → 0.39) 가 가장 크고 conv4 는 변화 없다 (m z=+1.0, 지니 z=+0.1).')
    w('- **어느 채널이 눌리나**: 벌점 기울기 λ·w_i²/R_ℓ 는 쏜 뉴런이든 안 쏜 뉴런이든 w² 만큼 걸린다. ours 는 w = 1 − c/max 라 약한 채널 '
      '(w≈1) 이 가장 세게 눌리고 1등 채널은 면제 → 약한 채널이 꺼지고 발화가 소수 채널에 모인다 (m 작음, 지니·상위 10% 몫 큼). '
      '`- 1- inversion` 은 w = c/max 라 1등 채널이 가장 세게 눌리고 약한 채널은 거의 면제 → 채널이 평준화된다 (m 큼, 지니 작음).')
    w(f'- R (가중 벌점 원값) 은 같은 스파이크에서 ours 와 같다 ({mm["R_rd"]["delta"]:+.1f}%, z={mm["R_rd"]["z"]:+.1f}). 층 배분은 conv2 몫 −2.9%p, conv3 몫 +2.9%p (z ±5.5~6.1). '
      '이미지당 침묵 −0.6%p (180~330K 대역에서만 유의, 위 참고).')
    w('')
    w('### 3b. 195 에폭 같은 비교 — 두 번째 `- 1- inversion` 런 (2.4e-2 s1, 중단) 으로 방향 재확인')
    w('')
    w('ours = 붕괴 아닌 런 (best val acc ≥ 95%, S30 배제 1.6e-2·2e-2 포함 — 기전 비교라 정확도 배제 기준을 쓰지 않음) 중 195 에폭 학습 모드 스파이크가 '
      '대상의 1/2~2배인 것. z = 예측 SE 기준, z(시드) = ours 같은 ρ 묶음의 195 에폭 합동 SD 기준 (자유도 17).')
    w('')
    w('| 런 | 195 에폭 tr_spk | ours n (자유도) | m: ours → noinv (z, z시드) | 채널 CV | 뉴런 지니 Δ (z시드) | 상위 10% Δ (z시드) | 침묵 % Δ (z시드) |')
    w('|---|---|---|---|---|---|---|---|')
    for o in res['mech_ep195']:
        m_ = o['m']
        if not m_:
            w(f'| {o["tag"]} | {sk(o["tr_spk"])} | {o["n_fit"]} | (ours 런 부족) | | | | |')
            continue
        w(f'| {o["tag"]} | {sk(o["tr_spk"])} | {o["n_fit"]} ({m_["cn_mean"]["df"]}) | {m_["cn_mean"]["ours_pred"]:.3f} → {m_["cn_mean"]["value"]:.3f} '
          f'({m_["cn_mean"]["z"]:+.0f}, {m_["cn_mean"]["z_seed"]:+.0f}) | {m_["cn_cv"]["ours_pred"]:.2f} → {m_["cn_cv"]["value"]:.2f} | '
          f'{m_["gini_sw"]["delta"]:+.3f} ({m_["gini_sw"]["z_seed"]:+.0f}) | {m_["top10_sw"]["delta"]:+.1f} ({m_["top10_sw"]["z_seed"]:+.0f}) | '
          f'{m_["dead_pct"]["delta"]:+.2f} ({m_["dead_pct"]["z_seed"]:+.1f}) |')
    w('')
    w(f'- 2.4e-2 s1 의 ours 비교군은 4런 (8e-3, 1.2e-2, 1.6e-2, 2e-2, 각 n=1) 이라 예측 SE 의 자유도가 2 다. 그래도 m·지니·상위 10% 의 차가 '
      'ours 시드 SD 의 30배 이상이라 방향은 분명하다.')
    sl = s195
    w(f'- **스파이크를 줄일 때 채널이 얼마나 모이나 (195 에폭)**: ours (n={sl["cn_mean"]["ours_n"]}, {sk(sl["cn_mean"]["ours_range"][0])}~'
      f'{sk(sl["cn_mean"]["ours_range"][1])}) 의 ln 스파이크당 기울기 m {sl["cn_mean"]["ours_b"]["est"]:+.3f} ±{sl["cn_mean"]["ours_b"]["se"]:.3f}, '
      f'지니 {sl["gini_sw"]["ours_b"]["est"]:+.3f} ±{sl["gini_sw"]["ours_b"]["se"]:.3f}, 상위 10% {sl["top10_sw"]["ours_b"]["est"]:+.1f} ±{sl["top10_sw"]["ours_b"]["se"]:.1f}; '
      f'`- 1- inversion` 두 런을 잇는 기울기 m {sl["cn_mean"]["noinv_two_point_b"]:+.3f}, 지니 {sl["gini_sw"]["noinv_two_point_b"]:+.3f}, '
      f'상위 10% {sl["top10_sw"]["noinv_two_point_b"]:+.1f}. 반전이 없으면 스파이크를 줄여도 집중이 ours 의 약 1/3 만큼만 커진다 '
      '(두 점·다른 ρ 의 같은 시드라 기울기의 불확실성은 못 잰다).')
    w('')
    w('### 3c. 학습 중 궤적 — 차이가 언제 생기나')
    w('')
    w(f'ours 7e-3 (n={len(res["mech_traj_ours7"])}, 착지 220,728 — `- 1- inversion` 1e-2 의 226,949 와 짝) 평균 vs `- 1- inversion` 두 런. 학습 모드.')
    w('')
    w('| 에폭 | tr_spk ours7 | tr_spk noinv 1e-2 | tr_spk noinv 2.4e-2 | m ours7 | m noinv 1e-2 | m noinv 2.4e-2 | 지니 ours7 | 지니 noinv 1e-2 | 지니 noinv 2.4e-2 |')
    w('|---|---|---|---|---|---|---|---|---|---|')

    def cell(row, nm, k, f):
        v = row.get(nm, {}).get(k)
        return f.format(v[0]) if v else '–'
    for e, row in res['mech_traj'].items():
        w(f'| {e} | {cell(row, "ours7", "tr_spk", "{:,.0f}")} | {cell(row, "noinv1e-2", "tr_spk", "{:,.0f}")} | '
          f'{cell(row, "noinv2.4e-2", "tr_spk", "{:,.0f}")} | {cell(row, "ours7", "cn_mean", "{:.3f}")} | '
          f'{cell(row, "noinv1e-2", "cn_mean", "{:.3f}")} | {cell(row, "noinv2.4e-2", "cn_mean", "{:.3f}")} | '
          f'{cell(row, "ours7", "gini_sw", "{:.3f}")} | {cell(row, "noinv1e-2", "gini_sw", "{:.3f}")} | '
          f'{cell(row, "noinv2.4e-2", "gini_sw", "{:.3f}")} |')
    w('')
    w('- 1·10 에폭에는 세 런이 같다 (규제가 아직 스파이크를 거의 못 줄인 때). 30 에폭부터 갈라져, 60 에폭 이후 ours 는 m 이 0.17~0.19 에 머물고 '
      '`- 1- inversion` 1e-2 는 60 에폭 0.26 에서 310 에폭 0.35 로 계속 오른다 — 반전이 없으면 학습이 진행될수록 채널이 더 고르게 된다 (n=1).')
    w('')

    # ------------------------------------------------------------ 4. npz
    w('## 4. 죽은 채널·활성 뉴런 (npz, 평가 모드)')
    w('')
    w(f'- `- 1- inversion` (inter) 의 npz: **{"없음" if not res["npz_noinv"] else ", ".join(res["npz_noinv"])}** (`_npz`, `_npz_t`, `deep1005/npz` 모두). '
      '채널을 실제로 덜 죽이는지는 측정 전이다.')
    w(f'- 측정했을 때의 기준값 — ours npz 곡선 ({NPZ_BAND[0] / 1e3:.0f}~{NPZ_BAND[1] / 1e3:.0f}K, n={tg["n_fit"]}: '
      + ', '.join(t.replace('r19c10-ours_ch-', '') for t in tg['tags']) + f') 의 {sk(nr["spikes"])} 예측:')
    w('')
    w('| 지표 | ours 예측 | 새 런 하나 예측 SE | n=1 로 p<0.05 가 되려면 필요한 차 |')
    w('|---|---|---|---|')
    for k, lb in (('dead_ch', '죽은 채널 (/4,736)'), ('macs_rm_pct', '제거 시 MACs 감소 %'), ('active_img', '이미지당 활성 뉴런'),
                  ('never_fired_pct', '테스트셋 전체 침묵 뉴런 %'), ('ch_gini_sw', '채널 지니'), ('ch_top10_sw', '채널 상위 10% 몫 (%)')):
        if k not in tg:
            continue
        q = tg[k]
        dg = {'dead_ch': 0, 'macs_rm_pct': 2, 'never_fired_pct': 2, 'ch_gini_sw': 3, 'ch_top10_sw': 2}.get(k, 3)
        if q['log']:
            w(f'| {lb} | {q["pred"]:,.0f} | ±{q["se_new"] * 100:.1f}% | ±{q["z_thresh"] * 100:.1f}% |')
        else:
            w(f'| {lb} | {q["pred"]:.{dg}f} | ±{q["se_new"]:.{dg}f} | ±{q["z_thresh"]:.{dg}f} |')
    w('')
    w('### 4b. intra 교차 확인 — `abl_noinv` 2.4e-2 vs `prop` 3e-3 (스파이크 맞춘 짝, npz)')
    w('')
    w(f'prop: {", ".join(ip["prop_tags"])} (138) / abl_noinv: {", ".join(ip["abl_tags"])} (23). Welch t (n={len(ip["prop_tags"])}·{len(ip["abl_tags"])}). '
      '**세 요소 제거 팔이고 서버가 팔과 겹친다.**')
    w('')
    w('| 지표 | prop 3e-3 | abl_noinv 2.4e-2 | 차 (abl − prop) | 95% CI | p |')
    w('|---|---|---|---|---|---|')
    for k, lb in (('npz_spk', 'npz 스파이크/이미지'), ('val_acc', 'val acc (이 6런)'), ('dead_ch', '죽은 채널'),
                  ('macs_rm_pct', '제거 시 MACs 감소 %'), ('active_img', '이미지당 활성 뉴런'),
                  ('never_fired_pct', '전체 침묵 뉴런 %'), ('ch_gini_sw', '채널 지니'), ('ch_top10_sw', '채널 상위 10% 몫 (%)')):
        if k not in ip:
            continue
        q = ip[k]
        if 'pct' in q:
            w(f'| {lb} | {q["prop"]:,.0f} | {q["abl"]:,.0f} | {q["pct"]:+.1f}% | [{q["lo"]:+.1f}, {q["hi"]:+.1f}]% | {fp(q["p"])} |')
        else:
            w(f'| {lb} | {q["prop"]:.3f} ±{q["prop_sd"]:.3f} | {q["abl"]:.3f} ±{q["abl_sd"]:.3f} | {q["diff"]:+.3f} | [{q["lo"]:+.3f}, {q["hi"]:+.3f}] | {fp(q["p"])} |')
    for s in STAGES:
        q = ip[f'dead_{s}']
        w(f'| 죽은 채널 {s} | {q["prop"]:.1f} | {q["abl"]:.1f} | {q["diff"]:+.1f} | [{q["lo"]:+.1f}, {q["hi"]:+.1f}] | {fp(q["p"])} |')
    w('')
    w(f'- 10-10 문서 4b 의 "죽은 채널 −100 ±11, 활성 뉴런 +13.5%" 재계산: {ip["dead_ch"]["diff"]:+.1f} ±{ip["dead_ch"]["se"]:.1f}, '
      f'{ip["active_img"]["pct"]:+.1f}% — 같다. npz 스파이크는 abl 쪽이 {ip["npz_spk"]["pct"]:+.1f}% 다르다 (원장 s_count 차는 '
      f'{(math.exp(it["pair"]["spk_diff"]["diff"]) - 1) * 100:+.1f}%). prop npz (n={ip["spk_adj"]["prop_npz_n"]}) 의 죽은 채널–ln 스파이크 기울기 '
      f'{ip["spk_adj"]["dead_slope"]["est"]:+.0f} ±{ip["spk_adj"]["dead_slope"]["se"]:.0f} 로 이 스파이크 차가 만들 죽은 채널 차는 '
      f'{ip["spk_adj"]["expected_dead_diff"]:+.1f} 개 — 관측 {ip["dead_ch"]["diff"]:+.0f} 은 스파이크 차로 설명되지 않는다.')
    w('')
    if res['intra_mech']:
        w('intra 학습 모드 분포 (prop 전체 vs abl_noinv 160~300K, x = 학습 모드 스파이크 ANCOVA d):')
        w('')
        w('| 지표 | n (prop / abl) | d (abl − prop) | 95% CI | p |')
        w('|---|---|---|---|---|')
        for k, lb in (('gini_sw', '뉴런 지니'), ('top10_sw', '상위 10% 몫 (%)'), ('dead_pct', '이미지당 침묵 %'),
                      ('share_conv2', '층 몫 conv2 (%)'), ('share_conv3', '층 몫 conv3 (%)')):
            q = res['intra_mech'][k]
            w(f'| {lb} | {q["n"][0]} / {q["n"][1]} | {q["d"]["est"]:+.3f} | [{q["d"]["lo"]:+.3f}, {q["d"]["hi"]:+.3f}] | {fp(q["d"]["p"])} |')
        w('')
        w('- 지니·상위 10%·침묵 % 는 inter 와 같은 방향 (덜 모이고 더 많은 뉴런이 쏨). 층 배분 (conv2→conv3 이동) 은 intra 에서 보이지 않는다.')
        w('')

    # ------------------------------------------------------------ 5. 시간
    w('## 5. 학습 시간 (에폭당 초, 런별 중앙값 → 묶음 중앙값; 1 에폭 제외; 연기시험 제외, 미완주·S30 배제 런 포함)')
    w('')
    w('| 팔 | 서버 | n | 중앙값 | 범위 |')
    w('|---|---|---|---|---|')
    for k in ('base', 'ours_ch', 'ch_noinv', 'prop', 'abl_noinv'):
        for s in ('138', '23'):
            q = tm.get(f'{k}|{s}')
            if q:
                w(f'| {LABEL[k]} | {s} | {q["n"]} | {q["median"]:.0f} | {q["min"]:.0f}~{q["max"]:.0f} |')
    w('')
    for c in res['time_concurrent']:
        if c['overlap']:
            v = np.array([o['sec_med'] for o in c['overlap']])
            w(f'- 같은 시각 비교 — {c["tag"]} ({c["gpu"]}, {c["sec_med"]:.0f}s) 와 학습 기간이 50% 이상 겹친 같은 서버 ours/base 런 {len(v)}개: '
              f'중앙값 {np.median(v):.0f}s (' + ', '.join(f'{o["tag"].replace("r19c10-", "")} {o["sec_med"]:.0f}' for o in c['overlap']) + ').')
        else:
            w(f'- 같은 시각 비교 — {c["tag"]} ({c["gpu"]}, {c["sec_med"]:.0f}s): 학습 기간이 50% 이상 겹친 같은 서버 ours/base 런 없음.')
    w('- 코드 차이는 `sc_rate = 1 − sc_norm` 대신 `sc_rate = sc_norm` 한 줄 (neurons.py:1047-1064) 이라 연산량이 같다. 측정값도 ours 와 같다.')
    w('')

    # ------------------------------------------------------------ 6. 판정
    w('## 6. 판정 가능 / 불가 / 필요한 측정')
    w('')
    w('| 질문 | 상태 | 근거 (n) |')
    w('|---|---|---|')
    w(f'| 같은 스파이크에서 정확도가 다른가 | **판정 불가** | n=1, d {ls["est"]:+.3f} [{ls["lo"]:+.2f}, {ls["hi"]:+.2f}]; 검출 하한 {pw["mde_by_n"]["1"]:.2f}%p (n=3 이면 {pw["mde_by_n"]["3"]:.2f}) |')
    w(f'| baseline 대비 무손실인가 | **판정 불가** | n=1, {vb["acc_diff"]:+.3f} [{vb["acc_lo"]:+.2f}, {vb["acc_hi"]:+.2f}] |')
    w(f'| 같은 ρ 에서 덜 줄이나 (ρ 눈금) | **판단 보류 (완주 n=1)** — 신호 큼, 두 런·두 시점 일치 | 1e-2: {lo["ratio_pct"]:+.0f}% (z={lo["z"]:+.1f}); 배율 {lo["factor"]:.2f} (1e-2) · {st["factor_at_E"]:.2f} (2.4e-2, 195 에폭) · {eo["factor"]:.2f} (1e-2, 195 에폭) |')
    w(f'| 채널을 고르게 만드나 (m, 채널 CV, 뉴런 지니, 상위 10%) | **판단 보류 (완주 n=1)** — 신호 큼, 두 런 일치 | 완주 1런 z {cm["z"]:+.0f}·{cv_["z"]:+.0f}·{gs["z"]:+.0f}·{tp["z"]:+.0f}; 중단 1런 (195 에폭) 같은 방향, 더 큼 |')
    w('| 채널을 덜 죽이나 (inter) | **측정 전** | npz 없음 |')
    w(f'| 채널을 덜 죽이나 (intra 세 요소 제거) | 유의 (n=3 vs 3, 단일 요소 아님) | {ip["dead_ch"]["diff"]:+.0f} ±{ip["dead_ch"]["se"]:.0f}, p={fp(ip["dead_ch"]["p"])} |')
    w('| 스텝 분포가 달라지나 | **측정 전** | npz_t 없음 |')
    w('| 2.4e-2 s1 은 실패인가 | **중단 (사용자 지시)** | 로그 (오류 없이 196 에폭 중간 끊김) + 큐 기록 + 같은 GPU 같은 시각 재투입 |')
    w(f'| 학습 시간 | 차이 없음 | sejong {tm["ch_noinv|23"]["median"]:.0f}s (n={tm["ch_noinv|23"]["n"]}) vs {tm["ours_ch|23"]["median"]:.0f}s (n={tm["ours_ch|23"]["n"]}) |')
    w('| 반전 단독의 intra·layer 판 (`abl_noinv_wc`, `lw1_noinv`) | 데이터 없음 | 0런 |')
    w('')
    w('**필요한 측정** (모두 GPU 작업 — 실행은 승인 사항, 이 분석에서는 돌리지 않음):')
    w('')
    w('1. `r19c10-ch_noinv-1e-2-s1` npz (스텝별 키 포함) 추출 — `extract_neuron_spikes.py`, best 가중치 `ep-0297.weights.h5` (sejong '
      '`/srv2/kyccj/EIP/paper/r19c10-ch_noinv-1e-2-s1/ResNet19_CIFAR10/ep-310_..._7/`). 테스트 1만 장 1회 추론, GPU 1장 약 17분 '
      '(10-10 ch_nofinal·ch_novmem 추출 실측). 세종에서 돌리거나 config+weights 를 읽기 전용 rsync 로 canus 에 복사. '
      '얻는 것: 죽은 채널·MACs·활성 뉴런 (4절 기준값과 비교), 채널 지니, 스텝 분포. 가장 싸고 "반전이 구조 희소성을 만드나" 에 바로 답한다.')
    w(f'2. `ch_noinv` 1e-2 s2·s3 (hold) 완주 — 학습 2런, 각 약 30시간. 같은 스파이크 정확도 검출 하한 {pw["mde_by_n"]["1"]:.2f} → {pw["mde_by_n"]["3"]:.2f}%p, '
      '기전 지표·ρ 배율에 noinv 자신의 시드 산포를 준다. 재투입 여부는 final_step 결정 (10-14 사전 등록 규칙) 에 묶여 있다.')
    w('3. (선택) `abl_noinv_wc` 1런 (약 30시간) — intra 교차 확인을 단일 요소로 바꾼다. 논문 어블레이션이 inter 기준이면 불필요.')
    w('')
    w('## 7. 10-10 문서 (ablation_mech_261010.md 4절) 대비 — 갱신·정정')
    w('')
    w('10-10 이후 `- 1- inversion` 새 완주 런은 없다 (s2·s3 hold). ours 비교 집합 (180~330K) 도 그대로 n=17 이다. 바뀐 것은 해석·정의와 새 지표다.')
    w('')
    w('| 10-10 문장 | 지금 | 종류 |')
    w('|---|---|---|')
    w(f'| 정확도 "+0.05 ±0.10 로 차이 없다" | ANCOVA d {ls["est"]:+.3f} [{ls["lo"]:+.2f}, {ls["hi"]:+.2f}] (n=1) — "차이 없다" 가 아니라 **판정 불가** | 정정 (표현) |')
    w(f'| "평균 w 0.34 vs 0.81" 을 분포 변화로 제시 | w 는 두 팔에서 정의가 반대 (w ↔ 1−w). 같은 양 m: {cm["ours_pred"]:.3f} → {cm["value"]:.3f}, 채널 CV {cv_["ours_pred"]:.2f} → {cv_["value"]:.2f} | 정정 |')
    w(f'| 지니 −0.153 (z=−27), 상위 10% −14.5%p (z=−19), 침묵 −0.62%p (z=−7.9), conv2 몫 −2.9 / conv3 +2.9%p | 같은 값 재현. x 를 학습 모드 스파이크로 바꿔도 같은 방향. 단 침묵 % 는 ours 전체 대역 (n=28) 으로 적합하면 z −1.2 — 대역 의존이라 결과로 쓰지 않음 | 재현 + 단서 |')
    w('| 지니·상위 10% 를 "이미지마다 층 안 뉴런 스파이크의 집중도" 로 정의 | 코드상 배치 (100장) 평균 뉴런별 스파이크의 집중도 (neurons.py:1449-1462) | 정정 (정의) |')
    w('| "채널 수(죽은 채널)는 이 팔 npz 가 없어 못 셌다" | 그대로 — 측정 기준값 (ours npz 곡선 예측 848 ±17) 추가 | 변화 없음 |')
    w(f'| (없음) | ρ 눈금 배율 {lo["factor"]:.2f} [{lo["factor_lo"]:.2f}, {lo["factor_hi"]:.2f}], 같은 스파이크에서 λ {lo["lam_ratio"]:.2f}배·L·R 같음 | 새로 |')
    w('| (없음) | 2.4e-2 s1 = 사용자 지시 중단 (실패 아님); 10-05 "ρ 약 8배" 추정은 intra 세 요소 배율을 옮긴 것 | 새로 |')
    w('| (없음) | 중단 런 (195 에폭) 으로 분포 방향 재확인; 195 에폭에서 스파이크를 줄일 때 집중이 ours 의 1/3 만 커짐 | 새로 |')
    w(f'| intra 4b: 죽은 채널 −100 ±11, 활성 뉴런 +13.5% | {ip["dead_ch"]["diff"]:+.1f} ±{ip["dead_ch"]["se"]:.1f}, {ip["active_img"]["pct"]:+.1f}% 재현; npz 스파이크 차로는 설명 안 됨 확인 | 재현 |')
    w('')
    w('## [LIMITATION]')
    w('')
    w('1. `- 1- inversion` 완주 런은 **n=1** (ρ 1e-2 s1, sejong). 정확도 차는 어느 방향으로도 말할 수 없다. 기전 지표·ρ 배율의 z 는 ours 시드 산포 기준이라 '
      'noinv 의 산포가 더 크면 과대평가된다. 중단 런 (2.4e-2 s1, 195 에폭) 으로는 방향만 재확인했고, 이 런도 s1·sejong 이다.')
    w('2. 서버: 두 noinv 런 모두 23. 정확도 ANCOVA 의 서버 항은 0 에 가깝지만 (e={:+.3f}), 기전 지표의 서버 효과는 따로 재지 않았다.'.format(anc['lin_srv']['coef']['e']['est']))
    w('3. 학습 모드 지표는 증강이 걸린 학습 배치 평균이다. 평가 모드 (npz) 로 같은 결론인지는 측정 1 이 필요하다. m·채널 CV 는 vmem readiness 가 c 에 섞인 '
      '값이라 순수 스파이크 채널 분포가 아니다.')
    w('4. "1등 채널을 누르면 과제 손실이 버텨 같은 압력에 덜 줄어든다" 는 해석이다. 채널별 과제 기울기·벌점 기울기를 재지 않았다.')
    w('5. ρ 배율은 ours ρ 별 평균을 ln-ln 구간 선형으로 이은 곡선 위의 값이다. 이웃 ρ 사이 곡률은 무시했고, 1e-2 의 같은 ρ 비교는 n=1 인 8e-3·1.2e-2 사이 보간, '
      '2.4e-2 의 같은 ρ 비교는 외삽이다.')
    w('6. intra 교차 확인 (`abl_noinv`) 은 반전·within_channel·vmem 세 요소를 함께 뺀 팔이고, 비교 구간의 abl 런은 전부 23, prop 런은 대부분 138 이다. '
      '반전 단독의 intra 증거는 없다 (`abl_noinv_wc` 0런).')
    w('7. best 에폭 기준은 310 에폭 최대값이라 시드 잡음을 위로 끌어올린다 (두 팔 공통). spikes 는 s_count 원값 — s_count 와 npz 총합의 불일치 (09-15 보류) 는 해소되지 않았다.')
    w('8. 학습 시간은 같은 서버의 다른 작업 부하에 좌우된다. `- 1- inversion` 은 n=2 (한 런은 195 에폭 중단) 이고 같은 시각 겹친 ours 런은 1개뿐이다.')
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refetch', action='store_true')
    a = ap.parse_args()
    runs, dups, inv = build_runs(a.refetch)
    res = analyze(runs)
    res['inv_generated'] = json.load(open(INV)).get('generated')
    res['dups'] = dups
    slim = [{k: v for k, v in r.items() if k not in ('traj', 'mech_traj')} for r in runs]
    with open(OUT_JSON, 'w') as f:
        json.dump({'runs': slim, 'result': res}, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    open(OUT_MD, 'w').write(write_md(runs, res, dups))
    print('wrote', OUT_JSON, OUT_MD)


if __name__ == '__main__':
    main()
