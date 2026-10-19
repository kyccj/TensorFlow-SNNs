#!/usr/bin/env python
"""R19 CIFAR-10 (ResNet-19, T=4): ours_inter_ch 의 vmem 요소 어블레이션 — 26-10-14.

질문 (사용자 10-14 "어블레이션 각 방법 분석도 해줄 수 있어?", 이 파일의 몫 = vmem)
  - vmem = 1-maxnorm 의 서열(채널 공간합 c)을 만들 때, 한 번도 안 쏜 위치에 막전위 준비도
    clip(V/Vth, 0, 1) 을 더한다 (silent_only, gain=1). `- vmem` (run 키 ch_novmem) 은 gain=0.
  - 10-10 이후 ch_novmem ρ 2e-3 s2·s3 (138) 이 완주해 그 ρ 가 n=3 이 됐다.
    10-10 의 "같은 스파이크 정확도 이득은 경향만 (-0.067 ±0.042, p=0.11, n=6)" 이 바뀌는가.

수치 정의 (원장·논문과 같음, nofinal_vs_inter_261014.py 와 같다)
  val acc = 310 에폭 중 최고 val_acc (같은 값이면 첫 에폭), spikes = 그 에폭 s_count (원값, 이미지당)
  train loss / train acc / val loss = 그 에폭 값 (train loss 는 규제 항 포함 'loss').
  값은 두 서버 런 폴더의 train.log 에서 직접 다시 읽는다 (원장은 교차검증에만).
    138 = /media/hdd1/kyccj/EIP/paper/<tag>/ (config 는 _paper/<tag>/config_sweep.py)
    23  = sejong /srv2/kyccj/EIP/paper/<tag>/ (config 는 ~/PycharmProjects/TensorFlow-SNNs/_paper/<tag>/)
          ssh 로 읽기만 한다. 결과는 scratchpad 캐시.
  팔 정체는 config 의 run_paper override 블록으로 확인 (단일 요인 차이만 허용).
  배제: 미완주 (310 에폭 미만), seed>=900 (연기시험), S30/S1<0.19 (원장 판정), 팔 불일치.

기전 지표 (학습 모드, reg_detail.csv / reg_neuron_detail.csv; ablation_mech_261010 과 같은 정의)
  w 평균  = sc_rate_mean 을 뉴런 수로 가중한 18층 평균 (마지막 에폭)
  R       = 18층 sc_loss 합 = Σ_l ‖S_l∘w_l‖₂
  w 퍼짐  = sc_rate_std (층 안 뉴런 간 SD) 의 뉴런 가중 평균
  침묵 %  = dead_neuron_ratio (이미지 하나에서 한 번도 안 쏜 뉴런 비율) 뉴런 가중
  지니·상위 10% = 층 안 뉴런 스파이크 집중도, 층 스파이크 가중
npz (테스트 1만 장 1회 추론, 평가 모드): _npz, _npz_t, scratchpad/deep1005/npz. - vmem 은 3e-3 s1 하나뿐.

쓰는 법:
  ~/anaconda3/envs/venv_1/bin/python analysis/ablation_vmem_261014.py [--refetch]
산출: 이 스크립트 + analysis/ablation_vmem_261014.md (문서는 이 스크립트 출력으로 작성).
      수치 원본 JSON 은 scratchpad/vmem1014/ablation_vmem_261014.json (과제 지시로 analysis/ 에 다른 파일을 쓰지 않음).
"""
import argparse, base64, glob, gzip, io, json, math, os, re, subprocess, sys
import numpy as np
from scipy import stats

ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
SCR = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
INV = SCR + '/led1007b/inventory.json'
CACHE = SCR + '/vmem1014'
OUT_JSON = CACHE + '/ablation_vmem_261014.json'
EPOCHS = 310
PATTERNS = ['r19c10-base---s1[1-6]', 'r19c10-ours_ch-*', 'r19c10-ch_novmem-*',
            'r19c10-prop-*', 'r19c10-abl_novmem-*', 'r19c10-lw1_novmem-*']
STORE = {'138': '/media/hdd1/kyccj/EIP/paper', '23': '/srv2/kyccj/EIP/paper'}
CFG = {'138': ROOT + '/_paper', '23': '/home/kyccj/PycharmProjects/TensorFlow-SNNs/_paper'}
NAME = {'base': 'baseline', 'ours_ch': 'ours_inter_ch', 'ch_novmem': '- vmem',
        'prop': 'ours_intra_ch', 'abl_novmem': 'ours_intra_ch - vmem', 'lw1_novmem': 'ours_layer_w1 - vmem'}
SPIKE_RANGE = (200e3, 420e3)      # nofinal_vs_inter_261014 와 같은 ours_ch 범위
NBOOT = 20000

LAYERS = ['conv1_conv_n'] + [f'conv2_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')] + \
         [f'conv3_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')] + \
         [f'conv4_block{b}_{s}_n' for b in (1, 2) for s in ('conv1', 'out')] + ['fc1_n']
NEU = {}
for l in LAYERS:
    NEU[l] = 256 if l == 'fc1_n' else (32 * 32 * 128 if l.startswith(('conv1', 'conv2')) else
                                      (16 * 16 * 256 if l.startswith('conv3') else 8 * 8 * 512))
STAGES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1']

# ----------------------------------------------------------------------------- 수집기
# 두 서버에서 같은 코드로 돈다 (sejong 에는 ssh stdin 으로 보낸다). 읽기만 한다.
COLLECTOR = r'''
import glob, gzip, json, os, re, sys, base64
def collect(store, cfgroot, patterns):
    # 런 폴더는 두 곳에 나뉘어 있다: 저장소(store: host.txt, reg_*.csv, 새 런의 train.log)와
    # 코드 사본(cfgroot: config_sweep.py, 옛 런의 train.log 실파일 / 새 런은 저장소로 가는 링크).
    # train.log 는 둘 중 에폭 줄이 더 많은 쪽, csv 는 저장소 우선.
    out = {}
    tags = set()
    for pat in patterns:
        for root in (store, cfgroot):
            tags |= {os.path.basename(d) for d in glob.glob(os.path.join(root, pat))}
    for tag in sorted(tags):
        ds, dc = os.path.join(store, tag), os.path.join(cfgroot, tag)
        r = {'host': None, 'lines': [], 'conf': '', 'rd': None, 'rnd': None, 'log_src': None}
        for d in (ds, dc):
            try:
                r['host'] = open(os.path.join(d, 'host.txt')).read().strip()
                break
            except Exception:
                pass
        for d in (dc, ds):
            lg = os.path.join(d, 'train.log')
            if os.path.exists(lg):
                txt = open(lg, errors='ignore').read().replace('\r', '\n')
                lines = [ln for ln in txt.split('\n') if 'val_loss' in ln and 'loss: ' in ln]
                if len(lines) > len(r['lines']):
                    r['lines'], r['log_src'] = lines, os.path.realpath(lg)
        cf = os.path.join(dc, 'config_sweep.py')
        if os.path.exists(cf):
            t = open(cf, errors='ignore').read()
            i, j = t.find('# >>> run_paper overrides'), t.find('# <<< run_paper overrides')
            r['conf'] = t[i:j + 30] if 0 <= i < j else ''
        for fn, k in (('reg_detail.csv', 'rd'), ('reg_neuron_detail.csv', 'rnd')):
            for d in (ds, dc):
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    r[k] = open(p).read()
                    break
        out[tag] = r
    return out
'''


def fetch(server, refetch):
    os.makedirs(CACHE, exist_ok=True)
    cache = os.path.join(CACHE, f'raw_{server}.json.gz')
    if refetch or not os.path.exists(cache):
        if server == '138':
            ns = {}
            exec(COLLECTOR, ns)
            data = ns['collect'](STORE['138'], CFG['138'], PATTERNS)
            with gzip.open(cache, 'wt') as f:
                json.dump(data, f)
        else:
            prog = COLLECTOR + (
                "\nsys.stdout.write(base64.b64encode(gzip.compress(json.dumps("
                f"collect({STORE['23']!r}, {CFG['23']!r}, {PATTERNS!r})).encode())).decode())\n")
            r = subprocess.run(['ssh', '-p', '23456', '-o', 'BatchMode=yes', 'kyccj@161.122.23.23', 'python3', '-'],
                               input=prog.encode(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
            with open(cache, 'wb') as f:
                f.write(base64.b64decode(r.stdout))
    with gzip.open(cache, 'rt') as f:
        return json.load(f)


# ----------------------------------------------------------------------------- 파싱
EPOCH_KV = re.compile(r'([A-Za-z_][\w\-]*): ([-\d.]+(?:[eE][+-]?\d+)?|nan)')
TIME = re.compile(r'\] - (\d+)s ')
ASSIGN = re.compile(r"^\s*conf\.([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.M)
TAGRE = re.compile(r'^r19c10-(base|ours_ch|ch_novmem|prop|abl_novmem|lw1_novmem)-(.*)-s(\d+)$')


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


def parse_epochs(lines):
    rows = []
    for ln in lines:
        k = ln.find('loss: ')
        d = {}
        for key, v in EPOCH_KV.findall(ln[k:]):
            try:
                d[key] = float(v)
            except ValueError:
                pass
        m = TIME.search(ln)
        d['sec'] = float(m.group(1)) if m else float('nan')
        if all(x in d for x in ('loss', 'acc', 'val_loss', 'val_acc', 's_count')):
            rows.append(d)
    return rows


def parse_csv(txt):
    """reg_detail / reg_neuron_detail → {epoch: {layer: {col: val}}}"""
    if not txt:
        return None
    lines = txt.strip().split('\n')
    cols = lines[0].split(',')
    out = {}
    for ln in lines[1:]:
        v = ln.split(',')
        if len(v) != len(cols):
            continue
        e = int(v[0])
        out.setdefault(e, {})[v[1]] = {c: float(x) for c, x in zip(cols[2:], v[2:])}
    return out


def arm_of(conf):
    """override 블록 → 팔 이름. 한 요소만 다른 팔만 인정한다."""
    g = lambda k, d=None: conf.get(k, d)
    if not g('reg_spike_out', False):
        return 'base'
    common = (g('reg_spike_out_sc') is True and g('reg_spike_out_sc_maxnorm') is True
              and not g('reg_spike_out_sc_maxnorm_plain', False)
              and g('reg_spike_out_wta_rev') is True
              and g('reg_spike_loss_ratio') is True and g('reg_spike_final_step') is True
              and float(g('reg_spike_loss_ratio_start_ep', 0) or 0) == 0
              and float(g('reg_spike_loss_ratio_end_ep', 0) or 0) == 0
              and g('reg_spike_maxnorm_no_inv', False) is False
              and float(g('reg_spike_shape_beta', 1.0)) == 1.0
              and not g('reg_spike_out_bpsr', False) and not g('reg_spike_accum_loss', False))
    if not common:
        return 'other'
    vm = float(g('reg_spike_vmem_gain', 0.0)) == 1.0 and g('reg_spike_vmem_silent_only') is True
    nv = float(g('reg_spike_vmem_gain', 0.0)) == 0.0
    grp, w1 = g('reg_spike_maxnorm_group', 'none'), g('reg_spike_wta_rev_w1', False) is True
    if grp == 'channel' and not w1:
        return 'ours_ch' if vm else ('ch_novmem' if nv else 'other')
    if grp == 'within_channel' and not w1:
        return 'prop' if vm else ('abl_novmem' if nv else 'other')
    if grp == 'none' and w1:
        return 'ours_layer_w1' if vm else ('lw1_novmem' if nv else 'other')
    return 'other'


def last_epoch_metrics(rd, rnd):
    """학습 모드 층 집계 (마지막 에폭). ablation_mech_261010.log_metrics 와 같은 정의."""
    if not rd:
        return {}
    e = max(rd)
    row = rd[e]
    ls = [l for l in LAYERS if l in row]
    if len(ls) != len(LAYERS):
        return {}
    out = {'tr_ep': e}
    spk = {l: row[l]['spike_count'] for l in ls}
    w = {l: row[l]['sc_rate_mean'] for l in ls}
    scl = {l: row[l]['sc_loss'] for l in ls}
    tot = sum(spk.values())
    nN = sum(NEU[l] for l in ls)
    out['tr_spk'] = tot
    out['w_mean'] = sum(w[l] * NEU[l] for l in ls) / nN
    out['R'] = sum(scl.values())
    for s in STAGES:
        ss = [l for l in ls if l.startswith(s)]
        out[f'share_{s}'] = 100 * sum(spk[l] for l in ss) / tot
        out[f'w_{s}'] = sum(w[l] * NEU[l] for l in ss) / sum(NEU[l] for l in ss)
        out[f'Rshare_{s}'] = 100 * sum(scl[l] for l in ss) / out['R'] if out['R'] > 0 else float('nan')
    out['w_layer'] = {l: w[l] for l in ls}
    out['spk_layer'] = {l: spk[l] for l in ls}
    out['R_layer'] = {l: scl[l] for l in ls}
    if rnd and e in rnd:
        d = rnd[e]
        out['w_sd'] = sum(d[l]['sc_rate_std'] * NEU[l] for l in ls) / nN
        out['dead_pct'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in ls) / nN
        out['gini_sw'] = sum(d[l]['gini'] * spk[l] for l in ls) / tot
        out['top10_sw'] = 100 * sum(d[l]['top10_share'] * spk[l] for l in ls) / tot
        out['dead_layer'] = {l: 100 * d[l]['dead_neuron_ratio'] for l in ls}
        out['wsd_layer'] = {l: d[l]['sc_rate_std'] for l in ls}
        for s in STAGES:
            ss = [l for l in ls if l.startswith(s)]
            out[f'dead_{s}'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in ss) / sum(NEU[l] for l in ss)
            sp = sum(spk[l] for l in ss)
            out[f'gini_{s}'] = sum(d[l]['gini'] * spk[l] for l in ss) / sp if sp > 0 else float('nan')
        inner = [l for l in ls if '_conv1_n' in l and l != 'conv1_conv_n']
        outer = [l for l in ls if '_out_n' in l]
        out['dead_inner'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in inner) / sum(NEU[l] for l in inner)
        out['dead_outer'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in outer) / sum(NEU[l] for l in outer)
        out['share_inner'] = 100 * sum(spk[l] for l in inner) / tot
        out['w_inner'] = sum(w[l] * NEU[l] for l in inner) / sum(NEU[l] for l in inner)
        out['w_outer'] = sum(w[l] * NEU[l] for l in outer) / sum(NEU[l] for l in outer)
        # 층 묶음별 '스파이크당 벌점 압력' 상대 지표 π = (R_g/spk_g)/(R/spk). 아래 analyze 의 주석 참고.
        for nm, ss in (('inner', inner), ('outer', outer), ('stem', ['conv1_conv_n'])):
            Rg, sg = sum(scl[l] for l in ss), sum(spk[l] for l in ss)
            out[f'lnpi_{nm}'] = math.log((Rg / sg) / (out['R'] / tot)) if out['R'] > 0 and sg > 0 and Rg > 0 else float('nan')
    return out


def traj_metrics(rd, epochs):
    """선택 에폭의 학습 모드 w 평균 · R · 스파이크."""
    out = {}
    if not rd:
        return out
    for e in epochs:
        if e not in rd:
            continue
        row = rd[e]
        ls = [l for l in LAYERS if l in row]
        nN = sum(NEU[l] for l in ls)
        out[e] = {'w_mean': sum(row[l]['sc_rate_mean'] * NEU[l] for l in ls) / nN,
                  'R': sum(row[l]['sc_loss'] for l in ls),
                  'tr_spk': sum(row[l]['spike_count'] for l in ls)}
    return out


TRAJ_EP = [0, 4, 9, 49, 99, 149, 198, 199, 200, 249, 309]


def build_runs(refetch):
    raw = {'138': fetch('138', refetch), '23': fetch('23', refetch)}
    inv = {r['tag']: r for r in json.load(open(INV))['runs']}
    runs, dups = [], []
    tags = sorted(set(raw['138']) | set(raw['23']))
    for tag in tags:
        m = TAGRE.match(tag)
        if not m:
            continue
        cands = []
        for srv in ('138', '23'):
            if tag in raw[srv]:
                rr = raw[srv][tag]
                cands.append((srv, rr, parse_epochs(rr['lines'])))
        if len(cands) > 1:
            dups.append({'tag': tag, 'epochs': {s: len(e) for s, _, e in cands}})
        srv, rr, rows = max(cands, key=lambda c: (len(c[2]), c[0] == '138'))
        key, knob, seed = m.group(1), m.group(2), int(m.group(3))
        conf = {k: _lit(v) for k, v in ASSIGN.findall(rr['conf'])}
        arm = arm_of(conf) if conf else 'no_conf'
        rho = conf.get('reg_spike_loss_ratio_target') if key != 'base' else None
        rec = {'tag': tag, 'key': key, 'knob': '-' if key == 'base' else knob.replace('p', '.'), 'seed': seed,
               'server': srv, 'host': rr['host'], 'arm_conf': arm, 'rho': rho, 'epochs': len(rows)}
        if rows:
            b = max(rows, key=lambda x: x['val_acc'])
            rec.update(best_epoch=rows.index(b) + 1, train_loss=b['loss'], train_acc=b['acc'] * 100,
                       val_loss=b['val_loss'], val_acc=b['val_acc'] * 100, spikes=b['s_count'])
            secs = np.array([r['sec'] for r in rows[1:]], float)     # 첫 에폭은 그래프 컴파일 포함
            secs = secs[np.isfinite(secs)]
            rec['sec_med'] = float(np.median(secs)) if len(secs) else None
            rec['sec_iqr'] = [float(np.percentile(secs, 25)), float(np.percentile(secs, 75))] if len(secs) else None
            if 'adp_lambda' in rows[-1]:
                lam = np.array([r.get('adp_lambda', np.nan) for r in rows])
                Rr = np.array([r.get('reg_R', np.nan) for r in rows])
                Lt = np.array([r.get('task_loss_clean', np.nan) for r in rows])
                rec.update(lam_mean=float(np.nanmean(lam)), lam_pre=float(np.nanmean(lam[:200])),
                           lam_post=float(np.nanmean(lam[200:])), lam_end=float(lam[-1]),
                           regR_end=float(Rr[-1]), Ltask_end=float(Lt[-1]),
                           lam_traj={e: float(lam[e]) for e in TRAJ_EP if e < len(lam)},
                           regR_traj={e: float(Rr[e]) for e in TRAJ_EP if e < len(Rr)})
        rd, rnd = parse_csv(rr['rd']), parse_csv(rr['rnd'])
        rec['mech'] = last_epoch_metrics(rd, rnd)
        rec['traj'] = traj_metrics(rd, TRAJ_EP)
        iv = inv.get(tag, {})
        reasons = []
        if seed >= 900:
            reasons.append(f'연기시험({len(rows)}ep)')
        elif len(rows) < EPOCHS:
            reasons.append(f'미완주({len(rows)}ep)')
        elif len(rows) > EPOCHS:
            reasons.append(f'에폭 줄 과다({len(rows)})')
        if key != 'base' and (rho is None or abs(float(rec['knob']) - float(rho)) > 1e-12):
            reasons.append(f'ρ 불일치(dir={rec["knob"]}, conf={rho})')
        if 'S30/S1' in (iv.get('exclude_reason') or ''):
            reasons.append([x for x in iv['exclude_reason'].split('; ') if 'S30' in x][0])
        if arm != key:
            reasons.append(f'팔 불일치(dir={key}, conf={arm})')
        hs = (rr['host'] or '').split('\t')[0]
        if hs and {'canus': '138', 'sejong': '23'}.get(hs) not in (None, srv):
            reasons.append(f'host.txt 불일치({hs})')
        rec['exclude_reason'] = '; '.join(reasons)
        rec['use'] = not reasons
        if iv and rows:
            rec['inv_check'] = {
                'val_acc_diff': None if iv.get('val_acc') is None else round(rec['val_acc'] - iv['val_acc'], 4),
                'spikes_diff': None if iv.get('spikes') is None else round(rec['spikes'] - iv['spikes'], 1),
                'server_match': iv.get('server') == srv}
        runs.append(rec)
    return runs, dups



# ----------------------------------------------------------------------------- 통계 도구
def tcrit(df):
    return stats.t.ppf(0.975, df)


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    d = a.mean() - b.mean()
    t = d / se
    return {'diff': d, 'se': se, 'df': df, 't': t, 'p': 2 * stats.t.sf(abs(t), df),
            'lo': d - tcrit(df) * se, 'hi': d + tcrit(df) * se, 'na': len(a), 'nb': len(b)}


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


def fieller(d, b, vd, vb, cdb, df):
    t2 = tcrit(df) ** 2
    A = b * b - t2 * vb
    B = -2 * (d * b - t2 * cdb)
    C = d * d - t2 * vd
    disc = B * B - 4 * A * C
    if A <= 0:
        return None, None
    r = math.sqrt(max(disc, 0))
    return (-B - r) / (2 * A), (-B + r) / (2 * A)


def mde(sd, n1, n2, target=0.8):
    df = n1 + n2 - 2
    se = sd * math.sqrt(1 / n1 + 1 / n2)
    tc = tcrit(df)
    lo, hi = 0.0, 10 * sd
    for _ in range(100):
        mid = (lo + hi) / 2
        ncp = mid / se
        pw = stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp)
        lo, hi = (mid, hi) if pw < target else (lo, mid)
    return hi


def n_needed_curve(delta, sd, var_curve, df, target=0.8):
    """곡선 ANCOVA 에서 팔 n 을 늘릴 때 |delta| 를 검정력 target 으로 볼 n (곡선은 지금대로)."""
    k = stats.t.ppf(0.975, df) + stats.t.ppf(target, df)
    need = (abs(delta) / k) ** 2 - var_curve
    if need <= 0:
        return None
    return int(math.ceil(sd ** 2 / need))


def line_pred(x, y, x0):
    X = np.c_[np.ones(len(x)), x]
    f = ols(X, y, ['a', 'b'])
    v = np.array([1.0, x0])
    m = float(v @ f['beta'])
    sm = float(np.sqrt(v @ f['cov'] @ v))
    return m, sm, math.sqrt(sm ** 2 + f['resid_sd'] ** 2), f


def msd(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float('nan')), len(v)


A = lambda rs, k: np.array([r[k] for r in rs], float)
M = lambda rs, k: np.array([r['mech'][k] for r in rs], float)


# ----------------------------------------------------------------------------- 1. 같은 스파이크 정확도
def ancova_acc(ours_rng, arm, label='arm'):
    """nofinal_vs_inter_261014 와 같은 모형. x = ln s − ln(팔 평균 스파이크)."""
    data = ours_rng + arm
    s0 = A(arm, 'spikes').mean()
    x = np.log(A(data, 'spikes')) - math.log(s0)
    y = A(data, 'val_acc')
    g = np.array([1.0 if r in arm else 0.0 for r in data])
    sv = np.array([1.0 if r['server'] == '23' else 0.0 for r in data])
    one = np.ones_like(x)
    models = {'lin': (np.c_[one, x, g], ['a', 'b', 'd']),
              'lin_srv': (np.c_[one, x, g, sv], ['a', 'b', 'd', 'e']),
              'quad': (np.c_[one, x, x ** 2, g], ['a', 'b', 'c', 'd']),
              'quad_srv': (np.c_[one, x, x ** 2, g, sv], ['a', 'b', 'c', 'd', 'e'])}
    out = {'n_ours': len(ours_rng), 'n_arm': len(arm), 's0': s0}
    fits = {}
    for nm, (X, names) in models.items():
        f = ols(X, y, names)
        fits[nm] = f
        out[nm] = {'coef': f['coef'], 'df': f['df'], 'n': f['n'], 'resid_sd': f['resid_sd']}
    # 같은 정확도에서 스파이크 비 (선형+서버): θ = d/b, 비 = exp(-θ) − 1
    f = fits['lin_srv']
    ib, idd = f['names'].index('b'), f['names'].index('d')
    b, d = f['beta'][ib], f['beta'][idd]
    vb, vd, cdb = f['cov'][ib, ib], f['cov'][idd, idd], f['cov'][ib, idd]
    lo, hi = fieller(d, b, vd, vb, cdb, f['df'])
    out['same_acc'] = {'theta': d / b, 'pct': (math.exp(-d / b) - 1) * 100,
                       'fieller_lo_pct': None if lo is None else (math.exp(-hi) - 1) * 100,
                       'fieller_hi_pct': None if lo is None else (math.exp(-lo) - 1) * 100}
    out['_fits'] = fits
    out['_X'] = models
    out['_y'] = y
    out['_data'] = data
    return out


def boot_d(res, nboot=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    data = res['_data']
    X, names = res['_X']['lin_srv']
    y = res['_y']
    io = np.array([i for i, r in enumerate(data) if r['key'] == 'ours_ch'])
    ia = np.array([i for i, r in enumerate(data) if r['key'] != 'ours_ch'])
    ds = []
    for _ in range(nboot):
        idx = np.r_[rng.choice(io, len(io)), rng.choice(ia, len(ia))]
        Xb = X[idx]
        if np.linalg.matrix_rank(Xb) < Xb.shape[1]:
            continue
        ds.append(np.linalg.lstsq(Xb, y[idx], rcond=None)[0][names.index('d')])
    ds = np.array(ds)
    return {'n': len(ds), 'lo': float(np.percentile(ds, 2.5)), 'hi': float(np.percentile(ds, 97.5)),
            'p_le0': float((ds >= 0).mean())}


def strip(o):
    return {k: v for k, v in o.items() if not k.startswith('_')}


def analyze(runs):
    use = [r for r in runs if r['use']]
    G = lambda k: sorted([r for r in use if r['key'] == k], key=lambda r: (float(r['knob']) if r['knob'] != '-' else 0, r['seed']))
    base, ours, nv, prop, anv = G('base'), G('ours_ch'), G('ch_novmem'), G('prop'), G('abl_novmem')
    ours_rng = [r for r in ours if SPIKE_RANGE[0] <= r['spikes'] <= SPIKE_RANGE[1]]
    res = {'n': {'base': len(base), 'ours_ch': len(ours), 'ours_ch_200_420K': len(ours_rng), 'ch_novmem': len(nv),
                 'prop': len(prop), 'abl_novmem': len(anv)}}

    # 합동 시드 SD (ours_ch, n>=2 인 ρ)
    by = {}
    for r in ours:
        by.setdefault(r['knob'], []).append(r)
    num = sum((len(v) - 1) * A(v, 'val_acc').var(ddof=1) for v in by.values() if len(v) > 1)
    dfp = sum(len(v) - 1 for v in by.values() if len(v) > 1)
    sd_pool = math.sqrt(num / dfp)
    res['pooled_sd'] = {'acc': sd_pool, 'df': dfp, 'rhos': sorted([k for k, v in by.items() if len(v) > 1], key=float)}

    # --- 1. 같은 스파이크 정확도
    sec1 = {}
    main = ancova_acc(ours_rng, nv)
    main['boot'] = boot_d(main)
    sec1['main'] = main
    # 런별 곡선 대비 (ours 200~420K 직선)
    fx, fy = np.log(A(ours_rng, 'spikes')), A(ours_rng, 'val_acc')
    per = []
    for r in nv:
        m, sm, sp, f = line_pred(fx, fy, math.log(r['spikes']))
        per.append({'tag': r['tag'], 'server': r['server'], 'spikes': r['spikes'], 'val_acc': r['val_acc'],
                    'pred': m, 'delta': r['val_acc'] - m, 'se': sp, 'z': (r['val_acc'] - m) / sp,
                    'new_1014': r['tag'] in ('r19c10-ch_novmem-2e-3-s2', 'r19c10-ch_novmem-2e-3-s3')})
    sec1['per_run'] = per
    # 부분집합 (모두 같은 ours 곡선)
    NEW = ('r19c10-ch_novmem-2e-3-s2', 'r19c10-ch_novmem-2e-3-s3')
    subsets = {
        '10-10 집합 (새 2런 제외)': [r for r in nv if r['tag'] not in NEW],
        '새 2런만 (10-10 이후, 사전 예측 검증)': [r for r in nv if r['tag'] in NEW],
        'ρ 2e-3 만 (n=3)': [r for r in nv if r['knob'] == '2e-3'],
        '138 런만': [r for r in nv if r['server'] == '138'],
        '23 런만': [r for r in nv if r['server'] == '23'],
        '3e-3 s1 (가장 낮은 런) 제외': [r for r in nv if r['tag'] != 'r19c10-ch_novmem-3e-3-s1'],
        '2e-3 s3 (96.42) 제외': [r for r in nv if r['tag'] != 'r19c10-ch_novmem-2e-3-s3'],
    }
    sub = {}
    for nm, arm in subsets.items():
        rr = ancova_acc(ours_rng, arm)
        sub[nm] = {'n_arm': len(arm), 'tags': [r['tag'] for r in arm],
                   'lin': rr['lin']['coef']['d'], 'lin_srv': rr['lin_srv']['coef']['d'] if len({r['server'] for r in ours_rng + arm}) > 1 else None}
    sec1['subsets'] = sub
    # LOO
    loo = []
    for r in nv:
        rr = ancova_acc(ours_rng, [q for q in nv if q is not r])
        loo.append({'drop': r['tag'], 'd': rr['lin_srv']['coef']['d']['est'], 'p': rr['lin_srv']['coef']['d']['p']})
    sec1['loo'] = loo
    # 서버 × 팔 상호작용 (- vmem 효과가 서버마다 다른가). 같은 ours 곡선, 선형 + 서버 + 팔 + 팔×서버.
    data = ours_rng + nv
    x = np.log(A(data, 'spikes')) - math.log(main['s0'])
    g = np.array([1.0 if r['key'] == 'ch_novmem' else 0.0 for r in data])
    sv = np.array([1.0 if r['server'] == '23' else 0.0 for r in data])
    fi = ols(np.c_[np.ones_like(x), x, g, sv, g * sv], A(data, 'val_acc'), ['a', 'b', 'd', 'e', 'dxe'])
    sec1['server_x_arm'] = {'d_138': fi['coef']['d'], 'dxe': fi['coef']['dxe'], 'n_nv': {'138': int(sum(g * (1 - sv))), '23': int(sum(g * sv))}}
    # ours 범위 민감도
    rngs = {}
    for lo_, hi_ in ((250e3, 420e3), (200e3, 450e3), (0, 1e9)):
        o2 = [r for r in ours if lo_ <= r['spikes'] <= hi_]
        rr = ancova_acc(o2, nv)
        rngs[f'{int(lo_/1e3)}~{int(hi_/1e3) if hi_ < 1e8 else "전체"}K'] = {
            'n_ours': len(o2), 'lin_srv': rr['lin_srv']['coef']['d'], 'quad_srv': rr['quad_srv']['coef']['d']}
    sec1['ranges'] = rngs
    # 국소 창: novmem 2e-3 평균 스파이크 ±8%
    s2 = A([r for r in nv if r['knob'] == '2e-3'], 'spikes').mean()
    win = {}
    for w in (0.05, 0.08, 0.12):
        sel = [r for r in ours if s2 * (1 - w) <= r['spikes'] <= s2 * (1 + w)]
        nv2 = [r for r in nv if r['knob'] == '2e-3']
        e = {'n_ours': len(sel), 'tags': [r['tag'] for r in sel], 'lo': s2 * (1 - w), 'hi': s2 * (1 + w)}
        if len(sel) >= 2:
            wa = welch(A(nv2, 'val_acc'), A(sel, 'val_acc'))
            ws = welch(np.log(A(nv2, 'spikes')), np.log(A(sel, 'spikes')))
            e.update(acc=wa, ours_acc=msd(A(sel, 'val_acc')), spk_pct=(math.exp(ws['diff']) - 1) * 100)
        win[w] = e
    sec1['window_2e3'] = {'s0': s2, 'win': win}
    # 검정력
    f = main['_fits']['lin_srv']
    se_d = f['coef']['d']['se']
    var_curve = max(se_d ** 2 - f['resid_sd'] ** 2 / len(nv), 0.0)
    k = stats.t.ppf(0.975, f['df']) + stats.t.ppf(0.8, f['df'])
    sec1['power'] = {'se_d': se_d, 'curve_se': math.sqrt(var_curve), 'mde_now': k * se_d,
                     'mde_by_n': {m: k * math.sqrt(sd_pool ** 2 / m + var_curve) for m in (3, 4, 8, 12, 20)},
                     'floor': k * math.sqrt(var_curve),
                     'n_for_obs': n_needed_curve(f['coef']['d']['est'], sd_pool, var_curve, f['df'])}
    res['sec1'] = {k_: (strip(v) if isinstance(v, dict) and '_fits' in v else v) for k_, v in sec1.items()}
    res['sec1']['main'] = strip(main)

    # --- 1b. intra 교차 확인 (prop vs abl_novmem). 섞지 않는다.
    ia = {}
    for nm, (lo_, hi_) in {'전 구간': (0, 1e9), '겹치는 구간 (prop 최소 스파이크 이상)': (min(A(prop, 'spikes')), 1e9)}.items():
        arm = [r for r in anv if lo_ <= r['spikes'] <= hi_]
        if len(arm) < 1:
            continue
        data = prop + arm
        x = np.log(A(data, 'spikes'))
        g = np.array([1.0 if r in arm else 0.0 for r in data])
        f2 = ols(np.c_[np.ones_like(x), x, g], A(data, 'val_acc'), ['a', 'b', 'd'])
        ia[nm] = {'n_prop': len(prop), 'n_abl': len(arm), 'd': f2['coef']['d'], 'b': f2['coef']['b'],
                  'resid_sd': f2['resid_sd'], 'abl_spk_range': [min(A(arm, 'spikes')), max(A(arm, 'spikes'))],
                  'prop_spk_range': [min(A(prop, 'spikes')), max(A(prop, 'spikes'))]}
    p4, a4 = [r for r in prop if r['knob'] == '4e-3'], [r for r in anv if r['knob'] == '4e-3']
    ia['rho4e3'] = {'acc': welch(A(a4, 'val_acc'), A(p4, 'val_acc')),
                    'lnspk': welch(np.log(A(a4, 'spikes')), np.log(A(p4, 'spikes'))),
                    'prop': [msd(A(p4, 'val_acc')), msd(A(p4, 'spikes'))],
                    'abl': [msd(A(a4, 'val_acc')), msd(A(a4, 'spikes'))]}
    res['intra'] = ia

    # --- 2. baseline 대비
    def vs_base(group):
        wa = welch(A(group, 'val_acc'), A(base, 'val_acc'))
        ws = welch(np.log(A(group, 'spikes')), np.log(A(base, 'spikes')))
        return {'n': len(group), 'acc': wa, 'spk_pct': (math.exp(ws['diff']) - 1) * 100,
                'spk_lo': (math.exp(ws['lo']) - 1) * 100, 'spk_hi': (math.exp(ws['hi']) - 1) * 100}
    res['vs_base'] = {'base': {'acc': msd(A(base, 'val_acc')), 'spk': msd(A(base, 'spikes'))},
                      'novmem_2e3': vs_base([r for r in nv if r['knob'] == '2e-3']),
                      'ours_2e3': vs_base([r for r in ours if r['knob'] == '2e-3']),
                      'ours_3e3': vs_base([r for r in ours if r['knob'] == '3e-3'])}
    # baseline 대비 런별 (n=1 런들)
    bm, bs, _ = msd(A(base, 'val_acc'))
    res['vs_base']['novmem_runs'] = [{'tag': r['tag'], 'dacc': r['val_acc'] - bm,
                                      'z': (r['val_acc'] - bm) / (bs * math.sqrt(1 + 1 / len(base))),
                                      'spk_pct': (r['spikes'] / A(base, 'spikes').mean() - 1) * 100} for r in nv]

    # --- 3. 같은 ρ / 같은 λ 에서 스파이크
    sec3 = {}
    o3 = [r for r in ours if 1e-3 <= r['rho'] <= 5e-3]
    for nm, xk in (('같은 ρ', 'rho'), ('같은 λ (310 평균)', 'lam_mean'), ('같은 λ (0~199 평균)', 'lam_pre'),
                   ('같은 λ (마지막)', 'lam_end')):
        data = o3 + nv
        x = np.log(A(data, xk))
        g = np.array([1.0 if r['key'] == 'ch_novmem' else 0.0 for r in data])
        y = np.log(A(data, 'spikes'))
        f3 = ols(np.c_[np.ones_like(x), x, g], y, ['a', 'b', 'd'])
        f3i = ols(np.c_[np.ones_like(x), x, g, x * g], y, ['a', 'b', 'd', 'i'])
        e = {'n_ours': len(o3), 'n_nv': len(nv), 'd': f3['coef']['d'], 'b': f3['coef']['b'],
             'pct': (math.exp(f3['coef']['d']['est']) - 1) * 100,
             'pct_lo': (math.exp(f3['coef']['d']['lo']) - 1) * 100, 'pct_hi': (math.exp(f3['coef']['d']['hi']) - 1) * 100,
             'slope_diff_p': f3i['coef']['i']['p']}
        if xk == 'rho':
            # ρ 눈금: - vmem 의 ρ 는 ours 의 몇 배 ρ 와 같은 스파이크인가 = exp(-d/b)
            # ln s_nv(ρ) = a + b ln ρ + d = ours 의 ln s(ρ') → ρ'/ρ = exp(d/b).
            # 즉 '- vmem 의 ρ' 는 ours 의 exp(d/b) 배 ρ 와 같은 스파이크다 (>1 이면 vmem 을 끄는 것이 ρ 를 키운 셈).
            b_, d_ = f3['beta'][1], f3['beta'][2]
            vb, vd, cdb = f3['cov'][1, 1], f3['cov'][2, 2], f3['cov'][1, 2]
            lo, hi = fieller(d_, b_, vd, vb, cdb, f3['df'])
            e['rho_mult'] = math.exp(d_ / b_)
            e['rho_mult_ci'] = None if lo is None else sorted([math.exp(lo), math.exp(hi)])
        sec3[nm] = e
    # λ 자체 (같은 ρ)
    data = o3 + nv
    x = np.log(A(data, 'rho'))
    g = np.array([1.0 if r['key'] == 'ch_novmem' else 0.0 for r in data])
    for nm, yk in (('λ 평균', 'lam_mean'), ('λ 마지막', 'lam_end'), ('R 마지막 (train.log reg_R)', 'regR_end'),
                   ('과제 손실 마지막', 'Ltask_end')):
        f3 = ols(np.c_[np.ones_like(x), x, g], np.log(A(data, yk)), ['a', 'b', 'd'])
        sec3['같은 ρ 의 ' + nm] = {'d': f3['coef']['d'], 'pct': (math.exp(f3['coef']['d']['est']) - 1) * 100}
    # ρ 2e-3 직접 (n=3 대 n=3)
    o2, n2 = [r for r in ours if r['knob'] == '2e-3'], [r for r in nv if r['knob'] == '2e-3']
    direct = {}
    for k_ in ('spikes', 'val_acc', 'lam_mean', 'lam_end', 'regR_end', 'Ltask_end'):
        if k_ in ('spikes', 'lam_mean', 'lam_end', 'regR_end'):
            w_ = welch(np.log(A(n2, k_)), np.log(A(o2, k_)))
            direct[k_] = {'ours': msd(A(o2, k_)), 'nv': msd(A(n2, k_)), 'pct': (math.exp(w_['diff']) - 1) * 100,
                          'lo': (math.exp(w_['lo']) - 1) * 100, 'hi': (math.exp(w_['hi']) - 1) * 100, 'p': w_['p']}
        else:
            w_ = welch(A(n2, k_), A(o2, k_))
            direct[k_] = {'ours': msd(A(o2, k_)), 'nv': msd(A(n2, k_)), 'diff': w_['diff'], 'lo': w_['lo'],
                          'hi': w_['hi'], 'p': w_['p']}
    sec3['rho2e3_direct'] = direct
    # intra: 같은 ρ
    data = [r for r in prop if 1.5e-3 <= r['rho'] <= 5e-3] + anv
    x = np.log(A(data, 'rho'))
    g = np.array([1.0 if r['key'] == 'abl_novmem' else 0.0 for r in data])
    f3 = ols(np.c_[np.ones_like(x), x, g], np.log(A(data, 'spikes')), ['a', 'b', 'd'])
    b_, d_ = f3['beta'][1], f3['beta'][2]
    sec3['intra 같은 ρ'] = {'n_prop': len(data) - len(anv), 'n_abl': len(anv), 'd': f3['coef']['d'], 'b': f3['coef']['b'],
                           'pct': (math.exp(d_) - 1) * 100, 'rho_mult': math.exp(d_ / b_)}
    res['sec3'] = sec3

    # --- 4. 기전 (학습 모드, 마지막 에폭)
    sec4 = {}
    om = [r for r in ours_rng if r['mech'].get('w_sd') is not None]
    nm_ = [r for r in nv if r['mech'].get('w_sd') is not None]
    KEYS = (['R', 'w_mean', 'w_conv1', 'w_conv2', 'w_conv3', 'w_conv4', 'w_fc1', 'w_inner', 'w_outer', 'w_sd',
             'Rshare_conv1', 'Rshare_conv2', 'Rshare_conv3', 'Rshare_conv4',
             'share_conv1', 'share_conv2', 'share_conv3', 'share_conv4', 'share_inner',
             'dead_pct', 'dead_inner', 'dead_outer', 'gini_sw', 'top10_sw',
             'dead_conv1', 'dead_conv2', 'dead_conv3', 'dead_conv4',
             'gini_conv1', 'gini_conv2', 'gini_conv3', 'gini_conv4'])
    NEW = ('r19c10-ch_novmem-2e-3-s2', 'r19c10-ch_novmem-2e-3-s3')
    anc = {}
    for k_ in KEYS:
        data = om + nm_
        x = np.log(M(data, 'tr_spk'))
        y = M(data, k_)
        if k_ == 'R':
            y = np.log(y)
        g = np.array([1.0 if r['key'] == 'ch_novmem' else 0.0 for r in data])
        f4 = ols(np.c_[np.ones_like(x), x, g], y, ['a', 'b', 'd'])
        # 복제 검사: ours 직선 대비 잔차, 10-10 런 6개 vs 새 2런
        xo, yo = np.log(M(om, 'tr_spk')), (np.log(M(om, k_)) if k_ == 'R' else M(om, k_))
        fo = ols(np.c_[np.ones_like(xo), xo], yo, ['a', 'b'])
        resid = {}
        for r in nm_:
            yy = math.log(r['mech'][k_]) if k_ == 'R' else r['mech'][k_]
            v = np.array([1.0, math.log(r['mech']['tr_spk'])])
            resid[r['tag']] = yy - float(v @ fo['beta'])
        old = [v for t, v in resid.items() if t not in NEW]
        new = [v for t, v in resid.items() if t in NEW]
        anc[k_] = {'d': f4['coef']['d'], 'b': f4['coef']['b'], 'n_ours': len(om), 'n_nv': len(nm_),
                   'resid_old_mean': float(np.mean(old)), 'resid_new': new, 'resid_sd_ours': fo['resid_sd']}
    sec4['ancova'] = anc
    # ρ 2e-3 직접 (n=3 대 n=3), 기전 지표
    o2m = [r for r in ours if r['knob'] == '2e-3']
    n2m = [r for r in nv if r['knob'] == '2e-3']
    dm = {}
    for k_ in ('tr_spk', 'w_mean', 'R', 'w_sd', 'dead_pct', 'dead_inner', 'gini_sw', 'top10_sw', 'share_inner',
               'w_conv1', 'w_conv2', 'w_conv3', 'w_conv4'):
        a_, b_ = M(n2m, k_), M(o2m, k_)
        if k_ in ('tr_spk', 'R'):
            w_ = welch(np.log(a_), np.log(b_))
            dm[k_] = {'ours': msd(b_), 'nv': msd(a_), 'pct': (math.exp(w_['diff']) - 1) * 100, 'p': w_['p'],
                      'lo': (math.exp(w_['lo']) - 1) * 100, 'hi': (math.exp(w_['hi']) - 1) * 100}
        else:
            w_ = welch(a_, b_)
            dm[k_] = {'ours': msd(b_), 'nv': msd(a_), 'diff': w_['diff'], 'p': w_['p'], 'lo': w_['lo'], 'hi': w_['hi']}
    sec4['rho2e3_direct'] = dm
    # 층별 Δw (같은 스파이크) 와 그 층의 침묵 비율·w 수준
    lay = []
    for l in LAYERS:
        data = om + nm_
        x = np.log(M(data, 'tr_spk'))
        y = np.array([r['mech']['w_layer'][l] for r in data])
        g = np.array([1.0 if r['key'] == 'ch_novmem' else 0.0 for r in data])
        f4 = ols(np.c_[np.ones_like(x), x, g], y, ['a', 'b', 'd'])
        yR = np.log([r['mech']['R_layer'][l] for r in data])
        fR = ols(np.c_[np.ones_like(x), x, g], yR, ['a', 'b', 'd'])
        ys = np.log([r['mech']['spk_layer'][l] for r in data])
        fs = ols(np.c_[np.ones_like(x), x, g], ys, ['a', 'b', 'd'])
        lay.append({'layer': l, 'dw': f4['coef']['d'], 'dlnR': fR['coef']['d'], 'dlnspk': fs['coef']['d'],
                    'w_ours': float(np.mean([r['mech']['w_layer'][l] for r in om])),
                    'dead_ours': float(np.mean([r['mech']['dead_layer'][l] for r in om])),
                    'wsd_ours': float(np.mean([r['mech']['wsd_layer'][l] for r in om]))})
    sec4['layers'] = lay
    conv = [e for e in lay if e['layer'] != 'fc1_n']
    dw = [e['dw']['est'] for e in conv]
    sec4['layer_corr'] = {
        'dw_vs_dead': stats.spearmanr(dw, [e['dead_ours'] for e in conv])._asdict(),
        'dw_vs_w': stats.spearmanr(dw, [e['w_ours'] for e in conv])._asdict(),
        'dw_vs_dlnspk': stats.spearmanr(dw, [e['dlnspk']['est'] for e in conv])._asdict(),
        'n_layers': len(conv)}
    # 궤적 (ρ 2e-3, n=3 대 n=3)
    tr = {}
    for e in TRAJ_EP:
        a_ = [r for r in n2m if e in r['traj']]
        b_ = [r for r in o2m if e in r['traj']]
        if len(a_) >= 2 and len(b_) >= 2:
            ent = {}
            for k_ in ('w_mean', 'R', 'tr_spk'):
                va, vb = np.array([r['traj'][e][k_] for r in a_]), np.array([r['traj'][e][k_] for r in b_])
                if k_ == 'w_mean':
                    w_ = welch(va, vb)
                    ent[k_] = {'ours': float(vb.mean()), 'nv': float(va.mean()), 'diff': w_['diff'], 'se': w_['se'], 'p': w_['p']}
                else:
                    w_ = welch(np.log(va), np.log(vb))
                    ent[k_] = {'ours': float(vb.mean()), 'nv': float(va.mean()), 'pct': (math.exp(w_['diff']) - 1) * 100, 'p': w_['p']}
            la = np.array([r['lam_traj'][e] for r in a_ if e in r.get('lam_traj', {})])
            lb = np.array([r['lam_traj'][e] for r in b_ if e in r.get('lam_traj', {})])
            if len(la) >= 2 and len(lb) >= 2:
                w_ = welch(np.log(la), np.log(lb))
                ent['lam'] = {'ours': float(lb.mean()), 'nv': float(la.mean()), 'pct': (math.exp(w_['diff']) - 1) * 100, 'p': w_['p']}
            tr[e] = ent
    sec4['traj_2e3'] = tr
    # intra 기전 (prop vs abl_novmem, 같은 학습 모드 스파이크)
    pm = [r for r in prop if r['mech'].get('w_sd') is not None]
    am = [r for r in anv if r['mech'].get('w_sd') is not None]
    ia4 = {}
    for k_ in ('R', 'w_mean', 'w_sd', 'dead_pct', 'top10_sw', 'gini_sw', 'share_conv1'):
        data = pm + am
        x = np.log(M(data, 'tr_spk'))
        y = np.log(M(data, k_)) if k_ == 'R' else M(data, k_)
        g = np.array([1.0 if r['key'] == 'abl_novmem' else 0.0 for r in data])
        f4 = ols(np.c_[np.ones_like(x), x, g], y, ['a', 'b', 'd'])
        ia4[k_] = {'d': f4['coef']['d'], 'n_prop': len(pm), 'n_abl': len(am)}
    sec4['intra'] = ia4
    # 층별 스파이크당 벌점 압력의 '배분'.
    # wta_rev 역전파에서 쏜 뉴런 i (층 l) 의 기울기 = λ w_i² / R_l, loss-ratio 로 λ = ρ L / R.
    # w 를 모든 뉴런에서 c 배 하면 λ→λ/c, w²→c²w², R_l→cR_l 이라 기울기가 그대로다 (척도 불변).
    # 그래서 '평균 w 가 낮다' 자체는 loss-ratio 아래에서 압력을 바꾸지 않고, 층·채널 사이의 **상대** 배분만 남는다.
    # 층 l 의 평균 스파이크당 기울기 ≈ ρ L R_l / (R · S̄_l · spk_l) (S̄ = 쏜 뉴런의 평균 스파이크 수, 두 팔에서 같다고 근사).
    # ρL 은 층 공통이므로 배분 지표 π_l = (R_l/spk_l)/(R/spk) 를 쓴다. ln π 차 > 0 이면 - vmem 에서 그 층이 상대적으로 더 눌린다.
    lp = []
    for l in LAYERS:
        data = om + nm_
        x = np.log(M(data, 'tr_spk'))
        y = np.array([math.log((r['mech']['R_layer'][l] / r['mech']['spk_layer'][l]) / (r['mech']['R'] / r['mech']['tr_spk']))
                      for r in data])
        g = np.array([1.0 if r['key'] == 'ch_novmem' else 0.0 for r in data])
        f4 = ols(np.c_[np.ones_like(x), x, g], y, ['a', 'b', 'd'])
        lp.append({'layer': l, 'dlnpi': f4['coef']['d'],
                   'dead_ours': float(np.mean([r['mech']['dead_layer'][l] for r in om]))})
    sec4['layer_pressure'] = lp
    conv = [e for e in lp if e['layer'] != 'fc1_n']
    sec4['layer_pressure_corr'] = {
        'dlnpi_vs_dead': stats.spearmanr([e['dlnpi']['est'] for e in conv], [e['dead_ours'] for e in conv])._asdict(),
        'dlnpi_vs_dw': stats.spearmanr([e['dlnpi']['est'] for e in conv], [e['dw']['est'] for e in sec4['layers'] if e['layer'] != 'fc1_n'])._asdict(),
        'n_layers': len(conv)}
    grp = {}
    for k_ in ('lnpi_inner', 'lnpi_outer', 'lnpi_stem'):
        data = om + nm_
        x = np.log(M(data, 'tr_spk'))
        g = np.array([1.0 if r['key'] == 'ch_novmem' else 0.0 for r in data])
        f4 = ols(np.c_[np.ones_like(x), x, g], M(data, k_), ['a', 'b', 'd'])
        xo, yo = np.log(M(om, 'tr_spk')), M(om, k_)
        fo = ols(np.c_[np.ones_like(xo), xo], yo, ['a', 'b'])
        rs = {r['tag']: r['mech'][k_] - float(np.array([1.0, math.log(r['mech']['tr_spk'])]) @ fo['beta']) for r in nm_}
        grp[k_] = {'d': f4['coef']['d'], 'ours_mean': float(np.mean(M(om, k_))), 'resid': rs}
    # 같은 학습 모드 스파이크에서 '안쪽 − 출력' 압력 차
    data = om + nm_
    x = np.log(M(data, 'tr_spk'))
    g = np.array([1.0 if r['key'] == 'ch_novmem' else 0.0 for r in data])
    f4 = ols(np.c_[np.ones_like(x), x, g], M(data, 'lnpi_inner') - M(data, 'lnpi_outer'), ['a', 'b', 'd'])
    grp['inner_minus_outer'] = {'d': f4['coef']['d']}
    # π 는 R_l 이 L2 노름이라 그 층 스파이크가 적을수록 기계적으로 커진다. 안쪽 층 스파이크 몫을
    # 공변량으로 넣어도 팔 차이가 남는지 (= 스파이크 배분이 아니라 w 배분에서 오는지) 확인.
    for k_, ctrl in (('lnpi_inner', 'share_inner'), ('lnpi_outer', 'share_inner'), ('lnpi_stem', 'share_conv1')):
        f4 = ols(np.c_[np.ones_like(x), x, M(data, ctrl), g], M(data, k_), ['a', 'b', 'c', 'd'])
        grp[k_ + '|ctrl_' + ctrl] = {'d': f4['coef']['d'], 'c': f4['coef']['c']}
    sec4['pressure_groups'] = grp
    res['sec4'] = sec4
    return res, base, ours, nv, prop, anv, ours_rng


# ----------------------------------------------------------------------------- 5. npz (테스트셋, 평가 모드)
NPZ_DIRS = [ROOT + '/_npz', ROOT + '/_npz_t', SCR + '/deep1005/npz']


def npz_metrics(path):
    z = np.load(path)
    ns, T = float(z['n_samples']), int(z['T'])
    tot = sum(z[l].sum() for l in LAYERS) / ns
    N = sum(z[l].size for l in LAYERS)
    silent = sum((z[l] == 0).sum() for l in LAYERS)
    fired = sum(z[l + '__fired'].sum() for l in LAYERS)
    out = {'npz_spk': float(tot), 'silent_pct': 100 * silent / N, 'active_img_pct': 100 * fired / (ns * N),
           'acc': float(z['acc'])}
    inner = [l for l in LAYERS if '_conv1_n' in l and l != 'conv1_conv_n']
    outer = [l for l in LAYERS if '_out_n' in l]
    out['silent_inner'] = float(sum((z[l] == 0).sum() for l in inner))
    out['silent_inner_c2'] = float(sum((z[l] == 0).sum() for l in inner if l.startswith('conv2')))
    out['silent_outer'] = float(sum((z[l] == 0).sum() for l in outer))
    out['silent_stem'] = float((z['conv1_conv_n'] == 0).sum())
    out['spk_inner'] = float(sum(z[l].sum() for l in inner) / ns)
    out['spk_outer'] = float(sum(z[l].sum() for l in outer) / ns)
    # 저발화 뉴런 (발화율 0 초과 1% 이하) 비율
    r_all = np.concatenate([(z[l] / (ns * T)).ravel() for l in LAYERS])
    out['low_pct'] = 100 * float(((r_all > 0) & (r_all <= 0.01)).mean())
    out['low5_pct'] = 100 * float(((r_all > 0) & (r_all <= 0.05)).mean())
    # 채널 단위
    dead, ch_tot, gsum, tsum, ssum = 0, 0, 0.0, 0.0, 0.0
    for l in LAYERS:
        x = z[l]
        c = x.sum((0, 1)) if x.ndim == 3 else x
        dead += int((c == 0).sum())
        ch_tot += c.size
        cs = np.sort(c)
        n = len(cs)
        sl = cs.sum()
        if sl > 0:
            gi = float((2 * np.arange(1, n + 1) - n - 1) @ cs / (n * sl))
            k = max(1, int(round(0.1 * n)))
            t10 = float(cs[-k:].sum() / sl)
            gsum += gi * sl
            tsum += t10 * sl
            ssum += sl
    out['dead_ch'] = dead
    out['ch_gini'] = gsum / ssum
    out['ch_top10'] = 100 * tsum / ssum
    return out


def npz_section(runs):
    import glob as _g
    by_tag = {}
    for d in NPZ_DIRS:
        for p in sorted(_g.glob(d + '/r19c10-*.npz')):
            t = os.path.basename(p)[:-4]
            by_tag.setdefault(t, p)          # 앞 디렉토리 우선 (_npz → _npz_t → deep1005)
    use = {r['tag']: r for r in runs if r['use']}
    sel = {t: p for t, p in by_tag.items() if t in use and use[t]['key'] in ('ours_ch', 'ch_novmem', 'base')}
    met = {}
    for t, p in sel.items():
        met[t] = npz_metrics(p)
        met[t]['path'] = p
        met[t]['key'] = use[t]['key']
        met[t]['ledger_spikes'] = use[t]['spikes']
        met[t]['val_acc'] = use[t]['val_acc']
        met[t]['server'] = use[t]['server']
    ours = [t for t in met if met[t]['key'] == 'ours_ch']
    nvs = [t for t in met if met[t]['key'] == 'ch_novmem']
    out = {'runs': met, 'n_ours': len(ours), 'novmem': nvs}
    keys = ['silent_pct', 'active_img_pct', 'low_pct', 'low5_pct', 'dead_ch', 'ch_gini', 'ch_top10',
            'silent_inner', 'silent_inner_c2', 'silent_outer', 'silent_stem', 'spk_inner', 'spk_outer']
    curve = {}
    o_in = [t for t in ours if 200e3 <= met[t]['ledger_spikes'] <= 420e3]
    for k_ in keys:
        x = np.log([met[t]['npz_spk'] for t in o_in])
        y = np.array([met[t][k_] for t in o_in], float)
        ent = {'n_ours': len(o_in)}
        for t in nvs:
            m, sm, sp, f = line_pred(x, y, math.log(met[t]['npz_spk']))
            z_ = (met[t][k_] - m) / sp
            ent[t] = {'obs': met[t][k_], 'pred': m, 'se': sp, 'z': z_, 'p': float(2 * stats.t.sf(abs(z_), f['df'])),
                      'df': f['df'], 'slope': float(f['beta'][1]), 'slope_p': f['coef']['b']['p'], 'resid_sd': f['resid_sd']}
        curve[k_] = ent
    out['curve'] = curve
    out['ours_in_tags'] = o_in
    # 같은 ρ 안 시드 산포 (스파이크 보정): k ~ ρ 고정효과 + b·ln(npz 스파이크), n>=2 인 ρ 만.
    # 10-10 보충 문서는 ρ 3e-3 세 시드의 SD (우연히 가장 작음) 로 z 를 냈다 → 여기서 합동 SD 로 다시 잰다.
    grp = {}
    for t in ours:
        grp.setdefault(t.split('-', 2)[2].rsplit('-s', 1)[0], []).append(t)
    grp = {k: v for k, v in grp.items() if len(v) >= 2}
    ks = sorted(grp)
    rows = [(k, t) for k in ks for t in grp[k]]
    wr = {'groups': {k: grp[k] for k in ks}}
    nv = nvs[0] if nvs else None
    for k_ in keys:
        X = np.array([[1.0 if k == kk else 0.0 for kk in ks] + [math.log(met[t]['npz_spk'])] for k, t in rows])
        y = np.array([met[t][k_] for _, t in rows], float)
        f = ols(X, y, ks + ['b'])
        ent = {'resid_sd': f['resid_sd'], 'df': f['df'], 'slope': float(f['beta'][-1]),
               'raw_sd_by_rho': {k: float(np.std([met[t][k_] for t in grp[k]], ddof=1)) for k in ks}}
        if nv and '3e-3' in ks:
            v = np.zeros(len(ks) + 1)
            v[ks.index('3e-3')] = 1.0
            v[-1] = math.log(met[nv]['npz_spk'])
            pred = float(v @ f['beta'])
            se = math.sqrt(f['resid_sd'] ** 2 + float(v @ f['cov'] @ v))
            z_ = (met[nv][k_] - pred) / se
            ent.update(obs=met[nv][k_], pred_same_rho=pred, se=se, z=z_, p=float(2 * stats.t.sf(abs(z_), f['df'])))
        wr[k_] = ent
    out['within_rho'] = wr
    return out


# ----------------------------------------------------------------------------- 6. 학습 시간
def timing(runs):
    use = [r for r in runs if r['use'] and r.get('sec_med')]
    out = {}
    for key in ('base', 'ours_ch', 'ch_novmem', 'prop', 'abl_novmem'):
        for srv in ('138', '23'):
            v = [r['sec_med'] for r in use if r['key'] == key and r['server'] == srv]
            if v:
                out[f'{key}|{srv}'] = {'n': len(v), 'median': float(np.median(v)), 'min': float(min(v)),
                                       'max': float(max(v)), 'runs': {r['tag']: r['sec_med'] for r in use
                                                                       if r['key'] == key and r['server'] == srv}}
    for srv in ('138', '23'):
        a = [r['sec_med'] for r in use if r['key'] == 'ch_novmem' and r['server'] == srv]
        b = [r['sec_med'] for r in use if r['key'] == 'ours_ch' and r['server'] == srv]
        if len(a) >= 2 and len(b) >= 2:
            w = welch(np.log(a), np.log(b))
            out[f'nv_vs_ours|{srv}'] = {'pct': (math.exp(w['diff']) - 1) * 100, 'lo': (math.exp(w['lo']) - 1) * 100,
                                        'hi': (math.exp(w['hi']) - 1) * 100, 'p': w['p'], 'n': [len(a), len(b)]}
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refetch', action='store_true')
    ap.add_argument('--collect-only', action='store_true')
    a = ap.parse_args()
    runs, dups = build_runs(a.refetch)
    if a.collect_only:
        for r in runs:
            print(r['tag'], r['server'], r['epochs'], r['arm_conf'], r.get('val_acc'), r.get('spikes'),
                  r.get('sec_med'), bool(r['mech']), r['exclude_reason'], r.get('inv_check'))
        print('dups', dups)
        return
    res, base, ours, nv, prop, anv, ours_rng = analyze(runs)
    res['npz'] = npz_section(runs)
    res['timing'] = timing(runs)
    res['dups'] = dups
    keep = ('tag', 'key', 'knob', 'seed', 'server', 'host', 'epochs', 'best_epoch', 'train_loss', 'train_acc',
            'val_loss', 'val_acc', 'spikes', 'sec_med', 'lam_mean', 'lam_end', 'regR_end', 'Ltask_end',
            'exclude_reason', 'use', 'inv_check')
    res['runs'] = [dict({k: r.get(k) for k in keep},
                        mech={k: v for k, v in r['mech'].items() if not isinstance(v, dict)}) for r in runs]
    bad = [r['tag'] for r in runs if r['use'] and r.get('inv_check') and
           (abs(r['inv_check']['val_acc_diff'] or 0) > 1e-6 or abs(r['inv_check']['spikes_diff'] or 0) > 0.5
            or not r['inv_check']['server_match'])]
    res['inv_mismatch'] = bad
    os.makedirs(CACHE, exist_ok=True)
    json.dump(jsonable(res), open(OUT_JSON, 'w'), indent=1, ensure_ascii=False)
    print('wrote', OUT_JSON, 'inv mismatch', bad)


if __name__ == '__main__':
    main()
