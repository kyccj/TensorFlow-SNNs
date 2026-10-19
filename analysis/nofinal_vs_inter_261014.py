#!/usr/bin/env python
"""R19 CIFAR-10 (ResNet-19, T=4): '- final_step (inter)' vs ours_inter_ch — 26-10-14.

질문 (사용자 10-14): "비슷한 성능에서도 스파이크가 적고 스파이크가 적은데 정확도가 높은데?"
→ ch_nofinal (ours_inter_ch 에서 final_step 만 끈 팔) ρ 2.2e-3 (n=3) 이 ours_inter_ch
  (run 키 ours_ch) 를 정확도-스파이크 평면에서 지배하는가.

수치 정의 (원장·논문과 같음)
  val acc  = 310 에폭 중 최고 val_acc (같은 값이면 첫 에폭)
  spikes   = 그 에폭의 s_count (원값, 이미지당)
  train loss / train acc / val loss = 그 에폭 값. train loss 는 규제 항 포함 'loss'.
값은 train.log 에서 직접 다시 읽는다 (원장 inventory.json 은 교차검증에만 쓴다).
  138 = 로컬 _paper/<tag>/train.log (HDD 심볼릭 링크)
  23  = sejong /srv2/kyccj/EIP/paper/<tag>/train.log 에서 ssh 로 에폭 줄만 추출 (읽기 전용)
팔 정체는 config_sweep.py 의 run_paper override 블록으로 확인한다.
배제: 미완주(310 에폭 미만), S30/S1<0.19 (원장 판정을 그대로 씀 — 해당 런은 전부 140K 이하라
      이 분석의 어느 창·범위에도 들지 않는다).

쓰는 법:
  ~/anaconda3/envs/venv_1/bin/python analysis/nofinal_vs_inter_261014.py \
      [--inv <inventory.json>] [--cache23 <path.gz>] [--refetch]
산출: analysis/nofinal_vs_inter_261014.json, analysis/nofinal_vs_inter_261014.md
"""
import argparse, glob, gzip, json, math, os, re, subprocess, sys
import numpy as np
from scipy import stats

ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
SCR = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
DEF_INV = SCR + '/led1007b/inventory.json'
DEF_CACHE = SCR + '/nf1014/ep23.txt.gz'
OUT_JSON = ROOT + '/analysis/nofinal_vs_inter_261014.json'
OUT_MD = ROOT + '/analysis/nofinal_vs_inter_261014.md'
EPOCHS = 310
PATTERNS = ['r19c10-base---s1[1-6]', 'r19c10-ours_ch-[0-9]*', 'r19c10-ch_nofinal-2p2e-3-s[0-9]*']
REMOTE = ('cd /srv2/kyccj/EIP/paper && for d in ' + ' '.join(PATTERNS) + '; do '
          '[ -f "$d/train.log" ] || continue; echo "@@RUN $d"; '
          'tr "\\r" "\\n" < "$d/train.log" | grep "val_loss" | grep -o "loss: .*"; '
          'echo "@@CONF $d"; '
          'sed -n "/# >>> run_paper overrides/,/# <<< run_paper overrides/p" '
          '~/PycharmProjects/TensorFlow-SNNs/_paper/$d/config_sweep.py; done')
EPOCH_KV = re.compile(r'([A-Za-z_][\w\-]*): ([\d.]+(?:[eE][+-]?\d+)?)')
ASSIGN = re.compile(r"^\s*conf\.([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.M)
NAME = re.compile(r'^r19c10-(base|ours_ch|ch_nofinal)-(.*)-s(\d+)$')
SPIKE_RANGE = (200e3, 420e3)          # ANCOVA 에 쓰는 ours_ch 범위 (과제 지정)
WINDOWS = (0.05, 0.08, 0.12)
NBOOT = 20000


# ----------------------------------------------------------------------------- 읽기
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
        if all(x in d for x in ('loss', 'acc', 'val_loss', 'val_acc', 's_count')):
            rows.append(d)
    return rows


def load_local():
    out = {}
    for pat in PATTERNS:
        for d in sorted(glob.glob(os.path.join(ROOT, '_paper', pat))):
            tag = os.path.basename(d)
            lg, cf = os.path.join(d, 'train.log'), os.path.join(d, 'config_sweep.py')
            if not os.path.exists(lg):
                continue
            txt = open(lg, errors='ignore').read().replace('\r', '\n')
            rows = parse_epoch_lines(l for l in txt.split('\n') if 'val_loss' in l)
            conf = parse_conf(open(cf, errors='ignore').read()) if os.path.exists(cf) else {}
            out[tag] = {'rows': rows, 'conf': conf, 'src': 'local'}
    return out


def load_sejong(cache, refetch):
    if refetch or not os.path.exists(cache):
        r = subprocess.run(['ssh', '-n', '-p', '23456', '-o', 'BatchMode=yes', 'kyccj@161.122.23.23', REMOTE],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with gzip.open(cache, 'wb') as f:
            f.write(r.stdout)
    txt = gzip.open(cache, 'rt', errors='ignore').read()
    out, tag, mode, buf = {}, None, None, {}
    for ln in txt.split('\n'):
        if ln.startswith('@@RUN '):
            tag, mode = ln[6:].strip(), 'run'
            buf[tag] = {'lines': [], 'conf': ''}
        elif ln.startswith('@@CONF '):
            tag, mode = ln[7:].strip(), 'conf'
        elif tag:
            if mode == 'run':
                buf[tag]['lines'].append(ln)
            else:
                buf[tag]['conf'] += ln + '\n'
    for tag, b in buf.items():
        out[tag] = {'rows': parse_epoch_lines(b['lines']),
                    'conf': parse_conf(b['conf']), 'src': 'sejong'}
    return out


def arm_check(conf):
    """override 블록 → 'base' / 'ours_ch' / 'ch_nofinal' / 기타. 단일 요인 차이만 허용."""
    if not conf.get('reg_spike_out', False):
        return 'base'
    common = (conf.get('reg_spike_out_sc') is True and conf.get('reg_spike_out_sc_maxnorm') is True
              and conf.get('reg_spike_maxnorm_group') == 'channel'
              and conf.get('reg_spike_loss_ratio') is True
              and float(conf.get('reg_spike_loss_ratio_start_ep', 0) or 0) == 0
              and float(conf.get('reg_spike_loss_ratio_end_ep', 0) or 0) == 0
              and conf.get('reg_spike_maxnorm_no_inv', False) is False
              and conf.get('reg_spike_wta_rev_w1', False) is False
              and float(conf.get('reg_spike_vmem_gain', 1.0)) == 1.0
              and float(conf.get('reg_spike_shape_beta', 1.0)) == 1.0
              and not conf.get('reg_spike_out_bpsr', False))
    if not common:
        return 'other'
    return 'ours_ch' if conf.get('reg_spike_final_step') is True else 'ch_nofinal'


def build_runs(inv_path, cache, refetch):
    raw = {}
    raw.update(load_local())
    sj = load_sejong(cache, refetch)
    dup = sorted(set(raw) & set(sj))
    raw.update({k: v for k, v in sj.items() if k not in raw})
    inv = {r['tag']: r for r in json.load(open(inv_path))['runs']} if os.path.exists(inv_path) else {}
    runs = []
    for tag, r in sorted(raw.items()):
        m = NAME.match(tag)
        key, knob, seed = m.group(1), m.group(2).replace('p', '.'), int(m.group(3))
        conf = r['conf']
        rms = str(conf.get('root_model_save', ''))
        # 서버 = 읽은 곳 (로컬 _paper = 138, sejong = 23). override 의 저장 경로가 있으면 교차검증.
        server = '138' if r['src'] == 'local' else '23'
        srv_rms = '138' if '/media/hdd1' in rms else ('23' if '/srv2' in rms else None)
        arm = arm_check(conf) if conf else 'no_conf'
        rho = conf.get('reg_spike_loss_ratio_target', None) if arm != 'base' else None
        if key == 'base':
            knob = '-'
        rows = r['rows']
        rec = {'tag': tag, 'key': key, 'knob': knob, 'seed': seed, 'server': server, 'arm_conf': arm,
               'rho': rho, 'epochs': len(rows), 'src': r['src']}
        if rows:
            b = max(rows, key=lambda x: x['val_acc'])          # 첫 최대 (원장과 같음)
            rec.update(best_epoch=rows.index(b) + 1, train_loss=b['loss'], train_acc=b['acc'] * 100,
                       val_loss=b['val_loss'], val_acc=b['val_acc'] * 100, spikes=b['s_count'],
                       last_val_acc=rows[-1]['val_acc'] * 100)
        iv = inv.get(tag, {})
        reasons = []
        if len(rows) < EPOCHS:
            reasons.append(f'연기시험({len(rows)}ep)' if seed >= 900 else f'미완주({len(rows)}ep)')
        elif len(rows) > EPOCHS:
            reasons.append(f'에폭 줄 과다({len(rows)})')
        if srv_rms and srv_rms != server:
            reasons.append(f'서버 불일치(읽은 곳 {server}, 저장 경로 {srv_rms})')
        if key != 'base' and (rho is None or abs(float(knob) - float(rho)) > 1e-12):
            reasons.append(f'ρ 불일치(dir={knob}, conf={rho})')
        if 'S30/S1' in (iv.get('exclude_reason') or ''):
            reasons.append([x for x in iv['exclude_reason'].split('; ') if 'S30' in x][0])
        if arm not in ('base', 'ours_ch', 'ch_nofinal') or key not in ('base', 'ours_ch', 'ch_nofinal') or \
                ({'base': 'base', 'ours_ch': 'ours_ch', 'ch_nofinal': 'ch_nofinal'}[key] != arm):
            reasons.append(f'팔 불일치(dir={key}, conf={arm})')
        rec['exclude_reason'] = '; '.join(reasons)
        rec['use'] = not reasons
        # 원장 교차검증
        if iv and rows:
            rec['inv_check'] = {
                'val_acc_diff': None if iv.get('val_acc') is None else round(rec['val_acc'] - iv['val_acc'], 4),
                'spikes_diff': None if iv.get('spikes') is None else round(rec['spikes'] - iv['spikes'], 1),
                'server_match': iv.get('server') == server}
        runs.append(rec)
    return runs, dup, inv


# ----------------------------------------------------------------------------- 통계
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
        co[nm] = {'est': beta[i], 'se': se[i], 'lo': beta[i] - tcrit(df) * se[i],
                  'hi': beta[i] + tcrit(df) * se[i], 't': t, 'p': 2 * stats.t.sf(abs(t), df)}
    return {'coef': co, 'df': df, 'n': n, 'resid_sd': math.sqrt(s2), 'beta': beta, 'cov': cov, 'names': names}


def fieller(d, b, vd, vb, cdb, df):
    """θ = d/b 의 Fieller 95% 구간. b 가 유의하지 않으면 유계가 아니다."""
    t2 = tcrit(df) ** 2
    A = b * b - t2 * vb
    B = -2 * (d * b - t2 * cdb)
    C = d * d - t2 * vd
    disc = B * B - 4 * A * C
    if A <= 0:
        return None, None, 'unbounded'
    r = math.sqrt(max(disc, 0))
    return (-B - r) / (2 * A), (-B + r) / (2 * A), 'bounded'


def power_2s(delta, sd, n):
    df = 2 * n - 2
    ncp = abs(delta) / (sd * math.sqrt(2 / n))
    tc = tcrit(df)
    return stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp)


def n_needed(delta, sd, target=0.8):
    if delta == 0:
        return None
    approx = 2 * (stats.norm.ppf(0.975) + stats.norm.ppf(target)) ** 2 * (sd / delta) ** 2
    if approx > 2000:
        return int(math.ceil(approx))
    for n in range(2, 5000):
        if power_2s(delta, sd, n) >= target:
            return n
    return None


def mde(sd, n1, n2, target=0.8):
    """두 표본 t (자유도 n1+n2-2) 에서 검정력 target 인 최소 차이."""
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


# ----------------------------------------------------------------------------- 분석
def analyze(runs):
    use = [r for r in runs if r['use']]
    base = [r for r in use if r['key'] == 'base']
    nof = [r for r in use if r['key'] == 'ch_nofinal' and r['knob'] == '2.2e-3']
    ours = [r for r in use if r['key'] == 'ours_ch']
    A = lambda rs, k: np.array([r[k] for r in rs], float)
    res = {'n': {'base': len(base), 'ch_nofinal': len(nof), 'ours_ch_all': len(ours)}}
    s0 = A(nof, 'spikes').mean()
    a0 = A(nof, 'val_acc').mean()
    res['nofinal_mean'] = {'spikes': s0, 'val_acc': a0, 'val_acc_sd': A(nof, 'val_acc').std(ddof=1),
                           'spikes_sd': A(nof, 'spikes').std(ddof=1), 'n': len(nof)}

    # 1. 국소 창
    win = []
    for w in WINDOWS:
        sel = [r for r in ours if s0 * (1 - w) <= r['spikes'] <= s0 * (1 + w)]
        e = {'window': w, 'lo': s0 * (1 - w), 'hi': s0 * (1 + w), 'n_ours': len(sel),
             'ours_tags': [r['tag'] for r in sel], 'ours_servers': [r['server'] for r in sel]}
        if len(sel) >= 2:
            wa = welch(A(nof, 'val_acc'), A(sel, 'val_acc'))
            ws = welch(np.log(A(nof, 'spikes')), np.log(A(sel, 'spikes')))
            e.update(acc=wa, ours_acc_mean=A(sel, 'val_acc').mean(), ours_acc_sd=A(sel, 'val_acc').std(ddof=1),
                     ours_spk_mean=A(sel, 'spikes').mean(),
                     spk_pct=(math.exp(ws['diff']) - 1) * 100, spk_pct_lo=(math.exp(ws['lo']) - 1) * 100,
                     spk_pct_hi=(math.exp(ws['hi']) - 1) * 100, spk_p=ws['p'])
        win.append(e)
    res['window'] = win

    # 2. ANCOVA
    rng_ours = [r for r in ours if SPIKE_RANGE[0] <= r['spikes'] <= SPIKE_RANGE[1]]
    data = rng_ours + nof
    L = np.log(A(data, 'spikes'))
    Lc0 = math.log(s0)                     # 중심 = nofinal 평균 스파이크 → 절편이 그 점의 ours 곡선 값
    x = L - Lc0
    acc = A(data, 'val_acc')
    nf = np.array([1.0 if r['key'] == 'ch_nofinal' else 0.0 for r in data])
    sv = np.array([1.0 if r['server'] == '23' else 0.0 for r in data])
    one = np.ones_like(x)
    models = {
        'lin': (np.c_[one, x, nf], ['a', 'b', 'd']),
        'lin_srv': (np.c_[one, x, nf, sv], ['a', 'b', 'd', 'e']),
        'quad': (np.c_[one, x, x ** 2, nf], ['a', 'b', 'c', 'd']),
        'quad_srv': (np.c_[one, x, x ** 2, nf, sv], ['a', 'b', 'c', 'd', 'e']),
    }
    anc = {}
    for nm, (X, names) in models.items():
        f = ols(X, acc, names)
        anc[nm] = f
    res['ancova_n'] = {'ours_ch': len(rng_ours), 'ch_nofinal': len(nof),
                       'ours_rhos': sorted({r['knob'] for r in rng_ours}, key=float)}
    res['ancova'] = {nm: {'coef': f['coef'], 'df': f['df'], 'n': f['n'], 'resid_sd': f['resid_sd']}
                     for nm, f in anc.items()}

    # 같은 정확도에서 스파이크 비: (a) 정방향 역산 θ = d/b (ours 곡선이 nofinal 정확도에 닿는 ln s 이동),
    # 비 = s_nof / s_ours(같은 acc) = exp(-θ). 선형 모형만 해석적, 2차는 부트스트랩.
    same_acc = {}
    for nm in ('lin', 'lin_srv'):
        f = anc[nm]
        i_b, i_d = f['names'].index('b'), f['names'].index('d')
        b, d = f['beta'][i_b], f['beta'][i_d]
        vb, vd, cdb = f['cov'][i_b, i_b], f['cov'][i_d, i_d], f['cov'][i_b, i_d]
        th = d / b
        se_th = abs(th) * math.sqrt(vd / d ** 2 + vb / b ** 2 - 2 * cdb / (d * b))
        lo_f, hi_f, kind = fieller(d, b, vd, vb, cdb, f['df'])
        same_acc[nm] = {'theta': th, 'ratio_pct': (math.exp(-th) - 1) * 100,
                        'delta_lo_pct': (math.exp(-(th + tcrit(f['df']) * se_th)) - 1) * 100,
                        'delta_hi_pct': (math.exp(-(th - tcrit(f['df']) * se_th)) - 1) * 100,
                        'fieller': kind,
                        'fieller_lo_pct': None if lo_f is None else (math.exp(-hi_f) - 1) * 100,
                        'fieller_hi_pct': None if lo_f is None else (math.exp(-lo_f) - 1) * 100}
    # 부트스트랩 (팔 안에서 런 재추출). nofinal n=3 재추출은 분산을 과소추정한다.
    rng = np.random.default_rng(0)
    io = np.array([i for i, r in enumerate(data) if r['key'] == 'ours_ch'])
    inf = np.array([i for i, r in enumerate(data) if r['key'] == 'ch_nofinal'])
    boot = {'lin_srv': [], 'quad_srv': [], 'lin_srv_d': [], 'quad_srv_d': []}
    fail = {'lin_srv': 0, 'quad_srv': 0}
    for _ in range(NBOOT):
        idx = np.r_[rng.choice(io, len(io)), rng.choice(inf, len(inf))]
        for nm in ('lin_srv', 'quad_srv'):
            X, names = models[nm]
            Xb, yb = X[idx], acc[idx]
            try:
                beta = np.linalg.lstsq(Xb, yb, rcond=None)[0]
            except np.linalg.LinAlgError:
                fail[nm] += 1
                continue
            if np.linalg.matrix_rank(Xb) < Xb.shape[1]:
                fail[nm] += 1
                continue
            bb = dict(zip(names, beta))
            boot[nm + '_d'].append(bb['d'])
            if nm == 'lin_srv':
                th = bb['d'] / bb['b']
            else:
                # ours 곡선 f(x)=a+bx+cx² 에서 f(x)=f(0)+d 인 x 중 0 에 가장 가까운 근
                c, b_, dd = bb['c'], bb['b'], bb['d']
                if abs(c) < 1e-12:
                    th = dd / b_
                else:
                    disc = b_ * b_ + 4 * c * dd
                    if disc < 0:
                        fail[nm] += 1
                        continue
                    roots = [(-b_ + s * math.sqrt(disc)) / (2 * c) for s in (1, -1)]
                    th = min(roots, key=abs)
            if abs(th) > 5:          # 곡선이 사실상 평평 → 비 정의 불가
                fail[nm] += 1
                continue
            boot[nm].append(th)
    for nm in ('lin_srv', 'quad_srv'):
        th = np.array(boot[nm])
        pct = (np.exp(-th) - 1) * 100
        same_acc.setdefault(nm, {})
        same_acc[nm].update({'boot_n': len(th), 'boot_fail': fail[nm],
                             'boot_med_pct': float(np.median(pct)),
                             'boot_lo_pct': float(np.percentile(pct, 2.5)),
                             'boot_hi_pct': float(np.percentile(pct, 97.5)),
                             'boot_d_lo': float(np.percentile(boot[nm + '_d'], 2.5)),
                             'boot_d_hi': float(np.percentile(boot[nm + '_d'], 97.5))})
    # 2차 모형 점추정 (부트스트랩 없이)
    f = anc['quad_srv']
    bb = dict(zip(f['names'], f['beta']))
    disc = bb['b'] ** 2 + 4 * bb['c'] * bb['d']
    if disc >= 0 and abs(bb['c']) > 1e-12:
        th = min([(-bb['b'] + s * math.sqrt(disc)) / (2 * bb['c']) for s in (1, -1)], key=abs)
        same_acc['quad_srv']['ratio_pct'] = (math.exp(-th) - 1) * 100
        same_acc['quad_srv']['local_slope_b'] = bb['b']
    res['same_acc'] = same_acc
    # (b) 역방향 회귀 (과제 지정). acc 에 시드 잡음이 있어 기울기가 감쇠 → 팔 계수는 두 팔의
    # 평균 ln s 차에 가깝게 끌린다. ours 범위 (200~420K) 설계에 따라 값이 바뀌므로 참고만.
    rev = {}
    for nm, X, names in (('rev', np.c_[one, acc - a0, nf], ['a', 'g', 'd']),
                         ('rev_srv', np.c_[one, acc - a0, nf, sv], ['a', 'g', 'd', 'e'])):
        fr = ols(X, L, names)
        dd = fr['coef']['d']
        rev[nm] = {'pct': (math.exp(dd['est']) - 1) * 100, 'lo_pct': (math.exp(dd['lo']) - 1) * 100,
                   'hi_pct': (math.exp(dd['hi']) - 1) * 100, 'p': dd['p'], 'g': fr['coef']['g'], 'n': fr['n']}
    # 역방향을 국소 창 (±8%) 안에서만
    sel = [r for r in ours if s0 * 0.92 <= r['spikes'] <= s0 * 1.08] + nof
    Ls = np.log(A(sel, 'spikes'))
    Xs = np.c_[np.ones(len(sel)), A(sel, 'val_acc') - a0,
               [1.0 if r['key'] == 'ch_nofinal' else 0.0 for r in sel]]
    fr = ols(Xs, Ls, ['a', 'g', 'd'])
    dd = fr['coef']['d']
    rev['rev_win8'] = {'pct': (math.exp(dd['est']) - 1) * 100, 'lo_pct': (math.exp(dd['lo']) - 1) * 100,
                       'hi_pct': (math.exp(dd['hi']) - 1) * 100, 'p': dd['p'], 'g': fr['coef']['g'], 'n': fr['n']}
    res['reverse'] = rev

    # 3. 지배 관계
    def dominates(p, q):          # p 가 q 를 지배: 스파이크 더 적고 정확도 더 높음 (둘 다 엄격)
        return p['spikes'] < q['spikes'] and p['val_acc'] > q['val_acc']
    dom = {'nof_dominated_by': {}, 'nof_dominates': {}}
    for r in nof:
        dom['nof_dominated_by'][r['tag']] = [o['tag'] for o in ours if dominates(o, r)]
        dom['nof_dominates'][r['tag']] = [o['tag'] for o in ours if dominates(r, o)]
    dom['ours_dominated_by_any_nof'] = sorted({o for v in dom['nof_dominates'].values() for o in v})
    dom['nof_dominated_by_any_ours'] = [t for t, v in dom['nof_dominated_by'].items() if v]
    res['dominance'] = dom
    # 조건 평균 곡선
    by = {}
    for r in ours:
        by.setdefault(r['knob'], []).append(r)
    pooled_num = sum((len(v) - 1) * A(v, 'val_acc').var(ddof=1) for v in by.values() if len(v) > 1)
    pooled_df = sum(len(v) - 1 for v in by.values() if len(v) > 1)
    sd_pool = math.sqrt(pooled_num / pooled_df)
    spk_pool = math.sqrt(sum((len(v) - 1) * np.log(A(v, 'spikes')).var(ddof=1) for v in by.values() if len(v) > 1)
                         / pooled_df)
    cond = sorted([{'rho': k, 'n': len(v), 'spikes': A(v, 'spikes').mean(), 'val_acc': A(v, 'val_acc').mean(),
                    'val_acc_sd': A(v, 'val_acc').std(ddof=1) if len(v) > 1 else None,
                    'servers': [r['server'] for r in v]} for k, v in by.items()], key=lambda c: c['spikes'])
    res['cond_means'] = cond
    res['pooled'] = {'acc_sd': sd_pool, 'df': pooled_df, 'ln_spk_sd': spk_pool,
                     'rhos': [k for k, v in by.items() if len(v) > 1]}

    def interp(pts, s):
        xs = [math.log(p['spikes']) for p in pts]
        x0 = math.log(s)
        for i in range(len(pts) - 1):
            if xs[i] <= x0 <= xs[i + 1]:
                w = (x0 - xs[i]) / (xs[i + 1] - xs[i])
                accv = (1 - w) * pts[i]['val_acc'] + w * pts[i + 1]['val_acc']
                se = sd_pool * math.sqrt((1 - w) ** 2 / pts[i]['n'] + w ** 2 / pts[i + 1]['n'])
                return accv, se, (pts[i]['rho'], pts[i + 1]['rho']), w
        return None, None, None, None

    def crossings(pts, a):
        out = []
        for i in range(len(pts) - 1):
            y1, y2 = pts[i]['val_acc'], pts[i + 1]['val_acc']
            if (y1 - a) * (y2 - a) <= 0 and y1 != y2:
                w = (a - y1) / (y2 - y1)
                x = math.log(pts[i]['spikes']) * (1 - w) + math.log(pts[i + 1]['spikes']) * w
                out.append({'spikes': math.exp(x), 'between': (pts[i]['rho'], pts[i + 1]['rho'])})
        return out

    curve = {}
    se_nof = sd_pool / math.sqrt(len(nof))
    se_nof_own = A(nof, 'val_acc').std(ddof=1) / math.sqrt(len(nof))
    for nm, pts in (('all', cond), ('n3', [c for c in cond if c['n'] >= 3])):
        accv, se, br, w = interp(pts, s0)
        cr = crossings(pts, a0)
        near = min(cr, key=lambda c: abs(math.log(c['spikes'] / s0))) if cr else None
        curve[nm] = {'n_points': len(pts), 'interp_acc': accv, 'interp_se': se, 'bracket': br, 'w': w,
                     'gap': a0 - accv, 'gap_se': math.sqrt(se ** 2 + se_nof ** 2),
                     'gap_se_own': math.sqrt(se ** 2 + se_nof_own ** 2),
                     'same_acc_crossings': cr,
                     'same_acc_nearest_spikes': near['spikes'] if near else None,
                     'same_acc_spk_pct': (s0 / near['spikes'] - 1) * 100 if near else None}
    res['mean_curve'] = curve

    # 4. 검정력
    obs = {'ancova_lin_srv_d': anc['lin_srv']['coef']['d']['est'],
           'ancova_quad_srv_d': anc['quad_srv']['coef']['d']['est'],
           'window8_diff': [w for w in win if w['window'] == 0.08][0]['acc']['diff'],
           'mean_curve_gap_n3': curve['n3']['gap']}
    pw = {}
    for k, dlt in obs.items():
        pw[k] = {'delta': dlt, 'n_per_group': n_needed(dlt, sd_pool)}
    pw['_mde_now'] = {'n_nof': len(nof), 'n_ours_win8': [w for w in win if w['window'] == 0.08][0]['n_ours'],
                      'mde': mde(sd_pool, len(nof), [w for w in win if w['window'] == 0.08][0]['n_ours'])}
    f = anc['lin_srv']
    se_d = f['coef']['d']['se']
    var_curve = max(se_d ** 2 - f['resid_sd'] ** 2 / len(nof), 0.0)
    k = stats.t.ppf(0.975, f['df']) + stats.t.ppf(0.8, f['df'])
    pw['_ancova_mde'] = {'curve_se': math.sqrt(var_curve), 'df': f['df'],
                         'by_n_nof': {str(m): k * math.sqrt(sd_pool ** 2 / m + var_curve) for m in (3, 4, 6, 10, 20)},
                         'floor': k * math.sqrt(var_curve)}
    pw['_mde_n4'] = {'mde_4v4': mde(sd_pool, 4, 4), 'mde_3v3': mde(sd_pool, 3, 3), 'mde_6v6': mde(sd_pool, 6, 6)}
    res['power'] = pw

    # 5. 서버
    b138 = [r for r in base if r['server'] == '138']
    b23 = [r for r in base if r['server'] == '23']
    srv = {'base_acc': welch(A(b23, 'val_acc'), A(b138, 'val_acc')),
           'base_lnspk': welch(np.log(A(b23, 'spikes')), np.log(A(b138, 'spikes'))),
           'base_n': {'138': len(b138), '23': len(b23)}}
    # ours_ch: ρ 고정효과 + 서버
    knobs = sorted(by, key=float)
    Xr = np.array([[1.0 if r['knob'] == k else 0.0 for k in knobs] + [1.0 if r['server'] == '23' else 0.0]
                   for r in ours])
    names = ['rho_' + k for k in knobs] + ['srv23']
    fa = ols(Xr, A(ours, 'val_acc'), names)
    fs = ols(Xr, np.log(A(ours, 'spikes')), names)
    srv['ours_acc_srv23'] = fa['coef']['srv23']
    srv['ours_lnspk_srv23'] = fs['coef']['srv23']
    srv['ours_n'] = {'138': sum(r['server'] == '138' for r in ours), '23': sum(r['server'] == '23' for r in ours),
                     'df': fa['df']}
    srv['ours_mixed_rhos'] = [k for k in knobs if len({r['server'] for r in by[k]}) > 1]
    # 서버 하위집합 ANCOVA (선형)
    sub = {}
    for s in ('138', '23'):
        dd = [r for r in data if r['server'] == s]
        Ls_ = np.log(A(dd, 'spikes')) - Lc0
        X = np.c_[np.ones(len(dd)), Ls_, [1.0 if r['key'] == 'ch_nofinal' else 0.0 for r in dd]]
        if sum(r['key'] == 'ch_nofinal' for r in dd) >= 1 and len(dd) > 4:
            f2 = ols(X, A(dd, 'val_acc'), ['a', 'b', 'd'])
            sub[s] = {'n_ours': sum(r['key'] == 'ours_ch' for r in dd),
                      'n_nof': sum(r['key'] == 'ch_nofinal' for r in dd), 'd': f2['coef']['d'], 'b': f2['coef']['b']}
    srv['ancova_by_server'] = sub
    res['server'] = srv

    # 6. baseline 대비
    def vs_base(group, label):
        wa = welch(A(group, 'val_acc'), A(base, 'val_acc'))
        ws = welch(np.log(A(group, 'spikes')), np.log(A(base, 'spikes')))
        return {'label': label, 'n': len(group), 'n_base': len(base), 'acc': wa,
                'spk_pct': (math.exp(ws['diff']) - 1) * 100,
                'spk_lo': (math.exp(ws['lo']) - 1) * 100, 'spk_hi': (math.exp(ws['hi']) - 1) * 100}
    w8 = [r for r in ours if s0 * 0.92 <= r['spikes'] <= s0 * 1.08]
    vb = {'base_mean': {'val_acc': A(base, 'val_acc').mean(), 'val_acc_sd': A(base, 'val_acc').std(ddof=1),
                        'spikes': A(base, 'spikes').mean(), 'n': len(base)},
          'nofinal': vs_base(nof, 'ch_nofinal 2.2e-3'),
          'ours_win8': vs_base(w8, 'ours_ch ±8% 창'),
          'ours_by_rho': [vs_base(by[k], f'ours_ch {k}') for k in knobs if len(by[k]) >= 2]}
    res['vs_base'] = vb
    return res, base, nof, ours


# ----------------------------------------------------------------------------- 보고
def fmt_p(p):
    return f'{p:.3f}' if p >= 0.001 else f'{p:.1e}'


def spk(x):
    return f'{x:,.0f}'


def write_md(runs, res, base, nof, ours, dup, inv_path, cache):
    L = []
    w = L.append
    nb, nn, no = len(base), len(nof), len(ours)
    s0 = res['nofinal_mean']['spikes']
    a0 = res['nofinal_mean']['val_acc']
    bm = res['vs_base']['base_mean']
    anc = res['ancova']
    sa = res['same_acc']
    win = {x['window']: x for x in res['window']}
    cv = res['mean_curve']
    pw = res['power']
    sv = res['server']
    dm = res['dominance'] if 'dominance' in res else None

    w('# `- final_step (inter)` vs `ours_inter_ch` — 같은 스파이크에서 정확도가 높은가 (R19 CIFAR-10, 26-10-14)')
    w('')
    w('- 질문 (사용자 10-14): "비슷한 성능에서도 스파이크가 적고 스파이크가 적은데 정확도가 높은데?" — '
      '`- final_step (inter)` (run 키 `ch_nofinal`, `ours_inter_ch` 에서 final_step 만 끔) 이 `ours_inter_ch` '
      '(run 키 `ours_ch`) 를 정확도-스파이크 평면에서 지배하는가.')
    w(f'- 정확도 = 310 에폭 중 best val acc, spikes = 그 에폭의 s_count (원값, 이미지당). train.log 에서 직접 다시 읽음. '
      f'팔은 config override 블록으로 확인 (두 팔의 설정 차이는 `reg_spike_final_step` 과 ρ 뿐).')
    w(f'- 대상: `ch_nofinal` ρ 2.2e-3 n={nn} (s1=23, s2·s3=138), `ours_ch` 완주·비배제 n={no} (ρ 1.5e-3~1.2e-2), '
      f'baseline n={nb} (138 3 · 23 3). `abl_nofinal` (intra 에서 final_step 끈 팔) 은 섞지 않음.')
    w(f'- 재현: `~/anaconda3/envs/venv_1/bin/python analysis/nofinal_vs_inter_261014.py` '
      f'(sejong 에폭 줄 캐시 `{cache}`, 없으면 ssh 로 다시 추출). 수치 원본 `analysis/nofinal_vs_inter_261014.json`.')
    w('')
    w('## 요약')
    w('')
    d8 = win[0.08]['acc']
    ls = anc['lin_srv']['coef']['d']
    qs = anc['quad_srv']['coef']['d']
    w(f'1. **국소 비교 (±8% 창)**: `- final_step` {a0:.2f}% (n={nn}) vs `ours_inter_ch` '
      f'{win[0.08]["ours_acc_mean"]:.2f}% (n={win[0.08]["n_ours"]}) → Δacc **{d8["diff"]:+.3f}%p**, '
      f'95% CI [{d8["lo"]:+.3f}, {d8["hi"]:+.3f}], p={fmt_p(d8["p"])}. 스파이크 {win[0.08]["spk_pct"]:+.1f}% '
      f'[{win[0.08]["spk_pct_lo"]:+.1f}, {win[0.08]["spk_pct_hi"]:+.1f}]. ±5%·±12% 창도 같은 그림 (1절 표).')
    w(f'2. **곡선 비교 (ANCOVA, ours 200~420K n={res["ancova_n"]["ours_ch"]} + nofinal n={nn})**: 같은 스파이크에서 '
      f'오프셋 d = **{ls["est"]:+.3f}%p** [{ls["lo"]:+.3f}, {ls["hi"]:+.3f}] p={fmt_p(ls["p"])} (선형+서버), '
      f'{qs["est"]:+.3f}%p [{qs["lo"]:+.3f}, {qs["hi"]:+.3f}] p={fmt_p(qs["p"])} (2차+서버). '
      f'같은 정확도에서 스파이크 비: {sa["lin_srv"]["ratio_pct"]:+.1f}% (선형, Fieller 구간 '
      + (f'[{sa["lin_srv"]["fieller_lo_pct"]:+.1f}, {sa["lin_srv"]["fieller_hi_pct"]:+.1f}]'
         if sa['lin_srv']['fieller'] == 'bounded' else '**무한** — 기울기 b 가 0 과 구별되지 않음')
      + f', 부트스트랩 [{sa["lin_srv"]["boot_lo_pct"]:+.1f}, {sa["lin_srv"]["boot_hi_pct"]:+.1f}]).')
    nd = res['dominance']
    w(f'3. **지배 관계**: 시드 단위로 `- final_step` 3런 중 {len(nd["nof_dominated_by_any_ours"])}런은 '
      f'그보다 스파이크 적고 정확도 높은 `ours_inter_ch` 런이 있다. 반대로 `- final_step` 런에 지배되는 '
      f'`ours_inter_ch` 런은 {len(nd["ours_dominated_by_any_nof"])}개 / {no}. 조건 평균 곡선 대비 '
      f'`- final_step` 평균점은 {cv["all"]["gap"]:+.3f}%p (모든 ρ 평균점) / {cv["n3"]["gap"]:+.3f}%p (n≥3 ρ 만) '
      f'± {cv["n3"]["gap_se"]:.3f}.')
    w(f'4. **검정력**: `ours_inter_ch` 같은 ρ 시드 간 합동 SD = {res["pooled"]["acc_sd"]:.3f}%p '
      f'(자유도 {res["pooled"]["df"]}, ρ {len(res["pooled"]["rhos"])}개). 관측 차 '
      f'{pw["ancova_lin_srv_d"]["delta"]:+.3f}%p 를 α=0.05·검정력 0.8 로 보려면 조건당 n≈'
      f'**{pw["ancova_lin_srv_d"]["n_per_group"]}**. 지금 (3 vs {pw["_mde_now"]["n_ours_win8"]}) 으로 볼 수 있는 최소 차는 '
      f'{pw["_mde_now"]["mde"]:.2f}%p. 곡선 ANCOVA 로도 ours 곡선을 지금대로 두면 nofinal n=4 에서 '
      f'{pw["_ancova_mde"]["by_n_nof"]["4"]:.2f}%p, n→∞ 여도 {pw["_ancova_mde"]["floor"]:.2f}%p 가 하한.')
    w(f'5. **서버**: baseline 23−138 = {sv["base_acc"]["diff"]:+.3f}%p (p={fmt_p(sv["base_acc"]["p"])}, 각 n=3), '
      f'스파이크 {(math.exp(sv["base_lnspk"]["diff"]) - 1) * 100:+.1f}% (p={fmt_p(sv["base_lnspk"]["p"])}). '
      f'`ours_inter_ch` (같은 ρ 고정) 23−138 = {sv["ours_acc_srv23"]["est"]:+.3f}%p (p={fmt_p(sv["ours_acc_srv23"]["p"])}), '
      f'스파이크 {(math.exp(sv["ours_lnspk_srv23"]["est"]) - 1) * 100:+.1f}% (p={fmt_p(sv["ours_lnspk_srv23"]["p"])}). '
      f'서버 항을 넣고 빼도 d 는 {anc["lin"]["coef"]["d"]["est"]:+.3f} → {ls["est"]:+.3f}%p — 결론 불변.')
    vn = res['vs_base']['nofinal']
    vo = res['vs_base']['ours_win8']
    w(f'6. **baseline 대비** (baseline {bm["val_acc"]:.2f}% @ {spk(bm["spikes"])}, n={nb}): `- final_step` '
      f'{vn["acc"]["diff"]:+.3f}%p [{vn["acc"]["lo"]:+.3f}, {vn["acc"]["hi"]:+.3f}] · 스파이크 {vn["spk_pct"]:+.1f}% (n={nn}); '
      f'근처 `ours_inter_ch` (±8%) {vo["acc"]["diff"]:+.3f}%p [{vo["acc"]["lo"]:+.3f}, {vo["acc"]["hi"]:+.3f}] · '
      f'{vo["spk_pct"]:+.1f}% (n={vo["n"]}). 둘 다 baseline 과 정확도가 구별되지 않는다.')
    w('')
    ests = [x['acc']['diff'] for x in res['window']] + [anc[m]['coef']['d']['est'] for m in anc] + \
           [cv['all']['gap'], cv['n3']['gap']]
    los = [x['acc']['lo'] for x in res['window']] + [anc[m]['coef']['d']['lo'] for m in anc]
    w('**판정**: 사용자 주장은 **방향은 맞지만 데이터로 지지되지 않는다.** '
      f'(i) 같은 스파이크에서 정확도: 모든 추정 (창 3개, ANCOVA 4개, 평균 곡선 2개) 이 {min(ests):+.2f}~{max(ests):+.2f}%p 로 '
      f'양수지만 시드 SD ({res["pooled"]["acc_sd"]:.2f}%p) 의 절반 이하이고 p 는 전부 0.19 이상. '
      f'(ii) 같은 정확도에서 스파이크: {sa["lin_srv"]["ratio_pct"]:+.0f}% 지만 95% 구간이 '
      f'[{sa["lin_srv"]["fieller_lo_pct"]:+.0f}, {sa["lin_srv"]["fieller_hi_pct"]:+.0f}]% 로 0 을 넓게 포함. '
      f'(iii) 시드 단위 지배는 양방향 — `- final_step` 이 ours 런 {len(nd["ours_dominated_by_any_nof"])}/{no} 개를 지배하지만, '
      'ours 7e-3 s2 (96.75 @ 220K) 한 런이 `- final_step` 3런을 모두 지배한다. '
      f'말할 수 있는 것은 **"final_step 을 꺼도 같은 스파이크에서 정확도가 떨어지지 않는다"** 까지다 '
      f'(n=3, 95% 하한 {min(los):+.2f}%p — 이보다 큰 손실은 배제). '
      '"이긴다" 를 말하려면 조건당 n 이 수십 개 필요하다 — 조건당 n=4 상한 (10-07 규칙) 으로는 이 크기의 차를 볼 수 없다.')
    w('')

    # ---------------- 0. 데이터 표
    w('## 0. 런별 값 (best 에폭)')
    w('')
    w('열: train loss (규제 포함 `loss`) → train acc → val loss → val acc → spikes, 뒤에 파생 (baseline 대비 스파이크). '
      '평균 행은 배제 행을 뺀 값, ± 는 시드 SD.')
    w('')
    w('| 팔 | ρ | seed | 서버 | best ep | train loss | train acc | val loss | val acc | spikes | spikes vs base | 비고 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|')
    bspk = bm['spikes']

    def row(r, label, rho):
        if 'val_acc' not in r:
            return f'| {label} | {rho} | s{r["seed"]} | {r["server"]} | – | – | – | – | – | – | – | {r["exclude_reason"]} |'
        return (f'| {label} | {rho} | s{r["seed"]} | {r["server"]} | {r["best_epoch"]} | {r["train_loss"]:.4f} | '
                f'{r["train_acc"]:.2f} | {r["val_loss"]:.4f} | {r["val_acc"]:.2f} | {spk(r["spikes"])} | '
                f'{(r["spikes"] / bspk - 1) * 100:+.1f}% | {r["exclude_reason"] or ""} |')

    def mean_row(rs, label, rho):
        A = lambda k: np.array([r[k] for r in rs], float)
        sd = f' ±{A("val_acc").std(ddof=1):.2f}' if len(rs) > 1 else ''
        ssd = f' ±{A("spikes").std(ddof=1):,.0f}' if len(rs) > 1 else ''
        return (f'| **{label} 평균** | {rho} | n={len(rs)} | {"/".join(sorted({r["server"] for r in rs}))} | | '
                f'{A("train_loss").mean():.4f} | {A("train_acc").mean():.2f} | {A("val_loss").mean():.4f} | '
                f'**{A("val_acc").mean():.2f}{sd}** | {A("spikes").mean():,.0f}{ssd} | '
                f'{(A("spikes").mean() / bspk - 1) * 100:+.1f}% | |')

    allb = sorted([r for r in runs if r['key'] == 'base'], key=lambda r: r['seed'])
    for r in allb:
        w(row(r, 'baseline', '–'))
    w(mean_row(base, 'baseline', '–'))
    alln = sorted([r for r in runs if r['key'] == 'ch_nofinal'], key=lambda r: r['seed'])
    for r in alln:
        w(row(r, '- final_step (inter)', '2.2e-3'))
    w(mean_row(nof, '- final_step (inter)', '2.2e-3'))
    allo = [r for r in runs if r['key'] == 'ours_ch']
    for k in sorted({r['knob'] for r in allo}, key=float):
        g = sorted([r for r in allo if r['knob'] == k], key=lambda r: r['seed'])
        for r in g:
            w(row(r, 'ours_inter_ch', k))
        gu = [r for r in g if r['use']]
        if gu:
            w(mean_row(gu, 'ours_inter_ch', k))
    w('')
    w('표에 없는 행 (원장 10-14 10:28 기준, 평균·검정 모두 제외): `- final_step (inter)` 진행 중 — '
      'ρ 1.4e-3 s1·s2 (138, 약 148/310 에폭), 1e-3·1.6e-3·1.8e-3 s1 (138, 시작 직후), 7e-4·2.8e-3 s1 (23, 시작 직후), '
      '3e-3 s901 (138, 2 에폭 연기시험). `ours_inter_ch` 3e-3 s5·8e-3 s2·s3 hold (미시작).')
    if dup:
        w(f'- 두 서버에 같은 이름의 런: {dup} — 로컬(138) 쪽을 씀.')
    mism = [r for r in runs if r.get('inv_check') and r['use'] and
            ((r['inv_check']['val_acc_diff'] or 0) != 0 or abs(r['inv_check']['spikes_diff'] or 0) > 1
             or not r['inv_check']['server_match'])]
    w(f'- 원장 (`inventory.json`) 교차검증: 쓰인 런 {sum(r["use"] for r in runs)}개 중 val acc·spikes·서버 불일치 '
      f'{len(mism)}건' + (': ' + ', '.join(r['tag'] for r in mism) if mism else '.'))
    w('')

    # ---------------- 1
    w(f'## 1. 국소 비교 — `- final_step` 평균 스파이크 {spk(s0)} (n={nn}) 중심 창')
    w('')
    w('Welch t. Δ = `- final_step` − `ours_inter_ch`. 스파이크 % 는 ln 스파이크 평균 차의 지수.')
    w('')
    w('| 창 | 스파이크 범위 | ours n (서버 138/23) | ours val acc | nofinal val acc (n=3) | Δacc %p | 95% CI | p | Δspikes | 95% CI |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for x in res['window']:
        a = x['acc']
        w(f'| ±{x["window"] * 100:.0f}% | {spk(x["lo"])}~{spk(x["hi"])} | {x["n_ours"]} '
          f'({x["ours_servers"].count("138")}/{x["ours_servers"].count("23")}) | '
          f'{x["ours_acc_mean"]:.3f} ±{x["ours_acc_sd"]:.3f} | {a0:.3f} ±{res["nofinal_mean"]["val_acc_sd"]:.3f} | '
          f'**{a["diff"]:+.3f}** | [{a["lo"]:+.3f}, {a["hi"]:+.3f}] | {fmt_p(a["p"])} | '
          f'{x["spk_pct"]:+.1f}% | [{x["spk_pct_lo"]:+.1f}, {x["spk_pct_hi"]:+.1f}] |')
    w('')
    for x in res['window']:
        w(f'- ±{x["window"] * 100:.0f}% 창 ours 런: ' + ', '.join(t.replace('r19c10-ours_ch-', '') for t in x['ours_tags']))
    w('- 창 안의 ours 평균 스파이크가 nofinal 과 ±3% 안이라 스파이크 보정 없이도 비교가 성립한다. '
      '세 창 모두 Δacc 는 시드 SD 의 절반 이하, CI 폭은 약 ±0.15~0.2%p.')
    w('')

    # ---------------- 2
    an = res['ancova_n']
    w(f'## 2. 곡선 비교 (ANCOVA) — ours 200K~420K n={an["ours_ch"]} (ρ {", ".join(an["ours_rhos"])}) + nofinal n={nn}')
    w('')
    w('`acc = a + b·x [+ c·x²] + d·[nofinal] + e·[server=23]`, x = ln(spikes) − ln(nofinal 평균 스파이크). '
      'd = 같은 스파이크에서의 정확도 오프셋.')
    w('')
    w('| 모형 | n | d (%p) | 95% CI | p | b (%p / ln) | b 의 p | c | e (서버 23) | 잔차 SD |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for nm, lab in (('lin', '선형'), ('lin_srv', '선형 + 서버'), ('quad', '2차'), ('quad_srv', '2차 + 서버')):
        f = anc[nm]
        c = f['coef']
        cc = f'{c["c"]["est"]:+.3f} (p={fmt_p(c["c"]["p"])})' if 'c' in c else '–'
        ee = f'{c["e"]["est"]:+.3f} (p={fmt_p(c["e"]["p"])})' if 'e' in c else '–'
        w(f'| {lab} | {f["n"]} | **{c["d"]["est"]:+.3f}** | [{c["d"]["lo"]:+.3f}, {c["d"]["hi"]:+.3f}] | '
          f'{fmt_p(c["d"]["p"])} | {c["b"]["est"]:+.3f} ±{c["b"]["se"]:.3f} | {fmt_p(c["b"]["p"])} | {cc} | {ee} | '
          f'{f["resid_sd"]:.3f} |')
    w('')
    w(f'- 잔차 SD ({anc["lin_srv"]["resid_sd"]:.3f}) 가 같은 ρ 시드 간 합동 SD ({res["pooled"]["acc_sd"]:.3f}) 와 비슷 → '
      '선형 곡선이 크게 어긋나지 않는다. 2차 항은 유의하지 않다.')
    w(f'- 부트스트랩 (팔 안 런 재추출 {NBOOT}회) d 95%: 선형+서버 [{sa["lin_srv"]["boot_d_lo"]:+.3f}, '
      f'{sa["lin_srv"]["boot_d_hi"]:+.3f}], 2차+서버 [{sa["quad_srv"]["boot_d_lo"]:+.3f}, {sa["quad_srv"]["boot_d_hi"]:+.3f}]. '
      'nofinal n=3 재추출이라 이 구간은 t 구간보다 좁게 나온다 (과소추정).')
    w('')
    w('### 같은 정확도에서 스파이크 차')
    w('')
    w('| 방법 | 스파이크 차 (nofinal / ours − 1) | 95% 구간 | 비고 |')
    w('|---|---|---|---|')
    for nm, lab in (('lin', '정방향 역산 (선형), θ=d/b'), ('lin_srv', '정방향 역산 (선형 + 서버)')):
        s = sa[nm]
        fl = (f'Fieller [{s["fieller_lo_pct"]:+.1f}, {s["fieller_hi_pct"]:+.1f}]' if s['fieller'] == 'bounded'
              else 'Fieller **무한** (b 비유의)')
        w(f'| {lab} | **{s["ratio_pct"]:+.1f}%** | 델타법 [{s["delta_lo_pct"]:+.1f}, {s["delta_hi_pct"]:+.1f}]; {fl}'
          + (f'; 부트스트랩 [{s["boot_lo_pct"]:+.1f}, {s["boot_hi_pct"]:+.1f}]' if 'boot_lo_pct' in s else '')
          + f' | b={anc[nm]["coef"]["b"]["est"]:+.3f}, p={fmt_p(anc[nm]["coef"]["b"]["p"])} |')
    q = sa['quad_srv']
    w(f'| 정방향 역산 (2차 + 서버) | {q.get("ratio_pct", float("nan")):+.1f}% | 부트스트랩 '
      f'[{q["boot_lo_pct"]:+.1f}, {q["boot_hi_pct"]:+.1f}] ({q["boot_fail"]}/{NBOOT} 회는 곡선이 평평해 근 없음) | |')
    for nm, lab in (('rev', '역회귀 ln s ~ acc + 팔'), ('rev_srv', '역회귀 + 서버'), ('rev_win8', '역회귀, ±8% 창 안만')):
        r = res['reverse'][nm]
        w(f'| {lab} (n={r["n"]}) | {r["pct"]:+.1f}% | [{r["lo_pct"]:+.1f}, {r["hi_pct"]:+.1f}], p={fmt_p(r["p"])} | '
          f'acc 기울기 g={r["g"]["est"]:+.3f} (p={fmt_p(r["g"]["p"])}) |')
    w('')
    gfull = res['reverse']['rev_srv']['g']['est']
    gwin = res['reverse']['rev_win8']['g']['est']
    binv = 1 / anc['lin_srv']['coef']['b']['est']
    w(f'- **역회귀는 주 결과로 쓰지 않는다.** acc 는 시드 잡음을 담은 설명변수라 기울기 g 가 감쇠된다: '
      f'정방향 기울기의 역수 1/b = {binv:.2f} 인데 역회귀 g = {gfull:.2f} (200~420K), ±8% 창 안에서는 {gwin:.2f} 로 거의 0. '
      'g 가 0 에 가까울수록 팔 계수는 "두 팔의 평균 ln 스파이크 차" 가 된다 — 창 안 값 '
      f'{res["reverse"]["rev_win8"]["pct"]:+.1f}% 는 창 안 스파이크 차 ({win[0.08]["spk_pct"]:+.1f}%) 와 같은 것이다. '
      '즉 ours 를 어느 범위에서 뽑았느냐가 답을 정한다.')
    w(f'- 정방향 역산 θ=d/b 가 일관된 정의다. 200~420K 에서 b 는 유의하다 (스파이크 2배에 '
      f'{anc["lin_srv"]["coef"]["b"]["est"] * math.log(2):.2f}%p, p={fmt_p(anc["lin_srv"]["coef"]["b"]["p"])}) 그래서 '
      'Fieller 구간이 유계지만, d 자체가 0 과 구별되지 않아 비의 구간이 −36%~+22% 로 넓다. 정확도 축 오프셋 d 를 주 결과로 쓴다.')
    w('')

    # ---------------- 3
    w('## 3. 지배 관계')
    w('')
    w('### (a) 시드 단위 — 엄격 지배 (스파이크 더 적고 AND 정확도 더 높음)')
    w('')
    w('| `- final_step` 런 | 서버 | val acc | spikes | 이 런을 지배하는 ours_inter_ch 런 | 이 런이 지배하는 ours_inter_ch 런 |')
    w('|---|---|---|---|---|---|')
    for r in nof:
        by_ = nd['nof_dominated_by'][r['tag']]
        dm_ = nd['nof_dominates'][r['tag']]
        cut = lambda t: t.replace('r19c10-ours_ch-', '')
        info = lambda t: next(f'{cut(t)} ({o["val_acc"]:.2f} @ {o["spikes"] / 1e3:.0f}K, {o["server"]})'
                              for o in ours if o['tag'] == t)
        w(f'| s{r["seed"]} | {r["server"]} | {r["val_acc"]:.2f} | {spk(r["spikes"])} | '
          f'{len(by_)}: ' + (', '.join(info(t) for t in by_) or '–') + ' | '
          f'{len(dm_)}: ' + (', '.join(info(t) for t in dm_) or '–') + ' |')
    w('')
    w(f'- `- final_step` 3런 중 ours 런에 지배되는 것 {len(nd["nof_dominated_by_any_ours"])}개; '
      f'`- final_step` 어느 한 런에라도 지배되는 ours 런 {len(nd["ours_dominated_by_any_nof"])}개 / {no}.')
    w('- 양방향 지배가 모두 존재한다. 정확도 범위 (96.4~96.9) 가 시드 SD 2~3배 폭이라 어떤 팔이든 운 좋은 시드 하나가 '
      '다른 팔의 여러 런을 "지배" 한다. 지배 개수는 표본 수와 시드 운에 좌우돼 효과 지표로 쓸 수 없다.')
    w('')
    w('### (b) 조건 평균 단위 — ours_inter_ch ρ 별 평균점 (스파이크 순)')
    w('')
    w('| ρ | n | 서버 | val acc 평균 | 시드 SD | spikes 평균 |')
    w('|---|---|---|---|---|---|')
    for c in res['cond_means']:
        sdv = '–' if c['val_acc_sd'] is None else f"{c['val_acc_sd']:.3f}"
        w(f'| {c["rho"]} | {c["n"]} | {"/".join(c["servers"])} | {c["val_acc"]:.3f} | {sdv} | {spk(c["spikes"])} |')
    w(f'| **- final_step 2.2e-3** | {nn} | 23/138/138 | **{a0:.3f}** | {res["nofinal_mean"]["val_acc_sd"]:.3f} | **{spk(s0)}** |')
    w('')
    w('| 평균 곡선 | 점 수 | 보간 구간 (ρ) | 곡선 값 @ nofinal 스파이크 | nofinal − 곡선 (%p) | ± SE (합동 SD) | ± SE (nofinal 자체 SD) | 같은 정확도 교차점 | 그 점 대비 스파이크 |')
    w('|---|---|---|---|---|---|---|---|---|')
    for nm, lab in (('all', '모든 ρ'), ('n3', 'n≥3 ρ 만')):
        c = cv[nm]
        cr = ', '.join(f'{spk(x["spikes"])} ({x["between"][0]}~{x["between"][1]})' for x in c['same_acc_crossings'])
        pct_s = '–' if c['same_acc_spk_pct'] is None else f"{c['same_acc_spk_pct']:+.1f}%"
        w(f'| {lab} | {c["n_points"]} | {c["bracket"][0]}~{c["bracket"][1]} | {c["interp_acc"]:.3f} | '
          f'**{c["gap"]:+.3f}** | {c["gap_se"]:.3f} | {c["gap_se_own"]:.3f} | {cr or "없음"} | '
          f'{pct_s} |')
    w('')
    w('- 보간은 ln(spikes) 축 선형. SE 는 두 이웃 평균점의 SE (합동 SD/√n) 와 nofinal 평균의 SE 를 전파.')
    w('- 평균 곡선 자체가 시드 잡음으로 들쭉날쭉하다 (ρ 4e-3 → 4.5e-3 → 5e-3 이 96.66 → 96.60 → 96.65, 위·아래·위). '
      f'nofinal 정확도 ({a0:.3f}) 를 가로지르는 곳은 4e-3~3e-3 사이 한 곳뿐인데, 4e-3 평균 (96.657, n=3) 이 nofinal 보다 '
      '0.006%p 낮아서 생긴 교차다. 4e-3 평균이 0.01%p 만 높았어도 교차점이 302K 아래로 내려와 "같은 정확도 스파이크 차" 가 '
      '−6.7% 에서 0 근처로 바뀐다 — 이 열의 값은 시드 하나에 좌우된다.')
    w('')

    # ---------------- 4
    w('## 4. 검정력')
    w('')
    w(f'- 합동 SD = **{res["pooled"]["acc_sd"]:.3f}%p** (`ours_inter_ch`, n≥2 인 ρ {len(res["pooled"]["rhos"])}개: '
      f'{", ".join(sorted(res["pooled"]["rhos"], key=float))}, 자유도 {res["pooled"]["df"]}). '
      f'ln 스파이크 합동 SD = {res["pooled"]["ln_spk_sd"] * 100:.1f}%.')
    w('- 두 표본 t, 조건당 같은 n, α=0.05 양측, 검정력 0.8.')
    w('')
    w('| 관측 차의 출처 | Δacc (%p) | 필요 n (조건당) |')
    w('|---|---|---|')
    for k, lab in (('ancova_lin_srv_d', 'ANCOVA d (선형 + 서버)'), ('ancova_quad_srv_d', 'ANCOVA d (2차 + 서버)'),
                   ('window8_diff', '±8% 창 Welch'), ('mean_curve_gap_n3', '평균 곡선 차 (n≥3 ρ)')):
        p = pw[k]
        w(f'| {lab} | {p["delta"]:+.3f} | **{p["n_per_group"]}** |')
    w('')
    m = pw['_mde_n4']
    w(f'- 현재 설계 (nofinal 3 vs ±8% 창 ours {pw["_mde_now"]["n_ours_win8"]}) 의 최소 검출 차: '
      f'**{pw["_mde_now"]["mde"]:.2f}%p**. 3 vs 3: {m["mde_3v3"]:.2f}, 4 vs 4: {m["mde_4v4"]:.2f}, 6 vs 6: {m["mde_6v6"]:.2f}%p.')
    am = pw['_ancova_mde']
    w(f'- 곡선 ANCOVA 의 최소 검출 차 (ours 곡선 n={res["ancova_n"]["ours_ch"]} 을 지금대로 두고 nofinal 반복만 늘릴 때; '
      f'곡선 자체의 SE {am["curve_se"]:.3f}%p, 합동 SD 사용): '
      + ', '.join(f'n={k} → {v:.2f}' for k, v in am['by_n_nof'].items())
      + f', n→∞ → {am["floor"]:.2f}%p. 관측 차 {pw["ancova_lin_srv_d"]["delta"]:+.3f}%p 는 nofinal 을 아무리 늘려도 '
      'ours 곡선 오차 때문에 이 설계로는 검출되지 않는다. 두 팔 모두 같은 스파이크 근처에 수십 런씩 필요하다.')
    w('')

    # ---------------- 5
    w('## 5. 서버 효과 (23 − 138)')
    w('')
    w('| 비교 | n (138 / 23) | Δacc %p | 95% CI | p | Δspikes | p |')
    w('|---|---|---|---|---|---|---|')
    ba, bs = sv['base_acc'], sv['base_lnspk']
    w(f'| baseline (Welch) | {sv["base_n"]["138"]} / {sv["base_n"]["23"]} | {ba["diff"]:+.3f} | '
      f'[{ba["lo"]:+.3f}, {ba["hi"]:+.3f}] | {fmt_p(ba["p"])} | {(math.exp(bs["diff"]) - 1) * 100:+.1f}% '
      f'[{(math.exp(bs["lo"]) - 1) * 100:+.1f}, {(math.exp(bs["hi"]) - 1) * 100:+.1f}] | {fmt_p(bs["p"])} |')
    oa, os_ = sv['ours_acc_srv23'], sv['ours_lnspk_srv23']
    w(f'| ours_inter_ch (ρ 고정효과 + 서버, 자유도 {sv["ours_n"]["df"]}) | {sv["ours_n"]["138"]} / {sv["ours_n"]["23"]} | '
      f'{oa["est"]:+.3f} | [{oa["lo"]:+.3f}, {oa["hi"]:+.3f}] | {fmt_p(oa["p"])} | '
      f'{(math.exp(os_["est"]) - 1) * 100:+.1f}% [{(math.exp(os_["lo"]) - 1) * 100:+.1f}, '
      f'{(math.exp(os_["hi"]) - 1) * 100:+.1f}] | {fmt_p(os_["p"])} |')
    lsc = anc['lin_srv']['coef']['e']
    w(f'| ANCOVA e (ours 200~420K + nofinal, 같은 스파이크) | – | {lsc["est"]:+.3f} | [{lsc["lo"]:+.3f}, {lsc["hi"]:+.3f}] | '
      f'{fmt_p(lsc["p"])} | – | – |')
    w('')
    w(f'- ours_inter_ch 의 서버 효과는 두 서버 런이 섞인 ρ ({", ".join(sv["ours_mixed_rhos"])}) 에서만 식별된다.')
    w('- `- final_step` 내부: s1 (23) ' + ', '.join(f'{r["val_acc"]:.2f} @ {spk(r["spikes"])}' for r in nof if r['server'] == '23')
      + ' / s2·s3 (138) ' + ', '.join(f'{r["val_acc"]:.2f} @ {spk(r["spikes"])}' for r in nof if r['server'] == '138') + '.')
    w('')
    w('서버별로 나눈 ANCOVA (선형, 서버 항 없음):')
    w('')
    w('| 서버 | ours n | nofinal n | d (%p) | 95% CI | p |')
    w('|---|---|---|---|---|---|')
    for s, x in sv['ancova_by_server'].items():
        w(f'| {s} | {x["n_ours"]} | {x["n_nof"]} | {x["d"]["est"]:+.3f} | [{x["d"]["lo"]:+.3f}, {x["d"]["hi"]:+.3f}] | '
          f'{fmt_p(x["d"]["p"])} |')
    w('')
    w(f'- 서버 차는 정확도·스파이크 모두 시드 잡음 범위다. 서버 항 유무로 d 가 {anc["lin"]["coef"]["d"]["est"]:+.3f} → '
      f'{anc["lin_srv"]["coef"]["d"]["est"]:+.3f}%p 로 거의 그대로이고, 서버별로 나눠도 부호가 같다 (23 쪽은 nofinal n=1). '
      '`- final_step` 의 서버 구성 (23 1 · 138 2) 은 결론을 바꾸지 않는다.')
    w('')

    # ---------------- 6
    vn_, vo_ = res['vs_base']['nofinal'], res['vs_base']['ours_win8']
    w(f'## 6. baseline 대비 (baseline {bm["val_acc"]:.2f} ±{bm["val_acc_sd"]:.2f}% @ {spk(bm["spikes"])}, n={nb})')
    w('')
    w('| 그룹 | n | val acc − baseline (%p) | 95% CI | p | spikes vs baseline | 95% CI |')
    w('|---|---|---|---|---|---|---|')
    for x in [res['vs_base']['nofinal'], res['vs_base']['ours_win8']] + res['vs_base']['ours_by_rho']:
        a = x['acc']
        w(f'| {x["label"]} | {x["n"]} | {a["diff"]:+.3f} | [{a["lo"]:+.3f}, {a["hi"]:+.3f}] | {fmt_p(a["p"])} | '
          f'{x["spk_pct"]:+.1f}% | [{x["spk_lo"]:+.1f}, {x["spk_hi"]:+.1f}] |')
    w('')
    w(f'- `- final_step` ({vn_["spk_pct"]:+.1f}%) 과 같은 스파이크의 `ours_inter_ch` ({vo_["spk_pct"]:+.1f}%) 는 둘 다 '
      'baseline 과 정확도가 0 과 구별되지 않는다 (각 p≥0.10). 같은 표에서 ours 5.5e-3 (−47%)·6e-3 (−51%) 은 손실이 유의하다 '
      '(p=0.03, 0.006; 7e-3 은 시드 산포가 커서 p=0.24) — 두 팔 모두 −40% 대가 무손실 경계 근처다.')
    w('- 두 팔의 baseline 대비 차 (−0.065 vs −0.118) 의 차이가 곧 1절의 +0.05%p 이고, 각자 CI 폭 (±0.13~0.15%p) 안이다.')
    w('')

    # ---------------- 한계
    w('## [LIMITATION]')
    w('')
    w(f'1. `- final_step` 은 **한 ρ (2.2e-3), n={nn}** 이다. 곡선이 아니라 한 점이라 "다른 스파이크 수준에서도 같은가" 는 말할 수 없다. '
      '진행 중인 ρ 1e-3~2.8e-3 런 (n=1~2 씩) 이 들어와도 점마다 n≤2 라 개별 점으로는 판단 불가, 곡선 ANCOVA 로 합쳐야 한다.')
    dd_ = anc['lin_srv']['coef']['d']
    w(f'2. 이 분석은 "차이가 없다" 를 증명하지 않는다. 같은 스파이크 오프셋의 95% CI 는 [{dd_["lo"]:+.2f}, {dd_["hi"]:+.2f}]%p — '
      '이보다 큰 손실·이득만 배제된다. 그 안의 우위·열위 (관측 +0.05%p 포함) 는 열려 있다.')
    w('3. best 에폭 기준은 310 에폭 중 최대값을 고르는 선택이라 시드 잡음을 위쪽으로 끌어올린다 (두 팔에 같게 작용하지만 '
      '팔마다 n 이 다르면 최대값의 기대치도 미세하게 다르다). 사용자 지시로 다른 기준 (마지막 N 에폭 평균 등) 은 주 결과로 쓰지 않았다.')
    w('4. spikes 는 원장의 s_count (이미지당, eval) 원값이다. s_count 와 뉴런별 npz 총합의 11% 불일치 (09-15 보류) 는 '
      '두 팔에 같은 정의를 쓰므로 비율 비교에는 영향이 적다고 보지만 해소되지 않았다.')
    w('5. 같은 ρ 에서 final_step 을 끄면 규제가 약 2배 세다 (2.2e-3 → 285K ≈ inter 4.5~5e-3). 이 비교는 같은 **스파이크** '
      '비교이지 같은 **설정 강도** 비교가 아니다. "final_step 이 같은 ρ 에서 덜 줄인다" 는 별개 사실이다.')
    w('6. 서버 효과는 baseline 각 n=3, ours 의 혼합 ρ 에서만 식별된다. 두 서버의 코드 차이 (세종 db3ea7f) 는 '
      'R19 경로 동일로 기록돼 있으나 (10-14 결정), 여기서 확인하지는 않았다.')
    w('7. S30/S1<0.19 배제 4런 (ours ρ 1.6e-2~5e-2) 은 원장 판정을 그대로 썼다. 전부 140K 이하·95.8% 이하라 이 분석의 '
      '창·범위·지배 판정 어디에도 들지 않는다.')
    open(OUT_MD, 'w').write('\n'.join(L) + '\n')


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items() if k not in ('cov', 'beta')}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inv', default=DEF_INV)
    ap.add_argument('--cache23', default=DEF_CACHE)
    ap.add_argument('--refetch', action='store_true')
    a = ap.parse_args()
    runs, dup, inv = build_runs(a.inv, a.cache23, a.refetch)
    res, base, nof, ours = analyze(runs)
    out = {'generated': '2026-10-14', 'definition': 'best val acc over 310 epochs (first max); spikes = s_count at that epoch',
           'inventory': a.inv, 'cache23': a.cache23,
           'runs': [{k: v for k, v in r.items()} for r in runs], 'results': res}
    json.dump(jsonable(out), open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)
    write_md(runs, res, base, nof, ours, dup, a.inv, a.cache23)
    print(f'wrote {OUT_JSON}\nwrote {OUT_MD}')


if __name__ == '__main__':
    main()
