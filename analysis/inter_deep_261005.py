"""ours_inter_ch 채택 근거와 '안 된 시도'의 통계·기전 재분석 (26-10-05).

주간 정리(docs/paper/weekly-261005-inter-ko.md) 4·5장의 '표 눈 비교'를 대체한다.
기존 분석 파일은 고치지 않는다. 산출물:
    analysis/inter_deep_261005.json   (그림용 데이터 포함)
    analysis/inter_deep_261005.md     (한국어 보고, 결론 먼저)

입력 (모두 읽기만):
  원장      scratchpad/ledger/inventory.json  (310 에폭 완주·seed<900·배제 아님 만 씀)
  에폭 로그  canus  /media/hdd1/kyccj/EIP/paper/<tag>/train.log  (없으면 _paper/<tag>/train.log)
            sejong /srv2/kyccj/EIP/paper/<tag>/train.log  -> scratch 로 '에폭 요약 줄만' 복사
            (deep1005/epochs138.txt, epochs138b.txt, logs23/epochs23.txt)
  층별 CSV  reg_detail.csv / reg_neuron_detail.csv (canus HDD, sejong 은 deep1005/logs23/<tag>/)
  npz      _npz/<tag>.npz (기존 82개) + deep1005/npz/<tag>.npz (이번에 CPU 로 새로 뽑은 12개)
  제거측정  scratchpad/prune_out/*.json (09-26) + deep1005/prune/*.json (이번, CPU)

통계 규약
  * 정확도는 원장의 val acc (= 베스트 에폭, CIFAR 에서는 val=test 1만 장). 스파이크는 그 에폭의 s_count.
  * 시드 SD 는 같은 (팔, ρ) 묶음 안 편차를 묶음 전체에서 합동(pooled)한 값.
  * 부트스트랩 2000회, seed 고정(20261005). 두 종류:
      seed      : ρ 점 안에서 시드를 복원추출 (n=1 점은 변동 0 -> 과소추정)
      pooled    : 각 런 = ρ 점 평균 + 합동 잔차에서 뽑은 잔차 (n=1 점에도 시드 산포를 준다) — 주 결과
  * 곡선 = 팔마다 ρ 점 평균을 ln(스파이크) 축에서 단조(isotonic, 스파이크가 늘면 정확도 비감소)로 맞춘 뒤 선형 보간.
    범위 밖은 외삽하지 않는다.

쓰는 법:  /home/kyccj/anaconda3/envs/venv_1/bin/python analysis/inter_deep_261005.py
"""
import json, os, re, glob, math
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
D = S + '/deep1005'
INV = S + '/ledger/inventory.json'
HDD = '/media/hdd1/kyccj/EIP/paper'
NPZ_DIRS = [os.path.join(ROOT, '_npz'), D + '/npz']
PRUNE_DIRS = [S + '/prune_out', D + '/prune']
OUT_JSON = os.path.join(ROOT, 'analysis', 'inter_deep_261005.json')
OUT_MD = os.path.join(ROOT, 'analysis', 'inter_deep_261005.md')
NB = 2000
RNG = np.random.default_rng(20261005)

NAME = {'ours_ch': 'ours_inter_ch', 'prop': 'ours_intra_ch', 'ours_w1': 'ours_intra_ch (w¹)',
        'ours_layer_w1': 'ours_layer (w¹)', 'ours_layer': 'ours_layer (w²)', 'l2': 'plain L2',
        'bpsr': 'BPSR', 'base': 'baseline'}
INTER = 'ours_inter_ch'
MAIN = ['ours_ch', 'prop', 'ours_w1', 'ours_layer_w1', 'l2', 'bpsr']          # 곡선 비교 팔
EXTRA = ['ours_layer']                                                       # 참고(주간 정리에 없음)
STAGES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc']
SKIP_L = {'n_in', 'predictions_n'}


def f2(x, d=2):
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def fk(x):
    return '–' if x is None or not np.isfinite(x) else f'{x / 1e3:,.0f}K'


def fi(x):
    return '–' if x is None or not np.isfinite(x) else f'{x:,.0f}'


# ======================================================================================
# 데이터
# ======================================================================================
INVALL = json.load(open(INV))
RUNS = INVALL['runs']
INVT = {r['tag']: r for r in RUNS}


def valid(r):
    return (r.get('status') == 'done' and r.get('epochs') == 310 and not r.get('exclude')
            and r['seed'] < 900 and r.get('val_acc') is not None and r.get('spikes'))


V = [r for r in RUNS if valid(r)]
EXCL = [r for r in RUNS if r['seed'] < 900 and r.get('key') in MAIN + EXTRA + ['base'] and not valid(r)
        and r.get('status') in ('done', 'excluded', 'stale')]


def runs(ds, key, knob=None):
    return [r for r in V if r['ds'] == ds and r['key'] == key and (knob is None or r['knob'] == knob)]


def groups(ds, key):
    g = defaultdict(list)
    for r in runs(ds, key):
        g[r['knob']].append(r)
    return g


# ---- 에폭 로그 -------------------------------------------------------------------------
def parse_epochs(fn):
    out, cur = {}, None
    if not os.path.exists(fn):
        return out
    for line in open(fn, errors='ignore'):
        line = line.rstrip('\n')
        if line.startswith('### '):
            cur = line[4:].strip(); out[cur] = []; continue
        if cur is None or 'val_acc:' not in line:
            continue
        d = {}
        for k, v in re.findall(r' - ([A-Za-z_\-0-9]+): ([-+0-9.eE]+|nan|inf)', line):
            try:
                d[k] = float(v)
            except ValueError:
                pass
        out[cur].append(d)
    return out


EPOCHS = {}
for fn, srv in ((D + '/epochs138.txt', '138'), (D + '/epochs138b.txt', '138'), (D + '/logs23/epochs23.txt', '23')):
    for k, v in parse_epochs(fn).items():
        if v:
            EPOCHS[(k, srv)] = v


def ep_of(r):
    return EPOCHS.get((r['tag'], r['server']))


def regcsv(r, name):
    p = (f"{HDD}/{r['tag']}/{name}" if r['server'] == '138' else f"{D}/logs23/{r['tag']}/{name}")
    if not os.path.exists(p):
        return None
    rows = defaultdict(dict)
    with open(p) as f:
        hdr = f.readline().strip().split(',')
        for line in f:
            x = line.strip().split(',')
            if len(x) != len(hdr):
                continue
            e = int(x[0]); rows[e][x[1]] = {h: float(v) for h, v in zip(hdr[2:], x[2:])}
    return rows


def best_epoch(r):
    ep = ep_of(r)
    va = [e['val_acc'] for e in ep]
    return int(np.argmax(va))          # 0-기준, 첫 최대 (ModelCheckpoint save_best_only 와 같다)


# ---- npz ------------------------------------------------------------------------------
def npz_path(tag):
    for d in NPZ_DIRS:
        p = os.path.join(d, tag + '.npz')
        if os.path.exists(p):
            return p
    return None


def stage_of(k):
    if k.startswith('conv1_'):
        return 'conv1'
    for s in ('conv2', 'conv3', 'conv4'):
        if k.startswith(s + '_'):
            return s
    return 'fc'


# ResNet19 fan-out (경계 패딩 반영). scratchpad/fanout.py 와 같은 규약:
# BN 접힘, identity shortcut 0 MAC, stage 입구 1x1 conv0 은 셈, GAP 는 fc1 에 접힘(스파이크 1개 = 256 시냅스).
def _reads(H, k, s, padding):
    if padding == 'same':
        out = -(-H // s); pad_total = max((out - 1) * s + k - H, 0); pb = pad_total // 2
    else:
        out = (H - k) // s + 1; pb = 0
    r = np.zeros(H)
    for i in range(H):
        lo = max(int(np.ceil((i + pb - k + 1) / s)), 0); hi = min((i + pb) // s, out - 1)
        if hi >= lo:
            r[i] = hi - lo + 1
    return r


def _cmap(H, k, s, pad, c):
    return np.outer(_reads(H, k, s, pad), _reads(H, k, s, pad)) * c


def build_fanout():
    M = {'conv1_conv_n': _cmap(32, 3, 1, 'same', 128) + _cmap(32, 1, 1, 'valid', 128)}
    st = [('conv2', 3, 128, 32), ('conv3', 3, 256, 16), ('conv4', 2, 512, 8)]
    for si, (nm, nb, C, sz) in enumerate(st):
        for b in range(1, nb + 1):
            M[f'{nm}_block{b}_conv1_n'] = _cmap(sz, 3, 1, 'same', C)
            if b < nb:
                M[f'{nm}_block{b}_out_n'] = _cmap(sz, 3, 1, 'same', C)
            elif si + 1 < len(st):
                _, _, nC, _ = st[si + 1]
                M[f'{nm}_block{b}_out_n'] = _cmap(sz, 3, 2, 'same', nC) + _cmap(sz, 1, 2, 'valid', nC)
            else:
                M[f'{nm}_block{b}_out_n'] = np.full((sz, sz), 256.0)
    M['fc1_n'] = np.full((256,), 10.0)
    return M


FAN = build_fanout()
FAN_MEAN = {k: float(v.mean()) for k, v in FAN.items()}

# prune JSON 의 가중치-층 사상 (params/MACs 오프라인 계산용; 같은 모델이므로 아무 JSON 이나)
_WL = None


def weight_layers():
    global _WL
    if _WL is None:
        for d in PRUNE_DIRS:
            for p in sorted(glob.glob(d + '/r19c10-*.json')):
                j = json.load(open(p))
                if 'weight_layers' in j:
                    _WL = j['weight_layers']; return _WL
    return _WL


def prune_cost(dead):
    """dead = {스파이킹 층: 죽은 채널 수}. prune_measure.py 와 같은 셈: 가중치 [.., cin, cout] 는
    src 채널이 죽거나 dst 채널이 죽으면 제거 (겹침 한 번). 개수만으로 정해진다."""
    pr = mr = pt = mt = 0
    for w in weight_layers():
        ks = w['ks']; cin, cout = ks[-2], ks[-1]; kk = int(np.prod(ks[:-2])) if len(ks) > 2 else 1
        di = dead.get(w['src'], 0); do = dead.get(w['dst'], 0)
        rem = (di * cout + do * cin - di * do) * kk
        tot = cin * cout * kk
        pr += rem + (do if w['bias'] else 0); pt += tot + (cout if w['bias'] else 0)
        mr += rem * w['hw']; mt += tot * w['hw']
    return 100 * pr / pt, 100 * mr / mt


NPZC = {}


def npz_metrics(tag):
    if tag in NPZC:
        return NPZC[tag]
    p = npz_path(tag)
    if p is None:
        NPZC[tag] = None; return None
    z = np.load(p)
    ns = float(z['n_samples'])
    tot = spk = sil = dead = nch = 0
    alive_pos = alive_pos_sil = 0
    ginis, act = [], []
    st = {s: dict(n=0, sil=0, ch=0, dead=0, spk=0.0, syn=0.0) for s in STAGES}
    layer = {}
    dead_by_layer = {}
    syn = 0.0
    conv_spk = 0.0; conv_alive = 0
    for k in z.files:
        if k.endswith('__fired') or k + '__fired' not in z.files or k in SKIP_L:
            continue
        c = z[k].astype(np.float64)
        n = c.size; s0 = int((c == 0).sum())
        tot += n; spk += c.sum(); sil += s0
        act.append(c[c > 0].ravel())
        sy = float((c.sum(-1) * FAN[k]).sum() / ns) if c.ndim == 3 else float((c * FAN[k]).sum() / ns)
        syn += sy
        s_ = st[stage_of(k)]
        s_['n'] += n; s_['sil'] += s0; s_['spk'] += c.sum() / ns; s_['syn'] += sy
        if c.ndim == 3:
            ch = c.reshape(-1, c.shape[-1]).sum(0); alive = ch > 0; d = int((~alive).sum())
            cc = c.reshape(-1, c.shape[-1])
            alive_pos += int(alive.sum()) * cc.shape[0]; alive_pos_sil += int((cc[:, alive] == 0).sum())
            g = np.sort(cc[:, alive], axis=0); m = g.shape[0]
            w_ = (2 * np.arange(1, m + 1) - m - 1)[:, None]
            gs = (w_ * g).sum(0) / (m * g.sum(0))
            ginis += list(gs)
            nc = c.shape[-1]
            conv_spk += c.sum() / ns; conv_alive += int(alive.sum())
        else:
            d = int((c == 0).sum()); nc = c.size
        dead += d; nch += nc; s_['ch'] += nc; s_['dead'] += d
        dead_by_layer[k] = d
        layer[k] = {'spk': c.sum() / ns, 'syn': sy, 'dead': d, 'ch': nc}
    a = np.sort(np.concatenate(act))[::-1]
    half = int(np.searchsorted(np.cumsum(a), a.sum() / 2) + 1)
    nfire = tot - sil
    pa, ma = prune_cost({k: v for k, v in dead_by_layer.items() if '_out_n' not in k})
    pb, mb = prune_cost(dead_by_layer)
    m = {'npz': p, 'spk': spk / ns, 'syn': syn, 'acc': float(z['acc']),
         'silent_pct': 100 * sil / tot, 'dead_ch': dead, 'channels': nch,
         'half_share_pct': 100 * half / nfire, 'alive_ch_gini': float(np.mean(ginis)),
         'alive_ch_silent_pct': 100 * alive_pos_sil / alive_pos,
         'stage': {s: {'silent_pct': 100 * v['sil'] / v['n'], 'dead': v['dead'], 'ch': v['ch'],
                       'spk': v['spk'], 'syn': v['syn']} for s, v in st.items()},
         'layer': layer, 'conv_spk': conv_spk, 'conv_alive_ch': conv_alive, 'params_pct_a': pa, 'macs_pct_a': ma, 'params_pct_b': pb, 'macs_pct_b': mb}
    NPZC[tag] = m
    return m


# ======================================================================================
# 통계 도구
# ======================================================================================
def ols(y, X):
    y = np.asarray(y, float); X = np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - np.linalg.matrix_rank(X)
    s2 = res @ res / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), res, dof, s2


def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for i, j in enumerate(o):
        run = max(run, (m - i) * ps[j]); adj[j] = min(1.0, run)
    return adj


def pooled_sd(rowsets, f=lambda r: r['val_acc']):
    ss = 0.0; df = 0
    for rs in rowsets:
        if len(rs) >= 2:
            v = np.array([f(r) for r in rs]); ss += ((v - v.mean()) ** 2).sum(); df += len(v) - 1
    return (math.sqrt(ss / df) if df else np.nan), df


def ancova(rows, ref=INTER, interaction=False, server=False, y=lambda r: r['val_acc'],
           x=lambda r: math.log(r['spikes'])):
    arms = []
    for r in rows:
        if r['_arm'] not in arms:
            arms.append(r['_arm'])
    others = [a for a in arms if a != ref]
    Y = np.array([y(r) for r in rows]); Xl = np.array([x(r) for r in rows])
    xm = Xl.mean()
    cols = [np.ones(len(rows)), Xl - xm]; names = ['a', 'b']
    for a in others:
        cols.append(np.array([1.0 if r['_arm'] == a else 0.0 for r in rows])); names.append(a)
    if server:
        cols.append(np.array([1.0 if r['server'] == '23' else 0.0 for r in rows])); names.append('server23')
    if interaction:
        for a in others:
            cols.append(np.array([(Xl[i] - xm) if r['_arm'] == a else 0.0 for i, r in enumerate(rows)])); names.append('bx:' + a)
    X = np.array(cols).T
    beta, se, res, dof, s2 = ols(Y, X)
    tcrit = stats.t.ppf(0.975, dof)
    co = {}
    for nm, b_, s_ in zip(names, beta, se):
        p_ = 2 * stats.t.sf(abs(b_ / s_), dof) if s_ > 0 else np.nan
        co[nm] = {'est': float(b_), 'se': float(s_), 'lo': float(b_ - tcrit * s_), 'hi': float(b_ + tcrit * s_), 'p': float(p_)}
    return {'coef': co, 'dof': int(dof), 'rss': float(res @ res), 'resid_sd': float(math.sqrt(s2)),
            'n': len(rows), 'n_arm': {a: sum(1 for r in rows if r['_arm'] == a) for a in arms},
            'others': others, 'resid': res.tolist()}


def tag_rows(ds, keys, lo, hi):
    out = []
    for k in keys:
        for r in runs(ds, k):
            if lo <= r['spikes'] <= hi:
                rr = dict(r); rr['_arm'] = NAME[k]; out.append(rr)
    return out


# ---- 곡선 (isotonic + 로그 보간) ----------------------------------------------------------
def pava(y, w):
    y = list(map(float, y)); w = list(map(float, w)); blocks = [[y[i], w[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks[i + 1]
            nw = a[1] + b[1]; blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / nw, nw, a[2] + b[2]]; del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, c in blocks:
        out += [v] * c
    return np.array(out)


def curve(pts):
    """pts = [(ln_spk, acc, n)] -> (xs, fitted) 오름차순."""
    pts = sorted(pts)
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts]); ws = np.array([p[2] for p in pts])
    return xs, pava(ys, ws)


def at(cv, x):
    xs, ys = cv
    if len(xs) == 0 or x < xs[0] - 1e-12 or x > xs[-1] + 1e-12:
        return np.nan
    return float(np.interp(x, xs, ys))


def group_points(ds, key, f_acc=lambda r: r['val_acc'], f_spk=lambda r: r['spikes']):
    pts = []
    for kn, rs in groups(ds, key).items():
        pts.append((float(np.mean([math.log(f_spk(r)) for r in rs])), float(np.mean([f_acc(r) for r in rs])), len(rs), kn,
                    [(math.log(f_spk(r)), f_acc(r)) for r in rs]))
    return sorted(pts)


def pooled_resid(ds, keys):
    ra, rs = [], []
    for k in keys:
        for kn, g in groups(ds, k).items():
            if len(g) >= 2:
                a = np.array([r['val_acc'] for r in g]); s = np.log([r['spikes'] for r in g]); c = math.sqrt(len(g) / (len(g) - 1))
                ra += list((a - a.mean()) * c); rs += list((s - s.mean()) * c)
    return np.array(ra), np.array(rs)


def boot_points(pts, kind, pool):
    """pts 의 각 ρ 점을 재표집해 [(ln_spk, acc, n)] 반환."""
    out = []
    for lx, ay, n, kn, raw in pts:
        if kind == 'seed':
            idx = RNG.integers(0, n, n); xs = np.array([raw[i][0] for i in idx]); ys = np.array([raw[i][1] for i in idx])
            out.append((xs.mean(), ys.mean(), n))
        else:
            ra, rs = pool; j = RNG.integers(0, len(ra), n)
            out.append((lx + rs[j].mean(), ay + ra[j].mean(), n))
    return out


# ======================================================================================
OUT = {'meta': {'generated': '2026-10-05', 'ledger': INVALL.get('generated'), 'n_boot': NB,
                'name_map': NAME, 'note': 'ours_intra_ch = prop(w², 원래 제안법). (w¹) 는 가중 1차.'}}
MD = []

# ---- 시드 SD ---------------------------------------------------------------------------
SD10, DF10 = pooled_sd([g for k in MAIN + EXTRA + ['base'] for g in groups('C10', k).values()])
SD100, DF100 = pooled_sd([g for k in MAIN + ['base'] for g in groups('C100', k).values()])
SDB10 = float(np.std([r['val_acc'] for r in runs('C10', 'base')], ddof=1))
SDB100 = float(np.std([r['val_acc'] for r in runs('C100', 'base')], ddof=1))
BASE10 = float(np.mean([r['val_acc'] for r in runs('C10', 'base')]))
BASE100 = float(np.mean([r['val_acc'] for r in runs('C100', 'base')]))
BASES10 = float(np.mean([r['spikes'] for r in runs('C10', 'base')]))
OUT['seed_sd'] = {'C10_pooled': SD10, 'C10_df': DF10, 'C100_pooled': SD100, 'C100_df': DF100,
                  'C10_base': SDB10, 'C10_base_n': len(runs('C10', 'base')), 'C100_base': SDB100,
                  'C100_base_n': len(runs('C100', 'base')), 'C10_base_mean': BASE10, 'C100_base_mean': BASE100,
                  'C10_base_spikes': BASES10}
POOL10 = pooled_resid('C10', MAIN + EXTRA + ['base'])
POOL100 = pooled_resid('C100', MAIN + ['base'])

# ======================================================================================
# A1. ANCOVA
# ======================================================================================
A = OUT['A'] = {}
A['A1'] = {}
for ds, keys, suf in (('C10', MAIN, ''), ('C10', MAIN + EXTRA, '+layer_w2'), ('C100', ['ours_ch', 'prop', 'ours_layer_w1', 'l2', 'bpsr'], '')):
    for lo, hi in ((250e3, 420e3), (180e3, 420e3), (320e3, 420e3)):
        rows = tag_rows(ds, keys, lo, hi)
        if not any(r['_arm'] == INTER for r in rows):
            continue
        m0 = ancova(rows)
        mi = ancova(rows, interaction=True)
        k_extra = len(m0['others'])
        F = ((m0['rss'] - mi['rss']) / k_extra) / (mi['rss'] / mi['dof']) if mi['dof'] > 0 else np.nan
        pF = float(stats.f.sf(F, k_extra, mi['dof'])) if np.isfinite(F) else np.nan
        # 모든 팔이 한 곡선인가 (팔 효과 전체 F)
        Y = np.array([r['val_acc'] for r in rows]); Xl = np.log([r['spikes'] for r in rows])
        _, _, res1, dof1, _ = ols(Y, np.array([np.ones(len(rows)), Xl]).T)
        Fa = ((res1 @ res1 - m0['rss']) / k_extra) / (m0['rss'] / m0['dof'])
        pFa = float(stats.f.sf(Fa, k_extra, m0['dof']))
        # lack of fit: 순수오차 = (팔, ρ) 묶음 안
        gg = defaultdict(list)
        for i, r in enumerate(rows):
            gg[(r['_arm'], r['knob'])].append(r['val_acc'])
        sspe = sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in gg.values()); dfpe = sum(len(v) - 1 for v in gg.values())
        dflof = m0['dof'] - dfpe
        Flof = ((m0['rss'] - sspe) / dflof) / (sspe / dfpe) if dflof > 0 and dfpe > 0 else np.nan
        pl = holm([m0['coef'][a]['p'] for a in m0['others']])
        for a, p_ in zip(m0['others'], pl):
            m0['coef'][a]['p_holm'] = float(p_)
        slopes = {}
        for a in [INTER] + m0['others']:
            sub = [r for r in rows if r['_arm'] == a]
            if len(set(r['knob'] for r in sub)) >= 2:
                b, se, *_r = ols([r['val_acc'] for r in sub], np.array([np.ones(len(sub)), np.log([r['spikes'] for r in sub])]).T)
                slopes[a] = {'b': float(b[1]), 'se': float(se[1]), 'n': len(sub)}
        # inter 대 나머지 전체 (단일 대비)
        D1 = ancova([dict(r, _arm=(INTER if r['_arm'] == INTER else 'others')) for r in rows])
        key = f'{ds}_{int(lo / 1e3)}-{int(hi / 1e3)}K{suf}'
        A['A1'][key] = {'inter_vs_pooled_others': D1['coef'].get('others'), 'common': {k: v for k, v in m0.items() if k != 'resid'}, 'F_interaction': float(F), 'p_interaction': pF,
                        'df_interaction': [k_extra, mi['dof']], 'F_arm': float(Fa), 'p_arm': pFa,
                        'pure_error_sd': float(math.sqrt(sspe / dfpe)) if dfpe else None, 'df_pe': dfpe,
                        'F_lof': float(Flof) if np.isfinite(Flof) else None,
                        'p_lof': float(stats.f.sf(Flof, dflof, dfpe)) if np.isfinite(Flof) else None,
                        'slopes': slopes,
                        'points': [{'arm': r['_arm'], 'knob': r['knob'], 'seed': r['seed'], 'server': r['server'],
                                    'val_acc': r['val_acc'], 'spikes': r['spikes']} for r in rows]}


# 검정력 (C10): 지금 데이터에서 inter 오프셋 SE 와, 0.10%p 를 80% 로 볼 때 필요한 배수
pw = {}
for key in ('C10_250-420K', 'C10_180-420K'):
    co = A['A1'][key]['common']['coef']
    for a in A['A1'][key]['common']['others']:
        se = co[a]['se']
        z = 0.10 / se
        pw[f'{key}|{a}'] = {'se': se, 'power_0.10': float(stats.norm.sf(1.96 - z) + stats.norm.cdf(-1.96 - z)),
                            'n_mult_for_80pct': float(((1.96 + 0.8416) * se / 0.10) ** 2)}
A['power_C10'] = pw

# ======================================================================================
# A2~A4. 곡선, 부트스트랩, 파레토, 하이퍼볼륨, 교차
# ======================================================================================
TARGETS = [340e3, 300e3, 240e3, 200e3]


def curves_for(ds, keys):
    return {NAME[k]: group_points(ds, k) for k in keys}


def hv(pts_sorted, lo=200e3, hi=450e3, ref=None, mode='stair'):
    """pts: [(ln_spk, acc, n)] 정렬. stair = 실제 점 계단, iso = 단조 보간 곡선 (위쪽은 마지막 점 유지).
    반환 = [lo, hi] 구간에서 (예산 x 로 얻을 수 있는 최고 정확도 − ref)⁺ 의 평균 (%p)."""
    grid = HV_GRID
    if mode == 'stair':
        pts_sorted = sorted(pts_sorted)
        xs = np.exp([p[0] for p in pts_sorted]); ys = np.maximum.accumulate(np.array([p[1] for p in pts_sorted]))
        i = np.searchsorted(xs, grid, side='right') - 1
        A_ = np.where(i >= 0, ys[np.clip(i, 0, None)], -np.inf)
    else:
        cv = curve([(p[0], p[1], p[2]) for p in pts_sorted]); xs = np.exp(cv[0]); ys = cv[1]
        A_ = np.where(grid >= xs[0], np.interp(grid, xs, ys), -np.inf)
    e = np.clip(A_ - ref, 0, None)
    return float(np.trapz(e, grid) / (hi - lo))      # 평균 초과 정확도 (%p)


HV_GRID = np.linspace(200e3, 450e3, 501)


def crossing(cv_a, cv_b):
    """a − b 가 높은 스파이크에서 + 이다가 낮은 쪽에서 − 로 바뀌는 첫 지점 (스파이크 수). 없으면 nan."""
    lo = max(cv_a[0][0], cv_b[0][0]); hi = min(cv_a[0][-1], cv_b[0][-1])
    if hi <= lo:
        return np.nan, 'no-overlap'
    g = np.linspace(hi, lo, 400)
    d = np.interp(g, *cv_a) - np.interp(g, *cv_b)
    if not (d > 0).any():
        return np.nan, 'a-never-above'
    i0 = int(np.argmax(d > 0))
    neg = np.where(d[i0 + 1:] < 0)[0]
    if len(neg) == 0:
        return np.nan, 'a-stays-above'
    i = i0 + 1 + neg[0]
    x = g[i - 1] + (g[i] - g[i - 1]) * d[i - 1] / (d[i - 1] - d[i])
    return float(math.exp(x)), 'cross'


for ds, keys, pool, base_acc in (('C10', MAIN, POOL10, BASE10),
                                 ('C100', ['ours_ch', 'prop', 'ours_layer_w1', 'l2', 'bpsr'], POOL100, BASE100)):
    P = curves_for(ds, keys)
    P = {a: p for a, p in P.items() if p}
    CV = {a: curve([(p[0], p[1], p[2]) for p in pts]) for a, pts in P.items()}
    others = [a for a in P if a != INTER]
    res = {'curves': {a: {'points': [{'knob': p[3], 'spikes': math.exp(p[0]), 'acc': p[1], 'n': p[2]} for p in pts],
                          'iso_spikes': np.exp(CV[a][0]).tolist(), 'iso_acc': CV[a][1].tolist()} for a, pts in P.items()}}
    # 점추정
    est = {t: {a: at(CV[a], math.log(t)) for a in P} for t in TARGETS}
    ref = base_acc - 0.3
    hv0 = {m: {a: hv(P[a], ref=ref, mode=m) for a in P} for m in ('stair', 'iso')}
    cr0 = {a: crossing(CV[INTER], CV[a]) for a in others}
    # 부트스트랩
    B = {kind: {'diff': {t: {a: [] for a in others} for t in TARGETS}, 'hv': {m: {a: [] for a in P} for m in ('stair', 'iso')},
                'cross': {a: [] for a in others}, 'level': {t: {a: [] for a in P} for t in TARGETS}} for kind in ('seed', 'pooled')}
    for kind in ('seed', 'pooled'):
        for _ in range(NB):
            bp = {a: boot_points(P[a], kind, pool) for a in P}
            bc = {a: curve(bp[a]) for a in P}
            for t in TARGETS:
                for a in P:
                    B[kind]['level'][t][a].append(at(bc[a], math.log(t)))
                for a in others:
                    B[kind]['diff'][t][a].append(at(bc[INTER], math.log(t)) - at(bc[a], math.log(t)))
            for m in ('stair', 'iso'):
                for a in P:
                    B[kind]['hv'][m][a].append(hv(sorted(bp[a]), ref=ref, mode=m))
            for a in others:
                B[kind]['cross'][a].append(crossing(bc[INTER], bc[a])[0])

    def summ(v, point):
        v = np.array(v, float); v = v[np.isfinite(v)]
        if len(v) < 50 or point is None or not np.isfinite(point):
            return None
        return {'est': float(point), 'lo': float(np.percentile(v, 2.5)), 'hi': float(np.percentile(v, 97.5)),
                'p_gt0': float((v > 0).mean()), 'sd': float(v.std())}
    res['at_spikes'] = {}
    for t in TARGETS:
        res['at_spikes'][str(int(t))] = {
            'level': {a: {'est': est[t][a], 'n_pts_bracket': None,
                          'pooled_sd': (float(np.nanstd(B['pooled']['level'][t][a])) if np.isfinite(est[t][a]) else None)} for a in P},
            'diff_inter_minus': {a: {kind: summ(B[kind]['diff'][t][a], est[t][INTER] - est[t][a]) for kind in ('seed', 'pooled')}
                                 for a in others}}
    res['hv_ref_acc'] = ref
    res['hv'] = {m: {a: {'est': hv0[m][a],
                         'pooled': summ(B['pooled']['hv'][m][a], hv0[m][a]),
                         'seed': summ(B['seed']['hv'][m][a], hv0[m][a])} for a in P} for m in ('stair', 'iso')}
    res['hv_diff_inter_minus'] = {m: {a: summ(np.array(B['pooled']['hv'][m][INTER]) - np.array(B['pooled']['hv'][m][a]),
                                              hv0[m][INTER] - hv0[m][a]) for a in others} for m in ('stair', 'iso')}
    res['crossing'] = {}
    for a in others:
        v = np.array(B['pooled']['cross'][a], float)
        fin = v[np.isfinite(v)]
        res['crossing'][a] = {'est': cr0[a][0], 'status': cr0[a][1], 'frac_cross': float(np.isfinite(v).mean()),
                              'median': float(np.median(fin)) if len(fin) else None,
                              'lo': float(np.percentile(fin, 2.5)) if len(fin) > 20 else None,
                              'hi': float(np.percentile(fin, 97.5)) if len(fin) > 20 else None}
    # 파레토 (ρ 점 평균 + baseline, 그리고 개별 런)
    allp = []
    for a, pts in P.items():
        for p in pts:
            allp.append({'arm': a, 'knob': p[3], 'spikes': math.exp(p[0]), 'acc': p[1], 'n': p[2]})
    bs = runs(ds, 'base')
    allp.append({'arm': 'baseline', 'knob': '-', 'spikes': float(np.mean([r['spikes'] for r in bs])),
                 'acc': float(np.mean([r['val_acc'] for r in bs])), 'n': len(bs)})

    def front(pp):
        f = []
        for p in pp:
            dom = any((q['spikes'] <= p['spikes'] and q['acc'] >= p['acc']) and (q['spikes'] < p['spikes'] or q['acc'] > p['acc']) for q in pp)
            if not dom:
                f.append(p)
        return sorted(f, key=lambda p: p['spikes'])
    res['pareto_means'] = front(allp)
    indiv = [{'arm': NAME[r['key']], 'knob': r['knob'], 'seed': r['seed'], 'spikes': r['spikes'], 'acc': r['val_acc'], 'n': 1}
             for k in keys + ['base'] for r in runs(ds, k)]
    res['pareto_runs'] = front(indiv)
    # 부트스트랩: 대역별 '전선 점유' 확률 (ρ 점 평균 기준)
    bands = [(180e3, 250e3), (250e3, 320e3), (320e3, 420e3)]
    occ = {f'{int(l/1e3)}-{int(h/1e3)}K': defaultdict(int) for l, h in bands}
    for _ in range(NB):
        pp = []
        for a, pts in P.items():
            for (lx, ay, n), p in zip(boot_points(pts, 'pooled', pool), pts):
                pp.append({'arm': a, 'spikes': math.exp(lx), 'acc': ay})
        fr = front(pp)
        for l, h in bands:
            seen = {p['arm'] for p in fr if l <= p['spikes'] < h}
            for a in seen:
                occ[f'{int(l/1e3)}-{int(h/1e3)}K'][a] += 1
    res['pareto_occupancy_pooled'] = {b: {a: v / NB for a, v in d.items()} for b, d in occ.items()}
    A[f'curves_{ds}'] = res

# C100 검정력: 두 표본 (σ=SD100), δ=0.15, 80%, α=0.05
z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
A['power_C100'] = {'sd': SD100, 'sd_base': SDB100, 'delta': 0.15,
                   'n_per_arm_two_sample': float(2 * (z * SD100 / 0.15) ** 2),
                   'n_per_arm_two_sample_base_sd': float(2 * (z * SDB100 / 0.15) ** 2),
                   'n_per_arm_delta_0.22': float(2 * (z * SD100 / 0.22) ** 2),
                   'note': 'ANCOVA 로 여러 ρ 점을 묶어도 팔당 총 런 수가 이 값에 가까워야 한다 (기울기 추정 비용 때문에 조금 더).'}

# ======================================================================================
# B. 층·단계별 스파이크 배분, SynOps
# ======================================================================================
Bo = OUT['B'] = {}
BAND_SETS = {
    '~340K': [('ours_ch', '3e-3'), ('ours_layer_w1', '1e-3'), ('ours_w1', '1e-3'), ('l2', '1e-7'), ('bpsr', '1e-10')],
    '~240K': [('ours_ch', '6e-3'), ('ours_layer_w1', '2e-3'), ('ours_w1', '2e-3'), ('prop', '3e-3'), ('l2', '2.5e-7'), ('bpsr', '3e-10')],
    '~200K': [('ours_ch', '8e-3'), ('ours_layer_w1', '3e-3'), ('ours_w1', '2.5e-3'), ('prop', '4e-3'), ('l2', '5e-7'), ('bpsr', '6e-10')],
}


def csv_layers(r):
    rd = regcsv(r, 'reg_detail.csv')
    if rd is None:
        return None
    e = best_epoch(r)
    if e not in rd:
        return None
    return {k: v['spike_count'] for k, v in rd[e].items() if k not in SKIP_L}


def alloc_from_layers(L):
    st = {s: 0.0 for s in STAGES}; sy = {s: 0.0 for s in STAGES}
    for k, v in L.items():
        st[stage_of(k)] += v; sy[stage_of(k)] += v * FAN_MEAN[k]
    return st, sy


base_csv = [csv_layers(r) for r in runs('C10', 'base')]
base_csv = [b for b in base_csv if b]
base_npz = [npz_metrics(r['tag']) for r in runs('C10', 'base')]
base_npz = [b for b in base_npz if b]


def agg_base(Ls, f):
    st = defaultdict(list); sy = defaultdict(list)
    for L in Ls:
        a, b = f(L)
        for s in STAGES:
            st[s].append(a[s]); sy[s].append(b[s])
    return {s: float(np.mean(st[s])) for s in STAGES}, {s: float(np.mean(sy[s])) for s in STAGES}


B0c = agg_base(base_csv, alloc_from_layers)
B0n = agg_base(base_npz, lambda m: ({s: m['stage'][s]['spk'] for s in STAGES}, {s: m['stage'][s]['syn'] for s in STAGES}))
Bo['baseline'] = {'csv': {'n': len(base_csv), 'stage_spk': B0c[0], 'stage_syn': B0c[1]},
                  'npz': {'n': len(base_npz), 'stage_spk': B0n[0], 'stage_syn': B0n[1]},
                  'fanout_mean': FAN_MEAN}
for band, items in BAND_SETS.items():
    bo = Bo[band] = {}
    for key, kn in items:
        rs = runs('C10', key, kn)
        for proto in ('csv', 'npz'):
            per = []
            for r in rs:
                if proto == 'csv':
                    L = csv_layers(r)
                    if not L:
                        continue
                    st, sy = alloc_from_layers(L)
                else:
                    m = npz_metrics(r['tag'])
                    if not m:
                        continue
                    st = {s: m['stage'][s]['spk'] for s in STAGES}; sy = {s: m['stage'][s]['syn'] for s in STAGES}
                per.append((r, st, sy))
            if not per:
                continue
            b_st, b_sy = (B0c if proto == 'csv' else B0n)
            tb = sum(b_st.values()); ty = sum(b_sy.values())
            rec = {'n': len(per), 'seeds': [p[0]['seed'] for p in per], 'servers': [p[0]['server'] for p in per],
                   'val_acc': float(np.mean([p[0]['val_acc'] for p in per])), 'spikes_ledger': float(np.mean([p[0]['spikes'] for p in per]))}
            spk_red = [100 * (1 - sum(p[1].values()) / tb) for p in per]
            syn_red = [100 * (1 - sum(p[2].values()) / ty) for p in per]
            rec['spk_red_pct'] = (float(np.mean(spk_red)), float(np.std(spk_red, ddof=1)) if len(per) > 1 else None)
            rec['syn_red_pct'] = (float(np.mean(syn_red)), float(np.std(syn_red, ddof=1)) if len(per) > 1 else None)
            rec['syn_per_spk'] = float(np.mean([sum(p[2].values()) / sum(p[1].values()) for p in per]))
            rec['stage_spk'] = {s: float(np.mean([p[1][s] for p in per])) for s in STAGES}
            rec['stage_red_share_pct'] = {}
            rec['stage_red_pct_of_base'] = {}
            rec['stage_syn_red_share_pct'] = {}
            for s in STAGES:
                d = [b_st[s] - p[1][s] for p in per]; dt = [tb - sum(p[1].values()) for p in per]
                rec['stage_red_share_pct'][s] = float(np.mean([100 * a / b for a, b in zip(d, dt)]))
                rec['stage_red_pct_of_base'][s] = float(np.mean([100 * (1 - p[1][s] / b_st[s]) for p in per]))
                dy = [b_sy[s] - p[2][s] for p in per]; dyt = [ty - sum(p[2].values()) for p in per]
                rec['stage_syn_red_share_pct'][s] = float(np.mean([100 * a / b for a, b in zip(dy, dyt)]))
            rec['_syn_red_list'] = syn_red; rec['_spk_red_list'] = spk_red
            rec['_synps_list'] = [sum(p[2].values()) / sum(p[1].values()) for p in per]
            bo.setdefault(NAME[key], {})[proto] = rec
# SynOps/스파이크 를 같은 대역에서 inter vs 다른 팔 (Welch, csv)
Bo['synps_tests'] = {}
for band in BAND_SETS:
    arms = Bo[band]
    if INTER not in arms or 'csv' not in arms[INTER]:
        continue
    xi = arms[INTER]['csv']['_synps_list']
    for a, d in arms.items():
        if a == INTER or 'csv' not in d:
            continue
        y = d['csv']['_synps_list']
        p = float(stats.ttest_ind(xi, y, equal_var=False).pvalue) if len(xi) >= 2 and len(y) >= 2 else None
        Bo['synps_tests'][f'{band}|{a}'] = {'inter': float(np.mean(xi)), 'other': float(np.mean(y)),
                                            'diff_pct': 100 * (np.mean(xi) / np.mean(y) - 1), 'p': p, 'n': [len(xi), len(y)]}
# SynOps 축 ANCOVA (csv SynOps, 250~420K): 축을 바꿔도 inter 오프셋이 같은가
syn_rows = []
for r in tag_rows('C10', MAIN, 250e3, 420e3):
    L = csv_layers(r)
    if L:
        r['_syn'] = sum(v * FAN_MEAN[k] for k, v in L.items()); r['_csvspk'] = sum(L.values()); syn_rows.append(r)
Bo['ancova_synops_250-420K'] = {k: v for k, v in ancova(syn_rows, x=lambda r: math.log(r['_syn'])).items() if k != 'resid'}
Bo['ancova_csvspk_250-420K'] = {k: v for k, v in ancova(syn_rows, x=lambda r: math.log(r['_csvspk'])).items() if k != 'resid'}

# ======================================================================================
# C. 기전: 정확도 잔차 vs 분포 지표
# ======================================================================================
Co = OUT['C'] = {}
MET = [('dead_ch', '죽은 채널'), ('half_share_pct', '50% 집중도 %'), ('silent_pct', '침묵 %'),
       ('alive_ch_gini', '산 채널 안 지니'), ('conv2_silent', 'conv2 침묵 %'), ('conv3_silent', 'conv3 침묵 %'),
       ('conv2_dead', 'conv2 죽은 채널'), ('conv3_dead', 'conv3 죽은 채널')]


def metv(m, k):
    if k.startswith('conv') and '_' in k:
        s, f = k.split('_'); return m['stage'][s]['silent_pct' if f == 'silent' else 'dead']
    return m[k]


for lo, hi in ((250e3, 420e3), (180e3, 420e3)):
    rows_all = tag_rows('C10', MAIN, lo, hi)
    Y = np.array([r['val_acc'] for r in rows_all]); Xl = np.log([r['spikes'] for r in rows_all])
    bb, *_ = ols(Y, np.array([np.ones(len(Y)), Xl]).T)
    rows = []
    for r in rows_all:
        m = npz_metrics(r['tag'])
        if m:
            r = dict(r); r['_m'] = m; r['_res'] = r['val_acc'] - (bb[0] + bb[1] * math.log(r['spikes'])); rows.append(r)
    key = f'{int(lo / 1e3)}-{int(hi / 1e3)}K'
    co = Co[key] = {'n': len(rows), 'n_arm': {a: sum(1 for r in rows if r['_arm'] == a) for a in set(r['_arm'] for r in rows)},
                    'curve_no_arm': {'a': float(bb[0]), 'b': float(bb[1])}, 'metrics': {}}
    arms = sorted(set(r['_arm'] for r in rows))
    lx = np.log([r['spikes'] for r in rows]); res = np.array([r['_res'] for r in rows])
    Darm = np.array([[1.0 if r['_arm'] == a else 0.0 for a in arms[1:]] for r in rows])
    Dint = np.array([1.0 if r['_arm'] == INTER else 0.0 for r in rows])
    acc = np.array([r['val_acc'] for r in rows])
    for mk, _ in MET:
        mv = np.array([metv(r['_m'], mk) for r in rows], float)
        # 부분상관 (ln 스파이크 통제), 전체
        Xc = np.array([np.ones(len(rows)), lx]).T
        _, _, e_m, _, _ = ols(mv, Xc); _, _, e_y, _, _ = ols(acc, Xc)
        r1, p1 = stats.pearsonr(e_m, e_y)
        # 팔 안 (ln 스파이크 + 팔 통제)
        Xa = np.column_stack([np.ones(len(rows)), lx, Darm])
        _, _, e_m2, _, _ = ols(mv, Xa); _, _, e_y2, _, _ = ols(acc, Xa)
        r2, p2 = stats.pearsonr(e_m2, e_y2)
        df2 = len(rows) - Xa.shape[1] - 1
        # 회귀 기울기 (정확도 %p / 지표 단위), 팔 통제
        b3, se3, *_ = ols(acc, np.column_stack([Xa, mv]))
        # 매개: inter 더미 효과가 지표를 넣으면 얼마나 줄어드는가
        X0 = np.column_stack([np.ones(len(rows)), lx, Dint]); X1 = np.column_stack([X0, mv])
        c0 = ols(acc, X0)[0][2]; c1 = ols(acc, X1)[0][2]
        bs = []
        idx_by = defaultdict(list)
        for i, r in enumerate(rows):
            idx_by[r['_arm']].append(i)
        for _ in range(NB):
            ii = np.concatenate([RNG.choice(v, len(v)) for v in idx_by.values()])
            try:
                a0 = ols(acc[ii], X0[ii])[0][2]; a1 = ols(acc[ii], X1[ii])[0][2]
                bs.append((a0, a1))
            except Exception:
                pass
        bs = np.array(bs)
        prop = (bs[:, 0] - bs[:, 1]) / np.where(np.abs(bs[:, 0]) < 1e-9, np.nan, bs[:, 0])
        co['metrics'][mk] = {'partial_r_spk': float(r1), 'p_spk': float(p1), 'n': len(rows),
                             'partial_r_within_arm': float(r2), 'p_within_arm': float(p2), 'df_within': int(df2),
                             'slope_within_arm': float(b3[-1]), 'slope_se': float(se3[-1]),
                             'inter_effect_total': float(c0), 'inter_effect_direct': float(c1),
                             'prop_mediated': float((c0 - c1) / c0) if abs(c0) > 1e-9 else None,
                             'prop_mediated_ci': [float(np.nanpercentile(prop, 2.5)), float(np.nanpercentile(prop, 97.5))],
                             'diff_ci': [float(np.percentile(bs[:, 0] - bs[:, 1], 2.5)), float(np.percentile(bs[:, 0] - bs[:, 1], 97.5))]}
    co['rows'] = [{'tag': r['tag'], 'arm': r['_arm'], 'knob': r['knob'], 'seed': r['seed'], 'server': r['server'],
                   'val_acc': r['val_acc'], 'spikes': r['spikes'], 'resid': r['_res'],
                   **{mk: metv(r['_m'], mk) for mk, _ in MET}} for r in rows]

# ======================================================================================
# D. 저스파이크 붕괴: 죽은 채널 곡선
# ======================================================================================
Do = OUT['D'] = {'runs': [], 'fits': {}, 'at': {}}
DKEYS = ['ours_ch', 'prop', 'ours_w1', 'ours_layer_w1', 'ours_layer', 'l2', 'bpsr', 'base']
for k in DKEYS:
    for r in runs('C10', k):
        m = npz_metrics(r['tag'])
        if not m:
            continue
        Do['runs'].append({'tag': r['tag'], 'arm': NAME[k], 'knob': r['knob'], 'seed': r['seed'], 'server': r['server'],
                           'train_loss': r['train_loss'], 'train_acc': r['train_acc'], 'val_loss': r['val_loss'],
                           'val_acc': r['val_acc'], 'spikes': r['spikes'], 'test_spk': m['spk'], 'test_acc': m['acc'],
                           'dead_ch': m['dead_ch'], 'silent_pct': m['silent_pct'], 'half_share_pct': m['half_share_pct'],
                           'alive_ch_gini': m['alive_ch_gini'], 'alive_ch_silent_pct': m['alive_ch_silent_pct'],
                           'stage_dead': {s: m['stage'][s]['dead'] for s in STAGES},
                           'stage_spk': {s: m['stage'][s]['spk'] for s in STAGES},
                           'syn': m['syn'], 'macs_pct_b': m['macs_pct_b'], 'params_pct_b': m['params_pct_b'],
                           'macs_pct_a': m['macs_pct_a'], 'params_pct_a': m['params_pct_a'], 'npz': m['npz'],
                           'conv_spk': m['conv_spk'], 'conv_alive_ch': m['conv_alive_ch']})
# 분해: conv 스파이크 = 산 conv 채널 수 × 산 채널당 스파이크.  baseline 대비 ln 감소를 두 항으로 나눈다.
bD = [r for r in Do['runs'] if r['arm'] == 'baseline']
lnA0 = float(np.mean([math.log(r['conv_alive_ch']) for r in bD])); lnQ0 = float(np.mean([math.log(r['conv_spk'] / r['conv_alive_ch']) for r in bD]))
for r in Do['runs']:
    dA = math.log(r['conv_alive_ch']) - lnA0; dQ = math.log(r['conv_spk'] / r['conv_alive_ch']) - lnQ0
    r['d_ln_alive'] = dA; r['d_ln_per_alive'] = dQ
    r['share_by_channel_death'] = dA / (dA + dQ) if abs(dA + dQ) > 1e-6 else None
Do['decomp'] = {'baseline_conv_alive_ch': math.exp(lnA0), 'baseline_spk_per_alive_ch': math.exp(lnQ0), 'n_base': len(bD), 'by_arm': {}}
for a in sorted(set(r['arm'] for r in Do['runs'])):
    rr = [r for r in Do['runs'] if r['arm'] == a and a != 'baseline']
    if len(rr) < 3:
        continue
    x = np.array([r['d_ln_alive'] + r['d_ln_per_alive'] for r in rr]); yA = np.array([r['d_ln_alive'] for r in rr])
    b, se, *_ = ols(yA, np.column_stack([np.ones(len(rr)), x]))
    sh = [r['share_by_channel_death'] for r in rr if r['share_by_channel_death'] is not None]
    Do['decomp']['by_arm'][a] = {'n': len(rr), 'elasticity_alive': float(b[1]), 'se': float(se[1]),
                                 'share_mean': float(np.mean(sh)), 'share_sd': float(np.std(sh, ddof=1)) if len(sh) > 1 else None}
# 분절 회귀: dead = a + b_hi·(lnS − ln300K)⁺ + b_lo·(lnS − ln300K)⁻   (팔별)
K300 = math.log(300e3)
for a in sorted(set(r['arm'] for r in Do['runs'])):
    rr = [r for r in Do['runs'] if r['arm'] == a and r['arm'] != 'baseline']
    if len(rr) < 4:
        continue
    lx = np.log([r['spikes'] for r in rr]); y = np.array([r['dead_ch'] for r in rr], float)
    hi_ = np.clip(lx - K300, 0, None); lo_ = np.clip(lx - K300, None, 0)
    rec = {'n': len(rr), 'spk_range': [float(min(r['spikes'] for r in rr)), float(max(r['spikes'] for r in rr))]}
    if (hi_ > 0).sum() >= 2 and (lo_ < 0).sum() >= 2:
        b, se, res, dof, s2 = ols(y, np.column_stack([np.ones(len(rr)), hi_, lo_]))
        # 기울기 차 검정
        X = np.column_stack([np.ones(len(rr)), hi_, lo_]); cov = s2 * np.linalg.pinv(X.T @ X)
        dd = b[2] - b[1]; sed = math.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])
        rec.update({'b_above300': float(b[1]), 'se_above': float(se[1]), 'b_below300': float(b[2]), 'se_below': float(se[2]),
                    'diff': float(dd), 'se_diff': float(sed), 'p_diff': float(2 * stats.t.sf(abs(dd / sed), dof)) if dof > 0 else None,
                    'dof': int(dof)})
    b, se, *_ = ols(y, np.column_stack([np.ones(len(rr)), lx]))
    rec['b_linear'] = float(b[1]); rec['se_linear'] = float(se[1])
    Do['fits'][a] = rec
# 같은 스파이크에서의 죽은 채널 (ρ 점 평균의 로그 보간, 단조 강제 없음)
for t in (400e3, 340e3, 300e3, 240e3, 200e3, 160e3):
    Do['at'][str(int(t))] = {}
    for a in sorted(set(r['arm'] for r in Do['runs'])):
        g = defaultdict(list)
        for r in Do['runs']:
            if r['arm'] == a:
                g[r['knob']].append(r)
        pts = sorted((np.mean([math.log(r['spikes']) for r in v]), np.mean([r['dead_ch'] for r in v]),
                      {s: np.mean([r['stage_dead'][s] for r in v]) for s in STAGES}, np.mean([r['val_acc'] for r in v]), len(v))
                     for v in g.values())
        xs = [p[0] for p in pts]
        if len(pts) < 2 or not (xs[0] <= math.log(t) <= xs[-1]):
            continue
        Do['at'][str(int(t))][a] = {'dead_ch': float(np.interp(math.log(t), xs, [p[1] for p in pts])),
                                    'stage_dead': {s: float(np.interp(math.log(t), xs, [p[2][s] for p in pts])) for s in STAGES},
                                    'val_acc': float(np.interp(math.log(t), xs, [p[3] for p in pts]))}
# inter: 정확도와 죽은 채널의 동시성 (ρ 점 순서대로)
g = defaultdict(list)
for r in Do['runs']:
    if r['arm'] == INTER:
        g[r['knob']].append(r)
seq = sorted(({'knob': k, 'n': len(v), 'spikes': float(np.mean([r['spikes'] for r in v])),
               'val_acc': float(np.mean([r['val_acc'] for r in v])), 'dead_ch': float(np.mean([r['dead_ch'] for r in v])),
               'stage_dead': {s: float(np.mean([r['stage_dead'][s] for r in v])) for s in STAGES},
               'half_share_pct': float(np.mean([r['half_share_pct'] for r in v])),
               'alive_ch_silent_pct': float(np.mean([r['alive_ch_silent_pct'] for r in v])),
               'macs_pct_b': float(np.mean([r['macs_pct_b'] for r in v]))} for k, v in g.items()), key=lambda d: -d['spikes'])
Do['inter_sequence'] = seq
# 죽은 채널 1개당 스파이크 감소 (한계 효율): inter 연속 ρ 점 사이 Δspikes/Δdead
Do['inter_marginal'] = [{'from': a['knob'], 'to': b['knob'], 'd_spikes': b['spikes'] - a['spikes'], 'd_dead': b['dead_ch'] - a['dead_ch'],
                         'd_acc': b['val_acc'] - a['val_acc']} for a, b in zip(seq, seq[1:])]

# ======================================================================================
# E. 학습 동역학
# ======================================================================================
Eo = OUT['E'] = {'arms': {}}
EREP = [('ours_ch', '3e-3'), ('ours_w1', '1e-3'), ('ours_layer_w1', '1e-3'), ('l2', '1e-7'), ('bpsr', '1e-10'), ('base', '-'),
        ('ours_ch', '6e-3'), ('ours_ch', '8e-3'), ('prop', '3e-3'), ('ours_w1', '2e-3'), ('ours_layer_w1', '2e-3')]
CHK = [1, 5, 10, 20, 30, 60, 100, 150, 200, 250, 280, 310]
DLAY = {'conv1': ['conv1_conv_n'], 'conv2': [f'conv2_block{b}_{t}_n' for b in (1, 2, 3) for t in ('conv1', 'out')],
        'conv3': [f'conv3_block{b}_{t}_n' for b in (1, 2, 3) for t in ('conv1', 'out')],
        'conv4': [f'conv4_block{b}_{t}_n' for b in (1, 2) for t in ('conv1', 'out')], 'fc': ['fc1_n']}
for key, kn in EREP:
    rs = runs('C10', key, kn)
    per = []
    for r in rs:
        ep = ep_of(r)
        nd = regcsv(r, 'reg_neuron_detail.csv')
        rd = regcsv(r, 'reg_detail.csv')
        if not ep:
            continue
        tr = {'tag': r['tag'], 'seed': r['seed'], 'server': r['server'],
              'val_acc': [e.get('val_acc', np.nan) * 100 for e in ep], 's_count': [e.get('s_count', np.nan) for e in ep],
              'lam': [e.get('adp_lambda', np.nan) for e in ep], 'R': [e.get('reg_R', np.nan) for e in ep],
              'task_loss': [e.get('task_loss_clean', e.get('loss', np.nan)) for e in ep], 'lr': [e.get('lr', np.nan) for e in ep],
              'loss_ratio': [e.get('loss_ratio', np.nan) for e in ep]}
        if nd:
            for s, ls in DLAY.items():
                tr['dead_' + s] = [float(np.mean([nd[e][l]['dead_neuron_ratio'] for l in ls if l in nd.get(e, {})])) if e in nd else np.nan
                                   for e in range(len(ep))]
        if rd:
            for s, ls in DLAY.items():
                tr['spk_' + s] = [float(sum(rd[e][l]['spike_count'] for l in ls if l in rd.get(e, {}))) if e in rd else np.nan
                                  for e in range(len(ep))]
        per.append(tr)
    if not per:
        continue
    nm = f'{NAME[key]} {kn}'
    rec = {'key': key, 'knob': kn, 'n': len(per), 'seeds': [p['seed'] for p in per], 'servers': [p['server'] for p in per],
           'val_acc_final': float(np.mean([r['val_acc'] for r in rs])), 'spikes_final': float(np.mean([r['spikes'] for r in rs]))}
    series = {}
    for f in ['val_acc', 's_count', 'lam', 'R', 'task_loss', 'loss_ratio'] + ['dead_' + s for s in STAGES] + ['spk_' + s for s in STAGES]:
        arr = [p[f] for p in per if f in p]
        if not arr:
            continue
        arr = np.array([a[:310] for a in arr if len(a) >= 310], float)
        if len(arr) == 0:
            continue
        series[f] = {'mean': np.nanmean(arr, 0).tolist(), 'sd': (np.nanstd(arr, 0, ddof=1).tolist() if len(arr) > 1 else None)}
    rec['series'] = series
    rec['_per'] = per
    # 죽음 시점: conv1/conv2 침묵 비율이 (최종 − 초기) 변화의 90% 에 처음 도달한 에폭
    t90 = {}
    for s in STAGES:
        f = 'dead_' + s
        if f not in series:
            continue
        v = np.array(series[f]['mean']); v0 = v[0]; v1 = np.nanmean(v[-10:])
        if abs(v1 - v0) < 1e-6:
            continue
        frac = (v - v0) / (v1 - v0)
        t90[s] = int(np.argmax(frac >= 0.9) + 1)
        t50 = int(np.argmax(frac >= 0.5) + 1)
        t90[s + '_t50'] = t50
    rec['t_dead'] = t90
    Eo['arms'][nm] = rec
# inter 3e-3 대비 정확도 격차의 구간별 평균: 시드마다 구간 평균을 낸 뒤 팔 간 Welch (에폭끼리 상관이 있으므로)
WIN = [(1, 30), (31, 100), (101, 200), (201, 280), (281, 310), (301, 310)]
Eo['gap_windows'] = {}
ref = Eo['arms'].get('ours_inter_ch 3e-3')
for nm, rec in Eo['arms'].items():
    if nm == 'ours_inter_ch 3e-3' or ref is None or rec['knob'] not in ('1e-3', '1e-7', '1e-10', '-'):
        continue
    gw = {'n': [ref['n'], rec['n']]}
    for a, b in WIN:
        xa = [np.mean(p_['val_acc'][a - 1:b]) for p_ in ref['_per']]; xb = [np.mean(p_['val_acc'][a - 1:b]) for p_ in rec['_per']]
        sa = [np.mean(p_['s_count'][a - 1:b]) for p_ in ref['_per']]; sb = [np.mean(p_['s_count'][a - 1:b]) for p_ in rec['_per']]
        se = math.sqrt(np.var(xa, ddof=1) / len(xa) + np.var(xb, ddof=1) / len(xb)) if len(xa) > 1 and len(xb) > 1 else None
        gw[f'{a}-{b}'] = {'acc_gap': float(np.mean(xa) - np.mean(xb)), 'se': se,
                          'p': (float(stats.ttest_ind(xa, xb, equal_var=False).pvalue) if len(xa) > 1 and len(xb) > 1 else None),
                          'spk_ratio_minus1_pct': float(100 * (np.mean(sa) / np.mean(sb) - 1))}
    Eo['gap_windows'][nm] = gw
# 최종 정확도 순위가 언제 정해지는가: 에폭 e 의 '지금까지 최고 val acc' (체크포인트 규칙과 같다) 의 inter − 다른 팔
Eo['bestsofar_gap'] = {}
for nm, rec in Eo['arms'].items():
    if nm == 'ours_inter_ch 3e-3' or ref is None or rec['knob'] not in ('1e-3', '1e-7', '1e-10', '-'):
        continue
    ra = np.array([np.maximum.accumulate(p_['val_acc'][:310]) for p_ in ref['_per']])
    rb = np.array([np.maximum.accumulate(p_['val_acc'][:310]) for p_ in rec['_per']])
    Eo['bestsofar_gap'][nm] = {str(e): float(ra[:, e - 1].mean() - rb[:, e - 1].mean()) for e in (200, 250, 280, 290, 300, 305, 310)}
# 언제 최고점이 나왔나 (베스트 에폭 분포)
Eo['best_epoch'] = {nm: [int(np.argmax(p_['val_acc']) + 1) for p_ in rec['_per']] for nm, rec in Eo['arms'].items()}

# ======================================================================================
# F. 실제 제거 이득
# ======================================================================================
Fo = OUT['F'] = {'runs': [], 'offline_check': []}
for d in PRUNE_DIRS:
    for p in sorted(glob.glob(d + '/r19c10-*.json')):
        if os.path.basename(p).startswith('test_'):
            continue
        j = json.load(open(p)); tag = j['tag']; iv = INVT.get(tag, {})
        rr = j['res'].get('train10k') or list(j['res'].values())[0]
        rec = {'tag': tag, 'arm': NAME.get(iv.get('key'), iv.get('key')), 'knob': iv.get('knob'), 'seed': iv.get('seed'),
               'server': iv.get('server'), 'train_loss': iv.get('train_loss'), 'train_acc': iv.get('train_acc'),
               'val_loss': iv.get('val_loss'), 'val_acc': iv.get('val_acc'), 'spikes': iv.get('spikes'),
               'acc_before': j['acc_base'], 'test_dead_ch': j.get('test_dead_ch'), 'src': p}
        for sc in ('a', 'b'):
            x = rr[sc]
            rec[sc] = {'n_dead': x['n_dead'], 'acc': x['acc'], 'd_acc': x['acc'] - j['acc_base'], 'params_pct': x['params_pct'],
                       'macs_pct': x['macs_pct'], 'n_dead_fired_test': x['n_dead_fired_test'],
                       'test_spk_lost_frac': x.get('test_spk_lost_frac')}
            # 오프라인 계산 검증 (개수만으로 params/MACs 가 재현되는가)
            pp, mm = prune_cost(x['per_layer_dead']) if 'per_layer_dead' in x else (None, None)
            if pp is not None:
                Fo['offline_check'].append({'tag': tag, 'sc': sc, 'params_json': x['params_pct'], 'params_offline': pp,
                                            'macs_json': x['macs_pct'], 'macs_offline': mm})
        Fo['runs'].append(rec)
# npz(테스트 판정) 기반 오프라인 MACs: 같은 스파이크 대역 비교용 (모든 팔 n=3)
Fo['offline_npz'] = {}
for band, items in BAND_SETS.items():
    for key, kn in items:
        vals = [npz_metrics(r['tag']) for r in runs('C10', key, kn)]
        vals = [(r, v) for r, v in zip(runs('C10', key, kn), vals) if v]
        if not vals:
            continue
        Fo['offline_npz'][f'{band}|{NAME[key]} {kn}'] = {
            'n': len(vals), 'spikes': float(np.mean([r['spikes'] for r, _ in vals])),
            'val_acc': float(np.mean([r['val_acc'] for r, _ in vals])),
            'dead_ch': (float(np.mean([v['dead_ch'] for _, v in vals])), float(np.std([v['dead_ch'] for _, v in vals], ddof=1)) if len(vals) > 1 else None),
            'macs_pct_b': (float(np.mean([v['macs_pct_b'] for _, v in vals])), float(np.std([v['macs_pct_b'] for _, v in vals], ddof=1)) if len(vals) > 1 else None),
            'params_pct_b': (float(np.mean([v['params_pct_b'] for _, v in vals])), float(np.std([v['params_pct_b'] for _, v in vals], ddof=1)) if len(vals) > 1 else None),
            'macs_pct_a': (float(np.mean([v['macs_pct_a'] for _, v in vals])), None)}
bm = [npz_metrics(r['tag']) for r in runs('C10', 'base')]
bm = [b for b in bm if b]
Fo['offline_npz']['baseline'] = {'n': len(bm), 'dead_ch': (float(np.mean([b['dead_ch'] for b in bm])), float(np.std([b['dead_ch'] for b in bm], ddof=1))),
                                 'macs_pct_b': (float(np.mean([b['macs_pct_b'] for b in bm])), float(np.std([b['macs_pct_b'] for b in bm], ddof=1)))}
# 같은 스파이크에서의 MACs% (npz 전 런 ANCOVA, 250~420K & 180~420K)
for lo, hi in ((250e3, 420e3), (180e3, 420e3)):
    rr = []
    for r in Do['runs']:
        if lo <= r['spikes'] <= hi and r['arm'] in [NAME[k] for k in MAIN]:
            x = dict(r); x['_arm'] = r['arm']; rr.append(x)
    if any(r['_arm'] == INTER for r in rr):
        Fo[f'ancova_macs_{int(lo/1e3)}-{int(hi/1e3)}K'] = {k: v for k, v in ancova(rr, y=lambda r: r['macs_pct_b']).items() if k != 'resid'}

# --- 26-10-05 추가 (F 마무리): inter 실측 행 + '같은 스파이크에서 inter 가 MACs 를 더 줄인다' 통계 ---
# (F1) 실측(학습셋 1만 장 판정) 집합 = npz 테스트 판정 집합인가, 실측 acc_base 와 원장 val acc 차이
Fo['measured_vs_npz'] = []
for rec in Fo['runs']:
    m = npz_metrics(rec['tag'])
    Fo['measured_vs_npz'].append({'tag': rec['tag'], 'arm': rec['arm'], 'dead_b_meas': rec['b']['n_dead'], 'dead_test_json': rec['test_dead_ch'],
                                  'dead_npz': m['dead_ch'] if m else None,
                                  'macs_b_meas': rec['b']['macs_pct'], 'macs_b_npz': m['macs_pct_b'] if m else None,
                                  'acc_base_meas': rec['acc_before'], 'val_acc_ledger': rec['val_acc']})
_da = [rec[sc]['d_acc'] for rec in Fo['runs'] for sc in ('a', 'b')]
Fo['dacc_summary'] = {'n_runs': len(Fo['runs']), 'n_inter': sum(r['arm'] == INTER for r in Fo['runs']),
                      'max_abs_dacc': float(np.max(np.abs(_da))), 'n_nonzero': int(sum(abs(x) > 1e-9 for x in _da)),
                      'fired_test_total': int(sum(r['b']['n_dead_fired_test'] for r in Fo['runs'])),
                      'n_meas_eq_npz': sum(1 for x in Fo['measured_vs_npz'] if x['dead_npz'] == x['dead_b_meas'] == x['dead_test_json']),
                      'max_abs_accbase_minus_ledger': float(max(abs(x['acc_base_meas'] - x['val_acc_ledger']) for x in Fo['measured_vs_npz']))}


def _ols_contrast(rr, yk, x0s):
    """y = a + b·(lnS − ln x0) + c·others + d·others·(lnS − ln x0).  c = x0 에서의 (다른 팔 − inter). 기울기 차이 허용."""
    out = {}
    for x0 in x0s:
        X = np.array([[1.0, math.log(r['spikes'] / x0), float(r['_arm'] != INTER), float(r['_arm'] != INTER) * math.log(r['spikes'] / x0)] for r in rr])
        Y = np.array([r[yk] for r in rr])
        beta, se, res, dof, s2 = ols(Y, X)
        tc = stats.t.ppf(0.975, dof)
        out[f'{int(x0 / 1e3)}K'] = {'est': float(beta[2]), 'se': float(se[2]), 'lo': float(beta[2] - tc * se[2]), 'hi': float(beta[2] + tc * se[2]),
                                    'p': float(2 * stats.t.sf(abs(beta[2] / se[2]), dof)), 'dof': int(dof),
                                    'slope_inter': float(beta[1]), 'slope_others': float(beta[1] + beta[3])}
    return out


# (F2) inter vs 다른 팔 합동 (npz 테스트 판정, 전 런): 공통 기울기 ANCOVA + 기울기 차이 + 서버 보정 + 고정 스파이크에서의 대비
# 250~420K 에서 다른 팔은 317K 아래에 없으므로 280K 대비는 외삽이라 넣지 않는다.
Fo['pooled_macs'] = {}
for lo, hi, x0s in ((300e3, 420e3, (340e3,)), (250e3, 420e3, (340e3,)), (180e3, 420e3, (340e3, 250e3, 200e3))):
    rr = [dict(r, _arm=(INTER if r['arm'] == INTER else 'others')) for r in Do['runs']
          if lo <= r['spikes'] <= hi and r['arm'] in [NAME[k] for k in MAIN]]
    if not any(r['_arm'] == INTER for r in rr) or not any(r['_arm'] == 'others' for r in rr):
        continue
    out = {'n': len(rr), 'n_inter': sum(r['_arm'] == INTER for r in rr),
           'n_arm_detail': {a: sum(1 for r in Do['runs'] if lo <= r['spikes'] <= hi and r['arm'] == a) for a in [NAME[k] for k in MAIN]},
           'spk_range_inter': [min(r['spikes'] for r in rr if r['_arm'] == INTER), max(r['spikes'] for r in rr if r['_arm'] == INTER)],
           'spk_range_others': [min(r['spikes'] for r in rr if r['_arm'] != INTER), max(r['spikes'] for r in rr if r['_arm'] != INTER)]}
    for yk in ('macs_pct_b', 'macs_pct_a', 'params_pct_b', 'dead_ch'):
        m0 = ancova(rr, y=lambda r, yk=yk: r[yk]); mi = ancova(rr, y=lambda r, yk=yk: r[yk], interaction=True)
        ms = ancova(rr, y=lambda r, yk=yk: r[yk], server=True)
        out[yk] = {'others_minus_inter': m0['coef']['others'], 'slope': m0['coef']['b'], 'resid_sd': m0['resid_sd'], 'dof': m0['dof'],
                   'slope_diff': mi['coef']['bx:others'], 'with_server': ms['coef']['others'], 'server23': ms['coef']['server23'],
                   'at': _ols_contrast(rr, yk, x0s)}
    Fo['pooled_macs'][f'{int(lo/1e3)}-{int(hi/1e3)}K'] = out

# (F3) 실측 JSON 만 쓴 회귀 (Δacc 측정이 있는 런만). 스파이크 범위가 거의 안 겹친다 -> 참고용
_rr = [dict(r, _arm=(INTER if r['arm'] == INTER else 'others')) for r in Fo['runs']]
Fo['measured_ancova'] = {'spk_range_inter': [min(r['spikes'] for r in _rr if r['_arm'] == INTER), max(r['spikes'] for r in _rr if r['_arm'] == INTER)],
                         'spk_range_others': [min(r['spikes'] for r in _rr if r['_arm'] != INTER), max(r['spikes'] for r in _rr if r['_arm'] != INTER)]}
for sc in ('a', 'b'):
    m0 = ancova(_rr, y=lambda r, sc=sc: r[sc]['macs_pct'])
    Fo['measured_ancova'][sc] = {'others_minus_inter': m0['coef']['others'], 'slope': m0['coef']['b'], 'n': m0['n'], 'n_arm': m0['n_arm'], 'dof': m0['dof']}

# (F4) ~340K 대역: inter 3e-3 (실측 n=3, npz 값과 동일) − 각 팔 (npz) Welch. 스파이크 보정 없음 (inter 쪽 스파이크가 더 많다)
Fo['band340'] = {}
_ib = [r for r in Do['runs'] if r['arm'] == INTER and r['knob'] == '3e-3']
for _nm, _kn in [(NAME[k], kn) for k, kn in BAND_SETS['~340K'] if NAME[k] != INTER]:
    _ob = [r for r in Do['runs'] if r['arm'] == _nm and r['knob'] == _kn]
    rec = {'n_inter': len(_ib), 'n_other': len(_ob)}
    for yk in ('macs_pct_b', 'macs_pct_a', 'params_pct_b', 'spikes', 'val_acc'):
        a_ = np.array([r[yk] for r in _ib]); b_ = np.array([r[yk] for r in _ob]); d_ = float(a_.mean() - b_.mean())
        if len(a_) > 1 and len(b_) > 1:
            va, vb = a_.var(ddof=1) / len(a_), b_.var(ddof=1) / len(b_); se_ = math.sqrt(va + vb)
            df_ = (va + vb) ** 2 / (va ** 2 / (len(a_) - 1) + vb ** 2 / (len(b_) - 1)); tc = stats.t.ppf(0.975, df_)
            rec[yk] = {'diff': d_, 'lo': d_ - tc * se_, 'hi': d_ + tc * se_, 'p': float(2 * stats.t.sf(abs(d_ / se_), df_)), 'df': float(df_)}
        else:
            rec[yk] = {'diff': d_}
    Fo['band340'][f'{_nm} {_kn}'] = rec

# ======================================================================================
# G. 안 된 시도
# ======================================================================================
Go = OUT['G'] = {'rows': []}
REFS = {'C10': {INTER: 'ours_ch', 'ours_intra_ch': 'prop', 'ours_intra_ch (w¹)': 'ours_w1', 'ours_layer (w¹)': 'ours_layer_w1',
                'plain L2': 'l2'}, 'C100': {INTER: 'ours_ch'}}
VARS = [('C10', 'ours_ch_st30', INTER, '늦은 시작 st30'), ('C10', 'ours_ch_st60', INTER, '늦은 시작 st60'),
        ('C10', 'ours_w1_st30', 'ours_intra_ch (w¹)', '늦은 시작 st30'), ('C10', 'ours_w1_st60', 'ours_intra_ch (w¹)', '늦은 시작 st60'),
        ('C10', 'ours_layer_w1_st30', 'ours_layer (w¹)', '늦은 시작 st30'), ('C10', 'ours_layer_w1_st60', 'ours_layer (w¹)', '늦은 시작 st60'),
        ('C100', 'ours_ch_st30', INTER, '늦은 시작 st30'),
        ('C10', 'ours_ch_taylor', INTER, 'Taylor 채널 중요도'), ('C100', 'ours_ch_taylor', INTER, 'Taylor 채널 중요도'),
        ('C10', 'ours_ch_b7', INTER, 'inter β=0.7'), ('C10', 'ours_ch_b4', INTER, 'inter β=0.4'),
        ('C10', 'sm', 'ours_intra_ch', '1-softmax'),
        ('C10', 'abl_novmem', 'ours_intra_ch', 'intra − vmem'), ('C10', 'abl_nofinal', 'ours_intra_ch', 'intra − final_step'),
        ('C10', 'abl_noinv', 'ours_intra_ch', 'intra − 1− inversion'), ('C10', 'abl_nolr', 'ours_intra_ch', 'intra − loss-ratio'),
        ('C10', 'l2_lr', 'plain L2', 'plain L2 + loss-ratio')]
REFCV = {}
for ds, mp in REFS.items():
    for nm, key in mp.items():
        pts = group_points(ds, key)
        pool = POOL10 if ds == 'C10' else POOL100
        cv = curve([(p[0], p[1], p[2]) for p in pts])
        bcv = [curve(boot_points(pts, 'pooled', pool)) for _ in range(NB)]
        REFCV[(ds, nm)] = (cv, bcv, pts)


def ref_pred(ds, nm, spk):
    cv, bcv, pts = REFCV[(ds, nm)]
    v = at(cv, math.log(spk))
    bv = np.array([at(b, math.log(spk)) for b in bcv], float)
    bv = bv[np.isfinite(bv)]
    return v, (float(bv.std()) if len(bv) > 100 else np.nan)


for ds, key, refname, label in VARS:
    sd = SD10 if ds == 'C10' else SD100
    for kn, rs in sorted(groups(ds, key).items(), key=lambda kv: -np.mean([r['spikes'] for r in kv[1]])):
        acc = float(np.mean([r['val_acc'] for r in rs])); spk = float(np.mean([r['spikes'] for r in rs])); n = len(rs)
        pred, pse = ref_pred(ds, refname, spk)
        # 같은 ρ 의 기준 팔 (있으면)
        same = runs(ds, REFS[ds][refname], kn)
        rec = {'ds': ds, 'variant': label, 'key': key, 'knob': kn, 'ref_arm': refname, 'n': n,
               'seeds': [r['seed'] for r in rs], 'servers': [r['server'] for r in rs],
               'train_loss': float(np.mean([r['train_loss'] for r in rs])), 'train_acc': float(np.mean([r['train_acc'] for r in rs])),
               'val_loss': float(np.mean([r['val_loss'] for r in rs])), 'val_acc': acc, 'spikes': spk,
               'indiv': [(r['seed'], r['server'], r['val_acc'], r['spikes']) for r in rs],
               'ref_pred': pred, 'ref_pred_se': pse,
               'same_rho': ({'n': len(same), 'val_acc': float(np.mean([r['val_acc'] for r in same])),
                             'spikes': float(np.mean([r['spikes'] for r in same]))} if same else None)}
        if np.isfinite(pred):
            se = math.sqrt(sd ** 2 / n + (pse if np.isfinite(pse) else 0) ** 2)
            rec.update({'resid': acc - pred, 'se': se, 'z': (acc - pred) / se})
        else:
            rec.update({'resid': None, 'se': None, 'z': None, 'note': '기준 곡선 범위 밖 (외삽 안 함)'})
        Go['rows'].append(rec)
# Taylor / st30 묶음
for lab, sel in (('Taylor', lambda r: r['key'] == 'ours_ch_taylor'),
                 ('st30 (세 팔·두 데이터)', lambda r: r['key'].endswith('_st30')),
                 ('st30 C10 inter 만', lambda r: r['key'] == 'ours_ch_st30' and r['ds'] == 'C10'),
                 ('가중 모양 변형 (sm, abl_*, β, l2_lr)', lambda r: r['key'] in ('sm', 'abl_novmem', 'abl_nofinal', 'abl_noinv', 'abl_nolr', 'ours_ch_b7', 'ours_ch_b4', 'l2_lr'))):
    rr = [r for r in Go['rows'] if sel(r) and r['z'] is not None]
    if not rr:
        continue
    zs = np.array([r['z'] for r in rr]); ws = np.array([1 / r['se'] ** 2 for r in rr]); res = np.array([r['resid'] for r in rr])
    # 역분산 가중 평균 잔차 (C10·C100 섞이면 %p 단위가 같지 않다는 점 주의)
    m = float((ws * res).sum() / ws.sum()); se = float(1 / math.sqrt(ws.sum()))
    Go.setdefault('pooled', {})[lab] = {'k': len(rr), 'n_runs': int(sum(r['n'] for r in rr)),
                                        'mean_resid_ivw': m, 'se': se, 'ci': [m - 1.96 * se, m + 1.96 * se],
                                        'stouffer_z': float(zs.sum() / math.sqrt(len(zs))),
                                        'p': float(2 * stats.norm.sf(abs(zs.sum() / math.sqrt(len(zs)))))}
# st30: 첫 시드 신호와 재현 시드
st = {}
for ds, key, refname in (('C10', 'ours_ch_st30', INTER), ('C10', 'ours_w1_st30', 'ours_intra_ch (w¹)'),
                         ('C10', 'ours_layer_w1_st30', 'ours_layer (w¹)')):
    for kn, rs in groups(ds, key).items():
        for r in rs:
            pred, pse = ref_pred(ds, refname, r['spikes'])
            same = runs(ds, REFS[ds][refname], kn)
            st.setdefault(f'{key} {kn}', []).append({'seed': r['seed'], 'server': r['server'], 'val_acc': r['val_acc'], 'spikes': r['spikes'],
                                                     'resid_curve': r['val_acc'] - pred if np.isfinite(pred) else None,
                                                     'diff_same_rho': (r['val_acc'] - np.mean([x['val_acc'] for x in same])) if same else None,
                                                     'same_rho_n': len(same)})
Go['st30_seeds'] = st
s1 = [v[0] for k, v in st.items() if v and v[0]['seed'] == 1 and v[0]['diff_same_rho'] is not None and k.endswith(('3e-3', '1e-3'))]
rep = [x for x in st.get('ours_ch_st30 3e-3', []) if x['seed'] != 1]
x3 = [r['val_acc'] for r in runs('C10', 'ours_ch_st30', '3e-3')]; y3 = [r['val_acc'] for r in runs('C10', 'ours_ch', '3e-3')]
xs3 = [r['spikes'] for r in runs('C10', 'ours_ch_st30', '3e-3')]; ys3 = [r['spikes'] for r in runs('C10', 'ours_ch', '3e-3')]
wt = stats.ttest_ind(x3, y3, equal_var=False)
wts = stats.ttest_ind(xs3, ys3, equal_var=False)
Go['st30_summary'] = {
    'first_seed_signal': {'k': len(s1), 'diffs': [x['diff_same_rho'] for x in s1], 'mean': float(np.mean([x['diff_same_rho'] for x in s1])),
                          'se_each': float(math.sqrt(SD10 ** 2 + SD10 ** 2 / 3)),
                          'z_pooled': float(np.mean([x['diff_same_rho'] for x in s1]) / (math.sqrt(SD10 ** 2 + SD10 ** 2 / 3) / math.sqrt(len(s1))))},
    'replication_seeds_inter3e-3': {'n': len(rep), 'diffs': [x['diff_same_rho'] for x in rep], 'resid_curve': [x['resid_curve'] for x in rep],
                                    'mean_diff': float(np.mean([x['diff_same_rho'] for x in rep])) if rep else None},
    'welch_inter3e-3': {'st30_mean': float(np.mean(x3)), 'st30_sd': float(np.std(x3, ddof=1)), 'n_st30': len(x3),
                        'base_mean': float(np.mean(y3)), 'base_sd': float(np.std(y3, ddof=1)), 'n_base': len(y3),
                        'diff': float(np.mean(x3) - np.mean(y3)), 't': float(wt.statistic), 'p': float(wt.pvalue),
                        'spk_diff_pct': float(100 * (np.mean(xs3) / np.mean(ys3) - 1)), 'p_spk': float(wts.pvalue),
                        'ci95': list(map(float, (np.mean(x3) - np.mean(y3) + np.array([-1, 1]) * stats.t.ppf(0.975, 3.9) *
                                                 math.sqrt(np.var(x3, ddof=1) / 3 + np.var(y3, ddof=1) / 3))))},
    's30_ratio': {'st30': float(np.mean([r['s30'] for r in runs('C10', 'ours_ch_st30', '3e-3')])),
                  'base': float(np.mean([r['s30'] for r in runs('C10', 'ours_ch', '3e-3')]))}}
# 승자의 저주 모의: 효과 0, 시드 SD=SD10 일 때 '세 팔 n=1 이 모두 +0.08 이상' 확률
sim = RNG.normal(0, 1, (200000, 3)) * math.sqrt(SD10 ** 2 + SD10 ** 2 / 3)
Go['st30_summary']['p_all3_ge_0.08_under_null'] = float((sim >= 0.08).all(1).mean())
Go['st30_summary']['p_mean3_ge_obs_under_null'] = float((sim.mean(1) >= Go['st30_summary']['first_seed_signal']['mean']).mean())

# ======================================================================================
# H. 견고성
# ======================================================================================
Ho = OUT['H'] = {}


def alt_rows(lo, hi, mode):
    out = []
    for r in tag_rows('C10', MAIN, 0, 1e9):
        ep = ep_of(r)
        if not ep:
            continue
        if mode == 'last':
            a = ep[-1]['val_acc'] * 100; s = ep[-1]['s_count']
        elif mode == 'last10':
            a = np.mean([e['val_acc'] for e in ep[-10:]]) * 100; s = np.mean([e['s_count'] for e in ep[-10:]])
        else:
            a = r['val_acc']; s = r['spikes']
        if lo <= s <= hi:
            x = dict(r); x['val_acc'] = float(a); x['spikes'] = float(s); x['_best_minus'] = r['val_acc'] - a; out.append(x)
    return out


Ho['checkpoint'] = {}
for mode in ('best', 'last', 'last10'):
    for lo, hi in ((250e3, 420e3), (180e3, 420e3), (320e3, 420e3)):
        rr = alt_rows(lo, hi, mode)
        rec = {k: v for k, v in ancova(rr).items() if k != 'resid'}
        rec['inter_vs_pooled_others'] = ancova([dict(r, _arm=(INTER if r['_arm'] == INTER else 'others')) for r in rr])['coef']['others']
        Ho['checkpoint'][f'{mode}|{int(lo/1e3)}-{int(hi/1e3)}K'] = rec
# best − last10 (낙관 편향) 팔별
opt = defaultdict(list)
for r in tag_rows('C10', MAIN + ['base'], 0, 1e9) + [dict(x, _arm='baseline') for x in runs('C10', 'base')]:
    ep = ep_of(r)
    if ep:
        opt[r['_arm']].append(r['val_acc'] - np.mean([e['val_acc'] for e in ep[-10:]]) * 100)
Ho['best_minus_last10'] = {a: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)), 'n': len(v)} for a, v in opt.items()}
# 베스트 에폭 위치 (1-기준) 팔별, 그리고 inter 대 나머지 (Mann-Whitney)
bep = defaultdict(list)
for r in tag_rows('C10', MAIN, 0, 1e9) + [dict(x, _arm='baseline') for x in runs('C10', 'base')]:
    if ep_of(r):
        bep[r['_arm']].append(best_epoch(r) + 1)
oth = [e for a, v in bep.items() if a not in (INTER, 'baseline') for e in v]
mw = stats.mannwhitneyu(bep[INTER], oth, alternative='two-sided')
Ho['best_epoch'] = {'by_arm': {a: {'median': float(np.median(v)), 'mean': float(np.mean(v)), 'n': len(v),
                                   'frac_le_300': float(np.mean(np.array(v) <= 300))} for a, v in bep.items()},
                    'inter_vs_others_mw_p': float(mw.pvalue)}
# 서버
b138 = [r['val_acc'] for r in runs('C10', 'base') if r['server'] == '138']; b23 = [r['val_acc'] for r in runs('C10', 'base') if r['server'] == '23']
wt = stats.ttest_ind(b23, b138, equal_var=False)
Ho['server'] = {'baseline_C10': {'138': (float(np.mean(b138)), len(b138)), '23': (float(np.mean(b23)), len(b23)),
                                 'diff_23_minus_138': float(np.mean(b23) - np.mean(b138)), 'p': float(wt.pvalue)}}
# 같은 (팔, ρ) 안 서버 고정효과
rows = [r for k in MAIN + EXTRA + ['base'] for r in runs('C10', k)]
gk = sorted(set((r['key'], r['knob']) for r in rows))
mixed = [g for g in gk if len(set(r['server'] for r in rows if (r['key'], r['knob']) == g)) == 2]
rr = [r for r in rows if (r['key'], r['knob']) in mixed]
X = np.column_stack([[1.0 if (r['key'], r['knob']) == g else 0.0 for r in rr] for g in mixed] +
                    [[1.0 if r['server'] == '23' else 0.0 for r in rr]])
b, se, res, dof, _ = ols([r['val_acc'] for r in rr], X)
Ho['server']['within_group_fe'] = {'groups': len(mixed), 'n': len(rr), 'est_23_minus_138': float(b[-1]), 'se': float(se[-1]),
                                   'p': float(2 * stats.t.sf(abs(b[-1] / se[-1]), dof)), 'dof': int(dof)}
Xs = np.column_stack([X[:, :-1], np.log([r['spikes'] for r in rr]), X[:, -1]])
bs_, ses_, *_ = ols([math.log(r['spikes']) for r in rr], np.column_stack([X[:, :-1], X[:, -1]]))
Ho['server']['within_group_fe_lnspk'] = {'est_23_minus_138_pct': float(100 * (math.exp(bs_[-1]) - 1)), 'se_ln': float(ses_[-1])}
for lo, hi in ((250e3, 420e3), (180e3, 420e3)):
    Ho['server'][f'ancova_with_server_{int(lo/1e3)}-{int(hi/1e3)}K'] = {k: v for k, v in ancova(tag_rows('C10', MAIN, lo, hi), server=True).items() if k != 'resid'}
# 스파이크 프로토콜
ratio = defaultdict(list)
for r in Do['runs']:
    ratio[r['arm']].append(r['test_spk'] / r['spikes'])
Ho['protocol_ratio_npz_over_ledger'] = {a: {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else None, 'n': len(v)}
                                        for a, v in ratio.items()}
for lo, hi in ((250e3, 420e3), (180e3, 420e3)):
    rr = [dict(r, _arm=r['arm']) for r in Do['runs'] if lo <= r['spikes'] <= hi and r['arm'] in [NAME[k] for k in MAIN]]
    if any(r['_arm'] == INTER for r in rr):
        Ho[f'protocol_{int(lo/1e3)}-{int(hi/1e3)}K'] = {
            'ledger_spk': {k: v for k, v in ancova(rr).items() if k != 'resid'},
            'npz_spk': {k: v for k, v in ancova(rr, x=lambda r: math.log(r['test_spk'])).items() if k != 'resid'}}

# 배제·미완주 (행은 남긴다)
OUT['excluded'] = [{'tag': r['tag'], 'server': r.get('server'), 'status': r.get('status'), 'epochs': r.get('epochs'),
                    'val_acc': r.get('val_acc'), 'spikes': r.get('spikes'), 's30': r.get('s30'),
                    'reason': r.get('exclude_reason') or r.get('status')} for r in EXCL]


# ======================================================================================
def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith('_')}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


json.dump(clean(OUT), open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)
print('json ->', OUT_JSON)
import pickle
pickle.dump(OUT, open(D + '/inter_deep_out.pkl', 'wb'))
