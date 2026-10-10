"""ours_inter_ch vs BPSR — 정확도·스파이크 수 말고 구조적 차이 (R19-C10, 26-10-09).

이미 계산된 것 (다시 내지 않고 인용만): 죽은 채널·MACs 제거·이미지당 활성 뉴런
(`struct_compare_261007`), 50% 집중도·드묾·산 채널 안 지니 (`npz_three_arms_261002`).
여기서 새로 재는 축:
  1. 층(단계)별 스파이크 배분과 baseline 대비 감소
  2. 분포 모양: 전체 뉴런 지니·상위 10%/1% 뉴런 몫, 채널 단위 지니·상위 10% 채널 몫,
     채널 안 공간 지니, 이미지 선택성(뉴런이 쏜 이미지 비율)
  3. 시간 대리: 발화한 (뉴런, 이미지) 쌍당 평균 발화 스텝 수 Σc/Σf (1~T=4)
  4. SynOps (위치별 정확한 fan-out, analysis/r19_fanout.py) 와 SynOps/스파이크
  5. val loss, train−val 차 (원장)
  6. 시드 간 SD (정확도·스파이크), S30/S1
통계: 두 팔만 넣은 공통 기울기 ANCOVA  y = a + b·ln(스파이크) + c·[inter].
  1차 = 원장 스파이크 300~420K, 2차 = 두 팔 유효 런 전부. 짝 비교 = inter 1.5e-3 vs BPSR 3e-11
  (원장 405K vs 412K, 각 n=3). baseline 은 기준 행으로만.

쓰는 법:  python analysis/inter_vs_bpsr_261009.py <원장 inventory.json>
"""
import json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r19_fanout import FANOUT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
INV = sys.argv[1] if len(sys.argv) > 1 else S + '/led1007b/inventory.json'
NPZ_DIRS = [os.path.join(ROOT, '_npz'), S + '/deep1005/npz']
OUT_JSON = os.path.join(ROOT, 'analysis', 'inter_vs_bpsr_261009.json')
OUT_MD = os.path.join(ROOT, 'analysis', 'inter_vs_bpsr_261009.md')
SKIP = {'n_in', 'predictions_n'}
BAND = (300e3, 420e3)
NAME = {'ours_ch': 'ours_inter_ch', 'bpsr': 'BPSR', 'base': 'baseline'}
STAGES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1']


PRUNE_DIRS = [S + '/prune_out', S + '/deep1005/prune']


def weight_layers():
    import glob
    for d in PRUNE_DIRS:
        for q in sorted(glob.glob(d + '/r19c10-*.json')):
            j = json.load(open(q))
            if 'weight_layers' in j:
                return j['weight_layers']
    raise SystemExit('weight_layers 없음')


WL = weight_layers()


def macs_removed_pct(dead):
    """죽은 채널을 물리 제거했을 때 줄어드는 밀집 MACs % (struct_compare_261007 시나리오 b 와 같은 셈)."""
    mr = mt = 0
    for w in WL:
        ks = w['ks']; cin, cout = ks[-2], ks[-1]; kk = int(np.prod(ks[:-2])) if len(ks) > 2 else 1
        di = dead.get(w['src'], 0); do = dead.get(w['dst'], 0)
        mr += (di * cout + do * cin - di * do) * kk * w['hw']; mt += cin * cout * kk * w['hw']
    return 100 * mr / mt


def stage_of(k):
    return k.split('_')[0]


def gini_sorted(v):
    v = np.sort(v.ravel())
    n = v.size; s = v.sum()
    if s == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float((2 * (i * v).sum()) / (n * s) - (n + 1) / n)


def top_share(v, q):
    v = np.sort(v.ravel())[::-1]
    k = max(1, int(round(q * v.size)))
    return float(100 * v[:k].sum() / v.sum())


def gini_cols(m):
    """m: (N, C) — 열마다 지니. 합 0 인 열은 nan."""
    v = np.sort(m, axis=0); n = v.shape[0]; s = v.sum(0)
    i = np.arange(1, n + 1)[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        g = (2 * (i * v).sum(0)) / (n * s) - (n + 1) / n
    g[s == 0] = np.nan
    return g


def npz_path(tag):
    for d in NPZ_DIRS:
        p = os.path.join(d, tag + '.npz')
        if os.path.exists(p):
            return p
    return None


def npz_metrics(p):
    z = np.load(p)
    ns = float(z['n_samples']); T = int(z['T'])
    allc, allf = [], []
    st_spk = {s: 0.0 for s in STAGES}; st_f = {s: 0.0 for s in STAGES}
    syn = 0.0
    ch_g, ch_t10, ch_w = [], [], []           # 층별 채널 지니, 상위 10% 채널 몫, 가중(층 스파이크)
    sp_g_num = sp_g_den = 0.0                 # 채널 안 공간 지니 (산 채널, 채널 스파이크 가중)
    dead_by = {}
    for k in z.files:
        # 26-10-10: npz 에 __t/__first_t/logits 등이 추가됐다. 층 키 = __fired 짝이 있는 키만.
        if k.endswith('__fired') or k + '__fired' not in z.files or k in SKIP or k in ('n_samples', 'T', 'acc'):
            continue
        c = z[k].astype(np.float64); f = z[k + '__fired'].astype(np.float64)
        allc.append(c.ravel()); allf.append(f.ravel())
        if c.ndim == 1:
            dead_by[k] = int((c == 0).sum())
        st = stage_of(k)
        st_spk[st] += c.sum(); st_f[st] += f.sum()
        syn += float((c * (FANOUT[k][:, :, None] if c.ndim == 3 else FANOUT[k])).sum())
        if c.ndim == 3:
            cc = c.reshape(-1, c.shape[-1])
            tot = cc.sum(0)
            dead_by[k] = int((tot == 0).sum())
            ch_g.append(gini_sorted(tot)); ch_t10.append(top_share(tot, 0.10)); ch_w.append(tot.sum())
            g = gini_cols(cc); alive = tot > 0
            sp_g_num += float((g[alive] * tot[alive]).sum()); sp_g_den += float(tot[alive].sum())
    c = np.concatenate(allc); f = np.concatenate(allf)
    spk = c.sum()
    act = f > 0
    p_img = f[act] / ns                        # 뉴런이 쏜 이미지 비율 (한 번이라도 쏜 뉴런만)
    nonsel = p_img >= 0.5
    ch_w = np.array(ch_w)
    m = {'npz_spk': spk / ns, 'test_acc': float(z['acc']), 'T': T,
         'gini_neuron': gini_sorted(c), 'top10_neuron': top_share(c, 0.10), 'top1_neuron': top_share(c, 0.01),
         'gini_ch_mean': float(np.mean(ch_g)), 'gini_ch_w': float((np.array(ch_g) * ch_w).sum() / ch_w.sum()),
         'top10_ch_mean': float(np.mean(ch_t10)),
         'gini_spatial_w': sp_g_num / sp_g_den,
         'p_img_mean': float(p_img.mean()), 'p_img_median': float(np.median(p_img)),
         'nonsel_pct': 100 * float(nonsel.sum()) / c.size,
         'nonsel_spk_share': 100 * float(c[act][nonsel].sum()) / spk,
         'steps_per_active': float(spk / f.sum()),
         'synops': syn / ns, 'synops_per_spk': syn / spk,
         'dead_ch': int(sum(dead_by.values())), 'macs_rm_pct': macs_removed_pct(dead_by),
         'active_img': float(f.sum() / ns)}
    for s in STAGES:
        m[f'spk_{s}'] = st_spk[s] / ns
        m[f'share_{s}'] = 100 * st_spk[s] / spk
        m[f'steps_{s}'] = st_spk[s] / st_f[s] if st_f[s] > 0 else float('nan')
    return m


# ---- 런 수집 -----------------------------------------------------------------------
inv_all = json.load(open(INV))
runs, excluded, no_npz = [], [], []
LF = ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes', 's30')
for r in inv_all['runs']:
    if r['ds'] != 'C10' or r['key'] not in NAME or r['seed'] >= 900:
        continue
    if r.get('status') == 'hold':
        continue
    ok = r.get('completed') and r['epochs'] == 310 and not r.get('exclude')
    p = npz_path(r['tag'])
    row = {'tag': r['tag'], 'key': r['key'], 'arm': NAME[r['key']], 'knob': r['knob'], 'seed': r['seed'],
           'server': r['server'], **{k: r.get(k) for k in LF}, 'npz': p is not None}
    if not ok:
        row['reason'] = r.get('exclude_reason') or f"미완주({r['epochs']}ep)"
        excluded.append(row); continue
    if p is not None:
        row.update(npz_metrics(p))
    else:
        no_npz.append(r['tag'])
    runs.append(row)


# ---- 통계 ----------------------------------------------------------------------------
def ancova(rows, y, xcol='npz_spk'):
    logy = y == 'active_img_pct'
    yy = 'active_img' if logy else y
    rr = [r for r in rows if r.get(yy) is not None and r.get(xcol) is not None
          and np.isfinite(r[yy])]
    Y = np.array([r[yy] for r in rr], float)
    if logy:
        Y = np.log(Y)
    x = np.log([r[xcol] for r in rr]); g = np.array([r['key'] == 'ours_ch' for r in rr], float)
    X = np.column_stack([np.ones_like(x), x, g])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta; dof = len(Y) - 3
    s2 = res @ res / dof; cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov)); t = beta / se
    p = 2 * stats.t.sf(np.abs(t), dof); tc = stats.t.ppf(0.975, dof)
    # 기울기 차이 (상호작용) F
    X2 = np.column_stack([X, x * g]); b2, *_ = np.linalg.lstsq(X2, Y, rcond=None)
    rs2 = Y - X2 @ b2; F = ((res @ res) - (rs2 @ rs2)) / ((rs2 @ rs2) / (len(Y) - 4))
    pF = float(stats.f.sf(F, 1, len(Y) - 4))
    # 기울기를 따로 둔 모형의 고정점 대비 (x0 = 1차 구간 npz 스파이크 기하 평균, 두 팔 모두 덮을 때)
    x0 = X0[xcol]
    sep = None
    ia, ib = g == 1, g == 0
    if x.min() <= np.log(x0) <= x.max() and min(x[ia].min(), x[ib].min()) <= np.log(x0) <= max(x[ia].max(), x[ib].max()):
        out = []
        for m in (ia, ib):
            Xm = np.column_stack([np.ones(m.sum()), x[m]]); bm, *_ = np.linalg.lstsq(Xm, Y[m], rcond=None)
            rm = Y[m] - Xm @ bm; dm = m.sum() - 2; s2m = rm @ rm / dm
            v = np.array([1, np.log(x0)]); cm = s2m * np.linalg.inv(Xm.T @ Xm)
            out.append((v @ bm, v @ cm @ v, dm))
        d = out[0][0] - out[1][0]; sed = np.sqrt(out[0][1] + out[1][1])
        dd = (out[0][1] + out[1][1]) ** 2 / (out[0][1] ** 2 / out[0][2] + out[1][1] ** 2 / out[1][2])
        sep = {'x0': x0, 'diff': float(d), 'se': float(sed), 'p': float(2 * stats.t.sf(abs(d / sed), dd))}
    return {'logy': logy, 'sep': sep, 'n': len(Y), 'n_inter': int(g.sum()), 'n_bpsr': int(len(Y) - g.sum()),
            'b': float(beta[1]), 'b_se': float(se[1]), 'c': float(beta[2]), 'c_se': float(se[2]),
            'c_lo': float(beta[2] - tc * se[2]), 'c_hi': float(beta[2] + tc * se[2]), 'c_p': float(p[2]),
            'slope_diff_p': pF, 'resid_sd': float(np.sqrt(s2))}


def welch(a, b):
    a = np.array(a, float); b = np.array(b, float)
    d = a.mean() - b.mean(); se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {'na': len(a), 'nb': len(b), 'diff': float(d), 'se': float(se), 'p': float(p)}


NPZ_Y = [('dead_ch', '죽은 채널 (/4,736)', 1), ('macs_rm_pct', '죽은 채널 제거 시 MACs 감소 %', 2),
         ('active_img_pct', '이미지당 활성 뉴런 (ln, % 차)', 2),
         ('gini_neuron', '뉴런 지니 (전체 뉴런)', 4), ('top10_neuron', '상위 10% 뉴런 몫 %', 2),
         ('top1_neuron', '상위 1% 뉴런 몫 %', 2),
         ('gini_ch_mean', '채널 지니 (17 conv 층 평균)', 4), ('gini_ch_w', '채널 지니 (층 스파이크 가중)', 4),
         ('top10_ch_mean', '상위 10% 채널 몫 % (층 평균)', 2), ('gini_spatial_w', '산 채널 안 공간 지니', 4),
         ('p_img_mean', '뉴런이 쏜 이미지 비율 (활성 뉴런 평균)', 4),
         ('nonsel_pct', '이미지 절반 이상에서 쏘는 뉴런 % (전체 대비)', 3),
         ('nonsel_spk_share', '그 뉴런들의 스파이크 몫 %', 2),
         ('steps_per_active', '발화한 (뉴런,이미지) 당 스텝 수 (/4)', 4),
         ('synops', 'SynOps/이미지', 0), ('synops_per_spk', 'SynOps/스파이크', 1)] + \
        [(f'share_{s}', f'{s} 스파이크 몫 %', 2) for s in STAGES] + \
        [(f'steps_{s}', f'{s} 스텝/활성쌍', 4) for s in STAGES]
LED_Y = [('val_loss', 'val loss', 4), ('train_loss', 'train loss', 4), ('val_acc', 'val acc', 3),
         ('gap_acc', 'train acc − val acc (%p)', 3), ('gap_loss', 'val loss − train loss', 4),
         ('s30', 'S30/S1', 4)]
for r in runs:
    r['gap_acc'] = r['train_acc'] - r['val_acc']; r['gap_loss'] = r['val_loss'] - r['train_loss']

two = [r for r in runs if r['key'] in ('ours_ch', 'bpsr')]
band = [r for r in two if BAND[0] <= r['spikes'] <= BAND[1]]
X0 = {'npz_spk': float(np.exp(np.mean(np.log([r['npz_spk'] for r in band if r['npz']])))),
      'spikes': float(np.exp(np.mean(np.log([r['spikes'] for r in band]))))}
res = {'npz': {}, 'ledger': {}, 'pair': {}, 'pair2': {}}
for y, _, _ in NPZ_Y:
    res['npz'][y] = {'band': ancova([r for r in band if r['npz']], y),
                     'all': ancova([r for r in two if r['npz']], y)}
for y, _, _ in LED_Y:
    res['ledger'][y] = {'band': ancova(band, y, 'spikes'), 'all': ancova(two, y, 'spikes')}
# npz 축 ANCOVA 에서 x 를 원장 스파이크로 (민감도)
res['npz_xledger'] = {y: ancova([r for r in band if r['npz']], y, 'spikes') for y, _, _ in NPZ_Y}


def grp(key, knob, need_npz=False):
    return [r for r in runs if r['key'] == key and r['knob'] == knob and (r['npz'] or not need_npz)]


PA, PB = grp('ours_ch', '1.5e-3', True), grp('bpsr', '3e-11', True)
PA2, PB2 = grp('ours_ch', '4e-3', True), grp('bpsr', '1e-10', True)
for y, _, _ in NPZ_Y + LED_Y + [('npz_spk', '', 0), ('spikes', '', 0)]:
    yy = 'active_img' if y == 'active_img_pct' else y
    res['pair'][y] = welch([r[yy] for r in PA], [r[yy] for r in PB])
    res['pair2'][y] = welch([r[yy] for r in PA2], [r[yy] for r in PB2])

# baseline 대비 단계별 감소 (npz)
base = [r for r in runs if r['key'] == 'base' and r['npz']]
bmean = {s: np.mean([r[f'spk_{s}'] for r in base]) for s in STAGES}
bmean['synops'] = np.mean([r['synops'] for r in base]); bmean['npz_spk'] = np.mean([r['npz_spk'] for r in base])
for r in runs:
    if r['npz']:
        for s in STAGES:
            r[f'red_{s}'] = 100 * (r[f'spk_{s}'] / bmean[s] - 1)
        r['red_total'] = 100 * (r['npz_spk'] / bmean['npz_spk'] - 1)
        r['red_synops'] = 100 * (r['synops'] / bmean['synops'] - 1)
for y in [f'red_{s}' for s in STAGES] + ['red_synops']:
    res['npz'][y] = {'band': ancova([r for r in band if r['npz']], y),
                     'all': ancova([r for r in two if r['npz']], y)}
    res['pair'][y] = welch([r[y] for r in PA], [r[y] for r in PB])
    res['pair2'][y] = welch([r[y] for r in PA2], [r[y] for r in PB2])

# 시드 간 SD (원장, 같은 ρ n≥3 묶음)
sd = {}
for key in ('ours_ch', 'bpsr', 'base'):
    gs = {}
    for r in runs:
        if r['key'] == key:
            gs.setdefault(r['knob'], []).append(r)
    rows_ = []
    for kb, g in gs.items():
        if len(g) < 3:
            continue
        a = np.array([x['val_acc'] for x in g]); s_ = np.array([x['spikes'] for x in g])
        rows_.append({'knob': kb, 'n': len(g), 'acc_sd': float(a.std(ddof=1)), 'spk_mean': float(s_.mean()),
                      'spk_cv': float(100 * s_.std(ddof=1) / s_.mean()),
                      'in_band': bool(BAND[0] <= s_.mean() <= BAND[1]),
                      'acc_ss': float(((a - a.mean()) ** 2).sum()),
                      'lspk_ss': float(((np.log(s_) - np.log(s_).mean()) ** 2).sum())})
    sd[key] = rows_


def pooled(rows_, band_only):
    rr = [x for x in rows_ if x['in_band'] or not band_only]
    dof = sum(x['n'] - 1 for x in rr)
    if dof == 0:
        return None
    return {'dof': dof, 'groups': len(rr), 'acc_sd': float(np.sqrt(sum(x['acc_ss'] for x in rr) / dof)),
            'spk_cv': float(100 * np.sqrt(sum(x['lspk_ss'] for x in rr) / dof))}


sd_pool = {k: {'all': pooled(v, False), 'band': pooled(v, True)} for k, v in sd.items()}
sd_test = {}
for which in ('all', 'band'):
    a, b = sd_pool['ours_ch'][which], sd_pool['bpsr'][which]
    if a and b:
        out = {}
        for m in ('acc_sd', 'spk_cv'):
            F = (a[m] / b[m]) ** 2
            p = 2 * min(stats.f.sf(F, a['dof'], b['dof']), stats.f.cdf(F, a['dof'], b['dof']))
            out[m] = {'ratio': a[m] / b[m], 'F': F, 'p': float(p), 'dof': (a['dof'], b['dof'])}
        sd_test[which] = out

json.dump({'generated_from': {'inventory': INV, 'inventory_generated': inv_all.get('generated'),
                              'npz_dirs': NPZ_DIRS},
           'definitions': __doc__, 'band': BAND, 'runs': runs, 'excluded': excluded, 'no_npz': no_npz,
           'baseline_npz_mean': {k: float(v) for k, v in bmean.items()}, 'stats': res,
           'seed_sd': sd, 'seed_sd_pooled': sd_pool, 'seed_sd_test': sd_test},
          open(OUT_JSON, 'w'), indent=1, default=float)
print('runs', len(runs), 'npz', sum(r['npz'] for r in runs), 'excluded', len(excluded), 'no npz', len(no_npz))


# ---- md ------------------------------------------------------------------------------
J = json.load(open(OUT_JSON))
ST = J['stats']; L = []
w = L.append


def gm(key, knob, col, npz=True):
    g = [r for r in runs if r['key'] == key and r['knob'] == knob and (r['npz'] or not npz) and r.get(col) is not None]
    v = np.array([r[col] for r in g], float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float('nan')), len(v)


def fmt(v, d):
    return f'{v:,.{d}f}'


def ci(a, d, scale=1.0, pct_log=False):
    if pct_log:
        c, lo, hi = (100 * (np.exp(a[k]) - 1) for k in ('c', 'c_lo', 'c_hi'))
        return f"{c:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}] p={a['c_p']:.2g}"
    return f"{a['c']*scale:+.{d}f} [{a['c_lo']*scale:+.{d}f}, {a['c_hi']*scale:+.{d}f}] p={a['c_p']:.2g}"


def pr(p, d, scale=1.0):
    return f"{p['diff']*scale:+.{d}f} ±{p['se']*scale:.{d}f} p={p['p']:.2g}"


base_led = [r for r in runs if r['key'] == 'base']
nb = ST['npz']['gini_neuron']['band']
w('# ours_inter_ch vs BPSR — 정확도·스파이크 수 밖의 구조적 차이 (R19-C10, 26-10-09)\n')
w(f"생성: `analysis/inter_vs_bpsr_261009.py` → `.json`, 이 문서. 원장 {J['generated_from']['inventory_generated']} 집계 "
  f"(310 에폭 완주·S30/S1≥0.19·seed<900), npz = 테스트 1만 장 1회 추론, 스파이킹 18층 (n_in·predictions_n 제외), T=4. "
  "ours_inter_ch 의 런 키는 `ours_ch`.\n")
w('## 요약\n')
S_ = ST
a_dead = S_['npz']['dead_ch']['band']; a_mac = S_['npz']['macs_rm_pct']['band']; a_act = S_['npz']['active_img_pct']['band']
a_syn = S_['npz']['red_synops']['band']; a_vl = S_['ledger']['val_loss']['band']; a_va = S_['ledger']['val_acc']['band']
a_gn = S_['npz']['gini_neuron']['band']; a_t10 = S_['npz']['top10_neuron']['band']; a_st = S_['npz']['steps_per_active']['band']
a_gc = S_['npz']['gini_ch_w']['band']
t_all = J['seed_sd_test']['all']; pa = J['seed_sd_pooled']
w(f"1. **inter 가 앞서는 축 = 채널 단위 구조 희소성과 이미지당 활성 뉴런.** 같은 스파이크 (원장 300~420K, inter n={a_dead['n_inter']} / BPSR n={a_dead['n_bpsr']}) 에서 "
  f"죽은 채널 {ci(a_dead, 1)} 개 (/4,736), 그 채널을 지웠을 때 MACs 감소 {ci(a_mac, 2)} %p, 이미지당 활성 뉴런 {ci(a_act, 1, pct_log=True)} %. "
  f"스파이크가 거의 같은 짝 (inter 1.5e-3 vs BPSR 3e-11, 원장 405K vs 412K, 각 n=3) 에서도 부호가 같다 (활성 뉴런 p<0.01, 죽은 채널·MACs 는 p=0.14·0.08 로 짝만으로는 유의하지 않음).")
w(f"2. **줄이는 방식이 다르다 (우열이 아니라 모양).** BPSR 은 발화하는 뉴런의 스텝 수를 고르게 깎고, inter 는 뉴런·채널을 통째로 끈다: "
  f"발화한 (뉴런, 이미지) 쌍당 발화 스텝 수 (최대 4) 가 inter 쪽이 {ci(a_st, 3)} 많고 (baseline 1.64 → 짝에서 inter 1.60, BPSR 1.54; conv1·conv2 에서 +0.12~0.13, conv4·fc1 에서는 차이 없음), "
  f"전체 뉴런 지니 {ci(a_gn, 3)}, 상위 10% 뉴런의 스파이크 몫 {ci(a_t10, 1)} %p, 채널 지니 {ci(a_gc, 3)} 로 스파이크가 더 소수에 몰린다. "
  "그 몫의 상당 부분은 이미지 절반 이상에서 쏘는 뉴런이 나른다 (3절).")
w(f"3. **차이 없는 축.** SynOps (fan-out 가중 시냅스 연산) 는 baseline 대비 감소율 차 {ci(a_syn, 2)} %p 로 통계적으로는 0 이 아니지만 0.4%p 이하라 실질 차이 없음 (전 구간 n=19/12 에서는 p=0.67). "
  f"층(단계)별 스파이크 배분은 1.3%p 안 (inter 가 conv1 을 덜, conv3 를 더 줄임). val loss {ci(a_vl, 4)}, val acc {ci(a_va, 3)} %p, train−val 차 (정확도·loss) 모두 CI 가 0 을 포함.")
w(f"4. **BPSR 이 앞서는 축 = 시드 재현성.** 같은 ρ 시드 간 SD (n≥3 묶음 합동): 정확도 inter {pa['ours_ch']['all']['acc_sd']:.3f}%p (자유도 {pa['ours_ch']['all']['dof']}) vs BPSR {pa['bpsr']['all']['acc_sd']:.3f}%p (자유도 {pa['bpsr']['all']['dof']}), "
  f"F 검정 p={t_all['acc_sd']['p']:.3f}; 착지 스파이크 변동계수 {pa['ours_ch']['all']['spk_cv']:.2f}% vs {pa['bpsr']['all']['spk_cv']:.2f}%, p={t_all['spk_cv']['p']:.3f} "
  f"(1차 구간만: {pa['ours_ch']['band']['spk_cv']:.2f}% vs {pa['bpsr']['band']['spk_cv']:.2f}%, p={J['seed_sd_test']['band']['spk_cv']['p']:.3f}, 자유도 9/4). 정확도 SD 는 경계선이고 BPSR 자유도가 8 이라 **아직 판단 불가**; 스파이크 착지는 BPSR 이 더 일정하다.")
w("5. **잴 수 없었던 축.** 타임스텝별 스파이크 분포·첫 스파이크 시각·보정 오차(ECE)·클래스별 정확도는 지금 npz (뉴런별 총합만 저장) 와 원장에 없다. 새 추론이 필요하다 (끝의 [LIMITATION]).\n")

w('## 정의\n')
w('- **같은 스파이크 비교 (ANCOVA)**: 두 팔만 넣고 `y = a + b·ln(npz 테스트 스파이크) + c·[inter]`. c = 같은 스파이크에서 inter − BPSR. '
  '1차 = 원장 스파이크 300~420K (inter n=11, BPSR n=6, npz 있는 런), 2차 = 두 팔 유효 npz 전 런 (n=19/12, 147~437K). '
  '기울기가 팔마다 다를 수 있어 (표의 "기울기 차 p") 기울기를 따로 둔 모형의 1차 구간 중앙 (npz 389K) 대비도 json `sep` 에 있다 — 전 지표에서 공통 기울기 c 와 소수 셋째 자리까지 같아 표에는 c 만 적었다.')
w('- **짝 비교**: inter 1.5e-3 vs BPSR 3e-11 (원장 405,469 vs 412,033, npz 430,731 vs 437,056, 각 n=3), Welch t. 모형 없이 거의 같은 스파이크끼리. '
  '보조 짝: inter 4e-3 vs BPSR 1e-10 (302K vs 319K, 각 n=3) — 스파이크가 5% 다르므로 json `pair2` 에만.')
w('- **죽은 채널**: 테스트 1만 장에서 공간 모든 위치가 한 번도 안 쏜 채널 (/4,736 = conv 4,480 + fc1 256). **MACs 감소 %**: 그 채널을 물리 제거할 때 줄어드는 밀집 MACs 비율 (`struct_compare_261007` 시나리오 b 와 같은 셈). '
  '**이미지당 활성 뉴런**: 이미지 한 장에서 T 스텝 동안 한 번이라도 쏜 뉴런 수 (ln 으로 회귀, % 로 환산).')
w('- **지니**: 0 = 모든 뉴런이 같은 수를 쏨, 1 = 한 뉴런이 전부. 전체 뉴런 지니는 1,442,048 개 뉴런의 총 스파이크로 (침묵 뉴런 포함). '
  '**채널 지니**: conv 17층 각각에서 채널별 총 스파이크의 지니, 층 스파이크로 가중 평균. **상위 10% 몫**: 가장 많이 쏜 10% 가 낸 스파이크 비율. '
  '**산 채널 안 공간 지니**: 산 채널 안에서 위치별 스파이크의 지니, 채널 스파이크 가중 평균.')
w('- **쏜 이미지 비율**: 한 번이라도 쏜 뉴런마다 (쏜 이미지 수 / 1만), 그 평균. 낮을수록 뉴런이 이미지를 가려서 쏜다 (선택성). '
  '**이미지 절반 이상에서 쏘는 뉴런**: 그 비율 ≥ 0.5 인 뉴런 (전체 뉴런 대비 %) 과 그 뉴런들이 낸 스파이크 몫.')
w('- **스텝/활성쌍** = Σ스파이크 / Σ(뉴런이 쏜 이미지 수) — 뉴런이 어떤 이미지에서 발화했을 때 T=4 스텝 중 평균 몇 스텝 쐈나 (1~4). 시간축 분포 자체는 아니고 그 대리.')
w('- **SynOps**: 스파이크 하나가 다음 연산에서 일으키는 곱셈-누산 수 (위치별 경계 포함 정확한 fan-out, `analysis/r19_fanout.py`; BN 접힘·identity shortcut 0·단계 입구 1×1 shortcut 포함·GAP 는 fc1 에 접힘). 이미지당 합.')
w('- 스파이크·정확도는 원장 (베스트 에폭 eval), 분포 지표는 npz (테스트 모드). npz/원장 스파이크 비는 1.05~1.07.\n')

w('## 1. 축별 결과 (inter − BPSR)\n')
w('| 축 | 지표 | baseline (n=3) | inter 1.5e-3 (n=3) | BPSR 3e-11 (n=3) | 짝 차 ±SE (n=3/3) | 같은 스파이크 1차 300~420K (n=11/6) | 기울기 차 p | 2차 전 구간 (n=19/12) | 판정 |')
w('|---|---|---|---|---|---|---|---|---|---|')
ROWS = [
    ('구조 (하드웨어)', 'dead_ch', '죽은 채널 (/4,736)', 1, 1, 'inter 우위'),
    ('구조 (하드웨어)', 'macs_rm_pct', '죽은 채널 제거 시 MACs 감소 %', 2, 1, 'inter 우위'),
    ('구조 (하드웨어)', 'active_img_pct', '이미지당 활성 뉴런', 1, 1, 'inter 우위'),
    ('연산량', 'red_synops', 'SynOps 감소율 vs baseline %', 2, 1, '실질 차이 없음 (<0.4%p)'),
    ('연산량', 'synops_per_spk', 'SynOps / 스파이크', 1, 1, '차이 없음'),
    ('분포 모양', 'gini_neuron', '뉴런 지니', 4, 1, '다름 (inter 가 더 몰림)'),
    ('분포 모양', 'top10_neuron', '상위 10% 뉴런 몫 %', 2, 1, '다름'),
    ('분포 모양', 'top1_neuron', '상위 1% 뉴런 몫 %', 2, 1, '다름'),
    ('분포 모양', 'gini_ch_w', '채널 지니 (가중)', 4, 1, '다름 (inter 가 더 몰림)'),
    ('분포 모양', 'top10_ch_mean', '상위 10% 채널 몫 % (층 평균)', 2, 1, '다름'),
    ('분포 모양', 'gini_spatial_w', '산 채널 안 공간 지니', 4, 1, '다름 (작음, inter 가 더 고름)'),
    ('선택성', 'p_img_mean', '쏜 이미지 비율 (활성 뉴런 평균)', 4, 1, '다름 (inter 가 약간 더 가려 쏨)'),
    ('선택성', 'nonsel_pct', '이미지 절반 이상에서 쏘는 뉴런 %', 2, 1, '짝에서 유의하지 않음'),
    ('선택성', 'nonsel_spk_share', '그 뉴런들의 스파이크 몫 %', 2, 1, '다름 (inter 가 큼)'),
    ('시간 대리', 'steps_per_active', '스텝/활성쌍 (최대 4)', 3, 1, '다름 (inter 가 큼)'),
]
for ax, y, nm, d, sc, verdict in ROWS:
    a, b = ST['npz'][y]['band'], ST['npz'][y]['all']
    yy = 'active_img' if y == 'active_img_pct' else y
    bm = np.mean([r[yy] for r in runs if r['key'] == 'base' and r['npz']])
    m1, s1, n1 = gm('ours_ch', '1.5e-3', yy); m2, s2, n2 = gm('bpsr', '3e-11', yy)
    pl = y == 'active_img_pct'
    pp = ST['pair'][y]
    if pl:
        pstr = f"{100*pp['diff']/m2:+.1f}% p={pp['p']:.2g}"
        dd = 0
    else:
        pstr = pr(pp, d); dd = d
    w(f"| {ax} | {nm} | {fmt(bm, dd)} | {fmt(m1, dd)} ±{fmt(s1, dd)} | {fmt(m2, dd)} ±{fmt(s2, dd)} | {pstr} | "
      f"{ci(a, d if not pl else 1, pct_log=pl)} | {a['slope_diff_p']:.2g} | {ci(b, d if not pl else 1, pct_log=pl)} | {verdict} |")
w('\n- 활성 뉴런 행: 값은 이미지당 뉴런 수, 차는 % (짝: BPSR 대비 %, ANCOVA: ln 오프셋을 % 로).')
w('- `struct_compare_261007` 의 5팔 공통 기울기 모형 (320~420K, n=31) 에서는 inter − BPSR = 죽은 채널 +39.4 [+21.5, +57.3], MACs +2.65 [+1.58, +3.72] %p, 활성 뉴런 −4.9 [−6.0, −3.8] %. 여기 두 팔 모형은 기울기를 두 팔에서만 추정해 값이 조금 크다. 부호·유의는 같다.')
w('- 기울기 차 p 가 작은 지표가 많다: inter 는 스파이크를 줄일수록 분포가 빠르게 몰리고 BPSR 은 거의 그대로다 (2차 전 구간 c 가 1차보다 큰 이유). 즉 차이는 스파이크를 더 줄일수록 커진다.\n')

w('## 2. 어디서 줄었나 — 단계별 스파이크 (npz, 이미지당)\n')
w('baseline 대비 감소율 %. 단계 = conv1 (1층), conv2 (6층, 32×32), conv3 (6층, 16×16), conv4 (4층, 8×8), fc1.\n')
w('| 그룹 | n | 전체 | conv1 | conv2 | conv3 | conv4 | fc1 | SynOps |')
w('|---|---|---|---|---|---|---|---|---|')
for key, kb, nm in [('ours_ch', '1.5e-3', 'inter 1.5e-3'), ('bpsr', '3e-11', 'BPSR 3e-11'), ('ours_ch', '4e-3', 'inter 4e-3'), ('bpsr', '1e-10', 'BPSR 1e-10')]:
    cells = []
    for c in ['red_total'] + [f'red_{s}' for s in STAGES] + ['red_synops']:
        m, s_, n = gm(key, kb, c); cells.append(f'{m:+.1f} ±{s_:.1f}')
    w(f"| {nm} | {n} | " + ' | '.join(cells) + ' |')
w('| baseline 절대값 (n=3) | 3 | ' + f"{bmean['npz_spk']:,.0f}" + ' | ' + ' | '.join(f"{bmean[s]:,.0f}" for s in STAGES) + f" | {bmean['synops']/1e6:,.0f}M |")
w('\n같은 스파이크 inter − BPSR (감소율 차 %p, 음수 = inter 가 더 줄임) 과 스파이크 몫 차 (%p):\n')
w('| 단계 | 감소율 차 1차 (n=11/6) | 감소율 차 짝 (n=3/3) | 몫 차 1차 | 몫 차 짝 | 스텝/활성쌍 차 1차 | 스텝/활성쌍 차 짝 |')
w('|---|---|---|---|---|---|---|')
for s in STAGES:
    w(f"| {s} | {ci(ST['npz'][f'red_{s}']['band'], 2)} | {pr(ST['pair'][f'red_{s}'], 2)} | {ci(ST['npz'][f'share_{s}']['band'], 2)} | "
      f"{pr(ST['pair'][f'share_{s}'], 2)} | {ci(ST['npz'][f'steps_{s}']['band'], 3)} | {pr(ST['pair'][f'steps_{s}'], 3)} |")
_act = 100 * (np.exp(ST['npz']['active_img_pct']['band']['c']) - 1)
_sb = np.mean([r['steps_per_active'] for r in runs if r['key'] == 'base' and r['npz']])
_s = {kb: gm(k, kb, 'steps_per_active')[0] for k, kb in [('ours_ch', '1.5e-3'), ('bpsr', '3e-11'), ('ours_ch', '4e-3'), ('bpsr', '1e-10')]}
w(f'\n- 가설 "inter 는 채널을 통째로 끄고 BPSR 은 발화율을 고르게 낮춘다" 는 **뉴런·이미지 단위에서 지지된다**: 같은 스파이크에서 inter 는 활성 뉴런이 적고 ({_act:+.1f}%) 활성 뉴런은 더 여러 스텝 쏜다 (+0.09 스텝). '
  f'BPSR 은 활성 뉴런을 남기고 스텝을 깎는다 — 스텝/활성쌍 baseline {_sb:.2f} (n=3) 에서 BPSR {_s["3e-11"]:.2f} → {_s["1e-10"]:.2f} (3e-11 → 1e-10), inter {_s["1.5e-3"]:.2f} → {_s["4e-3"]:.2f} (1.5e-3 → 4e-3), 각 n=3. '
  '차이는 conv1·conv2 (32×32 해상도) 에 몰려 있고 conv4·fc1 에서는 없다.')
w('- 같은 경향이 ρ 를 올릴수록 벌어진다 (6절 개별값): BPSR 은 이미지 절반 이상에서 쏘는 뉴런의 스파이크 몫이 36% → 27% → 17% → 12% (ρ 3e-11 → 6e-10) 로 빠르게 줄지만 뉴런 지니는 0.57~0.60 으로 거의 그대로다. inter 는 그 몫이 42% → 29% (1.5e-3 n=3 → 1.2e-2 n=1) 로 덜 줄고 지니는 0.61 → 0.78 로 오른다. BPSR 묶음은 각 n=3.')
w('- **층 배분은 거의 같다**: 단계별 몫 차가 1.3%p 이내. inter 는 conv1 을 덜 (+5.0%p 덜 줄임), conv3 를 더 (−2.3%p) 줄인다. 앞층 vs 뒷층 배분이 근본적으로 다르다고 말할 수 없다.\n')

w('## 3. 분포 모양과 선택성 — 읽는 법\n')
m1 = gm('ours_ch', '1.5e-3', 'nonsel_spk_share'); m2 = gm('bpsr', '3e-11', 'nonsel_spk_share'); mb = np.mean([r['nonsel_spk_share'] for r in runs if r['key'] == 'base' and r['npz']])
w(f"- inter 는 스파이크가 더 소수 뉴런·채널에 몰린다 (지니·상위 10% 몫 모두 BPSR 과 baseline 보다 큼). BPSR 의 지니는 baseline 보다 오히려 낮다 (0.570 vs 0.583) — 발화를 고르게 깎는 방법이라는 해석과 맞다.")
w(f"- 몰린 몫의 상당 부분은 **이미지 절반 이상에서 쏘는 뉴런** 이 낸다: 그 뉴런들의 스파이크 몫 inter {m1[0]:.1f}% vs BPSR {m2[0]:.1f}% (각 n=3), baseline {mb:.1f}%. "
  "뉴런 수는 둘 다 전체의 약 10% 로 같고 (짝 p=0.21), inter 에서는 그 뉴런들이 더 많이 쏜다. 한편 한 번이라도 쏜 뉴런의 평균 쏜 이미지 비율은 inter 가 약간 낮다 (−0.005~−0.008). "
  "→ inter 는 '대부분의 뉴런은 더 가려 쏘고, 소수의 상시 발화 뉴런이 더 많이 나르는' 구조다. 이것이 정확도나 일반화에 이득이라는 증거는 없다 (4절: val loss·train−val 차 차이 없음). **선택성이 inter 의 본질이라고 쓸 근거는 약하다** — 차이는 '어느 뉴런이 쏘나' 의 집중이지 이미지별 선택성의 큰 변화가 아니다.")
w('- 산 채널 안 공간 지니는 inter 가 약간 낮다 (−0.007): 살아남은 채널 안에서는 위치별 발화가 더 고르다. 채널 단위로 끄고 남은 채널은 고르게 쓰는 모양과 일치하지만 크기가 작다.\n')

w('## 4. 정확도 외 품질과 학습 (원장)\n')
w('| 지표 | baseline (n=6) | inter 1.5e-3 (n=3) | BPSR 3e-11 (n=3) | 짝 차 ±SE | 1차 300~420K (n=12/6) | 2차 전 구간 (n=28/12) | 판정 |')
w('|---|---|---|---|---|---|---|---|')
for y, nm, d in LED_Y:
    bv = [r[y] for r in base_led if r.get(y) is not None]; bm = np.mean(bv); bs = np.std(bv, ddof=1)
    m1, s1, _ = gm('ours_ch', '1.5e-3', y, False); m2, s2, _ = gm('bpsr', '3e-11', y, False)
    a, b = ST['ledger'][y]['band'], ST['ledger'][y]['all']
    v = '차이 없음' if a['c_lo'] <= 0 <= a['c_hi'] else ('inter 가 큼' if a['c'] > 0 else 'inter 가 작음')
    v = {'train_loss': '일관되지 않음 (짝 부호 반대)', 's30': '판단 불가 (구간마다 부호 다름)'}.get(y, v)
    w(f"| {nm} | {bm:.{d}f} ±{bs:.{d}f} | {m1:.{d}f} ±{s1:.{d}f} | {m2:.{d}f} ±{s2:.{d}f} | {pr(ST['pair'][y], d)} | {ci(a, d)} | {ci(b, d)} | {v} |")
w('\n- train loss 는 1차에서 inter 가 +0.006 크지만 (p=0.04) 짝에서는 −0.001 (p=0.81) — 일관되지 않다. 나머지 (val loss, val acc, 일반화 차) 는 차이 없음.')
w('- S30/S1 (30에폭 스파이크 / 1에폭 스파이크): 2차 전 구간에서 inter 가 +0.032 크다 (기울기 차 p<1e-16) — 같은 최종 스파이크라면 inter 가 학습 초반 스파이크를 덜 줄인다. 1차와 짝에서는 부호가 갈려 판단 불가.')
w('- ECE·클래스별 정확도: 저장된 로짓이 없어 계산 못 함.\n')

w('## 5. 시드 재현성 (원장, 같은 ρ 시드 n≥3 묶음)\n')
w('| 팔 | ρ | n | 서버 | val acc SD (%p) | 평균 스파이크 | 스파이크 변동계수 % | 1차 구간 |')
w('|---|---|---|---|---|---|---|---|')
for key in ('ours_ch', 'bpsr', 'base'):
    for x in J['seed_sd'][key]:
        srv = '/'.join(sorted({r['server'] for r in runs if r['key'] == key and r['knob'] == x['knob']}))
        w(f"| {NAME[key]} | {x['knob']} | {x['n']} | {srv} | {x['acc_sd']:.3f} | {x['spk_mean']:,.0f} | {x['spk_cv']:.2f} | {'○' if x['in_band'] else ''} |")
    for which in ('all', 'band'):
        q = J['seed_sd_pooled'][key][which]
        if q:
            w(f"| **{NAME[key]} 합동 ({'전 구간' if which == 'all' else '1차'})** | | **자유도 {q['dof']}** | | **{q['acc_sd']:.3f}** | | **{q['spk_cv']:.2f}** | |")
T_ = J['seed_sd_test']
w(f"\nF 검정 (inter/BPSR 분산비): 전 구간 정확도 SD 비 {T_['all']['acc_sd']['ratio']:.2f} p={T_['all']['acc_sd']['p']:.3f}, 스파이크 CV 비 {T_['all']['spk_cv']['ratio']:.2f} p={T_['all']['spk_cv']['p']:.3f} (자유도 17/8); "
  f"1차 정확도 {T_['band']['acc_sd']['ratio']:.2f} p={T_['band']['acc_sd']['p']:.2f}, 스파이크 CV {T_['band']['spk_cv']['ratio']:.2f} p={T_['band']['spk_cv']['p']:.3f} (자유도 9/4).\n")
w('- inter 의 정확도 시드 SD (0.11) 는 baseline (0.10, n=6) 과 같은 수준이고, BPSR 이 낮다 (0.06). BPSR 자유도 8 로는 0.06 vs 0.10 이 진짜 차이인지 판단 불가 (p=0.065).')
w('- 착지 스파이크는 BPSR 이 더 일정하다 (CV 1.1% vs 2.2%, 1차 0.7% vs 2.9%). inter 는 λ 를 loss 비로 매 스텝 조정하는 경로라 착지점이 더 흔들리는 것으로 보이나 원인은 확인하지 않았다. **이 축은 inter 가 불리하다.**\n')

w('## 6. 런 개별값 (팔·ρ 묶음, 묶음 아래 평균 ± SD)\n')
w('train/val 과 spikes 는 원장 (베스트 에폭), 이하는 npz. npz 없는 런은 원장 열만. `1차` = 원장 스파이크 300~420K.\n')
cols = [('npz_spk', 0), ('dead_ch', 0), ('macs_rm_pct', 1), ('active_img', 0), ('gini_neuron', 3), ('top10_neuron', 1),
        ('gini_ch_w', 3), ('p_img_mean', 3), ('nonsel_spk_share', 1), ('steps_per_active', 3), ('synops', 0)]
w('| 방법 | ρ | 시드 | 서버 | train loss | train acc | val loss | val acc | spikes | 1차 | npz 스파이크 | 죽은 채널 | MACs 감소 % | 활성 뉴런/이미지 | 뉴런 지니 | 상위10% 뉴런 몫 | 채널 지니 | 쏜 이미지 비율 | 상시발화 뉴런 몫 | 스텝/활성쌍 | SynOps (M) |')
w('|' + '---|' * 21)
def kf(k):
    return -1 if k == '-' else float(k)
for key in ('ours_ch', 'bpsr', 'base'):
    for kb in sorted({r['knob'] for r in runs if r['key'] == key}, key=kf):
        g = sorted([r for r in runs if r['key'] == key and r['knob'] == kb], key=lambda r: r['seed'])
        for r in g:
            npzc = []
            for c, d in cols:
                if not r['npz']:
                    npzc.append('')
                else:
                    v = r[c] / 1e6 if c == 'synops' else r[c]
                    npzc.append(f'{v:,.{d if c != "synops" else 1}f}')
            w(f"| {NAME[key]} | {kb} | s{r['seed']} | {r['server']} | {r['train_loss']:.4f} | {r['train_acc']:.2f} | {r['val_loss']:.4f} | {r['val_acc']:.2f} | "
              f"{r['spikes']:,.0f} | {'○' if BAND[0] <= r['spikes'] <= BAND[1] else ''} | " + ' | '.join(npzc) + ' |')
        if len(g) > 1:
            def ms(c, d, npz=False):
                v = np.array([r[c] for r in g if (r['npz'] or not npz)], float)
                if c == 'synops':
                    v = v / 1e6; d = 1
                if len(v) == 0:
                    return ''
                return f"**{v.mean():,.{d}f}** ±{v.std(ddof=1):,.{d}f}" if len(v) > 1 else f"**{v.mean():,.{d}f}**"
            nn = sum(r['npz'] for r in g)
            w(f"| **{NAME[key]} 평균** | {kb} | **n={len(g)}** (npz {nn}) | | {ms('train_loss', 4)} | {ms('train_acc', 2)} | {ms('val_loss', 4)} | {ms('val_acc', 2)} | {ms('spikes', 0)} | | "
              + ' | '.join(ms(c, d, True) for c, d in cols) + ' |')
w('\n## 7. 배제·npz 없음 (평균·회귀에서 뺌, 행은 남김)\n')
w('| 런 | 팔 | 서버 | spikes | val acc | 사유 | npz |')
w('|---|---|---|---|---|---|---|')
for e in J['excluded']:
    w(f"| {e['tag']} | {e['arm']} | {e['server']} | {e['spikes']:,.0f} | {e['val_acc']} | {e['reason']} | {'있음 (안 씀)' if e['npz'] else '없음'} |")
w(f"\n유효하지만 npz 가 없어 npz 지표에서 빠진 런 {len(J['no_npz'])}개 (원장 지표·시드 SD 에는 포함): " + ', '.join(f'`{t}`' for t in J['no_npz'])
  + '. 그중 1차 구간: `r19c10-ours_ch-3e-3-s4` (342K).\n')
w('## [LIMITATION]\n')
w('- **시간축은 재지 못했다.** npz 는 뉴런별로 (샘플×T) 합계와 "쏜 샘플 수" 만 저장한다. 타임스텝별 스파이크 분포, final_step 때문에 inter 가 마지막 스텝에 몰리는지, 첫 스파이크 시각은 이 자료로 답할 수 없다. 여기 "스텝/활성쌍" 은 발화 스텝 수의 평균일 뿐 어느 스텝인지 모른다. '
  '재려면 추출기 (`extract_neuron_spikes.py`) 에 스텝별 카운트와 로짓 저장을 더한 새 추론이 필요하다 (짝 6런 + baseline 3런, 런당 테스트 1만 장 1회). 이번 작업 범위 (읽기·집계) 밖이라 하지 않았다.')
w('- **ECE·클래스별 정확도도 같은 이유로 없다** (로짓 미저장). 원장에 있는 val loss 만 비교했다.')
w('- BPSR 은 1차 구간에 ρ 두 개 (3e-11 412K, 1e-10 319K) 뿐이다. 공통 기울기 ANCOVA 의 BPSR 쪽 기울기는 두 점 묶음으로 정해진다. 그래서 거의 같은 스파이크의 짝 비교 (n=3/3) 를 같이 냈다. 짝은 n=3 이라 죽은 채널·MACs 차는 짝만으로 유의하지 않다 (p=0.14, 0.08).')
w('- 분포 지표 상당수에서 기울기 차가 유의하다 — 차이는 스파이크가 낮을수록 커진다. 1차 c 는 그 구간 평균적 차이다.')
w('- SynOps 는 밀집 위치별 fan-out × 스파이크 수 근사다. 실제 하드웨어의 메모리 접근·채널 프루닝 이득은 측정하지 않았다. "MACs 감소 %" 는 죽은 채널을 지운 밀집 연산 기준 (제거 시 정확도 변화 0.00, 28런 실측, inter_deep F1).')
w('- 지표 수가 많다 (npz 30개 이상). 다중 비교 보정은 따로 하지 않았다. 판정 (우위·다름) 은 1차 CI 가 0 을 벗어나고 짝 비교의 부호가 같은 것만 적었고, 그 행들의 1차 p 는 모두 ≤2e-4 라 Bonferroni (지표 약 35개, 0.05/35 ≈ 1.4e-3) 를 적용해도 남는다. p 가 0.005~0.05 인 지표 (SynOps 감소율, conv3·conv4 몫) 는 판정에 쓰지 않았다.')
w('- 시드 SD 비교는 BPSR 자유도 8 (전 구간), 4 (1차) 로 작다. "BPSR 이 더 재현적" 은 스파이크 CV 에서만 1차 p=0.012 이고 정확도 SD 는 판단 불가.')
w('- 선택성 지표는 "쏜 이미지 비율" 하나로 쟀다. 클래스별 반응 (클래스 선택성) 은 표본별 기록이 없어 못 쟀다.')
w('- R19·CIFAR-10·T=4 한 설정이다. C100 은 BPSR 런이 없고 (결정 10-07: BPSR 비교는 R19 C10 에서만), 다른 모델로 일반화하지 않는다.')
open(OUT_MD, 'w').write('\n'.join(L) + '\n')
print('md', OUT_MD)
