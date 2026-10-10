"""경쟁 범위가 '스파이크가 어디에 남는지'를 바꾸는가 — 채널 단위 vs 뉴런 단위 (26-10-07).

질문 (양방향, 미리 정함)
  (a) ours_inter_ch (런 키 ours_ch) 가 채널 단위 지표에서 다른 팔보다 앞서는가
        - 죽은 채널 수 (테스트 1만 장에서 공간 전체가 한 번도 안 쏜 채널, /4,736)
        - 죽은 채널 제거로 줄어드는 MACs % (모든 스파이킹 층 = prune_measure.py 시나리오 b)
  (b) ours_intra_ch (런 키 ours_w1 = w¹, prop = w²) 가 뉴런 단위 지표에서 다른 팔보다 앞서는가
        - 이미지당 활성 뉴런 = Σ_층 Σ_뉴런 (그 뉴런이 쏜 표본 수) / 표본 수
          = 이미지 한 장에서 T 스텝 동안 한 번이라도 쏜 뉴런 수의 평균.
          스펙 A2 의 '활성 뉴런' (Σ N_l·(1 − dead_neuron_ratio_l), dead = 표본별 0 스파이크 비율) 과
          같은 정의를 테스트 모드 npz 로 잰 것이다 (A2 는 학습 모드 CSV).
      '앞선다' = 같은 스파이크에서 (a) 더 많이 죽이고 MACs 를 더 줄임 / (b) 활성 뉴런이 더 적음.

판정 규칙 (데이터 보기 전에 정함)
  * 모형: y = a_팔 + b·ln(spikes)  (공통 기울기 ANCOVA). 활성 뉴런은 스펙 A2 와 같게 ln(y).
  * 대상 팔 vs '나머지 팔 합동' 단일 대비의 95% CI 가 유리한 방향으로 0 을 벗어나고,
    개별 팔 대비의 부호가 모두 같은 방향이면 '성립'. 개별 대비는 Holm 보정 p 도 적는다.
  * 1차 구간 = 원장 스파이크 320~420K (주장 범위 −17~−35%), 2차 = 유효 npz 전 런.
    구간은 원장(eval s_count) 스파이크로 고르고, 회귀의 x 는 npz 테스트 스파이크 (지표와 같은 측정).
    원장 스파이크를 x 로 쓴 결과를 민감도로 함께 낸다.
  * baseline 은 회귀에서 빼고 기준 행으로만 둔다 (스파이크 한 점뿐, 구간 밖).
  * 팔: ours_inter_ch, ours_intra_ch (w¹), ours_intra_ch (w², 1차 구간엔 런 없음), ours_layer (w¹),
    plain L2, BPSR. ours_layer (w²) 는 민감도에만. Suetake 는 npz 없음. C100 은 npz 없음.

입력 (읽기만)
  원장  scratchpad/ledger/inventory.json (2026-10-06 16:14) — 310 에폭 완주·배제 아님만.
  npz   _npz/<tag>.npz + scratchpad/deep1005/npz/<tag>.npz
  가중치 층 사상  scratchpad/prune_out/*.json (MACs 계산; inter_deep_261005.py 와 같은 셈)

쓰는 법:  ~/anaconda3/envs/venv_1/bin/python analysis/struct_compare_261007.py
산출:     analysis/struct_compare_261007.json, analysis/struct_compare_261007.md
"""
import json, os, glob
from collections import defaultdict
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
INV = S + '/ledger/inventory.json'
NPZ_DIRS = [os.path.join(ROOT, '_npz'), S + '/deep1005/npz']
PRUNE_DIRS = [S + '/prune_out', S + '/deep1005/prune']
OUT_JSON = os.path.join(ROOT, 'analysis', 'struct_compare_261007.json')
OUT_MD = os.path.join(ROOT, 'analysis', 'struct_compare_261007.md')
SKIP_L = {'n_in', 'predictions_n'}
BAND1 = (320e3, 420e3)
X_FIX = 367e3            # 1차 구간 기하 중앙 (기울기를 따로 둘 때 대비를 내는 점)

NAME = {'ours_ch': 'ours_inter_ch', 'ours_w1': 'ours_intra_ch (w¹)', 'prop': 'ours_intra_ch (w²)',
        'ours_layer_w1': 'ours_layer (w¹)', 'ours_layer': 'ours_layer (w²)', 'l2': 'plain L2',
        'bpsr': 'BPSR', 'base': 'baseline'}
MAIN = ['ours_ch', 'ours_w1', 'prop', 'ours_layer_w1', 'l2', 'bpsr']
SENS = ['ours_layer']
INTER, INTRA1, INTRA2 = 'ours_ch', 'ours_w1', 'prop'


# ---- MACs (inter_deep_261005.py 와 같은 셈) -------------------------------------------
def weight_layers():
    for d in PRUNE_DIRS:
        for p in sorted(glob.glob(d + '/r19c10-*.json')):
            j = json.load(open(p))
            if 'weight_layers' in j:
                return j['weight_layers']
    raise SystemExit('weight_layers 없음')


WL = weight_layers()


def prune_cost(dead):
    pr = mr = pt = mt = 0
    for w in WL:
        ks = w['ks']; cin, cout = ks[-2], ks[-1]; kk = int(np.prod(ks[:-2])) if len(ks) > 2 else 1
        di = dead.get(w['src'], 0); do = dead.get(w['dst'], 0)
        rem = (di * cout + do * cin - di * do) * kk
        tot = cin * cout * kk
        pr += rem + (do if w['bias'] else 0); pt += tot + (cout if w['bias'] else 0)
        mr += rem * w['hw']; mt += tot * w['hw']
    return 100 * pr / pt, 100 * mr / mt


def npz_path(tag):
    for d in NPZ_DIRS:
        p = os.path.join(d, tag + '.npz')
        if os.path.exists(p):
            return p
    return None


def npz_metrics(p):
    z = np.load(p)
    ns = float(z['n_samples'])
    tot = spk = fired = sil = dead = nch = 0
    alive_n = alive_fired = 0           # 산 채널(및 fc 산 뉴런)에 속한 뉴런 수, 그 뉴런들의 이미지당 활성
    dead_by = {}
    for k in z.files:
        if k.endswith('__fired') or k + '__fired' not in z.files or k in SKIP_L:
            continue
        c = z[k].astype(np.float64); f = z[k + '__fired'].astype(np.float64)
        tot += c.size; spk += c.sum(); fired += f.sum(); sil += int((c == 0).sum())
        if c.ndim == 3:
            cc = c.reshape(-1, c.shape[-1]); ff = f.reshape(-1, f.shape[-1])
            alive = cc.sum(0) > 0
            d = int((~alive).sum()); nc = c.shape[-1]
            alive_n += int(alive.sum()) * cc.shape[0]; alive_fired += ff[:, alive].sum()
        else:
            alive = c > 0
            d = int((~alive).sum()); nc = c.size
            alive_n += int(alive.sum()); alive_fired += f[alive].sum()
        dead += d; nch += nc; dead_by[k] = d
    pa, ma = prune_cost({k: v for k, v in dead_by.items() if '_out_n' not in k})
    pb, mb = prune_cost(dead_by)
    return {'npz_spk': spk / ns, 'test_acc': float(z['acc']), 'neurons': tot, 'channels': nch,
            'active_img': fired / ns, 'active_pct': 100 * fired / ns / tot,
            'active_in_alive_pct': 100 * alive_fired / ns / alive_n,
            'silent_pct': 100 * sil / tot, 'dead_ch': dead,
            'macs_pct_b': mb, 'macs_pct_a': ma, 'params_pct_b': pb}


# ---- 런 수집 ---------------------------------------------------------------------------
inv_all = json.load(open(INV))
inv = {x['tag']: x for x in inv_all['runs']}
runs, excluded = [], []
for key in MAIN + SENS + ['base']:
    for tag, r in sorted(inv.items()):
        if r['ds'] != 'C10' or r['key'] != key or r['seed'] >= 900:
            continue
        p = npz_path(tag)
        ok = r['completed'] and r['epochs'] == 310 and not r['exclude']
        if not ok:
            if p is not None or (r.get('spikes') and BAND1[0] <= r['spikes'] <= BAND1[1]):
                excluded.append({'tag': tag, 'arm': NAME[key], 'server': r['server'], 'epochs': r['epochs'],
                                 'spikes': r.get('spikes'), 'val_acc': r.get('val_acc'),
                                 'reason': r.get('exclude_reason') or f"미완주({r['epochs']}ep)",
                                 'npz': p is not None})
            continue
        if p is None:
            continue
        m = npz_metrics(p)
        m.update(tag=tag, key=key, arm=NAME[key], knob=r['knob'], seed=r['seed'], server=r['server'],
                 **{k: r[k] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes', 's30')})
        runs.append(m)
no_npz_valid = sorted(t for t, r in inv.items() if r['ds'] == 'C10' and r['key'] in MAIN + SENS + ['suetake']
                      and r['completed'] and r['epochs'] == 310 and not r['exclude'] and r['seed'] < 900
                      and npz_path(t) is None)


# ---- 통계 -------------------------------------------------------------------------------
def ols(y, X):
    y = np.asarray(y, float); X = np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta; dof = len(y) - X.shape[1]
    s2 = res @ res / dof
    XtXi = np.linalg.inv(X.T @ X)
    return beta, s2 * XtXi, dof, res, XtXi


def holm(ps):
    o = np.argsort(ps); m = len(ps); out = [None] * m; run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (m - i) * ps[j])); out[j] = run
    return out


def ancova(rows, ycol, target, logy=False, xcol='npz_spk'):
    """y = a_팔 + b·ln(x). 대상 팔과 다른 팔 각각의 대비 (다른 − 대상), 대상 vs 나머지 합동 대비,
    기울기 이질성 F, 기울기를 따로 둔 모형의 X_FIX 점 대비."""
    arms = [target] + sorted({r['key'] for r in rows} - {target}, key=(MAIN + SENS).index)
    y = np.array([np.log(r[ycol]) if logy else r[ycol] for r in rows])
    lx = np.log([r[xcol] for r in rows])
    X = np.column_stack([np.ones(len(rows)), lx] + [[1.0 if r['key'] == a else 0.0 for r in rows] for a in arms[1:]])
    beta, cov, dof, res, _ = ols(y, X)
    tq = stats.t.ppf(0.975, dof)
    out = {'n': len(rows), 'n_by_arm': {NAME[a]: sum(r['key'] == a for r in rows) for a in arms},
           'dof': dof, 'b': [float(beta[1]), float(np.sqrt(cov[1, 1]))], 'resid_sd': float(np.sqrt(res @ res / dof)),
           'per_arm': {}}
    ps = []
    for i, a in enumerate(arms[1:]):
        j = i + 2; est = beta[j]; se = np.sqrt(cov[j, j]); p = 2 * stats.t.sf(abs(est / se), dof)
        out['per_arm'][NAME[a]] = {'est': float(est), 'se': float(se), 'lo': float(est - tq * se),
                                   'hi': float(est + tq * se), 'p': float(p)}
        ps.append(p)
    for (a, v), ph in zip(out['per_arm'].items(), holm(ps)):
        v['p_holm'] = float(ph)
    # 대상 vs 나머지 합동: 다른 팔 하나로 묶은 모형
    X2 = np.column_stack([np.ones(len(rows)), lx, [0.0 if r['key'] == target else 1.0 for r in rows]])
    b2, c2, d2, _, _ = ols(y, X2)
    se = np.sqrt(c2[2, 2]); t2 = stats.t.ppf(0.975, d2)
    out['pooled'] = {'est': float(b2[2]), 'se': float(se), 'lo': float(b2[2] - t2 * se), 'hi': float(b2[2] + t2 * se),
                     'p': float(2 * stats.t.sf(abs(b2[2] / se), d2)), 'dof': d2}
    # 기울기 이질성 + 기울기 따로 모형에서 X_FIX 대비 (다른 − 대상)
    Xs = np.column_stack([X] + [X[:, 1] * X[:, j] for j in range(2, X.shape[1])])
    bs, cs, ds_, rs, _ = ols(y, Xs)
    rss0, rss1 = res @ res, rs @ rs; q = Xs.shape[1] - X.shape[1]
    F = ((rss0 - rss1) / q) / (rss1 / ds_) if q > 0 and ds_ > 0 else np.nan
    out['slope_F'] = [float(F), float(stats.f.sf(F, q, ds_)) if np.isfinite(F) else None]
    L = np.log(X_FIX); k = X.shape[1] - 2; tq2 = stats.t.ppf(0.975, ds_)
    out['at_fix'] = {}
    xr = {a: (min(r[xcol] for r in rows if r['key'] == a), max(r[xcol] for r in rows if r['key'] == a)) for a in arms}
    for i, a in enumerate(arms[1:]):
        if not all(xr[b][0] <= X_FIX <= xr[b][1] for b in (target, a)):   # 외삽 안 함
            out['at_fix'][NAME[a]] = None; continue
        g = np.zeros(Xs.shape[1]); g[2 + i] = 1; g[2 + k + i] = L
        e = float(g @ bs); s = float(np.sqrt(g @ cs @ g))
        out['at_fix'][NAME[a]] = {'est': e, 'lo': e - tq2 * s, 'hi': e + tq2 * s, 'p': float(2 * stats.t.sf(abs(e / s), ds_))}
    if logy:      # 곱셈 인자로도
        for v in list(out['per_arm'].values()) + [out['pooled']]:
            v['ratio'] = [float(np.exp(v['est'])), float(np.exp(v['lo'])), float(np.exp(v['hi']))]
    return out


BANDS = {'1차 320-420K': lambda r: BAND1[0] <= r['spikes'] <= BAND1[1],
         '2차 전체': lambda r: True}
METRICS = {  # (열, log?, 대상 팔, 대상이 '앞선다'의 부호: 다른 − 대상 의 기대 부호)
    'dead_ch': ('dead_ch', False, INTER, -1),
    'macs_pct_b': ('macs_pct_b', False, INTER, -1),
    'active_img': ('active_img', True, INTRA1, +1),
}
# 같은 런 집합 2×2: 두 지표 × 두 대상 팔
reg = [r for r in runs if r['key'] in MAIN]
reg_s = [r for r in runs if r['key'] in MAIN + SENS]
out = {'generated_from': {'inventory': INV, 'inventory_generated': inv_all['generated'], 'npz_dirs': NPZ_DIRS},
       'definitions': __doc__, 'runs': runs, 'excluded': excluded, 'valid_without_npz': no_npz_valid,
       'analyses': {}}

for band, f in BANDS.items():
    rows = [r for r in reg if f(r)]
    A = out['analyses'][band] = {'n': len(rows), 'n_by_arm': {NAME[k]: sum(r['key'] == k for r in rows) for k in MAIN}}
    for target in (INTER, INTRA1, INTRA2):
        if not any(r['key'] == target for r in rows):
            continue
        for mname, (col, lg, _, _) in list(METRICS.items()) + [('active_in_alive_pct', ('active_in_alive_pct', False, None, 0))]:
            A[f'{mname}|target={NAME[target]}'] = ancova(rows, col, target, logy=lg)
            A[f'{mname}|target={NAME[target]}|x=원장'] = ancova(rows, col, target, logy=lg, xcol='spikes')
    rows_s = [r for r in reg_s if f(r)]
    if len(rows_s) > len(rows):
        for mname, (col, lg, tgt, _) in METRICS.items():
            A[f'{mname}|target={NAME[tgt]}|+ours_layer(w²)'] = ancova(rows_s, col, tgt, logy=lg)

# 스펙 A2 재현 확인: ln(활성) ~ ln(스파이크) + [prop] , prop vs plain L2 만, 전 구간
rr = [r for r in reg if r['key'] in ('prop', 'l2')]
out['spec_A2_check'] = ancova(rr, 'active_img', 'l2', logy=True)

# 그룹 요약 (팔·ρ)
grp = defaultdict(list)
for r in runs:
    grp[(r['key'], r['knob'])].append(r)
out['groups'] = []
for (k, kn), v in sorted(grp.items(), key=lambda kv: (([*MAIN, *SENS, 'base']).index(kv[0][0]), -np.mean([x['spikes'] for x in kv[1]]))):
    g = {'arm': NAME[k], 'knob': kn, 'n': len(v)}
    for c in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes', 'npz_spk', 'active_img', 'active_pct',
              'active_in_alive_pct', 'dead_ch', 'macs_pct_b', 'macs_pct_a'):
        a = np.array([x[c] for x in v], float)
        g[c] = [float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else None]
    out['groups'].append(g)

json.dump(out, open(OUT_JSON, 'w'), ensure_ascii=False, indent=1, default=float)
print('runs', len(runs), 'excluded', len(excluded), 'valid w/o npz', len(no_npz_valid))
for band, A in out['analyses'].items():
    print('==', band, A['n'], A['n_by_arm'])
    for k, v in A.items():
        if not isinstance(v, dict) or 'pooled' not in v or '원장' in k:
            continue
        pa = '; '.join(f"{a} {x['est']:+.4g} [{x['lo']:+.3g},{x['hi']:+.3g}] ph={x['p_holm']:.3f}" for a, x in v['per_arm'].items())
        print(f"{k}: pooled {v['pooled']['est']:+.4g} [{v['pooled']['lo']:+.3g},{v['pooled']['hi']:+.3g}] p={v['pooled']['p']:.3g} | slopeF p={v['slope_F'][1]} | {pa}")
print('A2', out['spec_A2_check']['per_arm'], out['spec_A2_check']['n_by_arm'])


# =====================================================================================
# 보고서 (md)
# =====================================================================================
def ci(v, d=2, pct=False):
    if v is None:
        return '범위 밖'
    if pct:   # ln 차이 -> %
        e, lo, hi = (100 * (np.exp(v[k]) - 1) for k in ('est', 'lo', 'hi'))
    else:
        e, lo, hi = v['est'], v['lo'], v['hi']
    ps = f"p={v['p']:.3f}" if v['p'] >= 0.001 else 'p<0.001'
    s = f"{e:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}] {ps}"
    if 'p_holm' in v:
        s += f" (Holm {v['p_holm']:.3f})"
    return s


def flip(v):   # '다른 − 대상' -> '대상 − 다른'
    if v is None:
        return None
    w = dict(v); w['est'], w['lo'], w['hi'] = -v['est'], -v['hi'], -v['lo']
    return w


B1, B2 = '1차 320-420K', '2차 전체'
A1, A2 = out['analyses'][B1], out['analyses'][B2]
L = []
w = L.append
w('# 경쟁 범위와 희소성의 구조: 채널 단위 vs 뉴런 단위 (R19-C10, 26-10-07)\n')
w(f"생성: `analysis/struct_compare_261007.py` → `.json`, 이 문서. 원장 {inv_all['generated']} 집계 (310 에폭 완주·S30/S1≥0.19·seed<900), "
  f"npz = 테스트 1만 장 1회 추론, 스파이킹 18층 (n_in·predictions_n 제외). **C100: npz 없음. Suetake: npz 없음** (C10 유효 런 n=2, 추출 안 됨).\n")
w('## 정의 (분석 전에 고정)\n')
w('- **죽은 채널**: 테스트 1만 장 전체에서 공간 모든 위치가 한 번도 안 쏜 채널 (/4,736 = conv 4,480 + fc1 256).')
w('- **MACs 제거 %**: 죽은 채널을 물리 제거했을 때 줄어드는 밀집 MACs 비율, 모든 스파이킹 층 (`prune_measure.py` 시나리오 b, `inter_deep_261005.py` 와 같은 셈). 제거해도 정확도 변화 0.00 (inter_deep F1, 실측 28런).')
w('- **이미지당 활성 뉴런**: 이미지 한 장에서 T 스텝 동안 한 번이라도 쏜 뉴런 수의 평균 = Σ(뉴런이 쏜 표본 수)/1만. 스펙 A2 의 활성 뉴런 (Σ N·(1−dead_neuron_ratio), 표본별 0 스파이크 비율) 과 같은 정의이고, 측정이 학습 모드 CSV 대신 테스트 모드 npz 다. 전체 뉴런 ' + f"{runs[0]['neurons']:,}" + ' 개 (모든 런 같음: ' + str(len({r['neurons'] for r in runs}) == 1) + ').')
w('- **산 채널 안 활성 %** (보조): 위 활성 뉴런을, 죽지 않은 채널(fc1 은 죽지 않은 뉴런) 에 속한 뉴런 수로 나눈 값. 채널 사망과 분리한 뉴런 단위 희소성. **판정에 쓰지 않는 보조 지표**다 (요청 지표가 아님).')
w('- 모형: `y = a_팔 + b·ln(스파이크)` 공통 기울기 ANCOVA. 활성 뉴런은 스펙 A2 와 같이 ln(y) → 결과를 % 로 환산. 스파이크 x 는 npz 테스트 스파이크 (지표와 같은 측정); 원장 스파이크를 x 로 쓴 결과는 민감도 절. 기울기가 팔마다 다를 수 있어 기울기를 따로 둔 모형의 367K (npz 축, 원장 약 340K) 대비도 낸다 — 두 팔 모두 그 스파이크를 덮을 때만.')
w('- **판정 규칙**: 대상 팔 vs 나머지 팔 합동 대비의 95% CI 가 유리한 방향으로 0 을 벗어나고, 개별 팔 대비의 점추정 부호가 전부 같은 방향이면 성립. 구간: **1차 = 원장 스파이크 320~420K** (주장 범위), 2차 = 유효 npz 전 런. baseline 은 회귀에서 빼고 기준 행으로만.\n')

# ---- 결론
ia = A1['macs_pct_b|target=ours_inter_ch']; idc = A1['dead_ch|target=ours_inter_ch']
na = A1['active_img|target=ours_intra_ch (w¹)']; nai = A1['active_img|target=ours_inter_ch']
ni2 = A2['active_img|target=ours_intra_ch (w²)']
w('## 0. 결론\n')
w(f"1. **(a) 성립.** 1차 구간 (n={A1['n']}: inter {A1['n_by_arm']['ours_inter_ch']}) 에서 같은 스파이크일 때 ours_inter_ch 는 나머지 팔 합동보다 "
  f"죽은 채널 {ci(flip(idc['pooled']), 1)} 개, MACs 제거 {ci(flip(ia['pooled']))} %p 더 많다. 개별 네 팔 모두 같은 부호. "
  f"단 ours_layer (w¹) 와는 공통 기울기 모형에서 구분되지 않는다 (죽은 채널 {ci(flip(idc['per_arm']['ours_layer (w¹)']), 1)}, MACs {ci(flip(ia['per_arm']['ours_layer (w¹)']))}); "
  f"기울기를 따로 둔 367K 대비는 MACs {ci(flip(ia['at_fix']['ours_layer (w¹)']))}. 2차 전체 (n={A2['n']}) 에서는 여섯 팔 전부와 Holm 후 유의.")
w(f"2. **(b) 불성립.** 1차 구간에서 ours_intra_ch (w¹) 의 이미지당 활성 뉴런은 나머지 팔 합동 대비 {ci(flip(na['pooled']), 1, pct=True)} % (음수 = intra 가 적음) 로 0 을 포함한다. "
  f"plain L2 ({ci(flip(na['per_arm']['plain L2']), 1, pct=True)} %)·BPSR ({ci(flip(na['per_arm']['BPSR']), 1, pct=True)} %) 보다는 적지만, "
  f"ours_inter_ch 보다는 오히려 {ci(na['per_arm']['ours_inter_ch'] and flip(na['per_arm']['ours_inter_ch']), 1, pct=True)} % (양수 = intra 가 많음) 이고 ours_layer (w¹) 와 같다. "
  f"기울기를 따로 둔 367K 에서는 intra 가 inter 보다 {ci(flip(na['at_fix']['ours_inter_ch']), 1, pct=True)} % 많다. "
  f"원래 제안법 w² (2차 전체만, n={A2['n_by_arm']['ours_intra_ch (w²)']}) 도 inter 와 {ci(flip(ni2['per_arm']['ours_inter_ch']), 1, pct=True)} % 로 같다.")
w(f"3. **이미지당 활성 뉴런이 가르는 것은 경쟁 범위가 아니라 'maxnorm 가중 계열 (inter·intra·layer) vs plain L2·BPSR' 이다.** 세 경쟁 범위 팔은 서로 1% 안이고, 셋 다 plain L2 보다 5~7%, BPSR 보다 4~7% 적다 (1차, 공통 기울기·367K 두 방식). "
  f"ours_inter_ch 를 대상으로 두면 inter 도 나머지 합동보다 {ci(flip(nai['pooled']), 1, pct=True)} % 적다 — 즉 inter 는 채널 지표와 뉴런 지표 **둘 다** 앞서거나 같다.")
a2 = out['spec_A2_check']['per_arm']['ours_intra_ch (w²)']
w(f"4. **스펙 A2 (−13.1%) 는 npz 테스트 모드로 재현된다:** ours_intra_ch (w²) vs plain L2 만 넣으면 {ci(a2, 1, pct=True)} % "
  f"(n={sum(out['spec_A2_check']['n_by_arm'].values())}: 각 9). 그러나 같은 정의로 inter 를 넣으면 w² 와 차이가 없으므로 (2번) **A2 는 intra 고유 성질이 아니다.**")
nb = A1['active_in_alive_pct|target=ours_intra_ch (w¹)']
w(f"5. 보조 지표 (판정 외): **산 채널 안 활성 %** 는 intra (w¹) 가 가장 낮다. 1차 나머지 합동 대비 {ci(flip(nb['pooled']))} %p, 개별 네 팔 전부 같은 부호, Holm 후 전부 p<0.05 "
  f"(inter {ci(flip(nb['per_arm']['ours_inter_ch']))}). 즉 '살아남은 채널 안에서 덜 쏜다' 는 intra 쪽 분리는 보인다. "
  f"다만 이 지표는 미리 정한 판정 지표가 아니고, 분모가 채널 사망에 따라 바뀌어 inter 쪽은 기계적으로 높아진다 (약한 채널이 빠지면 남은 채널의 활성 비율이 오른다). "
  f"plain L2·BPSR 은 intra 보다 채널을 덜 죽였는데도 이 값이 더 높으므로 (+0.8%p) 기계적 효과만은 아니다. 논문 근거로 쓰려면 이 지표를 사전 등록하고 별도 런 집합으로 확인해야 한다.")
w('6. **요지 문장 판정:** "경쟁 범위가 희소성의 구조를 고른다" 는 이 결과로 **쓸 수 없다.** 쓸 수 있는 것은 한쪽 — "채널끼리 경쟁(inter) 은 같은 스파이크에서 채널을 더 죽여 MACs 를 나머지 팔 합동보다 1.8%p (1차, n=31) 더 줄이고, 이미지당 활성 뉴런은 다른 경쟁 범위 팔보다 많지 않다" 이다.\n')

# ---- 2x2
w('## 1. 2×2: 지표 × 대상 팔 (같은 런 집합, 1차 320~420K)\n')
w(f"같은 {A1['n']} 런 (inter {A1['n_by_arm']['ours_inter_ch']}, intra (w¹) {A1['n_by_arm']['ours_intra_ch (w¹)']}, layer (w¹) {A1['n_by_arm']['ours_layer (w¹)']}, plain L2 {A1['n_by_arm']['plain L2']}, BPSR {A1['n_by_arm']['BPSR']}). "
  "칸 = 대상 팔 − 나머지 팔 합동, 같은 스파이크 (공통 기울기). 유리한 방향: 채널 지표는 +, 활성 뉴런은 −.\n")
w('| 대상 팔 | 죽은 채널 (개) | MACs 제거 (%p) | 이미지당 활성 뉴런 (%) | (보조) 산 채널 안 활성 (%p) |')
w('|---|---|---|---|---|')
for tname, tkey in (('ours_inter_ch', 'ours_inter_ch'), ('ours_intra_ch (w¹)', 'ours_intra_ch (w¹)')):
    w(f"| {tname} | {ci(flip(A1[f'dead_ch|target={tkey}']['pooled']), 1)} | {ci(flip(A1[f'macs_pct_b|target={tkey}']['pooled']))} | "
      f"{ci(flip(A1[f'active_img|target={tkey}']['pooled']), 1, pct=True)} | {ci(flip(A1[f'active_in_alive_pct|target={tkey}']['pooled']))} |")
d_ = A1['macs_pct_b|target=ours_inter_ch']['per_arm']['ours_intra_ch (w¹)']
w(f"| **inter − intra (w¹) 직접** | {ci(flip(A1['dead_ch|target=ours_inter_ch']['per_arm']['ours_intra_ch (w¹)']), 1)} | {ci(flip(d_))} | "
  f"{ci(flip(A1['active_img|target=ours_inter_ch']['per_arm']['ours_intra_ch (w¹)']), 1, pct=True)} | {ci(flip(A1['active_in_alive_pct|target=ours_inter_ch']['per_arm']['ours_intra_ch (w¹)']))} |")
w('\n이중 분리(double dissociation)라면 마지막 줄이 "채널 +, 활성 뉴런 +" 여야 한다 (inter 가 채널은 더 죽이고 활성 뉴런은 더 많음). 실제는 채널 +, 활성 뉴런 −(점추정) — inter 가 두 지표 모두 앞서거나 같다. 보조 지표에서만 부호가 갈린다.\n')

# ---- 상세 표
LAB = {'dead_ch': ('죽은 채널 (개)', 1, False), 'macs_pct_b': ('MACs 제거 (%p)', 2, False),
       'active_img': ('이미지당 활성 뉴런 (%)', 1, True), 'active_in_alive_pct': ('(보조) 산 채널 안 활성 (%p)', 2, False)}
for band, A in ((B1, A1), (B2, A2)):
    w(f"## {'2' if band == B1 else '3'}. {band} — 대상 팔 − 다른 팔 (같은 스파이크)\n")
    w(f"n={A['n']} (" + ', '.join(f'{k} {v}' for k, v in A['n_by_arm'].items()) + '). 각 칸: 공통 기울기 추정 [95% CI] p (Holm) / 367K 대비 (기울기 따로).\n')
    for tkey in ('ours_inter_ch', 'ours_intra_ch (w¹)', 'ours_intra_ch (w²)'):
        keys = [m for m in LAB if f'{m}|target={tkey}' in A]
        if not keys:
            continue
        others = list(A[f'{keys[0]}|target={tkey}']['per_arm'])
        w(f"### 대상 = {tkey}\n")
        w('| 지표 | b (ln 스파이크) | 기울기 차이 F p | 나머지 합동 | ' + ' | '.join(others) + ' |')
        w('|---|---|---|---|' + '---|' * len(others))
        for m in keys:
            v = A[f'{m}|target={tkey}']; lab, d, pc = LAB[m]
            cells = []
            for o in others:
                fx = v['at_fix'][o]
                cells.append(f"{ci(flip(v['per_arm'][o]), d, pc)} / 367K {ci(flip(fx), d, pc) if fx else '범위 밖'}")
            w(f"| {lab} | {v['b'][0]:.3g} ±{v['b'][1]:.2g} | {v['slope_F'][1]:.3g} | {ci(flip(v['pooled']), d, pc)} | " + ' | '.join(cells) + ' |')
        w('')
    sens = [k for k in A if k.endswith('|+ours_layer(w²)')]
    if sens:
        w('민감도 — ours_layer (w²) (n=5, 4e-3 한 ρ, 원장 190~202K) 를 팔로 추가:\n')
        for k in sens:
            v = A[k]; m = k.split('|')[0]; lab, d, pc = LAB[m]
            w(f"- {lab}, 대상 {k.split('|')[1][7:]}: 나머지 합동 {ci(flip(v['pooled']), d, pc)}; vs ours_layer (w²) {ci(flip(v['per_arm']['ours_layer (w²)']), d, pc)}")
        w('')

w('## 4. 민감도: 원장 스파이크(eval s_count)를 x 로\n')
w('npz/원장 스파이크 비가 팔마다 1.05~1.11 로 다르다 (inter_deep H3). x 를 원장으로 바꾼 공통 기울기 결과 (대상 − 나머지 합동):\n')
w('| 구간 | 대상 | 죽은 채널 | MACs 제거 %p | 이미지당 활성 뉴런 % |')
w('|---|---|---|---|---|')
for band, A in ((B1, A1), (B2, A2)):
    for tkey in ('ours_inter_ch', 'ours_intra_ch (w¹)', 'ours_intra_ch (w²)'):
        if f'dead_ch|target={tkey}|x=원장' not in A:
            continue
        w(f"| {band} | {tkey} | {ci(flip(A[f'dead_ch|target={tkey}|x=원장']['pooled']), 1)} | {ci(flip(A[f'macs_pct_b|target={tkey}|x=원장']['pooled']))} | {ci(flip(A[f'active_img|target={tkey}|x=원장']['pooled']), 1, pct=True)} |")
w('\n결론 같음.\n')

w('## 5. 스펙 A2 재현 (같은 정의, 테스트 모드)\n')
w(f"`ln(활성) ~ ln(스파이크) + [w²]`, ours_intra_ch (w²) vs plain L2 만, 전 구간: 곱셈 인자 {a2['ratio'][0]:.4f} [{a2['ratio'][1]:.4f}, {a2['ratio'][2]:.4f}] "
  f"= {100*(a2['ratio'][0]-1):+.1f}%, n=18 (각 9). 스펙 A2 (학습 모드 CSV): 0.8692, −13.1%, n=21. 정의가 같은 양을 재고 있다는 확인이다. "
  f"단 런 집합이 다르다 (A2 는 09-14 CSV 세트).\n")

# ---- 런 표
w('## 6. 런 개별값 (팔·ρ 묶음, 묶음 아래 평균 ± SD)\n')
w('train/val 열과 spikes 는 원장 (베스트 에폭), 이하 열은 npz. `1차` = 원장 스파이크 320~420K 포함 여부.\n')
w('| 방법 | ρ | 시드 | 서버 | train loss | train acc | val loss | val acc | spikes | 1차 | npz 스파이크/샘플 | test acc | 이미지당 활성 뉴런 | 활성 % | 산 채널 안 활성 % | 죽은 채널 | MACs 제거 % (b) | MACs 제거 % (a) |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for g in out['groups']:
    rs = sorted([r for r in runs if r['arm'] == g['arm'] and r['knob'] == g['knob']], key=lambda r: r['seed'])
    for r in rs:
        inb = '○' if BAND1[0] <= r['spikes'] <= BAND1[1] else ''
        w(f"| {r['arm']} | {r['knob']} | s{r['seed']} | {r['server']} | {r['train_loss']:.4f} | {r['train_acc']:.2f} | {r['val_loss']:.4f} | {r['val_acc']:.2f} | {r['spikes']:,.0f} | {inb} | "
          f"{r['npz_spk']:,.0f} | {r['test_acc']:.2f} | {r['active_img']:,.0f} | {r['active_pct']:.2f} | {r['active_in_alive_pct']:.2f} | {r['dead_ch']} | {r['macs_pct_b']:.1f} | {r['macs_pct_a']:.1f} |")
    if g['n'] > 1:
        def f(c, d=2, comma=False):
            m, s = g[c]
            return (f"**{m:,.0f}** ±{s:,.0f}" if comma else f"**{m:.{d}f}** ±{s:.{d}f}")
        w(f"| **{g['arm']} 평균** | {g['knob']} | **n={g['n']}** | | {f('train_loss', 4)} | {f('train_acc')} | {f('val_loss', 4)} | {f('val_acc')} | {f('spikes', comma=True)} | | "
          f"{f('npz_spk', comma=True)} | | {f('active_img', comma=True)} | {f('active_pct')} | {f('active_in_alive_pct')} | {f('dead_ch', 1)} | {f('macs_pct_b', 1)} | {f('macs_pct_a', 1)} |")

w('\n## 7. 배제·미완주 (평균·회귀에서 뺌, 행은 남김)\n')
w('| 런 | 팔 | 서버 | 에폭 | spikes | val acc | 사유 | npz |')
w('|---|---|---|---|---|---|---|---|')
for e in excluded:
    sp = f"{e['spikes']:,.0f}" if e['spikes'] else '–'
    w(f"| {e['tag']} | {e['arm']} | {e['server']} | {e['epochs']} | {sp} | {e['val_acc']} | {e['reason']} | {'있음' if e['npz'] else '없음'} |")
w(f"\n유효하지만 npz 가 없어 이 분석에 못 들어간 런 {len(no_npz_valid)}개: " + ', '.join(f'`{t}`' for t in no_npz_valid)
  + '. 그중 원장 스파이크가 1차 구간(320~420K)에 드는 것: '
  + ', '.join(f"`{t}` ({inv[t]['spikes']/1e3:.0f}K)" for t in no_npz_valid if BAND1[0] <= inv[t]['spikes'] <= BAND1[1])
  + ' — 추출하면 1차 n 이 늘어난다 (GPU 추론 필요, 이번엔 하지 않음).\n')

w('## [LIMITATION]\n')
w('- **1차 구간에 원래 제안법 w² (`prop`) 런이 없다** (prop 은 원장 163~296K). 1차의 intra 판정은 w¹ (n=5) 로만 했다. w² 는 2차 전체에서만 보았고 inter 와 활성 뉴런 차이가 없었다 (n=9 대 19).')
w('- 1차 구간의 plain L2·BPSR 은 각 n=4, intra (w¹) n=5. 개별 팔 대비의 검정력은 낮다. 판정은 합동 대비에 기댄다.')
w('- 이미지당 활성 뉴런은 테스트 모드 npz 다. 스펙 A2 는 학습 모드 CSV 라 절대값은 다르다 (같은 정의, 5절에서 비율 재현).')
w('- 활성 뉴런 회귀에서 팔별 기울기가 다르다 (기울기 차이 F p<0.001, 1차·2차). 공통 기울기 오프셋은 평균적 차이이고, 367K 대비가 그것을 보완한다. 두 방식의 부호는 같다.')
w('- 죽은 채널은 테스트셋 판정이다 (학습셋 판정과 28/28 런에서 같았음, inter_deep F1). MACs 는 타임스텝 1회분 밀집 연산 기준이고 스파이크 희소성에 따른 실제 연산량이 아니다.')
w('- npz 스파이크 총합과 `s_count` 의 불일치(결정 기록 09-15) 는 해소되지 않았다. 두 축 모두로 계산해 결론이 같음만 확인했다.')
w('- 보조 지표(산 채널 안 활성 %) 의 intra 우위는 사후에 판정 근거로 승격하면 안 된다. 미리 정한 판정 지표는 이미지당 활성 뉴런이다.')
w('- CIFAR-10·ResNet19·T=4 밖으로 일반화하지 않는다. C100·Suetake 는 npz 가 없어 판단 불가.')

open(OUT_MD, 'w').write('\n'.join(L) + '\n')
print('md ->', OUT_MD)
