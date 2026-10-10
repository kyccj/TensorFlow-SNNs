"""ours_inter_ch 실험 결과 정리 (26-10-09).

입력: 원장 inventory.json (TF R19 C10/C100 + VGGSNN, 10-09 12:56 집계), qk.json, sdt.json,
      QKFormer 런 폴더(summary.csv, spikes_*.json) 직접 확인.
산출: analysis/inter_results_261009.json (표 데이터 + 통계). md 는 이 json 을 바탕으로 손으로 쓴다.

쓰는 법:  python analysis/inter_results_261009.py <led 폴더>
"""
import csv, json, math, sys
from pathlib import Path
import numpy as np
from scipy import stats

LED = Path(sys.argv[1])
OUT = Path(__file__).with_suffix('.json')
inv = json.load(open(LED / 'inventory.json'))
R = inv['runs']
F = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes', 's30']


def kf(k):
    return float(k.replace('p', '.'))


def rows(ds, key):
    return [r for r in R if r['ds'] == ds and r['key'] == key]


def ok(r):
    return r['status'] == 'done' and r.get('completed') and not r.get('exclude') and r['seed'] < 900


def row_out(r):
    return {k: r.get(k) for k in ['tag', 'ds', 'arm', 'knob', 'seed', 'server', 'status', 'epochs'] + F +
            ['exclude_reason']}


def grp_stats(rs):
    o = {'n': len(rs)}
    for k in F:
        v = [r[k] for r in rs if r.get(k) is not None]
        o[k] = float(np.mean(v)) if v else None
        o[k + '_sd'] = float(np.std(v, ddof=1)) if len(v) > 1 else None
    return o


def pooled_sd(groups):
    ss, dof = 0.0, 0
    for g in groups:
        v = [r['val_acc'] for r in g]
        if len(v) > 1:
            ss += np.var(v, ddof=1) * (len(v) - 1)
            dof += len(v) - 1
    return math.sqrt(ss / dof), dof


def curve_table(ds, key, base):
    """ρ 마다 개별 + 평균. baseline 대비 spikes %, Δacc (Welch SE; n=1 은 합동 시드 SD)."""
    allr = rows(ds, key)
    knobs = sorted({r['knob'] for r in allr}, key=kf)
    groups = {k: [r for r in allr if r['knob'] == k and ok(r)] for k in knobs}
    sp, sp_dof = pooled_sd([g for g in groups.values()] + [base])
    bs = grp_stats(base)
    out = []
    for k in knobs:
        g = groups[k]
        e = {'knob': k, 'runs': [row_out(r) for r in sorted(allr, key=lambda r: r['seed'])
                                 if r['knob'] == k and r['seed'] < 900],
             'mean': grp_stats(g) if g else None}
        if g:
            m = e['mean']
            sd_g = m['val_acc_sd'] if m['n'] > 1 else sp
            se = math.sqrt(sd_g ** 2 / m['n'] + bs['val_acc_sd'] ** 2 / bs['n'])
            e['spikes_pct_vs_base'] = 100 * (m['spikes'] / bs['spikes'] - 1)
            e['dacc_vs_base'] = m['val_acc'] - bs['val_acc']
            e['dacc_se'] = se
        out.append(e)
    return {'baseline': {'runs': [row_out(r) for r in sorted(base, key=lambda r: r['seed'])], 'mean': bs},
            'groups': out, 'pooled_seed_sd': sp, 'pooled_seed_sd_dof': sp_dof}


# ---------------------------------------------------------------- C10 / C100 곡선
b10 = [r for r in rows('C10', 'base') if ok(r)]
b100 = [r for r in rows('C100', 'base') if ok(r)]
c10 = curve_table('C10', 'ours_ch', b10)
c100 = curve_table('C100', 'ours_ch', b100)


def fit_curve(runs, lo, hi):
    x = np.log([r['spikes'] for r in runs])
    y = np.array([r['val_acc'] for r in runs])
    X = np.column_stack([np.ones_like(x), x])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    res = y - X @ beta
    dof = len(y) - 2
    s2 = res @ res / dof
    return {'beta': beta, 'cov': XtXi * s2, 's': math.sqrt(s2), 'n': len(y), 'dof': dof,
            'lo': lo, 'hi': hi, 'xmin': float(min(r['spikes'] for r in runs)),
            'xmax': float(max(r['spikes'] for r in runs))}


def band(ds, key, lo, hi):
    return [r for r in rows(ds, key) if ok(r) and lo <= r['spikes'] <= hi]


def delta(fit, r):
    x = np.array([1.0, math.log(r['spikes'])])
    pred = float(x @ fit['beta'])
    se_fit = math.sqrt(x @ fit['cov'] @ x)
    se = math.sqrt(se_fit ** 2 + fit['s'] ** 2)     # 새 런 하나: 곡선 불확실성 + 시드 산포
    d = r['val_acc'] - pred
    p = 2 * stats.t.sf(abs(d / se), fit['dof'])
    inr = fit['lo'] <= r['spikes'] <= fit['hi']
    return {'pred': pred, 'delta': d, 'se': se, 'z': d / se, 'p': p, 'in_range': inr}


def mean_delta(fit, rs):
    """여러 런의 Δ 평균. 분산 = s²/n (런 산포) + 평균 예측의 곡선 분산 (상관 포함)."""
    X = np.column_stack([np.ones(len(rs)), np.log([r['spikes'] for r in rs])])
    xb = X.mean(0)
    d = float(np.mean([r['val_acc'] for r in rs]) - xb @ fit['beta'])
    se = math.sqrt(fit['s'] ** 2 / len(rs) + xb @ fit['cov'] @ xb)
    return {'n': len(rs), 'delta': d, 'se': se, 'z': d / se, 'p': 2 * stats.t.sf(abs(d / se), fit['dof'])}


fit10 = fit_curve(band('C10', 'ours_ch', 200e3, 420e3), 200e3, 420e3)
# 민감도: 10-07 과 같은 '300K 이상' 만 / 2차항
fit10_hi = fit_curve(band('C10', 'ours_ch', 280e3, 420e3), 280e3, 420e3)

ABL = [('ch_novmem', '- vmem'), ('ch_nofinal', '- final_step'), ('ch_nolr', '- loss-ratio'),
       ('ch_noinv', '- 1- inversion')]
DER = [('ours_ch_st30', 'ours_inter_ch st30 (규제 30에폭 늦게 시작)'),
       ('ours_ch_st60', 'ours_inter_ch st60 (60에폭 늦게 시작)'),
       ('ours_ch_taylor', 'ours_inter_ch Taylor (채널 서열 = Taylor 중요도)'),
       ('ours_ch_end40', 'ours_inter_ch end40 (마지막 40에폭 규제 해제)'),
       ('ours_chxwc', 'ours_inter_ch × intra (가중 = inter·intra 곱)'),
       ('ours_ch_b7', 'ours_inter_ch β=0.7 (1등 채널도 w=0.3)'),
       ('ours_ch_b4', 'ours_inter_ch β=0.4 (1등 채널도 w=0.6)')]


def arm_block(ds, specs, fit, fit2):
    out = []
    for key, name in specs:
        allr = sorted(rows(ds, key), key=lambda r: (kf(r['knob']), r['seed']))
        rs = []
        for r in allr:
            if r['seed'] >= 900:
                continue
            e = row_out(r)
            if ok(r):
                e['vs_inter'] = delta(fit, r)
                if fit2:
                    e['vs_inter_hi'] = delta(fit2, r)
            rs.append(e)
        good = [r for r in allr if ok(r)]
        blk = {'key': key, 'name': name, 'runs': rs}
        if len(good) > 1:
            blk['pooled'] = mean_delta(fit, good)
        # 같은 ρ 반복 그룹 평균
        gm = {}
        for k in sorted({r['knob'] for r in good}, key=kf):
            g = [r for r in good if r['knob'] == k]
            if len(g) > 1:
                gm[k] = {**grp_stats(g), 'vs_inter': mean_delta(fit, g)}
        blk['group_means'] = gm
        out.append(blk)
    return out


abl = arm_block('C10', ABL, fit10, fit10_hi)
der = arm_block('C10', DER, fit10, fit10_hi)
fit100 = fit_curve(band('C100', 'ours_ch', 280e3, 420e3), 280e3, 420e3)
der100 = arm_block('C100', [('ours_ch_st30', 'ours_inter_ch st30'), ('ours_ch_taylor', 'ours_inter_ch Taylor')],
                   fit100, None)

# ---------------------------------------------------------------- BPSR 비교 (C10)
inter_ok = [r for r in rows('C10', 'ours_ch') if ok(r)]
bpsr_ok = [r for r in rows('C10', 'bpsr') if ok(r)]


def ancova(a, b, lo, hi, nboot=4000, seed=0):
    """acc = a0 + b·ln(spikes) + c·[inter]. 같은 스파이크에서 inter−BPSR 정확도 차 c.
    같은 정확도에서 스파이크 비 = exp(−c/b); CI 는 런 단위 부트스트랩 (팔 안 복원추출)."""
    A = [r for r in a if lo <= r['spikes'] <= hi]
    B = [r for r in b if lo <= r['spikes'] <= hi]

    def one(A, B):
        y = np.array([r['val_acc'] for r in A + B])
        x = np.log([r['spikes'] for r in A + B])
        d = np.array([1.0] * len(A) + [0.0] * len(B))
        X = np.column_stack([np.ones_like(x), x, d])
        XtXi = np.linalg.inv(X.T @ X)
        be = XtXi @ X.T @ y
        res = y - X @ be
        dof = len(y) - 3
        s2 = res @ res / dof
        return be, np.sqrt(np.diag(XtXi) * s2), dof
    be, se, dof = one(A, B)
    rng = np.random.default_rng(seed)
    cs, rat = [], []
    for _ in range(nboot):
        Ab = [A[i] for i in rng.integers(0, len(A), len(A))]
        Bb = [B[i] for i in rng.integers(0, len(B), len(B))]
        try:
            bb, _, _ = one(Ab, Bb)
        except np.linalg.LinAlgError:
            continue
        cs.append(bb[2])
        rat.append(math.exp(-bb[2] / bb[1]) if bb[1] > 1e-9 else np.nan)
    rat = np.array(rat)
    fin = rat[np.isfinite(rat)]
    return {'band': [lo, hi], 'n_inter': len(A), 'n_bpsr': len(B), 'slope': be[1], 'slope_se': se[1],
            'slope_p': 2 * stats.t.sf(abs(be[1] / se[1]), dof),
            'c': be[2], 'c_se': se[2], 'c_p': 2 * stats.t.sf(abs(be[2] / se[2]), dof),
            'c_boot95': [float(np.percentile(cs, 2.5)), float(np.percentile(cs, 97.5))],
            'spike_ratio_same_acc': math.exp(-be[2] / be[1]) if be[1] > 0 else None,
            'spike_ratio_boot95': [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))],
            'boot_slope_nonpos_frac': float(np.mean(~np.isfinite(rat)))}


bpsr_cmp = {
    'ancova_300_420': ancova(inter_ok, bpsr_ok, 300e3, 420e3),
    'ancova_200_420': ancova(inter_ok, bpsr_ok, 200e3, 420e3),
    'ancova_180_300': ancova(inter_ok, bpsr_ok, 180e3, 300e3),
}
bp_groups = {}
for k in sorted({r['knob'] for r in bpsr_ok}, key=kf):
    g = [r for r in bpsr_ok if r['knob'] == k]
    bp_groups[k] = grp_stats(g)
bpsr_cmp['bpsr_groups'] = bp_groups
bpsr_cmp['bpsr_runs'] = [row_out(r) for r in sorted(rows('C10', 'bpsr'), key=lambda r: (kf(r['knob']), r['seed']))]
# 직접 비교: inter 3e-3 (n=4) vs BPSR 3e-11 (n=3) — 정확도가 거의 같은 두 점
i3 = [r for r in inter_ok if r['knob'] == '3e-3']
b311 = [r for r in bpsr_ok if r['knob'] == '3e-11']
bm = grp_stats(b311); im = grp_stats(i3)
# BPSR 곡선(ρ 평균, ln 축 선형 보간)에서 정확도 96.72 가 되는 스파이크
pts = sorted([(v['spikes'], v['val_acc']) for v in bp_groups.values()])
target = im['val_acc']
interp = None
for (s0, a0), (s1, a1) in zip(pts, pts[1:]):
    if min(a0, a1) <= target <= max(a0, a1) and a1 != a0:
        t = (target - a0) / (a1 - a0)
        interp = math.exp(math.log(s0) + t * (math.log(s1) - math.log(s0)))
bpsr_cmp['direct'] = {
    'inter_3e-3': im, 'bpsr_3e-11': bm,
    'spike_ratio': im['spikes'] / bm['spikes'],
    'dacc': im['val_acc'] - bm['val_acc'],
    'dacc_se': math.sqrt(im['val_acc_sd'] ** 2 / im['n'] + bm['val_acc_sd'] ** 2 / bm['n']),
    'bpsr_spikes_at_inter_acc_interp': interp,
    'spike_ratio_interp': im['spikes'] / interp if interp else None,
}


def interp_ratio(I, Bg):
    t_acc = np.mean([r['val_acc'] for r in I])
    pts = sorted((np.mean([r['spikes'] for r in g]), np.mean([r['val_acc'] for r in g])) for g in Bg)
    for (s0, a0), (s1, a1) in zip(pts, pts[1:]):
        if min(a0, a1) <= t_acc <= max(a0, a1) and a1 != a0:
            t = (t_acc - a0) / (a1 - a0)
            return np.mean([r['spikes'] for r in I]) / math.exp(math.log(s0) + t * (math.log(s1) - math.log(s0)))
    return np.nan     # BPSR 곡선 범위 밖 (가장 높은 BPSR 점보다 inter 가 정확하면 '비교 불가')


rng = np.random.default_rng(1)
Bg0 = [[r for r in bpsr_ok if r['knob'] == k] for k in bp_groups]
bs = []
for _ in range(4000):
    I = [i3[j] for j in rng.integers(0, len(i3), len(i3))]
    Bg = [[g[j] for j in rng.integers(0, len(g), len(g))] for g in Bg0]
    bs.append(interp_ratio(I, Bg))
bs = np.array(bs)
bpsr_cmp['direct']['spike_ratio_interp_boot95_finite'] = [float(np.nanpercentile(bs, 2.5)),
                                                          float(np.nanpercentile(bs, 97.5))]
bpsr_cmp['direct']['interp_out_of_range_frac'] = float(np.mean(~np.isfinite(bs)))

# ---------------------------------------------------------------- 진행 중 / 배제
running = [row_out(r) for r in R if r['status'] == 'running' and r['key'] in ('ours_ch', 'base')
           and r['ds'] in ('C100', 'DVS')]
excluded = [row_out(r) for r in R if r['key'] in {'ours_ch', 'base'} | {k for k, _ in ABL + DER}
            | {'ours_ch_st30', 'ours_ch_taylor'} and r['ds'] in ('C10', 'C100', 'DVS')
            and r['status'] in ('excluded', 'stale', 'done') and (r.get('exclude') or r['seed'] >= 900)]

# ---------------------------------------------------------------- QKFormer
QR = Path('/home/kyccj/runs/qkformer_wta_rev')
qk = json.load(open(LED / 'qk.json'))
qk_base_best = [j['acc'] for j in qk['jobs'] if j['name'].startswith('baseline') and j['status'] == 'done'
                and j['epochs'] == 410]


def summ(name):
    rs = [r for r in csv.DictReader(open(QR / name / 'summary.csv')) if r['epoch'] != 'epoch']
    b = max(rs, key=lambda r: float(r['eval_top1']))
    return {'epochs_logged': len(rs), 'best_epoch': int(b['epoch']), 'best_top1': float(b['eval_top1']),
            'best_eval_spikes': float(b['eval_spikes']), 'last_top1': float(rs[-1]['eval_top1']),
            'last_eval_spikes': float(rs[-1]['eval_spikes'])}


def sj(f):
    d = json.load(open(QR / f))
    return {'file': f, 'acc_remeasure': d['accuracy'], 'spikes': d['totals']['spikes_per_sample']}


qkd = {
    'baseline_best_top1': qk_base_best, 'baseline_best_mean': float(np.mean(qk_base_best)),
    'baseline_best_sd': float(np.std(qk_base_best, ddof=1)), 'baseline_n': len(qk_base_best),
    'baseline_spike_measure': [sj('spikes_base.json'), sj('spikes_base_s43.json')],
    'inter_7e-3': {**summ('ours_ch_rho7e-3'), 'measure': sj('spikes_ours_ch_rho7e-3.json')},
    'inter_1.2e-2': summ('ours_ch_rho1.2e-2'),
}
m = qkd['baseline_spike_measure']
qkd['baseline_remeasure_mean'] = float(np.mean([x['acc_remeasure'] for x in m]))
qkd['baseline_spikes_mean'] = float(np.mean([x['spikes'] for x in m]))
# 마지막 에폭 정확도 (best-of-410 은 위로 치우친다). base_s44 는 세종에만 있어 빠짐 -> n=6
QB = [Path('/home/kyccj/runs/qkformer_baseline/c10_baseline')] + [QR / f'base_s{i}' for i in (43, 45, 46, 47, 48)]
lasts = []
for d in QB:
    rs = [r for r in csv.DictReader(open(d / 'summary.csv')) if r['epoch'] != 'epoch']
    lasts.append(float(rs[-1]['eval_top1']))
qkd['baseline_last_top1'] = lasts
qkd['baseline_last_mean'] = float(np.mean(lasts)); qkd['baseline_last_sd'] = float(np.std(lasts, ddof=1))

# ---------------------------------------------------------------- SDTv3
sdt = json.load(open(LED / 'sdt.json'))
sdt1k = [j for j in sdt['jobs'] if j['ds'] == 'SDT1K']

out = {
    'generated_from': {'inventory': inv['generated'], 'led_dir': str(LED)},
    'C10_baseline_inter': c10, 'C100_baseline_inter': c100,
    'inter_fit_C10_200_420': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in fit10.items()},
    'inter_fit_C10_280_420': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in fit10_hi.items()},
    'inter_fit_C100_280_420': {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in fit100.items()},
    'ablation_C10': abl, 'derived_C10': der, 'derived_C100': der100,
    'bpsr_C10': bpsr_cmp, 'running': running, 'excluded': excluded,
    'qkformer_C10': qkd, 'sdtv3_imagenet_calib': sdt1k,
}
json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1, default=float)
print('wrote', OUT)
