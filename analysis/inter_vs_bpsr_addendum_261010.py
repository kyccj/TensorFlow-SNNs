"""inter vs BPSR 보충 — npz_t 9런으로 못 쟀던 축 (26-10-10).

입력: _npz_t/<tag>.npz 9개 (R19 C10 테스트 1만 장 1회 추론, extract_neuron_spikes.py)
      baseline s11·s12·s13 / BPSR 3e-11 s1·s2·s3 / ours_inter_ch(ours_ch) 1.5e-3 s1·s2·s3
      원장 = scratchpad led1007b/inventory.json (2026-10-10 13:50)
      대조 = analysis/ablation_mech_261010.json (스텝 몫·최초 발화·스텝별 정확도)
산출: analysis/inter_vs_bpsr_addendum_261010.json / .md

  1. 보정 (ECE 15구간, NLL, 최대확률 평균, 과신/소신, 최적 온도)
  2. 클래스별 정확도·혼동 행렬
  3. 스텝별 정확도 × 누적 스파이크 (logits_t = 출력 막전위 누적값)
  4. 층별 최초 발화 스텝 분포
  5. 같은 입력 예측 일치도·McNemar
"""
import os, json, itertools
import numpy as np
from scipy import stats
from scipy.special import log_softmax, softmax
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
INV = S + '/led1007b/inventory.json'
NPZ_T = os.path.join(ROOT, '_npz_t')
ABL = os.path.join(ROOT, 'analysis', 'ablation_mech_261010.json')
OUT_JSON = os.path.join(ROOT, 'analysis', 'inter_vs_bpsr_addendum_261010.json')
OUT_MD = os.path.join(ROOT, 'analysis', 'inter_vs_bpsr_addendum_261010.md')

ARMS = [('baseline', ['r19c10-base---s11', 'r19c10-base---s12', 'r19c10-base---s13']),
        ('BPSR 3e-11', ['r19c10-bpsr-3e-11-s1', 'r19c10-bpsr-3e-11-s2', 'r19c10-bpsr-3e-11-s3']),
        ('ours_inter_ch 1.5e-3', ['r19c10-ours_ch-1p5e-3-s1', 'r19c10-ours_ch-1p5e-3-s2', 'r19c10-ours_ch-1p5e-3-s3'])]
AN = [a for a, _ in ARMS]
B, P, I = AN
CLS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
SKIP = {'n_in', 'predictions_n'}
LAYERS = (['conv1_conv_n'] + [f'conv2_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')]
          + [f'conv3_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')]
          + [f'conv4_block{b}_{s}_n' for b in (1, 2) for s in ('conv1', 'out')] + ['fc1_n'])
NBINS = 15


def welch(a, b):
    """b − a. 차, SE, p, n."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = b.mean() - a.mean()
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    t = stats.ttest_ind(b, a, equal_var=False)
    return {'diff': float(d), 'se': se, 'p': float(t.pvalue), 'n': [len(b), len(a)]}


def ms(x):
    x = np.asarray(x, float)
    return {'mean': float(x.mean()), 'sd': float(x.std(ddof=1)) if len(x) > 1 else float('nan'), 'n': int(len(x))}


def calib(logits, y):
    lp = log_softmax(logits.astype(np.float64), 1)
    p = np.exp(lp)
    conf = p.max(1); pred = p.argmax(1); corr = (pred == y).astype(float)
    n = len(y)
    edges = np.linspace(0, 1, NBINS + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, NBINS - 1)
    ece = over = under = mce = 0.0
    bins = []
    for k in range(NBINS):
        m = idx == k
        if not m.any():
            bins.append(None); continue
        g = conf[m].mean() - corr[m].mean()
        w = m.sum() / n
        ece += w * abs(g); mce = max(mce, abs(g))
        if g > 0: over += w * g
        else: under += w * (-g)
        bins.append({'n': int(m.sum()), 'conf': float(conf[m].mean()), 'acc': float(corr[m].mean())})
    nll = float(-lp[np.arange(n), y].mean())
    q = np.full_like(p, 0.01); q[np.arange(n), y] += 0.9
    ls = float(-(q * lp).sum(1).mean())
    oh = np.zeros_like(p); oh[np.arange(n), y] = 1
    brier = float(((p - oh) ** 2).sum(1).mean())

    def f(lt):
        l2 = log_softmax(logits.astype(np.float64) / np.exp(lt), 1)
        return -l2[np.arange(n), y].mean()
    r = minimize_scalar(f, bounds=(-3, 3), method='bounded')
    temp = float(np.exp(r.x))
    # 정답/오답별 최대확률
    return {'ece': 100 * ece, 'ece_over': 100 * over, 'ece_under': 100 * under, 'mce': 100 * mce,
            'nll': nll, 'ls_loss': ls, 'brier': brier, 'conf_mean': 100 * float(conf.mean()),
            'acc': 100 * float(corr.mean()), 'gap': 100 * float(conf.mean() - corr.mean()),
            'conf_correct': 100 * float(conf[corr == 1].mean()), 'conf_wrong': 100 * float(conf[corr == 0].mean()),
            'temp_opt': temp, 'nll_temp': float(r.fun), 'bins': bins}


# ------------------------------------------------------------------ 로드
inv = {r['tag']: r for r in json.load(open(INV))['runs']}
inv_gen = json.load(open(INV)).get('generated')
abl = json.load(open(ABL))
R = {}
for arm, tags in ARMS:
    for t in tags:
        z = np.load(os.path.join(NPZ_T, t + '.npz'))
        y = z['labels'].astype(int); L = z['logits']; Lt = z['logits_t']; T = int(z['T'])
        o = {'arm': arm, 'ledger': {k: inv[t][k] for k in ('server', 'seed', 'epochs', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes', 'exclude')}}
        o['n_samples'] = int(z['n_samples']); o['T'] = T; o['npz_acc'] = float(z['acc'])
        o['check_logits_eq_last_step'] = float(np.abs(L - Lt[:, -1]).max())
        o['check_pred_eq_argmax'] = float((z['pred'] == L.argmax(1)).mean())
        o['cal'] = calib(L, y)
        pred = z['pred'].astype(int)
        o['pred'] = pred; o['y'] = y
        # 2 클래스
        C = np.zeros((10, 10), int)
        np.add.at(C, (y, pred), 1)
        o['conf_mat'] = C
        o['class_acc'] = (100 * np.diag(C) / C.sum(1)).tolist()
        o['class_sd'] = float(np.std(o['class_acc'], ddof=1))
        o['class_min'] = float(min(o['class_acc']))
        # 3 스텝 (누적)
        tot_t = np.zeros(T)
        first = {}
        for l in LAYERS:
            tot_t += z[l + '__t'].reshape(T, -1).sum(1)
            f = z[l + '__first_t'].astype(np.float64)
            first[l] = f
        o['spk_per_img_npz'] = float(tot_t.sum() / o['n_samples'])
        o['cum_share'] = (100 * np.cumsum(tot_t) / tot_t.sum()).tolist()
        o['step_share'] = (100 * tot_t / tot_t.sum()).tolist()
        o['acc_t'] = [100 * float((Lt[:, k].argmax(1) == y).mean()) for k in range(T)]
        o['agree_t_final'] = [100 * float((Lt[:, k].argmax(1) == pred).mean()) for k in range(T)]
        ct = [calib(Lt[:, k], y) for k in range(T)]
        o['ece_t'] = [c['ece'] for c in ct]; o['conf_t'] = [c['conf_mean'] for c in ct]; o['nll_t'] = [c['nll'] for c in ct]
        # 4 최초 발화 (층별)
        o['first'] = {}
        allf = np.zeros(T + 1)
        for l in LAYERS:
            f = first[l]; allf += f
            fired = f[:T].sum()
            o['first'][l] = {'share': (100 * f / f.sum()).tolist(),
                             'share_fired': (100 * f[:T] / fired).tolist() if fired > 0 else None,
                             'mean_t': float((np.arange(1, T + 1) * f[:T]).sum() / fired) if fired > 0 else None}
        o['first']['ALL'] = {'share': (100 * allf / allf.sum()).tolist(),
                             'share_fired': (100 * allf[:T] / allf[:T].sum()).tolist(),
                             'mean_t': float((np.arange(1, T + 1) * allf[:T]).sum() / allf[:T].sum())}
        R[t] = o

T = R[ARMS[0][1][0]]['T']
res = {'generated_from': {'npz_dir': NPZ_T, 'inventory': INV, 'inventory_generated': inv_gen, 'cross_check': ABL},
       'arms': {a: t for a, t in ARMS}}

# ------------------------------------------------------------------ 0 기본 표 / 대조
res['runs'] = {t: {k: v for k, v in o.items() if k not in ('pred', 'y', 'conf_mat', 'first')} | {'conf_mat': o['conf_mat'].tolist()}
               for t, o in R.items()}
xc = {}
for t, o in R.items():
    a = abl['final_step']['npz_t'][t]
    xc[t] = {'acc_t_maxdiff': float(np.max(np.abs(np.array(a['acc_t']) - o['acc_t']))),
             'step_share_maxdiff': float(np.max(np.abs(np.array(a['step_share']) - o['step_share']))),
             'mean_first_t_diff': float(abs(a['mean_first_t'] - o['first']['ALL']['mean_t'])),
             'npz_acc_vs_ledger': float(o['npz_acc'] - o['ledger']['val_acc'])}
res['cross_check'] = xc


def arm_vals(arm, fn):
    return [fn(R[t]) for t in dict(ARMS)[arm]]


def block(fn):
    out = {a: ms(arm_vals(a, fn)) for a in AN}
    out['inter-BPSR'] = welch(arm_vals(P, fn), arm_vals(I, fn))
    out['inter-baseline'] = welch(arm_vals(B, fn), arm_vals(I, fn))
    out['BPSR-baseline'] = welch(arm_vals(B, fn), arm_vals(P, fn))
    return out


# ------------------------------------------------------------------ 1 보정
CAL_KEYS = ['acc', 'conf_mean', 'gap', 'ece', 'ece_over', 'ece_under', 'mce', 'nll', 'ls_loss', 'brier',
            'conf_correct', 'conf_wrong', 'temp_opt', 'nll_temp']
res['calibration'] = {k: block(lambda o, k=k: o['cal'][k]) for k in CAL_KEYS}

# ------------------------------------------------------------------ 2 클래스
cls = {}
for c in range(10):
    cls[CLS[c]] = block(lambda o, c=c: o['class_acc'][c])
res['class_acc'] = cls
res['class_sd'] = block(lambda o: o['class_sd'])
res['class_min'] = block(lambda o: o['class_min'])
# 클래스별 Δ 의 분포: inter−BPSR 평균 Δ 10개
dcls = np.array([cls[CLS[c]]['inter-BPSR']['diff'] for c in range(10)])
res['class_delta_inter_bpsr'] = {'deltas': dcls.tolist(), 'sd_across_classes': float(dcls.std(ddof=1)),
                                 'n_p_lt_005': int(sum(cls[CLS[c]]['inter-BPSR']['p'] < 0.05 for c in range(10)))}
# 시드 잡음 기준: 같은 팔 안에서 시드 하나를 둘로 나눈 Δ 의 클래스 간 SD (3시드 → 쌍별 Δ)
ref = []
for a in AN:
    tags = dict(ARMS)[a]
    for t1, t2 in itertools.combinations(tags, 2):
        ref.append(float(np.std(np.array(R[t1]['class_acc']) - np.array(R[t2]['class_acc']), ddof=1)))
xs = []
for t1 in dict(ARMS)[I]:
    for t2 in dict(ARMS)[P]:
        xs.append(float(np.std(np.array(R[t1]['class_acc']) - np.array(R[t2]['class_acc']), ddof=1)))
res['class_delta_sd_single_pairs'] = {'within_arm_seed_pairs': ms(ref), 'inter_vs_bpsr_pairs': ms(xs)}
# 혼동 행렬
CM = {a: np.array([R[t]['conf_mat'] for t in dict(ARMS)[a]]) for a in AN}
res['conf_mat_mean'] = {a: CM[a].mean(0).tolist() for a in AN}
cells = []
for i in range(10):
    for j in range(10):
        if i == j:
            continue
        w = welch(CM[P][:, i, j], CM[I][:, i, j])
        wb = welch(CM[B][:, i, j], CM[I][:, i, j])
        cells.append({'true': CLS[i], 'pred': CLS[j], 'baseline': ms(CM[B][:, i, j]), 'BPSR': ms(CM[P][:, i, j]),
                      'inter': ms(CM[I][:, i, j]), 'inter-BPSR': w, 'inter-baseline': wb})
cells.sort(key=lambda c: -abs(c['inter-BPSR']['diff']))
res['conf_cells_top'] = cells[:8]
res['conf_cells_most_common'] = sorted(cells, key=lambda c: -(c['BPSR']['mean'] + c['inter']['mean']))[:6]
res['conf_cells_n_p_lt_005'] = int(sum(c['inter-BPSR']['p'] < 0.05 for c in cells))

# ------------------------------------------------------------------ 3 스텝
st = {}
for k in range(T):
    st[k + 1] = {'acc': block(lambda o, k=k: o['acc_t'][k]),
                 'acc_drop': block(lambda o, k=k: o['acc_t'][k] - o['acc_t'][-1]),
                 'cum_share': block(lambda o, k=k: o['cum_share'][k]),
                 'cum_spk_npz': block(lambda o, k=k: o['cum_share'][k] / 100 * o['spk_per_img_npz']),
                 'cum_spk_ledger': block(lambda o, k=k: o['cum_share'][k] / 100 * o['ledger']['spikes']),
                 'agree_final': block(lambda o, k=k: o['agree_t_final'][k]),
                 'ece': block(lambda o, k=k: o['ece_t'][k]),
                 'conf': block(lambda o, k=k: o['conf_t'][k])}
res['steps'] = st
res['spk_npz'] = block(lambda o: o['spk_per_img_npz'])
res['spk_ledger'] = block(lambda o: o['ledger']['spikes'])

# ------------------------------------------------------------------ 4 최초 발화
fl = {}
for l in LAYERS + ['ALL']:
    q = {}
    for k in range(T + 1):
        q[f's{k + 1}' if k < T else 'silent'] = block(lambda o, l=l, k=k: o['first'][l]['share'][k])
    q['mean_t'] = block(lambda o, l=l: o['first'][l]['mean_t'])
    for k in range(T):
        q[f'f{k + 1}'] = block(lambda o, l=l, k=k: o['first'][l]['share_fired'][k])
    fl[l] = q
res['first_fire'] = fl

# ------------------------------------------------------------------ 5 일치도
def agree(t1, t2):
    return 100 * float((R[t1]['pred'] == R[t2]['pred']).mean())


ag = {}
for a in AN:
    ag[a + ' 안'] = ms([agree(t1, t2) for t1, t2 in itertools.combinations(dict(ARMS)[a], 2)])
for a1, a2 in itertools.combinations(AN, 2):
    ag[a1 + ' × ' + a2] = ms([agree(t1, t2) for t1 in dict(ARMS)[a1] for t2 in dict(ARMS)[a2]])
res['agreement'] = ag
y = R[ARMS[0][1][0]]['y']
for t in R:
    assert (R[t]['y'] == y).all()
corr = {t: (R[t]['pred'] == y) for t in R}
trip = [(tb, tp, ti) for tb in dict(ARMS)[B] for tp in dict(ARMS)[P] for ti in dict(ARMS)[I]]
res['triples'] = {'all_correct': ms([int((corr[a] & corr[b] & corr[c]).sum()) for a, b, c in trip]),
                  'all_wrong': ms([int((~corr[a] & ~corr[b] & ~corr[c]).sum()) for a, b, c in trip]),
                  'n_triples': len(trip)}
allc = np.all([corr[t] for t in R], 0); allw = np.all([~corr[t] for t in R], 0)
res['nine_runs'] = {'all_correct': int(allc.sum()), 'all_wrong': int(allw.sum()), 'mixed': int((~allc & ~allw).sum())}


def mcn(ta, tb):
    """a 만 맞춤 = b01, b 만 맞춤 = b10. 정확 이항 (양측)."""
    only_a = int((corr[ta] & ~corr[tb]).sum()); only_b = int((~corr[ta] & corr[tb]).sum())
    p = float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b > 0 else 1.0
    return {'a': ta, 'b': tb, 'only_a': only_a, 'only_b': only_b, 'p': p}


mc = {}
for a1, a2 in ((I, P), (I, B), (P, B)):
    lst = [mcn(t1, t2) for t1 in dict(ARMS)[a1] for t2 in dict(ARMS)[a2]]
    mc[a1 + ' vs ' + a2] = {'pairs': lst, 'only_a': ms([x['only_a'] for x in lst]), 'only_b': ms([x['only_b'] for x in lst]),
                            'discord': ms([x['only_a'] + x['only_b'] for x in lst]),
                            'net': ms([x['only_a'] - x['only_b'] for x in lst]),
                            'n_p_lt_005': int(sum(x['p'] < 0.05 for x in lst))}
for a in AN:
    lst = [mcn(t1, t2) for t1, t2 in itertools.combinations(dict(ARMS)[a], 2)]
    mc[a + ' 시드끼리'] = {'pairs': lst, 'discord': ms([x['only_a'] + x['only_b'] for x in lst]),
                        'net_abs': ms([abs(x['only_a'] - x['only_b']) for x in lst]),
                        'n_p_lt_005': int(sum(x['p'] < 0.05 for x in lst))}
res['mcnemar'] = mc
# 이미지별 다수결 (3시드 중 맞춘 수)
K = {a: np.sum([corr[t] for t in dict(ARMS)[a]], 0) for a in AN}
res['seed_vote'] = {}
for a1, a2 in ((I, P), (I, B), (P, B)):
    x = int(((K[a1] == 3) & (K[a2] == 0)).sum()); w_ = int(((K[a1] == 0) & (K[a2] == 3)).sum())
    x2 = int(((K[a1] >= 2) & (K[a2] <= 1)).sum()); w2 = int(((K[a1] <= 1) & (K[a2] >= 2)).sum())
    res['seed_vote'][a1 + ' vs ' + a2] = {'a3_b0': x, 'a0_b3': w_, 'a_maj_b_min': x2, 'a_min_b_maj': w2,
                                          'p_maj': float(stats.binomtest(x2, x2 + w2, 0.5).pvalue)}
res['seed_vote']['hist'] = {a: np.bincount(K[a], minlength=4).tolist() for a in AN}


# ------------------------------------------------------------------ 저장
def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, float) and not np.isfinite(o): return None
    return o


json.dump(clean(res), open(OUT_JSON, 'w'), indent=1, ensure_ascii=False)

# ================================================================== MD
L_ = []
w = L_.append


def F(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return '—'
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


def MS(q, d=2):
    return f"{F(q['mean'], d)} ±{F(q['sd'], d)}"


def PV(p):
    return '—' if p is None else (f'{p:.2g}' if p >= 1e-3 else f'{p:.1e}')


def WD(q, d=2):
    return f"{F(q['diff'], d, True)} ±{F(q['se'], d)} (p={PV(q['p'])})"


def K_(x):
    return f'{x:,.0f}'


cal = res['calibration']
c_ece = cal['ece']; c_gap = cal['gap']; c_nll = cal['nll']; c_T = cal['temp_opt']
s2 = res['steps'][2]
mcIP = res['mcnemar'][f'{I} vs {P}']
fa = res['first_fire']['ALL']

w('# inter vs BPSR 보충 — 보정·클래스·스텝·최초 발화·예측 일치 (R19-C10, 26-10-10)\n')
w(f'생성: `analysis/inter_vs_bpsr_addendum_261010.py` → `.json`, 이 문서. 입력 = `_npz_t/` 9개 (테스트 1만 장 1회 추론, 런마다 원장 베스트 체크포인트), 원장 {inv_gen} 집계. '
  f'팔: baseline (s11·s12·s13), BPSR 3e-11 (s1·s2·s3), ours_inter_ch ρ 1.5e-3 (런 키 `ours_ch`, s1·s2·s3). **모든 팔 n=3.** '
  f'± 는 평균 행에서 시드 간 SD, 차이 행에서 Welch SE. p 는 Welch t (자유도 약 2~4). 그림 없음.\n')
w('## 요약\n')
w(f"1. **보정 (ECE = 신뢰도와 정확도의 평균 차이) — 세 팔 모두 소신 (자신 없게 맞춤), 팔 간 차이는 시드 산포 안.** "
  f"ECE baseline {MS(c_ece[B])}, BPSR {MS(c_ece[P])}, inter {MS(c_ece[I])} %p (각 n=3); inter−BPSR {WD(c_ece['inter-BPSR'])}, inter−baseline {WD(c_ece['inter-baseline'])}. "
  f"최대확률 평균이 정확도보다 {F(min(-c_gap[a]['mean'] for a in AN),1)}~{F(max(-c_gap[a]['mean'] for a in AN),1)}%p 낮다 (각 팔 n=3) — 학습의 label smoothing 0.1 이 만드는 정상 현상이다. NLL inter−BPSR {WD(c_nll['inter-BPSR'],4)}.")
cd = res['class_delta_inter_bpsr']; csd = res['class_sd']
w(f"2. **클래스별 정확도 — inter−BPSR 클래스별 Δ 10개 중 p<0.05 는 {cd['n_p_lt_005']}개 (보정 전), 클래스 편차 (클래스 10개 정확도의 SD) 도 차이 없음.** "
  f"클래스 편차 baseline {MS(csd[B])}, BPSR {MS(csd[P])}, inter {MS(csd[I])} %p; inter−BPSR {WD(csd['inter-BPSR'])}. "
  f"inter 한 런 − BPSR 한 런의 클래스별 Δ 산포 (SD {MS(res['class_delta_sd_single_pairs']['inter_vs_bpsr_pairs'])}) 가 같은 팔 시드끼리 ({MS(res['class_delta_sd_single_pairs']['within_arm_seed_pairs'])}) 와 같다. 가장 바뀐 혼동 쌍도 90칸 중 p<0.05 {res['conf_cells_n_p_lt_005']}칸 — 우연 기대치 (약 4.5칸) 보다 적다.")
w(f"3. **스텝별 정확도 (출력 막전위 누적, T'=k 에서 멈춤) — ablation_mech 6절 값과 일치 (최대 차 {F(max(v['acc_t_maxdiff'] for v in xc.values()),3)}%p).** "
  f"T'=2 (누적 스파이크 baseline {F(s2['cum_share'][B]['mean'],1)}% / BPSR {F(s2['cum_share'][P]['mean'],1)}% / inter {F(s2['cum_share'][I]['mean'],1)}%) 에서 정확도 손실 baseline {MS(s2['acc_drop'][B])}, BPSR {MS(s2['acc_drop'][P])}, inter {MS(s2['acc_drop'][I])} %p; "
  f"inter−BPSR {WD(s2['acc_drop']['inter-BPSR'])}. n=3 으로 **판단 불가.** T'=2 정확도의 절대값은 inter−BPSR {WD(s2['acc']['inter-BPSR'])} 이나, 그 시점 누적 스파이크가 inter 쪽이 {F((s2['cum_spk_ledger'][I]['mean']/s2['cum_spk_ledger'][P]['mean']-1)*100,1,True)}% 많아 (원장 환산 {K_(s2['cum_spk_ledger'][I]['mean'])} vs {K_(s2['cum_spk_ledger'][P]['mean'])}) 같은 스파이크 비교가 아니다.")
w(f"4. **최초 발화 스텝 — inter 가 BPSR 보다 일찍 쏜다 (18층 전체, 평균 최초 발화 스텝 inter−BPSR {WD(fa['mean_t']['inter-BPSR'],3)}).** "
  f"(뉴런·이미지) 쌍 중 t1 에 처음 쏜 몫 inter−BPSR {WD(fa['s1']['inter-BPSR'])}%p, 미발화 몫 {WD(fa['silent']['inter-BPSR'])}%p — inter 는 더 많은 쌍을 침묵시키면서 쏘는 쌍은 더 일찍 쏜다. 차이는 앞 층 (conv1·conv2) 에 몰려 있다 (4절 표). baseline 대비 inter 는 늦어짐이 BPSR 의 약 40% 수준.")
w(f"5. **같은 입력 예측 일치 — 팔 사이 불일치는 같은 팔 시드 사이 불일치와 같은 크기.** 예측 일치율 inter 시드끼리 {MS(res['agreement'][I + ' 안'])}%, BPSR 시드끼리 {MS(res['agreement'][P + ' 안'])}%, inter×BPSR {MS(res['agreement'][P + ' × ' + I])}% (각 3 / 3 / 9쌍). "
  f"McNemar (inter 한 런 vs BPSR 한 런, 9쌍): inter 만 맞춘 표본 {MS(mcIP['only_a'],1)}, BPSR 만 맞춘 {MS(mcIP['only_b'],1)}, p<0.05 인 쌍 {mcIP['n_p_lt_005']}/9. 9런 모두 맞춘 {K_(res['nine_runs']['all_correct'])}장, 모두 틀린 {K_(res['nine_runs']['all_wrong'])}장 (1만 장).")
w('')
w('## 정의\n')
w('- **ECE (expected calibration error)**: 신뢰도 (softmax 최대확률) 와 정확도의 평균 차이. 최대확률을 [0,1] 15개 등간격 구간으로 나눠 구간별 |평균 신뢰도 − 정확도| 를 표본 비율로 가중 합. 단위 %p. **과신 몫/소신 몫** = 신뢰도 > 정확도 인 구간 / < 인 구간의 기여 (합 = ECE).')
w('- **gap** = 최대확률 평균 − 정확도. 음수 = 소신 (자신 없게 맞춤).')
w('- **NLL** = −log p(정답), label smoothing 없이. **LS 손실** = 학습과 같은 label smoothing 0.1 교차엔트로피 — 원장 val loss 와 대조용.')
w('- **최적 온도 T\\***: logits/T\\* 로 테스트 NLL 이 최소가 되는 T\\*. T\\*<1 이면 소신 (날카롭게 해야 맞음), >1 이면 과신. 테스트셋으로 맞춘 기술 통계일 뿐이다.')
w('- **logits**: 출력층 (`predictions_n`) 막전위의 마지막 스텝 값 (`snn_out_proc=last_ts`). `logits_t[:,k]` 는 k+1 스텝까지 적분된 막전위 — 출력층이 적분기라 **누적값**이다 (`logits == logits_t[:,3]` 확인, 9런 최대 차 0). 그래서 "T\'=k 정확도" = k 스텝에서 멈췄을 때의 정확도.')
w('- **누적 스파이크 %**: 스파이킹 18층 (`n_in`, `predictions_n` 제외) 의 1~k 스텝 스파이크 / 4스텝 전체.')
w('- **최초 발화 몫**: (뉴런, 이미지) 쌍 중 스텝 k 에 처음 쏜 쌍의 비율. 미발화 = 4스텝 동안 한 번도 안 쏜 쌍. 평균 최초 발화 스텝 = 쏜 쌍만의 평균 (1~4).')
w('- **McNemar**: 같은 1만 장에서 a 만 맞춘 수 vs b 만 맞춘 수의 정확 이항 검정. **두 모델 (런) 사이의 차이**를 묻는 검정이지 방법 사이의 차이가 아니다 — 방법 비교의 단위는 시드 (n=3).')
w('- 스파이크: 원장 = 평가 모드·베스트 에폭 원장값, npz = 이 추출의 18층 합/이미지. 둘은 정의가 달라 npz 쪽이 4~6% 크다 (0절).')
w('')

# 0
w('## 0. 런 목록과 대조\n')
w('| 팔 | seed | 서버 | train loss | train acc | val loss | val acc | spikes (원장) | npz acc | spikes/이미지 (npz) | LS 손실 (npz) | acc_t 최대 차 vs ablation_mech | 스텝 몫 최대 차 | 평균 최초 발화 차 |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for a, tags in ARMS:
    for t in tags:
        o = R[t]; l = o['ledger']
        w(f"| {a} | s{l['seed']} | {l['server']} | {F(l['train_loss'],4)} | {F(l['train_acc'],2)} | {F(l['val_loss'],4)} | {F(l['val_acc'],2)} | {K_(l['spikes'])} | {F(o['npz_acc'],2)} | {K_(o['spk_per_img_npz'])} | {F(o['cal']['ls_loss'],4)} | {F(xc[t]['acc_t_maxdiff'],3)} | {F(xc[t]['step_share_maxdiff'],4)} | {F(xc[t]['mean_first_t_diff'],5)} |")
    g = lambda k: ms([R[t]['ledger'][k] for t in tags])
    w(f"| **{a} 평균** | **n=3** |  | {MS(g('train_loss'),4)} | {MS(g('train_acc'))} | {MS(g('val_loss'),4)} | **{MS(g('val_acc'))}** | **{K_(g('spikes')['mean'])} ±{K_(g('spikes')['sd'])}** | {MS(ms([R[t]['npz_acc'] for t in tags]))} | {K_(res['spk_npz'][a]['mean'])} ±{K_(res['spk_npz'][a]['sd'])} | {MS(ms([R[t]['cal']['ls_loss'] for t in tags]),4)} |  |  |  |")
w('')
w(f"- 배제 런 없음 (9런 모두 310 에폭 완주, 원장 exclude=False). npz acc 와 원장 val acc 차 최대 {F(max(abs(v['npz_acc_vs_ledger']) for v in xc.values()),2)}%p (baseline s11 96.72 vs 96.74 — 추출 체크포인트 재평가 차).")
w('- ablation_mech_261010 의 스텝 몫·평균 최초 발화·스텝별 정확도와 런별로 일치 (위 표 끝 세 열 = 절대 차). 같은 npz 에서 같은 정의로 다시 계산했다.')
w(f"- 원장 스파이크: inter {K_(res['spk_ledger'][I]['mean'])} vs BPSR {K_(res['spk_ledger'][P]['mean'])} (inter−BPSR {F(res['spk_ledger']['inter-BPSR']['diff']/res['spk_ledger'][P]['mean']*100,1,True)}%, p={PV(res['spk_ledger']['inter-BPSR']['p'])}), baseline {K_(res['spk_ledger'][B]['mean'])}. npz 로는 inter−BPSR {F(res['spk_npz']['inter-BPSR']['diff']/res['spk_npz'][P]['mean']*100,1,True)}% (p={PV(res['spk_npz']['inter-BPSR']['p'])}).")
w('')

# 1
w('## 1. 보정 — ECE·NLL·신뢰도\n')
w('| 팔 | seed | 서버 | val acc (npz) | 최대확률 평균 | gap | ECE | 과신 몫 | 소신 몫 | MCE | NLL | Brier | 정답 표본 신뢰도 | 오답 표본 신뢰도 | 최적 온도 T* |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for a, tags in ARMS:
    for t in tags:
        c = R[t]['cal']; l = R[t]['ledger']
        w(f"| {a} | s{l['seed']} | {l['server']} | {F(c['acc'])} | {F(c['conf_mean'])} | {F(c['gap'],2,True)} | {F(c['ece'])} | {F(c['ece_over'])} | {F(c['ece_under'])} | {F(c['mce'],1)} | {F(c['nll'],4)} | {F(c['brier'],4)} | {F(c['conf_correct'])} | {F(c['conf_wrong'])} | {F(c['temp_opt'],3)} |")
    q = lambda k, d=2: MS(cal[k][a], d)
    w(f"| **{a} 평균** | **n=3** |  | {q('acc')} | {q('conf_mean')} | {q('gap')} | **{q('ece')}** | {q('ece_over')} | {q('ece_under')} | {q('mce',1)} | **{q('nll',4)}** | {q('brier',4)} | {q('conf_correct')} | {q('conf_wrong')} | {q('temp_opt',3)} |")
w('')
w('차이 (Welch, 각 n=3):\n')
w('| 지표 | inter − BPSR | inter − baseline | BPSR − baseline |')
w('|---|---|---|---|')
for k, nm, d in (('ece', 'ECE (%p)', 2), ('gap', 'gap (%p)', 2), ('conf_mean', '최대확률 평균 (%)', 2), ('nll', 'NLL', 4),
                 ('brier', 'Brier', 4), ('conf_wrong', '오답 표본 신뢰도 (%)', 2), ('temp_opt', '최적 온도 T*', 3), ('acc', 'val acc (npz, %)', 2)):
    w(f"| {nm} | {WD(cal[k]['inter-BPSR'],d)} | {WD(cal[k]['inter-baseline'],d)} | {WD(cal[k]['BPSR-baseline'],d)} |")
w('')
w(f"- 방향: 세 팔 9런 모두 gap<0, 소신 몫 > 과신 몫, T*<1 — **소신**이다. label smoothing 0.1 로 학습하면 정답 확률 상한이 0.91 근처로 눌리므로 예상된 방향이다. 규제가 이 방향을 바꾸지 않는다.")
w(f"- 크기: 팔 간 ECE 차는 {F(abs(c_ece['inter-BPSR']['diff']),2)}%p (inter−BPSR) 로 시드 SD ({F(c_ece[P]['sd'],2)}~{F(c_ece[I]['sd'],2)}) 와 같은 수준. **보정에서 inter 와 BPSR 은 구분되지 않는다** (n=3).")
w(f"- 규제 두 방법 모두 baseline 보다 최대확률 평균이 약간 높다 (inter−baseline {WD(cal['conf_mean']['inter-baseline'])}, BPSR−baseline {WD(cal['conf_mean']['BPSR-baseline'])} %p, 각 n=3). 정확도 차 (+0.03~0.05) 보다 커서 소신이 아주 조금 줄지만 ECE 로는 보이지 않는다 (±0.1). 두 검정 모두 p≈0.01 이나 n=3/3 이고 사후 검정이라 경향으로만 둔다.")
w(f"- 오답 표본 신뢰도 (틀릴 때 얼마나 확신하나) 도 inter−BPSR {WD(cal['conf_wrong']['inter-BPSR'])} — 차이 없음.")
w('')

# 2
w('## 2. 클래스별 정확도\n')
w('| 클래스 | baseline (n=3) | BPSR (n=3) | inter (n=3) | inter − BPSR | inter − baseline |')
w('|---|---|---|---|---|---|')
for c in CLS:
    q = res['class_acc'][c]
    w(f"| {c} | {MS(q[B])} | {MS(q[P])} | {MS(q[I])} | {WD(q['inter-BPSR'])} | {WD(q['inter-baseline'])} |")
w(f"| **클래스 편차 (10클래스 SD)** | {MS(csd[B])} | {MS(csd[P])} | {MS(csd[I])} | {WD(csd['inter-BPSR'])} | {WD(csd['inter-baseline'])} |")
w(f"| **최저 클래스 정확도** | {MS(res['class_min'][B])} | {MS(res['class_min'][P])} | {MS(res['class_min'][I])} | {WD(res['class_min']['inter-BPSR'])} | {WD(res['class_min']['inter-baseline'])} |")
w('')
w('런별 클래스 정확도:\n')
w('| 팔 | seed | 서버 | ' + ' | '.join(CLS) + ' | 클래스 SD |')
w('|---|---|---|' + '---|' * 11)
for a, tags in ARMS:
    for t in tags:
        l = R[t]['ledger']
        w(f"| {a} | s{l['seed']} | {l['server']} | " + ' | '.join(F(x, 1) for x in R[t]['class_acc']) + f" | {F(R[t]['class_sd'])} |")
    w(f"| **{a} 평균** | **n=3** |  | " + ' | '.join(F(res['class_acc'][c][a]['mean'], 2) for c in CLS) + f" | {F(csd[a]['mean'])} |")
w('')
w(f"- 클래스별 Δ (inter−BPSR) 10개: " + ', '.join(f"{c} {F(d,2,True)}" for c, d in zip(CLS, cd['deltas'])) + f". 클래스 간 SD {F(cd['sd_across_classes'],2)}%p. p<0.05 는 {cd['n_p_lt_005']}/10 (10번 검정, 보정 전).")
w(f"- 시드 잡음 기준: 런 하나끼리 클래스별 Δ 의 SD — 같은 팔 시드끼리 {MS(res['class_delta_sd_single_pairs']['within_arm_seed_pairs'])} (9쌍), inter×BPSR {MS(res['class_delta_sd_single_pairs']['inter_vs_bpsr_pairs'])} (9쌍). 같은 크기라 **클래스 단위로 inter 가 BPSR 과 다르게 실수한다는 증거 없음.**")
w('- 클래스당 테스트 1,000장이라 클래스 정확도 하나의 이항 SE 는 약 0.4~1.0%p (cat 90% 기준 0.95%p). 클래스 단위 차 0.3%p 대는 런 하나로도 n=3 으로도 안 보인다.')
w('')
w(f'혼동 행렬 — inter−BPSR 평균 개수 차가 가장 큰 칸 (참 → 예측, 1만 장 중 개수, 각 n=3; 90칸 중 p<0.05 = {res["conf_cells_n_p_lt_005"]}칸):\n')
w('| 참 → 예측 | baseline | BPSR | inter | inter − BPSR | inter − baseline |')
w('|---|---|---|---|---|---|')
for c in res['conf_cells_top']:
    w(f"| {c['true']} → {c['pred']} | {MS(c['baseline'],1)} | {MS(c['BPSR'],1)} | {MS(c['inter'],1)} | {WD(c['inter-BPSR'],1)} | {WD(c['inter-baseline'],1)} |")
w('')
w('가장 흔한 혼동 (BPSR+inter 평균 개수 순):\n')
w('| 참 → 예측 | baseline | BPSR | inter | inter − BPSR |')
w('|---|---|---|---|---|')
for c in res['conf_cells_most_common']:
    w(f"| {c['true']} → {c['pred']} | {MS(c['baseline'],1)} | {MS(c['BPSR'],1)} | {MS(c['inter'],1)} | {WD(c['inter-BPSR'],1)} |")
w('')
w('- 90칸을 검정하면 p<0.05 가 우연으로 약 4.5칸 나온다. 위 칸들의 차는 8개 이하이고 시드 SD 와 같은 크기다 — **방향성 있는 혼동 변화로 읽지 않는다.**')
w('')

# 3
w("## 3. 스텝별 정확도 × 누적 스파이크 — T'=k 에서 멈추면\n")
w("`logits_t` 가 누적 막전위라 T'=k 는 \"k 스텝에서 추론을 멈춘\" 정확도다. 스파이크는 1~k 스텝 누적 (18층). 원장 환산 = 누적 몫 × 원장 스파이크.\n")
w("| 팔 | T' | n | 누적 스파이크 % | 스파이크/이미지 (npz) | 원장 환산 | val acc | T'=4 대비 | 최종 예측과 일치 % | ECE | 최대확률 평균 |")
w('|---|---|---|---|---|---|---|---|---|---|---|')
for a in AN:
    for k in range(1, T + 1):
        q = res['steps'][k]
        w(f"| {a} | {k} | 3 | {MS(q['cum_share'][a],1)} | {K_(q['cum_spk_npz'][a]['mean'])} | {K_(q['cum_spk_ledger'][a]['mean'])} | {MS(q['acc'][a])} | {MS(q['acc_drop'][a])} | {MS(q['agree_final'][a])} | {MS(q['ece'][a],1)} | {MS(q['conf'][a],1)} |")
w('')
w('런별 val acc (T\'=1/2/3/4):\n')
w("| 팔 | seed | 서버 | T'=1 | T'=2 | T'=3 | T'=4 | 누적 스파이크 % (1/2/3) |")
w('|---|---|---|---|---|---|---|---|')
for a, tags in ARMS:
    for t in tags:
        o = R[t]; l = o['ledger']
        w(f"| {a} | s{l['seed']} | {l['server']} | " + ' | '.join(F(x) for x in o['acc_t']) + ' | ' + ' / '.join(F(x, 1) for x in o['cum_share'][:3]) + ' |')
    w(f"| **{a} 평균** | **n=3** |  | " + ' | '.join(MS(res['steps'][k]['acc'][a]) for k in range(1, T + 1)) + ' | ' + ' / '.join(F(res['steps'][k]['cum_share'][a]['mean'], 1) for k in range(1, 4)) + ' |')
w('')
w('차이 (Welch, 각 n=3):\n')
w("| T' | 지표 | inter − BPSR | inter − baseline |")
w('|---|---|---|---|')
for k in (1, 2, 3):
    q = res['steps'][k]
    w(f"| {k} | val acc | {WD(q['acc']['inter-BPSR'])} | {WD(q['acc']['inter-baseline'])} |")
    w(f"| {k} | T'=4 대비 손실 | {WD(q['acc_drop']['inter-BPSR'])} | {WD(q['acc_drop']['inter-baseline'])} |")
    w(f"| {k} | 누적 스파이크 % | {WD(q['cum_share']['inter-BPSR'])} | {WD(q['cum_share']['inter-baseline'])} |")
w('')
w(f"- ablation_mech 6절과 같은 값이다 (0절 대조). 새로 붙인 것은 원장 환산 스파이크·최종 예측 일치·스텝별 ECE.")
w(f"- T'=2 에서 멈추면 스파이크를 약 {F(100-s2['cum_share'][I]['mean'],0)}% 덜 쓰고 (원장 환산 inter {K_(s2['cum_spk_ledger'][I]['mean'])} / BPSR {K_(s2['cum_spk_ledger'][P]['mean'])} / baseline {K_(s2['cum_spk_ledger'][B]['mean'])}) 정확도는 팔 평균으로 0.13%p 이내로 떨어진다 (런별 최대 손실 baseline s11 −0.29%p). 팔 간 차는 n=3 으로 **판단 불가** (inter−BPSR 손실 차 p={PV(s2['acc_drop']['inter-BPSR']['p'])}).")
w(f"- T'=1 은 정확도 40~67% 로 시드마다 크게 다르고 (SD 10~12%p) 방법과 무관하다. 스텝별 ECE 는 T'≤3 에서 크다 (T'=2 에서 최대 51~54%p, 정확도 96.6% 인데 최대확률 평균 43~46%) — 막전위가 덜 쌓여 softmax 가 평평한 소신이고, 보정 성질이 아니라 적분 길이 탓이다. T'=1 의 ECE (36~40) 는 정확도 자체가 낮아 작아 보일 뿐이다.")
w('- 이것은 "학습은 T=4, 추론만 짧게" 의 질문이다. 규제 방법 비교와 다른 축이고, T\'=2 추론으로 학습한 비교군 (T=2 학습) 은 없다.')
w('')

# 4
w('## 4. 최초 발화 스텝 — 층별\n')
w('(뉴런, 이미지) 쌍의 최초 발화 스텝 분포. 각 팔 n=3 평균 (시드 SD 는 json). 앞 4열은 전체 쌍 중 %, 미발화 포함 합 100.\n')
w('| 층 | 팔 | t1 % | t2 % | t3 % | t4 % | 미발화 % | 평균 최초 발화 스텝 (쏜 쌍) |')
w('|---|---|---|---|---|---|---|---|')
for l in ['ALL'] + LAYERS:
    for a in AN:
        q = res['first_fire'][l]
        nm = '**18층 전체**' if l == 'ALL' else l
        w(f"| {nm} | {a} | " + ' | '.join(F(q[f's{k}'][a]['mean'], 2) for k in range(1, T + 1)) + f" | {F(q['silent'][a]['mean'],2)} | {F(q['mean_t'][a]['mean'],3)} ±{F(q['mean_t'][a]['sd'],3)} |")
w('')
w('inter − BPSR (Welch, 각 n=3):\n')
w('| 층 | t1 몫 (%p) | t4 몫 (%p) | 미발화 (%p) | 쏜 쌍 중 t1 몫 (%p) | 평균 최초 발화 스텝 | (참고) inter−baseline 평균 최초 발화 | (참고) BPSR−baseline |')
w('|---|---|---|---|---|---|---|---|')
for l in ['ALL'] + LAYERS:
    q = res['first_fire'][l]
    nm = '**18층 전체**' if l == 'ALL' else l
    w(f"| {nm} | {WD(q['s1']['inter-BPSR'])} | {WD(q['s4']['inter-BPSR'])} | {WD(q['silent']['inter-BPSR'])} | {WD(q['f1']['inter-BPSR'])} | {WD(q['mean_t']['inter-BPSR'],3)} | {WD(q['mean_t']['inter-baseline'],3)} | {WD(q['mean_t']['BPSR-baseline'],3)} |")
w('')
nsig = sum(res['first_fire'][l]['mean_t']['inter-BPSR']['p'] < 0.05 for l in LAYERS)
neg = sum(res['first_fire'][l]['mean_t']['inter-BPSR']['diff'] < 0 for l in LAYERS)
w(f"- 18층 중 inter 의 평균 최초 발화가 BPSR 보다 이른 층 {neg}/18, 그중 p<0.05 {nsig}/18 (층마다 Welch, 보정 전).")
w(f"- 18층 전체: inter−BPSR {WD(fa['mean_t']['inter-BPSR'],3)} 스텝, inter−baseline {WD(fa['mean_t']['inter-baseline'],3)}, BPSR−baseline {WD(fa['mean_t']['BPSR-baseline'],3)}. 규제가 발화를 늦추는 것은 두 방법 공통이고, **같은 스파이크 감소에서 inter 가 덜 늦춘다.** ablation_mech 2b 와 같은 결론이다 (같은 npz).")
w('- 해석 한계: inter 에는 final_step (벌점 가중을 T 합산 발화로 정함) 이 있고 BPSR 에는 없지만, 두 방법은 가중 방식 전체가 다르므로 이 차이를 final_step 에 돌릴 수 없다. `- final_step` 런 npz 가 없다.')
w('')

# 5
w('## 5. 같은 입력에 대한 예측 일치\n')
w('| 비교 | 쌍 수 | 예측 일치율 % | 불일치 장수 (1만 장 중) |')
w('|---|---|---|---|')
for k, q in res['agreement'].items():
    w(f"| {k} | {q['n']} | {MS(q,2)} | {F((100-q['mean'])*100,0)} |")
w('')
tr = res['triples']; nr = res['nine_runs']
w(f"- 세 팔에서 한 런씩 뽑은 27조: 모두 맞춤 {MS(tr['all_correct'],1)}장, 모두 틀림 {MS(tr['all_wrong'],1)}장. 9런 전부: 모두 맞춤 {K_(nr['all_correct'])}, 모두 틀림 {K_(nr['all_wrong'])}, 엇갈림 {K_(nr['mixed'])}장.")
w('')
w('McNemar — a 만 맞춤 vs b 만 맞춤 (런 쌍별 정확 이항):\n')
w('| 비교 (a vs b) | 쌍 수 | a 만 맞춤 | b 만 맞춤 | 순차 (a−b) | 불일치 합 | p<0.05 쌍 |')
w('|---|---|---|---|---|---|---|')
for a1, a2 in ((I, P), (I, B), (P, B)):
    q = res['mcnemar'][f'{a1} vs {a2}']
    w(f"| {a1} vs {a2} | 9 | {MS(q['only_a'],1)} | {MS(q['only_b'],1)} | {MS(q['net'],1)} | {MS(q['discord'],1)} | {q['n_p_lt_005']}/9 |")
for a in AN:
    q = res['mcnemar'][a + ' 시드끼리']
    w(f"| (기준) {a} 시드끼리 | 3 | — | — | \\|순차\\| {MS(q['net_abs'],1)} | {MS(q['discord'],1)} | {q['n_p_lt_005']}/3 |")
w('')
w(f'inter vs BPSR 런 쌍 9개:\n')
w('| inter 런 | BPSR 런 | inter 만 맞춤 | BPSR 만 맞춤 | p |')
w('|---|---|---|---|---|')
for x in mcIP['pairs']:
    w(f"| {x['a'][-2:]} | {x['b'][-2:]} | {x['only_a']} | {x['only_b']} | {PV(x['p'])} |")
w('')
sv = res['seed_vote']
w('이미지별 3시드 다수결:\n')
w('| 비교 (a vs b) | a 3/3 맞춤 & b 0/3 | a 0/3 & b 3/3 | a 다수 맞춤 & b 다수 틀림 | 반대 | p (다수결, 이항) |')
w('|---|---|---|---|---|---|')
for a1, a2 in ((I, P), (I, B), (P, B)):
    q = sv[f'{a1} vs {a2}']
    w(f"| {a1} vs {a2} | {q['a3_b0']} | {q['a0_b3']} | {q['a_maj_b_min']} | {q['a_min_b_maj']} | {PV(q['p_maj'])} |")
w('')
w('3시드 중 맞춘 수 히스토그램 (0/1/2/3 시드, 1만 장): ' + '; '.join(f"{a} {sv['hist'][a]}" for a in AN))
w('')
w(f"- 팔 사이 불일치 (inter×BPSR {F((100-res['agreement'][P+' × '+I]['mean'])*100,0)}장) 는 같은 팔 시드 사이 불일치 ({F((100-res['agreement'][I+' 안']['mean'])*100,0)}·{F((100-res['agreement'][P+' 안']['mean'])*100,0)}장) 와 같은 크기다. **inter 는 BPSR 과 '다른 이미지를 맞추는' 모델이 아니다** — 시드를 바꾼 것만큼만 다르다.")
w('- 다수결 검정은 이미지를 단위로 하므로 이 6개 모델에 대한 진술이다. 방법 차이의 단위는 시드다.')
w('')

w('## [LIMITATION]\n')
w('- **모든 팔 n=3, 한 ρ/λ 점.** inter ρ 1.5e-3, BPSR λ 3e-11 한 쌍만 있다. 스파이크는 원장 기준 inter 가 BPSR 보다 약 1.6% 적어 완전히 같은 점은 아니다. 다른 스파이크 구간 (300K 대 등) 의 보정·클래스·일치도는 npz_t 가 없어 모른다.')
w('- Welch t 는 자유도 약 2~4 라 검정력이 낮다. 여기서 "차이 없음" 은 "시드 SD 수준보다 큰 차이는 없다" 는 뜻이지 동등성 증명이 아니다. 정확도 시드 SD 0.10~0.13%p 보다 작은 효과는 이 설계로 보이지 않는다.')
w('- 클래스 10개·혼동 90칸·층 18개 검정은 다중비교 보정을 하지 않았다. 개별 p<0.05 를 발견으로 읽지 않는다.')
w('- 보정은 label smoothing 0.1 학습 모델의 것이다. 소신 방향은 LS 의 결과이고, LS 없이 학습했을 때 규제가 보정을 어떻게 바꾸는지는 이 자료로 말할 수 없다. 최적 온도는 테스트셋에 맞춘 값이다.')
w('- 스텝별 정확도는 T=4 로 학습한 망을 짧게 추론한 것이다. 처음부터 T=2 로 학습한 망과의 비교가 아니다.')
w('- 최초 발화 차이를 final_step 등 특정 요소에 돌릴 수 없다 (`- final_step` npz 없음). inter 와 BPSR 은 가중 방식 전체가 다르다.')
w('- 테스트 1만 장 1회 추론 (결정적) 이다. 데이터 증강·다른 데이터셋 (C100, DVS) 으로 일반화되는지는 모른다.')
open(OUT_MD, 'w').write('\n'.join(L_) + '\n')
print('wrote', OUT_JSON, OUT_MD)
