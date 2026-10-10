"""ablation_mech_261010 보충 — 새 npz 2개로 두 미해결 질문을 닫는다 (R19-C10, 26-10-10).

질문 (사용자): "vmem 이 들어가면 같은 값에서 더 많이 발화를 하나?"
  1. vmem       — `- vmem` 3e-3 s1 (_npz_t) vs inter 3e-3 s1~s3 (_npz, 총합 키만), 같은 ρ.
                  어떤 뉴런이 더 쏘나: 발화율 구간별, 층별, 채널 단위.
  2. final_step — `- final_step` 2.2e-3 s1 (_npz_t) vs inter 1.5e-3 s1~3 / baseline s11~13 /
                  BPSR 3e-11 s1~3 (_npz_t): 스텝 몫, 최초 발화 스텝, 스텝별 정확도.

뉴런 대응에 대해: 독립 학습 망은 채널 번호가 정렬돼 있지 않다 (decisions 2026-09-20).
같은 시드 s1 끼리도 채널 단위 상관이 0 근처다 (이 스크립트의 'pairing_check').
그래서 (층, 위치, 채널) 로 뉴런을 짝짓지 않고, 층 안 발화율 **순위**로 짝짓는다
(순위 대응: 층마다 발화율을 정렬해 k 번째끼리 비교 = 분위수 비교). 순위 구간은
baseline s11~13 의 같은 순위 발화율 평균으로 정한다.

입력 (읽기 전용):
  _npz/r19c10-ours_ch-3e-3-s{1,2,3}.npz            (총합·__fired 만)
  _npz_t/*.npz                                     (스텝별 키 포함, 10-10 추출)
  scratchpad/led1007b/inventory.json               (원장, 10-10 14:20 집계)
쓰는 법:  ~/anaconda3/envs/venv_1/bin/python analysis/ablation_mech_addendum_261010.py   → .json, .md
"""
import json, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
INV = S + '/led1007b/inventory.json'
OUT_JSON = os.path.join(ROOT, 'analysis', 'ablation_mech_addendum_261010.json')
OUT_MD = os.path.join(ROOT, 'analysis', 'ablation_mech_addendum_261010.md')

STAGES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1']
LAYERS = ['conv1_conv_n'] + [f'conv2_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')] + \
         [f'conv3_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')] + \
         [f'conv4_block{b}_{s}_n' for b in (1, 2) for s in ('conv1', 'out')] + ['fc1_n']
BINS = [('0', None, 0.0), ('0~1%', 0.0, 0.01), ('1~5%', 0.01, 0.05), ('5~20%', 0.05, 0.20), ('20%+', 0.20, 9.0)]


def stage(l):
    return l.split('_')[0]


def P(tag, d='_npz_t'):
    return os.path.join(ROOT, d, tag + '.npz')


G = {
    'inter3': [('r19c10-ours_ch-3e-3-s%d' % i, P('r19c10-ours_ch-3e-3-s%d' % i, '_npz')) for i in (1, 2, 3)],
    'novmem': [('r19c10-ch_novmem-3e-3-s1', P('r19c10-ch_novmem-3e-3-s1'))],
    'base': [('r19c10-base---s%d' % i, P('r19c10-base---s%d' % i)) for i in (11, 12, 13)],
    'inter15': [('r19c10-ours_ch-1p5e-3-s%d' % i, P('r19c10-ours_ch-1p5e-3-s%d' % i)) for i in (1, 2, 3)],
    'bpsr': [('r19c10-bpsr-3e-11-s%d' % i, P('r19c10-bpsr-3e-11-s%d' % i)) for i in (1, 2, 3)],
    'nofinal': [('r19c10-ch_nofinal-2p2e-3-s1', P('r19c10-ch_nofinal-2p2e-3-s1'))],
}
NAME = {'inter3': 'ours_inter_ch 3e-3', 'novmem': '- vmem 3e-3', 'base': 'baseline',
        'inter15': 'ours_inter_ch 1.5e-3', 'bpsr': 'BPSR 3e-11', 'nofinal': '- final_step 2.2e-3'}

inv = {r['tag']: r for r in json.load(open(INV))['runs']}
Z = {}
for g, lst in G.items():
    for tag, p in lst:
        Z[tag] = np.load(p)


def zpred(v, arr):
    """n=1 값이 inter 시드 분포에서 얼마나 벗어나나. 새 런 하나의 예측 SD = SD·sqrt(1+1/n)."""
    a = np.asarray(arr, float); n = len(a); sd = a.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return float('nan'), float('nan')
    z = (v - a.mean()) / (sd * np.sqrt(1 + 1 / n))
    p = 2 * stats.t.sf(abs(z), n - 1)
    return float(z), float(p)


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); s = x.sum()
    if s <= 0:
        return float('nan')
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * s))


def rates(z, l):
    return z[l].astype(np.float64) / (float(z['n_samples']) * int(z['T']))


res = {'inputs': {g: [t for t, _ in lst] for g, lst in G.items()}}

# ---------------------------------------------------------------- 착지값
land = {}
for tag, z in Z.items():
    ns = float(z['n_samples'])
    spk = sum(z[l].sum() for l in LAYERS) / ns
    r = inv.get(tag, {})
    land[tag] = {k: r.get(k) for k in ('server', 'epochs', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes')}
    land[tag].update({'npz_spk': float(spk), 'npz_acc': float(z['acc'])})
res['landing'] = land

# ---------------------------------------------------------------- 0. 짝짓기 검사
pc = {}
for a, b in [('r19c10-ours_ch-3e-3-s1', 'r19c10-ch_novmem-3e-3-s1'), ('r19c10-ours_ch-3e-3-s1', 'r19c10-ours_ch-3e-3-s2'),
             ('r19c10-base---s11', 'r19c10-base---s12')]:
    rn, rc, rs = [], [], []
    for l in LAYERS:
        x = Z[a][l]; y = Z[b][l]
        rn.append(np.corrcoef(x.ravel(), y.ravel())[0, 1])
        if x.ndim == 3:
            rc.append(np.corrcoef(x.sum((0, 1)), y.sum((0, 1)))[0, 1])
            rs.append(np.corrcoef(x.sum(2).ravel(), y.sum(2).ravel())[0, 1])
    pc[f'{a} | {b}'] = {'neuron_r_median': float(np.median(rn)), 'channel_r_median': float(np.median(rc)),
                        'spatial_r_median': float(np.median(rs)), 'neuron_r_range': [float(min(rn)), float(max(rn))],
                        'channel_r_range': [float(min(rc)), float(max(rc))]}
res['pairing_check'] = pc

# ---------------------------------------------------------------- 1. vmem
NV = 'r19c10-ch_novmem-3e-3-s1'
I3 = [t for t, _ in G['inter3']]
BS = [t for t, _ in G['base']]
TOTN = sum(Z[NV][l].size for l in LAYERS)

# 1a-1 자기 발화율 구간 (짝짓기 없음): 뉴런 비율 %, 스파이크 몫 %
def own_bins(tag):
    z = Z[tag]
    r = np.concatenate([rates(z, l).ravel() for l in LAYERS])
    o = {}
    for nm, lo, hi in BINS:
        m = (r == 0) if lo is None else ((r > lo) & (r <= hi))
        o[nm] = {'neu_pct': float(100 * m.mean()), 'spk_pct': float(100 * r[m].sum() / r.sum())}
    return o


ob = {t: own_bins(t) for t in I3 + [NV] + BS}
sec1 = {'own_bins': ob}

# 1a-2 순위 대응: baseline 순위 구간
ref = {}
for l in LAYERS:
    ref[l] = np.mean([np.sort(rates(Z[t], l).ravel()) for t in BS], axis=0)
ref_all = np.concatenate([ref[l] for l in LAYERS])


def rank_bins(tag):
    xs = np.concatenate([np.sort(rates(Z[tag], l).ravel()) for l in LAYERS])
    o = {}
    for nm, lo, hi in BINS:
        m = (ref_all == 0) if lo is None else ((ref_all > lo) & (ref_all <= hi))
        o[nm] = {'n_neu': int(m.sum()), 'ref_rate': float(ref_all[m].mean()) if m.any() else float('nan'),
                 'rate': float(xs[m].mean()) if m.any() else float('nan'),
                 'silent_pct': float(100 * (xs[m] == 0).mean()) if m.any() else float('nan'),
                 'spk_per_img': float(xs[m].sum() * 4)}  # 이미지당 스파이크 = 발화율 합 × T
    return o


rb = {t: rank_bins(t) for t in I3 + [NV] + BS}
rbz = {}
for nm, *_ in BINS:
    rbz[nm] = {}
    for k in ('rate', 'silent_pct', 'spk_per_img'):
        arr = [rb[t][nm][k] for t in I3]
        v = rb[NV][nm][k]
        z, p = zpred(v, arr)
        rbz[nm][k] = {'inter_mean': float(np.mean(arr)), 'inter_sd': float(np.std(arr, ddof=1)), 'novmem': v,
                      'diff': v - float(np.mean(arr)), 'rel_pct': 100 * (v / np.mean(arr) - 1) if np.mean(arr) > 0 else float('nan'),
                      'z': z, 'p': p}
sec1['rank_bins'] = rb
sec1['rank_bins_z'] = rbz

# 1b 층별: 순위 대응에서 inter 가 더 쏘는 순위 수 vs 덜 쏘는 순위 수, 층 스파이크
lay = {}
for l in LAYERS:
    N = Z[NV][l].size
    nv = np.sort(rates(Z[NV], l).ravel())
    more, less, net = [], [], []
    for t in I3:
        x = np.sort(rates(Z[t], l).ravel())
        more.append(int((x > nv).sum())); less.append(int((x < nv).sum()))
        net.append(100 * (more[-1] - less[-1]) / N)
    nulln = []
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        xa = np.sort(rates(Z[I3[a]], l).ravel()); xb = np.sort(rates(Z[I3[b]], l).ravel())
        nulln.append(100 * ((xa > xb).sum() - (xa < xb).sum()) / N)
    spk_i = [Z[t][l].sum() / float(Z[t]['n_samples']) for t in I3]
    spk_v = Z[NV][l].sum() / float(Z[NV]['n_samples'])
    sil_i = [100 * (Z[t][l] == 0).mean() for t in I3]
    sil_v = 100 * (Z[NV][l] == 0).mean()
    lay[l] = {'N': N, 'inter_more': more, 'inter_less': less, 'net_pct': net,
              'null_abs_net_pct': [float(abs(v)) for v in nulln],
              'spk_inter': spk_i, 'spk_novmem': float(spk_v),
              'spk_rel_pct': 100 * (spk_v / np.mean(spk_i) - 1), 'spk_z': zpred(spk_v, spk_i)[0],
              'silent_inter': sil_i, 'silent_novmem': float(sil_v), 'silent_z': zpred(sil_v, sil_i)[0]}
sec1['layers'] = lay

# 1b' 층 종류별 묶음: stem (conv1_conv), 블록 안쪽 conv1 (각 잔차 블록의 첫 conv), 블록 출력 (out), fc1
def ltype(l):
    return 'stem' if l == 'conv1_conv_n' else ('fc1' if l == 'fc1_n' else ('inner' if l.endswith('conv1_n') else 'out'))


grp_l = {}
for l, o in lay.items():
    for key in (ltype(l), ltype(l) + ':' + stage(l)):
        a = grp_l.setdefault(key, {'N': 0, 'sil_i': np.zeros(3), 'sil_v': 0.0, 'spk_i': np.zeros(3), 'spk_v': 0.0})
        a['N'] += o['N']
        a['sil_i'] += np.array(o['silent_inter']) / 100 * o['N']; a['sil_v'] += o['silent_novmem'] / 100 * o['N']
        a['spk_i'] += np.array(o['spk_inter']); a['spk_v'] += o['spk_novmem']
for k, a in grp_l.items():
    zs, ps = zpred(a['sil_v'], a['sil_i']); zp, pp = zpred(a['spk_v'], a['spk_i'])
    grp_l[k] = {'N': a['N'], 'silent_inter': a['sil_i'].tolist(), 'silent_novmem': a['sil_v'],
                'silent_diff': a['sil_v'] - a['sil_i'].mean(), 'silent_z': zs, 'silent_p': ps,
                'spk_inter': a['spk_i'].tolist(), 'spk_novmem': a['spk_v'],
                'spk_rel_pct': 100 * (a['spk_v'] / a['spk_i'].mean() - 1), 'spk_z': zp, 'spk_p': pp}
sec1['by_type'] = grp_l

# 1c 채널 단위
def chan(tag):
    z = Z[tag]; ns = float(z['n_samples'])
    o = {'dead_ch': 0, 'dead_by_stage': {s: 0 for s in STAGES}, 'n_ch': 0}
    gw = tw = sw = 0.0
    gst = {s: [0.0, 0.0] for s in STAGES}
    act = 0.0; sil_neu = 0
    for l in LAYERS:
        c = z[l].astype(np.float64)
        ch = c.reshape(-1, c.shape[-1]).sum(0) if c.ndim == 3 else c
        d = int((ch == 0).sum()); o['dead_ch'] += d; o['dead_by_stage'][stage(l)] += d; o['n_ch'] += ch.size
        s = ch.sum(); g = gini(ch)
        k = max(1, int(round(0.1 * ch.size)))
        top = np.sort(ch)[::-1][:k].sum() / s
        gw += g * s; tw += top * s; sw += s
        gst[stage(l)][0] += g * s; gst[stage(l)][1] += s
        act += z[l + '__fired'].sum(); sil_neu += int((c == 0).sum())
    o['gini_ch'] = gw / sw; o['top10_ch_pct'] = 100 * tw / sw
    o['gini_ch_by_stage'] = {s: gst[s][0] / gst[s][1] for s in STAGES if gst[s][1] > 0}
    o['active_img_pct'] = 100 * act / ns / TOTN
    o['silent_neu_pct'] = 100 * sil_neu / TOTN
    return o


ch = {t: chan(t) for t in I3 + [NV] + BS}
chz = {}
for k in ['dead_ch', 'gini_ch', 'top10_ch_pct', 'active_img_pct', 'silent_neu_pct'] + \
         [f'dead_{s}' for s in STAGES[:4]] + [f'gini_{s}' for s in STAGES[:4]]:
    def get(t):
        if k.startswith('dead_') and k != 'dead_ch':
            return ch[t]['dead_by_stage'][k[5:]]
        if k.startswith('gini_') and k != 'gini_ch':
            return ch[t]['gini_ch_by_stage'][k[5:]]
        return ch[t][k]
    arr = [get(t) for t in I3]; v = get(NV)
    z_, p_ = zpred(v, arr)
    chz[k] = {'inter': arr, 'inter_mean': float(np.mean(arr)), 'inter_sd': float(np.std(arr, ddof=1)),
              'novmem': v, 'diff': v - float(np.mean(arr)), 'z': z_, 'p': p_,
              'base_mean': float(np.mean([get(t) for t in BS]))}
sec1['channel'] = ch
sec1['channel_z'] = chz
res['vmem'] = sec1

# ---------------------------------------------------------------- 2. final_step (시간축)
def tmetrics(tag):
    z = Z[tag]; ns = float(z['n_samples']); T = int(z['T'])
    tot_t = np.zeros(T); first = np.zeros(T + 1)
    st_t = {s: np.zeros(T) for s in STAGES}; st_f = {s: np.zeros(T + 1) for s in STAGES}
    for l in LAYERS:
        a = z[l + '__t'].reshape(T, -1).sum(1).astype(np.float64)
        f = z[l + '__first_t'].astype(np.float64)
        tot_t += a; first += f; st_t[stage(l)] += a; st_f[stage(l)] += f
    o = {'spk_per_img': float(tot_t.sum() / ns), 'step_share': (100 * tot_t / tot_t.sum()).tolist()}
    fired = first[:T].sum()
    o['first_share'] = (100 * first[:T] / fired).tolist()
    o['mean_first_t'] = float((np.arange(1, T + 1) * first[:T]).sum() / fired)
    o['silent_pair_pct'] = float(100 * first[T] / first.sum())
    for s in STAGES:
        o[f'last_{s}'] = float(100 * st_t[s][T - 1] / st_t[s].sum())
        o[f'first1_{s}'] = float(100 * st_t[s][0] / st_t[s].sum())
        ff = st_f[s][:T].sum()
        o[f'mft_{s}'] = float((np.arange(1, T + 1) * st_f[s][:T]).sum() / ff)
    lt = z['logits_t']; lab = z['labels']
    o['acc_t'] = [float(100 * (lt[:, t].argmax(1) == lab).mean()) for t in range(T)]
    o['acc_final'] = float(100 * (z['pred'] == lab).mean())
    return o


TM = {}
for g in ('base', 'inter15', 'bpsr', 'nofinal', 'novmem'):
    for t, _ in G[g]:
        TM[t] = tmetrics(t)
I15 = [t for t, _ in G['inter15']]; BP = [t for t, _ in G['bpsr']]; NF = 'r19c10-ch_nofinal-2p2e-3-s1'


def scal(o, k):
    if k.startswith('step'):
        return o['step_share'][int(k[-1]) - 1]
    if k.startswith('fs'):
        return o['first_share'][int(k[-1]) - 1]
    if k.startswith('acc'):
        return o['acc_t'][int(k[-1]) - 1]
    return o[k]


KEYS = ['spk_per_img', 'step1', 'step2', 'step3', 'step4', 'fs1', 'fs2', 'fs3', 'fs4', 'mean_first_t', 'silent_pair_pct'] + \
       [f'last_{s}' for s in STAGES] + [f'first1_{s}' for s in STAGES] + [f'mft_{s}' for s in STAGES] + ['acc1', 'acc2', 'acc3', 'acc4']
grp = {}
for g, tl in (('base', BS), ('inter15', I15), ('bpsr', BP)):
    grp[g] = {k: {'mean': float(np.mean([scal(TM[t], k) for t in tl])), 'sd': float(np.std([scal(TM[t], k) for t in tl], ddof=1))}
              for k in KEYS}
cmp2 = {}
for k in KEYS:
    arr = [scal(TM[t], k) for t in I15]
    v = scal(TM[NF], k); z_, p_ = zpred(v, arr)
    cmp2[k] = {'nofinal': v, 'novmem': scal(TM[NV], k), 'inter15_mean': float(np.mean(arr)),
               'diff_vs_inter15': v - float(np.mean(arr)), 'z': z_, 'p': p_,
               'diff_vs_base': v - grp['base'][k]['mean']}
# 스파이크 수준 보정 (방향만): baseline → inter 1.5e-3 두 묶음 평균을 ln(spikes) 직선으로 잇고 288K 에서 읽는다
trend = {}
xb = np.log(grp['base']['spk_per_img']['mean']); xi = np.log(grp['inter15']['spk_per_img']['mean'])
for k in ['step1', 'step4', 'fs1', 'fs4', 'mean_first_t'] + [f'last_{s}' for s in STAGES[:4]] + [f'first1_{s}' for s in STAGES[:4]]:
    yb = grp['base'][k]['mean']; yi = grp['inter15'][k]['mean']
    b = (yi - yb) / (xi - xb)
    # 예측 SE: 두 묶음 평균의 SE 전파 + 새 런 하나의 시드 SD (inter 1.5e-3)
    sdi = grp['inter15'][k]['sd']; sdb = grp['base'][k]['sd']
    o = {'slope_per_ln': float(b)}
    for nm, tg in (('nofinal', NF), ('novmem', NV)):
        d = (np.log(TM[tg]['spk_per_img']) - xi) / (xi - xb)
        pred = yi + d * (yi - yb)
        se = np.sqrt((1 + d) ** 2 * sdi ** 2 / 3 + d ** 2 * sdb ** 2 / 3 + sdi ** 2)
        obs = scal(TM[tg], k)
        o.update({f'pred_{nm}': float(pred), f'obs_{nm}': obs, f'resid_{nm}': obs - float(pred),
                  f'se_{nm}': float(se), f'z_{nm}': float((obs - pred) / se), f'extrap_{nm}': float(d)})
    trend[k] = o
res['final_step'] = {'runs': TM, 'groups': grp, 'nofinal_vs': cmp2, 'trend': trend}

json.dump(res, open(OUT_JSON, 'w'), indent=1, ensure_ascii=False, default=float)
print('wrote', OUT_JSON)

# ================================================================ md
I4L = ['r19c10-ours_ch-3e-3-s%d' % i for i in (1, 2, 3, 4)]
led_i4 = [inv[t]['spikes'] for t in I4L]
led_v = inv[NV]['spikes']
led_z = zpred(led_v, led_i4)
res['vmem']['ledger_spikes_3e3'] = {'inter_s1_4': led_i4, 'novmem': led_v, 'rel_pct': 100 * (led_v / np.mean(led_i4) - 1),
                                    'z': led_z[0], 'p': led_z[1]}
npz_i3 = [land[t]['npz_spk'] for t in I3]
npz_z = zpred(land[NV]['npz_spk'], npz_i3)
res['vmem']['npz_spikes'] = {'inter_s1_3': npz_i3, 'novmem': land[NV]['npz_spk'],
                             'rel_pct': 100 * (land[NV]['npz_spk'] / np.mean(npz_i3) - 1), 'z': npz_z[0], 'p': npz_z[1]}
json.dump(res, open(OUT_JSON, 'w'), indent=1, ensure_ascii=False, default=float)

L = []
w = L.append


def F(x, d=2, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    return (f'{x:+,.{d}f}' if sign else f'{x:,.{d}f}')


def K(x):
    return f'{x:,.0f}'


def Pv(p):
    return '—' if not np.isfinite(p) else (f'{p:.2g}' if p >= 0.001 else f'{p:.1e}')


v1 = res['vmem']; f2 = res['final_step']
cz = v1['channel_z']; bt = v1['by_type']; rbz = v1['rank_bins_z']; tr = f2['trend']; c2 = f2['nofinal_vs']
lz = v1['ledger_spikes_3e3']; nzs = v1['npz_spikes']

w('# ablation_mech_261010 보충 — `- vmem`·`- final_step` npz 로 두 미해결 질문 닫기 (R19-C10, 26-10-10)\n')
w('생성: `analysis/ablation_mech_addendum_261010.py` → `.json`, 이 문서. 원장 = `led1007b/inventory.json` (10-10 14:20 집계). '
  'npz = `_npz/` (inter 3e-3 s1~s3, 뉴런별 총합만), `_npz_t/` (나머지, 스텝별 키 포함, 10-10 14:28·14:45 추출분 포함). '
  '모두 테스트 1만 장 1회 추론 (평가 모드, 저장된 베스트 가중치).\n')
w('## 요약\n')
w(f"1. **vmem 은 같은 ρ 에서 총 스파이크를 거의 바꾸지 않고, 발화율이 낮은 뉴런을 살려 둔다.** ρ 3e-3 에서 `- vmem` (n=1) 의 스파이크는 inter (원장 n=4) 대비 {F(lz['rel_pct'],1,True)}% (z={F(lz['z'],1,True)}) 로 시드 산포 안이다. "
  f"대신 테스트셋 전체에서 한 번도 안 쏜 뉴런이 {F(cz['silent_neu_pct']['novmem'],2)}% vs {F(cz['silent_neu_pct']['inter_mean'],2)} ±{F(cz['silent_neu_pct']['inter_sd'],2)}% (inter npz n=3, z={F(cz['silent_neu_pct']['z'],1,True)}, 약 {K(cz['silent_neu_pct']['diff']/100*TOTN)}개) 로 늘고, "
  f"늘어난 침묵은 **잔차 블록 안쪽 conv1 층** (conv2 단계 +{K(bt['inner:conv2']['silent_diff'])}개, z={F(bt['inner:conv2']['silent_z'],1,True)}) 과 baseline 순위 발화율 0~5% 구간에 몰린다. 5~20%·20%+ 구간은 차이가 없다.")
w(f"2. **'문턱 근처 뉴런을 살린다' 는 '발화율 낮은 뉴런' 으로 읽으면 맞고, '중간 발화율 뉴런' 이나 '입력층' 으로 읽으면 틀린다.** 차이가 큰 곳은 0~1%·1~5% 구간 (침묵 몫 +{F(rbz['0~1%']['silent_pct']['diff'],1)}·+{F(rbz['1~5%']['silent_pct']['diff'],1)}%p, z≈3.5~3.7, 자유도 2 라 p≈0.07) 이고, stem conv1 은 z={F(bt['stem']['silent_z'],1,True)} 로 불분명하다. 채널 단위 (죽은 채널 {K(cz['dead_ch']['novmem'])} vs {F(cz['dead_ch']['inter_mean'],0)}, 지니, 상위 10% 몫) 는 구분되지 않는다.")
w(f"3. **final_step 을 빼면 스파이크가 뒤 스텝으로 조금 밀린다 — 방향만.** `- final_step` (n=1, {K(TM[NF]['spk_per_img'])}/이미지) 의 t4 몫은 {F(TM[NF]['step_share'][3],2)}% 로 inter 1.5e-3 (n=3, {K(f2['groups']['inter15']['spk_per_img']['mean'])}) 보다 {F(c2['step4']['diff_vs_inter15'],2,True)}%p 높지만, 그중 약 {F(tr['step4']['pred_nofinal']-f2['groups']['inter15']['step4']['mean'],1)}%p 는 스파이크 수준 차로 예상되는 몫이다. 추세 대비 잔차는 {F(tr['step4']['resid_nofinal'],2,True)} ±{F(tr['step4']['se_nofinal'],2)}%p (z={F(tr['step4']['z_nofinal'],1,True)}) 로 **판단 불가**, 단계별로 conv1 의 t4 몫만 {F(tr['last_conv1']['resid_nofinal'],2,True)}%p (z={F(tr['last_conv1']['z_nofinal'],1,True)}) 로 분명하다.")
w(f"4. **스텝별 정확도는 n=1 이라 판단 불가.** `- final_step` T'=2/3/4 = {F(TM[NF]['acc_t'][1])} / {F(TM[NF]['acc_t'][2])} / {F(TM[NF]['acc_t'][3])}. ablation_mech_261010.md 에서 바뀌는 문장은 6곳 (아래 3절). 핵심: 1d 의 '입력층 쪽 문턱 근처 뉴런' → '블록 안쪽 conv1 층의 저발화 뉴런', 2b 의 'final_step 고유 효과로 볼 근거 없음' → '빼면 조금 뒤로 밀린다 (방향만)', 요약 1 의 '같은 ρ 에서 5~6% 더 쏜다' 에 'ρ 3e-3 에서는 −1.2%' 단서.\n")

w('## 정의\n')
w('- **발화율** = 뉴런 하나의 테스트셋 총 스파이크 / (1만 장 × T=4). 0~1 사이.')
w('- **침묵 뉴런** = 테스트 1만 장 전체에서 한 번도 안 쏜 뉴런 (원 문서 1d 의 "이미지당 침묵" 과 다르다).')
w('- **이미지당 활성 뉴런 %** = Σ(그 뉴런이 쏜 이미지 수) / (1만 × 전체 뉴런 1,442,048).')
w('- **순위 대응**: 층마다 발화율을 오름차순 정렬해 같은 순위 k 끼리 비교 (분위수 비교). **순위 구간** = baseline s11~13 의 같은 순위 발화율 평균이 속한 구간 (0, 0~1%, 1~5%, 5~20%, 20%+).')
w('- **z (n=1 대 inter)**: (n=1 값 − inter 평균) / (inter 시드 SD·√(1+1/n)). inter 시드 n=3 이라 자유도 2 — |z|≥4.3 이어야 p<0.05 (t 분포). p 는 이 t 분포로 계산.')
w('- **채널**: conv 층의 채널 하나 = 공간 위치 전체 합. fc1 은 뉴런 하나가 채널. 18층 4,736 채널. **죽은 채널** = 테스트셋 합 0. **지니**·**상위 10% 몫** = 층 안 채널 합의 집중도를 층 스파이크로 가중 평균.')
w('- **층 종류**: stem = conv1_conv (1층), 블록 안쪽 = 각 잔차 블록의 첫 conv (`*_conv1_n`, 8층), 블록 출력 = `*_out_n` (8층), fc1.')
w('- **스텝 몫** = 스텝 t 스파이크 / 4스텝 합. **최초 발화 스텝** = (뉴런, 이미지) 쌍이 처음 쏜 스텝 (쏜 쌍만). **T\'=t 정확도** = `logits_t[:, t]` 의 argmax 정확도 (원 문서 6절과 같은 정의).\n')

w('## 0. 입력 런과 착지값\n')
w('| 팔 | seed | 서버 | train loss | train acc | val loss | val acc | spikes | npz spikes/이미지 | npz acc | npz 스텝별 키 |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
def lrow(nm, t, has_t):
    o = land[t]; sd = t.rsplit('-', 1)[1]
    w(f"| {nm} | {sd} | {o['server']} | {F(o['train_loss'],4)} | {F(o['train_acc'])} | {F(o['val_loss'],4)} | {F(o['val_acc'])} | {K(o['spikes'])} | {K(o['npz_spk'])} | {F(o['npz_acc'])} | {has_t} |")
def mrow(nm, tl):
    ks = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes', 'npz_spk', 'npz_acc']
    m = {k: (np.mean([land[t][k] for t in tl]), np.std([land[t][k] for t in tl], ddof=1)) for k in ks}
    w(f"| **{nm} 평균** | **n={len(tl)}** | | {F(m['train_loss'][0],4)} ±{F(m['train_loss'][1],4)} | {F(m['train_acc'][0])} ±{F(m['train_acc'][1])} | {F(m['val_loss'][0],4)} ±{F(m['val_loss'][1],4)} | **{F(m['val_acc'][0])} ±{F(m['val_acc'][1])}** | **{K(m['spikes'][0])} ±{K(m['spikes'][1])}** | {K(m['npz_spk'][0])} ±{K(m['npz_spk'][1])} | {F(m['npz_acc'][0])} | |")
for t in I3:
    lrow('ours_inter_ch 3e-3', t, '없음')
o4 = inv['r19c10-ours_ch-3e-3-s4']
w(f"| ours_inter_ch 3e-3 | s4 | {o4['server']} | {F(o4['train_loss'],4)} | {F(o4['train_acc'])} | {F(o4['val_loss'],4)} | {F(o4['val_acc'])} | {K(o4['spikes'])} | npz 없음 | — | — |")
mrow('ours_inter_ch 3e-3 (npz 있는 s1~s3)', I3)
w(f"| *ours_inter_ch 3e-3 원장 s1~s4* | *n=4* | | | | | *{F(np.mean([inv[t]['val_acc'] for t in I4L]))} ±{F(np.std([inv[t]['val_acc'] for t in I4L], ddof=1))}* | *{K(np.mean(led_i4))} ±{K(np.std(led_i4, ddof=1))}* | | | |")
lrow('- vmem 3e-3', NV, '있음')
for t in BS:
    lrow('baseline', t, '있음')
mrow('baseline', BS)
for t in I15:
    lrow('ours_inter_ch 1.5e-3', t, '있음')
mrow('ours_inter_ch 1.5e-3', I15)
for t in BP:
    lrow('BPSR 3e-11', t, '있음')
mrow('BPSR 3e-11', BP)
lrow('- final_step 2.2e-3', NF, '있음')
w('')
w(f"- npz 스파이크는 원장 (평가 모드, 베스트 에폭 검증값) 보다 5~8% 높다 — 같은 가중치의 테스트셋 1회 추론이라 프로토콜이 다르다. 비교는 npz 끼리만 한다.")
w(f"- 같은 ρ 3e-3 의 총 스파이크: 원장 `- vmem` {K(lz['novmem'])} vs inter s1~s4 {K(np.mean(led_i4))} ±{K(np.std(led_i4, ddof=1))} → {F(lz['rel_pct'],1,True)}% (z={F(lz['z'],1,True)}, p={Pv(lz['p'])}); "
  f"npz {K(nzs['novmem'])} vs s1~s3 {K(np.mean(npz_i3))} → {F(nzs['rel_pct'],1,True)}% (z={F(nzs['z'],1,True)}). **이 짝은 사실상 같은 스파이크 수준이다.** 아래 1절 차이는 총량이 아니라 배분의 차이다.")
w(f"- inter 3e-3 s4 는 npz 가 없어 (추출 안 됨) 1절의 inter 는 n=3 이다.\n")

w('## 1. vmem — 어떤 뉴런이 더 쏘나 (`- vmem` 3e-3 s1 vs inter 3e-3 s1~s3, 같은 ρ)\n')
w('### 1.0 뉴런을 위치로 짝지을 수 없다\n')
pcs = res['pairing_check']
w('| 짝 | 뉴런 단위 상관 (18층 중앙값, 범위) | 채널 합 상관 (중앙값, 범위) | 공간 위치 합 상관 (중앙값) |')
w('|---|---|---|---|')
for k, o in pcs.items():
    w(f"| {k.replace('r19c10-','').replace(' | ', ' vs ')} | {F(o['neuron_r_median'],3,True)} ({F(o['neuron_r_range'][0],2,True)}~{F(o['neuron_r_range'][1],2,True)}) | {F(o['channel_r_median'],3,True)} ({F(o['channel_r_range'][0],2,True)}~{F(o['channel_r_range'][1],2,True)}) | {F(o['spatial_r_median'],3)} |")
w('')
w('- 같은 시드 (s1) 의 inter 와 `- vmem` 도 채널 번호가 대응하지 않는다 (채널 합 상관 ≈0). 공간 위치만 대응한다 (≈0.9, 입력 영상 통계 때문). 그래서 "같은 층·같은 위치·같은 채널" 로 짝지은 뉴런별 차이는 의미가 없다 (decisions 2026-09-20 과 같은 이유). **이하 모두 순위 대응** 이다.\n')

w('### 1a. baseline 순위 구간별 발화율·침묵 비율\n')
rb = v1['rank_bins']
w('| 순위 구간 (baseline 발화율) | 뉴런 수 | baseline 발화율 평균 | inter 발화율 (n=3) | `- vmem` 발화율 | 비 | z | inter 침묵 % (n=3) | `- vmem` 침묵 % | 차 %p | z | p | 이미지당 스파이크 inter → `- vmem` |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for nm, *_ in BINS:
    a = rbz[nm]; r0 = rb[I3[0]][nm]
    w(f"| {nm} | {K(r0['n_neu'])} | {F(100*r0['ref_rate'],2)}% | {F(100*a['rate']['inter_mean'],3)}% ±{F(100*a['rate']['inter_sd'],3)} | {F(100*a['rate']['novmem'],3)}% | "
      f"{F(a['rate']['novmem']/a['rate']['inter_mean'],3) if nm!='0' else '—'} | {F(a['rate']['z'],1,True) if nm!='0' else '—'} | "
      f"{F(a['silent_pct']['inter_mean'],2)} ±{F(a['silent_pct']['inter_sd'],2)} | {F(a['silent_pct']['novmem'],2)} | {F(a['silent_pct']['diff'],2,True)} | {F(a['silent_pct']['z'],1,True)} | {Pv(a['silent_pct']['p'])} | "
      f"{K(a['spk_per_img']['inter_mean'])} → {K(a['spk_per_img']['novmem'])} |")
w('')
w('자기 발화율 구간 (짝짓기 없음): 뉴런 비율 % / 스파이크 몫 %')
w('')
w('| 런 | 0 | 0~1% | 1~5% | 5~20% | 20%+ |')
w('|---|---|---|---|---|---|')
for t in I3 + [NV] + BS:
    o = v1['own_bins'][t]
    w(f"| {t.replace('r19c10-','')} | " + ' | '.join(f"{F(o[b]['neu_pct'],2)} / {F(o[b]['spk_pct'],1)}" for b, *_ in BINS) + ' |')
w('')
w(f"- `- vmem` 은 baseline 기준 하위 순위 (0~1%, 1~5%) 에서 침묵이 늘고 발화율이 낮다 (침묵 +{F(rbz['0~1%']['silent_pct']['diff'],1)}·+{F(rbz['1~5%']['silent_pct']['diff'],1)}%p, 발화율 {F(100*(rbz['0~1%']['rate']['novmem']/rbz['0~1%']['rate']['inter_mean']-1),1,True)}·{F(100*(rbz['1~5%']['rate']['novmem']/rbz['1~5%']['rate']['inter_mean']-1),1,True)}%). "
  f"5~20% 는 발화율 {F(100*(rbz['5~20%']['rate']['novmem']/rbz['5~20%']['rate']['inter_mean']-1),1,True)}% (z={F(rbz['5~20%']['rate']['z'],1,True)}), 20%+ 는 {F(100*(rbz['20%+']['rate']['novmem']/rbz['20%+']['rate']['inter_mean']-1),1,True)}% 로 같다.")
w(f"- 각 z 는 3.5~3.7 이지만 자유도 2 라 p≈0.07 이다. 이 표 하나로는 유의하지 않다. 18층 전체 침묵 뉴런 비율 (1c, z={F(cz['silent_neu_pct']['z'],1,True)}, p={Pv(cz['silent_neu_pct']['p'])}) 이 같은 현상의 합계 지표다.")
w(f"- 스파이크 수로는 작다: 0~5% 구간에서 `- vmem` 이 덜 쏜 양은 이미지당 약 {K(rbz['0~1%']['spk_per_img']['inter_mean']+rbz['1~5%']['spk_per_img']['inter_mean']-rbz['0~1%']['spk_per_img']['novmem']-rbz['1~5%']['spk_per_img']['novmem'])} (전체 npz 스파이크의 약 {F(100*(rbz['0~1%']['spk_per_img']['inter_mean']+rbz['1~5%']['spk_per_img']['inter_mean']-rbz['0~1%']['spk_per_img']['novmem']-rbz['1~5%']['spk_per_img']['novmem'])/np.mean(npz_i3),1)}%). vmem 은 스파이크 총량보다 **쏘는 뉴런의 수** 를 바꾼다.\n")

w('### 1b. 층별 — 순위 대응에서 inter 가 더 쏘는 순위 vs 덜 쏘는 순위\n')
w('순 비율 = (inter 가 더 쏘는 순위 수 − 덜 쏘는 순위 수) / 층 뉴런 수, inter 시드 3개 각각. 대조 = inter 시드끼리 같은 계산의 |순 비율| (3쌍). 층 스파이크는 이미지당.')
w('')
w('| 층 | 뉴런 | 순 비율 % (inter s1/s2/s3 − `- vmem`) | 대조 \\|순 비율\\| % (inter 끼리 3쌍) | inter 층 스파이크 (n=3) | `- vmem` | 비 % | z | 침묵 % inter (n=3) | `- vmem` | z |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
for l, o in v1['layers'].items():
    flag = ' *' if (min(o['net_pct']) > max(o['null_abs_net_pct']) or max(o['net_pct']) < -max(o['null_abs_net_pct'])) else ''
    w(f"| {l[:-2]}{flag} | {K(o['N'])} | {' / '.join(F(x,1,True) for x in o['net_pct'])} | {' / '.join(F(x,1) for x in o['null_abs_net_pct'])} | "
      f"{K(np.mean(o['spk_inter']))} ±{K(np.std(o['spk_inter'],ddof=1))} | {K(o['spk_novmem'])} | {F(o['spk_rel_pct'],1,True)} | {F(o['spk_z'],1,True)} | "
      f"{F(np.mean(o['silent_inter']),2)} ±{F(np.std(o['silent_inter'],ddof=1),2)} | {F(o['silent_novmem'],2)} | {F(o['silent_z'],1,True)} |")
w('')
w('`*` = inter 세 시드 모두 같은 방향이고 대조의 최댓값보다 크다.')
w('')
w('층 종류별 합계:')
w('')
w('| 묶음 | 뉴런 | 침묵 뉴런 수 inter (n=3) | `- vmem` | 차 | z | p | 이미지당 스파이크 inter (n=3) | `- vmem` | 비 % | z | p |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|')
for k in ['stem', 'inner', 'inner:conv2', 'inner:conv3', 'inner:conv4', 'out', 'out:conv2', 'out:conv3', 'out:conv4', 'fc1']:
    o = bt[k]
    nm = {'stem': 'stem (conv1_conv)', 'inner': '블록 안쪽 conv1 (8층)', 'out': '블록 출력 (8층)', 'fc1': 'fc1'}.get(k, '　' + k.replace('inner:', '안쪽 ').replace('out:', '출력 '))
    w(f"| {nm} | {K(o['N'])} | {K(np.mean(o['silent_inter']))} ±{K(np.std(o['silent_inter'],ddof=1))} | {K(o['silent_novmem'])} | {F(o['silent_diff'],0,True)} | {F(o['silent_z'],1,True)} | {Pv(o['silent_p'])} | "
      f"{K(np.mean(o['spk_inter']))} ±{K(np.std(o['spk_inter'],ddof=1))} | {K(o['spk_novmem'])} | {F(o['spk_rel_pct'],1,True)} | {F(o['spk_z'],1,True)} | {Pv(o['spk_p'])} |")
w('')
w(f"- 순위 대응의 '더 쏘는 순위 수' 는 대부분 층에서 inter 시드끼리의 차와 같은 크기라 신호가 아니다. 세 시드가 모두 대조를 넘는 층은 표의 `*` 뿐이다.")
w(f"- 신호는 **블록 안쪽 conv1 층** 에 모인다: `- vmem` 이 이 8층에서 스파이크 {F(bt['inner']['spk_rel_pct'],1,True)}% (z={F(bt['inner']['spk_z'],1,True)}, p={Pv(bt['inner']['spk_p'])}), 침묵 뉴런 +{K(bt['inner']['silent_diff'])} (z={F(bt['inner']['silent_z'],1,True)}, p={Pv(bt['inner']['silent_p'])}) — 대부분 conv2 단계 (+{K(bt['inner:conv2']['silent_diff'])}). "
  f"블록 출력 8층은 스파이크 {F(bt['out']['spk_rel_pct'],1,True)}% (z={F(bt['out']['spk_z'],1,True)}), 침묵 {F(bt['out']['silent_diff'],0,True)} (z={F(bt['out']['silent_z'],1,True)}) 로 차이가 없다. "
  f"총 스파이크 차 ({K(nzs['novmem']-np.mean(npz_i3))}/이미지, 시드 산포 안) 는 거의 전부 블록 안쪽 층 ({K(bt['inner']['spk_novmem']-np.mean(bt['inner']['spk_inter']))}) 에서 나온다. 즉 vmem 이 더 쏘게 하는 곳은 블록 안쪽 층뿐이고, 그 양은 전체의 약 1.5% 라 총량 비교에서는 시드 산포에 묻힌다.")
w(f"- 블록 안쪽 conv1 은 발화율이 낮고 침묵 뉴런이 많은 층이다 (conv2 단계 inter 침묵: 안쪽 24~64%, 출력 0.3~18%). 1a 의 '하위 순위' 뉴런이 많이 모인 곳이다.\n")

w('### 1c. 채널 단위\n')
w('| 지표 | inter s1 / s2 / s3 | inter 평균 ±SD (n=3) | `- vmem` (n=1) | 차 | z | p | baseline 평균 (n=3, 참고) |')
w('|---|---|---|---|---|---|---|---|')
lab = {'dead_ch': '죽은 채널 (/4,736)', 'gini_ch': '채널 지니 (층 스파이크 가중)', 'top10_ch_pct': '상위 10% 채널 몫 %',
       'active_img_pct': '이미지당 활성 뉴런 %', 'silent_neu_pct': '침묵 뉴런 % (테스트셋 전체)',
       'dead_conv1': '죽은 채널 conv1 (/128)', 'dead_conv2': '죽은 채널 conv2 (/768)', 'dead_conv3': '죽은 채널 conv3 (/1,536)', 'dead_conv4': '죽은 채널 conv4 (/2,048)',
       'gini_conv1': '지니 conv1', 'gini_conv2': '지니 conv2', 'gini_conv3': '지니 conv3', 'gini_conv4': '지니 conv4'}
for k, nm in lab.items():
    o = cz[k]; d = 0 if k.startswith('dead') else (3 if 'gini' in k else 2)
    w(f"| {nm} | {' / '.join(F(x,d) for x in o['inter'])} | {F(o['inter_mean'],d)} ±{F(o['inter_sd'],d)} | {F(o['novmem'],d)} | {F(o['diff'],d,True)} | {F(o['z'],1,True)} | {Pv(o['p'])} | {F(o['base_mean'],d)} |")
w('')
w(f"- 채널 수준에서는 vmem 효과가 보이지 않는다. 죽은 채널 {F(cz['dead_ch']['diff'],0,True)} (z={F(cz['dead_ch']['z'],1,True)}), 지니 {F(cz['gini_ch']['diff'],3,True)}, 상위 10% 몫 {F(cz['top10_ch_pct']['diff'],2,True)}%p — 모두 시드 산포 안. "
  f"단계별 죽은 채널은 conv2 {F(cz['dead_conv2']['diff'],0,True)} (z={F(cz['dead_conv2']['z'],1,True)}, p={Pv(cz['dead_conv2']['p'])}), conv4 {F(cz['dead_conv4']['diff'],0,True)} (z={F(cz['dead_conv4']['z'],1,True)}, p={Pv(cz['dead_conv4']['p'])}) 로 서로 반대이고, 13개 지표 중 둘이라 결론으로 쓰지 않는다.")
w(f"- 늘어난 침묵 뉴런 ({F(cz['silent_neu_pct']['diff'],2,True)}%p ≈ {K(cz['silent_neu_pct']['diff']/100*TOTN)}개) 은 채널을 통째로 죽이지 않고 살아 있는 채널 안에 흩어져 있다.")
w(f"- 이미지당 활성 뉴런은 {F(cz['active_img_pct']['diff'],2,True)}%p (z={F(cz['active_img_pct']['z'],1,True)}) — 원 문서 1d 의 학습 모드 '이미지당 침묵 +0.07 (p=0.38)' 과 같은 결론 (이미지 하나 단위로는 안 보이고, 1만 장 누적에서 보인다).\n")

w('### 1d. 답 — "vmem 이 들어가면 같은 값에서 더 많이 발화하나?"\n')
w(f"- **스파이크 수로는: ρ 3e-3 에서는 아니다** (원장 {F(lz['rel_pct'],1,True)}%, z={F(lz['z'],1,True)}; n=1 대 n=4). 원 문서의 '같은 ρ 에서 −5.2%' 는 ρ 1e-3~5e-3 전체 회귀 (inter n=17 / `- vmem` n=6) 의 평균 오프셋이고, ρ 3e-3 한 점은 그 안에서 차이가 작은 쪽이다.")
w(f"- **쏘는 뉴런 수로는: 그렇다.** 같은 ρ·같은 총 스파이크에서 vmem 이 있으면 테스트셋에서 한 번이라도 쏘는 뉴런이 {F(cz['silent_neu_pct']['diff'],2)}%p (약 {K(cz['silent_neu_pct']['diff']/100*TOTN)}개) 많다 (`- vmem` 침묵 뉴런 z={F(cz['silent_neu_pct']['z'],1,True)}, p={Pv(cz['silent_neu_pct']['p'])}, inter n=3 대 n=1).")
w(f"- **어떤 뉴런이냐: 발화율 하위 (baseline 순위 0~5%) 이고 블록 안쪽 conv1 층 (특히 conv2 단계)** 이다. 중간 발화율 (5~20%) 과 상위 (20%+) 는 같다. 이 뉴런들의 막전위가 실제로 문턱 근처였는지는 npz 에 막전위가 없어 직접 확인하지 못했다 — '드물게 쏘는 뉴런을 살린다' 까지가 측정된 사실이다.\n")

w('## 2. final_step — 시간 효과 (`- final_step` 2.2e-3 s1 vs inter 1.5e-3 / baseline / BPSR)\n')
w(f"**같은 스파이크 짝이 아니다.** `- final_step` {K(TM[NF]['spk_per_img'])}/이미지 (원장 288K) vs inter 1.5e-3 {K(f2['groups']['inter15']['spk_per_img']['mean'])} (원장 405K), baseline {K(f2['groups']['base']['spk_per_img']['mean'])}, BPSR {K(f2['groups']['bpsr']['spk_per_img']['mean'])}. "
  f"스파이크를 줄이면 스텝 분포 자체가 뒤로 밀리므로 (원 문서 2b), 직접 차이에서 스파이크 수준 몫을 빼고 방향만 본다. "
  f"추세 = baseline (n=3) → inter 1.5e-3 (n=3) 두 묶음 평균을 ln(spikes) 직선으로 이어 `- final_step` 스파이크에서 읽은 값 (외삽 거리 = 두 묶음 간격의 {F(tr['step4']['extrap_nofinal'],1)}배). "
  f"± = 두 평균의 SE 전파 + 새 런 하나의 시드 SD. 직선 가정의 오차는 포함 안 됨 — 그래서 **final_step 이 켜진 `- vmem` 3e-3 ({K(TM[NV]['spk_per_img'])}) 을 검사점** 으로 같이 적는다 (vmem 도 빠져 있어 순수 대조는 아니다).\n")
w('### 2a. 전체 스텝 몫·최초 발화 스텝\n')
w('| 묶음 | n | spikes/이미지 (npz) | t1 % | t2 % | t3 % | t4 % | 최초 발화 t1 % | t2 % | t3 % | t4 % | 평균 최초 발화 스텝 | 침묵 (뉴런·이미지) % |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
def trow(nm, tl):
    def m(k):
        a = [scal(TM[t], k) for t in tl]
        return F(np.mean(a), 2) + (f" ±{F(np.std(a, ddof=1), 2)}" if len(a) > 1 else '')
    def m3(k):
        a = [scal(TM[t], k) for t in tl]
        return F(np.mean(a), 3) + (f" ±{F(np.std(a, ddof=1), 3)}" if len(a) > 1 else '')
    w(f"| {nm} | {len(tl)} | {K(np.mean([TM[t]['spk_per_img'] for t in tl]))} | {m('step1')} | {m('step2')} | {m('step3')} | {m('step4')} | {m('fs1')} | {m('fs2')} | {m('fs3')} | {m('fs4')} | {m3('mean_first_t')} | {m('silent_pair_pct')} |")
for t in BS: trow(t.replace('r19c10-', ''), [t])
trow('**baseline 평균**', BS)
for t in I15: trow(t.replace('r19c10-', ''), [t])
trow('**ours_inter_ch 1.5e-3 평균**', I15)
for t in BP: trow(t.replace('r19c10-', ''), [t])
trow('**BPSR 3e-11 평균**', BP)
trow('- vmem 3e-3 s1 (검사점, final_step 켜짐)', [NV])
trow('**- final_step 2.2e-3 s1**', [NF])
w('')
w('### 2b. `- final_step` − 스파이크 수준 추세\n')
w('| 지표 | inter 1.5e-3 (n=3) | `- final_step` (n=1) | 직접 차 (z) | 추세 예측 | 잔차 ± SE | z | 검사점 `- vmem` 잔차 ± SE (z) |')
w('|---|---|---|---|---|---|---|---|')
tl = {'step1': 't1 스파이크 몫 %', 'step4': 't4 스파이크 몫 %', 'fs1': '최초 발화 t1 몫 %', 'fs4': '최초 발화 t4 몫 %', 'mean_first_t': '평균 최초 발화 스텝',
      'last_conv1': 't4 몫 conv1 %', 'last_conv2': 't4 몫 conv2 %', 'last_conv3': 't4 몫 conv3 %', 'last_conv4': 't4 몫 conv4 %',
      'first1_conv1': 't1 몫 conv1 %', 'first1_conv2': 't1 몫 conv2 %', 'first1_conv3': 't1 몫 conv3 %', 'first1_conv4': 't1 몫 conv4 %'}
for k, nm in tl.items():
    d = 3 if k == 'mean_first_t' else 2
    o = tr[k]; c = c2[k]
    w(f"| {nm} | {F(c['inter15_mean'],d)} | {F(c['nofinal'],d)} | {F(c['diff_vs_inter15'],d,True)} ({F(c['z'],1,True)}) | {F(o['pred_nofinal'],d)} | {F(o['resid_nofinal'],d,True)} ±{F(o['se_nofinal'],d)} | {F(o['z_nofinal'],1,True)} | {F(o['resid_novmem'],d,True)} ±{F(o['se_novmem'],d)} ({F(o['z_novmem'],1,True)}) |")
w('')
w(f"- 직접 차는 크다 (t4 몫 {F(c2['step4']['diff_vs_inter15'],2,True)}%p, t1 몫 {F(c2['step1']['diff_vs_inter15'],2,True)}%p, 최초 발화 {F(c2['mean_first_t']['diff_vs_inter15'],3,True)} 스텝; |z|≥10) — 그러나 대부분 스파이크가 28% 적은 탓이다. 추세 예측과의 잔차는 t4 {F(tr['step4']['resid_nofinal'],2,True)} ±{F(tr['step4']['se_nofinal'],2)}, t1 {F(tr['step1']['resid_nofinal'],2,True)} ±{F(tr['step1']['se_nofinal'],2)}, 평균 최초 발화 {F(tr['mean_first_t']['resid_nofinal'],3,True)} 로 |z|≈1 이하.")
w(f"- 방향은 일관된다: **final_step 을 빼면 t4 몫↑, t1 몫↓ (스파이크가 뒤 스텝으로)**. 검사점 `- vmem` (final_step 켜짐) 은 t4 {F(tr['step4']['resid_novmem'],2,True)}, t1 {F(tr['step1']['resid_novmem'],2,True)} 로 반대쪽이다. "
  f"단계별로는 conv1 의 t4 몫 잔차 {F(tr['last_conv1']['resid_nofinal'],2,True)}%p (z={F(tr['last_conv1']['z_nofinal'],1,True)}) 가 검사점 ({F(tr['last_conv1']['resid_novmem'],2,True)}, z={F(tr['last_conv1']['z_novmem'],1,True)}) 의 약 {F(tr['last_conv1']['resid_nofinal']/tr['last_conv1']['resid_novmem'],0)}배로 가장 분명하다. 검사점 자체도 z=3 이 나오므로 SE 가 직선 가정 오차를 덜 담고 있다 — conv1 결과도 '방향' 으로만 읽는다.")
w(f"- 설계와 맞는 방향이다: final_step 이 꺼지면 매 스텝 누적 스파이크로 벌점을 다시 매기므로 (원 문서 2a 주의), 이른 스텝 스파이크가 남은 스텝 수만큼 여러 번 벌점에 들어간다. final_step 은 스파이크 시점과 무관하게 한 번만 센다.")
w(f"- 비슷한 스파이크의 BPSR (final_step 없음, 3e-11, n=3) 도 inter 1.5e-3 보다 t4 {F(f2['groups']['bpsr']['step4']['mean']-f2['groups']['inter15']['step4']['mean'],2,True)}%p, conv1 t4 몫 {F(f2['groups']['bpsr']['last_conv1']['mean']-f2['groups']['inter15']['last_conv1']['mean'],2,True)}%p 로 같은 방향이다 (다른 방법이라 보조 근거).\n")
w('### 2c. 스텝별 정확도 (logits_t)\n')
w("| 런·묶음 | n | T'=1 | T'=2 | T'=3 | T'=4 | T'=2 − T'=4 | T'=3 − T'=4 |")
w('|---|---|---|---|---|---|---|---|')
def arow(nm, tl):
    a = np.array([TM[t]['acc_t'] for t in tl])
    m = a.mean(0); s = a.std(0, ddof=1) if len(tl) > 1 else None
    d2 = a[:, 1] - a[:, 3]; d3 = a[:, 2] - a[:, 3]
    ff = lambda i: F(m[i]) + (f" ±{F(s[i])}" if s is not None else '')
    fd = lambda d: F(d.mean(), 2, True) + (f" ±{F(d.std(ddof=1))}" if len(d) > 1 else '')
    w(f"| {nm} | {len(tl)} | {ff(0)} | {ff(1)} | {ff(2)} | {ff(3)} | {fd(d2)} | {fd(d3)} |")
arow('baseline', BS); arow('ours_inter_ch 1.5e-3', I15); arow('BPSR 3e-11', BP); arow('- vmem 3e-3 s1', [NV]); arow('**- final_step 2.2e-3 s1**', [NF])
w('')
w(f"- `- final_step` 의 T'=2/3 손실 ({F(TM[NF]['acc_t'][1]-TM[NF]['acc_t'][3],2,True)}·{F(TM[NF]['acc_t'][2]-TM[NF]['acc_t'][3],2,True)}) 은 baseline (n=3) 범위 안이다. n=1 이고 시드 SD 0.10~0.13%p 보다 작은 차이라 **판단 불가**. 스파이크가 뒤로 밀린 것이 T' 절단 정확도를 해친다는 근거는 없다.\n")

w('## 3. ablation_mech_261010.md 와의 대조 — 바뀌는 문장\n')
w('| 위치 | 원 문장 (요지) | 바뀌는 내용 |')
w('|---|---|---|')
w(f"| 요약 1 / 1b 답 | 'vmem 이 있으면 같은 ρ 에서 inter 는 약 5~6% 더 쏜 채로 착지한다' | 회귀 평균 (ρ 1e-3~5e-3, n=17/6) 으로는 유지. 단 ρ 3e-3 한 점은 {F(lz['rel_pct'],1,True)}% (z={F(lz['z'],1,True)}) 로 시드 산포 안 — '모든 ρ 에서 5% 더' 로 읽으면 안 된다는 단서를 붙인다. |")
w(f"| 1b 둘째 줄 / [LIMITATION] 2번 | '뉴런 하나하나의 더 쏘는 뉴런은 npz 가 없어 재지 못했다' | 측정함. 단 뉴런 위치 짝짓기는 불가능 (같은 시드도 채널 상관 ≈0) → 순위 대응으로 대체. 결과: 같은 총 스파이크에서 vmem 이 있으면 한 번이라도 쏘는 뉴런 +{F(cz['silent_neu_pct']['diff'],2)}%p (z={F(cz['silent_neu_pct']['z'],1)}), 하위 순위 (0~5%)·블록 안쪽 conv1 층. |")
w(f"| 1d 방향 해석 | '\"vmem 이 더 쏘게 하는 뉴런 = 입력층 쪽의 문턱 근처 뉴런\" 가설과 맞지만 … npz 가 필요' | **'입력층 쪽' 은 틀림.** stem conv1 침묵 차는 z={F(bt['stem']['silent_z'],1,True)} 로 불분명하고, 차이는 conv2 단계의 블록 안쪽 conv1 층 (+{K(bt['inner:conv2']['silent_diff'])}개, z={F(bt['inner:conv2']['silent_z'],1,True)}). '발화율 중간 구간' 도 아님 — 하위 구간. |")
w(f"| 요약 2 마지막 문장 / 2b 셋째 줄 / [LIMITATION] 3번 | '`- final_step` npz 가 없어 요소 단독의 시간 효과는 재지 못했다' | 측정함 (n=1, 스파이크 수준 다름). final_step 을 빼면 스텝 분포가 뒤로 조금 밀린다 (t4 잔차 {F(tr['step4']['resid_nofinal'],2,True)} ±{F(tr['step4']['se_nofinal'],2)}, conv1 에서만 분명). |")
w(f"| 2b 둘째 줄 | '스파이크를 줄이면 t1·t2 발화가 늦어지는 것은 규제 일반 현상이고 final_step 고유의 효과로 볼 근거가 없다' | 앞 절반 유지. 뒤 절반은 수정: final_step 은 그 늦어짐을 **조금 줄이는 쪽** 으로 작용하는 것으로 보인다 (방향만, conv1 에서 가장 큼). inter 가 BPSR 보다 덜 뒤로 미는 차 (원 문서 −0.94%p) 의 일부가 final_step 에서 올 수 있다 — 크기 배분은 n=1 로 판단 불가. |")
w(f"| 요약 2 첫 문장 | 'final_step — 스파이크를 마지막 스텝으로 몰지 않는다' | 유지 (inter 전체는 baseline 과 거의 같다). 덧붙일 것: '빼면 오히려 조금 뒤로 몰린다'. |")
w('')
w('바뀌지 않는 것: 1c (vmem 이 w·R 을 낮춘다), 1e (같은 스파이크 정확도), 2a (final_step 은 같은 ρ 에서 덜 줄이게 한다, −22%), 3·4·5·6 절.\n')

w('## [LIMITATION]\n')
w('- **`- vmem`·`- final_step` 은 각 n=1** 이다. z 는 inter 시드 n=3 의 산포 (자유도 2) 기준이라 |z|<4.3 은 p≥0.05 다. 1절 결론은 합계 지표 (침묵 뉴런 %, 블록 안쪽 층 스파이크) 에만 기대고, 구간·층별 개별 값은 방향으로만 쓴다.')
w('- inter 3e-3 s4 의 npz 가 없어 1절 inter 는 n=3 (s1~s3) 이다. 원장 스파이크 비교만 n=4.')
w('- **뉴런 짝짓기는 순위 대응** 이다. "같은 뉴런이 더 쏜다" 가 아니라 "같은 분위의 뉴런이 더 쏜다" 다. 기능적 정렬 (클래스 반응으로 채널 매칭) 은 하지 않았다.')
w('- **막전위는 측정하지 않았다.** "문턱 근처" 는 발화율이 낮다는 것으로 대신했다. 침묵 뉴런의 막전위 분포를 보려면 추출기에 막전위 저장을 더해야 한다 (코드 변경 — eip-code 몫).')
w('- 2절은 같은 스파이크 짝이 아니다 (311K vs 431K npz). 추세는 두 묶음 평균을 잇는 직선의 외삽 (간격의 2배) 이고, 검사점 `- vmem` 에서도 |z|≈3 잔차가 나와 SE 가 작게 잡혀 있다. 크기는 말하지 않고 방향만 말한다. 같은 스파이크로 보려면 inter ρ 를 `- final_step` 288K 근처로 맞춘 런 (예: inter 5e-3 계열) 의 npz_t 가 필요하다.')
w('- 2절의 검사점 `- vmem` 도 vmem 이 빠진 팔이라 "final_step 켜짐" 의 순수 대조가 아니다.')
w('- 1b 의 층별 표는 18층 × 2지표를 검정했다. 다중 비교 보정은 하지 않았고, 개별 층 결과는 결론에 쓰지 않았다.')
w('- npz 는 테스트셋 1회 추론이고 원장 스파이크 (검증, 베스트 에폭) 보다 5~8% 높다. 원 문서의 학습 모드 지표와도 정의가 다르다.')
w('- R19·CIFAR-10·T=4, inter (ours_inter_ch) 한 방법만이다. intra 의 `- vmem` (원 문서: 같은 ρ −22.5%) 은 npz 가 없어 같은 분석을 못 했다.')
open(OUT_MD, 'w').write('\n'.join(L) + '\n')
print('wrote', OUT_MD)
