"""세 경쟁 범위 팔 + 대조군의 뉴런별 스파이크 분포 비교 (26-10-02).

R19-C10, 테스트 10,000장, 스파이킹 18층(n_in·predictions_n 제외) 전체 뉴런 기준.
`analysis/npz_silent_firing_w1.py` 를 본떴다 (그 파일은 고치지 않았다).

팔 (방법 이름 = 런 키)
  ours_inter_ch       = ours_ch        (채널끼리 경쟁)
  ours_layer (w1)     = ours_layer_w1  (층 전체 경쟁, w1 가중)
  ours_intra_ch (w1)  = ours_w1        (채널 안 경쟁, w1 가중)
  plain L2 = l2,  BPSR = bpsr,  baseline = base--

뉴런 분류
  침묵    테스트 전체에서 한 번도 발화하지 않음
  드묾    발화는 했지만 발화한 표본이 1% 미만 (100장 미만)
지표
  스파이크/샘플          테스트셋 기준 (학습 CSV 의 val spikes 와 프로토콜이 같지 않을 수 있다)
  활성 뉴런당 스파이크   스파이크/샘플 / (침묵 아닌 뉴런 수)
  50% 집중도            스파이크 절반을 내는 뉴런이 발화 뉴런 중 몇 %인가
  죽은 채널             공간 전체가 침묵인 채널 (4,736 = conv 4,480 + fc1 256; fc1 은 뉴런 = 채널)
  산 채널 안 침묵 %      conv 층에서 죽지 않은 채널에 속한 위치(뉴런) 중 침묵 비율
                        -> "살아남은 채널 안을 깎는가" 의 지표
  산 채널당 스파이크/샘플 conv 층 스파이크 / 산 conv 채널 수
단계: conv1 (conv1_conv) / conv2 (6층) / conv3 (6층) / conv4 (4층) / fc (fc1)

같은 스파이크에서의 비교: 대역 안 Welch t 와 함께, 대역을 합쳐
  y = a + b*ln(스파이크/샘플) + c_arm   (ANCOVA, 기준 팔 = ours_inter_ch) 를 낸다.

쓰는 법:  python analysis/npz_three_arms_261002.py
          -> analysis/npz_three_arms_261002.json, analysis/npz_three_arms_261002.md
"""
import json, os
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, '_npz')
INV = ('/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/'
       '291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad/ledger/inventory.json')
OUT_JSON = os.path.join(ROOT, 'analysis', 'npz_three_arms_261002.json')
OUT_MD = os.path.join(ROOT, 'analysis', 'npz_three_arms_261002.md')
SKIP = {'n_in', 'predictions_n'}
STAGES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc']

INTER, LAYER, INTRA = 'ours_inter_ch', 'ours_layer (w¹)', 'ours_intra_ch (w¹)'
BANDS = [
    ('약 330K', [(INTER, '3e-3', 'ours_ch-3e-3', [1, 2, 3]),
                (LAYER, '1e-3', 'ours_layer_w1-1e-3', [1, 2, 3]),
                (INTRA, '1e-3', 'ours_w1-1e-3', [1, 2, 3]),
                ('plain L2', '1e-7', 'l2-1e-7', [1, 2]),
                ('BPSR', '1e-10', 'bpsr-1e-10', [1, 2, 3])]),
    ('약 370K', [(INTER, '2e-3', 'ours_ch-2e-3', [1, 2, 3]),
                (LAYER, '7e-4', 'ours_layer_w1-7e-4', [1, 2, 3]),
                (INTRA, '7e-4', 'ours_w1-7e-4', [1, 2, 3])]),
    ('약 400K', [(INTER, '1.5e-3', 'ours_ch-1p5e-3', [1, 2, 3]),
                (LAYER, '5e-4', 'ours_layer_w1-5e-4', [1, 2, 3]),
                ('plain L2', '5e-8', 'l2-5e-8', [1, 2]),
                ('BPSR', '3e-11', 'bpsr-3e-11', [1, 2, 3])]),
    ('기준', [('baseline', '-', 'base--', [11, 12, 13])]),
]

inv = {x['tag']: x for x in json.load(open(INV))['runs']}


def stage_of(k):
    if k.startswith('conv1_'):
        return 'conv1'
    for s in ('conv2', 'conv3', 'conv4'):
        if k.startswith(s + '_'):
            return s
    return 'fc'


def gini(x):
    x = np.sort(x)
    n = x.size
    if n == 0 or x.sum() == 0:
        return np.nan
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


def metrics(path):
    z = np.load(path)
    ns = float(z['n_samples'])
    tot = spk = sil = rare = dead = nch = 0
    alive_pos = alive_pos_sil = 0
    conv_spk = conv_alive_ch = 0
    ginis = []
    act_counts = []
    stage = {s: dict(n=0, sil=0, ch=0, dead=0, spk=0.0) for s in STAGES}
    for k in z.files:
        if k.endswith('__fired') or k + '__fired' not in z.files or k in SKIP or k in ('n_samples', 'T', 'acc'):
            continue
        c = z[k].astype(np.float64)
        f = z[k + '__fired'].astype(np.float64)
        n = c.size
        s0 = int((c == 0).sum())
        r = int(((c > 0) & (f < 0.01 * ns)).sum())
        tot += n; spk += c.sum(); sil += s0; rare += r
        act_counts.append(c[c > 0].ravel())
        st = stage[stage_of(k)]
        st['n'] += n; st['sil'] += s0; st['spk'] += c.sum() / ns
        if c.ndim == 3:
            cc = c.reshape(-1, c.shape[-1])            # [위치, 채널]
            ch = cc.sum(0)
            alive = ch > 0
            d = int((~alive).sum())
            dead += d; nch += c.shape[-1]; st['ch'] += c.shape[-1]; st['dead'] += d
            alive_pos += int(alive.sum()) * cc.shape[0]
            alive_pos_sil += int((cc[:, alive] == 0).sum())
            conv_spk += ch.sum() / ns; conv_alive_ch += int(alive.sum())
            ginis += [gini(cc[:, j]) for j in np.where(alive)[0]]
        else:                                          # fc1: 뉴런 하나 = 채널 하나
            d = int((c == 0).sum())
            dead += d; nch += c.size; st['ch'] += c.size; st['dead'] += d
    act = np.sort(np.concatenate(act_counts))[::-1]
    half = int(np.searchsorted(np.cumsum(act), act.sum() / 2) + 1)
    nfire = tot - sil
    return {'neurons': tot, 'spk_per_sample': spk / ns, 'test_acc': float(z['acc']),
            'silent_pct': 100 * sil / tot, 'rare_pct': 100 * rare / tot,
            'spk_per_active': spk / ns / nfire, 'half_share_pct': 100 * half / nfire,
            'dead_ch': dead, 'channels': nch,
            'alive_ch_silent_pct': 100 * alive_pos_sil / alive_pos,
            'spk_per_alive_ch': conv_spk / conv_alive_ch,
            'alive_ch_gini': float(np.nanmean(ginis)),
            'stage_silent_pct': {s: 100 * v['sil'] / v['n'] for s, v in stage.items()},
            'stage_dead_ch': {s: v['dead'] for s, v in stage.items()},
            'stage_channels': {s: v['ch'] for s, v in stage.items()},
            'stage_spk': {s: v['spk'] for s, v in stage.items()}}


out = {'bands': {}, 'missing': [], 'pairs': [], 'ancova': {}}
allrows = []
for band, arms in BANDS:
    bo = out['bands'][band] = {}
    for name, knob, key, seeds in arms:
        rows = []
        for s in seeds:
            tag = f'r19c10-{key}-s{s}'
            p = os.path.join(NPZ, tag + '.npz')
            iv = inv.get(tag, {})
            if not os.path.exists(p):
                out['missing'].append(tag); print('npz 없음', tag); continue
            m = metrics(p)
            m.update(tag=tag, seed=s, server=iv.get('server', '?'),
                     **{k: iv.get(k) for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes')},
                     exclude=iv.get('exclude'), exclude_reason=iv.get('exclude_reason'))
            rows.append(m)
            allrows.append((band, name, m))
        bo[name] = {'knob': knob, 'runs': rows}


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else None), len(v)


KEYS = ['silent_pct', 'rare_pct', 'dead_ch', 'alive_ch_silent_pct', 'spk_per_alive_ch',
        'alive_ch_gini', 'spk_per_active', 'half_share_pct', 'spk_per_sample', 'test_acc']
for band, arms in out['bands'].items():
    names = list(arms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, B = arms[a]['runs'], arms[b]['runs']
            rec = {'band': band, 'a': a, 'b': b, 'n_a': len(A), 'n_b': len(B)}
            for k in KEYS:
                x = [r[k] for r in A]; y = [r[k] for r in B]
                d = float(np.mean(x) - np.mean(y)) if x and y else None
                se = (float(np.sqrt(np.var(x, ddof=1) / len(x) + np.var(y, ddof=1) / len(y)))
                      if len(x) > 1 and len(y) > 1 else None)
                p = (float(stats.ttest_ind(x, y, equal_var=False).pvalue)
                     if len(x) >= 3 and len(y) >= 3 else None)
                rec[k] = {'diff': d, 'se': se, 'p': p}
            out['pairs'].append(rec)

# ANCOVA: 같은 스파이크에서의 팔 오프셋 (기준 = ours_inter_ch), baseline 제외
arms_all = [INTER, LAYER, INTRA, 'plain L2', 'BPSR']
R = [(n, m) for _, n, m in allrows if n in arms_all]
if sum(1 for n, _ in R if n == INTER) == 0 or len(R) < 8:
    R = []
for k in ([] if not R else ['dead_ch', 'silent_pct', 'alive_ch_silent_pct', 'spk_per_alive_ch', 'alive_ch_gini', 'rare_pct', 'half_share_pct', 'spk_per_active', 'test_acc']):
    y = np.array([m[k] for _, m in R], float)
    X = [np.ones(len(R)), np.log([m['spk_per_sample'] for _, m in R])]
    cols = ['a', 'b_lnspk']
    for a in arms_all[1:]:
        if not any(n == a for n, _ in R):
            continue
        X.append(np.array([1.0 if n == a else 0.0 for n, _ in R])); cols.append(a)
    X = np.array(X).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = res @ res / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    pv = 2 * stats.t.sf(np.abs(tval), dof)
    out['ancova'][k] = {'n': len(y), 'dof': dof,
                        'n_per_arm': {a: sum(1 for n, _ in R if n == a) for a in arms_all},
                        'coef': {c: {'est': float(b_), 'se': float(s_), 'p': float(p_)}
                                 for c, b_, s_, p_ in zip(cols, beta, se, pv)}}


# ---------------- 마크다운 ----------------
L = []
L.append('# 세 경쟁 범위 팔 + 대조군: 뉴런별 스파이크 분포 (R19-C10, 26-10-02)\n')
L.append('테스트 10,000장, 스파이킹 18층 전 뉴런 (n_in·predictions_n 제외). 생성: `analysis/npz_three_arms_261002.py`.\n')
L.append('train/val 열과 val spikes 는 학습 원장(inventory.json, 베스트 체크포인트 에폭) 값, '
         '`test 스파이크/샘플`·`test acc` 이하 열은 npz(테스트셋 1회 추론) 값이다.\n')
EXCL = [t for t, r in inv.items() if r.get('exclude') and any(
    t.startswith(f'r19c10-{key}-s') for _, arms in BANDS for _, _, key, _ in arms)]
out['excluded'] = {t: inv[t].get('exclude_reason') for t in EXCL}
if EXCL:
    L.append('**배제 런 (npz 미추출, 평균에서 뺌):** ' + ', '.join(
        f"{t} ({inv[t].get('server')}, {inv[t].get('exclude_reason')}, val acc {inv[t].get('val_acc')})" for t in EXCL) + '\n')
if out['missing']:
    L.append('**npz 없음 (표에서 빠짐):** ' + ', '.join(out['missing']) + '\n')

F = [('train_loss', 'train loss', '{:.4f}'), ('train_acc', 'train acc', '{:.2f}'),
     ('val_loss', 'val loss', '{:.4f}'), ('val_acc', 'val acc', '{:.2f}'), ('spikes', 'spikes (원장)', '{:,.0f}'),
     ('spk_per_sample', 'test 스파이크/샘플', '{:,.0f}'), ('test_acc', 'test acc', '{:.2f}'),
     ('silent_pct', '침묵 %', '{:.2f}'), ('rare_pct', '드묾 %', '{:.2f}'),
     ('dead_ch', '죽은 채널 (/4736)', '{:.0f}'), ('alive_ch_silent_pct', '산 채널 안 침묵 %', '{:.2f}'),
     ('spk_per_alive_ch', '산 채널당 스파이크', '{:.1f}'), ('alive_ch_gini', '산 채널 안 지니', '{:.3f}'),
     ('spk_per_active', '활성 뉴런당 스파이크', '{:.3f}'), ('half_share_pct', '50% 집중도 %', '{:.2f}')]


def fmt(fm, v):
    return '–' if v is None else fm.format(v)


for band, arms in out['bands'].items():
    L.append(f'\n## {band}\n')
    L.append('| 방법 | λ | 시드 | 서버 | ' + ' | '.join(h for _, h, _ in F) + ' |')
    L.append('|' + '---|' * (4 + len(F)))
    for name, d in arms.items():
        rs = d['runs']
        for r in rs:
            L.append(f"| {name} | {d['knob']} | s{r['seed']} | {r['server']} | "
                     + ' | '.join(fmt(fm, r[k]) for k, _, fm in F) + ' |')
        if len(rs) > 1:
            cells = []
            for k, _, fm in F:
                m, s, n = ms([r[k] for r in rs])
                cells.append('–' if m is None else (f'**{fm.format(m)}**' + (f' ±{fm.format(s)}' if s is not None else '')))
            L.append(f"| **{name} 평균** | | **n={len(rs)}** | | " + ' | '.join(cells) + ' |')
    # 단계별
    L.append(f'\n### {band} — 단계별 침묵 % / 죽은 채널 (평균 ± SD)\n')
    L.append('| 방법 | n | ' + ' | '.join(f'{s} 침묵 %' for s in STAGES) + ' | '
             + ' | '.join(f'{s} 죽은 ch' for s in STAGES) + ' |')
    L.append('|' + '---|' * (2 + 2 * len(STAGES)))
    for name, d in arms.items():
        rs = d['runs']
        if not rs:
            continue
        c1 = []
        for s in STAGES:
            m, sd, n = ms([r['stage_silent_pct'][s] for r in rs])
            c1.append(f'{m:.2f}' + (f' ±{sd:.2f}' if sd is not None else ''))
        c2 = []
        for s in STAGES:
            m, sd, n = ms([r['stage_dead_ch'][s] for r in rs])
            c2.append(f'{m:.1f}' + (f' ±{sd:.1f}' if sd is not None else '') + f" /{rs[0]['stage_channels'][s]}")
        L.append(f'| {name} | {len(rs)} | ' + ' | '.join(c1) + ' | ' + ' | '.join(c2) + ' |')

L.append('\n## 대역 안 팔 간 차이 (a − b, 평균차 ± SE, Welch t 는 양쪽 n≥3 일 때만)\n')
PK = [('test_acc', 'test acc'), ('spk_per_sample', '스파이크/샘플'), ('silent_pct', '침묵 %'),
      ('dead_ch', '죽은 채널'), ('alive_ch_silent_pct', '산 채널 안 침묵 %'),
      ('spk_per_alive_ch', '산 채널당 스파이크'), ('rare_pct', '드묾 %'),
      ('spk_per_active', '활성 뉴런당'), ('half_share_pct', '50% 집중도')]
L.append('| 대역 | a | b | n_a | n_b | ' + ' | '.join(h for _, h in PK) + ' |')
L.append('|' + '---|' * (5 + len(PK)))
for rec in out['pairs']:
    if rec['n_a'] == 0 or rec['n_b'] == 0:
        continue
    cells = []
    for k, _ in PK:
        v = rec[k]
        if v['diff'] is None:
            cells.append('–'); continue
        fmd = '{:+,.0f}' if k == 'spk_per_sample' else '{:+.2f}'
        c = fmd.format(v['diff'])
        if v['se'] is not None:
            c += (' ±{:,.0f}' if k == 'spk_per_sample' else ' ±{:.2f}').format(v['se'])
        if v['p'] is not None:
            c += f" (p={v['p']:.3f})"
        cells.append(c)
    L.append(f"| {rec['band']} | {rec['a']} | {rec['b']} | {rec['n_a']} | {rec['n_b']} | " + ' | '.join(cells) + ' |')

L.append('\n## ANCOVA: 같은 테스트 스파이크에서의 팔 오프셋 (세 대역 합침, baseline 제외)\n')
L.append('y = a + b·ln(test 스파이크/샘플) + c_팔. 기준 팔 = ours_inter_ch. c 가 "같은 스파이크에서 ours_inter_ch 대비 차이".\n')
L.append('| 지표 | n | b (ln 스파이크) | ' + ' | '.join(f'c {a}' for a in arms_all[1:]) + ' |')
L.append('|' + '---|' * (3 + len(arms_all) - 1))
for k, a in out['ancova'].items():
    co = a['coef']
    cells = [f"{co['b_lnspk']['est']:+.2f} ±{co['b_lnspk']['se']:.2f} (p={co['b_lnspk']['p']:.3f})"]
    for arm in arms_all[1:]:
        if arm not in co:
            cells.append('–'); continue
        dg = 4 if k in ('alive_ch_gini', 'spk_per_active') else 2
        cells.append(f"{co[arm]['est']:+.{dg}f} ±{co[arm]['se']:.{dg}f} (p={co[arm]['p']:.3f}, n={a['n_per_arm'][arm]})")
    L.append(f"| {k} | {a['n']} (ours_inter_ch n={a['n_per_arm'][INTER]}) | " + ' | '.join(cells) + ' |')

json.dump(out, open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)
open(OUT_MD, 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
