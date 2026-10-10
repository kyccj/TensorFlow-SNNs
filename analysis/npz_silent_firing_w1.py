"""ours_layer_w1 vs ours_intra_ch: 침묵 뉴런과 발화 뉴런 (26-09-29).

R19-C10, 테스트 10,000장, 스파이킹 18층(n_in·predictions_n 제외) 전체 뉴런 기준.
스파이크 예산이 맞는 대역끼리만 비교한다 (침묵률은 스파이크를 적게 쓸수록 저절로 오른다).

뉴런 분류
  침묵    테스트 전체에서 한 번도 발화하지 않음
  드묾    발화는 했지만 발화한 표본이 1% 미만 (100장 미만)
  발화    나머지
발화 뉴런 쪽 지표
  활성 뉴런당 스파이크   전체 스파이크/샘플 ÷ (침묵 아닌 뉴런 수)
  50% 집중도            스파이크의 절반을 내는 뉴런이 발화 뉴런 중 몇 %인가 (작을수록 소수에 몰림)
죽은 채널: 공간 전체가 침묵인 채널 (4,736 채널 중)

쓰는 법:  python analysis/npz_silent_firing_w1.py  [결과 JSON 경로]
"""
import json, os, sys
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ = os.path.join(ROOT, '_npz')
INV = '/tmp/eip_ledger_live/inventory.json'
SKIP = {'n_in', 'predictions_n'}

BANDS = [
    ('약 195K', [('ours_layer_w1 3e-3', 'ours_layer_w1-3e-3', [1, 2, 3]),
                ('ours_intra_ch 4e-3', 'prop-4e-3', [1, 2, 3, 4, 5]),
                ('ours_layer 4e-3', 'ours_layer-4e-3', [1, 2, 3, 4, 5]),
                ('plain L2 5e-7', 'l2-5e-7', [1, 2, 3, 4, 5]),
                ('BPSR 6e-10', 'bpsr-6e-10', [1, 2, 3])]),
    ('약 220K', [('ours_layer_w1 2.5e-3', 'ours_layer_w1-2p5e-3', [1, 2, 3])]),
    ('약 240K', [('ours_layer_w1 2e-3', 'ours_layer_w1-2e-3', [1, 2, 3]),
                ('ours_intra_ch 3e-3', 'prop-3e-3', [2, 3, 4]),
                ('BPSR 3e-10', 'bpsr-3e-10', [1, 2, 3])]),
    ('기준', [('baseline', 'base--', [11])]),
]

inv = {x['tag']: x for x in json.load(open(INV))['runs']}


def metrics(path):
    z = np.load(path)
    ns = float(z['n_samples'])
    tot = spk = sil = rare = dead = nch = 0
    act_counts = []
    stage = {}
    for k in z.files:
        if k.endswith('__fired') or k + '__fired' not in z.files or k in SKIP:
            continue
        c = z[k].astype(np.float64)
        f = z[k + '__fired'].astype(np.float64)
        n = c.size
        s0 = int((c == 0).sum())
        r = int(((c > 0) & (f < 0.01 * ns)).sum())
        tot += n; spk += c.sum(); sil += s0; rare += r
        act_counts.append(c[c > 0].ravel())
        if c.ndim == 3:
            ch = c.reshape(-1, c.shape[-1]).sum(0)
            dead += int((ch == 0).sum()); nch += c.shape[-1]
        elif c.ndim == 1:                          # fc1: 뉴런 하나가 채널 하나 (4,736 = conv 4,480 + fc1 256)
            dead += int((c == 0).sum()); nch += c.size
        st = k.split('_')[0] if k.startswith(('conv', 'fc')) else k
        st = 'conv1+conv2' if st in ('conv1', 'conv2') else st
        a = stage.setdefault(st, [0, 0]); a[0] += s0; a[1] += n
    act = np.sort(np.concatenate(act_counts))[::-1]
    half = int(np.searchsorted(np.cumsum(act), act.sum() / 2) + 1)
    nfire = tot - sil
    return {'neurons': tot, 'spk_per_sample': spk / ns, 'silent': sil, 'silent_pct': 100 * sil / tot,
            'rare': rare, 'rare_pct': 100 * rare / tot, 'firing': nfire - rare,
            'firing_pct': 100 * (nfire - rare) / tot,
            'spk_per_active': spk / ns / nfire, 'half_share_pct': 100 * half / nfire,
            'dead_ch': dead, 'channels': nch,
            'stage_silent_pct': {k: 100 * v[0] / v[1] for k, v in stage.items()}}


out = {}
for band, arms in BANDS:
    for name, key, seeds in arms:
        rows = []
        for s in seeds:
            tag = f'r19c10-{key}-s{s}'
            p = os.path.join(NPZ, tag + '.npz')
            if not os.path.exists(p):
                print('npz 없음', tag); continue
            m = metrics(p)
            iv = inv.get(tag, {})
            m.update(seed=s, server=iv.get('server', '?'), val_acc=iv.get('val_acc'))
            rows.append(m)
        out[name] = {'band': band, 'runs': rows}

F = [('val_acc', 'val acc', '{:.2f}'), ('spk_per_sample', '스파이크/샘플', '{:,.0f}'),
     ('silent_pct', '침묵 %', '{:.2f}'), ('rare_pct', '드묾 %', '{:.2f}'), ('firing_pct', '발화 %', '{:.2f}'),
     ('spk_per_active', '활성 뉴런당 스파이크', '{:.3f}'), ('half_share_pct', '50% 집중도 %', '{:.2f}'),
     ('dead_ch', '죽은 채널', '{:.0f}')]
for band, arms in BANDS:
    print(f'\n## {band}')
    print('| arm | 시드 | 서버 | ' + ' | '.join(h for _, h, _ in F) + ' |')
    print('|' + '---|' * (3 + len(F)))
    for name, _, _ in arms:
        rs = out[name]['runs']
        for r in rs:
            print(f"| {name} | s{r['seed']} | {r['server']} | " + ' | '.join(fm.format(r[k]) if r[k] is not None else '–' for k, _, fm in F) + ' |')
        if len(rs) > 1:
            cells = []
            for k, _, fm in F:
                v = [r[k] for r in rs if r[k] is not None]
                cells.append(fm.format(np.mean(v)) + f' ±{np.std(v, ddof=1):.2f}' if k not in ('spk_per_sample',) else fm.format(np.mean(v)))
            print(f"| **{name} 평균** | **n={len(rs)}** | | " + ' | '.join(cells) + ' |')

print('\n## Welch t: ours_layer_w1 − 상대 (같은 대역)')
for band, arms in BANDS:
    w1 = [a for a in arms if a[0].startswith('ours_layer_w1')]
    if not w1 or len(arms) < 2:
        continue
    A = out[w1[0][0]]['runs']
    for name, _, _ in arms:
        if name == w1[0][0]:
            continue
        B = out[name]['runs']
        line = []
        for k in ('silent_pct', 'rare_pct', 'spk_per_active', 'half_share_pct', 'dead_ch'):
            a = [r[k] for r in A]; b = [r[k] for r in B]
            t, p = stats.ttest_ind(a, b, equal_var=False)
            line.append(f"{k} {np.mean(a) - np.mean(b):+.3f} (p={p:.3f})")
        print(f"- {band} vs {name}: " + ', '.join(line))

print('\n## 단계별 침묵 % (평균)')
stg = ['conv1+conv2', 'conv3', 'conv4', 'fc1']
print('| arm | ' + ' | '.join(stg) + ' |')
print('|' + '---|' * (1 + len(stg)))
for band, arms in BANDS:
    for name, _, _ in arms:
        rs = out[name]['runs']
        print(f'| {name} | ' + ' | '.join(f"{np.mean([r['stage_silent_pct'][s] for r in rs]):.2f}" for s in stg) + ' |')

if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], 'w'), ensure_ascii=False, indent=1)
