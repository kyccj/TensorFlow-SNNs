"""제안법 vs plain L2 — 같은 스파이크 예산에서 뉴런을 얼마나 쓰는가.

reg_neuron_detail.csv 의 dead_neuron_ratio 와 reg_detail.csv 의 spike_count 로
'활성 뉴런 수'와 '뉴런당 발화'를 만든다. 두 CSV 모두 최종 에폭 값을 쓴다.
주의 - dead_neuron_ratio 는 배치 100장 기준이다. 방법 간 비율 비교에는 유효하지만
절대값을 '영구히 죽은 뉴런'으로 읽으면 안 된다.
"""
import csv, os, re, math, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

STORE = '/media/hdd1/kyccj/EIP/paper'
OUT = os.path.dirname(os.path.abspath(__file__))

# ResNet19: 뉴런 층별 개수 (H*W*C, 이미지 1장 기준)
NEUR = {'conv1_conv_n': 32*32*128}
for b in (1, 2, 3):
    NEUR[f'conv2_block{b}_conv1_n'] = 32*32*128; NEUR[f'conv2_block{b}_out_n'] = 32*32*128
    NEUR[f'conv3_block{b}_conv1_n'] = 16*16*256; NEUR[f'conv3_block{b}_out_n'] = 16*16*256
for b in (1, 2):
    NEUR[f'conv4_block{b}_conv1_n'] = 8*8*512;  NEUR[f'conv4_block{b}_out_n'] = 8*8*512
NEUR['fc1_n'] = 256
ORDER = ['conv1_conv_n',
         'conv2_block1_conv1_n','conv2_block1_out_n','conv2_block2_conv1_n','conv2_block2_out_n',
         'conv2_block3_conv1_n','conv2_block3_out_n',
         'conv3_block1_conv1_n','conv3_block1_out_n','conv3_block2_conv1_n','conv3_block2_out_n',
         'conv3_block3_conv1_n','conv3_block3_out_n',
         'conv4_block1_conv1_n','conv4_block1_out_n','conv4_block2_conv1_n','conv4_block2_out_n',
         'fc1_n']

def last_epoch(rows, key):
    e = max(int(r['epoch']) for r in rows)
    return {r['layer']: r for r in rows if int(r['epoch']) == e}

def load(run):
    d = os.path.join(STORE, run)
    a, b = d + '/reg_detail.csv', d + '/reg_neuron_detail.csv'
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    ra, rb = list(csv.DictReader(open(a))), list(csv.DictReader(open(b)))
    if not ra or not rb:
        return None
    sp = {k: float(v['spike_count']) for k, v in last_epoch(ra, 'spike_count').items()}
    nd = last_epoch(rb, None)
    S = sum(sp.get(l, 0) for l in NEUR)
    A = sum((1 - float(nd[l]['dead_neuron_ratio'])) * NEUR[l] for l in NEUR if l in nd)
    if A <= 0:
        return None
    return dict(S=S, A=A, sp=sp,
                gini={l: float(nd[l]['gini']) for l in nd},
                top10={l: float(nd[l]['top10_share']) for l in nd},
                dead={l: float(nd[l]['dead_neuron_ratio']) for l in nd})

METH = {'base': ('규제 없음', '#888888', 'o'),
        'prop': ('제안법', '#c0392b', 'o'),
        'l2':   ('plain L2', '#2471a3', 's'),
        'sm':   ('1−softmax', '#8e44ad', '^')}

runs = {}
for d in sorted(glob.glob(STORE + '/r19c10-*')):
    n = os.path.basename(d)
    m = re.match(r'^r19c10-([a-z_0-9]+?)-', n)
    if not m or m.group(1) not in METH:
        continue
    r = load(n)
    if r:
        runs.setdefault(m.group(1), []).append(r)
for n in ('basemore-base_r19_c10_run3',):
    r = load(n)
    if r:
        runs.setdefault('base', []).append(r)

print({k: len(v) for k, v in runs.items()})

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24)

# ── (a) 활성 뉴런 vs 스파이크 ──────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for k in ('base', 'l2', 'sm', 'prop'):
    if k not in runs: continue
    lab, c, mk = METH[k]
    S = np.array([r['S'] for r in runs[k]]); A = np.array([r['A'] for r in runs[k]])
    ax.scatter(S/1e3, A/1e3, c=c, marker=mk, s=46, label=f'{lab} (n={len(S)})',
               edgecolor='white', linewidth=.6, zorder=3)
# 공통기울기 회귀 (prop vs l2)
X, Y, G = [], [], []
for k in ('prop', 'l2'):
    for r in runs[k]:
        X.append(math.log(r['S'])); Y.append(math.log(r['A'])); G.append(k)
X, Y, G = np.array(X), np.array(Y), np.array(G)
D = np.column_stack([np.ones_like(X), X, (G == 'prop').astype(float)])
b, *_ = np.linalg.lstsq(D, Y, rcond=None)
res = Y - D@b; dof = len(Y)-3
cov = (res@res/dof)*np.linalg.inv(D.T@D); se = math.sqrt(cov[2, 2])
xs = np.linspace(X.min(), X.max(), 50)
ax.plot(np.exp(xs)/1e3, np.exp(b[0]+b[1]*xs+b[2])/1e3, color=METH['prop'][1], lw=1.4, zorder=2)
ax.plot(np.exp(xs)/1e3, np.exp(b[0]+b[1]*xs)/1e3,       color=METH['l2'][1],  lw=1.4, zorder=2)
ratio = math.exp(b[2])
ax.set_xlabel('스파이크 수 (천)'); ax.set_ylabel('활성 뉴런 수 (천)')
ax.set_title(f'(a) 같은 스파이크 예산에서 쓰는 뉴런\n'
             f'제안법 = plain L2 × {ratio:.3f} ({100*(ratio-1):+.1f}%),  t={b[2]/se:.1f}', fontsize=11)
ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=.25)

# ── (b) 뉴런당 발화 ──────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
ks = [k for k in ('base', 'sm', 'l2', 'prop') if k in runs]
vals = [np.array([r['S']/r['A'] for r in runs[k]]) for k in ks]
pos = np.arange(len(ks))
ax.bar(pos, [v.mean() for v in vals],
       yerr=[v.std(ddof=1) if len(v) > 1 else 0 for v in vals],
       color=[METH[k][1] for k in ks], width=.6, capsize=4, alpha=.9)
for i, v in enumerate(vals):
    ax.scatter(np.full(len(v), i) + np.random.uniform(-.13, .13, len(v)), v,
               c='white', edgecolor='black', s=18, zorder=3, linewidth=.6)
    top = max(v.max(), v.mean() + (v.std(ddof=1) if len(v) > 1 else 0))
    ax.text(i, top + .022, f'{v.mean():.3f}', ha='center', fontsize=9.5, fontweight='bold')
t, p = stats.ttest_ind(np.array([r['S']/r['A'] for r in runs['prop']]),
                       np.array([r['S']/r['A'] for r in runs['l2']]), equal_var=False)
ax.set_xticks(pos); ax.set_xticklabels([f'{METH[k][0]}\n(n={len(runs[k])})' for k in ks], fontsize=9.5)
ax.set_ylabel('스파이크 / 활성 뉴런'); ax.set_ylim(1.0, 1.78)
ax.set_title(f'(b) 살아남은 뉴런이 얼마나 세게 쏘나\n제안법 − plain L2: t={t:.1f},  p={p:.1e}', fontsize=11)
ax.grid(alpha=.25, axis='y')

# ── 착지점이 가장 가까운 prop·l2 한 쌍 고르기 ──────────────────
best = min(((i, j) for i in range(len(runs['prop'])) for j in range(len(runs['l2']))),
           key=lambda ij: abs(runs['prop'][ij[0]]['S'] - runs['l2'][ij[1]]['S']))
P, L = runs['prop'][best[0]], runs['l2'][best[1]]
B = max(runs['base'], key=lambda r: r['S'])
pair = f"제안법 {P['S']/1e3:.0f}K  vs  plain L2 {L['S']/1e3:.0f}K (스파이크 차 {100*abs(P['S']/L['S']-1):.1f}%)"

# ── (c) 층별 지니 ────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
lay = [l for l in ORDER if l in P['gini'] and l in L['gini'] and l in B['gini']]
x = np.arange(len(lay))
for r, k in ((B, 'base'), (L, 'l2'), (P, 'prop')):
    ax.plot(x, [r['gini'][l] for l in lay], marker=METH[k][2], color=METH[k][1],
            label=METH[k][0], lw=1.5, ms=4.5)
ax.set_xticks(x); ax.set_xticklabels([l.replace('_conv1_n','').replace('_out_n','·out')
                                      .replace('_conv_n','').replace('_n','') for l in lay],
                                     rotation=60, ha='right', fontsize=7.5)
ax.set_ylabel('지니 계수 (뉴런 간 불균등)')
ax.set_title(f'(c) 층별 발화 불균등\n{pair}', fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=.25)

# ── (d) 층별 상위 10% 점유 ───────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
for r, k in ((B, 'base'), (L, 'l2'), (P, 'prop')):
    ax.plot(x, [r['top10'][l] for l in lay], marker=METH[k][2], color=METH[k][1],
            label=METH[k][0], lw=1.5, ms=4.5)
ax.set_xticks(x); ax.set_xticklabels([l.replace('_conv1_n','').replace('_out_n','·out')
                                      .replace('_conv_n','').replace('_n','') for l in lay],
                                     rotation=60, ha='right', fontsize=7.5)
ax.set_ylabel('상위 10% 뉴런의 스파이크 점유')
ax.set_title('(d) 소수 뉴런에 얼마나 몰리나', fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=.25)

fig.suptitle('R19-CIFAR10 — 뉴런 사용 방식의 차이 (같은 스파이크 예산)', fontsize=13.5, y=.975)
p1 = os.path.join(OUT, 'neuron_usage_26-09-14.png')
fig.savefig(p1, dpi=170, bbox_inches='tight')
print('저장:', p1)
print(f"회귀: 비율 {ratio:.4f}  SE {se:.4f}  t {b[2]/se:.1f}  n={len(Y)}")
print(f"쌍: prop S={P['S']:.0f} A={P['A']:.0f} | l2 S={L['S']:.0f} A={L['A']:.0f}")
