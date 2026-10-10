"""세 경쟁 범위 팔 + 대조군 파레토 곡선, R19-C10 / R19-C100 (26-10-05).

실제 정확도(y) x 실제 스파이크 수(x, test, 샘플당). 차이(delta)로 그리지 않는다.
선은 회귀선이 아니라 같은 팔의 knob(rho/lambda) 스윕 평균점을 스파이크 순으로 이은 것.
수치 출처: 원장 inventory.json (analysis/ledger, refresh.sh 산출물). 310 에폭 완주 런만,
연기시험(시드>=900) 제외. 배제 규칙(S30/S1 >= 0.19) 위반 점은 x 로만 표시하고 선에서 뺀다.
오차막대 = 그 점의 시드 SD / sqrt(n), n>=2 만. 속 빈 마커 = n=1.

쓰는 법:  python EIP_fig/pareto_three_arms_fig.py [inventory.json]
"""
import os, sys, json, statistics as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INV = sys.argv[1] if len(sys.argv) > 1 else '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad/ledger/inventory.json'
OUT = os.path.dirname(os.path.abspath(__file__)); VEC = os.path.join(OUT, 'vector'); STAMP = '26-10-05'
plt.rcParams['font.family'] = 'Noto Sans CJK JP'; plt.rcParams['axes.unicode_minus'] = False; plt.rcParams['font.size'] = 9

# (원장 key, 범례 이름, 색, 마커, 선)
ARMS = [
    ('l2',            'plain L2',                    '#7F7F7F', 'd', (0, (1, 1.4))),
    ('bpsr',          'BPSR (Yan 2022)',             '#CC79A7', 'v', (0, (3, 1.5))),
    ('prop',          'ours_intra_ch',               '#0072B2', 'o', '-'),
    ('ours_w1',       'ours_intra_ch (w¹)',          '#56B4E9', 'o', (0, (5, 2))),
    ('ours_layer_w1', 'ours_layer (w¹)',             '#D55E00', 's', '--'),
    ('ours_ch',       'ours_inter_ch',               '#009E73', '^', '-'),
]
XLIM = {'C10': (1.0e5, 5.4e5), 'C100': (1.6e5, 5.0e5)}
YLIM = {'C10': (95.7, 97.0), 'C100': (79.4, 81.2)}
XT = {'C10': [1e5, 1.5e5, 2e5, 3e5, 4e5, 5e5], 'C100': [2e5, 3e5, 4e5, 5e5]}

runs = [r for r in json.load(open(INV))['runs']
        if r['status'] in ('done', 'excluded') and r.get('epochs') == 310 and int(r['seed']) < 900
        and r.get('val_acc') is not None and r.get('spikes')]

def groups(ds, key):
    g = {}
    for r in runs:
        if r['ds'] == ds and r['key'] == key:
            g.setdefault(r['knob'], []).append(r)
    pts = []
    for knob, rs in g.items():
        ok = [r for r in rs if r['status'] == 'done']
        ex = len(ok) == 0
        use = ok if ok else rs
        acc = [r['val_acc'] for r in use]; sp = [r['spikes'] for r in use]
        sd = st.stdev(acc) if len(acc) > 1 else None
        pts.append((knob, st.mean(acc), st.mean(sp), len(use), sd, ex))
    return sorted(pts, key=lambda p: p[2])

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
summary = {}
for ax, ds in zip(axes, ('C10', 'C100')):
    b = [r for r in runs if r['ds'] == ds and r['key'] == 'base' and r['status'] == 'done']
    bacc = st.mean(r['val_acc'] for r in b); bsp = st.mean(r['spikes'] for r in b); bsd = st.stdev(r['val_acc'] for r in b)
    ax.axhspan(bacc - bsd, bacc + bsd, color='0.5', alpha=.10, lw=0, zorder=0)
    ax.axhline(bacc, color='0.45', ls=':', lw=1.0, zorder=1)
    ax.errorbar([bsp], [bacc], yerr=[bsd / np.sqrt(len(b))], fmt='*', ms=14, color='0.15', ecolor='0.15',
                elinewidth=1.1, capsize=2.5, zorder=6, label=f'baseline (n={len(b)})')
    summary[ds] = {'baseline': (round(bacc, 3), int(bsp), len(b), round(bsd, 3))}
    for key, name, color, mk, ls in ARMS:
        pts = [p for p in groups(ds, key) if XLIM[ds][0] <= p[2] <= XLIM[ds][1]]
        if not pts:
            continue
        summary[ds][name] = [(p[0], round(p[1], 3), int(p[2]), p[3], p[5]) for p in pts]
        line = [p for p in pts if not p[5]]
        ax.plot([p[2] for p in line], [p[1] for p in line], ls=ls, lw=1.4 if key == 'ours_ch' else 1.2,
                color=color, alpha=.95, zorder=3 if key == 'ours_ch' else 2, label=name)
        for _k, acc, sp, n, sd, ex in pts:
            if ex:
                ax.plot([sp], [acc], marker='x', ms=7, mew=1.4, lw=0, color=color, alpha=.55, zorder=3)
                continue
            ax.plot([sp], [acc], marker=mk, ms=7 if key != 'ours_ch' else 8, lw=0, color=color,
                    mfc=color if n >= 2 else 'white', mew=1.5, zorder=5 if key == 'ours_ch' else 4)
            if n >= 2 and sd is not None:
                ax.errorbar([sp], [acc], yerr=[sd / np.sqrt(n)], fmt='none', ecolor=color, elinewidth=1.0,
                            capsize=2.0, zorder=4)
    ax.set_xscale('log'); ax.set_xlim(*XLIM[ds]); ax.set_ylim(*YLIM[ds])
    ax.set_xticks(XT[ds]); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{int(v):,}'))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ''))
    ax.tick_params(axis='x', labelsize=8); ax.grid(alpha=.22, which='major', lw=.5)
    ax.set_xlabel('스파이크 수 (test, 샘플당)'); ax.set_ylabel('val accuracy (%)')
    ax.set_title(f'ResNet19 · {"CIFAR-10" if ds == "C10" else "CIFAR-100"} · T=4', fontsize=10)
axes[0].plot([], [], marker='o', ms=7, lw=0, color='0.35', mfc='white', mew=1.5, label='n=1')
axes[0].plot([], [], marker='x', ms=7, mew=1.4, lw=0, color='0.35', alpha=.7, label='배제 규칙 위반 (선에서 제외)')
for ax in axes:
    ax.legend(loc='lower right', fontsize=7.0, framealpha=.95)
fig.text(0.012, 0.012,
         '선은 회귀선이 아니라 각 팔의 ρ(λ) 스윕 평균점을 스파이크 순으로 이은 것. 오차막대 = 그 점의 시드 SD/√n (n≥2), 속 빈 마커 = n=1. '
         '회색 띠 = baseline ±1 SD.\n'
         'x = 배제 규칙(310에폭 AND S30/S1 ≥ 0.19) 위반. 늦은 시작·Taylor·어블레이션 팔은 제외. 원장 '
         + json.load(open(INV)).get('generated', '') + ' 집계.', fontsize=6.4, va='bottom', linespacing=1.45)
fig.tight_layout(rect=(0, 0.08, 1, 1))
os.makedirs(VEC, exist_ok=True)
for ext in ('png', 'pdf', 'svg'):
    p = os.path.join(OUT if ext == 'png' else VEC, f'pareto_three_arms_ko_{STAMP}.{ext}')
    fig.savefig(p, dpi=220 if ext == 'png' else None, bbox_inches='tight'); print('wrote', p)
json.dump(summary, open(os.path.join(OUT, f'_pareto_three_arms_{STAMP}.json'), 'w'), ensure_ascii=False, indent=1)
