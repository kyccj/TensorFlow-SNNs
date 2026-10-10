"""전체 팔 파레토 곡선, R19-C10 / R19-C100 (26-09-30).

실제 정확도(y) x 실제 스파이크 수(x). 차이(delta)로 그리지 않는다 (CLAUDE.md).
선은 회귀선이 아니라 각 팔의 rho/lambda 스윕 측정점(완주·유효)을 이은 것이다.
배제 규칙(310에폭 AND S30/S1 >= 0.19) 위반 점은 선에서 빼고 x 표시로만 남긴다 --
낮은 스파이크 쪽 모양을 보려는 것이지, 주장에 쓰는 점이 아니다.

수치 출처: 원장 inventory.json (analysis/ledger, 09-30 11:31 갱신) 을 그대로 집계한 것.
    python3 - <<EOF ... (ARMS 블록을 만든 명령은 스크립트 아래 DATA_NOTE 참고)

쓰는 법:  python EIP_fig/pareto_all_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = os.path.dirname(os.path.abspath(__file__))
VEC = os.path.join(OUT, 'vector')
STAMP = '26-09-30'
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

# 시드 간 val acc SD (%p). C10 은 기존 실측 묶음값, C100 은 baseline n=5 의 SD.
SEED_SD = {'C10': 0.117, 'C100': 0.20}

# (knob, val_acc, spikes, n, excluded)
BASELINE = {'C10': (96.73, 486444, 5), 'C100': (80.70, 449894, 5)}
ARMS = {
 'C10': {
  'ours_intra_ch (w²)': [('2e-3', 96.64, 290020, 2, 0), ('3e-3', 96.67, 234091, 7, 0),
                         ('4e-3', 96.59, 192092, 5, 0), ('5e-3', 96.42, 162846, 1, 0),
                         ('6e-3', 96.33, 136178, 1, 1), ('8e-3', 95.98, 105470, 1, 1)],
  'ours_intra_ch (w¹)': [('2e-3', 96.65, 239949, 3, 0), ('2.5e-3', 96.52, 204888, 3, 0),
                         ('3e-3', 96.41, 175242, 3, 0), ('4e-3', 96.14, 139476, 1, 1),
                         ('5e-3', 95.96, 107215, 1, 1)],
  'ours_layer (w¹)':    [('7e-4', 96.66, 371139, 3, 0), ('1e-3', 96.61, 334197, 3, 0),
                         ('1.5e-3', 96.66, 288425, 3, 0), ('2e-3', 96.69, 245418, 3, 0),
                         ('2.5e-3', 96.55, 220550, 5, 0), ('3e-3', 96.47, 195652, 3, 0),
                         ('4e-3', 96.32, 154571, 3, 1)],
  'ours_inter_ch':      [('2e-3', 96.73, 385705, 3, 0), ('3e-3', 96.74, 342044, 3, 0),
                         ('4e-3', 96.66, 302039, 3, 0), ('8e-3', 96.38, 200485, 1, 0),
                         ('1.2e-2', 96.30, 147495, 1, 0), ('1.6e-2', 95.81, 113360, 1, 1),
                         ('2e-2', 95.22, 88157, 1, 1)],
  'plain L2':           [('5e-8', 96.70, 387485, 2, 0), ('1e-7', 96.54, 323667, 2, 0),
                         ('1.5e-7', 96.68, 296829, 2, 0), ('2.5e-7', 96.59, 246178, 5, 0),
                         ('4.5e-7', 96.49, 194964, 3, 0), ('5e-7', 96.49, 185537, 5, 0),
                         ('7.5e-7', 96.31, 156533, 3, 1), ('1.5e-6', 95.99, 119539, 1, 1)],
  'BPSR':               [('3e-11', 96.73, 412032, 3, 0), ('1e-10', 96.66, 319481, 3, 0),
                         ('3e-10', 96.59, 237060, 3, 0), ('6e-10', 96.51, 190847, 3, 0),
                         ('1e-9', 96.39, 158352, 3, 1)],
 },
 'C100': {
  'ours_intra_ch (w²)': [('2e-3', 80.33, 276600, 1, 0), ('3e-3', 80.36, 220248, 2, 0),
                         ('4e-3', 79.91, 179743, 3, 1)],
  'ours_layer (w¹)':    [('7e-4', 80.49, 351081, 3, 0), ('1e-3', 80.67, 314917, 3, 0),
                         ('1.5e-3', 80.49, 267002, 3, 0), ('2e-3', 80.18, 243259, 3, 0),
                         ('2.5e-3', 79.79, 209723, 3, 1), ('3e-3', 79.82, 188226, 3, 1)],
  'plain L2':           [('2.5e-7', 80.30, 315486, 1, 0), ('3.5e-7', 80.28, 293278, 1, 0),
                         ('4e-7', 80.40, 266268, 1, 0), ('5e-7', 80.50, 255821, 3, 0),
                         ('6e-7', 80.25, 234110, 2, 0), ('7e-7', 80.25, 227633, 2, 0)],
  'BPSR':               [('3e-10', 80.35, 299000, 3, 0), ('6e-10', 80.27, 253869, 3, 0)],
 },
}

# Okabe-Ito + 마커 모양 병행
STYLE = {
    'ours_intra_ch (w²)': ('#0072B2', 'o', '-'),
    'ours_intra_ch (w¹)': ('#56B4E9', 'o', '-'),
    'ours_layer (w¹)':    ('#D55E00', 's', '--'),
    'ours_inter_ch':      ('#009E73', '^', '-.'),
    'plain L2':           ('#7F7F7F', 'd', (0, (1, 1.4))),
    'BPSR':               ('#CC79A7', 'v', (0, (3, 1.5))),
}
XT = {'C10': [1e5, 1.5e5, 2e5, 3e5, 4e5, 5e5], 'C100': [1.5e5, 2e5, 3e5, 4e5, 5e5]}
YL = {'C10': (95.7, 97.0), 'C100': (79.5, 81.1)}


def draw():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for ax, ds in zip(axes, ('C10', 'C100')):
        sd = SEED_SD[ds]
        bacc, bsp, bn = BASELINE[ds]
        ax.axhline(bacc, color='0.45', ls=':', lw=1.0, zorder=1)
        ax.errorbar([bsp], [bacc], yerr=[sd / np.sqrt(bn)], fmt='*', ms=14, color='0.15',
                    ecolor='0.15', elinewidth=1.1, capsize=2.5, zorder=6, label=f'baseline (n={bn})')
        for arm, pts in ARMS[ds].items():
            color, mk, ls = STYLE[arm]
            ok = [p for p in pts if not p[4]]
            ax.plot([p[2] for p in ok], [p[1] for p in ok], ls=ls, lw=1.3, color=color,
                    alpha=.9, zorder=2, label=arm)
            for _k, acc, s, n, ex in pts:
                if ex:
                    ax.plot([s], [acc], marker='x', ms=7, mew=1.4, lw=0, color=color, alpha=.55, zorder=3)
                    continue
                ax.plot([s], [acc], marker=mk, ms=7, lw=0, color=color,
                        mfc=color if n >= 2 else 'white', mew=1.5, zorder=4)
                if n >= 2:
                    ax.errorbar([s], [acc], yerr=[sd / np.sqrt(n)], fmt='none', ecolor=color,
                                elinewidth=1.1, capsize=2.2, zorder=4)
        ax.set_xscale('log')
        ax.set_xticks(XT[ds])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{int(v):,}'))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ''))
        ax.set_ylim(*YL[ds])
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(alpha=.22, which='major', lw=.5)
        ax.set_xlabel('스파이크 수 (test, 샘플당)')
        ax.set_ylabel('val accuracy (%)')
        ax.set_title(f'ResNet19 · {"CIFAR-10" if ds == "C10" else "CIFAR-100"}', fontsize=10)
    axes[0].plot([], [], marker='o', ms=7, lw=0, color='0.35', mfc='white', mew=1.5, label='n=1')
    axes[0].plot([], [], marker='x', ms=7, mew=1.4, lw=0, color='0.35', alpha=.7,
                 label='배제 규칙 위반 (선에서 제외)')
    axes[0].legend(loc='lower right', fontsize=7.2, framealpha=.95)
    axes[1].legend(loc='lower right', fontsize=7.2, framealpha=.95)
    fig.text(0.012, 0.012,
             '선은 회귀선이 아니라 각 팔의 ρ(λ) 스윕 측정점을 이은 것이다. 오차막대 = SEM '
             f'(시드 SD C10 {SEED_SD["C10"]}%p, C100 {SEED_SD["C100"]}%p), n≥2 만. 속 빈 마커 = n=1.\n'
             'x = 배제 규칙(310에폭 완주 AND S30/S1 ≥ 0.19) 위반 점 -- 모양만 보이고 주장에는 쓰지 않는다. '
             '원장 26-09-30 11:31 집계.',
             fontsize=6.4, va='bottom', linespacing=1.45)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    os.makedirs(VEC, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        p = os.path.join(OUT if ext == 'png' else VEC, f'pareto_all_ko_{STAMP}.{ext}')
        fig.savefig(p, dpi=220 if ext == 'png' else None, bbox_inches='tight')
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    draw()
