"""Method 절 개념도 2/2 (그림 2) — 가중치 함수 sc_rate(sc_norm).

실험 수치 없음. neurons.py:1020-1022 의 수식만 그린다:
  sc_norm = scp / max(그룹)                 in [0,1], 1=그룹 1등, 0=침묵
  sc_rate = 1 - beta*sc_norm                 (기본 beta=1.0, '1-maxnorm')
  sc_rate = sc_norm                          (reg_spike_maxnorm_no_inv=True, '- 1- inversion')

beta=0 은 sc_rate 가 항상 1(균일) -- plain L2 와 같은 자리(무가중)로 표시한다.

2026-09-21 수정 (사용자 피드백):
  - 열린 원(○)/채운 원(●) 의 의미를 범례에 명시했다. 이 둘은 임의 구분이 아니라
    실제로 다른 것을 나타낸다: 열린 원 = 그 지점에서 여러 곡선이 같은 값을 공유하는
    "공유점", 채운 원 = 그 값에 도달하는 곡선이 그 하나뿐인 "고유점".
    예: sc_norm=0 에서 beta=1.0/0.7/0.4/0 네 선이 모두 sc_rate=1 로 만난다(공유,
    열린 원). 반대로 sc_norm=0 에서 inversion 선만 sc_rate=0 이라(고유, 채운 원).
  - --lang en|ko 로 영문/국문 두 벌을 뽑는다. 영문이 논문 정본.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__))
DATE = '26-09-21'

FONT = {'ko': 'Noto Sans CJK JP', 'en': 'Liberation Sans'}

x = np.linspace(0, 1, 200)

# Okabe-Ito 색맹 안전 팔레트
C_OURS = '#000000'       # black, beta=1.0
C_B07 = '#E69F00'        # orange
C_B04 = '#56B4E9'        # sky blue
C_UNI = '#999999'        # gray, beta=0 / plain L2
C_INV = '#D55E00'        # vermillion, - 1- inversion

STR = {
    'ko': {
        'title': '가중치 함수 sc_rate(sc_norm) — illustrative',
        'xlabel': 'sc_norm  (0=침묵, 1=그룹 1등)',
        'ylabel': 'sc_rate  (규제 가중치)',
        'l_ours': r'$\beta{=}1.0$ (ours, 1-maxnorm)',
        'l_b07': r'$\beta{=}0.7$ (ours_ch_b7)',
        'l_b04': r'$\beta{=}0.4$ (ours_ch_b4)',
        'l_uni': r'$\beta{=}0$ (uniform = plain L2 자리)',
        'l_inv': r'$-$ 1$-$inversion (sc_rate$=$sc_norm)',
        'ann_conv': 'β 선 전부 1로 수렴',
        'ann_inv0': 'inversion만 0',
        'leg_open': '○ 공유점 (여러 곡선이 겹침)',
        'leg_filled': '● 고유점 (그 곡선만 도달)',
    },
    'en': {
        'title': 'Weight function sc_rate(sc_norm) — illustrative',
        'xlabel': 'sc_norm  (0=silent, 1=group winner)',
        'ylabel': 'sc_rate  (regularizer weight)',
        'l_ours': r'$\beta{=}1.0$ (ours, 1-maxnorm)',
        'l_b07': r'$\beta{=}0.7$ (ours_ch_b7)',
        'l_b04': r'$\beta{=}0.4$ (ours_ch_b4)',
        'l_uni': r'$\beta{=}0$ (uniform = plain L2 slot)',
        'l_inv': r'$-$ 1$-$inversion (sc_rate$=$sc_norm)',
        'ann_conv': 'all $\\beta$ curves converge to 1',
        'ann_inv0': 'only inversion is 0',
        'leg_open': r'$\circ$ shared point (multiple curves overlap)',
        'leg_filled': r'$\bullet$ unique point (only that curve)',
    },
}


def draw(lang):
    s = STR[lang]
    plt.rcParams['font.family'] = FONT[lang]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 8.5

    fig, ax = plt.subplots(figsize=(3.3, 3.95))

    lines = [
        (1.0, 1 - 1.0 * x, C_OURS, '-', s['l_ours']),
        (0.7, 1 - 0.7 * x, C_B07, '--', s['l_b07']),
        (0.4, 1 - 0.4 * x, C_B04, '-.', s['l_b04']),
        (0.0, np.ones_like(x), C_UNI, ':', s['l_uni']),
    ]

    for beta, y, c, ls, label in lines:
        ax.plot(x, y, color=c, linestyle=ls, linewidth=1.8, label=label, zorder=3)

    # inversion arm: sc_rate = sc_norm
    ax.plot(x, x, color=C_INV, linestyle='-', linewidth=1.8,
            label=s['l_inv'], zorder=3)

    # 침묵 지점(x=0): beta 선 4개가 전부 sc_rate=1 에서 겹친다 -> 공유점(open).
    # inversion 만 sc_rate=0 -> 그 값에 도달하는 곡선이 하나뿐 -> 고유점(filled).
    ax.scatter([0], [1], s=42, facecolor='white', edgecolor=C_OURS, zorder=5, linewidth=1.3)
    ax.scatter([0], [0], s=42, facecolor=C_INV, edgecolor=C_INV, zorder=5)
    ax.annotate(s['ann_conv'], xy=(0, 1), xytext=(0.05, 1.145),
                fontsize=6.8, color='#333333', ha='left', va='center')
    ax.annotate(s['ann_inv0'], xy=(0, 0), xytext=(0.05, -0.13),
                fontsize=6.8, color=C_INV, ha='left', va='center')

    # 1등 지점(x=1): beta=1.0 선만 sc_rate=0 에 닿는다 -> 고유점(filled).
    # inversion 과 beta=0(uniform) 선이 둘 다 sc_rate=1 에서 만난다 -> 공유점(open).
    ax.scatter([1], [0], s=42, facecolor=C_OURS, edgecolor=C_OURS, zorder=5)
    ax.scatter([1], [1], s=42, facecolor='white', edgecolor=C_INV, zorder=5, linewidth=1.3)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.18, 1.2)
    ax.set_xlabel(s['xlabel'])
    ax.set_ylabel(s['ylabel'])
    ax.axhline(0, color='#cccccc', lw=0.6, zorder=1)
    ax.axvline(0, color='#cccccc', lw=0.6, zorder=1)
    ax.grid(alpha=0.25, zorder=0)

    # 곡선 범례 + 열린/채운 원 의미를 알려주는 마커 범례를 함께 둔다.
    handles, labels = ax.get_legend_handles_labels()
    marker_handles = [
        Line2D([0], [0], marker='o', linestyle='none', markersize=6,
               markerfacecolor='white', markeredgecolor='#333333', markeredgewidth=1.2,
               label=s['leg_open']),
        Line2D([0], [0], marker='o', linestyle='none', markersize=6,
               markerfacecolor='#333333', markeredgecolor='#333333',
               label=s['leg_filled']),
    ]
    ax.legend(handles=handles + marker_handles, fontsize=6.3, loc='upper center',
              bbox_to_anchor=(0.5, -0.30), ncol=1, framealpha=0.92, handlelength=2.4)
    ax.set_title(s['title'], fontsize=8.8)

    fig.tight_layout()

    pdf = os.path.join(OUT, f'weight_function_{lang}_{DATE}.pdf')
    svg = os.path.join(OUT, f'weight_function_{lang}_{DATE}.svg')
    png = os.path.join(OUT, f'weight_function_{lang}_{DATE}.png')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(svg, bbox_inches='tight')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote', pdf, svg, png)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', choices=['en', 'ko', 'both'], default='ko')
    args = ap.parse_args()
    langs = ['en', 'ko'] if args.lang == 'both' else [args.lang]
    for lg in langs:
        draw(lg)
