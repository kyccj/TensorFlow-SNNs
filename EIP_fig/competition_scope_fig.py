"""Method 절 개념도 1/2 (그림 1) — 경쟁 범위(reg_spike_maxnorm_group) 3종.

실험 수치 없음. neurons.py:986-997 의 동작만 그린다:
  within_channel (ours_intra_ch)  reduce_max(scp, axis=[1,2])   채널마다 따로 최댓값
  none           (ours_layer)     reduce_max(scp, reduce_axis)  층 전체 하나의 최댓값
  channel        (ours_inter_ch)  ch=reduce_sum(scp,[1,2]); ch_max=reduce_max(ch,axis=3);
                                   broadcast_to(ch/ch_max, shape)  채널 공간합끼리 경쟁 후 broadcast

이름 대응 (2026-09-21 사용자 확정 — 나열 순서 그대로 배정하면 의미가 뒤집힌다):
  within_channel -> ours_intra_ch  (채널 "안"에서 겨룸)
  none           -> ours_layer
  channel        -> ours_inter_ch (채널 "끼리" 겨룸)

3x4 spike map(T=4, 채널 3개)을 하나 고정해 두고, 세 방식이 그 위에서
sc_rate = 1 - beta*sc_norm (beta=1.0, '1-maxnorm') 을 어떻게 다르게 배정하는지 보인다.
숫자는 설명용 예시(illustrative)다. 어떤 실험 결과도 반영하지 않는다.

2026-09-21 수정 (사용자 피드백):
  - 남색 칸 글자가 안 보이던 문제: 고정 rate>0.55 임계 대신, 실제 컬러맵 RGB의
    상대 휘도를 계산해 글자색을 뒤집는다.
  - 채널 간 최대값 격차를 키워(A=4, B=2, C=1) ours_intra_ch 대비 ours_layer/
    ours_inter_ch 의 차이가 더 많은 칸에서, 더 크게 드러나도록 예시 맵을 다시 짰다.
  - ours_layer 패널에는 ours_intra_ch 대비 값이 달라진 칸에 회색 삼각형(△) 표시를
    추가했다. ours_inter_ch 패널은 채널 전체(48칸 중 48칸)가 달라지므로 칸별 표시
    대신 "채널 전체가 균일값" 이라는 문구로 대신한다 (더 읽기 쉬움, 아래 판단 참조).
  - --lang en|ko 로 영문/국문 두 벌을 뽑는다. 영문이 논문 정본.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

OUT = os.path.dirname(os.path.abspath(__file__))
DATE = '26-09-21'

FONT = {'ko': 'Noto Sans CJK JP', 'en': 'Liberation Sans'}

# ── 예시 spike map (T=4). 채널 A/B/C, 각 4x4. 설명용 숫자, 실측 아님. ─────────
# 채널 간 활동량 격차를 크게 벌림: A 최고 4, B 최고 2, C 최고 1.
CH = {
    'A': np.array([[1, 0, 2, 0],
                    [0, 4, 0, 1],
                    [2, 0, 1, 0],
                    [0, 1, 0, 0]], dtype=float),
    'B': np.array([[0, 1, 0, 0],
                    [1, 0, 2, 0],
                    [0, 0, 0, 1],
                    [0, 2, 0, 0]], dtype=float),
    'C': np.array([[0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]], dtype=float),
}
ORDER = ['A', 'B', 'C']
BETA = 1.0  # sc_rate = 1 - beta*sc_norm ('1-maxnorm', ours 의 기본값)

own_max = {k: v.max() for k, v in CH.items()}
layer_max = max(own_max.values())
ch_sum = {k: v.sum() for k, v in CH.items()}
ch_sum_max = max(ch_sum.values())


def rate_within_channel(name):
    v = CH[name]
    return 1.0 - BETA * (v / own_max[name])


def rate_none(name):
    v = CH[name]
    return 1.0 - BETA * (v / layer_max)


def rate_channel(name):
    v = CH[name]
    flat = 1.0 - BETA * (ch_sum[name] / ch_sum_max)
    return np.full_like(v, flat)


RATE_FN = {'within_channel': rate_within_channel, 'none': rate_none, 'channel': rate_channel}

# panel order fixed: within_channel, none, channel — arm names per the table above.
PANEL_KEYS = ['within_channel', 'none', 'channel']
ARM_NAME = {'within_channel': 'ours_intra_ch', 'none': 'ours_layer', 'channel': 'ours_inter_ch'}
DESC = {
    'ko': {'within_channel': 'per-channel (채널마다 경쟁)',
           'none': 'layer-wide (층 전체가 경쟁)',
           'channel': 'channel-sum + broadcast (채널 합끼리 경쟁)'},
    'en': {'within_channel': 'per-channel',
           'none': 'layer-wide',
           'channel': 'channel-sum + broadcast'},
}

BASE_KEY = 'within_channel'  # diff marker baseline


def luminance(rgba):
    r, g, b = rgba[0], rgba[1], rgba[2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def draw(lang):
    plt.rcParams['font.family'] = FONT[lang]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 8

    cmap = plt.get_cmap('cividis')
    norm = Normalize(vmin=0.0, vmax=1.0)

    fig = plt.figure(figsize=(7.2, 3.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.5,
                           left=0.09, right=0.90, top=0.80, bottom=0.20)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])

    cell = 0.72
    gap_ch = 0.55

    base_rate = {name: RATE_FN[BASE_KEY](name) for name in ORDER}
    footnotes = {}

    for ax, key in zip(axes, PANEL_KEYS):
        rate_fn = RATE_FN[key]
        arm = ARM_NAME[key]
        desc = DESC[lang][key]
        ax.set_xlim(-0.3, 4 * cell + 0.3)
        ax.set_ylim(-0.55, 3 * (4 * cell + gap_ch) - gap_ch + 0.3)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f"{arm}\n{desc}", fontsize=8.3, linespacing=1.5, pad=8)

        n_diff = 0
        for ci, name in enumerate(ORDER):
            y0 = (2 - ci) * (4 * cell + gap_ch)
            v = CH[name]
            rate = rate_fn(name)
            bref = base_rate[name]
            for r in range(4):
                for c in range(4):
                    x = c * cell
                    y = y0 + (3 - r) * cell
                    color = cmap(norm(rate[r, c]))
                    ax.add_patch(Rectangle((x, y), cell, cell, facecolor=color,
                                            edgecolor='white', linewidth=0.8))
                    txt_color = 'white' if luminance(color) < 0.5 else 'black'
                    ax.text(x + cell / 2, y + cell / 2, f'{int(v[r, c])}',
                             ha='center', va='center', fontsize=7.2, color=txt_color)
                    if key == 'none' and abs(rate[r, c] - bref[r, c]) > 1e-9:
                        n_diff += 1
                        ax.scatter([x + cell - 0.11], [y + cell - 0.11], marker='^',
                                   s=26, facecolor='#D55E00', edgecolor='white',
                                   linewidth=0.4, zorder=6)
            ax.text(-0.28, y0 + 2 * cell, f'ch {name}', ha='right', va='center',
                     fontsize=8, fontweight='bold', rotation=90)

            if key in ('within_channel', 'none'):
                if key == 'within_channel':
                    winners = np.argwhere(v == own_max[name])
                else:
                    winners = np.argwhere(v == layer_max) if own_max[name] == layer_max \
                        else np.empty((0, 2), dtype=int)
                for (r, c) in winners:
                    x = c * cell
                    y = y0 + (3 - r) * cell
                    ax.add_patch(Rectangle((x, y), cell, cell, facecolor='none',
                                            edgecolor='#D55E00', linewidth=2.2))
            else:  # channel: 채널 통째로 표시, 1등 채널만 굵게
                is_winner_channel = ch_sum[name] == ch_sum_max
                lw = 2.6 if is_winner_channel else 1.0
                ec = '#D55E00' if is_winner_channel else '#888888'
                ax.add_patch(Rectangle((0, y0), 4 * cell, 4 * cell, facecolor='none',
                                        edgecolor=ec, linewidth=lw,
                                        linestyle='-' if is_winner_channel else '--'))

        # panel-specific diff footnote (recorded here, drawn later via fig.text so long
        # strings don't spill into the neighboring subplot)
        if key == 'none':
            footnotes[key] = ({'ko': f'▲ = ours_intra_ch 대비 다름\n(n={n_diff}/48칸)',
                                'en': f'▲ = differs from ours_intra_ch\n(n={n_diff}/48 cells)'}[lang],
                               '#D55E00')
        elif key == 'channel':
            footnotes[key] = ({'ko': '채널 전체 균일값\n(48칸 전부 다름)',
                                'en': 'channel-uniform\n(all 48 cells differ)'}[lang],
                               '#555555')

    for key, ax in zip(PANEL_KEYS, axes):
        if key in footnotes:
            text, color = footnotes[key]
            pos = ax.get_position()
            xc = (pos.x0 + pos.x1) / 2.0
            fig.text(xc, pos.y0 - 0.015, text, ha='center', va='top',
                      fontsize=6.4, color=color, linespacing=1.35)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label({'ko': 'sc_rate (0=보호, 1=최대 억제)',
                  'en': 'sc_rate (0=protected, 1=max suppression)'}[lang], fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    title = {'ko': "경쟁 범위에 따른 sc_rate 배정 (β=1.0, T=4, illustrative)",
              'en': "sc_rate assignment by competition scope (β=1.0, T=4, illustrative)"}[lang]
    fig.suptitle(title, fontsize=9.3, y=0.975)
    foot = {'ko': '주황 테두리 = 그 그룹의 1등(sc_rate=0). ours_inter_ch 패널만 채널 안 전부가 같은 값(broadcast).'
                  ' 숫자·수치는 illustrative.',
            'en': 'Orange border = the winner of that competition group (sc_rate=0). Only the'
                  ' ours_inter_ch panel broadcasts one value across the whole channel. Numbers are illustrative.'}[lang]
    fig.text(0.5, 0.015, foot, ha='center', fontsize=7.0)

    pdf = os.path.join(OUT, f'competition_scope_{lang}_{DATE}.pdf')
    svg = os.path.join(OUT, f'competition_scope_{lang}_{DATE}.svg')
    png = os.path.join(OUT, f'competition_scope_{lang}_{DATE}.png')
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
