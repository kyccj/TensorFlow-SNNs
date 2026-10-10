"""Method 절 개념도 4/5 (그림 4) — 규제를 어느 시간 단계에 거는가 (reg_spike_final_step).

실험 수치 없음. neurons.py:784-792, 1081-1090 의 동작만 그린다. T=4 예시.

  plain L2 (final_step=False): 매 타임스텝 t=1..T 마다 규제를 건다. 이때 sc_rate 는
    "그 시점까지의" spike_count 로 계산되므로, t=1 에서는 그 뉴런이 쏠 수 있는 최댓값이
    1 이라 sc_norm/sc_rate 가 0 또는 1 두 값으로만 무너진다(0/1 이분법). spike_reg 도
    그 시점의 순간 spike 값이다.

  ours (final_step=True): 규제를 t=T 한 번만 건다. 이때 spike_reg 는 t=1..T 의
    self.out(TensorArray) 을 전부 더한 값 -- 즉 시간축 전체를 적분한 뒤에 규제를
    건다. sc_rate 는 최종 spike_count(0..T 정수 전 구간)로 계산되어 0/1 이분법이
    생기지 않는다.

w1/w2(ours_w1) 기울기 경로 차이는 이 그림이 아니라 별도 그림(그림 5,
w1_vs_w2_gradient_fig.py)에서 다룬다 -- 캡션에서 그쪽을 가리킨다.

--lang en|ko 로 영문/국문 두 벌을 뽑는다. 영문이 논문 정본.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
DATE = '26-09-21'
FONT = {'ko': 'Noto Sans CJK JP', 'en': 'Liberation Sans'}

C_PLAIN = '#0072B2'   # blue
C_OURS = '#D55E00'    # vermillion
C_BRACE = '#555555'

T = 4
Y_PLAIN = 2.15
Y_OURS = 0.0
BRACE_Y = Y_OURS + 0.55

STR = {
    'ko': dict(
        suptitle='규제를 거는 시점 — plain L2 (매 스텝) vs ours (마지막 스텝 한 번), T=4',
        row_plain='plain L2\n(매 스텝마다)',
        row_ours='ours\n(마지막 스텝만)',
        plain_note='매 t 마다 규제.\n$t{=}1$ 은 그 순간까지의 최댓값이 1이라\nsc_rate가 0/1로만 무너진다',
        ours_note='$t{=}T$ 한 번만 규제.\n입력은 $t{=}1..T$ 누적 spike 합',
        brace_label='$t{=}1..T$ 를 적분한 뒤 규제',
        leg_plain='규제 적용 시점 (매 스텝)',
        leg_ours='규제 적용 시점 (마지막 스텝만)',
        foot='화살표·점=규제가 걸리는 시점. 수치는 illustrative. w¹ vs w² 기울기 경로 차이는 '
             '별도 그림(w¹ vs w² 기울기 경로)에서 다룬다.',
    ),
    'en': dict(
        suptitle='When the regularizer fires — plain L2 (every step) vs ours (final step once), T=4',
        row_plain='plain L2\n(every step)',
        row_ours='ours\n(final step only)',
        plain_note='regularized at every t.\nAt $t{=}1$ the max-so-far is 1,\nso sc_rate collapses to 0/1',
        ours_note='regularized once, at $t{=}T$.\nInput is the accumulated spike sum over $t{=}1..T$',
        brace_label='integrate $t{=}1..T$, then regularize',
        leg_plain='regularizer fires here (every step)',
        leg_ours='regularizer fires here (final step only)',
        foot='Arrows/dots mark when the regularizer fires. Values are illustrative. The w¹ vs w² '
             'gradient-path difference is covered in a separate figure (w¹ vs w² gradient path).',
    ),
}


def draw(lang):
    s = STR[lang]
    plt.rcParams['font.family'] = FONT[lang]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 8.3

    fig, ax = plt.subplots(figsize=(3.3, 3.75))

    ts = np.arange(1, T + 1)

    # timelines
    ax.plot([1, T], [Y_PLAIN, Y_PLAIN], color='#cccccc', lw=1.2, zorder=1)
    ax.plot([1, T], [Y_OURS, Y_OURS], color='#cccccc', lw=1.2, zorder=1)

    # plain L2: regularizer fires at every t
    for i, t in enumerate(ts):
        ax.scatter([t], [Y_PLAIN], s=70, facecolor=C_PLAIN, edgecolor='white',
                   linewidth=0.8, zorder=4,
                   label=s['leg_plain'] if i == 0 else None)

    # ours: regularizer fires only at t=T, fed by an accumulation over 1..T
    for t in ts[:-1]:
        ax.scatter([t], [Y_OURS], s=26, facecolor='white', edgecolor=C_OURS,
                   linewidth=1.1, zorder=4)
    ax.scatter([T], [Y_OURS], s=100, facecolor=C_OURS, edgecolor='white',
               linewidth=0.8, zorder=4, label=s['leg_ours'])

    # brace showing accumulation feeding the single final event (between the two rows)
    ax.plot([1, 1, T, T], [Y_OURS + 0.18, BRACE_Y, BRACE_Y, Y_OURS + 0.18],
            color=C_BRACE, lw=1.0, zorder=3)
    ax.text((1 + T) / 2, BRACE_Y + 0.06, s['brace_label'], ha='center', va='bottom',
             fontsize=6.3, color=C_BRACE)

    # row labels (left of t=1)
    ax.text(0.55, Y_PLAIN, s['row_plain'], ha='right', va='center', fontsize=7.4,
             color=C_PLAIN, fontweight='bold', linespacing=1.3)
    ax.text(0.55, Y_OURS, s['row_ours'], ha='right', va='center', fontsize=7.4,
             color=C_OURS, fontweight='bold', linespacing=1.3)

    # shared timestep labels, once, between the rows' own annotation bands
    for t in ts:
        ax.text(t, Y_OURS - 0.85, str(t), ha='center', va='center', fontsize=7.0,
                 color='#333333')
    ax.text(0.55, Y_OURS - 0.85, 't =', ha='right', va='center', fontsize=7.0,
             color='#333333')

    # annotations
    ax.annotate(s['plain_note'], xy=(1, Y_PLAIN), xytext=(1.15, Y_PLAIN + 0.62),
                fontsize=6.2, color=C_PLAIN, ha='left', va='bottom', linespacing=1.35,
                arrowprops=dict(arrowstyle='->', color=C_PLAIN, lw=0.8))
    ax.annotate(s['ours_note'], xy=(T, Y_OURS), xytext=(1.15, Y_OURS - 1.35),
                fontsize=6.2, color=C_OURS, ha='left', va='top', linespacing=1.35,
                arrowprops=dict(arrowstyle='->', color=C_OURS, lw=0.8,
                                 connectionstyle='arc3,rad=0.25'))

    ax.set_xlim(0.3, T + 0.6)
    ax.set_ylim(Y_OURS - 2.0, Y_PLAIN + 1.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ('top', 'right', 'left', 'bottom'):
        ax.spines[spine].set_visible(False)

    ax.legend(fontsize=6.0, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              framealpha=0.92, ncol=1)
    ax.set_title(s['suptitle'], fontsize=8.3, linespacing=1.4, pad=14)

    fig.text(0.5, 0.005, s['foot'], ha='center', fontsize=5.8, wrap=True)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    pdf = os.path.join(OUT, f'final_step_timing_{lang}_{DATE}.pdf')
    svg = os.path.join(OUT, f'final_step_timing_{lang}_{DATE}.svg')
    png = os.path.join(OUT, f'final_step_timing_{lang}_{DATE}.png')
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
