"""Method 절 개념도 5/5 (그림 5) — w1 vs w2 기울기 경로 (ours vs ours_w1).

실험 수치 없음. layers.py:2021-2093 의 두 custom_gradient 함수만 그린다. 두 경로는
forward 가 완전히 같다: out = sqrt(sum((spike*sc_rate)^2)). 차이는 그 곱셈이
tf.custom_gradient 로 감싼 상자 **안**이냐 **밖**이냐 하나뿐이다.

  ours (w^2, 기본값, reg_spike_wta_rev_w1=False):
    neurons.py:1091   sc_loss = spike_reg * sc_rate            # 곱셈이 상자 밖
    layers.py:2021    l2_norm_wta_rev(x, sc_rate, name)        # x = spike*sc_rate 를 받음
                       backward: dL/dx = sc_rate/||x||         # 상자 안에서 한 번
    상자를 나가면 autodiff 가 x=spike*sc_rate 의 곱셈을 다시 미분해 dx/dspike=sc_rate 를
    한 번 더 곱한다 -> dL/dspike = sc_rate^2/||x||  (docstring: "실측 확인").

  ours_w1 (w^1, reg_spike_wta_rev_w1=True):
    layers.py:2077    l2_norm_wta_rev_w1(spike, sc_rate, name)  # spike 를 직접 받음
                       x = spike*sc_rate 를 상자 **안**에서 계산
                       backward: dL/dspike = sc_rate/||x||      # 상자 안에서 한 번뿐

sc_rate 는 spike_count(tf.Variable)에서 나와 기울기가 없으므로 두 함수 모두
sc_rate 쪽 기울기는 None 이다. reg_spike_wta_rev_w1 기본값은 False(=w^2).

결론(2026-09-21 확정, 캡션에 명시): 두 경로는 정확도-스파이크 곡선, 뉴런 분포,
채널 수 전부에서 구별되지 않는다. w1 이 바꾸는 것은 사실상 rho 눈금뿐이다 -- 이
그림은 "무엇이 다른가"를 보이는 것이지 "무엇이 낫다"가 아니다.

--lang en|ko 로 영문/국문 두 벌을 뽑는다. 영문이 논문 정본.
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
DATE = '26-09-21'
FONT = {'ko': 'Noto Sans CJK JP', 'en': 'Liberation Sans'}

C_FWD = '#333333'
C_BWD = '#0072B2'     # blue, backward path
C_MARK = '#D55E00'    # vermillion, sc_rate multiplication marker
C_BOX = '#999999'
C_W1 = '#009E73'      # bluish green
C_W2 = '#CC79A7'      # reddish purple

STR = {
    'ko': dict(
        suptitle='w¹ vs w² 기울기 경로 — 곱셈이 custom_gradient 상자 안이냐 밖이냐',
        panel_w2='ours (w², 기본값)', panel_w1='ours_w1 (w¹)',
        n_spike='spike', n_x='x = spike·sc_rate', n_norm='||x||', n_loss='loss',
        n_spike1='spike', n_loss1='loss',
        box_label='custom_gradient 상자',
        fwd_label='forward (동일)', bwd_label='backward',
        mark1='×sc_rate\n(상자 안)', mark2='×sc_rate\n(상자 밖, autodiff)',
        mark1_w1='×sc_rate\n(상자 안, 한 번뿐)',
        total_w2=r'$\partial L/\partial\mathrm{spike} = \mathrm{sc\_rate}^2/\Vert x\Vert$  (두 번 곱함)',
        total_w1=r'$\partial L/\partial\mathrm{spike} = \mathrm{sc\_rate}/\Vert x\Vert$  (한 번 곱함)',
        curve_title='같은 sc_rate 에서 유효 가중치: $w$ vs $w^2$ (illustrative)',
        curve_x='sc_rate ($=w$, 발화 뉴런은 $w{<}1$)', curve_y='유효 가중치',
        curve_w1='$w$  (ours_w1 경로)', curve_w2='$w^2$  (ours 경로)',
        curve_note='$w{<}1$ 구간(발화 뉴런)에서 $w^2{<}w$\n→ 같은 $\\rho$ 에서 w¹ 이 발화 뉴런을 더 세게 누른다',
        foot='박스 = tf.custom_gradient 경계. 점선 화살표=backward. 곡선의 수치는 illustrative. '
             '2026-09-21 확정: 두 경로는 정확도-스파이크 곡선·뉴런 분포·채널 수에서 구별되지 않는다 — '
             '이 그림은 "무엇이 다른가"이지 "무엇이 낫다"가 아니다.',
    ),
    'en': dict(
        suptitle='w¹ vs w² gradient path — is the multiply inside or outside the custom_gradient box',
        panel_w2='ours (w², default)', panel_w1='ours_w1 (w¹)',
        n_spike='spike', n_x='x = spike·sc_rate', n_norm='||x||', n_loss='loss',
        n_spike1='spike', n_loss1='loss',
        box_label='custom_gradient box',
        fwd_label='forward (identical)', bwd_label='backward',
        mark1='×sc_rate\n(inside box)', mark2='×sc_rate\n(outside box, via autodiff)',
        mark1_w1='×sc_rate\n(inside box, only once)',
        total_w2=r'$\partial L/\partial\mathrm{spike} = \mathrm{sc\_rate}^2/\Vert x\Vert$  (multiplied twice)',
        total_w1=r'$\partial L/\partial\mathrm{spike} = \mathrm{sc\_rate}/\Vert x\Vert$  (multiplied once)',
        curve_title='Same sc_rate, effective weight: $w$ vs $w^2$ (illustrative)',
        curve_x='sc_rate ($=w$; fired neurons have $w{<}1$)', curve_y='effective weight',
        curve_w1='$w$  (ours_w1 path)', curve_w2='$w^2$  (ours path)',
        curve_note='For $w{<}1$ (fired neurons), $w^2{<}w$\n→ at the same $\\rho$, w¹ pushes harder on fired neurons',
        foot='Box = tf.custom_gradient boundary. Dashed arrows = backward. Curve values are illustrative. '
             'Confirmed 2026-09-21: the two paths are indistinguishable in accuracy-spike curves, neuron '
             'distributions, and channel counts — this figure shows what differs, not what is better.',
    ),
}


def draw_box_graph(ax, lang, arm):
    s = STR[lang]
    is_w2 = (arm == 'w2')

    node_y = 0.42
    fwd_y = node_y + 0.11
    bwd_y = node_y - 0.11
    mark_y = bwd_y - 0.09
    box_y0, box_y1 = node_y - 0.16, node_y + 0.16
    box_label_y = box_y1 + 0.03
    xs = {'spike': 0.10, 'x': 0.40, 'norm': 0.68, 'loss': 0.92}

    def node(x, label, w=0.16):
        b = FancyBboxPatch((x - w / 2, node_y - 0.075), w, 0.15,
                            boxstyle='round,pad=0.012,rounding_size=0.02',
                            linewidth=1.1, edgecolor=C_FWD, facecolor='white', zorder=5)
        ax.add_patch(b)
        ax.text(x, node_y, label, ha='center', va='center', fontsize=5.6, zorder=6)

    if is_w2:
        node(xs['spike'], s['n_spike'])
        node(xs['x'], s['n_x'], w=0.22)
        node(xs['norm'], s['n_norm'])
        node(xs['loss'], s['n_loss'])
        box_x0, box_x1 = xs['x'] + 0.11, xs['loss'] + 0.045
    else:
        node(xs['spike'], s['n_spike1'])
        node(xs['loss'], s['n_loss1'])
        box_x0, box_x1 = xs['spike'] - 0.045, xs['loss'] + 0.045

    # custom_gradient box
    box = FancyBboxPatch((box_x0, box_y0), box_x1 - box_x0, box_y1 - box_y0,
                          boxstyle='round,pad=0.006,rounding_size=0.015',
                          linewidth=1.3, edgecolor=C_BOX, facecolor='none',
                          linestyle='--', zorder=2)
    ax.add_patch(box)
    ax.text((box_x0 + box_x1) / 2, box_label_y, s['box_label'], ha='center',
             va='bottom', fontsize=5.3, color=C_BOX)

    # forward arrows (solid, above the node boxes)
    fwd_pairs = [('spike', 'x'), ('x', 'norm'), ('norm', 'loss')] if is_w2 else [('spike', 'loss')]
    for a, b in fwd_pairs:
        ax.annotate('', xy=(xs[b] - 0.08, fwd_y), xytext=(xs[a] + 0.08, fwd_y),
                     arrowprops=dict(arrowstyle='-|>', color=C_FWD, lw=1.0))

    # backward arrows (dashed, below the node boxes) with sc_rate multiplication markers
    if is_w2:
        ax.annotate('', xy=(xs['x'] + 0.06, bwd_y), xytext=(xs['norm'] - 0.06, bwd_y),
                     arrowprops=dict(arrowstyle='-|>', color=C_BWD, lw=1.1, linestyle='--'))
        ax.scatter([(xs['x'] + xs['norm']) / 2], [bwd_y], marker='o', s=30,
                   color=C_MARK, zorder=7)
        ax.text((xs['x'] + xs['norm']) / 2, mark_y, s['mark1'], ha='center', va='top',
                 fontsize=5.1, color=C_MARK, linespacing=1.2)

        ax.annotate('', xy=(xs['spike'] + 0.07, bwd_y), xytext=(xs['x'] - 0.10, bwd_y),
                     arrowprops=dict(arrowstyle='-|>', color=C_BWD, lw=1.1, linestyle='--'))
        ax.scatter([(xs['spike'] + xs['x']) / 2], [bwd_y], marker='o', s=30,
                   color=C_MARK, zorder=7)
        ax.text((xs['spike'] + xs['x']) / 2, mark_y, s['mark2'], ha='center', va='top',
                 fontsize=5.1, color=C_MARK, linespacing=1.2)
    else:
        ax.annotate('', xy=(xs['spike'] + 0.08, bwd_y), xytext=(xs['loss'] - 0.08, bwd_y),
                     arrowprops=dict(arrowstyle='-|>', color=C_BWD, lw=1.1, linestyle='--'))
        ax.scatter([(xs['spike'] + xs['loss']) / 2], [bwd_y], marker='o', s=30,
                   color=C_MARK, zorder=7)
        ax.text((xs['spike'] + xs['loss']) / 2, mark_y, s['mark1_w1'], ha='center', va='top',
                 fontsize=5.1, color=C_MARK, linespacing=1.2)

    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(0.0, 1.0)
    ax.axis('off')


def draw(lang):
    s = STR[lang]
    plt.rcParams['font.family'] = FONT[lang]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 7.6

    fig = plt.figure(figsize=(3.3, 4.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.3], hspace=0.55,
                           left=0.10, right=0.97, top=0.89, bottom=0.12)
    ax_top1 = fig.add_subplot(gs[0, 0])
    ax_top2 = fig.add_subplot(gs[1, 0])

    draw_box_graph(ax_top1, lang, 'w2')
    ax_top1.set_title(s['panel_w2'], fontsize=7.6, color=C_W2, fontweight='bold', pad=14)
    ax_top1.text(0.5, 0.97, s['total_w2'], transform=ax_top1.transAxes,
                  ha='center', va='top', fontsize=6.0)

    draw_box_graph(ax_top2, lang, 'w1')
    ax_top2.set_title(s['panel_w1'], fontsize=7.6, color=C_W1, fontweight='bold', pad=14)
    ax_top2.text(0.5, 0.97, s['total_w1'], transform=ax_top2.transAxes,
                  ha='center', va='top', fontsize=6.0)

    # bottom curve panel
    ax_c = fig.add_subplot(gs[2, 0])
    w = np.linspace(0, 1, 200)
    ax_c.plot(w, w, color=C_W1, lw=1.8, label=s['curve_w1'], zorder=3)
    ax_c.plot(w, w ** 2, color=C_W2, lw=1.8, linestyle='--', label=s['curve_w2'], zorder=3)
    ax_c.fill_between(w, w ** 2, w, color='#eeeeee', zorder=1)
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1.05)
    ax_c.set_xlabel(s['curve_x'], fontsize=6.6)
    ax_c.set_ylabel(s['curve_y'], fontsize=6.6)
    ax_c.set_title(s['curve_title'], fontsize=7.4, pad=6)
    ax_c.legend(fontsize=6.0, loc='upper left', framealpha=0.9)
    ax_c.tick_params(labelsize=6.0)
    ax_c.grid(alpha=0.25)
    ax_c.text(0.97, 0.06, s['curve_note'], transform=ax_c.transAxes, ha='right', va='bottom',
               fontsize=5.6, color='#444444', linespacing=1.3)

    fig.suptitle(s['suptitle'], fontsize=8.4, y=0.985, linespacing=1.4)
    fig.text(0.5, 0.005, s['foot'], ha='center', fontsize=5.5, wrap=True)

    pdf = os.path.join(OUT, f'w1_vs_w2_gradient_{lang}_{DATE}.pdf')
    svg = os.path.join(OUT, f'w1_vs_w2_gradient_{lang}_{DATE}.svg')
    png = os.path.join(OUT, f'w1_vs_w2_gradient_{lang}_{DATE}.png')
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
