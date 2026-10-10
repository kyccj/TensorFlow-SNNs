"""Method 절 개념도 3/5 (그림 3) — vmem readiness 가 침묵 뉴런의 sc_rate 뭉침을 어떻게 푸는지.

실험 수치 없음(비율은 illustrative). neurons.py:955-983 의 동작만 그린다.

배경 (neurons.py 주석, 실측): spike_count 는 0..T=4 정수라, T=4 에서 한 번도 안 쏜
뉴런은 전부 sc_rate 가 정확히 1.0 이 된다 -- 그리고 그게 층의 78~97% 다(실측, 이
그림에는 수치를 넣지 않는다 -- 캡션에만 적는다). wta_rev backward 는 그 전부에게
"같은" 기울기를 주므로 가중치가 사실상 균일해진다.

해법 (conf.reg_spike_vmem_gain, conf.reg_spike_vmem_silent_only):
  readiness = clip(vmem / vth, 0, 1)                 # vth=1.0
  scp = where(scp > 0, scp, gain * readiness)         # gain=1.0, silent_only=True

쏜 뉴런은 정수 spike_count 를 그대로 유지하고, 침묵 뉴런만 [0, gain) 에 연속적으로
퍼진다. 이 그림은 illustrative 뉴런 집단(N=60, 침묵 80%)에서 vmem 을 껐을 때/켰을
때 sc_rate 분포가 어떻게 갈리는지 히스토그램으로 비교한다.

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

# Okabe-Ito 색맹 안전 팔레트
C_FIRED = '#0072B2'    # blue
C_SILENT = '#999999'   # gray
C_SILENT_ON = '#D55E00'  # vermillion, vmem 적용 후 침묵 뉴런

rng = np.random.default_rng(7)

# ── illustrative 뉴런 집단 (N=60, T=4). 실측 78~97% 와 별개 -- 예시 값. ──────
N_FIRED = 12
N_SILENT = 48
T = 4
BETA = 1.0

# deterministic (not random draw) so group_max=4 is guaranteed and the discrete
# fired-neuron levels (0, 0.25, 0.5, 0.75) stay fixed and match the annotation text
fired_counts = np.array([1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4], dtype=float)
assert fired_counts.size == N_FIRED
# 침묵 뉴런의 vmem/vth: 문턱에서 먼 쪽에 몰리고 일부만 문턱 가까이 (illustrative)
readiness = rng.beta(1.5, 3.0, size=N_SILENT)

group_max = max(fired_counts.max(), 1)  # vmem 유무와 무관하게 동일 (readiness<1<=group_max)

sc_rate_fired = 1.0 - BETA * (fired_counts / group_max)

# vmem 없음: 침묵 뉴런 scp=0 -> sc_rate 정확히 1.0
sc_rate_silent_off = np.ones(N_SILENT)

# vmem 있음 (silent_only=True, gain=1.0): scp = gain*readiness
sc_rate_silent_on = 1.0 - BETA * (readiness / group_max)

STR = {
    'ko': dict(
        suptitle='vmem readiness 가 침묵 뉴런의 sc_rate 뭉침을 푼다 (illustrative, N=60, T=4)',
        panel_off='vmem 없음', panel_on='vmem 있음 (gain=1.0, 침묵 뉴런만)',
        xlabel='sc_rate (규제 가중치)', ylabel='뉴런 수',
        leg_fired='발화 뉴런 (정수 spike_count)',
        leg_silent_off='침묵 뉴런 (전부 sc_rate=1.0)',
        leg_silent_on='침묵 뉴런 (readiness로 퍼짐)',
        ann_off='침묵 뉴런 전부가 한 값에 뭉친다\n(역전파가 전부에게 같은 기울기를 준다)',
        ann_on='침묵 뉴런이 [0.75, 1.0) 에 연속적으로 퍼진다\n발화 뉴런 값(0, 0.25, 0.5, 0.75)과는 안 겹친다',
        foot='침묵 비율·readiness 분포는 illustrative(예시)다. 실측(78~97% 침묵)은 캡션 참조. '
             '침묵 뉴런만 나누는 이유·gain>1 의 위험도 캡션 참조.',
    ),
    'en': dict(
        suptitle='vmem readiness resolves the sc_rate pile-up of silent neurons (illustrative, N=60, T=4)',
        panel_off='vmem off', panel_on='vmem on (gain=1.0, silent neurons only)',
        xlabel='sc_rate (regularizer weight)', ylabel='# neurons',
        leg_fired='fired neurons (integer spike_count)',
        leg_silent_off='silent neurons (all sc_rate=1.0)',
        leg_silent_on='silent neurons (spread by readiness)',
        ann_off='all silent neurons pile up at one value\n(the backward pass gives them all the same gradient)',
        ann_on='silent neurons spread continuously over [0.75, 1.0)\nnever overlaps fired-neuron values (0, 0.25, 0.5, 0.75)',
        foot='The silent fraction and readiness distribution are illustrative. The measured silent '
             'fraction (78-97%) is in the caption, not this figure. See caption for why only silent '
             'neurons are split and the gain>1 ranking-inversion risk.',
    ),
}


def draw(lang):
    s = STR[lang]
    plt.rcParams['font.family'] = FONT[lang]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 8.3

    fig, axes = plt.subplots(2, 1, figsize=(3.3, 4.6), sharex=True,
                              gridspec_kw=dict(hspace=0.38))

    bins = np.linspace(0, 1, 41)

    # ── panel A: vmem off ──────────────────────────────────────────────
    ax = axes[0]
    ax.hist(sc_rate_fired, bins=bins, color=C_FIRED, edgecolor='white',
             linewidth=0.3, label=s['leg_fired'], zorder=3)
    ax.hist(sc_rate_silent_off, bins=bins, color=C_SILENT, edgecolor='white',
             linewidth=0.3, label=s['leg_silent_off'], zorder=3)
    ax.set_title(s['panel_off'], fontsize=8.6)
    ax.annotate(s['ann_off'], xy=(1.0, N_SILENT), xytext=(0.06, N_SILENT * 0.78),
                fontsize=6.4, color=C_SILENT, ha='left', va='top',
                arrowprops=dict(arrowstyle='->', color=C_SILENT, lw=0.9))
    ax.set_ylabel(s['ylabel'])
    ax.set_ylim(0, N_SILENT * 1.18)

    # ── panel B: vmem on ──────────────────────────────────────────────
    ax = axes[1]
    ax.hist(sc_rate_fired, bins=bins, color=C_FIRED, edgecolor='white',
             linewidth=0.3, label=s['leg_fired'], zorder=3)
    ax.hist(sc_rate_silent_on, bins=bins, color=C_SILENT_ON, edgecolor='white',
             linewidth=0.3, label=s['leg_silent_on'], zorder=3)
    ax.set_title(s['panel_on'], fontsize=8.6)
    ax.annotate(s['ann_on'], xy=(0.9, 6), xytext=(0.06, N_SILENT * 0.60),
                fontsize=6.4, color=C_SILENT_ON, ha='left', va='top',
                arrowprops=dict(arrowstyle='->', color=C_SILENT_ON, lw=0.9))
    ax.set_xlabel(s['xlabel'])
    ax.set_ylabel(s['ylabel'])
    ax.set_ylim(0, N_SILENT * 1.18)
    ax.set_xlim(-0.03, 1.05)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    for ax in axes:
        ax.legend(fontsize=6.0, loc='upper left', framealpha=0.92)
        ax.grid(alpha=0.25, axis='y', zorder=0)

    fig.suptitle(s['suptitle'], fontsize=8.6, y=0.995, linespacing=1.4)
    fig.text(0.5, 0.005, s['foot'], ha='center', fontsize=6.0, wrap=True)

    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    pdf = os.path.join(OUT, f'vmem_readiness_{lang}_{DATE}.pdf')
    svg = os.path.join(OUT, f'vmem_readiness_{lang}_{DATE}.svg')
    png = os.path.join(OUT, f'vmem_readiness_{lang}_{DATE}.png')
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
