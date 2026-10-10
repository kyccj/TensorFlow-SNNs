"""망 전체 영구 침묵률 (26-09-22).

한 층이 아니라 **스파이킹 18층 전부**(1,442,048 뉴런) 기준이다.
영구 침묵 = 테스트 10,000장 전체에서 한 번도 발화하지 않은 뉴런.

(a) 팔별 전역 침묵률. 시드가 여럿인 팔은 개별 점을 겹쳐 찍고 평균에 오차막대.
(b) 층별 침묵률. 규제가 어느 층을 끄는지 -- conv2 에서 갈리고 conv3 이후는 붙는다.
    `_out` 층(skip connection 합류점)은 baseline 에서 침묵률이 0 이라 따로 표시한다.

로렌츠 곡선은 뺐다.

수치는 npz 에서 직접 셌다 (EIP_fig/_silent_global_data.json).

쓰는 법:  python EIP_fig/silent_global_fig.py [--lang ko|en|both]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
STAMP = '26-09-22'
FONT = {'ko': 'Noto Sans CJK JP', 'en': 'Liberation Sans'}
N_NEU = 1442048
SRC = '/tmp/global_silent.json'

# 스파이크 예산이 맞는 팔만 그린다 (181,189 ~ 217,507 /샘플).
# 침묵률은 스파이크를 적게 쓸수록 자동으로 올라가므로 예산이 다르면 견줄 수 없다.
# 빠진 것: baseline 514,634 / plain L2 5e-8 405,496 / ours_inter_ch 2e-3 409,693.
MATCHED = ['plain L2 5e-7', 'ours_intra_ch 4e-3', 'ours_layer 4e-3',
           'ours_w1 3e-3', 'ours_intra_ch 5e-3', '- 1- inv 4e-2']
# baseline 은 규제가 없어 예산이 안 맞는다(514,634). 비교 대상이 아니라 출발점이므로
# 빗금과 구분선으로 떼어 놓고 (b) 에서는 점선으로 그린다.
REF = 'baseline'
OTHER = []

COLOR = {
    'baseline': '#222222', 'plain L2 5e-8': '#9F9F9F', 'plain L2 5e-7': '#7F7F7F',
    'ours_inter_ch 2e-3': '#009E73', 'ours_intra_ch 4e-3': '#0072B2',
    'ours_layer 4e-3': '#D55E00', 'ours_w1 3e-3': '#CC79A7',
    'ours_intra_ch 5e-3': '#56B4E9', '- 1- inv 4e-2': '#E69F00',
}
TXT = {
    'ko': dict(
        at='(a) 망 전체 영구 침묵률 — 스파이크 예산이 맞는 팔만', bt='(b) 층별 영구 침묵률',
        ay='영구 침묵 뉴런 (%)', by='영구 침묵 뉴런 (%)', bx='층',
        note='예산 불일치', seed='시드 개별값',
        ref='참조 (규제 없음)', band='스파이크 예산 181~218K',
        foot=(f'스파이킹 18층 {N_NEU:,} 뉴런. 영구 침묵 = 테스트 10,000장 전체에서 '
              '한 번도 발화하지 않음. 막대 안 숫자가 그 팔의 스파이크 수다.\n'
              '스파이크 예산이 맞는 팔만 그렸다 (181,189~217,507 /샘플). 침묵률은 스파이크를 '
              '적게 쓸수록 자동으로 올라가므로 예산이 다르면 견줄 수 없다.\n'
              'baseline 은 규제가 없어 예산(514,634)이 안 맞는다 — 비교 대상이 아니라 출발점이라 '
              '빗금으로 떼어 놓았다. 빠진 것 — plain L2 5e-8 405,496 / ours_inter_ch 2e-3 409,693. '
              'ours_inter_ch 는 rho=1.2e-2 가 이 대역(~193K 예측)에 들어올 예정이나 아직 미완주.\n'
              '오차막대 = 시드 SD (n>=2). (b) 음영 = `_out` 층(skip connection 합류점).')),
    'en': dict(
        at='(a) network-wide permanently silent', bt='(b) permanently silent by layer',
        ay='permanently silent neurons (%)', by='permanently silent neurons (%)', bx='layer',
        note='budget mismatch', seed='individual seeds',
        ref='reference (no reg.)', band='budget 181-218K',
        foot=(f'18 spiking layers, {N_NEU:,} neurons. Permanently silent = never fired on any '
              'of the 10,000 test images.\n'
              '(a) The left five share a spike budget (181-218K); the right four do not and must '
              'not be compared directly. Error bars = seed SD.\n'
              '(b) Budget-matched arms only. Shaded = `_out` layers (skip-connection joins), '
              'where baseline silence is 0.')),
}


def draw(lang, D):
    t = TXT[lang]
    plt.rcParams['font.family'] = FONT[lang]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 8.5

    fig, (axa, axb) = plt.subplots(
        2, 1, figsize=(7.0, 6.6), gridspec_kw={'height_ratios': [1, 1.25]})

    # ── (a) 전역 ────────────────────────────────────────────────────────
    order = [REF] + MATCHED
    for i, arm in enumerate(order):
        is_ref = arm == REF
        d = D[arm]
        vals = d['silent']
        m = float(np.mean(vals))
        c = COLOR[arm]
        axa.bar(i, m, width=.64, color=c, alpha=.45 if is_ref else 1.0,
                edgecolor='0.35' if is_ref else 'white', linewidth=.9,
                hatch='//' if is_ref else None, zorder=2)
        if len(vals) > 1:
            axa.errorbar(i, m, yerr=float(np.std(vals, ddof=1)), fmt='none',
                         ecolor='0.2', elinewidth=1.1, capsize=3, zorder=4)
            axa.plot([i] * len(vals), vals, marker='o', ms=2.6, lw=0,
                     color='0.15', alpha=.8, zorder=5)
        axa.text(i, m + 1.0, f'{m:.1f}', ha='center', fontsize=7.2, zorder=5)
        axa.text(i, 1.0, f"{np.mean(d['spikes']):,.0f}", ha='center', fontsize=5.8,
                 color='0.25' if is_ref else 'white', rotation=90, va='bottom', zorder=5)

    axa.axvline(0.5, color='0.55', lw=.9, ls=':')
    axa.text(0.0, 34.5, t['ref'], fontsize=6.6, color='0.35', ha='center')
    axa.text(3.5, 34.5, t['band'], fontsize=6.6, color='0.35', ha='center')
    axa.set_xticks(range(len(order)))
    axa.set_xticklabels([f"{a}\n(n={D[a]['n']})" for a in order],
                        rotation=32, ha='right', fontsize=6.9)
    axa.set_ylabel(t['ay'])
    axa.set_title(t['at'], fontsize=9.5, pad=6)
    axa.grid(alpha=.2, axis='y', lw=.5, zorder=0)
    axa.set_ylim(0, 38)

    # ── (b) 층별 ────────────────────────────────────────────────────────
    layers = list(D['baseline']['layer'].keys())
    short = [l[:-2].replace('conv', 'c').replace('_block', 'b').replace('_conv1', '_c1')
             for l in layers]
    x = np.arange(len(layers))
    for i, l in enumerate(layers):
        if l.endswith('_out_n'):
            axb.axvspan(i - .5, i + .5, color='0.9', zorder=0)
    vb = [float(np.mean(D[REF]['layer'][l])) for l in layers]
    axb.plot(x, vb, marker='o', ms=3.0, lw=1.2, ls=':', color=COLOR[REF],
             label=f"{REF} (n={D[REF]['n']}, 예산 불일치)", zorder=2)
    for arm in MATCHED:
        v = [float(np.mean(D[arm]['layer'][l])) for l in layers]
        axb.plot(x, v, marker='o', ms=3.4, lw=1.3, color=COLOR[arm],
                 label=f"{arm} (n={D[arm]['n']})", zorder=3)
    axb.set_xticks(x)
    axb.set_xticklabels(short, rotation=60, ha='right', fontsize=6.4)
    axb.set_ylabel(t['by']); axb.set_xlabel(t['bx'])
    axb.set_title(t['bt'], fontsize=9.5, pad=6)
    axb.grid(alpha=.2, axis='y', lw=.5)
    axb.legend(loc='upper right', fontsize=6.8, framealpha=.95)

    fig.text(0.012, 0.012, t['foot'], fontsize=6.2, va='bottom', linespacing=1.45)
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    for ext in ('pdf', 'svg', 'png'):
        p = os.path.join(OUT, f'silent_global_{lang}_{STAMP}.{ext}')
        fig.savefig(p, dpi=220 if ext == 'png' else None, bbox_inches='tight')
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='ko', choices=['ko', 'en', 'both'])
    a = ap.parse_args()
    D = json.load(open(SRC))
    json.dump(D, open(os.path.join(OUT, '_silent_global_data.json'), 'w'))
    for L in (['ko', 'en'] if a.lang == 'both' else [a.lang]):
        draw(L, D)
