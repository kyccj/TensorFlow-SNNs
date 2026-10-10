"""한 층의 뉴런 발화 분포 비교 (26-09-21).

층 선택은 추측이 아니라 실측이다: 스파이크가 맞춰진 네 팔(baseline / plain L2 5e-7 /
ours_intra_ch 4e-3 / ours_layer 4e-3)에서 18개 스파이킹 층의 gini 차와 영구침묵률 차를
전부 재서, 둘 다 1위인 conv2_block1_conv1 을 골랐다 (Δgini +0.125, Δ침묵 +13.2%p).
2위 conv2_block2_out 은 +0.121 / +3.9%p.

왼쪽  로렌츠 곡선 -- 뉴런을 활동량 오름차순으로 세웠을 때의 누적 스파이크 점유.
              대각선에서 멀수록 소수 뉴런에 몰려 있다. gini 가 그 면적이다.
오른쪽 영구 침묵률 -- 테스트 10,000장 전체에서 한 번도 발화하지 않은 뉴런의 비율.

ours_inter_ch 는 스파이크 예산이 맞지 않는다. 가장 센 rho 에서도 312,597 이라
이 대역(~190K)에 내려오지 못한다 -- 그 자체가 결과다. 범례에 예산을 병기하고
다른 선종으로 구분한다.

쓰는 법:  python EIP_fig/neuron_dist_layer_fig.py [--lang ko|en|both]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(os.path.dirname(OUT), '_npz')
STAMP = '26-09-21'
FONT = {'ko': 'Noto Sans CJK JP', 'en': 'Liberation Sans'}
LAYER = 'conv2_block1_conv1_n'

# (표시이름, npz, 완주 spikes, n, 예산이 맞는가)
RUNS = [
    ('baseline',      'r19c10-base---s11',        486445, 5, True),
    ('plain L2',      'r19c10-l2-5e-7-s5',        187911, 5, True),
    ('ours_intra_ch', 'r19c10-prop-4e-3-s4',      188208, 5, True),
    ('ours_layer',    'r19c10-ours_layer-4e-3-s2', 195989, 3, True),
    ('ours_inter_ch', 'r19c10-ours_ch-2e-3-s1',   382189, 1, False),
]
STYLE = {
    'baseline':      ('#222222', ':'),
    'plain L2':      ('#7F7F7F', (0, (1, 1.4))),
    'ours_intra_ch': ('#0072B2', '-'),
    'ours_layer':    ('#D55E00', '--'),
    'ours_inter_ch': ('#009E73', '-.'),
}
TXT = {
    'ko': dict(
        title=f'{LAYER} 뉴런 발화 분포 (뉴런 131,072개)',
        lx='뉴런 비율 (활동량 오름차순)', ly='누적 스파이크 점유',
        rx='', ry='영구 침묵 뉴런 (%)', diag='완전 균일',
        lt='(a) 로렌츠 곡선', rt='(b) 영구 침묵률',
        budget='스파이크', nolab='예산 불일치',
        foot=('영구 침묵 = 테스트 10,000장 전체에서 한 번도 발화하지 않은 뉴런. '
              'npz 는 팔당 n=1 이라 분포 지표에 시드 SD 가 없다\n'
              '(괄호 안 n 은 그 설정의 완주 런 수이고 분포는 그중 한 런에서 쟀다). '
              'ours_inter_ch 는 가장 센 ρ 에서도 312,597 이라 이 대역에 못 온다.')),
    'en': dict(
        title=f'Neuron firing distribution at {LAYER} (131,072 neurons)',
        lx='neuron fraction (sorted by activity)', ly='cumulative spike share',
        rx='', ry='permanently silent neurons (%)', diag='perfectly uniform',
        lt='(a) Lorenz curve', rt='(b) permanently silent',
        budget='spikes', nolab='budget mismatch',
        foot=('Permanently silent = never fired across all 10,000 test images. '
              'npz is n=1 per arm, so the distribution metrics carry no seed SD\n'
              '(the n in parentheses is the number of completed runs of that setting; '
              'the distribution was measured on one of them).\n'
              'ours_inter_ch cannot reach this budget: 312,597 even at its strongest rho.')),
}


def gini(x):
    x = np.sort(np.asarray(x, float).ravel())
    n, s = x.size, x.sum()
    return 0.0 if s <= 0 else float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


def draw(lang):
    t = TXT[lang]
    plt.rcParams['font.family'] = FONT[lang]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 8.5

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(7.0, 3.5), gridspec_kw={'width_ratios': [1.55, 1]})

    axl.plot([0, 1], [0, 1], color='0.75', lw=.9, ls='-', zorder=1)
    axl.text(.60, .53, t['diag'], fontsize=6.6, color='0.55', rotation=33,
             ha='center', va='center')

    names, sils, cols = [], [], []
    for name, tag, spikes, n, matched in RUNS:
        v = np.load(os.path.join(NPZ, tag + '.npz'))[LAYER].ravel().astype(float)
        color, ls = STYLE[name]

        x = np.sort(v)
        cum = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
        frac = np.linspace(0, 1, cum.size)
        lab = f"{name}  {spikes:,} {t['budget']} (n={n})"
        if not matched:
            lab += f"  ← {t['nolab']}"
        axl.plot(frac, cum, ls=ls, lw=1.6, color=color,
                 alpha=.95 if matched else .6, label=lab, zorder=3)

        names.append(name)
        sils.append(float((v == 0).mean() * 100))
        cols.append(color)

    axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    axl.set_xlabel(t['lx']); axl.set_ylabel(t['ly'])
    axl.set_title(t['lt'], fontsize=9, pad=6)
    axl.grid(alpha=.2, lw=.5)
    axl.legend(loc='upper left', fontsize=6.5, framealpha=.95)

    bars = axr.bar(range(len(names)), sils, color=cols, width=.62,
                   edgecolor='white', linewidth=.8)
    for b, s in zip(bars, sils):
        axr.text(b.get_x() + b.get_width() / 2, s + 1.2, f'{s:.1f}',
                 ha='center', fontsize=7.4)
    axr.set_xticks(range(len(names)))
    axr.set_xticklabels(names, rotation=38, ha='right', fontsize=7.2)
    axr.set_ylabel(t['ry'])
    axr.set_ylim(0, max(sils) * 1.22)
    axr.set_title(t['rt'], fontsize=9, pad=6)
    axr.grid(alpha=.2, axis='y', lw=.5)

    fig.suptitle(t['title'], fontsize=9.8, y=0.995)
    fig.text(0.012, 0.012, t['foot'], fontsize=6.2, va='bottom', linespacing=1.45)
    fig.tight_layout(rect=(0, 0.14, 1, 0.965))

    for ext in ('pdf', 'svg', 'png'):
        p = os.path.join(OUT, f'neuron_dist_layer_{lang}_{STAMP}.{ext}')
        fig.savefig(p, dpi=220 if ext == 'png' else None, bbox_inches='tight')
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='ko', choices=['ko', 'en', 'both'])
    a = ap.parse_args()
    for L in (['ko', 'en'] if a.lang == 'both' else [a.lang]):
        draw(L)
