"""ours 계열 세 팔 + baseline 파레토 곡선 (26-09-21).

실제 정확도(y) x 실제 스파이크 수(x). 차이(delta)로 그리지 않는다 (CLAUDE.md).

세 팔은 경쟁 범위만 다르다 -- loss-ratio, vmem 준비도, 마지막 스텝, 1- 역전,
w^2 기울기가 전부 같다.

선은 회귀선이 아니라 **각 팔의 rho 스윕을 실제 측정점끼리 이은 것**이다.
배제 규칙(310에폭 완주 AND S30/S1 >= 0.19)에 걸린 점은 그리지 않는다.

쓰는 법:  python EIP_fig/pareto_ours_family_fig.py [--lang ko|en|both]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = os.path.dirname(os.path.abspath(__file__))
STAMP = '26-09-21'
FONT = {'ko': 'Noto Sans CJK JP', 'en': 'Liberation Sans'}

# 시드 간 val acc SD (%p). CLAUDE.md 의 0.10~0.13 과 일치하는 실측 묶음값.
SEED_SD = 0.117

# ── 데이터 (eip-stats 확정본). (rho, val_acc, spikes, n) ────────────────────
# ours_layer 4e-3 은 09-21 12:39 에 s2/s3 가 완주해 n=1 -> n=3 (96.35 -> 96.50).
# ours_intra_ch 6e-3(S30/S1=0.175)·8e-3(0.135) 은 배제 규칙 위반이라 뺐다.
BASELINE = (96.73, 486445, 5)

ARMS = {
    'ours_intra_ch': [('2e-3', 96.64, 290020, 2), ('3e-3', 96.67, 234092, 7),
                      ('4e-3', 96.59, 192092, 5), ('5e-3', 96.42, 162846, 1)],
    'ours_layer':    [('1e-3', 96.67, 366001, 1), ('2e-3', 96.58, 290920, 1),
                      ('3e-3', 96.55, 234105, 3), ('4e-3', 96.50, 195989, 3)],
    'ours_inter_ch': [('2e-3', 96.77, 382189, 1), ('3e-3', 96.64, 347822, 1),
                      ('4e-3', 96.75, 312597, 1)],
    # 기존 방법. 곡선이 우리 계열에 고유하지 않다는 것이 이 팔의 역할이다.
    'plain L2':      [('5e-8', 96.77, 381250, 1), ('1e-7', 96.67, 324390, 1),
                      ('1.5e-7', 96.68, 290149, 1), ('2.5e-7', 96.59, 246178, 5),
                      ('4.5e-7', 96.49, 194964, 3), ('5e-7', 96.49, 185537, 5),
                      ('7.5e-7', 96.18, 160170, 1)],
}

# Okabe-Ito (색맹 안전) + 마커 모양 병행
STYLE = {
    'ours_intra_ch': ('#0072B2', 'o', '-'),
    'ours_layer':    ('#D55E00', 's', '--'),
    'ours_inter_ch': ('#009E73', '^', '-.'),
    'plain L2':      ('#7F7F7F', 'd', (0, (1, 1.4))),
}
DESC = {
    'ko': {'ours_intra_ch': '채널마다 경쟁', 'ours_layer': '층 전체가 경쟁',
           'ours_inter_ch': '채널 합끼리 경쟁', 'plain L2': '기존 방법'},
    'en': {'ours_intra_ch': 'per-channel', 'ours_layer': 'layer-wide',
           'ours_inter_ch': 'channel-sum', 'plain L2': 'prior work'},
}
TXT = {
    'ko': dict(x='스파이크 수 (test, 샘플당)', y='val accuracy (%)',
               base='baseline', n1='n=1',
               foot=('선은 회귀선이 아니라 각 팔의 ρ 스윕 측정점을 이은 것이다. '
                     '오차막대 = SEM (시드 SD 0.117%p), n≥2 만.\n'
                     '속 빈 마커 = n=1. 배제 규칙(310에폭 완주 AND S30/S1 ≥ 0.19) '
                     '위반 점은 그리지 않았다.')),
    'en': dict(x='spike count (test, per sample)', y='val accuracy (%)',
               base='baseline', n1='n=1',
               foot=('Lines connect measured points of each rho sweep, not a regression. '
                     'Error bars = SEM (seed SD 0.117%p), n>=2 only.\n'
                     'Open markers = n=1. Points violating the exclusion rule '
                     '(310 epochs AND S30/S1 >= 0.19) are not drawn.')),
}


def draw(lang):
    t = TXT[lang]
    plt.rcParams['font.family'] = FONT[lang]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    bacc, bsp, bn = BASELINE
    ax.axhline(bacc, color='0.45', ls=':', lw=1.0, zorder=1)
    ax.errorbar([bsp], [bacc], yerr=[SEED_SD / np.sqrt(bn)], fmt='*', ms=14,
                color='0.15', ecolor='0.15', elinewidth=1.1, capsize=2.5,
                zorder=5, label=f"{t['base']} (n={bn})")

    for arm, pts in ARMS.items():
        color, mk, ls = STYLE[arm]
        sp = [p[2] for p in pts]
        ac = [p[1] for p in pts]
        ax.plot(sp, ac, ls=ls, lw=1.3, color=color, alpha=.9, zorder=2,
                label=f"{arm} ({DESC[lang][arm]})")
        for _rho, acc, s, n in pts:
            ax.plot([s], [acc], marker=mk, ms=7, lw=0, color=color,
                    mfc=color if n >= 2 else 'white', mew=1.5, zorder=4)
            if n >= 2:
                ax.errorbar([s], [acc], yerr=[SEED_SD / np.sqrt(n)], fmt='none',
                            ecolor=color, elinewidth=1.2, capsize=2.5, zorder=4)

    ax.plot([], [], marker='o', ms=7, lw=0, color='0.35', mfc='white',
            mew=1.5, label=t['n1'])

    ax.set_xscale('log')
    ax.set_xlabel(t['x']); ax.set_ylabel(t['y'])
    ax.grid(alpha=.22, which='major', lw=.5)
    ax.set_xticks([1.5e5, 2e5, 3e5, 4e5, 5e5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{int(v):,}'))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ''))
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(loc='lower right', fontsize=7.6, framealpha=.95)

    fig.text(0.012, 0.012, t['foot'], fontsize=6.4, va='bottom', linespacing=1.45)
    fig.tight_layout(rect=(0, 0.11, 1, 1))

    for ext in ('pdf', 'svg', 'png'):
        p = os.path.join(OUT, f'pareto_ours_family_{lang}_{STAMP}.{ext}')
        fig.savefig(p, dpi=220 if ext == 'png' else None, bbox_inches='tight')
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='ko', choices=['ko', 'en', 'both'])
    a = ap.parse_args()
    for L in (['ko', 'en'] if a.lang == 'both' else [a.lang]):
        draw(L)
