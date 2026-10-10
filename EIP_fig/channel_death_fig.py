"""같은 스파이크에서 완전 침묵 채널 수 (26-09-22).

파레토 곡선(정확도 x 스파이크)에서는 모든 팔이 구별되지 않는다. 이 그림은
그 곡선이 못 보는 축이다: 같은 스파이크 예산에서 **통째로 죽는 채널의 수**.
채널이 통째로 죽으면 그 필터를 하드웨어에서 들어낼 수 있다 -- 위치 단위로
드문드문 죽는 것은 연산량을 줄이지 못한다.

완전 침묵 채널 = 테스트 10,000장 전체에서 그 채널의 모든 뉴런이 한 번도 발화하지 않음.
18개 스파이킹 층(n_in, predictions_n 제외)의 4,736개 채널에서 셌다.

수치는 npz 에서 직접 세어 넣었다 (EIP_fig/_channel_death_data.json 로 저장).
시드별 개별 점을 전부 그린다 -- 평균만 그리면 n=3 의 산포가 안 보인다.

쓰는 법:  python EIP_fig/channel_death_fig.py [--lang ko|en|both]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = os.path.dirname(os.path.abspath(__file__))
STAMP = '26-09-22'
FONT = {'ko': 'Noto Sans CJK JP', 'en': 'Liberation Sans'}
N_CHAN = 4736

# (표시이름, [(spikes/샘플, 죽은채널), ...])  npz 에서 직접 셈. 각 점이 한 시드.
DATA = {
    'baseline':      [(514634, 530)],
    'plain L2':      [(405496, 568), (207932, 778)],
    'ours_inter_ch': [(409693, 596)],
    'ours_intra_ch': [(216134, 803), (211853, 809), (215749, 854),
                      (209080, 828), (213590, 834), (181189, 868)],
    'ours_layer':    [(214002, 898), (214443, 920), (224076, 842)],
    'ours_w1':       [(199258, 878), (185464, 903), (200411, 903)],
}
STYLE = {
    'baseline':      ('#222222', '*', 13),
    'plain L2':      ('#7F7F7F', 'd', 8),
    'ours_inter_ch': ('#009E73', '^', 8),
    'ours_intra_ch': ('#0072B2', 'o', 7),
    'ours_layer':    ('#D55E00', 's', 7),
    'ours_w1':       ('#CC79A7', 'v', 7),
}
ORDER = ['baseline', 'plain L2', 'ours_inter_ch', 'ours_intra_ch', 'ours_layer', 'ours_w1']

TXT = {
    'ko': dict(x='스파이크 수 (test, 샘플당)', y=f'완전 침묵 채널 수 (전체 {N_CHAN:,})',
               title='같은 스파이크에서 통째로 죽는 채널',
               ref='plain L2 기준선',
               foot=('점 하나가 한 시드. 완전 침묵 채널 = 테스트 10,000장 전체에서 그 채널의 '
                     '모든 뉴런이 한 번도 발화하지 않음.\n'
                     '회색 선은 plain L2 두 점을 이은 기준선이다 (n=1 씩, 보간 근거가 약하다). '
                     '위로 떨어질수록 같은 예산에서 채널을 더 제거한 것이다.')),
    'en': dict(x='spike count (test, per sample)', y=f'fully silent channels (of {N_CHAN:,})',
               title='Channels removed entirely at matched spike budget',
               ref='plain L2 reference',
               foot=('One marker per seed. A channel is fully silent when none of its neurons '
                     'fired on any of the 10,000 test images.\n'
                     'The grey line connects the two plain L2 points (n=1 each; a weak basis for '
                     'interpolation). Higher means more channels removed at the same budget.')),
}


def draw(lang):
    t = TXT[lang]
    plt.rcParams['font.family'] = FONT[lang]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 9

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    # plain L2 기준선. 두 점뿐이라 외삽하지 않고 그 구간만 긋는다.
    l2 = sorted(DATA['plain L2'])
    ax.plot([p[0] for p in l2], [p[1] for p in l2], color='0.6', lw=1.2,
            ls='--', zorder=1, label=t['ref'])

    for arm in ORDER:
        pts = DATA[arm]
        color, mk, ms = STYLE[arm]
        n = len(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk, ms=ms,
                lw=0, color=color, mfc=color, mew=0, alpha=.9, zorder=3,
                label=f'{arm} (n={n})')

    ax.set_xscale('log')
    ax.set_xlabel(t['x']); ax.set_ylabel(t['y'])
    ax.set_title(t['title'], fontsize=10, pad=8)
    ax.grid(alpha=.22, which='major', lw=.5)
    ax.set_xticks([2e5, 3e5, 4e5, 5e5])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{int(v):,}'))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ''))
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(loc='upper right', fontsize=7.6, framealpha=.95)

    fig.text(0.012, 0.012, t['foot'], fontsize=6.4, va='bottom', linespacing=1.45)
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    for ext in ('pdf', 'svg', 'png'):
        p = os.path.join(OUT, f'channel_death_{lang}_{STAMP}.{ext}')
        fig.savefig(p, dpi=220 if ext == 'png' else None, bbox_inches='tight')
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--lang', default='ko', choices=['ko', 'en', 'both'])
    a = ap.parse_args()
    json.dump(DATA, open(os.path.join(OUT, '_channel_death_data.json'), 'w'), indent=1)
    for L in (['ko', 'en'] if a.lang == 'both' else [a.lang]):
        draw(L)
