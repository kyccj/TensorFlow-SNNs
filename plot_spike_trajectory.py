"""
Figure: 스파이크 수는 학습 중 3배 넘게 변하는데, 왜 초반에 한 번 재면 되는가.

왼쪽  : 원본 궤적 — 모델마다 값이 다르고, 학습 중 크게 줄어든다
오른쪽: 각자 최종값으로 나눈 궤적 — 모양이 겹친다
아래  : 실제로 epoch 5 값 × 0.33 을 해 보면 최종값이 맞는지 검산
"""

import csv
import collections
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'EIP_figure', 'spike_trajectory.png')

C = {'VGG-C10': '#2a78d6', 'VGG-C100': '#eb6834',
     'R19-C10': '#1baf7a', 'R19-C100': '#eda100'}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8983', '#e3e2dd'
SURFACE, HILITE = '#fcfcfb', '#fdf3e3'

ORDER = ['VGG-C10', 'VGG-C100', 'R19-C10', 'R19-C100']
NICE = {'VGG-C10': 'VGG · CIFAR-10', 'VGG-C100': 'VGG · CIFAR-100',
        'R19-C10': 'ResNet19 · CIFAR-10', 'R19-C100': 'ResNet19 · CIFAR-100'}
TRAJ = {'VGG-C10': 'EIP-SNN-26_baseline-no-reg',
        'VGG-C100': 'EIP-SNN-26_baseline-no-reg-vgg-c100',
        'R19-C10': 'EIP-SNN-26_baseline-no-reg-r19-c10',
        'R19-C100': 'EIP-SNN-26_baseline-no-reg-r19-c100'}
# 실제 auto-k 실행이 epoch 5 에서 측정한 값 (baseline 궤적과는 다른 실행)
AUTOK_S5 = {'VGG-C10': 247762, 'VGG-C100': 253876, 'R19-C10': 1517208, 'R19-C100': 1453854}
MEAS_EP, FACTOR = 5, 0.33


def traj(d):
    per = collections.defaultdict(float)
    for r in csv.DictReader(open(os.path.join(ROOT, d, 'reg_detail.csv'))):
        v = float(r['spike_count'])
        if v > 0:
            per[int(r['epoch'])] += v
    e = sorted(per)
    return np.array(e), np.array([per[x] for x in e])


T = {k: traj(TRAJ[k]) for k in ORDER}
FINAL = {k: T[k][1][-1] for k in ORDER}
AT5 = {k: T[k][1][int(np.where(T[k][0] == MEAS_EP)[0][0])] for k in ORDER}
RATIO = {k: FINAL[k] / AT5[k] for k in ORDER}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10, length=3, color=GRID)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


fig = plt.figure(figsize=(15, 9.4), facecolor=SURFACE)
axL = fig.add_axes([0.065, 0.505, 0.38, 0.285])
axR = fig.add_axes([0.565, 0.505, 0.38, 0.285])
axT = fig.add_axes([0.065, 0.035, 0.88, 0.335]); axT.axis('off')

# ---------------------------------------------------------------- 왼쪽: 원본
style(axL)
for k in ORDER:
    e, s = T[k]
    axL.plot(e, s / 1000, color=C[k], lw=2, zorder=3)
axL.set_yscale('log')
axL.set_xlim(0, 480)
axL.set_xlabel('학습 진행 (epoch)', fontsize=11, color=INK2)
axL.set_ylabel('스파이크 수 (천 개)', fontsize=11, color=INK2)
axL.set_title('① 값도 다르고, 학습 중에 크게 줄어든다',
              fontsize=13, color=INK, loc='left', pad=12)
# 곡선이 아키텍처별 두 무리로 겹치므로 무리 단위로만 표시
for keys, name, col in [(['R19-C10', 'R19-C100'], 'ResNet19', C['R19-C10']),
                        (['VGG-C10', 'VGG-C100'], 'VGG', C['VGG-C10'])]:
    s0 = sum(T[k][1][0] for k in keys) / 2
    s1 = sum(T[k][1][-1] for k in keys) / 2
    axL.annotate(f'{name}\n{s0/1000:,.0f}K → {s1/1000:,.0f}K',
                 xy=(322, s1 / 1000), fontsize=11, color=col,
                 va='center', ha='left', fontweight='bold', linespacing=1.6)
axL.text(0.02, -0.26, '시작할 때 스파이크 수는 최종값의 3배가 넘는다.\n'
                      '그대로 쓰면 λ가 3배 틀린다.',
         transform=axL.transAxes, fontsize=10.5, color=MUTED, va='top', linespacing=1.7)

# ---------------------------------------------------------------- 오른쪽: 정규화
style(axR)
for k in ORDER:
    e, s = T[k]
    axR.plot(e, s / FINAL[k], color=C[k], lw=2.2, zorder=3, label=NICE[k])
axR.axvline(MEAS_EP, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=2)
axR.axhline(1 / FACTOR, color=MUTED, lw=1.2, ls=(0, (2, 3)), zorder=1)
axR.set_xlim(0, 315)
axR.set_ylim(0.75, 3.7)
axR.set_xlabel('학습 진행 (epoch)', fontsize=11, color=INK2)
axR.set_ylabel('그 실행의 최종값 대비 배율', fontsize=11, color=INK2)
axR.set_title('② 각자 최종값으로 나누면 — 네 곡선이 겹친다',
              fontsize=13, color=INK, loc='left', pad=12)
axR.text(MEAS_EP + 8, 3.45, 'epoch 5\n여기서 한 번 잰다', fontsize=10.5,
         color=INK, va='top', fontweight='bold', linespacing=1.6)
axR.text(300, 1 / FACTOR + 0.13, f'{1/FACTOR:.2f}배', fontsize=11, color=MUTED, ha='right')
axR.legend(loc='upper right', bbox_to_anchor=(1.0, 0.80), frameon=False,
           fontsize=10, labelcolor=INK2, handlelength=1.5)
axR.text(0.02, -0.26,
         f'epoch 5 에서 네 곡선이 모두 최종값의 약 {1/FACTOR:.1f}배에 있다.\n'
         f'모델이 달라도 줄어드는 모양이 같다는 뜻이다.',
         transform=axR.transAxes, fontsize=10.5, color=MUTED, va='top', linespacing=1.7)

# ---------------------------------------------------------------- 아래: 검산표
axT.set_xlim(0, 12)
axT.set_ylim(-0.7, 5)
axT.text(0, 4.6, '③ 그래서 실제로 이렇게 쓴다 — epoch 5 에서 잰 값에 0.33 을 곱한다',
         fontsize=13, color=INK, fontweight='bold')
axT.text(0, 4.05, 'auto-k 실행이 epoch 5 에 실제로 측정한 값으로 검산한 것. '
                  '오른쪽 끝이 예측이 얼마나 맞았는지.',
         fontsize=10.5, color=MUTED)

XN, X5, XM, XP, XA, XE = 0.1, 3.5, 4.8, 6.0, 8.0, 10.1
axT.add_patch(plt.Rectangle((XP - 0.95, 0.55), 1.9, 2.85, facecolor=HILITE,
                            edgecolor='none', zorder=0))
axT.text(X5, 3.45, 'epoch 5 실측', ha='center', fontsize=11, color=INK2)
axT.text(XP, 3.45, '× 0.33 = 예측', ha='center', fontsize=11, color=INK, fontweight='bold')
axT.text(XA, 3.45, '실제 최종값', ha='center', fontsize=11, color=INK2)
axT.text(XE, 3.45, '빗나간 정도', ha='center', fontsize=11, color=INK2)

for i, k in enumerate(ORDER):
    y = 2.85 - i * 0.62
    pred = AUTOK_S5[k] * FACTOR
    err = 100 * (pred / FINAL[k] - 1)
    axT.plot([XN + 0.1], [y], 'o', color=C[k], ms=10, zorder=3)
    axT.text(XN + 0.35, y, NICE[k], ha='left', va='center', fontsize=11.5, color=INK)
    axT.text(X5, y, f'{AUTOK_S5[k]/1000:,.0f}K', ha='center', va='center', fontsize=12, color=INK)
    axT.text(XM, y, '×0.33 =', ha='center', va='center', fontsize=10.5, color=MUTED)
    axT.text(XP, y, f'{pred/1000:,.0f}K', ha='center', va='center',
             fontsize=13, color=INK, fontweight='bold')
    axT.text(XA, y, f'{FINAL[k]/1000:,.0f}K', ha='center', va='center', fontsize=12, color=INK)
    axT.text(XE, y, f'{err:+.1f}%', ha='center', va='center', fontsize=12.5,
             color=INK, fontweight='bold')

errs = [100 * (AUTOK_S5[k] * FACTOR / FINAL[k] - 1) for k in ORDER]
axT.text(0, 0.50,
         f'네 설정 모두 {min(errs):+.0f}% ~ {max(errs):+.0f}% 안에서 맞았다. '
         f'λ 를 정하는 데는 이 정도면 충분하다 — λ 가 20~30% 틀려도 최종 희소도는 2%p 안에서 벗어난다.\n'
         '※ 0.33 은 규제 없는 baseline 궤적에서 얻은 값이고, 위 검산은 그와 다른 실행(auto-k)의 '
         'epoch 5 측정치로 한 것이다.',
         fontsize=10.5, color=INK2, va='top', linespacing=1.8)

fig.suptitle('스파이크 수는 학습 중 3배 넘게 변한다 — 그런데 왜 초반에 한 번만 재면 되는가',
             fontsize=17, color=INK, x=0.065, ha='left', y=0.955, fontweight='bold')
fig.text(0.065, 0.895,
         'λ 를 정하려면 “이 모델이 결국 스파이크를 몇 개 쏠지”를 알아야 하는데, 학습을 시작하는 시점엔 알 수 없다.',
         fontsize=11.5, color=MUTED, ha='left', va='top')

fig.savefig(OUT, dpi=160, facecolor=SURFACE)
print('saved:', OUT)
for k in ORDER:
    print(f'  {k:9s} ep5={AT5[k]:>10,.0f}  final={FINAL[k]:>10,.0f}  '
          f'final/ep5={RATIO[k]:.3f}   auto-k ep5={AUTOK_S5[k]:>10,}  '
          f'오차={100*(AUTOK_S5[k]*FACTOR/FINAL[k]-1):+.1f}%')
