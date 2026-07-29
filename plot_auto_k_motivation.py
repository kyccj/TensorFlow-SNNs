"""
Figure: auto-k의 동기 — 처음 보는 사람도 읽을 수 있게.

왼쪽  : 모델마다 스파이크 수가 6배 다르다 (막대)
오른쪽: 필요한 lambda 에 그 스파이크 수를 곱해 보면 같은 값이 나온다 (실제 숫자 그대로)

통계 용어("산포", "정규화") 대신 곱셈식을 그대로 보여준다.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'EIP_figure', 'auto_k_motivation.png')

C = {'VGG-C10': '#2a78d6', 'VGG-C100': '#eb6834',
     'R19-C10': '#1baf7a', 'R19-C100': '#eda100'}
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8983', '#e3e2dd'
SURFACE = '#fcfcfb'
HILITE = '#fdf3e3'   # 곱셈 결과 열 배경

ORDER = ['VGG-C10', 'VGG-C100', 'R19-C10', 'R19-C100']
SHOW = {'VGG-C10': 'VGG', 'R19-C10': 'ResNet19', 'VGG-C100': 'VGG', 'R19-C100': 'ResNet19'}
BASE_SPIKE = {'VGG-C10': 78199, 'VGG-C100': 86274, 'R19-C10': 485715, 'R19-C100': 470387}
LAM = {'VGG-C10': 8.572e-8, 'VGG-C100': 1.983e-7, 'R19-C10': 1.397e-8, 'R19-C100': 5.048e-8}
K = {k: LAM[k] * BASE_SPIKE[k] for k in ORDER}

fig = plt.figure(figsize=(15, 7.4), facecolor=SURFACE)
ax1 = fig.add_axes([0.068, 0.16, 0.30, 0.52])
ax2 = fig.add_axes([0.435, 0.05, 0.555, 0.70])

# ============================================================ 왼쪽: 스파이크 수
ax1.set_facecolor(SURFACE)
for s in ('top', 'right'):
    ax1.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax1.spines[s].set_color(GRID)
ax1.tick_params(colors=INK2, labelsize=10, length=3, color=GRID)
ax1.grid(True, axis='y', color=GRID, lw=0.8)
ax1.set_axisbelow(True)

vals = [BASE_SPIKE[k] / 1000 for k in ORDER]
ax1.bar(range(4), vals, color=[C[k] for k in ORDER], width=0.6,
        edgecolor=SURFACE, linewidth=2)
for i, v in enumerate(vals):
    ax1.text(i, v + 12, f'{v:,.0f}K', ha='center', va='bottom',
             fontsize=12, color=INK, fontweight='bold')
ax1.set_xticks(range(4))
ax1.set_xticklabels(['VGG\nC10', 'VGG\nC100', 'ResNet19\nC10', 'ResNet19\nC100'],
                    fontsize=10, color=INK2, linespacing=1.5)
ax1.set_ylabel('학습된 모델이 뿜는 스파이크 수', fontsize=11, color=INK2)
ax1.set_ylim(0, 640)
ax1.annotate('', xy=(2.0, 558), xytext=(0.0, 558),
             arrowprops=dict(arrowstyle='<->', color=MUTED, lw=1.4))
ax1.text(1.0, 572, '6배', ha='center', fontsize=15, color=INK, fontweight='bold')
ax1.set_title('① ResNet19는 VGG보다 스파이크를 6배 많이 쏜다',
              fontsize=13, color=INK, loc='left', pad=14)
ax1.text(0.5, -0.28, '데이터셋(C10↔C100)은 거의 영향이 없다.',
         transform=ax1.transAxes, ha='center', fontsize=10.5, color=MUTED)

# ============================================================ 오른쪽: 곱셈식
ax2.set_facecolor(SURFACE)
ax2.set_xlim(0, 11.5)
ax2.set_ylim(0, 10)
ax2.axis('off')

XN, XL, XM, XS, XE, XK = 0.15, 3.1, 4.15, 5.6, 6.9, 8.3
ROWS = {'VGG-C10': 7.35, 'R19-C10': 6.35, 'VGG-C100': 3.55, 'R19-C100': 2.55}

ax2.text(0, 9.5, '② 그 스파이크 수를 λ에 곱해 보면 — 같은 값이 나온다',
         fontsize=13, color=INK, fontweight='bold')
ax2.text(0, 8.9, '아래는 모두 “스파이크를 30% 줄인다”는 똑같은 목표를 맞췄을 때의 값이다.',
         fontsize=10.5, color=MUTED)

# 열 제목
ax2.text(XL, 8.25, '필요한 λ', ha='center', fontsize=11.5, color=INK2)
ax2.text(XS, 8.25, '스파이크 수', ha='center', fontsize=11.5, color=INK2)
ax2.text(XK, 8.25, 'λ × 스파이크 수', ha='center', fontsize=11.5, color=INK, fontweight='bold')

# 결과 열 강조 배경
ax2.add_patch(plt.Rectangle((XK - 1.05, 1.95), 2.1, 6.6, facecolor=HILITE,
                            edgecolor='none', zorder=0))

for grp, ys, title in [(['VGG-C10', 'R19-C10'], 7.95, 'CIFAR-10'),
                       (['VGG-C100', 'R19-C100'], 4.15, 'CIFAR-100')]:
    ax2.text(XN, ys, title, fontsize=11.5, color=INK, fontweight='bold')
    for k in grp:
        y = ROWS[k]
        ax2.plot([XN + 0.12], [y], 'o', color=C[k], ms=11, zorder=3)
        ax2.text(XN + 0.45, y, SHOW[k], ha='left', va='center', fontsize=12, color=INK)
        ax2.text(XL, y, f'{LAM[k]*1e7:.2f} × 10$^{{-7}}$', ha='center', va='center',
                 fontsize=12.5, color=INK)
        ax2.text(XM, y, '×', ha='center', va='center', fontsize=12, color=MUTED)
        ax2.text(XS, y, f'{BASE_SPIKE[k]/1000:,.0f}K', ha='center', va='center',
                 fontsize=12.5, color=INK)
        ax2.text(XE, y, '=', ha='center', va='center', fontsize=12, color=MUTED)
        ax2.text(XK, y, f'{K[k]:.4f}', ha='center', va='center',
                 fontsize=14, color=INK, fontweight='bold')

    # 왼쪽: lambda 가 몇 배 다른지
    a, b = LAM[grp[0]], LAM[grp[1]]
    y0, y1 = ROWS[grp[0]], ROWS[grp[1]]
    ax2.annotate('', xy=(XL - 0.92, y1), xytext=(XL - 0.92, y0),
                 arrowprops=dict(arrowstyle='<->', color=INK, lw=1.6))
    ax2.text(XL - 1.02, (y0 + y1) / 2, f'{max(a,b)/min(a,b):.0f}배\n차이',
             ha='right', va='center', fontsize=12, color=INK,
             fontweight='bold', linespacing=1.4)

    # 오른쪽: 결과가 얼마나 같은지
    r = max(K[grp[0]], K[grp[1]]) / min(K[grp[0]], K[grp[1]])
    msg = '거의 같다' if r < 1.1 else f'{r:.1f}배로\n가까워짐'
    ax2.annotate('', xy=(XK + 1.18, y1), xytext=(XK + 1.18, y0),
                 arrowprops=dict(arrowstyle='-', color=INK, lw=1.6))
    ax2.text(XK + 1.3, (y0 + y1) / 2, msg, ha='left', va='center',
             fontsize=12.5, color=INK, fontweight='bold', linespacing=1.4)

# 두 데이터셋 사이에 남는 차이
ax2.plot([XN, 10.9], [5.35, 5.35], color=GRID, lw=1.2)
km10 = (K['VGG-C10'] * K['R19-C10']) ** 0.5
km100 = (K['VGG-C100'] * K['R19-C100']) ** 0.5
ax2.text(XN, 1.35,
         f'다만 데이터셋 사이에는 차이가 남는다 — CIFAR-10은 약 {km10:.3f}, '
         f'CIFAR-100은 약 {km100:.3f}으로 {km100/km10:.0f}배.\n'
         f'즉 모델이 바뀌어도 이 값은 그대로 쓸 수 있지만, 데이터셋이 바뀌면 다시 정해야 한다.',
         fontsize=11, color=INK2, va='top', linespacing=1.7)

ax2.text(XN, 0.15,
         '※ 각 값은 1회 실행에서 얻은 것이고 스파이크 수에 5% 정도의 실행 편차가 있다. '
         'ResNet19-C10의 λ는 측정한 두 점 사이를\n   50배 건너뛰어 추정한 값이라 넷 중 가장 불확실하다.',
         fontsize=9.5, color=MUTED, va='top', linespacing=1.6)

fig.suptitle('λ가 모델마다 달라 보였던 이유:  λ 안에 그 모델의 스파이크 수가 숨어 있었다',
             fontsize=17, color=INK, x=0.055, ha='left', y=0.945, fontweight='bold')
fig.text(0.055, 0.885,
         '규제 강도 λ는 “스파이크를 얼마나 억누를지” 정하는 값이다. '
         '같은 목표를 맞추는 데 모델마다 다른 λ가 필요해서 매번 찾아야 했다.',
         fontsize=11, color=MUTED, ha='left', va='top')

fig.savefig(OUT, dpi=160, facecolor=SURFACE)
print('saved:', OUT)
for k in ORDER:
    print(f'  {SHOW[k]:9s} {k:9s} lambda={LAM[k]:.3e}  spike={BASE_SPIKE[k]:,}  K={K[k]:.4f}')
