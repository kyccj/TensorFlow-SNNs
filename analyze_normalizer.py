"""
GPU 없이 하는 분석 두 가지.

B-1  lambda ∝ X^(-alpha) 에서 alpha 를 데이터로 맞춘다.
     auto-k 는 X=스파이크 수, alpha=1 을 가정하고 있다. 그게 맞는지 본다.

B-2  X 후보를 바꿔 본다 (스파이크 수 / 뉴런 수 / 파라미터 수).
     어느 축으로 나눠야 설정 간 차이가 가장 잘 사라지는지.

주의: 점이 4개(아키텍처 2 × 데이터셋 2)뿐이라 자유도가 매우 낮다.
"""

import math
import itertools

SET = ['VGG-C10', 'VGG-C100', 'R19-C10', 'R19-C100']

# 모델 규모 (학습 로그에서 추출)
NEURONS = {'VGG-C10': 280586, 'VGG-C100': 280676, 'R19-C10': 1445130, 'R19-C100': 1445220}
PARAMS = {'VGG-C10': 15255626, 'VGG-C100': 15301796, 'R19-C10': 12720010, 'R19-C100': 12743140}
SPIKES = {'VGG-C10': 78199, 'VGG-C100': 86274, 'R19-C10': 485715, 'R19-C100': 470387}
CLASSES = {'VGG-C10': 10, 'VGG-C100': 100, 'R19-C10': 10, 'R19-C100': 100}
RATE = {k: SPIKES[k] / NEURONS[k] for k in SET}

# (lambda, spike % of own baseline) — 6월 sweep 실측점
FR = {
    'VGG-C10':  [(1e-9, 95.7), (5e-8, 77.0), (1e-7, 68.0), (5e-7, 31.2), (1e-6, 18.9)],
    'VGG-C100': [(1e-9, 95.1), (5e-8, 89.9), (1e-7, 81.1), (5e-7, 55.0), (1e-6, 36.4)],
    'R19-C10':  [(1e-9, 97.7), (5e-8, 56.6), (1e-7, 41.6), (5e-7, 15.7), (1e-6, 7.1)],
    'R19-C100': [(1e-9, 100.9), (5e-8, 70.2), (1e-7, 55.8), (5e-7, 27.9), (1e-6, 20.0)],
}


def lam_at(k, target):
    pts = sorted(FR[k], key=lambda p: -p[1])
    for i in range(len(pts) - 1):
        (xa, ya), (xb, yb) = pts[i], pts[i + 1]
        if ya >= target >= yb:
            u = (ya - target) / (ya - yb)
            return math.exp(math.log(xa) + u * (math.log(xb) - math.log(xa)))
    return None


def fit_alpha(lams, X):
    """log lam = c - alpha*log X  를 최소제곱으로. (alpha, 잔차 산포배율) 반환."""
    xs = [math.log(X[k]) for k in SET]
    ys = [math.log(lams[k]) for k in SET]
    mx, my = sum(xs) / 4, sum(ys) / 4
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    alpha = -slope
    res = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    return alpha, math.exp(max(res) - min(res))


def spread_fixed(lams, X, alpha):
    v = [lams[k] * X[k] ** alpha for k in SET]
    return max(v) / min(v)


CAND = {'스파이크 수': SPIKES, '뉴런 수': NEURONS, '파라미터 수': PARAMS, '발화율(스파이크/뉴런)': RATE}
TARGETS = [80, 70, 60, 50]

print('=' * 78)
print('B-1  lambda ∝ X^(-alpha) 에서 alpha 맞추기   (X = 스파이크 수)')
print('=' * 78)
print(f"{'목표':>6s} {'lambda 산포':>11s} {'맞춘 alpha':>11s} {'그때 잔차':>10s} "
      f"{'alpha=1 고정':>13s}")
for t in TARGETS:
    lams = {k: lam_at(k, t) for k in SET}
    if any(v is None for v in lams.values()):
        continue
    a, r = fit_alpha(lams, SPIKES)
    raw = max(lams.values()) / min(lams.values())
    print(f"{t:5d}% {raw:10.1f}x {a:11.2f} {r:9.1f}x {spread_fixed(lams, SPIKES, 1.0):12.1f}x")

print()
print('=' * 78)
print('B-2  정규화 축 비교   (각 축마다 alpha 를 따로 맞춘 뒤 남는 산포)')
print('=' * 78)
print(f"{'축':>22s} " + ' '.join(f'{t:>4d}%' for t in TARGETS) + '   (맞춘 alpha)')
for name, X in CAND.items():
    cells, alphas = [], []
    for t in TARGETS:
        lams = {k: lam_at(k, t) for k in SET}
        if any(v is None for v in lams.values()):
            cells.append('  --'); continue
        a, r = fit_alpha(lams, X)
        cells.append(f'{r:4.1f}'); alphas.append(a)
    am = sum(alphas) / len(alphas) if alphas else float('nan')
    print(f'{name:>22s} ' + ' '.join(f'{c}x' for c in cells) + f'   (a={am:.2f})')

print()
print('=' * 78)
print('B-2b  남은 잔차가 클래스 수와 관계있나  (스파이크 수로 정규화한 뒤)')
print('=' * 78)
for t in TARGETS:
    lams = {k: lam_at(k, t) for k in SET}
    if any(v is None for v in lams.values()):
        continue
    K = {k: lams[k] * SPIKES[k] for k in SET}
    c10 = (K['VGG-C10'] * K['R19-C10']) ** 0.5
    c100 = (K['VGG-C100'] * K['R19-C100']) ** 0.5
    arch10 = max(K['VGG-C10'], K['R19-C10']) / min(K['VGG-C10'], K['R19-C10'])
    arch100 = max(K['VGG-C100'], K['R19-C100']) / min(K['VGG-C100'], K['R19-C100'])
    print(f'  목표 {t}%:  같은 데이터셋 안 아키텍처 차이 = ×{arch10:.2f} / ×{arch100:.2f}   '
          f'|  데이터셋 사이 = ×{c100/c10:.2f}')

print()
print('=' * 78)
print('B-2c  두 인자 모델  lambda ∝ 스파이크^(-a) × 클래스수^(-b)')
print('      (점 4개에 파라미터 3개 -> 자유도 1. 참고용일 뿐 검증 아님)')
print('=' * 78)
for t in TARGETS:
    lams = {k: lam_at(k, t) for k in SET}
    if any(v is None for v in lams.values()):
        continue
    # 정규방정식 (설계행렬 [1, -logS, -logC])
    import numpy as np
    A = np.array([[1.0, -math.log(SPIKES[k]), -math.log(CLASSES[k])] for k in SET])
    y = np.array([math.log(lams[k]) for k in SET])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    c, a, b = sol
    pred = A @ sol
    res = y - pred
    print(f'  목표 {t}%:  a={a:5.2f}  b={b:5.2f}   잔차 산포 ×{math.exp(max(res)-min(res)):.2f}')
