"""F1 -- 가중치 설계 (weighting design) 19종의 정확도-스파이크 산점도와 단일 적합 곡선.

A8(스펙 §2): "가중치 설계 19종이 같은 정확도-스파이크 곡선에 떨어진다"의 유일한 시각적
증거. 축은 실제 검증 정확도(%)와 실제 스파이크 수 -- 정규화·상대값 축을 쓰지 않는다
(사용자 지시, 2026-09-14).

**데이터 범위 -- VGG16/CIFAR10, T1·F2-F5(ResNet19/CIFAR10)와 다른 조합이다.** 이유:
설계 다양성(softmax/maxnorm/entropy/p-norm/그룹축/vmem 등)을 실제로 스윕한 것은
2026년 7-8월의 VGG16-CIFAR10 탐색 단계였다. ResNet19-CIFAR10 쪽에는 이 탐색이
`_r19_c10_maxnorm`(1-maxnorm)·`_sweep_wta_rev_r19_c10`(1-softmax)·`_entropy_sweep`
(entropy) 세 갈래뿐이라 19종을 채울 수 없다 (본 스크립트 실행 로그에 근거 기록).
서로 다른 아키텍처를 한 곡선에 섞지 않는다 -- VGG16 은 ~95%, ResNet19 는 ~97% 대가
정확도 상한이라 실제 값 축에서 두 아키텍처를 같은 곡선에 놓으면 A8 주장 자체가
성립하지 않는다. 캡션에 이 범위를 명시한다.

**데이터 위치 정정 (2026-09-14, task-13 실행 중 발견)** -- task-13-brief.md 는
"archive 에는 571개 중 15개만 train.log 를 가진다"(`find -maxdepth 2`)고 적었지만
그 find 는 `archive/<family>/<run>/train.log` 구조에서 `<run>` 한 단계를 못 본다.
`-maxdepth 3` 으로 다시 재면 489개다. 이 스크립트는 archive 를 directly 읽는다.

**대표점 선정 규칙 (그리기 전에 고정, docs/paper/figures-tables.md 에도 기록)**:
각 설계(design)마다, 사전 등록 선별 규칙(완주 + S30/S1 >= 0.19, 스펙 §12-1)을
통과한 런 중 **검증 정확도가 가장 높은 런**을 대표점으로 쓴다 -- 즉 그 설계가 낼 수
있는 가장 온건한(정확도를 가장 덜 깎는) 설정. 이 규칙은 설계마다 다른 판단을 넣지
않고 기계적으로 적용된다. 유효 런이 하나뿐이면 그것이 대표점이다. 유효 런이 아예
없는 설계(beta, 레거시 channel-wise, accum_loss)는 그림에서 빠진다 -- 아래 print
로그에 어떤 설계가 빠졌는지, 왜인지 남긴다.
"""
import os, sys, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

REPO = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
ARCH = '/media/hdd1/kyccj/EIP/archive'
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
import collect_paper as cp

# (표시 이름, 후보 런 디렉토리 목록, 설계 설명)
# 후보 여러 개는 같은 설계를 여러 λ 로 스윕한 것 -- 대표점 선정 규칙이 하나를 고른다.
DESIGNS = [
    ('baseline (규제 없음)',
     [f'{REPO}/_baseline_vgg_c10_trajectory'],
     '규제 없음 -- 곡선의 기준점 (가중 설계 축 자체는 아니지만 λ=0 끝점으로 포함)'),
    ('1-softmax (그룹 없음)',
     [f'{REPO}/_sweep_wta_rev/lambda_{v}' for v in
      ('1e-07', '1e-09', '1e-10', '5e-08', '1e-06', '5e-07', '1e-05', '5e-06')],
     'reg_spike_out_sc_sm, sm_group=none -- 논문 1-softmax 비교군과 같은 기본형'),
    ('1-maxnorm (그룹 없음)',
     [f'{ARCH}/_compare_maxnorm/maxnorm_lmb_{v}' for v in ('1e-07', '1e-08', '1e-06', '3e-07')],
     'reg_spike_out_sc_maxnorm, maxnorm_group=none'),
    ('1-maxnorm (없음) + encourage',
     [f'{ARCH}/_vgg_c10_maxnorm/maxnorm_enc_lmb_{v}' for v in ('1e-07', '1e-08', '1e-06', '3e-07')],
     '1-maxnorm 에 승자 발화 촉진 항(encourage) 추가'),
    ('1-softmax (없음) + encourage',
     [f'{ARCH}/_encourage/softmax_enc_lmb_{v}' for v in ('1e-07', '1e-08', '1e-06', '3e-07')],
     '1-softmax 에 encourage 항 추가'),
    ('1-maxnorm (채널 내)',
     [f'{ARCH}/_mxg/mx_wc_{v}' for v in ('1e-7', '3e-7', '5e-7')],
     'maxnorm_group=within_channel -- 채널마다 최고 발화 뉴런은 항상 보호'),
    ('1-maxnorm (채널 내) + vmem',
     [f'{ARCH}/_vmem/g{g}_5e-7' for g in ('1', '2', '4')],
     '문턱 근접도(vmem) 를 spike_count 에 더해 침묵 뉴런을 세분'),
    ('1-maxnorm (채널 내) + vmem + silent_only',
     [f'{ARCH}/_vmsil/g{g}' for g in ('100', '050')],
     'vmem 항을 침묵 뉴런에만 적용 -- 제안법이 최종적으로 쓰는 조합의 구성 요소'),
    ('p-norm (연속 축, p=4/8/16)',
     [f'{ARCH}/_pnorm/pn_p{v}' for v in ('4', '8', '16')],
     'sc_rate = 1 - sc/||sc||_p -- p->1 이 1-softmax, p->inf 가 1-maxnorm'),
    ('entropy WTA',
     [f'{ARCH}/_entropy_sweep/vgg_c10_lmb_{v}' for v in ('1e-07', '1e-08', '1e-06', '3e-07')],
     'sc_rate 를 -(1+log(p)) 엔트로피 기울기에서 뽑음'),
    ('plain softmax (반전 없음)',
     [f'{ARCH}/_softmax_cmp/smplain_lmb_{v}' for v in ('1e-6', '1e-7', '1e-8', '3e-7')],
     'sm_plain -- 1- 반전을 빼 많이 쏘는 뉴런이 더 세게 벌점받는 반대 방향'),
    ('constant coefficient',
     [f'{ARCH}/_sc_one/scone_lmb_{v}' for v in ('1e-7', '1e-8', '1e-6', '3e-7')],
     'sc_rate = 1 고정 -- 뉴런 간 차등이 전혀 없는 균일 가중치 대조군'),
    ('1-softmax (채널 내 그룹)',
     [f'{ARCH}/_grp/wc_a{v}' for v in ('1', '0.2', '0.5')],
     'sm_group=within_channel -- 채널마다 독립적으로 정규화'),
    ('layer_cost=synops',
     [f'{ARCH}/_synops/so_{v}' for v in ('3e-7', '5e-7')],
     '층별 벌점을 SynOps(다음 층 연산량)로 가중 -- 스파이크 수가 아니라 에너지 프록시를 최적화'),
]

# 유효 데이터가 아예 없어 그림에서 빠진 설계 (경과 기록용, 대표점 선정에 관여하지 않음)
DROPPED = {
    'shape_beta (β 스윕)': f'{ARCH}/_beta/* -- 6개 런 전부 S30/S1<0.19 로 배제, 유효 런 0',
    '레거시 channel-wise (row)': f'{ARCH}/_channel_wise_wta/* -- 3개 런 전부 0에폭(미완주), 유효 런 0',
    'accum_loss (norm-of-sum)': f'{ARCH}/_accum/* -- 4개 런 전부 S30/S1<0.19 로 배제, 유효 런 0',
    '1-softmax (row 그룹) + flat 대조': (
        f'{ARCH}/_grp_smoke/* -- S30/S1 은 통과하나 검증 정확도가 23.9~30.9%로 붕괴, '
        '정상 학습 궤적이 아니라 대표점으로 쓰지 않음 (flat 대조는 그룹 구조 자체를 고립시키는 '
        '진단용 대조군이지 시험 대상 설계가 아니다)'),
}


def representative(name, cands):
    """유효(완주+S30/S1>=0.19) 런 중 검증 정확도가 가장 높은 것을 고른다."""
    recs = [cp.collect(d) for d in cands if os.path.isdir(d)]
    valid = [r for r in recs if not r['exclude'] and r['best_val_acc'] != '']
    if not valid:
        return None
    best = max(valid, key=lambda r: r['best_val_acc'])
    return dict(name=name, acc=best['best_val_acc'], spikes=best['s_count'],
                run=best['run_dir'], n_valid=len(valid), n_total=len(recs))


rows = []
for name, cands, desc in DESIGNS:
    r = representative(name, cands)
    if r is None:
        print(f'[빠짐] {name}: 후보 {len(cands)}개 중 유효 런 0 -- {desc}')
        continue
    rows.append(r)
    print(f"{name:32s}  acc={r['acc']:6.2f}  spikes={r['spikes']:9.0f}  "
          f"(유효 {r['n_valid']}/{r['n_total']}, 대표 런 {os.path.basename(r['run'])})")

for name, why in DROPPED.items():
    print(f'[빠짐] {name}: {why}')

print(f'\nF1 에 실제로 찍히는 설계 수: {len(rows)} (baseline 포함)')

# ── 단일 로그-선형 곡선 적합 (설계 구분 없이 하나의 회귀선 -- A8 의 핵심 주장) ──
S = np.array([r['spikes'] for r in rows])
Acc = np.array([r['acc'] for r in rows])
X = np.log(S)
D = np.column_stack([np.ones_like(X), X])
b, *_ = np.linalg.lstsq(D, Acc, rcond=None)
pred = D @ b
resid = Acc - pred
ss_res = float(resid @ resid)
ss_tot = float(((Acc - Acc.mean()) ** 2).sum())
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
resid_sd = math.sqrt(ss_res / (len(Acc) - 2)) if len(Acc) > 2 else float('nan')
print(f"단일 곡선 적합: acc = {b[0]:.3f} + {b[1]:.4f}*ln(spikes),  R^2={r2:.3f},  "
      f"잔차 sd={resid_sd:.3f} (n={len(Acc)})")

# ── 그림 ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.6, 6.6))
markers = ['o', 's', '^', 'v', 'D', 'P', 'X', 'h', '<', '>', '8', 'p', 'd']
cmap = plt.get_cmap('tab20')
for i, r in enumerate(rows):
    is_base = r['name'].startswith('baseline')
    ax.scatter(r['spikes'] / 1e3, r['acc'],
               marker='*' if is_base else markers[i % len(markers)],
               s=190 if is_base else 85,
               color='#888888' if is_base else cmap(i / max(1, len(rows) - 1)),
               edgecolor='white', linewidth=0.7, zorder=3, label=r['name'])

xs = np.linspace(X.min() - 0.15, X.max() + 0.15, 100)
ax.plot(np.exp(xs) / 1e3, b[0] + b[1] * xs, color='#222222', lw=1.6, ls='--', zorder=2,
        label=f'단일 적합 곡선 (R²={r2:.2f})')

ax.set_xscale('log')
ax.set_xlabel('스파이크 수 (천, log축)')
ax.set_ylabel('검증 정확도 (%)')
ax.set_title('VGG16-CIFAR10 -- 가중치 설계 %d종이 같은 정확도-스파이크 곡선에 떨어진다'
              % (len(rows) - 1), fontsize=11.5)
ax.legend(fontsize=7.6, loc='upper center', bbox_to_anchor=(0.5, -0.13),
          ncol=2, framealpha=0.95)
ax.grid(alpha=0.25, which='both')
fig.tight_layout()

p_pdf = os.path.join('/home/kyccj/PycharmProjects/TensorFlow-SNNs/paper/figures', 'f1_pareto_curve.pdf')
os.makedirs(os.path.dirname(p_pdf), exist_ok=True)
fig.savefig(p_pdf, format='pdf', bbox_inches='tight')
p_png = os.path.join(OUT, 'pareto_curve_26-09-14.png')
fig.savefig(p_png, dpi=170, bbox_inches='tight')
print('저장:', p_pdf)
print('저장:', p_png)
