"""10-20 조기 종료 그림 (검토 반영판). R19 CIFAR-10, T=4.

수치는 eip-stats 산출물만 읽는다 (원본 실험 CSV 를 읽지 않는다):
    analysis/early_exit_261020.json
      cpu.runs.<런>.trunc_200   -- 왼쪽 실선 (T' 로 다시 돌림, T=4 BN 통계 고정) 과 파선 (T=4 런의 스텝별 판독)
      recal.aug.runs.<런>.T     -- 왼쪽 점선 (BN 재추정, 앞 1000장)
      runs.<태그>.curve/npz_*   -- 오른쪽 조기 종료 곡선과 런별 T=4 값
      arms.<팔>.npz_*           -- 오른쪽 팔 평균 +- 시드 SD, real_max_saving_pct

규약: 실제 정확도(y) x 실제 스파이크 수(x). 차이(delta)로 그리지 않는다.
  왼쪽 y 는 0~103 전 범위. 오른쪽 y 는 96.0~97.0 만 보인다 (그림 안에 적는다).

쓰는 법:  ~/anaconda3/envs/venv_1/bin/python EIP_fig/an_261020/early_exit_fig.py
  -> EIP_fig/an_261020/early_exit_ko_26-10-20.png  (검토 전 판은 *_pre-review.png 로 보관)
"""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'EIP_fig', 'an_261020', 'early_exit_ko_26-10-20.png')
R = json.load(open(os.path.join(ROOT, 'analysis', 'early_exit_261020.json')))

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.edgecolor'] = '#555555'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

INK, INK2, GRID = '#222222', '#555555', '#e6e6e6'
COL = {'baseline': '#222222', 'ours_inter_ch': '#009E73'}       # an_261019_fig.py 와 같은 범주 색
MK = {'baseline': 'o', 'ours_inter_ch': 'D'}
KF = FuncFormatter(lambda v, _: f'{v / 1e3:.0f}K')

# 왼쪽에 그리는 CPU 체크포인트 (json 에 trunc_200 이 있는 것만; 없으면 건너뜀)
CPU_RUNS = ['base11', 'base12', 'base13', 'ours2', 'ours3', 'ours4']

fig = plt.figure(figsize=(11.0, 6.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.24, 0.26], width_ratios=[1.05, 1.0],
                      left=0.06, right=0.985, top=0.86, bottom=0.01, hspace=0.34, wspace=0.17)
ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
lg1, lg2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
nt1, nt2 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
for a in (lg1, lg2, nt1, nt2):
    a.axis('off')

# ------------------------------------------------------------------ 왼쪽
cpu = R['cpu']['runs']
rc = R['recal']['aug']['runs']
used = []
for nm in CPU_RUNS:
    if nm not in cpu or not cpu[nm]['trunc_200']:
        continue
    used.append(nm)
    t, arm = cpu[nm]['trunc_200'], cpu[nm]['arm']
    c = COL[arm]
    if nm in rc:                                              # BN 재추정 (앞 1000장): 점선 + 삼각형
        TT = rc[nm]['T']
        ks = [k for k in ('1', '2', '3', '4') if k in TT]
        ax1.plot([TT[k]['spk'] for k in ks], [TT[k]['acc'] for k in ks], ls=':', lw=1.6, color=c, zorder=4)
        ax1.scatter([TT[k]['spk'] for k in ks], [TT[k]['acc'] for k in ks], s=46, marker='^', color=c,
                    edgecolors='white', linewidths=0.8, zorder=6)
    ks = ('1', '2', '3', '4')
    xs_p = [t[k]['spk_prefix_T4'] for k in ks]
    ys_p = [t[k]['acc_readout_T4'] for k in ks]
    xs_r = [t[k]['spk_run'] for k in ks]
    ys_r = [t[k]['acc_run'] for k in ks]
    ax1.plot(xs_p, ys_p, ls=(0, (4, 3)), lw=1.2, color=c, alpha=0.75, zorder=2)
    ax1.scatter(xs_p, ys_p, s=34, marker=MK[arm], facecolors='white', edgecolors=c, linewidths=1.3, zorder=3)
    ax1.plot(xs_r, ys_r, ls='-', lw=1.8, color=c, zorder=4)
    ax1.scatter(xs_r, ys_r, s=36, marker=MK[arm], color=c, edgecolors='white', linewidths=0.8, zorder=5)
assert used == ['base11', 'base12', 'ours4'], used            # 그림 문구가 이 3개를 전제한다

# 지점 라벨 (T'=다시 돌린 스텝 수, 스텝 t = T=4 런의 t 번째 스텝). 선과 겹치지 않는 자리.
for txt, xy in (("T'=1", (24e3, 17)), ("T'=2", (153e3, 3.5)), ("T'=3", (325e3, 84))):
    ax1.text(*xy, txt, fontsize=8, color=INK2)
for txt, xy in (('스텝 1', (24e3, 40)), ('스텝 2', (126e3, 101)), ('스텝 3', (300e3, 101))):
    ax1.text(*xy, txt, fontsize=8, color=INK2)
b11 = cpu['base11']['trunc_200']['4']
ax1.text(b11['spk_run'] - 38e3, b11['acc_run'] - 9, "T'=4", fontsize=8, color=INK2)
ax1.set_xlim(0, 560e3)
ax1.set_ylim(0, 103)
ax1.set_xlabel('이미지당 스파이크 수')
ax1.set_ylabel('테스트 정확도 (%)')
ax1.set_title("재학습 없이 스텝 수를 줄여 다시 돌린 값 vs T=4 런 안의 스텝별 판독\n"
              "(T'=다시 돌린 스텝 수, T=4 가 원래 학습·평가 스텝 수)", fontsize=9.5, loc='left', color=INK)
ax1.xaxis.set_major_formatter(KF)
ax1.grid(True, color=GRID, lw=0.6)

# ------------------------------------------------------------------ 오른쪽
for tag, row in R['runs'].items():
    arm = row['arm']
    cur = sorted(row['curve'], key=lambda q: q['spk_real'])   # 임계값을 낮출수록 왼쪽 (스파이크 적음)
    ax2.plot([q['spk_real'] for q in cur], [q['acc'] for q in cur], lw=1.2, color=COL[arm], alpha=0.85, zorder=3)
    ax2.scatter([row['npz_spk']], [row['npz_acc']], s=40, marker=MK[arm], color=COL[arm],
                edgecolors='white', linewidths=0.8, zorder=5)
for arm in ('baseline', 'ours_inter_ch'):
    a = R['arms'][arm]
    ax2.errorbar([a['npz_spk']['mean']], [a['npz_acc']['mean']], xerr=[a['npz_spk']['sd']], yerr=[a['npz_acc']['sd']],
                 fmt=MK[arm], ms=9, mfc='white', mec=COL[arm], mew=1.6, ecolor=COL[arm], elinewidth=1.0,
                 capsize=2, zorder=6)
Y0, Y1 = 96.0, 97.0
ax2.set_xlim(330e3, 530e3)
ax2.set_ylim(Y0, Y1)
ax2.set_xlabel('이미지당 스파이크 수 (테스트 1만 장, 평가 모드)')
ax2.set_ylabel(f'테스트 정확도 (%)  [{Y0:.1f}~{Y1:.1f} 만 표시]')
ax2.xaxis.set_major_formatter(KF)
ax2.grid(True, color=GRID, lw=0.6)
sb, so = R['arms']['baseline']['real_max_saving_pct'], R['arms']['ours_inter_ch']['real_max_saving_pct']
n_b, n_o = R['arms']['baseline']['n'], R['arms']['ours_inter_ch']['n']
ax2.set_title("실제로 낼 수 있는 조기 종료 (몸통 4스텝은 다 돌고 출력만 일찍 읽음)\n"
              f"마지막 두 층의 뒤 스텝만 아낄 수 있음 (최대 baseline {sb['mean']:.2f}%, "
              f"ours_inter_ch {so['mean']:.2f}%)",
              fontsize=9.5, loc='left', color=INK)

# ------------------------------------------------------------------ 범례 (조건 표기) — 그림 아래
def proxy(**kw):
    return Line2D([], [], **kw)

c_leg = [proxy(color=COL['baseline'], marker='o', lw=0, ms=7, label='baseline'),
         proxy(color=COL['ours_inter_ch'], marker='D', lw=0, ms=7, label='ours_inter_ch 3e-3')]
l_c = lg1.legend(handles=c_leg, loc='upper left', ncol=2, frameon=False, fontsize=8.5,
                 bbox_to_anchor=(0.0, 1.02), handletextpad=0.3, columnspacing=1.6)
lg1.add_artist(l_c)
s_leg = [proxy(color=INK2, lw=1.8, marker='o', ms=5, label="실선 ●◆   T' 스텝만 다시 돌림, T=4 BN 통계 고정 (앞 200장)"),
         proxy(color=INK2, lw=1.6, ls=':', marker='^', ms=6, label="점선 ▲   T' 스텝만 다시 돌림, BN 재추정 (앞 1000장, T'=2~4)"),
         proxy(color=INK2, lw=1.2, ls=(0, (4, 3)), marker='o', ms=5, mfc='white',
               label="파선 ○   T=4 를 끝까지 돌린 뒤 스텝 t 출력을 읽음 (다시 돌린 값 아님, 앞 200장)")]
lg1.legend(handles=s_leg, loc='upper left', ncol=1, frameon=False, fontsize=8, labelspacing=0.45,
           bbox_to_anchor=(0.0, 0.76), handletextpad=0.6)

r_leg = [proxy(color=INK2, lw=1.2, label="선: 신뢰도 임계값을 훑은 조기 종료 (런마다 하나)"),
         proxy(color=INK2, marker='o', lw=0, ms=5, label=f"작은 점: 런의 T=4 값 (반복 횟수 {n_b} + {n_o})"),
         proxy(color=INK2, marker='o', lw=0, ms=8, mfc='white', mew=1.5,
               label=f"큰 속 빈 점: 팔 평균 ± 시드 SD (각 반복 횟수 {n_b}, {n_o})")]
lg2.legend(handles=r_leg, loc='upper left', ncol=1, frameon=False, fontsize=8, labelspacing=0.45,
           bbox_to_anchor=(0.0, 1.02), handletextpad=0.6)

# ------------------------------------------------------------------ 설명 (n, 자름) — 그림 아래
lo = min(q['acc'] for r in R['runs'].values() for q in r['curve'])
hi_lo = max(min(q['acc'] for q in r['curve']) for r in R['runs'].values())
nt1.text(0.0, 1.0,
         "런 3개 (baseline s11·s12, ours_inter_ch 3e-3 s4), 각 반복 횟수 1, 서버 138, CPU 로 직접 돌림. "
         "BN 재추정 = BN 이동통계만\n"
         "학습셋 5000장 (증강) 으로 다시 잡음. 런 3개 (반복 횟수 1) 라 추세나 팔 간 차이는 말할 수 없다.",
         transform=nt1.transAxes, ha='left', va='top', fontsize=7.5, color=INK2, linespacing=1.45)
nt2.text(0.0, 1.0,
         f"신뢰도 = 스텝 t 누적 출력의 최대 softmax 확률. 이미지마다 처음 임계값 이상인 스텝에서 종료.\n"
         f"세로축은 {Y0:.1f}~{Y1:.1f} 만 표시 (선은 임계값을 낮출수록 아래로 떨어져 틀 밖으로 나감, "
         f"곡선 최저 {lo:.0f}~{hi_lo:.0f}%).\n"
         f"절약 최대 = baseline {sb['mean']:.2f} ± {sb['sd']:.2f}%, "
         f"ours_inter_ch {so['mean']:.2f} ± {so['sd']:.2f}% (시드 SD).\n"
         f"서버 138, 18층 npz 스파이크.",
         transform=nt2.transAxes, ha='left', va='top', fontsize=7.5, color=INK2, linespacing=1.45)

fig.suptitle("R19 CIFAR-10 T=4: 시간 평균 BN 망이라 스텝 t 출력은 't 에서 멈춘 출력' 이 아니다 (26-10-20)",
             fontsize=10.5, x=0.01, y=0.985, ha='left', color=INK)
fig.savefig(OUT, dpi=200, facecolor='white')
print('[fig]', OUT)
