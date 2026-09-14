"""에폭에 따른 λ 와 정확도 — loss-ratio(제안법) vs 고정 λ.

제안법은 ρ=4e-3 를 주면 λ = ρ·L_task/R 로 매 에폭 다시 풀린다.
비교군은 같은 방법에서 loss-ratio 만 끄고 λ 를 고정한 abl_nolr.
ep200 에서 cutmix 가 꺼져(conf.mix_off_iter = 500*200) 과제 손실이 급락하고,
loss-ratio 가 거기에 반응해 λ 를 44% 낮춘다.
"""
import re, os, glob
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
P = '/home/kyccj/PycharmProjects/TensorFlow-SNNs/_paper'
OUT = os.path.dirname(os.path.abspath(__file__))
MIX_OFF = 200

def get(run, pat, n=310):
    f = os.path.join(P, run, 'train.log')
    t = open(f, errors='ignore').read().replace('\r', '\n')
    v = [float(x) for x in re.findall(pat, t)]
    return v[:n] if len(v) >= n else None

VA = r' - val_acc: ([0-9.eE+-]+)'
LA = r' - adp_lambda: ([0-9.eE+-]+)'

prop = [os.path.basename(d) for d in sorted(glob.glob(P + '/r19c10-prop-4e-3-s*'))]
pl = np.array([v for v in (get(r, LA) for r in prop) if v is not None])
pv = np.array([v for v in (get(r, VA) for r in prop) if v is not None]) * 100
FIX = {'3e-7': ('r19c10-abl_nolr-3e-7-s1', 3e-7, '#2471a3'),
       '5e-7': ('r19c10-abl_nolr-5e-7-s1', 5e-7, '#27ae60')}
fv = {k: (np.array(get(run, VA)) * 100, lam, c) for k, (run, lam, c) in FIX.items()
      if get(run, VA) is not None}
ep = np.arange(1, 311)
RED = '#c0392b'

fig, axes = plt.subplots(2, 1, figsize=(10, 8.4), sharex=True,
                         gridspec_kw=dict(hspace=0.16))

# ── (a) λ ────────────────────────────────────────────────
ax = axes[0]
ax.plot(ep, pl.mean(0), color=RED, lw=1.9, label=f'제안법 (loss-ratio, ρ=4e-3)  n={len(pl)}', zorder=4)
ax.fill_between(ep, pl.min(0), pl.max(0), color=RED, alpha=.17, lw=0, zorder=2)
for k, (_, lam, c) in sorted(FIX.items()):
    if k in fv:
        ax.axhline(lam, color=c, lw=1.6, ls='--', label=f'고정 λ = {k}', zorder=3)
ax.axhline(pl.mean(), color=RED, lw=1.0, ls=':', zorder=3)
ax.text(150, pl.mean(), f'시간평균 {pl.mean():.2e}', color=RED, fontsize=9,
        va='bottom', ha='center', bbox=dict(fc='white', ec='none', alpha=.75, pad=1.5))
ax.axvline(MIX_OFF, color='#555', lw=1.1, ls='-.', zorder=1)
ax.text(MIX_OFF + 4, ax.get_ylim()[1], ' cutmix off (ep200)', fontsize=9, color='#555',
        va='top', rotation=0)
ax.set_ylabel('λ (규제 계수)')
ax.set_title('(a) 에폭에 따른 λ — loss-ratio 는 과제 손실에 맞춰 λ 를 다시 푼다', fontsize=11.5)
ax.legend(fontsize=9, loc='lower center', ncol=3, framealpha=.95); ax.grid(alpha=.25)

# ── (b) val_acc ──────────────────────────────────────────
ax = axes[1]
ax.plot(ep, pv.mean(0), color=RED, lw=1.9, label=f'제안법 (ρ=4e-3)  n={len(pv)}', zorder=4)
ax.fill_between(ep, pv.min(0), pv.max(0), color=RED, alpha=.17, lw=0, zorder=2)
for k, (v, lam, c) in sorted(fv.items()):
    ax.plot(ep, v, color=c, lw=1.5, ls='--', label=f'고정 λ = {k}  n=1', zorder=3)
ax.axvline(MIX_OFF, color='#555', lw=1.1, ls='-.', zorder=1)
ax.set_xlabel('에폭'); ax.set_ylabel('검증 정확도 (%)')
ax.set_xlim(1, 310); ax.set_ylim(20, 99.5)
ax.set_title('(b) 에폭에 따른 검증 정확도', fontsize=11.5)
ax.legend(fontsize=9, loc='upper left', framealpha=.95); ax.grid(alpha=.25)

# 후반 확대
ins = ax.inset_axes([0.55, 0.09, 0.42, 0.46])
ins.plot(ep, pv.mean(0), color=RED, lw=1.6)
ins.fill_between(ep, pv.min(0), pv.max(0), color=RED, alpha=.17, lw=0)
for k, (v, lam, c) in sorted(fv.items()):
    ins.plot(ep, v, color=c, lw=1.3, ls='--')
ins.axvline(MIX_OFF, color='#555', lw=1.0, ls='-.')
ins.set_xlim(190, 310); ins.set_ylim(92.5, 97.2)
ins.tick_params(labelsize=7.5); ins.grid(alpha=.3)
ins.set_title('ep190–310 확대', fontsize=8.5)
ax.indicate_inset_zoom(ins, edgecolor='#999')

fig.suptitle('R19-CIFAR10 — 적응 λ 와 고정 λ', fontsize=13.5, y=.955)
p = os.path.join(OUT, 'lambda_epoch_26-09-14.png')
fig.savefig(p, dpi=170, bbox_inches='tight')
print('저장:', p)

# F6 -- 부록, §9-B 게이트 대기 (ep201 급락과 LR 스케줄의 겹침 미확인).
# 게이트가 열리기 전에도 PDF 는 만들어 둔다 -- \input 자체를 빼는 것으로 게이트를 지킨다
# (paper/sections/experiments.tex 참조).
PAPER_FIG = '/home/kyccj/PycharmProjects/TensorFlow-SNNs/paper/figures'
os.makedirs(PAPER_FIG, exist_ok=True)
p6 = os.path.join(PAPER_FIG, 'f6_lambda_epoch_appendix.pdf')
fig.savefig(p6, format='pdf', bbox_inches='tight')
print('저장:', p6)
print(f"λ: ep1 {pl.mean(0)[0]:.2e}  ep100 {pl.mean(0)[99]:.2e}  ep200 {pl.mean(0)[199]:.2e}  "
      f"ep201 {pl.mean(0)[200]:.2e}  ep310 {pl.mean(0)[309]:.2e}  시간평균 {pl.mean():.2e}")
print(f"최종 val_acc: 제안법 {pv.mean(0)[309]:.2f} / " +
      " / ".join(f"고정 {k} {v[309]:.2f}" for k, (v, _, _) in sorted(fv.items())))
