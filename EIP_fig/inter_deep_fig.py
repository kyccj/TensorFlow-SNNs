"""inter 심층 비교 그림 3개 (R19-C10, 26-10-05). 수치 출처: analysis/inter_deep_261005.json 만 읽는다.

1. inter_deep_curves  : 정확도 x 스파이크 + 단조 곡선 + inter 와 교차점(중앙값, 95% CI)
2. inter_deep_forest  : 같은 스파이크에서 inter - 다른 팔 (pooled 95% CI)
3. inter_deep_resid   : 정확도 잔차 vs dead_ch / alive_ch_gini (250-420K, 팔 색)
4. inter_deep_macs    : 같은 스파이크에서 죽은 채널 제거로 줄어드는 MACs % (F 절)

쓰는 법:  python EIP_fig/inter_deep_fig.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__)); VEC = os.path.join(HERE, 'vector'); STAMP = '26-10-05'
J = json.load(open(os.path.join(HERE, '..', 'analysis', 'inter_deep_261005.json')))
plt.rcParams['font.family'] = 'Noto Sans CJK JP'; plt.rcParams['axes.unicode_minus'] = False; plt.rcParams['font.size'] = 9

ARMS = {  # json 팔 이름: (색, 마커, 선)
    'plain L2':           ('#7F7F7F', 'd', (0, (1, 1.4))),
    'BPSR':               ('#CC79A7', 'v', (0, (3, 1.5))),
    'ours_intra_ch':      ('#0072B2', 'o', '-'),
    'ours_intra_ch (w¹)': ('#56B4E9', 'o', (0, (5, 2))),
    'ours_layer (w¹)':    ('#D55E00', 's', '--'),
    'ours_inter_ch':      ('#009E73', '^', '-'),
}
OTHERS = ['ours_intra_ch', 'ours_intra_ch (w¹)', 'ours_layer (w¹)', 'plain L2', 'BPSR']

def save(fig, name):
    os.makedirs(VEC, exist_ok=True)
    for ext in ('png', 'pdf', 'svg'):
        p = os.path.join(HERE if ext == 'png' else VEC, f'{name}_ko_{STAMP}.{ext}')
        fig.savefig(p, dpi=220 if ext == 'png' else None, bbox_inches='tight'); print('wrote', p)
    plt.close(fig)

def caption(fig, text, bottom=0.09):
    fig.text(0.012, 0.012, text, fontsize=6.4, va='bottom', linespacing=1.45)

cc = J['A']['curves_C10']; sd = J['seed_sd']

# ---------------------------------------------------------------- 1
fig, ax = plt.subplots(figsize=(8.2, 5.4))
bacc, bsp, bsd, bn = sd['C10_base_mean'], sd['C10_base_spikes'], sd['C10_base'], sd['C10_base_n']
ax.axhspan(bacc - bsd, bacc + bsd, color='0.85', alpha=.7, lw=0, zorder=0)
ax.axhline(bacc, color='0.45', ls=':', lw=1, zorder=1)
ax.plot([bsp], [bacc], marker='*', ms=14, color='0.15', lw=0, zorder=7, label=f'baseline (n={bn}) ±1 SD')
for a in OTHERS:  # 교차점 띠: inter 와 해당 팔의 곡선이 만나는 스파이크
    cr = cc['crossing'][a]; col = ARMS[a][0]
    weak = cr['est'] is None
    ax.axvspan(cr['lo'], cr['hi'], color=col, alpha=.07 if weak else .12, lw=0, zorder=0,
               hatch='///' if weak else None, ec=col if weak else None)
    ax.axvline(cr['median'], color=col, lw=1.0, ls=(0, (2, 2)), alpha=.9, zorder=1)
for a in ['plain L2', 'BPSR', 'ours_intra_ch', 'ours_intra_ch (w¹)', 'ours_layer (w¹)', 'ours_inter_ch']:
    col, mk, ls = ARMS[a]; c = cc['curves'][a]; main = a == 'ours_inter_ch'
    ax.plot(c['iso_spikes'], c['iso_acc'], ls=ls, lw=1.6 if main else 1.2, color=col, zorder=3 if main else 2)
    pts = c['points']
    n1 = [p for p in pts if p['n'] < 2]; nm = [p for p in pts if p['n'] >= 2]
    kw = dict(marker=mk, ms=8 if main else 6.5, lw=0, color=col, mew=1.4, zorder=5 if main else 4)
    if nm: ax.plot([p['spikes'] for p in nm], [p['acc'] for p in nm], mfc=col, **kw)
    if n1: ax.plot([p['spikes'] for p in n1], [p['acc'] for p in n1], mfc='white', **kw)
    ax.plot([], [], ls=ls, marker=mk, color=col, mfc=col, lw=1.4, ms=6,
            label=f"{a} (점 {len(pts)}개 중 n=1 인 점 {len(n1)}개)")
ax.plot([], [], marker='o', lw=0, color='0.35', mfc='white', mew=1.4, label='속 빈 마커 = n=1 (ρ 점 평균, 그 외는 n≥2)')
ax.fill_between([], [], [], color='0.6', alpha=.25, label='세로 띠 = inter 와의 교차점 95% CI, 점선 = 중앙값 (팔 색)')
ax.set_xscale('log'); ax.set_xlim(1.2e5, 5.4e5); ax.set_ylim(95.7, 97.0)
ax.set_xticks([1.5e5, 2e5, 3e5, 4e5, 5e5]); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{int(v):,}'))
ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ''))
ax.grid(alpha=.22, lw=.5); ax.set_xlabel('스파이크 수 (test, 샘플당, 로그축)'); ax.set_ylabel('val accuracy (%)')
ax.set_title('ResNet19 · CIFAR-10 · T=4 · 단조 곡선과 inter 교차점', fontsize=10)
ax.legend(loc='lower right', fontsize=6.8, framealpha=.95)
caption(fig, '선 = 팔별 ρ 점 평균을 단조(isotonic) 보정한 곡선. 점 = ρ 별 시드 평균, 속 빈 마커 = n=1. 회색 띠 = baseline ±1 SD (n=6).\n'
        '빗금 띠 = ours_intra_ch: 곡선이 겹치는 구간이 짧아 부트스트랩 중 교차 비율 16%뿐 (점추정 없음), 참고용. inter 의 5e-3 이하 점은 전부 n=1.\n'
        '출처: inter_deep_261005.json  A.curves_C10.{curves.<팔>.points/iso_spikes/iso_acc, crossing.<팔>.{median,lo,hi}}, seed_sd.C10_base*.')
fig.tight_layout(rect=(0, 0.085, 1, 1)); save(fig, 'inter_deep_curves')

# ---------------------------------------------------------------- 2
fig, ax = plt.subplots(figsize=(8.2, 5.2))
levels = ['340000', '300000', '240000', '200000']
w = 0.14; y0 = np.arange(len(levels))[::-1] * 1.0
for gi, lv in enumerate(levels):
    dm = cc['at_spikes'][lv]['diff_inter_minus']
    for ai, a in enumerate(OTHERS):
        p = dm[a]['pooled']; col, mk, _ = ARMS[a]
        y = y0[gi] + (2 - ai) * w
        if p is None:
            ax.text(0.02, y, '데이터 없음', fontsize=6.5, va='center', color=col, alpha=.8); continue
        ax.errorbar([p['est']], [y], xerr=[[p['est'] - p['lo']], [p['hi'] - p['est']]], fmt=mk, color=col,
                    ms=6, capsize=2.5, elinewidth=1.3, mfc=col, zorder=3,
                    )
    if lv in ('240000', '200000'):
        ax.text(0.99, y0[gi] + 0.42, 'inter 는 n=1 점 사이 보간', transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=7, style='italic', color='0.3')
ax.axvline(0, color='k', lw=1)
for g in range(len(levels) - 1): ax.axhline(y0[g] - 0.5, color='0.85', lw=.6)
ax.set_yticks(y0); ax.set_yticklabels([f'{int(l)//1000}K' for l in levels])
ax.set_ylim(-0.6, len(levels) - 0.35); ax.set_xlim(-0.45, 0.35)
ax.set_xlabel('정확도 차이: inter − 다른 팔 (%p, 같은 스파이크에서; 0 보다 크면 inter 가 높음)')
ax.set_ylabel('기준 스파이크 수 (test, 샘플당)')
ax.set_title('ResNet19 · CIFAR-10 · 같은 스파이크에서 ours_inter_ch − 다른 팔', fontsize=10)
ax.grid(axis='x', alpha=.22, lw=.5)
from matplotlib.lines import Line2D
hs = [Line2D([], [], marker=ARMS[a][1], color=ARMS[a][0], lw=1.3, ms=6) for a in OTHERS]
ax.legend(hs, OTHERS, title='비교 상대 (위에서 아래 순서)', loc='upper right', bbox_to_anchor=(1.0, 0.99), fontsize=7, title_fontsize=7.5, framealpha=.95)
caption(fig, '점 = pooled 부트스트랩 추정, 막대 = 95% CI (시드 산포 포함). 차이(delta) 그림이므로 위의 정확도×스파이크 그림과 함께 읽을 것.\n'
        '340K·300K 에는 ours_intra_ch 곡선이 닿지 않아 비교 없음. 240K·200K 는 inter 가 n=1 점뿐인 구간의 보간이라 CI 가 시드 재현을 보장하지 않음.\n'
        '출처: inter_deep_261005.json  A.curves_C10.at_spikes.<340000|300000|240000|200000>.diff_inter_minus.<팔>.pooled.{est,lo,hi}.', 0)
fig.tight_layout(rect=(0, 0.085, 1, 1)); save(fig, 'inter_deep_forest')

# ---------------------------------------------------------------- 3
C = J['C']['250-420K']; rows = C['rows']; M = C['metrics']
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
for ax, key, xl in [(axes[0], 'dead_ch', '죽은 채널 수 (dead_ch, 총 4,736 중)'),
                    (axes[1], 'alive_ch_gini', '산 채널 안 지니 (alive_ch_gini)')]:
    for a, (col, mk, _) in ARMS.items():
        rs = [r for r in rows if r['arm'] == a]
        if not rs: continue
        ax.plot([r[key] for r in rs], [r['resid'] for r in rs], marker=mk, lw=0, ms=6.5, color=col, mfc=col, mew=1.2,
                alpha=.85, label=f'{a} (n={len(rs)})')
    ax.axhline(0, color='0.4', lw=.8); ax.set_ylim(-0.40, 0.27)
    m = M[key]; pv = m['p_within_arm']; ptxt = '< 0.001' if pv < 0.001 else f'= {pv:.3f}'
    ax.text(0.03, 0.04, f"팔 안 부분상관 (ln 스파이크·팔 통제)\nr = {m['partial_r_within_arm']:.2f}, p {ptxt} (df={m['df_within']})",
            transform=ax.transAxes, fontsize=8, va='bottom', bbox=dict(fc='white', ec='0.6', alpha=.9))
    ax.set_xlabel(xl); ax.set_ylabel('정확도 잔차 (%p)'); ax.grid(alpha=.22, lw=.5)
axes[0].legend(fontsize=7, loc='upper right', framealpha=.95)
fig.suptitle(f"ResNet19 · CIFAR-10 · 250-420K 스파이크 · 정확도 잔차 vs 분포 지표 (런 n={C['n']}, 점 하나 = 런 하나)", fontsize=10)
caption(fig, '잔차 = val acc − (공통 곡선 93.26 + 0.267·ln 스파이크, 팔 항 없음). 점은 모두 개별 런 (n=1). 관찰 상관이며 인과 아님: 팔 안 상관은 시드 간 차이에서 나옴.\n'
        'ours_intra_ch (w²) 는 npz 가 없어 이 대역에 없음. 출처: inter_deep_261005.json  C.250-420K.rows[].{arm,resid,dead_ch,alive_ch_gini}, C.250-420K.metrics.<지표>.{partial_r_within_arm,p_within_arm,df_within}.')
fig.tight_layout(rect=(0, 0.08, 1, 0.95)); save(fig, 'inter_deep_resid')

# ---------------------------------------------------------------- 4  (F 절만 사용)
F = J['F']
ARMS4 = dict(ARMS); ARMS4['ours_layer (w²)'] = ('#D55E00', 'D', (0, (5, 2)))
groups = {}  # (arm, knob) -> dict(spk, y, sd, n, src)
byg = {}
for r in F['runs']: byg.setdefault((r['arm'], r['knob']), []).append(r)
for (a, k), rs in byg.items():
    y = np.array([r['b']['macs_pct'] for r in rs])
    groups[(a, k)] = dict(spk=np.mean([r['spikes'] for r in rs]), y=y.mean(), sd=y.std(ddof=1) if len(y) > 1 else None, n=len(rs), src='meas')
for key, v in F['offline_npz'].items():
    if key == 'baseline': continue
    lv, rest = key.split('|'); a, k = rest.rsplit(' ', 1)
    if (a, k) in groups: continue  # 실측과 같은 (팔, ρ) 는 실측으로 표시
    groups[(a, k)] = dict(spk=v['spikes'], y=v['macs_pct_b'][0], sd=v['macs_pct_b'][1] if v['n'] > 1 else None, n=v['n'], src='npz')
fig, ax = plt.subplots(figsize=(8.4, 5.6))
bm, bsd_ = F['offline_npz']['baseline']['macs_pct_b']; bn4 = F['offline_npz']['baseline']['n']
ax.axhline(bm, color='0.35', ls=':', lw=1.2, zorder=1)
ax.text(1.22e5, bm + 0.5, f'baseline: 죽은 채널 자체 {bm:.1f}% (n={bn4})', fontsize=7.5, color='0.25', va='bottom')
for a in ['plain L2', 'BPSR', 'ours_intra_ch', 'ours_intra_ch (w¹)', 'ours_layer (w¹)', 'ours_layer (w²)', 'ours_inter_ch']:
    col, mk, ls = ARMS4[a]; main = a == 'ours_inter_ch'
    gs = [g for (aa, _k), g in groups.items() if aa == a]
    if not gs: continue
    for g in gs:
        kw = dict(marker=mk, ms=8 if main else 6.5, color=col, mew=1.5, zorder=5 if main else 4)
        face = col if g['n'] >= 2 else 'white'
        edge = 'k' if g['src'] == 'meas' else col
        if g['sd'] is not None: ax.errorbar([g['spk']], [g['y']], yerr=[g['sd']], fmt='none', ecolor=col, elinewidth=1.1, capsize=2, zorder=3)
        ax.plot([g['spk']], [g['y']], lw=0, mfc=face, mec=edge, **kw)
        ax.annotate(f"n={g['n']}", (g['spk'], g['y']), textcoords='offset points', xytext=(5, 5), fontsize=5.8, color='0.3')
    if len(gs) >= 3:  # ln 스파이크 회귀 (ρ 묶음 평균 위, 점 3개 이상일 때만)
        x = np.log([g['spk'] for g in gs]); y = np.array([g['y'] for g in gs]); b1, b0 = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 20); ax.plot(np.exp(xx), b0 + b1 * xx, color=col, ls=ls, lw=.9, alpha=.8, zorder=2)
    ngr = len(gs)
    ax.plot([], [], ls=ls if ngr >= 3 else 'none', marker=mk, color=col, mfc=col, lw=1, ms=6, label=f'{a} (ρ 묶음 {ngr}개' + ('' if ngr >= 3 else ', 회귀선 없음') + ')')
ax.plot([], [], marker='o', lw=0, color='0.5', mfc='0.5', mec='k', mew=1.5, ms=6, label='검은 테두리 = 실측 (F.runs, 런 단위)')
ax.plot([], [], marker='o', lw=0, color='0.5', mfc='0.5', mec='0.5', mew=1.5, ms=6, label='테두리 팔 색 = 오프라인 npz 계산 (F.offline_npz)')
ax.plot([], [], marker='o', lw=0, color='0.4', mfc='white', mew=1.5, ms=6, label='속 빈 마커 = n=1,  막대 = ±1 SD')
ax.set_xscale('log'); ax.set_xlim(1.2e5, 4.6e5); ax.set_ylim(8, 46)
ax.set_xticks([1.5e5, 2e5, 3e5, 4e5]); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f'{int(v):,}'))
ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _p: ''))
ax.grid(alpha=.22, lw=.5); ax.set_xlabel('스파이크 수 (test, 샘플당, 로그축)')
ax.set_ylabel('죽은 채널 제거로 줄어드는 MACs (%)  [시나리오 (b): 모든 스파이킹 층]')
ax.set_title('ResNet19 · CIFAR-10 · T=4 · 같은 스파이크에서 제거 가능한 MACs', fontsize=10)
ax.legend(loc='upper right', fontsize=6.5, framealpha=.95)
P = F['pooled_macs']['300-420K']; m = P['macs_pct_b']; o = m['others_minus_inter']; at = m['at']['340K']
M = F['measured_ancova']['b']['others_minus_inter']; pp = '< 0.001' if o['p'] < .001 else f"= {o['p']:.3f}"
box = (f"스파이크 보정 회귀 (ln 스파이크, 300~420K, 런 n={P['n']}, inter {P['n_inter']})\n"
       f"inter − 다른 팔: {-o['est']:+.1f}%p [{-o['hi']:+.1f}, {-o['lo']:+.1f}], p {pp}\n"
       f"340K 에서: {-at['est']:+.1f}%p [{-at['hi']:+.1f}, {-at['lo']:+.1f}]\n"
       f"실측만 (n={F['measured_ancova']['b']['n']}, 스파이크 범위 불일치): {-M['est']:+.1f}%p [{-M['hi']:+.1f}, {-M['lo']:+.1f}], p = {M['p']:.2f}")
ax.text(0.985, 0.03, box, transform=ax.transAxes, fontsize=6.8, ha='right', va='bottom', bbox=dict(fc='white', ec='0.6', alpha=.95))
caption(fig, '제거한 채널은 테스트에서 한 번도 쏘지 않아 정확도 변화는 전 런 0.00 (실측 28런) — 팔 차이는 MACs 에서만 보인다. MACs 는 타임스텝 1회분 밀집 연산 기준.\n'
        '점 = ρ 묶음 평균, 얇은 선 = 팔별 ln 스파이크 회귀 (묶음 3개 이상인 팔만). 오프라인 npz 와 실측은 같은 (팔, ρ) 이면 실측으로 표시. w¹/w² 는 팔 이름 그대로.\n'
        '출처: inter_deep_261005.json  F.runs[].b.macs_pct, F.offline_npz.<키>.{n,spikes,macs_pct_b}, F.pooled_macs.300-420K.macs_pct_b, F.measured_ancova.b.')
fig.tight_layout(rect=(0, 0.085, 1, 1)); save(fig, 'inter_deep_macs')
