"""10-19 분석 3종 (결과 요약 / 방법 분석 / ablation) 그림 6장.

수치는 eip-stats 산출물 json 만 읽는다 (원본 실험 CSV 를 읽지 않는다):
    analysis/results_261019.json   -- 그림 1, 2 (원장 고정 사본 10-19 16:38)
    analysis/method_261019.json    -- 그림 3
    analysis/ablation_261019.json  -- 그림 5, 6 의 통계값 (342K 직접 비교, 학습 궤적, rho 눈금)
    그림 4 의 점·평균은 results json 의 conds (그림 1, 2 와 같은 n), 통계값은 ablation json.

규약: 실제 정확도(y) x 실제 스파이크 수(x). 차이(delta)로 그리지 않는다.
  baseline = 평균 +- 시드 SD 띠. 개별 런 = 작은 점 (○ 서버 138, □ 서버 23).
  조건 평균 = 큰 마커 + 시드 SD 막대 (반복 횟수 2 이상). 속 빈 마커 = 반복 횟수 1.
  y 축은 정확도 일부 구간만 표시한다 (각 패널 안에 구간을 적는다).

그림 7 (ablation 구조, 평가 모드 npz): 342K 비교점 각 팔의 런별 값 (가로 = npz 스파이크, 세로 = 구조 지표 실제 값).
  수치는 ablation json 의 npz.per_run (런별), npz.<팔>.<지표> (평균·SD·곡선 ANCOVA·보정 직접) 만 읽는다.

쓰는 법:  ~/anaconda3/envs/venv_1/bin/python EIP_fig/an_261019_fig.py [그림 번호 ...]
"""
import os, json, math, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator, NullFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'EIP_fig', 'an_261019')
os.makedirs(OUT, exist_ok=True)
STAMP = '26-10-19'
R = json.load(open(os.path.join(ROOT, 'analysis', 'results_261019.json')))
M = json.load(open(os.path.join(ROOT, 'analysis', 'method_261019.json')))
A = json.load(open(os.path.join(ROOT, 'analysis', 'ablation_261019.json')))

plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.edgecolor'] = '#555555'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

INK = '#222222'
INK2 = '#555555'
GRID = '#e6e6e6'

# 범주 색 (validate_palette.js --pairs all: 3색 통과, 5색 통과 -- CVD 6~8 구간이라 마커 모양·직접 라벨을 함께 쓴다)
C_BASE = '#222222'
C_OURS = '#009E73'
C_L2 = '#E69F00'
C_BPSR = '#CC79A7'
C_VMEM = '#0072B2'
C_FINAL = '#D55E00'
C_INV = '#E69F00'
C_LR = '#CC79A7'

METHOD = {  # 이름: (색, 마커, 선모양)
    'ours_inter_ch': (C_OURS, 'o', '-'),
    'plain L2': (C_L2, 'D', (0, (4, 2))),
    'BPSR': (C_BPSR, 'v', (0, (1, 1.6))),
}
ABL = {  # 팔: (색, 마커)
    'ours_inter_ch': (C_OURS, 'o'),
    '- vmem': (C_VMEM, 's'),
    '- final_step': (C_FINAL, 'v'),
    '- 1- inversion': (C_INV, 'D'),
    '- loss-ratio': (C_LR, 'P'),
}
SRV_MARK = {'138': 'o', '23': 's'}


def kfmt(x, _=None):
    return f'{x / 1e3:.0f}K'


def mfmt(x, _=None):
    return f'{x / 1e6:.2f}M'


def conds_of(block, arms=None):
    """results json 블록의 조건 목록 (arm, knob, n, 평균, SD, runs)."""
    out = []
    for c in block['conds']:
        if arms is None or c['arm'] in arms:
            out.append(c)
    return out


def knob_val(k):
    try:
        return float(k)
    except Exception:
        return float('nan')


def plot_runs(ax, runs, color, zorder=3, size=15, alpha=.5):
    """개별 런: 작은 점. 서버 138 = 원, 23 = 네모."""
    for r in runs:
        mk = SRV_MARK.get(str(r.get('server')), 'o')
        ax.scatter([r['spikes']], [r['val_acc']], s=size, marker=mk, color=color, alpha=alpha,
                   linewidths=0, zorder=zorder)


def plot_mean(ax, c, color, marker, ms=7.5, zorder=5, capsize=2.2, lw=1.1):
    """조건 평균: 반복 횟수 2 이상 = 채운 마커 + 시드 SD 막대, 1 = 속 빈 마커."""
    n = c['n']
    x, y = c['spikes'], c['val_acc']
    if n >= 2 and c.get('val_acc_sd') is not None:
        ax.errorbar([x], [y], yerr=[c['val_acc_sd']], fmt='none', ecolor=color, elinewidth=lw,
                    capsize=capsize, zorder=zorder - 1)
        ax.plot([x], [y], marker=marker, ms=ms, color=color, mfc=color, mec='white', mew=.7, lw=0,
                zorder=zorder)
    else:
        ax.plot([x], [y], marker=marker, ms=ms, color=color, mfc='white', mec=color, mew=1.5, lw=0,
                zorder=zorder)


def plot_curve(ax, cs, color, marker, ls='-', line=True, lw=1.1, ms=7.5, run_alpha=.5, zorder=3):
    cs = sorted(cs, key=lambda c: c['spikes'])
    if line and len(cs) > 1:
        ax.plot([c['spikes'] for c in cs], [c['val_acc'] for c in cs], ls=ls, lw=lw, color=color,
                alpha=.85, zorder=zorder)
    for c in cs:
        plot_runs(ax, c['runs'], color, zorder=zorder, alpha=run_alpha)
    for c in cs:
        plot_mean(ax, c, color, marker, ms=ms, zorder=zorder + 2)


def plot_baseline(ax, b, runs=None, label=True, vline=True):
    """baseline: 평균 +- 시드 SD 띠 (반복 횟수 2 이상), 평균선, baseline 스파이크 위치의 세로 점선."""
    if b['n'] >= 2:
        ax.axhspan(b['acc'] - b['acc_sd'], b['acc'] + b['acc_sd'], color='#8a8a8a', alpha=.20, lw=0, zorder=0)
        ax.axhline(b['acc'], color='#666666', ls=':', lw=.9, zorder=1)
    if vline:
        ax.axvline(b['spikes'], color='#888888', ls=(0, (1, 2)), lw=.8, zorder=1)
    if runs:
        plot_runs(ax, runs, C_BASE, zorder=3, alpha=.5)
    if b['n'] >= 2:
        ax.errorbar([b['spikes']], [b['acc']], yerr=[b['acc_sd']], fmt='none', ecolor=C_BASE, elinewidth=1.2,
                    capsize=2.5, zorder=5)
        ax.plot([b['spikes']], [b['acc']], marker='*', ms=13, color=C_BASE, mec='white', mew=.6, lw=0, zorder=6)
    else:
        ax.plot([b['spikes']], [b['acc']], marker='*', ms=13, color=C_BASE, mfc='white', mec=C_BASE, mew=1.3,
                lw=0, zorder=6)


def set_ylim_from(ax, ys, pad=.10, lo=None, hi=None):
    a, b = min(ys), max(ys)
    d = (b - a) or 1.0
    lo = a - pad * d if lo is None else lo
    hi = b + pad * d if hi is None else hi
    ax.set_ylim(lo, hi)
    return lo, hi


def trunc_note(ax, ylim, fmt='{:.1f}'):
    ax.text(0.012, 0.012, 'y 축은 ' + fmt.format(ylim[0]) + '~' + fmt.format(ylim[1]) + '%만 표시 (0 에서 시작하지 않음)',
            transform=ax.transAxes, fontsize=7, color=INK2, va='bottom', ha='left', zorder=10,
            bbox=dict(fc='white', ec='none', alpha=.75, pad=1.2))


def style_ax(ax):
    ax.grid(True, color=GRID, lw=.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK2, length=3)


def base_label(b):
    if b['n'] >= 2:
        return f"baseline {b['acc']:.2f} ±{b['acc_sd']:.2f} (반복 횟수 {b['n']})"
    return f"baseline {b['acc']:.2f} (반복 횟수 {b['n']})"


def footer(fig, text, y=0.005):
    """그림 아래 출처·주의 글. y = 글 윗변 (그림 좌표). 그림 너비에 맞춰 줄바꿈."""
    width = int(fig.get_size_inches()[0] * 15.5)
    fig.text(0.012, y, textwrap.fill(text, width), fontsize=7, color=INK2, ha='left', va='top')


def label_pts(ax, cs, offs=None, fs=6.8, fmt=lambda k: k, color=INK2):
    """조건 평균 옆에 ρ 라벨. offs = {knob: (dx, dy, ha)}; 없으면 위·아래 번갈아."""
    cs = sorted(cs, key=lambda c: c['spikes'])
    offs = offs or {}
    for i, c in enumerate(cs):
        dx, dy, ha = offs.get(c['knob'], (0, 9 if i % 2 == 0 else -12, 'center'))
        ax.annotate(fmt(c['knob']), (c['spikes'], c['val_acc']), xytext=(dx, dy), textcoords='offset points',
                    fontsize=fs, color=color, ha=ha, va='center', zorder=9)


def legend_handles_common(arms):
    hs = []
    for name, col, mk in arms:
        hs.append(Line2D([0], [0], marker=mk, color=col, mfc=col, mec='white', mew=.7, ms=8, lw=1.0, label=name))
    return hs


def save(fig, name):
    p = os.path.join(OUT, f'{name}_ko_{STAMP}.png')
    fig.savefig(p, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', p)
    return p


# ---------------------------------------------------------------------------------------------
# 그림 1. 설정별 정확도 x 스파이크 (작은 패널 6개)
# ---------------------------------------------------------------------------------------------
def fig1():
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    axes = axes.ravel()
    spec = [
        ('R19 · CIFAR-10 (T=4, 310 에폭)', R['r19_c10'], 'k'),
        ('R19 · CIFAR-100 (T=4, 310 에폭)', R['r19_c100'], 'k'),
        ('VGGSNN · CIFAR10-DVS 첫 프레임 (200 에폭)', R['dvs_f0'], 'k'),
        ('QKFormer · CIFAR-10 (410 에폭, 재측정)', R['qk_c10'], 'k'),
        ('QKFormer · CIFAR-100 (410 에폭, 재측정)', R['qk_c100'], 'k'),
    ]
    sumry = {}
    for ax, (title, blk, _) in zip(axes[:5], spec):
        b = blk['base']
        cs_all = conds_of(blk)
        base_c = [c for c in cs_all if c['arm'] == 'baseline'][0]
        ours = [c for c in cs_all if c['arm'] == 'ours_inter_ch']
        plot_baseline(ax, b, runs=base_c['runs'])
        plot_curve(ax, ours, C_OURS, 'o')
        ys = [r['val_acc'] for c in ours + [base_c] for r in c['runs']] + [b['acc'] - b['acc_sd'], b['acc'] + b['acc_sd']]
        yl = set_ylim_from(ax, ys, pad=.10)
        xs = [r['spikes'] for c in ours + [base_c] for r in c['runs']]
        span = max(xs) - min(xs)
        ax.set_xlim(min(xs) - .06 * span, max(xs) + .06 * span)
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.set_title(title + '\n' + base_label(b), fontsize=9.5, color=INK, loc='left')
        ax.set_xlabel('스파이크 수 (이미지당)', fontsize=8.5, color=INK2)
        ax.set_ylabel('정확도 (%)', fontsize=8.5, color=INK2)
        style_ax(ax)
        trunc_note(ax, yl, '{:.1f}')
        sumry[title] = {'base_n': b['n'], 'ours_conds': [(c['knob'], c['n']) for c in ours]}
    # ρ 라벨 (겹침 방지: R19 C10 은 일부 ρ 만 적는다)
    c10 = sorted([c for c in conds_of(R['r19_c10']) if c['arm'] == 'ours_inter_ch'], key=lambda c: c['spikes'])
    skip10 = {'4.5e-3', '5e-3', '6e-3'}
    show10 = [c for c in c10 if c['knob'] not in skip10]
    label_pts(axes[0], show10, offs={
        '1.2e-2': (0, 10, 'center'), '8e-3': (0, -11, 'center'), '7e-3': (0, 10, 'center'), '5.5e-3': (-4, 11, 'center'),
        '4e-3': (0, 11, 'center'), '3e-3': (0, -12, 'center'), '2e-3': (-4, -13, 'center'), '1.5e-3': (6, 12, 'center')}, fs=6.5)
    axes[0].text(0.985, 0.03, '라벨은 ρ (겹치는 4.5e-3·5e-3·6e-3 은 생략)', transform=axes[0].transAxes, fontsize=6.8,
                 color=INK2, ha='right', va='bottom')
    label_pts(axes[1], [c for c in conds_of(R['r19_c100']) if c['arm'] == 'ours_inter_ch'],
              offs={'3e-3': (0, 10, 'center'), '2.5e-3': (0, -12, 'center'), '2e-3': (0, 10, 'center'),
                    '1.5e-3': (0, 10, 'center'), '1.2e-3': (0, -12, 'center'), '1e-3': (0, 10, 'center')}, fs=6.5)
    label_pts(axes[2], [c for c in conds_of(R['dvs_f0']) if c['arm'] == 'ours_inter_ch'],
              offs={'2e-2': (14, 10, 'center'), '1e-2': (0, 10, 'center'), '5e-3': (0, -12, 'center')}, fs=6.8)
    label_pts(axes[3], [c for c in conds_of(R['qk_c10']) if c['arm'] == 'ours_inter_ch'],
              offs={'2e-2': (0, 13, 'center'), '1.2e-2': (26, -6, 'left'), '7e-3': (0, -12, 'center'),
                    '5e-3': (0, -12, 'center'), '3e-3': (6, 12, 'center'), '2e-3': (10, -12, 'center'),
                    '1e-3': (0, 12, 'center')}, fs=6.5)
    label_pts(axes[4], [c for c in conds_of(R['qk_c100']) if c['arm'] == 'ours_inter_ch'],
              offs={'1.2e-2': (0, 14, 'center'), '6e-3': (24, -10, 'left')}, fs=7)

    # SDTv3
    ax = axes[5]
    sdt = [r for r in R['sdt_main'] if str(r['dir']).startswith('main_') and not str(r['dir']).startswith('main_kd')]
    base = [r for r in sdt if r['knob'] == '-'][0]
    ours = [r for r in sdt if r['knob'] != '-']
    ax.axvline(base['spikes'], color='#888888', ls=(0, (1, 2)), lw=.8, zorder=1)
    ax.plot([base['spikes']], [base['val_acc']], marker='*', ms=13, color=C_BASE, mfc='white', mec=C_BASE, mew=1.3,
            lw=0, zorder=6)
    for r in ours:
        ax.plot([r['spikes']], [r['val_acc']], marker='o', ms=7.5, color=C_OURS, mfc='white', mec=C_OURS, mew=1.5,
                lw=0, zorder=5)
        ax.annotate('ρ ' + r['knob'], (r['spikes'], r['val_acc']), xytext=(0, 9), textcoords='offset points',
                    fontsize=7, color=INK2, ha='center')
    ys = [r['val_acc'] for r in sdt]
    yl = set_ylim_from(ax, ys, pad=.10)
    ax.set_xlim(1.25e6, 2.12e6)
    ax.xaxis.set_major_formatter(FuncFormatter(mfmt))
    ax.set_title('SDTv3 171M · ImageNet-1k 파인튜닝 (30 에폭, 마지막 에폭 값)\nbaseline(대조) 반복 횟수 1 — 시드 SD 미측정이라 띠 없음',
                 fontsize=9.5, color=INK, loc='left')
    ax.set_xlabel('스파이크 수 (이미지당)', fontsize=8.5, color=INK2)
    ax.set_ylabel('정확도 (%)', fontsize=8.5, color=INK2)
    style_ax(ax)
    trunc_note(ax, yl, '{:.1f}')
    ax.text(0.98, 0.04, '모든 점 반복 횟수 1 (서버 23)\n진행 중 KD 4런은 뺐다', transform=ax.transAxes, fontsize=7,
            color=INK2, ha='right', va='bottom')

    hs = [
        Patch(fc='#8a8a8a', alpha=.35, label='baseline 평균 ± 시드 SD (반복 횟수 2 이상일 때)'),
        Line2D([0], [0], marker='*', color=C_BASE, ms=12, lw=0, label='baseline 평균 (속 빈 별 = 반복 횟수 1)'),
        Line2D([0], [0], marker='o', color=C_OURS, mfc=C_OURS, mec='white', ms=8, lw=1.0,
               label='ours_inter_ch 조건 평균 (막대 = 시드 SD, 반복 횟수 2 이상)'),
        Line2D([0], [0], marker='o', color=C_OURS, mfc='white', mec=C_OURS, mew=1.5, ms=8, lw=0,
               label='속 빈 마커 = 반복 횟수 1'),
        Line2D([0], [0], marker='o', color='#777777', alpha=.6, ms=5, lw=0, label='개별 런 (서버 138)'),
        Line2D([0], [0], marker='s', color='#777777', alpha=.6, ms=5, lw=0, label='개별 런 (서버 23)'),
    ]
    fig.legend(handles=hs, loc='lower center', ncol=3, fontsize=8.3, frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle('설정별 정확도 × 스파이크 — baseline 대비 ours_inter_ch (10-19 고정 사본, 실제 값)', fontsize=12,
                 color=INK, x=0.012, ha='left', y=0.995)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    footer(fig, '수치: analysis/results_261019.json (원장 고정 사본 10-19 16:38). R19·VGGSNN 스파이크 = 원장 s_count, QK = 테스트셋 fp32 재측정, '
                'SDTv3 = 원장 기록값 — 설정끼리 가로축 값을 직접 비교하지 않는다. 원장 배제 런·진행 중 런은 그리지 않았다.', y=-0.057)
    return save(fig, 'acc_spikes_settings')


# ---------------------------------------------------------------------------------------------
# 그림 2. R19 C10 / C100: ours_inter_ch vs plain L2 vs BPSR
# ---------------------------------------------------------------------------------------------
def fig2():
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.9))
    items = [
        (axes[0], 'R19 · CIFAR-10', R['r19_c10'], 'C10_methods', '200000-420000', (1.30e5, 5.15e5)),
        (axes[1], 'R19 · CIFAR-100', R['r19_c100'], 'C100_methods', '220000-420000', (2.10e5, 4.75e5)),
    ]
    for ax, title, blk, akey, bkey, xl in items:
        b = blk['base']
        cs_all = conds_of(blk)
        base_c = [c for c in cs_all if c['arm'] == 'baseline'][0]
        plot_baseline(ax, b, runs=base_c['runs'])
        allys = [b['acc'] - b['acc_sd'], b['acc'] + b['acc_sd']]
        for name, (col, mk, ls) in METHOD.items():
            cs = [c for c in cs_all if c['arm'] == name]
            plot_curve(ax, cs, col, mk, ls=ls, zorder=3 if name != 'ours_inter_ch' else 4)
            allys += [r['val_acc'] for c in cs for r in c['runs']]
        yl = set_ylim_from(ax, allys, pad=.07)
        ax.set_xlim(*xl)
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.set_title(f'{title}\n{base_label(b)}', fontsize=10, color=INK, loc='left')
        ax.set_xlabel('스파이크 수 (이미지당, 원장 s_count)', fontsize=8.8, color=INK2)
        ax.set_ylabel('정확도 (best val acc, %)', fontsize=8.8, color=INK2)
        style_ax(ax)
        trunc_note(ax, yl, '{:.1f}')
        # 같은 스파이크 비교 (ANCOVA) 결과 글
        a = R['ancova'][akey][bkey]
        nb = a['n_by_arm']
        off = a['offsets']
        lines = [f"같은 스파이크 비교 ({int(a['band'][0] / 1000)}~{int(a['band'][1] / 1000)}K, 런 {a['N']}개, 서버 항 포함)"
                 f" — ours_inter_ch(런 {nb['ours_inter_ch']}개) 대비:"]
        for nm in ('BPSR', 'plain L2'):
            o = off[nm]
            lines.append(f"  {nm} (런 {nb[nm]}개): {o['c']:+.3f}%p [{o['lo']:+.3f}, {o['hi']:+.3f}], p={o['p']:.2f}")
        lines.append('  → 세 방법의 차이는 검출되지 않음 (같다는 뜻은 아님)')
        ax.text(0.5, -0.15, '\n'.join(lines), transform=ax.transAxes, fontsize=7.8, color=INK, ha='center', va='top')
    # 표1 지점 (C10, 각 반복 횟수 4): 검은 고리로 표시하고 값은 구석에 한 번에 적는다
    ax = axes[0]
    t1 = [('ours_inter_ch', '3e-3', 'ρ'), ('BPSR', '1e-10', 'λ'), ('plain L2', '1e-7', 'λ')]
    cs_all = conds_of(R['r19_c10'])
    tl = ['고리 = 표1 비교점 (각 반복 횟수 4):']
    for nm, kn, sym in t1:
        c = [c for c in cs_all if c['arm'] == nm and c['knob'] == kn][0]
        ax.plot([c['spikes']], [c['val_acc']], marker='o', ms=17, mfc='none', mec='#111111', mew=1.0, lw=0, zorder=9)
        tl.append(f"  {nm} {sym} {kn}: {c['val_acc']:.2f} @ {c['spikes'] / 1e3:.0f}K")
    ax.text(0.985, 0.05, '\n'.join(tl), transform=ax.transAxes, fontsize=7.4, color=INK, ha='right', va='bottom',
            bbox=dict(fc='white', ec='#cccccc', lw=.5, alpha=.9, pad=2.5), zorder=10)
    hs = legend_handles_common([('ours_inter_ch', C_OURS, 'o'), ('plain L2', C_L2, 'D'), ('BPSR', C_BPSR, 'v')])
    hs += [
        Patch(fc='#8a8a8a', alpha=.35, label='baseline 평균 ± 시드 SD'),
        Line2D([0], [0], marker='o', color='#555555', mfc='#555555', mec='white', ms=8, lw=0,
               label='채운 마커 = 조건 평균 (막대 = 시드 SD)'),
        Line2D([0], [0], marker='o', color='#555555', mfc='white', mec='#555555', mew=1.5, ms=8, lw=0,
               label='속 빈 마커 = 반복 횟수 1'),
        Line2D([0], [0], marker='o', color='#777777', alpha=.6, ms=5, lw=0, label='개별 런: ○ 서버 138'),
        Line2D([0], [0], marker='s', color='#777777', alpha=.6, ms=5, lw=0, label='개별 런: □ 서버 23'),
    ]
    fig.legend(handles=hs, loc='lower center', ncol=5, fontsize=8.3, frameon=False, bbox_to_anchor=(0.5, -0.115))
    fig.suptitle('R19 에서 세 규제 방법의 정확도 × 스파이크 (선 = 조건 평균을 스파이크 순으로 이은 것, 회귀선 아님)',
                 fontsize=11.5, color=INK, x=0.012, ha='left', y=1.0)
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    footer(fig, '수치: analysis/results_261019.json (ancova, conds). 원장 배제 런(S30/S1<0.19)·진행 중 런·ablation 팔은 그리지 않았다. '
                'ρ = ours_inter_ch 손잡이, λ = plain L2·BPSR 손잡이.', y=-0.128)
    return save(fig, 'r19_methods')


# ---------------------------------------------------------------------------------------------
# 그림 3. 구조 (npz, 테스트 1만 장) — 같은 스파이크 근처에서 팔 비교
# ---------------------------------------------------------------------------------------------
def fig3():
    rows = M['npz_rows']
    t1 = M['q3']['table1']['mean']
    b1 = M['q3']['npz']['B1_300_420K']
    xr = b1['dead_ch']['x_range']
    nm_map = {'ours_inter_ch': 'ours_inter_ch', 'plain L2': 'plain L2', 'BPSR': 'BPSR', 'baseline': 'baseline'}
    style = {'ours_inter_ch': (C_OURS, 'o'), 'plain L2': (C_L2, 'D'), 'BPSR': (C_BPSR, 'v'), 'baseline': (C_BASE, '*')}
    metrics = [
        ('dead_ch', '죽은 채널 수 (전체 4,736개 중)', lambda v: v, lambda v: f'{v:.0f}', '개'),
        ('macs_a', '지울 수 있는 밀집 MACs (%, 시나리오 a)', lambda v: v, lambda v: f'{v:.1f}', '%p'),
        ('active_img', '이미지당 활성 뉴런 (천 개, 전체 1,442K 중)', lambda v: v / 1e3, lambda v: f'{v / 1e3:.0f}', ''),
        ('gini_neuron', '뉴런 지니 (0 = 고름, 1 = 한곳에 몰림)', lambda v: v, lambda v: f'{v:.2f}', ''),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.2))
    for ax, (key, ylabel, tf, _f, unit) in zip(axes.ravel(), metrics):
        ax.axvspan(xr[0], xr[1], color='#d9ecff', alpha=.55, lw=0, zorder=0)
        for arm, (col, mk) in style.items():
            rs = [r for r in rows if r['arm'] == arm]
            for r in rs:
                m_ = SRV_MARK.get(str(r['server']), 'o') if arm != 'baseline' else SRV_MARK.get(str(r['server']), 'o')
                ax.scatter([r['npz_spk']], [tf(r[key])], s=17, marker=m_, color=col, alpha=.5, linewidths=0, zorder=3)
        for arm, (col, mk) in style.items():
            tm = t1.get(arm)
            if tm is None:
                continue
            (ym, ysd, n), (xm, xsd, _n) = tm[key], tm['npz_spk']
            mfc = col
            ax.errorbar([xm], [tf(ym)], xerr=[xsd], yerr=[tf(ysd)], fmt='none', ecolor=col, elinewidth=1.1,
                        capsize=2.2, zorder=5)
            ax.plot([xm], [tf(ym)], marker=(mk if arm != 'baseline' else '*'), ms=(10 if arm != 'baseline' else 14),
                    color=col, mfc=mfc, mec='white', mew=.8, lw=0, zorder=6)
        allv = [tf(r[key]) for r in rows]
        lo, hi = min(allv), max(allv)
        d = hi - lo
        ax.set_ylim(lo - .08 * d, hi + .20 * d)
        ax.set_xlim(1.0e5, 5.35e5)
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.set_xlabel('npz 스파이크 수 (테스트 1만 장, 이미지당; 원장 s_count 축과 다름)', fontsize=8.5, color=INK2)
        ax.set_ylabel(ylabel, fontsize=8.8, color=INK2)
        style_ax(ax)
        c = b1[key]['contrast']
        n = {'o': c['BPSR']['n_ref'], 'b': c['BPSR']['n_o'], 'l': c['plain L2']['n_o']}
        if key == 'active_img':
            tb = f"{c['BPSR']['pct']:+.1f}% [{c['BPSR']['pct_ci95'][0]:+.1f}, {c['BPSR']['pct_ci95'][1]:+.1f}]"
            tl = f"{c['plain L2']['pct']:+.1f}% [{c['plain L2']['pct_ci95'][0]:+.1f}, {c['plain L2']['pct_ci95'][1]:+.1f}]"
        elif key == 'gini_neuron':
            tb = f"{c['BPSR']['c']:+.3f} [{c['BPSR']['ci95'][0]:+.3f}, {c['BPSR']['ci95'][1]:+.3f}]"
            tl = f"{c['plain L2']['c']:+.3f} [{c['plain L2']['ci95'][0]:+.3f}, {c['plain L2']['ci95'][1]:+.3f}]"
        else:
            tb = f"{c['BPSR']['c']:+.1f}{unit} [{c['BPSR']['ci95'][0]:+.1f}, {c['BPSR']['ci95'][1]:+.1f}]"
            tl = f"{c['plain L2']['c']:+.1f}{unit} [{c['plain L2']['ci95'][0]:+.1f}, {c['plain L2']['ci95'][1]:+.1f}]"
        ax.text(0.015, 0.985, f"같은 스파이크 (파란 띠 {xr[0] / 1e3:.0f}~{xr[1] / 1e3:.0f}K, ours_inter_ch 런 {n['o']}개 기준)\n"
                              f"  ours_inter_ch − BPSR (런 {n['b']}개): {tb}\n"
                              f"  ours_inter_ch − plain L2 (런 {n['l']}개): {tl}",
                transform=ax.transAxes, fontsize=7.4, color=INK, ha='left', va='top', zorder=10,
                bbox=dict(fc='white', ec='none', alpha=.85, pad=1.5))
    hs = legend_handles_common([('ours_inter_ch', C_OURS, 'o'), ('plain L2', C_L2, 'D'), ('BPSR', C_BPSR, 'v')])
    hs += [Line2D([0], [0], marker='*', color=C_BASE, ms=12, lw=0, label='baseline (반복 횟수 3)'),
           Patch(fc='#d9ecff', alpha=.9, label='같은 스파이크 비교 구간 (모형 fit 범위)'),
           Line2D([0], [0], marker='o', color='#555555', mec='white', ms=9, lw=0,
                  label='큰 마커 = 표1 점 평균 ± SD (각 반복 횟수 4, baseline 3)'),
           Line2D([0], [0], marker='o', color='#777777', alpha=.6, ms=5, lw=0, label='작은 점 = 개별 런 (○ 138, □ 23)')]
    fig.legend(handles=hs, loc='lower center', ncol=4, fontsize=8.3, frameon=False, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle('구조: 같은 스파이크에서 ours_inter_ch 는 죽은 채널·지울 수 있는 MACs 가 더 많고 활성 뉴런은 적다 — 보조 근거 (B 등급)',
                 fontsize=11.5, color=INK, x=0.012, ha='left', y=0.998)
    fig.tight_layout(rect=(0, 0.01, 1, 0.965))
    footer(fig, '수치: analysis/method_261019.json (npz_rows 47런: baseline 3, ours_inter_ch 20, plain L2 11, BPSR 13; q3.table1; q3.npz.B1_300_420K). '
                '정확도 차는 검출되지 않으므로 이 그림은 "구조가 다르다"만 보인다. maxnorm 계열 공통 성질이라 ours_inter_ch 고유 근거는 아니다.', y=-0.057)
    return save(fig, 'structure_matched')


# ---------------------------------------------------------------------------------------------
# 그림 4. ablation: 정확도 × 스파이크 (342K 비교점 5묶음 + ours_inter_ch 곡선)
# ---------------------------------------------------------------------------------------------
PT_KNOB = {'ours_inter_ch': '3e-3', '- vmem': '2e-3', '- final_step': '1.4e-3', '- 1- inversion': '4.5e-3',
           '- loss-ratio': '1.25e-7'}
ARM_KEY = {'- vmem': 'ch_novmem', '- final_step': 'ch_nofinal', '- 1- inversion': 'ch_noinv', '- loss-ratio': 'ch_nolr'}


def fig4():
    blk = R['r19_c10']
    cs_all = conds_of(blk)
    b = blk['base']
    base_c = [c for c in cs_all if c['arm'] == 'baseline'][0]
    fig = plt.figure(figsize=(15.5, 7.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.17)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])
    ours = [c for c in cs_all if c['arm'] == 'ours_inter_ch']
    allys = [b['acc'] - b['acc_sd'], b['acc'] + b['acc_sd']]
    for ax in (axL, axR):
        plot_baseline(ax, b, runs=base_c['runs'] if ax is axL else None, vline=(ax is axL))
    # 왼쪽: 전체 곡선
    plot_curve(axL, ours, C_OURS, 'o', ms=7, zorder=3)
    allys += [r['val_acc'] for c in ours for r in c['runs']]
    for arm, (col, mk) in ABL.items():
        if arm == 'ours_inter_ch':
            continue
        cs = [c for c in cs_all if c['arm'] == arm]
        for c in cs:
            full = (c['n'] >= 4)
            if full:
                plot_runs(axL, c['runs'], col, zorder=4, alpha=.5)
                plot_mean(axL, c, col, mk, ms=9, zorder=7)
            else:
                plot_runs(axL, c['runs'], col, zorder=2, alpha=.28)
                # 반복 횟수 4 미만은 연하게
                x, y = c['spikes'], c['val_acc']
                if c['n'] >= 2 and c.get('val_acc_sd') is not None:
                    axL.errorbar([x], [y], yerr=[c['val_acc_sd']], fmt='none', ecolor=col, elinewidth=.9, capsize=2,
                                 alpha=.55, zorder=3)
                    axL.plot([x], [y], marker=mk, ms=6.5, color=col, mfc=col, alpha=.5, mec='white', mew=.5, lw=0, zorder=4)
                else:
                    axL.plot([x], [y], marker=mk, ms=6.5, color=col, mfc='white', mec=col, mew=1.2, alpha=.65, lw=0,
                             zorder=4)
            allys += [r['val_acc'] for r in c['runs']]
    yl = set_ylim_from(axL, allys, pad=.06)
    axL.set_xlim(1.40e5, 5.2e5)
    axL.xaxis.set_major_formatter(FuncFormatter(kfmt))
    axL.set_title('(가) 전체: ours_inter_ch 곡선 위에 끈 팔을 얹은 모습\n' + base_label(b), fontsize=10, color=INK, loc='left')
    axL.set_xlabel('스파이크 수 (이미지당, 원장 s_count)', fontsize=8.8, color=INK2)
    axL.set_ylabel('정확도 (best val acc, %)', fontsize=8.8, color=INK2)
    style_ax(axL)
    trunc_note(axL, yl, '{:.1f}')
    # 342K 확대 창 표시
    axL.axvspan(328e3, 384e3, color='#d9ecff', alpha=.6, lw=0, zorder=0)
    axL.annotate('(나) 확대 구간', (356e3, yl[1] - .01), xytext=(0, 0), textcoords='offset points', fontsize=7.5,
                 color=INK2, ha='center', va='top')

    # 오른쪽: 342K 확대
    pts = {}
    for arm, kn in PT_KNOB.items():
        c = [c for c in cs_all if c['arm'] == arm and c['knob'] == kn][0]
        pts[arm] = c
    for arm, c in pts.items():
        col, mk = ABL[arm]
        plot_runs(axR, c['runs'], col, zorder=3, size=26, alpha=.55)
        plot_mean(axR, c, col, mk, ms=11, zorder=6, capsize=3.2, lw=1.4)
    # 라벨 위치 (겹침 방지)
    lab = {
        'ours_inter_ch': (16, 66, 'left'),
        '- vmem': (14, -40, 'left'),
        '- final_step': (4, -30, 'left'),
        '- 1- inversion': (-8, -58, 'right'),
        '- loss-ratio': (70, 14, 'left'),
    }
    for arm, c in pts.items():
        col, mk = ABL[arm]
        ddx, ddy, ha = lab[arm]
        knob = c['knob']
        sym = 'λ' if arm == '- loss-ratio' else 'ρ'
        axR.annotate(f"{arm}\n{sym} {knob} (반복 {c['n']})\n{c['val_acc']:.2f} @ {c['spikes'] / 1e3:.0f}K",
                     (c['spikes'], c['val_acc']), xytext=(ddx, ddy), textcoords='offset points', fontsize=7.6,
                     color=INK, ha=ha, va='center', zorder=10,
                     arrowprops=dict(arrowstyle='-', color='#999999', lw=.6))
    ys = [r['val_acc'] for c in pts.values() for r in c['runs']] + [b['acc'] - b['acc_sd'], b['acc'] + b['acc_sd']]
    yl2 = set_ylim_from(axR, ys, pad=.20)
    axR.set_ylim(yl2[0], yl2[1] + .10)
    yl2 = axR.get_ylim()
    axR.set_xlim(328e3, 384e3)
    axR.xaxis.set_major_formatter(FuncFormatter(kfmt))
    axR.set_title('(나) 342K 근처 비교점 (각 반복 횟수 4) 확대 — 가로축 328K~384K 만 표시\n' + base_label(b) + ' (띠, baseline 위치 489K 는 범위 밖)',
                  fontsize=10, color=INK, loc='left')
    axR.set_xlabel('스파이크 수 (이미지당, 원장 s_count)', fontsize=8.8, color=INK2)
    axR.set_ylabel('정확도 (best val acc, %)', fontsize=8.8, color=INK2)
    style_ax(axR)
    trunc_note(axR, yl2, '{:.2f}')
    # 비교 글 (ablation json, 342K 직접 비교 4 대 4 Welch)
    ref = pts['ours_inter_ch']
    lines = [f"기준 ours_inter_ch ρ 3e-3 대비 정확도 차\n(반복 횟수 4 대 4, Welch, 95% 구간; 최소 검출 차 {A['acc']['ch_novmem']['mde_direct_4v4']:.2f}%p)"]
    for arm in ('- vmem', '- final_step', '- 1- inversion', '- loss-ratio'):
        w = A['acc'][ARM_KEY[arm]]['direct_welch']
        hp = A['acc'][ARM_KEY[arm]]['holm_p_primary'] if 'holm_p_primary' in A['acc'][ARM_KEY[arm]] else None
        lines.append(f"{arm}: {w['d']:+.3f}%p [{w['lo']:+.2f}, {w['hi']:+.2f}], p={w['p']:.2f}")
    lines.append('→ 네 팔 모두 구간이 0 을 포함 (검출되지 않음, 없다는 뜻은 아님)')
    axR.text(0.992, 0.992, '\n'.join(lines), transform=axR.transAxes, fontsize=7.6, color=INK, ha='right', va='top', zorder=11,
             bbox=dict(fc='white', ec='#cccccc', lw=.5, alpha=.95, pad=3))

    hs = [Line2D([0], [0], marker=ABL[k][1], color=ABL[k][0], mfc=ABL[k][0], mec='white', mew=.7, ms=8, lw=0, label=k)
          for k in ABL]
    hs += [Patch(fc='#8a8a8a', alpha=.35, label='baseline 평균 ± 시드 SD'),
           Line2D([0], [0], marker='*', color=C_BASE, ms=12, lw=0, label='baseline 평균'),
           Line2D([0], [0], marker='o', color='#555555', mfc='white', mec='#555555', mew=1.4, ms=7, lw=0,
                  label='속 빈·연한 마커 = 반복 횟수 4 미만'),
           Line2D([0], [0], marker='o', color='#777777', alpha=.6, ms=5, lw=0, label='개별 런 ○138 □23')]
    fig.legend(handles=hs, loc='lower center', ncol=5, fontsize=8.3, frameon=False, bbox_to_anchor=(0.5, -0.075))
    fig.suptitle('ablation (R19 CIFAR-10): 구성요소를 하나씩 끈 팔 — 342K 근처 비교점 5묶음 (반복 횟수 4) 과 ours_inter_ch 곡선',
                 fontsize=11.5, color=INK, x=0.012, ha='left', y=1.0)
    fig.tight_layout(rect=(0, 0.0, 1, 0.95))
    pv = [x for x in R['ablation_342k'] if x['arm'].startswith('- vmem')][0]
    pf = [x for x in R['ablation_342k'] if x['arm'].startswith('- final_step')][0]
    dv = pv['dspk'] * 100
    footer(fig, '수치: 점·평균 analysis/results_261019.json (conds, ablation_342k), 4 대 4 비교 analysis/ablation_261019.json (acc.*.direct_welch). '
                f'선 = ours_inter_ch 조건 평균을 이은 것 (모형 아님). - vmem 2e-3 은 기준보다 스파이크가 {dv:.1f}% 많은 곳에 착지했다 (같은 스파이크 비교 아님). '
                f"같은 비교를 results 보고서는 합동 시드 SD 로 검정해 p={pv['p']:.3f} (- vmem), p={pf['p']:.3f} (- final_step) 로 적었고 (Holm 보정 후 {pv['p_holm']:.2f} / {pf['p_holm']:.2f}), 위 구간·p 는 Welch 다.", y=-0.087)
    return save(fig, 'ablation_342k')


# ---------------------------------------------------------------------------------------------
# 그림 5. ablation 기전: 1- inversion (발화 집중), loss-ratio (벌점 세기)
# ---------------------------------------------------------------------------------------------
def fig5():
    mt = A['mech_traj']['ch_noinv']
    tr = A['traj']['ch_nolr']
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6))
    # (가) 뉴런 지니, (나) m
    ep_g = sorted(int(k) for k in mt['gini_sw'].keys())
    def series(d, who):
        xs = sorted(int(k) for k in d.keys())
        m = np.array([d[str(e)][who][0] for e in xs]); s = np.array([d[str(e)][who][1] for e in xs])
        p = [d[str(e)]['p'] if 'p' in d[str(e)] else None for e in xs]
        return np.array(xs), m, s, p
    for ax, key, ylabel, ttl in [
        (axes[0, 0], 'gini_sw', '뉴런 지니 (층 스파이크로 가중; 0=고름, 1=몰림)', '(가) 발화가 얼마나 몰려 있나'),
        (axes[0, 1], 'm_mean', 'm = 채널 평균 c/max c (클수록 채널 사이 발화가 고름)', '(나) 채널 사이 발화 비'),
    ]:
        for who, col, mk, lab in [('ours', C_OURS, 'o', 'ours_inter_ch ρ 3e-3 (반복 횟수 4)'),
                                  ('arm', C_INV, 'D', '- 1- inversion ρ 4.5e-3 (반복 횟수 4)')]:
            xs, m, s, p = series(mt[key], who)
            ax.fill_between(xs, m - s, m + s, color=col, alpha=.18, lw=0)
            ax.plot(xs, m, color=col, lw=1.4, marker=mk, ms=6, mec='white', mew=.6, label=lab)
        ax.axvline(200, color='#888888', ls=(0, (3, 2)), lw=.9)
        ax.set_xlabel('에폭 (0 기준)', fontsize=8.8, color=INK2)
        ax.set_ylabel(ylabel, fontsize=8.3, color=INK2)
        ax.set_title(ttl, fontsize=10, color=INK, loc='left')
        style_ax(ax)
        yl = ax.get_ylim()
        ax.text(203, yl[0] + .45 * (yl[1] - yl[0]), '혼합 증강 해제\n(에폭 200)', fontsize=7.2, color=INK2, va='center')
        ax.legend(fontsize=7.8, frameon=False, loc='upper left' if key == 'm_mean' else 'lower right',
                  bbox_to_anchor=(0.30, 0.99) if key == 'm_mean' else (1.0, 0.0))
        # 마지막 에폭 값
        xs, m_o, s_o, _ = series(mt[key], 'ours'); _, m_a, s_a, _ = series(mt[key], 'arm')
        ax.annotate(f'{m_o[-1]:.3f}', (xs[-1], m_o[-1]), xytext=(4, 0), textcoords='offset points', fontsize=7.5, color=C_OURS,
                    va='center')
        ax.annotate(f'{m_a[-1]:.3f}', (xs[-1], m_a[-1]), xytext=(4, 0), textcoords='offset points', fontsize=7.5, color='#b87a00',
                    va='center')
        ax.set_xlim(-8, 335)
    # (다) 실효 rho, (라) lambda
    grp = ['혼합 증강 구간\n(에폭 100~199)', '해제 뒤\n(에폭 200~309)']
    r1 = tr['rho_eff_post_over_pre']
    for ax, kpre, kpost, ylabel, ttl, fmtv, yfmt in [
        (axes[1, 0], 'rho_eff_pre', 'rho_eff_post', '실효 ρ = 벌점 ÷ 과제 손실',
         f"(다) 벌점의 상대 세기 (실효 ρ)\n해제 뒤 ÷ 해제 전: λ 고정 ×{r1['arm'][0]:.2f} ±{r1['arm'][1]:.2f}, loss-ratio 켜짐 ×{r1['ours'][0]:.3f} ±{r1['ours'][1]:.3f} (각 반복 횟수 4)",
         lambda v: f'{v * 1e3:.2f}e-3', lambda v, _: f'{v * 1e3:.2f}e-3'),
        (axes[1, 1], 'lam_pre', 'lam_post', 'λ (벌점 가중치, 단위 1e-7)', '(라) λ: loss-ratio 가 켜지면 λ 가 과제 손실을 따라 내려간다',
         lambda v: f'{v * 1e7:.2f}e-7', lambda v, _: f'{v * 1e7:.1f}'),
    ]:
        for who, col, mk, lab in [('ours', C_OURS, 'o', 'ours_inter_ch ρ 3e-3 — loss-ratio 켜짐 (λ 가 따라 바뀜)'),
                                  ('arm', C_LR, 'P', '- loss-ratio λ 1.25e-7 — λ 고정')]:
            dx = -0.05 if who == 'ours' else 0.05
            y0, s0 = tr[kpre][who]; y1, s1 = tr[kpost][who]
            xs = np.array([0 + dx, 1 + dx])
            ax.errorbar(xs, [y0, y1], yerr=[s0, s1], fmt='none', ecolor=col, elinewidth=1.4, capsize=3.5, zorder=4)
            ax.plot(xs, [y0, y1], color=col, lw=1.4, marker=mk, ms=9, mec='white', mew=.7, label=lab, zorder=5)
            offs = ([(-10, 0, 'right'), (10, 0, 'left')] if who == 'ours' else
                    ([(10, 0, 'left'), (10, 0, 'left')] if kpre == 'rho_eff_pre' else [(0, -15, 'center'), (0, -15, 'center')]))
            for (x_, y_), (ox, oy, ha) in zip(zip(xs, [y0, y1]), offs):
                ax.annotate(fmtv(y_), (x_, y_), xytext=(ox, oy), textcoords='offset points',
                            fontsize=8, color=INK, ha=ha, va='center')
        ax.set_xticks([0, 1]); ax.set_xticklabels(grp, fontsize=8.5)
        ax.set_xlim(-0.55, 1.55)
        ax.yaxis.set_major_formatter(FuncFormatter(yfmt))
        ax.yaxis.get_offset_text().set_visible(False)
        ax.set_ylabel(ylabel, fontsize=8.8, color=INK2)
        ax.set_title(ttl, fontsize=9.6, color=INK, loc='left')
        style_ax(ax)
        yl = ax.get_ylim(); ax.set_ylim(yl[0] - .05 * (yl[1] - yl[0]), yl[1] + .16 * (yl[1] - yl[0]))
        ax.legend(fontsize=7.8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.0))
    fig.suptitle('ablation 기전 (R19 CIFAR-10, 342K 근처 각 반복 횟수 4): - 1- inversion 은 발화가 몰리는 정도를, - loss-ratio 는 벌점 세기의 일정함을 바꾼다',
                 fontsize=11.2, color=INK, x=0.012, ha='left', y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.965))
    sr = A['traj']['ch_noinv']['s_ratio']
    ai = A['acc']['ch_noinv']['lin_srv_all']
    footer(fig, '수치: analysis/ablation_261019.json (mech_traj.ch_noinv, traj.ch_nolr, acc.ch_noinv). 띠·막대 = 시드 SD. (가)(나) 는 학습 모드 통계 (그 에폭 학습 배치). '
                f"정확도 차는 검출되지 않았다 (같은 스파이크 차 {ai['d']:+.2f}%p [{ai['lo']:+.2f}, {ai['hi']:+.2f}], 없다는 뜻은 아님). "
                f"- 1- inversion 의 총 스파이크는 에폭 50 에 {sr['49']['pct']:+.1f}% 였다가 에폭 310 에 {sr['309']['pct']:+.1f}% 로 같아진다. "
                'vmem 의 가중 w 이동은 정의상 값이라 그리지 않았다.', y=0.027)
    return save(fig, 'ablation_mechanism')


# ---------------------------------------------------------------------------------------------
# 그림 6. ablation: 같은 ρ 에서의 스파이크 (ρ 눈금)
# ---------------------------------------------------------------------------------------------
def fig6():
    blk = R['r19_c10']
    cs_all = conds_of(blk)
    b = blk['base']
    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    ax.axhspan(b['spikes'] - blk['conds'][0].get('spikes_sd', 0) if False else 0, 0, alpha=0)  # no-op
    bc = [c for c in cs_all if c['arm'] == 'baseline'][0]
    ax.axhspan(bc['spikes'] - bc['spikes_sd'], bc['spikes'] + bc['spikes_sd'], color='#8a8a8a', alpha=.20, lw=0, zorder=0)
    ax.axhline(bc['spikes'], color='#666666', ls=':', lw=.9, zorder=1)
    ax.text(1.05e-3 * 0.93, bc['spikes'], f"baseline {bc['spikes'] / 1e3:.0f}K ± {bc['spikes_sd'] / 1e3:.0f}K (반복 횟수 {bc['n']})",
            fontsize=7.8, color=INK2, va='bottom', ha='left')
    for arm, (col, mk) in ABL.items():
        if arm == '- loss-ratio':
            continue
        cs = sorted([c for c in cs_all if c['arm'] == arm], key=lambda c: knob_val(c['knob']))
        ax.plot([knob_val(c['knob']) for c in cs], [c['spikes'] for c in cs], color=col, lw=1.2,
                ls='-' if arm == 'ours_inter_ch' else (0, (4, 2)), alpha=.85, zorder=3)
        for c in cs:
            x = knob_val(c['knob'])
            for r in c['runs']:
                ax.scatter([x * (1 + .012 * (1 if SRV_MARK.get(str(r['server'])) == 's' else -1))], [r['spikes']], s=16,
                           marker=SRV_MARK.get(str(r['server']), 'o'), color=col, alpha=.45, linewidths=0, zorder=3)
            sd = c.get('spikes_sd')
            if c['n'] >= 2 and sd is not None:
                ax.errorbar([x], [c['spikes']], yerr=[sd], fmt='none', ecolor=col, elinewidth=1.1, capsize=2.4, zorder=4)
                ax.plot([x], [c['spikes']], marker=mk, ms=8.5, color=col, mfc=col, mec='white', mew=.7, lw=0, zorder=6)
            else:
                ax.plot([x], [c['spikes']], marker=mk, ms=8.5, color=col, mfc='white', mec=col, mew=1.5, lw=0, zorder=6)
    # loss-ratio: 같은 스파이크를 내는 ours rho (가로 구간)
    lr = A['rho']['ch_nolr']
    lrc = [c for c in cs_all if c['arm'] == '- loss-ratio'][0]
    col, mk = ABL['- loss-ratio']
    ax.errorbar([lr['rho_equiv']], [lrc['spikes']], xerr=[[lr['rho_equiv'] - lr['rho_equiv_lo']], [lr['rho_equiv_hi'] - lr['rho_equiv']]],
                yerr=[lrc['spikes_sd']], fmt='none', ecolor=col, elinewidth=1.4, capsize=3, zorder=5)
    ax.plot([lr['rho_equiv']], [lrc['spikes']], marker=mk, ms=10, color=col, mfc=col, mec='white', mew=.7, lw=0, zorder=7)
    ax.annotate(f"- loss-ratio (λ 고정 1.25e-7, 반복 횟수 {lrc['n']}): {lrc['spikes'] / 1e3:.0f}K\n같은 스파이크를 내는 ours ρ = {lr['rho_equiv'] * 1e3:.2f}e-3 "
                f"[{lr['rho_equiv_lo'] * 1e3:.2f}, {lr['rho_equiv_hi'] * 1e3:.2f}]e-3",
                (lr['rho_equiv'], lrc['spikes']), xytext=(1.03e-3, 205e3), textcoords='data', fontsize=7.6, color=INK, ha='left',
                va='center', zorder=10, arrowprops=dict(arrowstyle='-', color='#999999', lw=.6))
    ref_spk = [c for c in cs_all if c['arm'] == 'ours_inter_ch' and c['knob'] == '3e-3'][0]['spikes']
    ax.axhline(ref_spk, color=C_OURS, ls=(0, (1, 2)), lw=.8, alpha=.7, zorder=1)
    ax.text(1.31e-2, ref_spk, f'ours ρ 3e-3 착지 {ref_spk / 1e3:.0f}K', fontsize=7.4, color='#007a59', ha='right', va='bottom')
    # 직접 라벨
    ours = sorted([c for c in cs_all if c['arm'] == 'ours_inter_ch'], key=lambda c: knob_val(c['knob']))
    lbl = {'ours_inter_ch': (C_OURS, 'ours_inter_ch', (1.6e-3, None)),
           }
    ax.set_xscale('log')
    ticks = [1e-3, 1.5e-3, 2e-3, 3e-3, 4.5e-3, 6e-3, 1e-2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1e}'.replace('e-0', 'e-').replace('.0e', 'e')))
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.set_xlim(.9e-3, 1.35e-2)
    ax.set_ylim(130e3, 520e3)
    ax.set_xlabel('ρ (로그 눈금; - loss-ratio 는 ρ 가 아니라 λ 고정이라 ours 로 환산한 값을 찍음)', fontsize=8.8, color=INK2)
    ax.set_ylabel('스파이크 수 (이미지당, 원장 s_count)', fontsize=8.8, color=INK2)
    style_ax(ax)
    ax.text(0.012, 0.012, 'y 축은 130K~520K 만 표시 (0 에서 시작하지 않음)', transform=ax.transAxes, fontsize=7, color=INK2,
            va='bottom', ha='left', bbox=dict(fc='white', ec='none', alpha=.75, pad=1.2))
    # F 요약 (경험적 보간)
    f_txt = ['같은 스파이크에 필요한 ρ 의 비 F = (ours ρ) ÷ (끈 팔 ρ), 경험적 보간 (모형 없음):']
    for arm, k in (('- vmem', 'ch_novmem'), ('- final_step', 'ch_nofinal'), ('- 1- inversion', 'ch_noinv')):
        it = [x for x in A['rho'][k]['interp'] if x.get('F') is not None]
        n_ = {x['rho']: x['n_arm'] for x in it}
        txt = ', '.join(f"ρ {x['rho']} (반복 {x['n_arm']}) ×{x['F']:.2f}" for x in it if x['n_arm'] >= 3)
        f_txt.append(f"  {arm}: {txt}")
    dv = A['rho']['ch_novmem']['direct_same_rho'][0]
    f_txt.append(f"  - vmem 은 같은 ρ {dv['rho']} 직접 비교가 {dv['pct']:+.1f}% [{dv['pct_lo']:+.1f}, {dv['pct_hi']:+.1f}], p={dv['p']:.2f} "
                 f"(반복 횟수 {dv['n_arm']} 대 {dv['n_ours']}) 라 눈금 이동은 불확실하다")
    ax.text(0.5, -0.14, '\n'.join(f_txt), transform=ax.transAxes, fontsize=7.8, color=INK, ha='center', va='top')
    hs = [Line2D([0], [0], marker=ABL[k][1], color=ABL[k][0], mfc=ABL[k][0], mec='white', mew=.7, ms=8, lw=1.0,
                 ls='-' if k == 'ours_inter_ch' else (0, (4, 2)), label=k) for k in ABL]
    hs += [Patch(fc='#8a8a8a', alpha=.35, label='baseline 평균 ± 시드 SD'),
           Line2D([0], [0], marker='o', color='#555555', mfc='white', mec='#555555', mew=1.4, ms=7, lw=0, label='속 빈 마커 = 반복 횟수 1'),
           Line2D([0], [0], marker='o', color='#777777', alpha=.6, ms=5, lw=0, label='개별 런 ○138 □23 (겹침 방지로 가로로 살짝 어긋남)')]
    fig.legend(handles=hs, loc='lower center', ncol=4, fontsize=8.1, frameon=False, bbox_to_anchor=(0.5, -0.115))
    ax.set_title('ablation: 같은 ρ 에서 끄면 스파이크가 어떻게 달라지나 (R19 CIFAR-10) — - final_step·- 1- inversion 은 ρ 눈금이 크게 달라진다',
                 fontsize=10.5, color=INK, loc='left')
    fig.tight_layout()
    footer(fig, '수치: analysis/results_261019.json (conds), analysis/ablation_261019.json (rho.*.interp, rho.ch_nolr). 막대 = 시드 SD. '
                '이 그림은 정확도가 아니라 스파이크 눈금을 보인다 — 정확도 차는 그림 4.', y=-0.128)
    return save(fig, 'ablation_rho_scale')


# ---------------------------------------------------------------------------------------------
# 그림 7. ablation 구조 (평가 모드 npz, 테스트 1만 장): 342K 비교점 각 팔의 런별 값
# ---------------------------------------------------------------------------------------------
ARM_NPZ = [('- vmem', 'ch_novmem'), ('- final_step', 'ch_nofinal'), ('- 1- inversion', 'ch_noinv'),
           ('- loss-ratio', 'ch_nolr')]


def _excl0(o):
    return o['lo'] > 0 or o['hi'] < 0


def _flag(blk, n_arm):
    """보고서 5.5 의 기준: 곡선 ANCOVA 와 보정 직접 둘 다 구간이 0 제외 + ANCOVA 팔 안 Holm < 0.05."""
    an, adj = blk['ancova'], blk['adj']
    if _excl0(an) and _excl0(adj) and an['holm'] < 0.05:
        return '기준 충족'
    if n_arm < 4:
        return f'판단 불가 (반복 {n_arm})'
    if _excl0(an) or _excl0(adj):
        return '판단 불가 (한쪽 분석만 0 제외)'
    return '검출 안 됨'


def fig7():
    N = A['npz']
    PR = N['per_run']
    srv = {r['tag']: str(r['server']) for r in A['runs']}
    XLO, XHI = 295e3, 440e3
    ours_tags = [t for t in N['ours_curve_tags'] if '-ours_ch-3e-3-' in t]
    other_ours = [t for t in N['ours_curve_tags'] if t not in ours_tags]
    # (키, 제목, 런별 값 함수, 통계 형식, 단위 글)
    spec = [
        ('m_eval', '(가) m: 채널 사이 발화 비 (스파이크만, 클수록 채널 사이가 고름)', lambda p: p['m_eval'], '{:+.3f}', ''),
        ('n_gini_sw', '(나) 뉴런 지니 (0 = 고름, 1 = 한곳에 몰림)', lambda p: p['n_gini_sw'], '{:+.3f}', ''),
        ('silent_pct', '(다) 침묵 뉴런 (%, 1만 장 전체에서 한 번도 안 쏜 비율)', lambda p: p['silent_pct'], '{:+.2f}', '%p'),
        ('dead_ch_conv', '(라) 죽은 채널 (conv 4,480개 중, 1만 장에서 한 번도 안 쏨)', lambda p: p['dead_ch_conv'], '{:+.1f}', '개'),
        ('active_img_pct', '(마) 이미지당 활성 뉴런 (%, 이미지 하나에서 쏜 비율의 평균)', lambda p: p['active_img_pct'], '{:+.2f}', '%p'),
        ('t_share4', '(바) 스텝 4 에 쏜 몫 (%, conv+fc1 합)', lambda p: p['t_share'][3] if 't_share' in p else None, '{:+.2f}', '%p'),
    ]
    fig = plt.figure(figsize=(16.0, 14.2))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.26, left=0.055, right=0.99, top=0.935, bottom=0.26)
    for i, (key, title, getv, sfmt, unit) in enumerate(spec):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        allv = []
        # 회색: ours 다른 ρ 의 런 (곡선 ANCOVA 의 기준 점)
        for t in other_ours:
            p = PR[t]; v = getv(p)
            if v is None or not (XLO <= p['npz_spk'] <= XHI):
                continue
            ax.scatter([p['npz_spk']], [v], s=17, marker=SRV_MARK.get(srv[t], 'o'), color='#9a9a9a', alpha=.55,
                       linewidths=0, zorder=2)
            allv.append(v)
        # 기준: ours ρ 3e-3 / 끈 팔 342K 세트 런별
        groups = [('ours_inter_ch', ours_tags, C_OURS, 'o')]
        for (nm, ak) in ARM_NPZ:
            groups.append((nm, N[ak]['tags_pt'], ABL[nm][0], ABL[nm][1]))
        for nm, tags, col, mk in groups:
            for t in tags:
                p = PR[t]; v = getv(p)
                if v is None:
                    continue
                ax.scatter([p['npz_spk']], [v], s=26, marker=SRV_MARK.get(srv[t], 'o'), color=col, alpha=.6,
                           linewidths=0, zorder=3)
                allv.append(v)
        # 곡선 ANCOVA 에만 들어간 다른 ρ 의 끈 팔 런 (연하게)
        for nm, ak in ARM_NPZ:
            col = ABL[nm][0]
            for t in N[ak]['tags_all']:
                if t in N[ak]['tags_pt']:
                    continue
                p = PR[t]; v = getv(p)
                if v is None or not (XLO <= p['npz_spk'] <= XHI):
                    continue
                ax.scatter([p['npz_spk']], [v], s=22, marker=SRV_MARK.get(srv[t], 'o'), facecolor='white',
                           edgecolor=col, linewidths=.9, alpha=.8, zorder=3)
                allv.append(v)
        # 평균 +- SD (가로 SD 포함)
        for nm, ak in [('ours_inter_ch', None)] + ARM_NPZ:
            col, mk = ABL[nm]
            if key == 't_share4':
                st = N['ch_noinv']['t_share']['steps'][3]
                if ak is None:
                    ym, ys = st['ours']
                else:
                    ym, ys = N[ak]['t_share']['steps'][3]['arm']
            else:
                if ak is None:
                    ym, ys = N['ours_pt'][key]
                else:
                    ym, ys = N[ak][key]['arm']
            xm, xs = N['ours_pt']['npz_spk'] if ak is None else N[ak]['npz_spk']['arm']
            ax.errorbar([xm], [ym], xerr=[xs], yerr=[ys], fmt='none', ecolor=col, elinewidth=1.1, capsize=2.2, zorder=5)
            ax.plot([xm], [ym], marker=mk, ms=10 if nm != '- loss-ratio' else 11, color=col, mfc=col, mec='white',
                    mew=.8, lw=0, zorder=6)
            allv += [ym - ys, ym + ys]
        lo, hi = min(allv), max(allv)
        d = (hi - lo) or 1.0
        ax.set_ylim(lo - .08 * d, hi + .08 * d)
        ax.set_xlim(XLO, XHI)
        ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
        ax.set_xlabel('npz 스파이크 수 (테스트 1만 장, 이미지당)', fontsize=8.2, color=INK2)
        ax.set_title(title, fontsize=9.2, color=INK, loc='left')
        style_ax(ax)
        # 통계 글
        if key == 't_share4':
            lines = ['끈 팔 − ours_inter_ch 스텝 4 몫 (%p, Welch 95% 구간, 보정 없음):']
            for nm, ak in ARM_NPZ:
                st = N[ak]['t_share']['steps'][3]
                dd = st['direct']
                lines.append(f"{nm} (반복 {dd['na']} 대 {dd['nb']}): {dd['d']:+.2f} [{dd['lo']:+.2f}, {dd['hi']:+.2f}], p={dd['p']:.2g}")
            lines.append('ours_inter_ch 는 새 추출기 3런 (s2~s4) 만 스텝 배열이 있다')
        else:
            lines = ['끈 팔 − ours_inter_ch, 같은 npz 스파이크 (곡선 ANCOVA 95% 구간,',
                     '  n = 끈 팔 / ours 곡선 19):']
            for nm, ak in ARM_NPZ:
                b = N[ak][key]
                an = b['ancova']
                lines.append(f"{nm} (n={an['n_arm']}): " + sfmt.format(an['d']) + ' [' + sfmt.format(an['lo']) +
                             ', ' + sfmt.format(an['hi']) + f"] {unit} → {_flag(b, an['n_arm'])}")
        ax.text(0.0, -0.20, '\n'.join(lines), transform=ax.transAxes, fontsize=7.1, color=INK, ha='left', va='top')
    hs = [Line2D([0], [0], marker=ABL[k][1], color=ABL[k][0], mfc=ABL[k][0], mec='white', mew=.7, ms=9, lw=0, label=k)
          for k in ABL]
    hs += [Line2D([0], [0], marker='o', color='#9a9a9a', alpha=.7, ms=5, lw=0,
                  label='회색 = ours_inter_ch 다른 ρ (이 창 안은 1.5e-3~5e-3), 곡선 ANCOVA 의 기준'),
           Line2D([0], [0], marker='o', color='#555555', mfc='white', mec='#555555', mew=1.0, ms=6, lw=0,
                  label='속 빈 작은 마커 = 곡선 ANCOVA 에만 들어간 다른 ρ 의 끈 팔 런'),
           Line2D([0], [0], marker='o', color='#777777', alpha=.6, ms=5, lw=0, label='작은 점 = 런 하나 (○ 서버 138, □ 서버 23)'),
           Line2D([0], [0], marker='o', color='#555555', mec='white', ms=10, lw=0, label='큰 마커 = 평균 ± SD (가로·세로)')]
    fig.legend(handles=hs, loc='lower center', ncol=4, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, 0.075))
    fig.suptitle('ablation 구조 (평가 모드, R19 CIFAR-10): 같은 스파이크에서 기준을 넘어 달라진 것은 - 1- inversion 뿐 '
                 '(- loss-ratio 는 반복 3 이라 판단 불가)', fontsize=11.8, color=INK, x=0.012, ha='left', y=0.985)
    footer(fig, '수치: analysis/ablation_261019.json (npz.per_run 런별 값, npz.<팔>.<지표> 평균·SD·ancova·adj, npz.<팔>.t_share). '
                '큰 마커 = 342K 비교점 (ours_inter_ch ρ 3e-3 반복 4, - vmem ρ 2e-3 반복 4, - final_step ρ 1.4e-3 반복 4, '
                '- 1- inversion ρ 4.5e-3 반복 4, - loss-ratio λ 1.25e-7 반복 3: s1 은 세종 런이라 추출 안 함). '
                '기준 충족 = 곡선 ANCOVA 와 스파이크 보정 직접 비교 둘 다 구간이 0 제외 + ANCOVA 팔 안 Holm < 0.05. '
                '검출 안 됨은 같다는 뜻이 아니다. 끈 팔 17런은 모두 10월 시작, ours 곡선 19런은 9월 13 / 10월 6 이고 이 실행 시기는 통제하지 않았다 '
                '(보고서 5.5 시기 점검). 가로축은 295K~440K 만 표시 (ours 곡선의 220~283K 점, ρ 5.5e-3~8e-3 은 밖), '
                'y 축은 패널마다 값 범위만 표시 (0 에서 시작하지 않음). baseline (반복 3, 507K) 은 스파이크가 40% 많아 같은 스파이크 비교가 아니라 뺐다.',
           y=0.052)
    return save(fig, 'ablation_structure_eval')


if __name__ == '__main__':
    import sys
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', '6', '7']
    fn = {'1': fig1, '2': fig2, '3': fig3, '4': fig4, '5': fig5, '6': fig6, '7': fig7}
    for w in which:
        fn[w]()
