"""파레토 그림 5장 (26-10-20). 수치는 analysis/results_261019.json (eip-stats 확정본) 만 읽는다.

원본 CSV·원장 인벤토리를 직접 읽지 않는다. 평균·SD·반복 횟수는 json 의 conds 값을 그대로 쓴다.
x = 샘플당 스파이크 수 (원값), y = val acc (원값). 차이(delta)로 그리지 않는다.
스타일은 EIP_fig/pareto_261012/plot.py 의 fig() 와 같다 (색·마커·밴드·선).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
SRC = ROOT + '/analysis/results_261019.json'
OUT = os.path.dirname(os.path.abspath(__file__)) + '/'
DATE = '26-10-20'
D = json.load(open(SRC))

# ---- 스타일 (pareto_261012/plot.py 와 동일 팔레트, plain L2 = 팔레트 3번째 aqua) ----
MS = 11
SURF = '#fcfcfb'; INK = '#1f1f1d'; INK2 = '#5f5e58'; GRID = '#e6e5e0'
OURS = '#2a78d6'; BPSR = '#eb6834'; L2C = '#1baf7a'; BASE = '#6b6a64'
EXCL = '#9a9992'
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
plt.rcParams.update({'font.family': ['Noto Sans CJK JP', 'DejaVu Sans'], 'font.size': 10,
                     'axes.unicode_minus': False})

S30_MIN = 0.19  # 원장 배제 기준 (S30/S1 < 0.19 는 그리지 않는다)


def group(conds, arm):
    """json conds 에서 한 팔의 조건들을 스파이크 오름차순으로. 평균·SD·n 은 json 값."""
    out = []
    for c in conds:
        if c['arm'] != arm:
            continue
        for r in c['runs']:
            s30 = r.get('s30')
            assert s30 is None or s30 >= S30_MIN, ('배제 기준 미달 런이 conds 에 있음', r['tag'])
        out.append(dict(k=c['knob'], n=c['n'], a=c['val_acc'], sa=(c['val_acc_sd'] or 0.0), s=c['spikes'],
                        pts=[(r['spikes'], r['val_acc']) for r in c['runs']]))
        assert len(c['runs']) == c['n']
    return sorted(out, key=lambda d: d['s'])


def base_of(block, base_conds_arm='baseline'):
    b = block['base']
    bc = [c for c in block['conds'] if c['arm'] == base_conds_arm][0]
    assert bc['n'] == b['n']
    return dict(n=b['n'], a=b['acc'], sa=b['acc_sd'], s=b['spikes'], ss=bc['spikes_sd'] or 0.0,
                pts=[(r['spikes'], r['val_acc']) for r in bc['runs']])


def fig(title, base, arms, fname, hi=None, excl=None, notes='', legend_loc='lower right',
        size=(8.6, 5.5), note_hi=None, label_all=True):
    """arms: [(label, groups, color, marker, knobname)].
    hi: {label: set(knob)} 표1 지점 (고리 + 굵은 라벨).
    excl: [(spikes, acc, text)] 배제 런 (× 표시, 선에 연결 안 함).
    note_hi: {(label,knob): 추가 라벨 문구}"""
    hi = hi or {}
    note_hi = note_hi or {}
    f, ax = plt.subplots(figsize=size, dpi=200)
    f.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    bm, bsd = base['a'], base['sa']
    ax.axhspan(bm - bsd, bm + bsd, color=BASE, alpha=0.10, lw=0)
    ax.axhline(bm, color=BASE, lw=1, ls=(0, (4, 3)))
    ax.scatter([p[0] for p in base['pts']], [p[1] for p in base['pts']], s=14, color=BASE, alpha=0.35, lw=0, zorder=2)
    ax.errorbar([base['s']], [bm], xerr=[base['ss']], yerr=[bsd], fmt='D', ms=8, color=BASE, mec=SURF, mew=2,
                ecolor=BASE, elinewidth=1.5, capsize=0, zorder=4)

    # (x, y, 반지름 px) — 라벨이 피해야 할 점들
    marks = [(base['s'], bm, 16)] + [(p[0], p[1], 5) for p in base['pts']]
    labels = []  # 배치할 라벨
    lines_px = [[(base['s'], bm - bsd), (base['s'], bm + bsd)], [(base['s'] - base['ss'], bm), (base['s'] + base['ss'], bm)]]
    for lab, g, col, mk, knobname in arms:
        xs = [d['s'] for d in g]; ys = [d['a'] for d in g]
        for d in g:
            ax.scatter([p[0] for p in d['pts']], [p[1] for p in d['pts']], s=14, color=col, alpha=0.30, lw=0, zorder=2)
            for p in d['pts']:
                marks.append((p[0], p[1], 5))
        ax.plot(xs, ys, color=col, lw=2, zorder=3)
        lines_px.append(list(zip(xs, ys)))
        for gi, d in enumerate(g):
            if d['n'] >= 2:
                ax.errorbar([d['s']], [d['a']], yerr=[d['sa']], fmt=mk, ms=MS, color=col, mec=SURF, mew=1.6,
                            ecolor=col, elinewidth=1.2, capsize=0, zorder=4)
                tcol = SURF
            else:  # 반복 1회 = 속 빈 점
                ax.errorbar([d['s']], [d['a']], fmt=mk, ms=MS, mfc=SURF, mec=col, mew=1.8, color=col,
                            ecolor=col, capsize=0, zorder=4)
                tcol = col
            # 점 안에 반복 횟수
            ax.annotate(str(d['n']), (d['s'], d['a']), xytext=(0, -1.3 if mk == '^' else -0.2), textcoords='offset points',
                        ha='center', va='center', fontsize=6.6, fontweight='bold', color=tcol, zorder=6)
            marks.append((d['s'], d['a'], 15))
            if d['n'] >= 2:
                lines_px.append([(d['s'], d['a'] - d['sa']), (d['s'], d['a'] + d['sa'])])  # 오차 막대도 피한다
            is_hi = d['k'] in hi.get(lab, set())
            if is_hi:
                ax.scatter([d['s']], [d['a']], s=470, facecolors='none', edgecolors=INK, linewidths=1.2, zorder=5)
            ends = gi == 0 or gi == len(g) - 1
            if not (label_all or is_hi or ends):
                continue
            txt = f"{knobname} {d['k']}"
            if is_hi:
                txt += ' · 표1'
            if (lab, d['k']) in note_hi:
                txt += ' · ' + note_hi[(lab, d['k'])]
            labels.append(dict(x=d['s'], y=d['a'], t=txt, col=col, bold=is_hi))
    if excl:
        for (sx, sy, t) in excl:
            ax.scatter([sx], [sy], marker='X', s=70, color=EXCL, edgecolors=SURF, linewidths=1.2, zorder=4)
            marks.append((sx, sy, 12))
            labels.append(dict(x=sx, y=sy, t=t, col=EXCL, bold=False))

    # 축 범위: 데이터 + 오차 막대 + 밴드 기준. 위아래 여백만 둔다 (y 는 0 부터가 아님 -> 아래 notes 에 적음)
    ally = [bm - bsd, bm + bsd] + [p[1] for p in base['pts']]
    allx = [base['s'] - base['ss'], base['s'] + base['ss']] + [p[0] for p in base['pts']]
    for lab, g, *_ in arms:
        for d in g:
            ally += [d['a'] - d['sa'], d['a'] + d['sa']] + [p[1] for p in d['pts']]
            allx += [p[0] for p in d['pts']] + [d['s']]
    if excl:
        ally += [e[1] for e in excl]; allx += [e[0] for e in excl]
    y0, y1 = min(ally), max(ally); x0, x1 = min(allx), max(allx)
    py = (y1 - y0) * 0.12; px_ = (x1 - x0) * 0.07
    ax.set_ylim(y0 - py, y1 + py); ax.set_xlim(x0 - px_ * 1.4, x1 + px_)

    ax.set_title(title, loc='left', color=INK, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('샘플당 스파이크 수', color=INK2); ax.set_ylabel('val acc (%)', color=INK2)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f'{v/1000:.0f}K'))

    h = [Line2D([0], [0], color=BASE, marker='D', ls=(0, (4, 3)), ms=7, label='baseline (평균 ± SD)')]
    h += [Line2D([0], [0], color=c, marker=m, lw=2, ms=7, label=l) for l, _, c, m, _k in arms]
    if hi:
        h += [Line2D([0], [0], marker='o', ls='', ms=11, mfc='none', mec=INK, mew=1.2, label='표 1 지점')]
    leg = ax.legend(handles=h, frameon=False, loc=legend_loc, fontsize=9, labelcolor=INK2)

    f.tight_layout(rect=[0, 0.135, 1, 1])
    f.canvas.draw()
    labels.insert(0, dict(x=base['s'], y=bm, t=f"baseline (반복 횟수 {base['n']})", col=INK2, bold=False, fs=9, first=True, force_leader=True))
    lines_px.append([(ax.get_xlim()[0], bm), (ax.get_xlim()[1], bm)])  # baseline 점선도 피한다
    place_labels(f, ax, labels, marks, lines_px, leg)
    f.text(0.012, 0.012, notes, fontsize=7.6, color=INK2, va='bottom', ha='left', linespacing=1.5)
    path = OUT + fname
    f.savefig(path, facecolor=SURF)
    plt.close(f)
    print('saved', path)


def place_labels(f, ax, labels, marks, lines, leg, fs=7.4):
    """라벨을 겹치지 않게 놓는다. 후보 위치를 돌며 (다른 라벨·점·선·범례·baseline 문구)와 겹침이 가장 적은 곳."""
    r = f.canvas.get_renderer()
    k = f.dpi / 72.0
    T = ax.transData
    axbb = ax.get_window_extent(r)
    placed = [tuple(leg.get_window_extent(r).extents)]
    mpx = [(*T.transform((x, y)), rad) for x, y, rad in marks]
    # 선분을 3px 간격으로 샘플
    samples = []
    for pts in lines:
        P = [T.transform(p) for p in pts]
        for (a, b) in zip(P[:-1], P[1:]):
            n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 3))
            for t in np.linspace(0, 1, n):
                samples.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
    samples = np.array(samples) if samples else np.zeros((0, 2))

    probe = f.text(0, 0, 'x', fontsize=fs)
    def size_of(t, bold, fsz):
        probe.set_text(t); probe.set_fontweight('bold' if bold else 'normal'); probe.set_fontsize(fsz)
        bb = probe.get_window_extent(r)
        return bb.width, bb.height

    def ov(a, b):
        w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
        return w * h if (w > 0 and h > 0) else 0.0

    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    # 가까운 칸부터. 바깥 칸은 지시선(leader)을 그린다.
    ring = [(1.0, 0), (1.9, 0), (2.9, 1), (4.0, 1), (5.2, 1)]
    order = sorted(range(len(labels)), key=lambda i: (not labels[i].get('first', False), labels[i]['x']))
    for i in order:
        L = labels[i]
        fsz = L.get('fs', fs)
        w, h = size_of(L['t'], L['bold'], fsz)
        cx, cy = T.transform((L['x'], L['y']))
        best = None
        for (dist, leader) in (ring[1:] if L.get('first') else ring):
            for (dx, dy) in dirs:
                gap = 9.0 * dist * k
                ox = dx * gap * (0.78 if dx and dy else 1.0)
                oy = dy * gap * (0.78 if dx and dy else 1.0)
                ax0 = cx + ox
                ay0 = cy + oy
                x0 = ax0 if dx > 0 else (ax0 - w if dx < 0 else ax0 - w / 2)
                y0 = ay0 if dy > 0 else (ay0 - h if dy < 0 else ay0 - h / 2)
                box = (x0, y0, x0 + w, y0 + h)
                sc = 0.0
                for p in placed:
                    sc += 3.0 * ov(box, p)
                for (mx, my, rad) in mpx:
                    sc += 4.0 * ov((box[0] - 7, box[1] - 4, box[2] + 7, box[3] + 4), (mx - rad, my - rad, mx + rad, my + rad))
                if len(samples):
                    ins = ((samples[:, 0] > box[0]) & (samples[:, 0] < box[2]) & (samples[:, 1] > box[1]) & (samples[:, 1] < box[3])).sum()
                    sc += 25.0 * ins
                out = w * h - ov(box, tuple(axbb.extents))
                sc += 8.0 * out
                if leader or L.get('force_leader'):   # 지시선이 이미 놓인 라벨을 가로지르면 감점
                    for t in np.linspace(0.15, 1.0, 9):
                        qx = cx + ox * t; qy = cy + oy * t
                        for pb in placed:
                            if pb[0] <= qx <= pb[2] and pb[1] <= qy <= pb[3]:
                                sc += 150.0
                sc += 40.0 * dist                      # 가까울수록 선호
                if dx == 0 and dy == -1: sc += 15      # 같은 거리면 위/오른쪽 우선
                if best is None or sc < best[0]:
                    best = (sc, dx, dy, dist, leader, box, ox, oy)
            if best is not None and best[0] < 60 * ring[0][0] + 5:
                break  # 충분히 깨끗하면 바깥 칸은 보지 않는다
        sc, dx, dy, dist, leader, box, ox, oy = best
        placed.append(box)
        ha = 'left' if dx > 0 else ('right' if dx < 0 else 'center')
        va = 'bottom' if dy > 0 else ('top' if dy < 0 else 'center')
        ap = dict(arrowstyle='-', color=(BASE if L.get('force_leader') else L['col']), lw=0.8, alpha=0.8, shrinkA=0, shrinkB=8) if (leader or L.get('force_leader')) else None
        ax.annotate(L['t'], (L['x'], L['y']), xytext=(ox / k, oy / k), textcoords='offset points', ha=ha, va=va,
                    fontsize=fsz, color=INK if L['bold'] else INK2, fontweight='bold' if L['bold'] else 'normal',
                    arrowprops=ap, zorder=6)
    probe.remove()


NOTE_A = ('큰 점 = 같은 ρ/λ 의 평균 ± SD (점 안 숫자 = 반복 횟수), 작은 점 = 개별 런, 속 빈 점 = 반복 횟수 1. '
          '왼쪽 위일수록 좋음.\n')
NOTE_Y = 'y축은 0 부터가 아니라 데이터 구간만 확대함. 배제 런(S30/S1 < 0.19)·진행 중 런은 그리지 않음.'


def check(cond, msg):
    if not cond:
        raise SystemExit('수치 불일치: ' + msg)


# ============ 1. ResNet-19 CIFAR-10 ============
b = D['r19_c10']
base = base_of(b)
check(base['n'] == 6 and abs(base['a'] - 96.73) < 0.005 and abs(base['sa'] - 0.10) < 0.005 and round(base['s']) == 489129, 'r19_c10 base')
o = group(b['conds'], 'ours_inter_ch'); p = group(b['conds'], 'BPSR'); l = group(b['conds'], 'plain L2')
check((len(o), len(p), len(l)) == (11, 4, 6), f'r19_c10 조건 수 {len(o), len(p), len(l)}')
check([d['n'] for d in p if d['k'] == '1e-10'] == [4] and round([d['s'] for d in p if d['k'] == '1e-10'][0]) == 322503, 'BPSR 1e-10')
fig('ResNet-19 · CIFAR-10 — ours_inter_ch vs BPSR vs plain L2 vs baseline', base,
    [('ours_inter_ch', o, OURS, 'o', 'ρ'), ('BPSR', p, BPSR, 's', 'λ'), ('plain L2', l, L2C, '^', 'λ')],
    f'pareto_r19_c10_ko_{DATE}.png',
    hi={'ours_inter_ch': {'3e-3'}, 'BPSR': {'1e-10'}, 'plain L2': {'1e-7'}},
    notes=NOTE_A + '표 1 = ours_inter_ch ρ 3e-3 · BPSR λ 1e-10 · plain L2 λ 1e-7 (각 반복 횟수 4). 값 표기는 팔마다 양 끝 점과 표 1 지점만 (ours_inter_ch = ρ, 나머지 = λ).\n' + NOTE_Y,
    label_all=False,
    legend_loc='lower right', size=(9.2, 5.8))

# ============ 2. ResNet-19 CIFAR-100 ============
b = D['r19_c100']
base = base_of(b)
check(base['n'] == 6 and abs(base['a'] - 80.65) < 0.005 and abs(base['sa'] - 0.22) < 0.005 and round(base['s']) == 450599, 'r19_c100 base')
o = group(b['conds'], 'ours_inter_ch'); p = group(b['conds'], 'BPSR'); l = group(b['conds'], 'plain L2')
check((len(o), len(p), len(l)) == (6, 3, 8), f'r19_c100 조건 수 {len(o), len(p), len(l)}')
check([d['n'] for d in o if d['k'] == '1e-3'] == [4] and round([d['s'] for d in o if d['k'] == '1e-3'][0]) == 395380, 'ours 1e-3')
fig('ResNet-19 · CIFAR-100 — ours_inter_ch vs BPSR vs plain L2 vs baseline', base,
    [('ours_inter_ch', o, OURS, 'o', 'ρ'), ('BPSR', p, BPSR, 's', 'λ'), ('plain L2', l, L2C, '^', 'λ')],
    f'pareto_r19_c100_ko_{DATE}.png',
    notes=NOTE_A + '값 표기는 팔마다 양 끝 점만 (ours_inter_ch = ρ, 나머지 = λ).\n' + NOTE_Y,
    label_all=False,
    legend_loc='lower right', size=(9.2, 5.8))

# ============ 3. QKFormer CIFAR-10 (fp32 재측정) ============
b = D['qk_c10']
base = base_of(b)
check(base['n'] == 6 and abs(base['a'] - 96.02) < 0.005 and abs(base['sa'] - 0.14) < 0.005 and round(base['s']) == 936126, 'qk_c10 base')
o = group(b['conds'], 'ours_inter_ch')
check(sorted(d['k'] for d in o) == sorted(['1e-3', '2e-3', '3e-3', '5e-3', '7e-3', '1.2e-2', '2e-2']), 'qk_c10 ρ 목록')
d12 = [d for d in o if d['k'] == '1.2e-2'][0]
check(d12['n'] == 4 and round(d12['s']) == 662441 and abs(d12['a'] - 95.965) < 0.001, 'qk_c10 1.2e-2')
fig('QKFormer · CIFAR-10 — ours_inter_ch vs baseline (fp32 재측정)', base,
    [('ours_inter_ch', o, OURS, 'o', 'ρ')],
    f'pareto_qk_c10_ko_{DATE}.png',
    note_hi={('ours_inter_ch', '2e-2'): '+3 진행 중'},
    notes=NOTE_A + 'baseline = 시드 42·43·45·46·47·48 (s44 는 재측정값이 없어 제외). 스파이크는 fp32 재측정값.\n' + NOTE_Y,
    legend_loc='lower left', size=(8.6, 5.5))

# ============ 4. QKFormer CIFAR-100 ============
b = D['qk_c100']
base = base_of(b)
check(base['n'] == 4 and abs(base['a'] - 80.96) < 0.005 and abs(base['sa'] - 0.14) < 0.006 and round(base['s']) == 1013205, 'qk_c100 base')
o = group(b['conds'], 'ours_inter_ch')
check(sorted(d['k'] for d in o) == ['1.2e-2', '6e-3'] and all(d['n'] == 4 for d in o), 'qk_c100 ρ')
fig('QKFormer · CIFAR-100 — ours_inter_ch vs baseline', base,
    [('ours_inter_ch', o, OURS, 'o', 'ρ')],
    f'pareto_qk_c100_ko_{DATE}.png',
    notes=NOTE_A + '스파이크는 fp32 재측정값.\n' + NOTE_Y,
    legend_loc='lower right', size=(8.6, 5.5))

# ============ 5. VGGSNN CIFAR10-DVS 첫 프레임 레시피 ============
b = D['dvs_f0']
base = base_of(b)
check(base['n'] == 4 and abs(base['a'] - 82.31) < 0.005 and abs(base['sa'] - 0.54) < 0.005 and round(base['s']) == 262018, 'dvs_f0 base')
o = group(b['conds'], 'ours_inter_ch')
check(sorted(d['k'] for d in o) == sorted(['1e-3', '2e-3', '3e-3', '5e-3', '1e-2', '2e-2']), 'dvs ρ')
check([d['n'] for d in sorted(o, key=lambda d: d['s'])] == [1, 4, 4, 4, 4, 4], 'dvs n')
ex = b['excluded']
check(len(ex) == 1 and ex[0]['knob'] == '3e-2', 'dvs excluded')
excl = [(ex[0]['spikes'], ex[0]['val_acc'], f"ρ {ex[0]['knob']} · 배제 (S30/S1 < 0.19)")]
fig('VGGSNN · CIFAR10-DVS — 첫 프레임 레시피, ours_inter_ch vs baseline', base,
    [('ours_inter_ch', o, OURS, 'o', 'ρ')],
    f'pareto_dvs_f0_ko_{DATE}.png', excl=excl,
    notes=NOTE_A + '× = 배제 런(곡선에 연결 안 함). 검증셋이 992장이라 정확도는 약 0.1%p 단위로 계단.\n' + NOTE_Y,
    legend_loc='lower right', size=(8.6, 5.5))
