"""booktabs LaTeX 표 생성기 (Task 11, 2026-09-14).

`collect_paper.py` 가 낸 CSV (`_results/paper.csv` 등)를 읽어 `.tex` 조각을 낸다.
Task 12·13 이 같은 스크립트에 표를 덧붙인다 — 표 하나마다 `TABLES` 딕셔너리에
`--table` 이름과 생성 함수를 등록하고, 행 서식은 `render_group`/`fmt_*` 를 재사용한다.

열 순서는 스펙 §6 이 고정한다: train loss -> train acc -> val loss -> val acc -> spikes.
반복 런(같은 방법·같은 강도)은 개별 행을 모두 내고, 그룹 바로 아래 줄에 평균 행을 낸다
(그룹↔평균 사이는 `\\midrule` 이 아니라 `\\cmidrule`; 그룹과 그룹 사이는 `\\midrule`).

사용례:
    python paper/tables/gen_tables.py --in _results/paper.csv --table t1
"""
import argparse
import csv
import os
import re
from collections import OrderedDict

# ---------------------------------------------------------------------------
# 공통 서식 유틸 — Task 12/13 이 그대로 재사용한다.
# ---------------------------------------------------------------------------

# run_dir 안의 방법·강도 표기를 논문 표기로 바꾼다 (docs/paper/method-terms.md).
METHOD_LABELS = {
    'base': 'No regularisation',
    'prop': 'Ours',
    'l2': 'Plain $L_2$',
    'sm': '1-softmax',
}


def fmt_knob(raw):
    """'3e-3' / '2p5e-7' -> '3\\times10^{-3}' 형태의 과학적 표기.

    디렉토리 이름은 소수점을 못 쓰므로 'p' 로 대신한다 (2p5e-7 = 2.5e-7).
    """
    raw = re.sub(r'(\d)p(\d)', r'\1.\2', raw)
    m = re.match(r'^([0-9.]+)e([+-]?\d+)$', raw)
    if not m:
        return raw
    mantissa, exponent = m.group(1), int(m.group(2))
    return f'{mantissa}\\times10^{{{exponent}}}'


def classify(run_dir):
    """run_dir 경로에서 (그룹 키, 표에 쓸 조건 이름) 을 뽑는다.

    baseline 5런은 다섯 디렉토리에 흩어져 있어(스펙 §2 A1) 공통 접두사가 없다 —
    'base'(정본 서술) 또는 'baseline'(디렉토리명) 부분 문자열로 잡는다.
    """
    m = re.search(r'-(prop|l2|sm)-(.+?)-s\d+$', run_dir)
    if m:
        method, knob_raw = m.group(1), m.group(2)
        label = METHOD_LABELS[method]
        knob = fmt_knob(knob_raw)
        if method == 'prop':
            cond = f'{label} ($\\rho={knob}$)'
        else:
            cond = f'{label} ($\\lambda={knob}$)'
        return (method, knob_raw), cond
    if 'base' in run_dir.lower():
        return ('base', None), METHOD_LABELS['base']
    raise ValueError(f'run_dir 를 방법으로 분류하지 못함: {run_dir}')


def fmt_loss(v):
    return f'{float(v):.4f}'


def fmt_acc(v, decimals=2):
    return f'{float(v):.{decimals}f}'


def fmt_spikes(v):
    return f'{round(float(v)):,}'


def render_row(cond, r):
    return (f'{cond} & {fmt_loss(r["train_loss"])} & {fmt_acc(r["train_acc"])} & '
            f'{fmt_loss(r["val_loss"])} & {fmt_acc(r["best_val_acc"])} & '
            f'{fmt_spikes(r["s_count"])} \\\\')


def render_mean_row(n):
    def inner(rows):
        mean = lambda k: sum(float(r[k]) for r in rows) / len(rows)
        return (f'Mean ($n={n}$) & {fmt_loss(mean("train_loss"))} & '
                f'{fmt_acc(mean("train_acc"), 3)} & {fmt_loss(mean("val_loss"))} & '
                f'{fmt_acc(mean("best_val_acc"), 3)} & {fmt_spikes(mean("s_count"))} \\\\')
    return inner


def render_group(cond, rows, lines):
    """그룹의 개별 행 + (n>1 이면) 평균 행을 `lines` 에 덧붙인다."""
    for r in rows:
        lines.append(render_row(cond, r))
    if len(rows) > 1:
        lines.append('\\cmidrule{1-6}')
        lines.append(render_mean_row(len(rows))(rows))


def group_rows(rows):
    """분류 키의 첫 등장 순서를 유지하며 묶는다 (입력이 미리 그룹 인접이 아니어도 안전)."""
    groups = OrderedDict()
    for r in rows:
        key, cond = classify(r['run_dir'])
        groups.setdefault(key, {'cond': cond, 'rows': []})['rows'].append(r)
    return groups


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


HEADER = (
    'Condition & Train loss & Train acc (\\%) & Val loss & Val acc (\\%) & Spikes \\\\'
)


def wrap_table(caption, label, body_lines):
    return '\n'.join([
        '\\begin{table}',
        '\\centering',
        '\\small',
        '\\begin{tabular}{lccccc}',
        '\\toprule',
        HEADER,
        '\\midrule',
        *body_lines,
        '\\bottomrule',
        '\\end{tabular}',
        f'\\caption{{{caption}}}',
        f'\\label{{{label}}}',
        '\\end{table}',
        '',
    ])


# ---------------------------------------------------------------------------
# T1 — R19-C10 주 비교표 (스펙 §2 A1/A1b, figures-tables.md T1)
# ---------------------------------------------------------------------------

def gen_t1(rows, out_path):
    groups = group_rows(rows)
    lines = []
    first = True
    for key, g in groups.items():
        if not first:
            lines.append('\\midrule')
        first = False
        render_group(g['cond'], g['rows'], lines)
    caption = (
        'ResNet-19 / CIFAR-10, best-val\\_acc epoch, eval/test spike counts. '
        'Repeated runs show every individual run with the group mean directly '
        'below (pre-registered selection rules, spec \\S 2 A1/\\S 12-1).'
    )
    tex = wrap_table(caption, 'tab:t1_main', lines)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f'저장: {out_path}')


# ---------------------------------------------------------------------------
# T3 -- Ablation, tier A: same rho=4e-3 (스펙 §12-4 tier A, §2 A5/A6/A9,
# figures-tables.md T3, docs/paper/method-terms.md for row names)
# ---------------------------------------------------------------------------

# 논문 표기는 docs/paper/method-terms.md 오른쪽 열만 쓴다 (코드 플래그를 본문/표에 노출 금지).
ABL_LABELS = {
    'prop': 'Ours',
    'abl_novmem': 'w/o potential',
    'abl_nofinal': 'w/o final-step',
    'abl_noinv': 'w/o inversion',
    'abl_nolr': 'fixed $\\lambda$',
}

# 정확도 열에 쓰는 개별 런 — 선별 규칙(스펙 §12-1) 통과분. prop 4e-3 은 Task 11 의 T1 과
# 동일한 4런(s1,s2,s4,s5; s3 는 규칙 3 으로 배제, runs-t1.txt 참조)을 그대로 쓴다.
# abl_novmem-s5 는 305에폭(규칙 1), abl_noinv-s4 는 0에폭(규칙 1)으로 collect_paper.py 가
# 이미 배제했다 — 규칙 3(상위 4개) 이 아니라 규칙 1(미완주) 적용.
T3_TIER_A_RUNS = OrderedDict([
    ('prop', ['_paper/r19c10-prop-4e-3-s1', '_paper/r19c10-prop-4e-3-s2',
              '_paper/r19c10-prop-4e-3-s4', '_paper/r19c10-prop-4e-3-s5']),
    ('abl_novmem', ['_paper/r19c10-abl_novmem-4e-3-s1', '_paper/r19c10-abl_novmem-4e-3-s2',
                     '_paper/r19c10-abl_novmem-4e-3-s3', '_paper/r19c10-abl_novmem-4e-3-s4']),
    ('abl_nofinal', ['_paper/r19c10-abl_nofinal-4e-3-s1', '_paper/r19c10-abl_nofinal-4e-3-s2']),
    ('abl_noinv', ['_paper/r19c10-abl_noinv-4e-3-s1', '_paper/r19c10-abl_noinv-4e-3-s3']),
])

# 스파이크당 활성 뉴런 (스펙 §8 "활성 뉴런 축 전체 표", T9 후보) — 정확도 표와는 별도의
# 뉴런 단위 추출(reg_neuron_detail.csv)에서 나온 값이라 n 이 위 표와 다르다. n<4 인 행은
# sd 를 적지 않는다(값을 지어내지 않는다) — n 만 적는다.
T3_NEURON_USAGE = {
    'prop':        dict(n=11, per_neuron=1.4790, sd=0.0157),
    'abl_novmem':  dict(n=3,  per_neuron=1.4860, sd=None),
    'abl_nofinal': dict(n=2,  per_neuron=1.4478, sd=None),
    'abl_noinv':   dict(n=2,  per_neuron=1.5300, sd=None),
}

# 곡선(ANCOVA) 대비 정확도 기여 — 확정된 것은 vmem 항 하나뿐이다 (스펙 §2 A6). 나머지
# 두 같은-rho 팔은 이 통계를 낸 적이 없어 값을 지어내지 않고 대시로 둔다.
T3_CURVE_OFFSET = {
    'abl_novmem': '$-0.222$ ($3.5\\sigma$, $n{=}4$)',
}

T3_HEADER = (
    'Condition & Train loss & Train acc (\\%) & Val loss & Val acc (\\%) & Spikes & '
    '$\\Delta$ vs curve (pp) & Spikes / neuron \\\\'
)


def render_row_t3(cond, r):
    base = render_row(cond, r)[:-3]  # ' \\\\' 를 떼고 두 열을 더 붙인다
    return base + ' & -- & -- \\\\'


def render_mean_row_t3(n, curve_txt, neuron_txt):
    def inner(rows):
        mean = lambda k: sum(float(r[k]) for r in rows) / len(rows)
        base = (f'Mean ($n={n}$) & {fmt_loss(mean("train_loss"))} & '
                f'{fmt_acc(mean("train_acc"), 3)} & {fmt_loss(mean("val_loss"))} & '
                f'{fmt_acc(mean("best_val_acc"), 3)} & {fmt_spikes(mean("s_count"))}')
        return base + f' & {curve_txt} & {neuron_txt} \\\\'
    return inner


def _fmt_neuron_usage(arm):
    u = T3_NEURON_USAGE[arm]
    if u['sd'] is not None:
        return f'{u["per_neuron"]:.4f} $\\pm$ {u["sd"]:.4f} ($n{{=}}{u["n"]}$)'
    return f'{u["per_neuron"]:.4f} ($n{{=}}{u["n"]}$)'


def gen_t3(rows, out_path):
    by_dir = {r['run_dir']: r for r in rows}
    lines = []
    first = True
    for arm, run_dirs in T3_TIER_A_RUNS.items():
        if not first:
            lines.append('\\midrule')
        first = False
        cond = ABL_LABELS[arm]
        arm_rows = []
        for rd in run_dirs:
            if rd not in by_dir:
                raise ValueError(f'T3_TIER_A_RUNS 의 런이 CSV 에 없음: {rd}')
            r = by_dir[rd]
            lines.append(render_row_t3(cond, r))
            arm_rows.append(r)
        curve_txt = T3_CURVE_OFFSET.get(arm, '--')
        neuron_txt = _fmt_neuron_usage(arm) if arm in T3_NEURON_USAGE else '--'
        lines.append('\\cmidrule{1-8}')
        lines.append(render_mean_row_t3(len(arm_rows), curve_txt, neuron_txt)(arm_rows))

    body = '\n'.join([
        '\\begin{table}',
        '\\centering',
        '\\small',
        '\\begin{tabular}{lccccccc}',
        '\\toprule',
        T3_HEADER,
        '\\midrule',
        *lines,
        '\\bottomrule',
        '\\end{tabular}',
        '\\caption{Ablation, tier A: all arms at the same $\\rho=4\\times10^{-3}$ '
        '(spec \\S 12-4 tier A). Removing a component changes the neuron budget $R$, '
        'so the auto-solved strength moves the landing point in spike count even '
        'though $\\rho$ is held fixed -- that movement is itself the result (spec '
        '\\S 2 A9). $\\Delta$ vs curve is the accuracy offset against the fitted '
        'accuracy-spike curve (spec \\S 2 A6); only the sub-threshold term ablation '
        'has this statistic computed at this sample size, so the other two arms show '
        '\\texttt{--}. Spikes / neuron is drawn from a separate neuron-level '
        'extraction (spec \\S 8) whose run count differs from the accuracy columns '
        'and is cited per row; arms with $n<4$ report $n$ only, with no fabricated '
        'standard deviation.}',
        '\\label{tab:t3_ablation}',
        '\\end{table}',
        '',
    ])

    gate_note = '\n'.join([
        '',
        '% GATE: T3-matched -- tier B (spike-matched at ~192K) is designed but the',
        '% 12 runs it needs have NOT been executed and are NOT approved (spec \\S 12-4).',
        '% Targets, once run, go through the same collect_paper.py + gen_tables.py path:',
        '%   w/o potential   (abl_novmem)  rho ~= 2.6e-3  -> target ~192K spikes',
        '%   w/o final-step  (abl_nofinal) rho ~= 2.9e-3  -> target ~192K spikes',
        '%   fixed $\\lambda$ (abl_nolr)    lambda ~= 3.2e-7 -> target ~192K spikes',
        '%   w/o inversion   (abl_noinv)   NOT matched -- its unmatched landing point',
        '%     of 437,011 against the ~192K the other arms land at (at the same rho=4e-3',
        '%     as Ours) is the evidence that the inversion is a precondition for the',
        '%     method to work at all, not a tunable knob (spec \\S 2 A5).',
        '% Do not fill this block in without a re-approval -- see spec \\S 12-4.',
        '',
    ])

    tex = body + gate_note
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f'저장: {out_path}')


# ---------------------------------------------------------------------------
# T4 -- 학습 비용 (ms/step), 스펙 §2 A7 표를 그대로 전재. 파싱할 로그가 없고
# (310에폭 평균, train.log 타임스탬프 기반 수기 계산), 재생성 스크립트도 없다 --
# `rows` 인자는 다른 표와의 CLI 일관성을 위해서만 받고 쓰지 않는다.
# ---------------------------------------------------------------------------

# (조건, ms/step, s/epoch, 총 시간(h), baseline 대비 %) -- 스펙 §2 A7, 310에폭 평균,
# 런 내 sd 25~30ms (~4%).
T4_ROWS = [
    ('No regularisation', 631, 316, 27.2, None),
    (ABL_LABELS['abl_noinv'], 675, 338, 29.1, 7.0),
    (ABL_LABELS['prop'], 695, 348, 30.0, 10.0),
    ('Plain $L_2$', 712, 357, 30.7, 12.8),
    ('1-softmax', 759, 380, 32.7, 20.2),
    (ABL_LABELS['abl_nofinal'] + ' (per-time-step application)', 841, 421, 36.2, 33.2),
]


def gen_t4(rows, out_path):
    lines = []
    for cond, ms, s_ep, hours, pct in T4_ROWS:
        pct_txt = '--' if pct is None else f'+{pct:.1f}\\%'
        lines.append(f'{cond} & {ms} & {s_ep} & {hours:.1f}h & {pct_txt} \\\\')
    header = 'Condition & ms/step & s/epoch & Total time & vs.\\ baseline \\\\'
    tex = '\n'.join([
        '\\begin{table}',
        '\\centering',
        '\\small',
        '\\begin{tabular}{lcccc}',
        '\\toprule',
        header,
        '\\midrule',
        *lines,
        '\\bottomrule',
        '\\end{tabular}',
        '\\caption{Training cost, 310-epoch average (spec \\S 2 A7); per-run standard '
        'deviation is 25--30\\,ms ($\\approx$4\\%). Turning on any regulariser costs '
        '7--33\\% over the unregularised baseline; the proposed regulariser at 10\\% '
        'is cheaper than every other regularised comparison arm but not cheaper than '
        'the unregularised baseline. The last row applies the spike-count loss at '
        'every time step instead of once at the final step, isolating the '
        'single-step application\'s contribution to the 10\\% figure.}',
        '\\label{tab:t4_cost}',
        '\\end{table}',
        '',
    ])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f'저장: {out_path}')


# ---------------------------------------------------------------------------
# T8 -- 데이터셋 일반성 (게이트: G3, G6, G8 -- 2026-09-14 기준 전부 미충족).
# CIFAR-100 조건당 n=1(baseline 만 n=3) 이라 곡선을 못 그린다 (스펙 §2 B2). Spikformer
# /CIFAR10-DVS 는 제안법으로 재실행된 적이 없고(C4), SDT-V3 는 이식만 끝났다(C5).
# 이 표는 "일반성 주장의 근거"가 아니라 "지금 유일하게 있는 측정값"이다.
# ---------------------------------------------------------------------------

T8_C100_RUNS = OrderedDict([
    ('base', ['_paper/r19c100-base---s1', '_paper/r19c100-base---s2',
              '_paper/r19c100-base---s3']),
    ('l2', ['_paper/r19c100-l2-5e-7-s1']),
    ('prop', ['_paper/r19c100-prop-3e-3-s1']),
])

T8_C100_LABELS = {
    'base': 'No regularisation',
    'l2': 'Plain $L_2$ ($\\lambda=5\\times10^{-7}$)',
    'prop': 'Ours ($\\rho=3\\times10^{-3}$)',
}


def gen_t8(rows, out_path):
    by_dir = {r['run_dir']: r for r in rows}
    lines = []
    first = True
    for arm, run_dirs in T8_C100_RUNS.items():
        if not first:
            lines.append('\\midrule')
        first = False
        cond = T8_C100_LABELS[arm]
        arm_rows = [by_dir[rd] for rd in run_dirs]
        for r in arm_rows:
            lines.append(render_row(cond, r))
        if len(arm_rows) > 1:
            lines.append('\\cmidrule{1-6}')
            lines.append(render_mean_row(len(arm_rows))(arm_rows))
    tex = wrap_table(
        'CIFAR-100 / ResNet-19, the only generality measurement available as of '
        '2026-09-14 (spec \\S 2 B2). Baseline has $n=3$; Plain $L_2$ and the proposed '
        'regulariser have $n=1$ each, so no error bar is reported for them and no '
        'accuracy-spike curve can be fit -- the landing points do not match '
        '(211,350 vs.\\ 253,049 spikes). \\textbf{This table is not evidence of '
        'generality}: gate G3 (CIFAR-100, $n\\geq3$ per condition) is unmet, and '
        'gates G6 (Spikformer / CIFAR10-DVS re-run with the proposed regulariser) '
        'and G8 (SDT-V3, 173M) have no data at all (spec \\S 2 C4, C5). Kept out of '
        'the main body until a gate opens (\\texttt{figures-tables.md} T8).',
        'tab:t8_generality', lines)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(tex)
    print(f'저장: {out_path}')


TABLES = {
    't1': ('t1_main.tex', gen_t1),
    't3': ('t3_ablation.tex', gen_t3),
    't4': ('t4_cost.tex', gen_t4),
    't8': ('t8_generality.tex', gen_t8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='infile', required=True, help='collect_paper.py 가 낸 CSV')
    ap.add_argument('--table', required=True, choices=sorted(TABLES),
                     help='생성할 표')
    ap.add_argument('--out', default='', help='출력 .tex 경로 (기본: paper/tables/<표>.tex)')
    a = ap.parse_args()

    default_name, fn = TABLES[a.table]
    out_path = a.out or os.path.join('paper', 'tables', default_name)
    rows = read_csv(a.infile)
    fn(rows, out_path)


if __name__ == '__main__':
    main()
