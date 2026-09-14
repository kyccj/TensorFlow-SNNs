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


TABLES = {
    't1': ('t1_main.tex', gen_t1),
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
