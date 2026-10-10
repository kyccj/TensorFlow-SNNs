"""정확도-스파이크 ANCOVA (26-09-21 재계산).

    acc = a + b*ln(spikes) [+ b2*ln(spikes)^2] + sum_m c_m*[arm_m]

기준 팔은 인자로 준다. 네 모델을 돌린다: (완주만 / 배제적용) x (log-linear / 곡률).

입력은 `collect_paper_v2.py` 산출 CSV (v1 은 설정 파싱이 깨져 팔 배정이 어긋난다).
같은 이름 런이 두 서버에 있으면 0에폭 껍데기를 버린다 — 큐가 집어갔다가 다른 쪽에서 돈 흔적.
"""
import argparse, csv, itertools, os, sys
import numpy as np
from scipy import stats


def load(paths):
    rows = []
    for p in paths:
        rows += list(csv.DictReader(open(p)))
    # 0에폭 껍데기 제거 (같은 basename 의 실런이 있을 때)
    by = {}
    for r in rows:
        by.setdefault(os.path.basename(r['run_dir']), []).append(r)
    out = []
    for k, v in by.items():
        real = [r for r in v if int(r['epochs'] or 0) > 0]
        out += (real if real else v[:1])
    return out


def ols(y, X):
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ X.T @ y
    resid = y - X @ beta
    n, p = X.shape
    dof = n - np.linalg.matrix_rank(X)
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(XtXi) * s2)
    return beta, se, resid @ resid, dof


def fit(rows, ref, quad, label, arms_order=None):
    arms = sorted({r['arm'] for r in rows})
    if ref not in arms:
        print(f'  [{label}] 기준 팔 {ref} 없음 — 건너뜀')
        return None
    others = [a for a in arms if a != ref]
    if arms_order:
        others = [a for a in arms_order if a in others] + [a for a in others if a not in arms_order]
    y = np.array([float(r['val_acc']) for r in rows])
    ls = np.log(np.array([float(r['spikes']) for r in rows]))
    lc = ls - ls.mean()                       # 중심화: 곡률 모델에서 b 해석을 평균점 기울기로
    cols = [np.ones(len(rows)), lc] + ([lc ** 2] if quad else [])
    names = ['intercept', 'ln(spikes)'] + (['ln(spikes)^2'] if quad else [])
    for a in others:
        cols.append(np.array([1.0 if r['arm'] == a else 0.0 for r in rows]))
        names.append(a)
    X = np.column_stack(cols)
    beta, se, rss_f, dof = ols(y, X)
    # 팔 블록 F 검정
    k = len(names) - len(others)
    Xr = X[:, :k]
    _, _, rss_r, _ = ols(y, Xr)
    q = len(others)
    F = ((rss_r - rss_f) / q) / (rss_f / dof)
    p = 1 - stats.f.cdf(F, q, dof)
    print(f'\n  [{label}]  n={len(rows)}  기준={ref}  잔차SD={np.sqrt(rss_f/dof):.4f}  dof={dof}')
    print(f'    {"항":22s} {"추정":>9} {"SE":>8} {"sigma":>7}')
    for nm, b, s in zip(names, beta, se):
        if nm == 'intercept':
            continue
        print(f'    {nm:22s} {b:9.4f} {s:8.4f} {abs(b/s):7.2f}')
    print(f'    팔 블록 F({q},{dof})={F:.3f}  p={p:.4f}')
    return {'names': names, 'beta': beta, 'se': se, 'n': len(rows), 'F': F, 'p': p, 'dof': dof}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csvs', nargs='+')
    ap.add_argument('--ref', default='plain L2')
    ap.add_argument('--dataset', default='CIFAR10')
    ap.add_argument('--arms', default='')
    a = ap.parse_args()
    rows = [r for r in load(a.csvs)
            if r['dataset'] == a.dataset and r['val_acc'] and r['spikes']]
    if a.arms:
        keep = set(a.arms.split(','))
        rows = [r for r in rows if r['arm'] in keep]
    comp = [r for r in rows if r['completed'] == '1']
    excl = [r for r in comp if not (r['S30_over_S1'] and float(r['S30_over_S1']) < 0.19)]
    print(f'{a.dataset}: 전체 {len(rows)}  완주 {len(comp)}  배제적용 {len(excl)}')
    for nm, sub in (('완주만', comp), ('배제적용', excl)):
        cnt = {}
        for r in sub:
            cnt[r['arm']] = cnt.get(r['arm'], 0) + 1
        print(f'  {nm}: ' + '  '.join(f'{k}={v}' for k, v in sorted(cnt.items())))
    res = {}
    for nm, sub in (('완주만', comp), ('배제적용', excl)):
        for q in (False, True):
            tag = f'{nm} / {"곡률" if q else "log-linear"}'
            res[tag] = fit(sub, a.ref, q, tag)
    return res


if __name__ == '__main__':
    main()
