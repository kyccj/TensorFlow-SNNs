"""층별 gini ANCOVA 의 귀무 분포 (26-09-21 갱신판).

`npz_dist_perm_null.py` 와 같은 귀무(ours 10런을 무작위 2분할)를 쓰되 두 가지를 고친다.

1. **관측값을 하드코딩하지 않고 현장에서 다시 계산한다.** 구판은 (0.0077, 7, 5.77) 을
   소스에 박아뒀다. ours_layer 가 6런->8런이 되면 관측값이 바뀌는데 박아두면 안 잡힌다.

2. **검정력 크기를 맞춘 대조를 같이 낸다.** 귀무는 ours 10런을 4~6 : 6~4 로 쪼개 총 10런이고,
   관측은 ours_layer 8런 vs ours 10런 = 총 18런이다. 표본이 더 크면 SE 가 작아져
   `|t|>=2 인 층 수` 가 저절로 늘어난다 — 즉 이 통계량은 귀무 대비 유리하게 편향된다.
   그래서 관측 쪽도 같은 크기(k : 10-k)로 부분표집해 같은 검정력에서 다시 비교한다.
   `|c 중앙값|` 은 점추정이라 이 편향이 훨씬 덜하다.

한계: 귀무는 ours(within_channel) 내부의 시드·rho 변동만 담는다. 팔이 바뀔 때 생기는
계통 차이는 담지 못한다. 또 18개 층은 서로 독립이 아니라 Stouffer z 는 과대평가다.
"""
import sys
sys.path.insert(0, '/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis')
exec(open('/home/kyccj/PycharmProjects/TensorFlow-SNNs/analysis/npz_dist_ancova.py')
     .read().split("tbl('ours_ch'")[0])
import numpy as np

REPS = 300
rng = np.random.default_rng(11)


def ancova_sets(SA, SB, layer):
    x, y, g = [], [], []
    for S, gv in [(SB, 0), (SA, 1)]:
        for r, n, s in S:
            sc = s.loc[layer, 'spike_count']
            if sc <= 0:
                return None
            x.append(np.log(sc)); y.append(n.loc[layer, 'gini']); g.append(gv)
    x = np.array(x); y = np.array(y); g = np.array(g, float)
    X = np.column_stack([np.ones_like(x), x, g])
    if np.linalg.matrix_rank(X) < 3:
        return None
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - 3
    if dof < 2:
        return None
    s2 = res @ res / dof
    se = np.sqrt((s2 * np.linalg.pinv(X.T @ X))[2, 2])
    return beta[2], beta[2] / se


def stat(SA, SB):
    cs, ts = [], []
    for l in LS:
        r = ancova_sets(SA, SB, l)
        if r is None:
            continue
        cs.append(r[0]); ts.append(r[1])
    cs = np.array(cs); ts = np.array(ts)
    return np.median(cs), int((np.abs(ts) >= 2).sum()), np.sum(ts) / np.sqrt(len(ts))


def spread_ok(S, key='conv1_conv_n'):
    v = [np.log(s.loc[key, 'spike_count']) for _, _, s in S]
    return max(v) - min(v) >= 0.02


R = D['ours']
# ---- 귀무: ours 를 무작위 2분할 ----
meds, n2, zz = [], [], []
while len(meds) < REPS:
    idx = rng.permutation(len(R))
    k = int(rng.integers(4, 7))
    SA = [R[i] for i in idx[:k]]; SB = [R[i] for i in idx[k:]]
    if not (spread_ok(SA) and spread_ok(SB)):
        continue
    try:
        m, c2, z = stat(SA, SB)
    except Exception:
        continue
    meds.append(abs(m)); n2.append(c2); zz.append(abs(z))
meds = np.array(meds); n2 = np.array(n2); zz = np.array(zz)

print(f'=== 귀무: ours(within_channel) 10런을 무작위 2분할, {len(meds)}회 ===')
print(f'  |c 중앙값|   : 중앙 {np.median(meds):.4f}  95% {np.quantile(meds,0.95):.4f}  최대 {meds.max():.4f}')
print(f'  |t|>=2 층수  : 중앙 {np.median(n2):.1f}  95% {np.quantile(n2,0.95):.1f}  최대 {n2.max()}')
print(f'  |Stouffer z| : 중앙 {np.median(zz):.2f}  95% {np.quantile(zz,0.95):.2f}  최대 {zz.max():.2f}')

# ---- 관측 (전체 표본) ----
print(f'\n=== 관측 (현장 재계산, 하드코딩 아님) ===')
obs = {}
for nm, A, B in [('ours_layer vs ours', 'ours_layer', 'ours'),
                 ('ours_w1 vs ours', 'ours_w1', 'ours')]:
    m, c2, z = stat(D[A], D[B])
    obs[nm] = (abs(m), c2, abs(z))
    pm = (meds >= abs(m)).mean(); pc = (n2 >= c2).mean(); pz = (zz >= abs(z)).mean()
    print(f'  {nm:22s} n={len(D[A])}v{len(D[B])}  |c중앙|={abs(m):.4f} p={pm:.3f} | '
          f'|t|>=2 층={c2:2d} p={pc:.3f} | |z|={abs(z):.2f} p={pz:.3f}')

# ---- 검정력 크기를 맞춘 관측 ----
print(f'\n=== 크기 맞춘 관측 (k : 10-k 로 부분표집, 귀무와 같은 dof), {REPS}회 ===')
for nm, A, B in [('ours_layer vs ours', 'ours_layer', 'ours'),
                 ('ours_w1 vs ours', 'ours_w1', 'ours')]:
    mm, cc, zzz = [], [], []
    tries = 0
    while len(mm) < REPS and tries < REPS * 40:
        tries += 1
        k = int(rng.integers(4, 7))
        if k > len(D[A]) or (10 - k) > len(D[B]):
            continue
        SA = [D[A][i] for i in rng.choice(len(D[A]), k, replace=False)]
        SB = [D[B][i] for i in rng.choice(len(D[B]), 10 - k, replace=False)]
        if not (spread_ok(SA) and spread_ok(SB)):
            continue
        try:
            m, c2, z = stat(SA, SB)
        except Exception:
            continue
        mm.append(abs(m)); cc.append(c2); zzz.append(abs(z))
    mm = np.array(mm); cc = np.array(cc); zzz = np.array(zzz)
    pm = np.array([(meds >= v).mean() for v in mm])
    pc = np.array([(n2 >= v).mean() for v in cc])
    pz = np.array([(zz >= v).mean() for v in zzz])
    print(f'  {nm:22s} ({len(mm)}회)')
    print(f'    |c중앙|  중앙 {np.median(mm):.4f}  -> p 중앙 {np.median(pm):.3f}  '
          f'(p<0.05 인 비율 {(pm<0.05).mean():.2f})')
    print(f'    |t|>=2층 중앙 {np.median(cc):.1f}     -> p 중앙 {np.median(pc):.3f}  '
          f'(p<0.05 인 비율 {(pc<0.05).mean():.2f})')
    print(f'    |z|      중앙 {np.median(zzz):.2f}     -> p 중앙 {np.median(pz):.3f}  '
          f'(p<0.05 인 비율 {(pz<0.05).mean():.2f})')
