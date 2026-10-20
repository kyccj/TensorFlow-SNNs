#!/usr/bin/env python
"""R19 CIFAR-10 (ResNet-19, T=4) ours_inter_ch 구성요소 어블레이션 갱신 — 26-10-19.

요청 (사용자 10-19): "방법 분석 ... ablation 분석도 해봐야할거같은데 진행해줄래?"
  10-14 분석 (analysis/ablation_261014.md) 뒤로 342K 비교점의 네 팔이 모두 n=4 가 됐다
  (- vmem 2e-3, - final_step 1.4e-3, - 1- inversion 4.5e-3, - loss-ratio λ 1.25e-7).
  구성요소마다 (1) 같은 스파이크 정확도, (2) ρ 눈금 이동, (3) 학습 시간, (4) 기전을 다시 잰다.

런 선택과 n 은 원장 스냅샷 (10-19 16:38) 이 정한다:
  scratchpad/an1019/snapshot/inventory.json — completed, epochs==310, exclude=False, seed<900.
  per-run 세부값 (train.log 에폭 줄, reg_detail.csv, reg_neuron_detail.csv, config override) 은
  두 서버 런 폴더에서 읽기만 한다 (sejong 은 ssh stdin 수집기). 캐시: scratchpad/an1019/raw_*.json.gz
  best 에폭 val acc / spikes 를 다시 계산해 스냅샷과 대조한다 (불일치면 멈춘다).

수치 정의
  val acc = 310 에폭 중 best val_acc (같으면 첫 에폭), spikes = 그 에폭 s_count (평가 모드, 이미지당 원값)
  train loss = 그 에폭 'loss' (규제 항 포함), train acc, val loss = 그 에폭 값
  학습 모드 기전 지표 = reg_detail / reg_neuron_detail 의 마지막 에폭 (0-기준 309), 18층.
  λ_used[e] = 에폭 e 에 쓰인 λ (loss-ratio 런: 에폭 e-1 끝 adp_lambda, 고정 λ: 상수)
  R[e] = Σ_l sc_loss (배치 평균, λ 곱하기 전), L[e] = loss − λ_used·R (loss-ratio 런은 task_loss_clean)
  실효 ρ[e] = λ_used·R/L. 혼합 증강 해제 = 0-기준 에폭 200.

쓰는 법:  ~/anaconda3/envs/venv_1/bin/python analysis/ablation_261019.py [--refetch]
산출: analysis/ablation_261019.json (수치 원본), scratchpad/an1019/tables_261019.md (표 조각)

10-19 검토 반영 (eip-critic): JSON 'review' (두 오프셋 모형, 342K 세트 Holm, 마지막 10에폭·마지막 에폭
정확도 민감도, 기준 쪽 하나 빼기, vmem 층별 m 차와 침묵 비율 Spearman, 지니 대 baseline 의 스파이크 차,
학습 시간 짝의 런 수, loss-ratio 런별 배율, 9월/10월 실행 시기, npz 추출기 판), 'ckpt_check',
rho.*.interp (경험적 보간 F). 2차 F 풀이의 예외 분기 부호 수정 (현재 적합에서는 실행 안 됨, 결과 불변).

10-19 21시 갱신 (사용자 "추출 끝나면 ablation 다시 계산해서 노션 고쳐줘"):
  npz (테스트 1만 장, 평가 모드) 가 342K 세트에 거의 다 생겼다 (55 서버 추출 14개 + - vmem 2e-3 s1 1개).
  5절 npz 를 다시 짰다 — 'npz' (팔마다 342K 직접 / 스파이크 보정 직접 / ours npz 곡선 ANCOVA, 팔 안 Holm),
  'npz_vmem_rho' (같은 ρ 에서 vmem 이 어디서 더 쏘나: 직접 2e-3 과 ρ 2차 모형, 평가 모드 + 학습 모드),
  'npz_extractor_check' (같은 체크포인트를 옛/새 추출기로 뽑은 값 대조), 'npz_acc_check' (npz 정확도 = best val acc).
  정확도·ρ·궤적·학습 시간 절은 바꾸지 않았다 (같은 원자료 캐시, 같은 스냅샷).

10-19 22시 (21시 판 eip-critic 2차 검토 반영): JSON 'review21' 추가 — 다른 키의 값은 그대로.
  (a) ours 의 ρ 반응을 단계별로 (같은 ρ 의 단계 차가 ρ 눈금 이동으로 설명되는지),
  (b) 같은 npz 스파이크에서 단계 배분 (ln 단계 스파이크 ANCOVA, 4단계 Holm, 보간 확인),
  (c) 실행 시기 (9월/10월 시작) 항 — ours 안 효과와 팔 ANCOVA 에 넣었을 때,
  (d) 1− inversion 기준 통과 수, (e) 스텝 지표 32개 다중 비교, (f) 스텝 4 몫과 스파이크 수,
  (g) 추출기 판 대조 8쌍, (h) 학습 모드 `- vmem` 런과 npz 런의 겹침, (i) 죽은 채널 비율.
"""
import argparse, base64, glob, gzip, json, math, os, re, subprocess, sys
from itertools import combinations
import numpy as np
from scipy import stats

ROOT = '/home/kyccj/PycharmProjects/TensorFlow-SNNs'
SCR = '/tmp/claude-1002/-home-kyccj-PycharmProjects-TensorFlow-SNNs/291eebe0-2201-4ffb-bf39-2fb01ccd5a4a/scratchpad'
A = SCR + '/an1019'
SNAP = A + '/snapshot/inventory.json'
OUT_JSON = ROOT + '/analysis/ablation_261019.json'
OUT_TAB = A + '/tables_261019.md'
NPZ_DIRS = [A + '/npz_abl', A + '/npz', ROOT + '/_npz_t', ROOT + '/_npz', SCR + '/deep1005/npz']
EPOCHS, MIX_OFF = 310, 200
KEYS = ['base', 'ours_ch', 'ch_novmem', 'ch_nofinal', 'ch_noinv', 'ch_nolr']
ARMS = ['ch_novmem', 'ch_nofinal', 'ch_noinv', 'ch_nolr']
NAME = {'base': 'baseline', 'ours_ch': 'ours_inter_ch', 'ch_novmem': '- vmem', 'ch_nofinal': '- final_step',
        'ch_noinv': '- 1- inversion', 'ch_nolr': '- loss-ratio'}
# 342K 비교점 (각 n=4) — 결정 기록 10-14 18:1x / 10-17 18:45
POINT = {'ours_ch': '3e-3', 'ch_novmem': '2e-3', 'ch_nofinal': '1.4e-3', 'ch_noinv': '4.5e-3', 'ch_nolr': '1.25e-7'}
PATTERNS = ['r19c10-base---s1[1-6]'] + [f'r19c10-{k}-*' for k in KEYS[1:]]
STORE = {'138': '/media/hdd1/kyccj/EIP/paper', '23': '/srv2/kyccj/EIP/paper'}
CFG = {'138': ROOT + '/_paper', '23': '/home/kyccj/PycharmProjects/TensorFlow-SNNs/_paper'}
SPIKE_RANGE = (200e3, 420e3)
NBOOT = 20000

LAYERS = ['conv1_conv_n'] + [f'conv2_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')] + \
         [f'conv3_block{b}_{s}_n' for b in (1, 2, 3) for s in ('conv1', 'out')] + \
         [f'conv4_block{b}_{s}_n' for b in (1, 2) for s in ('conv1', 'out')] + ['fc1_n']
SHAPE = {}
for _l in LAYERS:
    SHAPE[_l] = (256,) if _l == 'fc1_n' else ((32, 32, 128) if _l.startswith(('conv1', 'conv2')) else
                                             ((16, 16, 256) if _l.startswith('conv3') else (8, 8, 512)))
NEU = {l: int(np.prod(s)) for l, s in SHAPE.items()}
STAGES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1']
INNER = [l for l in LAYERS if '_conv1_n' in l and l != 'conv1_conv_n']
OUTER = [l for l in LAYERS if '_out_n' in l]

# ----------------------------------------------------------------------------- 수집기 (읽기만)
COLLECTOR = r'''
import glob, gzip, json, os, re, sys, base64
def collect(store, cfgroot, patterns):
    out = {}
    tags = set()
    for pat in patterns:
        for root in (store, cfgroot):
            tags |= {os.path.basename(d) for d in glob.glob(os.path.join(root, pat))}
    for tag in sorted(tags):
        ds, dc = os.path.join(store, tag), os.path.join(cfgroot, tag)
        r = {'host': None, 'lines': [], 'conf': '', 'rd': None, 'rnd': None, 'ckpt': []}
        for d in (ds, dc):
            try:
                r['host'] = open(os.path.join(d, 'host.txt')).read().strip(); break
            except Exception:
                pass
        for d in (dc, ds):
            lg = os.path.join(d, 'train.log')
            if os.path.exists(lg):
                txt = open(lg, errors='ignore').read().replace('\r', '\n')
                lines = [ln for ln in txt.split('\n') if 'val_loss' in ln and 'loss: ' in ln]
                if len(lines) > len(r['lines']):
                    r['lines'] = lines
        cf = os.path.join(dc, 'config_sweep.py')
        if os.path.exists(cf):
            t = open(cf, errors='ignore').read()
            i, j = t.find('# >>> run_paper overrides'), t.find('# <<< run_paper overrides')
            r['conf'] = t[i:j + 30] if 0 <= i < j else ''
        for fn, k in (('reg_detail.csv', 'rd'), ('reg_neuron_detail.csv', 'rnd')):
            for d in (ds, dc):
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    r[k] = open(p).read(); break
        r['ckpt'] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ds, '*', '*', 'ep-*.weights.h5')))
        out[tag] = r
    return out
'''


def fetch(server, refetch):
    os.makedirs(A, exist_ok=True)
    cache = os.path.join(A, f'raw_{server}.json.gz')
    if refetch or not os.path.exists(cache):
        if server == '138':
            ns = {}
            exec(COLLECTOR, ns)
            data = ns['collect'](STORE['138'], CFG['138'], PATTERNS)
            with gzip.open(cache, 'wt') as f:
                json.dump(data, f)
        else:
            prog = COLLECTOR + (
                "\nsys.stdout.write(base64.b64encode(gzip.compress(json.dumps("
                f"collect({STORE['23']!r}, {CFG['23']!r}, {PATTERNS!r})).encode())).decode())\n")
            r = subprocess.run(['ssh', '-p', '23456', '-o', 'BatchMode=yes', 'kyccj@161.122.23.23', 'python3', '-'],
                               input=prog.encode(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
            with open(cache, 'wb') as f:
                f.write(base64.b64decode(r.stdout))
    with gzip.open(cache, 'rt') as f:
        return json.load(f)


# ----------------------------------------------------------------------------- 파싱
EPOCH_KV = re.compile(r'([A-Za-z_][\w\-]*): ([-\d.]+(?:[eE][+-]?\d+)?|nan)')
TIME = re.compile(r'\] - (\d+)s ')
ASSIGN = re.compile(r"^\s*conf\.([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.M)
TAGRE = re.compile(r'^r19c10-(base|ours_ch|ch_novmem|ch_nofinal|ch_noinv|ch_nolr)-(.*)-s(\d+)$')


def _lit(s):
    s = s.split('#')[0].strip()
    if s[:1] in ("'", '"'):
        return s.strip('\'"')
    if s in ('True', 'False'):
        return s == 'True'
    try:
        return float(s)
    except ValueError:
        return s


def parse_epochs(lines):
    rows = []
    for ln in lines:
        k = ln.find('loss: ')
        d = {}
        for key, v in EPOCH_KV.findall(ln[k:]):
            try:
                d[key] = float(v)
            except ValueError:
                pass
        m = TIME.search(ln)
        d['sec'] = float(m.group(1)) if m else float('nan')
        if all(x in d for x in ('loss', 'acc', 'val_loss', 'val_acc', 's_count')):
            rows.append(d)
    return rows


def parse_csv(txt):
    """reg_detail / reg_neuron_detail → {epoch: {layer: {col: val}}}"""
    if not txt:
        return None
    lines = txt.strip().split('\n')
    cols = lines[0].split(',')
    out = {}
    for ln in lines[1:]:
        v = ln.split(',')
        if len(v) != len(cols):
            continue
        out.setdefault(int(v[0]), {})[v[1]] = {c: float(x) for c, x in zip(cols[2:], v[2:])}
    return out


def arm_of(conf):
    """override 블록 → 팔. inter 기준에서 한 요소만 다른 팔만 인정."""
    g = lambda k, d=None: conf.get(k, d)
    if not g('reg_spike_out', False):
        return 'base'
    if not (g('reg_spike_out_sc') is True and g('reg_spike_out_sc_maxnorm') is True
            and not g('reg_spike_out_sc_maxnorm_plain', False) and g('reg_spike_out_wta_rev') is True
            and g('reg_spike_maxnorm_group') == 'channel' and not g('reg_spike_wta_rev_w1', False)
            and float(g('reg_spike_loss_ratio_start_ep', 0) or 0) == 0
            and float(g('reg_spike_loss_ratio_end_ep', 0) or 0) == 0
            and float(g('reg_spike_shape_beta', 1.0)) == 1.0 and not g('reg_spike_out_bpsr', False)
            and not g('reg_spike_accum_loss', False)):
        return 'other'
    vm = float(g('reg_spike_vmem_gain', 0.0)) == 1.0 and g('reg_spike_vmem_silent_only') is True
    nv = float(g('reg_spike_vmem_gain', 0.0)) == 0.0
    lr = g('reg_spike_loss_ratio') is True
    fs = g('reg_spike_final_step') is True
    ni = g('reg_spike_maxnorm_no_inv', False) is True
    off = [nv, not fs, ni, not lr]
    if sum(off) > 1 or not (vm or nv):
        return 'other'
    if nv:
        return 'ch_novmem'
    if not fs:
        return 'ch_nofinal'
    if ni:
        return 'ch_noinv'
    if not lr:
        return 'ch_nolr'
    return 'ours_ch'


def mech_epoch(rd, rnd, e):
    """학습 모드 층 집계 (에폭 e). ablation_vmem_261014.last_epoch_metrics 와 같은 정의 + 추가."""
    if not rd or e not in rd:
        return {}
    row = rd[e]
    ls = [l for l in LAYERS if l in row]
    if len(ls) != len(LAYERS):
        return {}
    o = {}
    spk = {l: row[l]['spike_count'] for l in ls}
    w = {l: row[l]['sc_rate_mean'] for l in ls}
    scl = {l: row[l]['sc_loss'] for l in ls}
    tot = sum(spk.values())
    nN = sum(NEU[l] for l in ls)
    o['tr_spk'] = tot
    o['w_mean'] = sum(w[l] * NEU[l] for l in ls) / nN
    o['R'] = sum(scl.values())
    for s in STAGES:
        ss = [l for l in ls if l.startswith(s)]
        o[f'share_{s}'] = 100 * sum(spk[l] for l in ss) / tot
        o[f'w_{s}'] = sum(w[l] * NEU[l] for l in ss) / sum(NEU[l] for l in ss)
        o[f'Rshare_{s}'] = 100 * sum(scl[l] for l in ss) / o['R'] if o['R'] > 0 else float('nan')
    o['share_inner'] = 100 * sum(spk[l] for l in INNER) / tot
    o['w_inner'] = sum(w[l] * NEU[l] for l in INNER) / sum(NEU[l] for l in INNER)
    o['w_outer'] = sum(w[l] * NEU[l] for l in OUTER) / sum(NEU[l] for l in OUTER)
    o['Rshare_inner'] = 100 * sum(scl[l] for l in INNER) / o['R'] if o['R'] > 0 else float('nan')
    o['w_layer'] = {l: w[l] for l in ls}
    o['spk_layer'] = {l: spk[l] for l in ls}
    o['R_layer'] = {l: scl[l] for l in ls}
    if rnd and e in rnd:
        d = rnd[e]
        o['w_sd'] = sum(d[l]['sc_rate_std'] * NEU[l] for l in ls) / nN
        o['dead_pct'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in ls) / nN
        o['gini_sw'] = sum(d[l]['gini'] * spk[l] for l in ls) / tot
        o['top10_sw'] = 100 * sum(d[l]['top10_share'] * spk[l] for l in ls) / tot
        o['dead_layer'] = {l: 100 * d[l]['dead_neuron_ratio'] for l in ls}
        for s in STAGES:
            ss = [l for l in ls if l.startswith(s)]
            o[f'dead_{s}'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in ss) / sum(NEU[l] for l in ss)
        o['dead_inner'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in INNER) / sum(NEU[l] for l in INNER)
        o['dead_outer'] = 100 * sum(d[l]['dead_neuron_ratio'] * NEU[l] for l in OUTER) / sum(NEU[l] for l in OUTER)
    return o


def knob_val(k):
    return float(k.replace('p', '.'))


def build_runs(refetch):
    raw = {'138': fetch('138', refetch), '23': fetch('23', refetch)}
    snap = {r['tag']: r for r in json.load(open(SNAP))['runs']}
    runs, dups, mism = [], [], []
    for tag in sorted(set(raw['138']) | set(raw['23'])):
        m = TAGRE.match(tag)
        if not m:
            continue
        cands = [(s, raw[s][tag], parse_epochs(raw[s][tag]['lines'])) for s in ('138', '23') if tag in raw[s]]
        if len(cands) > 1:
            dups.append({'tag': tag, 'epochs': {s: len(e) for s, _, e in cands}})
        srv, rr, rows = max(cands, key=lambda c: (len(c[2]), c[0] == '138'))
        key, knob, seed = m.group(1), m.group(2), int(m.group(3))
        sv = snap.get(tag)
        conf = {k: _lit(v) for k, v in ASSIGN.findall(rr['conf'])}
        arm = arm_of(conf) if conf else 'no_conf'
        rec = {'tag': tag, 'key': key, 'knob': '-' if key == 'base' else knob.replace('p', '.'), 'seed': seed,
               'server': srv, 'host': (rr['host'] or '').split('\t')[0], 'arm_conf': arm, 'epochs': len(rows),
               'ckpt': rr.get('ckpt', [])}
        if key == 'ch_nolr':
            rec['lam_fixed'] = conf.get('reg_spike_out_const')
        elif key != 'base':
            rec['rho'] = conf.get('reg_spike_loss_ratio_target')
        # 스냅샷이 정한 사용 여부
        use = bool(sv and sv.get('completed') and sv.get('epochs') == EPOCHS and not sv.get('exclude')
                   and seed < 900)
        rec['snap_reason'] = '' if use else (('스냅샷에 없음' if not sv else (sv.get('exclude_reason') or
                                               ('연기시험' if seed >= 900 else f"미완주/보류({sv.get('epochs')}ep, {sv.get('status')})"))))
        if rows:
            b = max(rows, key=lambda x: x['val_acc'])
            bi = rows.index(b)
            rec.update(best_epoch=bi + 1, train_loss=b['loss'], train_acc=b['acc'] * 100, val_loss=b['val_loss'],
                       val_acc=b['val_acc'] * 100, spikes=b['s_count'])
            secs = np.array([r['sec'] for r in rows[1:]], float)
            secs = secs[np.isfinite(secs)]
            rec['sec_med'] = float(np.median(secs)) if len(secs) else None
            rec['rows'] = rows
        rd, rnd = parse_csv(rr['rd']), parse_csv(rr['rnd'])
        rec['rd'], rec['rnd'] = rd, rnd
        # 체크포인트 에폭 = best 에폭 대조 (10-19 검토 추가). ckpt 는 저장소 폴더에서만 모은다.
        ck_eps = sorted(int(m_.group(1)) for m_ in (re.search(r'ep-(\d+)', c) for c in rec['ckpt']) if m_)
        rec['ckpt_epochs'] = ck_eps
        rec['ckpt_match'] = (rec.get('best_epoch') in ck_eps) if ck_eps else None
        rec['store_dir'] = (os.path.isdir(os.path.join(STORE['138'], tag)) if srv == '138' else None)
        rec['has_rd'] = rd is not None
        if use:
            # 스냅샷 대조
            if not rows or abs(rec['val_acc'] - sv['val_acc']) > 0.006 or abs(rec['spikes'] - sv['spikes']) > 1.0 \
                    or sv['server'] != srv or len(rows) != EPOCHS:
                mism.append({'tag': tag, 'mine': (rec.get('val_acc'), rec.get('spikes'), srv, len(rows)),
                             'snap': (sv['val_acc'], sv['spikes'], sv['server'], sv['epochs'])})
            if arm != key:
                mism.append({'tag': tag, 'arm_conf': arm})
            if key not in ('base', 'ch_nolr') and abs(float(rec['rho']) - knob_val(knob)) > 1e-12:
                mism.append({'tag': tag, 'rho_conf': rec['rho']})
        rec['use'] = use
        runs.append(rec)
    return runs, dups, mism




# ----------------------------------------------------------------------------- 통계 도구
def tq(df, q=0.975):
    return float(stats.t.ppf(q, df))


def ols(X, y):
    X, y = np.asarray(X, float), np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    res = y - X @ beta
    df = len(y) - np.linalg.matrix_rank(X)
    s2 = float(res @ res / df) if df > 0 else float('nan')
    cov = s2 * XtX_inv
    return {'beta': beta, 'cov': cov, 'se': np.sqrt(np.diag(cov)), 'df': df, 's2': s2, 'res': res, 'XtX_inv': XtX_inv}


def est(b, se, df):
    t = b / se if se > 0 else float('nan')
    p = float(2 * stats.t.sf(abs(t), df)) if np.isfinite(t) else float('nan')
    h = tq(df) * se
    return {'d': float(b), 'se': float(se), 'lo': float(b - h), 'hi': float(b + h), 'p': p, 'df': int(df)}


def welch(a, b):
    """평균 a − 평균 b, Welch."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if va + vb > 0 else len(a) + len(b) - 2
    o = est(a.mean() - b.mean(), se, df)
    o['df'] = float(df)
    o['na'], o['nb'] = len(a), len(b)
    return o


def pooled_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    df = len(a) + len(b) - 2
    sp2 = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / df
    o = est(a.mean() - b.mean(), math.sqrt(sp2 * (1 / len(a) + 1 / len(b))), df)
    o['sp'] = math.sqrt(sp2)
    return o


def perm_p(a, b):
    """정확 순열 검정 (두 쪽, 평균 차)."""
    a, b = list(a), list(b)
    allv = a + b
    obs = abs(np.mean(a) - np.mean(b))
    cnt = tot = 0
    for idx in combinations(range(len(allv)), len(a)):
        s = set(idx)
        x = [allv[i] for i in idx]
        y = [allv[i] for i in range(len(allv)) if i not in s]
        tot += 1
        cnt += abs(np.mean(x) - np.mean(y)) >= obs - 1e-12
    return cnt / tot


def mde_t(se, df, alpha=0.05, power=0.8):
    """두 쪽 t 검정에서 검정력 power 로 잡히는 최소 차 (비중심 t 로 푼다). se = 차의 표준오차."""
    tc = tq(df, 1 - alpha / 2)
    lo, hi = 0.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        pw = stats.nct.sf(tc, df, mid) + stats.nct.cdf(-tc, df, mid)
        lo, hi = (mid, hi) if pw < power else (lo, mid)
    return float(hi * se)


def fieller_ratio(a, b, va, vb, cab, df, level=0.95):
    """a/b 의 Fieller 구간."""
    t2 = tq(df, 0.5 + level / 2) ** 2
    A_ = b * b - t2 * vb
    B_ = -2 * (a * b - t2 * cab)
    C_ = a * a - t2 * va
    disc = B_ * B_ - 4 * A_ * C_
    if A_ <= 0 or disc < 0:
        return (float('nan'), float('nan'))
    r1, r2 = (-B_ - math.sqrt(disc)) / (2 * A_), (-B_ + math.sqrt(disc)) / (2 * A_)
    return (min(r1, r2), max(r1, r2))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [None] * len(ps)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def msd(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float('nan'))


# ----------------------------------------------------------------------------- 1. 같은 스파이크 정확도
X0 = math.log(342075.0)        # ours_inter_ch 3e-3 (n=4) 평균 스파이크 — 중심


def ancova(ref, arm, yf, xf, quad=False, srv=True, slope_int=False):
    """y = a + b·x (+ c·x²) + e·[23] + Δ·[arm] (+ g·x·[arm]).  x = xf(r) − 중심."""
    rows = [(r, 0) for r in ref] + [(r, 1) for r in arm]
    X, y = [], []
    for r, g in rows:
        x = xf(r)
        v = [1.0, x]
        if quad:
            v.append(x * x)
        if srv:
            v.append(1.0 if r['server'] == '23' else 0.0)
        v.append(float(g))
        if slope_int:
            v.append(x * g)
        X.append(v)
        y.append(yf(r))
    X = np.array(X)
    f = ols(X, y)
    k = X.shape[1] - 1 - (1 if slope_int else 0)
    o = est(f['beta'][k], f['se'][k], f['df'])
    o.update(n_ref=len(ref), n_arm=len(arm), slope=float(f['beta'][1]), slope_se=float(f['se'][1]),
             resid_sd=math.sqrt(f['s2']), c_arm=float(f['XtX_inv'][k, k]))
    if slope_int:
        o['slope_int'] = est(f['beta'][-1], f['se'][-1], f['df'])
    if srv:
        ks = 2 + (1 if quad else 0)
        o['srv23'] = est(f['beta'][ks], f['se'][ks], f['df'])
    return o


def acc_y(r):
    return r['val_acc']


def lnspk_c(r):
    return math.log(r['spikes']) - X0


def acc_section(G, sd_pool, df_pool):
    """팔마다 같은 스파이크 정확도: ANCOVA 여러 형태 + 하나 빼기 + 342K 직접 비교 + 최소 검출 차."""
    ours = G['ours_curve']
    out = {}
    prim_p = {}
    for a in ARMS:
        arm_all = G[a]
        arm_pt = [r for r in arm_all if r['knob'] == POINT[a]]
        o = {}
        o['lin_srv_all'] = ancova(ours, arm_all, acc_y, lnspk_c)
        o['quad_srv_all'] = ancova(ours, arm_all, acc_y, lnspk_c, quad=True)
        o['lin_nosrv_all'] = ancova(ours, arm_all, acc_y, lnspk_c, srv=False)
        o['lin_srv_pt'] = ancova(ours, arm_pt, acc_y, lnspk_c)
        o['quad_srv_pt'] = ancova(ours, arm_pt, acc_y, lnspk_c, quad=True)
        a138 = [r for r in arm_all if r['server'] == '138']
        a23 = [r for r in arm_all if r['server'] == '23']
        if len(a138) >= 2:
            o['lin_srv_138only'] = ancova(ours, a138, acc_y, lnspk_c)
        if len(a23) >= 1:
            o['lin_srv_23only'] = ancova(ours, a23, acc_y, lnspk_c)
        # 서버×팔 상호작용 (끈 팔이 두 서버에 다 있을 때): y = … + Δ·arm + h·arm·[23]
        if a138 and a23:
            rows = [(r, 0) for r in ours] + [(r, 1) for r in arm_all]
            Xi = np.array([[1.0, lnspk_c(r), 1.0 if r['server'] == '23' else 0.0, float(g),
                            float(g) * (1.0 if r['server'] == '23' else 0.0)] for r, g in rows])
            fi = ols(Xi, [r['val_acc'] for r, _ in rows])
            o['srv_x_arm'] = est(fi['beta'][4], fi['se'][4], fi['df'])
        # ours 곡선 범위를 좁힘 (국소): 250~420K
        loc = [r for r in ours if 250e3 <= r['spikes'] <= 420e3]
        o['lin_srv_all_ref250'] = ancova(loc, arm_all, acc_y, lnspk_c)
        o['n_ref250'] = len(loc)
        # 기울기 상호작용 (끈 팔이 두 스파이크 수준 이상일 때)
        if len({r['knob'] for r in arm_all}) >= 2:
            o['slope_int'] = ancova(ours, arm_all, acc_y, lnspk_c, slope_int=True)['slope_int']
        # 하나 빼기 — 끈 팔 쪽
        loo = []
        for i in range(len(arm_all)):
            sub = arm_all[:i] + arm_all[i + 1:]
            if not sub:
                continue
            e_ = ancova(ours, sub, acc_y, lnspk_c)
            loo.append({'drop': arm_all[i]['tag'], 'd': e_['d'], 'p': e_['p'], 'lo': e_['lo'], 'hi': e_['hi']})
        o['loo_arm'] = loo
        loo_pt = []
        for i in range(len(arm_pt)):
            sub = arm_pt[:i] + arm_pt[i + 1:]
            e_ = ancova(ours, sub, acc_y, lnspk_c)
            loo_pt.append({'drop': arm_pt[i]['tag'], 'd': e_['d'], 'p': e_['p'], 'lo': e_['lo'], 'hi': e_['hi']})
        o['loo_pt'] = loo_pt
        # 342K 직접 비교의 하나 빼기 (끈 팔 4런 중 하나씩)
        o['loo_direct'] = [welch([r['val_acc'] for j, r in enumerate(arm_pt) if j != i],
                                 [r['val_acc'] for r in G['ours_pt']])['d'] for i in range(len(arm_pt))]
        # 하나 빼기 — ours 곡선 쪽
        loo_o = [ancova(ours[:i] + ours[i + 1:], arm_all, acc_y, lnspk_c) for i in range(len(ours))]
        o['loo_ours_range'] = [min(x['d'] for x in loo_o), max(x['d'] for x in loo_o)]
        o['loo_ours_p_range'] = [min(x['p'] for x in loo_o), max(x['p'] for x in loo_o)]
        # 342K 직접 (n=4 대 n=4)
        o3 = G['ours_pt']
        ya, yo = [r['val_acc'] for r in arm_pt], [r['val_acc'] for r in o3]
        o['direct_welch'] = welch(ya, yo)
        o['direct_pooled'] = pooled_t(ya, yo)
        o['direct_perm_p'] = perm_p(ya, yo)
        # 스파이크 보정 직접 비교: ours 곡선 기울기 b 로 각 런을 342,075 로 옮긴다 (b 의 불확실성 전파)
        Xc = np.array([[1.0, lnspk_c(r), 1.0 if r['server'] == '23' else 0.0] for r in ours])
        fc = ols(Xc, [r['val_acc'] for r in ours])
        b, vb = fc['beta'][1], fc['cov'][1, 1]
        adj = lambda rs: [r['val_acc'] - b * lnspk_c(r) for r in rs]
        aa, ao = adj(arm_pt), adj(o3)
        dx = np.mean([lnspk_c(r) for r in arm_pt]) - np.mean([lnspk_c(r) for r in o3])
        pt = pooled_t(aa, ao)
        se = math.sqrt(pt['se'] ** 2 + dx * dx * vb)
        o['direct_adj'] = est(np.mean(aa) - np.mean(ao), se, pt['df'])
        o['direct_adj']['dx_ln'] = float(dx)
        o['direct_adj']['slope'] = float(b)
        # 최소 검출 차 (α=0.05, 검정력 0.8)
        o['mde_direct_4v4'] = mde_t(sd_pool * math.sqrt(1 / len(arm_pt) + 1 / len(o3)), len(arm_pt) + len(o3) - 2)
        c = o['lin_srv_all']['c_arm']
        o['mde_ancova_all'] = mde_t(sd_pool * math.sqrt(c), o['lin_srv_all']['df'])
        o['mde_ancova_pt'] = mde_t(sd_pool * math.sqrt(o['lin_srv_pt']['c_arm']), o['lin_srv_pt']['df'])
        # 관측 차를 검정력 0.8 로 보려면 조건당 몇 런 (직접 비교, 같은 n)
        d_obs = abs(o['direct_adj']['d'])
        nn = None
        for n in range(2, 2001):
            if mde_t(sd_pool * math.sqrt(2 / n), 2 * n - 2) <= d_obs:
                nn = n
                break
        o['n_needed_direct'] = nn
        out[a] = o
        prim_p[a] = o['lin_srv_all']['p']
    hp = holm([prim_p[a] for a in ARMS])
    for a, h in zip(ARMS, hp):
        out[a]['holm_p_primary'] = h
    return out


def verdict(o):
    """판정: 기여 있음 / 검출 안 됨 / 판단 불가 (과제 정의)."""
    prim = o['lin_srv_all']
    n = prim['n_arm']
    robust = all(x['p'] < 0.05 for x in o['loo_arm']) and o['quad_srv_all']['p'] < 0.05 \
        and o['lin_nosrv_all']['p'] < 0.05 and o['direct_adj']['p'] < 0.05
    if n <= 2:
        return '판단 불가 (n≤2)'
    if prim['p'] < 0.05 and robust:
        return '기여 있음'
    if prim['p'] < 0.05:
        return '판단 불가 (경계선: 명목 유의, 하나 빼기·서버·직접 비교에서 유지 안 됨)'
    if o['lin_srv_pt']['p'] < 0.05:
        # 10-19 검토: 342K 세트는 사전 계획한 설계점이라 '사후 부분집합' 이 아니다. 1차 분석을 미리 정하지 않아
        # 분석 선택에 따라 결론이 갈리므로 경계선으로 둔다.
        return '판단 불가 (경계선: 1차 분석 사전 미지정 — 곡선 전체는 CI 가 0 포함, 342K 세트는 명목 유의, 모형 없는 직접 비교는 0 포함)'
    pt = o['lin_srv_pt']
    return (f"검출 안 됨 (곡선 전체 n={prim['n_arm']} CI {prim['lo']:+.3f}~{prim['hi']:+.3f}; "
            f"342K 세트 n={pt['n_arm']} CI {pt['lo']:+.3f}~{pt['hi']:+.3f}; 동등성 마진 없음)")


# ----------------------------------------------------------------------------- 2. ρ 눈금 (같은 ρ 의 스파이크)
def _rho_fit(ours, arm, quad, c0, srv=True):
    X, y = [], []
    for r, g in [(r, 0) for r in ours] + [(r, 1) for r in arm]:
        x = math.log(float(r['rho'])) - c0
        v = [1.0, x] + ([x * x] if quad else []) + ([1.0 if r['server'] == '23' else 0.0] if srv else []) + [float(g)]
        X.append(v)
        y.append(math.log(r['spikes']))
    return ols(np.array(X), y), np.array(X)


def interp_F(ours_all, arm):
    """경험적 보간 F (모형 없음, 10-19 검토 추가): ours 의 ρ 묶음 평균 (ln ρ, 평균 ln spikes) 을 꺾은선으로 잇고,
    끈 팔 ρ 묶음의 평균 ln spikes 에 해당하는 ours ρ 를 읽는다. F = 그 ours ρ / 끈 팔 ρ. 서버 보정 없음.
    ours 스파이크 범위 밖이면 None (외삽 안 함)."""
    byr = {}
    for r in ours_all:
        byr.setdefault(float(r['rho']), []).append(math.log(r['spikes']))
    pts = sorted((math.log(k), float(np.mean(v)), len(v)) for k, v in byr.items())
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if any(ys[i + 1] >= ys[i] for i in range(len(ys) - 1)):
        return {'error': 'ours 묶음 평균이 ρ 에 단조 감소가 아님'}
    res = []
    for k_ in sorted({r['knob'] for r in arm}, key=float):
        v = [math.log(r['spikes']) for r in arm if r['knob'] == k_]
        y = float(np.mean(v))
        row = {'rho': k_, 'n_arm': len(v), 'arm_spk_gmean': math.exp(y), 'F': None, 'rho_ours_equiv': None}
        for i in range(len(ys) - 1):
            if ys[i] >= y >= ys[i + 1]:
                t = (y - ys[i]) / (ys[i + 1] - ys[i])
                xr = xs[i] + t * (xs[i + 1] - xs[i])
                row['rho_ours_equiv'] = math.exp(xr)
                row['F'] = math.exp(xr) / float(k_)
                row['bracket'] = [math.exp(xs[i]), math.exp(xs[i + 1])]
                row['bracket_n'] = [pts[i][2], pts[i + 1][2]]
                break
        res.append(row)
    return res


def rho_section(G):
    ours_all = G['ours_all']
    out = {}
    for a, win in (('ch_novmem', (1.5e-3, 5e-3)), ('ch_nofinal', (1.5e-3, 3e-3)), ('ch_noinv', (3e-3, 1.2e-2))):
        arm = G[a]
        o = {'window': win}
        rho_pt = float(POINT[a])
        c0 = math.log(rho_pt)
        win_ref = [r for r in ours_all if win[0] - 1e-12 <= float(r['rho']) <= win[1] + 1e-12]
        rmin, rmax = min(float(r['rho']) for r in ours_all), max(float(r['rho']) for r in ours_all)
        arm_in = [r for r in arm if rmin - 1e-12 <= float(r['rho']) <= rmax + 1e-12]
        specs = {'lin_window': (win_ref, False, arm), 'quad_all': (ours_all, True, arm), 'lin_all': (ours_all, False, arm),
                 'quad_all_overlap': (ours_all, True, arm_in)}
        for nm, (ref, quad, arm_) in specs.items():
            if not arm_:
                continue
            f, X = _rho_fit(ref, arm_, quad, c0)
            k = X.shape[1] - 1
            e = est(f['beta'][k], f['se'][k], f['df'])
            e['pct'], e['pct_lo'], e['pct_hi'] = [100 * (math.exp(v) - 1) for v in (e['d'], e['lo'], e['hi'])]
            e['n_ref'], e['n_arm'] = len(ref), len(arm_)
            # 같은 스파이크를 내는 ours ρ 의 배율 (arm ρ 대비): 선형이면 ln F = δ/c, 2차면 수치로
            # (c<0, δ<0 → F>1. 10-19 검토: 주석의 옛 부호 −δ/c 를 바로잡음, 계산식은 원래 δ/c)
            c, vc, cdc = f['beta'][1], f['cov'][1, 1], f['cov'][1, k]
            # 배율 F = (같은 스파이크를 내는 ours ρ) / (끈 팔 ρ). ln F = x* 는 b1·x* (+ b2·x*²) = δ 의 해.
            # F>1: 끈 팔이 같은 ρ 에서 더 세게 줄인다 (ours 는 ρ 를 F 배 올려야 같은 스파이크).
            if not quad:
                lnF = e['d'] / c
                lo, hi = fieller_ratio(f['beta'][k], c, f['cov'][k, k], vc, cdc, f['df'])
                e['rho_factor'], e['rho_factor_lo'], e['rho_factor_hi'] = math.exp(lnF), math.exp(lo), math.exp(hi)
            else:
                def solve(beta):
                    # f(x*) = f(0) + δ, x* = ln(ρ*/ρ_pt). f(x) = b1 x + b2 x²
                    b1, b2, d = beta[1], beta[2], beta[k]
                    disc = b1 * b1 + 4 * b2 * d
                    if b2 == 0 or disc < 0:
                        return d / b1        # 선형 근사 (10-19 검토: 옛 부호 −d/b1 수정. 현재 적합은 disc>0 이라 안 탐)
                    r1, r2 = (-b1 + math.sqrt(disc)) / (2 * b2), (-b1 - math.sqrt(disc)) / (2 * b2)
                    return r1 if abs(r1) < abs(r2) else r2
                x0 = solve(f['beta'])
                e['disc'] = float(f['beta'][1] ** 2 + 4 * f['beta'][2] * f['beta'][k])
                # 국소 기울기 (끈 팔 ρ 점에서의 d ln spk / d ln ρ) 와 그 점이 ours ρ 범위 안인지
                e['local_slope_at_pt'] = float(f['beta'][1])
                e['pt_inside_ours_rho'] = bool(rmin - 1e-12 <= rho_pt <= rmax + 1e-12)
                e['arm_rho_outside_ours'] = sorted({r['knob'] for r in arm_ if not (rmin - 1e-12 <= float(r['rho']) <= rmax + 1e-12)})
                g = np.zeros(len(f['beta']))
                for j in (1, 2, k):
                    bb = f['beta'].copy()
                    h = 1e-6 * max(1.0, abs(bb[j]))
                    bb[j] += h
                    g[j] = (solve(bb) - x0) / h
                se = math.sqrt(g @ f['cov'] @ g)
                tt = tq(f['df'])
                e['rho_factor'], e['rho_factor_lo'], e['rho_factor_hi'] = math.exp(x0), math.exp(x0 - tt * se), math.exp(x0 + tt * se)
            o[nm] = e
        # 같은 ρ 직접 (두 팔 모두 런이 있는 ρ)
        direct = []
        for k_ in sorted({r['knob'] for r in arm}, key=float):
            aa = [r['spikes'] for r in arm if r['knob'] == k_]
            oo = [r['spikes'] for r in ours_all if abs(float(r['rho']) - float(k_)) < 1e-12]
            if aa and oo:
                d = {'rho': k_, 'n_arm': len(aa), 'n_ours': len(oo), 'arm_mean': float(np.mean(aa)),
                     'ours_mean': float(np.mean(oo)), 'pct': 100 * (np.mean(aa) / np.mean(oo) - 1)}
                if len(aa) >= 2 and len(oo) >= 2:
                    w_ = welch(np.log(aa), np.log(oo))
                    d.update(pct_lo=100 * (math.exp(w_['lo']) - 1), pct_hi=100 * (math.exp(w_['hi']) - 1), p=w_['p'])
                direct.append(d)
        o['direct_same_rho'] = direct
        o['interp'] = interp_F(ours_all, arm)
        out[a] = o
    # loss-ratio: 고정 λ 의 착지에 해당하는 ours ρ (역예측, ours 2e-3~4e-3 직선)
    ref = [r for r in ours_all if 2e-3 - 1e-12 <= float(r['rho']) <= 4e-3 + 1e-12]
    X = np.array([[1.0, math.log(float(r['rho'])) - math.log(3e-3), 1.0 if r['server'] == '23' else 0.0] for r in ref])
    f = ols(X, [math.log(r['spikes']) for r in ref])
    nl = G['ch_nolr']
    ybar = np.mean([math.log(r['spikes']) - f['beta'][2] * (r['server'] == '23') for r in nl])
    vy = np.var([math.log(r['spikes']) for r in nl], ddof=1) / len(nl)
    a_, b_ = ybar - f['beta'][0], f['beta'][1]
    va = vy + f['cov'][0, 0]
    lo, hi = fieller_ratio(a_, b_, va, f['cov'][1, 1], -f['cov'][0, 1], f['df'])
    out['ch_nolr'] = {'rho_equiv': 3e-3 * math.exp(a_ / b_), 'rho_equiv_lo': 3e-3 * math.exp(lo),
                      'rho_equiv_hi': 3e-3 * math.exp(hi), 'n_ref': len(ref), 'n_arm': len(nl)}
    return out


# ----------------------------------------------------------------------------- 3. 에폭 궤적 (λ · R · 실효 ρ · 스파이크)
TRAJ_EP = [0, 4, 9, 49, 99, 149, 199, 200, 209, 249, 299, 309]
PRE, POST = (100, 200), (200, 310)


def traj(r):
    rows = r['rows']
    if len(rows) != EPOCHS:
        return None
    s = np.array([x['s_count'] for x in rows])
    loss = np.array([x['loss'] for x in rows])
    rd = r['rd']
    Rrd = np.array([sum(rd[e][l]['sc_loss'] for l in LAYERS) if rd and e in rd else np.nan for e in range(EPOCHS)])
    o = {'s': s}
    if r['key'] == 'base':
        return o
    if 'adp_lambda' in rows[-1]:
        lam = np.r_[np.nan, np.array([x['adp_lambda'] for x in rows])[:-1]]
        o['lam'] = lam
        o['R'] = np.array([x['reg_R'] for x in rows])
        o['L'] = np.array([x['task_loss_clean'] for x in rows])
        o['rho_eff'] = np.array([x['loss_ratio'] for x in rows])
        o['R_rd_vs_log'] = float(np.nanmedian(np.abs(Rrd / o['R'] - 1)))
    else:
        lam = np.full(EPOCHS, float(r['lam_fixed']))
        o['lam'] = lam
        o['R'] = Rrd
        o['L'] = loss - lam * Rrd
        o['rho_eff'] = lam * Rrd / o['L']
    return o


def traj_summary(t):
    s = t['s']
    o = {'s_pre10': float(s[MIX_OFF - 10:MIX_OFF].mean()), 's_last10': float(s[-10:].mean())}
    o['mix_resp'] = o['s_last10'] / o['s_pre10'] - 1
    o['s_at'] = {e: float(s[e]) for e in TRAJ_EP}
    if 'lam' in t:
        for k in ('lam', 'R', 'L', 'rho_eff'):
            v = t[k]
            o[f'{k}_pre'] = float(np.nanmean(v[PRE[0]:PRE[1]]))
            o[f'{k}_post'] = float(np.nanmean(v[POST[0]:POST[1]]))
            o[f'{k}_end'] = float(np.nanmean(v[-10:]))
            o[f'{k}_at'] = {e: float(v[e]) for e in TRAJ_EP if np.isfinite(v[e])}
        o['rho_eff_post_over_pre'] = o['rho_eff_post'] / o['rho_eff_pre']
        o['lam_post_over_pre'] = o['lam_post'] / o['lam_pre']
        o['L_drop_mix'] = float(t['L'][MIX_OFF + 1:MIX_OFF + 11].mean() / t['L'][MIX_OFF - 10:MIX_OFF].mean() - 1)
        o['R_drop_mix'] = float(t['R'][MIX_OFF + 1:MIX_OFF + 11].mean() / t['R'][MIX_OFF - 10:MIX_OFF].mean() - 1)
    return o


def traj_section(G):
    """342K 비교점: 끈 팔 n=4 대 ours 3e-3 n=4. 에폭별 ln 스파이크 차와 λ·실효 ρ 요약."""
    out = {}
    o3 = G['ours_pt']
    T = {r['tag']: traj(r) for r in G['all_used']}
    S = {tag: traj_summary(t) for tag, t in T.items() if t is not None}
    out['per_run'] = S
    for a in ARMS:
        ap = [r for r in G[a] if r['knob'] == POINT[a]]
        o = {}
        # 에폭별 스파이크 비 (Welch, ln)
        o['s_ratio'] = {}
        for e in TRAJ_EP:
            w_ = welch([math.log(T[r['tag']]['s'][e]) for r in ap], [math.log(T[r['tag']]['s'][e]) for r in o3])
            o['s_ratio'][e] = {'pct': 100 * (math.exp(w_['d']) - 1), 'lo': 100 * (math.exp(w_['lo']) - 1),
                               'hi': 100 * (math.exp(w_['hi']) - 1), 'p': w_['p']}
        for k in ('mix_resp', 'lam_pre', 'lam_post', 'lam_end', 'R_end', 'L_end', 'rho_eff_pre', 'rho_eff_post',
                  'rho_eff_end', 'rho_eff_post_over_pre', 'lam_post_over_pre', 'L_drop_mix', 'R_drop_mix'):
            va = [S[r['tag']][k] for r in ap if k in S[r['tag']]]
            vo = [S[r['tag']][k] for r in o3 if k in S[r['tag']]]
            if len(va) >= 2 and len(vo) >= 2:
                o[k] = {'arm': msd(va), 'ours': msd(vo), 'welch': welch(va, vo)}
        out[a] = o
    # 같은 ρ 궤적 (두 팔 모두 런이 있는 ρ): - vmem 2e-3 (n=4) 대 ours 2e-3 (n=3), - 1- inversion 4.5e-3 대 ours 4.5e-3
    out['same_rho'] = {}
    for a, k_ in (('ch_novmem', '2e-3'), ('ch_noinv', '4.5e-3')):
        ap = [r for r in G[a] if r['knob'] == k_]
        oo = [r for r in G['ours_curve'] if r['knob'] == k_]
        o = {'rho': k_, 'n_arm': len(ap), 'n_ours': len(oo), 's_ratio': {}}
        for e in TRAJ_EP:
            w_ = welch([math.log(T[r['tag']]['s'][e]) for r in ap], [math.log(T[r['tag']]['s'][e]) for r in oo])
            o['s_ratio'][e] = {'pct': 100 * (math.exp(w_['d']) - 1), 'lo': 100 * (math.exp(w_['lo']) - 1),
                               'hi': 100 * (math.exp(w_['hi']) - 1), 'p': w_['p']}
        for k in ('lam_pre', 'lam_post', 'lam_end', 'R_end', 'L_end', 'mix_resp'):
            va = [S[r['tag']][k] for r in ap]
            vo = [S[r['tag']][k] for r in oo]
            o[k] = {'arm': msd(va), 'ours': msd(vo), 'welch': welch(va, vo)}
        out['same_rho'][a] = o
    # 같은 스파이크에서 λ_end · R_end · L_end (ours 곡선 대비 ANCOVA, ln 척도) — loss-ratio 팔만
    out['same_spk'] = {}
    for a in ('ch_novmem', 'ch_nofinal', 'ch_noinv'):
        o = {}
        for k in ('lam_end', 'R_end', 'L_end', 'lam_pre', 'lam_post'):
            yf = lambda r, k=k: math.log(S[r['tag']][k])
            e_ = ancova(G['ours_curve'], G[a], yf, lnspk_c)
            e_['ratio'], e_['ratio_lo'], e_['ratio_hi'] = math.exp(e_['d']), math.exp(e_['lo']), math.exp(e_['hi'])
            o[k] = e_
        out['same_spk'][a] = o
    # baseline 해제 반응 (n=6)
    bm = [S[r['tag']]['mix_resp'] for r in G['base']]
    out['base_mix_resp'] = msd(bm)
    out['R_rd_vs_log_max'] = max(t.get('R_rd_vs_log', 0) for t in T.values() if t)
    return out, T


# ----------------------------------------------------------------------------- 4. 학습 모드 기전 (마지막 에폭)
MECH_KEYS = ['m_mean', 'w_mean', 'w_sd', 'lnR', 'dead_pct', 'gini_sw', 'top10_sw',
             'share_conv1', 'share_conv2', 'share_conv3', 'share_conv4', 'share_inner',
             'Rshare_conv1', 'Rshare_conv2', 'Rshare_conv3', 'Rshare_conv4', 'Rshare_inner',
             'dead_conv1', 'dead_conv2', 'dead_conv3', 'dead_conv4', 'dead_inner', 'dead_outer',
             'w_conv1', 'w_conv2', 'w_conv3', 'w_conv4', 'w_inner', 'w_outer', 'lnpi_inner']


def mech_of(r):
    if not r['rd']:
        return None
    e = EPOCHS - 1
    o = mech_epoch(r['rd'], r['rnd'], e)
    if not o:
        return None
    # m = mean(c/max c) — 두 팔에서 같은 양. 켜짐: w = 1 − m, 꺼짐(- 1- inversion): w = m.
    inv = r['key'] == 'ch_noinv'
    o['m_mean'] = o['w_mean'] if inv else 1 - o['w_mean']
    for s in STAGES + ['inner', 'outer']:
        if f'w_{s}' in o:
            o[f'm_{s}'] = o[f'w_{s}'] if inv else 1 - o[f'w_{s}']
    o['lnR'] = math.log(o['R']) if o['R'] > 0 else float('nan')
    Ri = sum(o['R_layer'][l] for l in INNER)
    si = sum(o['spk_layer'][l] for l in INNER)
    o['lnpi_inner'] = math.log((Ri / si) / (o['R'] / o['tr_spk'])) if Ri > 0 and si > 0 else float('nan')
    return o


def mech_section(G):
    """학습 모드 지표: (a) 342K 직접 n=4 대 n=4, (b) ours 곡선 대비 ANCOVA (공변량 ln 학습 모드 스파이크)."""
    M = {r['tag']: mech_of(r) for r in G['all_used']}
    out = {'per_run': {t: {k: v for k, v in m.items() if not isinstance(v, dict)} for t, m in M.items() if m}}
    ours = [r for r in G['ours_curve'] if M.get(r['tag'])]
    o3 = [r for r in G['ours_pt'] if M.get(r['tag'])]
    keys = MECH_KEYS + ['m_conv1', 'm_conv2', 'm_conv3', 'm_conv4', 'm_inner', 'm_outer']
    for a in ARMS:
        ap = [r for r in G[a] if r['knob'] == POINT[a] and M.get(r['tag'])]
        aa = [r for r in G[a] if M.get(r['tag'])]
        res = {'n_pt': len(ap), 'n_all': len(aa), 'n_ref': len(ours), 'n_o3': len(o3)}
        for k in keys:
            va = [M[r['tag']].get(k) for r in ap]
            vo = [M[r['tag']].get(k) for r in o3]
            if None in va or None in vo or not all(np.isfinite(va + vo)):
                continue
            d = {'arm': msd(va), 'ours': msd(vo), 'direct': welch(va, vo)}
            xf = lambda r: math.log(M[r['tag']]['tr_spk']) - math.log(np.mean([M[x['tag']]['tr_spk'] for x in o3]))
            yf = lambda r, k=k: M[r['tag']][k]
            try:
                d['ancova'] = ancova(ours, aa, yf, xf)
            except Exception:
                pass
            res[k] = d
        # 층별 (18층) w·dead·스파이크 몫 직접 비교
        lay = {}
        for l in LAYERS:
            row = {}
            for nm, f in (('m', lambda m, l=l: (m['w_layer'][l] if a == 'ch_noinv' else 1 - m['w_layer'][l])),
                          ('dead', lambda m, l=l: m['dead_layer'][l]),
                          ('share', lambda m, l=l: 100 * m['spk_layer'][l] / m['tr_spk']),
                          ('Rshare', lambda m, l=l: 100 * m['R_layer'][l] / m['R'])):
                va = [f(M[r['tag']]) for r in ap]
                vo = [f(M[r['tag']]) for r in o3]
                row[nm] = {'arm': msd(va), 'ours': msd(vo), 'direct': welch(va, vo)}
            lay[l] = row
        res['layers'] = lay
        out[a] = res
    # ours 자체의 m 은 o3 의 m 과 아래 ours m (켜짐, w=1−m) 과 비교
    out['ours_pt_m'] = msd([M[r['tag']]['m_mean'] for r in o3])
    return out, M


def mech_traj_section(G, M_unused=None):
    """학습 모드 지표의 에폭 궤적 (342K 비교점, n=4 대 n=4)."""
    eps = [0, 9, 49, 99, 149, 199, 209, 249, 309]
    out = {}
    def at(r, e):
        o = mech_epoch(r['rd'], r['rnd'], e) if r['rd'] else {}
        if not o:
            return None
        inv = r['key'] == 'ch_noinv'
        o['m_mean'] = o['w_mean'] if inv else 1 - o['w_mean']
        return o
    cache = {}
    for r in G['ours_pt'] + [x for a in ARMS for x in G[a] if x['knob'] == POINT[a]]:
        cache[r['tag']] = {e: at(r, e) for e in eps}
    for a in ARMS:
        ap = [r for r in G[a] if r['knob'] == POINT[a]]
        res = {}
        for k in ('m_mean', 'dead_pct', 'gini_sw', 'top10_sw', 'share_conv1', 'share_conv4', 'R', 'tr_spk'):
            res[k] = {}
            for e in eps:
                va = [cache[r['tag']][e][k] for r in ap if cache[r['tag']][e]]
                vo = [cache[r['tag']][e][k] for r in G['ours_pt'] if cache[r['tag']][e]]
                if len(va) >= 2 and len(vo) >= 2:
                    if k in ('R', 'tr_spk'):
                        w_ = welch(np.log(va), np.log(vo))
                        res[k][e] = {'arm': msd(va), 'ours': msd(vo), 'pct': 100 * (math.exp(w_['d']) - 1), 'p': w_['p']}
                    else:
                        res[k][e] = {'arm': msd(va), 'ours': msd(vo), 'd': welch(va, vo)['d'], 'p': welch(va, vo)['p']}
        out[a] = res
    return out


# ----------------------------------------------------------------------------- 5. npz (테스트 1만 장, 평가 모드)
CONV = [l for l in LAYERS if l != 'fc1_n']


def gini(v):
    v = np.sort(np.asarray(v, float).ravel())
    n = len(v)
    if v.sum() <= 0:
        return float('nan')
    return float((2 * np.arange(1, n + 1) - n - 1) @ v / (n * v.sum()))


def npz_metrics(path):
    z = np.load(path)
    ns, T = int(z['n_samples']), int(z['T'])
    o = {'path': path, 'acc': float(z['acc']) if 'acc' in z else None}
    tot = {l: z[l].astype(np.float64) for l in LAYERS}
    allspk = sum(v.sum() for v in tot.values())
    o['npz_spk'] = allspk / ns
    N = sum(v.size for v in tot.values())
    o['silent_pct'] = 100 * sum((v == 0).sum() for v in tot.values()) / N
    o['active_img_pct'] = 100 * sum(z[l + '__fired'].astype(np.float64).sum() for l in LAYERS) / (ns * N)
    dead = 0
    per = {}
    mw, gw, tw = 0.0, 0.0, 0.0
    for l in CONV:
        c = tot[l].sum(axis=(0, 1))
        C = len(c)
        d = int((c == 0).sum())
        dead += d
        srt = np.sort(c)[::-1]
        k10 = max(1, int(round(0.1 * C)))
        m_l = float((c / c.max()).mean()) if c.max() > 0 else float('nan')
        per[l] = {'dead_ch': d, 'C': C, 'm': m_l, 'ch_gini': gini(c), 'top10ch': float(srt[:k10].sum() / c.sum()),
                  'spk': float(tot[l].sum() / ns), 'silent_pct': float(100 * (tot[l] == 0).mean()),
                  'low_ch': int((c < 0.01 * c.max()).sum())}
        sw = tot[l].sum()
        mw += m_l * tot[l].size
        gw += per[l]['ch_gini'] * sw
        tw += per[l]['top10ch'] * sw
    convN = sum(tot[l].size for l in CONV)
    convS = sum(tot[l].sum() for l in CONV)
    o['dead_ch_conv'] = dead
    o['dead_ch_conv_plus_fc'] = dead + int((tot['fc1_n'] == 0).sum())
    o['low_ch_conv'] = sum(per[l]['low_ch'] for l in CONV)
    o['m_eval'] = mw / convN
    o['ch_gini_sw'] = gw / convS
    o['top10ch_sw'] = 100 * tw / convS
    for s in STAGES[:-1]:
        ls = [l for l in CONV if l.startswith(s)]
        o[f'dead_ch_{s}'] = sum(per[l]['dead_ch'] for l in ls)
        o[f'm_eval_{s}'] = sum(per[l]['m'] * tot[l].size for l in ls) / sum(tot[l].size for l in ls)
        o[f'ch_gini_{s}'] = sum(per[l]['ch_gini'] * tot[l].sum() for l in ls) / sum(tot[l].sum() for l in ls)
        o[f'share_{s}'] = 100 * sum(tot[l].sum() for l in ls) / allspk
    o['dead_ch_inner'] = sum(per[l]['dead_ch'] for l in INNER)
    # 21시 추가: 뉴런 단위 집중도 (테스트셋 합, 층 스파이크 가중 — 학습 모드 gini_sw 와 같은 가중, conv 17층),
    # 층·단계별 이미지당 스파이크, 침묵 (테스트셋 전체에서 0) 과 이미지당 활성 뉴런
    ng = nt = 0.0
    for l in CONV:
        v = tot[l].ravel()
        srt = np.sort(v)[::-1]
        k10 = max(1, int(round(0.1 * v.size)))
        per[l]['n_gini'] = gini(v)
        per[l]['n_top10'] = float(srt[:k10].sum() / v.sum()) if v.sum() > 0 else float('nan')
        per[l]['active_img_pct'] = float(100 * z[l + '__fired'].astype(np.float64).sum() / (ns * v.size))
        ng += per[l]['n_gini'] * v.sum()
        nt += per[l]['n_top10'] * v.sum()
    o['n_gini_sw'] = ng / convS
    o['n_top10_sw'] = 100 * nt / convS
    for s in STAGES[:-1]:
        ls = [l for l in CONV if l.startswith(s)]
        nN = sum(tot[l].size for l in ls)
        o[f'spk_{s}'] = float(sum(tot[l].sum() for l in ls) / ns)
        o[f'silent_{s}'] = float(100 * sum((tot[l] == 0).sum() for l in ls) / nN)
        o[f'active_{s}'] = float(sum(per[l]['active_img_pct'] * tot[l].size for l in ls) / nN)
        o[f'n_gini_{s}'] = float(sum(per[l]['n_gini'] * tot[l].sum() for l in ls) / sum(tot[l].sum() for l in ls))
    o['spk_fc1'] = float(tot['fc1_n'].sum() / ns)
    o['layers'] = per
    # 시간 단계 (있으면): 스텝별 몫과 최초 발화 스텝
    if LAYERS[0] + '__t' in z:
        st = np.zeros(T)
        stg = {s: np.zeros(T) for s in STAGES}
        for l in LAYERS:
            v = z[l + '__t'].astype(np.float64).reshape(T, -1).sum(axis=1)
            st += v
            stg[[s for s in STAGES if l.startswith(s)][0]] += v
        o['t_share'] = list(100 * st / st.sum())
        o['t_share_stage'] = {s: list(100 * v / v.sum()) for s, v in stg.items() if v.sum() > 0}
        ft = sum(z[l + '__first_t'].astype(np.float64) for l in LAYERS)
        fired = ft[:T].sum()
        o['first_t_share'] = list(100 * ft[:T] / fired)       # 발화한 (표본, 뉴런) 쌍 중 처음 쏜 스텝
    return o


def npz_find(tag):
    for d in NPZ_DIRS:
        p = os.path.join(d, tag + '.npz')
        if os.path.exists(p):
            try:
                np.load(p)['n_samples']
                return p
            except Exception:
                continue
    return None


NPZ_STAGE_KEYS = [f'{p}_{s}' for p in ('dead_ch', 'm_eval', 'ch_gini', 'n_gini', 'share', 'silent', 'active', 'spk')
                  for s in STAGES[:-1]]
NPZ_KEYS = ['npz_spk', 'silent_pct', 'active_img_pct', 'dead_ch_conv', 'dead_ch_conv_plus_fc', 'low_ch_conv',
            'm_eval', 'ch_gini_sw', 'top10ch_sw', 'n_gini_sw', 'n_top10_sw', 'dead_ch_inner'] + NPZ_STAGE_KEYS
# 팔 안 Holm 묶음 (21시 주 지표 12개 — 활성·침묵·죽은 채널·집중도·단계 배분·m)
NPZ_PRIMARY = ['m_eval', 'ch_gini_sw', 'top10ch_sw', 'n_gini_sw', 'n_top10_sw', 'silent_pct', 'active_img_pct',
               'dead_ch_conv', 'share_conv1', 'share_conv2', 'share_conv3', 'share_conv4']
NPZ_LAYER_KEYS = ('dead_ch', 'm', 'ch_gini', 'top10ch', 'silent_pct', 'n_gini', 'active_img_pct', 'share')


def adj_direct(ref, arm_pt, o3, yf, xf):
    """스파이크 보정 직접 비교: ours npz 곡선 (선형 + 서버 항) 의 기울기 b 로 각 런을 ours 3e-3 평균 npz 스파이크로
    옮긴 뒤 Welch. SE 에 (Δx)²·var(b) 를 더한다 (곡선 기울기 불확실성 전파)."""
    Xc = np.array([[1.0, xf(r), 1.0 if r['server'] == '23' else 0.0] for r in ref])
    fc = ols(Xc, [yf(r) for r in ref])
    b, vb = float(fc['beta'][1]), float(fc['cov'][1, 1])
    aa = [yf(r) - b * xf(r) for r in arm_pt]
    ao = [yf(r) - b * xf(r) for r in o3]
    w_ = welch(aa, ao)
    dx = float(np.mean([xf(r) for r in arm_pt]) - np.mean([xf(r) for r in o3]))
    o = est(float(np.mean(aa) - np.mean(ao)), math.sqrt(w_['se'] ** 2 + dx * dx * vb), w_['df'])
    o.update(dx_ln=dx, slope=b, slope_se=math.sqrt(vb), na=len(aa), nb=len(ao))
    return o


def _ln_pct(va, vo):
    w_ = welch(np.log(va), np.log(vo))
    return {'pct': 100 * (math.exp(w_['d']) - 1), 'lo': 100 * (math.exp(w_['lo']) - 1),
            'hi': 100 * (math.exp(w_['hi']) - 1), 'p': w_['p'], 'df': w_['df'], 'arm': msd(va), 'ours': msd(vo),
            'na': len(va), 'nb': len(vo)}


def npz_section(G):
    tags = [r['tag'] for r in G['all_used']]
    met = {}
    for t in tags:
        p = npz_find(t)
        if p:
            met[t] = npz_metrics(p)
    out = {'have': {t: met[t]['path'] for t in met}}
    out['per_run'] = {t: {k: v for k, v in m.items() if k != 'layers'} for t, m in met.items()}
    ours_c = [r for r in G['ours_curve'] if r['tag'] in met]
    o3 = [r for r in G['ours_pt'] if r['tag'] in met]
    base = [r for r in G['base'] if r['tag'] in met]
    out['n_ours_curve'], out['n_o3'], out['n_base'] = len(ours_c), len(o3), len(base)
    out['ours_curve_tags'] = [r['tag'] for r in ours_c]
    x0 = math.log(np.mean([met[r['tag']]['npz_spk'] for r in o3]))
    out['x0_npz_spk'] = math.exp(x0)
    xf = lambda r: math.log(met[r['tag']]['npz_spk']) - x0
    lshare = lambda m, l: 100 * m['layers'][l]['spk'] / m['npz_spk']
    for a in ARMS:
        ap = [r for r in G[a] if r['knob'] == POINT[a] and r['tag'] in met]
        aa = [r for r in G[a] if r['tag'] in met]
        res = {'n_pt': len(ap), 'n_all': len(aa), 'tags_pt': [r['tag'] for r in ap], 'tags_all': [r['tag'] for r in aa],
               'servers_pt': [r['server'] for r in ap],
               'missing_pt': [r['tag'] for r in G[a] if r['knob'] == POINT[a] and r['tag'] not in met]}
        for k in NPZ_KEYS:
            yf = lambda r, k=k: met[r['tag']][k]
            va = [yf(r) for r in ap]
            vo = [yf(r) for r in o3]
            d = {'arm': msd(va) if va else None, 'ours': msd(vo) if vo else None, 'arm_vals': va, 'ours_vals': vo}
            if k == 'npz_spk':
                if len(va) >= 2 and len(vo) >= 2:
                    d['direct'] = welch(va, vo)
                    d['ln_pct'] = _ln_pct(va, vo)
                res[k] = d
                continue
            if len(va) >= 2 and len(vo) >= 2:
                d['direct'] = welch(va, vo)
                d['adj'] = adj_direct(ours_c, ap, o3, yf, xf)
            if aa and len(ours_c) >= 5:
                d['ancova'] = ancova(ours_c, aa, yf, xf)
                if ap and len(ap) != len(aa):
                    d['ancova_pt'] = ancova(ours_c, ap, yf, xf)
            res[k] = d
        for src in ('ancova', 'adj', 'direct'):
            ks = [k for k in NPZ_PRIMARY if src in res[k]]
            if ks:
                for k, h in zip(ks, holm([res[k][src]['p'] for k in ks])):
                    res[k][src]['holm'] = h
        # 시간 단계
        tt_a = [met[r['tag']] for r in ap if 't_share' in met[r['tag']]]
        tt_o = [met[r['tag']] for r in o3 if 't_share' in met[r['tag']]]
        if tt_a and tt_o:
            res['t_share'] = {'n_arm': len(tt_a), 'n_ours': len(tt_o), 'steps': [],
                              'ours_tags': [r['tag'] for r in o3 if 't_share' in met[r['tag']]]}
            for j in range(4):
                va = [m['t_share'][j] for m in tt_a]
                vo = [m['t_share'][j] for m in tt_o]
                row = {'arm': msd(va), 'ours': msd(vo)}
                if len(va) >= 2 and len(vo) >= 2:
                    row['direct'] = welch(va, vo)
                res['t_share']['steps'].append(row)
            res['t_share_stage'] = {}
            for s in STAGES:
                rows_ = []
                for j in range(4):
                    va = [m['t_share_stage'][s][j] for m in tt_a if s in m['t_share_stage']]
                    vo = [m['t_share_stage'][s][j] for m in tt_o if s in m['t_share_stage']]
                    row = {'arm': msd(va), 'ours': msd(vo)}
                    if len(va) >= 2 and len(vo) >= 2:
                        row['direct'] = welch(va, vo)
                    rows_.append(row)
                res['t_share_stage'][s] = rows_
            res['first_t'] = []
            for j in range(4):
                va = [m['first_t_share'][j] for m in tt_a]
                vo = [m['first_t_share'][j] for m in tt_o]
                row = {'arm': msd(va), 'ours': msd(vo)}
                if len(va) >= 2 and len(vo) >= 2:
                    row['direct'] = welch(va, vo)
                res['first_t'].append(row)
        # 층별 (17 conv 층, 342K 직접 + 스파이크 보정)
        lay = {}
        for l in CONV:
            row = {}
            for k in NPZ_LAYER_KEYS:
                yf = (lambda r, l=l: lshare(met[r['tag']], l)) if k == 'share' else \
                     (lambda r, l=l, k=k: met[r['tag']]['layers'][l][k])
                va = [yf(r) for r in ap]
                vo = [yf(r) for r in o3]
                if va and vo:
                    row[k] = {'arm': msd(va), 'ours': msd(vo)}
                    if len(va) >= 2 and len(vo) >= 2:
                        row[k]['direct'] = welch(va, vo)
                        row[k]['adj'] = adj_direct(ours_c, ap, o3, yf, xf)
            lay[l] = row
        res['layers'] = lay
        out[a] = res
    out['base'] = {k: msd([met[r['tag']][k] for r in base]) for k in NPZ_KEYS} if len(base) >= 2 else None
    out['ours_pt'] = {k: msd([met[r['tag']][k] for r in o3]) for k in NPZ_KEYS} if len(o3) >= 2 else None
    tb = [met[r['tag']] for r in base if 't_share' in met[r['tag']]]
    out['base_t_share'] = {'n': len(tb), 'steps': [msd([m['t_share'][j] for m in tb]) for j in range(4)],
                           'first_t': [msd([m['first_t_share'][j] for m in tb]) for j in range(4)]} if len(tb) >= 2 else None
    return out, met


def npz_vmem_rho(G, met, M):
    """같은 ρ 에서 vmem 이 있으면 더 쏘나, 어디서 — 두 팔 모두 스파이크만으로 잰 값.
    (a) 평가 모드 npz 직접: ours 2e-3 대 `- vmem` 2e-3. (b) 평가 모드 ρ 2차 모형: ours npz 전체 + `- vmem` npz 전체.
    (c) 학습 모드 (reg_detail spike_count, 마지막 에폭) 직접 2e-3. (d) 학습 모드 ρ 2차 모형: ours 전체 + `- vmem` 전체.
    (e) 네트워크 쪽 m (스파이크만, 채널 테스트셋 합) 을 같은 ρ 에서 비교."""
    out = {}
    o2 = [r for r in G['ours_all'] if r['knob'] == '2e-3']
    v2 = [r for r in G['ch_novmem'] if r['knob'] == '2e-3']
    o2n = [r for r in o2 if r['tag'] in met]
    v2n = [r for r in v2 if r['tag'] in met]
    out['n'] = {'ours2e3_snapshot': len(o2), 'novmem2e3_snapshot': len(v2), 'ours2e3_npz': len(o2n),
                'novmem2e3_npz': len(v2n), 'ours_tags': [(r['tag'], r['server']) for r in o2n],
                'arm_tags': [(r['tag'], r['server']) for r in v2n]}
    stg = STAGES[:-1]
    d = {}
    for k in ['npz_spk'] + [f'spk_{s}' for s in stg] + ['spk_fc1']:
        d[k] = _ln_pct([met[r['tag']][k] for r in v2n], [met[r['tag']][k] for r in o2n])
    # 단계별 기여: (끈 − 켜짐) 이미지당 스파이크 차와 총 차에서의 몫
    tot_diff = np.mean([met[r['tag']]['npz_spk'] for r in v2n]) - np.mean([met[r['tag']]['npz_spk'] for r in o2n])
    d['contrib'] = {}
    for s in stg + ['fc1']:
        dd = np.mean([met[r['tag']][f'spk_{s}'] for r in v2n]) - np.mean([met[r['tag']][f'spk_{s}'] for r in o2n])
        d['contrib'][s] = {'diff_per_img': float(dd), 'frac_of_total': float(dd / tot_diff) if tot_diff else None}
    d['contrib']['total_diff_per_img'] = float(tot_diff)
    for k in ['m_eval'] + [f'm_eval_{s}' for s in stg] + ['silent_pct', 'active_img_pct', 'dead_ch_conv', 'n_gini_sw',
                                                         'ch_gini_sw', 'top10ch_sw'] + [f'share_{s}' for s in stg]:
        va = [met[r['tag']][k] for r in v2n]
        vo = [met[r['tag']][k] for r in o2n]
        d[k] = {'arm': msd(va), 'ours': msd(vo), 'direct': welch(va, vo)}
    lay = {}
    for l in CONV:
        va = [met[r['tag']]['layers'][l]['spk'] for r in v2n]
        vo = [met[r['tag']]['layers'][l]['spk'] for r in o2n]
        lay[l] = {'spk': _ln_pct(va, vo),
                  'm': welch([met[r['tag']]['layers'][l]['m'] for r in v2n], [met[r['tag']]['layers'][l]['m'] for r in o2n])}
    d['layers'] = lay
    out['eval_direct'] = d

    # (b)·(d) ρ 2차 모형: y = a + c·u + c2·u² + e·[23] + δ·[- vmem], u = ln ρ − ln 2e-3
    c0 = math.log(2e-3)
    def rho_model(ours_rs, arm_rs, yf, log=True):
        rows = [(r, 0) for r in ours_rs] + [(r, 1) for r in arm_rs]
        X = np.array([[1.0, math.log(float(r['rho'])) - c0, (math.log(float(r['rho'])) - c0) ** 2,
                       1.0 if r['server'] == '23' else 0.0, float(g)] for r, g in rows])
        y = [math.log(yf(r)) if log else yf(r) for r, _ in rows]
        f = ols(X, y)
        e_ = est(f['beta'][4], f['se'][4], f['df'])
        if log:
            e_.update(pct=100 * (math.exp(e_['d']) - 1), pct_lo=100 * (math.exp(e_['lo']) - 1),
                      pct_hi=100 * (math.exp(e_['hi']) - 1))
        e_.update(n_ours=len(ours_rs), n_arm=len(arm_rs))
        return e_
    on = [r for r in G['ours_all'] if r['tag'] in met]
    vn = [r for r in G['ch_novmem'] if r['tag'] in met]
    mb = {'n_ours': len(on), 'n_arm': len(vn), 'arm_tags': [r['tag'] for r in vn]}
    for k in ['npz_spk'] + [f'spk_{s}' for s in stg]:
        mb[k] = rho_model(on, vn, lambda r, k=k: met[r['tag']][k])
    for k in ['m_eval'] + [f'm_eval_{s}' for s in stg] + ['silent_pct', 'n_gini_sw']:
        mb[k] = rho_model(on, vn, lambda r, k=k: met[r['tag']][k], log=False)
    mb['layers'] = {l: rho_model(on, vn, lambda r, l=l: met[r['tag']]['layers'][l]['spk']) for l in CONV}
    # 같은 모형을 snapshot s_count 로 (npz 가 있는 런만) — npz 총합과 s_count 의 차이 점검
    mb['s_count_same_runs'] = rho_model(on, vn, lambda r: r['spikes'])
    out['eval_rho_model'] = mb

    # (c) 학습 모드 직접 (마지막 에폭 reg_detail spike_count, 층 합)
    stage_sp = lambda m, s: sum(v for l, v in m['spk_layer'].items() if l.startswith(s))
    o2m = [r for r in o2 if M.get(r['tag'])]
    v2m = [r for r in v2 if M.get(r['tag'])]
    tdir = {'n_ours': len(o2m), 'n_arm': len(v2m)}
    tdir['tr_spk'] = _ln_pct([M[r['tag']]['tr_spk'] for r in v2m], [M[r['tag']]['tr_spk'] for r in o2m])
    for s in STAGES:
        tdir[f'spk_{s}'] = _ln_pct([stage_sp(M[r['tag']], s) for r in v2m], [stage_sp(M[r['tag']], s) for r in o2m])
    out['train_direct'] = tdir
    # (d) 학습 모드 ρ 2차 모형
    om = [r for r in G['ours_all'] if M.get(r['tag'])]
    vm = [r for r in G['ch_novmem'] if M.get(r['tag'])]
    tm = {'n_ours': len(om), 'n_arm': len(vm)}
    tm['tr_spk'] = rho_model(om, vm, lambda r: M[r['tag']]['tr_spk'])
    for s in STAGES:
        tm[f'spk_{s}'] = rho_model(om, vm, lambda r, s=s: stage_sp(M[r['tag']], s))
    tm['layers'] = {l: rho_model(om, vm, lambda r, l=l: M[r['tag']]['spk_layer'][l]) for l in LAYERS}
    out['train_rho_model'] = tm
    return out


def npz_extractor_check():
    """같은 체크포인트 (ours 3e-3 s2·s3) 를 10-02 옛 추출기 (_npz) 와 10-19 새 추출기 (npz_abl) 로 뽑은 값 대조."""
    out = {}
    for s in (2, 3):
        t = f'r19c10-ours_ch-3e-3-s{s}'
        po, pn = os.path.join(ROOT, '_npz', t + '.npz'), os.path.join(A, 'npz_abl', t + '.npz')
        if not (os.path.exists(po) and os.path.exists(pn)):
            continue
        zo, zn = np.load(po), np.load(pn)
        out[t] = {'max_abs_diff_counts': max(float(np.abs(zo[l].astype(np.float64) - zn[l].astype(np.float64)).max())
                                             for l in LAYERS),
                  'max_abs_diff_fired': max(float(np.abs(zo[l + '__fired'].astype(np.float64) -
                                                         zn[l + '__fired'].astype(np.float64)).max()) for l in LAYERS),
                  'acc_old': float(zo['acc']), 'acc_new': float(zn['acc']),
                  'n_samples': [int(zo['n_samples']), int(zn['n_samples'])],
                  'keys_old': len(zo.files), 'keys_new': len(zn.files),
                  'new_has_step': LAYERS[0] + '__t' in zn.files, 'old_has_step': LAYERS[0] + '__t' in zo.files}
    return out


def npz_acc_check(G, met):
    """npz 정확도 (테스트 1만 장, 추출 때 계산) 와 best val acc 대조 — 체크포인트가 best 에폭인지."""
    rows = []
    for r in G['all_used']:
        if r['tag'] in met and met[r['tag']]['acc'] is not None:
            rows.append({'tag': r['tag'], 'npz_acc': met[r['tag']]['acc'], 'val_acc': r['val_acc'],
                         'diff': met[r['tag']]['acc'] - r['val_acc'],
                         'npz_over_scount': met[r['tag']]['npz_spk'] / r['spikes']})
    d = [abs(x['diff']) for x in rows]
    return {'n': len(rows), 'max_abs_diff': max(d) if d else None,
            'n_gt_0p015': sum(1 for x in d if x > 0.015), 'rows': rows}


# ----------------------------------------------------------------------------- 6. 학습 시간 (같은 서버·같은 시기 짝)
def timeline(r, finished):
    secs = np.array([x['sec'] for x in r['rows']], float)
    if not np.all(np.isfinite(secs)) or not finished:
        return None
    import datetime as dt
    end = dt.datetime.strptime(finished, '%Y-%m-%d %H:%M').timestamp()
    ends = end - np.r_[np.cumsum(secs[::-1])[::-1][1:], 0.0]
    starts = ends - secs
    return starts, ends, secs


def timing_section(G):
    snap = {r['tag']: r for r in json.load(open(SNAP))['runs']}
    TL = {}
    for r in G['all_used']:
        t = timeline(r, snap[r['tag']].get('finished'))
        if t:
            TL[r['tag']] = t
    out = {'per_run': {r['tag']: {'sec_med': r['sec_med'], 'server': r['server'], 'key': r['key'],
                                  'start': float(TL[r['tag']][0][0]) if r['tag'] in TL else None}
                       for r in G['all_used']}}
    # 짝: 같은 서버, 겹치는 구간의 에폭이 두 런 모두 30개 이상. 그 구간 안 각 런의 에폭 시간 중앙값의 비.
    pairs = []
    tags = [r for r in G['all_used'] if r['tag'] in TL]
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            a, b = tags[i], tags[j]
            if a['server'] != b['server'] or a['key'] == b['key']:
                continue
            sa, ea, xa = TL[a['tag']]
            sb, eb, xb = TL[b['tag']]
            lo, hi = max(sa[1], sb[1]), min(ea[-1], eb[-1])
            if hi <= lo:
                continue
            ma = (sa >= lo) & (ea <= hi)
            mb = (sb >= lo) & (eb <= hi)
            ma[0] = mb[0] = False
            if ma.sum() < 30 or mb.sum() < 30:
                continue
            pairs.append({'a': a['tag'], 'b': b['tag'], 'ka': a['key'], 'kb': b['key'], 'server': a['server'],
                          'n_ep': [int(ma.sum()), int(mb.sum())], 'sec': [float(np.median(xa[ma])), float(np.median(xb[mb]))],
                          'ratio': float(np.median(xa[ma]) / np.median(xb[mb]))})
    out['pairs'] = pairs
    # 팔별 요약: 끈 팔 대 다른 키 (final_step 켜진 쪽)
    summ = {}
    for a in ARMS + ['ours_ch']:
        rr = []
        for p in pairs:
            if p['ka'] == a and p['kb'] != a and p['kb'] != 'base':
                rr.append((p['kb'], p['ratio'], p['server']))
            elif p['kb'] == a and p['ka'] != a and p['ka'] != 'base':
                rr.append((p['ka'], 1 / p['ratio'], p['server']))
        summ[a] = rr
    out['summary'] = summ
    return out


# ----------------------------------------------------------------------------- 7. 검토 반영 (10-19 eip-critic)
def _two_offset(ref, arm_pt, arm_rest, yf=acc_y, xf=lnspk_c):
    """y = a + b·x + e·[23] + d_pt·[끈 팔 비교점 세트] + d_rest·[끈 팔 나머지]. 두 오프셋과 그 차."""
    rows = [(r, 0, 0) for r in ref] + [(r, 1, 0) for r in arm_pt] + [(r, 0, 1) for r in arm_rest]
    X = np.array([[1.0, xf(r), 1.0 if r['server'] == '23' else 0.0, float(p), float(q)] for r, p, q in rows])
    f = ols(X, [yf(r) for r, _, _ in rows])
    c = np.array([0, 0, 0, 1.0, -1.0])
    dd = float(c @ f['beta'])
    se = math.sqrt(float(c @ f['cov'] @ c))
    return {'d_pt': est(f['beta'][3], f['se'][3], f['df']), 'd_rest': est(f['beta'][4], f['se'][4], f['df']),
            'diff': est(dd, se, f['df']), 'n_ref': len(ref), 'n_pt': len(arm_pt), 'n_rest': len(arm_rest),
            'rest_knobs': sorted({r['knob'] for r in arm_rest}, key=float)}


def _acc_last10(r):
    return 100 * float(np.mean([x['val_acc'] for x in r['rows'][-10:]]))


def _lnspk_last10(r):
    return math.log(float(np.mean([x['s_count'] for x in r['rows'][-10:]]))) - X0


def _acc_last(r):
    return 100 * r['rows'][-1]['val_acc']


def _lnspk_last(r):
    return math.log(r['rows'][-1]['s_count']) - X0


def review_section(G, res, T, M):
    out = {}
    ours = G['ours_curve']
    o3 = G['ours_pt']
    # (1) 두 오프셋 모형 — 비교점 세트 대 나머지 (기울기 상호작용과 구별)
    out['two_offset'] = {}
    for a in ARMS:
        ap = [r for r in G[a] if r['knob'] == POINT[a]]
        rest = [r for r in G[a] if r['knob'] != POINT[a]]
        if ap and rest:
            out['two_offset'][a] = _two_offset(ours, ap, rest)
    # (2) 342K 세트 분석을 1차로 볼 때의 Holm (네 팔; loss-ratio 는 세트 = 전체)
    pts = [res['acc'][a]['lin_srv_pt']['p'] for a in ARMS]
    out['holm_pt'] = dict(zip(ARMS, holm(pts)))
    out['p_pt'] = dict(zip(ARMS, pts))
    # (3) 정확도 지표 민감도: 마지막 10에폭 평균, 마지막 에폭
    out['acc_metric'] = {}
    for nm, yf, xf in (('last10', _acc_last10, _lnspk_last10), ('last10_xbest', _acc_last10, lnspk_c),
                       ('last', _acc_last, _lnspk_last)):
        oo = {}
        for a in ARMS:
            ap = [r for r in G[a] if r['knob'] == POINT[a]]
            e_all = ancova(ours, G[a], yf, xf)
            e_pt = ancova(ours, ap, yf, xf)
            loo = [ancova(ours, G[a][:i] + G[a][i + 1:], yf, xf) for i in range(len(G[a]))]
            a23 = [r for r in G[a] if r['server'] == '23']
            oo[a] = {'all': e_all, 'pt': e_pt, 'direct_welch': welch([yf(r) for r in ap], [yf(r) for r in o3]),
                     'loo_n_p_gt_05': sum(1 for x in loo if x['p'] > 0.05), 'loo_n': len(loo),
                     'loo_d_range': [min(x['d'] for x in loo), max(x['d'] for x in loo)],
                     'loo_p_range': [min(x['p'] for x in loo), max(x['p'] for x in loo)],
                     'srv23only': ancova(ours, a23, yf, xf) if a23 else None}
        oo['holm_all'] = dict(zip(ARMS, holm([oo[a]['all']['p'] for a in ARMS])))
        oo['holm_pt'] = dict(zip(ARMS, holm([oo[a]['pt']['p'] for a in ARMS])))
        out['acc_metric'][nm] = oo
    # (4) 기준 쪽 하나 빼기: ours 3e-3 4런 중 하나씩 빼고 342K 직접 비교 / baseline 비교
    out['loo_ours_side'] = {}
    for a in ARMS:
        ap = [r['val_acc'] for r in G[a] if r['knob'] == POINT[a]]
        rows_ = []
        for i, r in enumerate(o3):
            rest = [x['val_acc'] for j, x in enumerate(o3) if j != i]
            w_ = welch(ap, rest)
            rows_.append({'drop': r['tag'], 'd': w_['d'], 'p': w_['p'], 'lo': w_['lo'], 'hi': w_['hi']})
        out['loo_ours_side'][a] = rows_
    bv = [r['val_acc'] for r in G['base']]
    out['ours_pt_vs_base_loo'] = [{'drop': r['tag'], 'd': welch([x['val_acc'] for j, x in enumerate(o3) if j != i], bv)['d']}
                                  for i, r in enumerate(o3)]
    # (5) vmem 층별 m 차와 침묵 비율 (17 conv 층)
    conv = [l for l in LAYERS if l != 'fc1_n']
    ours_m = [r for r in ours if M.get(r['tag'])]
    o3m = [r for r in o3 if M.get(r['tag'])]
    nv_all = [r for r in G['ch_novmem'] if M.get(r['tag'])]
    nv_pt = [r for r in nv_all if r['knob'] == POINT['ch_novmem']]
    x0m = math.log(np.mean([M[r['tag']]['tr_spk'] for r in o3m]))
    xf = lambda r: math.log(M[r['tag']]['tr_spk']) - x0m
    lay = {}
    for l in conv:
        yf = lambda r, l=l: 1 - M[r['tag']]['w_layer'][l]       # 켜짐·`- vmem` 모두 w = 1 − m
        e_ = ancova(ours_m, nv_all, yf, xf)
        lay[l] = {'m_gain_ancova': -e_['d'], 'lo': -e_['hi'], 'hi': -e_['lo'], 'p': e_['p'],
                  'm_gain_direct': float(np.mean([yf(r) for r in o3m]) - np.mean([yf(r) for r in nv_pt])),
                  'silent_ours3e3': float(np.mean([M[r['tag']]['dead_layer'][l] for r in o3m])),
                  'silent_novmem_pt': float(np.mean([M[r['tag']]['dead_layer'][l] for r in nv_pt]))}
    g = [lay[l]['m_gain_ancova'] for l in conv]
    gd = [lay[l]['m_gain_direct'] for l in conv]
    so = [lay[l]['silent_ours3e3'] for l in conv]
    sn = [lay[l]['silent_novmem_pt'] for l in conv]
    sp = lambda a_, b_: dict(zip(('rho', 'p'), map(float, stats.spearmanr(a_, b_))))
    out['vmem_layer_m_vs_silent'] = {
        'note': 'm_gain = ours_inter_ch − `- vmem` (양수 = vmem 이 m 을 올림). 침묵 = 학습 모드 마지막 에폭 dead_neuron_ratio (%), 층 평균',
        'n_layers': len(conv), 'n_ref': len(ours_m), 'n_arm': len(nv_all), 'n_o3': len(o3m), 'n_arm_pt': len(nv_pt),
        'layers': lay,
        'spearman_gain_ancova_vs_silent_ours': sp(g, so),
        'spearman_gain_ancova_vs_silent_novmem': sp(g, sn),
        'spearman_gain_direct_vs_silent_ours': sp(gd, so),
        'spearman_negdiff_vs_silent_ours': sp([-v for v in g], so)}
    # (6) 지니: baseline 과 끈 팔은 같은 스파이크가 아님 — 차이와 ours 곡선 기울기로 본 스파이크 몫
    bm = [r for r in G['base'] if M.get(r['tag'])]
    ni = [r for r in G['ch_noinv'] if r['knob'] == POINT['ch_noinv'] and M.get(r['tag'])]
    X = np.array([[1.0, math.log(M[r['tag']]['tr_spk']), 1.0 if r['server'] == '23' else 0.0] for r in ours_m])
    fg = ols(X, [M[r['tag']]['gini_sw'] for r in ours_m])
    tb = float(np.mean([M[r['tag']]['tr_spk'] for r in bm]))
    tn = float(np.mean([M[r['tag']]['tr_spk'] for r in ni]))
    to = float(np.mean([M[r['tag']]['tr_spk'] for r in o3m]))
    out['gini_vs_base'] = {
        'base': {'n': len(bm), 'gini': msd([M[r['tag']]['gini_sw'] for r in bm]), 'tr_spk': tb,
                 'missing_rd': [r['tag'] for r in G['base'] if not M.get(r['tag'])]},
        'ours_pt': {'n': len(o3m), 'gini': msd([M[r['tag']]['gini_sw'] for r in o3m]), 'tr_spk': to},
        'noinv_pt': {'n': len(ni), 'gini': msd([M[r['tag']]['gini_sw'] for r in ni]), 'tr_spk': tn},
        'ours_slope_per_ln_spk': est(fg['beta'][1], fg['se'][1], fg['df']),
        'base_over_noinv_spk': tb / tn - 1,
        'gini_shift_from_spk_gap': float(fg['beta'][1] * math.log(tb / tn))}
    # (7) 학습 시간 짝 — 런 수 병기
    pairs = res['timing']['pairs']
    def pair_set(off, on_keys):
        sel = []
        for p in pairs:
            if p['ka'] == off and p['kb'] in on_keys:
                sel.append((p['b'], p['a'], 1 / p['ratio'], p['kb']))     # (on, off, on/off)
            elif p['kb'] == off and p['ka'] in on_keys:
                sel.append((p['a'], p['b'], p['ratio'], p['ka']))
        if not sel:
            return None
        rr = np.array([s[2] for s in sel])
        return {'n_pairs': len(sel), 'n_off_runs': len({s[1] for s in sel}), 'n_on_runs': len({s[0] for s in sel}),
                'off_runs': sorted({s[1] for s in sel}), 'on_over_off_median': float(np.median(rr)),
                'on_over_off_range': [float(rr.min()), float(rr.max())],
                'by_on_key': {k: {'n': int(sum(1 for s in sel if s[3] == k)),
                                  'median': float(np.median([s[2] for s in sel if s[3] == k]))}
                              for k in sorted({s[3] for s in sel})}}
    out['timing'] = {'vmem': pair_set('ch_novmem', {'ours_ch', 'ch_noinv', 'ch_nolr'}),
                     'final_step': pair_set('ch_nofinal', {'ours_ch', 'ch_noinv', 'ch_nolr'})}
    secs = {}
    for k in KEYS:
        v = [(r['sec_med'], r['server']) for r in G['all_used'] if r['key'] == k and r.get('sec_med')]
        if v:
            secs[k] = {s: [min(x for x, sv in v if sv == s), max(x for x, sv in v if sv == s), sum(1 for _, sv in v if sv == s)]
                       for s in ('138', '23') if any(sv == s for _, sv in v)}
    out['timing']['sec_med_range_by_key_server'] = secs
    # (8) loss-ratio: 해제 뒤 실효 ρ 배율 — 런별, 새 3런만
    S = res['traj']['per_run']
    nl = sorted(G['ch_nolr'], key=lambda r: r['seed'])
    out['nolr_ratio'] = {r['tag']: {'server': r['server'], 'rho_eff_pre': S[r['tag']]['rho_eff_pre'],
                                    'rho_eff_post': S[r['tag']]['rho_eff_post'],
                                    'ratio': S[r['tag']]['rho_eff_post_over_pre'], 'mix_resp': S[r['tag']]['mix_resp']}
                         for r in nl}
    # 고정 λ 런의 L 복원 (L = loss − λ·R_rd) 검증: loss-ratio 런에서 같은 식을 task_loss_clean 과 대조
    errs = {}
    for r in o3:
        t_ = T.get(r['tag'])
        if not t_ or 'lam' not in t_:
            continue
        rd = r['rd']
        Rrd = np.array([sum(rd[e][l]['sc_loss'] for l in LAYERS) if rd and e in rd else np.nan for e in range(EPOCHS)])
        loss = np.array([x['loss'] for x in r['rows']])
        Lrec = loss - t_['lam'] * Rrd
        rel = np.abs(Lrec[1:] / t_['L'][1:] - 1)
        errs[r['tag']] = {'max': float(np.nanmax(rel)), 'median': float(np.nanmedian(rel))}
    out['L_recon_check_ours_pt'] = errs
    new3 = [S[r['tag']]['rho_eff_post_over_pre'] for r in nl if r['seed'] != 1]
    out['nolr_ratio_new3'] = {'n': len(new3), 'min': min(new3), 'max': max(new3), 'mean_sd': msd(new3)}
    # (9) 실행 시기: ours 안의 10월 대 9월 (시작 시각 기준)
    st = res['timing']['per_run']
    import datetime as dt
    oct1 = dt.datetime(2026, 10, 1).timestamp()
    month = {t: ('oct' if v['start'] >= oct1 else 'sep') for t, v in st.items() if v.get('start')}
    out['start_month'] = {k: {m_: sum(1 for r in G['all_used'] if r['key'] == k and month.get(r['tag']) == m_)
                              for m_ in ('sep', 'oct')} for k in KEYS}
    oa = [r for r in G['ours_all'] if r['tag'] in month]
    c0 = math.log(3e-3)
    Xr = np.array([[1.0, math.log(float(r['rho'])) - c0, (math.log(float(r['rho'])) - c0) ** 2,
                    1.0 if r['server'] == '23' else 0.0, 1.0 if month[r['tag']] == 'oct' else 0.0] for r in oa])
    fr = ols(Xr, [math.log(r['spikes']) for r in oa])
    e_ = est(fr['beta'][4], fr['se'][4], fr['df'])
    e_.update(pct=100 * (math.exp(e_['d']) - 1), pct_lo=100 * (math.exp(e_['lo']) - 1), pct_hi=100 * (math.exp(e_['hi']) - 1))
    oc = [r for r in ours if r['tag'] in month]
    Xa = np.array([[1.0, lnspk_c(r), 1.0 if r['server'] == '23' else 0.0, 1.0 if month[r['tag']] == 'oct' else 0.0] for r in oc])
    fa = ols(Xa, [r['val_acc'] for r in oc])
    out['oct_vs_sep_ours'] = {'lnspk_at_rho': e_, 'n_oct_rho': int(Xr[:, 4].sum()), 'n_sep_rho': int(len(oa) - Xr[:, 4].sum()),
                              'acc_at_spk': est(fa['beta'][3], fa['se'][3], fa['df']),
                              'n_oct_acc': int(Xa[:, 3].sum()), 'n_sep_acc': int(len(oc) - Xa[:, 3].sum())}
    # (10) npz: ours 3e-3 기준 4런의 추출기 판 (파일 경로·스텝별 배열 유무)
    nz = res['npz']
    out['npz_ours_pt'] = {r['tag']: {'path': nz['have'].get(r['tag']),
                                     'has_step': 't_share' in nz['per_run'].get(r['tag'], {}),
                                     'dead_ch_conv': nz['per_run'].get(r['tag'], {}).get('dead_ch_conv'),
                                     'silent_pct': nz['per_run'].get(r['tag'], {}).get('silent_pct'),
                                     'npz_over_scount': (nz['per_run'][r['tag']]['npz_spk'] / r['spikes'])
                                     if r['tag'] in nz['per_run'] else None}
                          for r in o3}
    # (11) 1- inversion 정확도 CI 모음 (판정 세트 n=4 와 전체 n=5)
    acc = res['acc']['ch_noinv']
    out['noinv_acc_cis'] = {k: {x: acc[k][x] for x in ('d', 'lo', 'hi', 'p')} for k in
                            ('lin_srv_all', 'lin_srv_pt', 'direct_welch', 'direct_adj')}
    return out


# ----------------------------------------------------------------------------- 8. 21시 판 검토 반영 (eip-critic 2차)
def _fit_coef(rows, yf, cols, k):
    """rows 에 대해 y = X·β 를 적합하고 열 k 의 추정을 낸다. cols(r) = 설명변수 목록 (상수항 포함)."""
    X = np.array([cols(r) for r in rows], float)
    f = ols(X, [yf(r) for r in rows])
    return est(f['beta'][k], f['se'][k], f['df']), f


def review21_section(G, res, met, M):
    """21시 판에 대한 검토 지적을 직접 다시 계산한다 (검토자 수치를 그대로 쓰지 않음).
    (a) ours 의 ρ 반응을 단계별로 — ρ 1.5e-3 (n=3) → 4e-3 (n=3) 에서 줄어든 npz 스파이크의 단계별 몫,
        평가 모드 ρ 2차 모형 (ours npz + `- vmem` npz) 의 ρ 2e-3 국소 기울기, '순수한 ρ 이동' 으로 본 단계별 예측 vs 관측.
    (b) 같은 npz 스파이크에서 단계 배분: ln 단계 스파이크 = a + b·ln npz 총합 + e·[23] + Δ·[끈 팔] (ours npz 곡선 n=19 대
        끈 팔 npz 전체), 4단계 Holm. 모형 없는 확인: ours 2e-3·3e-3 묶음 평균을 ln 공간에서 잇고 끈 팔 342K 세트 평균에서 읽음.
    (c) 실행 시기 (시작 시각 9월/10월): ours npz 곡선 안에서 10월 − 9월 (선형·2차 + 서버), 팔 ANCOVA 에 10월 항을 넣은 값.
    (d) 1− inversion 주 지표 12개의 기준 통과 수 (두 정의).
    (e) 스텝 지표 32개 (4팔 × 스텝 몫 4 + 처음 쏜 스텝 4) 의 Bonferroni·Holm.
    (f) 스텝 4 몫이 npz 스파이크 수 차이만으로 움직이는 크기 (ours 스텝 배열 런의 기울기).
    (g) 추출기 판 대조를 같은 체크포인트 모든 쌍으로 확대.
    (h) 학습 모드 `- vmem` 런 중 npz 가 없는 런 (두 측정의 겹침)."""
    out = {}
    stg = STAGES[:-1]
    nz = res['npz']
    ours_c = [r for r in G['ours_curve'] if r['tag'] in met]
    o3 = [r for r in G['ours_pt'] if r['tag'] in met]
    x0 = math.log(nz['x0_npz_spk'])
    xf = lambda r: math.log(met[r['tag']]['npz_spk']) - x0
    srv = lambda r: 1.0 if r['server'] == '23' else 0.0
    # ---- (a) ours 의 ρ 반응, 단계별
    oa = {}
    for k in ('1.5e-3', '2e-3', '3e-3', '4e-3'):
        oa[k] = [r for r in G['ours_all'] if r['knob'] == k and r['tag'] in met]
    lo_, hi_ = oa['1.5e-3'], oa['4e-3']
    tot = np.mean([met[r['tag']]['npz_spk'] for r in lo_]) - np.mean([met[r['tag']]['npz_spk'] for r in hi_])
    a = {'n_lo': len(lo_), 'n_hi': len(hi_), 'rho_lo': '1.5e-3', 'rho_hi': '4e-3', 'total_drop_per_img': float(tot),
         'share_of_drop_pct': {}}
    for s in stg + ['fc1']:
        dd = np.mean([met[r['tag']][f'spk_{s}'] for r in lo_]) - np.mean([met[r['tag']][f'spk_{s}'] for r in hi_])
        a['share_of_drop_pct'][s] = float(100 * dd / tot)
    on = [r for r in G['ours_all'] if r['tag'] in met]
    vn = [r for r in G['ch_novmem'] if r['tag'] in met]
    c0 = math.log(2e-3)
    u = lambda r: math.log(float(r['rho'])) - c0
    cols_j = lambda g: (lambda r: [1.0, u(r), u(r) ** 2, srv(r), float(g(r))])
    isarm = lambda r: r['key'] == 'ch_novmem'
    slope, eff, slope_o = {}, {}, {}
    for k in ['npz_spk'] + [f'spk_{s}' for s in stg]:
        yf = lambda r, k=k: math.log(met[r['tag']][k])
        e1, f = _fit_coef(on + vn, yf, cols_j(isarm), 1)
        slope[k] = e1
        eff[k] = est(f['beta'][4], f['se'][4], f['df'])
        eo, _ = _fit_coef(on, yf, lambda r: [1.0, u(r), u(r) ** 2, srv(r)], 1)
        slope_o[k] = eo
    d_tot = eff['npz_spk']['d']
    dl = d_tot / slope['npz_spk']['d']      # 총합 차 (`- vmem` − ours) 를 ours 곡선 위의 ρ 이동으로 옮긴 값 (ln ρ, 양수 = ρ 를 올린 것과 같음)
    a['eval_rho_model'] = {'n_ours': len(on), 'n_arm': len(vn), 'center': '2e-3',
                           'local_slope_joint': {k: v for k, v in slope.items()},
                           'local_slope_ours_only': {k: v for k, v in slope_o.items()},
                           'delta_ln_rho_equiv': float(dl),
                           'pred_pct_pure_rho_shift': {k: float(100 * (math.exp(slope[k]['d'] * dl) - 1)) for k in slope},
                           'obs_pct': {k: float(100 * (math.exp(eff[k]['d']) - 1)) for k in eff}}
    # 같은 계산을 학습 모드 (reg_detail 마지막 에폭, ours 전체 + `- vmem` 전체) 로
    om = [r for r in G['ours_all'] if M.get(r['tag'])]
    vm = [r for r in G['ch_novmem'] if M.get(r['tag'])]
    stage_sp = lambda m, s: sum(v for l, v in m['spk_layer'].items() if l.startswith(s))
    tsl, tef = {}, {}
    for k in ['tr_spk'] + [f'spk_{s}' for s in stg]:
        yf = (lambda r: math.log(M[r['tag']]['tr_spk'])) if k == 'tr_spk' else \
             (lambda r, s=k[4:]: math.log(stage_sp(M[r['tag']], s)))
        e1, f = _fit_coef(om + vm, yf, cols_j(isarm), 1)
        tsl[k], tef[k] = e1, est(f['beta'][4], f['se'][4], f['df'])
    dlt = tef['tr_spk']['d'] / tsl['tr_spk']['d']
    a['train_rho_model'] = {'n_ours': len(om), 'n_arm': len(vm), 'local_slope_joint': tsl,
                            'delta_ln_rho_equiv': float(dlt),
                            'pred_pct_pure_rho_shift': {k: float(100 * (math.exp(tsl[k]['d'] * dlt) - 1)) for k in tsl},
                            'obs_pct': {k: float(100 * (math.exp(tef[k]['d']) - 1)) for k in tef}}
    out['a_ours_rho_by_stage'] = a
    # ---- (b) 같은 npz 스파이크에서 단계 배분 (ln 단계 스파이크 ANCOVA), 네 팔
    b = {}
    for arm in ARMS:
        aa = [r for r in G[arm] if r['tag'] in met]
        ap = [r for r in aa if r['knob'] == POINT[arm]]
        row = {'n_ref': len(ours_c), 'n_arm': len(aa)}
        ps = []
        for s in stg:
            yf = lambda r, s=s: math.log(met[r['tag']][f'spk_{s}'])
            e_ = ancova(ours_c, aa, yf, xf)
            e_.update(pct=100 * (math.exp(e_['d']) - 1), pct_lo=100 * (math.exp(e_['lo']) - 1),
                      pct_hi=100 * (math.exp(e_['hi']) - 1))
            row[s] = e_
            ps.append(e_['p'])
        for s, h in zip(stg, holm(ps)):
            row[s]['holm4'] = h
        # 모형 없는 확인: ours 묶음 평균 (ln npz, ln 단계) 꺾은선에서 끈 팔 342K 세트 평균 ln npz 위치를 읽음
        gx = {k: np.mean([math.log(met[r['tag']]['npz_spk']) for r in v]) for k, v in oa.items() if len(v) >= 2}
        xa = float(np.mean([math.log(met[r['tag']]['npz_spk']) for r in ap]))
        kk = sorted(gx, key=lambda k: gx[k])
        br = [(kk[i], kk[i + 1]) for i in range(len(kk) - 1) if gx[kk[i]] <= xa <= gx[kk[i + 1]]]
        if br:
            k1, k2 = br[0]
            wt = (xa - gx[k1]) / (gx[k2] - gx[k1])
            row['interp'] = {'between': [k1, k2], 'npz_spk_arm_pt': float(math.exp(xa)), 'n_arm_pt': len(ap),
                             'n_ours': [len(oa[k1]), len(oa[k2])]}
            for s in stg:
                g1 = np.mean([math.log(met[r['tag']][f'spk_{s}']) for r in oa[k1]])
                g2 = np.mean([math.log(met[r['tag']][f'spk_{s}']) for r in oa[k2]])
                ya = np.mean([math.log(met[r['tag']][f'spk_{s}']) for r in ap])
                row['interp'][s] = float(100 * (math.exp(ya - (g1 + wt * (g2 - g1))) - 1))
        b[arm] = row
    out['b_stage_alloc_same_spk'] = b
    # ---- (c) 실행 시기
    st = res['timing']['per_run']
    import datetime as dt
    oct1 = dt.datetime(2026, 10, 1).timestamp()
    month = {t: ('oct' if v['start'] >= oct1 else 'sep') for t, v in st.items() if v.get('start')}
    isoct = lambda r: 1.0 if month.get(r['tag']) == 'oct' else 0.0
    c = {'ours_curve': [(r['tag'], r['server'], month.get(r['tag'])) for r in ours_c],
         'n_oct': int(sum(isoct(r) for r in ours_c)), 'n_sep': int(sum(1 - isoct(r) for r in ours_c)),
         'arms_month': {arm: sorted({month.get(r['tag']) for r in G[arm] if r['tag'] in met}) for arm in ARMS},
         'ours_pt_month': {r['tag']: month.get(r['tag']) for r in o3},
         'start_iso': {r['tag']: dt.datetime.fromtimestamp(st[r['tag']]['start']).strftime('%m-%d')
                       for r in ours_c + [x for arm in ARMS for x in G[arm] if x['tag'] in met] if st[r['tag']].get('start')}}
    keys_c = NPZ_PRIMARY
    c['within_ours'] = {}
    for k in keys_c:
        yf = lambda r, k=k: met[r['tag']][k]
        el, _ = _fit_coef(ours_c, yf, lambda r: [1.0, xf(r), srv(r), isoct(r)], 3)
        eq, _ = _fit_coef(ours_c, yf, lambda r: [1.0, xf(r), xf(r) ** 2, srv(r), isoct(r)], 4)
        c['within_ours'][k] = {'lin': el, 'quad': eq}
    c['arm_with_month'] = {}
    for arm in ARMS:
        aa = [r for r in G[arm] if r['tag'] in met]
        isa = lambda r: 1.0 if r['key'] == arm else 0.0
        row = {}
        for k in keys_c:
            yf = lambda r, k=k: met[r['tag']][k]
            e0 = nz[arm][k]['ancova']
            el, _ = _fit_coef(ours_c + aa, yf, lambda r: [1.0, xf(r), srv(r), isoct(r), isa(r)], 4)
            eq, _ = _fit_coef(ours_c + aa, yf, lambda r: [1.0, xf(r), xf(r) ** 2, srv(r), isoct(r), isa(r)], 5)
            row[k] = {'no_month': {x: e0[x] for x in ('d', 'lo', 'hi', 'p')}, 'lin_month': el, 'quad_month': eq}
        for mdl in ('lin_month', 'quad_month'):                # 팔 안 Holm (주 지표 12개), 시기 항을 넣은 모형
            for k, h in zip(keys_c, holm([row[k][mdl]['p'] for k in keys_c])):
                row[k][mdl]['holm'] = h
        c['arm_with_month'][arm] = row
    out['c_start_month'] = c
    # ---- (d) 1− inversion 주 지표 기준 통과 수
    ni = nz['ch_noinv']
    crit1 = [k for k in NPZ_PRIMARY if ni[k]['ancova']['holm'] < 0.05 and (ni[k]['adj']['lo'] > 0 or ni[k]['adj']['hi'] < 0)]
    crit2 = [k for k in crit1 if ni[k]['adj']['holm'] < 0.05]
    out['d_noinv_pass'] = {'n_primary': len(NPZ_PRIMARY), 'pass_ancova_holm_and_adj_ci': crit1,
                           'pass_both_holm': crit2,
                           'adj_holm': {k: ni[k]['adj']['holm'] for k in NPZ_PRIMARY},
                           'ancova_holm': {k: ni[k]['ancova']['holm'] for k in NPZ_PRIMARY}}
    # ---- (e) 스텝 지표 32개 다중 비교
    tests = []
    for arm in ARMS:
        r_ = nz[arm]
        if 't_share' not in r_:
            continue
        for j in range(4):
            tests.append((arm, f'step{j + 1}_share', r_['t_share']['steps'][j]['direct']))
            tests.append((arm, f'first_t{j + 1}', r_['first_t'][j]['direct']))
    hb = holm([t[2]['p'] for t in tests])
    out['e_step_multi'] = {'n_tests': len(tests),
                           'rows': [{'arm': t[0], 'metric': t[1], 'd': t[2]['d'], 'p': t[2]['p'],
                                     'bonf': min(1.0, t[2]['p'] * len(tests)), 'holm': h} for t, h in zip(tests, hb)]}
    # ---- (f) 스텝 4 몫과 npz 스파이크 (ours 스텝 배열 런)
    ot = [r for r in G['ours_all'] if r['tag'] in met and 't_share' in met[r['tag']]]
    ef, _ = _fit_coef(ot, lambda r: met[r['tag']]['t_share'][3], lambda r: [1.0, xf(r)], 1)
    dln = math.log(1 + nz['ch_noinv']['npz_spk']['ln_pct']['pct'] / 100)     # 342K 세트 직접 ln 차 (n=4 대 4)
    out['f_step4_vs_spk'] = {'ours_runs': [(r['tag'], round(met[r['tag']]['t_share'][3], 2)) for r in ot],
                             'slope_per_ln_spk': ef, 'noinv_ln_npz_diff': dln,
                             'pred_shift_pp': float(ef['d'] * dln),
                             'noinv_step4_runs': [round(met[r['tag']]['t_share'][3], 2) for r in G['ch_noinv']
                                                  if r['knob'] == POINT['ch_noinv'] and r['tag'] in met]}
    # ---- (g) 추출기 판 대조 (같은 체크포인트, 모든 쌍)
    g = {}
    pairs = [(f'r19c10-ours_ch-3e-3-s{s}', os.path.join(ROOT, '_npz'), os.path.join(A, 'npz_abl')) for s in (2, 3)] + \
            [(f'r19c10-ours_ch-1p5e-3-s{s}', os.path.join(ROOT, '_npz'), os.path.join(ROOT, '_npz_t')) for s in (1, 2, 3)] + \
            [(f'r19c10-base---s{s}', os.path.join(ROOT, '_npz'), os.path.join(ROOT, '_npz_t')) for s in (11, 12, 13)]
    for t, do, dn in pairs:
        po, pn = os.path.join(do, t + '.npz'), os.path.join(dn, t + '.npz')
        if not (os.path.exists(po) and os.path.exists(pn)):
            g[t] = None
            continue
        zo, zn = np.load(po), np.load(pn)
        g[t] = {'max_abs_diff_counts': max(float(np.abs(zo[l].astype(np.float64) - zn[l].astype(np.float64)).max())
                                           for l in LAYERS),
                'max_abs_diff_fired': max(float(np.abs(zo[l + '__fired'].astype(np.float64) -
                                                       zn[l + '__fired'].astype(np.float64)).max()) for l in LAYERS),
                'acc': [float(zo['acc']), float(zn['acc'])], 'old': po, 'new': pn}
    out['g_extractor_pairs'] = g
    # ---- (h) 학습 모드 `- vmem` 런과 npz 런의 겹침
    tv = [r['tag'] for r in G['ch_novmem'] if M.get(r['tag'])]
    out['h_vmem_overlap'] = {'train_mode': tv, 'npz': [t for t in tv if t in met], 'train_only': [t for t in tv if t not in met],
                             'ours_train_mode_n': len([r for r in G['ours_curve'] if M.get(r['tag'])]),
                             'ours_npz_n': len(ours_c)}
    # ---- (i) 1− inversion 죽은 채널: 4,480 중 비율, vmem 보정 직접 CI 반폭 대비
    dn_ = nz['ch_noinv']['dead_ch_conv']
    dv_ = nz['ch_novmem']['dead_ch_conv']
    out['i_dead'] = {'n_conv_ch': int(sum(SHAPE[l][-1] for l in CONV)), 'noinv_ancova': dn_['ancova']['d'],
                     'noinv_pct_of_ch': 100 * dn_['ancova']['d'] / sum(SHAPE[l][-1] for l in CONV),
                     'ours_pt_dead_pct': 100 * nz['ours_pt']['dead_ch_conv'][0] / sum(SHAPE[l][-1] for l in CONV),
                     'ours_pt_silent_pct': nz['ours_pt']['silent_pct'][0],
                     'vmem_adj_halfwidth': (dv_['adj']['hi'] - dv_['adj']['lo']) / 2,
                     'vmem_adj_halfwidth_over_noinv_adj': ((dv_['adj']['hi'] - dv_['adj']['lo']) / 2) / abs(dn_['adj']['d'])}
    return out


# ----------------------------------------------------------------------------- 묶음
def groups(runs):
    use = [r for r in runs if r['use']]
    G = {'all_used': use}
    G['base'] = [r for r in use if r['key'] == 'base']
    G['ours_all'] = [r for r in use if r['key'] == 'ours_ch']
    G['ours_curve'] = [r for r in G['ours_all'] if SPIKE_RANGE[0] <= r['spikes'] <= SPIKE_RANGE[1]]
    G['ours_pt'] = [r for r in G['ours_all'] if r['knob'] == POINT['ours_ch']]
    for a in ARMS:
        G[a] = [r for r in use if r['key'] == a]
    return G


def pooled_sd(groups_):
    ss, df = 0.0, 0
    for g in groups_:
        v = np.array(g, float)
        if len(v) >= 2:
            ss += ((v - v.mean()) ** 2).sum()
            df += len(v) - 1
    return math.sqrt(ss / df), df


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return [jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refetch', action='store_true')
    args = ap.parse_args()
    runs, dups, mism = build_runs(args.refetch)
    if mism:
        print('스냅샷 불일치 — 중단', mism)
        sys.exit(1)
    G = groups(runs)
    res = {'generated': __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M'),
           'snapshot': json.load(open(SNAP))['generated'], 'dups': dups,
           'n': {k: len(G[k]) for k in ['base', 'ours_all', 'ours_curve', 'ours_pt'] + ARMS}}
    # 합동 시드 SD
    byk = {}
    for r in G['ours_curve']:
        byk.setdefault(r['knob'], []).append(r['val_acc'])
    sd_o, df_o = pooled_sd(byk.values())
    arm_g = {}
    for a in ARMS:
        for r in G[a]:
            arm_g.setdefault((a, r['knob']), []).append(r['val_acc'])
    sd_all, df_all = pooled_sd(list(byk.values()) + list(arm_g.values()))
    res['seed_sd'] = {'ours': [sd_o, df_o], 'ours_plus_arms': [sd_all, df_all],
                      'groups_arm': {f'{a}|{k}': msd(v) for (a, k), v in arm_g.items()}}
    res['acc'] = acc_section(G, sd_o, df_o)
    for a in ARMS:
        res['acc'][a]['verdict'] = verdict(res['acc'][a])
    # ours 3e-3 대 baseline
    res['ours_pt_vs_base'] = welch([r['val_acc'] for r in G['ours_pt']], [r['val_acc'] for r in G['base']])
    res['rho'] = rho_section(G)
    tr, T = traj_section(G)
    res['traj'] = tr
    res['mech'], M = mech_section(G)
    res['mech_traj'] = mech_traj_section(G)
    res['npz'], NM = npz_section(G)
    res['npz_vmem_rho'] = npz_vmem_rho(G, NM, M)
    res['npz_extractor_check'] = npz_extractor_check()
    res['npz_acc_check'] = npz_acc_check(G, NM)
    res['timing'] = timing_section(G)
    res['runs'] = [{k: r.get(k) for k in ('tag', 'key', 'knob', 'seed', 'server', 'epochs', 'best_epoch', 'train_loss',
                                          'train_acc', 'val_loss', 'val_acc', 'spikes', 'sec_med', 'use', 'snap_reason',
                                          'rho', 'lam_fixed', 'ckpt', 'ckpt_epochs', 'ckpt_match', 'store_dir',
                                          'has_rd')} for r in runs]
    U = G['all_used']
    res['ckpt_check'] = {'n_used': len(U), 'match': sum(1 for r in U if r['ckpt_match'] is True),
                         'mismatch': [r['tag'] for r in U if r['ckpt_match'] is False],
                         'no_ckpt': [{'tag': r['tag'], 'store_dir': r['store_dir'], 'has_rd': r['has_rd']}
                                     for r in U if r['ckpt_match'] is None]}
    res['review'] = review_section(G, res, T, M)
    res['review21'] = review21_section(G, res, NM, M)
    json.dump(jsonable(res), open(OUT_JSON, 'w'), ensure_ascii=False, indent=1)
    write_tables(runs, G, res)
    print('wrote', OUT_JSON, OUT_TAB)
    return res


# ----------------------------------------------------------------------------- 표 조각
def fnum(x, d=2):
    return '–' if x is None or (isinstance(x, float) and not math.isfinite(x)) else f'{x:.{d}f}'


def write_tables(runs, G, res):
    L = []
    w = L.append
    base_mean = np.mean([r['spikes'] for r in G['base']])
    secd = {t: v['sec_med'] for t, v in res['timing']['per_run'].items()}
    w('| 팔 | ρ / λ | seed | 서버 | train loss | train acc | val loss | val acc | spikes | spikes vs base | best ep | s/에폭 | 비고 |')
    w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    def rowline(r, note=''):
        return (f"| {NAME[r['key']]} | {r['knob']} | s{r['seed']} | {r['server']} | {fnum(r.get('train_loss'), 4)} | "
                f"{fnum(r.get('train_acc'))} | {fnum(r.get('val_loss'), 4)} | {fnum(r.get('val_acc'))} | "
                f"{r['spikes']:,.0f} | {100 * (r['spikes'] / base_mean - 1):+.1f}% | {r.get('best_epoch')} | "
                f"{fnum(secd.get(r['tag']), 0)} | {note} |") if r.get('val_acc') is not None else \
               f"| {NAME[r['key']]} | {r['knob']} | s{r['seed']} | {r['server']} | – | – | – | – | – | – | – | – | {note} |"
    def meanline(rs, label):
        from decimal import Decimal, ROUND_HALF_UP
        f = lambda k: float(Decimal(repr(round(float(np.mean([r[k] for r in rs])), 6))).quantize(
            Decimal('0.0001'), rounding=ROUND_HALF_UP))
        fa = lambda: Decimal(repr(round(float(np.mean([r['val_acc'] for r in rs])), 6))).quantize(
            Decimal('0.01'), rounding=ROUND_HALF_UP)
        sa = np.std([r['val_acc'] for r in rs], ddof=1) if len(rs) > 1 else float('nan')
        ss = np.std([r['spikes'] for r in rs], ddof=1) if len(rs) > 1 else float('nan')
        srv = '/'.join(sorted({r['server'] for r in rs}))
        return (f"| **{label} 평균** | {rs[0]['knob']} | n={len(rs)} | {srv} | {f('train_loss'):.4f} | {f('train_acc'):.2f} | "
                f"{f('val_loss'):.4f} | **{fa()} ±{sa:.2f}** | {f('spikes'):,.0f} ±{ss:,.0f} | "
                f"{100 * (f('spikes') / base_mean - 1):+.1f}% | | | |")
    for key in ['base', 'ours_ch'] + ARMS:
        rs = [r for r in runs if r['key'] == key]
        knobs = sorted({r['knob'] for r in rs}, key=lambda k: 0 if k == '-' else float(k))
        for k in knobs:
            grp = sorted([r for r in rs if r['knob'] == k], key=lambda r: r['seed'])
            used = [r for r in grp if r['use']]
            for r in grp:
                if r['use']:
                    note = ''
                    if key == 'ours_ch' and not (SPIKE_RANGE[0] <= r['spikes'] <= SPIKE_RANGE[1]):
                        note = '200~420K 밖 — 정확도 곡선에서 제외 (ρ 눈금에만 사용)'
                    w(rowline(r, note))
                elif r['seed'] < 900 and r.get('val_acc') is not None:
                    w(rowline(r, '배제: ' + r['snap_reason']))
                elif r['seed'] < 900:
                    w(rowline(r, '배제: ' + r['snap_reason']))
                else:
                    w(rowline(r, '배제: 2에폭 연기시험'))
            if len(used) >= 2:
                w(meanline(used, NAME[key]))
    open(OUT_TAB, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
