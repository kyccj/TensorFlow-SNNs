"""논문용 실험 결과 집계 (26-09-03).

기존 `collect_results.py` 와 다르다 — 저쪽은 **마지막 에폭**을 카테고리별로 훑어 보여주고,
여기는 **최고 val_acc 시점**과 선별 판정(S30/S1, 완주 여부)을 CSV 로 낸다. 둘 다 쓴다.

`train.log` 를 매번 임시 파이썬으로 파싱하다 실수가 반복돼서 하나로 고정한다.
논문 표와 선별 판정에 필요한 것만 뽑아 CSV 한 장으로 만든다.

    python collect_results.py _p0/'*' _shoulder/'*' --out _results/p0.csv
    python collect_results.py '_vmlow/*' --print

인자는 run 디렉토리(= `config_sweep.py` 와 `train.log` 가 있는 곳) 또는 그 glob.

**뽑는 것**
  설정   model / dataset / T / 방법 플래그 / lambda / rho / seed
  성능   최고 val_acc 와 그 시점의 train loss·acc, val loss, s_count
  진행   완주 에폭 수, 완주 여부(>=300)
  선별   S30/S1 — 첫 30에폭에 학습 모드 스파이크가 1에폭의 몇 배로 줄었는가.
         0.19 미만이면 정확도 손상이 예측된다 (189런 회고, LOO 정밀도 0.94, flags.py:860-868).
         proc.py:1254 의 브레이크와 같은 정의 — `reg_detail.csv` 의 층별 spike_count 중
         양수만 합친다 (n_in / predictions 는 음수 마커라 제외).

**주의** — `s_count`(train.log)는 eval/test 프로토콜 스파이크 수이고,
`reg_detail.csv` 의 spike_count 는 학습 모드다. 둘은 값이 다르며 섞으면 안 된다.
S30/S1 은 학습 모드, 논문 표의 스파이크 수는 eval 쪽을 쓴다.
"""
import argparse, csv, glob, os, re, sys

EPOCH_LINE = re.compile(r'([A-Za-z_][\w\-]*): ([\d.]+(?:[eE][+-]?\d+)?)')

# config_sweep.py 에서 읽을 설정 — (열 이름, 정규식). 주석 줄은 제외한다.
CONF_KEYS = [
    ('model',        r"^conf\.model\s*=\s*'([^']+)'"),
    ('dataset',      r"^conf\.dataset\s*=\s*'([^']+)'"),
    ('time_step',    r"^\s*conf\.time_step\s*=\s*(\S+)"),
    ('reg_on',       r"^\s*conf\.reg_spike_out\s*=\s*(True|False)"),
    ('lambda',       r"^\s*conf\.reg_spike_out_const\s*=\s*(\S+)"),
    ('wta_rev',      r"^\s*conf\.reg_spike_out_wta_rev\s*=\s*(True)"),
    ('sc_sm',        r"^\s*conf\.reg_spike_out_sc_sm\s*=\s*(True)"),
    ('maxnorm',      r"^\s*conf\.reg_spike_out_sc_maxnorm\s*=\s*(True)"),
    ('maxnorm_plain', r"^\s*conf\.reg_spike_out_sc_maxnorm_plain\s*=\s*(True)"),
    ('sm_plain',     r"^\s*conf\.reg_spike_out_sm_plain\s*=\s*(True)"),
    ('sc_one',       r"^\s*conf\.reg_spike_out_sc_one\s*=\s*(True)"),
    ('mx_group',     r"^\s*conf\.reg_spike_maxnorm_group\s*=\s*'([^']+)'"),
    ('vmem_gain',    r"^\s*conf\.reg_spike_vmem_gain\s*=\s*(\S+)"),
    ('silent_only',  r"^\s*conf\.reg_spike_vmem_silent_only\s*=\s*(True)"),
    ('final_step',   r"^\s*conf\.reg_spike_final_step\s*=\s*(True)"),
    ('accum',        r"^\s*conf\.reg_spike_accum_loss\s*=\s*(True)"),
    ('loss_ratio',   r"^\s*conf\.reg_spike_loss_ratio\s*=\s*(True)"),
    ('rho',          r"^\s*conf\.reg_spike_loss_ratio_target\s*=\s*(\S+)"),
    ('R_per_step',   r"^\s*conf\.reg_spike_R_per_step\s*=\s*(True|False)"),
    ('seed',         r"^\s*conf\.run_seed\s*=\s*(\S+)"),
    ('lr_brake',     r"^\s*conf\.reg_spike_lr_brake\s*=\s*(True)"),
]


def read_conf(run_dir):
    p = os.path.join(run_dir, 'config_sweep.py')
    out = {k: '' for k, _ in CONF_KEYS}
    if not os.path.exists(p):
        return out
    txt = open(p, errors='ignore').read()
    # `else:   # previous work` 아래는 실행되지 않는 죽은 가지다. 거기에도
    # conf.reg_spike_out_const 가 있어서 안 자르면 lambda 를 5E-9 로 잘못 읽는다.
    dead = re.search(r'^\s*else:\s*#\s*previous work', txt, re.M)
    if dead:
        txt = txt[:dead.start()]
    for key, pat in CONF_KEYS:
        ms = re.findall(pat, txt, re.M)
        if ms:
            out[key] = ms[-1]          # 마지막 대입이 유효값
    # 활성 exp_set_name (root_model_save 대입 앞의 마지막 것)
    i = txt.find('conf.root_model_save=conf.exp_set_name')
    ms = re.findall(r"^conf\.exp_set_name\s*=\s*'([^']*)'", txt[:i] if i > 0 else txt, re.M)
    out['save_name'] = ms[-1] if ms else ''
    return out


def planned_epochs(run_dir, default=310):
    """train.log 의 'Epoch n/N' 에서 N. 없으면 default.

    completed 를 len(rows) >= 300 으로 재던 것이 버그였다 — 302 에폭에서 멈춘 런이
    완주로 잡혀 사전 등록한 '310 완주 실패 -> 배제' 규칙을 그냥 통과했다."""
    fp = os.path.join(run_dir, 'train.log')
    if not os.path.exists(fp):
        return default
    m = re.search(r'Epoch\s+\d+/(\d+)', open(fp, errors='ignore').read())
    return int(m.group(1)) if m else default


def read_epochs(run_dir):
    """train.log 의 에폭 줄들을 dict 목록으로."""
    p = os.path.join(run_dir, 'train.log')
    if not os.path.exists(p):
        return []
    txt = open(p, errors='ignore').read().replace('\r', '\n')
    rows = []
    for line in txt.split('\n'):
        if 'val_loss' not in line:
            continue
        d = {}
        for k, v in EPOCH_LINE.findall(line):
            try:
                d[k] = float(v)
            except ValueError:
                pass
        if 'val_acc' in d:
            rows.append(d)
    return rows


def spike_ratio(run_dir, save_name, at_epoch=30):
    """S(at_epoch)/S(first) — 학습 모드 스파이크. proc.py:1254 와 같은 정의.

    26-09-14 에 실험 산출물을 /media/hdd1/kyccj/EIP 로 옮겼다. 예전에는 reg_detail.csv 가
    저장소 루트의 <save_name>/ 에 있었으나 지금은 STORE/<save_name>/ 이다. 둘 다 본다 —
    안 그러면 S30/S1 이 비고 사전 등록한 배제 규칙이 통째로 무력해진다.
    """
    STORE = '/media/hdd1/kyccj/EIP/paper'
    ARCH = '/media/hdd1/kyccj/EIP/archive'
    cands = [save_name, run_dir]
    if save_name:
        cands += [os.path.join(STORE, save_name), os.path.join(ARCH, save_name)]
    for base in cands:
        if not base:
            continue
        p = os.path.join(base, 'reg_detail.csv')
        if not os.path.exists(p):
            continue
        tot = {}
        for r in csv.DictReader(open(p)):
            try:
                e = int(r['epoch']); v = float(r['spike_count'])
            except (KeyError, ValueError):
                continue
            if v > 0:                                  # n_in / predictions 는 음수 마커
                tot[e] = tot.get(e, 0.0) + v
        if not tot:
            continue
        eps = sorted(tot)
        # 브레이크와 같게, 합이 양수인 첫 에폭을 기준(S1)으로 잡는다
        first = next((e for e in eps if tot[e] > 0), None)
        cand = [e for e in eps if e <= at_epoch and e >= (first if first is not None else 0)]
        if first is None or not cand:
            continue
        near = cand[-1]
        return tot[first], tot[near], near, tot[near] / tot[first]
    return None, None, None, None


def collect(run_dir):
    conf = read_conf(run_dir)
    rows = read_epochs(run_dir)
    rec = {'run_dir': run_dir}
    rec.update(conf)
    rec['epochs'] = len(rows)
    rec['completed'] = int(len(rows) >= planned_epochs(run_dir))
    if rows:
        b = max(rows, key=lambda x: x['val_acc'])
        bi = rows.index(b)
        rec['best_epoch'] = bi + 1
        rec['best_val_acc'] = round(b['val_acc'] * 100, 4)
        rec['s_count'] = round(b.get('s_count', float('nan')), 1)
        rec['val_loss'] = round(b.get('val_loss', float('nan')), 4)
        rec['train_loss'] = round(b.get('loss', float('nan')), 4)
        rec['train_acc'] = round(b.get('acc', 0.0) * 100, 4)
        last = rows[-1]
        rec['final_train_loss'] = round(last.get('loss', float('nan')), 4)
        rec['final_train_acc'] = round(last.get('acc', 0.0) * 100, 4)
    else:
        for k in ('best_epoch', 'best_val_acc', 's_count', 'val_loss', 'train_loss',
                  'train_acc', 'final_train_loss', 'final_train_acc'):
            rec[k] = ''
    s1, s30, at, ratio = spike_ratio(run_dir, conf.get('save_name', ''))
    rec['S1'] = round(s1, 1) if s1 else ''
    rec['S30'] = round(s30, 1) if s30 else ''
    rec['S30_at_epoch'] = at if at is not None else ''
    rec['S30_over_S1'] = round(ratio, 4) if ratio else ''
    # 선별 규칙 (사전 선언): 미완주 또는 S30/S1 < 0.19 이면 배제 후보
    reasons = []
    if not rec['completed']:
        reasons.append(f"미완주({rec['epochs']}ep)")
    if ratio is not None and ratio < 0.19:
        reasons.append(f'S30/S1={ratio:.3f}<0.19')
    rec['exclude'] = int(bool(reasons))
    rec['exclude_reason'] = '; '.join(reasons)
    return rec


COLS = (['run_dir', 'save_name', 'model', 'dataset', 'time_step', 'epochs', 'completed',
         'best_epoch', 'best_val_acc', 's_count', 'val_loss', 'train_loss', 'train_acc',
         'final_train_loss', 'final_train_acc',
         'S1', 'S30', 'S30_at_epoch', 'S30_over_S1', 'exclude', 'exclude_reason']
        + [k for k, _ in CONF_KEYS if k not in ('model', 'dataset', 'time_step')])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='run 디렉토리 또는 glob')
    ap.add_argument('--out', default='', help='CSV 저장 경로')
    ap.add_argument('--print', dest='show', action='store_true', help='요약을 화면에')
    a = ap.parse_args()

    dirs = []
    for p in a.paths:
        hits = sorted(glob.glob(p)) or ([p] if os.path.isdir(p) else [])
        dirs += [d for d in hits if os.path.isdir(d)]
    dirs = [d for d in dict.fromkeys(dirs)]
    if not dirs:
        raise SystemExit('디렉토리를 못 찾음')

    recs = [collect(d) for d in dirs]

    if a.out:
        os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
        with open(a.out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
            w.writeheader()
            w.writerows(recs)
        print(f'저장: {a.out}  ({len(recs)}행)')

    if a.show or not a.out:
        print(f'{"run":38s} {"ep":>4} {"val_acc":>8} {"s_count":>10} '
              f'{"S30/S1":>7} {"배제":>4}  설정')
        for r in recs:
            tag = []
            if r['final_step']: tag.append('final_step')
            if r['maxnorm']: tag.append(f"1-mx/{r['mx_group'] or 'none'}")
            if r['vmem_gain'] not in ('', '0.0'): tag.append(f"vmem{r['vmem_gain']}")
            if r['accum']: tag.append('accum')
            if r['loss_ratio']: tag.append(f"rho={r['rho']}")
            elif r['lambda']: tag.append(f"λ={r['lambda']}")
            if r['sc_sm'] and not r['maxnorm']: tag.append('1-sm')
            if r['seed'] not in ('', '-1'): tag.append(f"seed={r['seed']}")
            print(f"{r['run_dir'][:38]:38s} {r['epochs']:4d} "
                  f"{str(r['best_val_acc']):>8} {str(r['s_count']):>10} "
                  f"{str(r['S30_over_S1']):>7} {'X' if r['exclude'] else '':>4}  "
                  f"{' '.join(tag)}")
        ok = [r for r in recs if not r['exclude'] and r['best_val_acc'] != '']
        if len(ok) > 1:
            import statistics as st
            acc = [r['best_val_acc'] for r in ok]
            sp = [r['s_count'] for r in ok]
            print(f'\n유효 {len(ok)}런  정확도 평균 {st.mean(acc):.3f} sd {st.stdev(acc):.3f}  '
                  f'스파이크 평균 {st.mean(sp):.0f}')


if __name__ == '__main__':
    main()
