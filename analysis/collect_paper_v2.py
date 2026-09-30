"""논문용 집계 v2 (26-09-21) — `collect_paper.py` 의 설정 파싱 버그를 고친다.

**왜 새로 쓰나.** `collect_paper.py` 는 `else:   # previous work` (225행) 아래를 죽은 가지로
보고 통째로 잘라낸다. 그런데 `run_paper.py` 가 실제로 적용한 값은 그보다 **아래**,
`# >>> run_paper overrides` ~ `# <<< run_paper overrides` (239~268행) 블록에 있다.
즉 진짜 설정이 잘려나가고 템플릿의 낡은 값이 읽힌다. 실측 피해:

    r19c10-ours_layer-4e-3-s1   실제 rho=4e-3, mx_group='none'
                                 구버전 보고: rho=0.003, mx_group=''  ← 팔 정체가 뒤바뀜

ANCOVA 의 팔 배정이 통째로 어긋나므로 v2 를 쓴다. 성능·S30/S1 파싱은 v1 과 동일 정의.

**팔 판정은 override 블록으로 한다.** 디렉토리 이름과 교차검증해서 불일치하면 행에 표시한다.
서버는 override 블록의 `root_model_save` 경로로 판정한다 (`/media/hdd1`=138, `/srv2`=23).
`host.txt` 는 실제로는 어느 런에도 없다.
"""
import argparse, csv, glob, os, re, time

EPOCH_LINE = re.compile(r'([A-Za-z_][\w\-]*): ([\d.]+(?:[eE][+-]?\d+)?)')
OPEN_MARK = '# >>> run_paper overrides'
CLOSE_MARK = '# <<< run_paper overrides'
ASSIGN = re.compile(r"^\s*conf\.([A-Za-z_]\w*)\s*=\s*(.+?)\s*$", re.M)


def _lit(s):
    s = s.split('#')[0].strip()
    if s.startswith(("'", '"')):
        return s.strip('\'"')
    if s in ('True', 'False'):
        return s == 'True'
    try:
        return float(s)
    except ValueError:
        return s


def read_conf(run_dir):
    """override 블록을 권위로 읽는다. 없으면 죽은 가지 위쪽으로 대체하고 표시."""
    p = os.path.join(run_dir, 'config_sweep.py')
    if not os.path.exists(p):
        return {}, 'no_config'
    txt = open(p, errors='ignore').read()
    i, j = txt.find(OPEN_MARK), txt.find(CLOSE_MARK)
    if i >= 0 and j > i:
        return {k: _lit(v) for k, v in ASSIGN.findall(txt[i:j])}, 'override'
    dead = re.search(r'^\s*else:\s*#\s*previous work', txt, re.M)
    body = txt[:dead.start()] if dead else txt
    return {k: _lit(v) for k, v in ASSIGN.findall(body)}, 'fallback_no_override'


def arm_from_conf(c):
    """override 설정 → 팔 이름. 이름 규약은 2026-09-21 확정본.

    09-30: loss-ratio 의 규제 시작을 늦춘 팔(run_paper 의 *_st30/*_st60)은 원 팔 이름 뒤에
    ' st30' 처럼 붙인다. start_ep 만 다르고 나머지 설정이 원 팔과 같아서, 붙이지 않으면
    원 팔에 섞여 들어간다. 기존 런은 전부 start_ep=0 이라 이름이 바뀌지 않는다.
    """
    arm = _arm_core(c)
    st = c.get('reg_spike_loss_ratio_start_ep', 0)
    if c.get('reg_spike_loss_ratio', False) and isinstance(st, (int, float)) and st > 0:
        arm += f' st{int(st)}'
    return arm


def _arm_core(c):
    if not c.get('reg_spike_out', False):
        return 'baseline'
    if c.get('reg_spike_out_bpsr', False):
        return 'BPSR'
    if not c.get('reg_spike_out_sc', False):
        return 'plain L2 + rho' if c.get('reg_spike_loss_ratio', False) else 'plain L2'
    # out_sc = True : ours 계열 / 1-softmax / 어블레이션
    if not c.get('reg_spike_out_sc_maxnorm', False):
        # 1- 역전이 없다 (sc/max 그대로). rho 제어기 유무로 갈린다.
        return '- 1- inversion' if c.get('reg_spike_loss_ratio', False) else '1-softmax'
    grp = c.get('reg_spike_maxnorm_group', '')
    w1 = c.get('reg_spike_wta_rev_w1', False)
    # 09-26: 어블레이션은 어느 팔에서 뺐는지 붙인다. prop(within, w2) 에서 뺀 것은 옛 이름,
    # 확정 제안법 ours_layer_w1(none, w1) 에서 뺀 것은 'lw1 - ...'. 섞이면 어블레이션 표가 틀린다.
    base = 'lw1 ' if (grp == 'none' and w1) else ''
    if c.get('reg_spike_vmem_gain', 1.0) == 0.0:
        return base + '- vmem'
    if not c.get('reg_spike_final_step', False):
        return base + '- final_step'
    if not c.get('reg_spike_loss_ratio', False):
        return base + '- loss-ratio'
    if c.get('reg_spike_maxnorm_no_inv', False):
        return base + ('- 1- inversion' if base else '- 1- inversion (wc)')   # 반전만 뺀 단일요인
    # 09-25: w1 판정이 경쟁 범위보다 먼저여서 ours_layer_w1(none+w1)이 ours_w1(within+w1)로
    # 묶였다. 경쟁 범위와 기울기를 함께 본다.
    if grp == 'within_channel':
        return 'ours_w1' if w1 else 'ours_intra_ch'
    if grp == 'none':
        return 'ours_layer_w1' if w1 else 'ours_layer'
    if grp == 'channel':
        b = c.get('reg_spike_shape_beta', 1.0)
        return 'ours_inter_ch' if b == 1.0 else f'ours_ch_b{int(round(b * 10))}'
    return f'unknown(grp={grp})'


# 디렉토리 이름 접두사 → 기대 팔 (교차검증용)
DIR_ARM = {
    'base': 'baseline', 'prop': 'ours_intra_ch', 'ours_layer': 'ours_layer',
    'ours_ch': 'ours_inter_ch', 'ours_ch_b7': 'ours_ch_b7', 'ours_ch_b4': 'ours_ch_b4',
    'ours_w1': 'ours_w1', 'ours_layer_w1': 'ours_layer_w1', 'l2': 'plain L2', 'l2_lr': 'plain L2 + rho', 'bpsr': 'BPSR',
    'sm': '1-softmax', 'abl_novmem': '- vmem', 'abl_nofinal': '- final_step',
    'abl_noinv': '- 1- inversion', 'abl_noinv_wc': '- 1- inversion (wc)', 'lw1_novmem': 'lw1 - vmem', 'lw1_nofinal': 'lw1 - final_step', 'lw1_nolr': 'lw1 - loss-ratio', 'lw1_noinv': 'lw1 - 1- inversion', 'abl_nolr': '- loss-ratio',
    # 09-30 규제 시작 지연 (loss-ratio start_ep)
    'ours_w1_st30': 'ours_w1 st30', 'ours_w1_st60': 'ours_w1 st60',
    'ours_ch_st30': 'ours_inter_ch st30', 'ours_ch_st60': 'ours_inter_ch st60',
    'ours_layer_w1_st30': 'ours_layer_w1 st30', 'ours_layer_w1_st60': 'ours_layer_w1 st60',
}
NAME = re.compile(r'^r19c(10|100)-(.+?)-([^-]*)-s(\d+)$')


def parse_name(run_dir):
    m = NAME.match(os.path.basename(run_dir.rstrip('/')))
    if not m:
        return {}
    ds, key, knob, seed = m.groups()
    return {'ds_name': 'CIFAR' + ds, 'dir_key': key,
            'dir_knob': knob.replace('p', '.'), 'seed': int(seed)}


def planned_epochs(run_dir, default=310):
    fp = os.path.join(run_dir, 'train.log')
    if not os.path.exists(fp):
        return default
    m = re.search(r'Epoch\s+\d+/(\d+)', open(fp, errors='ignore').read())
    return int(m.group(1)) if m else default


def read_epochs(run_dir):
    p = os.path.join(run_dir, 'train.log')
    if not os.path.exists(p):
        return []
    rows = []
    for line in open(p, errors='ignore').read().replace('\r', '\n').split('\n'):
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


def spike_ratio(run_dir, save_name, stores, at_epoch=30):
    """S(30)/S(first) — 학습 모드. proc.py:1254 와 같은 정의."""
    cands = [run_dir] + ([os.path.join(s, save_name) for s in stores] if save_name else [])
    for base in cands:
        p = os.path.join(base, 'reg_detail.csv')
        if not os.path.exists(p):
            continue
        tot = {}
        for r in csv.DictReader(open(p)):
            try:
                e, v = int(r['epoch']), float(r['spike_count'])
            except (KeyError, ValueError):
                continue
            if v > 0:
                tot[e] = tot.get(e, 0.0) + v
        if not tot:
            continue
        eps = sorted(tot)
        first = next((e for e in eps if tot[e] > 0), None)
        cand = [e for e in eps if first is not None and first <= e <= at_epoch]
        if first is None or not cand:
            continue
        near = cand[-1]
        return tot[first], tot[near], near, tot[near] / tot[first]
    return None, None, None, None


COLS = ['run_dir', 'server', 'dataset', 'arm', 'knob', 'seed', 'epochs', 'completed',
        'train_loss', 'train_acc', 'val_loss', 'val_acc', 'spikes',
        'best_epoch', 'S1', 'S30', 'S30_over_S1', 'exclude', 'exclude_reason',
        'conf_source', 'arm_dir', 'arm_mismatch', 'rho', 'lambda', 'mx_group', 'shape_beta', 'finished']


def collect(run_dir, stores):
    c, src = read_conf(run_dir)
    nm = parse_name(run_dir)
    arm = arm_from_conf(c) if c else 'no_config'
    arm_dir = DIR_ARM.get(nm.get('dir_key', ''), '?')
    save = os.path.basename(str(c.get('root_model_save', '')).rstrip('/')) or os.path.basename(run_dir)
    rms = str(c.get('root_model_save', ''))
    server = '138' if '/media/hdd1' in rms else ('23' if '/srv2' in rms else '?')
    rec = {'run_dir': run_dir, 'server': server, 'dataset': nm.get('ds_name', ''),
           'arm': arm, 'arm_dir': arm_dir,
           'arm_mismatch': int(arm_dir != '?' and arm_dir != arm),
           'seed': nm.get('seed', ''), 'conf_source': src,
           'rho': c.get('reg_spike_loss_ratio_target', ''),
           'lambda': c.get('reg_spike_out_const', ''),
           'mx_group': c.get('reg_spike_maxnorm_group', ''),
           'shape_beta': c.get('reg_spike_shape_beta', '')}
    # knob: rho 가 있으면 rho, 없으면 lambda
    rec['knob'] = rec['rho'] if c.get('reg_spike_loss_ratio', False) else rec['lambda']
    rows = read_epochs(run_dir)
    rec['epochs'] = len(rows)
    rec['completed'] = int(len(rows) >= planned_epochs(run_dir))
    # 종료 시각 = train.log 가 마지막으로 쓰인 때 (완주 런만). 원장의 '종료 시간' 칸 (09-29)
    try:
        rec['finished'] = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(os.path.join(run_dir, 'train.log')))) if rec['completed'] else ''
    except OSError:
        rec['finished'] = ''
    if rows:
        b = max(rows, key=lambda x: x['val_acc'])
        rec['best_epoch'] = rows.index(b) + 1
        rec['val_acc'] = round(b['val_acc'] * 100, 4)
        rec['spikes'] = round(b.get('s_count', float('nan')), 1)
        rec['val_loss'] = round(b.get('val_loss', float('nan')), 4)
        rec['train_loss'] = round(b.get('loss', float('nan')), 4)
        rec['train_acc'] = round(b.get('acc', 0.0) * 100, 4)
    else:
        for k in ('best_epoch', 'val_acc', 'spikes', 'val_loss', 'train_loss', 'train_acc'):
            rec[k] = ''
    s1, s30, at, ratio = spike_ratio(run_dir, save, stores)
    rec['S1'] = round(s1, 1) if s1 else ''
    rec['S30'] = round(s30, 1) if s30 else ''
    rec['S30_over_S1'] = round(ratio, 4) if ratio else ''
    reasons = []
    if not rec['completed']:
        reasons.append(f"미완주({rec['epochs']}ep)")
    if ratio is not None and ratio < 0.19:
        reasons.append(f'S30/S1={ratio:.3f}<0.19')
    rec['exclude'] = int(bool(reasons))
    rec['exclude_reason'] = '; '.join(reasons)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--store', action='append', default=[])
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    stores = a.store or ['/media/hdd1/kyccj/EIP/paper', '/media/hdd1/kyccj/EIP/archive']
    dirs = []
    for p in a.paths:
        dirs += [d for d in (sorted(glob.glob(p)) or [p]) if os.path.isdir(d)]
    dirs = list(dict.fromkeys(dirs))
    recs = [collect(d, stores) for d in dirs]
    out = a.out or '/dev/stdout'
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True) if a.out else None
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction='ignore')
        w.writeheader()
        w.writerows(recs)
    bad = [r for r in recs if r['arm_mismatch'] or r['conf_source'] != 'override']
    if bad:
        import sys
        print(f'[경고] 이름-설정 불일치 또는 override 없음 {len(bad)}건', file=sys.stderr)
        for r in bad:
            print(f"  {r['run_dir']}  conf={r['arm']}  dir={r['arm_dir']}  src={r['conf_source']}",
                  file=sys.stderr)


if __name__ == '__main__':
    main()
