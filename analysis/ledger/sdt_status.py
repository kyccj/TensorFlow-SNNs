"""SDTv3 (Spike-Driven Transformer V3) 런 상태 -> <작업폴더>/sdt.json.

원장의 'SDT' 탭과 'SDTv3' 진행 칸이 이것을 읽는다. 런은 sejong(23)에 있고 **읽기만** 한다
(ssh 한 번으로 log.txt / pid.txt / train.log 를 읽어 온다 — 아무것도 쓰거나 실행하지 않는다).
ssh 가 실패·초과하면 빈 목록을 쓴다 (원장 갱신 전체가 죽지 않게).
쓰는 법:  python analysis/ledger/sdt_status.py <작업폴더>

새 학습 묶음(ImageNet 전체 본 런 등)은 ROOTS 에 한 줄 더하면 된다. 폴더 이름 규칙:
  <접두>_base_g<gpu>            -> baseline (규제 없음)
  <접두>_rho<ρ>_g<gpu>          -> ours_inter_ch, knob = ρ
  <접두>_ctrl / <접두>_rho<ρ>   -> (SDT1K) _g<gpu> 없음. gpu 는 meta.txt 의 gpus= (없으면 '0-7'). ctrl = baseline
  <접두>_kd_ctrl / <접두>_kd_rho<ρ> -> 지식 증류(KD) 런 (10-15~, main_kd.sh). knob 에 '+KD' 를 붙인다
"""
import datetime as dt, json, os, re, subprocess, sys
from pathlib import Path

HOST = ('-p', '23456', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'kyccj@161.122.23.23')
ROOTS = [
    # ds: 원장 탭 키 / divisor: log.txt 에 test_spikes 가 없을 때 train.log 'Total spikes' 에 곱하는 값
    #   (옛 코드 8641bad 는 50000 고정 분모로 나눔 -> in100 val 5000장이면 x10 해야 이미지당 값)
    {'ds': 'SDT', 'dir': '/srv2/kyccj/EIP/sdtv3_prep_1007', 'glob': 'sub100_*', 'total': 30, 'divisor': 10},
    # ImageNet-1k 전체 (10-08~): 새 코드라 log.txt 에 test_spikes(이미지당)가 있다 -> divisor 1.
    #   calib_* = 공식 171M ckpt 에서 출발하는 3에폭 보정(--epochs 30 --stop_epoch 3 -> log.txt 3행이 완료),
    #   main_*  = 본 런 30에폭 (대조·ρ0.3·1.0·3.0, 2 GPU 쌍). 이름에 _g<gpu> 가 없다. smoke_*, eval_headonly 는 glob 에 안 걸린다.
    {'ds': 'SDT1K', 'dir': '/srv2/kyccj/EIP/sdtv3_full', 'glob': 'calib_*', 'total': 3, 'divisor': 1},
    {'ds': 'SDT1K', 'dir': '/srv2/kyccj/EIP/sdtv3_full', 'glob': 'main_*', 'total': 30, 'divisor': 1},   # 본 런 (10-09, NVLink 쌍 2 GPU, 디렉토리 main_*)
]
ARM = {'base': 'baseline', 'rho': 'ours_inter_ch'}

REMOTE = r"""
import glob, json, os, re, sys, time
roots = ROOTS_JSON
out = []
now = time.time()
def alive(pid, name):
    try:
        return name in open('/proc/%s/cmdline' % pid).read().replace('\0', ' ')
    except Exception:
        return False
def rd(p):
    try:
        return open(p, errors='ignore').read()
    except Exception:
        return ''
for R in roots:
    for d in sorted(glob.glob(os.path.join(R['dir'], R['glob']))):
        if not any(os.path.isfile(d + '/' + f) for f in ('log.txt', 'pid.txt', 'meta.txt')):
            continue
        name = os.path.basename(d)
        rows = []
        for l in rd(d + '/log.txt').splitlines():
            try:
                j = json.loads(l)
            except Exception:
                continue
            if 'test_acc1' in j:        # 규제 런의 첫 행 {"reg_init":..., "epoch": -1} 은 에폭이 아님 (이게 "4/3" 표기의 원인)
                rows.append(j)
        meta = rd(d + '/meta.txt')
        pm = re.findall(r'\d+', rd(d + '/pid.txt').split('\n')[0]) if rd(d + '/pid.txt') else []
        if not pm:                                      # SDT1K: pid.txt 없이 meta.txt 첫 줄 'pid=<torchrun 부모>'
            pm = re.findall(r'\bpid=(\d+)', meta)
        pid = pm[-1] if pm else None
        seed = re.search(r'--seed[ =](\d+)', meta)
        gm = re.search(r'\bgpus=(\S+)', meta)          # main_*: 2 GPU 쌍 (launcher 가 meta 첫 줄에 기록)
        tl = rd(d + '/train.log').splitlines()
        # train.log: 에폭마다 '* Acc@1 ... / Total spikes' 가 (모델, EMA) 순서로 두 번 -> 짝수 번째가 모델
        sp = [float(m.group(1)) for m in (re.match(r'\s*Total spikes:\s*([\d.]+)', l) for l in tl) if m][0::2]
        ts = []
        for l in tl:
            m = re.match(r'\[(\d+):(\d+):(\d+)\.\d+\] \* Acc@1', l)
            if m:
                ts.append(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)))
        ts = ts[0::2]
        gaps = [(b - a) % 86400 for a, b in zip(ts, ts[1:])][-3:]
        ep_s = sorted(gaps)[len(gaps) // 2] if gaps else None
        slim = []
        for i, r in enumerate(rows):
            s = r.get('test_spikes')
            if s is None and i < len(sp):
                s = sp[i] * R['divisor']
            slim.append([r.get('train_loss') if r.get('train_task_loss') is None else r['train_task_loss'],
                         r.get('train_acc1'), r.get('test_loss'), r.get('test_acc1'), s])
        try:
            mt = os.path.getmtime(d + '/log.txt')
        except Exception:
            mt = None
        out.append({'ds': R['ds'], 'name': name, 'total': R['total'], 'seed': int(seed.group(1)) if seed else 0,
                    'alive': bool(pid and alive(pid, name)), 'pid': pid, 'rows': slim, 'ep_s': ep_s, 'mtime': mt,
                    'gpus': gm.group(1) if gm else None,
                    'done_file': os.path.exists(d + '/DONE')})   # 런처가 정상 종료 시 touch (2에폭 보정처럼 total 보다 짧게 끝낸 런도 done)
print(json.dumps({'now': now, 'runs': out}))
"""


def main():
    W = Path(sys.argv[1])
    res = {'runs': [], 'now': None}
    try:
        p = subprocess.run(['ssh', *HOST, 'python3 -'], input='ROOTS_JSON = ' + json.dumps(ROOTS) + '\n' + REMOTE,
                           capture_output=True, text=True, timeout=90)
        res = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:                              # 타임아웃·접속 실패·파싱 실패 모두 빈 목록
        print('sdt: 원격 읽기 실패, 빈 목록', type(e).__name__)
    jobs = []
    now = dt.datetime.fromtimestamp(res['now']) if res.get('now') else None
    for r in res['runs']:
        # 접미 (_lr4x 등) 는 knob 뒤에 붙여 표시 — 없으면 ρ 1.0 과 ρ 1.0_lr4x 가 같은 칸에 겹친다
        m = re.match(r'^(.*?)_(ctrl|base|rho([0-9.e+-]+?))(?:_(?!g\d+$)([a-z][a-z0-9]*))?(?:_g(\d+))?$', r['name'])
        if not m:
            continue
        arm = ARM['base' if m.group(2) in ('base', 'ctrl') else 'rho']
        knob = m.group(3) or '-'
        if m.group(4):
            knob += ' ' + m.group(4)
        if r['ds'] == 'SDT1K' and m.group(1) == 'calib':   # 보정과 본 런이 같은 칸에 겹치지 않게 knob 에 표시
            knob = (knob if knob != '-' else '0') + ' (보정)'
        if re.search(r'(?:^|_)kd$', m.group(1)):          # main_kd_*: 접두가 'main_kd'
            # 안 붙이면 main_kd_ctrl 이 main_ctrl 과 같은 tag(sdt1k-baseline---s0)·칸이 돼 평균에 섞인다.
            # 공백 대신 '+': tag 가 원장 artifact db 문서 id(notes/<tag>)로 쓰여 공백이 허용되지 않는다
            knob = (knob if knob != '-' else '0') + '+KD'
        rows, total = r['rows'], r['total']
        ep = len(rows)
        if r.get('done_file'):
            total = ep                                      # 의도한 길이만큼 끝난 런: n/n 으로 표시
        st = 'running' if r['alive'] else 'done' if ep >= total else 'stale'
        last = rows[-1] if rows else [None] * 5
        accs = [x[3] for x in rows if x[3] is not None]
        eta = None
        if st == 'running' and now and ep:
            per = r['ep_s']
            if per:
                eta = (now + dt.timedelta(seconds=per * (total - ep))).strftime('%m-%d %H:%M')
        jobs.append({'ds': r['ds'], 'arm': arm, 'knob': knob, 'seed': r['seed'], 'server': '23',
                     'gpu': m.group(5) or r.get('gpus') or '0-7',
                     'status': st, 'dir': r['name'], 'epochs': ep, 'total': total, 'dir': r['name'],
                     'train_loss': last[0], 'train_acc': last[1], 'val_loss': last[2],
                     'val_acc': max(accs) if accs else None, 'val_last': last[3], 'spikes': last[4],
                     'eta': eta, 'epoch_s': r['ep_s'],
                     'finished': dt.datetime.fromtimestamp(r['mtime']).strftime('%Y-%m-%d %H:%M') if (st == 'done' and r['mtime']) else None})
    # 원자적 교체: 쓰다 죽어도 반쯤 쓴 sdt.json 이 남아 build_ledger.py 의 json.load 를 깨지 않게
    tmp = W / 'sdt.json.tmp'
    with open(tmp, 'w') as f:
        json.dump({'jobs': jobs}, f, ensure_ascii=False)
    os.replace(tmp, W / 'sdt.json')
    print('sdt:', [(j['arm'], j['knob'], j['status'], j['epochs']) for j in jobs])


main()
