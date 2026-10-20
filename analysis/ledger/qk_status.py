"""QKFormer 전용 실행기(~/runs/qkformer_wta_rev/queue_qk.py) 상태 -> <작업폴더>/qk.json.

원장의 'QKFormer' 칸이 이것을 읽는다. canus(138)에서 돌리고, sejong(23)·neuro(55) 런은 ssh 로 읽기만 한다.
55 는 queue_qk.py 가 아니라 손으로 띄운 런이다 (예: 55:~/runs/qkformer_wta_rev/qk_seeds55.sh).
쓰는 법:  python analysis/ledger/qk_status.py <작업폴더>
"""
import datetime as dt, json, re, subprocess, sys
from pathlib import Path

QR = Path('/home/kyccj/runs/qkformer_wta_rev')
BASE = Path('/home/kyccj/runs/qkformer_baseline/c10_baseline')
TOTAL = 410                   # 400 + cooldown 10
EPOCH_S = 290                 # GPU 한 장에 2개씩 (09-28 실측 약 287s). 혼자면 140s
EPOCH_S_55 = 140              # 55 는 GPU 한 장에 한 런 (같은 A6000). 에폭 3개 이상 돌면 실측으로 바꾼다

ps = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout.splitlines()
runner = next((l.split()[0] for l in ps if len(l.split()) > 1 and l.split()[1].endswith('python') and 'queue_qk.py' in l), None)


def gpu_of(pid):
    r = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'],
                       capture_output=True, text=True, timeout=60).stdout
    u = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                       capture_output=True, text=True, timeout=60).stdout
    idx = {b.strip(): a.strip() for a, b in (l.split(',') for l in u.splitlines() if l)}
    for l in r.splitlines():
        p, g = [x.strip() for x in l.split(',')]
        if p == str(pid):
            return idx.get(g)
    return None


def proc(pat):
    for l in ps:
        if pat in l and 'python' in l:
            return int(l.split()[0])
    return None


def best(run_dir):
    """summary.csv 의 eval_top1 최고값 (열 이름으로 찾음)."""
    s = run_dir / 'summary.csv'
    if not s.exists():
        return None, 0
    rows = s.read_text().splitlines()
    h = rows[0].split(',')
    if 'eval_top1' not in h:
        return None, len(rows) - 1
    k = h.index('eval_top1')
    # --resume 으로 이어 돌리면 timm 이 헤더 행을 한 번 더 쓴다 (10-08) -> 헤더 반복 행은 건너뜀
    data = [r for r in rows[1:] if r and not r.startswith(h[0] + ',')]
    v = [float(r.split(',')[k]) for r in data]
    return (max(v) if v else None), len(data)


def best_from_log(log):
    m = re.findall(r'Best metric: ([\d.]+)', log.read_text(errors='ignore')) if log.exists() else []
    return float(m[-1]) if m else None


def spikes(name):
    j = QR / f'spikes_{name}.json'
    if not j.exists():
        return None
    return json.loads(j.read_text())['totals']['spikes_per_sample']


now = dt.datetime.now()
jobs = []
# 끝난 학습 (비교 기준)
jobs.append({'name': 'baseline', 'kind': '학습', 'status': 'done', 'epochs': best(BASE)[1],
             'acc': best(BASE)[0], 'spikes': spikes('base'), 'gpu': None, 'eta': None,
             'finished': dt.datetime.fromtimestamp((BASE / 'summary.csv').stat().st_mtime).strftime('%Y-%m-%d %H:%M')})
jobs.append({'name': 'ours_layer_w1 ρ 1e-2', 'kind': '학습', 'status': 'done', 'epochs': TOTAL,
             'acc': best_from_log(QR / 'ours_layer_w1_rho1e-2.log'), 'spikes': spikes('rho1e-2'),
             'gpu': None, 'eta': None,
             'finished': dt.datetime.fromtimestamp((QR / 'ours_layer_w1_rho1e-2' / 'summary.csv').stat().st_mtime).strftime('%Y-%m-%d %H:%M')})
# 실행기 대기열 (queue_qk.py 의 JOBS 순서)
for name, pat in (('eval_rho1e-2', 'ours_layer_w1_rho1e-2 '), ('eval_base', 'c10_baseline')):
    out = QR / f'spikes_{name[5:]}.json'
    pid = next((int(l.split()[0]) for l in ps if 'measure_spikes' in l and pat.strip() in l), None)
    st = 'done' if out.exists() else 'running' if pid else 'todo'
    jobs.append({'name': ('스파이크 측정 — ' + ('ρ 1e-2' if 'rho' in name else 'baseline')),
                 'kind': '측정', 'status': st, 'epochs': None, 'acc': None, 'spikes': spikes(name[5:]),
                 'gpu': gpu_of(pid) if pid else None, 'eta': None})
sys.path.insert(0, str(QR))
import queue_qk                                  # 실행기의 작업 목록(ALL_JOBS, 서버 칸 포함)을 그대로 읽는다
# 10-10 사용자 결정으로 대기 스크립트(wait_launch_*.sh)가 띄우는 런 — queue_qk.py 의 목록에 없으므로 여기서 보탠다.
# (queue_qk.py 에 넣으면 그 실행기가 재시작될 때 다시 띄우려 들 수 있어 원장에서만 안다.)
EXTRA_JOBS = [(n, 'train', None, 'canus') for n in
              ['ours_ch_rho1.2e-2_s43', 'ours_ch_rho1.2e-2_s44', 'ours_ch_rho1.2e-2_s45',
               'c100_ours_ch_rho1.2e-2_s43', 'c100_ours_ch_rho1.2e-2_s44', 'c100_ours_ch_rho1.2e-2_s45', 'c100_ours_ch_rho1.2e-2_s46',
               'c100_ours_ch_rho6e-3_s43',
               # 10-13 사용자 승인: C10 무손실 범위 ρ 사다리 (시드 42, GPU 4·5)
               'ours_ch_rho1e-3', 'ours_ch_rho2e-3', 'ours_ch_rho3e-3', 'ours_ch_rho5e-3',
               # 10-14 사용자 'C100 inter 6e-3 도 돌려봐야 해': n=1 → 4 (canus GPU 6·7)
               'c100_ours_ch_rho6e-3_s44', 'c100_ours_ch_rho6e-3_s45', 'c100_ours_ch_rho6e-3_s46',
               # 10-17 사용자 'QK 2e-2 큐에 넣어줘': C10 ρ 2e-2 (시드 42), wait_launch_c10_rho2e-2.sh 가 빈 GPU 에 띄움
               'ours_ch_rho2e-2']]
# 10-18 ρ 2e-2 무손실 (−44%) → 시드 43~45 (n=4). 처음엔 canus 의 wait_launch_c10_rho2e-2_seeds.sh 로 띄우려 했으나
# 10-19 사용자 'QK 도 55 에서 돌려도 돼' 로 서버 55(neuro)로 옮겼다 (그 스크립트는 '사용 안 함' 처리됨).
# s43 -> 55 GPU 6, s44 -> GPU 7 은 바로, s45 는 GPU 8·9 가 빌 때 (55 의 qk_seeds55.sh). 그래서 호스트 '55'.
EXTRA_JOBS += [(n, 'train', None, '55') for n in
               ['ours_ch_rho2e-2_s43', 'ours_ch_rho2e-2_s44', 'ours_ch_rho2e-2_s45']]
ALL_JOBS = list(queue_qk.ALL_JOBS) + EXTRA_JOBS

# sejong(23)·neuro(55) 런은 그쪽에서 읽는다: 이름 -> [pid, gpu, 에폭, 최고 eval_top1, summary.csv mtime, 프로세스 경과초]
# 경과초는 --resume 으로 이어 돌린 런이면 None (에폭 수와 경과 시간이 어긋나 속도를 못 잰다)
REMOTE = r"""
import json, re, subprocess, sys
from pathlib import Path
QR = Path('/home/kyccj/runs/qkformer_wta_rev')
ps = subprocess.run(['ps','-eo','pid,args'], capture_output=True, text=True).stdout.splitlines()
u = {b.strip(): a.strip() for a, b in (l.split(',') for l in subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'], capture_output=True, text=True).stdout.splitlines() if l)}
apps = {p.strip(): u.get(g.strip()) for p, g in (l.split(',') for l in subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'], capture_output=True, text=True).stdout.splitlines() if l)}
out = {'runner': next((l.split()[0] for l in ps if len(l.split()) > 1 and l.split()[1].endswith('python') and 'queue_qk.py' in l), None)}


def gpu_of(pid, n):
    # 프로세스가 받은 CUDA_VISIBLE_DEVICES (한 장일 때) -> nvidia-smi 의 pid-GPU 대응 -> 띄운 스크립트 로그의 'on GPU N'
    try:
        for kv in Path('/proc/%s/environ' % pid).read_bytes().split(b'\0'):
            if kv.startswith(b'CUDA_VISIBLE_DEVICES=') and kv[21:].decode().isdigit():
                return kv[21:].decode()
    except OSError:
        pass
    if apps.get(pid) is not None:
        return apps[pid]
    for lg in sorted(QR.glob('qk_seeds*.log')):
        m = re.findall(r'launched ' + re.escape(n) + r' .* on GPU (\d+)', lg.read_text(errors='ignore'))
        if m:
            return m[-1]
    return None


def summary(f):
    # epoch 열이 숫자인 행만 센다. --resume 으로 이어 돌리면 timm 이 헤더 행을 한 번 더 쓰고 (10-08),
    # 쓰다 만 마지막 줄도 있을 수 있다 -> 헤더 반복·빈 줄·깨진 행은 건너뛴다
    r = f.read_text().splitlines()
    if not r:
        return None, 0
    h = r[0].split(','); k = h.index('eval_top1') if 'eval_top1' in h else None
    ep, v = 0, []
    for x in r[1:]:
        c = x.split(',')
        try:
            float(c[0])
            if len(c) < len(h):
                continue
            ep += 1
            if k is not None:
                v.append(float(c[k]))
        except ValueError:
            continue
    return (max(v) if v else None), ep


for n in sys.argv[1:]:
    # --resume 재개는 --experiment 뒤에 인자가 더 붙는다 -> 끝맺음이 아니라 토큰(뒤 공백 포함)으로 찾는다
    line = next((l for l in ps if 'python' in l and ('--experiment ' + n + ' ') in (l.rstrip() + ' ')), None)
    pid = line.split()[0] if line else None
    f = QR / n / 'summary.csv'; acc, ep = summary(f) if f.exists() else (None, 0)
    et = None
    if pid and '--resume' not in line:
        r = subprocess.run(['ps', '-o', 'etimes=', '-p', pid], capture_output=True, text=True).stdout.strip()
        et = int(r) if r.isdigit() else None
    out[n] = [pid, gpu_of(pid, n) if pid else None, ep, acc, f.stat().st_mtime if f.exists() else None, et]
print(json.dumps(out))
"""


def remote(label, target, py, names):
    """원격 서버에서 REMOTE 를 돌려 {이름: [...]} 를 받는다. 실패해도 원장은 계속 -- 경고 한 줄만 (refresh.sh 와 같은 관례).
    실패하면 그 서버 런은 진행 정보 없이(todo/0에폭) 보이므로, 조용히 넘기지 않고 이유를 남긴다."""
    if not names:
        return {}
    r = None
    try:
        r = subprocess.run(['ssh', '-p', '23456', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', target,
                            f'{py} - ' + ' '.join(names)], input=REMOTE, capture_output=True, text=True, timeout=90)
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        err = ' | '.join(r.stderr.strip().splitlines()[-2:]) if r is not None else ''
        print(f'[경고] 서버 {label} QKFormer 읽기 실패 — 그 서버 런은 진행 정보 없이 표시됨 '
              f'({type(e).__name__}, ssh rc={r.returncode if r is not None else "-"}) {err}')
        return {}


sej = remote('23', 'kyccj@161.122.23.23', 'python3', [n for n, k, _, h in ALL_JOBS if k == 'train' and h == 'sejong'])
# 55 의 python: 로그인 PATH 에 기대지 않고 시스템 python3 (3.10) 를 직접 쓴다
neu = remote('55', 'kyccj@161.122.23.55', '/usr/bin/python3', [n for n, k, _, h in ALL_JOBS if k == 'train' and h == '55'])

for name, kind, _, host in ALL_JOBS:
    if kind != 'train':
        continue
    et = None
    if host in ('sejong', '55'):
        pid, gpu, ep, acc, *mt = (sej if host == 'sejong' else neu).get(name, [None, None, 0, None, None])
        mt, et = (mt + [None, None])[:2]
    else:
        # --resume 재개(10-08)는 --experiment 뒤에 인자가 더 붙는다 -> 끝맺음이 아니라 토큰으로 찾는다
        pid = next((int(l.split()[0]) for l in ps if 'python' in l and f'--experiment {name}' in (l.rstrip() + ' ').replace(f'--experiment {name} ', f'--experiment {name}\0')
                    and f'--experiment {name}\0' in (l.rstrip() + ' ').replace(f'--experiment {name} ', f'--experiment {name}\0')), None)
        acc, ep = best(QR / name)
        gpu = gpu_of(pid) if pid else None
        _f = QR / name / 'summary.csv'
        mt = _f.stat().st_mtime if _f.exists() else None
    st = 'done' if ep >= TOTAL else 'running' if pid else ('stale' if ep else 'todo')
    sec = EPOCH_S
    if host == '55':
        sec = et / ep if (et and ep >= 3) else EPOCH_S_55
    eta = (now + dt.timedelta(seconds=(TOTAL - ep) * sec)).strftime('%m-%d %H:%M') if pid else None
    label = name.replace('ours_layer_w1_rho', 'ours_layer_w1 ρ ').replace('base_s', 'baseline s').replace('l2_3e-6_s', 'plain L2 λ 3e-6 s')
    label = label.replace('_s', ' s') if label.startswith('ours') else label
    jobs.append({'name': label, 'kind': '학습', 'status': st, 'epochs': ep, 'acc': acc,
                 'spikes': spikes(name), 'gpu': gpu, 'eta': eta, 'server': {'sejong': '23', '55': '55'}.get(host, '138'),
                 # 종료 시각 = summary.csv 마지막 기록 (완주 런만, 09-29)
                 'finished': dt.datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M') if (mt and st == 'done') else None})
runner = ' / '.join(x for x in (f'138 pid {runner}' if runner else None,
                                f'23 pid {sej["runner"]}' if sej.get('runner') else None,
                                f'55 pid {neu["runner"]}' if neu.get('runner') else None) if x) or None

tail = (QR / 'queue_qk.log').read_text().splitlines()[-1:] if (QR / 'queue_qk.log').exists() else []
json.dump({'runner': runner, 'total': TOTAL, 'last': tail[0] if tail else '', 'jobs': jobs},
          open(Path(sys.argv[1]) / 'qk.json', 'w'), ensure_ascii=False)
print(f'qk: runner={runner} jobs={[(j["name"], j["status"]) for j in jobs]}')
