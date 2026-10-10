"""QKFormer 전용 실행기(~/runs/qkformer_wta_rev/queue_qk.py) 상태 -> <작업폴더>/qk.json.

원장의 'QKFormer' 칸이 이것을 읽는다. canus 에서만 돈다 (QKFormer 는 sejong 에 없음).
쓰는 법:  python analysis/ledger/qk_status.py <작업폴더>
"""
import datetime as dt, json, re, subprocess, sys
from pathlib import Path

QR = Path('/home/kyccj/runs/qkformer_wta_rev')
BASE = Path('/home/kyccj/runs/qkformer_baseline/c10_baseline')
TOTAL = 410                   # 400 + cooldown 10
EPOCH_S = 290                 # GPU 한 장에 2개씩 (09-28 실측 약 287s). 혼자면 140s

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
               'c100_ours_ch_rho1.2e-2_s43', 'c100_ours_ch_rho1.2e-2_s44', 'c100_ours_ch_rho1.2e-2_s45', 'c100_ours_ch_rho1.2e-2_s46']]
ALL_JOBS = list(queue_qk.ALL_JOBS) + EXTRA_JOBS

# sejong 런은 그쪽에서 읽는다: 이름 -> [pid, gpu, 에폭, 최고 eval_top1]
REMOTE = r"""
import json, subprocess, sys
from pathlib import Path
QR = Path('/home/kyccj/runs/qkformer_wta_rev')
ps = subprocess.run(['ps','-eo','pid,args'], capture_output=True, text=True).stdout.splitlines()
u = {b.strip(): a.strip() for a, b in (l.split(',') for l in subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'], capture_output=True, text=True).stdout.splitlines() if l)}
apps = {p.strip(): u.get(g.strip()) for p, g in (l.split(',') for l in subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'], capture_output=True, text=True).stdout.splitlines() if l)}
out = {'runner': next((l.split()[0] for l in ps if len(l.split()) > 1 and l.split()[1].endswith('python') and 'queue_qk.py' in l), None)}
for n in sys.argv[1:]:
    pid = next((l.split()[0] for l in ps if 'python' in l and l.rstrip().endswith('--experiment ' + n)), None)
    f = QR / n / 'summary.csv'; acc, ep = None, 0
    if f.exists():
        r = f.read_text().splitlines(); h = r[0].split(','); ep = len(r) - 1
        if 'eval_top1' in h:
            v = [float(x.split(',')[h.index('eval_top1')]) for x in r[1:] if x]; acc = max(v) if v else None
    out[n] = [pid, apps.get(pid) if pid else None, ep, acc, f.stat().st_mtime if f.exists() else None]
print(json.dumps(out))
"""
sej_names = [n for n, k, _, h in ALL_JOBS if k == 'train' and h == 'sejong']
try:
    sej = json.loads(subprocess.run(['ssh', '-p', '23456', '-o', 'BatchMode=yes', 'kyccj@161.122.23.23',
                                     'python3 - ' + ' '.join(sej_names)], input=REMOTE, capture_output=True,
                                    text=True, timeout=90).stdout.strip().splitlines()[-1])
except Exception:
    sej = {}

for name, kind, _, host in ALL_JOBS:
    if kind != 'train':
        continue
    if host == 'sejong':
        pid, gpu, ep, acc, *mt = sej.get(name, [None, None, 0, None, None])
        mt = mt[0] if mt else None
    else:
        # --resume 재개(10-08)는 --experiment 뒤에 인자가 더 붙는다 -> 끝맺음이 아니라 토큰으로 찾는다
        pid = next((int(l.split()[0]) for l in ps if 'python' in l and f'--experiment {name}' in (l.rstrip() + ' ').replace(f'--experiment {name} ', f'--experiment {name}\0')
                    and f'--experiment {name}\0' in (l.rstrip() + ' ').replace(f'--experiment {name} ', f'--experiment {name}\0')), None)
        acc, ep = best(QR / name)
        gpu = gpu_of(pid) if pid else None
        _f = QR / name / 'summary.csv'
        mt = _f.stat().st_mtime if _f.exists() else None
    st = 'done' if ep >= TOTAL else 'running' if pid else ('stale' if ep else 'todo')
    eta = (now + dt.timedelta(seconds=(TOTAL - ep) * EPOCH_S)).strftime('%m-%d %H:%M') if pid else None
    label = name.replace('ours_layer_w1_rho', 'ours_layer_w1 ρ ').replace('base_s', 'baseline s').replace('l2_3e-6_s', 'plain L2 λ 3e-6 s')
    label = label.replace('_s', ' s') if label.startswith('ours') else label
    jobs.append({'name': label, 'kind': '학습', 'status': st, 'epochs': ep, 'acc': acc,
                 'spikes': spikes(name), 'gpu': gpu, 'eta': eta, 'server': '23' if host == 'sejong' else '138',
                 # 종료 시각 = summary.csv 마지막 기록 (완주 런만, 09-29)
                 'finished': dt.datetime.fromtimestamp(mt).strftime('%Y-%m-%d %H:%M') if (mt and st == 'done') else None})
runner = ' / '.join(x for x in (f'138 pid {runner}' if runner else None,
                                f'23 pid {sej["runner"]}' if sej.get('runner') else None) if x) or None

tail = (QR / 'queue_qk.log').read_text().splitlines()[-1:] if (QR / 'queue_qk.log').exists() else []
json.dump({'runner': runner, 'total': TOTAL, 'last': tail[0] if tail else '', 'jobs': jobs},
          open(Path(sys.argv[1]) / 'qk.json', 'w'), ensure_ascii=False)
print(f'qk: runner={runner} jobs={[(j["name"], j["status"]) for j in jobs]}')
