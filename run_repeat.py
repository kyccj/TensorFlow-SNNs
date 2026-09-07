"""반복 실행기 — 기존 run의 config_sweep.py를 그대로 복사해 GPU/이름만 바꾼다.

조건을 새로 조립하지 않고 원본 config를 복사하므로 '같은 조건'이 보장된다.
(런처를 다시 쓸 때 치환 하나를 빠뜨려 조건이 달라지는 사고를 막는 목적)

JOBS: (원본 config 경로, 새 이름)
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_rep')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
RESERVED = {6, 7}

JOBS = [
    ('_grp/ch_a7',        'ch_a7_r2'),     # 채널 간 α=7  (지금 n=1 → n=3)
    ('_grp/ch_a7',        'ch_a7_r3'),
    ('_grp/wc_a0.5',      'wc_a05_r2'),    # 채널 내 α=0.5 (지금 n=1 → n=3)
    ('_grp/wc_a0.5',      'wc_a05_r3'),
    ('_grp/ch_flat',      'ch_flat_r2'),   # 기준점 94.56 이 운인지
    ('_meeting/lr_full',  'lr_full_r2'),   # 94.35 가 운인지 (대수적으로는 상대비와 동일해야 함)
]


# 활성(주석 아닌) 대입만 잡는다.
# 26-08-21 사고: 주석 처리된 #conf.exp_set_name= 줄을 먼저 치환해버려 활성 줄이 남았고,
# root_model_save 가 원본 이름 그대로여서 원본 체크포인트를 덮어썼다.
RE_NAME = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_GPU  = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)


def active_name(txt):
    """root_model_save 가 잡히는 시점(= conf.root_model_save 대입 직전)의 exp_set_name"""
    i = txt.find('conf.root_model_save=conf.exp_set_name')
    if i < 0:
        raise RuntimeError('root_model_save 줄을 못 찾음')
    ms = RE_NAME.findall(txt[:i])
    if not ms:
        raise RuntimeError('활성 exp_set_name 대입이 없음')
    return ms[-1].split("'")[1]


def make_config(gpu_id, src, name):
    c = open(os.path.join(PROJECT_ROOT, src, 'config_sweep.py')).read()
    c2, n = RE_GPU.subn(f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"', c)
    if n != 1:
        raise RuntimeError(f'{name}: GPU 줄이 {n}개 (1개여야 함)')
    c3, n = RE_NAME.subn(f"conf.exp_set_name='rep-{name}'", c2)
    if n < 1:
        raise RuntimeError(f'{name}: 활성 exp_set_name 줄 없음')
    return c3


def verify(path, src, name, gpu):
    a = open(path).read()
    b = open(os.path.join(PROJECT_ROOT, src, 'config_sweep.py')).read()
    norm = lambda s: RE_NAME.sub('', RE_GPU.sub('', s))
    if norm(a) != norm(b):
        raise RuntimeError(f'{name}: 원본과 GPU/이름 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{name}: GPU 지정 안 됨')
    # 핵심: 저장 경로를 정하는 '유효' 이름이 바뀌었는지
    eff = active_name(a)
    if eff != f'rep-{name}':
        raise RuntimeError(f'{name}: 유효 exp_set_name 이 {eff!r} — 원본을 덮어쓸 위험')
    if active_name(b) == eff:
        raise RuntimeError(f'{name}: 원본과 저장 경로가 같음')


def preflight(jobs):
    """시작 전에 저장 경로가 비어 있는지 확인 — 덮어쓰기 사고 재발 방지"""
    bad = []
    for src, name in jobs:
        tgt = os.path.join(PROJECT_ROOT, f'rep-{name}')
        if os.path.exists(tgt) and any(f.endswith('.weights.h5')
                                       for _, _, fs in os.walk(tgt) for f in fs):
            bad.append(tgt)
        if not os.path.exists(os.path.join(PROJECT_ROOT, src, 'config_sweep.py')):
            bad.append(f'{src} (원본 config 없음)')
    if bad:
        raise SystemExit('중단 — 저장 경로에 이미 가중치가 있음:\n  ' + '\n  '.join(bad))
    print('사전 점검 통과: 저장 경로 충돌 없음', flush=True)


def run_one(gpu, src, name):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    open(cfg, 'w').write(make_config(gpu, src, name))
    verify(cfg, src, name, gpu)
    m = open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')).read()
    open(os.path.join(d, 'main_sweep.py'), 'w').write(
        m.replace('from config_snn_training import config', 'from config_sweep import config'))
    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}  (원본 {src})", flush=True)
    with open(os.path.join(d, 'train.log'), 'w') as lf:
        rc = subprocess.Popen([PYTHON, os.path.join(d, 'main_sweep.py')], cwd=PROJECT_ROOT,
                              stdout=lf, stderr=subprocess.STDOUT, env=env).wait()
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] DONE  {name} rc={rc}", flush=True)


def free_gpus(exclude, max_mib=200):
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            timeout=60).decode()
    except Exception:
        return []
    res = []
    for line in out.strip().split('\n'):
        idx, mem = [x.strip() for x in line.split(',')]
        idx = int(idx)
        if idx in RESERVED or idx in exclude:
            continue
        if int(mem) <= max_mib:
            res.append(idx)
    return res


def main():
    sel = sys.argv[1:]
    pending = [j for j in JOBS if not sel or j[1] in sel]
    busy, threads = set(), []
    os.makedirs(SWEEP_DIR, exist_ok=True)
    preflight(pending)
    print(f'--- repeats: {len(pending)} runs ---', flush=True)
    while pending:
        for g in free_gpus(busy):
            if not pending:
                break
            busy.add(g)
            src, name = pending.pop(0)

            def lane(g=g, s=src, n=name):
                try:
                    run_one(g, s, n)
                finally:
                    busy.discard(g)
            t = threading.Thread(target=lane); t.start(); threads.append(t)
            time.sleep(45)
        if pending:
            time.sleep(300)
    for t in threads:
        t.join()
    print('--- repeats drained ---', flush=True)


if __name__ == '__main__':
    main()
