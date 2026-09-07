"""채널별 가중치의 자기 λ 곡선 그리기 (26-08-21 등록)

왜 — 지금까지 채널별 방법은 **λ=3e-7 한 점**밖에 없다. 점 하나를 전역의 29점짜리
곡선과 비교하는 셈이라 "곡선이 낫다"고 말할 수 없다.
더 중요한 건, λ=3e-7이 착지하는 33~35K는 V16-C10 **무료 예산 경계(44,959, 잔량 57.5%)
바깥**이다. 즉 지금까지 본 건 "손실 구간에서 덜 잃는다"이지 "무손실 구간이 넓어진다"가 아니다.
목적이 '정확도 유지 + 스파이크 감소'이므로 후자를 봐야 한다.

설계 — 두 방법 × λ 3점. 기존 3e-7과 합치면 각 4점 곡선이 된다.
    λ=1e-7  : 전역 대조 94.91% @ 53,067 (무료 예산 안쪽)
    λ=2e-7  : 무료 예산 경계(44,959) 부근에 착지할 것으로 예상 <- 핵심 점
    λ=5e-7  : 전역 대조 94.04% @ 24,364 (깊은 쪽)

원본 config를 복사해 GPU / 이름 / λ 세 가지만 바꾼다. λ는 192행(활성 분기)만 교체하고
225행(else 분기, 실행 안 됨)은 건드리지 않는다.

판정 — 채널별 곡선이 전역 곡선보다 위에 있고, 특히 잔량 57.5% 부근에서 무손실(94.84% 이상,
baseline 95.00 - 2sd)을 유지하면 **무료 예산 경계가 밀린 것**이다.
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_lcurve')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
RESERVED = {6, 7}

JOBS = [                       # (원본, 새 이름, 새 λ)
    ('_grp/wc_a0.5', 'wc05_2e-7', '2E-7'),   # 채널 내 α=0.5 — 무료 예산 경계 부근 (핵심)
    ('_grp/ch_a7',   'ch7_2e-7',  '2E-7'),   # 채널 간 α=7   — 같은 지점
    ('_grp/wc_a0.5', 'wc05_1e-7', '1E-7'),
    ('_grp/ch_a7',   'ch7_1e-7',  '1E-7'),
    ('_grp/wc_a0.5', 'wc05_5e-7', '5E-7'),
    ('_grp/ch_a7',   'ch7_5e-7',  '5E-7'),
]

RE_NAME = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_GPU  = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)
RE_LMB  = re.compile(r'^(\s*)conf\.reg_spike_out_const\s*=\s*(\S+)', re.M)


def active_name(txt):
    i = txt.find('conf.root_model_save=conf.exp_set_name')
    if i < 0:
        raise RuntimeError('root_model_save 줄을 못 찾음')
    ms = RE_NAME.findall(txt[:i])
    if not ms:
        raise RuntimeError('활성 exp_set_name 대입이 없음')
    return ms[-1].split("'")[1]


def lambdas(txt):
    return [m.group(2) for m in RE_LMB.finditer(txt)]


def make_config(gpu_id, src, name, lmb):
    c = open(os.path.join(PROJECT_ROOT, src, 'config_sweep.py')).read()
    c, n = RE_GPU.subn(f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"', c)
    if n != 1:
        raise RuntimeError(f'{name}: GPU 줄이 {n}개')
    c, n = RE_NAME.subn(f"conf.exp_set_name='lc-{name}'", c)
    if n < 1:
        raise RuntimeError(f'{name}: 활성 exp_set_name 없음')
    # λ 는 '첫 번째' 활성 줄만 (두 번째는 else 분기라 실행되지 않음)
    c, n = RE_LMB.subn(lambda m: f'{m.group(1)}conf.reg_spike_out_const={lmb}', c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: lambda 치환 실패')
    return c


def verify(path, src, name, gpu, lmb):
    a = open(path).read()
    b = open(os.path.join(PROJECT_ROOT, src, 'config_sweep.py')).read()
    norm = lambda s: RE_LMB.sub('', RE_NAME.sub('', RE_GPU.sub('', s)))
    if norm(a) != norm(b):
        raise RuntimeError(f'{name}: GPU/이름/λ 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{name}: GPU 지정 안 됨')
    eff = active_name(a)
    if eff != f'lc-{name}':
        raise RuntimeError(f'{name}: 유효 exp_set_name 이 {eff!r} — 원본 덮어쓸 위험')
    if active_name(b) == eff:
        raise RuntimeError(f'{name}: 원본과 저장 경로가 같음')
    la, lb = lambdas(a), lambdas(b)
    if len(la) != len(lb):
        raise RuntimeError(f'{name}: lambda 줄 개수가 바뀜 ({len(lb)}→{len(la)})')
    if la[0] != lmb:
        raise RuntimeError(f'{name}: 첫 lambda 가 {la[0]} (기대 {lmb})')
    if la[1:] != lb[1:]:
        raise RuntimeError(f'{name}: 두 번째 이후 lambda 가 바뀜 {lb[1:]}→{la[1:]}')


def preflight(jobs):
    bad = []
    for src, name, lmb in jobs:
        tgt = os.path.join(PROJECT_ROOT, f'lc-{name}')
        if os.path.exists(tgt) and any(f.endswith('.weights.h5')
                                       for _, _, fs in os.walk(tgt) for f in fs):
            bad.append(tgt)
        if not os.path.exists(os.path.join(PROJECT_ROOT, src, 'config_sweep.py')):
            bad.append(f'{src} (원본 config 없음)')
    if bad:
        raise SystemExit('중단 — 저장 경로 충돌:\n  ' + '\n  '.join(bad))
    print('사전 점검 통과: 저장 경로 충돌 없음', flush=True)


def run_one(gpu, src, name, lmb):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    open(cfg, 'w').write(make_config(gpu, src, name, lmb))
    verify(cfg, src, name, gpu, lmb)
    m = open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')).read()
    open(os.path.join(d, 'main_sweep.py'), 'w').write(
        m.replace('from config_snn_training import config', 'from config_sweep import config'))
    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}  (원본 {src}, λ={lmb})", flush=True)
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
    print(f'--- lambda curve: {len(pending)} runs ---', flush=True)
    while pending:
        for g in free_gpus(busy):
            if not pending:
                break
            busy.add(g)
            src, name, lmb = pending.pop(0)

            def lane(g=g, s=src, n=name, l=lmb):
                try:
                    run_one(g, s, n, l)
                finally:
                    busy.discard(g)
            t = threading.Thread(target=lane); t.start(); threads.append(t)
            time.sleep(45)
        if pending:
            time.sleep(300)
    for t in threads:
        t.join()
    print('--- lambda curve drained ---', flush=True)


if __name__ == '__main__':
    main()
