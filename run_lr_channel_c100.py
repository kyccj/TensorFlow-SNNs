"""loss-ratio × 채널 간 가중치 결합 — CIFAR-100 (26-08-23 등록)

V16-C10 쌍 결과 (같은 조건, 방금 종료):
    전역      94.43% / 42,113 (잔량 53.9%)
    채널 간   94.40% / 40,815 (잔량 52.2%)
    => 차이 −0.03%p. **고정 λ 에서 보이던 +0.3%p 가 loss-ratio 에서는 사라졌다.**

같은 조건을 CIFAR-100 으로 옮겨 확인한다. 구조는 같고 데이터셋만 다르므로 α=7 의 의미가 유지된다.

ρ=1e-3 착지 예상 — 잔량 48~52% (C10 쌍의 52~54% 와 같은 구간)
    기존 무브레이크 실측: 2.6e-4→79.8%, 5.8e-4→66.3%, 1.2e-3→45.1%

주의
  - 이 구간에서 C100 은 손실이 크다 (ρ=1.2e-3 에서 −4.39%p). 두 조건 모두 크게 잃을 것이다
  - C100 은 반복 편차가 크다 (ρ=5.8e-4 에서 sd 0.70, n=6). **쌍 하나로는 0.5%p 미만 차이를 못 가린다**
  - 따라서 이 실험이 답할 수 있는 것은 "C10 처럼 차이가 없다" 또는 "C100 에서는 크게 갈린다" 뿐이다

--- 이하 원래 설명 ---
loss-ratio × 채널 간 가중치 결합 (26-08-23 등록)

두 갈래를 합친다.
  loss-ratio  : λ 를 스윕 없이 자동으로 정한다 (설정마다 최적 λ 가 17배 다른 문제를 해결)
  채널 간 α=7 : λ=2e-7 에서 95.03% / 40,213 (잔량 51.4%) — 무손실로 절반 감소 (n=1, 반복 중)

ρ = 1e-3 을 쓴다. 관심 구간(잔량 50% 안팎)에 들어가는 가장 단순한 값이다.
  V16-C10 실측 : ρ=1.2e-3 -> 잔량 49.8%,  ρ=5.8e-4 -> 69.5%
  R19-C10 실측 : ρ=5.8e-4 -> 68.1%,       ρ=3e-3   -> 23.6%
  => ρ=1e-3 이면 두 설정 모두 잔량 50~55% 근처에 착지할 것으로 본다.
     (1.2e-3 처럼 소수 두 자리로 맞추면 '맞춘 값'으로 보이므로 단순한 값을 쓴다)

대조군은 같은 ρ 로 전역 가중치를 돌려 1:1 로 만든다. V16-C10 은 ρ=1.2e-3 전역 실측
(94.74% @ 38,960) 이 있지만 ρ 가 달라졌으므로 전역도 ρ=1e-3 으로 새로 돌린다.

브레이크는 끈다 (소스가 무브레이크이고, 브레이크 효과 자체가 미확정이라 변수를 늘리지 않는다).

소스 config: _rho_ladder/rl_c10_12_nb
  이미 loss-ratio + α=7 + VGG16-CIFAR10 + 무브레이크. 여기서 바꾸는 것은
  GPU / 이름 / ρ / (채널가중치 켬) / (R19 로 교체) 다섯 가지뿐이고, 그 외 차이가 없음을 검증한다.
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_lrch100')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
RESERVED = {6, 7}
SRC = '_rho_ladder/rl_c10_12_nb'
RHO = '0.001'

# (이름, 채널가중치 켬?, ResNet19 로 바꿈?)
JOBS = [
    ('v16c100_ch',   True,  True),
    ('v16c100_glob', False, True),
]

RE_GPU   = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)
RE_NAME  = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_ALPHA = re.compile(r'^(\s*)conf\.reg_spike_out_alpha=.*$', re.M)
RE_MODEL = re.compile(r"^conf\.model\s*=\s*'[^']*'", re.M)
RE_DATA  = re.compile(r"^conf\.dataset\s*=\s*'[^']*'", re.M)
RE_RHO   = re.compile(r'^(\s*)conf\.reg_spike_loss_ratio_target\s*=\s*\S+', re.M)
RE_SMG   = re.compile(r"^\s*conf\.reg_spike_sm_group\s*=\s*'channel'$\n?", re.M)


def active_name(txt):
    i = txt.find('conf.root_model_save=conf.exp_set_name')
    if i < 0:
        raise RuntimeError('root_model_save 줄을 못 찾음')
    ms = RE_NAME.findall(txt[:i])
    if not ms:
        raise RuntimeError('활성 exp_set_name 없음')
    return ms[-1].split("'")[1]


def make_config(gpu, name, chan, r19):
    c = open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()
    def sub(rx, rep, why, cnt=1):
        nonlocal c
        c, n = rx.subn(rep, c, count=cnt)
        if n != cnt:
            raise RuntimeError(f'{name}: {why} 치환 {n}회 (기대 {cnt})')
    sub(RE_GPU,  f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu}"', 'GPU')
    sub(RE_NAME, f"conf.exp_set_name='lrc100-{name}'",            'exp_set_name')
    sub(RE_RHO,  lambda m: f'{m.group(1)}conf.reg_spike_loss_ratio_target = {RHO}', 'rho')
    if chan:                       # α 줄 바로 뒤에 채널 간 묶음을 켠다
        sub(RE_ALPHA, lambda m: m.group(0) + f"\n{m.group(1)}conf.reg_spike_sm_group = 'channel'",
            'sm_group')
    if r19:                      # 여기서는 'CIFAR-100 으로 바꿈' 의 뜻
        sub(RE_DATA, "conf.dataset='CIFAR100'", 'dataset')
    return c


def verify(path, name, gpu, chan, r19):
    a = open(path).read()
    b = open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()
    # 허용한 항목만 지운 뒤 나머지가 완전히 같아야 한다
    strip = lambda s: RE_SMG.sub('', RE_RHO.sub('', RE_DATA.sub('', RE_MODEL.sub(
        '', RE_NAME.sub('', RE_GPU.sub('', s))))))
    if strip(a) != strip(b):
        raise RuntimeError(f'{name}: 허용 항목 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{name}: GPU 지정 안 됨')
    eff = active_name(a)
    if eff != f'lrc100-{name}':
        raise RuntimeError(f'{name}: 유효 exp_set_name 이 {eff!r}')
    if active_name(b) == eff:
        raise RuntimeError(f'{name}: 원본과 저장 경로가 같음')
    if not re.search(rf'^\s*conf\.reg_spike_loss_ratio_target = {re.escape(RHO)}$', a, re.M):
        raise RuntimeError(f'{name}: rho 가 {RHO} 아님')
    if bool(RE_SMG.search(a)) != chan:
        raise RuntimeError(f'{name}: sm_group 상태 불일치 (기대 {chan})')
    if not re.search(r"^conf\.model='VGG16'$", a, re.M):
        raise RuntimeError(f'{name}: model 이 VGG16 아님')
    want_d = 'CIFAR100' if r19 else 'CIFAR10'
    if not re.search(rf"^conf\.dataset='{want_d}'$", a, re.M):
        raise RuntimeError(f'{name}: dataset 이 {want_d} 아님')
    if not re.search(r'^\s*conf\.reg_spike_loss_ratio = True$', a, re.M):
        raise RuntimeError(f'{name}: loss_ratio 꺼져 있음')
    if re.search(r'^\s*conf\.reg_spike_lr_brake = True$', a, re.M):
        raise RuntimeError(f'{name}: 브레이크가 켜져 있음')
    if re.search(r'^\s*conf\.reg_spike_out_sc_maxnorm', a, re.M):
        raise RuntimeError(f'{name}: maxnorm 이 켜져 있음')


def preflight(jobs):
    bad = []
    for name, _c, _r in jobs:
        t = os.path.join(PROJECT_ROOT, f'lrc100-{name}')
        if os.path.exists(t) and any(f.endswith('.weights.h5')
                                     for _, _, fs in os.walk(t) for f in fs):
            bad.append(t)
    if not os.path.exists(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')):
        bad.append(f'{SRC} (원본 없음)')
    if bad:
        raise SystemExit('중단 — 저장 경로 충돌:\n  ' + '\n  '.join(bad))
    print('사전 점검 통과: 저장 경로 충돌 없음', flush=True)


def run_one(gpu, name, chan, r19):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    open(cfg, 'w').write(make_config(gpu, name, chan, r19))
    verify(cfg, name, gpu, chan, r19)
    m = open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')).read()
    open(os.path.join(d, 'main_sweep.py'), 'w').write(
        m.replace('from config_snn_training import config', 'from config_sweep import config'))
    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name} "
          f"(채널가중치={chan}, CIFAR100={r19}, ρ={RHO})", flush=True)
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
    pending = [j for j in JOBS if not sel or j[0] in sel]
    busy, threads = set(), []
    os.makedirs(SWEEP_DIR, exist_ok=True)
    preflight(pending)
    print(f'--- loss-ratio x channel: {len(pending)} runs (ρ={RHO}) ---', flush=True)
    while pending:
        for g in free_gpus(busy):
            if not pending:
                break
            busy.add(g)
            name, chan, r19 = pending.pop(0)

            def lane(g=g, n=name, c=chan, r=r19):
                try:
                    run_one(g, n, c, r)
                finally:
                    busy.discard(g)
            t = threading.Thread(target=lane); t.start(); threads.append(t)
            time.sleep(45)
        if pending:
            time.sleep(300)
    for t in threads:
        t.join()
    print('--- loss-ratio x channel drained ---', flush=True)


if __name__ == '__main__':
    main()
