"""wta_rev backward 분모에 하한 c 를 넣어 '유한 규제' 확인 (26-08-25 등록)

문제 — l2_norm_wta_rev 의 backward 는

    grad = lambda * sc_rate / ||x||,     ||x|| = sqrt( sum over FIRING neurons of sc_rate^2 )

분모가 **현재 발화량**이다. 규제가 먹혀서 스파이크가 줄면 ||x|| 가 작아지고 뉴런당
pressure 가 오히려 커진다. 즉 성공할수록 세지는 가속 페달이다.
수치 확인 (합성 층, N=65536, sc_rate 는 침묵=1 / 발화=U(0,0.75)):

    floor    발화 20%      발화 5%       증폭
    0.0      1.692e-02    3.837e-02     2.27배   <- 현재
    0.04     1.229e-02    1.715e-02     1.40배
    0.15     8.110e-03    9.577e-03     1.18배
    0.6      4.457e-03    4.900e-03     1.10배

깊은 lambda 에서 완만히 나빠지지 않고 93%대로 무너지는 것이 이것 때문인지 본다.

방법 — 분모에 하한을 더한다.

    grad = sc_rate / sqrt( sum(x^2) + f * sum(sc_rate^2) )

sum(sc_rate^2) 는 **모든** 뉴런에 대한 합이라 학습 내내 ~0.8~1.0*N 으로 안정적이다
(뉴런의 78~97% 가 침묵이고 1-maxnorm 에서 침묵은 sc_rate=1). f=0 이면 현재 동작 그대로.
forward 는 건드리지 않았으므로 R 값과 loss-ratio 는 그대로 비교 가능하다.

주의 — floor 는 증폭만 죽이는 게 아니라 **전체 gradient 크기도 줄인다** (f=0.6 에서 3.8배).
그래서 같은 lambda 로 돌리면 "유해진 것"이 아니라 "약해진 것"이 된다. lambda 를 올려
보정한 뒤 **곡선끼리** 비교해야 한다. 아래 lambda 는 위 표의 초기 gradient 비로 보정한 값이다.

대조군 (같은 방법, f=0, 채널 내 1-maxnorm, V16-C10):
    lambda=1e-7  94.72% @ 49,963
    lambda=3e-7  94.46% @ 29,646
    lambda=5e-7  94.53% @ 20,710   <- 현재 최고 깊은 점
    전역 계열은 ~15K 에서 93.3% 로 붕괴 (lambda=1e-6)

판정 — floor 곡선이 같은 스파이크에서 f=0 곡선보다 위에 있으면 자기증폭이 원인이었다.
특히 15K 부근에서 94%대를 유지하면 확정.
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_floor')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
RESERVED = {6, 7}          # juyun 님 GPU — 절대 쓰지 않는다

SRC = '_mxg/mx_wc_5e-7'    # 채널 내 1-maxnorm, wta_rev, log_detail 켜짐

#        이름            floor   lambda   (lambda 는 초기 gradient 비로 보정)
JOBS = [
    ('f015_1e-6', '0.15', '1.0E-6'),   # f=0 의 5e-7 착지점(20,710) 과 맞물릴 예상
    ('f015_2e-6', '0.15', '2.0E-6'),   # 더 깊게
    ('f015_4e-6', '0.15', '4.0E-6'),   # f=0 이면 붕괴할 압력
    ('f060_2e-6', '0.60', '2.0E-6'),   # f=0.6 의 매칭점
    ('f060_4e-6', '0.60', '4.0E-6'),
    ('f060_8e-6', '0.60', '8.0E-6'),
]

RE_NAME  = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_GPU   = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)
RE_LMB   = re.compile(r'^(\s*)conf\.reg_spike_out_const\s*=\s*(\S+)', re.M)
RE_MXG   = re.compile(r"^(\s*)conf\.reg_spike_maxnorm_group\s*=\s*'within_channel'", re.M)
RE_FLOOR = re.compile(r"^\s*conf\.reg_spike_wta_rev_floor\s*=\s*(\S+)", re.M)


def active_name(txt):
    """주석 처리된 exp_set_name 이 아니라 실제로 마지막에 대입되는 값을 읽는다.
    (08-21 에 주석 줄을 치환해서 원본 체크포인트를 날린 사고 재발 방지)"""
    i = txt.find('conf.root_model_save=conf.exp_set_name')
    if i < 0:
        raise RuntimeError('root_model_save 줄을 못 찾음')
    ms = RE_NAME.findall(txt[:i])
    if not ms:
        raise RuntimeError('활성 exp_set_name 대입이 없음')
    return ms[-1].split("'")[1]


def lambdas(txt):
    return [m.group(2) for m in RE_LMB.finditer(txt)]


def make_config(gpu_id, name, floor, lmb):
    c = open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()
    c, n = RE_GPU.subn(f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"', c)
    if n != 1:
        raise RuntimeError(f'{name}: GPU 줄이 {n}개')
    c, n = RE_NAME.subn(f"conf.exp_set_name='flr-{name}'", c)
    if n < 1:
        raise RuntimeError(f'{name}: 활성 exp_set_name 없음')
    # lambda 는 '첫 번째' 활성 줄만 (두 번째 이후는 else 분기라 실행되지 않음)
    c, n = RE_LMB.subn(lambda m: f'{m.group(1)}conf.reg_spike_out_const={lmb}', c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: lambda 치환 실패')
    # floor 를 within_channel 줄 바로 뒤에 넣는다 (활성 블록 안이 보장됨)
    c, n = RE_MXG.subn(lambda m: (f"{m.group(1)}conf.reg_spike_maxnorm_group = 'within_channel'\n"
                                  f"{m.group(1)}conf.reg_spike_wta_rev_floor = {floor}"), c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: floor 삽입 실패')
    return c


def verify(path, name, gpu, floor, lmb):
    a = open(path).read()
    b = open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()
    norm = lambda s: RE_LMB.sub('', RE_NAME.sub('', RE_GPU.sub('', s)))
    a_cmp = RE_FLOOR.sub('', a)            # 새로 넣은 줄은 '비교용 사본'에서만 지운다
    if norm(a_cmp).replace('\n\n', '\n') != norm(b).replace('\n\n', '\n'):
        raise RuntimeError(f'{name}: GPU/이름/lambda/floor 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{name}: GPU 지정 안 됨')
    eff = active_name(a)
    if eff != f'flr-{name}':
        raise RuntimeError(f'{name}: 유효 exp_set_name 이 {eff!r} — 원본 덮어쓸 위험')
    if active_name(b) == eff:
        raise RuntimeError(f'{name}: 원본과 저장 경로가 같음')
    fs = RE_FLOOR.findall(a)
    if fs != [floor]:
        raise RuntimeError(f'{name}: floor 가 {fs} (기대 [{floor!r}])')
    # 주석이 붙은 줄도 있으므로 $ 로 닫지 않는다 (예: '...=True    # revised WTA')
    for pat, why in [(r"^\s*conf\.reg_spike_out_wta_rev\s*=\s*True\b", 'wta_rev'),
                     (r"^\s*conf\.reg_spike_out_sc_maxnorm\s*=\s*True\b", '1-maxnorm'),
                     (r"^\s*conf\.reg_spike_maxnorm_group\s*=\s*'within_channel'", '채널 내'),
                     (r"^\s*conf\.reg_spike_log_detail\s*=\s*True\b", 'log_detail')]:
        if not re.search(pat, a, re.M):
            raise RuntimeError(f'{name}: {why} 설정 안 됨')
    for bad, why in [(r"^\s*conf\.reg_spike_out_sc_maxnorm_plain\s*=\s*True\b", 'maxnorm_plain'),
                     (r"^\s*conf\.reg_spike_loss_ratio\s*=\s*True\b", 'loss_ratio'),
                     (r"^\s*conf\.reg_spike_starget\s*=\s*True\b", 'starget')]:
        if re.search(bad, a, re.M):
            raise RuntimeError(f'{name}: {why} 가 켜져 있음')
    la, lb = lambdas(a), lambdas(b)
    if len(la) != len(lb):
        raise RuntimeError(f'{name}: lambda 줄 개수가 바뀜 ({len(lb)}→{len(la)})')
    if la[0] != lmb:
        raise RuntimeError(f'{name}: 첫 lambda 가 {la[0]} (기대 {lmb})')
    if la[1:] != lb[1:]:
        raise RuntimeError(f'{name}: 두 번째 이후 lambda 가 바뀜 {lb[1:]}→{la[1:]}')


def preflight(jobs):
    bad = []
    if not os.path.exists(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')):
        bad.append(f'{SRC} (원본 config 없음)')
    for name, floor, lmb in jobs:
        tgt = os.path.join(PROJECT_ROOT, f'flr-{name}')
        if os.path.exists(tgt) and any(f.endswith('.weights.h5')
                                       for _, _, fs in os.walk(tgt) for f in fs):
            bad.append(tgt)
    # floor 플래그가 실제로 존재하는지 (flags.py 반영 안 됐으면 조용히 무시되고 f=0 이 됨)
    if not re.search(r"reg_spike_wta_rev_floor", open(os.path.join(PROJECT_ROOT, 'flags.py')).read()):
        bad.append('flags.py 에 reg_spike_wta_rev_floor 가 없음')
    if not re.search(r"reg_spike_wta_rev_floor",
                     open(os.path.join(PROJECT_ROOT, 'lib_snn', 'layers.py')).read()):
        bad.append('lib_snn/layers.py 에 floor 반영 안 됨')
    if bad:
        raise SystemExit('중단:\n  ' + '\n  '.join(bad))
    print('사전 점검 통과: 저장 경로 충돌 없음, floor 플래그 반영됨', flush=True)


def run_one(gpu, name, floor, lmb):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    open(cfg, 'w').write(make_config(gpu, name, floor, lmb))
    verify(cfg, name, gpu, floor, lmb)
    m = open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')).read()
    open(os.path.join(d, 'main_sweep.py'), 'w').write(
        m.replace('from config_snn_training import config', 'from config_sweep import config'))
    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}  (floor={floor}, λ={lmb})",
          flush=True)
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
    print(f'--- wta_rev floor sweep: {len(pending)} runs ---', flush=True)
    while pending:
        for g in free_gpus(busy):
            if not pending:
                break
            job = pending.pop(0)
            busy.add(g)

            def worker(g=g, job=job):
                try:
                    run_one(g, *job)
                finally:
                    busy.discard(g)

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            threads.append(t)
            time.sleep(90)      # 같은 GPU 를 두 job 이 잡는 것 방지
        if pending:
            time.sleep(60)
    for t in threads:
        t.join()
    print('--- 전부 완료 ---', flush=True)


if __name__ == '__main__':
    main()
