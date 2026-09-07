"""1-maxnorm 완화 — beta 스윕 (26-08-26 등록)

    sc_rate = 1 - beta * (sc/max)      beta=1 이 현재 1-maxnorm (비트 단위 동일)

왜 완화하나 — 1-maxnorm 은 **재현성이 나쁘다**. 같은 lambda 를 반복했을 때 착지점이:
    1-softmax (사실상 균일, sc_rate 표준편차 0.0001) : 0.7 ~ 2.9%
    1-maxnorm (beta=1)                              : 3 ~ 37%
lambda=1e-6 에서는 두 run 이 14,532 / 9,975 로 갈라져 93.46% / 83.41% 가 나왔다.
원인은 수렴 후 max 불안정이 아니라 **10~20에폭 초기 폭주**다 (두 run 이 ep10 까지 같다가
ep20 에 5배 벌어지고 영영 못 돌아옴 — 활동이 떨어지면 surrogate 가 0 이라 회복 불가).

beta=0 은 안정적이고 beta=1 은 불안정하므로, 그 사이에 **차등은 남기고 흔들림만 줄이는**
지점이 있는지 본다.

mean(sc_rate) = 1 - beta*mean(sc/max) 이고 mean(sc/max)~0.04 이라
beta 0~1 에서 mean 이 0.956~1.000, **4.4% 밖에 안 변한다.** 실효 lambda 오염이 거의 없어
mean1 보정 없이 그대로 비교할 수 있다 (그래서 beta=1 이 기존 run 과 동일하게 유지된다).

설정 — 채널 내 1-maxnorm, lambda=5e-7 고정, gain/floor 없음.
대조군 beta=1 : 94.53% @ 20,710 인데 **n=1 이다.** 어제 vmem gain=2 의 95.05% 가
n=3 에서 94.77% 로 내려앉은 것을 봤으므로, 대조군부터 반복해야 한다.
그래서 beta=1 을 2회 더 넣어 n=3 으로 만든다.

규제 없음 95.00% (sd 0.11, n=5) @ 78,789.  전역 1-softmax 곡선이 보간 기준선.

판정 — 세 가지를 본다.
  (1) beta=1 의 n=3 평균이 94.53% 를 유지하나? (아니면 그것도 위쪽 끝값이었다)
  (2) beta<1 이 착지점 분산을 줄이나?
  (3) beta<1 이 같은 스파이크에서 정확도를 유지하나?
반복 편차 sd 0.17~0.28%p.
"""
import subprocess, os, threading, time, sys, re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(PROJECT_ROOT, '_beta')
PYTHON = '/home/kyccj/anaconda3/envs/venv_1/bin/python'
CUDA_LD_PATH = '/home/kyccj/anaconda3/envs/venv_1/lib'
RESERVED = {6, 7}          # juyun 님 GPU — 절대 쓰지 않는다

SRC = '_mxg/mx_wc_5e-7'    # 채널 내 1-maxnorm, wta_rev, log_detail 켜짐, λ=5e-7

#        이름        gain    lambda
JOBS = [
    ('b100_r2', '1.0',  '5E-7'),   # 대조군 반복 (현재 n=1, 94.53% @ 20,710)
    ('b100_r3', '1.0',  '5E-7'),   # 대조군 반복
    ('b050',    '0.5',  '5E-7'),   # 완화 중간
    ('b025',    '0.25', '5E-7'),   # 많이 완화
    ('b075',    '0.75', '5E-7'),   # 살짝 완화
    ('b000',    '0.0',  '5E-7'),   # 균일 앵커 (차등 0)
]

RE_NAME = re.compile(r"^conf\.exp_set_name\s*=\s*'[^']*'", re.M)
RE_GPU  = re.compile(r'^os\.environ\["CUDA_VISIBLE_DEVICES"\]\s*=\s*"\d+"', re.M)
RE_LMB  = re.compile(r'^(\s*)conf\.reg_spike_out_const\s*=\s*(\S+)', re.M)
RE_MXG  = re.compile(r"^(\s*)conf\.reg_spike_maxnorm_group\s*=\s*'within_channel'", re.M)
RE_GAIN = re.compile(r"^\s*conf\.reg_spike_shape_beta\s*=\s*(\S+)", re.M)


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


def make_config(gpu_id, name, gain, lmb):
    c = open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()
    c, n = RE_GPU.subn(f'os.environ["CUDA_VISIBLE_DEVICES"]="{gpu_id}"', c)
    if n != 1:
        raise RuntimeError(f'{name}: GPU 줄이 {n}개')
    c, n = RE_NAME.subn(f"conf.exp_set_name='bt-{name}'", c)
    if n < 1:
        raise RuntimeError(f'{name}: 활성 exp_set_name 없음')
    c, n = RE_LMB.subn(lambda m: f'{m.group(1)}conf.reg_spike_out_const={lmb}', c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: lambda 치환 실패')
    c, n = RE_MXG.subn(lambda m: (f"{m.group(1)}conf.reg_spike_maxnorm_group = 'within_channel'\n"
                                  f"{m.group(1)}conf.reg_spike_shape_beta = {gain}"), c, count=1)
    if n != 1:
        raise RuntimeError(f'{name}: gain 삽입 실패')
    return c


def verify(path, name, gpu, gain, lmb):
    a = open(path).read()
    b = open(os.path.join(PROJECT_ROOT, SRC, 'config_sweep.py')).read()
    norm = lambda s: RE_LMB.sub('', RE_NAME.sub('', RE_GPU.sub('', s)))
    a_cmp = RE_GAIN.sub('', a)             # 새로 넣은 줄은 '비교용 사본'에서만 지운다
    if norm(a_cmp).replace('\n\n', '\n') != norm(b).replace('\n\n', '\n'):
        raise RuntimeError(f'{name}: GPU/이름/lambda/gain 외의 차이가 있음')
    if f'"CUDA_VISIBLE_DEVICES"]="{gpu}"' not in a:
        raise RuntimeError(f'{name}: GPU 지정 안 됨')
    eff = active_name(a)
    if eff != f'bt-{name}':
        raise RuntimeError(f'{name}: 유효 exp_set_name 이 {eff!r} — 원본 덮어쓸 위험')
    if active_name(b) == eff:
        raise RuntimeError(f'{name}: 원본과 저장 경로가 같음')
    gs = RE_GAIN.findall(a)
    if gs != [gain]:
        raise RuntimeError(f'{name}: beta 가 {gs} (기대 [{gain!r}])')
    # 주석이 붙은 줄도 있으므로 $ 로 닫지 않는다
    for pat, why in [(r"^\s*conf\.reg_spike_out_wta_rev\s*=\s*True\b", 'wta_rev'),
                     (r"^\s*conf\.reg_spike_out_sc_maxnorm\s*=\s*True\b", '1-maxnorm'),
                     (r"^\s*conf\.reg_spike_maxnorm_group\s*=\s*'within_channel'", '채널 내'),
                     (r"^\s*conf\.reg_spike_log_detail\s*=\s*True\b", 'log_detail')]:
        if not re.search(pat, a, re.M):
            raise RuntimeError(f'{name}: {why} 설정 안 됨')
    for bad, why in [(r"^\s*conf\.reg_spike_out_sc_maxnorm_plain\s*=\s*True\b", 'maxnorm_plain'),
                     (r"^\s*conf\.reg_spike_loss_ratio\s*=\s*True\b", 'loss_ratio'),
                     (r"^\s*conf\.reg_spike_starget\s*=\s*True\b", 'starget'),
                     (r"^\s*conf\.reg_spike_wta_rev_floor\s*=", 'floor'),
                     (r"^\s*conf\.reg_spike_vmem_gain\s*=", 'vmem gain'),
                     (r"^\s*conf\.reg_spike_layer_cost\s*=", 'layer_cost (이번 실험과 섞이면 안 됨)')]:
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
    for name, gain, lmb in jobs:
        tgt = os.path.join(PROJECT_ROOT, f'bt-{name}')
        if os.path.exists(tgt) and any(f.endswith('.weights.h5')
                                       for _, _, fs in os.walk(tgt) for f in fs):
            bad.append(tgt)
    # 플래그가 실제로 존재하는지 (반영 안 됐으면 조용히 무시되고 gain=0 이 된다)
    if not re.search(r"reg_spike_shape_beta", open(os.path.join(PROJECT_ROOT, 'flags.py')).read()):
        bad.append('flags.py 에 reg_spike_shape_beta 가 없음')
    if not re.search(r"reg_spike_shape_beta",
                     open(os.path.join(PROJECT_ROOT, 'lib_snn', 'neurons.py')).read()):
        bad.append('lib_snn/neurons.py 에 beta 반영 안 됨')
    if bad:
        raise SystemExit('중단:\n  ' + '\n  '.join(bad))
    print('사전 점검 통과: 저장 경로 충돌 없음, beta 플래그 반영됨', flush=True)


def run_one(gpu, name, gain, lmb):
    d = os.path.join(SWEEP_DIR, name)
    os.makedirs(d, exist_ok=True)
    cfg = os.path.join(d, 'config_sweep.py')
    open(cfg, 'w').write(make_config(gpu, name, gain, lmb))
    verify(cfg, name, gpu, gain, lmb)
    m = open(os.path.join(PROJECT_ROOT, 'main_snn_training.py')).read()
    open(os.path.join(d, 'main_sweep.py'), 'w').write(
        m.replace('from config_snn_training import config', 'from config_sweep import config'))
    env = os.environ.copy()
    env['PYTHONPATH'] = d + ':' + PROJECT_ROOT + ':' + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = CUDA_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')
    env['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/home/kyccj/anaconda3/envs/venv_1'
    print(f"[{time.strftime('%m-%d %H:%M')}] [GPU {gpu}] START {name}  (β={gain}, λ={lmb})",
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
    print(f'--- 1-maxnorm beta sweep: {len(pending)} runs ---', flush=True)
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
